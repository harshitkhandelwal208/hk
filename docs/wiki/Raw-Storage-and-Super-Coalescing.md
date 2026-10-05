# Raw Storage and Payload Alignment

> **Status:** the container supports raw (unquantized) tensors and configurable payload alignment. What alignment buys you is narrower than earlier versions of this page claimed; this page states what the code does and what has been measured.

---

## Raw storage

`f32`, `f16`, `bf16`, `fp8`, integer and `bool` tensors are stored as their plain bit patterns. Opening a file maps it with `mmap` (`MapViewOfFile` on Windows); `HKReader.getRawF32/F16/BF16/Int8` return slices straight into the mapping, with no decode step and no copy. Pages are faulted in by the OS on first touch.

That is the full meaning of "zero decoding overhead": reading a raw tensor costs nothing beyond page faults. It does **not** mean inference with raw weights is free. A bf16 or f16 model is simply the largest and slowest-to-stream form, because decode speed is bounded by memory bandwidth (see [Benchmarks and Performance](Benchmarks-and-Performance)). Quantized formats move fewer bytes per token and are what the engine is tuned for; the engine decodes them inside its dot-product kernels, not as a separate dequantization pass.

Setting the header flag `RAW_WEIGHT_STORAGE` marks a file as holding raw weights. It is informational.

---

## Payload alignment

The writer starts every tensor payload (and each tensor's scale and residual buffers) on a multiple of the file's `alignment`, and records it in the header.

| Alignment | Preset | Intended for |
| :--- | :--- | :--- |
| 128 | `DEFAULT_ALIGNMENT_BYTES` | Default for files written by the Zig tools |
| 4096 | `UNIVERSAL_PAGE_ALIGNMENT_BYTES` | Page-granular mapping on common x86-64 and Linux ARM systems. Default for Python `save_raw`. |
| 16384 | `APPLE_SILICON_ALIGNMENT_BYTES` | 16 KiB pages (Apple Silicon) |
| 65536 | `DIRECT_DMA_ALIGNMENT_BYTES` | 64 KiB allocation granularity (Windows) |

Because 4096 and 16384 are multiples of 128, a file aligned to either also satisfies 128-byte alignment. That arithmetic is the only "universal" property here.

What is and isn't implemented:

- **Implemented:** the writer honors the alignment; `hk verify` checks that all payloads start on a 128-byte boundary and fit inside the file; the header flags `TILE_ALIGNED`, `FLEXIBLE_ALIGNMENT` and `UNIVERSAL_PAGE_ALIGNED` reflect the chosen value.
- **Not required by the engine.** CPU inference works with any alignment. The CPU kernels use unaligned loads, and the Vulkan backend copies weights into device buffers (re-laid out for the GPU) rather than using the mapped file in place.
- **Not implemented:** zero-copy GPU buffer creation from the mapped file (for example Metal `newBufferWithBytesNoCopy`). There is no Metal, ROCm/HIP, CUDA or NPU backend in the engine, so alignment for those targets is a property of the file, not something any code here exploits. Whether page alignment makes a difference for a third-party runtime that does zero-copy mapping has not been measured.

Choose the alignment for the file you will actually use: the default is fine for the engine; use 4096 or larger if another tool maps the file and wants page-aligned buffers.

---

## `shared_ref` and `null_ref`

- **`shared_ref` (`0x31`)**: a TOC entry that reuses the offsets of an earlier tensor. Tied embeddings (`token_embd` / `lm_head`) are stored once, and both names resolve to the same bytes. The Python `torch`/`numpy` savers create these by detecting identical `data_ptr`s; the writer requires the target to appear earlier in the TOC.
- **`null_ref` (`0x30`)**: a tensor with no stored bytes (`data_offset = data_size = 0`); the reader expands it to zeros on request. The shape and name remain, so architecture checks still pass.

---

## Python

The Python package's raw store (`hk.raw`) needs the native library built from this repo (see [Python API Reference](Python-API-Reference)):

```python
import torch
from hk.raw import save_raw, load_raw, HKRawWeightStore

weights = {
    "layer0.gate.weight": torch.randn(2048, 4096, dtype=torch.bfloat16),
    "layer0.up.weight": torch.randn(2048, 4096, dtype=torch.bfloat16),
}
save_raw("model_raw.hk", weights, alignment=4096)   # filename first

tensors = load_raw("model_raw.hk")                  # dict of tensors backed by the mapping

with HKRawWeightStore("model_raw.hk") as store:     # also exposes alignment info and gemv
    print(store.alignment, store.is_universal_page_aligned)
    y = store.gemv("layer0.gate.weight", torch.randn(4096))
```

Notes:

- `save_raw(filename, tensors, metadata=None, alignment=4096, split_index=0, split_count=1)`.
- `to_amd_rocm`, `to_intel_npu`, `to_apple_metal` and `to_nvidia_tensor_core` in `hk.raw` only return a contiguous copy of the tensor. They do not align memory or talk to any device API; treat them as compatibility stubs.
- `gemv` uses the native kernels for `bf16`/`f16`/`int8`/`f32` and a NumPy fallback otherwise.

---

## Verifying a file

```bash
hk verify model.hk
```

Checks the magic and version, that the payload offset and each tensor offset are 128-byte aligned, and that no tensor extends past the end of the file. It does not verify content checksums (the header `checksum` field is reserved and unused).
