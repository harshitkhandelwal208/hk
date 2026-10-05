# Storage, Sparsity, Sharding and Metadata Editing

> **Status:** this page covers container features. Several are *storage-only*: the file can hold the data and the reader can decode it to f32, but the inference engine does not use it (called out per section). Everything below refers to code in `src/sparsity.zig`, `src/metadata.zig`, `src/reader.zig`, `python/hk/quantization.py`, `python/hk/pruning.py` and `python/hk/torch.py`.

---

## 1. 2:4 structured sparsity (storage only)

A 2:4 tensor keeps 2 of every 4 consecutive values (the two largest magnitudes) and zeroes the rest.

**What exists**

- `sparsity.encodeStructured2_4_F32` / `decodeStructured2_4_F32` pack and unpack it. Payload: `[index metadata: 4 bits per group of 4, padded to 4 bytes][kept values as f32]`. For f32 input this is about 8.5 bytes per group instead of 16, roughly a 47% size reduction. The values themselves are unchanged, so decoding reproduces the pruned tensor bit for bit.
- The reader decodes `sparse_2_4` (and f32 tensors tagged `structured_2_4`), `sparse_f16`, bitmask-sparse and BSR tensors to dense f32 on request (`HKReader.dequantizeToF32`).
- Python: `hk.quantization.make_2_4_sparse` (prune, with optional norm-preserving scale correction), `pack_2_4` / `unpack_2_4` (call the native library), and `hk.pruning.prune_structured_2_4(model)` for a whole `nn.Module`.

**What does not exist**

- The inference engine has no sparse kernels. A 2:4 tensor is not read in its packed form during generation, and loading one into the engine is an error (`vecdot.supported` is false). Pruned weights only save disk space unless you decode them.
- Hardware sparse-tensor-core execution (NVIDIA Ampere+) is not used by anything in this repository. Pruning to 2:4 does not make HK inference faster, and pruning without fine-tuning generally costs accuracy.

```python
import torch
from hk.quantization import make_2_4_sparse, pack_2_4

w = torch.randn(2048, 4096)
sparse_w, _ = make_2_4_sparse(w)        # keep top-2 magnitudes in each group of 4
packed = pack_2_4(sparse_w)             # needs the native library
```

---

## 2. Tile layouts (declared, not produced)

The TOC has a `tile_layout` field with values for 16x16, 16x8, 32x16, 32x32 and 64x64 tiles. The converters and writers emit `row_major`, and the engine does not read pre-tiled files: it repacks weights into its own register tiles per thread at run time, directly from the mapped row-major data (see [Hardware and Kernels](Hardware-and-Kernels)). The field exists so other tools can describe a layout; no tiling gain is claimed.

---

## 3. In-place metadata editing

```bash
hk metadata list model.hk
hk metadata get  model.hk general.name
hk metadata set  model.hk general.version "1.1.0"
```

`set` infers the type from the value (JSON object/array, `true`/`false`, integer, float, else string), re-serializes the metadata block, and then:

- **If metadata + TOC still fit before the first tensor payload** (the file's alignment padding), it overwrites the header, metadata and TOC in place. Tensor bytes are never touched, so this takes milliseconds regardless of file size. With the default 128-byte alignment there is little spare room (at most 127 bytes beyond the existing metadata), so growing a long value (for example a chat template) often does not fit; files written with 4096-byte or larger alignment have more room.
- **Otherwise** it appends the new metadata block at the end of the file and points the header at it. The old block stays as dead bytes. This is still fast and does not move weights, but it is refused (`MetadataExceedsPaddingWithAppendix`) if the file already has appendix records, because those run to end of file.

Neither path has been tested for crash safety: the header write and the metadata write are separate operations, so an interruption in between can leave the file inconsistent. Work on a copy for important files.

---

## 4. Multi-file sharding

- Header: `IS_SHARDED` flag, `split_index`, `split_count` (see [Format Specification](Format-Specification)).
- Python `hk.torch.save_sharded_file(tensors, filename_pattern=..., max_shard_size=...)` splits tensors across files and writes an index manifest (`<name>.hk.index.json`, with a `weight_map` from tensor name to shard file); `load_sharded_file` reads it back.
- `hk.raw.save_sharded_raw(base_path, shards, ...)` writes caller-supplied shard dicts as `<stem>-00001-of-0000N.hk`; `load_sharded_raw(paths)` merges them.
- Hugging Face downloads that are sharded safetensors are handled by the converter (see [Downloading Models](Downloading-Models)); the engine itself loads a single `.hk` or `.gguf`.

---

## 5. Quantization formats

Which formats the engine can *run* is listed in [Compatibility](Compatibility). Summary:

- **GGUF-compatible formats** (legacy `Q4_0`...`Q8_0`, K-quants `Q2_K`...`Q6_K`, I-quants, `TQ1_0`/`TQ2_0`, `MXFP4`, `NVFP4`): stored byte-for-byte as in GGUF and executed by the engine's integer dot-product kernels. These are the supported inference formats.
- **HK dual-mode (`dq4`, `dq8`, ...)**: a 4-bit NF4-style base plus an optional residual stream that the reader can add back to approximate the original values (`HKReader.dequantizeToF32(entry, with_residual, out)`). Produced by the Python tools (`quantize_nf4_dual_mode`). Storage and offline use only; the engine does not run them, and no accuracy figure is claimed here.
