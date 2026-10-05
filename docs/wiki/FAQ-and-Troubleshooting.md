# FAQ and Troubleshooting

## Questions

### Can I run it without a GPU?
Yes. The CPU engine is the main one. It picks the fastest kernels your processor supports when it starts (see [Hardware and Kernels](Hardware-and-Kernels.md)).

### How do I use the GPU?
Pass `-ngl 99` (any number above 0) or set `HK_GPU=1`. hk prints either `[gpu] running on <device> (Vulkan)` or why it did not: no Vulkan driver, no usable device, a format the GPU cannot decode yet, or a model that does not fit in device memory. Partial offload is not implemented, so a model larger than your VRAM runs on the CPU.

### Does it work on Apple Silicon?
It builds for macOS on ARM and uses the NEON kernels (with the dot product extension when the chip has it). That path has been cross compiled and its portable logic tested, but **not run on a Mac**. There is no Metal backend. If you try it, `zig build test -Doptimize=ReleaseFast` and `hk run model.hk "hi" --temp 0` are the two commands that tell us whether it works; please report what they print.

### Why is my first prompt slow?
The model file is memory mapped, so the first pass reads it from disk. After that the pages are in the operating system's cache. `hk benchmark model.hk` separates opening, first touch and warm reads.

### How much memory does it use?
The "memory" figure printed after each answer splits resident memory into private (owned by the process) and mapped (the model file in the page cache, which the OS can drop). A few tens of MiB private is normal. The KV cache is allocated as the context fills, not up front.

### Which models work?
Llama, Qwen2 and Qwen3 style dense models. See [Compatibility](Compatibility.md).

### How does `.hk` compare with GGUF and safetensors?
- **GGUF** stores quantized blocks and metadata. A `.hk` file made from a GGUF holds the same blocks byte for byte; the container adds a table of contents, 128 byte aligned payloads, an optional append-only version history, and room for metadata edits in place.
- **safetensors** stores float tensors with a JSON header. `.hk` can hold the same tensors, and `hk export -f safetensors` writes them back (quantized tensors become F16).
- The research features of the container (growth, lineage, sparsity) are described in their own pages and are experimental.

## Troubleshooting

### `error: UnsupportedArchitecture` or "architecture ... is not supported"
The model family is outside what the engine runs. See [Compatibility](Compatibility.md).

### `hk pull` says "no checksum available to verify"
The Hub did not list a SHA-256 for that file (some safetensors shards). The download completed and was converted; it just could not be checked.

### `hk pull` says a token is needed
Set `HF_TOKEN` to a token that can read the repository.

### The GPU is not used
Read the `[gpu] not used: ...` line. Common causes: no Vulkan loader installed (`libvulkan.so.1`), the driver lacks 8 and 16 bit storage or subgroup arithmetic, the model has a format without a GPU kernel (see the table in [Compatibility](Compatibility.md)), or the model does not fit in VRAM. `hk hardware-profile` shows whether a device was found.

### Results differ between machines or kernel levels
With `--temp 0` the output is deterministic on one machine. Across instruction set levels the integer arithmetic is identical, and the float part differs only in summation order, so a text can occasionally diverge after many tokens when two candidates are almost tied. To check a level, compare perplexity, which is stable (`HK_KERNELS=avx2 hk-probe m.hk --ppl ids.u32`).

### Windows: "file in use" after loading with Python
Memory mapped files cannot be deleted while mapped. `hk.torch.safe_open` is a context manager that unmaps on exit; use `with safe_open(...) as f:` and delete the file afterwards.

### Tests
`zig build test` runs the unit tests with the portable kernels. `zig build test -Doptimize=ReleaseFast` adds every instruction set level the machine supports. `zig build test-e2e -Doptimize=ReleaseFast` runs the command line, server and downloader end to end. `pytest tests` covers the Python package. GPU tests skip themselves when no Vulkan device exists.
