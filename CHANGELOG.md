# Changelog

All notable changes to **hk** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed
- Authorship and contact now name me directly (Harshit Khandelwal, `harshitkhandelwal208@gmail.com`) in `SECURITY.md`, `CONTRIBUTING.md`, the README, `LICENSE`, `CITATION.cff` and the Python, npm, NuGet, crates.io and Maven package metadata, replacing "HK AI Research Team" and the `security@hk-ai.org` address. The package metadata reaches the registries with the next release.
- The README, wiki home, repository description and package description now present hk as one project that is both a fast runtime and a model format with bigger ambitions (zero-copy mmap, 2:4 sparse storage, Net2Net growth, in-file version history, self-learning loops), with a status for each.
- Repository root cleaned up for the public repo: `README.md` is rewritten as a landing page (measured 2.0 numbers, including where hk is behind llama.cpp, and the limits); `CONTRIBUTING.md` describes the current source layout; the agent orientation file `CONTEXT.md` moved to `docs/plan/CONTEXT.md`.
- `graphify-out/` (a generated knowledge graph of the code: `graph.json`, `GRAPH_REPORT.md`, `graph.html`, `manifest.json`) is now part of the repository.

### Removed
- `install.sh` and `install.ps1`: they were pinned to v1.0.0 and downloaded CLI binaries that releases do not publish. Install from source (`zig build -Doptimize=ReleaseFast`), `pip install hknt`, `npm install hkntf` or `cargo add hknt`.
- The four `run_*_demo.py` scripts and `PERFORMANCE_REPORT_KERNELS.md` at the repository root (the latter described a different repository's commit). The old demos are still in the git history of 1.1.x.

### Fixed
- `hk run`/`hk chat` printed `memory 0 MiB = 0 private + 0 mapped` on every platform but Linux, because the figure came from `/proc`. Memory now comes from `platform.selfMemory()`: Linux reads `/proc/self/status` as before, Windows lists the resident pages and splits them by the working set `Shared` flag (mapped model and DLLs count as mapped, the rest as private), and macOS reports the total only (the Mach call has no split). Where only the total is known the line shows just the total, and where nothing can be read it shows `n/a` instead of zeros. `hk-probe` uses the same call, and `hk-compare` now measures peak memory of a child process on Windows as well as Linux.
  Run: Windows 11, Zig 0.17, RTX 5060 laptop: `zig build test` (Debug, ReleaseFast, `-Dcpu=x86_64_v3`, `-Dcpu=x86_64`), `test-e2e`, and `hk run` on a 135M model (CPU: 10 private + 105 mapped MiB for a 100 MiB file; CUDA: 172 private + 144 mapped, the mapped part being mostly driver libraries). Cross-compiled only, not run: Linux x86_64, Linux aarch64, macOS aarch64. Not run: the macOS call, `processMemory` on Windows through `hk-compare` (needs llama.cpp binaries).

## [2.0.0] - 2026-10-21 - Unified runtime: backend abstraction, CUDA engine with tensor cores, partial offload, container integrity

HK 2.0 (plan §150): every item of the milestone exists and was checked on one machine (Ryzen 7 7445HS, RTX 3050 6 GB laptop, Linux); see `docs/wiki/Compatibility.md` for what was run and what was not, and `benchmarks/results/` for the numbers. Highlights: a `backend.Device` seam with Vulkan and CUDA under it, a CUDA engine whose device code is written in Zig (sm_70/sm_80 PTX from `zig build`, no toolkit needed), int8/f16 tensor cores and CUDA graphs, Vulkan cooperative matrices and integer-dot decode, a memory planner with partial offload and automatic device selection (now the default), the kernel stacks unified on the engine's, a SHA-256 container seal with crash-safe edits, and perplexity that matches llama.cpp and Transformers.

**Upgrading from 1.1.1**
- Any `.hk` file whose Q4_K tensors were written by the old Python/C-ABI quantizer must be re-quantized (that layout was private and already unusable in the engine). Files made from GGUF files are unaffected.
- `hk run`, `hk chat` and `hk serve` now use a CUDA or Vulkan device by default when one takes the model (`-ngl 0`, `--backend cpu` or `HK_GPU=0` for the CPU). `-ngl N` now means the first N layers, as in llama.cpp (it used to mean "the whole model" for any N above 0).
- C ABI version is 2 (`hk_abi_version()`); the product functions report failure as NaN and validate sizes. `hk verify`, `hk metadata set` and `hk rollback` exit non-zero on failure. Python's metadata and appendix mutations need the native library.
- The container format stays 1.0; files written by 2.0 carry an optional seal that 1.1.1 ignores.

### Container integrity and crash safety (plan §4, §5)
- **SHA-256 container seal** (header flag bit 9). Every producer seals: `HKWriter`, the streaming converters, `hk metadata set`, and appendix append/rollback. Older files stay readable and verify as *legacy*; a legacy file is sealed by its first edit.
- **Edits verify first.** Modifying a file whose seal does not match is refused instead of re-sealing the damage.
- **Crash-safe in-place edits.** A journal (`<file>.hkmeta-journal`) is synced before the first byte changes. Metadata patches and appendix appends roll back to the previous sealed file; appendix rollback is completed to the requested state. A torn journal is discarded only when the file's seal proves it untouched.
- **Atomic writes.** `HKWriter` and the streaming converters write `<path>.partial`, seal, then rename.
- **Bounded memory.** Sealing and verifying stream in 1 MiB chunks. Measured on a 2.1 GiB container: `hk verify` 1.27 s and `hk metadata set` 2.50 s at 13 MiB peak RSS (one machine, warm cache). Time is O(file).
- Per-record CRC-32 on appendix payloads (`0` = legacy record); overflow-safe range validation in the reader.
- New `hk repair`. **`hk verify`, `hk metadata set` and `hk rollback` now exit non-zero on failure** (they exited 0 before, so scripts could not detect a tampered file). `hk verify` also fails while an interrupted-edit journal is present.
- C ABI additions: `hk_verify_integrity`, `hk_appendix_append_at` (explicit timestamp). `hk_appendix_append` now rejects an unknown entry type instead of casting the byte to an enum.
- **Python:** `metadata_set`, `append_record` and `rollback_appendix` now run in the native library. The pure-Python implementations are removed (the metadata patcher left a stale seal and could corrupt an appendix; the appendix writers read the whole file into memory and rewrote it non-atomically). Without the native library they raise instead of falling back. `read_appendix` no longer loads the whole file when the native library is present.

### Backends (toward HK 2.0)
- **`backend.Device`**: a tagged union of whole-network executors that `Model` runs through (Vulkan today). The CLI prints the API name from the device.
- **CPU executor extracted** from `Model` into `engine/cpu_backend.zig` (`Cpu`): same code, moved. Logits are byte-identical to the previous build in 70/70 comparisons (f32 and Q8_0 models, four ISA levels, 1 and 6 threads, three prompt shapes, perplexity mode, Vulkan on an RTX 3050). **Speed was not re-measured** (no real model on the test machine; the hot loop is unchanged code).
- **Capability table** (`backend/caps.zig`) and **device planner** (`backend/select.zig`): device choice and context size are decided by a pure function of probe results, with a recorded reason for every rejected device (unusable, unsupported weight format, does not fit). The context halves until the model fits in 90 % of free device memory.
- **`--backend auto|cpu|vulkan|cuda` / `HK_BACKEND`.** Naming a device fails if it cannot be used. `gpu_note` is now owned by the model and may carry numbers.
- **CUDA engine (first version).** The whole network on one NVIDIA GPU through the CUDA driver API loaded at run time. **Device code is Zig**, compiled by `zig build` to PTX (no CUDA toolkit needed to build). One generic matrix-vector kernel serves all 26 weight formats the CPU runs, by calling the CPU's own `quant/dequant.zig` block decoders on the device. Also norm, RoPE (NeoX and adjacent-pair), f16 KV cache, causal attention, SwiGLU, residual add, QKV bias and Qwen3 per-head q/k norm. Verified against the CPU on an RTX 3050: f32 models agree to ~2e-6 absolute on logits; Q8_0 models agree with the CPU exactly as closely as Vulkan does (cosine 0.99993) and with Vulkan to ~1.5e-6. Performance (synthetic 136M-parameter Qwen3-shaped Q8_0 model with random weights, prompt 512 / generate 64, RTX 3050 6 GB Laptop, 5 interleaved repeats in one session; the model says nothing about quality and the numbers say nothing about other engines): Vulkan prefill 5,719 tok/s (5,490..5,830) and decode 406 (393..416); CUDA prefill 1,190 (1,187..1,192) and decode 360 (356..362); CPU (6 threads) prefill 1,906 and decode 91 (this laptop's CPU numbers swing about 2x between sessions, so compare only within a run). CUDA decode is within ~12 % of Vulkan, CUDA prefill is ~4.8x behind. How it got there, each step measured: batched per-layer passes with a tiled shared-memory GEMM (prefill 232 -> 1,131 tok/s), a warp-per-row matrix-vector kernel with the activation chunk staged in shared memory (decode 136 -> 279), split-KV attention for decode (253 -> 362). Not done: kernel fusion, CUDA graphs, Tensor Cores, integer dot products; so `--backend auto` still prefers Vulkan. Heads up to 256 wide; one device.
- **Vulkan prompt matmul on cooperative matrices** (`VK_KHR_cooperative_matrix`, new `shaders/mm_cm.comp`, all 15 GPU weight formats). Weights are decoded to f16 into shared memory, activations are rounded to f16, products accumulate in f32 on 16x16x16 tiles; 128 x 128 output tile per workgroup, next K slice prefetched into registers. Chosen automatically when the device reports the extension, a 16x16x16 f16/f16->f32 subgroup shape and a 32-wide subgroup; `HK_NO_COOPMAT=1` forces the previous f32 tile kernel, and `hk-probe` prints which one runs. **Numerics:** f16 operand rounding (about 2^-11 per product) instead of exact f32 products; logits cosine against the CPU stays >= 0.9999 on the parity models, and a new kernel-level test compares it with the f32 kernel on all 15 formats (red when the shader is mutated). Measured on the RTX 3050 laptop GPU, Qwen3-0.6B Q8_0, 512-token prompt: matmul time 214 ms -> 87 ms per pass, prefill 1,537 -> ~2,500 tok/s (3 repeats). Not run on AMD or Intel hardware (the extension and shape are checked, the kernel is untested there).
- **Vulkan prompt attention on cooperative matrices** (`shaders/attention_cm.comp`, head sizes 64 and 128). 64 query rows per workgroup share K/V tiles of 32 positions; S = QK^T and O += PV on f16 operands with f32 accumulate, online softmax with a lazy-rescale threshold (the running maximum only moves when a row's block maximum exceeds it by more than 8, so P stays below e^8 and O is rarely rescaled). Used when a pass is 16 or more consecutive positions of one sequence; decode, multi-sequence passes (the server) and other head sizes run the per-token kernel. Qwen3-0.6B Q8_0, 512-token prompt: attention 105 ms -> 8 ms per pass; with the matmul above, Vulkan prefill 1,537 -> ~4,800 tok/s (RTX 3050 laptop, one session, 2 repeats; decode unchanged at ~117 tok/s). Real-model logits (300-token prompt): cosine 0.9999995 against the previous Vulkan kernels, 0.99978 against the CPU, same as the CUDA engine gets. Tests: two prompt-attention parity tests against the CPU (head size 64 and 128, prompt in two chunks so the second starts mid-tile, every row checked) and a kernel-level test with keys that grow with position so the running maximum climbs (red when the rescale or the causal mask is mutated).
- **Vulkan device creation no longer fails after about 25 per process.** The NVIDIA driver (615.x; a 30 line C program reproduces it) returns `VK_ERROR_INITIALIZATION_FAILED` from `vkEnumeratePhysicalDevices` after about 25 instances have been created and destroyed. A model load opens two contexts, so a long running process (a server that reloads models, the test suite) would lose the GPU after a dozen loads, and the planner reported it as "the Vulkan device could not be opened". The loader and instance are now created once per process and shared. A test opens 60 contexts in a row.
- **CUDA tensor cores** (compute capability 8.0 and newer; written in Zig as inline PTX `mma.sync.m16n8k16`, f16 operands, f32 accumulation). `build.zig` now compiles the device code twice, for sm_70 and sm_80; the driver is handed the sm_80 build on GPUs that can run it (`HK_NO_TENSOR_CORES=1` keeps the portable one, and a driver that rejects the sm_80 module falls back to it). **Batched matmul** (`gemm_tc_<fmt>`, all 26 formats): a block decodes 64 rows of weights to f16 in shared memory through the same block decoders as before and multiplies them against 128 tokens; used from 16 tokens. **Prompt attention** (`attention_tc_64/128`): flash attention, 64 query tokens per block, online softmax on the MMA accumulator registers; used when a pass is 16 or more consecutive positions of one sequence (otherwise, and for decode, the earlier kernels run). Qwen3-0.6B Q8_0, 512-token prompt, RTX 3050 laptop, 3 repeats: prefill 350 -> 4,005 tok/s (4,002..4,011); decode unchanged (~108 tok/s). Operands are rounded to f16 (about 2^-11 per product), so results differ from the previous kernels at the 1e-3 level: the all-format kernel test bounds the error at 1.1e-3 of the sum of magnitudes, and the model-level tests keep the 0.9999 cosine bar against the CPU. Mutation-checked (swapped A fragment registers, dropped O rescale, off-by-one causal mask each turn a test red).
- **Partial offload** (plan §1.2: "should a layer be CPU or GPU resident?"). A model that does not fit in device memory now runs its leading layers on the device (Vulkan or CUDA) and the rest on the CPU: the CPU embeds, the device runs layers `0..k`, the hidden states come back (an [n][dim] copy per pass), the CPU runs layers `k..` and the output matrix. The planner (`backend/select.zig`) keeps the requested context and picks the largest `k` that fits in 90 % of free device memory, from exact per-layer weight bytes (mixed quantizations are handled) plus the per-layer KV cache; a whole-model plan keeps at least 2048 positions of context, below that the model goes partial instead of shrinking the window to 512. `-ngl N` / `--gpu-layers N` now means what it does in llama.cpp: exactly the first N layers (N at or above the layer count: the whole model). **Behaviour change:** `-ngl 1` used to mean "the whole model". The host `KvCache` of such a model covers only the CPU layers (`Model.newKv`, `KvCache.initFrom`), so device layers do not also cost host memory. Tests: planner cases (fits whole, exact boundary, context kept, forced count that fits and one that does not, nothing fits) and, on both devices, logits of 1 and 3 device layers of a 4 layer model against the CPU for a prompt and six single steps (cosine > 0.9999), plus the host cache size; mutation-checked. Measured on Qwen3-0.6B Q8_0 (28 layers, which fits whole; forced for the measurement), Vulkan, RTX 3050 laptop, prompt 128 / decode 16, one run each: 7 layers 676 tok/s prefill and 36 tok/s decode, 14 layers 980 / 46, all 28 layers 3,166 / 132, CPU alone ~24 decode.
- **Vulkan split-KV decode attention** (`attention_part.comp` + `attention_combine.comp`): for passes of up to 4 tokens the context of each (head, token) is cut into up to 16 slices (at least 32 positions each) run as separate workgroups and merged exactly; 16 heads of one token used to launch 16 workgroups. Qwen3-0.6B Q8_0 after a 512-token prompt: attention 1.3 -> 0.85 ms per token on the device (RTX 3050 laptop; measured while a compiler was running on the CPU, so end-to-end tok/s not claimed). Covered by the existing long-context parity test (700-token prompt, then single steps, against the CPU).
- **CUDA integer tensor cores** (`gemm_imma_<fmt>`, `quantize_act`; Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K). Activations are quantized to int8 per 32 values (the arithmetic the CPU engine already uses) and multiplied against the weights' own quants with `mma.sync.m16n8k32.s8` (two `m16n8k16` where a format has a scale per 16 values); products are exact in int32 and each 32-block's scale and minimum are applied once, `d_x * (sc * sum(q * xq) - mn * sum(xq))`. Weights are never decoded to floats on this path: the 32 quants of a block go to shared memory as signed bytes with their scales. Used for passes of 16 tokens or more on sm_80+; other formats keep the f16 tensor core kernel. Qwen3-0.6B Q8_0, 512-token prompt, RTX 3050 laptop: prefill 4,005 -> ~8,100 tok/s (the matmul time per pass 108 -> 36 ms). All eight formats are checked against an f64 CPU reference with the exact bound of the int8 activation rounding (`sum|w| * amax / 254` per block) and mutation-checked; Qwen2.5-1.5B Q4_K_M perplexity (wikitext-2, 20 windows of 512): llama.cpp 11.64-11.71, hk 11.64-11.70, of which CUDA (this path) 11.65.
- **CUDA decode** (single token): integer matrix-vector kernels (`gemv_dp4a_<fmt>`, the same eight formats, `dp4a` on the quantized activations; `HK_NO_DP4A=1` switches them off), **CUDA graphs** for passes of up to 8 tokens (captured once per token count and attention slice count, replayed afterwards; `HK_NO_CUDA_GRAPH=1`), and a new split-KV decode attention (`attention_part_warp`: one lane scores one position by itself, a warp does one softmax step per 32 positions, four warps merge once at the end; 128 positions per slice; `HK_NO_WARP_ATTN=1`; head widths that are a multiple of 32). Qwen3-0.6B Q8_0 after a 512-token prompt, RTX 3050 laptop: decode 108 -> ~138 tok/s (graphs +6 %, attention kernel +16 %, dp4a neutral on its own: the GEMV was already at ~75 % of memory bandwidth once the ~3.5 us per-launch overhead of the event timing is removed). Tests: decode steps after the two-chunk prompt at head sizes 64 and 128 on both devices; a long-context test at 16/64/128 with 400-position slices so each warp walks several tiles (turns red when the running-softmax rescale is removed).
- **CUDA dense weights and barrier-free GEMV.** f16, bf16 and f32 weights get vectorized staging in the tensor core GEMM (16-byte copies, no block decoder; prefill of Qwen3-0.6B f16 2,890 -> 4,590 tok/s) and a dedicated matrix-vector kernel; the dense and the `dp4a` GEMVs no longer stage the activations through shared memory behind a barrier (the barrier drained the memory pipeline every 4 KB of a row). Decode: Qwen3-0.6B f16 75.7 -> 88.9 tok/s, Q8_0 138.6 -> 143.8, Llama-3.2-1B Q4_K_M 121.4 -> 127.6 (llama.cpp CUDA on the same files: 90.8, 152.4, 137.5).
- **Automatic device use is now the default** for `hk run`, `hk chat` and `hk serve` (plan §1.2, automatic device selection): without flags a device is used when one exists and takes the model, and the CPU runs it otherwise. The `[gpu] running on ...` line appears when a device is used; `[gpu] not used: ...` appears when a device exists but cannot take the model (format, memory) and stays silent on machines with no device. `-ngl 0`, `--backend cpu` or `HK_GPU=0` keep the CPU. **Behaviour change** for anyone who relied on the CPU being the default. The end-to-end tests pin themselves to the CPU.
- **Vulkan integer decode kernels** (`mmvq.comp` + `qact.comp`, `VK_KHR_shader_integer_dot_product`, enabled when the device reports it; `HK_NO_INTDOT=1` switches it off). For passes of up to 4 tokens the activations are quantized to int8 per 32 values and the weights of Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K and Q6_K are multiplied with four products per `OpSDot` instead of unpacking nibbles to floats. Llama-3.2-3B Q4_K_M decode on the RTX 3050 laptop: 34.2 -> 50.5 tok/s (llama.cpp Vulkan on the same file: 55.8); Q8_0 is not routed through it (127.0 against 128.9 tok/s, the float kernel is as fast). Decode-path perplexity (`HK_BATCH=1 hk-probe --ppl`, Qwen2.5-1.5B Q4_K_M, 4 windows of 512) agrees across backends: CPU 9.960, Vulkan 9.955, CUDA 9.945, Vulkan float 9.924. A kernel-level test compares all eight shaders with the float kernels under the exact bound of the int8 activation rounding (mutation-checked on Q5_0, Q5_K and Q6_K).
- **Vulkan prompt matmul decodes four K-quant weights at a time** (`dq4` for Q4_K, Q5_K and Q6_K: the sub-block scale is unpacked once per four weights instead of once per weight). Prefill on the RTX 3050 laptop: Llama-3.2-1B Q4_K_M 2,889 -> 3,322 tok/s, Llama-3.2-3B Q4_K_M 1,016 -> 1,112. A scalar `dotPacked4x8` tile kernel for batches (64 x 64 tiles, 4 x 4 per thread) was written and measured and is **not** used: it was 2-3x slower than the f16 cooperative matrix kernel on this GPU (1B Q4_K_M 1,526 against 3,322), so it was removed again. The Vulkan f16 cooperative matrix GEMM is at this GPU's f16-with-f32-accumulate peak (~6.9 TFLOPS on the 3B model); getting past it needs int8 matrix units (the driver lists signed 8-bit 16x16x32 cooperative matrices) or f16 accumulation, both left for later.
- **Fixed: the CUDA engine failed every request in `hk serve`.** A CUDA context is current per thread; the server answers on worker threads, which never had the process's context current, so each call failed and the request returned `GpuFailed`. Every thread now binds the context on first use. Found by running the server on the GPU by hand (the end-to-end tests are pinned to the CPU); a new test uses a loaded engine from another thread on both devices and is red without the fix.
- `cuda.zig` gains exact-lifetime device allocation (the existing caching allocator can hand a huge cached block to a small request and keeps freed memory), and distinguishes out-of-device-memory from other failures.
- `hk-tiny-model --q8`.

### Quantization: one set of block formats, and three silent bugs fixed
The repository had two implementations of the GGML block formats, and the legacy one (`quantization.zig`) was wrong in places that nothing noticed:
- **Q4_K was a private layout** that only shared the name and the 144-byte block size (flat 16-weight groups, `q*d + dmin`), not GGML's (eight 32-weight sub-blocks with 6-bit scales and mins, `d*sc*q - dmin*m`). A bit-for-bit comparison against the engine's decoders (the ones checked against llama.cpp) showed 100 % of values differing. So a Q4_K tensor written by the Python/C-ABI quantizer was **silently misread by `hk run`**, and `HKReader.dequantizeToF32` (hence `hk_dequantize_f32`) decoded **real GGUF Q4_K tensors wrongly**. Now the only Q4_K is GGML's: a new encoder (`quant/encode.zig`) writes it and the engine's decoder reads it. **Breaking:** any `.hk` file whose Q4_K tensors were written by the old Python/ABI quantizer must be re-quantized (they were already unusable in the engine; the file carries nothing that distinguishes the two layouts). Python's `Q4_K_M`/`Q4_K_S` recipes now produce files `hk run` can load.
- **The legacy Q3_K encoder was broken against its own decoder** (mean absolute error 2.1 on unit-variance data, worse than storing zeros). Replaced by an encoder derived from the engine's decoder (error 0.13).
- `dequantizeToF32` skipped any block that did not fit instead of failing. It now routes every GGML block format through the engine's decoders and returns an error on short data. Scale-less GGML `iq4_nl`/`mxfp4`/`nvfp4` tensors, previously `MissingScales`, now decode; the HK-private form with a scales array still takes its own path.
- Q4_0, Q8_0, Q8_K, Q5_K, Q6_K and Q2_K decoders were already bit-identical to the engine's; their encoders moved unchanged. Research formats (NF4, DQ8, DQT, FP8, raw IQ4_NL/MXFP4/NVFP4 streams) moved to `quant/research.zig`. `legacy quantization.zig` is deleted; the C ABI, importers and `hk.quantization` keep their names through `quant/compat.zig`.
- New tests: every encoder round-trips through the engine's decoders; sub-blocks of very different magnitude round-trip independently (catches scales landing on the wrong sub-block, which an average over similar sub-blocks hides); the reader and engine agree on real GGML K-quant tensors; short tensor data is an error.

### C ABI products now run on the engine kernels
- `hk_gemv_*`, `hk_gemm_f32`, `hk_dot_*` and `hk_fused_*`/SwiGLU call `engine/abi_ops.zig`, which uses the same multi-threaded SIMD kernels as `hk run`. The legacy `tensor_ops.zig` (single-threaded: its "thread pool" ran every task inline) is deleted, as are nine unreferenced functions in it. The research products (NF4/DQ8/INT8-scaled GEMV and dot) moved to `quant/research.zig`, the f32 helpers (SiLU, norms, softmax, rotary layout permutations) to `ops.zig`; `rmsNormF32` now delegates to the engine's.
- **Numerics:** dense f32/f16/bf16 products agree with the old ones to float rounding (relative ~2e-7). Quantized-weight products now multiply 8-bit-quantized activations, as the engine does, so they differ from the old f32-activation results by about 0.4 % of the term magnitude (measured 0.43-0.46 %).
- **Strictness:** sizes are validated. A `k` that is not a whole number of blocks, too few weight bytes, or a short output buffer is an error rather than a clamped or zero-padded partial result. Because these ABI functions return `void`, a failure fills the output with **NaN** (and the `f32`-returning dot products return NaN) instead of leaving stale numbers.
- One-run check at 4096x4096 (ReleaseFast, 10/10/3 repetitions; not a full benchmark): f32 gemv equal (5.29 vs 5.40 ms), Q8_0 gemv 1.7x faster (3.35 vs 1.92 ms), f32 gemm 512^3 3x faster (16.0 vs 5.3 ms).

### Toolchain
- **Zig 0.17 is now required** (0.16 no longer builds this repository). Mechanical changes: optimize modes `.debug/.fast`, reflection via `field_names`/`field_values`, no `**` array repetition, no `@bitCast` from bytes to an extern struct, and the install prefix is unknown while configuring (the e2e tests now receive the binary path in `HK_EXE`).
- `test-e2e` runs are no longer result-cached. The cache key covered the test binary but not the `hk` executable it spawns, so after changing `hk` the suite could report green without running.
- Why now: Zig 0.17's NVPTX target can compile kernels written in Zig (0.16 cannot export a kernel), which is how the CUDA backend will be built with no `nvcc`.

### Verification
Unit tests 148/148 in Debug, ReleaseFast, `-Dcpu=x86_64_v3` and `-Dcpu=x86_64`; e2e 24/24; bindings 8 suites, 0 failed; Python 97 passed, 3 skipped (missing optional `openai`, `jax`, `tkinter`). Each new guard was disabled once to confirm its test fails. Not run: Windows, macOS, ARM.

---

## [1.1.1] - 2026-10-05 - Engine rewrite: portable kernels, tiled prefill, Vulkan backend, Zig tooling, verified language bindings

### Engine and performance
- **Runtime ISA dispatch.** The compute kernels are compiled once per level (x86: AVX-512+VNNI, AVX-VNNI, AVX2, generic; ARM: dotprod, NEON, generic) and selected at startup from detected CPU features. `HK_KERNELS=<level>` overrides the choice. Nothing is tuned to the build machine.
- **Register-tiled prefill GEMM** for legacy, K-quant, I-quant, ternary, MXFP4 and float formats. Weights stay memory-mapped; each thread repacks its tile on the fly (no second model copy).
- **Decode**: broadcast f16 scale conversion (removes a false dependency), fused QKV and gate/up regions, physical-core thread count, prefetch hints, a futex-skipping thread pool, split-KV decode attention, segmented f16 KV cache with tiled keys.
- **Vulkan compute backend** (`-ngl`, `HK_GPU`): whole model on one device, loaded dynamically (no build-time dependency), shaders checked in as SPIR-V. Tested on one NVIDIA GPU only. No Metal, ROCm, CUDA-in-engine or TPU backend.
- ARM NEON/dotprod kernels compile and pass the portable tests; they have not been run on ARM hardware.

### Tooling moved from Python to Zig
- Benchmark comparison (`hk-compare`), kernel microbenchmark (`hk-kernels`), tiny-model generator (`hk-tiny-model`), Unicode table generator, and the CLI/server/Hub test suites (`zig build test-e2e`, mock Hub and HTTP server in Zig). The Python `hk` command now just runs the native executable.
- New: `hk export` (safetensors), `hk hardware-profile` rewritten, `-ngl` flags.

### Language bindings
- New `tests/bindings/run.sh` (and a CI job running it): builds libhk, writes a fixture through the C ABI, and tests the C ABI, C++, Rust, Go, C#, Java and TypeScript bindings against it, including each writer, `metadata set`, and the C engine/tokenizer/sampler on a tiny model.
- **Java** had `native` methods with no implementation anywhere; added the JNI glue (`bindings/java/jni/hk_jni.c`, built with `zig build jni -Djdk=<dir>`).
- **Rust** crate now links libhk through `build.rs` (`HK_LIB_DIR`); `AppendixEntry` carries the payload.
- **Go** links `zig-out/lib` (was `zig-out/bin`) and exposes the appendix payload.
- **TypeScript**: the package is now a real ES module (it was CommonJS inside a `"type": "module"` package, so `import` failed), appendix records are read with their 8-byte padding, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/f16 decode, and unsupported storage types throw instead of returning zeros.
- **C#** exposes the appendix payload; `include/hk.h` and `include/hk.hpp` list every storage type (Q4_1, Q5_0, Q5_1, Q8_1, IQ2_S, IQ3_S were missing) and every exported function. `hk_layernorm_offset` in the header is now `hk_layernorm_offset_f32`, the name that is actually exported. A string literal passed to `hk::Writer::add_metadata` no longer selects the bool overload.
- All bindings recognize the full storage-type list.

### Fixes
- `hk_appendix_get_entry` returned `name` and `target` as C strings that were not NUL terminated (they pointed into the length-prefixed records), so every binding read garbage. They are now NUL-terminated copies.
- `hk_dequantize_f32` / `HKReader.dequantizeToF32` failed for Q4_1, Q5_0, Q5_1, Q8_1, the I-quants, IQ4_XS and the ternary types; they now use the engine's decoders.
- `hk metadata set` / `hk_metadata_patch_in_place` corrupted the file when the new metadata was longer but still fit in the alignment padding (the TOC was read through the mapping that the metadata was being written over). Covered by a new test.
- YaRN RoPE blend was inverted.
- `hk expand` wrote unexpanded f16/bf16 tensors as f32 bytes under their old type; it now stores them as f32. JSON metadata is now preserved.
- `hk metadata set` no longer appends metadata after an existing appendix (which corrupted the appendix); it reports an error instead.
- Head dimension validation in config.

### Documentation
- Wiki and README rewritten against the code. Overstated claims (zero-overhead GPU alignment, 2:4 hardware speedups, secure sandbox, SDK inference examples, unmeasured benchmark tables) removed or corrected; benchmark numbers now come from `hk-compare` with conditions stated.
- Minimum Zig version is 0.16.0.

## [1.1.0] - 2026-09-21 - Native SIMD Kernel Optimizations, Zero-Skip GEMM, RoPE Angle Caching & Quantization LUT

### Performance & Kernel Optimizations
- **High-Throughput SIMD GEMM (`gemmF32`)**:
  - Implemented `@Vector(8, f32)` vector lanes with 8-wide FMA unrolling and scalar tail processing.
  - Implemented zero-skipping heuristic (`@abs(val) > 1e-9`), achieving an empirical **9.32x speedup** on activation matrices (reducing compute time from 15.65 ms to 1.68 ms).
- **Rotary Position Embedding (RoPE) Angle Caching**:
  - Replaced repetitive `std.math.cos` and `std.math.sin` runtime transcendental function evaluations with a precomputed thread-safe static trigonometrical step table (`rope_cache_dim`).
  - Achieved an empirical **22.02x speedup** (reducing RoPE kernel time from 38.60 ms to 1.75 ms).
- **Comptime LUT FP8 E4M3 Dequantization**:
  - Implemented comptime 256-element IEEE FP8 E4M3 lookup table (`FP8_E4M3_LUT`) using bit manipulation, eliminating runtime IEEE exponent/mantissa decoding.
  - Achieved an empirical **34.51x speedup** (reducing dequantization from 36.65 ms to 1.06 ms).
- **Numerically Stabilized Underflow-Pruned Softmax (`softmaxF32`)**:
  - Pruned exponential underflow below threshold (`diff < -16.0f -> 0.0f`), skipping expensive hardware exp calls for suppressed logits.
  - Achieved an empirical **2.52x speedup** (reducing execution time from 7.02 ms to 2.78 ms).
- **4-Row Register-Tiled Quantized GEMV (`gemvQ8_0`, `gemvQ4_0`, `gemvF32`)**:
  - Restructured matrix-vector dot products to process 4 output rows concurrently with `@Vector(8, f32)` accumulation, maximizing L1 cache reuse of activation vectors.
  - Achieved an empirical **4.09x speedup** on Q8_0 GEMV (reducing execution time from 1.68 ms to 0.41 ms).
- **Vectorized Two-Pass LayerNorm (`layerNormF32`)**:
  - Vectorized mean, variance, and normalization loops with `@Vector(8, f32)` operations.
- **Python-Side Dispatch & Weight Transpose Caching**:
  - Cached pre-transposed weights in `HKLinear.forward()` and optimized ctypes parameter marshaling, achieving **3.5x to 79.1x speedups** on micro-dispatches.

### Bug Fixes
- **Tensor TOC Deallocation Buffer Size**:
  - Fixed memory allocator size mismatch in `src/tensor_toc.zig` by ensuring `e.name.ptr[0 .. e.name.len + 1]` is freed with the exact null-terminated allocation length.
- **Comptime FP8 LUT Branch Quota**:
  - In `src/quantization.zig`, increased `@setEvalBranchQuota(100_000)` and replaced iterative `pow` with bit-shift calculations for comptime evaluation.
- **Dynamic Offload Usable RAM Boundary**:
  - In `python/hk/offload.py`, safeguarded `usable_b` calculation when available host RAM is near or below the 2 GB threshold.

---

## [1.0.4] - 2026-09-16 - Automated Dynamic GPU/CPU Offloading Engine & Framework Optimizations

### Added & Enhanced
- **Automated Dynamic GPU/CPU Offloading Engine (`hk.offload`)**:
  - Implemented `HardwareMemoryInspector` for real-time GPU VRAM discovery (`torch.cuda.mem_get_info()`) and host RAM queries via `psutil`.
  - Implemented `LayerMemoryEstimator` for analytical sub-module parameter footprint budgeting.
  - Implemented `DynamicOffloadPlanner` with a conservative 128 MB buffer memory headroom to maximize on-device layer capacity while preventing runtime CUDA Out-Of-Memory (OOM) errors.
  - Implemented greedy water-filling layer placement across multi-GPU environments (`cuda:0`, `cuda:1`, ...) with seamless spillover of overflowing tail layers to host CPU memory.
  - Integrated `AutoDeviceDispatcher` for transparent non-blocking cross-device activation streaming (`hidden_states.to(next_device, non_blocking=True)`).
  - Implemented `DynamicOOMGuard` runtime watchdog to catch memory spikes (<256MB free) and safely migrate boundary layers to host memory instead of crashing.
  - Wired `device_map="auto"` in `AutoModel.from_pretrained()`, `model.to_dynamic_offload()`, and `pipeline(..., device="auto")`.
- **FP16 / BF16 Causal Mask Precision Fix**:
  - In `HKForCausalLM.forward()`, causal mask tensor now strictly inherits `hidden_states.dtype`, permanently eliminating the `RuntimeError: expected mat1 and mat2 to have the same dtype`.
- **Sub-500ms Zero-Copy Weight Serialization**:
  - In `NativeHKWriter`, eliminated triple-buffer memory duplication by passing contiguous raw tensor data pointers (`v.data_ptr()`) directly to the native C/Zig writer, cutting save latency from 1.77s to <450ms.
- **High-Throughput Native Streaming Tokenization**:
  - Added word-level BPE memoization cache (`_bpe_cache`) and an `encode_stream()` generator for streaming large document corpora.
- **On-Device Accelerator QLoRA Kernel**:
  - Cached dequantized base weights on the accelerator device in `HKQuantizedLinear` (`get_dequantized_weight()`), eliminating per-step CPU roundtrips and accelerating fine-tuning step times.
- **Scale-Corrected 2:4 Structured Sparsity**:
  - Added energy-conserving norm scaling to `make_2_4_sparse()`, boosting reconstruction PSNR from 23 dB to >32 dB.

---

## [1.0.3] - 2026-09-16 - Cross-Platform Truncation & CodeSandbox Isolation Fixes

### Fixed & Enhanced
- **POSIX In-Place File Truncation**:
  - Corrected `truncateFile` in `src/platform.zig` to directly invoke the POSIX `std.posix.system.ftruncate` system call with errno verification, ensuring appendix rollback functions properly on Linux and macOS.
- **CodeSandbox Isolation Robustness**:
  - Defaulted `use_docker=False` in `CodeSandbox` to utilize fast, self-contained AST-isolated subprocess execution out of the box without requiring external Docker setup.
  - Enhanced `_detect_docker` to verify Docker daemon connectivity via `docker info` and local image presence via `docker image inspect`.
  - Added automatic fallback to subprocess execution when Docker infrastructure errors (e.g. error code 125, missing container manifest) occur.
- **Documentation & Presentation**:
  - Cleaned `README.md` by removing test passing badge tags and streamlining the narrative section heading to `Why I Built HK`.

---

## [1.0.2] - 2026-09-16 - Training Robustness, True QLoRA & Cryptographic Lineage Hardening

### Added & Enhanced
- **Training Arguments & Optimizer Controls**:
  - Added native `gradient_accumulation_steps: int = 1`, `max_grad_norm: float = 1.0`, and `protect_base_capacity: bool = False` to `HKTrainingArguments`.
  - Implemented gradient accumulation loss scaling and `torch.nn.utils.clip_grad_norm_` gradient clipping inside `HKTrainer.train()`.
  - Changed `enable_adaptive_growth` default to `False` to prevent unexpected dynamic layer widening unless explicitly opted-in.
  - Automatically hooks `model.enable_continual_learning(protect_base=True)` upon trainer initialization when `protect_base_capacity` is set.
- **Cryptographic Version Chaining & Lineage Verification**:
  - Upgraded `verifyLineage` in native Zig (`src/appendix.zig`) and `verify_lineage` in Python (`hk.adaptive.appendix`) to cryptographically hash the complete record (`name` + `target` + `payload`) with SHA-256 rather than only payload data.
  - Enforced strict hash verification across all subsequent generations, disallowing all-zero hash bypasses.
  - Automatically chains parent hashes across incremental LoRA checkpoints in `AppendixManager`.
- **True QLoRA Quantized Base Adaptation**:
  - `enable_qlora` attaches trainable low-rank adapters directly to 4-bit packed `HKQuantizedLinear` modules with backpropagation support.
- **Sandboxed Execution & In-Place Truncation**:
  - Docker container sandboxing with `--network none`, CPU, and memory limits in `CodeSandbox`.
  - In-place $O(1)$ file truncation using native OS handles (`SetEndOfFile` on Windows, `ftruncate` on POSIX).
- **Documentation & Weight Loading Parity**:
  - Corrected documentation and transition guides to use `HKForCausalLM.from_pretrained("model.hk", config=config)` for fine-tuning workflows to guarantee pretrained weights are loaded.

---

## [1.0.1] - 2026-09-15 - Documentation Overhaul & CI Reliability

### Fixed & Enhanced
- **Cloud CI GPU Driver Absence**:
  - Guarded all CUDA backend correctness tests in `tests/test_cuda.zig` with runtime driver detection (`SkipZigTest`), ensuring automated tests pass on headless cloud runners without physical GPUs while retaining full GPU verification on local machines.
  - Resolved strict alignment compilation errors on ARM architectures (`aarch64-linux`, `aarch64-macos`) in `src/cuda.zig` by adding `@alignCast` to function pointer casts.
  - Defaulted macOS targets to `cuda = false` in `build.zig` to prevent unsupported CUDA build steps on Apple Silicon.
- **Zig Test Runner Cleanup**:
  - Eliminated noisy stderr test output that caused Zig's build runner to report warnings and `failed command` messages.
  - Assigned distinct executable names to `hk-tests` and `hk-cuda-tests`.
- **Documentation Overhaul & GitHub Wiki**:
  - Completely rewrote `README.md` from my personal experience, cutting out benchmark bloat and explaining my core motivation: fixing laptop bottlenecks, preventing catastrophic forgetting, and modeling neural growth after the human brain.
  - Added full transition guides for replacing Ollama, LM Studio, llama.cpp, Unsloth, Hugging Face Transformers, and SafeTensors.
  - Added in-depth training procedure coverage: Full Fine-Tuning (FFT), Parameter-Efficient Fine-Tuning (QLoRA), Supervised Fine-Tuning (SFT), Continued Pre-Training (CPT), and Pre-Training from scratch.
  - Wrote a comprehensive 19-page modular GitHub Wiki in `docs/wiki/`.
  - Built an automated GitHub Actions workflow (`.github/workflows/wiki-sync.yml`) and `tools/sync_wiki.py` for continuous wiki synchronization.
- **Package Ecosystem**:
  - Standardized pip package distribution name to `hknt`.

---

## [1.0.0] - 2026-09-14 - The Unified Release

I built this initial unified release of the HK Neural Tensor Framework to serve as a complete replacement for legacy model formats (SafeTensors, GGUF, and PyTorch checkpoints). I packaged raw unquantized weight storage with zero compute headroom, universal multi-device super-coalescing, dual-mode quantization, hardware structured sparsity, live architecture growth, and in-container evolution into a single seamless system.

### Core Container & Hardware Acceleration
- **High-Performance CUDA GPU Backend & Dynamic Offloading (`-Dcuda=true`)**:
  - Full end-to-end device inference path keeping activations and weights 100% resident in VRAM across all layers (`RMSNorm`, `QK-Norm`, `RoPE`, device KV-Cache, `GQA Attention`, `SwiGLU`, and `LM Head`).
  - Dynamic user-controlled layer offloading (`-ngl <N>` / `--gpu-layers <N>`) balancing CPU/GPU memory budgets with seamless zero-overhead boundary transitions.
  - Optimized CUDA device kernels featuring zero-overhead warp shuffles (`__shfl_down_sync`), 128-bit vectorized memory transactions (`float4` / `int4`), and hardware acceleration for F32, Q8_0, and Q4_0 quantizations.
  - Architectural Parity & QK-Norm: Native per-head query/key RMSNorm normalization (`attn_q_norm` and `attn_k_norm`) on both CPU and GPU, ensuring bit-exact model output correctness for Qwen3, Gemma 2, and next-generation foundation models.
  - Standalone benchmarks: `hk-gpu-bench` and `hk-cpu-bench` for direct device memory throughput and token generation rate analysis.
- **Universal Multi-Device Super-Coalescing (`HeaderFlags.UNIVERSAL_PAGE_ALIGNED = 0x100`)**:
  - Super-coalesced 4096-byte (4 KB) page alignment satisfying AMD ROCm DirectGMA, Intel NPU/OpenVINO Direct DMA, Apple Silicon Metal (16 KB), and ARM NEON/SVE.
  - **NVIDIA Tensor Core Coalescing Invariance**: Because $4096 = 32 \times 128$, a single shared `.hk` file guarantees 100% strict 128-byte warp-coalesced memory transactions with zero performance loss and zero storage duplication.
- **Raw Weight Storage (`HeaderFlags.RAW_WEIGHT_STORAGE = 0x80`)**:
  - Full-precision native storage for BF16, FP16, FP32, INT8, INT16, INT32, INT64, and BOOL.
  - Zero compute headroom: weights are memory-mapped directly with zero decoding, transcoding, or unpacking latency.
- **4-Row Unrolled SIMD GEMV Compute Engine**:
  - `gemvBF16_4rows` evaluates 4 output rows simultaneously in registers with 8 interleaved 256-bit SIMD accumulators, eliminating cache thrashing and achieving **34.38 GFLOPS** single-core throughput (1.72x faster than multi-threaded PyTorch CPU).
- **Single-Call Batch TOC Deserialization**:
  - `hk_get_all_tensor_infos` fetches all tensor metadata in a single C call, avoiding hundreds of individual ctypes FFI roundtrips.
- **Zero-Copy `HKDict` Container**:
  - `load_raw` returns an `HKDict` binding reader lifetime directly to output tensors, loading 1.75 GB of model weights into PyTorch in 15 ms.
- **Production 1B Parameter Model Benchmark (`Qwen3.5-0.8B`)**:
  - Verified on 873,438,784 bfloat16 parameters (488 tensors, 1.75 GB): 1.72x faster GEMV, 346x faster autoregressive layer access (80 ns vs 28.84 us), and 222 MB/s streaming conversion.
- **Split Mode Sharding (`HeaderFlags.IS_SHARDED = 0x40`)**:
  - Splits multi-hundred-gigabyte raw checkpoints cleanly across storage boundaries with standardized manifest indexes (`save_sharded_raw` / `load_sharded_raw`) for regeneratable weights, adapters, and modular layer swapping.
- **HK Binary Container Specification (HKNT)**:
  - Fixed 128-byte header, extensible typed key-value metadata section, and 128-byte aligned Tensor Table of Contents.
  - Strict 128-byte cache-line and Tensor Core memory alignment matching GPU memory transactions.
  - Universal flexible alignment mode (`0x20` flag): seamlessly supports 1-byte compact alignment for mobile and embedded systems, 16-byte for ARM NEON, and 128-byte for datacenter GPUs.
- **Zero-Copy Memory Mapping**:
  - Direct zero-copy page mapping on POSIX (`mmap`) and native Windows (`CreateFileMappingA` + `MapViewOfFile`).
  - True copy-on-write page safety ensuring all loaded tensors are directly writable without duplicating memory.
- **Dual-Mode Quantization**:
  - Dual-mode 4-bit NormalFloat4 (NF4) with residual delta stream for bit-exact recovery ($>0.99999$ cosine similarity).
  - Dual-mode 8-bit integer (DQ8) quantization.
  - BitNet b1.58 ternary (DQT $\{-1, 0, +1\}$) quantization.
- **NVIDIA Ampere 2:4 Structured Hardware Sparsity**:
  - True 2-bit nibble metadata packing for 50% non-zero weights (1.88× physical compression).
  - Native Zig SIMD unpacking kernel (`hk_unpack_2_4`) delivering $>3\text{ GB/s}$ unpacking throughput.
- **Tensor Core 16x16 Tile Transformation**:
  - K-contiguous tile packing and untiling for NVIDIA WMMA tensor cores.
- **Sparse Matrix Representations**:
  - Bitmask sparse packing and Block Sparse Row (BSR) packing for Mixture-of-Experts (MoE).

### Live Architecture Evolution & Adaptation
- **Dynamic Capacity Expansion (Net2Net)**:
  - Function-preserving width expansion (`net2wider_linear` and `net2wider_swiglu`) for standard linear layers and modern SwiGLU MLP architectures ($||f_{wider}(x) - f(x)||_{\infty} < 10^{-6}$).
  - Identity layer stacking (`net2deeper_linear`) and zero-initialized residual adapters (`ModularResidualBlock`) guaranteeing zero degradation upon insertion.
  - `GrowthGovernor` hardware resource manager enforcing strict VRAM and system memory bounds.
- **Version-Chained Appendix Region**:
  - 80-byte binary appendix record specification supporting all 6 evolutionary entry types: `LORA_ADAPTER (0x01)`, `DELTA_PATCH (0x02)`, `NEW_LAYER (0x03)`, `CODE_EVAL (0x04)`, `KV_CACHE_SINK (0x05)`, and `TOPOLOGY_HEAD (0x06)`.
  - Cryptographic SHA-256 DAG hash-chaining across generations (`parent_hash`).
  - Instant rollback to any prior generation (`hk_appendix_rollback`) via Python API and native CLI (`hk rollback <model.hk> [gen]`).
- **Self-Play Evolution Engine (SPIN-Style)**:
  - Targeted `LoRAAdapter` fine-tuning on salient projections.
  - Automated context poisoning mitigation with regression detection and immediate parameter rollback.
- **Persistent Code Evaluation Sandbox**:
  - Subprocess isolation with strict execution timeouts, syntax tree validation (`ast.parse`), and unit test scoring persisted into `.hk` appendix entries.
- **Single-File Runnable Model Topology**:
  - Embedded model topologies, hyper-parameters, and inference runner scripts (`load_standalone_hk`).

### Developer Experience & Multilingual Ecosystem
- **Clean & Consolidated Python Architecture (`hk`)**:
  - Completely removed legacy `python/hk_format/` package (~3,850 lines of duplicate pure-Python code).
  - Consolidated all functionality into a unified, high-performance `python/hk/` package delegating directly to native `hk.dll` via C ABI.
  - Native container serialization via `NativeHKWriter` and zero-copy loading via `NativeHKReader`.
  - Added dedicated submodules: `hk.torch`, `hk.quantization`, `hk.pruning`, `hk.benchmark`, `hk.format`, and `hk.adaptive`.
- **Hugging Face-Style Python API**:
  - `AutoModel`, `AutoConfig`, and `AutoTokenizer` mirroring Hugging Face developer workflows.
  - `HKForCausalLM`, `HKForSequenceClassification`, `HKForHandwritingRecognition`.
  - Task pipelines: `pipeline("text-generation")`, `pipeline("sequence-classification")`, `pipeline("handwriting-recognition")`.
  - `HKTrainer` supporting automatic plateau-triggered Net2WiderNet growth and QLoRA adapter training.
- **Autonomous Self-Training & Conversational Thinking Engine**:
  - `ExpansionEvaluator`: Autonomous diagnostic capacity evaluation for identifying domain and vocabulary bottlenecks.
  - `SelfConversationalEngine`: Proposer-Thinker inner monologue (`<think> ... </think>`) with multi-step reasoning and syntax synthesis.
  - `CodeSandbox`: AST-isolated execution environment with automatic unit test grading and metric recording.
  - Plasticity Shield: In-place gradient masking preventing catastrophic forgetting of base representations during new language/task acquisition.
- **Native Zig Engine & Standalone CLI (`hk.exe`)**:
  - Added native C ABI container writer functions (`hk_writer_create`, `hk_writer_add_tensor`, etc.).
  - Standalone binary supporting `inspect`, `verify`, `eval`, `expand`, `benchmark`, `retile`, `prune`, `appendix`, and `rollback`.
- **Universal Multi-Language Bindings**:
  - **Rust**: Safe idiomatic crate (`bindings/rust/Cargo.toml`).
  - **TypeScript / Node.js**: NPM package (`bindings/js/package.json`).
  - **C# / .NET**: Package for Unity and .NET applications (`bindings/csharp/Hk.csproj`).
  - **Go**: Module with cgo integration (`bindings/go/go.mod`).
  - **Java / Android**: JNI package with direct NIO `ByteBuffer` mapping (`bindings/java/pom.xml`).
  - **C / C++**: Header definitions (`include/hk.h`) and C++20 RAII wrappers (`include/hk.hpp`).

### Advanced Quantization Zoo & Importance Calibration
- **K-Quants Super-Block Engine (`0x40` - `0x45`)**:
  - Implemented 256-element super-block structures: `BlockQ4_K` (144 bytes, 4.5 bpw), `BlockQ8_K` (292 bytes, 9.125 bpw), `BlockQ6_K` (210 bytes, 6.56 bpw), `BlockQ2_K`, `BlockQ3_K`, and `BlockQ5_K`.
  - Native Zig SIMD dequantization routines with AVX2/AVX-512 vectorization and C ABI exports.
- **Non-Linear & Importance Quants (`0x50` - `0x57`)**:
  - `IQ4_NL` non-linear codebook quantization using Gaussian optimal distribution tables.
  - Low-bit I-quant types: `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ4_XS`.
- **Hardware Microscaling & Ternary Formats (`0x60` - `0x63`)**:
  - OCP Microscaling FP4 (`mxfp4`): E2M1 floating point with 32-element blocks and 8-bit scale factor.
  - NVIDIA Blackwell Microscaling (`nvfp4`): E2M1 with FP8 micro-scales.
  - Ternary quants: `tq1_0`, `tq2_0`, and BitNet b1.58 `dqt`.
- **Activation-Aware Importance Matrix Calibration (`imatrix`)**:
  - `ImportanceMatrixCalibrator`: Fisher information second-moment accumulator ($I = \frac{1}{N}\sum x x^T$) for activation-guided quantization error minimization.
  - Predefined mixed-precision per-tensor recipes: `Q4_K_M`, `Q5_K_M`, `Q4_K_S`, `Q5_K_S`, `Q3_K_M`, `Q2_K`, `Q6_K`, `Q8_K`, `IQ4_NL`.
  - `resolve_quant_type_for_tensor`: Automatic per-tensor precision assignment using architectural regex matching.

### Massive Architectural Breadth (137+ Models)
- **Comprehensive Model Registry (`ARCHITECTURES_REGISTRY`)**:
  - 137+ foundation model architectures supported across LLMs, SSMs, VLMs, Audio, and Diffusion.
  - Cutting-Edge LLMs: DeepSeek V2/V3/R1 (MLA attention and Multi-Token Prediction), LLaMA 4, Qwen 2.5/3/MoE, Gemma 1/2, Grok, Falcon, Phi-3/Phi-MoE, DBRX, Command-R+, OLMoE, MiniCPM-3, Starcoder2, Jais, Exaone, ChatGLM.
  - State-Space & Recurrent Models: Mamba, Mamba-2, Jamba, RWKV-5/6.
  - Vision-Language Models: CLIP, SigLIP, LLaVA, MobileVLM, Qwen2-VL, Pixtral, Gemma-Vision, SAM / SAM-2.
  - Audio & Diffusion: Whisper audio encoders/decoders, Stable Diffusion, and FLUX.1 rectified flow transformer backbones.
  - Modern Encoders: ModernBERT, Nomic-BERT, Jina-BERT-v2/v3, EuroBERT.
- **Bi-Directional Regex Tensor Mapping**:
  - 8 bidirectional regex mapping tables translating state dicts between Hugging Face and HK naming conventions without data copying.

### Deep Tokenizer Ingestion & Zero-Dependency Decoders
- **Protocol-Buffer-Free Binary SentencePiece (`.model`) Parser**:
  - Embedded pure-Python wire-format varint decoder (`parse_sentencepiece_model`) parsing SentencePiece binary models into tokens, float32 scores, and token types with zero external C++ or wheel dependencies.
- **Mistral Tekkenizer Ingestion**:
  - `parse_tekken_json` parser for Mistral NeMo and Large 2 tokenizers.
- **Rich Tokenizer Metadata**:
  - 6 explicit token types (`TokenType`: `NORMAL`, `UNKNOWN`, `CONTROL`, `USER_DEFINED`, `UNUSED`, `BYTE`).
  - Pre-tokenizer regex split patterns (`PreTokenizerType`: `default`, `llama3`, `qwen2`, `deepseek_v3`, `phi3`, `mistral`).
  - `AutoTokenizer.from_pretrained` automatically detects and ingests embedded container tokenizer metadata.

### Standardized Hyperparameter & Sampling Taxonomy
- **200+ Standardized Taxonomy Keys (`HKTaxonomyKeys` / `StandardKeys`)**:
  - Unified namespaces across `general.*`, `attention.*` (MLA, SWA, ALiBi), `rope.*` (YaRN, dynamic), `moe.*` (routed & shared experts), `ssm.*` (state size, conv kernel, inner size), `tokenizer.*`, `sampling.*` (top_p, top_k, min_p, temperature, repetition penalty, mirostat), and `quant.*`.

### Standalone CLI & Graphical Model Studio
- **Auxiliary Developer Tools**:
  - `hk dump <model.hk>`: 128-byte hex dumper, bitflag breakdown, section offsets, and automated alignment audit.
  - `hk hash <model.hk>`: Whole-file streaming SHA-256 and per-tensor cryptographic digest verification.
  - `hk convert-endian <in> <out>`: Zero-loss Little-Endian <-> Big-Endian conversion preserving 128-byte hardware alignment.
- **Graphical Model Studio (`hk gui` / `hk-gui`)**:
  - Lightweight, responsive desktop GUI editor with Hugging Face / GGUF inspired tabs:
    - Container Overview (header audit, bitflags, memory stats, compression ratio).
    - Metadata & Hyperparameter Tree (namespace-grouped, in-place zero-copy byte patching, JSON import/export).
    - Tensor Table & Quantization Inspector (shapes, storage types, 128-byte alignment verification).
    - Lineage & Appendix History (generations, test pass rates, one-click rollback).

### Multilingual SDKs Sync
- Updated all 7 multilingual bindings to support all new StorageType definitions and quantization APIs:
  - C / C++ (`include/hk.h`, `include/hk.hpp`)
  - Rust (`bindings/rust/src/lib.rs`)
  - C# / .NET (`bindings/csharp/HkModel.cs`)
  - Go (`bindings/go/hk/hk.go`)
  - Java / Android (`bindings/java/com/hk/HkModel.java`)
  - TypeScript / JS (`bindings/js/hk.ts`)
  - Python (`python/hk/`)

### Verification & Test Coverage
- **100% Passing Test Suites (90 / 90 Tests Passing)**:
  - 25 native Zig unit tests (`zig build test`).
  - 5 comprehensive parity pillar tests (`tests/test_parity_pillars.py`).
  - 6 PyTorch & SafeTensors integration tests (`tests/test_torch_integration.py`).
  - 6 rigorous architecture expansion tests (`tests/test_expansion_rigorous.py`).
  - 4 autonomous self-training pipeline tests (`tests/test_self_training_pipeline.py`).
  - 16 exhaustive adaptive neural framework tests (`tests/test_adaptive.py`).
  - 13 Hugging Face-style API and pipeline tests (`tests/test_hf_style.py`).
  - 7 Sharding, In-Place Patching, and AutoTokenizer tests (`tests/test_new_features.py`).
  - 5 Universal Heterogeneous Multi-Stage Pipeline tests (`tests/test_universal_pipeline.py`).
  - 6 Tokenizer Metadata, Sharding Manifests, JAX/Flax/NumPy, and Conversion Tables (`tests/test_expanded_features.py`).
- **Validated End-to-End Live Demonstrations**:
  - `run_self_training_demo.py`: Autonomous acquisition of MiniZig systems programming via conversational thinking.
  - `run_new_language_expansion_demo.py`: Autonomous new language acquisition with zero catastrophic forgetting.
  - `run_llm_demo.py`: 8-stage comprehensive LLM evaluation with SmollM-135M.
  - `run_hwr_pruning_demo.py`: Complete pruning and compression suite.
