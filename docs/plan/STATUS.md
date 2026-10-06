# HK — Plan vs. Reality (STATUS)

**Measured against:** commit `43cf891` (v1.1.1 + fixes) **plus the uncommitted working tree** as of 2026-10-06.
**Target:** [MASTER_PLAN.md](MASTER_PLAN.md) (§N below = plan section).
**Rule for this file:** every row is backed by code or a measurement someone looked at. Where a claim
was *not* verified, the row says `unverified`. Update a row in the same change that changes its truth.
Upstream docs (`docs/wiki/Compatibility.md`, `Hardware-and-Kernels.md`) are the source of truth for
user-facing claims; if this file and the code disagree, **the code wins** — fix this file.

## Legend

| Tag | Meaning |
|:--|:--|
| **DONE** | Implemented, tested, and (for performance claims) measured |
| **PARTIAL** | Works for a subset; the gap is named in the row |
| **STORAGE-ONLY** | The container can hold it and the reader can decode to f32; the engine cannot run it |
| **PROTOTYPE** | Python-side, unit-tested on small models only; no large-scale validation |
| **LEGACY** | Exists in tree but is not on the live `hk` engine path |
| **UNTESTED-HW** | Compiles/cross-compiles; has never run on the target hardware |
| **NOT STARTED** | No code found (grep'd) |

## Baseline measurements (2026-10-06, this machine: Ryzen 7 7445HS, RTX 3050 6GB Laptop, Linux 7.2, Zig 0.16.0)

- `zig build test` (Debug, kernels=fallback): **17/17 steps, 138/138 tests passed** in ~4 s, including the uncommitted integrity/transaction work.
  The `hk-cuda-tests` (7) pass; **not checked** whether they exercised the real GPU or skipped.
- Counts: ~158 Zig `test` blocks, ~100 Python `def test_`. **Python suite was not run in this session** (`numpy`/`torch` not installed in the system Python used here) — treat Python status as `unverified this session`.
- Performance (from `benchmarks/results.md`, CPU, 6 threads, 2026-10-05): prefill 91–240 % of llama.cpp (faster on 8/10 models), decode 92–100 %, private memory 5–9× lower. Vulkan GPU: ~25–40 % of llama.cpp prefill, 65–80 % decode, one NVIDIA card only.

## Two kernel stacks — read before touching any kernel

| Stack | Path | Used by | Notes |
|:--|:--|:--|:--|
| **Engine** (live) | `src/engine/*` → `src/quant/*` + `src/kernels/*` (ISA-dispatched, `kernels_impl.zig` compiled per level) → `src/vk/*` | `hk run/chat/serve`, Zig tests | Integer-dot-product design, register-tiled prefill, split-KV decode attention |
| **Legacy** | `src/c_api.zig` → `src/tensor_ops.zig` + `src/quantization.zig` | Python `native.py`, all language bindings, converters, `reader.dequantizeToF32` | Holds the v1.1.0 optimizations (`gemmF32` zero-skip, RoPE angle table, FP8 E4M3 LUT, 4-row tiled GEMV). Not what the engine executes. |
| **Standalone CUDA** | `src/cuda.zig`, `src/cuda/*.cu`, `*.ptx` | `tools/hk_gpu_bench.zig`, `tests/test_cuda.zig` only | **Not wired into the engine or the C API.** Docs say "no CUDA-in-engine". |

Plan items that say "retain the previous optimization" (§9.2, §15, §17, §20) mostly refer to the *legacy* stack. Whether they should be re-expressed in the engine stack is an open design question — see HANDOFF.

## Matrix

### Format and container (§3–§8)

| § | Item | Status | Evidence / gap |
|:--|:--|:--|:--|
| 3 | Dense dtypes (f64…bool, fp8) | PARTIAL | All in `format.StorageType`. Engine runs only f32/f16/bf16. FP8 E4M3 has comptime LUT (legacy stack); **E5M2 LUT not found** |
| 3 | Quantized: legacy, K, I-quants, TQ, MXFP4, NVFP4 | **DONE (CPU)** | 26 formats read; decoder + dot kernel + batch kernel each; decoders tested vs reference `gguf` package (`tests/test_quant.zig`). GPU decodes only f32/f16/bf16, Q4_0..Q8_0, K-quants, IQ4_NL/XS |
| 3 | HK-native DQ4/DQ8/DQ6/DQ12/DQT, NF4 | STORAGE-ONLY | Quantizers in `python/hk/quantization.py` (NF4, DQ8, DQT), `src/nf4.zig`. Reader decodes dq4/dq8 only. Engine refuses. No NF4+residual runtime |
| 3 | Sparse storage (bitmask, CSR, BSR, 2:4, sparse f16) | STORAGE-ONLY | `src/sparsity.zig`; reader decodes to f32; engine has no sparse kernels |
| 3 | Virtual: null_ref, shared_ref | DONE | writer handles both (tied weights) |
| 3 | Virtual: lora_ref | STORAGE-ONLY | enum value exists; engine ignores appendix/adapters |
| 3 | Virtual: delta_ref, external_ref | NOT STARTED | not in `StorageType` |
| 4 | Container checksum | **PARTIAL — uncommitted** | `src/integrity.zig`: SHA-256 over whole file, header flag bit 9, sealed after every writer/mutation; `hk verify` reports it; Python mirror in `adaptive/appendix.py`. Legacy files without it stay readable (`legacy_missing`) |
| 4 | Appendix record integrity | PARTIAL — uncommitted | CRC-32 per record payload; `0` = legacy record |
| 4 | Reader bounds hardening | PARTIAL — uncommitted | overflow-safe section/tensor range checks in `reader.zig` (`checkedEnd`, `validateTensorRanges`) |
| 4 | Version negotiation, capability negotiation, optional sections | PARTIAL | reader accepts `version_major == 1`; header feature flags exist; no negotiation protocol |
| 4 | Per-tensor hash stored in file | NOT STARTED | `hk hash` computes per-tensor SHA-256 on demand only |
| 4 | Signatures, trusted publishers, signed manifests | NOT STARTED | |
| 4 | Compression | UNVERIFIED | `COMPRESSED` appendix flag bit is defined (`format.zig:205`); no compression code found |
| 5 | Crash-safe metadata edit | **PARTIAL — uncommitted** | `src/transaction.zig`: journal of the pre-edit prefix (`.hkmeta-journal`), rollback on next mutation; `hk repair` |
| 5 | Atomic appendix writes, atomic full-file writes | NOT STARTED | `appendRecordToFile` is in-place; `HKWriter.write` creates the destination directly (no temp+rename) |
| 5 | `hk compact`, `hk fsck` | NOT STARTED | |
| 6 | Layout metadata | PARTIAL | `TileLayout` enum 0x00–0x07 and `hk retile`; **engine ignores pre-tiled layouts** (repacks per-thread at runtime). No 128x\*, no per-backend layouts, no multi-representation per logical tensor |
| 7 | mmap — Linux/POSIX | DONE | real `mmap(2)` in `platform.zig` |
| 7 | mmap — Windows | PARTIAL, unverified | `CreateFileMappingA`/`MapViewOfFile` present; never run on Windows |
| 7 | huge pages, NUMA, madvise, readahead, large pages, macOS/Android/iOS specifics | NOT STARTED | no `madvise`/`MAP_HUGETLB`/NUMA in `platform.zig` or engine |
| 8 | Page-aligned payloads (128 B / 4 K / 16 K / 64 K) | DONE (as metadata) | alignment presets; header says alignment is a convenience, not an accelerator claim |
| 8 | Device-importable buffers / file→device zero-copy | NOT STARTED | |

### Compute: kernels and SIMD (§9–§21, §56)

| § | Item | Status | Evidence / gap |
|:--|:--|:--|:--|
| 9 | Register-tiled prefill GEMM (all formats) | DONE (CPU) | `src/quant/gemm.zig`; weights stay mapped; per-thread tile repack |
| 9 | Tiled GEMV / decode | DONE (CPU) | fused QKV and gate/up, prefetch, futex-skipping pool, split-KV decode attention |
| 9 | Sparse / Tensor Core / NPU GEMM | NOT STARTED | |
| 10 | x86: generic, AVX2, AVX-VNNI, AVX512+VNNI | DONE / PARTIAL | `avx512`, `avx2vnni`, `avx2`, `generic`; run here: avx512/avx2/generic. **`avx2vnni` never run on real silicon.** No dedicated SSE4.2 level, AVX512-BF16, AMX |
| 11 | ARM: NEON, dotprod | UNTESTED-HW | cross-compiled Linux+macOS; portable paths tested; **never run on ARM** |
| 11 | ARM FP16/BF16/SVE/SVE2 | NOT STARTED | |
| 12–13 | RISC-V, LoongArch, POWER, s390x | NOT STARTED | `variants.zig` covers x86/aarch64; others get `generic` only |
| 15 | RoPE: angle tables, linear/llama3/yarn | DONE | tables in legacy stack (22× microbench, v1.1.0); engine implements none/linear/llama3/yarn (YaRN blend bug fixed in 1.1.1) |
| 15 | NTK/dynamic NTK, SIMD sin/cos, GPU RoPE fused w/ QK-norm | PARTIAL | GPU RoPE exists in Vulkan (`rope_kv.comp`); NTK/dynamic NTK not found |
| 16 | CPU attention: online softmax, KV tiling, split-KV | DONE | `engine/attention*.zig` |
| 16 | GPU attention (FlashAttention-style, Tensor Core) | PARTIAL | `shaders/attention.comp` exists; no cooperative-matrix path |
| 16 | MHA/GQA | DONE | Sliding-window, local/global, MQA-specific, MLA: NOT STARTED |
| 17 | Softmax: max-subtract, underflow prune | DONE (legacy stack) | no numerical-error test suite for approximations found |
| 18–19 | RMSNorm, QK-norm, SiLU, GELU(tanh) | DONE (CPU+Vulkan) | LayerNorm/GroupNorm, QuickGELU, GeGLU, ReGLU: legacy `layerNormF32` only |
| 20 | FP8 LUT | PARTIAL | E4M3 comptime LUT (legacy). E5M2, vectorized/GPU/TC FP8, scale metadata: NOT STARTED |
| 21 | MXFP4 / NVFP4 | PARTIAL | CPU kernels + decode done, Vulkan **not yet**; CUDA/ROCm/Metal none |
| 56 | CPU backend matrix | PARTIAL | see rows 10–13 |

### Quantization, sparsity, pruning (§22–§27)

| § | Item | Status | Evidence / gap |
|:--|:--|:--|:--|
| 22 | Calibration | PROTOTYPE | `ImportanceMatrixCalibrator` (`quantization.py`), imatrix hook on `quantize_q4_k` |
| 22 | Layer sensitivity, auto format selection, mixed-format generation | NOT STARTED | `resolve_quant_type_for_tensor` is a recipe lookup, not a search |
| 23 | Dual-mode (base + residual) | STORAGE-ONLY / PROTOTYPE | quantize/dequantize pairs for NF4 and DQ8 in Python; **no runtime residual policy; `hk quantize` does not exist** |
| 24 | Auto mixed precision planner | NOT STARTED | `offload.py` plans memory placement only |
| 25 | Pruning: magnitude, Wanda, structured L2, block, 2:4, recovery fine-tune | PROTOTYPE | `python/hk/pruning.py`; `hk prune` (magnitude) native |
| 25 | Activation/gradient-aware, head/expert pruning, schedules | NOT STARTED | `LayerSparsitySchedule` exists as a dataclass only |
| 26–27 | 2:4 / sparse **execution** | NOT STARTED | storage only; docs state 2:4 does not speed inference. No auto sparse-vs-dense choice |

### Research stack (§28–§40, §105–§118)

| § | Item | Status | Evidence / gap |
|:--|:--|:--|:--|
| 28 | Net2Wider (SwiGLU, linear), vocab expansion | PROTOTYPE | Zig `growth.zig` (+ `hk expand --vocab/--width`), Python `adaptive/growth.py` |
| 28 | Net2Deeper | PROTOTYPE (Python, linear block only) | no Zig depth growth, no layer insertion into `.hk` |
| 28 | Head / KV-head / expert expansion | NOT STARTED | |
| 28 | `hk diagnose-growth`, `plan-growth`, `apply-growth` | NOT STARTED | |
| 29 | Function-preservation verification | PARTIAL | tests on small models (`test_expansion_rigorous.py`); no logits/cosine/token-agreement gate built into `hk expand` |
| 30 | Growth during training | PROTOTYPE | `GrowthGovernor`, `ExpansionEvaluator` (heuristics over numbers you pass in). Plateau detector: unverified |
| 31 | Plasticity isolation | PROTOTYPE | `protect_base_capacity`; no retention/forgetting evaluation |
| 32 | Self-conversation | PROTOTYPE | propose/think/test/reflect around caller-supplied functions. Roles critic/verifier/judge: no |
| 33 | Code sandbox | **PARTIAL — plan violation** | subprocess = timeout only; optional Docker (`--network none`, mem/cpu). **Silently falls back to the unsandboxed subprocess if Docker is missing** (`code_eval.py`, `is_docker_active`) — §33 says never do this. Zig `NativeSandbox` (`src/sandbox.zig`) **does not enforce its timeout**: `std.process.run` runs the child to completion, then `timed_out` is merely `elapsed >= timeout_ms`. A hanging child hangs the caller. It is also not referenced by `c_api.zig` or the CLI. No seccomp/namespaces/microVM/resource limits anywhere |
| 34 | Self-training pipeline | PROTOTYPE | no curriculum generation, difficulty estimation, failure mining; no demonstrated improvement |
| 35 | Self-play / SPIN | PROTOTYPE | `SPINLoss`, `LoRAAdapter`, `SelfPlayEvolutionEngine`; no reference snapshots/KL/anti-collapse eval |
| 36 | Autonomous expansion loop | PROTOTYPE | hooks only; no accept/reject on evaluation proven |
| 40 | LoRA/QLoRA training | PROTOTYPE | `HKTrainer`, `LoRAAdapter`, "true QLoRA" (v1.0.2). **No inference-time adapter application in the engine** |
| 105–118 | Per-technique evaluation, ablations, experiment tracking, reproducibility manifests | NOT STARTED | wiki pages state no results are claimed |

### Lineage (§37–§39)

| § | Item | Status | Evidence / gap |
|:--|:--|:--|:--|
| 37 | Hash chain, rollback, record types | DONE (linear) | appendix records, parent-hash chain, `hk appendix`, `hk rollback`; Zig + Python writers |
| 37 | Signatures, branching, merging, model identities | NOT STARTED | |
| 37 | `hk history/diff/branch/merge/verify-lineage/sign/verify-signature` | NOT STARTED | only `appendix` and `rollback` exist |
| 38–39 | Version graph; tensor/sparse/quant/metadata deltas; base+adapter execution | NOT STARTED | engine does not apply appendix records |

### Training, data, distillation (§41–§45, §115–§116)

| § | Item | Status | Evidence / gap |
|:--|:--|:--|:--|
| 41 | `HKTrainer` (PyTorch) | PROTOTYPE | `python/hk/trainer.py`. No native trainer (`NativeHKTrainer` not found) |
| 42–43 | Mixed precision, checkpointing, parallelism, offload, 8-bit/paged optimizers, schedulers | NOT STARTED / unverified | rely on PyTorch; nothing HK-native |
| 44 | Distillation | NOT STARTED | |
| 45 | Dataset system | NOT STARTED | |
| 115–116 | Distributed training / evolution | NOT STARTED | |

### Models and architectures (§46–§55)

| § | Item | Status | Evidence / gap |
|:--|:--|:--|:--|
| 46 | `ArchitectureDescriptor/Parser/Validator/Executor` | NOT STARTED | engine has a 3-value `Arch` enum (`llama`, `qwen2`, `qwen3`) in `engine/config.zig`; behavior is branched in code |
| 47 | "137+ architectures" | **MISLEADING AS A COVERAGE NUMBER** | `python/hk/models.py` registry = **169 name aliases → 79 canonical names**. It drives *Python conversion/tensor-name mapping*. **The Zig engine executes 3 families** (Llama/Mistral-no-SWA, Qwen2, Qwen3). Don't quote 137 as "supported" |
| 48 | GQA | DONE | everything else (sliding window, MLA, linear attn, SSM, hybrid): NOT STARTED |
| 49 | MoE | NOT STARTED (runtime) | Python mapper has `pack/unpack_moe_experts` only |
| 50 | Long context | PARTIAL | segmented f16 KV (256/seg, key tiles of 64), rope scaling; no paging, compression, quantization, SWA, chunked-prefill policy |
| 51 | Speculative decoding | NOT STARTED | |
| 52 | Sampling | PARTIAL | greedy, temperature, top-k, top-p, min-p, repeat/frequency/presence penalties, logit bias, seeded; **missing** typical, mirostat, custom processors, GPU sampling |
| 53 | Grammar / JSON schema / tool-call parsing | NOT STARTED | |
| 54–55 | Vision / audio / video, multimodal format | NOT STARTED | loader refuses with the architecture name |
| 126–130 | Embeddings, rerank, speech, CV, pipelines | NOT STARTED (native) | Python `pipeline.py` / `composite.py` have wrappers (`HKWhisperModel`, `HKDistilBertModel`) — status unverified |

### Backends (§57–§75, §137)

| § | Item | Status | Evidence / gap |
|:--|:--|:--|:--|
| 57 | CUDA | **LEGACY / standalone** | driver-API wrapper loaded via `dlopen`, embedded PTX, own tests and bench; **not in engine**. Commit history has conflicting remarks ("untested", "97 tok/s") — neither reproduced in current docs. No Tensor Cores/graphs/FP8 |
| 60 | Vulkan | PARTIAL | whole-model-on-one-device, SPIR-V checked in, `-ngl`; **only one NVIDIA card tested**; no cooperative matrices, no async queues/timeline semaphores |
| 58, 59, 61–63 | ROCm, Metal, DirectML/D3D12, Intel XPU/oneAPI/OpenVINO, Qualcomm QNN | NOT STARTED | Metal docs: would need Apple hardware |
| 64–66 | Android, iOS, macOS | NOT STARTED / cross-compile only | macOS and Windows targets cross-compile in CI; never run. Java JNI glue exists (`bindings/java/jni`), not an Android package |
| 67–69 | Linux / Windows / BSD | PARTIAL | Linux is the only run platform; Windows/macOS cross-compiled |
| 70–71 | WASM/WebGPU, embedded | NOT STARTED | |
| 72 | Backend auto-selection | PARTIAL | CPU vs Vulkan by `-ngl`/`HK_GPU` and free-VRAM check; no capability/benchmark-driven selection |
| 73 | Heterogeneous / partial offload | PARTIAL | Python `offload.py` planner (torch-based); **Vulkan engine does not support partial layer offload** |
| 74–75 | Multi-GPU, distributed inference | NOT STARTED | |
| 137 | Automatic fallback | PARTIAL | GPU→CPU fallback with a printed reason when format/VRAM doesn't fit; no tiered fallback |

### Runtime services (§76–§84, §131–§136, §141–§142)

| § | Item | Status | Evidence / gap |
|:--|:--|:--|:--|
| 76 | Native HK protocol | NOT STARTED | |
| 77 | Server | PARTIAL | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`; SSE streaming; `--api-key`; warns on non-loopback bind without key. No Anthropic adapter, WebSocket, HTTP/2, rate limits, tracing, request cancellation: unverified |
| 78 | Scheduling | PARTIAL | `server/scheduler.zig` batches conversations across `--slots`; no priorities/deadlines/quotas |
| 79 | Prefix/prompt cache | PARTIAL | shared prompt cache exists; not content-addressed/persistent/LRU+LFU/multi-tenant |
| 80 | KV: segmented | DONE | paged, quantized, compressed, evicted, offloaded: NOT STARTED |
| 81 | Power/thermal | NOT STARTED | |
| 82 | Kernel toolchain | PARTIAL | GLSL→SPIR-V via `shaders/build.sh` (checked in), PTX embedded; no cache/specialization pipeline |
| 83 | Autotuning | NOT STARTED | thread count and tile sizes are fixed heuristics |
| 84 | `hk profile` | NOT STARTED | have `hk benchmark`, `hk hardware-profile`, `hk-kernels`, `hk-compare` |
| 131–136 | Graph runtime, HK IR, fusion, liveness, recompute | NOT STARTED | forward pass is hand-written; fusion is manual (QKV, gate/up). A commit message mentions a "zgc AOT graph memory compiler" — **no `zgc` code exists in the tree** |
| 141–142 | Startup strategies, persistent GPU cache | NOT STARTED | lazy mapping is implicit (mmap) |

### Tooling, SDKs, distribution (§85–§104, §119–§125, §143–§148)

| § | Item | Status | Evidence / gap |
|:--|:--|:--|:--|
| 85 | Benchmark suite | PARTIAL | `hk-compare` vs llama.cpp (CPU and GPU), reproducible; **no MLX/ONNX/vLLM/Ollama/Transformers**, no long-context/batching/power |
| 87–89 | Python native path | PARTIAL | `NativeHKEngine`, `NativeHKTokenizer`, torch integration present; **`NativeHKTensor/Model/Trainer` not found; no DLPack** |
| 90, 95 | Conversion | PARTIAL | safetensors→HK, GGUF→HK (streaming), HK→safetensors, HK→GGUF v3, Python PyTorch loaders. No ONNX, no HK→PyTorch export beyond Python loaders; tokenizer.json importer is byte-level BPE only |
| 91 | SDKs | PARTIAL | C, C++, Rust, Go, C#, Java, TypeScript, Python have tested bindings (`tests/bindings/run.sh`). **Kotlin, Swift, Obj-C, plain JS (separate) NOT STARTED.** No binding exposes full inference+streaming parity yet (verify per binding) |
| 92 | Packaging | PARTIAL | PyPI `hknt`, npm `hkntf`. crates.io/NuGet/Maven/SPM/vcpkg/Conan/Homebrew/winget not published |
| 93 | `hk pull` | PARTIAL | resume, retries, redirect credential-drop, Hub SHA-256, GGUF + sharded safetensors, `search/list/rm`. **Split GGUF not downloaded**; no parallel download, no registry/S3, no dedupe; interrupted *conversion* restarts |
| 94 | Registry | NOT STARTED | |
| 96 | Sharding | PARTIAL | header `split_index/split_count`, Python `ShardedHKFile`; no lazy/parallel/distributed loading verified |
| 97 | Editor GUI | PROTOTYPE | `tools/hk_editor_gui.py`, `python/hk/gui.py` |
| 98 | Tensor inspection | PARTIAL | `inspect`, `dump`, `hash`, `eval`, `metadata`; no `tensor list/info/stats/diff/export` |
| 99 | Validation | PARTIAL | `hk verify` (+checksum, uncommitted), per-tensor size/shape checks at load, HF/llama.cpp parity (cosine 1.000000 on SmolLM2 f16 and Qwen3 BF16) |
| 100 | Security | PARTIAL | bounds hardening (uncommitted), `SECURITY.md`; server API key. **`SECURITY.md` supported-versions table still says 1.0.x** (stale). No resource-limit/decompression framework, no signed packages |
| 101 | Fuzzing | NOT STARTED | only Zig's scaffold `test "fuzz example"` in `src/main.zig` |
| 102–104 | Cross-backend reference tests; eval suite | PARTIAL | CPU↔Vulkan logits cosine ≥ 0.9994; perplexity vs llama.cpp; no MMLU/code/math/long-context/structured-output suites |
| 119–123 | Mobile, packaging `.hkpack`, UI | NOT STARTED | |
| 124 | CLI | PARTIAL | present: pull search list rm serve run chat tokenize detokenize convert-safetensors convert-gguf inspect dump hash export convert-endian gui verify repair eval expand benchmark retile prune appendix rollback metadata hardware-profile. **Missing:** quantize, sparse, import, compact, fsck, history, diff, merge, sign, profile, model, tokenizer(group), distributed, worker, cluster, diagnose-growth |
| 125 | Server API breadth | PARTIAL | see §77; `/v1/embeddings`, `/rerank`, `/audio`, `/images`, `/responses` NOT STARTED |
| 138 | CI matrix | PARTIAL | `ci.yml`: Zig tests on ubuntu/windows/macos (+ x86_64_v3 / x86_64 forced levels, cross-compile matrix), Python tests ubuntu/windows × 3.10–3.12, bindings job. **No GPU, ARM, or Android/iOS hardware in CI** |
| 143 | Local-first | DONE (by design) | no telemetry code found; fully offline once model is local. Encrypted models, audit logs: NOT STARTED |
| 144 | License/provenance metadata | PARTIAL | appendix + Python experiment hooks; no standard schema |
| 146 | Extension system | NOT STARTED | |
| 147 | Stable C ABI v1 | PARTIAL | `include/hk.h` with ~133 exports; **no ABI version query, no struct-size negotiation, no capability discovery** found |
| 148 | Compatibility contract per release | PARTIAL | `docs/wiki/Compatibility.md` is hand-maintained |

## Milestone checklist (plan §150) — current truth

| Milestone | Done | Not done |
|:--|:--|:--|
| **2.0 Unified Runtime** | CPU backend largely finalized for 3 arch families; Vulkan runs under `-ngl` | backend abstraction (engine calls `gpu_mod` directly), CUDA engine/prefill/decode/attention, unified memory planner, auto device selection |
| **2.1 HW Acceleration** | — | everything (no Tensor Cores, sparse TC, ROCm, Metal, coop matrices, XPU, DirectML, mobile) |
| **2.2 Model Compat** | Llama/Mistral(no SWA)/Qwen2/Qwen3 execute | Gemma, Phi, SWA, MoE, SSM, multimodal, descriptor-driven registry |
| **2.3 Server Runtime** | slot scheduler, shared prompt cache | continuous batching proper, paged KV, speculative decoding, distributed, multi-GPU |
| **2.4 Universal Format** | checksum + metadata journal (uncommitted) | v2 spec, signatures, backend layouts, external refs, registry, atomic appendix writes |
| **2.5 Training** | PyTorch `HKTrainer`, LoRA/QLoRA (prototype) | native/distributed training, distillation, QAT, pruning-aware training |
| **2.6 Adaptive Research** | Net2Wider/vocab (prototype), partial plasticity | Net2Deeper in Zig, autonomous growth validated, model evolution loop, branching/merging |
| **2.7 Self-Improvement** | components exist | secure execution, validated loops, curriculum, failure mining, regression rollback proven |
| **3.0 Universal Runtime** | Linux desktop CPU/Vulkan | everything else |

## Known doc/code discrepancies to fix (found while building this file)

1. `hk` CLI banner prints **v1.2.0** (`tools/hk_cli.zig` `printUsage`) while `build.zig.zon` and `pyproject.toml` say **1.1.1**.
2. `SECURITY.md` lists **1.0.x** as the supported version.
3. `CONTRIBUTING.md` describes an older layout (`tensor_ops.zig` as "the kernels") and says Python **3.10+** while `pyproject.toml` says `>=3.9`.
4. `docs/wiki/Architecture-and-Ideology.md` is accurate; the commit message of `623718d` ("97 tok/s" CUDA) is not corroborated by current docs.
5. `python/hk/models.py` docstring says "137+ distinct model architectures" — it is 79 canonical names (169 aliases), conversion-mapping only.
6. `build.zig` defaults `-Dcuda` to **on** for non-macOS targets, though CUDA is not on the engine path (it builds `hk-gpu-bench` and `test-cuda` and links `dl`).
