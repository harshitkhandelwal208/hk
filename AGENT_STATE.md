# HK Neural Tensor Framework — Agent State & Implementation Tracker

## Session Info
- **Project**: HK Neural Tensor Framework
- **Active Branch**: `feat/hk-2.1-rocm`
- **Status**: ACTIVE — 100% COMPLETE FOR MILESTONES 2.1 THROUGH 3.0 (UNIVERSAL NEURAL RUNTIME)
- **Target Plan**: `docs/plan/MASTER_PLAN.md` (Milestones 2.1 - 3.0 Universal Runtime)
- **Last Verified Test**: 250 passed, 7 skipped (ReleaseFast, Debug, x86_64_v3, x86_64; 26/26 e2e; 8/8 bindings suites)

## Milestone 2.1: Hardware Acceleration & Zero-Dependency Python (100% COMPLETE)
- [x] Abstracted inline-PTX in `src/cuda_device.zig` behind per-architecture primitives (`src/gpu_prims.zig`).
- [x] ROCm HSA code object compilation (`gfx10_3_generic`, `gfx11_generic`, `gfx12_generic`) in `build.zig`.
- [x] Dynamic HIP runtime loader with HSA code object embedding (`src/hip.zig`).
- [x] Implemented WMMA (`v_wmma_f32_16x16x16_f16`) matrix core kernels for prompt processing across all 26 formats.
- [x] GFX12 split barriers (`llvm.amdgcn.s.barrier`) and discrete memory wait counters in `src/gpu_prims.zig`.
- [x] HIP stream graph capture API in `src/hip.zig`.
- [x] Zero-dependency Python distribution (`dependencies = []`, plan §1.6).
- [x] HK's own tensor and autograd core `hk.tensor` (plan §161) with reverse-mode tape, buffer protocol, and native C ABI ops.
- [x] Continuous synchronization with `docs/plan/HANDOFF.md` and `docs/plan/STATUS.md`.

## Milestone 2.2: Model Compatibility & Architecture Registry (100% COMPLETE)
- [x] Native Architecture Registry (`src/models/registry.zig`):
  - Added comprehensive descriptors for LLaMA, Mistral, Gemma, Gemma 2, Qwen 2, Qwen 3, Phi 3, DeepSeek.
  - Parameterized canonical families, default attention types, normalization types, activation functions (SiLU vs GeGLU), and vocabulary metadata.
- [x] Multi-Model Execution & Memory Planner (`src/engine/planner.zig`):
  - Validated Multi-Head (MHA), Grouped-Query (GQA), and Multi-Query Attention (MQA) parameter invariants (`n_heads % n_kv_heads == 0`).
  - Validated RoPE scaling invariants for linear, LLaMA 3 frequency band interpolation, and YaRN scaling bounds.
  - Validated vocabulary mapping and token indexing ranges against vocabulary table bounds.
  - Calculated layer-by-layer activation buffers, KV cache footprints, and peak memory requirements.
- [x] Memory-Aware Placement Planner & Offloading Engine (`src/backend/select.zig`):
  - Defined `PlacementPolicy` (`cpu_only`, `whole_device`, `prefix_layers`).
  - Added granular `MemoryBreakdown` and query methods `canFit()`, `layerTarget()`, and `describeBreakdown()`.
  - Added streaming prefill chunking (`chunk_size`) and `recomputation_advised` flag.
- [x] Vulkan Cooperative Matrices (`VK_KHR_cooperative_matrix`):
  - Verified SPIR-V compute shaders (`shaders/mm_cm.comp`, `shaders/attention_cm.comp`) in `src/vk/spv/` with subgroup 16x16x16 matrix products.
  - Verified numerical parity against reference CPU implementations.

## Milestone 2.3: Paged KV Cache & Attention Optimization (100% COMPLETE)
- [x] Paged Block Allocation (`src/engine/kv_cache.zig` & `src/engine/kv.zig`):
  - Implemented `PagedBlock`, `PagedBlockPool`, and `PagedKVCache` to eliminate memory fragmentation.
  - Implemented copy-on-write sequence forking (`fork()`) for prefix caching and shared prompt state reuse.
  - Implemented chunked prefill (`appendChunk()`) for bounded memory usage during long context ingest.
  - Implemented sliding-window attention start tracking (`activeWindowStart()`).
- [x] GPU FlashAttention Compute Shaders:
  - Implemented fused QK dot product, online softmax with running max/sum, and cross-subgroup shared memory value reduction (`src/vk/shaders/flash_attention.comp` and `shaders/flash_attention.comp`).
  - Compiled SPIR-V binary (`src/vk/spv/flash_attention.spv`) and integrated into `shaders/build.sh`.
- [x] Speculative Decoding Engine (`src/engine/speculative.zig`, plan §51):
  - Implemented `SpeculativeVerifier` using exact Leviathan et al. rejection sampling and residual distribution resampling.
  - Implemented `DynamicDraftScheduler` adapting draft count based on empirical acceptance rate EMA.
  - Supported execution topologies: `same_device`, `cpu_draft_gpu_target`, `npu_draft_gpu_target`, `small_model_large_model`.
  - Zero-heap allocation in verification loops with bounded preallocated buffers.
- [x] Continuous Verification:
  - Passed `zig build test` (228/235 pass, 7 skipped ROCm device tests).
  - Passed `zig build test -Doptimize=ReleaseFast`.
  - Passed `zig build test-e2e`.
  - Passed `tests/bindings/run.sh` (8/8 language bindings pass).
  - Passed `python3 tests/test_zero_dependency.py`.
  - Passed `python3 docs/graph/gen_graph.py --check` (0 errors).

## Milestone 2.4: Universal Model Format & Model Registry (100% COMPLETE)
- [x] Robust V2 Format Specification (`src/format.zig`, `src/tensor_toc.zig`, `src/reader.zig`, `src/writer.zig`):
  - Added `VERSION_MAJOR_V2` (2) and `HAS_V2_EXTENSIONS` flag for backward-compatible version negotiation.
  - Added `HAS_PER_TENSOR_HASH` flag and 32-byte cryptographic SHA-256 hash per tensor in `TensorEntry` and TOC v2 format.
  - Added virtual reference storage types `delta_ref` (0x33) and `external_ref` (0x34).
  - Added layout metadata `tile_128x128` (0x08) and `backend_layout` (0x09) with hardware backend layout identifier.
  - Validated fail-closed bounds checking and backward compatibility: V1 files deserialize seamlessly.
- [x] Crash-Safe Mutation & Atomic Writes:
  - Implemented `HKWriter.writeAtomic` using temporary `.partial` file, checksum sealing, and atomic POSIX `rename`.
- [x] Universal Model Registry & External Shard Resolver (`src/registry.zig`, plan §94, §150):
  - Implemented `ModelIdentity` (namespace, name, version, commit) and URI formatting.
  - Implemented `ModelRegistry` with unique registration, architecture and storage indexing, and query methods.
  - Implemented `resolveExternalReference` with directory traversal protection against hostile relative paths.
  - Implemented `verifyEntry` validating container SHA-256 and Ed25519 publisher signatures.

## Milestone 2.7: Self-Improvement Research, Hardened Sandbox & SPIN (100% COMPLETE)
- [x] Hardened Native Execution Sandbox (`src/sandbox.zig`, plan §33, §150):
  - Hard timeout enforcement via `std.process.run` with process termination, verified killing hanging subprocesses with exit code 124 in 50ms without hanging the caller.
  - Memory output buffer limits (`max_output_bytes`), enforcing process termination with exit code 137 on stream overflow.
  - Fail-closed non-silent Docker isolation in `python/hk/adaptive/code_eval.py`: prohibited silent fallback to unsandboxed host process.
- [x] Self-Play Fine-Tuning & Preference Optimization (`src/self_play.zig`, plan §35, §150):
  - Exact SPIN (Self-Play Fine-Tuning) logistic preference loss and analytical policy gradients for winning and losing log probabilities.
  - Reference snapshot comparison with exact Kullback-Leibler (KL) divergence regularization.
  - Anti-collapse monitor computing Shannon entropy $H(P_\theta)$ with automated rejection of collapsed distributions.
  - `SelfPlayEngine` with regression rollback safeguards.
- [x] Autonomous Curriculum & Failure Mining (`src/curriculum.zig`, plan §32, §34, §150):
  - Multi-role cognitive deliberation (`ConversationRole`: proposer, thinker, coder, critic, verifier, reflector, judge).
  - Validation gating: strictly requiring verifier and judge approval before converting traces into `VerifiedSyntheticPair`.
  - Failure mining with exponential priority weighting (`FailureMiner`) to prioritize hard regression cases for targeted retraining.
  - Staged dynamic `CurriculumScheduler` (beginner -> intermediate -> advanced -> expert).
- [x] Continuous Verification:
  - Passed `zig build test` (244/251 pass, 7 skipped ROCm device tests).
  - Passed `zig build test -Doptimize=ReleaseFast`.
  - Passed `zig build test-e2e` (26/26 tests passed).
  - Passed `tests/bindings/run.sh` (8/8 language bindings pass).
  - Passed `python3 tests/test_zero_dependency.py`.
  - Passed `python3 docs/graph/gen_graph.py --check` (0 errors).


- [x] Net2Net Model Surgery (`src/growth.zig`, plan §28, §30, §150):
  - Implemented `net2Wider` (exact zero-init bitwise equivalence verified).
  - Implemented `net2Deeper` identity matrix layer initialization preserving representations.
  - Implemented `net2WiderSwiGLU` for GLU/SwiGLU architectures.
  - Implemented `expandVocab`, `applyPlasticityMaskRows`, and `applyPlasticityMaskCols`.
- [x] Model Lineage DAG & Weight Merging (`src/lineage.zig`, plan §37-§39, §150):
  - Implemented `LineageGraph`, `CommitNode`, `LineageBranch` with Merkle parent hash linking.
  - Supported root commits, linear incremental commits, and multi-parent merge commits.
  - Full DAG cycle detection and topological sorting in `verifyDAG`.
  - Implemented `mergeWeightsLinear` (linear interpolation between parent weight sets).
  - Implemented `mergeWeightsTaskArithmetic` ($\theta_{merged} = \theta_{base} + \lambda_A \tau_A + \lambda_B \tau_B$).
  - Implemented `mergeWeightsSlerp` (spherical linear interpolation between multidimensional weight vectors).
- [x] Autonomous Model Evolution Engine (`src/evolution.zig`, plan §36, §150):
  - Implemented `EvolutionCandidate` tracking candidate architectures, weights, eval loss, generation, parent commits.
  - Implemented `EvolutionEngine` autonomous growth & evaluation loop coordinating `growth.zig`, `GrowthGovernor`, `RegressionGovernor`, and `LineageGraph`.
  - Enforced fail-closed acceptance/rollback gates on hardware memory budget and evaluation loss.
- [x] Continuous Verification:
  - Passed `zig build test` (234/241 pass, 7 skipped ROCm device tests).
  - Passed `zig build test -Doptimize=ReleaseFast`.
  - Passed `zig build test-e2e` (26/26 tests passed).
  - Passed `tests/bindings/run.sh` (8/8 language bindings pass).
  - Passed `python3 tests/test_zero_dependency.py`.
  - Passed `python3 docs/graph/gen_graph.py --check` (0 errors).

- [x] Native Autograd & Exact Derivatives (`src/training.zig`, plan §41, §150, §161):
  - Exact reverse-mode backward kernels for GEMM, SiLU, RMSNorm, Softmax, and Cross-Entropy loss.
  - Finite-difference `gradcheck` test suite validating mathematical derivatives to within relative tolerance.
- [x] Frozen-Base LoRA & QLoRA Adaptation:
  - Implemented `LoRALayer` supporting frozen quantized base weights with low-rank adapters A and B.
  - Zero memory overhead for base model weights during gradient accumulation.
- [x] Decoupled AdamW Optimizer & Schedules:
  - Implemented `AdamWOptimizer` with first/second moment tracking, bias correction, and decoupled weight decay.
- [x] Knowledge Distillation & Quantization-Aware Training:
  - Implemented `computeDistillationLoss` combining hard CE ground truth with soft temperature-scaled teacher-student KL divergence.
  - Implemented Straight-Through Estimator (`fakeQuantizeSTE`) for QAT.
- [x] Continuous Verification:
  - Passed `zig build test` (231/238 pass, 7 skipped ROCm device tests).
  - Passed `zig build test -Doptimize=ReleaseFast`.
  - Passed `zig build test-e2e`.
  - Passed `tests/bindings/run.sh` (8/8 language bindings pass).
  - Passed `python3 tests/test_zero_dependency.py`.
  - Passed `python3 docs/graph/gen_graph.py --check` (0 errors).

## Milestone 3.0: Universal Neural Runtime & Multi-Architecture Expansion (100% COMPLETE)
- [x] Universal Hardware-Neutral Principle:
  - Ensured HK is not specialized or tuned for a single architecture (e.g. LLaMA only) or single vendor silicon (e.g. NVIDIA).
  - Maintained universal execution across CPU, Vulkan, ROCm/HIP, and CUDA with identical numerical invariants.
- [x] Native Mixture of Experts (MoE) Architecture & Routing Core (`src/models/moe.zig`):
  - Implemented `MoeConfig`, `MoeRouter` (Top-K selection with softmax or sigmoid normalization, invariant $\sum w_k = 1.0$).
  - Implemented `dispatchMoe` with DeepSeek-style shared expert residual pathway.
  - Zero-alloc, bounded scratch buffer execution.
- [x] Native Multimodal Vision & Audio Transformers (`src/models/multimodal.zig`):
  - Vision Transformer (ViT / CLIP) 2D patch extraction and linear embedding with spatial positional embeddings.
  - Vision-to-Language projection bridge (`projectVisionToText`).
  - 1D Temporal Convolution subsampling front-end for Audio models (Whisper `applyAudio1DConvSubsampling`).
- [x] Universal Architecture Configuration & Registry (`src/engine/config.zig`, `src/models/registry.zig`):
  - Extended `Arch` enum: `.moe`, `.clip_vision`, `.whisper_audio`.
  - Added canonical descriptors for Mixtral, DeepSeek-MoE, CLIP, and Whisper.
  - Added `expert_count` and `experts_per_tok` metadata fields for MoE planning.
- [x] Multi-Model Execution & Memory Planner (`src/engine/planner.zig`):
  - Updated `planExecution` with accurate MoE weight scaling for routed experts.
  - Verified memory footprint parity between dense and MoE configurations.
- [x] Continuous Verification:
  - Passed `zig build test` (250/257 pass, 7 skipped ROCm device tests).
  - Passed `zig build test -Doptimize=ReleaseFast`.
  - Passed `zig build test -Dcpu=x86_64_v3`.
  - Passed `zig build test -Dcpu=x86_64`.
  - Passed `zig build test-e2e` (26/26 tests passed).
  - Passed `tests/bindings/run.sh` (8/8 language bindings pass).
  - Passed `python3 tests/test_zero_dependency.py`.
  - Passed `python3 docs/graph/gen_graph.py --check` (80 curated nodes, 210 files, 447 edges, 0 errors).



