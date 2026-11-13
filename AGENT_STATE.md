# HK Neural Tensor Framework — Agent State & Implementation Tracker

## Session Info
- **Project**: HK Neural Tensor Framework
- **Active Branch**: `feat/hk-2.1-rocm`
- **Status**: ACTIVE — 100% COMPLETE FOR MILESTONES 2.1, 2.2, & 2.3
- **Target Plan**: `docs/plan/MASTER_PLAN.md` (Milestones 2.1, 2.2, 2.3 Server Runtime)
- **Last Verified Test**: 223 passed, 7 skipped (ReleaseFast, Debug, x86_64_v3, x86_64; 24/24 e2e; 8/8 bindings suites)

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
- [x] Continuous Verification:
  - Passed `zig build test` (223/230 pass, 7 skipped ROCm device tests).
  - Passed `zig build test -Doptimize=ReleaseFast`.
  - Passed `zig build test-e2e`.
  - Passed `python3 tests/test_zero_dependency.py`.
  - Passed `python3 docs/graph/gen_graph.py --check` (0 errors).

