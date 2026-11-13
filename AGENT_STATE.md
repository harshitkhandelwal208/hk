# HK Neural Tensor Framework — Agent State & Implementation Tracker

## Session Info
- **Project**: HK Neural Tensor Framework
- **Active Branch**: `feat/hk-2.1-rocm`
- **Status**: ACTIVE
- **Target Plan**: `docs/plan/MASTER_PLAN.md` (Milestone 2.1 Hardware Acceleration)
- **Last Verified Test**: 197 passed, 7 skipped (ROCm parity tests awaiting AMD HW)

## Current Sprint Tasks (Milestone 2.1)
- [x] Abstracted inline-PTX in `src/cuda_device.zig` behind per-architecture primitives (`src/gpu_prims.zig`).
- [x] ROCm HSA code object compilation (`gfx11-generic`) in `build.zig`.
- [x] Dynamic HIP runtime loader (`src/hip.zig`).
- [x] Implement WMMA (`v_wmma_f32_16x16x16_f16`) kernels for prompt processing across all 26 formats.
- [x] Build `gfx10_3_generic` and `gfx12_generic` target objects and multi-target code object fallback.
- [x] GFX12 split barriers and discrete memory wait counters in `src/gpu_prims.zig`.
- [x] HIP stream graph capture API in `src/hip.zig`.
- [x] Zero-dependency Python distribution (`dependencies = []`, plan §1.6).
- [x] HK's own tensor and autograd core `hk.tensor` (plan §161) with buffer protocol and native C ABI ops.
- [x] Continuous synchronization with `docs/plan/HANDOFF.md` and `docs/plan/STATUS.md`.
