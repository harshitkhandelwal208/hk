# Hardware and Kernels

## One binary, every instruction set level

The compute kernels (integer dot products for every quantization format, the batched matrix multiply, attention) are compiled several times, once per instruction set level of the processor family, and linked into the same program. When hk starts it asks the processor what it supports and uses the best level. Nothing depends on the machine that built the binary.

| Architecture | Levels, best first |
|:---|:---|
| x86-64 | `avx512` (AVX-512 with VNNI), `avx2vnni`, `avx2`, `generic` |
| AArch64 | `dotprod` (NEON with the dot product extension), `neon` |
| other | `generic` |

`hk hardware-profile` prints the level in use and the levels the binary contains. To force another level, for testing or to measure it, set `HK_KERNELS=avx2` (or any name above) before running.

All levels compute exactly the same integers. The dot product primitive has one definition, the weight operand is at most 127 in magnitude (or signed, handled separately), and each level implements it with whatever instruction is best: `vpdpbusd` on AVX-512 VNNI and AVX-VNNI, `vpmaddubsw` plus `vpmaddwd` on AVX2 (the 16 bit intermediate cannot overflow under that bound), `sdot` on ARM, and a portable widening multiply elsewhere. Two checks enforce this: `zig build test` runs the portable path on every platform, and the quant tests run every compiled-in level the machine supports against the same reference.

```bash
zig build test -Doptimize=ReleaseFast                    # every level this CPU can run
zig build test -Doptimize=ReleaseFast -Dcpu=x86_64_v3    # the AVX2 code in the library itself
zig build test -Doptimize=ReleaseFast -Dcpu=x86_64       # the portable code in the library itself
```

### What has been run

- **x86-64**: `avx512`, `avx2` and `generic` run here and give identical perplexity. `avx2vnni` needs a processor with AVX-VNNI but no AVX-512 (Intel Alder Lake and Raptor Lake desktop and mobile parts); it is built and the dispatcher selects it by feature bits, but no such processor was available.
- **AArch64**: the `dotprod` and `neon` levels compile for Linux and macOS, including the inline `sdot` and `tbl` instructions, and the portable code paths they share with x86 are tested. **They have not been run on ARM hardware.** The first run on an Apple Silicon Mac or a Graviton instance is the real test; `hk run model.hk "hi" --temp 0` should print the same text as with `HK_KERNELS=neon`, and `zig build test -Doptimize=ReleaseFast` should pass.
- **Windows and macOS on x86-64**: cross compiled, not run.

## Threads

The default thread count is the number of physical cores, not logical CPUs: a second hardware thread on a core adds contention to kernels that are limited by memory bandwidth and synchronization. On processors with two kinds of cores (Intel hybrid parts, ARM big.LITTLE) only the fast cores are counted, using `cpu_capacity` on Linux and the performance level count on macOS. Override with `--threads`.

## Memory

- Weights are never copied. The `.hk` file is memory mapped and the kernels read it in place, so a model costs address space, not memory; pages come in as layers are first used.
- Prompt processing unpacks one tile of 16 rows (8 on AVX2) at a time into a small per thread buffer, a few hundred KiB, instead of keeping a repacked copy of the model. llama.cpp keeps repacked copies of the weights.
- The KV cache is f16 and is allocated in segments of 256 positions as the context fills.

## GPU

A Vulkan compute backend runs the whole network on one device: embedding lookup, every layer, and the final norm in one command buffer per forward pass, with the key and value caches on the device. Enable it with `-ngl N` (any N above 0) or `HK_GPU=1`.

```
$ hk run model.hk "Hello" -ngl 99
[gpu] running on NVIDIA GeForce RTX 3050 6GB Laptop GPU (Vulkan)
```

- **Portable by construction.** Vulkan runs on NVIDIA, AMD, Intel and, through MoltenVK, Apple GPUs. The loader is opened at run time, so a machine without a driver starts normally and uses the CPU. Shaders are written in GLSL (`shaders/`), compiled to SPIR-V and checked in (`src/vk/spv/`), so building hk needs no Vulkan SDK; `shaders/build.sh` recompiles them with `glslc`.
- **What is verified.** On an NVIDIA RTX 3050 Laptop GPU: logits agree with the CPU engine to a cosine similarity of 0.9994 or better for every format the GPU decodes (the CPU quantizes activations to 8 bits and the GPU does not, so exact equality is not expected), unit tests compare the two engines on tiny models, and the command line and server end to end tests pass with `HK_GPU=1`. Nothing was run on an AMD, Intel or Apple GPU.
- **Fit.** hk checks the free memory the driver reports (90 percent of it may be used). A model that fits whole runs whole, with the context halved if that helps (never below 2048 positions when partial offload is the alternative). One that does not fit runs its leading layers on the device and the rest, the embedding and the output matrix on the CPU: the planner takes the largest prefix of layers whose weights and KV cache fit at the requested context, and `-ngl N` sets the number yourself. Per pass the hidden states cross once in each direction (a few megabytes for a prompt).
- **Serving.** Each server slot gets its own region of the device KV cache (`hk serve -ngl 1 --slots 4`).
- **Formats.** The formats marked in [Compatibility](Compatibility.md).
- **Speed.** Prompts run on cooperative matrices (`VK_KHR_cooperative_matrix`, f16 operands with f32 accumulation) when the device reports a 16x16x16 shape and 32-wide subgroups, and fall back to an f32 tiled kernel otherwise (`HK_NO_COOPMAT=1` forces it). Decode of a few tokens uses `VK_KHR_shader_integer_dot_product` for the nibble and K-quant formats (`HK_NO_INTDOT=1` turns it off) and a split-KV attention kernel. Numbers against llama.cpp are in [Compatibility](Compatibility.md).

### Not supported

- **Metal.** Apple Silicon runs the CPU kernels (NEON, untested) or the Vulkan backend through MoltenVK (untested). A Metal backend would need an Apple machine to build and check.
- **CUDA.** `--backend cuda` (and `auto` on a card with compute capability 8.0+) runs on one NVIDIA GPU through the CUDA driver API (loaded at run time; no CUDA toolkit is needed to build or run hk). The device code is written in Zig and compiled to PTX by `zig build`, for sm_70 and sm_80; the sm_80 build adds tensor cores: `mma.sync` int8 kernels for Q8_0, Q4_0/1, Q5_0/1, Q4_K, Q5_K and Q6_K prompts (activations quantized to int8 per 32 values, the arithmetic of the CPU engine), f16 tensor-core kernels for every other format, flash attention for prompts, `dp4a` matrix-vector kernels, a warp-per-32-positions split-KV decode attention and CUDA graphs for decode. Every one of the 26 formats runs (the fallback kernels call the CPU's block decoders). Checked against the CPU on one RTX 3050 Laptop (llama, Qwen2, Qwen3; head sizes 16, 64, 128) and against llama.cpp on perplexity; `HK_NO_TENSOR_CORES`, `HK_NO_DP4A`, `HK_NO_CUDA_GRAPH` and `HK_NO_WARP_ATTN` switch the pieces off. Below compute capability 8.0 only the portable kernels exist (untested on such cards).
- **ROCm (AMD).** `--backend rocm` runs the network on one AMD GPU through the HIP runtime (`libamdhip64`, loaded at run time; no ROCm toolkit is needed to build hk). The device code is the same Zig source as the CUDA kernels (`src/cuda_device.zig`), compiled for AMD by `zig build` into one code object for the generic gfx11 target, which covers RDNA3 and RDNA3.5 (gfx1100 to gfx1103); the engine is the shared one in `src/gpu/engine.zig`. It uses the 4-way int8 dot product (`v_dot4_i32_iu8`) for single-token decode and the portable tiled kernels for prompts; there are no matrix-core (WMMA) kernels, no graph capture, and RDNA1/2/4 and the wave64 GPUs (GCN, CDNA) are not supported. **Status: implemented, compiled and unit tested for planning; not run on AMD hardware**, so correctness and speed are unmeasured. `auto` tries it after Vulkan and CUDA. An RDNA3 iGPU that ROCm does not list can often be used with `HSA_OVERRIDE_GFX_VERSION=11.0.0`; the user needs access to `/dev/kfd` (the `render` group).
- **TPUs.** Google TPUs are programmed through XLA and have no runtime that a program like this can drive directly. There is no TPU backend and none is planned.
- **NPUs.** No backend.
