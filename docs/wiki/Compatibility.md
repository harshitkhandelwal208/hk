# Compatibility

What runs, in which format, on which hardware, and how well it is tested.

## Models

| Family | Status | Notes |
|:---|:---|:---|
| Llama 2 / 3, Mistral without sliding window, SmolLM2 (`llama`) | runs | rope scaling: none, linear, llama3, yarn |
| Qwen2, Qwen2.5 (`qwen2`) | runs | bias on the q, k, v projections |
| Qwen3 (`qwen3`) | runs | per head q and k norm |
| Gemma, Phi, Falcon, Mamba, mixture of experts, vision and audio models | not supported | the loader refuses them with a message naming the architecture |

The engine is a dense decoder with grouped query attention, RMS norm and a gated MLP (SiLU). Head width up to 256. Logits are checked against Hugging Face Transformers (cosine similarity 1.000000 on SmolLM2 f16 and Qwen3 BF16) and perplexity against llama.cpp.

## Weight formats

26 storage formats are read. Every one of them has a decoder, a CPU dot product kernel and a batch kernel; the GPUs decode the formats marked.

| Format | CPU | Vulkan | CUDA |
|:---|:---:|:---:|:---:|
| f32, f16, bf16 | yes | yes | yes |
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | yes | yes | yes |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | yes | yes | yes |
| IQ4_NL, IQ4_XS | yes | yes | yes |
| IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S | yes | not yet | yes |
| TQ1_0, TQ2_0, MXFP4, NVFP4 | yes | not yet | yes |

CUDA runs every format the CPU runs because its kernels call the CPU's own block decoders; each of the 26 is checked against an f64 reference on the GPU (`tests/test_cuda.zig`). Only Q8_0 (and the dense types) have been timed on CUDA. A model that uses a format the chosen GPU cannot decode runs on the CPU, and hk says so (with `--backend vulkan|cuda` it fails instead). Quantized weights are decoded exactly: each decoder is tested against the reference `gguf` package on blocks of random data (`tests/test_quant.zig`).

## Tokenizers and chat templates

Byte level BPE and SentencePiece vocabularies, checked against llama.cpp on 301,948 tokens of real text. Chat templates run on a built-in Jinja subset, checked against Python's jinja2. A template that uses something outside the subset is reported, and the chat falls back to a plain transcript.

## Hardware

| Target | Status |
|:---|:---|
| x86-64 with AVX-512 and VNNI | tested (Ryzen 7 7445HS) |
| x86-64 with AVX2 | tested by forcing the level on the same machine (`HK_KERNELS=avx2`) |
| x86-64 without AVX2 | portable level, tested by forcing it; slow |
| x86-64 with AVX-VNNI and AVX2 only (Alder Lake and later) | built; has not run on such a processor |
| AArch64 (Apple Silicon, Graviton, Raspberry Pi 4/5) | built and cross compiled for Linux and macOS; **not run on ARM hardware** |
| Windows, macOS | cross compiled; not run |
| NVIDIA GPU through Vulkan | tested on one card (RTX 3050 Laptop) |
| AMD, Intel and Apple GPUs through Vulkan | not tested |
| CUDA | first engine (`--backend cuda`): all 26 CPU weight formats, llama/qwen2/qwen3, one NVIDIA GPU; verified against the CPU on one RTX 3050; early performance (see Hardware and Kernels) |
| ROCm (AMD, RDNA3+) | `--backend rocm`: the same Zig device code compiled for amdgcn and the shared GPU engine; compiles, planner logic unit tested; **not run on AMD hardware** |
| Metal, TPU | no backend |

See [Hardware and Kernels](Hardware-and-Kernels.md) for how the kernels are chosen and what has and has not been verified.

## Release contract

What a release promises, per the project plan (section 148). Every row says what was run.

### Backends

| Backend | Selected by | Verified | Known gaps |
|:---|:---|:---|:---|
| CPU | default; `--backend cpu` | unit tests at four instruction-set levels; logits byte-identical before/after the executor extraction in 70 comparisons; llama/qwen2/qwen3; perplexity within 0.3 % of llama.cpp and Transformers | ARM and Windows not run on hardware |
| Vulkan | `-ngl N`, `--backend vulkan` | parity with the CPU for llama, wide MHA, qwen2 and qwen3 in f32 and Q8_0, multi-sequence slots, a 700-token context, head sizes 16/64/128, partial offload; kernel-level tests of the cooperative-matrix GEMM (15 formats) and attention against the f32 kernels; one RTX 3050 Laptop | one device; 15 of 26 formats; K-quant decode is ~40 % behind llama.cpp's Vulkan backend; cooperative matrices need a 32-wide subgroup (NVIDIA; AMD/Intel not run) |
| CUDA | `--backend cuda`; first choice in `auto` on compute capability 8.0+ | the same parity set, all 26 formats against an f64 reference (f32, f16 tensor core and int8 tensor core kernels each), head sizes 16/64/128, partial offload, graph replay; one RTX 3050 Laptop (sm_86) | one device; the integer tensor core and `dp4a` kernels cover 8 formats (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K), the rest use f16 tensor cores; prompts of K-quant models are ~30 % behind llama.cpp's CUDA backend; sm_70/75 devices get the portable kernels (no tensor core path), untested |

`--backend auto` (the default when a GPU flag is given) tries CUDA first when the card has tensor cores (compute capability 8.0 and newer), because it was ahead of Vulkan on prompts and on decode for every format tried (see the benchmark below), then Vulkan, and keeps the CPU if neither fits, printing why. Otherwise Vulkan goes first. A model that does not fit in device memory runs its leading layers on the device and the rest on the CPU (`-ngl N` fixes the number of layers; without it the planner takes as many as fit in 90 % of free device memory, keeping the requested context).

### Versions

| What | Value | Compatibility |
|:---|:---|:---|
| Container format | 1.0 (header `version_major` 1) | This release adds optional fields only: header flag bit 9 and the SHA-256 seal, a CRC-32 on new appendix records, and an edit journal beside the file. Reading the 1.1.1 source (not run) shows its reader checks only the magic and major version, ignores bit 9, never reads the checksum field and never verifies the appendix CRC, so it opens files written by this release. An older binary that *edits* a sealed file leaves the seal stale, which `hk verify` then reports. This release reads files without a seal. |
| C ABI | 2 (`HK_ABI_VERSION`, `hk_abi_version()`) | Version 1 is the unversioned ABI of 1.x. Incompatible changes in 2: Q4_K tensors written by the old quantizer must be re-quantized; the product functions report failure as NaN; sizes are validated. There is no struct-size negotiation or capability discovery yet. |
| Python package | follows the release | `metadata_set`, `append_record` and `rollback_appendix` need the native library |

### Known limitations

- Dense decoders only: llama, mistral without sliding window, qwen2, qwen3. No MoE, sliding window, state-space or multimodal models.
- One device per run, plus the CPU for layers that do not fit (partial offload runs the *leading* layers on the device; the context is kept, or a whole-model plan halves it but not below 2048 positions before going partial).
- Run and verified on one Linux x86-64 machine with one NVIDIA GPU. Windows, macOS, ARM, AMD and Intel GPUs, Metal, ROCm and multi-GPU have not been run (ROCm is implemented and compiles; see above).
- No signatures: the seal detects accidental damage and an unfinished edit, not tampering (anyone can recompute it).
- Sealing and verifying are O(file) in time (about 1.7 GiB/s here) at a constant 13 MiB of memory.
- All speed numbers come from one laptop (Ryzen 7 7445HS, RTX 3050 6 GB Laptop, 6 CPU threads); a different GPU or CPU can reorder the engines. The GPU clocks of this laptop vary by tens of percent between sessions, so only numbers from one interleaved run are compared.

### Benchmark suite

Everything is reproducible from the repository:

```bash
zig build -Doptimize=ReleaseFast
zig-out/bin/hk-tiny-model m.gguf --q8 --arch qwen3 --dim 1024 --layers 12 --heads 8 --kv 2 --ffn 2816 --ctx 2048
zig-out/bin/hk convert-gguf m.gguf m.hk
HK_BACKEND=cpu|vulkan|cuda zig-out/bin/hk-probe m.hk --bench 512 64     # prefill and decode tok/s
HK_GPU_PROFILE=1  ... # host launch time and whole-pass GPU time;  HK_CUDA_EVENTS=1 ... # per-kernel GPU time
zig-out/bin/hk-compare --llama-bin <dir> --pair m.hk:m.gguf --threads 6 --repeats 5   # against llama.cpp, with a real model
```

Measured with `benchmarks/suite.py` (all engines interleaved, 3 repeats, median and min..max, prompt 512, generate 128, 6 threads) at hk `57948cf` against llama.cpp `d806193` on real GGUF files that both engines load byte for byte; the full tables with memory are in `benchmarks/results/2026-10-21b-four-models.md`:

| Model (GGUF) | Engine | Prompt tok/s | Generation tok/s |
|:---|:---|--:|--:|
| Qwen3-0.6B Q8_0 | hk CUDA / llama.cpp CUDA | 7,577 / 7,786 | 143.2 / 152.2 |
| | hk Vulkan / llama.cpp Vulkan | 4,252 / 5,779 | 121.3 / 147.1 |
| | hk CPU / llama.cpp CPU | 613 / 383 | 23.5 / 24.5 |
| Llama-3.2-1B Q4_K_M | hk CUDA / llama.cpp CUDA | 4,697 / 6,185 | 126.6 / 137.1 |
| | hk Vulkan / llama.cpp Vulkan | 3,005 / 5,905 | 118.4 / 132.4 |
| | hk CPU / llama.cpp CPU | 343 / 313 | 20.1 / 21.1 |
| Qwen2.5-1.5B Q4_K_M | hk CUDA / llama.cpp CUDA | 3,306 / 4,287 | 100.8 / 103.1 |
| | hk Vulkan / llama.cpp Vulkan | 2,057 / 3,756 | 71.9 / 104.4 |
| | hk CPU / llama.cpp CPU | 226 / 235 | 16.6 / 17.3 |
| Llama-3.2-3B Q4_K_M | hk CUDA / llama.cpp CUDA | 1,774 / 2,410 | 52.8 / 56.7 |
| | hk Vulkan / llama.cpp Vulkan | 1,091 / 2,335 | 48.7 / 55.8 |
| | hk CPU / llama.cpp CPU | 85 / 109 (74..124 / 106..111) | 8.1 / 8.5 |

In words: hk's CUDA backend is level with llama.cpp's on Q8_0 prompts (97 %) and 6 percent behind on decode; on the K-quant models it is 74 to 77 percent on prompts and 92 to 98 percent on decode. hk's Vulkan backend is 74 percent (Q8_0) down to 47 percent (K-quants) on prompts and 69 to 89 percent on decode. On the CPU hk is faster on Q8_0 prompts (+60 %), within a few percent on decode, and mixed on K-quant prompts (96 % and 110 % on two models, 78 % with a wide range on the third), with a smaller resident set (2.0 GiB against 3.4 GiB for the 3B model). The CPU numbers of this laptop swing by about 2x between sessions, so compare engines only within a table row. Accuracy (`benchmarks/perplexity.py`, wikitext-2 test, 20 windows of 512 tokens, the llama-perplexity protocol): Qwen3-0.6B Q8_0 perplexity llama.cpp 23.67-23.68, hk 23.64-23.68, Transformers f32 on the original weights 23.73; Qwen2.5-1.5B Q4_K_M llama.cpp 11.64-11.71, hk 11.64-11.70. Transformers on the GPU was not measured here.
