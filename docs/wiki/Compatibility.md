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

26 storage formats are read. Every one of them has a decoder, a CPU dot product kernel and a batch kernel; the GPU decodes the formats marked.

| Format | CPU | GPU (Vulkan) |
|:---|:---:|:---:|
| f32, f16, bf16 | yes | yes |
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | yes | yes |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | yes | yes |
| IQ4_NL, IQ4_XS | yes | yes |
| IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S | yes | not yet |
| TQ1_0, TQ2_0, MXFP4, NVFP4 | yes | not yet |

A model that uses a format the GPU cannot decode runs on the CPU, and hk says so. Quantized weights are decoded exactly: each decoder is tested against the reference `gguf` package on blocks of random data (`tests/test_quant.zig`).

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
| Metal, ROCm, CUDA, TPU | no backend |

See [Hardware and Kernels](Hardware-and-Kernels.md) for how the kernels are chosen and what has and has not been verified.
