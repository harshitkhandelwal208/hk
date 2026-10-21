<div align="center">

# hk

**Local LLMs that run lean, load instantly and keep getting smarter.**

A from-scratch LLM runtime and model format in Zig: zero-copy mmap loading, CPU / CUDA / Vulkan engines, an OpenAI-compatible server, 2:4 sparsity, Net2Net model growth, in-file version history and self-learning loops, in one project.

[![CI](https://github.com/harshitkhandelwal208/hk/actions/workflows/ci.yml/badge.svg)](https://github.com/harshitkhandelwal208/hk/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/harshitkhandelwal208/hk)](https://github.com/harshitkhandelwal208/hk/releases)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

[The pitch](#the-pitch) · [Quick start](#quick-start) · [What is inside](#what-is-inside) · [Why hk](#why-hk) · [How fast is it](#how-fast-is-it) · [What it does not do yet](#what-it-does-not-do-yet) · [Install](#install) · [Docs](#docs)

</div>

---

## The pitch

I'm building hk because I think running a model on your own machine should feel like opening a file, not like moving house. Loading a 1B model took gigabytes of RAM for no good reason, getting a GGUF into another tool meant another conversion step, and a model was something you converted once and never touched again. I wanted a runtime where memory is the first thing that gets engineered, and a model format that can grow, remember its own history and be edited safely.

It's a personal project. In one release cycle I went from a CPU engine to a full 2.0: my own CUDA engine (kernels written in Zig, no CUDA toolkit needed), a Vulkan engine, partial GPU offload, a sealed crash-safe container and a planner that picks the device for you. Prompt processing on the CUDA engine went from 350 to 7,577 tokens/s on Qwen3-0.6B on a laptop RTX 3050, with perplexity matching llama.cpp and Hugging Face Transformers. On the CPU it reads long prompts up to 2.4x faster than llama.cpp while using 5 to 9 times less private memory.

The bigger bet is what comes after fast inference: a model file that carries its own version history, can be widened or given new vocabulary without retraining from zero, stores 2:4 sparse weights, and feeds a loop where the model generates, tests and learns from its own work. Some of that is working today, some is a prototype, and the table below says which is which. The plan runs through hardware backends (ROCm, Metal), more model families and a distributed server; 2.0 is the foundation, not the finish line.

## Quick start

You need [Zig 0.17](https://ziglang.org/download/) and nothing else.

```bash
git clone https://github.com/harshitkhandelwal208/hk && cd hk
zig build -Doptimize=ReleaseFast

./zig-out/bin/hk run  Qwen/Qwen3-0.6B "What is the capital of France?" --chat --temp 0
./zig-out/bin/hk chat Qwen/Qwen3-0.6B
./zig-out/bin/hk serve Qwen/Qwen3-0.6B --port 8080     # OpenAI-compatible API
```

The first command downloads the model, converts it while it downloads and keeps only the `.hk` file. hk uses a CUDA or Vulkan GPU on its own when there is one that fits the model, and runs the rest of the layers on the CPU when the model is too big for the card. `-ngl 0` (or `--backend cpu`) forces the CPU.

A GGUF repository works the same way and picks Q4_K_M unless you ask for another quant (`hk pull bartowski/SmolLM2-135M-Instruct-GGUF:Q8_0`). Already have a GGUF file? `hk convert-gguf model.gguf model.hk`.

## What is inside

hk is two things in one repository: a **runtime** that is fast and frugal with memory, and a **container** that carries the bigger ideas. The status column is the honest part.

| | What it is | Status |
|:---|:---|:---|
| **Zero-copy mmap loading** | Weights are memory mapped from the `.hk` file and never copied; the KV cache grows in small segments; converter and downloader use bounded buffers | Working, measured: 5 to 9× less private memory than llama.cpp on the CPU |
| **Streaming Hub → `.hk`** | `hk pull` converts GGUF and safetensors while downloading (resumable, SHA-256 checked) | Working |
| **CPU, CUDA and Vulkan engines** | Tensor cores and integer dot products, partial offload for models bigger than the card, CUDA kernels written in Zig (no toolkit needed) | Working; tested on one NVIDIA laptop GPU |
| **OpenAI-compatible server** | Streaming, many conversations at once, shared prompt cache | Working |
| **26 quantization formats** | f32 down to 1 and 2 bit IQ and ternary types, one definition per block format, checked against llama.cpp | Working |
| **Sealed, crash-safe container** | SHA-256 seal, journaled in-place metadata edits, verify and repair | Working |
| **2:4 structured sparsity** | Keeps 2 of every 4 weights; stored in the file and decoded on load | Storage only: no sparse kernels yet, so it does not speed up inference (sparse tensor cores are planned work) |
| **Net2Net model growth** | Widen a model or grow its vocabulary without losing what it learned (`hk expand`) | Prototype: unit tested, not validated on large models |
| **In-file version history** | A hash-chained history of changes stored inside the `.hk` file, with rollback | Working for append and rollback; branching and merging are not built |
| **Self-learning loops for LLMs** | Python toolkit that generates solutions, tests them and trains on what passed | Experimental: building blocks, no result yet that it improves a model |

Each research item has a page in the [wiki](docs/wiki/Home.md) that says what is tested and what is not.

## Why hk

- **Memory first.** Weights are memory mapped from the file and never copied, the KV cache grows in small segments, and the downloader and converter use bounded buffers. On the CPU, hk's private memory while generating was 5 to 9 times lower than llama.cpp's on most models I measured (30 MiB against 182 on Qwen3-0.6B).
- **One step from the Hub to a running model.** `hk pull` streams, converts, checks the Hub's SHA-256, resumes a dropped connection and never keeps the original file.
- **CPU, CUDA and Vulkan from one code base.** The CUDA kernels are written in Zig and compiled to PTX during the build, so there is no CUDA toolkit to install. Vulkan uses cooperative matrices and integer dot products when the driver has them.
- **A real server.** `/v1/chat/completions` and `/v1/completions` with streaming, several conversations at once, a shared prompt cache, API keys and a `/metrics` route.
- **Careful with files you did not make.** Model files, tokenizers, chat templates and HTTP input are treated as hostile: bounds are checked before use, and errors name the tensor or field.
- **Honest numbers.** Every claim here has a measurement or a test behind it, and the [limits](#what-it-does-not-do-yet) are listed.

Supported models: Llama, Qwen2 and Qwen3 style dense models (Llama 2 and 3, Mistral without sliding window, SmolLM2, Qwen2.5, Qwen3), in 26 storage formats from f32 down to the 1 and 2 bit IQ and ternary types. Tokenizers (byte-level BPE and SentencePiece) were checked against llama.cpp on 301,948 tokens of real text, and chat templates against Python's jinja2.

## How fast is it

Measured against llama.cpp on the same GGUF bytes, 3 to 5 interleaved repeats, one machine (Ryzen 7 7445HS, RTX 3050 6 GB laptop GPU). Differences under about 5 percent are noise. Method, variance and memory are in [benchmarks/](benchmarks/) and the [Benchmarks page](docs/wiki/Benchmarks-and-Performance.md); `hk-compare` reproduces the CPU numbers.

| | CPU | CUDA | Vulkan |
|:---|:---|:---|:---|
| Prompt processing | faster on 8 of 10 models (up to 2.4×), tied on f16 | Q8_0 97 %, Q4_K_M 74 to 77 % of llama.cpp | 47 to 74 % |
| Decode (tokens/s) | 92 to 99 % | 92 to 98 % | 69 to 89 % |
| Memory | private memory 5 to 9× lower | host memory about the same; more VRAM than llama.cpp (1.46 GB against 0.82 on Qwen3-0.6B) | host memory not lower (1.06 GB against 0.83 on Qwen3-0.6B); more VRAM |

So: memory is a clear win, CPU prompt processing is mostly ahead, CUDA is close, and Vulkan, especially on K-quant models, is the slow backend. Perplexity agrees with llama.cpp and Transformers (Qwen3-0.6B Q8_0 on wikitext-2: 23.64 to 23.68 for hk, 23.67 to 23.68 for llama.cpp, 23.73 for Transformers on the original weights). These are one laptop's numbers: read them as "same ballpark, not yet equal".

## What it does not do yet

I would rather you hear this from me than find it out.

- Only Llama, Qwen2 and Qwen3 style dense models run. No Gemma, Phi, sliding window attention, mixture of experts or vision models.
- GPU support was tested on **one NVIDIA laptop GPU**. Nothing was run on AMD, Intel or Apple GPUs. A ROCm backend for AMD GPUs (RDNA3 and newer) is written and compiles, but has not been run on AMD hardware; there is no Metal or TPU backend.
- The ARM kernels compile and pass the portable tests but have not been run on ARM hardware; the older x86 levels have not been benchmarked on older CPUs.
- The `tokenizer.json` importer handles byte-level BPE only, so a safetensors repository with a SentencePiece or Unigram tokenizer is refused with a message (its GGUF version works).
- Split GGUF files (`-00001-of-00003.gguf`) are not downloaded yet; no grammar or JSON-schema constrained decoding; no tool-call parsing; an interrupted conversion restarts from the beginning (the download resumes).

The full list per model, format and platform is on the [Compatibility page](docs/wiki/Compatibility.md).

## Install

| What | How |
|:---|:---|
| CLI and server | build from source: `zig build -Doptimize=ReleaseFast` (Zig 0.17.0 or newer) |
| Python | `pip install hknt` (bundles the native library and CLI) |
| Node.js | `npm install hkntf` |
| Rust | `cargo add hknt` |
| C, C++, Go, C#, Java | bindings in [bindings/](bindings/) over the C header [include/hk.h](include/hk.h); examples in [examples/](examples/) |

Tests: `zig build test` (Zig), `zig build test-e2e` (the built CLI, server and downloader against a mock Hub), `bash tests/bindings/run.sh` (every language binding) and `pytest tests` (Python).

## Docs

Start with [Getting Started](docs/wiki/Getting-Started.md). The [wiki](docs/wiki/Home.md) also covers the [CLI](docs/wiki/CLI-Reference.md), [downloading models](docs/wiki/Downloading-Models.md), [serving](docs/wiki/Inference-and-Serving.md), [hardware and kernels](docs/wiki/Hardware-and-Kernels.md) and the [file format](docs/wiki/Format-Specification.md). Release notes are in the [CHANGELOG](CHANGELOG.md). Contributions are welcome: see [CONTRIBUTING](CONTRIBUTING.md). Security reports: [SECURITY](SECURITY.md).

The repository also ships a [knowledge graph](graphify-out/GRAPH_REPORT.md) of the code (`graphify-out/`), handy for finding your way around.

## Contact

hk is built by me, Harshit Khandelwal. Questions, ideas, bug reports and "I want to help" are all welcome: open an issue or a Discussion, or email `harshitkhandelwal208@gmail.com`.

## License

Apache 2.0, see [LICENSE](LICENSE).
