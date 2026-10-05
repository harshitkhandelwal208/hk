# hk

A local LLM runtime and model container, written in Zig. One static binary pulls a model from Hugging Face, converts it to the `.hk` format while it downloads, and runs it, either in your terminal or as an OpenAI compatible server.

I built it because running models locally on an ordinary laptop was a bad experience. Loading a 1B model took gigabytes of RAM for no good reason, and getting a GGUF file into any other tool meant another conversion step. hk is my attempt at the version where memory is the thing that gets treated seriously.

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## What works today

- **Pull and convert in one step.** `hk pull owner/name` streams a model from the Hub and writes a `.hk` file as the bytes arrive. GGUF and safetensors repositories both work, including sharded ones. You never keep the original file. The download is checked against the Hub's SHA-256, resumes after a dropped connection, and the converter's memory use stays flat no matter how big the model is.
- **Run it.** `hk run`, `hk chat` and `hk serve` use a CPU engine written for this project. Weights are memory mapped from the file, so nothing is copied into RAM.
- **OpenAI compatible server.** `/v1/chat/completions` and `/v1/completions` with streaming, several conversations at once, a shared prompt cache, API key support and a `/metrics` route.
- **Models.** Llama, Qwen2 and Qwen3 style dense models (this covers Llama 2 and 3, Mistral without sliding window, SmolLM2, Qwen2.5 and Qwen3). 26 storage formats, from f32 down to the 1 and 2 bit IQ and ternary types.
- **Tokenizers and chat templates.** Byte-level BPE and SentencePiece, checked against llama.cpp on 301,948 tokens of real text. Chat templates run on a built-in Jinja subset, checked against Python's jinja2.

## Try it

```bash
zig build -Doptimize=ReleaseFast        # needs Zig 0.16.0 or newer

./zig-out/bin/hk pull Qwen/Qwen3-0.6B
./zig-out/bin/hk run Qwen/Qwen3-0.6B "What is the capital of France?" --chat --temp 0
./zig-out/bin/hk chat Qwen/Qwen3-0.6B
./zig-out/bin/hk serve Qwen/Qwen3-0.6B --port 8080
```

A GGUF repository works the same way and picks Q4_K_M unless you ask for another quant:

```bash
hk pull bartowski/SmolLM2-135M-Instruct-GGUF:Q8_0
```

Already have a GGUF on disk? `hk convert-gguf model.gguf model.hk`.

## How it compares with llama.cpp

Measured on a Ryzen 7 7445HS (6 cores), CPU only, 6 threads, same models. Medians of 5 alternating runs; the machine was not idle, so differences under about 5 percent are noise. Full table, method and the GPU comparison are in [benchmarks/results.md](benchmarks/results.md) and the [Benchmarks page](docs/wiki/Benchmarks-and-Performance.md); `hk-compare` reproduces them.

| | hk vs llama.cpp (CPU) |
|:---|:---|
| Private memory while generating | 5 to 9 times lower for most models (8 to 10 MiB against 45 to 109 on SmolLM2-135M, 30 against 182 on Qwen3-0.6B) |
| Prompt processing | faster on 8 of 10 models (106 to 240 percent), tied on f16, 91 percent on IQ4_XS |
| Decode speed | slightly behind: 92 to 99 percent, tied on f16 |

So memory is a clear win, prompt processing is mostly ahead, and decode is a few percent behind. I have not closed that gap. The Vulkan GPU backend works but is much slower than llama.cpp's Vulkan backend (roughly 25 to 40 percent on prompts, 65 to 80 percent on decode, one NVIDIA laptop GPU).

## What it does not do yet

I would rather you hear this from me.

- Only Llama, Qwen2 and Qwen3 style dense models run. No Gemma, Phi, Mistral with sliding window, mixture of experts, or vision models.
- The tokenizer.json importer handles byte-level BPE only, so a safetensors repository with a SentencePiece or Unigram tokenizer is refused with a message. Its GGUF version will work.
- GPU support is a Vulkan backend (`-ngl`, `HK_GPU=1`) that runs a whole model on one device. It was tested on one NVIDIA card only and is slower than llama.cpp's. There is no Metal, ROCm, CUDA-in-engine or TPU backend.
- The CPU kernels are built for each instruction set level (AVX-512 VNNI, AVX2, generic on x86; NEON with and without dot product on ARM) and chosen at startup. The ARM kernels pass the portable tests but have not been run on ARM hardware, and the older x86 levels have not been benchmarked on real older CPUs.
- Split GGUF files (`-00001-of-00003.gguf`) are not downloaded yet.
- No grammar or JSON-schema constrained decoding, and no tool-call parsing.
- Interrupted conversions restart from the beginning. The download itself resumes, the conversion does not.

## The other half of the project

The `.hk` container and the Python package carry a number of research features that predate the engine work above: growing a model's layers and vocabulary (Net2Net), a version history stored inside the file with hash chaining and rollback, 2:4 structured sparsity, in-place metadata edits, and a training and self-play pipeline in Python. They work in their unit tests. I have not validated them against large models or compared them with other training stacks, so treat them as experimental. Each has a page in the [wiki](docs/wiki/Home.md) that says what is tested and what is not.

## Install

| What | How |
|:---|:---|
| CLI and server | build from source: `zig build -Doptimize=ReleaseFast` (Zig 0.16.0 or newer) |
| Python package | `pip install hknt` (1.1.1 and later bundle the current native library and CLI; 1.1.0 has the old engine) |
| Node.js | `npm install hkntf` |
| C, C++, Rust, Go, C#, Java | bindings in [bindings/](bindings/) over the C header [include/hk.h](include/hk.h) |

Tests: `zig build test` for the Zig side, `zig build test-e2e` for the CLI, server and downloader tests against the built binary, and `pytest tests` for the Python package.

## Docs

Start with [Getting Started](docs/wiki/Getting-Started.md). The rest of the [wiki](docs/wiki/Home.md) covers the [CLI](docs/wiki/CLI-Reference.md), [downloading models](docs/wiki/Downloading-Models.md), [serving](docs/wiki/Inference-and-Serving.md), the [file format](docs/wiki/Format-Specification.md), and what each model, format and platform supports in the [compatibility page](docs/wiki/Compatibility.md).

## License

Apache 2.0, see [LICENSE](LICENSE).
