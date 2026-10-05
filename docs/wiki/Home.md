# hk

hk is a local LLM runtime and a model container, written in Zig. One binary downloads a model from Hugging Face, converts it to the `.hk` format while it downloads, and runs it in a terminal or behind an OpenAI compatible server.

This wiki says what each part does, how to use it, and how well it works, including where it does not work yet. If a page makes a claim, the claim has a number or a test behind it, or it says that it does not.

## Start here

| If you want to | Read |
|:---|:---|
| install it and run a model | [Getting Started](Getting-Started.md) |
| get a model from Hugging Face | [Downloading Models](Downloading-Models.md) |
| know every command and flag | [CLI Reference](CLI-Reference.md) |
| run it as an API server | [Inference and Serving](Inference-and-Serving.md) |
| see what runs and what does not | [Compatibility](Compatibility.md) |
| understand the speed and memory numbers | [Benchmarks and Performance](Benchmarks-and-Performance.md) |
| know how it uses your CPU or GPU | [Hardware and Kernels](Hardware-and-Kernels.md) |
| replace another tool with it | [Transition Guide](Transition-Guide.md) |
| fix a problem | [FAQ and Troubleshooting](FAQ-and-Troubleshooting.md) |

## What it is good at

- **Memory.** Weights are memory mapped from the file and never copied, the KV cache grows in small segments as the context fills, and the converter and downloader use bounded buffers. A running model costs a few tens of MiB of private memory on top of the file. llama.cpp, measured the same way, costs several times more.
- **Prompt processing.** On the CPU, hk reads long prompts faster than llama.cpp on eight of ten test models, ties on f16, and is 9 percent behind on IQ4_XS. See [Benchmarks and Performance](Benchmarks-and-Performance.md).
- **Decode speed** is the open gap: 92 to 99 percent of llama.cpp on the same models (tied on f16).
- **One binary for every instruction set level.** The compute kernels are built once for each level of the processor family (AVX-512 with VNNI, AVX2, plain x86-64; NEON with and without dot product on ARM) and the right one is picked when the program starts. Nothing is tuned to the machine it was built on.
- **Downloading.** `hk pull` converts as it downloads, resumes after a dropped connection, and checks the Hub's SHA-256 before the file is put in place.

## What it does not do yet

- It runs Llama, Qwen2 and Qwen3 style dense models. Gemma, Phi, sliding window attention, mixture of experts and vision models are not supported.
- The GPU backend uses Vulkan. It runs a whole model on one device, and it has been tested on one NVIDIA card. There is no Metal, ROCm or TPU backend. See [Hardware and Kernels](Hardware-and-Kernels.md).
- The ARM kernels compile and pass the portable tests, but nobody has run them on ARM hardware yet.

## The research half

The `.hk` container and the Python package carry features that predate the engine: growing a model's width, depth and vocabulary, a hash chained version history stored inside the file, 2:4 structured sparsity, in-place metadata edits and a self-training loop. They work in their unit tests. They have not been validated on large models or compared with other training stacks, so each page about them says what is tested. See [Dynamic Architecture Growth](Dynamic-Architecture-Growth.md), [In-Container Version Lineage](In-Container-Version-Lineage.md), [Storage and Sparsity](Storage-and-Sparsity.md), [Training and Fine-Tuning](Training-and-Fine-Tuning.md) and [Autonomous Self-Training](Autonomous-Self-Training.md).
