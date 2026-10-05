# Design

This page explains how the parts fit together and why they are built the way they are. It describes what is implemented.

## Why it exists

Running a language model locally on an ordinary laptop was a bad experience: loading a 1B model took gigabytes of RAM for no good reason, and getting a file from one tool into another meant another conversion step. hk is an attempt at the version where memory is treated seriously, and where there is one small program instead of a stack of runtimes.

## The pieces

```
Hub / GGUF / safetensors ──► converter ──► .hk container ──► engine ──► CLI, server, C library
                                                │                │
                                                │                ├─ CPU kernels (one build per instruction set level)
                                                │                └─ Vulkan kernels (GPU)
                                                └──► research tools: growth, lineage, sparsity, training (Python)
```

| Part | Where | What it does |
|:---|:---|:---|
| Container | `src/format.zig`, `reader.zig`, `writer.zig` | A 128 byte header, a table of contents, 128 byte aligned payloads, an optional appendix. See [Format Specification](Format-Specification.md). |
| Converters | `src/convert/`, `src/gguf.zig` | Stream GGUF and safetensors into `.hk` with bounded memory; write GGUF and safetensors back out. |
| Downloader | `src/hub/` | HTTP with resume, retries, redirects that drop credentials, SHA-256 checking. |
| Tokenizer, templates | `src/tokenizer*`, `src/chat/` | Byte level BPE and SentencePiece; a Jinja subset for chat templates. |
| Engine | `src/engine/` | Weights as views of the mapped file, a forward pass that allocates nothing, a KV cache that grows in segments. |
| Kernels | `src/quant/`, `src/kernels/` | Dot products and matrix multiplies for every format, built once per instruction set level. |
| GPU backend | `src/vk/`, `shaders/` | A small Vulkan compute runtime and the shaders. |
| Server | `src/server/` | HTTP, OpenAI style JSON, a scheduler that batches conversations. |
| C library | `src/c_api.zig`, `include/hk.h` | What the bindings and the Python package call. |

## Decisions and their reasons

**Weights are views, not copies.** The engine never reads a tensor into its own memory. A weight matrix is a slice of the mapped file plus a storage type and a shape. Loading costs address space; pages arrive as layers are first used; the operating system can drop them under pressure. This is why a running model costs tens of MiB of private memory.

**Quantized weights stay quantized.** Matrices are multiplied in their stored block format. Activations are quantized to 8 bits per block, so the inner loop is an integer dot product. This is the same scheme GGML uses, which keeps results comparable with it.

**Batches get their own kernel.** Token generation reads each weight once per token, so it is limited by memory bandwidth. A prompt can reuse every weight many times, so it is limited by compute, and it uses a register tiled kernel that unpacks a tile of weight rows into rows-as-lanes layout once and runs every token of the batch against it. The unpacked tile lives in a small per thread buffer, so there is no second copy of the model.

**Kernels are chosen at run time.** The same source is compiled for each instruction set level, and a table of function pointers is filled when the program starts. All levels share one definition of the integer arithmetic, so they agree exactly, and the fast paths can be tested against the portable one on any machine.

**The KV cache is a set of segments.** Positions are allocated in groups of 256, keys stored transposed in tiles of 64 so attention scores sixteen positions per vector. Memory follows the context actually used.

**Alignment.** Payloads in a `.hk` file start on 128 byte boundaries (the header's alignment field can also say 4096 or 16384). That makes every tensor start on a cache line and lets the file be memory mapped without shifting; it is a convenience for the mapped CPU path, not a claim about any particular accelerator.

**Everything checkable is checked.** Each tensor's shape and exact byte size are verified when a model loads, with the tensor's name in the error. Untrusted input is never turned into an enum without checking it. Downloads are verified against the Hub's hash.

## What is not designed yet

Mixture of experts, sliding window attention, speculative decoding, grammar constrained sampling, multi device and multi node execution, and GPU backends other than Vulkan.
