# Getting Started

## Build

hk needs [Zig 0.16](https://ziglang.org/download/) and nothing else. There are no system libraries to install.

```bash
git clone https://github.com/harshitkhandelwal208/hk.git
cd hk
zig build -Doptimize=ReleaseFast
```

This builds the command line tool `zig-out/bin/hk`, the C library `zig-out/lib/libhk.so` (`.dylib` or `.dll` on other systems), and a few developer tools. A full release build compiles the kernels once for every instruction set level of your processor family, which takes a couple of minutes. For quick development builds, `-Dkernels=fallback` builds only the portable level.

The binary is not tied to the machine that built it: it picks the fastest kernels your processor supports when it starts. `hk hardware-profile` shows what it found.

## Run a model

```bash
./zig-out/bin/hk pull Qwen/Qwen3-0.6B                       # download and convert
./zig-out/bin/hk run Qwen/Qwen3-0.6B "What is the capital of France?" --chat --temp 0
./zig-out/bin/hk chat Qwen/Qwen3-0.6B                       # interactive
./zig-out/bin/hk serve Qwen/Qwen3-0.6B --port 8080          # OpenAI compatible API
```

Models are cached under `~/.cache/hk` (set `HK_HOME` to move it). A GGUF repository works the same way and picks Q4_K_M unless you name another quant:

```bash
hk pull bartowski/SmolLM2-135M-Instruct-GGUF:Q8_0
```

If you already have a GGUF file, convert it once with `hk convert-gguf model.gguf model.hk` and pass the `.hk` path instead of a repository name. Details are in [Downloading Models](Downloading-Models.md).

## Use the GPU

```bash
hk run model.hk "Hello" -ngl 99          # Vulkan, whole model on the GPU if it fits
```

Without `-ngl` (and without `HK_GPU=1` in the environment) everything runs on the CPU. When the GPU cannot be used, hk says why and carries on with the CPU. See [Hardware and Kernels](Hardware-and-Kernels.md).

## Check that it works

```bash
zig build test                              # unit tests, portable kernels, Debug build
zig build test -Doptimize=ReleaseFast       # the same with every instruction set level compiled in
zig build test-e2e -Doptimize=ReleaseFast   # the command line, the server and the downloader, end to end
```

The end to end tests start `hk` as a separate process on a tiny generated model, run a mock Hugging Face server that can drop connections and send wrong checksums, and talk to `hk serve` over HTTP. Add `HK_GPU=1` to run them on the GPU.

## Python

The Python package wraps the C library and adds the research features (growth, lineage, sparsity, training):

```bash
cp zig-out/lib/libhk.so python/hk/
pip install -e .
```

`pip install hknt` installs the published wheel, which is older than the code in this repository. The `hk` command of the Python package finds and runs the native executable; set `HK_BIN_DIR=zig-out/bin` if it is not next to the package. See the [Python API Reference](Python-API-Reference.md).

## Other languages

Bindings for C, C++, Rust, Go, C#, Java and TypeScript are in `bindings/` and call the C library through `include/hk.h`. See [Multi-Language SDKs](Multi-Language-SDKs.md).
