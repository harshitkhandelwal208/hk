# Getting Started

## Build

hk needs [Zig 0.17](https://ziglang.org/download/) and nothing else. There are no system libraries to install.

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
hk run model.hk "Hello"                  # uses a CUDA or Vulkan device when there is one that takes the model
hk run model.hk "Hello" -ngl 20          # exactly the first 20 layers on the device, the rest on the CPU
hk run model.hk "Hello" -ngl 0           # CPU only (also --backend cpu, HK_GPU=0)
```

hk picks a device by itself: CUDA when the card has tensor cores (compute capability 8.0 and newer), otherwise Vulkan. A model that does not fit in device memory runs its leading layers on the device and the rest on the CPU. hk prints `[gpu] running on ...` when it uses a device, and says why when a device exists but could not take the model. See [Hardware and Kernels](Hardware-and-Kernels.md).

## Check that it works

```bash
zig build test                              # unit tests, portable kernels, Debug build
zig build test -Doptimize=ReleaseFast       # the same with every instruction set level compiled in
zig build test-e2e -Doptimize=ReleaseFast   # the command line, the server and the downloader, end to end
```

The end to end tests start `hk` as a separate process on a tiny generated model, run a mock Hugging Face server that can drop connections and send wrong checksums, and talk to `hk serve` over HTTP. Add `HK_GPU=1` to run them on the GPU.

## Python

```bash
pip install hknt
hk --help
```

That is the whole install: no dependencies, and no separate download. The wheel for your platform (Linux x86-64 and ARM64, macOS on Apple silicon, Windows x86-64) carries the native library and the `hk` executable, and `import hk` imports neither NumPy nor PyTorch. On another platform pip falls back to the source package, which has no native build; build one with Zig and point `HK_BIN_DIR` at it.

The Python tensor layers (`HKForCausalLM`, `HKTrainer`, `hk.torch`, `hk.numpy`, the pruning and growth helpers) still use PyTorch and NumPy for now, so they are optional extras: `pip install "hknt[torch]"` or `pip install "hknt[numpy]"`. Touching one without its package raises an error that says which extra to install. Replacing PyTorch with HK's own tensor library is on the roadmap (plan section 161).

To work on the Python package from a checkout: build the native code (`zig build -Doptimize=ReleaseFast`), then `pip install -e ".[dev]"` and set `HK_LIB_DIR=zig-out/lib` and `HK_BIN_DIR=zig-out/bin` (they are also found automatically when you run from the repository root). See the [Python API Reference](Python-API-Reference.md).

## Other languages

Bindings for C, C++, Rust, Go, C#, Java and TypeScript are in `bindings/` and call the C library through `include/hk.h`. See [Multi-Language SDKs](Multi-Language-SDKs.md).
