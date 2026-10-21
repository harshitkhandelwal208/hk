# Contributing to hk

Thank you for your interest in contributing to hk. Kernels, backends, model support, bindings, documentation and benchmarks are all welcome. Please keep claims honest: say what you ran, on which hardware, and mark anything you did not run as "not run".

---

## 1. Be kind

hk is a personal project of mine (Harshit Khandelwal). I'd like it to be a friendly place to work: please be respectful and constructive in issues and pull requests. If something goes wrong, or you just want to talk about an idea before you start, email me at `harshitkhandelwal208@gmail.com` or open a Discussion.

---

## 2. Architecture Overview

- **`src/`**: the engine, written in Zig 0.17.
  - `format.zig`, `convert/`: the `.hk` container (header, TOC, metadata, seal) and the streaming converters (GGUF, safetensors).
  - `quant/`: the 26 storage formats. `blocks.zig` defines each block once, `dequant.zig` decodes (checked against llama.cpp), `encode.zig` encodes, `vecdot.zig`/`gemm.zig` are the CPU kernels.
  - `kernels/`, `engine/`: the CPU engine (model, KV cache, sampler, sessions) and the kernels it runs, built per instruction-set level.
  - `backend/`: the device interface and `select.zig`, the pure planner that picks CPU, Vulkan or CUDA and decides how many layers go to the device.
  - `cuda/`, `cuda_device.zig`, `cuda.zig`: the CUDA engine; its kernels are written in Zig and compiled to PTX at build time (no `nvcc`).
  - `vk/`, `shaders/`: the Vulkan engine and its GLSL compute shaders (the compiled SPIR-V is checked in; `shaders/build.sh` regenerates it).
  - `hub/`, `server/`, `tokenizer/`, `chat/`: model download, the OpenAI-compatible server, tokenizers and chat templates.
  - `c_api.zig`: the stable C interface exported as `hk.dll`, `libhk.so` or `libhk.dylib`, running on the same kernels.
- **`tools/`**: the `hk` CLI and the benchmark and probe tools.
- **`include/`**: the C and C++ headers (`hk.h`, `hk.hpp`).
- **`python/hk/`**: the Python package, a thin layer over the C interface plus PyTorch and Hugging Face integration.
- **`bindings/`**: Rust, C#, Go, Java and TypeScript bindings.
- **`tests/`**: Zig test suites, end-to-end tests that run the built binary, the binding tests and pytest.
- **`benchmarks/`**: the comparison scripts against llama.cpp and Transformers, and the recorded results.
- **`docs/`**: the wiki (`docs/wiki`), the plan and project state (`docs/plan`) and a queryable map of the code (`docs/graph`). `graphify-out/` is a generated knowledge graph of the repository.

For a map of the project and its working rules, start with [AGENTS.md](AGENTS.md) and [docs/plan/CONTEXT.md](docs/plan/CONTEXT.md).

---

## 3. Development Environment Setup

### Prerequisites
- **Zig**: `0.17.x`. Verify with `zig version`. (0.16 no longer builds this repository: 0.17 renamed the optimize modes, moved reflection to `field_names`/`field_values`, and the install prefix is no longer known while configuring.)
- **Python**: Python `3.10`+ (Python 3.12 recommended).
- **C/C++ Compiler**: Clang, GCC, or MSVC (optional, Zig acts as C compiler).
- **Rust** (optional, for Rust SDK): `rustc` & `cargo`.
- **.NET SDK** (optional, for C# SDK): .NET 8.0+.
- **JDK** (optional, for Java SDK): Java 17+.

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/harshitkhandelwal208/hk.git
   cd hk
   ```

2. **Build the native Zig engine and run native tests**:
   ```bash
   zig build
   zig build test --summary all
   ```

3. **Install the Python package in editable mode**:
   ```bash
   pip install -e ".[dev]"      # pytest plus the optional torch and numpy layers; hknt itself needs no dependencies
   export HK_LIB_DIR=$PWD/zig-out/lib HK_BIN_DIR=$PWD/zig-out/bin
   ```

4. **Verify the Python CLI**:
   ```bash
   python -m hk.cli --help
   hk --help
   ```

5. **Run the Python test suite**:
   ```bash
   pytest tests/
   ```

---

## 4. Testing Guidelines

Any PR touching native code or quantization kernels **must** satisfy:
- **Numerical Parity**: Quantizers and dequantizers must match reference implementations within acceptable epsilon (`max_discrepancy == 0.0` for bit-identical dequantization, or $\le 10^{-4}$ for floating point math).
- **Zero Failures, Zero Warnings**: All tests in `tests/` must pass with 0 pytest warnings and 0 errors.
- **Cross-Platform Portability**: File I/O and SIMD intrinsics must compile cleanly across Windows (`x86_64`), Linux (`x86_64`, `aarch64`), and macOS (`aarch64`).
  ```bash
  zig build -Dtarget=x86_64-linux
  zig build -Dtarget=aarch64-linux
  zig build -Dtarget=aarch64-macos
  zig build -Dtarget=x86_64-windows
  ```

---

## 5. Submitting a Pull Request

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feat/my-quant-kernel
   ```
2. **Commit your changes**:
   - Write clear, concise commit messages.
   - Reference any relevant issues (e.g. `Fixes #123`).
3. **Run all tests locally before opening the PR**:
   ```bash
   zig build test
   pytest tests/
   ```
4. **Open a Pull Request** against `main`:
   - Fill out the PR template completely.
   - CI will automatically run multi-platform tests and multilingual SDK verification.
