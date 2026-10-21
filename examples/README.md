# HK Framework Examples

This directory provides working, zero-friction examples of the HK Neural Tensor Framework across multiple programming languages and environments.

---

## Directory Index

| Directory / Script | Language | Description |
| ------------------ | -------- | ----------- |
| [`quickstart_transcode.py`](quickstart_transcode.py) | Python | End-to-end GGUF $\leftrightarrow$ HK transcoding, packed SIMD `HKQuantizedLinear` inference, and metadata verification. |
| [`c_cpp/`](c_cpp/) | C / C++ | Loading `.hk` model containers, extracting tensors via C API (`hk.h`) and modern C++ RAII wrapper (`hk.hpp`). |
| [`csharp/`](csharp/) | C# (.NET) | Pure C# memory-mapped reader (`HkModel.cs`) extracting metadata, headers, and tensor shapes. |
| [`java/`](java/) | Java | Java NIO reader (`HkModel.java`) demonstrating cross-platform enterprise JVM tensor ingestion. |

---

## Running the Python Quickstart

```bash
# Ensure dependencies are installed
pip install "hknt[torch]" gguf safetensors

# Run the end-to-end transcoding & inference demo
python examples/quickstart_transcode.py
```

## Running the C/C++ Example

```bash
cd examples/c_cpp
# Build using CMake or Zig as C compiler:
zig cc -I../../include -L../../zig-out/lib main.c -lhk -o hk_c_example
./hk_c_example
```

## Running the C# Example

```bash
cd examples/csharp
dotnet run
```

## Running the Java Example

```bash
cd examples/java
javac -cp ../../bindings/java com/hk/HkModel.java App.java
java -cp ../../bindings/java:. App
```
