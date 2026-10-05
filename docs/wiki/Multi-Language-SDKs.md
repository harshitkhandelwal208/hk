# Language Bindings and the C API

> **Status:** the bindings are **container readers/writers**: open an `.hk` file, list tensors, read metadata, read or dequantize tensor data, work with the appendix. They are thin wrappers over the C ABI in `include/hk.h`. **None of the bindings (Rust, Go, C#, Java, TypeScript) exposes the inference engine**, and there is no `HkContext` / `forward_step` / GPU-layer API in them; earlier versions of this page showed examples that did not exist.
>
> What exists for inference from other languages is the C API (`hk_engine_*`, `hk_tokenizer_*`, `hk_sample_token`), which you can call directly through your language's FFI, and which the Python package wraps (`NativeHKEngine`, `NativeHKTokenizer`).
>
> **Tested:** `tests/bindings/run.sh` builds the library, writes a fixture file through the C ABI (two tensors, metadata of every type, one appendix record), and runs a test for each of C, C++, Rust, Go, C#, Java and TypeScript against it. Each test also writes a file with that language's writer and reads it back, applies `metadata set`, and checks a few math helpers. CI runs it on Linux with every toolchain installed. Not covered: Windows and macOS builds of the bindings, Android, and real large models.

---

## The C API (`include/hk.h`, `include/hk.hpp`)

Link against the shared library produced by `zig build -Doptimize=ReleaseFast` (`libhk.so` / `libhk.dylib` / `hk.dll`). Groups of functions:

| Area | Functions |
| :--- | :--- |
| Reader | `hk_open`, `hk_close`, `hk_get_tensor_count`, `hk_get_tensor_info`, `hk_get_all_tensor_infos`, `hk_get_tensor_data` / `_residual` / `_scales`, `hk_dequantize_f32`, `hk_get_metadata_{string,int,float,bool}`, `hk_reader_is_sharded`, `hk_reader_get_split_{index,count}` |
| Writer | `hk_writer_create`, `hk_writer_add_metadata_*`, `hk_writer_add_tensor`, `hk_writer_set_sharding`, `hk_writer_set_raw_storage`, `hk_writer_write_to_file`, `hk_writer_destroy` |
| Metadata edit | `hk_metadata_patch_in_place` |
| Appendix | `hk_appendix_get_count`, `hk_appendix_get_entry`, `hk_appendix_append`, `hk_appendix_rollback` |
| Quantization helpers | block quantize/dequantize for NF4, DQ8, DQT, Q4_0, Q8_0, Q4_K, Q8_K, Q6_K, Q2_K, IQ4_NL, MXFP4, NVFP4; 2:4 pack/unpack; 16x16 tiling |
| Math helpers | `hk_dot_product_f32`, `hk_gemv_*`, `hk_gemm_f32`, fused NF4/DQ8 GEMV, SwiGLU/RMSNorm/SiLU forward pieces |
| Growth | `hk_net2wider`, `hk_net2deeper`, `hk_net2wider_swiglu`, `hk_expand_vocab`, plasticity masks |
| Tokenizer | `hk_tokenizer_load_from_file`, `_free`, `_get_vocab_size`, `_encode`, `_decode` |
| Engine | `hk_engine_load_from_file`, `_last_error`, `_free`, `_get_vocab_size`, `_get_context_size`, `_reset_cache`, `hk_engine_forward_tokens`, `hk_engine_forward` |
| Sampling | `hk_sample_token` (temperature, top-k, top-p, min-p, repeat penalty, seed) |

The engine functions run Llama, Qwen2 and Qwen3 style dense models (see [Compatibility](Compatibility)) on the CPU kernels. The C API does **not** currently select the Vulkan backend. Positions are explicit: pass the position of the first token you feed, and call `hk_engine_reset_cache` before a new conversation. Handles are not thread-safe; `hk_engine_last_error` is global.

Minimal C inference sketch:

```c
#include "hk.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    hk_engine_t* e = hk_engine_load_from_file("model.hk");
    if (!e) { char msg[512]; hk_engine_last_error(msg, sizeof msg); fprintf(stderr, "%s\n", msg); return 1; }
    uint32_t vocab = hk_engine_get_vocab_size(e);
    float* logits = malloc(vocab * sizeof *logits);
    uint32_t prompt[] = {1, 2, 3};                       /* ids from hk_tokenizer_encode */
    if (hk_engine_forward_tokens(e, prompt, 3, 0, logits) == 0) {
        uint32_t next = hk_sample_token(logits, vocab, 0.0f, 0, 1.0f, 0.0f, 1.0f, prompt, 3, 0);
        printf("next token: %u\n", next);
    }
    free(logits);
    hk_engine_free(e);
    return 0;
}
```

`include/hk.hpp` provides C++20 wrappers (`hk::Model`, `hk::Tensor`, `hk::Writer`, and free functions for the math, growth and forward helpers) over the same reader/writer/helper functions.

---

## What each binding covers

All of them: open a file, tensor count and info (name, storage type, shape, layout, sparsity, block size), raw data / scales / residual bytes, dequantize to f32, typed metadata getters, sharding info. Rust, Go, C# and Java load the shared library; the TypeScript package parses the container itself.

| Language | Location | Notes |
| :--- | :--- | :--- |
| Rust | `bindings/rust/` (crate `hknt`; `build.rs` links `libhk`, set `HK_LIB_DIR` to its directory) | `HkModel::open`, `get_tensor`, `HkTensor::{dequantize, raw_data, as_raw_f32, ...}`, appendix entries and `rollback`, `patch_metadata_in_place`. |
| Go | `bindings/go/hk/` (cgo) | `hk.Open`, `Model.GetTensor`, `Tensor.Dequantize`, `GetMetadata*`, sharding info. |
| C# / .NET | `bindings/csharp/` (`DllImport`) | `HkModel` and tensor/appendix/hardware structs. |
| Java | `bindings/java/` (JNI) | `HkModel`, `HkTensor`, `HkWriter`. Needs the JNI glue library: `zig build jni -Djdk=/path/to/jdk` produces `libhkjni` next to `libhk`; put both on `java.library.path`. |
| TypeScript / JavaScript | `bindings/js/` (npm package `hkntf`, ES module) | Pure TypeScript parser over an `ArrayBuffer` (`HkModel.fromArrayBuffer`, `loadFromUrl`), plus `HkWriter` and a small pipeline-manifest helper. No native dependency, so it also works in browsers; it reads the whole file into memory rather than memory-mapping it. `dequantizeToF32` handles f32, f16, bf16, the integer types, NF4, and Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0; any other storage type (K-quants, I-quants, ...) throws instead of returning zeros. |

Example (Rust):

```rust
use hknt::HkModel;

fn main() -> Result<(), String> {
    let model = HkModel::open("model.hk")?;
    println!("{} tensors", model.tensor_count());
    if let Some(t) = model.get_tensor(0) {
        println!("{} {:?} {:?}", t.name(), t.storage_type(), t.shape());
        let f32s = t.dequantize(true)?;        // decoded copy
        println!("first value {}", f32s[0]);
    }
    println!("arch = {:?}", model.get_metadata_string("general.architecture"));
    Ok(())
}
```

Example (TypeScript):

```typescript
import { readFileSync } from "node:fs";
import { HkModel } from "hkntf";

const buf = readFileSync("model.hk");
const model = HkModel.fromArrayBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
console.log(model.tensors.size, "tensors; sharded:", model.isSharded);
console.log(model.metadata.get("general.architecture"));
```

Check each binding's source for exact signatures before depending on it. The C# package is published as `hknt` on NuGet.
