# HK Binary Format Specification (`.hk`)

> **Status:** this page is derived directly from `src/format.zig`, `src/tensor_toc.zig`, `src/metadata.zig`, `src/writer.zig` and `src/appendix.zig`. If it ever disagrees with the code, the code wins. All integers are little-endian.
>
> Format version: **1.0**. Readers accept any file whose `magic` matches and whose `version_major` is `1`.

---

## 1. File layout

```
+-----------------------------------------------------------+
| File header                       (128 bytes, fixed)      |
+-----------------------------------------------------------+
| Metadata key/value section        (variable)              |
+-----------------------------------------------------------+
| Tensor table of contents (TOC)    (variable length)       |
+-----------------------------------------------------------+
| Zero padding up to `alignment`                            |
+-----------------------------------------------------------+
| Tensor payloads, each start aligned to `alignment`        |
|   [data] [scales?] [residual?]  per tensor                |
+-----------------------------------------------------------+
| Appendix records (optional, append-only)                  |
+-----------------------------------------------------------+
```

Section offsets are not implied by the layout; readers must use the offsets stored in the header. The writer emits metadata directly after the header, then the TOC, then pads to the first aligned offset for tensor data.

---

## 2. File header (128 bytes)

| Offset | Field | Type | Meaning |
| :--- | :--- | :--- | :--- |
| `0x00` | `magic` | `[4]u8` | `48 4B 4E 54` (`"HKNT"`) |
| `0x04` | `version_major` | `u16` | `1` |
| `0x06` | `version_minor` | `u16` | `0` |
| `0x08` | `flags` | `u32` | See below |
| `0x0C` | `alignment` | `u16` | Payload alignment in bytes used by the writer |
| `0x0E` | `split_index` | `u16` | Shard index (0-based) |
| `0x10` | `tensor_count` | `u64` | Number of TOC entries |
| `0x18` | `metadata_kv_count` | `u64` | Number of metadata pairs |
| `0x20` | `metadata_offset` | `u64` | Absolute offset of the metadata section |
| `0x28` | `metadata_size` | `u64` | Metadata byte length |
| `0x30` | `tensor_toc_offset` | `u64` | Absolute offset of the TOC |
| `0x38` | `tensor_toc_size` | `u64` | TOC byte length |
| `0x40` | `tensor_data_offset` | `u64` | Absolute offset of the first aligned payload |
| `0x48` | `appendix_offset` | `u64` | Offset of the first appendix record, `0` if none |
| `0x50` | `checksum` | `u64` | Reserved. **Currently always written as `0` and not verified.** |
| `0x58` | `split_count` | `u16` | Total shards, `>= 1` |
| `0x5A` | `reserved1` | `[38]u8` | Zero |

### Flags

| Bit | Name | Meaning |
| :--- | :--- | :--- |
| 0 | `LITTLE_ENDIAN` | Always set |
| 1 | `HAS_APPENDIX` | Set when `appendix_offset != 0` |
| 2 | `HAS_QUANT_TABLE` | Reserved |
| 3 | `SPARSITY_2_4` | Reserved/informational |
| 4 | `TILE_ALIGNED` | Set when `alignment` is a multiple of 128 |
| 5 | `FLEXIBLE_ALIGNMENT` | Set when `alignment` is not a multiple of 128 |
| 6 | `IS_SHARDED` | File is one shard of a multi-file model |
| 7 | `RAW_WEIGHT_STORAGE` | Informational: raw unquantized storage |
| 8 | `UNIVERSAL_PAGE_ALIGNED` | Set when `alignment >= 4096` |

Alignment presets in `format.zig`: 128 B (default), 4096 B (page), 16384 B (Apple Silicon page size) and 65536 B (64 KiB granularity). The alignment only affects where payloads start; it does not change how they are read. The inference engine memory-maps payloads in place, so a page-aligned file lets the OS map tensors without copies, but nothing in the engine requires a particular value.

---

## 3. Metadata section

A sequence of `metadata_kv_count` entries, each:

| Field | Type |
| :--- | :--- |
| `key_len` | `u16` |
| `key` | `key_len` bytes (UTF-8) |
| `type` | `u8` |
| `value_len` | `u32` |
| `value` | `value_len` bytes |

Value types:

| Tag | Name | Encoding |
| :--- | :--- | :--- |
| `0x01` | string | UTF-8 bytes |
| `0x02` | int64 | 8 bytes, little-endian |
| `0x03` | float64 | 8 bytes, IEEE-754 |
| `0x04` | bool | 1 byte |
| `0x05` | json | UTF-8 JSON text |
| `0x06` | bytes | opaque |

Model hyperparameters, tokenizer data and chat templates live here (for converted GGUF/safetensors models the keys follow the GGUF naming such as `general.architecture` and `<arch>.block_count`; see `src/convert/`).

---

## 4. Tensor table of contents

`tensor_count` variable-length entries, back to back. There is **no fixed entry size**.

| Field | Type | Notes |
| :--- | :--- | :--- |
| `name_len` | `u16` | Max name length is 128 |
| `name` | bytes | |
| `storage_type` | `u8` | See section 5 |
| `tile_layout` | `u8` | See section 6 |
| `sparsity_type` | `u8` | See section 7 |
| `ndim` | `u8` | At most 8 |
| `shape` | `ndim × u64` | Only `ndim` dimensions are stored |
| `data_offset` | `u64` | Absolute offset of the payload |
| `data_size` | `u64` | Payload bytes |
| `residual_offset` | `u64` | `0` if none |
| `residual_size` | `u64` | |
| `scale_offset` | `u64` | `0` if none |
| `scale_size` | `u64` | |
| `block_size` | `u16` | Quantization block size (default 32) |
| `sparsity_ratio` | `u32` | `f32` bit pattern |

Special cases handled by the writer:

- `null_ref`: `data_offset = 0`, `data_size = 0`; no bytes are stored.
- `shared_ref`: copies the offsets of an earlier tensor (tied weights), so both names point at the same bytes.
- Each of `data`, `scales` and `residual` begins at its own aligned offset.

---

## 5. Storage types (`storage_type`)

| Value | Name | Notes |
| :--- | :--- | :--- |
| `0x00`–`0x0E` | `f32, f16, bf16, fp8_e4m3, fp8_e5m2, int8, int32, int64, uint8, bool, int16, uint16, uint32, uint64, f64` | Raw dense types, in that order |
| `0x10`–`0x14` | `dq4, dq8, dq6, dq12, dqt` | HK "dual-mode" quantization containers |
| `0x15`–`0x1A` | `q4_0, q8_0, q4_1, q5_0, q5_1, q8_1` | GGUF-compatible legacy block formats |
| `0x20`–`0x23` | `sparse_f16, sparse_dq8, sparse_2_4, sparse_dq4_2_4` | Sparse storage |
| `0x30`–`0x32` | `null_ref, shared_ref, lora_ref` | Virtual references |
| `0x40`–`0x45` | `q2_k … q6_k, q8_k` | K-quants, 256-weight super-blocks |
| `0x50`–`0x58` | `iq1_s, iq1_m, iq2_xxs, iq2_xs, iq3_xxs, iq4_nl, iq4_xs, iq2_s, iq3_s` | I-quants |
| `0x60`–`0x63` | `tq1_0, tq2_0, mxfp4, nvfp4` | Ternary and microscaling |

**Which of these the inference engine runs** is a separate question from which can be stored. The engine's quantized kernels (`vecdot.supported`) cover `f32`, `f16`, `bf16`, the legacy GGUF block formats (`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `iq4_nl`), all K-quants except `q8_k`, the I-quants, `tq1_0`/`tq2_0`, `mxfp4` and `nvfp4`. A model containing any other weight type fails to load with an error rather than producing garbage. The reader (`HKReader.dequantizeToF32`) can additionally decode `dq4`, `dq8`, `sparse_f16`, `sparse_2_4`, bitmask/CSR sparsity and `null_ref` to f32, which is what the conversion, editing and training tools use; the engine does not run those directly. See [Compatibility](Compatibility).

---

## 6. Tile layouts (`tile_layout`)

`0x00 row_major`, `0x01 col_major`, `0x02 tile_16x16`, `0x03 tile_16x8`, `0x04 tile_32x16`, `0x05 block_sparse_2_4`, `0x06 tile_32x32`, `0x07 tile_64x64`.

Files produced by the converters use `row_major`. The engine does its own register-tile repacking at run time (per thread, from the mmap'd weights), so it does not read pre-tiled layouts.

## 7. Sparsity types (`sparsity_type`)

`0x00 none`, `0x01 bitmask`, `0x02 csr`, `0x03 structured_2_4`, `0x04 physical_pruned`, `0x05 bsr`. Converted models use `none`.

---

## 8. Appendix region

If `appendix_offset != 0`, records are laid out back to back starting there until end of file. Appending a record never moves existing bytes: the tool pads the file to 8 bytes, writes the record, and (only for the first record) patches `appendix_offset` and the `HAS_APPENDIX` flag in the header.

Each record:

```
AppendixRecordHeader   (80 bytes)
name                   (name_len bytes)
target                 (target_len bytes)
data                   (data_size bytes)
zero padding to a multiple of 8 bytes
```

Header layout:

| Offset | Field | Type |
| :--- | :--- | :--- |
| 0 | `entry_type` | `u8` |
| 1 | `flags` | `u8` (bit0 ACTIVE, bit1 COMPRESSED) |
| 2 | `name_len` | `u16` |
| 4 | `generation` | `u32` |
| 8 | `timestamp` | `u64` |
| 16 | `parent_hash` | `[32]u8` |
| 48 | `metric_loss, metric_acc, metric_pass, metric_custom` | `4 × f32` |
| 64 | `target_len` | `u16` |
| 66 | `reserved` | `u16` |
| 68 | `data_crc32` | `u32`, currently always `0` |
| 72 | `data_size` | `u64` |

Entry types: `0x01 lora_adapter`, `0x02 delta_patch`, `0x03 new_layer`, `0x04 code_eval`, `0x05 kv_cache_sink`, `0x06 topology_head`.

### Lineage hash

`parent_hash` of record *n* is the SHA-256 of record *n-1*'s `name ‖ target ‖ data`. The verifier also accepts the older form, SHA-256 of the payload only. The first record's `parent_hash` is not checked. See [In-Container Version Lineage](In-Container-Version-Lineage).

> The inference engine does **not** apply appendix records when running a model; it reads only the base tensors. Appendix handling lives in the lineage/training tooling.

---

## 9. Reading a file

1. Map the file; check `magic` and `version_major`.
2. Parse `metadata_kv_count` pairs at `metadata_offset`.
3. Parse `tensor_count` entries at `tensor_toc_offset`.
4. Tensor `i` bytes are `file[data_offset .. data_offset + data_size]`; scales and residual likewise.
5. If `appendix_offset != 0`, parse records from there.

Readers should bounds-check every offset and size against the file length; `src/reader.zig` does.
