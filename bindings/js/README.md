# hkntf: TypeScript / JavaScript reader and writer for `.hk` files

A dependency-free ES module that parses and builds HK model containers in Node.js, browsers and web workers. It is a **container** library: it reads tensors, metadata and the appendix, and writes new files. It does not run models.

The whole file is read into an `ArrayBuffer` (no memory mapping), so it suits headers, metadata, small models and tooling rather than multi-gigabyte weights.

## Install

```bash
npm install hkntf
```

## Read a file

```typescript
import { readFileSync } from "node:fs";
import { HkModel, StorageType } from "hkntf";

const buf = readFileSync("model.hk");
const model = HkModel.fromArrayBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));

console.log(model.metadata.get("general.architecture"));
for (const [name, t] of model.tensors) {
  console.log(name, t.shape, StorageType[t.storageType]);
}

const entry = model.getTensor("token_embd.weight");
if (entry) {
  const values = model.dequantizeToF32(entry);   // Float32Array
  const raw = model.getRawTensorBytes(entry);    // Uint8Array view, no copy
}
```

In a browser use `await HkModel.loadFromUrl(url)`.

`dequantizeToF32` decodes f32, f16, bf16, the integer types, NF4, and the GGUF block formats Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0. Any other storage type (K-quants, I-quants, ternary, MXFP4, sparse formats) **throws**; use the native library for those.

Other members: `model.appendixEntries`, `model.alignment`, `model.isSharded`, `model.splitIndex`, `model.splitCount`, `model.patchMetadataInPlace(key, value)`.

## Write a file

```typescript
import { HkWriter, StorageType, TileLayout, SparsityType } from "hkntf";

const w = new HkWriter(4096);                       // payload alignment in bytes
w.addMetadataString("general.name", "demo");
w.addMetadataInt("answer", 42);
w.addTensor("w", StorageType.F32, TileLayout.RowMajor, SparsityType.None, [2],
            new Uint8Array(new Float32Array([1.5, 2.5]).buffer));
const bytes: Uint8Array = w.build();
```

## Tests

`npm test` builds the package and runs `test/test.mjs`, which needs a fixture written by `tests/bindings/c_abi.c`; from the repository root `tests/bindings/run.sh` does all of that.

## License

Apache-2.0
