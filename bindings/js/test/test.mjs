// Reads the fixture written by tests/bindings/c_abi.c, then writes a file with HkWriter and reads
// it back. Run with HK_FIXTURE=<fixture.hk>. If HK_CLI points at the hk executable, the written
// file is also checked with `hk verify`.
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { HkModel, HkWriter, StorageType, TileLayout, SparsityType } from "../dist/hk.js";

let failures = 0;
const check = (ok, what) => {
  if (!ok) {
    failures++;
    console.error("FAIL " + what);
  }
};

const fixture = process.env.HK_FIXTURE;
if (!fixture) {
  console.log("HK_FIXTURE not set, skipping");
  process.exit(0);
}

const toArrayBuffer = (buf) => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);

const m = HkModel.fromArrayBuffer(toArrayBuffer(readFileSync(fixture)));
check(m.tensors.size === 2, "tensor count");
const t = m.getTensor("w.f32");
check(t && t.storageType === StorageType.F32 && t.shape.join(",") === "2,3", "w.f32 entry");
const f = m.dequantizeToF32(t);
check(f[0] === 1 && f[5] === 6, "f32 values");
const q = m.getTensor("w.q8");
check(q && q.storageType === StorageType.Q8_0, "q8 entry");
check(Math.abs(m.dequantizeToF32(q)[0] + 4) < 0.02, "q8 values");
check(m.metadata.get("general.name") === "fixture", "string metadata");
check(m.metadata.get("answer") === 42, "int metadata");
check(m.metadata.get("pi") === 3.5, "float metadata");
check(m.metadata.get("flag") === true, "bool metadata");
check(m.alignment === 4096 && !m.isSharded && m.splitCount === 1, "header");
check(m.appendixEntries.length === 1, "appendix count");
const e = m.appendixEntries[0];
check(e.name === "gen1" && e.target === "w.f32" && e.generation === 1 && e.dataSize === 13, "appendix entry");

const w = new HkWriter(128);
w.addMetadataString("k", "v");
w.addMetadataInt("n", 7);
w.addTensor("t", StorageType.F32, TileLayout.RowMajor, SparsityType.None, [2], new Uint8Array(new Float32Array([1.5, 2.5]).buffer));
const bytes = w.build();
const back = HkModel.fromArrayBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
check(back.metadata.get("k") === "v" && back.metadata.get("n") === 7, "written metadata");
const v = back.dequantizeToF32(back.getTensor("t"));
check(v[0] === 1.5 && v[1] === 2.5, "written values");

// patchMetadataInPlace: growing a value inside the padding must leave the tensors readable.
{
  const pw = new HkWriter(4096);
  pw.addMetadataString("k", "v");
  pw.addTensor("t", StorageType.F32, TileLayout.RowMajor, SparsityType.None, [2], new Uint8Array(new Float32Array([1.5, 2.5]).buffer));
  const pb = pw.build();
  const pm = HkModel.fromArrayBuffer(pb.buffer.slice(pb.byteOffset, pb.byteOffset + pb.byteLength));
  check(pm.patchMetadataInPlace("k", "a much longer value than before"), "patch fits in the padding");
  const re = HkModel.fromArrayBuffer(pm.getArrayBuffer());
  check(re.metadata.get("k") === "a much longer value than before", "patched metadata");
  check(re.dequantizeToF32(re.getTensor("t"))[1] === 2.5, "tensors survive the patch");
}

if (process.env.HK_CLI) {
  const dir = mkdtempSync(join(tmpdir(), "hk-js-"));
  const path = join(dir, "w.hk");
  writeFileSync(path, bytes);
  const r = spawnSync(process.env.HK_CLI, ["verify", path], { encoding: "utf8" });
  const out = r.stdout + r.stderr; // the CLI reports on stderr
  check(out.includes("ALL CHECKS PASSED"), "hk verify accepts the file written by HkWriter");
}

console.log(failures === 0 ? "js ok" : failures + " failures");
process.exit(failures === 0 ? 0 : 1);
