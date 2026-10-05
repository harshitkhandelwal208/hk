/**
 * HK TypeScript / JavaScript / WASM SDK
 * Zero-dependency pure TypeScript parser, builder, and WebAssembly interface for browsers and Node.js.
 */

export enum StorageType {
  F32 = 0x00,
  F16 = 0x01,
  BF16 = 0x02,
  FP8_E4M3 = 0x03,
  FP8_E5M2 = 0x04,
  INT8 = 0x05,
  INT32 = 0x06,
  INT64 = 0x07,
  UINT8 = 0x08,
  BOOL = 0x09,
  INT16 = 0x0A,
  UINT16 = 0x0B,
  UINT32 = 0x0C,
  UINT64 = 0x0D,
  F64 = 0x0E,
  DQ4 = 0x10,
  NF4 = 0x10,
  DQ8 = 0x11,
  DQ6 = 0x12,
  DQ12 = 0x13,
  DQT = 0x14,
  Q4_0 = 0x15,
  Q8_0 = 0x16,
  Q4_1 = 0x17,
  Q5_0 = 0x18,
  Q5_1 = 0x19,
  Q8_1 = 0x1a,
  SparseF16 = 0x20,
  SparseDQ8 = 0x21,
  Sparse24 = 0x22,
  SparseDQ4_2_4 = 0x23,
  NullRef = 0x30,
  SharedRef = 0x31,
  LoRARef = 0x32,
  Q2_K = 0x40,
  Q3_K = 0x41,
  Q4_K = 0x42,
  Q5_K = 0x43,
  Q6_K = 0x44,
  Q8_K = 0x45,
  IQ1_S = 0x50,
  IQ1_M = 0x51,
  IQ2_XXS = 0x52,
  IQ2_XS = 0x53,
  IQ3_XXS = 0x54,
  IQ4_NL = 0x55,
  IQ4_XS = 0x56,
  IQ2_S = 0x57,
  IQ3_S = 0x58,
  TQ1_0 = 0x60,
  TQ2_0 = 0x61,
  MXFP4 = 0x62,
  NVFP4 = 0x63,
}

export const DEFAULT_ALIGNMENT_BYTES = 128;
export const UNIVERSAL_PAGE_ALIGNMENT_BYTES = 4096;
export const APPLE_SILICON_ALIGNMENT_BYTES = 16384;
export const DIRECT_DMA_ALIGNMENT_BYTES = 65536;

export enum TileLayout {
  ROW_MAJOR = 0x00,
  COL_MAJOR = 0x01,
  TILE_16X16 = 0x02,
  TILE_16X8 = 0x03,
  TILE_32X16 = 0x04,
  BLOCK_SPARSE_2_4 = 0x05,
  TILE_32X32 = 0x06,
  TILE_64X64 = 0x07,
}

export enum SparsityType {
  NONE = 0x00,
  BITMASK = 0x01,
  CSR = 0x02,
  STRUCTURED_2_4 = 0x03,
  PHYSICAL_PRUNED = 0x04,
  BSR = 0x05,
}

export enum AppendixType {
  LORA_ADAPTER = 0x01,
  DELTA_PATCH = 0x02,
  NEW_LAYER = 0x03,
  CODE_EVAL = 0x04,
  KV_CACHE_SINK = 0x05,
  TOPOLOGY_HEAD = 0x06,
}

export const HeaderFlags = {
  LITTLE_ENDIAN: 1 << 0,
  HAS_APPENDIX: 1 << 1,
  HAS_QUANT_TABLE: 1 << 2,
  SPARSITY_2_4: 1 << 3,
  TILE_ALIGNED: 1 << 4,
  FLEXIBLE_ALIGNMENT: 1 << 5,
  IS_SHARDED: 1 << 6,
  RAW_WEIGHT_STORAGE: 1 << 7,
  UNIVERSAL_PAGE_ALIGNED: 1 << 8,
} as const;

// 16 NF4 standard codebook values
const NF4_CODEBOOK = new Float32Array([
  -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
  -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
  0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
  0.4407079219818115, 0.5626170039176941, 0.7229568362236023, 1.0,
]);

export interface TensorEntry {
  name: string;
  storageType: StorageType;
  tileLayout: TileLayout;
  sparsityType: SparsityType;
  shape: number[];
  dataOffset: number;
  dataSize: number;
  residualOffset: number;
  residualSize: number;
  scaleOffset: number;
  scaleSize: number;
  blockSize: number;
  sparsityRatio: number;
}

export interface AppendixEntry {
  entryType: AppendixType;
  flags: number;
  generation: number;
  timestamp: bigint;
  parentHash: Uint8Array;
  metricLoss: number;
  metricAcc: number;
  metricPass: number;
  metricCustom: number;
  name: string;
  target: string;
  dataOffset: number;
  dataSize: number;
}

/** IEEE half precision to number. */
function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

export class HkModel {
  private buffer: ArrayBuffer;
  private view: DataView;
  public metadata: Map<string, any> = new Map();
  public tensors: Map<string, TensorEntry> = new Map();
  public appendixEntries: AppendixEntry[] = [];
  public alignment: number = 128;
  public flags: number = 0;
  public splitIndex: number = 0;
  public splitCount: number = 1;
  public isSharded: boolean = false;

  private constructor(buffer: ArrayBuffer) {
    this.buffer = buffer;
    this.view = new DataView(buffer);
    this.parseHeaderAndTOC();
  }

  public static async loadFromUrl(url: string): Promise<HkModel> {
    const res = await fetch(url);
    const buf = await res.arrayBuffer();
    return new HkModel(buf);
  }

  public static fromArrayBuffer(buffer: ArrayBuffer): HkModel {
    return new HkModel(buffer);
  }

  public getArrayBuffer(): ArrayBuffer {
    return this.buffer;
  }

  private parseHeaderAndTOC(): void {
    const magic = String.fromCharCode(
      this.view.getUint8(0),
      this.view.getUint8(1),
      this.view.getUint8(2),
      this.view.getUint8(3)
    );
    if (magic !== "HKNT") {
      throw new Error(`Invalid HK file magic: ${magic}`);
    }

    const versionMajor = this.view.getUint16(4, true);
    if (versionMajor !== 1) {
      throw new Error(`Unsupported version: ${versionMajor}`);
    }

    this.flags = this.view.getUint32(8, true);
    this.alignment = this.view.getUint16(12, true);
    this.splitIndex = this.view.getUint16(14, true);
    const tensorCount = Number(this.view.getBigUint64(16, true));
    const metaCount = Number(this.view.getBigUint64(24, true));
    const metaOffset = Number(this.view.getBigUint64(32, true));
    const metaSize = Number(this.view.getBigUint64(40, true));
    const tocOffset = Number(this.view.getBigUint64(48, true));
    const tocSize = Number(this.view.getBigUint64(56, true));
    const appendixOffset = Number(this.view.getBigUint64(72, true));
    this.splitCount = this.view.getUint16(88, true);
    if (this.splitCount === 0) {
      this.splitCount = 1;
    }
    this.isSharded = (this.flags & HeaderFlags.IS_SHARDED) !== 0;

    // Parse Metadata
    if (metaSize > 0) {
      let mPos = metaOffset;
      const decoder = new TextDecoder();
      for (let i = 0; i < metaCount; i++) {
        const kLen = this.view.getUint16(mPos, true);
        mPos += 2;
        const key = decoder.decode(new Uint8Array(this.buffer, mPos, kLen));
        mPos += kLen;
        const tag = this.view.getUint8(mPos);
        mPos += 1;
        const vLen = this.view.getUint32(mPos, true);
        mPos += 4;

        if (tag === 0x01) {
          this.metadata.set(key, decoder.decode(new Uint8Array(this.buffer, mPos, vLen)));
        } else if (tag === 0x02) {
          this.metadata.set(key, Number(this.view.getBigInt64(mPos, true)));
        } else if (tag === 0x03) {
          this.metadata.set(key, this.view.getFloat64(mPos, true));
        } else if (tag === 0x04) {
          this.metadata.set(key, this.view.getUint8(mPos) !== 0);
        } else if (tag === 0x05) {
          const jsonStr = decoder.decode(new Uint8Array(this.buffer, mPos, vLen));
          try {
            this.metadata.set(key, JSON.parse(jsonStr));
          } catch {
            this.metadata.set(key, jsonStr);
          }
        } else if (tag === 0x06) {
          this.metadata.set(key, new Uint8Array(this.buffer.slice(mPos, mPos + vLen)));
        }
        mPos += vLen;
      }
    }

    // Parse TOC
    let tPos = tocOffset;
    const decoder = new TextDecoder();
    for (let i = 0; i < tensorCount; i++) {
      const nLen = this.view.getUint16(tPos, true);
      tPos += 2;
      const name = decoder.decode(new Uint8Array(this.buffer, tPos, nLen));
      tPos += nLen;

      const stype = this.view.getUint8(tPos);
      const tlayout = this.view.getUint8(tPos + 1);
      const sparseType = this.view.getUint8(tPos + 2);
      const ndim = this.view.getUint8(tPos + 3);
      tPos += 4;

      const shape: number[] = [];
      for (let d = 0; d < ndim; d++) {
        shape.push(Number(this.view.getBigUint64(tPos, true)));
        tPos += 8;
      }

      const dOff = Number(this.view.getBigUint64(tPos, true));
      const dSz = Number(this.view.getBigUint64(tPos + 8, true));
      const rOff = Number(this.view.getBigUint64(tPos + 16, true));
      const rSz = Number(this.view.getBigUint64(tPos + 24, true));
      const sOff = Number(this.view.getBigUint64(tPos + 32, true));
      const sSz = Number(this.view.getBigUint64(tPos + 40, true));
      const bSz = this.view.getUint16(tPos + 48, true);
      const sRatio = this.view.getFloat32(tPos + 50, true);
      tPos += 54;

      this.tensors.set(name, {
        name,
        storageType: stype as StorageType,
        tileLayout: tlayout as TileLayout,
        sparsityType: sparseType as SparsityType,
        shape,
        dataOffset: dOff,
        dataSize: dSz,
        residualOffset: rOff,
        residualSize: rSz,
        scaleOffset: sOff,
        scaleSize: sSz,
        blockSize: bSz,
        sparsityRatio: sRatio,
      });
    }

    // Parse Appendix Records if present
    if (appendixOffset > 0 && appendixOffset + 80 <= this.buffer.byteLength) {
      let aPos = appendixOffset;
      while (aPos + 80 <= this.buffer.byteLength) {
        const entryType = this.view.getUint8(aPos);
        if (entryType === 0) break;
        const flags = this.view.getUint8(aPos + 1);
        const nameLen = this.view.getUint16(aPos + 2, true);
        const gen = this.view.getUint32(aPos + 4, true);
        const timestamp = this.view.getBigUint64(aPos + 8, true);
        const parentHash = new Uint8Array(this.buffer.slice(aPos + 16, aPos + 48));
        const metricLoss = this.view.getFloat32(aPos + 48, true);
        const metricAcc = this.view.getFloat32(aPos + 52, true);
        const metricPass = this.view.getFloat32(aPos + 56, true);
        const metricCustom = this.view.getFloat32(aPos + 60, true);
        const targetLen = this.view.getUint16(aPos + 64, true);
        const dataSize = Number(this.view.getBigUint64(aPos + 72, true));
        aPos += 80;

        const name = decoder.decode(new Uint8Array(this.buffer, aPos, nameLen));
        aPos += nameLen;
        const target = targetLen > 0 ? decoder.decode(new Uint8Array(this.buffer, aPos, targetLen)) : "";
        aPos += targetLen;

        const dataOffset = aPos;
        aPos += dataSize;
        aPos = Math.ceil(aPos / 8) * 8; // records are padded to 8 bytes

        this.appendixEntries.push({
          entryType: entryType as AppendixType,
          flags,
          generation: gen,
          timestamp,
          parentHash,
          metricLoss,
          metricAcc,
          metricPass,
          metricCustom,
          name,
          target,
          dataOffset,
          dataSize,
        });
      }
    }
  }

  public getTensor(name: string): TensorEntry | undefined {
    return this.tensors.get(name);
  }

  public dequantizeToF32(entry: TensorEntry, withResidual = true): Float32Array {
    let totalElements = 1;
    for (const d of entry.shape) totalElements *= d;
    const output = new Float32Array(totalElements);

    if (entry.storageType === StorageType.NullRef) {
      return output;
    }

    if (entry.storageType === StorageType.F32) {
      const src = new Float32Array(this.buffer, entry.dataOffset, totalElements);
      output.set(src);
      return output;
    }

    if (entry.storageType === StorageType.DQ4) {
      const packed = new Uint8Array(this.buffer, entry.dataOffset, entry.dataSize);
      const scales = new Float32Array(this.buffer, entry.scaleOffset, entry.scaleSize / 4);
      const blockSize = entry.blockSize || 32;

      let elemIdx = 0;
      for (let i = 0; i < packed.length && elemIdx < totalElements; i++) {
        const byte = packed[i];
        const low = byte & 0x0f;
        const high = (byte >> 4) & 0x0f;

        const blkIdx = Math.floor(elemIdx / blockSize);
        const scale = scales[blkIdx] || 1.0;
        output[elemIdx++] = NF4_CODEBOOK[low] * scale;

        if (elemIdx < totalElements) {
          const blkIdx2 = Math.floor(elemIdx / blockSize);
          const scale2 = scales[blkIdx2] || 1.0;
          output[elemIdx++] = NF4_CODEBOOK[high] * scale2;
        }
      }

      // Add residual if requested
      if (withResidual && entry.residualSize > 0) {
        const res = new Float32Array(this.buffer, entry.residualOffset, entry.residualSize / 4);
        for (let i = 0; i < totalElements && i < res.length; i++) {
          output[i] += res[i];
        }
      }
      return output;
    }

    if (entry.storageType === StorageType.BF16) {
      const u16 = new Uint16Array(this.buffer, entry.dataOffset, totalElements);
      const u32 = new Uint32Array(1);
      const f32 = new Float32Array(u32.buffer);
      for (let i = 0; i < totalElements; i++) {
        u32[0] = u16[i] << 16;
        output[i] = f32[0];
      }
      return output;
    }

    if (entry.storageType === StorageType.INT8) {
      const src = new Int8Array(this.buffer, entry.dataOffset, totalElements);
      for (let i = 0; i < totalElements; i++) output[i] = src[i];
      return output;
    }

    if (entry.storageType === StorageType.UINT8) {
      const src = new Uint8Array(this.buffer, entry.dataOffset, totalElements);
      for (let i = 0; i < totalElements; i++) output[i] = src[i];
      return output;
    }

    if (entry.storageType === StorageType.INT16) {
      const src = new Int16Array(this.buffer, entry.dataOffset, totalElements);
      for (let i = 0; i < totalElements; i++) output[i] = src[i];
      return output;
    }

    if (entry.storageType === StorageType.INT32) {
      const src = new Int32Array(this.buffer, entry.dataOffset, totalElements);
      for (let i = 0; i < totalElements; i++) output[i] = src[i];
      return output;
    }

    if (entry.storageType === StorageType.F64) {
      const src = new Float64Array(this.buffer, entry.dataOffset, totalElements);
      for (let i = 0; i < totalElements; i++) output[i] = Number(src[i]);
      return output;
    }

    if (entry.storageType === StorageType.F16) {
      for (let i = 0; i < totalElements; i++) output[i] = halfToFloat(this.view.getUint16(entry.dataOffset + i * 2, true));
      return output;
    }

    if (this.dequantizeLegacyBlocks(entry, totalElements, output)) {
      return output;
    }

    // Anything else (K-quants, I-quants, ternary, MXFP4, sparse formats, ...) is not decoded in
    // TypeScript. Failing is better than returning zeros that look like real weights.
    throw new Error(
      `dequantizeToF32: storage type ${StorageType[entry.storageType] ?? entry.storageType} is not supported by the TypeScript reader; use the native library`
    );
  }

  /** Decodes the GGUF-compatible 32-element block formats. Returns false for other types. */
  private dequantizeLegacyBlocks(entry: TensorEntry, total: number, out: Float32Array): boolean {
    const v = this.view;
    const base = entry.dataOffset;
    const blocks = Math.ceil(total / 32);
    const put = (b: number, j: number, x: number) => {
      const idx = b * 32 + j;
      if (idx < total) out[idx] = x;
    };
    switch (entry.storageType) {
      case StorageType.Q8_0:
        for (let b = 0; b < blocks; b++) {
          const o = base + b * 34;
          const d = halfToFloat(v.getUint16(o, true));
          for (let j = 0; j < 32; j++) put(b, j, v.getInt8(o + 2 + j) * d);
        }
        return true;
      case StorageType.Q4_0:
        for (let b = 0; b < blocks; b++) {
          const o = base + b * 18;
          const d = halfToFloat(v.getUint16(o, true));
          for (let j = 0; j < 16; j++) {
            const q = v.getUint8(o + 2 + j);
            put(b, j, ((q & 0x0f) - 8) * d);
            put(b, j + 16, ((q >> 4) - 8) * d);
          }
        }
        return true;
      case StorageType.Q4_1:
        for (let b = 0; b < blocks; b++) {
          const o = base + b * 20;
          const d = halfToFloat(v.getUint16(o, true));
          const m = halfToFloat(v.getUint16(o + 2, true));
          for (let j = 0; j < 16; j++) {
            const q = v.getUint8(o + 4 + j);
            put(b, j, (q & 0x0f) * d + m);
            put(b, j + 16, (q >> 4) * d + m);
          }
        }
        return true;
      case StorageType.Q5_0:
        for (let b = 0; b < blocks; b++) {
          const o = base + b * 22;
          const d = halfToFloat(v.getUint16(o, true));
          const qh = v.getUint32(o + 2, true);
          for (let j = 0; j < 16; j++) {
            const q = v.getUint8(o + 6 + j);
            const lo = (q & 0x0f) | (((qh >>> j) << 4) & 0x10);
            const hi = (q >> 4) | ((qh >>> (j + 12)) & 0x10);
            put(b, j, (lo - 16) * d);
            put(b, j + 16, (hi - 16) * d);
          }
        }
        return true;
      case StorageType.Q5_1:
        for (let b = 0; b < blocks; b++) {
          const o = base + b * 24;
          const d = halfToFloat(v.getUint16(o, true));
          const m = halfToFloat(v.getUint16(o + 2, true));
          const qh = v.getUint32(o + 4, true);
          for (let j = 0; j < 16; j++) {
            const q = v.getUint8(o + 8 + j);
            const lo = (q & 0x0f) | (((qh >>> j) << 4) & 0x10);
            const hi = (q >> 4) | ((qh >>> (j + 12)) & 0x10);
            put(b, j, lo * d + m);
            put(b, j + 16, hi * d + m);
          }
        }
        return true;
      default:
        return false;
    }
  }

  public isRawWeightStorage(): boolean {
    return (this.flags & HeaderFlags.RAW_WEIGHT_STORAGE) !== 0;
  }

  public isUniversalPageAligned(): boolean {
    return (this.flags & HeaderFlags.UNIVERSAL_PAGE_ALIGNED) !== 0;
  }

  public isTensorCoreAligned(): boolean {
    return (this.alignment % 128) === 0;
  }

  public getRawTensorBytes(entry: TensorEntry): Uint8Array {
    return new Uint8Array(this.buffer, entry.dataOffset, entry.dataSize);
  }

  /**
   * Updates or inserts a metadata string in-place directly in the container buffer without moving tensor data.
   * Succeeds if there is enough space between the header and tensor_data_offset.
   */
  public patchMetadataInPlace(key: string, val: string): boolean {
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();

    // Read current header offsets
    const metaOffset = Number(this.view.getBigUint64(32, true));
    const metaSize = Number(this.view.getBigUint64(40, true));
    const tocOffset = Number(this.view.getBigUint64(48, true));
    const tocSize = Number(this.view.getBigUint64(56, true));
    const dataOffset = Number(this.view.getBigUint64(64, true));
    const headerSize = 128;

    // Build updated map of metadata raw entries
    const items: Array<{ key: string; type: number; valBytes: Uint8Array }> = [];
    let updated = false;

    if (metaSize > 0) {
      let mPos = metaOffset;
      const metaCount = Number(this.view.getBigUint64(24, true));
      for (let i = 0; i < metaCount; i++) {
        const kLen = this.view.getUint16(mPos, true);
        mPos += 2;
        const curKey = decoder.decode(new Uint8Array(this.buffer, mPos, kLen));
        mPos += kLen;
        const tag = this.view.getUint8(mPos);
        mPos += 1;
        const vLen = this.view.getUint32(mPos, true);
        mPos += 4;

        if (curKey === key) {
          const newVBytes = encoder.encode(val);
          items.push({ key: curKey, type: 0x01, valBytes: newVBytes });
          updated = true;
        } else {
          const vSlice = new Uint8Array(this.buffer.slice(mPos, mPos + vLen));
          items.push({ key: curKey, type: tag, valBytes: vSlice });
        }
        mPos += vLen;
      }
    }

    if (!updated) {
      items.push({ key, type: 0x01, valBytes: encoder.encode(val) });
    }

    // Serialize new metadata
    let newMetaSize = 0;
    const serializedParts: Uint8Array[] = [];
    for (const item of items) {
      const kBytes = encoder.encode(item.key);
      const kHead = new Uint8Array(2);
      new DataView(kHead.buffer).setUint16(0, kBytes.length, true);
      const tag = new Uint8Array([item.type]);
      const vHead = new Uint8Array(4);
      new DataView(vHead.buffer).setUint32(0, item.valBytes.length, true);

      serializedParts.push(kHead, kBytes, tag, vHead, item.valBytes);
      newMetaSize += 2 + kBytes.length + 1 + 4 + item.valBytes.length;
    }

    // Verify layout fits in padding before tensor_data_offset
    const newTocOffset = headerSize + newMetaSize;
    if (newTocOffset + tocSize > dataOffset) {
      return false; // Insufficient space before tensor payload
    }

    // Copy existing TOC
    const tocBytes = new Uint8Array(this.buffer.slice(tocOffset, tocOffset + tocSize));

    // Write new metadata starting at headerSize (128)
    const bufView = new Uint8Array(this.buffer);
    let writePos = headerSize;
    for (const part of serializedParts) {
      bufView.set(part, writePos);
      writePos += part.length;
    }

    // Write TOC immediately after new metadata
    bufView.set(tocBytes, newTocOffset);

    // Update Header
    this.view.setBigUint64(24, BigInt(items.length), true);
    this.view.setBigUint64(32, BigInt(headerSize), true);
    this.view.setBigUint64(40, BigInt(newMetaSize), true);
    this.view.setBigUint64(48, BigInt(newTocOffset), true);

    // Update cached metadata map
    this.metadata.set(key, val);
    return true;
  }
}

/**
 * HK Container Builder / Writer for TypeScript & Node.js
 */
export class HkWriter {
  private alignment: number;
  private splitIndex: number = 0;
  private splitCount: number = 1;
  private isSharded: boolean = false;
  private rawWeightStorage: boolean = false;
  private metadata: Array<{ key: string; type: number; val: any }> = [];
  private tensors: Array<{
    name: string;
    storageType: StorageType;
    tileLayout: TileLayout;
    sparsityType: SparsityType;
    shape: number[];
    data: Uint8Array;
    sparsityRatio: number;
  }> = [];

  constructor(alignment = 128) {
    this.alignment = alignment;
  }

  public setAlignment(alignment: number): void {
    this.alignment = alignment;
  }

  public setRawWeightStorage(enabled: boolean): void {
    this.rawWeightStorage = enabled;
  }

  public setSharding(splitIndex: number, splitCount: number): void {
    this.splitIndex = splitIndex;
    this.splitCount = splitCount;
    this.isSharded = splitCount > 1;
  }

  public addMetadataString(key: string, val: string): void {
    this.metadata.push({ key, type: 0x01, val });
  }

  public addMetadataInt(key: string, val: number | bigint): void {
    this.metadata.push({ key, type: 0x02, val: BigInt(val) });
  }

  public addMetadataFloat(key: string, val: number): void {
    this.metadata.push({ key, type: 0x03, val });
  }

  public addMetadataBool(key: string, val: boolean): void {
    this.metadata.push({ key, type: 0x04, val });
  }

  public addMetadataJson(key: string, val: any): void {
    this.metadata.push({ key, type: 0x05, val: JSON.stringify(val) });
  }

  public addMetadataBytes(key: string, val: Uint8Array): void {
    this.metadata.push({ key, type: 0x06, val });
  }

  public addTensor(
    name: string,
    storageType: StorageType,
    tileLayout: TileLayout,
    sparsityType: SparsityType,
    shape: number[],
    data: Uint8Array,
    sparsityRatio = 0.0
  ): void {
    this.tensors.push({
      name,
      storageType,
      tileLayout,
      sparsityType,
      shape,
      data,
      sparsityRatio,
    });
  }

  public build(): Uint8Array {
    const encoder = new TextEncoder();

    // 1. Serialize Metadata Section
    const metaParts: Uint8Array[] = [];
    for (const m of this.metadata) {
      const kBytes = encoder.encode(m.key);
      const kHeader = new Uint8Array(2);
      new DataView(kHeader.buffer).setUint16(0, kBytes.length, true);
      const tag = new Uint8Array([m.type]);

      let valBytes: Uint8Array;
      if (m.type === 0x01 || m.type === 0x05) {
        valBytes = encoder.encode(m.val);
      } else if (m.type === 0x02) {
        valBytes = new Uint8Array(8);
        new DataView(valBytes.buffer).setBigInt64(0, m.val, true);
      } else if (m.type === 0x03) {
        valBytes = new Uint8Array(8);
        new DataView(valBytes.buffer).setFloat64(0, m.val, true);
      } else if (m.type === 0x04) {
        valBytes = new Uint8Array([m.val ? 1 : 0]);
      } else if (m.type === 0x06) {
        valBytes = m.val instanceof Uint8Array ? m.val : new Uint8Array(m.val);
      } else {
        valBytes = encoder.encode(String(m.val));
      }

      const vHeader = new Uint8Array(4);
      new DataView(vHeader.buffer).setUint32(0, valBytes.length, true);

      metaParts.push(kHeader, kBytes, tag, vHeader, valBytes);
    }
    const metaSize = metaParts.reduce((acc, p) => acc + p.length, 0);
    const metaBuf = new Uint8Array(metaSize);
    let mPos = 0;
    for (const p of metaParts) {
      metaBuf.set(p, mPos);
      mPos += p.length;
    }

    // 2. Compute TOC Size
    let tocSize = 0;
    for (const t of this.tensors) {
      const nBytes = encoder.encode(t.name);
      tocSize += 2 + nBytes.length + 4 + t.shape.length * 8 + 54;
    }

    // 3. Layout Offsets
    const headerSize = 128;
    const metaOffset = headerSize;
    const tocOffset = metaOffset + metaSize;
    let dataOffset = tocOffset + tocSize;
    if (dataOffset % this.alignment !== 0) {
      dataOffset += this.alignment - (dataOffset % this.alignment);
    }

    // 4. Compute data offsets for tensors
    let currentDataOffset = dataOffset;
    const tensorOffsets: number[] = [];
    for (const t of this.tensors) {
      if (currentDataOffset % this.alignment !== 0) {
        currentDataOffset += this.alignment - (currentDataOffset % this.alignment);
      }
      tensorOffsets.push(currentDataOffset);
      currentDataOffset += t.data.length;
    }

    const totalFileSize = currentDataOffset;
    const outBuf = new Uint8Array(totalFileSize);
    const view = new DataView(outBuf.buffer);

    // 5. Write File Header
    outBuf[0] = 0x48; // 'H'
    outBuf[1] = 0x4b; // 'K'
    outBuf[2] = 0x4e; // 'N'
    outBuf[3] = 0x54; // 'T'
    view.setUint16(4, 1, true); // Version Major
    view.setUint16(6, 0, true); // Version Minor
    let flags = HeaderFlags.LITTLE_ENDIAN;
    if ((this.alignment % 128) === 0) {
      flags |= HeaderFlags.TILE_ALIGNED;
    }
    if ((this.alignment % 4096) === 0) {
      flags |= HeaderFlags.UNIVERSAL_PAGE_ALIGNED;
    }
    if (this.rawWeightStorage) {
      flags |= HeaderFlags.RAW_WEIGHT_STORAGE;
    }
    if (this.isSharded) {
      flags |= HeaderFlags.IS_SHARDED;
    }
    view.setUint32(8, flags, true);
    view.setUint16(12, this.alignment, true);
    view.setUint16(14, this.splitIndex, true);
    view.setBigUint64(16, BigInt(this.tensors.length), true);
    view.setBigUint64(24, BigInt(this.metadata.length), true);
    view.setBigUint64(32, BigInt(metaOffset), true);
    view.setBigUint64(40, BigInt(metaSize), true);
    view.setBigUint64(48, BigInt(tocOffset), true);
    view.setBigUint64(56, BigInt(tocSize), true);
    view.setBigUint64(64, BigInt(dataOffset), true);
    view.setBigUint64(72, 0n, true); // appendix_offset
    view.setBigUint64(80, 0n, true); // checksum
    view.setUint16(88, this.splitCount, true);

    // Write Metadata
    outBuf.set(metaBuf, metaOffset);

    // Write TOC
    let tWritePos = tocOffset;
    for (let i = 0; i < this.tensors.length; i++) {
      const t = this.tensors[i];
      const nBytes = encoder.encode(t.name);
      view.setUint16(tWritePos, nBytes.length, true);
      tWritePos += 2;
      outBuf.set(nBytes, tWritePos);
      tWritePos += nBytes.length;

      view.setUint8(tWritePos, t.storageType);
      view.setUint8(tWritePos + 1, t.tileLayout);
      view.setUint8(tWritePos + 2, t.sparsityType);
      view.setUint8(tWritePos + 3, t.shape.length);
      tWritePos += 4;

      for (const dim of t.shape) {
        view.setBigUint64(tWritePos, BigInt(dim), true);
        tWritePos += 8;
      }

      view.setBigUint64(tWritePos, BigInt(tensorOffsets[i]), true); // data_offset
      view.setBigUint64(tWritePos + 8, BigInt(t.data.length), true); // data_size
      view.setBigUint64(tWritePos + 16, 0n, true); // residual_offset
      view.setBigUint64(tWritePos + 24, 0n, true); // residual_size
      view.setBigUint64(tWritePos + 32, 0n, true); // scale_offset
      view.setBigUint64(tWritePos + 40, 0n, true); // scale_size
      view.setUint16(tWritePos + 48, 32, true); // block_size
      view.setFloat32(tWritePos + 50, t.sparsityRatio, true); // sparsity_ratio
      tWritePos += 54;
    }

    // Write Data
    for (let i = 0; i < this.tensors.length; i++) {
      outBuf.set(this.tensors[i].data, tensorOffsets[i]);
    }

    return outBuf;
  }
}

/**
 * Universal Multi-Stage Pipeline Support for Heterogeneous Models
 */
export type PipelineModality =
  | "audio_transcription"
  | "vision_ocr"
  | "token_analysis"
  | "sequence_classification"
  | "text_generation"
  | "generic";

export interface PipelineStageConfig {
  name: string;
  modality?: PipelineModality;
  modelPath?: string;
  inputMapping?: Record<string, string>;
  outputMapping?: Record<string, string>;
  params?: Record<string, any>;
}

export interface PipelineManifest {
  pipeline_type: "universal" | "composite";
  stages: PipelineStageConfig[];
  context_strategy?: "blackboard" | "linear_pipe" | "isolated";
  metadata?: Record<string, any>;
}

export type StageHandler = (input: any, context: PipelineContext) => Promise<any> | any;

export class PipelineContext {
  public blackboard: Map<string, any> = new Map();
  public stageTrace: Array<{ stage: string; timestamp: number; outputKeys: string[] }> = [];

  constructor(initialData: Record<string, any> = {}) {
    for (const [k, v] of Object.entries(initialData)) {
      this.blackboard.set(k, v);
    }
  }

  public get(key: string, defaultValue: any = undefined): any {
    return this.blackboard.has(key) ? this.blackboard.get(key) : defaultValue;
  }

  public set(key: string, value: any): void {
    this.blackboard.set(key, value);
  }

  public toObject(): Record<string, any> {
    const obj: Record<string, any> = {};
    for (const [k, v] of this.blackboard.entries()) {
      obj[k] = v;
    }
    return obj;
  }
}

export class PipelineStage {
  public name: string;
  public modality: PipelineModality;
  public handler: StageHandler;
  public inputMapping: Record<string, string>;
  public outputMapping: Record<string, string>;
  public params: Record<string, any>;

  constructor(options: {
    name: string;
    modality?: PipelineModality;
    handler: StageHandler;
    inputMapping?: Record<string, string>;
    outputMapping?: Record<string, string>;
    params?: Record<string, any>;
  }) {
    this.name = options.name;
    this.modality = options.modality || "generic";
    this.handler = options.handler;
    this.inputMapping = options.inputMapping || {};
    this.outputMapping = options.outputMapping || {};
    this.params = options.params || {};
  }

  public async execute(stageInput: any, ctx: PipelineContext): Promise<any> {
    const rawOut = await this.handler(stageInput, ctx);

    // Blackboard propagation & mapping
    if (typeof rawOut === "object" && rawOut !== null && !Array.isArray(rawOut)) {
      for (const [outKey, outVal] of Object.entries(rawOut)) {
        const mappedKey = this.outputMapping[outKey] || outKey;
        ctx.set(mappedKey, outVal);
      }
    } else {
      ctx.set(this.name, rawOut);
    }

    ctx.stageTrace.push({
      stage: this.name,
      timestamp: Date.now(),
      outputKeys: typeof rawOut === "object" && rawOut !== null && !Array.isArray(rawOut) ? Object.keys(rawOut) : [this.name],
    });

    return rawOut;
  }
}

export class UniversalPipeline {
  public stages: PipelineStage[] = [];
  public contextStrategy: "blackboard" | "linear_pipe" | "isolated" = "blackboard";

  public addStage(stage: PipelineStage): this {
    this.stages.push(stage);
    return this;
  }

  public async execute(initialInput: any, context?: PipelineContext): Promise<PipelineContext> {
    const ctx =
      context ||
      new PipelineContext(
        typeof initialInput === "object" && initialInput !== null && !Array.isArray(initialInput)
          ? initialInput
          : { input: initialInput }
      );

    let currentInput = initialInput;
    for (const stage of this.stages) {
      let stageIn = currentInput;
      if (Object.keys(stage.inputMapping).length > 0) {
        stageIn = {};
        for (const [stageArg, ctxKey] of Object.entries(stage.inputMapping)) {
          stageIn[stageArg] = ctx.get(ctxKey);
        }
      }
      currentInput = await stage.execute(stageIn, ctx);
    }

    return ctx;
  }
}

