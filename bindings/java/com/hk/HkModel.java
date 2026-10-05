package com.hk;

import java.io.Closeable;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * HK Java / Android SDK
 * Reader and writer for HK containers over the C ABI (through JNI). It does not run inference.
 */
public final class HkModel implements Closeable, AutoCloseable {

    static {
        // libhk is the C library; libhkjni (built with `zig build jni -Djdk=...`) is the JNI glue.
        // Both must be on java.library.path, or load them yourself with System.load() first.
        try {
            System.loadLibrary("hk");
            System.loadLibrary("hkjni");
        } catch (UnsatisfiedLinkError e) {
            // Left for the caller: the first native call then fails with the same error.
        }
    }

    public static final class StorageType {
        public static final byte F32 = 0x00;
        public static final byte F16 = 0x01;
        public static final byte BF16 = 0x02;
        public static final byte FP8_E4M3 = 0x03;
        public static final byte FP8_E5M2 = 0x04;
        public static final byte INT8 = 0x05;
        public static final byte INT32 = 0x06;
        public static final byte INT64 = 0x07;
        public static final byte UINT8 = 0x08;
        public static final byte BOOL = 0x09;
        public static final byte INT16 = 0x0A;
        public static final byte UINT16 = 0x0B;
        public static final byte UINT32 = 0x0C;
        public static final byte UINT64 = 0x0D;
        public static final byte F64 = 0x0E;
        public static final byte DQ4 = 0x10;
        public static final byte NF4 = 0x10;
        public static final byte DQ8 = 0x11;
        public static final byte DQ6 = 0x12;
        public static final byte DQ12 = 0x13;
        public static final byte DQT = 0x14;
        public static final byte Q4_0 = 0x15;
        public static final byte Q8_0 = 0x16;
        public static final byte Q4_1 = 0x17;
        public static final byte Q5_0 = 0x18;
        public static final byte Q5_1 = 0x19;
        public static final byte Q8_1 = 0x1A;
        public static final byte SPARSE_F16 = 0x20;
        public static final byte SPARSE_DQ8 = 0x21;
        public static final byte SPARSE_2_4 = 0x22;
        public static final byte SPARSE_DQ4_2_4 = 0x23;
        public static final byte NULL_REF = 0x30;
        public static final byte SHARED_REF = 0x31;
        public static final byte LORA_REF = 0x32;
        public static final byte Q2_K = 0x40;
        public static final byte Q3_K = 0x41;
        public static final byte Q4_K = 0x42;
        public static final byte Q5_K = 0x43;
        public static final byte Q6_K = 0x44;
        public static final byte Q8_K = 0x45;
        public static final byte IQ1_S = 0x50;
        public static final byte IQ1_M = 0x51;
        public static final byte IQ2_XXS = 0x52;
        public static final byte IQ2_XS = 0x53;
        public static final byte IQ3_XXS = 0x54;
        public static final byte IQ4_NL = 0x55;
        public static final byte IQ4_XS = 0x56;
        public static final byte IQ2_S = 0x57;
        public static final byte IQ3_S = 0x58;
        public static final byte TQ1_0 = 0x60;
        public static final byte TQ2_0 = 0x61;
        public static final byte MXFP4 = 0x62;
        public static final byte NVFP4 = 0x63;
    }

    public static final class Constants {
        public static final int FLAG_IS_SHARDED = 0x40;
        public static final int FLAG_RAW_WEIGHT_STORAGE = 1 << 7;
        public static final int FLAG_UNIVERSAL_PAGE_ALIGNED = 1 << 8;

        public static final long DEFAULT_ALIGNMENT_BYTES = 128;
        public static final long UNIVERSAL_PAGE_ALIGNMENT_BYTES = 4096;
        public static final long APPLE_SILICON_ALIGNMENT_BYTES = 16384;
        public static final long DIRECT_DMA_ALIGNMENT_BYTES = 65536;
    }

    public static final class TileLayout {
        public static final byte ROW_MAJOR = 0x00;
        public static final byte COL_MAJOR = 0x01;
        public static final byte TILE_16X16 = 0x02;
        public static final byte TILE_16X8 = 0x03;
        public static final byte TILE_32X16 = 0x04;
        public static final byte BLOCK_SPARSE_2_4 = 0x05;
        public static final byte TILE_32X32 = 0x06;
        public static final byte TILE_64X64 = 0x07;
    }

    public static final class SparsityType {
        public static final byte NONE = 0x00;
        public static final byte BITMASK = 0x01;
        public static final byte CSR = 0x02;
        public static final byte STRUCTURED_2_4 = 0x03;
        public static final byte PHYSICAL_PRUNED = 0x04;
        public static final byte BSR = 0x05;
    }

    public static final class AppendixType {
        public static final byte LORA_ADAPTER = 0x01;
        public static final byte DELTA_PATCH = 0x02;
        public static final byte NEW_LAYER = 0x03;
        public static final byte CODE_EVAL = 0x04;
        public static final byte KV_CACHE_SINK = 0x05;
        public static final byte TOPOLOGY_HEAD = 0x06;
    }

    private long nativeHandle;
    private boolean isClosed = false;

    private HkModel(long handle) {
        this.nativeHandle = handle;
    }

    public static HkModel open(String path) throws FileNotFoundException {
        long handle = nativeOpen(path);
        if (handle == 0) {
            throw new FileNotFoundException("Failed to open HK model file: " + path);
        }
        return new HkModel(handle);
    }

    public long getTensorCount() {
        checkClosed();
        return nativeGetTensorCount(nativeHandle);
    }

    public HkTensor getTensor(long index) {
        checkClosed();
        return nativeGetTensor(nativeHandle, index);
    }

    public String getMetadataString(String key) {
        checkClosed();
        return nativeGetMetadataString(nativeHandle, key);
    }

    public Long getMetadataInt(String key) {
        checkClosed();
        long[] out = new long[1];
        if (nativeGetMetadataInt(nativeHandle, key, out) == 0) {
            return out[0];
        }
        return null;
    }

    public Double getMetadataFloat(String key) {
        checkClosed();
        double[] out = new double[1];
        if (nativeGetMetadataFloat(nativeHandle, key, out) == 0) {
            return out[0];
        }
        return null;
    }

    public Boolean getMetadataBool(String key) {
        checkClosed();
        int[] out = new int[1];
        if (nativeGetMetadataBool(nativeHandle, key, out) == 0) {
            return out[0] != 0;
        }
        return null;
    }

    public boolean isSharded() {
        checkClosed();
        return nativeIsSharded(nativeHandle) != 0;
    }

    public int getSplitIndex() {
        checkClosed();
        return nativeGetSplitIndex(nativeHandle);
    }

    public int getSplitCount() {
        checkClosed();
        return nativeGetSplitCount(nativeHandle);
    }

    public static boolean patchMetadataInPlace(String path, String key, String val) {
        return nativePatchMetadataInPlace(path, key, val) == 0;
    }

    public long getAppendixCount() {
        checkClosed();
        return nativeGetAppendixCount(nativeHandle);
    }

    public HkAppendixEntry getAppendixEntry(long index) {
        checkClosed();
        return nativeGetAppendixEntry(nativeHandle, index);
    }

    public boolean isRawStorage() {
        checkClosed();
        return nativeIsRawStorage(nativeHandle) != 0;
    }

    public boolean isUniversalPageAligned() {
        checkClosed();
        return nativeIsUniversalPageAligned(nativeHandle) != 0;
    }

    public int getFileAlignment() {
        checkClosed();
        return nativeGetFileAlignment(nativeHandle);
    }

    public boolean isTensorCoreAligned() {
        return (getFileAlignment() % 128) == 0;
    }

    public static void rollback(String path, int targetGeneration) throws IOException {
        int res = nativeRollback(path, targetGeneration);
        if (res != 0) {
            throw new IOException("Failed to rollback " + path + " to generation " + targetGeneration);
        }
    }

    @Override
    public synchronized void close() {
        if (!isClosed) {
            if (nativeHandle != 0) {
                nativeClose(nativeHandle);
                nativeHandle = 0;
            }
            isClosed = true;
        }
    }

    private void checkClosed() {
        if (isClosed) {
            throw new IllegalStateException("HkModel is closed.");
        }
    }

    public static class HkTensor {
        private final long modelHandle;
        private final long index;
        private final String name;
        private final byte storageType;
        private final byte tileLayout;
        private final byte sparsityType;
        private final short blockSize;
        private final float sparsityRatio;
        private final long[] shape;

        public HkTensor(long modelHandle, long index, String name, byte storageType, byte tileLayout, byte sparsityType, short blockSize, float sparsityRatio, long[] shape) {
            this.modelHandle = modelHandle;
            this.index = index;
            this.name = name;
            this.storageType = storageType;
            this.tileLayout = tileLayout;
            this.sparsityType = sparsityType;
            this.blockSize = blockSize;
            this.sparsityRatio = sparsityRatio;
            this.shape = shape;
        }

        public String getName() { return name; }
        public byte getStorageType() { return storageType; }
        public byte getTileLayout() { return tileLayout; }
        public byte getSparsityType() { return sparsityType; }
        public short getBlockSize() { return blockSize; }
        public float getSparsityRatio() { return sparsityRatio; }
        public long[] getShape() { return shape.clone(); }

        public long getElementCount() {
            long count = 1;
            for (long dim : shape) count *= dim;
            return count;
        }

        public FloatBuffer dequantize(boolean withResidual) {
            long count = getElementCount();
            FloatBuffer buffer = ByteBuffer.allocateDirect((int) (count * 4))
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();

            int res = nativeDequantizeF32(modelHandle, index, withResidual ? 1 : 0, buffer, count);
            if (res != 0) {
                throw new RuntimeException("Dequantization failed for tensor " + name);
            }
            return buffer;
        }

        public ByteBuffer getRawData() {
            return nativeGetTensorData(modelHandle, index);
        }

        public ByteBuffer getRawResidual() {
            return nativeGetTensorResidual(modelHandle, index);
        }

        public ByteBuffer getRawScales() {
            return nativeGetTensorScales(modelHandle, index);
        }

        public boolean isRaw() {
            return storageType == StorageType.F32 ||
                   storageType == StorageType.F16 ||
                   storageType == StorageType.BF16 ||
                   storageType == StorageType.FP8_E4M3 ||
                   storageType == StorageType.FP8_E5M2 ||
                   storageType == StorageType.INT8 ||
                   storageType == StorageType.INT16 ||
                   storageType == StorageType.INT32 ||
                   storageType == StorageType.INT64 ||
                   storageType == StorageType.UINT8 ||
                   storageType == StorageType.UINT16 ||
                   storageType == StorageType.UINT32 ||
                   storageType == StorageType.UINT64 ||
                   storageType == StorageType.BOOL ||
                   storageType == StorageType.F64;
        }

        public ByteBuffer getRawBytes() {
            return nativeGetTensorRawPtr(modelHandle, index);
        }
    }

    public static class HkAppendixEntry {
        public byte entryType;
        public byte flags;
        public int generation;
        public long timestamp;
        public byte[] parentHash;
        public float metricLoss;
        public float metricAcc;
        public float metricPass;
        public float metricCustom;
        public String name;
        public String target;
        public long dataSize;

        public HkAppendixEntry(byte entryType, byte flags, int generation, long timestamp, byte[] parentHash, float metricLoss, float metricAcc, float metricPass, float metricCustom, String name, String target, long dataSize) {
            this.entryType = entryType;
            this.flags = flags;
            this.generation = generation;
            this.timestamp = timestamp;
            this.parentHash = parentHash;
            this.metricLoss = metricLoss;
            this.metricAcc = metricAcc;
            this.metricPass = metricPass;
            this.metricCustom = metricCustom;
            this.name = name;
            this.target = target;
            this.dataSize = dataSize;
        }
    }

    public static final class HkWriter implements Closeable, AutoCloseable {
        private long nativeWriterHandle;
        private boolean isClosed = false;

        public HkWriter(long alignment) {
            this.nativeWriterHandle = nativeWriterCreate(alignment);
            if (this.nativeWriterHandle == 0) {
                throw new RuntimeException("Failed to create native HkWriter");
            }
        }

        public void setSharding(int splitIndex, int splitCount) {
            checkClosed();
            nativeWriterSetSharding(nativeWriterHandle, splitIndex, splitCount);
        }

        public void setRawStorage(boolean enabled) {
            checkClosed();
            nativeWriterSetRawStorage(nativeWriterHandle, enabled ? 1 : 0);
        }

        public void addMetadataString(String key, String val) {
            checkClosed();
            if (nativeWriterAddMetadataString(nativeWriterHandle, key, val) != 0) {
                throw new RuntimeException("Failed to add string metadata: " + key);
            }
        }

        public void addMetadataInt(String key, long val) {
            checkClosed();
            if (nativeWriterAddMetadataInt(nativeWriterHandle, key, val) != 0) {
                throw new RuntimeException("Failed to add int metadata: " + key);
            }
        }

        public void addMetadataFloat(String key, double val) {
            checkClosed();
            if (nativeWriterAddMetadataFloat(nativeWriterHandle, key, val) != 0) {
                throw new RuntimeException("Failed to add float metadata: " + key);
            }
        }

        public void addMetadataBool(String key, boolean val) {
            checkClosed();
            if (nativeWriterAddMetadataBool(nativeWriterHandle, key, val ? 1 : 0) != 0) {
                throw new RuntimeException("Failed to add bool metadata: " + key);
            }
        }

        public void addTensor(String name, byte storageType, byte tileLayout, byte sparsityType, long[] shape, ByteBuffer data, float sparsityRatio) {
            checkClosed();
            if (shape.length > 8) {
                throw new IllegalArgumentException("HK format supports at most 8 dimensions");
            }
            if (nativeWriterAddTensor(nativeWriterHandle, name, storageType, tileLayout, sparsityType, (byte) shape.length, shape, data, data.remaining(), sparsityRatio) != 0) {
                throw new RuntimeException("Failed to add tensor: " + name);
            }
        }

        public void writeToFile(String path) throws IOException {
            checkClosed();
            if (nativeWriterWriteToFile(nativeWriterHandle, path) != 0) {
                throw new IOException("Failed to write HK container file to: " + path);
            }
        }

        @Override
        public synchronized void close() {
            if (!isClosed) {
                if (nativeWriterHandle != 0) {
                    nativeWriterDestroy(nativeWriterHandle);
                    nativeWriterHandle = 0;
                }
                isClosed = true;
            }
        }

        private void checkClosed() {
            if (isClosed) throw new IllegalStateException("HkWriter is closed");
        }
    }

    // Native JNI functions
    private static native long nativeOpen(String path);
    private static native void nativeClose(long handle);
    private static native long nativeGetTensorCount(long handle);
    private static native HkTensor nativeGetTensor(long handle, long index);
    private static native ByteBuffer nativeGetTensorData(long handle, long index);
    private static native ByteBuffer nativeGetTensorResidual(long handle, long index);
    private static native ByteBuffer nativeGetTensorScales(long handle, long index);
    private static native int nativeDequantizeF32(long handle, long index, int withResidual, FloatBuffer outBuf, long count);
    private static native String nativeGetMetadataString(long handle, String key);
    private static native int nativeGetMetadataInt(long handle, String key, long[] outVal);
    private static native int nativeGetMetadataFloat(long handle, String key, double[] outVal);
    private static native int nativeGetMetadataBool(long handle, String key, int[] outVal);
    private static native int nativeIsSharded(long handle);
    private static native int nativeGetSplitIndex(long handle);
    private static native int nativeGetSplitCount(long handle);
    private static native int nativePatchMetadataInPlace(String path, String key, String val);
    private static native long nativeGetAppendixCount(long handle);
    private static native HkAppendixEntry nativeGetAppendixEntry(long handle, long index);
    private static native int nativeRollback(String path, int targetGeneration);
    private static native int nativeIsRawStorage(long handle);
    private static native int nativeIsUniversalPageAligned(long handle);
    private static native int nativeGetFileAlignment(long handle);
    private static native ByteBuffer nativeGetTensorRawPtr(long handle, long index);

    // Native JNI Writer functions
    private static native long nativeWriterCreate(long alignment);
    private static native void nativeWriterDestroy(long writerHandle);
    private static native void nativeWriterSetSharding(long writerHandle, int splitIndex, int splitCount);
    private static native void nativeWriterSetRawStorage(long writerHandle, int enabled);
    private static native int nativeWriterAddMetadataString(long writerHandle, String key, String val);
    private static native int nativeWriterAddMetadataInt(long writerHandle, String key, long val);
    private static native int nativeWriterAddMetadataFloat(long writerHandle, String key, double val);
    private static native int nativeWriterAddMetadataBool(long writerHandle, String key, int val);
    private static native int nativeWriterAddTensor(long writerHandle, String name, byte storageType, byte tileLayout, byte sparsityType, byte ndim, long[] shape, ByteBuffer data, long dataLen, float sparsityRatio);
    private static native int nativeWriterWriteToFile(long writerHandle, String path);
}

