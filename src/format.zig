const std = @import("std");

/// HK File Format Magic Bytes: "HKNT" (0x48, 0x4B, 0x4E, 0x54)
pub const MAGIC: [4]u8 = .{ 0x48, 0x4B, 0x4E, 0x54 };
pub const VERSION_MAJOR: u16 = 1;
pub const VERSION_MINOR: u16 = 0;
pub const DEFAULT_ALIGNMENT_BYTES: usize = 128; // Default NVIDIA Tensor Core coalescing alignment
pub const ALIGNMENT_BYTES: usize = 128;
pub const UNIVERSAL_PAGE_ALIGNMENT_BYTES: usize = 4096; // Standard Multi-Device Page Alignment (AMD ROCm, Intel NPU, x86_64, Linux ARM)
pub const APPLE_SILICON_ALIGNMENT_BYTES: usize = 16384; // Apple Silicon Metal zero-copy GPU page alignment (16 KB)
pub const DIRECT_DMA_ALIGNMENT_BYTES: usize = 65536; // 64 KB hugepage / Windows allocation granularity

pub const HeaderFlags = struct {
    pub const LITTLE_ENDIAN: u32 = 1 << 0;
    pub const HAS_APPENDIX: u32 = 1 << 1;
    pub const HAS_QUANT_TABLE: u32 = 1 << 2;
    pub const SPARSITY_2_4: u32 = 1 << 3;
    pub const TILE_ALIGNED: u32 = 1 << 4;
    pub const FLEXIBLE_ALIGNMENT: u32 = 1 << 5; // Relaxed alignment for arbitrary portable devices
    pub const IS_SHARDED: u32 = 1 << 6; // Multi-file sharding flag
    pub const RAW_WEIGHT_STORAGE: u32 = 1 << 7; // Raw unquantized IEEE weights for zero compute headroom
    pub const UNIVERSAL_PAGE_ALIGNED: u32 = 1 << 8; // Super-coalesced hardware page aligned (4KB / 16KB / 64KB)
};

pub const StorageType = enum(u8) {
    // Dense float & int types
    f32 = 0x00,
    f16 = 0x01,
    bf16 = 0x02,
    fp8_e4m3 = 0x03,
    fp8_e5m2 = 0x04,
    int8 = 0x05,
    int32 = 0x06,
    int64 = 0x07,
    uint8 = 0x08,
    bool = 0x09,
    int16 = 0x0A,
    uint16 = 0x0B,
    uint32 = 0x0C,
    uint64 = 0x0D,
    f64 = 0x0E,

    // Dual-mode quantized types
    dq4 = 0x10, // 4-bit dual-mode (NF4/INT4 base + block scale + optional residual)
    dq8 = 0x11, // 8-bit quantized
    dq6 = 0x12, // 6-bit quantized
    dq12 = 0x13, // 12-bit quantized
    dqt = 0x14, // Ternary {-1, 0, +1} (BitNet b1.58)

    // Standard baseline quantizations (GGUF compatible)
    q4_0 = 0x15, // Standard 4-bit baseline (32 weights, 18 bytes: FP16 d + 16 bytes qs)
    q8_0 = 0x16, // Standard 8-bit baseline (32 weights, 34 bytes: FP16 d + 32 bytes qs)
    q4_1 = 0x17, // Standard 4-bit baseline with offset
    q5_0 = 0x18, // Standard 5-bit baseline
    q5_1 = 0x19, // Standard 5-bit baseline with offset
    q8_1 = 0x1A, // Standard 8-bit baseline with offset

    // Sparse types
    sparse_f16 = 0x20,
    sparse_dq8 = 0x21,
    sparse_2_4 = 0x22, // Ampere 2:4 structured sparse (FP16/BF16)
    sparse_dq4_2_4 = 0x23, // 2:4 structured sparse with 4-bit quantization

    // Virtual references
    null_ref = 0x30, // Fully pruned/zero tensor (0 payload bytes)
    shared_ref = 0x31, // Shared pointer to another tensor (tied weights)
    lora_ref = 0x32, // Base tensor + low rank adaptation factors

    // K-Quants (super-block 256 weights with 16 sub-blocks, scales & min offsets)
    q2_k = 0x40, // 2-bit super-block quantization (2.5625 bpw)
    q3_k = 0x41, // 3-bit super-block quantization (3.4375 bpw)
    q4_k = 0x42, // 4-bit super-block quantization (4.5 bpw)
    q5_k = 0x43, // 5-bit super-block quantization (5.5 bpw)
    q6_k = 0x44, // 6-bit super-block quantization (6.5625 bpw)
    q8_k = 0x45, // 8-bit super-block quantization (8.5 bpw)

    // I-Quants (importance matrix non-linear codebook vector quantization)
    iq1_s = 0x50, // 1-bit importance quant (1.56 bpw)
    iq1_m = 0x51, // 1-bit importance quant (1.75 bpw)
    iq2_xxs = 0x52, // 2-bit extra-extra-small (2.06 bpw)
    iq2_xs = 0x53, // 2-bit extra-small (2.31 bpw)
    iq3_xxs = 0x54, // 3-bit extra-extra-small (3.06 bpw)
    iq4_nl = 0x55, // 4-bit non-linear codebook (4.5 bpw)
    iq4_xs = 0x56, // 4-bit extra-small (4.25 bpw)
    iq2_s = 0x57, // 2-bit small (2.5 bpw)
    iq3_s = 0x58, // 3-bit small (3.44 bpw)

    // Ternary & Microscaling
    tq1_0 = 0x60, // 1.58-bit ternary quantization
    tq2_0 = 0x61, // 2-bit ternary quantization
    mxfp4 = 0x62, // OCP Microscaling FP4 (E2M1 with E8M0 scale)
    nvfp4 = 0x63, // NVIDIA Blackwell FP4 (E2M1 with FP8 scale)

    pub fn isRaw(self: StorageType) bool {
        return switch (self) {
            .f32, .f16, .bf16, .fp8_e4m3, .fp8_e5m2, .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .f64, .bool => true,
            else => false,
        };
    }

    pub fn elementSize(self: StorageType) usize {
        return switch (self) {
            .f32, .int32, .uint32 => 4,
            .f16, .bf16, .int16, .uint16 => 2,
            .int8, .uint8, .bool, .fp8_e4m3, .fp8_e5m2 => 1,
            .int64, .uint64, .f64 => 8,
            else => 1,
        };
    }
};

pub const TileLayout = enum(u8) {
    row_major = 0x00,
    col_major = 0x01,
    tile_16x16 = 0x02, // 16x16 Ampere WMMA tile (K-contiguous)
    tile_16x8 = 0x03,
    tile_32x16 = 0x04,
    block_sparse_2_4 = 0x05,
    tile_32x32 = 0x06,
    tile_64x64 = 0x07,
};

pub const SparsityType = enum(u8) {
    none = 0x00,
    bitmask = 0x01,
    csr = 0x02,
    structured_2_4 = 0x03,
    physical_pruned = 0x04, // Physically reduced dimension matrices
    bsr = 0x05,
};

/// 128-Byte Fixed File Header (aligned to ALIGNMENT_BYTES)
pub const FileHeader = extern struct {
    magic: [4]u8 = MAGIC,
    version_major: u16 = VERSION_MAJOR,
    version_minor: u16 = VERSION_MINOR,
    flags: u32 = HeaderFlags.LITTLE_ENDIAN | HeaderFlags.TILE_ALIGNED,
    alignment: u16 = 128,
    split_index: u16 = 0, // Sharding: current shard index (0-based)
    tensor_count: u64 = 0,
    metadata_kv_count: u64 = 0,
    metadata_offset: u64 = 0,
    metadata_size: u64 = 0,
    tensor_toc_offset: u64 = 0,
    tensor_toc_size: u64 = 0,
    tensor_data_offset: u64 = 0,
    appendix_offset: u64 = 0,
    checksum: u64 = 0,
    split_count: u16 = 1, // Sharding: total number of shards (>= 1)
    reserved1: [38]u8 = @as([38]u8, @splat(0)),

    pub fn isValid(self: *const FileHeader) bool {
        return std.mem.eql(u8, &self.magic, &MAGIC) and self.version_major == VERSION_MAJOR;
    }
};

comptime {
    if (@sizeOf(FileHeader) != 128) {
        @compileError(std.fmt.comptimePrint("FileHeader size must be 128 bytes, got {}", .{@sizeOf(FileHeader)}));
    }
}

pub const MAX_DIMS: usize = 8;
pub const MAX_NAME_LEN: usize = 128;

pub const TensorEntry = struct {
    name: []const u8,
    storage_type: StorageType,
    tile_layout: TileLayout,
    sparsity_type: SparsityType,
    ndim: u8,
    shape: [MAX_DIMS]u64,
    data_offset: u64, // Absolute byte offset in file (128-byte aligned)
    data_size: u64, // Primary payload byte size
    residual_offset: u64 = 0, // Byte offset to residual recovery buffer (0 if none)
    residual_size: u64 = 0, // Byte size of residual recovery buffer
    scale_offset: u64 = 0, // Byte offset to per-block scale factors (0 if none)
    scale_size: u64 = 0, // Byte size of scales
    block_size: u16 = 32, // Quantization block size (default 32)
    sparsity_ratio: f32 = 0.0, // Pruning sparsity ratio [0.0, 1.0)
};

pub const MetadataValueType = enum(u8) {
    val_string = 0x01,
    val_int64 = 0x02,
    val_float64 = 0x03,
    val_bool = 0x04,
    val_json = 0x05,
    val_bytes = 0x06,
};

pub const AppendixEntryType = enum(u8) {
    lora_adapter = 0x01,
    delta_patch = 0x02,
    new_layer = 0x03,
    code_eval = 0x04,
    kv_cache_sink = 0x05,
    topology_head = 0x06,
};

pub const AppendixFlags = struct {
    pub const ACTIVE: u8 = 1 << 0;
    pub const COMPRESSED: u8 = 1 << 1;
};

pub const AppendixMetrics = extern struct {
    loss: f32 = 0.0,
    accuracy: f32 = 0.0,
    pass_rate: f32 = 0.0,
    custom: f32 = 0.0,
};

pub const AppendixRecordHeader = extern struct {
    entry_type: u8,
    flags: u8 = AppendixFlags.ACTIVE,
    name_len: u16,
    generation: u32,
    timestamp: u64,
    parent_hash: [32]u8 = @as([32]u8, @splat(0)),
    metric_loss: f32 = 0.0,
    metric_acc: f32 = 0.0,
    metric_pass: f32 = 0.0,
    metric_custom: f32 = 0.0,
    target_len: u16 = 0,
    reserved: u16 = 0,
    data_crc32: u32 = 0,
    data_size: u64,
};

comptime {
    if (@sizeOf(AppendixRecordHeader) != 80) {
        @compileError(std.fmt.comptimePrint("AppendixRecordHeader size must be 80 bytes, got {}", .{@sizeOf(AppendixRecordHeader)}));
    }
}

pub const AppendixRecord = struct {
    entry_type: AppendixEntryType,
    flags: u8 = AppendixFlags.ACTIVE,
    name: []const u8,
    target: []const u8 = "",
    generation: u32,
    timestamp: u64,
    parent_hash: [32]u8 = @as([32]u8, @splat(0)),
    metrics: AppendixMetrics = .{},
    data: []const u8,
};

