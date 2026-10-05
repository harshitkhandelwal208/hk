//! On disk block layouts for every quantized format HK can store, plus the table of block
//! sizes. All layouts match the GGML formats byte for byte so tensors can be moved between
//! containers without re-encoding.
//!
//! Blocks are always read through `*align(1)` pointers: a block can start at any byte offset
//! (for example 17 byte MXFP4 blocks), so nothing here assumes alignment.

const std = @import("std");
const format = @import("../format.zig");

pub const QK_K: usize = 256;

pub const BlockQ4_0 = extern struct { d: f16, qs: [16]u8 };
pub const BlockQ4_1 = extern struct { d: f16, m: f16, qs: [16]u8 };
pub const BlockQ5_0 = extern struct { d: f16, qh: [4]u8, qs: [16]u8 };
pub const BlockQ5_1 = extern struct { d: f16, m: f16, qh: [4]u8, qs: [16]u8 };
pub const BlockQ8_0 = extern struct { d: f16, qs: [32]i8 };

pub const BlockQ2K = extern struct { scales: [16]u8, qs: [64]u8, d: f16, dmin: f16 };
pub const BlockQ3K = extern struct { hmask: [32]u8, qs: [64]u8, scales: [12]u8, d: f16 };
pub const BlockQ4K = extern struct { d: f16, dmin: f16, scales: [12]u8, qs: [128]u8 };
pub const BlockQ5K = extern struct { d: f16, dmin: f16, scales: [12]u8, qh: [32]u8, qs: [128]u8 };
pub const BlockQ6K = extern struct { ql: [128]u8, qh: [64]u8, scales: [16]i8, d: f16 };
/// Activation side block for K-quant dot products. Never stored in files.
pub const BlockQ8K = extern struct { d: f32, qs: [256]i8, bsums: [16]i16 };

pub const BlockIQ2XXS = extern struct { d: f16, qs: [64]u8 };
pub const BlockIQ2XS = extern struct { d: f16, qs: [64]u8, scales: [8]u8 };
pub const BlockIQ2S = extern struct { d: f16, qs: [32]u8, signs: [32]u8, qh: [8]u8, scales: [8]u8 };
pub const BlockIQ3XXS = extern struct { d: f16, qs: [96]u8 };
pub const BlockIQ3S = extern struct { d: f16, qs: [64]u8, qh: [8]u8, signs: [32]u8, scales: [4]u8 };
pub const BlockIQ1S = extern struct { d: f16, qs: [32]u8, qh: [16]u8 };
pub const BlockIQ1M = extern struct { qs: [32]u8, qh: [16]u8, scales: [8]u8 };
pub const BlockIQ4NL = extern struct { d: f16, qs: [16]u8 };
pub const BlockIQ4XS = extern struct { d: f16, scales_h: [2]u8, scales_l: [4]u8, qs: [128]u8 };

pub const BlockTQ1_0 = extern struct { qs: [48]u8, qh: [4]u8, d: f16 };
pub const BlockTQ2_0 = extern struct { qs: [64]u8, d: f16 };

pub const BlockMXFP4 = extern struct { e: u8, qs: [16]u8 };
pub const BlockNVFP4 = extern struct { d: [4]u8, qs: [32]u8 };

comptime {
    const expect = struct {
        fn size(comptime T: type, comptime n: usize) void {
            if (@sizeOf(T) != n) @compileError(std.fmt.comptimePrint("{s} must be {d} bytes, is {d}", .{ @typeName(T), n, @sizeOf(T) }));
        }
    };
    expect.size(BlockQ4_0, 18);
    expect.size(BlockQ4_1, 20);
    expect.size(BlockQ5_0, 22);
    expect.size(BlockQ5_1, 24);
    expect.size(BlockQ8_0, 34);
    expect.size(BlockQ2K, 84);
    expect.size(BlockQ3K, 110);
    expect.size(BlockQ4K, 144);
    expect.size(BlockQ5K, 176);
    expect.size(BlockQ6K, 210);
    expect.size(BlockQ8K, 292);
    expect.size(BlockIQ2XXS, 66);
    expect.size(BlockIQ2XS, 74);
    expect.size(BlockIQ2S, 82);
    expect.size(BlockIQ3XXS, 98);
    expect.size(BlockIQ3S, 110);
    expect.size(BlockIQ1S, 50);
    expect.size(BlockIQ1M, 56);
    expect.size(BlockIQ4NL, 18);
    expect.size(BlockIQ4XS, 136);
    expect.size(BlockTQ1_0, 54);
    expect.size(BlockTQ2_0, 66);
    expect.size(BlockMXFP4, 17);
    expect.size(BlockNVFP4, 36);
}

pub const Info = struct {
    /// Weights per block. 1 for plain float and integer types.
    elems: u16,
    /// Bytes per block.
    bytes: u16,
};

/// Block geometry for a storage type, or null when the type is not a fixed size block type
/// (sparse, reference and dual mode containers).
pub fn info(t: format.StorageType) ?Info {
    return switch (t) {
        .f32, .int32, .uint32 => .{ .elems = 1, .bytes = 4 },
        .f16, .bf16, .int16, .uint16 => .{ .elems = 1, .bytes = 2 },
        .int8, .uint8, .bool, .fp8_e4m3, .fp8_e5m2 => .{ .elems = 1, .bytes = 1 },
        .int64, .uint64, .f64 => .{ .elems = 1, .bytes = 8 },
        .q4_0 => .{ .elems = 32, .bytes = @sizeOf(BlockQ4_0) },
        .q4_1 => .{ .elems = 32, .bytes = @sizeOf(BlockQ4_1) },
        .q5_0 => .{ .elems = 32, .bytes = @sizeOf(BlockQ5_0) },
        .q5_1 => .{ .elems = 32, .bytes = @sizeOf(BlockQ5_1) },
        .q8_0 => .{ .elems = 32, .bytes = @sizeOf(BlockQ8_0) },
        .q2_k => .{ .elems = QK_K, .bytes = @sizeOf(BlockQ2K) },
        .q3_k => .{ .elems = QK_K, .bytes = @sizeOf(BlockQ3K) },
        .q4_k => .{ .elems = QK_K, .bytes = @sizeOf(BlockQ4K) },
        .q5_k => .{ .elems = QK_K, .bytes = @sizeOf(BlockQ5K) },
        .q6_k => .{ .elems = QK_K, .bytes = @sizeOf(BlockQ6K) },
        .q8_k => .{ .elems = QK_K, .bytes = @sizeOf(BlockQ8K) },
        .iq2_xxs => .{ .elems = QK_K, .bytes = @sizeOf(BlockIQ2XXS) },
        .iq2_xs => .{ .elems = QK_K, .bytes = @sizeOf(BlockIQ2XS) },
        .iq2_s => .{ .elems = QK_K, .bytes = @sizeOf(BlockIQ2S) },
        .iq3_xxs => .{ .elems = QK_K, .bytes = @sizeOf(BlockIQ3XXS) },
        .iq3_s => .{ .elems = QK_K, .bytes = @sizeOf(BlockIQ3S) },
        .iq1_s => .{ .elems = QK_K, .bytes = @sizeOf(BlockIQ1S) },
        .iq1_m => .{ .elems = QK_K, .bytes = @sizeOf(BlockIQ1M) },
        .iq4_nl => .{ .elems = 32, .bytes = @sizeOf(BlockIQ4NL) },
        .iq4_xs => .{ .elems = QK_K, .bytes = @sizeOf(BlockIQ4XS) },
        .tq1_0 => .{ .elems = QK_K, .bytes = @sizeOf(BlockTQ1_0) },
        .tq2_0 => .{ .elems = QK_K, .bytes = @sizeOf(BlockTQ2_0) },
        .mxfp4 => .{ .elems = 32, .bytes = @sizeOf(BlockMXFP4) },
        .nvfp4 => .{ .elems = 64, .bytes = @sizeOf(BlockNVFP4) },
        else => null,
    };
}

pub const SizeError = error{ UnsupportedType, NotBlockAligned };

/// Exact byte size of `n_elems` weights of type `t`. Fails instead of rounding: a tensor whose
/// element count is not a whole number of blocks is corrupt or mis-typed, and silently padding
/// it would shift every later tensor in the file.
pub fn byteSize(t: format.StorageType, n_elems: u64) SizeError!u64 {
    const i = info(t) orelse return error.UnsupportedType;
    if (n_elems % i.elems != 0) return error.NotBlockAligned;
    return (n_elems / i.elems) * i.bytes;
}

test "block size table matches the reference" {
    try std.testing.expectEqual(@as(u64, 54), try byteSize(.tq1_0, 256));
    try std.testing.expectEqual(@as(u64, 66), try byteSize(.tq2_0, 256));
    try std.testing.expectEqual(@as(u64, 50), try byteSize(.iq1_s, 256));
    try std.testing.expectError(error.NotBlockAligned, byteSize(.q4_k, 100));
    try std.testing.expectError(error.UnsupportedType, byteSize(.sparse_2_4, 256));
}
