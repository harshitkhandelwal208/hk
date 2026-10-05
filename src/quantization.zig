const std = @import("std");
const format = @import("format.zig");

/// 16-point NormalFloat-4 (NF4) quantile lookup table
/// Normalized to [-1.0, 1.0] from standard normal distribution N(0, 1) quantiles
pub const NF4_TABLE: [16]f32 = .{
    -1.0000000,
    -0.6961928,
    -0.5250731,
    -0.3949175,
    -0.2844414,
    -0.1847734,
    -0.0910500,
    0.0000000,
    0.0795803,
    0.1609302,
    0.2461123,
    0.3379152,
    0.4407098,
    0.5626170,
    0.7229568,
    1.0000000,
};

/// Finds the closest NF4 code (0..15) for a normalized value in [-1.0, 1.0]
pub fn findClosestNF4(val: f32) u4 {
    var best_idx: u4 = 0;
    var min_dist: f32 = std.math.inf(f32);

    inline for (0..16) |i| {
        const dist = @abs(val - NF4_TABLE[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = @intCast(i);
        }
    }
    return best_idx;
}

/// Quantizes a block of floats to NF4 with optional residual tracking.
/// packed_out receives ceil(block.len / 2) bytes (2 4-bit codes per byte).
pub fn quantizeBlockNF4(
    block: []const f32,
    packed_out: []u8,
    residual_out: ?[]f32,
) f32 {
    if (block.len == 0) return 0.0;

    var max_abs: f32 = 0.0;
    for (block) |x| {
        const a = @abs(x);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0) max_abs = 1e-8;

    const inv_scale = 1.0 / max_abs;

    for (0..block.len) |i| {
        const norm = std.math.clamp(block[i] * inv_scale, -1.0, 1.0);
        const code = findClosestNF4(norm);

        const byte_idx = i / 2;
        if (i % 2 == 0) {
            packed_out[byte_idx] = @as(u8, code);
        } else {
            packed_out[byte_idx] |= (@as(u8, code) << 4);
        }

        if (residual_out) |res| {
            const deq_val = NF4_TABLE[code] * max_abs;
            res[i] = block[i] - deq_val;
        }
    }

    return max_abs;
}

/// Dequantizes packed 4-bit NF4 bytes to 32-bit floats using block scale.
pub fn dequantizeBlockNF4(
    packed_in: []const u8,
    scale: f32,
    count: usize,
    out: []f32,
) void {
    var i: usize = 0;
    // Process 4 elements at a time with vectorization where possible
    while (i + 4 <= count) : (i += 4) {
        const b0 = packed_in[i / 2];
        const b1 = packed_in[(i / 2) + 1];
        const c0: u4 = @truncate(b0 & 0x0F);
        const c1: u4 = @truncate((b0 >> 4) & 0x0F);
        const c2: u4 = @truncate(b1 & 0x0F);
        const c3: u4 = @truncate((b1 >> 4) & 0x0F);

        const v: @Vector(4, f32) = .{
            NF4_TABLE[c0],
            NF4_TABLE[c1],
            NF4_TABLE[c2],
            NF4_TABLE[c3],
        };
        const s_vec: @Vector(4, f32) = @splat(scale);
        const res_vec = v * s_vec;
        out[i..][0..4].* = res_vec;
    }

    while (i < count) : (i += 1) {
        const byte_idx = i / 2;
        const code: u4 = if (i % 2 == 0)
            @truncate(packed_in[byte_idx] & 0x0F)
        else
            @truncate((packed_in[byte_idx] >> 4) & 0x0F);

        out[i] = NF4_TABLE[code] * scale;
    }
}

/// Quantizes a block of floats to symmetric 8-bit integers (DQ8) with optional residual delta.
pub fn quantizeBlockDQ8(
    block: []const f32,
    out_i8: []i8,
    residual_out: ?[]f32,
) f32 {
    if (block.len == 0) return 0.0;

    var max_abs: f32 = 0.0;
    for (block) |x| {
        const a = @abs(x);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0) max_abs = 1e-8;

    const scale = max_abs / 127.0;
    const inv_scale = 127.0 / max_abs;

    for (0..block.len) |i| {
        const q_float = std.math.clamp(block[i] * inv_scale, -127.0, 127.0);
        const q_int: i8 = @intFromFloat(@round(q_float));
        out_i8[i] = q_int;

        if (residual_out) |res| {
            const deq = @as(f32, @floatFromInt(q_int)) * scale;
            res[i] = block[i] - deq;
        }
    }

    return scale;
}

/// Dequantizes signed 8-bit integers (DQ8) back to 32-bit floats.
pub fn dequantizeBlockDQ8(
    in_i8: []const i8,
    scale: f32,
    count: usize,
    out: []f32,
) void {
    var i: usize = 0;
    while (i + 4 <= count) : (i += 4) {
        const q: @Vector(4, f32) = .{
            @floatFromInt(in_i8[i + 0]),
            @floatFromInt(in_i8[i + 1]),
            @floatFromInt(in_i8[i + 2]),
            @floatFromInt(in_i8[i + 3]),
        };
        const s: @Vector(4, f32) = @splat(scale);
        out[i..][0..4].* = q * s;
    }
    while (i < count) : (i += 1) {
        out[i] = @as(f32, @floatFromInt(in_i8[i])) * scale;
    }
}

/// BitNet b1.58 Ternary Quantization: weights in {-1, 0, +1}
/// 2 bits per weight:
///   00 -> 0
///   01 -> +1
///   10 -> -1
/// Scale = average absolute value: alpha = (1/N) * sum(|W|)
pub fn quantizeBlockDQT(
    block: []const f32,
    packed_out: []u8,
    residual_out: ?[]f32,
) f32 {
    if (block.len == 0) return 0.0;

    var sum_abs: f64 = 0.0;
    for (block) |x| {
        sum_abs += @abs(x);
    }
    const scale: f32 = @floatCast(sum_abs / @as(f64, @floatFromInt(block.len)));
    const eff_scale = if (scale == 0.0) 1e-8 else scale;
    const thresh = eff_scale * 0.5;

    @memset(packed_out, 0);

    for (0..block.len) |i| {
        var code: u2 = 0; // 00 -> 0
        var val: f32 = 0.0;

        if (block[i] > thresh) {
            code = 1; // 01 -> +1
            val = eff_scale;
        } else if (block[i] < -thresh) {
            code = 2; // 10 -> -1
            val = -eff_scale;
        }

        const byte_idx = i / 4;
        const bit_shift: u3 = @intCast((i % 4) * 2);
        packed_out[byte_idx] |= (@as(u8, code) << bit_shift);

        if (residual_out) |res| {
            res[i] = block[i] - val;
        }
    }

    return eff_scale;
}

/// Dequantizes packed 2-bit ternary weights back to 32-bit floats using block scale.
pub fn dequantizeBlockDQT(
    packed_in: []const u8,
    scale: f32,
    count: usize,
    out: []f32,
) void {
    for (0..count) |i| {
        const byte_idx = i / 4;
        const bit_shift: u3 = @intCast((i % 4) * 2);
        const code: u2 = @truncate((packed_in[byte_idx] >> bit_shift) & 0x03);

        const val: f32 = switch (code) {
            1 => scale,
            2 => -scale,
            else => 0.0,
        };
        out[i] = val;
    }
}

/// Dequantizes FP16 values to FP32
pub fn dequantizeF16(in_bytes: []const u8, count: usize, out: []f32) void {
    const f16_slice: []const f16 = @as([*]const f16, @ptrCast(@alignCast(in_bytes.ptr)))[0..count];
    for (0..count) |i| {
        out[i] = @floatCast(f16_slice[i]);
    }
}

/// Dequantizes BF16 values to FP32
pub fn dequantizeBF16(in_bytes: []const u8, count: usize, out: []f32) void {
    const u16_slice: []const u16 = @as([*]const u16, @ptrCast(@alignCast(in_bytes.ptr)))[0..count];
    for (0..count) |i| {
        const u32_bits = @as(u32, u16_slice[i]) << 16;
        out[i] = @bitCast(u32_bits);
    }
}

fn computeFP8_E4M3_LUT() [256]f32 {
    @setEvalBranchQuota(100000);
    var table: [256]f32 = undefined;
    for (0..256) |idx| {
        const b: u8 = @intCast(idx);
        const sign: f32 = if ((b & 0x80) != 0) -1.0 else 1.0;
        const exp = (b >> 3) & 0x0F;
        const mant = b & 0x07;

        if (exp == 0) {
            // Subnormal: (-1)^sign * 2^(-6) * (mant / 8)
            // 2^(-6) = 0.015625
            table[idx] = sign * 0.015625 * (@as(f32, @floatFromInt(mant)) / 8.0);
        } else if (exp == 15 and mant == 7) {
            // NaN in E4M3
            table[idx] = std.math.nan(f32);
        } else {
            // Normalized: (-1)^sign * 2^(exp - 7) * (1 + mant / 8)
            const exp_i: i32 = @as(i32, @intCast(exp)) - 7;
            const factor: f32 = if (exp_i >= 0)
                @floatFromInt(@as(u32, 1) << @intCast(exp_i))
            else
                1.0 / @as(f32, @floatFromInt(@as(u32, 1) << @intCast(-exp_i)));
            table[idx] = sign * factor * (1.0 + @as(f32, @floatFromInt(mant)) / 8.0);
        }
    }
    return table;
}

pub const FP8_E4M3_TABLE: [256]f32 = computeFP8_E4M3_LUT();

/// Dequantizes FP8 E4M3 to FP32 using comptime-generated lookup table (sign: 1, exp: 4, mantissa: 3, bias: 7)
pub fn dequantizeFP8_E4M3(in_bytes: []const u8, count: usize, out: []f32) void {
    const limit = @min(@min(in_bytes.len, count), out.len);
    var i: usize = 0;
    while (i + 8 <= limit) : (i += 8) {
        out[i + 0] = FP8_E4M3_TABLE[in_bytes[i + 0]];
        out[i + 1] = FP8_E4M3_TABLE[in_bytes[i + 1]];
        out[i + 2] = FP8_E4M3_TABLE[in_bytes[i + 2]];
        out[i + 3] = FP8_E4M3_TABLE[in_bytes[i + 3]];
        out[i + 4] = FP8_E4M3_TABLE[in_bytes[i + 4]];
        out[i + 5] = FP8_E4M3_TABLE[in_bytes[i + 5]];
        out[i + 6] = FP8_E4M3_TABLE[in_bytes[i + 6]];
        out[i + 7] = FP8_E4M3_TABLE[in_bytes[i + 7]];
    }
    while (i < limit) : (i += 1) {
        out[i] = FP8_E4M3_TABLE[in_bytes[i]];
    }
}

// ---------------------------------------------------------------------------
// Standard Baseline Quantizations: Q4_0 and Q8_0 (32 weights per block)
// ---------------------------------------------------------------------------
pub const QK4_0: usize = 32;
pub const QK8_0: usize = 32;

/// Q4_0: 32 elements per block, 18 bytes total.
/// 2-byte FP16 scale + 16 bytes of 4-bit nibbles (low nibble 0..15, high nibble 16..31).
pub const BlockQ4_0 = extern struct {
    d: f16 = 0,
    qs: [16]u8 = @as([16]u8, @splat(0)),
};

comptime {
    if (@sizeOf(BlockQ4_0) != 18) {
        @compileError(std.fmt.comptimePrint("BlockQ4_0 size must be 18 bytes, got {}", .{@sizeOf(BlockQ4_0)}));
    }
}

pub fn quantizeBlockQ4_0(weights: []const f32, block: *BlockQ4_0) void {
    var max_abs: f32 = 0.0;
    var max_val: f32 = 0.0;
    for (weights) |w| {
        const a = @abs(w);
        if (a > max_abs) {
            max_abs = a;
            max_val = w;
        }
    }
    if (max_abs == 0.0) {
        block.d = 0;
        @memset(&block.qs, 0);
        return;
    }

    const d: f32 = max_val / -8.0;
    const d_f16: f16 = @floatCast(d);
    block.d = d_f16;
    const d_eff: f32 = @floatCast(d_f16);
    const id: f32 = if (d_eff != 0.0) 1.0 / d_eff else 0.0;

    for (0..16) |i| {
        const w0 = if (i < weights.len) weights[i] else 0.0;
        const w1 = if (i + 16 < weights.len) weights[i + 16] else 0.0;
        const q0_f = std.math.clamp(std.math.trunc(w0 * id + 8.5), 0.0, 15.0);
        const q1_f = std.math.clamp(std.math.trunc(w1 * id + 8.5), 0.0, 15.0);
        const q0: u8 = @intFromFloat(q0_f);
        const q1: u8 = @intFromFloat(q1_f);
        block.qs[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
    }
}

pub fn dequantizeBlockQ4_0(block: *const BlockQ4_0, count: usize, out: []f32) void {
    const d: f32 = @floatCast(block.d);
    const n = @min(count, 32);
    const n_low = @min(n, 16);
    for (0..n_low) |i| {
        const q: i8 = @as(i8, @intCast(block.qs[i] & 0x0F)) - 8;
        out[i] = @as(f32, @floatFromInt(q)) * d;
    }
    if (n > 16) {
        for (16..n) |i| {
            const q: i8 = @as(i8, @intCast((block.qs[i - 16] >> 4) & 0x0F)) - 8;
            out[i] = @as(f32, @floatFromInt(q)) * d;
        }
    }
}

/// Q8_0: 32 elements per block, 34 bytes total.
/// 2-byte FP16 scale + 32 bytes of 8-bit signed integers.
pub const BlockQ8_0 = extern struct {
    d: f16 = 0,
    qs: [32]i8 = @as([32]i8, @splat(0)),
};

comptime {
    if (@sizeOf(BlockQ8_0) != 34) {
        @compileError(std.fmt.comptimePrint("BlockQ8_0 size must be 34 bytes, got {}", .{@sizeOf(BlockQ8_0)}));
    }
}

pub fn quantizeBlockQ8_0(weights: []const f32, block: *BlockQ8_0) void {
    var max_abs: f32 = 0.0;
    for (weights) |w| {
        const a = @abs(w);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0) {
        block.d = 0;
        @memset(&block.qs, 0);
        return;
    }

    const d_f32 = max_abs / 127.0;
    const d_f16: f16 = @floatCast(d_f32);
    block.d = d_f16;
    const d_eff: f32 = @floatCast(d_f16);
    const id: f32 = if (d_eff != 0.0) 1.0 / d_eff else 0.0;

    for (0..@min(weights.len, 32)) |i| {
        const q = std.math.clamp(@round(weights[i] * id), -128.0, 127.0);
        block.qs[i] = @intFromFloat(q);
    }
    if (weights.len < 32) {
        @memset(block.qs[weights.len..32], 0);
    }
}

pub fn dequantizeBlockQ8_0(block: *const BlockQ8_0, count: usize, out: []f32) void {
    const d: f32 = @floatCast(block.d);
    const n = @min(count, 32);
    var i: usize = 0;
    while (i + 4 <= n) : (i += 4) {
        const q: @Vector(4, f32) = .{
            @floatFromInt(block.qs[i + 0]),
            @floatFromInt(block.qs[i + 1]),
            @floatFromInt(block.qs[i + 2]),
            @floatFromInt(block.qs[i + 3]),
        };
        const s: @Vector(4, f32) = @splat(d);
        out[i..][0..4].* = q * s;
    }
    while (i < n) : (i += 1) {
        out[i] = @as(f32, @floatFromInt(block.qs[i])) * d;
    }
}

// ---------------------------------------------------------------------------
// K-Quants (Q2_K .. Q8_K) Super-Block Structures & Kernels
// ---------------------------------------------------------------------------
pub const QK_K: usize = 256;

/// Q4_K: 256 weights per super-block (16 sub-blocks of 16 weights)
/// Scales and mins packed into 12 bytes; 4-bit weights packed into 128 bytes.
pub const BlockQ4_K = extern struct {
    d: f16 = 0,
    dmin: f16 = 0,
    scales: [12]u8 = @as([12]u8, @splat(0)),
    qs: [QK_K / 2]u8 = @as([QK_K / 2]u8, @splat(0)),
};

pub fn quantizeSuperBlockQ4_K(weights: []const f32, block: *BlockQ4_K) void {
    var min_all: f32 = std.math.inf(f32);
    var max_all: f32 = -std.math.inf(f32);
    for (weights) |w| {
        if (w < min_all) min_all = w;
        if (w > max_all) max_all = w;
    }
    if (max_all <= min_all) {
        block.d = 0;
        block.dmin = 0;
        @memset(&block.scales, 0);
        @memset(&block.qs, 0);
        return;
    }

    const range = max_all - min_all;
    const super_scale = range / 15.0;
    block.d = @floatCast(super_scale);
    block.dmin = @floatCast(min_all);

    const inv_scale = 15.0 / range;
    for (0..weights.len) |i| {
        const q_val = std.math.clamp((weights[i] - min_all) * inv_scale, 0.0, 15.0);
        const code: u4 = @intFromFloat(@round(q_val));
        const byte_idx = i / 2;
        if (i % 2 == 0) {
            block.qs[byte_idx] = @as(u8, code);
        } else {
            block.qs[byte_idx] |= (@as(u8, code) << 4);
        }
    }
    @memset(&block.scales, 1);
}

pub fn dequantizeSuperBlockQ4_K(block: *const BlockQ4_K, count: usize, out: []f32) void {
    const d: f32 = @floatCast(block.d);
    const dmin: f32 = @floatCast(block.dmin);
    const n = @min(count, QK_K);

    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        const byte_idx = i / 2;
        const b = block.qs[byte_idx..][0..8];
        inline for (0..8) |j| {
            const raw = b[j];
            out[i + 2 * j] = @as(f32, @floatFromInt(raw & 0x0F)) * d + dmin;
            out[i + 2 * j + 1] = @as(f32, @floatFromInt(raw >> 4)) * d + dmin;
        }
    }
    while (i < n) : (i += 1) {
        const byte_idx = i / 2;
        const code: u4 = if (i % 2 == 0)
            @truncate(block.qs[byte_idx] & 0x0F)
        else
            @truncate((block.qs[byte_idx] >> 4) & 0x0F);

        out[i] = @as(f32, @floatFromInt(code)) * d + dmin;
    }
}

/// Q8_K: 256 weights per super-block (8-bit quantization with super-scale)
pub const BlockQ8_K = extern struct {
    d: f32 = 0,
    qs: [QK_K]i8 = @as([QK_K]i8, @splat(0)),
    bsums: [16]i16 = @as([16]i16, @splat(0)),
};

pub fn quantizeSuperBlockQ8_K(weights: []const f32, block: *BlockQ8_K) void {
    var max_abs: f32 = 0.0;
    for (weights) |w| {
        const a = @abs(w);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0) max_abs = 1e-8;

    const scale = max_abs / 127.0;
    block.d = scale;
    const inv_scale = 127.0 / max_abs;

    for (0..weights.len) |i| {
        const q = std.math.clamp(weights[i] * inv_scale, -127.0, 127.0);
        block.qs[i] = @intFromFloat(@round(q));
    }

    // Compute sub-block sums for fast GEMV dot-products
    for (0..16) |sb| {
        var sum: i32 = 0;
        for (0..16) |j| {
            sum += block.qs[sb * 16 + j];
        }
        block.bsums[sb] = @intCast(std.math.clamp(sum, -32768, 32767));
    }
}

pub fn dequantizeSuperBlockQ8_K(block: *const BlockQ8_K, count: usize, out: []f32) void {
    const scale = block.d;
    const n = @min(count, QK_K);
    var i: usize = 0;
    while (i + 4 <= n) : (i += 4) {
        const q: @Vector(4, f32) = .{
            @floatFromInt(block.qs[i + 0]),
            @floatFromInt(block.qs[i + 1]),
            @floatFromInt(block.qs[i + 2]),
            @floatFromInt(block.qs[i + 3]),
        };
        const s: @Vector(4, f32) = @splat(scale);
        out[i..][0..4].* = q * s;
    }
    while (i < n) : (i += 1) {
        out[i] = @as(f32, @floatFromInt(block.qs[i])) * scale;
    }
}

/// Helper to unpack 6-bit scales and mins from 12-byte packed buffer for Q4_K and Q5_K
pub fn getScaleMinK4(scales: *const [12]u8, sc: *[8]u8, min: *[8]u8) void {
    for (0..4) |j| {
        const d_byte = scales[j];
        const m_byte = scales[j + 4];
        const md_byte = scales[j + 8];
        sc[j] = d_byte & 0x3F;
        sc[j + 4] = (md_byte & 0x0F) | ((d_byte >> 2) & 0x30);
        min[j] = m_byte & 0x3F;
        min[j + 4] = ((md_byte >> 4) & 0x0F) | ((m_byte >> 2) & 0x30);
    }
}

/// Helper to pack 6-bit scales and mins into 12-byte buffer for Q4_K and Q5_K
pub fn setScaleMinK4(sc: *const [8]u8, min: *const [8]u8, scales: *[12]u8) void {
    for (0..4) |j| {
        const sc0 = sc[j] & 0x3F;
        const sc1 = sc[j + 4] & 0x3F;
        const m0 = min[j] & 0x3F;
        const m1 = min[j + 4] & 0x3F;
        scales[j] = sc0 | ((sc1 & 0x30) << 2);
        scales[j + 4] = m0 | ((m1 & 0x30) << 2);
        scales[j + 8] = (sc1 & 0x0F) | ((m1 & 0x0F) << 4);
    }
}

/// Q5_K: 256 weights per super-block (8 sub-blocks of 32 elements).
/// 176 bytes total: d (2) + dmin (2) + scales (12) + qh (32) + qs (128).
pub const BlockQ5_K = extern struct {
    d: f16 = 0,
    dmin: f16 = 0,
    scales: [12]u8 = @as([12]u8, @splat(0)),
    qh: [32]u8 = @as([32]u8, @splat(0)),
    qs: [QK_K / 2]u8 = @as([QK_K / 2]u8, @splat(0)),
};

comptime {
    if (@sizeOf(BlockQ5_K) != 176) {
        @compileError(std.fmt.comptimePrint("BlockQ5_K size must be 176 bytes, got {}", .{@sizeOf(BlockQ5_K)}));
    }
}

pub fn quantizeSuperBlockQ5_K(weights: []const f32, block: *BlockQ5_K) void {
    var min_all: f32 = std.math.inf(f32);
    var max_all: f32 = -std.math.inf(f32);
    for (weights) |w| {
        if (w < min_all) min_all = w;
        if (w > max_all) max_all = w;
    }
    if (max_all <= min_all) {
        block.d = 0;
        block.dmin = 0;
        @memset(&block.scales, 0);
        @memset(&block.qh, 0);
        @memset(&block.qs, 0);
        return;
    }

    var sb_min: [8]f32 = undefined;
    var sb_max: [8]f32 = undefined;
    for (0..8) |sb| {
        var smin: f32 = std.math.inf(f32);
        var smax: f32 = -std.math.inf(f32);
        const start = sb * 32;
        const end = @min(start + 32, weights.len);
        for (weights[start..end]) |w| {
            if (w < smin) smin = w;
            if (w > smax) smax = w;
        }
        sb_min[sb] = if (smin == std.math.inf(f32)) 0.0 else smin;
        sb_max[sb] = if (smax == -std.math.inf(f32)) 0.0 else smax;
    }

    var max_range: f32 = 0.0;
    for (0..8) |sb| {
        const r = sb_max[sb] - sb_min[sb];
        if (r > max_range) max_range = r;
    }
    const d_val = max_range / (31.0 * 63.0);
    const d_eff = if (d_val == 0) 1e-8 else d_val;
    block.d = @floatCast(d_eff);

    var max_min: f32 = 0.0;
    for (0..8) |sb| {
        const am = @abs(sb_min[sb]);
        if (am > max_min) max_min = am;
    }
    const dmin_val = max_min / 63.0;
    const dmin_eff = if (dmin_val == 0) 1e-8 else dmin_val;
    block.dmin = @floatCast(dmin_eff);

    var sc: [8]u8 = @as([8]u8, @splat(0));
    var min: [8]u8 = @as([8]u8, @splat(0));
    for (0..8) |sb| {
        const r = sb_max[sb] - sb_min[sb];
        const sc_f = std.math.clamp(@round(r / (31.0 * d_eff)), 0.0, 63.0);
        sc[sb] = @intFromFloat(sc_f);
        const min_f = std.math.clamp(@round(-sb_min[sb] / dmin_eff), 0.0, 63.0);
        min[sb] = @intFromFloat(min_f);
    }
    setScaleMinK4(&sc, &min, &block.scales);

    @memset(&block.qh, 0);
    @memset(&block.qs, 0);

    for (0..8) |sb| {
        const start = sb * 32;
        const end = @min(start + 32, weights.len);
        const sub_d = @as(f32, @floatCast(block.d)) * @as(f32, @floatFromInt(sc[sb]));
        const sub_m = @as(f32, @floatCast(block.dmin)) * @as(f32, @floatFromInt(min[sb]));
        const inv_d = if (sub_d != 0) 1.0 / sub_d else 0.0;

        const pair_idx = sb / 2;
        const is_high = (sb % 2) == 1;
        const qs_offset = pair_idx * 32;

        for (start..end) |i| {
            const idx_in_sb = i - start;
            const q_float = std.math.clamp(@round((weights[i] + sub_m) * inv_d), 0.0, 31.0);
            const q_val: u8 = @intFromFloat(q_float);
            const ql = q_val & 0x0F;
            const qh = (q_val >> 4) & 0x01;

            if (is_high) {
                block.qs[qs_offset + idx_in_sb] |= (ql << 4);
            } else {
                block.qs[qs_offset + idx_in_sb] = ql;
            }
            block.qh[idx_in_sb] |= (qh << @intCast(sb));
        }
    }
}

pub fn dequantizeSuperBlockQ5_K(block: *const BlockQ5_K, count: usize, out: []f32) void {
    const d: f32 = @floatCast(block.d);
    const dmin: f32 = @floatCast(block.dmin);
    var sc: [8]u8 = undefined;
    var min: [8]u8 = undefined;
    getScaleMinK4(&block.scales, &sc, &min);

    const n = @min(count, QK_K);

    for (0..8) |sb| {
        const sb_start = sb * 32;
        if (sb_start >= n) break;
        const sb_end = @min(sb_start + 32, n);
        const sub_d = d * @as(f32, @floatFromInt(sc[sb]));
        const sub_m = dmin * @as(f32, @floatFromInt(min[sb]));

        const pair_idx = sb / 2;
        const is_high = (sb % 2) == 1;
        const qs_offset = pair_idx * 32;

        for (sb_start..sb_end) |i| {
            const idx_in_sb = i - sb_start;
            const q_byte = block.qs[qs_offset + idx_in_sb];
            const ql: u8 = if (is_high) (q_byte >> 4) & 0x0F else q_byte & 0x0F;
            const qh_bit: u8 = (block.qh[idx_in_sb] >> @intCast(sb)) & 0x01;
            const q_val = ql | (qh_bit << 4);

            out[i] = sub_d * @as(f32, @floatFromInt(q_val)) - sub_m;
        }
    }
}

/// Q3_K: 256 weights per super-block (16 sub-blocks of 16 weights).
/// 110 bytes total: hmask (32) + qs (64) + scales (12) + d (2).
pub const BlockQ3_K = extern struct {
    hmask: [32]u8 = @as([32]u8, @splat(0)),
    qs: [QK_K / 4]u8 = @as([QK_K / 4]u8, @splat(0)),
    scales: [12]u8 = @as([12]u8, @splat(0)),
    d: f16 = 0,
};

comptime {
    if (@sizeOf(BlockQ3_K) != 110) {
        @compileError(std.fmt.comptimePrint("BlockQ3_K size must be 110 bytes, got {}", .{@sizeOf(BlockQ3_K)}));
    }
}

pub fn quantizeSuperBlockQ3_K(weights: []const f32, block: *BlockQ3_K) void {
    var max_abs: f32 = 0.0;
    for (weights) |w| {
        const a = @abs(w);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0) {
        block.d = 0;
        @memset(&block.hmask, 0);
        @memset(&block.qs, 0);
        @memset(&block.scales, 0);
        return;
    }

    var sb_max: [16]f32 = @as([16]f32, @splat(0.0));
    for (0..16) |sb| {
        const start = sb * 16;
        const end = @min(start + 16, weights.len);
        var smax: f32 = 0.0;
        for (weights[start..end]) |w| {
            const a = @abs(w);
            if (a > smax) smax = a;
        }
        sb_max[sb] = smax;
    }

    const d_val = max_abs / (4.0 * 31.0);
    const d_eff = if (d_val == 0) 1e-8 else d_val;
    block.d = @floatCast(d_eff);

    var scales: [16]i8 = undefined;
    for (0..16) |sb| {
        const sc_f = std.math.clamp(@round(sb_max[sb] / (4.0 * d_eff)), 0.0, 31.0);
        scales[sb] = @intFromFloat(sc_f);
    }

    // Pack 16 scales (each stored as scale + 32 in 6 bits)
    @memset(&block.scales, 0);
    for (0..16) |sb| {
        const v: u8 = @intCast(@as(i16, scales[sb]) + 32);
        const lscale = v & 0x0F;
        const hscale = (v >> 4) & 0x03;

        const l_shift: u3 = if (sb < 8) 0 else 4;
        block.scales[sb % 8] |= (lscale << l_shift);

        const h_col = sb % 4;
        const h_row: u3 = @intCast(sb / 4);
        const h_shift: u3 = h_row * 2;
        block.scales[8 + h_col] |= (hscale << h_shift);
    }

    @memset(&block.hmask, 0);
    @memset(&block.qs, 0);

    for (0..16) |sb| {
        const sb_start = sb * 16;
        const sb_end = @min(sb_start + 16, weights.len);
        const dl = @as(f32, @floatCast(block.d)) * @as(f32, @floatFromInt(scales[sb]));
        const inv_dl = if (dl != 0) 1.0 / dl else 0.0;

        const qs_base: usize = if (sb < 8) 0 else 32;
        const sb_mod8 = sb % 8;
        const qs_offset: usize = if (sb_mod8 % 2 == 1) 16 else 0;
        const shift: u3 = @intCast((sb_mod8 / 2) * 2);

        const hm_offset: usize = if (sb % 2 == 1) 16 else 0;
        const hm_bit: u3 = @intCast(sb / 2);

        for (sb_start..sb_end) |i| {
            const idx_in_sb = i - sb_start;
            const q_f = std.math.clamp(@round(weights[i] * inv_dl), -4.0, 3.0);
            const q_int: i8 = @intFromFloat(q_f);

            var ql: u8 = 0;
            var qh_bit: u8 = 0;
            if (q_int >= 0) {
                ql = @intCast(q_int);
                qh_bit = 0; // qh = 0 => (0 ^ 1) = 1 in dequant
            } else {
                ql = @intCast(q_int + 4);
                qh_bit = 1; // qh = 1 => (1 ^ 1) = 0 in dequant
            }

            const qs_idx = qs_base + qs_offset + idx_in_sb;
            block.qs[qs_idx] |= ((ql & 0x03) << shift);

            const hm_idx = hm_offset + idx_in_sb;
            block.hmask[hm_idx] |= (qh_bit << hm_bit);
        }
    }
}

pub fn dequantizeSuperBlockQ3_K(block: *const BlockQ3_K, count: usize, out: []f32) void {
    const d: f32 = @floatCast(block.d);
    const n = @min(count, QK_K);

    var scales: [16]i8 = undefined;
    for (0..16) |sb| {
        const l_shift: u3 = if (sb < 8) 0 else 4;
        const l_byte = block.scales[sb % 8];
        const lscale = (l_byte >> l_shift) & 0x0F;

        const h_col = sb % 4;
        const h_row: u3 = @intCast(sb / 4);
        const h_byte = block.scales[8 + h_col];
        const h_shift: u3 = h_row * 2;
        const hscale = (h_byte >> h_shift) & 0x03;

        scales[sb] = @as(i8, @intCast(lscale | (hscale << 4))) - 32;
    }

    for (0..16) |sb| {
        const sb_start = sb * 16;
        if (sb_start >= n) break;
        const sb_end = @min(sb_start + 16, n);
        const dl = d * @as(f32, @floatFromInt(scales[sb]));

        const qs_base: usize = if (sb < 8) 0 else 32;
        const sb_mod8 = sb % 8;
        const qs_offset: usize = if (sb_mod8 % 2 == 1) 16 else 0;
        const shift: u3 = @intCast((sb_mod8 / 2) * 2);

        const hm_offset: usize = if (sb % 2 == 1) 16 else 0;
        const hm_bit: u3 = @intCast(sb / 2);

        for (sb_start..sb_end) |i| {
            const idx_in_sb = i - sb_start;
            const q_byte = block.qs[qs_base + qs_offset + idx_in_sb];
            const ql: u8 = (q_byte >> shift) & 0x03;

            const hm_byte = block.hmask[hm_offset + idx_in_sb];
            const qh_bit: u8 = ((hm_byte >> hm_bit) & 0x01) ^ 0x01;
            const q = @as(i8, @intCast(ql)) - (@as(i8, @intCast(qh_bit)) << 2);

            out[i] = dl * @as(f32, @floatFromInt(q));
        }
    }
}

/// Q6_K: 256 weights per super-block (6 bits per weight: 4-bit low + 2-bit high)
pub const BlockQ6_K = extern struct {
    ql: [QK_K / 2]u8 = @as([QK_K / 2]u8, @splat(0)),
    qh: [QK_K / 4]u8 = @as([QK_K / 4]u8, @splat(0)),
    scales: [16]i8 = @as([16]i8, @splat(0)),
    d: f16 = 0,
};

comptime {
    if (@sizeOf(BlockQ6_K) != 210) {
        @compileError(std.fmt.comptimePrint("BlockQ6_K size must be 210 bytes, got {}", .{@sizeOf(BlockQ6_K)}));
    }
}

pub fn quantizeSuperBlockQ6_K(weights: []const f32, block: *BlockQ6_K) void {
    var max_all: f32 = 0.0;
    for (weights) |w| {
        const a = @abs(w);
        if (a > max_all) max_all = a;
    }
    if (max_all == 0.0) {
        block.d = 0;
        @memset(&block.ql, 0);
        @memset(&block.qh, 0);
        @memset(&block.scales, 0);
        return;
    }

    var sb_max: [16]f32 = @as([16]f32, @splat(0.0));
    for (0..16) |sb| {
        const start = sb * 16;
        const end = @min(start + 16, weights.len);
        var smax: f32 = 0.0;
        for (weights[start..end]) |w| {
            const a = @abs(w);
            if (a > smax) smax = a;
        }
        sb_max[sb] = smax;
    }

    const d_val = max_all / (31.0 * 127.0);
    const d_eff = if (d_val == 0) 1e-8 else d_val;
    block.d = @floatCast(d_eff);

    for (0..16) |sb| {
        const sc_f = std.math.clamp(@round(sb_max[sb] / (31.0 * d_eff)), 1.0, 127.0);
        block.scales[sb] = @intFromFloat(sc_f);
    }

    @memset(&block.ql, 0);
    @memset(&block.qh, 0);

    for (0..16) |sb| {
        const start = sb * 16;
        const end = @min(start + 16, weights.len);
        const sub_scale = @as(f32, @floatCast(block.d)) * @as(f32, @floatFromInt(block.scales[sb]));
        const inv_scale = if (sub_scale != 0) 1.0 / sub_scale else 0.0;

        const h: usize = start / 128;
        const rem = start % 128;
        const chunk32 = rem / 32;
        const is_high = chunk32 >= 2;
        const ql_chunk: usize = chunk32 % 2;
        const offset_in_32: usize = rem % 32;
        const ql_base: usize = h * 64 + ql_chunk * 32 + offset_in_32;
        const qh_base: usize = h * 32 + offset_in_32;
        const qh_shift: u3 = @intCast(chunk32 * 2);

        for (start..end) |i| {
            const j = i - start;
            const q_f = std.math.clamp(@round(weights[i] * inv_scale), -32.0, 31.0);
            const val6: i8 = @intFromFloat(q_f);
            const code: u8 = @intCast(val6 + 32);

            const low_nibble = code & 0x0F;
            const high_bits = (code >> 4) & 0x03;

            if (is_high) {
                block.ql[ql_base + j] |= (low_nibble << 4);
            } else {
                block.ql[ql_base + j] = low_nibble;
            }

            block.qh[qh_base + j] |= (high_bits << qh_shift);
        }
    }
}

pub fn dequantizeSuperBlockQ6_K(block: *const BlockQ6_K, count: usize, out: []f32) void {
    const d: f32 = @floatCast(block.d);
    const n = @min(count, QK_K);

    for (0..16) |sb| {
        const start = sb * 16;
        if (start >= n) break;
        const end = @min(start + 16, n);
        const sub_scale = @as(f32, @floatFromInt(block.scales[sb])) * d;

        const h: usize = start / 128;
        const rem = start % 128;
        const chunk32 = rem / 32;
        const is_high = chunk32 >= 2;
        const ql_chunk: usize = chunk32 % 2;
        const offset_in_32: usize = rem % 32;
        const ql_base: usize = h * 64 + ql_chunk * 32 + offset_in_32;
        const qh_base: usize = h * 32 + offset_in_32;
        const qh_shift: u3 = @intCast(chunk32 * 2);

        for (start..end) |i| {
            const j = i - start;
            const ql_byte = block.ql[ql_base + j];
            const ql_val: u8 = if (is_high) (ql_byte >> 4) & 0x0F else ql_byte & 0x0F;
            const qh_byte = block.qh[qh_base + j];
            const qh_val: u8 = (qh_byte >> qh_shift) & 0x03;
            const val6: i8 = @as(i8, @intCast(ql_val | (qh_val << 4))) - 32;
            out[i] = @as(f32, @floatFromInt(val6)) * sub_scale;
        }
    }
}

/// Q2_K: 256 weights per super-block (2 bits per weight + 16 sub-block scales/mins)
pub const BlockQ2_K = extern struct {
    scales: [16]u8 = @as([16]u8, @splat(0)),
    qs: [QK_K / 4]u8 = @as([QK_K / 4]u8, @splat(0)),
    d: f16 = 0,
    dmin: f16 = 0,
};

comptime {
    if (@sizeOf(BlockQ2_K) != 84) {
        @compileError(std.fmt.comptimePrint("BlockQ2_K size must be 84 bytes, got {}", .{@sizeOf(BlockQ2_K)}));
    }
}

pub fn quantizeSuperBlockQ2_K(weights: []const f32, block: *BlockQ2_K) void {
    var min_all: f32 = std.math.inf(f32);
    var max_all: f32 = -std.math.inf(f32);
    for (weights) |w| {
        if (w < min_all) min_all = w;
        if (w > max_all) max_all = w;
    }
    if (max_all <= min_all) {
        block.d = 0;
        block.dmin = 0;
        @memset(&block.scales, 0);
        @memset(&block.qs, 0);
        return;
    }

    var sb_min: [16]f32 = undefined;
    var sb_max: [16]f32 = undefined;
    for (0..16) |sb| {
        var smin: f32 = std.math.inf(f32);
        var smax: f32 = -std.math.inf(f32);
        const start = sb * 16;
        const end = @min(start + 16, weights.len);
        for (weights[start..end]) |w| {
            if (w < smin) smin = w;
            if (w > smax) smax = w;
        }
        sb_min[sb] = if (smin == std.math.inf(f32)) 0.0 else smin;
        sb_max[sb] = if (smax == -std.math.inf(f32)) 0.0 else smax;
    }

    var max_range: f32 = 0.0;
    for (0..16) |sb| {
        const r = sb_max[sb] - sb_min[sb];
        if (r > max_range) max_range = r;
    }
    const d_val = max_range / (3.0 * 15.0);
    const d_eff = if (d_val == 0) 1e-8 else d_val;
    block.d = @floatCast(d_eff);

    var max_min: f32 = 0.0;
    for (0..16) |sb| {
        const am = @abs(sb_min[sb]);
        if (am > max_min) max_min = am;
    }
    const dmin_val = max_min / 15.0;
    const dmin_eff = if (dmin_val == 0) 1e-8 else dmin_val;
    block.dmin = @floatCast(dmin_eff);

    @memset(&block.scales, 0);
    @memset(&block.qs, 0);

    for (0..16) |sb| {
        const r = sb_max[sb] - sb_min[sb];
        const sc_f = std.math.clamp(@round(r / (3.0 * d_eff)), 0.0, 15.0);
        const sc_val: u8 = @intFromFloat(sc_f);
        const min_f = std.math.clamp(@round(-sb_min[sb] / dmin_eff), 0.0, 15.0);
        const min_val: u8 = @intFromFloat(min_f);
        block.scales[sb] = (sc_val & 0x0F) | ((min_val & 0x0F) << 4);

        const sub_scale = @as(f32, @floatCast(block.d)) * @as(f32, @floatFromInt(sc_val));
        const sub_min = @as(f32, @floatCast(block.dmin)) * @as(f32, @floatFromInt(min_val));
        const inv_scale = if (sub_scale != 0) 1.0 / sub_scale else 0.0;

        const start = sb * 16;
        const end = @min(start + 16, weights.len);
        const h: usize = start / 128;
        const rem = start % 128;
        const chunk32 = rem / 32;
        const shift: u3 = @intCast(chunk32 * 2);
        const offset_in_32: usize = rem % 32;
        const qs_base: usize = h * 32 + offset_in_32;

        for (start..end) |i| {
            const j = i - start;
            const q_f = std.math.clamp(@round((weights[i] + sub_min) * inv_scale), 0.0, 3.0);
            const code: u2 = @intFromFloat(q_f);

            block.qs[qs_base + j] |= (@as(u8, code) << shift);
        }
    }
}

pub fn dequantizeSuperBlockQ2_K(block: *const BlockQ2_K, count: usize, out: []f32) void {
    const d: f32 = @floatCast(block.d);
    const dmin: f32 = @floatCast(block.dmin);
    const n = @min(count, QK_K);

    for (0..16) |sb| {
        const start = sb * 16;
        if (start >= n) break;
        const end = @min(start + 16, n);
        const sub_scale = d * @as(f32, @floatFromInt(block.scales[sb] & 0x0F));
        const sub_min = dmin * @as(f32, @floatFromInt(block.scales[sb] >> 4));

        const h: usize = start / 128;
        const rem = start % 128;
        const chunk32 = rem / 32;
        const shift: u3 = @intCast(chunk32 * 2);
        const offset_in_32: usize = rem % 32;
        const qs_base: usize = h * 32 + offset_in_32;

        for (start..end) |i| {
            const j = i - start;
            const q_byte = block.qs[qs_base + j];
            const code: u2 = @truncate((q_byte >> shift) & 0x03);
            out[i] = @as(f32, @floatFromInt(code)) * sub_scale - sub_min;
        }
    }
}

// ---------------------------------------------------------------------------
// I-Quants: Non-Linear Codebook Vector Quantization (IQ4_NL)
// ---------------------------------------------------------------------------
pub const IQ4NL_TABLE: [16]f32 = .{
    -1.0000000, -0.7903226, -0.6129032, -0.4677419,
    -0.3387097, -0.2258065, -0.1290323, -0.0483871,
     0.0483871,  0.1290323,  0.2258065,  0.3387097,
     0.4677419,  0.6129032,  0.7903226,  1.0000000,
};

pub fn findClosestIQ4NL(val: f32) u4 {
    var best: u4 = 0;
    var min_d = std.math.inf(f32);
    inline for (0..16) |i| {
        const dist = @abs(val - IQ4NL_TABLE[i]);
        if (dist < min_d) {
            min_d = dist;
            best = @intCast(i);
        }
    }
    return best;
}

pub fn quantizeBlockIQ4_NL(block: []const f32, packed_out: []u8) f32 {
    if (block.len == 0) return 0.0;
    var max_abs: f32 = 0.0;
    for (block) |x| {
        const a = @abs(x);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0) max_abs = 1e-8;
    const inv_scale = 1.0 / max_abs;

    for (0..block.len) |i| {
        const norm = std.math.clamp(block[i] * inv_scale, -1.0, 1.0);
        const code = findClosestIQ4NL(norm);
        const byte_idx = i / 2;
        if (i % 2 == 0) {
            packed_out[byte_idx] = @as(u8, code);
        } else {
            packed_out[byte_idx] |= (@as(u8, code) << 4);
        }
    }
    return max_abs;
}

pub fn dequantizeBlockIQ4_NL(packed_in: []const u8, scale: f32, count: usize, out: []f32) void {
    for (0..count) |i| {
        const byte_idx = i / 2;
        const code: u4 = if (i % 2 == 0)
            @truncate(packed_in[byte_idx] & 0x0F)
        else
            @truncate((packed_in[byte_idx] >> 4) & 0x0F);

        out[i] = IQ4NL_TABLE[code] * scale;
    }
}

// ---------------------------------------------------------------------------
// Microscaling Formats: MXFP4 (OCP) and NVFP4 (Blackwell)
// ---------------------------------------------------------------------------
pub const FP4_E2M1_TABLE: [16]f32 = .{
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

/// OCP Microscaling MXFP4: 32 elements per block with E8M0 scale factor
pub fn dequantizeBlockMXFP4(packed_fp4: []const u8, e8m0_scale: u8, count: usize, out: []f32) void {
    const scale_exp = @as(f32, @floatFromInt(e8m0_scale)) - 127.0;
    const scale = std.math.pow(f32, 2.0, scale_exp);

    for (0..count) |i| {
        const byte_idx = i / 2;
        const code: u4 = if (i % 2 == 0)
            @truncate(packed_fp4[byte_idx] & 0x0F)
        else
            @truncate((packed_fp4[byte_idx] >> 4) & 0x0F);

        out[i] = FP4_E2M1_TABLE[code] * scale;
    }
}

/// NVIDIA Blackwell NVFP4: 16 elements per block with FP8 E4M3 scale factor
pub fn dequantizeBlockNVFP4(packed_fp4: []const u8, fp8_scale: u8, count: usize, out: []f32) void {
    var scale_arr: [1]f32 = undefined;
    const scale_bytes = [_]u8{fp8_scale};
    dequantizeFP8_E4M3(&scale_bytes, 1, &scale_arr);
    const scale = scale_arr[0];

    for (0..count) |i| {
        const byte_idx = i / 2;
        const code: u4 = if (i % 2 == 0)
            @truncate(packed_fp4[byte_idx] & 0x0F)
        else
            @truncate((packed_fp4[byte_idx] >> 4) & 0x0F);

        out[i] = FP4_E2M1_TABLE[code] * scale;
    }
}

