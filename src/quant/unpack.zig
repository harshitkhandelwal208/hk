//! Unpacking quantized blocks into a canonical integer form shared by the batch kernel and the
//! single token kernels of the formats that have no hand written dot product.
//!
//! "Legacy" blocks hold 32 weights with one float scale: weight = d * vals[j] (+ m for the
//! formats with an offset). "K" blocks hold 256 weights with one float scale per block and one
//! integer scale per 16 weights: weight = d * sc[g] * vals[j] - e * mn[g]. Values are signed
//! bytes for the codebook formats and small unsigned numbers for the plain K formats; whichever
//! the caller expects is documented at its use.

const std = @import("std");
const format = @import("../format.zig");
const blocks = @import("blocks.zig");
const dequant = @import("dequant.zig");
const tables = @import("tables.zig");
const isa = @import("isa.zig");

const StorageType = format.StorageType;

inline fn h(x: f16) f32 {
    return @floatCast(x);
}

inline fn blk(comptime B: type, row: []const u8, i: usize) *align(1) const B {
    return @ptrCast(row[i * @sizeOf(B) ..].ptr);
}

pub const LegacyBlock = struct { vals: [32]u8, d: f32, m: f32 };

pub inline fn unpackLegacy(comptime t: StorageType, row: []const u8, i: usize) LegacyBlock {
    var o: LegacyBlock = undefined;
    o.m = 0;
    switch (t) {
        .q8_0 => {
            const b = blk(blocks.BlockQ8_0, row, i);
            o.d = h(b.d);
            o.vals = @bitCast(b.qs);
        },
        .q4_0, .q4_1 => {
            const b = if (t == .q4_0) blk(blocks.BlockQ4_0, row, i) else blk(blocks.BlockQ4_1, row, i);
            o.d = h(b.d);
            if (t == .q4_1) o.m = h(b.m);
            const q: @Vector(16, u8) = b.qs;
            o.vals = std.simd.join(q & @as(@Vector(16, u8), @splat(15)), q >> @as(@Vector(16, u8), @splat(4)));
        },
        .q5_0, .q5_1 => {
            const b = if (t == .q5_0) blk(blocks.BlockQ5_0, row, i) else blk(blocks.BlockQ5_1, row, i);
            o.d = h(b.d);
            if (t == .q5_1) o.m = h(b.m);
            const qh = std.mem.readInt(u32, &b.qh, .little);
            for (0..16) |j| {
                const j5: u5 = @intCast(j);
                const xh0: u32 = ((qh >> j5) << 4) & 0x10;
                const xh1: u32 = (qh >> (j5 + 12)) & 0x10;
                o.vals[j] = @intCast((@as(u32, b.qs[j] & 0x0F)) | xh0);
                o.vals[j + 16] = @intCast((@as(u32, b.qs[j] >> 4)) | xh1);
            }
        },
        .mxfp4 => {
            const b = blk(blocks.BlockMXFP4, row, i);
            o.d = dequant.e8m0Half(b.e);
            const q: @Vector(16, u8) = b.qs;
            const idx = std.simd.join(q & @as(@Vector(16, u8), @splat(15)), q >> @as(@Vector(16, u8), @splat(4)));
            o.vals = isa.lookup16(fp4_tbl, idx);
        },
        .iq4_nl => {
            const b = blk(blocks.BlockIQ4NL, row, i);
            o.d = h(b.d);
            for (0..16) |j| {
                o.vals[j] = @bitCast(tables.kvalues_iq4nl[b.qs[j] & 0x0F]);
                o.vals[j + 16] = @bitCast(tables.kvalues_iq4nl[b.qs[j] >> 4]);
            }
        },
        else => unreachable,
    }
    return o;
}


pub const KBlock = struct { vals: [256]u8, sc: [16]i32, mn: [16]i32, d: f32, e: f32 };

pub inline fn unpackK(comptime t: StorageType, row: []const u8, i: usize) KBlock {
    var o: KBlock = undefined;
    switch (t) {
        .q2_k => {
            const b = blk(blocks.BlockQ2K, row, i);
            o.d = h(b.d);
            o.e = h(b.dmin);
            for (0..16) |g| {
                o.sc[g] = b.scales[g] & 0x0F;
                o.mn[g] = b.scales[g] >> 4;
            }
            // Group g = n * 8 + shift * 2 + half; its 16 values come from qs[32n + 16 half ..].
            for (0..2) |n| for (0..4) |s| for (0..2) |hf| {
                const g = n * 8 + s * 2 + hf;
                for (0..16) |l| o.vals[g * 16 + l] = (b.qs[32 * n + 16 * hf + l] >> @intCast(2 * s)) & 3;
            };
        },
        .q3_k => {
            const b = blk(blocks.BlockQ3K, row, i);
            o.d = h(b.d);
            o.e = o.d;
            var scales: [16]i8 = undefined;
            dequant.q3kScales(&b.scales, &scales);
            for (0..16) |g| {
                const s: i32 = @as(i32, scales[g]) - 32;
                o.sc[g] = s;
                o.mn[g] = 4 * s;
            }
            for (0..2) |n| for (0..4) |s| for (0..2) |hf| {
                const g = n * 8 + s * 2 + hf;
                const mask: u8 = @as(u8, 1) << @intCast(n * 4 + s);
                for (0..16) |l| {
                    const lo: u8 = (b.qs[32 * n + 16 * hf + l] >> @intCast(2 * s)) & 3;
                    const hi: u8 = if (b.hmask[16 * hf + l] & mask != 0) 4 else 0;
                    o.vals[g * 16 + l] = lo + hi;
                }
            };
        },
        .q4_k, .q5_k => {
            const b = if (t == .q4_k) blk(blocks.BlockQ4K, row, i) else blk(blocks.BlockQ5K, row, i);
            o.d = h(b.d);
            o.e = h(b.dmin);
            for (0..8) |j| {
                var sc: u8 = undefined;
                var mn: u8 = undefined;
                dequant.scaleMinK4(j, &b.scales, &sc, &mn);
                o.sc[2 * j] = sc;
                o.sc[2 * j + 1] = sc;
                o.mn[2 * j] = mn;
                o.mn[2 * j + 1] = mn;
            }
            for (0..4) |c| for (0..32) |l| {
                var lo: u8 = b.qs[32 * c + l] & 0x0F;
                var hi: u8 = b.qs[32 * c + l] >> 4;
                if (t == .q5_k) {
                    if (b.qh[l] & (@as(u8, 1) << @intCast(2 * c)) != 0) lo += 16;
                    if (b.qh[l] & (@as(u8, 2) << @intCast(2 * c)) != 0) hi += 16;
                }
                o.vals[64 * c + l] = lo;
                o.vals[64 * c + 32 + l] = hi;
            };
        },
        .q6_k => {
            const b = blk(blocks.BlockQ6K, row, i);
            o.d = h(b.d);
            o.e = o.d;
            for (0..16) |g| {
                o.sc[g] = b.scales[g];
                o.mn[g] = 32 * @as(i32, b.scales[g]);
            }
            for (0..2) |n| for (0..32) |l| {
                const h8: u8 = b.qh[32 * n + l];
                const l0 = b.ql[64 * n + l];
                const l1 = b.ql[64 * n + l + 32];
                o.vals[128 * n + l] = (l0 & 0x0F) | ((h8 & 3) << 4);
                o.vals[128 * n + l + 32] = (l1 & 0x0F) | (((h8 >> 2) & 3) << 4);
                o.vals[128 * n + l + 64] = (l0 >> 4) | (((h8 >> 4) & 3) << 4);
                o.vals[128 * n + l + 96] = (l1 >> 4) | (((h8 >> 6) & 3) << 4);
            };
        },
        .iq4_xs => {
            const b = blk(blocks.BlockIQ4XS, row, i);
            o.d = h(b.d);
            o.e = o.d;
            const scales_h = std.mem.readInt(u16, &b.scales_h, .little);
            for (0..8) |ib| {
                const lo: u32 = (b.scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0x0F;
                const hi: u32 = (scales_h >> @intCast(2 * ib)) & 3;
                const ls: i32 = @as(i32, @intCast(lo | (hi << 4))) - 32;
                o.sc[2 * ib] = ls;
                o.sc[2 * ib + 1] = ls;
                o.mn[2 * ib] = 0;
                o.mn[2 * ib + 1] = 0;
                const qs = b.qs[16 * ib ..][0..16];
                for (0..16) |j| {
                    o.vals[32 * ib + j] = @bitCast(tables.kvalues_iq4nl[qs[j] & 0x0F]);
                    o.vals[32 * ib + j + 16] = @bitCast(tables.kvalues_iq4nl[qs[j] >> 4]);
                }
            }
        },
        .iq2_xxs => {
            const b = blk(blocks.BlockIQ2XXS, row, i);
            o.d = h(b.d) * 0.125;
            o.e = 0;
            o.mn = @splat(0);
            for (0..8) |ib| {
                const aux1 = std.mem.readInt(u32, b.qs[8 * ib + 4 ..][0..4], .little);
                const s: i32 = 1 + 2 * @as(i32, @intCast(aux1 >> 28));
                o.sc[2 * ib] = s;
                o.sc[2 * ib + 1] = s;
                for (0..4) |l| {
                    const grid = &tables.iq2xxs_grid[b.qs[8 * ib + l]];
                    const signs = tables.ksigns_iq2xs[(aux1 >> @intCast(7 * l)) & 127];
                    _ = grid;
                    put8(&o.vals, 32 * ib + 8 * l, applySigns(grid8(tables.iq2xxs_grid, b.qs[8 * ib + l]), signs));
                }
            }
        },
        .iq2_xs => {
            const b = blk(blocks.BlockIQ2XS, row, i);
            o.d = h(b.d) * 0.125;
            o.e = 0;
            o.mn = @splat(0);
            for (0..8) |ib| {
                o.sc[2 * ib] = 1 + 2 * @as(i32, b.scales[ib] & 0x0F);
                o.sc[2 * ib + 1] = 1 + 2 * @as(i32, b.scales[ib] >> 4);
                for (0..4) |l| {
                    const q: u16 = std.mem.readInt(u16, b.qs[2 * (4 * ib + l) ..][0..2], .little);
                    const grid = &tables.iq2xs_grid[q & 511];
                    const signs = tables.ksigns_iq2xs[q >> 9];
                    _ = grid;
                    put8(&o.vals, 32 * ib + 8 * l, applySigns(grid8(tables.iq2xs_grid, q & 511), signs));
                }
            }
        },
        .iq2_s => {
            const b = blk(blocks.BlockIQ2S, row, i);
            o.d = h(b.d) * 0.125;
            o.e = 0;
            o.mn = @splat(0);
            for (0..8) |ib| {
                o.sc[2 * ib] = 1 + 2 * @as(i32, b.scales[ib] & 0x0F);
                o.sc[2 * ib + 1] = 1 + 2 * @as(i32, b.scales[ib] >> 4);
                for (0..4) |l| {
                    const hi: u32 = (@as(u32, b.qh[ib]) << @intCast(8 - 2 * l)) & 0x300;
                    const grid = &tables.iq2s_grid[b.qs[4 * ib + l] | hi];
                    const signs = b.signs[4 * ib + l];
                    _ = grid;
                    put8(&o.vals, 32 * ib + 8 * l, applySigns(grid8(tables.iq2s_grid, b.qs[4 * ib + l] | hi), signs));
                }
            }
        },
        .iq3_xxs => {
            const b = blk(blocks.BlockIQ3XXS, row, i);
            o.d = h(b.d) * 0.25;
            o.e = 0;
            o.mn = @splat(0);
            const scales_and_signs = b.qs[64..];
            for (0..8) |ib| {
                const aux = std.mem.readInt(u32, scales_and_signs[4 * ib ..][0..4], .little);
                const s: i32 = 1 + 2 * @as(i32, @intCast(aux >> 28));
                o.sc[2 * ib] = s;
                o.sc[2 * ib + 1] = s;
                for (0..4) |l| {
                    const signs = tables.ksigns_iq2xs[(aux >> @intCast(7 * l)) & 127];
                    const g1 = &tables.iq3xxs_grid[b.qs[8 * ib + 2 * l + 0]];
                    const g2 = &tables.iq3xxs_grid[b.qs[8 * ib + 2 * l + 1]];
                    _ = g1;
                    _ = g2;
                    const lo: u64 = grid4(tables.iq3xxs_grid, b.qs[8 * ib + 2 * l + 0]);
                    const hi: u64 = grid4(tables.iq3xxs_grid, b.qs[8 * ib + 2 * l + 1]);
                    put8(&o.vals, 32 * ib + 8 * l, applySigns(lo | (hi << 32), signs));
                }
            }
        },
        .iq3_s => {
            const b = blk(blocks.BlockIQ3S, row, i);
            o.d = h(b.d);
            o.e = 0;
            o.mn = @splat(0);
            for (0..8) |ib| {
                const nib = (b.scales[ib / 2] >> @intCast(4 * (ib & 1))) & 0x0F;
                const s: i32 = 1 + 2 * @as(i32, nib);
                o.sc[2 * ib] = s;
                o.sc[2 * ib + 1] = s;
            }
            for (0..32) |g| {
                const e0 = 2 * g;
                const e1 = 2 * g + 1;
                const idx0: u32 = b.qs[e0] | (@as(u32, (b.qh[e0 / 8] >> @intCast(e0 % 8)) & 1) << 8);
                const idx1: u32 = b.qs[e1] | (@as(u32, (b.qh[e1 / 8] >> @intCast(e1 % 8)) & 1) << 8);
                const g1 = &tables.iq3s_grid[idx0];
                const g2 = &tables.iq3s_grid[idx1];
                const signs = b.signs[g];
                const lo: u64 = grid4(tables.iq3s_grid, idx0);
                const hi: u64 = grid4(tables.iq3s_grid, idx1);
                _ = g1;
                _ = g2;
                put8(&o.vals, 8 * g, applySigns(lo | (hi << 32), signs));
            }
        },
        .iq1_s => {
            const b = blk(blocks.BlockIQ1S, row, i);
            // weight = d * (2 s + 1) * (grid + delta) = (d / 8) * (2 s + 1) * (8 grid + 8 delta)
            comptime std.debug.assert(tables.iq1s_delta == 0.125);
            o.d = h(b.d) * 0.125;
            o.e = 0;
            o.mn = @splat(0);
            for (0..8) |ib| {
                const qh: u16 = std.mem.readInt(u16, b.qh[2 * ib ..][0..2], .little);
                const s: i32 = 1 + 2 * @as(i32, (qh >> 12) & 7);
                o.sc[2 * ib] = s;
                o.sc[2 * ib + 1] = s;
                const delta: i32 = if (qh & 0x8000 != 0) -1 else 1;
                for (0..4) |l| {
                    const idx: u32 = b.qs[4 * ib + l] | (@as(u32, (qh >> @intCast(3 * l)) & 7) << 8);
                    const grid = &tables.iq1s_grid[idx];
                    o.vals[32 * ib + 8 * l ..][0..8].* = @bitCast(gridDelta(grid, delta));
                }
            }
        },
        .iq1_m => {
            const b = blk(blocks.BlockIQ1M, row, i);
            var scw: [4]u16 = undefined;
            for (0..4) |k| scw[k] = std.mem.readInt(u16, b.scales[2 * k ..][0..2], .little);
            const scale_bits: u16 = (scw[0] >> 12) | ((scw[1] >> 8) & 0x00F0) | ((scw[2] >> 4) & 0x0F00) | (scw[3] & 0xF000);
            o.d = h(@bitCast(scale_bits)) * 0.125;
            o.e = 0;
            o.mn = @splat(0);
            for (0..8) |ib| {
                const w = scw[ib / 2];
                const base: u4 = @intCast(6 * (ib % 2));
                o.sc[2 * ib] = 1 + 2 * @as(i32, (w >> base) & 7);
                o.sc[2 * ib + 1] = 1 + 2 * @as(i32, (w >> (base + 3)) & 7);
                const qh0 = b.qh[2 * ib];
                const qh1 = b.qh[2 * ib + 1];
                const idx = [4]u32{
                    b.qs[4 * ib + 0] | ((@as(u32, qh0) << 8) & 0x700),
                    b.qs[4 * ib + 1] | ((@as(u32, qh0) << 4) & 0x700),
                    b.qs[4 * ib + 2] | ((@as(u32, qh1) << 8) & 0x700),
                    b.qs[4 * ib + 3] | ((@as(u32, qh1) << 4) & 0x700),
                };
                const delta = [4]i32{
                    if (qh0 & 0x08 != 0) -1 else 1,
                    if (qh0 & 0x80 != 0) -1 else 1,
                    if (qh1 & 0x08 != 0) -1 else 1,
                    if (qh1 & 0x80 != 0) -1 else 1,
                };
                for (0..4) |l| {
                    const grid = &tables.iq1s_grid[idx[l]];
                    o.vals[32 * ib + 8 * l ..][0..8].* = @bitCast(gridDelta(grid, delta[l]));
                }
            }
        },
        .tq1_0 => {
            const b = blk(blocks.BlockTQ1_0, row, i);
            o.d = h(b.d);
            o.e = 0;
            o.sc = @splat(1);
            o.mn = @splat(0);
            const pow3 = [_]u8{ 1, 3, 9, 27, 81 };
            var p: usize = 0;
            const q32: @Vector(32, u8) = b.qs[0..32].*;
            inline for (0..5) |n| {
                o.vals[p..][0..32].* = ternaryVec(32, q32, pow3[n]);
                p += 32;
            }
            const q16: @Vector(16, u8) = b.qs[32..48].*;
            inline for (0..5) |n| {
                o.vals[p..][0..16].* = ternaryVec(16, q16, pow3[n]);
                p += 16;
            }
            const q4: @Vector(4, u8) = b.qh;
            inline for (0..4) |n| {
                o.vals[p..][0..4].* = ternaryVec(4, q4, pow3[n]);
                p += 4;
            }
        },
        .tq2_0 => {
            const b = blk(blocks.BlockTQ2_0, row, i);
            o.d = h(b.d);
            o.e = 0;
            o.sc = @splat(1);
            o.mn = @splat(0);
            var p: usize = 0;
            var j: usize = 0;
            while (j < 64) : (j += 32) for (0..4) |l| for (0..32) |k| {
                const q: i32 = (b.qs[j + k] >> @intCast(2 * l)) & 3;
                o.vals[p] = @bitCast(@as(i8, @intCast(q - 1)));
                p += 1;
            };
        },
        else => unreachable,
    }
    return o;
}


/// A codebook value with the sign bit `j` of `signs` applied, as a signed byte pattern.
inline fn signedGrid(v: u8, signs: u8, j: usize) u8 {
    const neg = signs & tables.kmask_iq2xs[j] != 0;
    const x: i8 = @intCast(v);
    return @bitCast(if (neg) -x else x);
}

/// One ternary digit (-1, 0, 1) of a TQ1_0 byte, as a signed byte pattern.
inline fn ternaryDigit(q: u8, mul: u8) u8 {
    const v: u8 = q *% mul;
    const t: u32 = (@as(u32, v) * 3) >> 8;
    return @bitCast(@as(i8, @intCast(@as(i32, @intCast(t)) - 1)));
}

/// The FP4 codebook in both 16 byte halves, for `isa.lookup16`.
const fp4_tbl: isa.U32 = blk: {
    var t: [32]u8 = undefined;
    for (0..16) |i| {
        t[i] = @bitCast(tables.kvalues_fp4[i]);
        t[i + 16] = t[i];
    }
    break :blk t;
};

/// 0xFF in byte j where bit j of the index is set. Turns a sign byte into a byte mask.
const sign_masks: [256]u64 = blk: {
    @setEvalBranchQuota(100_000);
    var t: [256]u64 = undefined;
    for (0..256) |i| {
        var m: u64 = 0;
        for (0..8) |j| {
            if (i & (1 << j) != 0) m |= @as(u64, 0xFF) << @intCast(8 * j);
        }
        t[i] = m;
    }
    break :blk t;
};

/// Eight grid bytes (all between 1 and 127) with signs applied, negating byte j when bit j of
/// `signs` is set: (x ^ 0xFF) + 1 per byte, which never carries because x is not zero.
inline fn applySigns(g: u64, signs: u8) u64 {
    const m = sign_masks[signs];
    return (g ^ m) +% (m & 0x0101010101010101);
}

inline fn grid8(comptime table: anytype, idx: usize) u64 {
    return std.mem.readInt(u64, &table[idx], .little);
}

inline fn grid4(comptime table: anytype, idx: usize) u32 {
    return std.mem.readInt(u32, &table[idx], .little);
}

inline fn put8(vals: *[256]u8, at: usize, v: u64) void {
    std.mem.writeInt(u64, vals[at..][0..8], v, .little);
}

/// 8 * grid + delta for eight grid values of an IQ1 codebook entry.
inline fn gridDelta(grid: *const [8]i8, delta: i32) @Vector(8, i8) {
    const g: @Vector(8, i8) = grid.*;
    return g * @as(@Vector(8, i8), @splat(8)) + @as(@Vector(8, i8), @splat(@as(i8, @intCast(delta))));
}

/// Ternary digit (-1, 0, 1) for each byte of `q` at power of three `mul`: ((q * mul) * 3) >> 8 - 1.
inline fn ternaryVec(comptime n: usize, q: @Vector(n, u8), mul: u8) @Vector(n, u8) {
    const v: @Vector(n, u8) = q *% @as(@Vector(n, u8), @splat(mul));
    const t: @Vector(n, u16) = (@as(@Vector(n, u16), v) * @as(@Vector(n, u16), @splat(3))) >> @as(@Vector(n, u4), @splat(8));
    return @bitCast(@as(@Vector(n, i8), @intCast(@as(@Vector(n, i16), @intCast(t)) - @as(@Vector(n, i16), @splat(1)))));
}
