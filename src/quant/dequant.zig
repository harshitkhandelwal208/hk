//! Exact decoders for every quantized format. These are the reference implementation the fast
//! dot product kernels are tested against, and the path used for formats without a fast kernel.
//!
//! Every function decodes one block into a caller supplied buffer of `elems` floats. Nothing
//! here allocates, so decoding a whole tensor needs only a block sized stack buffer.

const std = @import("std");
const format = @import("../format.zig");
const blocks = @import("blocks.zig");
const tables = @import("tables.zig");

const StorageType = format.StorageType;
const f32_ = f32;

inline fn h(x: f16) f32_ {
    return @floatCast(x);
}

inline fn blk(comptime B: type, src: []const u8) *align(1) const B {
    std.debug.assert(src.len >= @sizeOf(B));
    return @ptrCast(src.ptr);
}

pub fn q4_0(src: []const u8, out: *[32]f32_) void {
    const b = blk(blocks.BlockQ4_0, src);
    const d = h(b.d);
    for (0..16) |j| {
        out[j] = @as(f32_, @floatFromInt(@as(i32, b.qs[j] & 0x0F) - 8)) * d;
        out[j + 16] = @as(f32_, @floatFromInt(@as(i32, b.qs[j] >> 4) - 8)) * d;
    }
}

pub fn q4_1(src: []const u8, out: *[32]f32_) void {
    const b = blk(blocks.BlockQ4_1, src);
    const d = h(b.d);
    const m = h(b.m);
    for (0..16) |j| {
        out[j] = @as(f32_, @floatFromInt(b.qs[j] & 0x0F)) * d + m;
        out[j + 16] = @as(f32_, @floatFromInt(b.qs[j] >> 4)) * d + m;
    }
}

pub fn q5_0(src: []const u8, out: *[32]f32_) void {
    const b = blk(blocks.BlockQ5_0, src);
    const d = h(b.d);
    const qh = std.mem.readInt(u32, &b.qh, .little);
    for (0..16) |j| {
        const j5: u5 = @intCast(j);
        const xh0: u32 = ((qh >> j5) << 4) & 0x10;
        const xh1: u32 = (qh >> (j5 + 12)) & 0x10;
        const x0: i32 = @as(i32, @intCast((@as(u32, b.qs[j] & 0x0F)) | xh0)) - 16;
        const x1: i32 = @as(i32, @intCast((@as(u32, b.qs[j] >> 4)) | xh1)) - 16;
        out[j] = @as(f32_, @floatFromInt(x0)) * d;
        out[j + 16] = @as(f32_, @floatFromInt(x1)) * d;
    }
}

pub fn q5_1(src: []const u8, out: *[32]f32_) void {
    const b = blk(blocks.BlockQ5_1, src);
    const d = h(b.d);
    const m = h(b.m);
    const qh = std.mem.readInt(u32, &b.qh, .little);
    for (0..16) |j| {
        const j5: u5 = @intCast(j);
        const xh0: u32 = ((qh >> j5) << 4) & 0x10;
        const xh1: u32 = (qh >> (j5 + 12)) & 0x10;
        const x0: u32 = (@as(u32, b.qs[j] & 0x0F)) | xh0;
        const x1: u32 = (@as(u32, b.qs[j] >> 4)) | xh1;
        out[j] = @as(f32_, @floatFromInt(x0)) * d + m;
        out[j + 16] = @as(f32_, @floatFromInt(x1)) * d + m;
    }
}

pub fn q8_0(src: []const u8, out: *[32]f32_) void {
    const b = blk(blocks.BlockQ8_0, src);
    const d = h(b.d);
    for (0..32) |j| out[j] = @as(f32_, @floatFromInt(b.qs[j])) * d;
}

pub fn q2_k(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockQ2K, src);
    const d = h(b.d);
    const min = h(b.dmin);
    var is: usize = 0;
    var o: usize = 0;
    var q: usize = 0;
    var n: usize = 0;
    while (n < 256) : (n += 128) {
        var shift: u8 = 0;
        for (0..4) |_| {
            var sc = b.scales[is];
            is += 1;
            var dl = d * @as(f32_, @floatFromInt(sc & 0x0F));
            var ml = min * @as(f32_, @floatFromInt(sc >> 4));
            for (0..16) |l| {
                out[o] = dl * @as(f32_, @floatFromInt((b.qs[q + l] >> @as(u3, @intCast(shift))) & 3)) - ml;
                o += 1;
            }
            sc = b.scales[is];
            is += 1;
            dl = d * @as(f32_, @floatFromInt(sc & 0x0F));
            ml = min * @as(f32_, @floatFromInt(sc >> 4));
            for (0..16) |l| {
                out[o] = dl * @as(f32_, @floatFromInt((b.qs[q + l + 16] >> @as(u3, @intCast(shift))) & 3)) - ml;
                o += 1;
            }
            shift += 2;
        }
        q += 32;
    }
}

/// Unpacks the sixteen 6 bit scales of a Q3_K block into signed values (stored with +32 bias).
pub fn q3kScales(packed_scales: *const [12]u8, scales: *[16]i8) void {
    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;
    var aux: [4]u32 = undefined;
    inline for (0..3) |i| aux[i] = std.mem.readInt(u32, packed_scales[i * 4 ..][0..4], .little);
    const tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    inline for (0..4) |i| {
        const bytes: [4]u8 = @bitCast(std.mem.nativeToLittle(u32, aux[i]));
        inline for (0..4) |k| scales[i * 4 + k] = @bitCast(bytes[k]);
    }
}

pub fn q3_k(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockQ3K, src);
    const d_all = h(b.d);
    var scales: [16]i8 = undefined;
    q3kScales(&b.scales, &scales);
    var is: usize = 0;
    var o: usize = 0;
    var q: usize = 0;
    var m: u8 = 1;
    var n: usize = 0;
    while (n < 256) : (n += 128) {
        var shift: u8 = 0;
        for (0..4) |_| {
            var dl = d_all * @as(f32_, @floatFromInt(@as(i32, scales[is]) - 32));
            is += 1;
            for (0..16) |l| {
                const lo: i32 = (b.qs[q + l] >> @as(u3, @intCast(shift))) & 3;
                const hi: i32 = if (b.hmask[l] & m != 0) 0 else 4;
                out[o] = dl * @as(f32_, @floatFromInt(lo - hi));
                o += 1;
            }
            dl = d_all * @as(f32_, @floatFromInt(@as(i32, scales[is]) - 32));
            is += 1;
            for (0..16) |l| {
                const lo: i32 = (b.qs[q + l + 16] >> @as(u3, @intCast(shift))) & 3;
                const hi: i32 = if (b.hmask[l + 16] & m != 0) 0 else 4;
                out[o] = dl * @as(f32_, @floatFromInt(lo - hi));
                o += 1;
            }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

/// Splits the packed 6 bit scale and min pair for sub block `j` of a Q4_K or Q5_K block.
pub inline fn scaleMinK4(j: usize, q: *const [12]u8, sc: *u8, mn: *u8) void {
    if (j < 4) {
        sc.* = q[j] & 63;
        mn.* = q[j + 4] & 63;
    } else {
        sc.* = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        mn.* = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

pub fn q4_k(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockQ4K, src);
    const d = h(b.d);
    const min = h(b.dmin);
    var is: usize = 0;
    var o: usize = 0;
    var q: usize = 0;
    var j: usize = 0;
    while (j < 256) : (j += 64) {
        var sc: u8 = undefined;
        var mn: u8 = undefined;
        scaleMinK4(is, &b.scales, &sc, &mn);
        const d1 = d * @as(f32_, @floatFromInt(sc));
        const m1 = min * @as(f32_, @floatFromInt(mn));
        scaleMinK4(is + 1, &b.scales, &sc, &mn);
        const d2 = d * @as(f32_, @floatFromInt(sc));
        const m2 = min * @as(f32_, @floatFromInt(mn));
        for (0..32) |l| {
            out[o + l] = d1 * @as(f32_, @floatFromInt(b.qs[q + l] & 0x0F)) - m1;
            out[o + 32 + l] = d2 * @as(f32_, @floatFromInt(b.qs[q + l] >> 4)) - m2;
        }
        o += 64;
        q += 32;
        is += 2;
    }
}

pub fn q5_k(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockQ5K, src);
    const d = h(b.d);
    const min = h(b.dmin);
    var is: usize = 0;
    var o: usize = 0;
    var q: usize = 0;
    var hbit1: u8 = 1;
    var hbit2: u8 = 2;
    var j: usize = 0;
    while (j < 256) : (j += 64) {
        var sc: u8 = undefined;
        var mn: u8 = undefined;
        scaleMinK4(is, &b.scales, &sc, &mn);
        const d1 = d * @as(f32_, @floatFromInt(sc));
        const m1 = min * @as(f32_, @floatFromInt(mn));
        scaleMinK4(is + 1, &b.scales, &sc, &mn);
        const d2 = d * @as(f32_, @floatFromInt(sc));
        const m2 = min * @as(f32_, @floatFromInt(mn));
        for (0..32) |l| {
            const lo: u32 = (b.qs[q + l] & 0x0F) + @as(u32, if (b.qh[l] & hbit1 != 0) 16 else 0);
            const hi: u32 = (b.qs[q + l] >> 4) + @as(u32, if (b.qh[l] & hbit2 != 0) 16 else 0);
            out[o + l] = d1 * @as(f32_, @floatFromInt(lo)) - m1;
            out[o + 32 + l] = d2 * @as(f32_, @floatFromInt(hi)) - m2;
        }
        o += 64;
        q += 32;
        is += 2;
        hbit1 <<= 2;
        hbit2 <<= 2;
    }
}

pub fn q6_k(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockQ6K, src);
    const d = h(b.d);
    var o: usize = 0;
    var ql: usize = 0;
    var qh: usize = 0;
    var sc: usize = 0;
    var n: usize = 0;
    while (n < 256) : (n += 128) {
        for (0..32) |l| {
            const is = l / 16;
            const lo0: i32 = b.ql[ql + l] & 0x0F;
            const lo1: i32 = b.ql[ql + l + 32] & 0x0F;
            const hi0: i32 = b.ql[ql + l] >> 4;
            const hi1: i32 = b.ql[ql + l + 32] >> 4;
            const h8: i32 = b.qh[qh + l];
            const q1 = (lo0 | (((h8 >> 0) & 3) << 4)) - 32;
            const q2 = (lo1 | (((h8 >> 2) & 3) << 4)) - 32;
            const q3 = (hi0 | (((h8 >> 4) & 3) << 4)) - 32;
            const q4 = (hi1 | (((h8 >> 6) & 3) << 4)) - 32;
            out[o + l + 0] = d * @as(f32_, @floatFromInt(@as(i32, b.scales[sc + is + 0]) * q1));
            out[o + l + 32] = d * @as(f32_, @floatFromInt(@as(i32, b.scales[sc + is + 2]) * q2));
            out[o + l + 64] = d * @as(f32_, @floatFromInt(@as(i32, b.scales[sc + is + 4]) * q3));
            out[o + l + 96] = d * @as(f32_, @floatFromInt(@as(i32, b.scales[sc + is + 6]) * q4));
        }
        o += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

pub fn q8_k(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockQ8K, src);
    for (0..256) |j| out[j] = b.d * @as(f32_, @floatFromInt(b.qs[j]));
}

inline fn signOf(sign_byte: u8, j: usize) f32_ {
    return if (sign_byte & tables.kmask_iq2xs[j] != 0) -1.0 else 1.0;
}

pub fn iq2_xxs(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockIQ2XXS, src);
    const d = h(b.d);
    const qs_bytes = &b.qs;
    var o: usize = 0;
    for (0..8) |ib32| {
        const grid_idx = qs_bytes[8 * ib32 ..][0..4];
        const aux1 = std.mem.readInt(u32, qs_bytes[8 * ib32 + 4 ..][0..4], .little);
        const db = d * (0.5 + @as(f32_, @floatFromInt(aux1 >> 28))) * 0.25;
        for (0..4) |l| {
            const grid = &tables.iq2xxs_grid[grid_idx[l]];
            const signs = tables.ksigns_iq2xs[(aux1 >> @intCast(7 * l)) & 127];
            for (0..8) |j| out[o + j] = db * @as(f32_, @floatFromInt(grid[j])) * signOf(signs, j);
            o += 8;
        }
    }
}

pub fn iq2_xs(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockIQ2XS, src);
    const d = h(b.d);
    var o: usize = 0;
    for (0..8) |ib32| {
        const db0 = d * (0.5 + @as(f32_, @floatFromInt(b.scales[ib32] & 0x0F))) * 0.25;
        const db1 = d * (0.5 + @as(f32_, @floatFromInt(b.scales[ib32] >> 4))) * 0.25;
        for (0..4) |l| {
            const q: u16 = std.mem.readInt(u16, b.qs[2 * (4 * ib32 + l) ..][0..2], .little);
            const grid = &tables.iq2xs_grid[q & 511];
            const signs = tables.ksigns_iq2xs[q >> 9];
            const db = if (l < 2) db0 else db1;
            for (0..8) |j| out[o + j] = db * @as(f32_, @floatFromInt(grid[j])) * signOf(signs, j);
            o += 8;
        }
    }
}

pub fn iq2_s(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockIQ2S, src);
    const d = h(b.d);
    var o: usize = 0;
    for (0..8) |ib32| {
        const db0 = d * (0.5 + @as(f32_, @floatFromInt(b.scales[ib32] & 0x0F))) * 0.25;
        const db1 = d * (0.5 + @as(f32_, @floatFromInt(b.scales[ib32] >> 4))) * 0.25;
        for (0..4) |l| {
            const hi: u32 = (@as(u32, b.qh[ib32]) << @intCast(8 - 2 * l)) & 0x300;
            const grid = &tables.iq2s_grid[b.qs[4 * ib32 + l] | hi];
            const signs = b.signs[4 * ib32 + l];
            const db = if (l < 2) db0 else db1;
            for (0..8) |j| out[o + j] = db * @as(f32_, @floatFromInt(grid[j])) * signOf(signs, j);
            o += 8;
        }
    }
}

pub fn iq3_xxs(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockIQ3XXS, src);
    const d = h(b.d);
    const scales_and_signs = b.qs[64..];
    var o: usize = 0;
    for (0..8) |ib32| {
        const aux = std.mem.readInt(u32, scales_and_signs[4 * ib32 ..][0..4], .little);
        const db = d * (0.5 + @as(f32_, @floatFromInt(aux >> 28))) * 0.5;
        for (0..4) |l| {
            const signs = tables.ksigns_iq2xs[(aux >> @intCast(7 * l)) & 127];
            const g1 = &tables.iq3xxs_grid[b.qs[8 * ib32 + 2 * l + 0]];
            const g2 = &tables.iq3xxs_grid[b.qs[8 * ib32 + 2 * l + 1]];
            for (0..4) |j| {
                out[o + j] = db * @as(f32_, @floatFromInt(g1[j])) * signOf(signs, j);
                out[o + j + 4] = db * @as(f32_, @floatFromInt(g2[j])) * signOf(signs, j + 4);
            }
            o += 8;
        }
    }
}

pub fn iq3_s(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockIQ3S, src);
    const d = h(b.d);
    var o: usize = 0;
    // One group is 8 weights: two 4 value grid entries sharing a sign byte. 32 groups per block,
    // four groups per scale nibble.
    for (0..32) |g| {
        const scale_nibble = (b.scales[g / 8] >> @intCast(4 * ((g / 4) & 1))) & 0x0F;
        const db = d * @as(f32_, @floatFromInt(1 + 2 * @as(u32, scale_nibble)));
        const e0 = 2 * g;
        const e1 = 2 * g + 1;
        const idx0: u32 = b.qs[e0] | (@as(u32, (b.qh[e0 / 8] >> @intCast(e0 % 8)) & 1) << 8);
        const idx1: u32 = b.qs[e1] | (@as(u32, (b.qh[e1 / 8] >> @intCast(e1 % 8)) & 1) << 8);
        const g1 = &tables.iq3s_grid[idx0];
        const g2 = &tables.iq3s_grid[idx1];
        const signs = b.signs[g];
        for (0..4) |j| {
            out[o + j] = db * @as(f32_, @floatFromInt(g1[j])) * signOf(signs, j);
            out[o + j + 4] = db * @as(f32_, @floatFromInt(g2[j])) * signOf(signs, j + 4);
        }
        o += 8;
    }
}

pub fn iq1_s(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockIQ1S, src);
    const d = h(b.d);
    var o: usize = 0;
    for (0..8) |ib| {
        const qh: u16 = std.mem.readInt(u16, b.qh[2 * ib ..][0..2], .little);
        const dl = d * @as(f32_, @floatFromInt(2 * @as(u32, (qh >> 12) & 7) + 1));
        const delta: f32_ = if (qh & 0x8000 != 0) -tables.iq1s_delta else tables.iq1s_delta;
        for (0..4) |l| {
            const idx: u32 = b.qs[4 * ib + l] | (@as(u32, (qh >> @intCast(3 * l)) & 7) << 8);
            const grid = &tables.iq1s_grid[idx];
            for (0..8) |j| out[o + j] = dl * (@as(f32_, @floatFromInt(grid[j])) + delta);
            o += 8;
        }
    }
}

pub fn iq1_m(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockIQ1M, src);
    var sc: [4]u16 = undefined;
    for (0..4) |i| sc[i] = std.mem.readInt(u16, b.scales[2 * i ..][0..2], .little);
    // The f16 scale is spread over the top nibble of each of the four 16 bit scale words.
    const scale_bits: u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const d = h(@bitCast(scale_bits));
    var o: usize = 0;
    for (0..8) |ib| {
        const w = sc[ib / 2];
        const base: u4 = @intCast(6 * (ib % 2));
        const dl1 = d * @as(f32_, @floatFromInt(2 * @as(u32, (w >> base) & 7) + 1));
        const dl2 = d * @as(f32_, @floatFromInt(2 * @as(u32, (w >> (base + 3)) & 7) + 1));
        const qh0 = b.qh[2 * ib];
        const qh1 = b.qh[2 * ib + 1];
        const idx = [4]u32{
            b.qs[4 * ib + 0] | ((@as(u32, qh0) << 8) & 0x700),
            b.qs[4 * ib + 1] | ((@as(u32, qh0) << 4) & 0x700),
            b.qs[4 * ib + 2] | ((@as(u32, qh1) << 8) & 0x700),
            b.qs[4 * ib + 3] | ((@as(u32, qh1) << 4) & 0x700),
        };
        const delta = [4]f32_{
            if (qh0 & 0x08 != 0) -tables.iq1s_delta else tables.iq1s_delta,
            if (qh0 & 0x80 != 0) -tables.iq1s_delta else tables.iq1s_delta,
            if (qh1 & 0x08 != 0) -tables.iq1s_delta else tables.iq1s_delta,
            if (qh1 & 0x80 != 0) -tables.iq1s_delta else tables.iq1s_delta,
        };
        for (0..4) |l| {
            const dl = if (l < 2) dl1 else dl2;
            const grid = &tables.iq1s_grid[idx[l]];
            for (0..8) |j| out[o + j] = dl * (@as(f32_, @floatFromInt(grid[j])) + delta[l]);
            o += 8;
        }
    }
}

pub fn iq4_nl(src: []const u8, out: *[32]f32_) void {
    const b = blk(blocks.BlockIQ4NL, src);
    const d = h(b.d);
    for (0..16) |j| {
        out[j] = d * @as(f32_, @floatFromInt(tables.kvalues_iq4nl[b.qs[j] & 0x0F]));
        out[j + 16] = d * @as(f32_, @floatFromInt(tables.kvalues_iq4nl[b.qs[j] >> 4]));
    }
}

pub fn iq4_xs(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockIQ4XS, src);
    const d = h(b.d);
    const scales_h = std.mem.readInt(u16, &b.scales_h, .little);
    var o: usize = 0;
    for (0..8) |ib| {
        const lo: u32 = (b.scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0x0F;
        const hi: u32 = (scales_h >> @intCast(2 * ib)) & 3;
        const ls: i32 = @as(i32, @intCast(lo | (hi << 4))) - 32;
        const dl = d * @as(f32_, @floatFromInt(ls));
        const qs = b.qs[16 * ib ..][0..16];
        for (0..16) |j| {
            out[o + j] = dl * @as(f32_, @floatFromInt(tables.kvalues_iq4nl[qs[j] & 0x0F]));
            out[o + j + 16] = dl * @as(f32_, @floatFromInt(tables.kvalues_iq4nl[qs[j] >> 4]));
        }
        o += 32;
    }
}

const pow3 = [_]u8{ 1, 3, 9, 27, 81 };

inline fn ternary(q: u8, mul: u8) f32_ {
    const v: u8 = q *% mul;
    const t: u32 = (@as(u32, v) * 3) >> 8;
    return @as(f32_, @floatFromInt(@as(i32, @intCast(t)) - 1));
}

pub fn tq1_0(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockTQ1_0, src);
    const d = h(b.d);
    var o: usize = 0;
    for (0..5) |n| {
        for (0..32) |m| {
            out[o] = ternary(b.qs[m], pow3[n]) * d;
            o += 1;
        }
    }
    for (0..5) |n| {
        for (0..16) |m| {
            out[o] = ternary(b.qs[32 + m], pow3[n]) * d;
            o += 1;
        }
    }
    for (0..4) |n| {
        for (0..4) |m| {
            out[o] = ternary(b.qh[m], pow3[n]) * d;
            o += 1;
        }
    }
}

pub fn tq2_0(src: []const u8, out: *[256]f32_) void {
    const b = blk(blocks.BlockTQ2_0, src);
    const d = h(b.d);
    var o: usize = 0;
    var j: usize = 0;
    while (j < 64) : (j += 32) {
        for (0..4) |l| {
            for (0..32) |k| {
                const q: i32 = (b.qs[j + k] >> @intCast(2 * l)) & 3;
                out[o] = @as(f32_, @floatFromInt(q - 1)) * d;
                o += 1;
            }
        }
    }
}

pub inline fn e8m0Half(e: u8) f32_ {
    const bits: u32 = if (e < 2) (@as(u32, 0x00200000) << @intCast(e)) else (@as(u32, e) - 1) << 23;
    return @bitCast(bits);
}

pub fn mxfp4(src: []const u8, out: *[32]f32_) void {
    const b = blk(blocks.BlockMXFP4, src);
    const d = e8m0Half(b.e);
    for (0..16) |j| {
        out[j] = @as(f32_, @floatFromInt(tables.kvalues_fp4[b.qs[j] & 0x0F])) * d;
        out[j + 16] = @as(f32_, @floatFromInt(tables.kvalues_fp4[b.qs[j] >> 4])) * d;
    }
}

/// Unsigned E4M3 scale (bias 7) with the 0.5 factor of the doubled FP4 value convention.
inline fn ue4m3Half(x: u8) f32_ {
    if (x == 0 or x == 0x7F) return 0;
    const exp: i32 = (x >> 3) & 0xF;
    const man: f32_ = @floatFromInt(x & 7);
    const raw: f32_ = if (exp == 0)
        man * 0x1p-9
    else
        (1.0 + man / 8.0) * std.math.pow(f32_, 2.0, @as(f32_, @floatFromInt(exp - 7)));
    return raw * 0.5;
}

pub fn nvfp4(src: []const u8, out: *[64]f32_) void {
    const b = blk(blocks.BlockNVFP4, src);
    for (0..4) |s| {
        const d = ue4m3Half(b.d[s]);
        for (0..8) |j| {
            const byte = b.qs[s * 8 + j];
            out[s * 16 + j] = @as(f32_, @floatFromInt(tables.kvalues_fp4[byte & 0x0F])) * d;
            out[s * 16 + 8 + j] = @as(f32_, @floatFromInt(tables.kvalues_fp4[byte >> 4])) * d;
        }
    }
}

pub const DecodeError = error{ UnsupportedType, BadLength };

/// Decodes `dst.len` weights from `src`. `dst.len` must be a whole number of blocks and `src`
/// must hold exactly that many blocks. Plain float types are converted element wise.
pub fn dequantizeRow(t: StorageType, src: []const u8, dst: []f32_) DecodeError!void {
    const i = blocks.info(t) orelse return error.UnsupportedType;
    if (dst.len % i.elems != 0) return error.BadLength;
    if (src.len != (dst.len / i.elems) * i.bytes) return error.BadLength;

    switch (t) {
        .f32 => for (dst, 0..) |*o, k| {
            o.* = @bitCast(std.mem.readInt(u32, src[k * 4 ..][0..4], .little));
        },
        .f16 => for (dst, 0..) |*o, k| {
            o.* = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, src[k * 2 ..][0..2], .little))));
        },
        .bf16 => for (dst, 0..) |*o, k| {
            o.* = @bitCast(@as(u32, std.mem.readInt(u16, src[k * 2 ..][0..2], .little)) << 16);
        },
        .q4_0 => decodeBlocks(32, i.bytes, src, dst, q4_0),
        .q4_1 => decodeBlocks(32, i.bytes, src, dst, q4_1),
        .q5_0 => decodeBlocks(32, i.bytes, src, dst, q5_0),
        .q5_1 => decodeBlocks(32, i.bytes, src, dst, q5_1),
        .q8_0 => decodeBlocks(32, i.bytes, src, dst, q8_0),
        .iq4_nl => decodeBlocks(32, i.bytes, src, dst, iq4_nl),
        .mxfp4 => decodeBlocks(32, i.bytes, src, dst, mxfp4),
        .nvfp4 => decodeBlocks(64, i.bytes, src, dst, nvfp4),
        .q2_k => decodeBlocks(256, i.bytes, src, dst, q2_k),
        .q3_k => decodeBlocks(256, i.bytes, src, dst, q3_k),
        .q4_k => decodeBlocks(256, i.bytes, src, dst, q4_k),
        .q5_k => decodeBlocks(256, i.bytes, src, dst, q5_k),
        .q6_k => decodeBlocks(256, i.bytes, src, dst, q6_k),
        .q8_k => decodeBlocks(256, i.bytes, src, dst, q8_k),
        .iq2_xxs => decodeBlocks(256, i.bytes, src, dst, iq2_xxs),
        .iq2_xs => decodeBlocks(256, i.bytes, src, dst, iq2_xs),
        .iq2_s => decodeBlocks(256, i.bytes, src, dst, iq2_s),
        .iq3_xxs => decodeBlocks(256, i.bytes, src, dst, iq3_xxs),
        .iq3_s => decodeBlocks(256, i.bytes, src, dst, iq3_s),
        .iq1_s => decodeBlocks(256, i.bytes, src, dst, iq1_s),
        .iq1_m => decodeBlocks(256, i.bytes, src, dst, iq1_m),
        .iq4_xs => decodeBlocks(256, i.bytes, src, dst, iq4_xs),
        .tq1_0 => decodeBlocks(256, i.bytes, src, dst, tq1_0),
        .tq2_0 => decodeBlocks(256, i.bytes, src, dst, tq2_0),
        else => return error.UnsupportedType,
    }
}

fn decodeBlocks(
    comptime elems: usize,
    block_bytes: usize,
    src: []const u8,
    dst: []f32_,
    comptime f: fn ([]const u8, *[elems]f32_) void,
) void {
    const n = dst.len / elems;
    for (0..n) |k| {
        f(src[k * block_bytes ..][0..block_bytes], dst[k * elems ..][0..elems]);
    }
}
