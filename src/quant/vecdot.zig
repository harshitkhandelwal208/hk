//! Dot products between one quantized weight row and one activation vector.
//!
//! Weights stay in their compressed form in memory. The activation vector is quantized once
//! to 8 bits (Q8 for the legacy formats, Q8_K for the K and IQ formats) and every row dots
//! against that, which turns the inner loop into small integer multiplies. This is the same
//! scheme GGML uses, so results are comparable to it. Formats without an integer kernel fall
//! back to decoding one block at a time into a stack buffer and a float dot, so every format
//! is supported and no path allocates.

const std = @import("std");
const format = @import("../format.zig");
const blocks = @import("blocks.zig");
const dequant = @import("dequant.zig");
const unpack = @import("unpack.zig");
const tables = @import("tables.zig");

const StorageType = format.StorageType;
const QK_K = blocks.QK_K;

/// 32 weight activation block. Keeps the plain sum of the int8 values so the formats that add
/// a per block offset (Q4_1, Q5_1) can apply it without another pass.
pub const BlockA8 = extern struct {
    d: f32,
    s: i32,
    qs: [32]i8,
};

pub const BlockQ8K = blocks.BlockQ8K;

inline fn h(x: f16) f32 {
    return @floatCast(x);
}

inline fn blk(comptime B: type, row: []const u8, i: usize) *align(1) const B {
    return @ptrCast(row[i * @sizeOf(B) ..].ptr);
}


// ---------------------------------------------------------------------------------------
// Integer dot primitive
// ---------------------------------------------------------------------------------------

const isa = @import("isa.zig");
pub const V8i = isa.V8i;
pub const V8f = isa.V8f;
pub const U32 = isa.U32;
pub const S32 = isa.S32;
pub const has_vnni = isa.has_vnni;
const dpu = isa.dpu;
const dps = isa.dps;
const dpw = isa.dpw;
/// Name of the instruction set level this build of the kernels uses.
pub const isa_name = isa.name;

inline fn fcvt(v: V8i) V8f {
    return @floatFromInt(v);
}

inline fn splatF(x: f32) V8f {
    return @splat(x);
}

inline fn splatI(x: i32) V8i {
    return @splat(x);
}

/// `x` in lane 0 and zero elsewhere. Row dots sum all lanes at the end, so a per block
/// correction that applies once must live in a single lane.
inline fn lane0(x: i32) V8i {
    return .{ x, 0, 0, 0, 0, 0, 0, 0 };
}

/// An f16 scale as a vector with the value in every lane. Converting a scalar f16 makes the
/// compiler merge the result into whatever vector register is free, which here is usually an
/// accumulator, and that false dependency turns the whole block loop into one long chain.
/// Broadcasting first and converting the vector has no such dependency.
inline fn hv(x: f16) V8f {
    const v: @Vector(8, f16) = @splat(x);
    return @floatCast(v);
}

inline fn load32(comptime T: type, p: anytype) @Vector(32, T) {
    return @as(*align(1) const [32]T, @ptrCast(p)).*;
}

/// Scale vector holding `lo` in lanes 0..3 and `hi` in lanes 4..7: the two 16 element halves
/// of a 32 element chunk, each covered by four lanes of `dp`.
inline fn pair(lo: i32, hi: i32) V8i {
    return .{ lo, lo, lo, lo, hi, hi, hi, hi };
}


const V16s = @Vector(16, i16);

/// The IQ4 codebook, repeated in both halves, for `isa.lookup16`.
const iq4nl_tbl: U32 = blk: {
    var t: [32]u8 = undefined;
    for (0..16) |i| {
        const v: u8 = @bitCast(tables.kvalues_iq4nl[i]);
        t[i] = v;
        t[i + 16] = v;
    }
    break :blk t;
};

/// Zero extends 16 bytes to two vectors of 8 i32 (sign extends for i8 input).
inline fn widen16(comptime T: type, v: @Vector(16, T)) [2]V8i {
    const w: @Vector(16, i32) = @intCast(v);
    return .{
        @shuffle(i32, w, undefined, V8i{ 0, 1, 2, 3, 4, 5, 6, 7 }),
        @shuffle(i32, w, undefined, V8i{ 8, 9, 10, 11, 12, 13, 14, 15 }),
    };
}

/// Lanes 0..3 hold element 2j of `w` and lanes 4..7 hold element 2j + 1: the scale pairing of
/// `dp` for the two 16 element halves of a 32 element chunk.
inline fn pairAt(w: V8i, comptime j: usize) V8i {
    return @shuffle(i32, w, undefined, V8i{ 2 * j, 2 * j, 2 * j, 2 * j, 2 * j + 1, 2 * j + 1, 2 * j + 1, 2 * j + 1 });
}

inline fn bsumsOf(a: *const BlockQ8K) V16s {
    return @as(*align(1) const [16]i16, @ptrCast(&a.bsums)).*;
}

inline fn dotsum(a: V16s, b: V16s) V8i {
    return dpw(@splat(0), @bitCast(a), @bitCast(b));
}

inline fn signApply(w: S32, y: S32) S32 {
    const neg = w < @as(S32, @splat(0));
    return @select(i8, neg, -%y, y);
}

pub fn quantizeA8(x: []const f32, out: []BlockA8) void {
    std.debug.assert(x.len == out.len * 32);
    for (out, 0..) |*o, b| {
        const v: @Vector(32, f32) = x[b * 32 ..][0..32].*;
        const amax = @reduce(.Max, @abs(v));
        const d = amax / 127.0;
        const id: f32 = if (d != 0) 1.0 / d else 0.0;
        const scaled = v * @as(@Vector(32, f32), @splat(id));
        const q: @Vector(32, i32) = @intFromFloat(@round(scaled));
        const q8: @Vector(32, i8) = @intCast(q);
        o.d = d;
        o.qs = q8;
        o.s = @reduce(.Add, q);
    }
}

pub fn quantizeQ8K(x: []const f32, out: []BlockQ8K) void {
    std.debug.assert(x.len == out.len * QK_K);
    for (out, 0..) |*o, b| {
        const src = x[b * QK_K ..][0..QK_K];
        var amax: f32 = 0;
        var max: f32 = 0;
        for (src) |v| {
            if (@abs(v) > amax) {
                amax = @abs(v);
                max = v;
            }
        }
        if (amax == 0) {
            o.d = 0;
            o.qs = @splat(0);
            o.bsums = @splat(0);
            continue;
        }
        const iscale = -127.0 / max;
        for (0..QK_K) |j| {
            const v: i32 = @intFromFloat(@round(iscale * src[j]));
            o.qs[j] = @intCast(@min(127, v));
        }
        for (0..16) |j| {
            var s: i32 = 0;
            for (0..16) |k| s += o.qs[j * 16 + k];
            o.bsums[j] = @intCast(s);
        }
        o.d = 1.0 / iscale;
    }
}

/// Which activation format a weight type pairs with.
pub const ActKind = enum { a8, q8k, f32 };

pub fn actKind(t: StorageType) ActKind {
    return switch (t) {
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .iq4_nl, .mxfp4 => .a8,
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .iq4_xs, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m, .tq1_0, .tq2_0 => .q8k,
        else => .f32,
    };
}

/// Per vector activation scratch. One instance is reused for every matrix that consumes the
/// same input, and only the formats actually needed are prepared.
pub const Act = struct {
    x: []const f32,
    a8: []BlockA8,
    q8k: []BlockQ8K,
    has_a8: bool = false,
    has_q8k: bool = false,

    /// `a8` and `q8k` must be sized for the longest input: len/32 and len/256 blocks.
    pub fn init(x: []const f32, a8_buf: []BlockA8, q8k_buf: []BlockQ8K) Act {
        return .{ .x = x, .a8 = a8_buf[0 .. x.len / 32], .q8k = q8k_buf[0 .. x.len / QK_K] };
    }

    pub fn prepare(self: *Act, kind: ActKind) void {
        switch (kind) {
            .a8 => if (!self.has_a8) {
                quantizeA8(self.x, self.a8);
                self.has_a8 = true;
            },
            .q8k => if (!self.has_q8k) {
                quantizeQ8K(self.x, self.q8k);
                self.has_q8k = true;
            },
            .f32 => {},
        }
    }
};


// ---------------------------------------------------------------------------------------
// Kernels. Each computes N activation vectors against one weight row in a single pass, so a
// weight block is loaded and unpacked once and feeds N independent accumulator chains.
// ---------------------------------------------------------------------------------------

fn A8s(comptime N: usize) type {
    return [N][]const BlockA8;
}
fn KS(comptime N: usize) type {
    return [N][]const BlockQ8K;
}

fn dotQ8_0(comptime N: usize, row: []const u8, act: A8s(N)) [N]f32 {
    var accf: [N]V8f = @splat(@splat(0));
    for (0..act[0].len) |i| {
        const w = blk(blocks.BlockQ8_0, row, i);
        const q = load32(i8, &w.qs);
        const dv = hv(w.d);
        inline for (0..N) |n| {
            const a = &act[n][i];
            const di = dps(@splat(0), q, load32(i8, &a.qs));
            accf[n] = @mulAdd(V8f, fcvt(di), dv * splatF(a.d), accf[n]);
        }
    }
    var out: [N]f32 = undefined;
    inline for (0..N) |n| out[n] = @reduce(.Add, accf[n]);
    return out;
}

/// The 32 unsigned 4 bit values of a Q4_0 / Q4_1 / IQ4 style block, low nibbles first.
inline fn nibbles(qs: *align(1) const [16]u8) U32 {
    const q: @Vector(16, u8) = qs.*;
    const lo = q & @as(@Vector(16, u8), @splat(0x0F));
    const hi = q >> @as(@Vector(16, u8), @splat(4));
    return std.simd.join(lo, hi);
}

/// Unsigned 5 bit values of a Q5_0 / Q5_1 block: nibbles with the fifth bit from `qh`.
inline fn nibbles5(qs: *align(1) const [16]u8, qh: u32) U32 {
    if (comptime isa.has_fast_lookup) {
        // Weight j takes bit j of qh. Broadcast qh, pick byte j / 8 into lane j, isolate bit j % 8.
        const bytes: U32 = @bitCast(@as(@Vector(8, u32), @splat(qh)));
        const pick: U32 = comptime blk: {
            var t: [32]u8 = undefined;
            for (0..32) |j| t[j] = @intCast((j / 8) % 4 + 0 * j);
            // Inside each 128 bit half the four bytes of qh sit at 0..3; the high half needs 2, 3.
            for (0..16) |j| t[j] = @intCast(j / 8);
            for (16..32) |j| t[j] = @intCast(2 + (j - 16) / 8);
            break :blk t;
        };
        const bitmask: U32 = comptime blk: {
            var t: [32]u8 = undefined;
            for (0..32) |j| t[j] = @as(u8, 1) << @intCast(j % 8);
            break :blk t;
        };
        const sel = isa.lookup16(bytes, pick) & bitmask;
        const set = sel == @as(U32, @splat(0));
        const h5: U32 = @select(u8, set, @as(U32, @splat(0)), @as(U32, @splat(0x10)));
        return nibbles(qs) | h5;
    }
    const idx: @Vector(32, u5) = comptime blk: {
        var r: [32]u5 = undefined;
        for (0..32) |j| r[j] = @intCast(j);
        break :blk r;
    };
    const bits: @Vector(32, u32) = (@as(@Vector(32, u32), @splat(qh)) >> idx) & @as(@Vector(32, u32), @splat(1));
    const h5: U32 = @intCast(bits << @as(@Vector(32, u5), @splat(4)));
    return nibbles(qs) | h5;
}

fn dotQ4_0(comptime N: usize, row: []const u8, act: A8s(N)) [N]f32 {
    var accf: [N]V8f = @splat(@splat(0));
    for (0..act[0].len) |i| {
        const w = blk(blocks.BlockQ4_0, row, i);
        const q = nibbles(&w.qs);
        const dv = hv(w.d);
        inline for (0..N) |n| {
            const a = &act[n][i];
            // Stored values are q - 8; the dot used q, so remove 8 * sum(y).
            const di = dpu(@splat(0), q, load32(i8, &a.qs)) - lane0(8 * a.s);
            accf[n] = @mulAdd(V8f, fcvt(di), dv * splatF(a.d), accf[n]);
        }
    }
    var out: [N]f32 = undefined;
    inline for (0..N) |n| out[n] = @reduce(.Add, accf[n]);
    return out;
}

fn dotQ4_1(comptime N: usize, row: []const u8, act: A8s(N)) [N]f32 {
    var accf: [N]V8f = @splat(@splat(0));
    var extra: [N]V8f = @splat(@splat(0));
    for (0..act[0].len) |i| {
        const w = blk(blocks.BlockQ4_1, row, i);
        const q = nibbles(&w.qs);
        const dv = hv(w.d);
        const mv = hv(w.m);
        inline for (0..N) |n| {
            const a = &act[n][i];
            const di = dpu(@splat(0), q, load32(i8, &a.qs));
            accf[n] = @mulAdd(V8f, fcvt(di), dv * splatF(a.d), accf[n]);
            extra[n] = @mulAdd(V8f, mv, splatF(a.d) * fcvt(splatI(a.s)), extra[n]);
        }
    }
    var out: [N]f32 = undefined;
    // The offset term is added to every lane, so it is counted eight times; divide it back out.
    inline for (0..N) |n| out[n] = @reduce(.Add, accf[n]) + @reduce(.Add, extra[n]) * (1.0 / 8.0);
    return out;
}

fn dotQ5_0(comptime N: usize, row: []const u8, act: A8s(N)) [N]f32 {
    var accf: [N]V8f = @splat(@splat(0));
    for (0..act[0].len) |i| {
        const w = blk(blocks.BlockQ5_0, row, i);
        const q = nibbles5(&w.qs, std.mem.readInt(u32, &w.qh, .little));
        const dv = hv(w.d);
        inline for (0..N) |n| {
            const a = &act[n][i];
            const di = dpu(@splat(0), q, load32(i8, &a.qs)) - lane0(16 * a.s);
            accf[n] = @mulAdd(V8f, fcvt(di), dv * splatF(a.d), accf[n]);
        }
    }
    var out: [N]f32 = undefined;
    inline for (0..N) |n| out[n] = @reduce(.Add, accf[n]);
    return out;
}

fn dotQ5_1(comptime N: usize, row: []const u8, act: A8s(N)) [N]f32 {
    var accf: [N]V8f = @splat(@splat(0));
    var extra: [N]V8f = @splat(@splat(0));
    for (0..act[0].len) |i| {
        const w = blk(blocks.BlockQ5_1, row, i);
        const q = nibbles5(&w.qs, std.mem.readInt(u32, &w.qh, .little));
        const dv = hv(w.d);
        const mv = hv(w.m);
        inline for (0..N) |n| {
            const a = &act[n][i];
            const di = dpu(@splat(0), q, load32(i8, &a.qs));
            accf[n] = @mulAdd(V8f, fcvt(di), dv * splatF(a.d), accf[n]);
            extra[n] = @mulAdd(V8f, mv, splatF(a.d) * fcvt(splatI(a.s)), extra[n]);
        }
    }
    var out: [N]f32 = undefined;
    inline for (0..N) |n| out[n] = @reduce(.Add, accf[n]) + @reduce(.Add, extra[n]) * (1.0 / 8.0);
    return out;
}

fn dotIQ4NL(comptime N: usize, row: []const u8, act: A8s(N)) [N]f32 {
    var accf: [N]V8f = @splat(@splat(0));
    for (0..act[0].len) |i| {
        const w = blk(blocks.BlockIQ4NL, row, i);
        const q: S32 = @bitCast(isa.lookup16(iq4nl_tbl, nibbles(&w.qs)));
        const dv = hv(w.d);
        inline for (0..N) |n| {
            const a = &act[n][i];
            const di = dps(@splat(0), q, load32(i8, &a.qs));
            accf[n] = @mulAdd(V8f, fcvt(di), dv * splatF(a.d), accf[n]);
        }
    }
    var out: [N]f32 = undefined;
    inline for (0..N) |n| out[n] = @reduce(.Add, accf[n]);
    return out;
}

// ---------------------------------------------------------------------------------------
// K formats against Q8_K
// ---------------------------------------------------------------------------------------

fn dotQ2K(comptime N: usize, row: []const u8, act: KS(N)) [N]f32 {
    var accf: [N]V8f = @splat(@splat(0));
    var corr: [N]V8f = @splat(@splat(0));
    for (0..act[0].len) |i| {
        const w = blk(blocks.BlockQ2K, row, i);
        const sc16: @Vector(16, u8) = w.scales;
        const lows = widen16(u8, sc16 & @as(@Vector(16, u8), @splat(0x0F)));
        const mins: V16s = @intCast(sc16 >> @as(@Vector(16, u8), @splat(4)));
        const d = hv(w.d);
        const dm = hv(w.dmin);
        var acci: [N]V8i = @splat(@splat(0));
        inline for (0..2) |half| {
            const q = load32(u8, w.qs[half * 32 ..].ptr);
            inline for (0..4) |j| {
                const qv: U32 = (q >> @as(U32, @splat(2 * j))) & @as(U32, @splat(3));
                const sv = pairAt(lows[half], j);
                inline for (0..N) |n| {
                    acci[n] += dpu(@splat(0), qv, load32(i8, act[n][i].qs[half * 128 + j * 32 ..].ptr)) * sv;
                }
            }
        }
        inline for (0..N) |n| {
            const a = &act[n][i];
            accf[n] = @mulAdd(V8f, fcvt(acci[n]), d * splatF(a.d), accf[n]);
            corr[n] = @mulAdd(V8f, fcvt(dotsum(bsumsOf(a), mins)), dm * splatF(a.d), corr[n]);
        }
    }
    var out: [N]f32 = undefined;
    inline for (0..N) |n| out[n] = @reduce(.Add, accf[n]) - @reduce(.Add, corr[n]);
    return out;
}

fn dotQ3K(comptime N: usize, row: []const u8, act: KS(N)) [N]f32 {
    var accf: [N]V8f = @splat(@splat(0));
    for (0..act[0].len) |i| {
        const w = blk(blocks.BlockQ3K, row, i);
        var scales: [16]i8 = undefined;
        dequant.q3kScales(&w.scales, &scales);
        const s8: @Vector(16, i8) = scales;
        const s32 = widen16(i8, s8 - @as(@Vector(16, i8), @splat(32)));
        const s16: V16s = @as(V16s, @intCast(s8)) - @as(V16s, @splat(32));
        var acci: [N]V8i = @splat(@splat(0));
        const hm = load32(u8, &w.hmask);
        inline for (0..2) |half| {
            const q = load32(u8, w.qs[half * 32 ..].ptr);
            inline for (0..4) |j| {
                const k = half * 4 + j;
                const lo: U32 = (q >> @as(U32, @splat(2 * j))) & @as(U32, @splat(3));
                const hb: U32 = ((hm >> @as(U32, @splat(k))) & @as(U32, @splat(1))) << @as(U32, @splat(2));
                const qv = lo + hb;
                const sv = pairAt(s32[half], j);
                inline for (0..N) |n| {
                    acci[n] += dpu(@splat(0), qv, load32(i8, act[n][i].qs[half * 128 + j * 32 ..].ptr)) * sv;
                }
            }
        }
        const d = hv(w.d);
        // The stored value is (lo + 4 * bit) - 4, the dot used lo + 4 * bit: remove 4 * sc * sum(y).
        inline for (0..N) |n| {
            const a = &act[n][i];
            const bias = dotsum(bsumsOf(a), s16) * splatI(4);
            accf[n] = @mulAdd(V8f, fcvt(acci[n] - bias), d * splatF(a.d), accf[n]);
        }
    }
    var out: [N]f32 = undefined;
    inline for (0..N) |n| out[n] = @reduce(.Add, accf[n]);
    return out;
}

/// Unpacks the eight 6 bit scales and eight 6 bit mins of a Q4_K / Q5_K block with a few word
/// operations (the same trick GGML uses). Returns scales in `sc` and mins in `mn`.
inline fn scalesMinsK4(packed_s: *const [12]u8, sc: *@Vector(8, u8), mn: *@Vector(8, u8)) void {
    const k1: u32 = 0x3f3f3f3f;
    const k2: u32 = 0x0f0f0f0f;
    const k3: u32 = 0x03030303;
    var u: [4]u32 = undefined;
    inline for (0..3) |i| u[i] = std.mem.readInt(u32, packed_s[i * 4 ..][0..4], .little);
    u[3] = ((u[2] >> 4) & k2) | (((u[1] >> 6) & k3) << 4);
    const aux = u[1] & k1;
    u[1] = (u[2] & k2) | (((u[0] >> 6) & k3) << 4);
    u[2] = aux;
    u[0] &= k1;
    sc.* = @bitCast([2]u32{ u[0], u[1] });
    mn.* = @bitCast([2]u32{ u[2], u[3] });
}

/// Each of eight values repeated twice, as 16 lanes (a min per 32 weights becomes one per 16).
inline fn dup16(v: @Vector(8, u8)) V16s {
    const w: @Vector(8, i16) = @intCast(v);
    return @shuffle(i16, w, undefined, @Vector(16, i32){ 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7 });
}

inline fn k4Block(comptime N: usize, comptime has_high: bool, w: anytype, act: KS(N), i: usize, accf: *[N]V8f, corr: *[N]V8f) void {
    var sc8: @Vector(8, u8) = undefined;
    var mn8: @Vector(8, u8) = undefined;
    scalesMinsK4(&w.scales, &sc8, &mn8);
    const sc: V8i = @intCast(sc8);
    const mins = dup16(mn8);
    var acci: [N]V8i = @splat(@splat(0));
    const qh = if (has_high) load32(u8, &w.qh) else undefined;
    inline for (0..4) |p| {
        const q = load32(u8, w.qs[p * 32 ..].ptr);
        var lo = q & @as(U32, @splat(0x0F));
        var hi = q >> @as(U32, @splat(4));
        if (has_high) {
            lo |= ((qh >> @as(U32, @splat(2 * p))) & @as(U32, @splat(1))) << @as(U32, @splat(4));
            hi |= ((qh >> @as(U32, @splat(2 * p + 1))) & @as(U32, @splat(1))) << @as(U32, @splat(4));
        }
        const s0 = @shuffle(i32, sc, undefined, @as(V8i, @splat(2 * p)));
        const s1 = @shuffle(i32, sc, undefined, @as(V8i, @splat(2 * p + 1)));
        inline for (0..N) |n| {
            const a = &act[n][i];
            acci[n] += dpu(@splat(0), lo, load32(i8, a.qs[p * 64 ..].ptr)) * s0 +
                dpu(@splat(0), hi, load32(i8, a.qs[p * 64 + 32 ..].ptr)) * s1;
        }
    }
    const d = hv(w.d);
    const dm = hv(w.dmin);
    inline for (0..N) |n| {
        const a = &act[n][i];
        accf[n] = @mulAdd(V8f, fcvt(acci[n]), d * splatF(a.d), accf[n]);
        corr[n] = @mulAdd(V8f, fcvt(dotsum(bsumsOf(a), mins)), dm * splatF(a.d), corr[n]);
    }
}

fn dotQ4K(comptime N: usize, row: []const u8, act: KS(N)) [N]f32 {
    var accf: [N]V8f = @splat(@splat(0));
    var corr: [N]V8f = @splat(@splat(0));
    for (0..act[0].len) |i| k4Block(N, false, blk(blocks.BlockQ4K, row, i), act, i, &accf, &corr);
    var out: [N]f32 = undefined;
    inline for (0..N) |n| out[n] = @reduce(.Add, accf[n]) - @reduce(.Add, corr[n]);
    return out;
}

fn dotQ5K(comptime N: usize, row: []const u8, act: KS(N)) [N]f32 {
    var accf: [N]V8f = @splat(@splat(0));
    var corr: [N]V8f = @splat(@splat(0));
    for (0..act[0].len) |i| k4Block(N, true, blk(blocks.BlockQ5K, row, i), act, i, &accf, &corr);
    var out: [N]f32 = undefined;
    inline for (0..N) |n| out[n] = @reduce(.Add, accf[n]) - @reduce(.Add, corr[n]);
    return out;
}

fn dotQ6K(comptime N: usize, row: []const u8, act: KS(N)) [N]f32 {
    var accf: [N]V8f = @splat(@splat(0));
    for (0..act[0].len) |i| {
        const w = blk(blocks.BlockQ6K, row, i);
        const s8: @Vector(16, i8) = w.scales;
        const s32 = widen16(i8, s8);
        const s16: V16s = @intCast(s8);
        var acci: [N]V8i = @splat(@splat(0));
        inline for (0..2) |half| {
            const ql0 = load32(u8, w.ql[half * 64 ..].ptr);
            const ql1 = load32(u8, w.ql[half * 64 + 32 ..].ptr);
            const qh = load32(u8, w.qh[half * 32 ..].ptr);
            const m4 = @as(U32, @splat(0x0F));
            const m3 = @as(U32, @splat(3));
            const q1 = (ql0 & m4) | ((qh & m3) << @as(U32, @splat(4)));
            const q2 = (ql1 & m4) | (((qh >> @as(U32, @splat(2))) & m3) << @as(U32, @splat(4)));
            const q3 = (ql0 >> @as(U32, @splat(4))) | (((qh >> @as(U32, @splat(4))) & m3) << @as(U32, @splat(4)));
            const q4 = (ql1 >> @as(U32, @splat(4))) | (((qh >> @as(U32, @splat(6))) & m3) << @as(U32, @splat(4)));
            const qv = [4]U32{ q1, q2, q3, q4 };
            inline for (0..4) |g| {
                const sv = pairAt(s32[half], g);
                inline for (0..N) |n| {
                    acci[n] += dpu(@splat(0), qv[g], load32(i8, act[n][i].qs[half * 128 + g * 32 ..].ptr)) * sv;
                }
            }
        }
        const d = hv(w.d);
        inline for (0..N) |n| {
            const a = &act[n][i];
            // Stored values are q - 32; the dot used q: remove 32 * scale * sum(y).
            const bias = dotsum(bsumsOf(a), s16) * splatI(32);
            accf[n] = @mulAdd(V8f, fcvt(acci[n] - bias), d * splatF(a.d), accf[n]);
        }
    }
    var out: [N]f32 = undefined;
    inline for (0..N) |n| out[n] = @reduce(.Add, accf[n]);
    return out;
}

fn dotIQ4XS(comptime N: usize, row: []const u8, act: KS(N)) [N]f32 {
    var accf: [N]V8f = @splat(@splat(0));
    for (0..act[0].len) |i| {
        const w = blk(blocks.BlockIQ4XS, row, i);
        const scales_h = std.mem.readInt(u16, &w.scales_h, .little);
        var ls: [8]i32 = undefined;
        inline for (0..8) |ib| {
            const lo: u32 = (w.scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0x0F;
            const hi: u32 = (scales_h >> @intCast(2 * ib)) & 3;
            ls[ib] = @as(i32, @intCast(lo | (hi << 4))) - 32;
        }
        const lsv: V8i = ls;
        var acci: [N]V8i = @splat(@splat(0));
        inline for (0..8) |ib| {
            const q: S32 = @bitCast(isa.lookup16(iq4nl_tbl, nibbles(@ptrCast(w.qs[16 * ib ..].ptr))));
            const sv = @shuffle(i32, lsv, undefined, @as(V8i, @splat(ib)));
            inline for (0..N) |n| {
                acci[n] += dps(@splat(0), q, load32(i8, act[n][i].qs[32 * ib ..].ptr)) * sv;
            }
        }
        const d = hv(w.d);
        inline for (0..N) |n| {
            accf[n] = @mulAdd(V8f, fcvt(acci[n]), d * splatF(act[n][i].d), accf[n]);
        }
    }
    var out: [N]f32 = undefined;
    inline for (0..N) |n| out[n] = @reduce(.Add, accf[n]);
    return out;
}

// ---------------------------------------------------------------------------------------
// Exact fallback: decode a block, dot with float activations
// ---------------------------------------------------------------------------------------

inline fn dotF32(a: []const f32, b: []const f32) f32 {
    const L = 16;
    var acc: @Vector(L, f32) = @splat(0);
    var i: usize = 0;
    while (i + L <= a.len) : (i += L) {
        const va: @Vector(L, f32) = a[i..][0..L].*;
        const vb: @Vector(L, f32) = b[i..][0..L].*;
        acc = @mulAdd(@Vector(L, f32), va, vb, acc);
    }
    var s = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) s += a[i] * b[i];
    return s;
}

fn dotGeneric(comptime t: StorageType, comptime N: usize, row: []const u8, xs: [N][]const f32) [N]f32 {
    const info = comptime blocks.info(t).?;
    const elems = info.elems;
    var buf: [256]f32 = undefined;
    var sum: [N]f32 = @splat(0);
    const n_blocks = xs[0].len / elems;
    for (0..n_blocks) |i| {
        dequant.dequantizeRow(t, row[i * info.bytes ..][0..info.bytes], buf[0..elems]) catch unreachable;
        inline for (0..N) |n| sum[n] += dotF32(buf[0..elems], xs[n][i * elems ..][0..elems]);
    }
    return sum;
}

// Plain float rows. Loads go through `align(1)` pointers because rows can start anywhere.
fn dotF32Row(row: []const u8, x: []const f32) f32 {
    const L = 16;
    var acc: @Vector(L, f32) = @splat(0);
    var i: usize = 0;
    while (i + L <= x.len) : (i += L) {
        const w: @Vector(L, f32) = @as(*align(1) const [L]f32, @ptrCast(row[i * 4 ..].ptr)).*;
        const v: @Vector(L, f32) = x[i..][0..L].*;
        acc = @mulAdd(@Vector(L, f32), w, v, acc);
    }
    var s = @reduce(.Add, acc);
    while (i < x.len) : (i += 1) s += @as(f32, @bitCast(std.mem.readInt(u32, row[i * 4 ..][0..4], .little))) * x[i];
    return s;
}

fn dotF16Row(row: []const u8, x: []const f32) f32 {
    const L = 16;
    var acc: @Vector(L, f32) = @splat(0);
    var i: usize = 0;
    while (i + L <= x.len) : (i += L) {
        const raw: @Vector(L, u16) = @as(*align(1) const [L]u16, @ptrCast(row[i * 2 ..].ptr)).*;
        const w: @Vector(L, f32) = @floatCast(@as(@Vector(L, f16), @bitCast(raw)));
        const v: @Vector(L, f32) = x[i..][0..L].*;
        acc = @mulAdd(@Vector(L, f32), w, v, acc);
    }
    var s = @reduce(.Add, acc);
    while (i < x.len) : (i += 1) {
        const bits = std.mem.readInt(u16, row[i * 2 ..][0..2], .little);
        s += @as(f32, @floatCast(@as(f16, @bitCast(bits)))) * x[i];
    }
    return s;
}

fn dotBF16Row(row: []const u8, x: []const f32) f32 {
    const L = 16;
    var acc: @Vector(L, f32) = @splat(0);
    var i: usize = 0;
    while (i + L <= x.len) : (i += L) {
        const raw: @Vector(L, u16) = @as(*align(1) const [L]u16, @ptrCast(row[i * 2 ..].ptr)).*;
        const w: @Vector(L, f32) = @bitCast(@as(@Vector(L, u32), raw) << @as(@Vector(L, u5), @splat(16)));
        const v: @Vector(L, f32) = x[i..][0..L].*;
        acc = @mulAdd(@Vector(L, f32), w, v, acc);
    }
    var s = @reduce(.Add, acc);
    while (i < x.len) : (i += 1) {
        const bits: u32 = std.mem.readInt(u16, row[i * 2 ..][0..2], .little);
        s += @as(f32, @bitCast(bits << 16)) * x[i];
    }
    return s;
}

/// Dots one weight row against N prepared activations at once. `row` is exactly one row of
/// `acts[0].x.len` weights and every activation must have been prepared with
/// `prepare(actKind(t))`.
/// Formats without a hand written kernel: the block is unpacked into signed bytes and per 16
/// weight integer scales (see unpack.zig) and then dotted like the K formats.
fn dotCanonK(comptime t: StorageType, comptime N: usize, row: []const u8, act: KS(N)) [N]f32 {
    var accf: [N]V8f = @splat(@splat(0));
    for (0..act[0].len) |i| {
        const kb = unpack.unpackK(t, row, i);
        var acci: [N]V8i = @splat(@splat(0));
        inline for (0..8) |c| {
            const w: S32 = @bitCast(load32(u8, kb.vals[32 * c ..].ptr));
            const sv = pair(kb.sc[2 * c], kb.sc[2 * c + 1]);
            inline for (0..N) |n| {
                acci[n] += dps(@splat(0), w, load32(i8, act[n][i].qs[32 * c ..].ptr)) * sv;
            }
        }
        const dv: V8f = @splat(kb.d);
        inline for (0..N) |n| accf[n] = @mulAdd(V8f, fcvt(acci[n]), dv * splatF(act[n][i].d), accf[n]);
    }
    var out: [N]f32 = undefined;
    inline for (0..N) |n| out[n] = @reduce(.Add, accf[n]);
    return out;
}

fn dotCanonLegacy(comptime t: StorageType, comptime N: usize, row: []const u8, act: A8s(N)) [N]f32 {
    var accf: [N]V8f = @splat(@splat(0));
    for (0..act[0].len) |i| {
        const lb = unpack.unpackLegacy(t, row, i);
        const w: S32 = @bitCast(load32(u8, &lb.vals));
        const dv: V8f = @splat(lb.d);
        inline for (0..N) |n| {
            const a = &act[n][i];
            accf[n] = @mulAdd(V8f, fcvt(dps(@splat(0), w, load32(i8, &a.qs))), dv * splatF(a.d), accf[n]);
        }
    }
    var out: [N]f32 = undefined;
    inline for (0..N) |n| out[n] = @reduce(.Add, accf[n]);
    return out;
}

pub fn dotRowN(comptime N: usize, t: StorageType, row: []const u8, acts: [N]*const Act) [N]f32 {
    var a8: [N][]const BlockA8 = undefined;
    var qk: [N][]const BlockQ8K = undefined;
    var xs: [N][]const f32 = undefined;
    inline for (0..N) |n| {
        a8[n] = acts[n].a8;
        qk[n] = acts[n].q8k;
        xs[n] = acts[n].x;
    }
    return switch (t) {
        .q8_0 => dotQ8_0(N, row, a8),
        .q4_0 => dotQ4_0(N, row, a8),
        .q4_1 => dotQ4_1(N, row, a8),
        .q5_0 => dotQ5_0(N, row, a8),
        .q5_1 => dotQ5_1(N, row, a8),
        .iq4_nl => dotIQ4NL(N, row, a8),
        .q2_k => dotQ2K(N, row, qk),
        .q3_k => dotQ3K(N, row, qk),
        .q4_k => dotQ4K(N, row, qk),
        .q5_k => dotQ5K(N, row, qk),
        .q6_k => dotQ6K(N, row, qk),
        .iq4_xs => dotIQ4XS(N, row, qk),
        .f32 => blk: {
            var o: [N]f32 = undefined;
            inline for (0..N) |n| o[n] = dotF32Row(row, xs[n]);
            break :blk o;
        },
        .f16 => blk: {
            var o: [N]f32 = undefined;
            inline for (0..N) |n| o[n] = dotF16Row(row, xs[n]);
            break :blk o;
        },
        .bf16 => blk: {
            var o: [N]f32 = undefined;
            inline for (0..N) |n| o[n] = dotBF16Row(row, xs[n]);
            break :blk o;
        },
        .mxfp4 => dotCanonLegacy(.mxfp4, N, row, a8),
        inline .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m, .tq1_0, .tq2_0 => |tt| dotCanonK(tt, N, row, qk),
        .nvfp4 => dotGeneric(.nvfp4, N, row, xs),
        else => unreachable, // callers validate the type up front with `supported`
    };
}

pub fn dotRow(t: StorageType, row: []const u8, act: *const Act) f32 {
    return dotRowN(1, t, row, .{act})[0];
}

/// True when `dotRow` can handle `t`. Loaders call this so an unsupported tensor type is a
/// load error rather than silently producing zeros.
pub fn supported(t: StorageType) bool {
    return switch (t) {
        .f32, .f16, .bf16, .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .iq4_nl, .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .iq4_xs, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m, .tq1_0, .tq2_0, .mxfp4, .nvfp4 => true,
        else => false,
    };
}
