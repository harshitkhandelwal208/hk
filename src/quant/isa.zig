//! Instruction set specific primitives for the integer kernels, chosen at compile time.
//!
//! Every kernel in `vecdot` and `gemm` is written against the handful of operations here. The
//! operations have one exact definition each, so every instruction set produces bit identical
//! integer results and only the speed differs:
//!
//!   x86 with VNNI      vpdpbusd / vpdpwssd, one instruction per 32 products
//!   x86 with AVX2      vpmaddubsw + vpmaddwd, exact because the unsigned operand is at most 127
//!   AArch64 dotprod    sdot, which is also exact because the "unsigned" operand is at most 127
//!   anything else      widening multiply and add, written with portable vectors
//!
//! The engine builds the kernels once per instruction set level and picks one at run time, so a
//! single binary is fast on every machine it meets.

const std = @import("std");
const builtin = @import("builtin");

pub const V8i = @Vector(8, i32);
pub const V8f = @Vector(8, f32);
pub const U32 = @Vector(32, u8);
pub const S32 = @Vector(32, i8);

const arch = builtin.cpu.arch;
const features = builtin.cpu.features;

/// Zig's unoptimized code generator does not enable the vector subtarget features for inline
/// assembly, so Debug builds always use the portable path. It is also the path the unit tests of
/// the generic variant exercise.
const asm_ok = builtin.mode != .Debug;

fn x86has(comptime f: std.Target.x86.Feature) bool {
    return arch == .x86_64 and std.Target.x86.featureSetHas(features, f);
}

fn armhas(comptime f: std.Target.aarch64.Feature) bool {
    return arch == .aarch64 and std.Target.aarch64.featureSetHas(features, f);
}

pub const has_vnni = asm_ok and ((x86has(.avx512vnni) and x86has(.avx512vl)) or x86has(.avxvnni));
pub const has_avx2 = asm_ok and x86has(.avx2);
pub const has_avx512 = asm_ok and x86has(.avx512f) and x86has(.avx512vl);
pub const has_dotprod = asm_ok and armhas(.dotprod);
pub const has_neon = arch == .aarch64 and asm_ok;

/// Short name of the instruction set level this build of the kernels targets.
pub const name: []const u8 = if (has_vnni and has_avx512) "avx512-vnni" else if (has_vnni) "avx2-vnni" else if (has_avx2) "avx2" else if (has_dotprod) "neon-dotprod" else if (has_neon) "neon" else "generic";

/// Rows and tokens of the register tile of the batch kernel. The accumulators of one tile have
/// to stay in registers: 32 vector registers allow 2 x 4, 16 allow 1 x 4.
pub const tile_row_groups: usize = if (has_avx512 or has_neon) 2 else 1;
pub const tile_tokens: usize = 4;

// ---------------------------------------------------------------------------------------
// Dot products
// ---------------------------------------------------------------------------------------

/// acc[i] += sum of the four products a[4i + k] * b[4i + k], a unsigned and at most 127,
/// b signed within [-127, 127]. Under those bounds every instruction set gives the same exact
/// result (the bounds keep the 16 bit intermediate of the AVX2 path from saturating).
pub inline fn dpu(acc: V8i, a: U32, b: S32) V8i {
    if (comptime has_vnni) {
        var r = acc;
        asm ("vpdpbusd %[b], %[a], %[r]"
            : [r] "+x" (r),
            : [a] "x" (a),
              [b] "x" (b),
        );
        return r;
    } else if (comptime has_avx2) {
        const ones: @Vector(16, i16) = @splat(1);
        const m: @Vector(16, i16) = asm ("vpmaddubsw %[b], %[a], %[r]"
            : [r] "=x" (-> @Vector(16, i16)),
            : [a] "x" (a),
              [b] "x" (b),
        );
        const s: V8i = asm ("vpmaddwd %[o], %[m], %[r]"
            : [r] "=x" (-> V8i),
            : [m] "x" (m),
              [o] "x" (ones),
        );
        return acc + s;
    } else if (comptime has_dotprod) {
        const lo = sdot16(@shuffle(i32, acc, undefined, @Vector(4, i32){ 0, 1, 2, 3 }), half(i8, @as(S32, @bitCast(a)), 0), half(i8, b, 0));
        const hi = sdot16(@shuffle(i32, acc, undefined, @Vector(4, i32){ 4, 5, 6, 7 }), half(i8, @as(S32, @bitCast(a)), 1), half(i8, b, 1));
        return @shuffle(i32, lo, hi, V8i{ 0, 1, 2, 3, -1, -2, -3, -4 });
    } else {
        return acc + dpEmulated(a, b);
    }
}

/// Like `dpu` with a signed first operand: any value in [-128, 127] against b in [-127, 127].
pub inline fn dps(acc: V8i, w: S32, b: S32) V8i {
    if (comptime has_dotprod) {
        return dpu(acc, @bitCast(w), b);
    } else if (comptime has_vnni or has_avx2) {
        // |w| is at most 128, and 128 * 127 * 2 still fits the AVX2 path's 16 bit intermediate.
        const neg = w < @as(S32, @splat(0));
        const aw: U32 = @abs(w);
        return dpu(acc, aw, @select(i8, neg, -%b, b));
    } else {
        return acc + dpEmulatedS(w, b);
    }
}

inline fn dpEmulated(a: U32, b: S32) V8i {
    const p: @Vector(32, i32) = @as(@Vector(32, i32), a) * @as(@Vector(32, i32), b);
    return sumQuads(p);
}

inline fn dpEmulatedS(a: S32, b: S32) V8i {
    const p: @Vector(32, i32) = @as(@Vector(32, i32), a) * @as(@Vector(32, i32), b);
    return sumQuads(p);
}

inline fn sumQuads(p: @Vector(32, i32)) V8i {
    return @shuffle(i32, p, undefined, lane_masks[0]) + @shuffle(i32, p, undefined, lane_masks[1]) +
        @shuffle(i32, p, undefined, lane_masks[2]) + @shuffle(i32, p, undefined, lane_masks[3]);
}

/// Shuffle masks picking every fourth product, evaluated once at file scope.
const lane_masks: [4]V8i = .{
    .{ 0, 4, 8, 12, 16, 20, 24, 28 },
    .{ 1, 5, 9, 13, 17, 21, 25, 29 },
    .{ 2, 6, 10, 14, 18, 22, 26, 30 },
    .{ 3, 7, 11, 15, 19, 23, 27, 31 },
};

const half_masks: [2]@Vector(16, i32) = blk: {
    var m: [2][16]i32 = undefined;
    for (0..2) |w| for (0..16) |i| {
        m[w][i] = @intCast(w * 16 + i);
    };
    break :blk .{ m[0], m[1] };
};

inline fn half(comptime T: type, v: @Vector(32, T), comptime which: usize) @Vector(16, T) {
    return @shuffle(T, v, undefined, half_masks[which]);
}

inline fn sdot16(acc: @Vector(4, i32), a: @Vector(16, i8), b: @Vector(16, i8)) @Vector(4, i32) {
    var r = acc;
    asm ("sdot %[r].4s, %[a].16b, %[b].16b"
        : [r] "+w" (r),
        : [a] "w" (a),
          [b] "w" (b),
    );
    return r;
}

/// acc[i] += sum of two adjacent i16 x i16 products, 8 lanes of 2 products each. Operands
/// are passed as 8 lanes of packed i16 pairs.
pub inline fn dpw(acc: V8i, a: V8i, b: V8i) V8i {
    if (comptime has_vnni) {
        var r = acc;
        asm ("vpdpwssd %[b], %[a], %[r]"
            : [r] "+x" (r),
            : [a] "x" (a),
              [b] "x" (b),
        );
        return r;
    } else if (comptime has_avx2) {
        const s: V8i = asm ("vpmaddwd %[b], %[a], %[r]"
            : [r] "=x" (-> V8i),
            : [a] "x" (a),
              [b] "x" (b),
        );
        return acc + s;
    } else {
        const a16: @Vector(16, i16) = @bitCast(a);
        const b16: @Vector(16, i16) = @bitCast(b);
        const p: @Vector(16, i32) = @as(@Vector(16, i32), a16) * @as(@Vector(16, i32), b16);
        const even = @shuffle(i32, p, undefined, V8i{ 0, 2, 4, 6, 8, 10, 12, 14 });
        const odd = @shuffle(i32, p, undefined, V8i{ 1, 3, 5, 7, 9, 11, 13, 15 });
        return acc + even + odd;
    }
}

// ---------------------------------------------------------------------------------------
// Table lookup
// ---------------------------------------------------------------------------------------

/// result[i] = table[idx[i]] for indices below 16 (larger values are not defined: x86 zeroes
/// the lane, ARM does too past 15), with the 16 entry table repeated in both 16 byte halves of
/// `tbl`. This is how a 4 bit codebook is applied to 32 indices at once.
pub inline fn lookup16(tbl: U32, idx: U32) U32 {
    if (comptime has_avx2) {
        return asm ("vpshufb %[i], %[t], %[r]"
            : [r] "=x" (-> U32),
            : [t] "x" (tbl),
              [i] "x" (idx),
        );
    } else if (comptime has_neon) {
        const t: @Vector(16, u8) = half(u8, tbl, 0);
        const lo: @Vector(16, u8) = asm ("tbl %[r].16b, {%[t].16b}, %[i].16b"
            : [r] "=w" (-> @Vector(16, u8)),
            : [t] "w" (t),
              [i] "w" (half(u8, idx, 0)),
        );
        const hi: @Vector(16, u8) = asm ("tbl %[r].16b, {%[t].16b}, %[i].16b"
            : [r] "=w" (-> @Vector(16, u8)),
            : [t] "w" (t),
              [i] "w" (half(u8, idx, 1)),
        );
        return std.simd.join(lo, hi);
    } else {
        const t: [32]u8 = tbl;
        const x: [32]u8 = idx;
        var r: [32]u8 = undefined;
        for (0..32) |i| r[i] = t[(i & 16) + (x[i] & 15)];
        return r;
    }
}

/// True when `lookup16` is a single instruction (or two) rather than a scalar loop.
pub const has_fast_lookup = has_avx2 or has_neon;

// ---------------------------------------------------------------------------------------
// Tests: the instruction set in use must agree with the portable definition.
// ---------------------------------------------------------------------------------------

test "dot primitives equal the portable definition" {
    var prng = std.Random.DefaultPrng.init(0x15A);
    const rand = prng.random();
    for (0..200) |_| {
        var a: [32]u8 = undefined;
        var w: [32]i8 = undefined;
        var b: [32]i8 = undefined;
        for (&a) |*x| x.* = rand.uintAtMost(u8, 127);
        for (&w) |*x| x.* = rand.int(i8);
        for (&b) |*x| x.* = rand.intRangeAtMost(i8, -127, 127);
        // Extremes stress the 16 bit intermediate of the AVX2 path.
        if (rand.boolean()) for (0..32) |i| {
            a[i] = 127;
            b[i] = if (rand.boolean()) 127 else -127;
            w[i] = if (rand.boolean()) -128 else 127;
        };
        var acc_arr: [8]i32 = undefined;
        for (&acc_arr) |*x| x.* = rand.intRangeAtMost(i32, -1000, 1000);
        const acc: V8i = acc_arr;
        try std.testing.expectEqual(acc + dpEmulated(a, b), dpu(acc, a, b));
        try std.testing.expectEqual(acc + dpEmulatedS(w, b), dps(acc, w, b));
    }
}

test "pair dot and table lookup equal the portable definition" {
    var prng = std.Random.DefaultPrng.init(0x7AB);
    const rand = prng.random();
    for (0..100) |_| {
        var a_arr: [8]i32 = undefined;
        var b_arr: [8]i32 = undefined;
        var acc_arr: [8]i32 = undefined;
        for (0..8) |i| {
            a_arr[i] = @bitCast(rand.int(u32));
            b_arr[i] = @bitCast(rand.int(u32));
            acc_arr[i] = rand.intRangeAtMost(i32, -1000, 1000);
        }
        const a: V8i = a_arr;
        const b: V8i = b_arr;
        const acc: V8i = acc_arr;
        const a16: [16]i16 = @bitCast(a);
        const b16: [16]i16 = @bitCast(b);
        var want: [8]i32 = acc_arr;
        for (0..8) |i| want[i] += @as(i32, a16[2 * i]) * b16[2 * i] + @as(i32, a16[2 * i + 1]) * b16[2 * i + 1];
        const want_v: V8i = want;
        try std.testing.expectEqual(want_v, dpw(acc, a, b));

        var t16: [16]u8 = undefined;
        for (&t16) |*x| x.* = rand.int(u8);
        const tbl: U32 = t16 ++ t16;
        var idx: [32]u8 = undefined;
        for (&idx) |*x| x.* = rand.uintAtMost(u8, 15);
        const got: [32]u8 = lookup16(tbl, idx);
        for (0..32) |i| try std.testing.expectEqual(t16[idx[i]], got[i]);
    }
}
