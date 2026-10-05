//! Elementwise and normalization kernels. None of them allocate.

const std = @import("std");
const config_mod = @import("config.zig");

pub const V = 16;
pub const Vf = @Vector(V, f32);

/// out = x * w / rms(x), with rms computed in f32.
pub fn rmsNorm(out: []f32, x: []const f32, w: []const f32, eps: f32) void {
    std.debug.assert(out.len == x.len and w.len == x.len);
    var acc: Vf = @splat(0);
    var i: usize = 0;
    while (i + V <= x.len) : (i += V) {
        const v: Vf = x[i..][0..V].*;
        acc = @mulAdd(Vf, v, v, acc);
    }
    var ss = @reduce(.Add, acc);
    while (i < x.len) : (i += 1) ss += x[i] * x[i];
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    const vinv: Vf = @splat(inv);
    i = 0;
    while (i + V <= x.len) : (i += V) {
        const v: Vf = x[i..][0..V].*;
        const wv: Vf = w[i..][0..V].*;
        out[i..][0..V].* = v * vinv * wv;
    }
    while (i < x.len) : (i += 1) out[i] = x[i] * inv * w[i];
}

/// In place per head RMS norm, used by Qwen3 on queries and keys before RoPE.
pub fn headNorm(x: []f32, w: []const f32, head_dim: usize, eps: f32) void {
    var h: usize = 0;
    while (h < x.len) : (h += head_dim) {
        const s = x[h..][0..head_dim];
        rmsNorm(s, s, w, eps);
    }
}

/// Vectorized exp. Zig lowers `@exp` on a vector to one libm call per lane, which dominates
/// SwiGLU and softmax in prompt processing. This is range reduction to [-ln2/2, ln2/2], a
/// degree 6 polynomial, and a scale by 2^n built from the exponent bits. Relative error is
/// about 1.2e-7, at the limit of f32, and inputs are clamped so the result never overflows
/// to infinity or denormals.
pub inline fn vexp(x: Vf) Vf {
    const lo: Vf = @splat(-87.0);
    const hi: Vf = @splat(88.0);
    const xc = @min(@max(x, lo), hi);
    const n = @round(xc * @as(Vf, @splat(1.44269504089)));
    // ln2 split into a high part that is exact in f32 and a small correction.
    var f = xc - n * @as(Vf, @splat(0.693359375));
    f = f - n * @as(Vf, @splat(-2.12194440e-4));
    var p: Vf = @splat(1.0 / 720.0);
    p = @mulAdd(Vf, p, f, @as(Vf, @splat(1.0 / 120.0)));
    p = @mulAdd(Vf, p, f, @as(Vf, @splat(1.0 / 24.0)));
    p = @mulAdd(Vf, p, f, @as(Vf, @splat(1.0 / 6.0)));
    p = @mulAdd(Vf, p, f, @as(Vf, @splat(0.5)));
    p = @mulAdd(Vf, p, f, @as(Vf, @splat(1.0)));
    p = @mulAdd(Vf, p, f, @as(Vf, @splat(1.0)));
    const ni: @Vector(V, i32) = @intFromFloat(n);
    const bits: @Vector(V, i32) = (ni + @as(@Vector(V, i32), @splat(127))) << @as(@Vector(V, u5), @splat(23));
    return p * @as(Vf, @bitCast(bits));
}

pub inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// gate = silu(gate) * up
pub fn swiglu(gate: []f32, up: []const f32) void {
    std.debug.assert(gate.len == up.len);
    var i: usize = 0;
    while (i + V <= gate.len) : (i += V) {
        const g: Vf = gate[i..][0..V].*;
        const u: Vf = up[i..][0..V].*;
        const one: Vf = @splat(1.0);
        gate[i..][0..V].* = (g / (one + vexp(-g))) * u;
    }
    while (i < gate.len) : (i += 1) gate[i] = silu(gate[i]) * up[i];
}

pub fn addInPlace(x: []f32, y: []const f32) void {
    std.debug.assert(x.len == y.len);
    var i: usize = 0;
    while (i + V <= x.len) : (i += V) {
        const a: Vf = x[i..][0..V].*;
        const b: Vf = y[i..][0..V].*;
        x[i..][0..V].* = a + b;
    }
    while (i < x.len) : (i += 1) x[i] += y[i];
}

/// Inverse frequencies for rotary embeddings, base^(-2i/n_rot), stretched according to `r`.
/// `factors` is an optional per-frequency divisor, which is how llama.cpp files carry llama3
/// scaling (the `rope_freqs.weight` tensor).
pub fn ropeInvFreq(out: []f32, base: f32, n_rot: usize, r: config_mod.RopeScaling, factors: ?[]const f32) void {
    std.debug.assert(out.len == n_rot / 2);
    if (factors) |f| std.debug.assert(f.len == out.len);
    const nr: f64 = @floatFromInt(n_rot);
    const b: f64 = base;
    const factor: f64 = r.factor;

    // YaRN: the band of dimensions that is blended between stretched and original frequencies.
    var low: f64 = 0;
    var high: f64 = 0;
    if (r.kind == .yarn) {
        const ctx: f64 = @floatFromInt(r.orig_ctx);
        const corr = struct {
            fn dim(n: f64, ctx_: f64, rotations: f64, base_: f64) f64 {
                return n * @log(ctx_ / (rotations * 2.0 * std.math.pi)) / (2.0 * @log(base_));
            }
        }.dim;
        low = @max(0.0, @floor(corr(nr, ctx, r.beta_fast, b)));
        high = @min(nr - 1.0, @ceil(corr(nr, ctx, r.beta_slow, b)));
        if (low == high) high += 0.001;
    }

    for (out, 0..) |*o, i| {
        const exponent = @as(f64, @floatFromInt(2 * i)) / nr;
        const inv = std.math.pow(f64, b, -exponent);
        var v: f64 = switch (r.kind) {
            .none => inv,
            .linear => inv / factor,
            .llama3 => blk: {
                const old_ctx: f64 = @floatFromInt(r.orig_ctx);
                const wavelen = 2.0 * std.math.pi / inv;
                const low_wavelen = old_ctx / r.low_freq_factor;
                const high_wavelen = old_ctx / r.high_freq_factor;
                if (wavelen < high_wavelen) break :blk inv;
                if (wavelen > low_wavelen) break :blk inv / factor;
                const smooth = (old_ctx / wavelen - r.low_freq_factor) / (r.high_freq_factor - r.low_freq_factor);
                break :blk (1.0 - smooth) * inv / factor + smooth * inv;
            },
            .yarn => blk: {
                const ramp = std.math.clamp((@as(f64, @floatFromInt(i)) - low) / (high - low), 0.0, 1.0);
                // High frequency dimensions (ramp 0) keep the original frequency, low ones are stretched.
                const keep_original = 1.0 - ramp;
                break :blk (inv / factor) * (1.0 - keep_original) + inv * keep_original;
            },
        };
        if (factors) |f| v /= f[i];
        o.* = @floatCast(v);
    }
}

/// Multiplier applied to every cos and sin. Only YaRN changes it: 0.1 ln(factor) + 1, times the
/// configured attention factor.
pub fn ropeMscale(r: config_mod.RopeScaling) f32 {
    if (r.kind != .yarn) return 1.0;
    return r.attn_factor * @as(f32, @floatCast(1.0 + 0.1 * @log(@as(f64, r.factor))));
}

/// Fills cos and sin for one position. Computed once per token and shared by every head.
pub fn ropeAngles(cos_out: []f32, sin_out: []f32, inv_freq: []const f32, pos: usize, mscale: f32) void {
    const p: f32 = @floatFromInt(pos);
    for (inv_freq, 0..) |f, i| {
        const a = p * f;
        cos_out[i] = @cos(a) * mscale;
        sin_out[i] = @sin(a) * mscale;
    }
}

/// Rotates the first `2 * cos.len` dimensions of every head in `x` in place.
pub fn ropeApply(x: []f32, head_dim: usize, cos: []const f32, sin: []const f32, style: config_mod.RopeStyle) void {
    const half = cos.len;
    var h: usize = 0;
    while (h < x.len) : (h += head_dim) {
        const hv = x[h..][0..head_dim];
        switch (style) {
            .norm => for (0..half) |i| {
                const a = hv[2 * i];
                const b = hv[2 * i + 1];
                hv[2 * i] = a * cos[i] - b * sin[i];
                hv[2 * i + 1] = a * sin[i] + b * cos[i];
            },
            .neox => for (0..half) |i| {
                const a = hv[i];
                const b = hv[i + half];
                hv[i] = a * cos[i] - b * sin[i];
                hv[i + half] = a * sin[i] + b * cos[i];
            },
        }
    }
}

/// Converts f32 to f16 with round to nearest even. Written for slices so the compiler can
/// use hardware conversion where it exists.
pub fn toF16(dst: []f16, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    var i: usize = 0;
    while (i + V <= src.len) : (i += V) {
        const v: Vf = src[i..][0..V].*;
        const h: @Vector(V, f16) = @floatCast(v);
        dst[i..][0..V].* = h;
    }
    while (i < src.len) : (i += 1) dst[i] = @floatCast(src[i]);
}

/// Dot of an f32 vector with an f16 vector.
pub inline fn dotF32F16(a: []const f32, b: []const f16) f32 {
    std.debug.assert(a.len == b.len);
    var acc: Vf = @splat(0);
    var i: usize = 0;
    while (i + V <= a.len) : (i += V) {
        const va: Vf = a[i..][0..V].*;
        const vb: Vf = @floatCast(@as(@Vector(V, f16), b[i..][0..V].*));
        acc = @mulAdd(Vf, va, vb, acc);
    }
    var s = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) s += a[i] * @as(f32, @floatCast(b[i]));
    return s;
}

/// acc += w * v for an f16 vector v.
pub inline fn axpyF16(acc: []f32, w: f32, v: []const f16) void {
    std.debug.assert(acc.len == v.len);
    const vw: Vf = @splat(w);
    var i: usize = 0;
    while (i + V <= acc.len) : (i += V) {
        const a: Vf = acc[i..][0..V].*;
        const b: Vf = @floatCast(@as(@Vector(V, f16), v[i..][0..V].*));
        acc[i..][0..V].* = @mulAdd(Vf, vw, b, a);
    }
    while (i < acc.len) : (i += 1) acc[i] += w * @as(f32, @floatCast(v[i]));
}

test "rmsNorm matches the definition" {
    var x = [_]f32{ 2, -2, 2, -2 };
    const w = [_]f32{ 1, 1, 1, 1 };
    var out: [4]f32 = undefined;
    rmsNorm(&out, &x, &w, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 1), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1), out[1], 1e-6);
}

test "rope rotates by the requested angle in both styles" {
    var norm = [_]f32{ 1, 0, 0, 1 };
    var neox = [_]f32{ 1, 0, 0, 1 };
    const cos = [_]f32{ 0, 0 };
    const sin = [_]f32{ 1, 1 };
    // 90 degrees for both pairs.
    ropeApply(&norm, 4, &cos, &sin, .norm);
    ropeApply(&neox, 4, &cos, &sin, .neox);
    // norm pairs (x0,x1) and (x2,x3): (1,0)->(0,1), (0,1)->(-1,0)
    try std.testing.expectApproxEqAbs(@as(f32, 0), norm[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), norm[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1), norm[2], 1e-6);
    // neox pairs (x0,x2) and (x1,x3): (1,0)->(0,1), (0,1)->(-1,0)
    try std.testing.expectApproxEqAbs(@as(f32, 0), neox[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1), neox[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), neox[2], 1e-6);
}

test "f16 helpers agree with plain f32 math" {
    var a: [37]f32 = undefined;
    var b32: [37]f32 = undefined;
    var b16: [37]f16 = undefined;
    for (&a, &b32, 0..) |*x, *y, i| {
        x.* = @floatFromInt(i);
        y.* = 0.5 * @as(f32, @floatFromInt(i)) - 3;
    }
    toF16(&b16, &b32);
    var want: f32 = 0;
    for (a, b32) |x, y| want += x * y;
    try std.testing.expectApproxEqAbs(want, dotF32F16(&a, &b16), 1e-2);
    var acc: [37]f32 = @splat(1);
    axpyF16(&acc, 2, &b16);
    try std.testing.expectApproxEqAbs(1 + 2 * b32[5], acc[5], 1e-3);
}

test "vexp is accurate across the useful range" {
    var worst: f64 = 0;
    var x: f32 = -80;
    while (x < 80) : (x += 0.37) {
        var lane_arr: [V]f32 = undefined;
        for (0..V) |i| lane_arr[i] = x + @as(f32, @floatFromInt(i)) * 0.011;
        const lane: Vf = lane_arr;
        const got_arr: [V]f32 = vexp(lane);
        for (0..V) |i| {
            const want = @exp(@as(f64, lane_arr[i]));
            const rel = @abs((@as(f64, got_arr[i]) - want) / want);
            worst = @max(worst, rel);
        }
    }
    try std.testing.expect(worst < 3e-7);
    // Extreme inputs saturate instead of producing inf or nan.
    const big = vexp(@splat(1000.0));
    const small = vexp(@splat(-1000.0));
    try std.testing.expect(std.math.isFinite(big[0]) and big[0] > 1e30);
    try std.testing.expect(small[0] >= 0 and small[0] < 1e-30);
}

// Expected values come from Hugging Face Transformers (ROPE_INIT_FUNCTIONS) for head width 64.
fn expectFreqs(r: config_mod.RopeScaling, base: f32, probes: []const [2]f32) !void {
    var out: [32]f32 = undefined;
    ropeInvFreq(&out, base, 64, r, null);
    for (probes) |p| {
        const i: usize = @intFromFloat(p[0]);
        try std.testing.expectApproxEqRel(p[1], out[i], 2e-6);
    }
}

test "llama3 rope scaling matches transformers" {
    try expectFreqs(.{ .kind = .llama3, .factor = 8, .orig_ctx = 8192, .low_freq_factor = 1, .high_freq_factor = 4 }, 500000.0, &.{
        .{ 0, 1.0 },           .{ 1, 6.636012793e-01 }, .{ 7, 5.666961893e-02 }, .{ 8, 3.760603070e-02 },
        .{ 12, 7.292665076e-03 }, .{ 16, 5.248460220e-04 }, .{ 20, 3.428102355e-05 }, .{ 24, 6.647869668e-06 },
        .{ 31, 3.767322596e-07 },
    });
}

test "yarn rope scaling matches transformers, magnitude included" {
    const r = config_mod.RopeScaling{ .kind = .yarn, .factor = 4, .orig_ctx = 4096 };
    try expectFreqs(r, 10000.0, &.{
        .{ 0, 1.0 },           .{ 1, 7.498942018e-01 }, .{ 7, 1.333521456e-01 }, .{ 8, 1.000000015e-01 },
        .{ 12, 2.797399648e-02 }, .{ 16, 6.538461894e-03 }, .{ 20, 1.337886788e-03 }, .{ 24, 2.500000119e-04 },
        .{ 31, 3.333803761e-05 },
    });
    try std.testing.expectApproxEqRel(@as(f32, 1.138629436), ropeMscale(r), 1e-6);
    try std.testing.expectEqual(@as(f32, 1.0), ropeMscale(.{ .kind = .llama3, .factor = 8, .orig_ctx = 8192 }));
}

test "linear scaling and plain rope" {
    try expectFreqs(.{ .kind = .linear, .factor = 4 }, 10000.0, &.{
        .{ 0, 0.25 }, .{ 1, 1.874735504e-01 }, .{ 16, 2.499999944e-03 }, .{ 31, 3.333803761e-05 * 1.0 },
    });
    var plain: [32]f32 = undefined;
    ropeInvFreq(&plain, 10000.0, 64, .{}, null);
    try std.testing.expectEqual(@as(f32, 1.0), plain[0]);
    try std.testing.expectApproxEqRel(@as(f32, 4.0 * 3.333803761e-05), plain[31], 1e-5);
}

test "frequency factors divide the result" {
    var factors: [32]f32 = undefined;
    for (&factors, 0..) |*f, i| f.* = 1.0 + @as(f32, @floatFromInt(i));
    var with: [32]f32 = undefined;
    var without: [32]f32 = undefined;
    ropeInvFreq(&with, 10000.0, 64, .{}, &factors);
    ropeInvFreq(&without, 10000.0, 64, .{}, null);
    for (with, without, factors) |w, o, f| try std.testing.expectApproxEqRel(o / f, w, 1e-6);
}
