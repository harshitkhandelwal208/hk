//! Register tiled matrix times matrix for batches of tokens (prompt processing).
//!
//! `vecdot` computes one weight row against a few activation vectors, so every integer dot
//! ends in a horizontal sum and every block pays a scale multiply per token. Here a tile of 16
//! weight rows is first unpacked into scratch memory with the rows as SIMD lanes: one vector
//! then holds the same four columns of eight different rows, and one `vpdpbusd` against a
//! broadcast of four activation bytes advances eight output rows at once. Scales are vectors
//! too (one lane per row), so the float work per block is paid once for eight rows and no
//! reduction is needed at the end.
//!
//! The unpack is done per tile, on the fly, into a small per thread buffer. Weights are never
//! copied wholesale, so the model stays a memory mapped file and its private memory stays
//! tiny; the unpack is amortized over every token in the batch.
//!
//! Three kernel families cover every format:
//!   legacy  32 weight blocks with one float scale (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL)
//!   kquant  256 weight super blocks with 16 weight integer scales (Q2_K .. Q6_K, IQ4_XS)
//!   float   everything else (f32, f16, bf16, IQ2/IQ3/IQ1, ternary, FP4): decoded to f32

const std = @import("std");
const format = @import("../format.zig");
const blocks = @import("blocks.zig");
const dequant = @import("dequant.zig");
const unpack = @import("unpack.zig");
const tables = @import("tables.zig");
const vecdot = @import("vecdot.zig");
const isa = @import("isa.zig");

const StorageType = format.StorageType;
const V8i = vecdot.V8i;
const V8f = vecdot.V8f;
const U32 = vecdot.U32;
const S32 = vecdot.S32;
const BlockA8 = vecdot.BlockA8;
const BlockQ8K = vecdot.BlockQ8K;
const dpu = isa.dpu;
const dps = isa.dps;
const dpw = isa.dpw;

/// Rows per SIMD register and per tile.
pub const lanes = 8;
/// Row groups per tile: as many as the register file holds accumulators for.
pub const row_groups = isa.tile_row_groups;
pub const tile_rows = lanes * row_groups;
/// Tokens handled by one micro kernel call.
pub const tile_tokens = isa.tile_tokens;
comptime {
    std.debug.assert(tile_tokens == 4); // the dispatch over a partial last group covers 1 to 4
}

pub const Kind = enum { legacy, kquant, float };

pub fn kindOf(t: StorageType) Kind {
    return switch (t) {
        .q8_0, .q4_0, .q4_1, .q5_0, .q5_1, .iq4_nl, .mxfp4 => .legacy,
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .iq4_xs, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m, .tq1_0, .tq2_0 => .kquant,
        else => .float,
    };
}

/// Bytes of scratch one thread needs to unpack a 16 row tile of `cols` columns.
pub fn scratchBytes(t: StorageType, cols: usize) usize {
    const per_group: usize = switch (kindOf(t)) {
        .legacy => (cols / 32) * 320,
        .kquant => (cols / 256) * 2880,
        .float => cols * lanes * 4,
    };
    return per_group * row_groups;
}

/// True when this module can run `t`. Block formats need whole blocks per row.
pub fn supported(t: StorageType) bool {
    return vecdot.supported(t);
}

// ---------------------------------------------------------------------------------------
// Small vector helpers
// ---------------------------------------------------------------------------------------

inline fn loadU(p: [*]const u8) U32 {
    return @as(*align(32) const [32]u8, @ptrCast(@alignCast(p))).*;
}

inline fn loadI(p: [*]const u8) V8i {
    return @as(*align(32) const [8]i32, @ptrCast(@alignCast(p))).*;
}

inline fn loadF(p: [*]const u8) V8f {
    return @as(*align(32) const [8]f32, @ptrCast(@alignCast(p))).*;
}

/// Four consecutive bytes repeated in every 32 bit lane.
inline fn bcast4(p: *const i8) S32 {
    const w: u32 = @as(*align(1) const u32, @ptrCast(p)).*;
    return @bitCast(@as(@Vector(8, u32), @splat(w)));
}

inline fn bcastI(w: u32) V8i {
    return @bitCast(@as(@Vector(8, u32), @splat(w)));
}

inline fn h(x: f16) f32 {
    return @floatCast(x);
}

/// A signed weight vector with its magnitude and sign mask precomputed once, so the cost is
/// shared by every token of the tile on instruction sets whose dot needs an unsigned operand.
const Signed = struct { w: S32, a: U32, neg: @Vector(32, bool) };

inline fn prepSigned(v: U32) Signed {
    const s: S32 = @bitCast(v);
    return .{ .w = s, .a = @abs(s), .neg = s < @as(S32, @splat(0)) };
}

inline fn dotSigned(acc: V8i, p: Signed, y: S32) V8i {
    if (comptime isa.has_dotprod or !(isa.has_vnni or isa.has_avx2)) return dps(acc, p.w, y);
    return dpu(acc, p.a, @select(i8, p.neg, -%y, y));
}

inline fn blk(comptime B: type, row: []const u8, i: usize) *align(1) const B {
    return @ptrCast(row[i * @sizeOf(B) ..].ptr);
}

// ---------------------------------------------------------------------------------------
// Micro kernels
// ---------------------------------------------------------------------------------------

/// RG groups of 8 rows against T tokens over `nb` 32 weight blocks.
/// Weights: `qw` holds u8 values as [block][k4][row][4 bytes]. A weight is
/// `dv * (u - zp)` plus `mv` when `has_m`.
fn kernelLegacy(
    comptime RG: usize,
    comptime T: usize,
    comptime zp: i32,
    comptime has_m: bool,
    comptime signed: bool,
    nb: usize,
    qw: [RG][*]const u8,
    dv: [RG][*]const u8,
    mv: [RG][*]const u8,
    act: [T][*]const BlockA8,
) [RG][T]V8f {
    @setEvalBranchQuota(100_000);
    var accf: [RG][T]V8f = @splat(@splat(@splat(0)));
    var b: usize = 0;
    while (b < nb) : (b += 1) {
        var acci: [RG][T]V8i = @splat(@splat(@splat(0)));
        inline for (0..8) |j| {
            var w: [RG]U32 = undefined;
            var sw: [RG]Signed = undefined;
            inline for (0..RG) |g| {
                w[g] = loadU(qw[g] + (b * 8 + j) * 32);
                if (comptime signed) sw[g] = prepSigned(w[g]);
            }
            inline for (0..T) |t| {
                const y = bcast4(&act[t][b].qs[4 * j]);
                inline for (0..RG) |g| {
                    acci[g][t] = if (comptime signed) dotSigned(acci[g][t], sw[g], y) else dpu(acci[g][t], w[g], y);
                }
            }
        }
        inline for (0..T) |t| {
            const a = &act[t][b];
            const dy: V8f = @splat(a.d);
            const corr: V8i = @splat(zp * a.s);
            const dys: V8f = @splat(a.d * @as(f32, @floatFromInt(a.s)));
            inline for (0..RG) |g| {
                var v = acci[g][t];
                if (comptime zp != 0) v -= corr;
                accf[g][t] = @mulAdd(V8f, @floatFromInt(v), loadF(dv[g] + b * 32) * dy, accf[g][t]);
                if (comptime has_m) accf[g][t] = @mulAdd(V8f, loadF(mv[g] + b * 32), dys, accf[g][t]);
            }
        }
    }
    return accf;
}

/// RG groups of 8 rows against T tokens over `nsb` 256 weight super blocks. A weight is
/// `dv * sc * u - ev * mn`, with `sc` and `mn` integers per 16 weights.
fn kernelK(
    comptime RG: usize,
    comptime T: usize,
    comptime signed: bool,
    nsb: usize,
    qw: [RG][*]const u8,
    sc: [RG][*]const u8,
    mp: [RG][*]const u8,
    dv: [RG][*]const u8,
    ev: [RG][*]const u8,
    act: [T][*]const BlockQ8K,
) [RG][T]V8f {
    @setEvalBranchQuota(100_000);
    var accf: [RG][T]V8f = @splat(@splat(@splat(0)));
    var sb: usize = 0;
    while (sb < nsb) : (sb += 1) {
        var acc_sb: [RG][T]V8i = @splat(@splat(@splat(0)));
        var acc_mn: [RG][T]V8i = @splat(@splat(@splat(0)));
        inline for (0..16) |gi| {
            var w: [RG][4]U32 = undefined;
            var sw: [RG][4]Signed = undefined;
            var s: [RG]V8i = undefined;
            inline for (0..RG) |g| {
                inline for (0..4) |jj| {
                    w[g][jj] = loadU(qw[g] + (sb * 64 + gi * 4 + jj) * 32);
                    if (comptime signed) sw[g][jj] = prepSigned(w[g][jj]);
                }
                s[g] = loadI(sc[g] + (sb * 16 + gi) * 32);
            }
            inline for (0..T) |t| {
                var y: [4]S32 = undefined;
                inline for (0..4) |jj| y[jj] = bcast4(&act[t][sb].qs[gi * 16 + jj * 4]);
                inline for (0..RG) |g| {
                    var d4: V8i = @splat(0);
                    inline for (0..4) |jj| d4 = if (comptime signed) dotSigned(d4, sw[g][jj], y[jj]) else dpu(d4, w[g][jj], y[jj]);
                    acc_sb[g][t] += d4 * s[g];
                }
            }
        }
        inline for (0..8) |gp| {
            var m: [RG]V8i = undefined;
            inline for (0..RG) |g| m[g] = loadI(mp[g] + (sb * 8 + gp) * 32);
            inline for (0..T) |t| {
                const bs = bcastI(@as(*align(1) const u32, @ptrCast(&act[t][sb].bsums[2 * gp])).*);
                inline for (0..RG) |g| acc_mn[g][t] = dpw(acc_mn[g][t], m[g], bs);
            }
        }
        inline for (0..T) |t| {
            const dy: V8f = @splat(act[t][sb].d);
            inline for (0..RG) |g| {
                accf[g][t] = @mulAdd(V8f, @floatFromInt(acc_sb[g][t]), loadF(dv[g] + sb * 32) * dy, accf[g][t]);
                accf[g][t] = @mulAdd(V8f, @floatFromInt(acc_mn[g][t]), -(loadF(ev[g] + sb * 32) * dy), accf[g][t]);
            }
        }
    }
    return accf;
}

/// Plain f32: weights as [k][row] floats, activations as float vectors.
fn kernelFloat(
    comptime RG: usize,
    comptime T: usize,
    cols: usize,
    w: [RG][*]const u8,
    x: [T][*]const f32,
) [RG][T]V8f {
    var accf: [RG][T]V8f = @splat(@splat(@splat(0)));
    var k: usize = 0;
    while (k < cols) : (k += 1) {
        var wv: [RG]V8f = undefined;
        inline for (0..RG) |g| wv[g] = loadF(w[g] + k * 32);
        inline for (0..T) |t| {
            const xb: V8f = @splat(x[t][k]);
            inline for (0..RG) |g| accf[g][t] = @mulAdd(V8f, wv[g], xb, accf[g][t]);
        }
    }
    return accf;
}

// ---------------------------------------------------------------------------------------
// Unpacking one row of a tile
// ---------------------------------------------------------------------------------------

/// Zero point of the unsigned values for a legacy type, or 0 when the offset is the float `m`.
fn legacyZeroPoint(t: StorageType) i32 {
    return switch (t) {
        .q4_0 => 8,
        .q5_0 => 16,
        else => 0,
    };
}

/// Types whose weights are signed bytes. They feed the signed dot, so there is no zero point.
fn legacySigned(t: StorageType) bool {
    return t == .q8_0 or t == .iq4_nl or t == .mxfp4;
}

fn legacyHasM(t: StorageType) bool {
    return t == .q4_1 or t == .q5_1;
}

fn repackLegacy(comptime t: StorageType, base: [*]u8, nb: usize, r: usize, row: ?[]const u8) void {
    const qw: [*]u32 = @ptrCast(@alignCast(base));
    const dv: [*]f32 = @ptrCast(@alignCast(base + nb * 256));
    const mv: [*]f32 = @ptrCast(@alignCast(base + nb * 288));
    for (0..nb) |b| {
        var blkv: unpack.LegacyBlock = .{ .vals = @splat(0), .d = 0, .m = 0 };
        if (row) |rw| blkv = unpack.unpackLegacy(t, rw, b);
        inline for (0..8) |j| qw[(b * 8 + j) * 8 + r] = std.mem.readInt(u32, blkv.vals[4 * j ..][0..4], .little);
        dv[b * 8 + r] = blkv.d;
        mv[b * 8 + r] = blkv.m;
    }
}

fn repackK(comptime t: StorageType, base: [*]u8, nsb: usize, r: usize, row: ?[]const u8) void {
    const qw: [*]u32 = @ptrCast(@alignCast(base));
    const sc: [*]i32 = @ptrCast(@alignCast(base + nsb * 2048));
    const mp: [*]u32 = @ptrCast(@alignCast(base + nsb * 2560));
    const dv: [*]f32 = @ptrCast(@alignCast(base + nsb * 2816));
    const ev: [*]f32 = @ptrCast(@alignCast(base + nsb * 2848));
    for (0..nsb) |s| {
        var kb: unpack.KBlock = .{ .vals = @splat(0), .sc = @splat(0), .mn = @splat(0), .d = 0, .e = 0 };
        if (row) |rw| kb = unpack.unpackK(t, rw, s);
        for (0..64) |k4| qw[(s * 64 + k4) * 8 + r] = std.mem.readInt(u32, kb.vals[4 * k4 ..][0..4], .little);
        for (0..16) |g| sc[(s * 16 + g) * 8 + r] = kb.sc[g];
        for (0..8) |gp| {
            const lo: u32 = @as(u16, @bitCast(@as(i16, @intCast(kb.mn[2 * gp]))));
            const hi: u32 = @as(u16, @bitCast(@as(i16, @intCast(kb.mn[2 * gp + 1]))));
            mp[(s * 8 + gp) * 8 + r] = lo | (hi << 16);
        }
        dv[s * 8 + r] = kb.d;
        ev[s * 8 + r] = kb.e;
    }
}

/// Decodes one row into the [k][row] float layout.
fn repackFloat(t: StorageType, base: [*]u8, cols: usize, r: usize, row: ?[]const u8) void {
    const w: [*]f32 = @ptrCast(@alignCast(base));
    const rw = row orelse {
        for (0..cols) |k| w[k * 8 + r] = 0;
        return;
    };
    switch (t) {
        .f32, .f16, .bf16 => {
            const L = 16;
            var k: usize = 0;
            while (k + L <= cols) : (k += L) {
                const v: @Vector(L, f32) = switch (t) {
                    .f32 => @as(*align(1) const [L]f32, @ptrCast(rw[k * 4 ..].ptr)).*,
                    .f16 => @floatCast(@as(@Vector(L, f16), @bitCast(@as(*align(1) const [L]u16, @ptrCast(rw[k * 2 ..].ptr)).*))),
                    else => @bitCast(@as(@Vector(L, u32), @as(*align(1) const [L]u16, @ptrCast(rw[k * 2 ..].ptr)).*) << @as(@Vector(L, u5), @splat(16))),
                };
                inline for (0..L) |j| w[(k + j) * 8 + r] = v[j];
            }
            while (k < cols) : (k += 1) {
                var one: [1]f32 = undefined;
                dequant.dequantizeRow(t, rw[k * blocks.info(t).?.bytes ..][0..blocks.info(t).?.bytes], &one) catch unreachable;
                w[k * 8 + r] = one[0];
            }
        },
        else => {
            const info = blocks.info(t).?;
            var buf: [256]f32 = undefined;
            const n_blocks = cols / info.elems;
            for (0..n_blocks) |b| {
                dequant.dequantizeRow(t, rw[b * info.bytes ..][0..info.bytes], buf[0..info.elems]) catch unreachable;
                for (0..info.elems) |j| w[(b * info.elems + j) * 8 + r] = buf[j];
            }
        },
    }
}

// ---------------------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------------------

pub const Args = struct {
    t: StorageType,
    data: []const u8,
    rows: usize,
    cols: usize,
    row_bytes: usize,
    acts: []const vecdot.Act,
    y: []f32,
    bias: ?[]const f32,
};

/// Computes y[t * rows + r] = dot(weights row r, acts[t]) + bias[r] for the rows in
/// [row_start, row_end) rounded out to tiles. `scratch` must hold `scratchBytes(t, cols)`.
pub fn tiles(a: *const Args, scratch: []align(64) u8, tile_start: usize, tile_end: usize) void {
    switch (kindOf(a.t)) {
        .legacy => switch (a.t) {
            inline .q8_0, .q4_0, .q4_1, .q5_0, .q5_1, .iq4_nl, .mxfp4 => |tt| runTiles(.legacy, tt, a, scratch, tile_start, tile_end),
            else => unreachable,
        },
        .kquant => switch (a.t) {
            inline .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .iq4_xs, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m, .tq1_0, .tq2_0 => |tt| runTiles(.kquant, tt, a, scratch, tile_start, tile_end),
            else => unreachable,
        },
        .float => runTiles(.float, .f32, a, scratch, tile_start, tile_end),
    }
}

/// `t` is the comptime storage type for the integer families and ignored for `.float`, whose
/// decoder reads the runtime type from `a`.
fn runTiles(
    comptime kind: Kind,
    comptime t: StorageType,
    a: *const Args,
    scratch: []align(64) u8,
    tile_start: usize,
    tile_end: usize,
) void {
    const per_group = scratchBytes(a.t, a.cols) / 2;
    const n_tok = a.acts.len;
    for (tile_start..tile_end) |tile| {
        const r0 = tile * tile_rows;
        const n_rows: usize = @min(tile_rows, a.rows - r0);
        const n_groups: usize = (n_rows + lanes - 1) / lanes;

        // Unpack the rows of this tile; missing rows of the last group become zeros.
        inline for (0..row_groups) |g| {
            if (g < n_groups) {
                const base = scratch.ptr + g * per_group;
                for (0..lanes) |r| {
                    const gr = g * lanes + r;
                    const row: ?[]const u8 = if (gr < n_rows) a.data[(r0 + gr) * a.row_bytes ..][0..a.row_bytes] else null;
                    switch (kind) {
                        .legacy => repackLegacy(t, base, a.cols / 32, r, row),
                        .kquant => repackK(t, base, a.cols / 256, r, row),
                        .float => repackFloat(a.t, base, a.cols, r, row),
                    }
                }
            }
        }

        var t0: usize = 0;
        while (t0 < n_tok) {
            const cnt: usize = @min(tile_tokens, n_tok - t0);
            switch (cnt) {
                inline 1, 2, 3, 4 => |c| tokenGroup(kind, t, a, scratch, per_group, n_groups, r0, n_rows, t0, c),
                else => unreachable,
            }
            t0 += cnt;
        }
    }
}

inline fn tokenGroup(
    comptime kind: Kind,
    comptime t: StorageType,
    a: *const Args,
    scratch: []align(64) u8,
    per_group: usize,
    n_groups: usize,
    r0: usize,
    n_rows: usize,
    t0: usize,
    comptime T: usize,
) void {
    switch (n_groups) {
        inline 1, 2 => |RG| if (RG <= row_groups) {
            var base: [RG][*]const u8 = undefined;
            inline for (0..RG) |g| base[g] = scratch.ptr + g * per_group;
            const out = group(kind, t, RG, T, a, base, t0);
            store(RG, T, a, out, r0, n_rows, t0);
        } else unreachable,
        else => unreachable,
    }
}

inline fn group(
    comptime kind: Kind,
    comptime t: StorageType,
    comptime RG: usize,
    comptime T: usize,
    a: *const Args,
    base: [RG][*]const u8,
    t0: usize,
) [RG][T]V8f {
    switch (kind) {
        .legacy => {
            const nb = a.cols / 32;
            var act: [T][*]const BlockA8 = undefined;
            inline for (0..T) |k| act[k] = a.acts[t0 + k].a8.ptr;
            var qw: [RG][*]const u8 = undefined;
            var dv: [RG][*]const u8 = undefined;
            var mv: [RG][*]const u8 = undefined;
            inline for (0..RG) |g| {
                qw[g] = base[g];
                dv[g] = base[g] + nb * 256;
                mv[g] = base[g] + nb * 288;
            }
            return switch (t) {
                inline .q8_0, .q4_0, .q4_1, .q5_0, .q5_1, .iq4_nl, .mxfp4 => |tt| kernelLegacy(RG, T, comptime legacyZeroPoint(tt), comptime legacyHasM(tt), comptime legacySigned(tt), nb, qw, dv, mv, act),
                else => unreachable,
            };
        },
        .kquant => {
            const nsb = a.cols / 256;
            var act: [T][*]const BlockQ8K = undefined;
            inline for (0..T) |k| act[k] = a.acts[t0 + k].q8k.ptr;
            var qw: [RG][*]const u8 = undefined;
            var sc: [RG][*]const u8 = undefined;
            var mp: [RG][*]const u8 = undefined;
            var dv: [RG][*]const u8 = undefined;
            var ev: [RG][*]const u8 = undefined;
            inline for (0..RG) |g| {
                qw[g] = base[g];
                sc[g] = base[g] + nsb * 2048;
                mp[g] = base[g] + nsb * 2560;
                dv[g] = base[g] + nsb * 2816;
                ev[g] = base[g] + nsb * 2848;
            }
            return kernelK(RG, T, t != .q2_k and t != .q3_k and t != .q4_k and t != .q5_k and t != .q6_k, nsb, qw, sc, mp, dv, ev, act);
        },
        .float => {
            var x: [T][*]const f32 = undefined;
            inline for (0..T) |k| x[k] = a.acts[t0 + k].x.ptr;
            return kernelFloat(RG, T, a.cols, base, x);
        },
    }
}

inline fn store(comptime RG: usize, comptime T: usize, a: *const Args, out: [RG][T]V8f, r0: usize, n_rows: usize, t0: usize) void {
    inline for (0..T) |k| {
        inline for (0..RG) |g| {
            const row0 = r0 + g * lanes;
            const dst = a.y[(t0 + k) * a.rows + row0 ..];
            if (g * lanes + lanes <= n_rows) {
                var v = out[g][k];
                if (a.bias) |bs| v += @as(*align(1) const [8]f32, @ptrCast(bs[row0..].ptr)).*;
                @as(*align(1) [8]f32, @ptrCast(dst.ptr)).* = v;
            } else {
                const left = n_rows - g * lanes;
                const arr: [8]f32 = out[g][k];
                var i: usize = 0;
                while (i < left) : (i += 1) dst[i] = arr[i] + (if (a.bias) |bs| bs[row0 + i] else 0);
            }
        }
    }
}
