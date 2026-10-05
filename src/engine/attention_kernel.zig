//! Attention kernels: the per tile work and the task that drives it. Compiled once per
//! instruction set level (see kernels/impl.zig); `attention.zig` orchestrates the tasks.
//!
//! Causal attention over the f16 KV cache.
//!
//! Keys are scored sixteen positions per vector: the cache holds them transposed, so one
//! broadcast query component times one vector of sixteen keys is a single fused multiply add and
//! no horizontal sums are needed. Up to four query rows (a kv head's group of heads, or several
//! consecutive prompt tokens) share every key and value load. The softmax is online, so a
//! sequence of any length needs only a few KiB of stack.
//!
//! Work is split by (block of items, kv head, slice of positions). Prompt processing gets its
//! parallelism from the blocks. Single token decoding has few blocks, so the positions are cut
//! into slices that are merged afterwards, which keeps every core busy at long contexts.

const std = @import("std");
const config_mod = @import("config.zig");
const kv_mod = @import("kv.zig");
const ops = @import("ops.zig");
const pool_mod = @import("../pool.zig");

const Vf = ops.Vf;
const V = ops.V;
const KvCache = kv_mod.KvCache;
const Item = kv_mod.Item;

/// Positions per tile. A divisor of the segment length, so a tile never crosses a segment.
pub const tile = kv_mod.key_tile;
const NT = tile / V;
pub const chunk_rows = 4;
/// Most query rows one task carries state for.
pub const max_rows = 16;
/// Widest head supported (after padding to 16).
pub const max_head_dim = 256;

comptime {
    std.debug.assert(kv_mod.seg_len % tile == 0);
}

const neg_inf = -std.math.inf(f32);

pub const Job = struct {
    cfg: config_mod.Config,
    layer: usize,
    items: []const Item,
    q: []const f32,
    out: []f32,
    scale: f32,
    group: usize,
    /// Heads of a group handled by one task, and the number of such tasks per kv head.
    hpt: usize,
    n_hs: usize,
    hd_pad: usize,
    items_per_block: usize,
    n_blocks: usize,
    n_slices: usize,
    slice_len: usize,
    partials: []f32,
};

pub fn task(ctx: *anyopaque, start: usize, end: usize) void {
    const job: *const Job = @ptrCast(@alignCast(ctx));
    const per_block = job.cfg.n_kv_heads * job.n_hs * job.n_slices;
    const total = job.n_blocks * per_block;
    for (start..end) |id| {
        // Later blocks see more positions, so hand those out first.
        const rid = total - 1 - id;
        const block = rid / per_block;
        const rem = rid % per_block;
        const g = rem / (job.n_hs * job.n_slices);
        const hs = (rem / job.n_slices) % job.n_hs;
        const slice = rem % job.n_slices;
        const first = block * job.items_per_block;
        const last = @min(first + job.items_per_block, job.items.len);
        var a = first;
        while (a < last) {
            var b = a + 1;
            while (b < last and job.items[b].kv == job.items[a].kv) b += 1;
            processRun(job, a, b, g, hs, slice);
            a = b;
        }
    }
}

fn processRun(job: *const Job, a: usize, b: usize, g: usize, hs: usize, slice: usize) void {
    const cfg = job.cfg;
    const hd = cfg.head_dim;
    const hdp = job.hd_pad;
    const G = job.group;
    // This task's heads: `hn` of them starting at head `h0` of the group.
    const h0 = hs * job.hpt;
    const hn = @min(job.hpt, G - h0);
    const rows = (b - a) * hn;
    const kv = job.items[a].kv;
    const qd = cfg.qDim();

    var qs: [max_rows * max_head_dim]f32 align(64) = undefined;
    var acc: [max_rows * max_head_dim]f32 align(64) = undefined;
    var m: [max_rows]f32 = @splat(neg_inf);
    var l: [max_rows]f32 = @splat(0);
    var npos: [max_rows]usize = undefined;

    var max_np: usize = 0;
    for (0..rows) |r| {
        const item = a + r / hn;
        const head = g * G + h0 + r % hn;
        const src = job.q[item * qd + head * hd ..][0..hd];
        const dst = qs[r * hdp ..][0..hdp];
        for (dst[0..hd], src) |*d, s| d.* = s * job.scale;
        @memset(dst[hd..], 0);
        @memset(acc[r * hdp ..][0..hdp], 0);
        npos[r] = @as(usize, job.items[item].pos) + 1;
        max_np = @max(max_np, npos[r]);
    }

    const s0 = slice * job.slice_len;
    const s1 = @min(s0 + job.slice_len, max_np);
    // Start every fetch of this slice up front. Tiles are contiguous runs, so the memory system
    // streams them while the first tiles are computed instead of stalling on each in turn.
    {
        var q0 = s0;
        while (q0 < s1) : (q0 += tile) {
            const kb: [*]const u8 = @ptrCast(kv.kTile(job.layer, g, q0));
            const vb: [*]const u8 = @ptrCast(kv.vRows(job.layer, g, q0));
            var off: usize = 0;
            while (off < tile * hdp * 2) : (off += 64) {
                @prefetch(kb + off, .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(vb + off, .{ .rw = .read, .locality = 3, .cache = .data });
            }
        }
    }
    var p0 = s0;
    while (p0 < s1) : (p0 += tile) {
        const kt = kv.kTile(job.layer, g, p0);
        const vt = kv.vRows(job.layer, g, p0);
        var c: usize = 0;
        while (c < rows) : (c += chunk_rows) {
            const R = @min(chunk_rows, rows - c);
            var live = false;
            for (c..c + R) |r| live = live or npos[r] > p0;
            if (!live) continue;
            switch (R) {
                inline 1, 2, 3, 4 => |RR| tileChunk(RR, hdp, qs[c * hdp ..].ptr, acc[c * hdp ..].ptr, m[c..].ptr, l[c..].ptr, npos[c..].ptr, p0, kt, vt),
                else => unreachable,
            }
        }
    }

    for (0..rows) |r| {
        const item = a + r / hn;
        const head = g * G + h0 + r % hn;
        const row_acc = acc[r * hdp ..][0..hdp];
        if (job.n_slices == 1) {
            const inv = 1.0 / l[r];
            for (job.out[item * qd + head * hd ..][0..hd], row_acc[0..hd]) |*o, v| o.* = v * inv;
        } else {
            const slot = job.partials[((item * cfg.n_heads + head) * job.n_slices + slice) * (hdp + 2) ..][0 .. hdp + 2];
            slot[0] = m[r];
            slot[1] = l[r];
            @memcpy(slot[2..], row_acc);
        }
    }
}

inline fn loadK(p: [*]const f16) Vf {
    return @floatCast(@as(*align(2) const @Vector(V, f16), @ptrCast(p)).*);
}

/// One tile of keys and values for R query rows: scores, online softmax update, and the
/// weighted value sum.
inline fn tileChunk(
    comptime R: usize,
    hdp: usize,
    qs: [*]const f32,
    acc: [*]f32,
    m: [*]f32,
    l: [*]f32,
    npos: [*]const usize,
    p0: usize,
    kt: [*]const f16,
    vt: [*]const f16,
) void {
    // Value rows past the longest row of the chunk may hold anything, so never read them.
    var max_np: usize = 0;
    inline for (0..R) |r| max_np = @max(max_np, npos[r]);
    const jn = @min(tile, max_np - p0);

    var s: [R][tile]f32 align(64) = undefined;

    // Scores: s[r][j] = sum_d q[r][d] * k[d][j].
    {
        var sc: [R][NT]Vf = @splat(@splat(@splat(0)));
        var d: usize = 0;
        while (d < hdp) : (d += 1) {
            var kv: [NT]Vf = undefined;
            inline for (0..NT) |v| kv[v] = loadK(kt + d * tile + v * V);
            inline for (0..R) |r| {
                const qb: Vf = @splat(qs[r * hdp + d]);
                inline for (0..NT) |v| sc[r][v] = @mulAdd(Vf, qb, kv[v], sc[r][v]);
            }
        }
        inline for (0..R) |r| inline for (0..NT) |v| {
            s[r][v * V ..][0..V].* = sc[r][v];
        };
    }

    // Online softmax per row. Positions at or past the row's length are masked out.
    const iota: @Vector(V, i32) = comptime blk: {
        var t: [V]i32 = undefined;
        for (0..V) |i| t[i] = @intCast(i);
        break :blk t;
    };
    inline for (0..R) |r| {
        const valid: i32 = @intCast(@min(npos[r] -| p0, tile));
        var sv: [NT]Vf = undefined;
        var tmax: Vf = @splat(neg_inf);
        inline for (0..NT) |v| {
            const idx = iota + @as(@Vector(V, i32), @splat(@as(i32, @intCast(v * V))));
            const ok = idx < @as(@Vector(V, i32), @splat(valid));
            sv[v] = @select(f32, ok, @as(Vf, s[r][v * V ..][0..V].*), @as(Vf, @splat(neg_inf)));
            tmax = @max(tmax, sv[v]);
        }
        const tile_max = @reduce(.Max, tmax);
        if (tile_max == neg_inf) {
            inline for (0..NT) |v| s[r][v * V ..][0..V].* = @as(Vf, @splat(0));
        } else {
            const new_max = @max(m[r], tile_max);
            if (m[r] != new_max) {
                const corr: f32 = if (m[r] == neg_inf) 0 else @exp(m[r] - new_max);
                const cv: Vf = @splat(corr);
                var i: usize = 0;
                while (i < hdp) : (i += V) {
                    const cur: Vf = acc[r * hdp + i ..][0..V].*;
                    acc[r * hdp + i ..][0..V].* = cur * cv;
                }
                l[r] *= corr;
            }
            const nm: Vf = @splat(new_max);
            var psum: Vf = @splat(0);
            inline for (0..NT) |v| {
                const idx = iota + @as(@Vector(V, i32), @splat(@as(i32, @intCast(v * V))));
                const ok = idx < @as(@Vector(V, i32), @splat(valid));
                const e = @select(f32, ok, ops.vexp(sv[v] - nm), @as(Vf, @splat(0)));
                s[r][v * V ..][0..V].* = e;
                psum += e;
            }
            l[r] += @reduce(.Add, psum);
            m[r] = new_max;
        }
    }

    // Weighted values: acc[r][d] += sum_j p[r][j] * v[j][d], four vectors of d at a time.
    var d0: usize = 0;
    while (d0 + 4 * V <= hdp) : (d0 += 4 * V) {
        var a: [R][4]Vf = undefined;
        inline for (0..R) |r| inline for (0..4) |i| {
            a[r][i] = acc[r * hdp + d0 + i * V ..][0..V].*;
        };
        for (0..jn) |j| {
            var vv: [4]Vf = undefined;
            inline for (0..4) |i| vv[i] = loadK(vt + j * hdp + d0 + i * V);
            inline for (0..R) |r| {
                const pb: Vf = @splat(s[r][j]);
                inline for (0..4) |i| a[r][i] = @mulAdd(Vf, pb, vv[i], a[r][i]);
            }
        }
        inline for (0..R) |r| inline for (0..4) |i| {
            acc[r * hdp + d0 + i * V ..][0..V].* = a[r][i];
        };
    }
    while (d0 < hdp) : (d0 += V) {
        var a: [R]Vf = undefined;
        inline for (0..R) |r| a[r] = acc[r * hdp + d0 ..][0..V].*;
        for (0..jn) |j| {
            const vv = loadK(vt + j * hdp + d0);
            inline for (0..R) |r| a[r] = @mulAdd(Vf, @as(Vf, @splat(s[r][j])), vv, a[r]);
        }
        inline for (0..R) |r| acc[r * hdp + d0 ..][0..V].* = a[r];
    }
}

