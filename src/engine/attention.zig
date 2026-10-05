//! Causal attention orchestration: splits the work over the thread pool and merges sliced
//! decoding results. The per tile kernels live in `attention_kernel.zig`, which is built once
//! per instruction set level and reached through `kernels.get()`.

const std = @import("std");
const config_mod = @import("config.zig");
const kv_mod = @import("kv.zig");
const pool_mod = @import("../pool.zig");
const kernels = @import("../kernels.zig");
const kernel = @import("attention_kernel.zig");

const KvCache = kv_mod.KvCache;
const Item = kv_mod.Item;
const Job = kernel.Job;
const tile = kernel.tile;
const max_head_dim = kernel.max_head_dim;
const chunk_rows = kernel.chunk_rows;
const max_rows = kernel.max_rows;
const neg_inf = -std.math.inf(f32);

/// Growable scratch for the partial results of sliced decoding.
pub const Partials = struct {
    buf: []f32 = &.{},

    pub fn deinit(self: *Partials, allocator: std.mem.Allocator) void {
        if (self.buf.len != 0) allocator.free(self.buf);
        self.buf = &.{};
    }

    fn ensure(self: *Partials, allocator: std.mem.Allocator, n: usize) !void {
        if (self.buf.len >= n) return;
        if (self.buf.len != 0) allocator.free(self.buf);
        self.buf = &.{};
        self.buf = try allocator.alloc(f32, n);
    }
};

pub fn supportsHeadDim(hd: usize) bool {
    return (hd + 15) / 16 * 16 <= max_head_dim;
}

/// out[t][h * hd ..] = softmax(q . K^T * scale) V for every item `t` and head `h`.
/// `q` and `out` are [item][n_heads * head_dim]. All keys up to each item's position must
/// already be in the cache.
pub fn run(
    pool: *pool_mod.Pool,
    allocator: std.mem.Allocator,
    partials: *Partials,
    cfg: config_mod.Config,
    layer: usize,
    items: []const Item,
    q: []const f32,
    out: []f32,
) error{OutOfMemory}!void {
    const hd_pad = items[0].kv.hd_pad;
    const group = cfg.n_heads / cfg.n_kv_heads;
    const hpt = @min(group, max_rows);
    const n_hs = (group + hpt - 1) / hpt;
    const ipb: usize = @max(1, @min(chunk_rows, max_rows / hpt));
    const n_blocks = (items.len + ipb - 1) / ipb;

    var max_pos: usize = 0;
    for (items) |it| max_pos = @max(max_pos, @as(usize, it.pos) + 1);

    // Cut the positions into slices when there are too few tasks to feed every thread.
    const threads = pool.size();
    var n_slices: usize = 1;
    var slice_len: usize = max_pos;
    const base_tasks = n_blocks * cfg.n_kv_heads * n_hs;
    if (base_tasks < 4 * threads and max_pos >= 2 * tile) {
        const want = (4 * threads + base_tasks - 1) / base_tasks;
        const by_len = max_pos / tile;
        n_slices = @max(1, @min(@min(want, by_len), 16));
        slice_len = (((max_pos + n_slices - 1) / n_slices) + tile - 1) / tile * tile;
        n_slices = (max_pos + slice_len - 1) / slice_len;
    }

    var part: []f32 = &.{};
    if (n_slices > 1) {
        try partials.ensure(allocator, items.len * cfg.n_heads * n_slices * (hd_pad + 2));
        part = partials.buf;
    }

    var job = Job{
        .cfg = cfg,
        .layer = layer,
        .items = items,
        .q = q,
        .out = out,
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim))),
        .group = group,
        .hpt = hpt,
        .n_hs = n_hs,
        .hd_pad = hd_pad,
        .items_per_block = ipb,
        .n_blocks = n_blocks,
        .n_slices = n_slices,
        .slice_len = slice_len,
        .partials = part,
    };
    pool.parallelFor(n_blocks * cfg.n_kv_heads * n_hs * n_slices, 1, &job, kernels.get().attn_task);
    if (n_slices > 1) merge(&job);
}

/// Combines the per slice partial softmax states of every (item, head).
fn merge(job: *const Job) void {
    const cfg = job.cfg;
    const hd = cfg.head_dim;
    const hdp = job.hd_pad;
    const qd = cfg.qDim();
    for (0..job.items.len) |item| {
        for (0..cfg.n_heads) |head| {
            const base = (item * cfg.n_heads + head) * job.n_slices * (hdp + 2);
            var big: f32 = neg_inf;
            for (0..job.n_slices) |s| big = @max(big, job.partials[base + s * (hdp + 2)]);
            var total: f32 = 0;
            const o = job.out[item * qd + head * hd ..][0..hd];
            @memset(o, 0);
            for (0..job.n_slices) |s| {
                const slot = job.partials[base + s * (hdp + 2) ..][0 .. hdp + 2];
                if (slot[1] == 0) continue;
                const w = @exp(slot[0] - big);
                total += w * slot[1];
                for (o, slot[2..][0..hd]) |*x, v| x.* += w * v;
            }
            const inv = 1.0 / total;
            for (o) |*x| x.* *= inv;
        }
    }
}

// ---------------------------------------------------------------------------------------
// Tests: compare against a plain f64 implementation reading the same cache.
// ---------------------------------------------------------------------------------------

fn testConfig(n_heads: usize, n_kv: usize, hd: usize) config_mod.Config {
    return .{
        .arch = .llama, .n_layers = 1, .dim = n_heads * hd, .ffn_dim = 8, .n_heads = n_heads,
        .n_kv_heads = n_kv, .head_dim = hd, .vocab = 8, .ctx_train = 4096, .rms_eps = 1e-5,
        .rope_base = 10000, .rope_dim = hd, .rope_style = .neox, .rope_scaling = .{},
        .ffn_act = .silu, .qk_norm = false, .qkv_bias = false,
    };
}

fn referenceAttention(cfg: config_mod.Config, kv: *const KvCache, item: Item, head: usize, q: []const f32, out: []f32) void {
    const hd = cfg.head_dim;
    const g = head / (cfg.n_heads / cfg.n_kv_heads);
    const n_pos = @as(usize, item.pos) + 1;
    var scores: [1024]f64 = undefined;
    var mx: f64 = -std.math.inf(f64);
    for (0..n_pos) |p| {
        const kt = kv.kTile(0, g, p / tile * tile);
        var s: f64 = 0;
        for (0..hd) |d| s += @as(f64, q[d]) * @as(f64, @floatCast(kt[d * tile + p % tile]));
        s /= @sqrt(@as(f64, @floatFromInt(hd)));
        scores[p] = s;
        mx = @max(mx, s);
    }
    var sum: f64 = 0;
    for (0..n_pos) |p| {
        scores[p] = @exp(scores[p] - mx);
        sum += scores[p];
    }
    for (0..hd) |d| {
        var o: f64 = 0;
        for (0..n_pos) |p| o += scores[p] * @as(f64, @floatCast(kv.vRows(0, g, p)[d]));
        out[d] = @floatCast(o / sum);
    }
}

fn checkAttention(n_heads: usize, n_kv: usize, hd: usize, n_items: usize, first_pos: usize, threads: usize) !void {
    const allocator = std.testing.allocator;
    const cfg = testConfig(n_heads, n_kv, hd);
    var pool = try pool_mod.Pool.init(allocator, threads);
    defer pool.deinit();
    var kv = try KvCache.init(allocator, cfg, 1024);
    defer kv.deinit();
    try kv.ensure(first_pos + n_items);
    var prng = std.Random.DefaultPrng.init(0xA77E);
    const rand = prng.random();

    const kd = cfg.kvDim();
    const kbuf = try allocator.alloc(f32, kd);
    defer allocator.free(kbuf);
    const vbuf = try allocator.alloc(f32, kd);
    defer allocator.free(vbuf);
    for (0..first_pos + n_items) |p| {
        for (kbuf) |*x| x.* = rand.floatNorm(f32);
        for (vbuf) |*x| x.* = rand.floatNorm(f32);
        for (0..n_kv) |g| kv.write(0, g, p, kbuf[g * hd ..][0..hd], vbuf[g * hd ..][0..hd]);
    }

    const items = try allocator.alloc(Item, n_items);
    defer allocator.free(items);
    for (items, 0..) |*it, i| it.* = .{ .kv = &kv, .token = 0, .pos = @intCast(first_pos + i) };
    const qd = cfg.qDim();
    const q = try allocator.alloc(f32, n_items * qd);
    defer allocator.free(q);
    for (q) |*x| x.* = rand.floatNorm(f32);
    const out = try allocator.alloc(f32, n_items * qd);
    defer allocator.free(out);

    var partials = Partials{};
    defer partials.deinit(allocator);
    try run(&pool, allocator, &partials, cfg, 0, items, q, out);

    var want: [max_head_dim]f32 = undefined;
    for (items, 0..) |it, i| {
        for (0..n_heads) |h| {
            referenceAttention(cfg, &kv, it, h, q[i * qd + h * hd ..][0..hd], want[0..hd]);
            for (0..hd) |d| {
                const got = out[i * qd + h * hd + d];
                if (!(@abs(got - want[d]) <= 2e-4 * @max(1.0, @abs(want[d])))) {
                    std.debug.print("heads {d}/{d} hd {d} item {d} pos {d} head {d} dim {d}: got {d} want {d}\n", .{ n_heads, n_kv, hd, i, it.pos, h, d, got, want[d] });
                    return error.AttentionMismatch;
                }
            }
        }
    }
}

test "attention matches a plain implementation" {
    // A prompt batch: many rows share tiles, ragged tile edges, groups of three heads.
    try checkAttention(6, 2, 64, 130, 0, 4);
    // Continuing a prompt past a segment boundary.
    try checkAttention(6, 2, 64, 40, 240, 3);
    // Single token decoding at long context uses sliced positions.
    try checkAttention(6, 2, 64, 1, 700, 6);
    try checkAttention(9, 3, 64, 1, 563, 6);
    // Plain multi head, head width that needs padding, and a wide group.
    try checkAttention(4, 4, 80, 9, 100, 2);
    try checkAttention(20, 1, 32, 3, 300, 4);
    try checkAttention(8, 2, 128, 5, 650, 1);
}
