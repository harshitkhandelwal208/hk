//! The CPU executor: thread pool, scratch buffers and the per-layer loop.
//!
//! Memory behaviour is the point of this file:
//!   * weights are views into the mapped container, never copied or converted;
//!   * the KV cache is f16 and grows in segments of `seg_len` positions as the context fills,
//!     so resident size follows the real context length rather than `n_ctx`, whatever the
//!     kernel's transparent huge page setting is;
//!   * attention is a single pass with an online softmax, so its scratch is a few hundred
//!     bytes on the stack no matter how long the context is;
//!   * every buffer a forward pass needs is allocated once in `init`.
//!
//! This is the same code that used to live inline in `Model`; it moved here so the CPU is a peer of the
//! device backends rather than a special case. It owns everything only the CPU needs (pool, scratch, rope
//! angle tables, profile) and borrows the weights per call, so the struct can move freely.

const std = @import("std");
const pool_mod = @import("../pool.zig");
const quant = @import("../quant.zig");
const config_mod = @import("config.zig");
const weights_mod = @import("weights.zig");
const matmul = @import("matmul.zig");
const kv_mod = @import("kv.zig");
const attention_mod = @import("attention.zig");
const kernels = @import("../kernels.zig");
const ops = @import("ops.zig");

pub const Config = config_mod.Config;
pub const Item = kv_mod.Item;
const vecdot = quant.vecdot;
const Mat = weights_mod.Mat;

pub const Error = error{OutOfMemory};

/// Wall time per phase of the forward pass, filled in only when `Cpu.profile.enabled` is set.
pub const Profile = struct {
    enabled: bool = false,
    embed: i128 = 0,
    norm: i128 = 0,
    prepare: i128 = 0,
    qkv: i128 = 0,
    rope: i128 = 0,
    kv: i128 = 0,
    attn: i128 = 0,
    wo: i128 = 0,
    ffn_in: i128 = 0,
    act: i128 = 0,
    ffn_down: i128 = 0,
};

pub const Cpu = struct {
    allocator: std.mem.Allocator,
    cfg: Config,
    pool: pool_mod.Pool,
    n_batch: usize,

    /// Borrowed from the model; the model keeps them alive for as long as this executor runs.
    inv_freq: []const f32,
    rope_mscale: f32,
    cos: []f32,
    sin: []f32,

    x: []f32,
    xb: []f32,
    tmp: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    att: []f32,
    gate: []f32,
    up: []f32,
    logits: []f32,
    /// Output rows for `logitsMany`, grown on demand. Sized by the number of rows asked for, not
    /// by the batch size, since a row is vocabulary sized.
    logits_many: []f32 = &.{},

    acts: []vecdot.Act,
    a8_pool: []vecdot.BlockA8,
    q8k_pool: []vecdot.BlockQ8K,
    a8_stride: usize,
    q8k_stride: usize,
    profile: Profile = .{},
    attn_part: attention_mod.Partials = .{},

    pub fn init(
        allocator: std.mem.Allocator,
        cfg: Config,
        n_threads: usize,
        n_batch: usize,
        inv_freq: []const f32,
        rope_mscale: f32,
    ) (std.Thread.SpawnError || error{OutOfMemory})!Cpu {
        var pool = try pool_mod.Pool.init(allocator, n_threads);
        errdefer pool.deinit();

        const nb = n_batch;
        const half = cfg.rope_dim / 2;
        // One angle table per token of a batch.
        const cos = try allocator.alloc(f32, nb * half);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, nb * half);
        errdefer allocator.free(sin);

        const max_in = @max(cfg.dim, @max(cfg.qDim(), cfg.ffn_dim));
        const a8_stride = max_in / 32 + 1;
        const q8k_stride = max_in / 256 + 1;

        var c = Cpu{
            .allocator = allocator,
            .cfg = cfg,
            .pool = pool,
            .n_batch = nb,
            .inv_freq = inv_freq,
            .rope_mscale = rope_mscale,
            .cos = cos,
            .sin = sin,
            .x = undefined,
            .xb = undefined,
            .tmp = undefined,
            .q = undefined,
            .k = undefined,
            .v = undefined,
            .att = undefined,
            .gate = undefined,
            .up = undefined,
            .logits = undefined,
            .acts = undefined,
            .a8_pool = undefined,
            .q8k_pool = undefined,
            .a8_stride = a8_stride,
            .q8k_stride = q8k_stride,
        };

        // Scratch. Allocated one by one so a failure frees exactly what exists.
        var made: usize = 0;
        errdefer c.freeScratch(made);
        c.x = try allocator.alloc(f32, nb * cfg.dim);
        made += 1;
        c.xb = try allocator.alloc(f32, nb * cfg.dim);
        made += 1;
        c.tmp = try allocator.alloc(f32, nb * cfg.dim);
        made += 1;
        c.q = try allocator.alloc(f32, nb * cfg.qDim());
        made += 1;
        c.k = try allocator.alloc(f32, nb * cfg.kvDim());
        made += 1;
        c.v = try allocator.alloc(f32, nb * cfg.kvDim());
        made += 1;
        c.att = try allocator.alloc(f32, nb * cfg.qDim());
        made += 1;
        c.gate = try allocator.alloc(f32, nb * cfg.ffn_dim);
        made += 1;
        c.up = try allocator.alloc(f32, nb * cfg.ffn_dim);
        made += 1;
        c.logits = try allocator.alloc(f32, cfg.vocab);
        made += 1;
        c.acts = try allocator.alloc(vecdot.Act, nb);
        made += 1;
        c.a8_pool = try allocator.alloc(vecdot.BlockA8, nb * a8_stride);
        made += 1;
        c.q8k_pool = try allocator.alloc(vecdot.BlockQ8K, nb * q8k_stride);
        made += 1;
        return c;
    }

    fn freeScratch(self: *Cpu, made: usize) void {
        const a = self.allocator;
        if (made > 0) a.free(self.x);
        if (made > 1) a.free(self.xb);
        if (made > 2) a.free(self.tmp);
        if (made > 3) a.free(self.q);
        if (made > 4) a.free(self.k);
        if (made > 5) a.free(self.v);
        if (made > 6) a.free(self.att);
        if (made > 7) a.free(self.gate);
        if (made > 8) a.free(self.up);
        if (made > 9) a.free(self.logits);
        if (made > 10) a.free(self.acts);
        if (made > 11) a.free(self.a8_pool);
        if (made > 12) a.free(self.q8k_pool);
    }

    pub fn deinit(self: *Cpu) void {
        self.freeScratch(13);
        self.allocator.free(self.cos);
        self.allocator.free(self.sin);
        if (self.logits_many.len != 0) self.allocator.free(self.logits_many);
        self.attn_part.deinit(self.allocator);
        self.pool.deinit();
    }

    /// Points `n` activation slots at consecutive rows of `src` (row length `len`).
    fn initActs(self: *Cpu, src: []const f32, len: usize, n: usize) void {
        for (0..n) |t| {
            self.acts[t] = vecdot.Act.init(
                src[t * len ..][0..len],
                self.a8_pool[t * self.a8_stride ..],
                self.q8k_pool[t * self.q8k_stride ..],
            );
        }
    }

    inline fn tick(self: *const Cpu) i128 {
        if (!self.profile.enabled) return 0;
        return std.Io.Timestamp.now(std.Options.debug_io, .awake).nanoseconds;
    }

    inline fn lap(self: *Cpu, comptime field: []const u8, since: *i128) void {
        if (!self.profile.enabled) return;
        const now = std.Io.Timestamp.now(std.Options.debug_io, .awake).nanoseconds;
        @field(self.profile, field) += now - since.*;
        since.* = now;
    }

    fn prepare(self: *Cpu, n: usize, mats: []const Mat) void {
        var need_a8 = false;
        var need_q8k = false;
        for (mats) |m| switch (vecdot.actKind(m.t)) {
            .a8 => need_a8 = true,
            .q8k => need_q8k = true,
            .f32 => {},
        };
        const k = kernels.get();
        if (need_a8) k.prepare(self.acts[0..n], .a8);
        if (need_q8k) k.prepare(self.acts[0..n], .q8k);
    }

    /// Runs one forward pass over `items`, which may come from different sequences. The hidden
    /// state of item `i` ends up in `x[i]`; read predictions with `logitsFor(i)`. The caller has
    /// already checked batch size, context room and token range.
    pub fn forward(self: *Cpu, w: *const weights_mod.Weights, items: []const Item) Error!void {
        try self.embed(w, items);
        try self.forwardFrom(w, items, 0);
    }

    /// Embeds the tokens of `items` into `x` and makes room in their caches. The first half of `forward`; partial
    /// offload hands `x` to a device for the leading layers and then calls `forwardFrom`.
    pub fn embed(self: *Cpu, w: *const weights_mod.Weights, items: []const Item) Error!void {
        const cfg = self.cfg;
        for (items) |it| try it.kv.ensure(it.pos + 1);
        for (items, 0..) |it, t| {
            const row = w.tok_embd.data[it.token * w.tok_embd.rowBytes() ..][0..w.tok_embd.rowBytes()];
            quant.dequantizeRow(w.tok_embd.t, row, self.x[t * cfg.dim ..][0..cfg.dim]) catch unreachable;
        }
    }

    /// Runs layers `first_layer ..` over the hidden states already in `x`. The caches of `items` must already be
    /// sized (`embed` does that) and must hold layers from `first_layer` on.
    pub fn forwardFrom(self: *Cpu, w: *const weights_mod.Weights, items: []const Item, first_layer: usize) Error!void {
        const n = items.len;
        const cfg = self.cfg;

        const half = cfg.rope_dim / 2;
        for (items, 0..) |it, t| ops.ropeAngles(self.cos[t * half ..][0..half], self.sin[t * half ..][0..half], self.inv_freq, it.pos, self.rope_mscale);

        const qd = cfg.qDim();
        const kd = cfg.kvDim();

        for (w.layers[first_layer..], first_layer..) |*l, li| {
            // Attention block.
            var t0 = self.tick();
            kernels.get().rms_norm_rows(self.xb, self.x, l.attn_norm, cfg.rms_eps, n, cfg.dim);
            self.lap("norm", &t0);
            self.initActs(self.xb, cfg.dim, n);
            self.prepare(n, &.{ l.wq, l.wk, l.wv });
            self.lap("prepare", &t0);
            matmul.runParts(&self.pool, &.{
                .{ .mat = l.wq, .y = self.q, .bias = l.bq },
                .{ .mat = l.wk, .y = self.k, .bias = l.bk },
                .{ .mat = l.wv, .y = self.v, .bias = l.bv },
            }, self.acts[0..n], &.{});
            self.lap("qkv", &t0);

            kernels.get().qk_rope(&.{
                .q = self.q,
                .k = self.k,
                .n = n,
                .q_dim = qd,
                .kv_dim = kd,
                .head_dim = cfg.head_dim,
                .q_norm = l.q_norm,
                .k_norm = l.k_norm,
                .eps = cfg.rms_eps,
                .cos = self.cos,
                .sin = self.sin,
                .half = half,
                .style = cfg.rope_style,
            });

            self.lap("rope", &t0);
            // All keys and values of the batch go in first so a token can see its predecessors.
            for (items, 0..) |it, t| {
                for (0..cfg.n_kv_heads) |g| {
                    it.kv.write(li, g, it.pos, self.k[t * kd + g * cfg.head_dim ..][0..cfg.head_dim], self.v[t * kd + g * cfg.head_dim ..][0..cfg.head_dim]);
                }
            }
            self.lap("kv", &t0);
            self.pool.hintNext(l.wo.data);
            try self.attention(li, items);
            self.lap("attn", &t0);

            self.initActs(self.att, qd, n);
            self.prepare(n, &.{l.wo});
            matmul.runParts(&self.pool, &.{.{ .mat = l.wo, .y = self.tmp }}, self.acts[0..n], l.w_gate.data);
            kernels.get().add_in_place(self.x[0 .. n * cfg.dim], self.tmp[0 .. n * cfg.dim]);
            self.lap("wo", &t0);

            // Feed forward block.
            kernels.get().rms_norm_rows(self.xb, self.x, l.ffn_norm, cfg.rms_eps, n, cfg.dim);
            self.initActs(self.xb, cfg.dim, n);
            self.prepare(n, &.{ l.w_gate, l.w_up });
            matmul.runParts(&self.pool, &.{
                .{ .mat = l.w_gate, .y = self.gate },
                .{ .mat = l.w_up, .y = self.up },
            }, self.acts[0..n], l.w_down.data);
            self.lap("ffn_in", &t0);
            switch (cfg.ffn_act) {
                .silu => kernels.get().swiglu(self.gate[0 .. n * cfg.ffn_dim], self.up[0 .. n * cfg.ffn_dim]),
                .gelu_tanh => ops.geglu(self.gate[0 .. n * cfg.ffn_dim], self.up[0 .. n * cfg.ffn_dim]),
            }
            self.lap("act", &t0);

            self.initActs(self.gate, cfg.ffn_dim, n);
            self.prepare(n, &.{l.w_down});
            const after = if (li + 1 < w.layers.len) w.layers[li + 1].wq.data else w.output.data;
            matmul.runParts(&self.pool, &.{.{ .mat = l.w_down, .y = self.tmp }}, self.acts[0..n], after);
            kernels.get().add_in_place(self.x[0 .. n * cfg.dim], self.tmp[0 .. n * cfg.dim]);
            self.lap("ffn_down", &t0);
        }
    }

    /// Next token logits for the token at index `t` of the last pass. The slice is reused by the next call.
    pub fn logitsFor(self: *Cpu, w: *const weights_mod.Weights, t: usize) []const f32 {
        const cfg = self.cfg;
        kernels.get().rms_norm_rows(self.xb, self.x[t * cfg.dim ..][0..cfg.dim], w.out_norm, cfg.rms_eps, 1, cfg.dim);
        self.initActs(self.xb, cfg.dim, 1);
        self.prepare(1, &.{w.output});
        matmul.run(&self.pool, w.output, self.acts[0..1], self.logits, null);
        return self.logits;
    }

    /// Predictions for several items of the last pass at once. The output matrix is read a
    /// single time for all of them, so serving N sequences costs one sweep of it per step, not
    /// N. Row `k` of the result (vocabulary sized) belongs to `indices[k]`. The caller has checked
    /// `indices.len <= n_batch`.
    pub fn logitsMany(self: *Cpu, w: *const weights_mod.Weights, indices: []const usize) Error![]const f32 {
        const cfg = self.cfg;
        const n = indices.len;
        if (self.logits_many.len < n * cfg.vocab) {
            if (self.logits_many.len != 0) self.allocator.free(self.logits_many);
            self.logits_many = &.{};
            self.logits_many = try self.allocator.alloc(f32, n * cfg.vocab);
        }
        for (indices, 0..) |idx, k| {
            kernels.get().rms_norm_rows(self.xb[k * cfg.dim ..][0..cfg.dim], self.x[idx * cfg.dim ..][0..cfg.dim], w.out_norm, cfg.rms_eps, 1, cfg.dim);
        }
        self.initActs(self.xb, cfg.dim, n);
        self.prepare(n, &.{w.output});
        matmul.run(&self.pool, w.output, self.acts[0..n], self.logits_many, null);
        return self.logits_many[0 .. n * cfg.vocab];
    }

    fn attention(self: *Cpu, layer: usize, items: []const Item) Error!void {
        try attention_mod.run(&self.pool, self.allocator, &self.attn_part, self.cfg, layer, items, self.q, self.att);
    }
};
