//! Decoder only transformer inference on the CPU.
//!
//! Memory behaviour is the point of this file:
//!   * weights are views into the mapped container, never copied or converted;
//!   * the KV cache is f16 and grows in segments of `seg_len` positions as the context fills,
//!     so resident size follows the real context length rather than `n_ctx`, whatever the
//!     kernel's transparent huge page setting is;
//!   * attention is a single pass with an online softmax, so its scratch is a few hundred
//!     bytes on the stack no matter how long the context is;
//!   * every buffer a forward pass needs is allocated once in `init`.

const std = @import("std");
const reader_mod = @import("../reader.zig");
const pool_mod = @import("../pool.zig");
const quant = @import("../quant.zig");
const config_mod = @import("config.zig");
const weights_mod = @import("weights.zig");
const matmul = @import("matmul.zig");
const kv_mod = @import("kv.zig");
const attention_mod = @import("attention.zig");
const kernels = @import("../kernels.zig");
const cores = @import("../cores.zig");
const gpu_mod = @import("../vk/engine.zig");
const ops = @import("ops.zig");

pub const Config = config_mod.Config;
pub const Diag = config_mod.Diag;
const vecdot = quant.vecdot;
const Mat = weights_mod.Mat;

pub const Options = struct {
    /// Total threads including the caller. 0 picks one per physical core.
    n_threads: usize = 0,
    /// Context window to reserve. 0 uses the smaller of the trained length and 8192.
    n_ctx: usize = 0,
    /// Most tokens processed in one pass. Larger is faster for prompts and uses more scratch.
    n_batch: usize = 256,
    /// Whether to run the network on a GPU (through Vulkan).
    gpu: GpuMode = .off,
    /// Sequences that may use the GPU at once (conversations of a server).
    gpu_slots: usize = 1,
    /// Which Vulkan device, or null for the best one.
    gpu_device: ?usize = null,
};

pub const GpuMode = enum {
    /// CPU only.
    off,
    /// Use a GPU when there is one and the whole model fits; otherwise stay on the CPU quietly.
    auto,
    /// Use a GPU, and fail to load when that is not possible.
    on,
};

pub const Error = error{
    OutOfMemory,
    ContextFull,
    BatchTooLarge,
    TokenOutOfRange,
    GpuFailed,
};

pub const InitError = config_mod.ConfigError || weights_mod.LoadError || std.Thread.SpawnError || error{ OutOfMemory, GpuFailed };

pub const KvCache = kv_mod.KvCache;
pub const Item = kv_mod.Item;

/// Wall time per phase of the forward pass, filled in only when `Model.profile` is set.
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

pub const Model = struct {
    allocator: std.mem.Allocator,
    cfg: Config,
    weights: weights_mod.Weights,
    pool: pool_mod.Pool,
    n_ctx: usize,
    n_batch: usize,

    /// The cache used by `forward`, for callers with a single conversation.
    kv: KvCache,

    inv_freq: []f32,
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
    /// The GPU engine, when the network runs on a device.
    gpu: ?*gpu_mod.Engine = null,
    /// Why the GPU is not in use, when it was asked for and could not be.
    gpu_note: ?[]const u8 = null,

    pub fn init(
        allocator: std.mem.Allocator,
        reader: *const reader_mod.HKReader,
        opts: Options,
        diag: *Diag,
    ) InitError!Model {
        const vocab = try weights_mod.Weights.vocabRows(reader, diag);
        const cfg = try config_mod.fromMetadata(&reader.metadata_map, vocab, diag);

        var w = try weights_mod.Weights.load(allocator, reader, cfg, diag);
        errdefer w.deinit();

        _ = kernels.get(); // pick the instruction set level before any worker runs
        const n_threads = if (opts.n_threads != 0) opts.n_threads else defaultThreads();
        var pool = try pool_mod.Pool.init(allocator, n_threads);
        errdefer pool.deinit();

        const n_ctx = if (opts.n_ctx != 0) opts.n_ctx else @min(cfg.ctx_train, 8192);
        const nb = @max(1, opts.n_batch);

        var kv = try KvCache.init(allocator, cfg, n_ctx);
        errdefer kv.deinit();

        const half = cfg.rope_dim / 2;
        const inv_freq = try allocator.alloc(f32, half);
        errdefer allocator.free(inv_freq);
        ops.ropeInvFreq(inv_freq, cfg.rope_base, cfg.rope_dim, cfg.rope_scaling, w.rope_freqs);
        // One angle table per token of a batch.
        const cos = try allocator.alloc(f32, nb * half);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, nb * half);
        errdefer allocator.free(sin);

        const max_in = @max(cfg.dim, @max(cfg.qDim(), cfg.ffn_dim));
        const a8_stride = max_in / 32 + 1;
        const q8k_stride = max_in / 256 + 1;

        var m = Model{
            .allocator = allocator,
            .cfg = cfg,
            .weights = w,
            .pool = pool,
            .n_ctx = n_ctx,
            .n_batch = nb,
            .kv = kv,
            .inv_freq = inv_freq,
            .rope_mscale = ops.ropeMscale(cfg.rope_scaling),
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
        errdefer m.freeScratch(made);
        m.x = try allocator.alloc(f32, nb * cfg.dim);
        made += 1;
        m.xb = try allocator.alloc(f32, nb * cfg.dim);
        made += 1;
        m.tmp = try allocator.alloc(f32, nb * cfg.dim);
        made += 1;
        m.q = try allocator.alloc(f32, nb * cfg.qDim());
        made += 1;
        m.k = try allocator.alloc(f32, nb * cfg.kvDim());
        made += 1;
        m.v = try allocator.alloc(f32, nb * cfg.kvDim());
        made += 1;
        m.att = try allocator.alloc(f32, nb * cfg.qDim());
        made += 1;
        m.gate = try allocator.alloc(f32, nb * cfg.ffn_dim);
        made += 1;
        m.up = try allocator.alloc(f32, nb * cfg.ffn_dim);
        made += 1;
        m.logits = try allocator.alloc(f32, cfg.vocab);
        made += 1;
        m.acts = try allocator.alloc(vecdot.Act, nb);
        made += 1;
        m.a8_pool = try allocator.alloc(vecdot.BlockA8, nb * a8_stride);
        made += 1;
        m.q8k_pool = try allocator.alloc(vecdot.BlockQ8K, nb * q8k_stride);
        made += 1;

        if (opts.gpu != .off) m.tryGpu(opts, diag) catch |e| {
            if (opts.gpu == .on) {
                diag.set("GPU requested but not usable: {s}", .{m.gpu_note orelse @errorName(e)});
                return error.GpuFailed;
            }
        };
        return m;
    }

    /// Starts the GPU engine. On failure `gpu_note` says why and the model stays on the CPU.
    fn tryGpu(self: *Model, opts: Options, diag: *Diag) !void {
        _ = diag;
        // Reserve a context window that fits next to the weights.
        var ctx = self.n_ctx;
        const free: u64 = blk: {
            var probe = @import("../vk/vk.zig").Context.init(self.allocator, opts.gpu_device) catch |e| {
                self.gpu_note = switch (e) {
                    error.NoVulkan => "no Vulkan driver found",
                    error.NoDevice => "no Vulkan device found",
                    error.MissingFeature => "the Vulkan device lacks a feature the kernels need",
                    else => "the Vulkan device could not be opened",
                };
                return e;
            };
            defer probe.deinit();
            if (probe.caps.is_cpu) {
                self.gpu_note = "only a software Vulkan device exists";
                return error.NoDevice;
            }
            break :blk probe.freeDeviceBytes();
        };
        const gopts = gpu_mod.Options{ .n_batch = self.n_batch, .n_ctx = ctx, .slots = opts.gpu_slots, .device = opts.gpu_device };
        var go = gopts;
        while (gpu_mod.Engine.bytesNeeded(&self.weights, self.cfg, go) > free / 10 * 9 and ctx > 512) {
            ctx /= 2;
            go.n_ctx = ctx;
        }
        if (gpu_mod.Engine.bytesNeeded(&self.weights, self.cfg, go) > free / 10 * 9) {
            self.gpu_note = "the model does not fit in device memory";
            return error.OutOfDeviceMemory;
        }
        const eng = try self.allocator.create(gpu_mod.Engine);
        errdefer self.allocator.destroy(eng);
        eng.* = gpu_mod.Engine.init(self.allocator, &self.weights, self.cfg, self.inv_freq, self.rope_mscale, go) catch |e| {
            self.gpu_note = switch (e) {
                error.UnsupportedType => "a weight format has no GPU kernel yet",
                error.UnsupportedShape => "a tensor shape is not supported on the GPU",
                error.OutOfDeviceMemory => "the model does not fit in device memory",
                else => "the GPU engine failed to start",
            };
            return e;
        };
        self.gpu = eng;
        self.n_ctx = go.n_ctx;
    }

    /// Forgets a cache that is going away, so a GPU slot it held can be reused.
    pub fn forget(self: *Model, kv: *const KvCache) void {
        if (self.gpu) |g| g.release(kv);
    }

    fn freeScratch(self: *Model, made: usize) void {
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

    pub fn deinit(self: *Model) void {
        if (self.gpu) |g| {
            g.deinit();
            self.allocator.destroy(g);
        }
        self.freeScratch(13);
        self.allocator.free(self.cos);
        self.allocator.free(self.sin);
        self.allocator.free(self.inv_freq);
        self.kv.deinit();
        if (self.logits_many.len != 0) self.allocator.free(self.logits_many);
        self.attn_part.deinit(self.allocator);
        self.pool.deinit();
        self.weights.deinit();
    }

    fn defaultThreads() usize {
        // Decoding is memory bound, so hyper threads add contention and little throughput.
        return cores.physicalCores(std.Options.debug_io);
    }

    /// Points `n` activation slots at consecutive rows of `src` (row length `len`).
    fn initActs(self: *Model, src: []const f32, len: usize, n: usize) void {
        for (0..n) |t| {
            self.acts[t] = vecdot.Act.init(
                src[t * len ..][0..len],
                self.a8_pool[t * self.a8_stride ..],
                self.q8k_pool[t * self.q8k_stride ..],
            );
        }
    }

    inline fn tick(self: *const Model) i128 {
        if (!self.profile.enabled) return 0;
        return std.Io.Timestamp.now(std.Options.debug_io, .awake).nanoseconds;
    }

    inline fn lap(self: *Model, comptime field: []const u8, since: *i128) void {
        if (!self.profile.enabled) return;
        const now = std.Io.Timestamp.now(std.Options.debug_io, .awake).nanoseconds;
        @field(self.profile, field) += now - since.*;
        since.* = now;
    }

    fn prepare(self: *Model, n: usize, mats: []const Mat) void {
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

    /// Runs `tokens` through the network at positions `pos .. pos + tokens.len` and leaves the
    /// hidden state of every token in `self.x`. Call `logitsFor` to read out predictions.
    pub fn forward(self: *Model, tokens: []const u32, pos: usize) Error!void {
        const n = tokens.len;
        if (n == 0) return;
        if (n > self.n_batch) return error.BatchTooLarge;
        if (pos + n > self.n_ctx) return error.ContextFull;
        // Scratch for the item list lives on the stack for ordinary batches.
        var small: [64]Item = undefined;
        const items: []Item = if (n <= small.len) small[0..n] else try self.allocator.alloc(Item, n);
        defer if (n > small.len) self.allocator.free(items);
        for (items, tokens, 0..) |*it, tok, t| it.* = .{ .kv = &self.kv, .token = tok, .pos = @intCast(pos + t) };
        try self.forwardItems(items);
    }

    /// Runs one forward pass over `items`, which may come from different sequences. The hidden
    /// state of item `i` ends up in `x[i]`; read predictions with `logitsFor(i)`.
    pub fn forwardItems(self: *Model, items: []const Item) Error!void {
        const n = items.len;
        const cfg = self.cfg;
        if (n == 0) return;
        if (n > self.n_batch) return error.BatchTooLarge;
        for (items) |it| {
            if (it.pos >= it.kv.n_ctx) return error.ContextFull;
            if (it.token >= cfg.vocab) return error.TokenOutOfRange;
            if (self.gpu == null) try it.kv.ensure(it.pos + 1);
        }
        if (self.gpu) |g| {
            g.forward(items) catch |e| return switch (e) {
                error.OutOfMemory => error.OutOfMemory,
                error.NoGpuSlot => error.GpuFailed,
                else => error.GpuFailed,
            };
            return;
        }

        for (items, 0..) |it, t| {
            const row = self.weights.tok_embd.data[it.token * self.weights.tok_embd.rowBytes() ..][0..self.weights.tok_embd.rowBytes()];
            quant.dequantizeRow(self.weights.tok_embd.t, row, self.x[t * cfg.dim ..][0..cfg.dim]) catch unreachable;
        }

        const half = cfg.rope_dim / 2;
        for (items, 0..) |it, t| ops.ropeAngles(self.cos[t * half ..][0..half], self.sin[t * half ..][0..half], self.inv_freq, it.pos, self.rope_mscale);

        const qd = cfg.qDim();
        const kd = cfg.kvDim();

        for (self.weights.layers, 0..) |*l, li| {
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
            kernels.get().swiglu(self.gate[0 .. n * cfg.ffn_dim], self.up[0 .. n * cfg.ffn_dim]);
            self.lap("act", &t0);

            self.initActs(self.gate, cfg.ffn_dim, n);
            self.prepare(n, &.{l.w_down});
            const after = if (li + 1 < self.weights.layers.len) self.weights.layers[li + 1].wq.data else self.weights.output.data;
            matmul.runParts(&self.pool, &.{.{ .mat = l.w_down, .y = self.tmp }}, self.acts[0..n], after);
            kernels.get().add_in_place(self.x[0 .. n * cfg.dim], self.tmp[0 .. n * cfg.dim]);
            self.lap("ffn_down", &t0);
        }
    }

    /// Next token logits for the token at index `t` of the last `forward` batch. The slice is
    /// reused by the next call.
    pub fn logitsFor(self: *Model, t: usize) []const f32 {
        if (self.gpu) |g| return g.logits(&.{t}) catch @panic("the GPU failed while reading logits");
        const cfg = self.cfg;
        kernels.get().rms_norm_rows(self.xb, self.x[t * cfg.dim ..][0..cfg.dim], self.weights.out_norm, cfg.rms_eps, 1, cfg.dim);
        self.initActs(self.xb, cfg.dim, 1);
        self.prepare(1, &.{self.weights.output});
        matmul.run(&self.pool, self.weights.output, self.acts[0..1], self.logits, null);
        return self.logits;
    }

    /// Predictions for several items of the last pass at once. The output matrix is read a
    /// single time for all of them, so serving N sequences costs one sweep of it per step, not
    /// N. Row `k` of the result (vocabulary sized) belongs to `indices[k]`.
    pub fn logitsMany(self: *Model, indices: []const usize) Error![]const f32 {
        const cfg = self.cfg;
        const n = indices.len;
        if (n == 0) return &.{};
        if (n > self.n_batch) return error.BatchTooLarge;
        if (self.gpu) |g| {
            if (n > g.max_rows) return error.BatchTooLarge;
            return g.logits(indices) catch error.GpuFailed;
        }
        if (self.logits_many.len < n * cfg.vocab) {
            if (self.logits_many.len != 0) self.allocator.free(self.logits_many);
            self.logits_many = &.{};
            self.logits_many = try self.allocator.alloc(f32, n * cfg.vocab);
        }
        for (indices, 0..) |idx, k| {
            kernels.get().rms_norm_rows(self.xb[k * cfg.dim ..][0..cfg.dim], self.x[idx * cfg.dim ..][0..cfg.dim], self.weights.out_norm, cfg.rms_eps, 1, cfg.dim);
        }
        self.initActs(self.xb, cfg.dim, n);
        self.prepare(n, &.{self.weights.output});
        matmul.run(&self.pool, self.weights.output, self.acts[0..n], self.logits_many, null);
        return self.logits_many[0 .. n * cfg.vocab];
    }

    fn attention(self: *Model, layer: usize, items: []const Item) Error!void {
        try attention_mod.run(&self.pool, self.allocator, &self.attn_part, self.cfg, layer, items, self.q, self.att);
    }
};
