//! Decoder only transformer inference.
//!
//! `Model` owns what every executor shares (configuration, the mapped weights, the default KV cache, rope
//! frequencies) and hands each forward pass to an executor: the CPU one in `cpu_backend.zig`, or a device
//! backend when the whole network fits on one. Memory behaviour of the CPU path is described there.

const std = @import("std");
const reader_mod = @import("../reader.zig");
const quant = @import("../quant.zig");
const config_mod = @import("config.zig");
const weights_mod = @import("weights.zig");
const cpu_mod = @import("cpu_backend.zig");
const kv_mod = @import("kv.zig");
const kernels = @import("../kernels.zig");
const cores = @import("../cores.zig");
const gpu_mod = @import("../vk/engine.zig");
const backend_mod = @import("../backend/backend.zig");
const cuda_engine = @import("../cuda/engine.zig");
const rocm_engine = @import("../rocm/engine.zig");
const ops = @import("ops.zig");

pub const Config = config_mod.Config;
pub const Diag = config_mod.Diag;

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
    /// Which target may run the network. Only consulted when `gpu` is not `.off`.
    backend: backend_mod.select.Choice = .auto,
    /// Run exactly this many leading layers on the device and the rest on the CPU (`--gpu-layers`). Null lets the
    /// planner decide: the whole network when it fits, otherwise as many leading layers as do.
    gpu_layers: ?usize = null,
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

pub const Profile = cpu_mod.Profile;

fn cuda_engine_tensor_cores() bool {
    return @import("../cuda.zig").hasTensorCores();
}

pub const Model = struct {
    allocator: std.mem.Allocator,
    cfg: Config,
    weights: weights_mod.Weights,
    n_ctx: usize,
    n_batch: usize,

    /// The cache used by `forward`, for callers with a single conversation.
    kv: KvCache,

    inv_freq: []f32,
    rope_mscale: f32,

    /// The CPU executor. It exists even when a device runs the network, as before this split; creating
    /// it lazily would save its pool and scratch on GPU runs (a follow-up, not done here).
    cpu: cpu_mod.Cpu,
    /// The device backend, when the network runs on one (the CPU path lives in this struct).
    gpu: ?backend_mod.Device = null,
    /// Why the GPU is not in use, when it was asked for and could not be. Owned by the model.
    gpu_note: ?[]const u8 = null,
    /// Whether `gpu_note` is worth telling a user who did not ask for a device: true when a device exists but cannot
    /// take the model (format, memory), false when there simply is none.
    gpu_note_notable: bool = false,
    /// When a device runs only the leading layers (partial offload), how many; 0 when it runs the whole network or
    /// there is no device. The CPU runs the layers after these, the embedding and the output.
    gpu_layers: usize = 0,

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

        const n_ctx = if (opts.n_ctx != 0) opts.n_ctx else @min(cfg.ctx_train, 8192);
        const nb = @max(1, opts.n_batch);

        var kv = try KvCache.init(allocator, cfg, n_ctx);
        errdefer kv.deinit();

        const half = cfg.rope_dim / 2;
        const inv_freq = try allocator.alloc(f32, half);
        errdefer allocator.free(inv_freq);
        ops.ropeInvFreq(inv_freq, cfg.rope_base, cfg.rope_dim, cfg.rope_scaling, w.rope_freqs);
        const rope_mscale = ops.ropeMscale(cfg.rope_scaling);

        var cpu = try cpu_mod.Cpu.init(allocator, cfg, n_threads, nb, inv_freq, rope_mscale);
        errdefer cpu.deinit();

        var m = Model{
            .allocator = allocator,
            .cfg = cfg,
            .weights = w,
            .n_ctx = n_ctx,
            .n_batch = nb,
            .kv = kv,
            .inv_freq = inv_freq,
            .rope_mscale = rope_mscale,
            .cpu = cpu,
        };

        if (opts.gpu != .off) m.tryGpu(opts, diag) catch |e| {
            if (opts.gpu == .on) {
                diag.set("GPU requested but not usable: {s}", .{m.gpu_note orelse @errorName(e)});
                m.setNote(null);
                return error.GpuFailed;
            }
        };
        return m;
    }

    fn setNote(self: *Model, text: ?[]const u8) void {
        if (self.gpu_note) |old| self.allocator.free(old);
        self.gpu_note = if (text) |t| self.allocator.dupe(u8, t) catch null else null;
    }

    /// Plans and starts the device engine. On failure `gpu_note` says why and the model stays on the CPU.
    fn tryGpu(self: *Model, opts: Options, diag: *Diag) !void {
        _ = diag;
        const vopts = gpu_mod.Options{ .n_batch = self.n_batch, .n_ctx = self.n_ctx, .slots = opts.gpu_slots, .device = opts.gpu_device };
        const copts = cuda_engine.Options{ .n_batch = self.n_batch, .n_ctx = self.n_ctx, .slots = opts.gpu_slots };
        const ropts = rocm_engine.Options{ .n_batch = self.n_batch, .n_ctx = self.n_ctx, .slots = opts.gpu_slots };
        // `--gpu-layers 0` is the CPU.
        if (opts.gpu_layers) |n| if (n == 0) {
            self.setNote("0 layers were requested on the device");
            return error.NoDevice;
        };
        // Order of preference, from measurement (benchmarks/results, RTX 3050 laptop, four models): with tensor cores
        // (compute capability 8.0+) CUDA is ahead of Vulkan on prompts and on decode for every format tried, so it goes
        // first there; on anything older, or without CUDA, Vulkan goes first (the CUDA kernels below sm_80 are the
        // portable tiled ones, about 5x slower than Vulkan's on prompts). `--backend` overrides the order.
        const vk_cand = backend_mod.probeVulkan(self.allocator, &self.weights, self.cfg, vopts);
        const cuda_cand = backend_mod.probeCuda(&self.weights, self.cfg, copts);
        // ROCm goes last in `auto`: it has no matrix-core kernels yet, so on a machine that also has a Vulkan driver the
        // Vulkan backend (cooperative matrices) is the better bet; ROCm is chosen there with `--backend rocm`.
        const rocm_cand = backend_mod.probeRocm(&self.weights, self.cfg, ropts);
        const cuda_first = cuda_cand.unusable == null and cuda_engine_tensor_cores();
        const candidates = if (cuda_first)
            [_]backend_mod.select.Candidate{ cuda_cand, vk_cand, rocm_cand }
        else
            [_]backend_mod.select.Candidate{ vk_cand, cuda_cand, rocm_cand };
        const plan = backend_mod.select.plan(.{ .choice = opts.backend, .n_ctx = self.n_ctx, .candidates = &candidates, .force_layers = opts.gpu_layers });

        if (plan.target == .cpu) {
            var buf: [320]u8 = undefined;
            var w: std.Io.Writer = .fixed(&buf);
            if (plan.rejected().len > 0) {
                backend_mod.select.writeReason(plan.rejected()[0], &w) catch {};
            } else {
                w.writeAll("the CPU was requested") catch {};
            }
            self.setNote(w.buffered());
            for (plan.rejected()) |r| {
                if (r.why != .unavailable) self.gpu_note_notable = true;
            }
            return error.NoDevice;
        }

        switch (plan.target) {
            .vulkan => {
                var go = vopts;
                go.n_ctx = plan.n_ctx;
                go.layers = plan.layers;
                const eng = try self.allocator.create(gpu_mod.Engine);
                errdefer self.allocator.destroy(eng);
                eng.* = gpu_mod.Engine.init(self.allocator, &self.weights, self.cfg, self.inv_freq, self.rope_mscale, go) catch |e| {
                    self.gpu_note_notable = true;
                    self.setNote(switch (e) {
                        error.UnsupportedType => "a weight format has no GPU kernel yet",
                        error.UnsupportedShape => "a tensor shape is not supported on the GPU",
                        error.OutOfDeviceMemory => "the model does not fit in device memory",
                        else => "the GPU engine failed to start",
                    });
                    return e;
                };
                self.gpu = .{ .vulkan = eng };
                self.n_ctx = go.n_ctx;
                self.setPartial(plan.layers);
            },
            .cuda => {
                var go = copts;
                go.n_ctx = plan.n_ctx;
                go.layers = plan.layers;
                const eng = try self.allocator.create(cuda_engine.Engine);
                errdefer self.allocator.destroy(eng);
                eng.* = cuda_engine.Engine.init(self.allocator, &self.weights, self.cfg, self.inv_freq, self.rope_mscale, go) catch |e| {
                    self.gpu_note_notable = true;
                    self.setNote(switch (e) {
                        error.UnsupportedType => "a weight format has no CUDA kernel yet",
                        error.UnsupportedShape => "a tensor shape is not supported on CUDA",
                        error.OutOfDeviceMemory => "the model does not fit in device memory",
                        else => "the CUDA engine failed to start",
                    });
                    return e;
                };
                self.gpu = .{ .cuda = eng };
                self.n_ctx = go.n_ctx;
                self.setPartial(plan.layers);
            },
            .rocm => {
                var go = ropts;
                go.n_ctx = plan.n_ctx;
                go.layers = plan.layers;
                const eng = try self.allocator.create(rocm_engine.Engine);
                errdefer self.allocator.destroy(eng);
                eng.* = rocm_engine.Engine.init(self.allocator, &self.weights, self.cfg, self.inv_freq, self.rope_mscale, go) catch |e| {
                    self.gpu_note_notable = true;
                    self.setNote(switch (e) {
                        error.UnsupportedType => "a weight format has no ROCm kernel yet",
                        error.UnsupportedShape => "a tensor shape is not supported on ROCm",
                        error.OutOfDeviceMemory => "the model does not fit in device memory",
                        else => "the ROCm engine failed to start",
                    });
                    return e;
                };
                self.gpu = .{ .rocm = eng };
                self.n_ctx = go.n_ctx;
                self.setPartial(plan.layers);
            },
            .cpu => unreachable,
        }
    }

    /// Records that a device runs `layers` leading layers (0: the whole network). The default cache then holds only the
    /// layers the CPU runs.
    fn setPartial(self: *Model, layers: usize) void {
        self.gpu_layers = layers;
        self.kv.first_layer = layers;
    }

    /// The first layer whose keys and values live in a host `KvCache`: caches made for this model must start there.
    pub fn kvFirstLayer(self: *const Model) usize {
        return self.gpu_layers;
    }

    /// A cache for one more sequence, sized for this model and its device split.
    pub fn newKv(self: *const Model, allocator: std.mem.Allocator, n_ctx: usize) !KvCache {
        return KvCache.initFrom(allocator, self.cfg, @min(n_ctx, self.n_ctx), self.gpu_layers);
    }

    /// Forgets a cache that is going away, so a GPU slot it held can be reused.
    pub fn forget(self: *Model, kv: *const KvCache) void {
        if (self.gpu) |g| g.release(kv);
    }

    pub fn deinit(self: *Model) void {
        if (self.gpu) |g| g.destroy(self.allocator);
        self.setNote(null);
        self.cpu.deinit();
        self.allocator.free(self.inv_freq);
        self.kv.deinit();
        self.weights.deinit();
    }

    fn defaultThreads() usize {
        // Decoding is memory bound, so hyper threads add contention and little throughput.
        return cores.physicalCores(std.Options.debug_io);
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
    /// state of item `i` ends up in the executor's activations; read predictions with `logitsFor(i)`.
    pub fn forwardItems(self: *Model, items: []const Item) Error!void {
        const n = items.len;
        const cfg = self.cfg;
        if (n == 0) return;
        if (n > self.n_batch) return error.BatchTooLarge;
        for (items) |it| {
            if (it.pos >= it.kv.n_ctx) return error.ContextFull;
            if (it.token >= cfg.vocab) return error.TokenOutOfRange;
        }
        if (self.gpu) |g| {
            if (self.gpu_layers != 0) return self.forwardHybrid(g, items);
            g.forward(items) catch |e| return switch (e) {
                error.OutOfMemory => error.OutOfMemory,
                error.NoGpuSlot => error.GpuFailed,
                else => error.GpuFailed,
            };
            return;
        }
        try self.cpu.forward(&self.weights, items);
    }

    /// Partial offload: the CPU embeds, the device runs the leading layers, the CPU runs the rest and (later) the output.
    fn forwardHybrid(self: *Model, g: backend_mod.Device, items: []const Item) Error!void {
        const n = items.len;
        try self.cpu.embed(&self.weights, items);
        g.forwardPartial(items, self.cpu.x[0 .. n * self.cfg.dim]) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.GpuFailed,
        };
        try self.cpu.forwardFrom(&self.weights, items, self.gpu_layers);
    }

    /// Next token logits for the token at index `t` of the last `forward` batch. The slice is
    /// reused by the next call.
    pub fn logitsFor(self: *Model, t: usize) []const f32 {
        if (self.gpu) |g| if (self.gpu_layers == 0) return g.logits(&.{t}) catch @panic("the GPU failed while reading logits");
        return self.cpu.logitsFor(&self.weights, t);
    }

    /// Predictions for several items of the last pass at once. The output matrix is read a
    /// single time for all of them, so serving N sequences costs one sweep of it per step, not
    /// N. Row `k` of the result (vocabulary sized) belongs to `indices[k]`.
    pub fn logitsMany(self: *Model, indices: []const usize) Error![]const f32 {
        const n = indices.len;
        if (n == 0) return &.{};
        if (n > self.n_batch) return error.BatchTooLarge;
        if (self.gpu) |g| if (self.gpu_layers == 0) {
            if (n > g.maxRows()) return error.BatchTooLarge;
            return g.logits(indices) catch error.GpuFailed;
        };
        return self.cpu.logitsMany(&self.weights, indices);
    }
};
