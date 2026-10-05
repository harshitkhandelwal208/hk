//! The transformer forward pass on a GPU, through Vulkan compute.
//!
//! Weights are uploaded once, in the byte layout they have in the `.hk` file, and the whole pass
//! (embedding lookup, every layer, the final norm) is recorded into one command buffer and run
//! with a single submission. The key and value caches live on the device too, one region per
//! sequence ("slot"). The CPU sees the logits of the tokens it asks for, nothing else.
//!
//! Supported weight formats are those with a decoder in shaders/common.glsl. A model that uses
//! any other format is refused with `error.UnsupportedType`, and the caller keeps using the CPU.

const std = @import("std");
const vk = @import("vk.zig");
const c = vk.c;
const format = @import("../format.zig");
const config_mod = @import("../engine/config.zig");
const weights_mod = @import("../engine/weights.zig");
const kv_mod = @import("../engine/kv.zig");

const StorageType = format.StorageType;
const Config = config_mod.Config;
const Mat = weights_mod.Mat;

pub const Error = error{ UnsupportedType, UnsupportedShape, NoGpuSlot, OutOfMemory } || vk.Error;

pub const Options = struct {
    /// Most tokens in one forward pass.
    n_batch: usize = 256,
    /// Context length reserved per sequence.
    n_ctx: usize = 2048,
    /// Sequences that can use the device at once.
    slots: usize = 1,
    /// Index of the Vulkan device, or null to pick the best one.
    device: ?usize = null,
};

/// The format name used in shader file names, or null when the GPU cannot decode the format.
fn typeName(t: StorageType) ?[]const u8 {
    return switch (t) {
        .f32 => "f32",
        .f16 => "f16",
        .bf16 => "bf16",
        .q4_0 => "q4_0",
        .q4_1 => "q4_1",
        .q5_0 => "q5_0",
        .q5_1 => "q5_1",
        .q8_0 => "q8_0",
        .iq4_nl => "iq4_nl",
        .q2_k => "q2_k",
        .q3_k => "q3_k",
        .q4_k => "q4_k",
        .q5_k => "q5_k",
        .q6_k => "q6_k",
        .iq4_xs => "iq4_xs",
        else => null,
    };
}

pub fn supports(t: StorageType) bool {
    return typeName(t) != null;
}

fn spirv(comptime kind: []const u8, t: StorageType) ?[]const u8 {
    inline for (.{ "f32", "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "iq4_xs" }) |n| {
        if (std.mem.eql(u8, typeName(t) orelse "", n)) return @embedFile("spv/" ++ kind ++ "_" ++ n ++ ".spv");
    }
    return null;
}

const MmvPush = extern struct { rows: u32, cols: u32, row_bytes: u32, x_stride: u32, y_stride: u32, has_bias: u32, x_off: u32, y_off: u32, lpr: u32, mode: u32 };
const MmPush = extern struct { rows: u32, cols: u32, row_bytes: u32, x_stride: u32, y_stride: u32, has_bias: u32, x_off: u32, y_off: u32, n_tokens: u32, mode: u32 };
const EmbedPush = extern struct { dim: u32, row_bytes: u32 };
const NormPush = extern struct { dim: u32, eps: f32 };
const RopePush = extern struct { n_tokens: u32, n_heads: u32, n_kv: u32, head_dim: u32, half_rot: u32, neox: u32, qk_norm: u32, n_ctx: u32, eps: f32, mscale: f32 };
const AttnPush = extern struct { n_tokens: u32, n_heads: u32, n_kv: u32, head_dim: u32, n_ctx: u32, scale: f32 };
const CountPush = extern struct { n: u32 };

/// A weight matrix on the device.
const DevMat = struct {
    buf: vk.Buffer,
    t: StorageType,
    rows: u32,
    cols: u32,
    row_bytes: u32,
    set: c.VkDescriptorSet = null,
    /// For the interleaved gate/up matrix: the set whose output is the pair buffer.
    set_pair: c.VkDescriptorSet = null,
};

const Layer = struct {
    attn_norm: vk.Buffer,
    ffn_norm: vk.Buffer,
    wq: DevMat,
    wk: DevMat,
    wv: DevMat,
    wo: DevMat,
    /// Gate and up rows interleaved in one matrix when both have the same format; then `up` is
    /// not used and `fused_gu` is set.
    gate: DevMat,
    up: DevMat,
    fused_gu: bool,
    down: DevMat,
    bq: ?vk.Buffer,
    bk: ?vk.Buffer,
    bv: ?vk.Buffer,
    q_norm: ?vk.Buffer,
    k_norm: ?vk.Buffer,
    kc: vk.Buffer,
    vc: vk.Buffer,
    // Descriptor sets of the fixed buffers each step reads and writes.
    s_norm1: c.VkDescriptorSet = null,
    s_rope: c.VkDescriptorSet = null,
    s_attn: c.VkDescriptorSet = null,
    s_add1: c.VkDescriptorSet = null,
    s_norm2: c.VkDescriptorSet = null,
    s_swiglu: c.VkDescriptorSet = null,
    s_swiglu_pair: c.VkDescriptorSet = null,
    s_add2: c.VkDescriptorSet = null,
};

fn nowSecs() f64 {
    return @as(f64, @floatFromInt(std.Io.Timestamp.now(std.Options.debug_io, .awake).nanoseconds)) / 1e9;
}

pub const Kind = enum(u8) { embed, norm, mmv, mm, rope, attn, swiglu, add };

pub const Engine = struct {
    allocator: std.mem.Allocator,
    ctx: vk.Context,
    cfg: Config,
    n_batch: u32,
    n_ctx: u32,
    n_slots: u32,
    layers: []Layer,
    tok_embd: DevMat,
    output: DevMat,
    tied_output: bool,
    out_norm: vk.Buffer,
    inv_freq: vk.Buffer,
    dummy: vk.Buffer,
    mscale: f32,

    // Activations, each n_batch rows.
    x: vk.Buffer,
    xb: vk.Buffer,
    q: vk.Buffer,
    k: vk.Buffer,
    v: vk.Buffer,
    att: vk.Buffer,
    tmp: vk.Buffer,
    gate: vk.Buffer,
    up: vk.Buffer,
    /// Interleaved gate and up results of a batch, before the activation.
    gu: vk.Buffer,
    xn: vk.Buffer,
    /// Positions, kv slots and token ids of the current batch, written by the CPU.
    par: vk.Buffer,
    toks: vk.Buffer,
    /// Logit rows requested at once.
    logits_dev: vk.Buffer,
    logits_host: vk.Buffer,
    max_rows: u32,

    pipes_mmv: [128]?vk.Pipeline = @splat(null),
    pipes_mm: [128]?vk.Pipeline = @splat(null),
    pipes_embed: [128]?vk.Pipeline = @splat(null),
    p_norm: vk.Pipeline,
    p_rope: vk.Pipeline,
    p_attn: vk.Pipeline,
    p_swiglu: vk.Pipeline,
    p_swiglu_pair: vk.Pipeline,
    p_add: vk.Pipeline,

    s_embed: c.VkDescriptorSet = null,
    s_final_norm: c.VkDescriptorSet = null,
    cb: c.VkCommandBuffer,
    n_last: u32 = 0,
    /// Seconds spent recording command buffers and waiting for the device, summed over forward
    /// passes and logits reads; filled in only when HK_GPU_PROFILE is set.
    prof_record: f64 = 0,
    prof_wait: f64 = 0,
    prof_calls: u64 = 0,
    owners: []?*const kv_mod.KvCache,
    /// GPU side timing per kernel kind, enabled by HK_GPU_PROFILE.
    tq: c.VkQueryPool = null,
    tq_n: u32 = 0,
    tq_kinds: [2048]u8 = undefined,
    kind_ns: [8]f64 = @splat(0),

    fn dev(self: *Engine, size: u64) !vk.Buffer {
        return self.ctx.createBuffer(size, .device);
    }

    fn f32Buffer(ctx: *vk.Context, data: []const f32) !vk.Buffer {
        const b = try ctx.createBuffer(data.len * 4, .device);
        try ctx.upload(b, 0, std.mem.sliceAsBytes(data));
        return b;
    }

    /// Bytes per block in the layout the shaders read (see shaders/common.glsl): blocks start on
    /// 4 byte boundaries and the small f16 scales of the legacy formats are widened to f32.
    fn gpuBlockBytes(t: StorageType) u32 {
        return switch (t) {
            .q8_0 => 36,
            .q4_0, .iq4_nl => 20,
            .q4_1, .q5_0 => 24,
            .q5_1 => 28,
            .q6_k => 212,
            .q3_k => 112,
            else => @intCast(@import("../quant.zig").info(t).?.bytes),
        };
    }

    fn f16at(b: []const u8, o: usize) f32 {
        return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, b[o..][0..2], .little))));
    }

    fn put32(dst: []u8, o: usize, v: f32) void {
        std.mem.writeInt(u32, dst[o..][0..4], @bitCast(v), .little);
    }

    /// Converts `n` blocks from the file layout to the GPU layout.
    fn repack(t: StorageType, src: []const u8, dst: []u8, n: usize) void {
        const sb: usize = @import("../quant.zig").info(t).?.bytes;
        const db: usize = gpuBlockBytes(t);
        for (0..n) |i| {
            const s = src[i * sb ..][0..sb];
            const d = dst[i * db ..][0..db];
            switch (t) {
                .q8_0 => {
                    put32(d, 0, f16at(s, 0));
                    @memcpy(d[4..36], s[2..34]);
                },
                .q4_0, .iq4_nl => {
                    put32(d, 0, f16at(s, 0));
                    @memcpy(d[4..20], s[2..18]);
                },
                .q4_1 => {
                    put32(d, 0, f16at(s, 0));
                    put32(d, 4, f16at(s, 2));
                    @memcpy(d[8..24], s[4..20]);
                },
                .q5_0 => {
                    put32(d, 0, f16at(s, 0));
                    @memcpy(d[4..8], s[2..6]);
                    @memcpy(d[8..24], s[6..22]);
                },
                .q5_1 => {
                    put32(d, 0, f16at(s, 0));
                    put32(d, 4, f16at(s, 2));
                    @memcpy(d[8..12], s[4..8]);
                    @memcpy(d[12..28], s[8..24]);
                },
                else => {
                    @memcpy(d[0..sb], s);
                    @memset(d[sb..], 0);
                },
            }
        }
    }

    fn matBuffer(ctx: *vk.Context, m: Mat, allocator: std.mem.Allocator) !DevMat {
        if (!supports(m.t)) return error.UnsupportedType;
        const info = @import("../quant.zig").info(m.t).?;
        if (m.cols % info.elems != 0) return error.UnsupportedShape;
        // The kernels decode eight weights per step for plain floats and a block for the rest.
        if (info.elems == 1 and m.cols % 8 != 0) return error.UnsupportedShape;
        if (m.cols % 32 != 0) return error.UnsupportedShape;

        const per_row_blocks: usize = if (info.elems == 1) m.cols / 8 else m.cols / info.elems;
        // Plain floats are read in groups of eight elements: the block is 8 elements wide.
        const src_block: usize = if (info.elems == 1) 8 * info.bytes else info.bytes;
        const dst_block: usize = if (info.elems == 1) src_block else gpuBlockBytes(m.t);
        const row_bytes = per_row_blocks * dst_block;
        const total = row_bytes * m.rows;
        const b = try ctx.createBuffer(total, .device);
        errdefer ctx.destroyBuffer(b);
        if (dst_block == src_block) {
            try ctx.upload(b, 0, m.data);
        } else {
            const chunk_rows = @max(1, (8 << 20) / row_bytes);
            const tmp = try allocator.alloc(u8, chunk_rows * row_bytes);
            defer allocator.free(tmp);
            const src_row = per_row_blocks * src_block;
            var r: usize = 0;
            while (r < m.rows) : (r += chunk_rows) {
                const n_rows = @min(chunk_rows, m.rows - r);
                repack(m.t, m.data[r * src_row ..][0 .. n_rows * src_row], tmp, n_rows * per_row_blocks);
                try ctx.upload(b, r * row_bytes, tmp[0 .. n_rows * row_bytes]);
            }
        }
        return .{ .buf = b, .t = m.t, .rows = @intCast(m.rows), .cols = @intCast(m.cols), .row_bytes = @intCast(row_bytes) };
    }

    /// Two matrices of the same shape and format stored as one with their rows interleaved
    /// (a0, b0, a1, b1, ...). The decode kernel computes both rows of a pair in one pass.
    fn matBufferPair(ctx: *vk.Context, a: Mat, b2: Mat, allocator: std.mem.Allocator) !DevMat {
        var out = try matBuffer(ctx, a, allocator);
        errdefer ctx.destroyBuffer(out.buf);
        ctx.destroyBuffer(out.buf);
        const info = @import("../quant.zig").info(a.t).?;
        const per_row_blocks: usize = if (info.elems == 1) a.cols / 8 else a.cols / info.elems;
        const src_block: usize = if (info.elems == 1) 8 * info.bytes else info.bytes;
        const dst_block: usize = if (info.elems == 1) src_block else gpuBlockBytes(a.t);
        const row_bytes = per_row_blocks * dst_block;
        const src_row = per_row_blocks * src_block;
        const buf = try ctx.createBuffer(2 * row_bytes * a.rows, .device);
        errdefer ctx.destroyBuffer(buf);
        const chunk_rows = @max(1, (4 << 20) / row_bytes);
        const tg = try allocator.alloc(u8, chunk_rows * row_bytes);
        defer allocator.free(tg);
        const tu = try allocator.alloc(u8, chunk_rows * row_bytes);
        defer allocator.free(tu);
        const mixed = try allocator.alloc(u8, 2 * chunk_rows * row_bytes);
        defer allocator.free(mixed);
        var r: usize = 0;
        while (r < a.rows) : (r += chunk_rows) {
            const n_rows = @min(chunk_rows, a.rows - r);
            if (dst_block == src_block) {
                @memcpy(tg[0 .. n_rows * row_bytes], a.data[r * src_row ..][0 .. n_rows * src_row]);
                @memcpy(tu[0 .. n_rows * row_bytes], b2.data[r * src_row ..][0 .. n_rows * src_row]);
            } else {
                repack(a.t, a.data[r * src_row ..][0 .. n_rows * src_row], tg, n_rows * per_row_blocks);
                repack(a.t, b2.data[r * src_row ..][0 .. n_rows * src_row], tu, n_rows * per_row_blocks);
            }
            for (0..n_rows) |k| {
                @memcpy(mixed[2 * k * row_bytes ..][0..row_bytes], tg[k * row_bytes ..][0..row_bytes]);
                @memcpy(mixed[(2 * k + 1) * row_bytes ..][0..row_bytes], tu[k * row_bytes ..][0..row_bytes]);
            }
            try ctx.upload(buf, 2 * r * row_bytes, mixed[0 .. 2 * n_rows * row_bytes]);
        }
        out.buf = buf;
        out.row_bytes = @intCast(row_bytes);
        out.rows = @intCast(a.rows); // rows of output; the buffer holds twice as many weight rows
        return out;
    }

    /// Bytes of device memory a model needs, so a caller can decide before uploading anything.
    pub fn bytesNeeded(w: *const weights_mod.Weights, cfg: Config, o: Options) u64 {
        var total: u64 = 0;
        for (w.layers) |l| {
            for ([_]Mat{ l.wq, l.wk, l.wv, l.wo, l.w_gate, l.w_up, l.w_down }) |m| total += m.data.len;
        }
        total += w.tok_embd.data.len;
        if (!w.tied_output) total += w.output.data.len;
        const kv_per_layer: u64 = 2 * @as(u64, o.slots) * o.n_ctx * cfg.n_kv_heads * cfg.head_dim * 2;
        total += kv_per_layer * cfg.n_layers;
        const act: u64 = @as(u64, o.n_batch) * (3 * cfg.dim + 2 * cfg.qDim() + 2 * cfg.kvDim() + 2 * cfg.ffn_dim) * 4;
        total += act + (64 << 20);
        return total;
    }

    pub fn init(
        allocator: std.mem.Allocator,
        w: *const weights_mod.Weights,
        cfg: Config,
        inv_freq: []const f32,
        mscale: f32,
        o: Options,
    ) Error!Engine {
        if (cfg.head_dim > 256 or cfg.head_dim % 2 != 0) return error.UnsupportedShape;
        for (w.layers) |l| {
            for ([_]Mat{ l.wq, l.wk, l.wv, l.wo, l.w_gate, l.w_up, l.w_down }) |m| if (!supports(m.t)) return error.UnsupportedType;
        }
        if (!supports(w.tok_embd.t) or !supports(w.output.t)) return error.UnsupportedType;

        var self: Engine = undefined;
        self.allocator = allocator;
        self.ctx = try vk.Context.init(allocator, o.device);
        self.owners = &.{};
        self.layers = &.{};
        errdefer self.deinitPartial();
        const need = bytesNeeded(w, cfg, o);
        if (need > self.ctx.freeDeviceBytes()) return error.OutOfDeviceMemory;

        self.cfg = cfg;
        self.n_batch = @intCast(o.n_batch);
        self.n_ctx = @intCast(o.n_ctx);
        self.n_slots = @intCast(o.slots);
        self.mscale = mscale;
        self.max_rows = @intCast(@max(o.slots, 8));
        self.pipes_mmv = @splat(null);
        self.pipes_mm = @splat(null);
        self.pipes_embed = @splat(null);
        self.n_last = 0;
        self.tq = null;
        self.tq_n = 0;
        self.kind_ns = @splat(0);
        self.s_embed = null;
        self.s_final_norm = null;
        self.owners = try allocator.alloc(?*const kv_mod.KvCache, o.slots);
        @memset(self.owners, null);
        const cx = &self.ctx;

        self.p_norm = try cx.createPipeline(@embedFile("spv/rmsnorm.spv"), 3, @sizeOf(NormPush));
        self.p_rope = try cx.createPipeline(@embedFile("spv/rope_kv.spv"), 9, @sizeOf(RopePush));
        self.p_attn = try cx.createPipeline(@embedFile("spv/attention.spv"), 5, @sizeOf(AttnPush));
        self.p_swiglu = try cx.createPipeline(@embedFile("spv/swiglu.spv"), 2, @sizeOf(CountPush));
        self.p_add = try cx.createPipeline(@embedFile("spv/add.spv"), 2, @sizeOf(CountPush));
        self.p_swiglu_pair = try cx.createPipeline(@embedFile("spv/swiglu_pair.spv"), 2, @sizeOf(CountPush));

        const nb: u64 = self.n_batch;
        self.x = try cx.createBuffer(nb * cfg.dim * 4, .device);
        self.xb = try cx.createBuffer(nb * cfg.dim * 4, .device);
        self.xn = try cx.createBuffer(nb * cfg.dim * 4, .device);
        self.tmp = try cx.createBuffer(nb * cfg.dim * 4, .device);
        self.q = try cx.createBuffer(nb * cfg.qDim() * 4, .device);
        self.att = try cx.createBuffer(nb * cfg.qDim() * 4, .device);
        self.k = try cx.createBuffer(nb * cfg.kvDim() * 4, .device);
        self.v = try cx.createBuffer(nb * cfg.kvDim() * 4, .device);
        self.gate = try cx.createBuffer(nb * cfg.ffn_dim * 4, .device);
        self.up = try cx.createBuffer(nb * cfg.ffn_dim * 4, .device);
        self.gu = try cx.createBuffer(nb * 2 * cfg.ffn_dim * 4, .device);
        self.par = try cx.createBuffer(nb * 8, .upload);
        self.toks = try cx.createBuffer(nb * 4, .upload);
        self.logits_dev = try cx.createBuffer(@as(u64, self.max_rows) * cfg.vocab * 4, .device);
        self.logits_host = try cx.createBuffer(@as(u64, self.max_rows) * cfg.vocab * 4, .readback);
        self.dummy = try cx.createBuffer(64, .device);
        self.out_norm = try f32Buffer(cx, w.out_norm);
        self.inv_freq = try f32Buffer(cx, inv_freq);
        self.cb = cx.xfer_cb;

        self.tok_embd = try matBuffer(cx, w.tok_embd, allocator);
        self.tied_output = w.tied_output;
        self.output = if (w.tied_output) self.tok_embd else try matBuffer(cx, w.output, allocator);

        self.layers = try allocator.alloc(Layer, w.layers.len);
        var made: usize = 0;
        errdefer {
            for (self.layers[0..made]) |l| self.freeLayer(l);
            allocator.free(self.layers);
            self.layers = &.{};
        }
        const kv_bytes: u64 = @as(u64, o.slots) * o.n_ctx * cfg.n_kv_heads * cfg.head_dim * 2;
        for (w.layers, 0..) |l, i| {
            self.layers[i] = .{
                .attn_norm = try f32Buffer(cx, l.attn_norm),
                .ffn_norm = try f32Buffer(cx, l.ffn_norm),
                .wq = try matBuffer(cx, l.wq, allocator),
                .wk = try matBuffer(cx, l.wk, allocator),
                .wv = try matBuffer(cx, l.wv, allocator),
                .wo = try matBuffer(cx, l.wo, allocator),
                .gate = undefined,
                .up = undefined,
                .fused_gu = false,
                .down = try matBuffer(cx, l.w_down, allocator),
                .bq = if (l.bq) |b| try f32Buffer(cx, b) else null,
                .bk = if (l.bk) |b| try f32Buffer(cx, b) else null,
                .bv = if (l.bv) |b| try f32Buffer(cx, b) else null,
                .q_norm = if (l.q_norm) |b| try f32Buffer(cx, b) else null,
                .k_norm = if (l.k_norm) |b| try f32Buffer(cx, b) else null,
                .kc = try cx.createBuffer(kv_bytes, .device),
                .vc = try cx.createBuffer(kv_bytes, .device),
            };
            const dl = &self.layers[i];
            const fuse = dl.wq.rows != 0 and w.layers[i].w_gate.t == w.layers[i].w_up.t and w.layers[i].w_gate.rows == w.layers[i].w_up.rows and w.layers[i].w_gate.cols == w.layers[i].w_up.cols;
            if (fuse) {
                dl.gate = try matBufferPair(cx, w.layers[i].w_gate, w.layers[i].w_up, allocator);
                dl.up = dl.gate;
                dl.fused_gu = true;
            } else {
                dl.gate = try matBuffer(cx, w.layers[i].w_gate, allocator);
                dl.up = try matBuffer(cx, w.layers[i].w_up, allocator);
            }
            made += 1;
        }

        try self.buildSets();
        return self;
    }

    fn freeLayer(self: *Engine, l: Layer) void {
        const cx = &self.ctx;
        cx.destroyBuffer(l.attn_norm);
        cx.destroyBuffer(l.ffn_norm);
        for ([_]DevMat{ l.wq, l.wk, l.wv, l.wo, l.gate, l.down }) |m| cx.destroyBuffer(m.buf);
        if (!l.fused_gu) cx.destroyBuffer(l.up.buf);
        for ([_]?vk.Buffer{ l.bq, l.bk, l.bv, l.q_norm, l.k_norm }) |b| if (b) |bb| cx.destroyBuffer(bb);
        cx.destroyBuffer(l.kc);
        cx.destroyBuffer(l.vc);
    }

    /// Releases what `init` created when it fails halfway.
    fn deinitPartial(self: *Engine) void {
        self.ctx.deinit();
        if (self.owners.len != 0) self.allocator.free(self.owners);
    }

    pub fn deinit(self: *Engine) void {
        _ = self.ctx.f.deviceWaitIdle(self.ctx.device);
        for (self.layers) |l| self.freeLayer(l);
        self.allocator.free(self.layers);
        const cx = &self.ctx;
        for ([_]vk.Buffer{ self.x, self.xb, self.xn, self.tmp, self.q, self.att, self.k, self.v, self.gate, self.up, self.gu, self.par, self.toks, self.logits_dev, self.logits_host, self.dummy, self.out_norm, self.inv_freq }) |b| cx.destroyBuffer(b);
        cx.destroyBuffer(self.tok_embd.buf);
        if (!self.tied_output) cx.destroyBuffer(self.output.buf);
        for (self.pipes_mmv) |p| if (p) |pp| cx.destroyPipeline(pp);
        for (self.pipes_mm) |p| if (p) |pp| cx.destroyPipeline(pp);
        for (self.pipes_embed) |p| if (p) |pp| cx.destroyPipeline(pp);
        for ([_]vk.Pipeline{ self.p_norm, self.p_rope, self.p_attn, self.p_swiglu, self.p_swiglu_pair, self.p_add }) |p| cx.destroyPipeline(p);
        self.allocator.free(self.owners);
        self.ctx.deinit();
    }

    pub fn deviceName(self: *const Engine) []const u8 {
        return self.ctx.caps.deviceName();
    }

    fn mmvPipe(self: *Engine, t: StorageType) !vk.Pipeline {
        const i = @intFromEnum(t);
        if (self.pipes_mmv[i]) |p| return p;
        const p = try self.ctx.createPipeline(spirv("mmv", t) orelse return error.UnsupportedType, 4, @sizeOf(MmvPush));
        self.pipes_mmv[i] = p;
        return p;
    }

    fn mmPipe(self: *Engine, t: StorageType) !vk.Pipeline {
        const i = @intFromEnum(t);
        if (self.pipes_mm[i]) |p| return p;
        const p = try self.ctx.createPipeline(spirv("mm", t) orelse return error.UnsupportedType, 4, @sizeOf(MmPush));
        self.pipes_mm[i] = p;
        return p;
    }

    fn embedPipe(self: *Engine, t: StorageType) !vk.Pipeline {
        const i = @intFromEnum(t);
        if (self.pipes_embed[i]) |p| return p;
        const p = try self.ctx.createPipeline(spirv("embed", t) orelse return error.UnsupportedType, 3, @sizeOf(EmbedPush));
        self.pipes_embed[i] = p;
        return p;
    }

    fn matSet(self: *Engine, m: *DevMat, input: vk.Buffer, output: vk.Buffer, bias: ?vk.Buffer) !void {
        const p = try self.mmvPipe(m.t);
        _ = try self.mmPipe(m.t);
        m.set = try self.ctx.makeSet(p, &.{ .{ .buf = m.buf }, .{ .buf = input }, .{ .buf = output }, .{ .buf = bias orelse self.dummy } });
    }

    fn buildSets(self: *Engine) !void {
        const cx = &self.ctx;
        const ep = try self.embedPipe(self.tok_embd.t);
        self.s_embed = try cx.makeSet(ep, &.{ .{ .buf = self.tok_embd.buf }, .{ .buf = self.toks }, .{ .buf = self.x } });
        self.s_final_norm = try cx.makeSet(self.p_norm, &.{ .{ .buf = self.x }, .{ .buf = self.out_norm }, .{ .buf = self.xn } });
        try self.matSet(&self.output, self.xn, self.logits_dev, null);
        for (self.layers) |*l| {
            l.s_norm1 = try cx.makeSet(self.p_norm, &.{ .{ .buf = self.x }, .{ .buf = l.attn_norm }, .{ .buf = self.xb } });
            try self.matSet(&l.wq, self.xb, self.q, l.bq);
            try self.matSet(&l.wk, self.xb, self.k, l.bk);
            try self.matSet(&l.wv, self.xb, self.v, l.bv);
            l.s_rope = try cx.makeSet(self.p_rope, &.{
                .{ .buf = self.q },     .{ .buf = self.k },                 .{ .buf = self.v },
                .{ .buf = self.par },   .{ .buf = self.inv_freq },          .{ .buf = l.q_norm orelse self.dummy },
                .{ .buf = l.k_norm orelse self.dummy }, .{ .buf = l.kc }, .{ .buf = l.vc },
            });
            l.s_attn = try cx.makeSet(self.p_attn, &.{ .{ .buf = self.q }, .{ .buf = self.par }, .{ .buf = l.kc }, .{ .buf = l.vc }, .{ .buf = self.att } });
            try self.matSet(&l.wo, self.att, self.x, null);
            l.s_norm2 = try cx.makeSet(self.p_norm, &.{ .{ .buf = self.x }, .{ .buf = l.ffn_norm }, .{ .buf = self.xb } });
            if (l.fused_gu) {
                // Decode writes the activated values straight into `gate`; a batch writes the
                // interleaved pairs into `gu` and a second kernel applies the activation.
                try self.matSet(&l.gate, self.xb, self.gate, null);
                const p = try self.mmvPipe(l.gate.t);
                l.gate.set_pair = try cx.makeSet(p, &.{ .{ .buf = l.gate.buf }, .{ .buf = self.xb }, .{ .buf = self.gu }, .{ .buf = self.dummy } });
                l.s_swiglu_pair = try cx.makeSet(self.p_swiglu_pair, &.{ .{ .buf = self.gu }, .{ .buf = self.gate } });
            } else {
                try self.matSet(&l.gate, self.xb, self.gate, null);
                try self.matSet(&l.up, self.xb, self.up, null);
                l.s_swiglu = try cx.makeSet(self.p_swiglu, &.{ .{ .buf = self.gate }, .{ .buf = self.up } });
            }
            try self.matSet(&l.down, self.gate, self.x, null);
        }
    }

    /// The device slot of a sequence's cache, assigning one on first use.
    pub fn slotFor(self: *Engine, kv: *const kv_mod.KvCache) Error!u32 {
        for (self.owners, 0..) |o, i| if (o == kv) return @intCast(i);
        for (self.owners, 0..) |o, i| {
            if (o == null) {
                self.owners[i] = kv;
                return @intCast(i);
            }
        }
        return error.NoGpuSlot;
    }

    /// Frees the slot of a cache that is going away.
    pub fn release(self: *Engine, kv: *const kv_mod.KvCache) void {
        for (self.owners) |*o| if (o.* == kv) {
            o.* = null;
        };
    }

    /// Records a dispatch, and a timestamp after it when profiling.
    fn disp(self: *Engine, kind: Kind, p: vk.Pipeline, set: c.VkDescriptorSet, push: []const u8, x: u32, y: u32, z: u32) void {
        self.ctx.dispatch(self.cb, p, set, push, x, y, z);
        if (self.tq != null and self.tq_n < self.tq_kinds.len) {
            self.ctx.f.cmdWriteTimestamp(self.cb, c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, self.tq, self.tq_n + 1);
            self.tq_kinds[self.tq_n] = @intFromEnum(kind);
            self.tq_n += 1;
        }
    }

    fn barrier(self: *Engine) void {
        self.ctx.barrier(self.cb);
    }

    /// Lanes that share one row in the decode kernel: enough units of work per lane to hide
    /// latency, few enough that every lane has some.
    fn lanesPerRow(m: *const DevMat) u32 {
        const info = @import("../quant.zig").info(m.t).?;
        const elems: u32 = if (info.elems == 1) 8 else @intCast(info.elems);
        const parts: u32 = switch (m.t) {
            .q2_k, .q3_k, .q4_k, .q5_k => 4,
            .q6_k, .iq4_xs => 8,
            else => 1,
        };
        const units = (m.cols / elems) * parts;
        var lpr: u32 = 4;
        while (lpr < 32 and lpr * 8 <= units) lpr *= 2;
        return lpr;
    }

    fn mat(self: *Engine, m: *const DevMat, n: u32, x_off: u32, y_off: u32, has_bias: bool, mode: u32) void {
        if (n <= 4) {
            const p = self.pipes_mmv[@intFromEnum(m.t)].?;
            const lpr = lanesPerRow(m);
            const push = MmvPush{ .rows = m.rows, .cols = m.cols, .row_bytes = m.row_bytes, .x_stride = m.cols, .y_stride = m.rows, .has_bias = @intFromBool(has_bias), .x_off = x_off, .y_off = y_off, .lpr = lpr, .mode = mode };
            self.disp(.mmv, p, m.set, std.mem.asBytes(&push), (m.rows + (128 / lpr) - 1) / (128 / lpr), n, 1);
        } else {
            const p = self.pipes_mm[@intFromEnum(m.t)].?;
            const push = MmPush{ .rows = m.rows, .cols = m.cols, .row_bytes = m.row_bytes, .x_stride = m.cols, .y_stride = m.rows, .has_bias = @intFromBool(has_bias), .x_off = x_off, .y_off = y_off, .n_tokens = n, .mode = mode };
            self.disp(.mm, p, m.set, std.mem.asBytes(&push), (m.rows + 63) / 64, (n + 63) / 64, 1);
        }
    }

    /// Gate and up projections of a layer with the interleaved matrix.
    fn gateUp(self: *Engine, l: *const Layer, n: u32) void {
        const m = &l.gate;
        if (n <= 4) {
            const lpr = lanesPerRow(m);
            const p = self.pipes_mmv[@intFromEnum(m.t)].?;
            const push = MmvPush{ .rows = m.rows, .cols = m.cols, .row_bytes = m.row_bytes, .x_stride = m.cols, .y_stride = m.rows, .has_bias = 0, .x_off = 0, .y_off = 0, .lpr = lpr, .mode = 2 };
            self.disp(.mmv, p, m.set, std.mem.asBytes(&push), (m.rows + (128 / lpr) - 1) / (128 / lpr), n, 1);
        } else {
            const p = self.pipes_mm[@intFromEnum(m.t)].?;
            const push = MmPush{ .rows = 2 * m.rows, .cols = m.cols, .row_bytes = m.row_bytes, .x_stride = m.cols, .y_stride = 2 * m.rows, .has_bias = 0, .x_off = 0, .y_off = 0, .n_tokens = n, .mode = 0 };
            self.disp(.mm, p, m.set_pair, std.mem.asBytes(&push), (2 * m.rows + 63) / 64, (n + 63) / 64, 1);
            self.barrier();
            const count = CountPush{ .n = n * m.rows };
            self.disp(.swiglu, self.p_swiglu_pair, l.s_swiglu_pair, std.mem.asBytes(&count), (count.n + 255) / 256, 1, 1);
        }
    }

    /// Runs the whole network on `n` tokens. The items name the sequence (cache), token and
    /// position of each; items of one cache must be consecutive positions in increasing order.
    pub fn forward(self: *Engine, items: []const kv_mod.Item) Error!void {
        const n: u32 = @intCast(items.len);
        std.debug.assert(n >= 1 and n <= self.n_batch);
        const cfg = self.cfg;
        const par: [*]u32 = @ptrCast(@alignCast(self.par.mapped.?));
        const toks: [*]u32 = @ptrCast(@alignCast(self.toks.mapped.?));
        for (items, 0..) |it, t| {
            if (it.pos >= self.n_ctx) return error.OutOfMemory;
            par[t] = it.pos;
            par[n + t] = try self.slotFor(it.kv);
            toks[t] = it.token;
        }

        const cx = &self.ctx;
        const cb = self.cb;
        const t_start = nowSecs();
        try cx.begin(cb, true);
        if (self.tq != null) {
            cx.f.cmdResetQueryPool(cb, self.tq, 0, @intCast(self.tq_kinds.len + 1));
            cx.f.cmdWriteTimestamp(cb, c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, self.tq, 0);
            self.tq_n = 0;
        }

        const ep = self.pipes_embed[@intFromEnum(self.tok_embd.t)].?;
        const eb = EmbedPush{ .dim = @intCast(cfg.dim), .row_bytes = self.tok_embd.row_bytes };
        self.disp(.embed, ep, self.s_embed, std.mem.asBytes(&eb), (@as(u32, @intCast(cfg.dim)) + 127) / 128, n, 1);
        self.barrier();

        const norm = NormPush{ .dim = @intCast(cfg.dim), .eps = cfg.rms_eps };
        const rope = RopePush{
            .n_tokens = n,
            .n_heads = @intCast(cfg.n_heads),
            .n_kv = @intCast(cfg.n_kv_heads),
            .head_dim = @intCast(cfg.head_dim),
            .half_rot = @intCast(cfg.rope_dim / 2),
            .neox = @intFromBool(cfg.rope_style == .neox),
            .qk_norm = @intFromBool(cfg.qk_norm),
            .n_ctx = self.n_ctx,
            .eps = cfg.rms_eps,
            .mscale = self.mscale,
        };
        const attn = AttnPush{ .n_tokens = n, .n_heads = @intCast(cfg.n_heads), .n_kv = @intCast(cfg.n_kv_heads), .head_dim = @intCast(cfg.head_dim), .n_ctx = self.n_ctx, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim))) };
        const count_dim = CountPush{ .n = n * @as(u32, @intCast(cfg.dim)) };
        const count_ffn = CountPush{ .n = n * @as(u32, @intCast(cfg.ffn_dim)) };

        for (self.layers) |*l| {
            self.disp(.norm, self.p_norm, l.s_norm1, std.mem.asBytes(&norm), n, 1, 1);
            self.barrier();
            // The three projections read the same input and write different buffers.
            self.mat(&l.wq, n, 0, 0, l.bq != null, 0);
            self.mat(&l.wk, n, 0, 0, l.bk != null, 0);
            self.mat(&l.wv, n, 0, 0, l.bv != null, 0);
            self.barrier();
            self.disp(.rope, self.p_rope, l.s_rope, std.mem.asBytes(&rope), @intCast(cfg.n_heads + 2 * cfg.n_kv_heads), n, 1);
            self.barrier();
            self.disp(.attn, self.p_attn, l.s_attn, std.mem.asBytes(&attn), @intCast(cfg.n_heads), n, 1);
            self.barrier();
            // The output projection adds its result to the residual stream itself.
            self.mat(&l.wo, n, 0, 0, false, 1);
            self.barrier();
            self.disp(.norm, self.p_norm, l.s_norm2, std.mem.asBytes(&norm), n, 1, 1);
            self.barrier();
            if (l.fused_gu) {
                self.gateUp(l, n);
            } else {
                self.mat(&l.gate, n, 0, 0, false, 0);
                self.mat(&l.up, n, 0, 0, false, 0);
                self.barrier();
                self.disp(.swiglu, self.p_swiglu, l.s_swiglu, std.mem.asBytes(&count_ffn), (count_ffn.n + 255) / 256, 1, 1);
            }
            self.barrier();
            self.mat(&l.down, n, 0, 0, false, 1);
            self.barrier();
        }
        _ = count_dim;

        // The final norm, for every row, so any of them can be asked for afterwards.
        self.disp(.norm, self.p_norm, self.s_final_norm, std.mem.asBytes(&norm), n, 1, 1);
        try cx.end(cb);
        const t_rec = nowSecs();
        try cx.submitAndWait(cb);
        self.prof_record += t_rec - t_start;
        self.prof_wait += nowSecs() - t_rec;
        self.prof_calls += 1;
        self.n_last = n;
        if (self.tq != null) self.collectTimestamps();
    }

    fn collectTimestamps(self: *Engine) void {
        var res: [2049]u64 = undefined;
        const n = self.tq_n + 1;
        _ = self.ctx.f.getQueryPoolResults(self.ctx.device, self.tq, 0, n, n * 8, &res, 8, c.VK_QUERY_RESULT_64_BIT | c.VK_QUERY_RESULT_WAIT_BIT);
        const period: f64 = self.ctx.caps.timestamp_period;
        for (0..self.tq_n) |i| {
            const d = @as(f64, @floatFromInt(res[i + 1] - res[i])) * period;
            self.kind_ns[self.tq_kinds[i]] += d;
        }
    }

    /// Enables per kernel kind timing on the device (see `kind_ns`).
    pub fn enableProfiling(self: *Engine) !void {
        self.tq = try self.ctx.createTimestampPool(@intCast(self.tq_kinds.len + 1));
    }

    /// Logits of the given tokens of the last pass, one vocabulary sized row each, in the order
    /// of `indices`. The slice stays valid until the next call.
    pub fn logits(self: *Engine, indices: []const usize) Error![]const f32 {
        std.debug.assert(indices.len >= 1 and indices.len <= self.max_rows);
        const cfg = self.cfg;
        const cx = &self.ctx;
        const cb = self.cb;
        try cx.begin(cb, true);
        for (indices, 0..) |idx, k| {
            std.debug.assert(idx < self.n_last);
            self.mat(&self.output, 1, @intCast(idx * cfg.dim), @intCast(k * cfg.vocab), false, 0);
            self.barrier();
        }
        cx.copy(cb, self.logits_dev, self.logits_host, 0, 0, @as(u64, indices.len) * cfg.vocab * 4);
        try cx.end(cb);
        const t0 = nowSecs();
        try cx.submitAndWait(cb);
        self.prof_wait += nowSecs() - t0;
        const out: [*]const f32 = @ptrCast(@alignCast(self.logits_host.mapped.?));
        return out[0 .. indices.len * cfg.vocab];
    }
};
