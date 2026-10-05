//! Weight tensors as zero copy views into the memory mapped container.
//!
//! Matrices are never copied or converted: a `Mat` is a slice of the mapped file plus its
//! storage type and shape, so loading a model costs address space, not memory, and pages are
//! only read when a layer is first used. Small vectors (norm weights, biases) are the one
//! exception: they are used as f32 on every token, so when the file stores them in another
//! type they are converted once into a small owned buffer.
//!
//! Every tensor is checked on load against the shape the architecture demands and against
//! the exact byte size its type implies. A truncated or mislabelled tensor fails here, with
//! its name, instead of reading past the end of the file during generation.

const std = @import("std");
const format = @import("../format.zig");
const reader_mod = @import("../reader.zig");
const quant = @import("../quant.zig");
const config_mod = @import("config.zig");

const Config = config_mod.Config;
const Diag = config_mod.Diag;

pub const Mat = struct {
    data: []const u8,
    t: format.StorageType,
    /// Output features.
    rows: usize,
    /// Input features.
    cols: usize,

    pub fn rowBytes(self: Mat) usize {
        const info = quant.info(self.t).?;
        return (self.cols / info.elems) * info.bytes;
    }
};

pub const Layer = struct {
    attn_norm: []const f32,
    wq: Mat,
    wk: Mat,
    wv: Mat,
    wo: Mat,
    bq: ?[]const f32 = null,
    bk: ?[]const f32 = null,
    bv: ?[]const f32 = null,
    q_norm: ?[]const f32 = null,
    k_norm: ?[]const f32 = null,
    ffn_norm: []const f32,
    w_gate: Mat,
    w_up: Mat,
    w_down: Mat,
};

pub const LoadError = error{
    MissingTensor,
    BadTensorShape,
    UnsupportedTensorType,
    BadTensorSize,
    OutOfMemory,
};

pub const Weights = struct {
    allocator: std.mem.Allocator,
    layers: []Layer,
    tok_embd: Mat,
    out_norm: []const f32,
    /// Output projection. Aliases `tok_embd` when the model ties them.
    output: Mat,
    tied_output: bool,
    /// Per-frequency divisors for rotary embeddings, when the file carries them (llama.cpp
    /// files with llama3 rope scaling do).
    rope_freqs: ?[]const f32 = null,
    /// Buffers created by converting non f32 vectors; freed on deinit.
    owned: std.ArrayList([]f32) = .empty,

    pub fn deinit(self: *Weights) void {
        for (self.owned.items) |b| self.allocator.free(b);
        self.owned.deinit(self.allocator);
        self.allocator.free(self.layers);
    }

    /// Reads the token embedding shape without loading anything else. The vocabulary size is
    /// defined by this tensor, and the configuration needs it before the rest can be checked.
    pub fn vocabRows(reader: *const reader_mod.HKReader, diag: *Diag) LoadError!usize {
        const e = reader.toc.find("token_embd.weight") orelse {
            diag.set("tensor 'token_embd.weight' not found", .{});
            return error.MissingTensor;
        };
        if (e.ndim != 2) {
            diag.set("'token_embd.weight' must be 2 dimensional, has {d}", .{e.ndim});
            return error.BadTensorShape;
        }
        return @intCast(e.shape[0]);
    }

    pub fn load(
        allocator: std.mem.Allocator,
        reader: *const reader_mod.HKReader,
        cfg: Config,
        diag: *Diag,
    ) LoadError!Weights {
        var w = Weights{
            .allocator = allocator,
            .layers = &.{},
            .tok_embd = undefined,
            .out_norm = undefined,
            .output = undefined,
            .tied_output = false,
        };
        errdefer {
            for (w.owned.items) |b| allocator.free(b);
            w.owned.deinit(allocator);
        }

        w.tok_embd = try mat(reader, "token_embd.weight", cfg.vocab, cfg.dim, diag);
        w.out_norm = try vec(&w, reader, "output_norm.weight", cfg.dim, diag);
        if (reader.toc.find("rope_freqs.weight") != null) {
            w.rope_freqs = try vec(&w, reader, "rope_freqs.weight", cfg.rope_dim / 2, diag);
            for (w.rope_freqs.?) |f| if (!(f > 0 and std.math.isFinite(f))) {
                diag.set("'rope_freqs.weight' holds a value that is not a positive number", .{});
                return error.BadTensorSize;
            };
        }

        if (reader.toc.find("output.weight") != null) {
            w.output = try mat(reader, "output.weight", cfg.vocab, cfg.dim, diag);
        } else {
            w.output = w.tok_embd;
            w.tied_output = true;
        }

        const layers = try allocator.alloc(Layer, cfg.n_layers);
        errdefer allocator.free(layers);

        var name_buf: [96]u8 = undefined;
        for (layers, 0..) |*l, i| {
            const n = struct {
                fn f(buf: []u8, layer: usize, comptime suffix: []const u8) []const u8 {
                    return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ layer, suffix }) catch unreachable;
                }
            }.f;

            l.attn_norm = try vec(&w, reader, n(&name_buf, i, "attn_norm.weight"), cfg.dim, diag);
            l.wq = try mat(reader, n(&name_buf, i, "attn_q.weight"), cfg.qDim(), cfg.dim, diag);
            l.wk = try mat(reader, n(&name_buf, i, "attn_k.weight"), cfg.kvDim(), cfg.dim, diag);
            l.wv = try mat(reader, n(&name_buf, i, "attn_v.weight"), cfg.kvDim(), cfg.dim, diag);
            l.wo = try mat(reader, n(&name_buf, i, "attn_output.weight"), cfg.dim, cfg.qDim(), diag);
            l.ffn_norm = try vec(&w, reader, n(&name_buf, i, "ffn_norm.weight"), cfg.dim, diag);
            l.w_gate = try mat(reader, n(&name_buf, i, "ffn_gate.weight"), cfg.ffn_dim, cfg.dim, diag);
            l.w_up = try mat(reader, n(&name_buf, i, "ffn_up.weight"), cfg.ffn_dim, cfg.dim, diag);
            l.w_down = try mat(reader, n(&name_buf, i, "ffn_down.weight"), cfg.dim, cfg.ffn_dim, diag);

            l.bq = null;
            l.bk = null;
            l.bv = null;
            l.q_norm = null;
            l.k_norm = null;
            if (cfg.qkv_bias) {
                l.bq = try vec(&w, reader, n(&name_buf, i, "attn_q.bias"), cfg.qDim(), diag);
                l.bk = try vec(&w, reader, n(&name_buf, i, "attn_k.bias"), cfg.kvDim(), diag);
                l.bv = try vec(&w, reader, n(&name_buf, i, "attn_v.bias"), cfg.kvDim(), diag);
            }
            if (cfg.qk_norm) {
                l.q_norm = try vec(&w, reader, n(&name_buf, i, "attn_q_norm.weight"), cfg.head_dim, diag);
                l.k_norm = try vec(&w, reader, n(&name_buf, i, "attn_k_norm.weight"), cfg.head_dim, diag);
            }
        }
        w.layers = layers;
        return w;
    }
};

fn mat(reader: *const reader_mod.HKReader, name: []const u8, rows: usize, cols: usize, diag: *Diag) LoadError!Mat {
    const e = reader.toc.find(name) orelse {
        diag.set("tensor '{s}' not found", .{name});
        return error.MissingTensor;
    };
    if (e.ndim != 2 or e.shape[0] != rows or e.shape[1] != cols) {
        diag.set("tensor '{s}' has shape [{d}, {d}] (ndim {d}), expected [{d}, {d}]", .{ name, e.shape[0], e.shape[1], e.ndim, rows, cols });
        return error.BadTensorShape;
    }
    if (!quant.vecdot.supported(e.storage_type)) {
        diag.set("tensor '{s}' uses storage type '{s}', which this build cannot run", .{ name, @tagName(e.storage_type) });
        return error.UnsupportedTensorType;
    }
    const info = quant.info(e.storage_type).?;
    if (cols % info.elems != 0) {
        diag.set("tensor '{s}' has {d} columns, not a multiple of the {s} block size {d}", .{ name, cols, @tagName(e.storage_type), info.elems });
        return error.BadTensorShape;
    }
    const want: u64 = @as(u64, rows) * (cols / info.elems) * info.bytes;
    if (e.data_size != want) {
        diag.set("tensor '{s}' holds {d} bytes, {s} [{d}, {d}] needs {d}", .{ name, e.data_size, @tagName(e.storage_type), rows, cols, want });
        return error.BadTensorSize;
    }
    const data = reader.getTensorData(e) catch {
        diag.set("tensor '{s}' points outside the file", .{name});
        return error.BadTensorSize;
    };
    return .{ .data = data, .t = e.storage_type, .rows = rows, .cols = cols };
}

/// A 1 dimensional float vector. Returned straight from the mapping when the file stores it
/// as aligned f32, otherwise converted once into an owned buffer.
fn vec(w: *Weights, reader: *const reader_mod.HKReader, name: []const u8, len: usize, diag: *Diag) LoadError![]const f32 {
    const e = reader.toc.find(name) orelse {
        diag.set("tensor '{s}' not found", .{name});
        return error.MissingTensor;
    };
    var n: u64 = 1;
    for (0..e.ndim) |i| n *= e.shape[i];
    if (n != len) {
        diag.set("tensor '{s}' has {d} elements, expected {d}", .{ name, n, len });
        return error.BadTensorShape;
    }
    const data = reader.getTensorData(e) catch {
        diag.set("tensor '{s}' points outside the file", .{name});
        return error.BadTensorSize;
    };
    switch (e.storage_type) {
        .f32 => {
            if (data.len != len * 4) {
                diag.set("tensor '{s}' holds {d} bytes, f32 [{d}] needs {d}", .{ name, data.len, len, len * 4 });
                return error.BadTensorSize;
            }
            if (std.mem.isAligned(@intFromPtr(data.ptr), @alignOf(f32))) {
                return @as([*]const f32, @ptrCast(@alignCast(data.ptr)))[0..len];
            }
        },
        .f16, .bf16 => {
            if (data.len != len * 2) {
                diag.set("tensor '{s}' holds {d} bytes, {s} [{d}] needs {d}", .{ name, data.len, @tagName(e.storage_type), len, len * 2 });
                return error.BadTensorSize;
            }
        },
        else => {
            diag.set("vector tensor '{s}' must be f32, f16 or bf16, found '{s}'", .{ name, @tagName(e.storage_type) });
            return error.UnsupportedTensorType;
        },
    }
    const out = try w.allocator.alloc(f32, len);
    errdefer w.allocator.free(out);
    quant.dequantizeRow(e.storage_type, data, out) catch unreachable;
    try w.owned.append(w.allocator, out);
    return out;
}
