//! The table of hot kernels one instruction set level provides.
//!
//! Everything here is called per task or per token vector, never per element, so the indirect
//! call costs nothing measurable. Signatures use slices, pointers and integers only: vector
//! values are passed differently depending on the instruction set, and the table is shared
//! between code built for different ones.

const std = @import("std");
const format = @import("../format.zig");
const pool_mod = @import("../pool.zig");
const vecdot = @import("../quant/vecdot.zig");
const gemm = @import("../quant/gemm.zig");
const config_mod = @import("../engine/config.zig");

/// One matrix against a batch of activations, row at a time (small batches and token generation).
pub const RowsArgs = struct {
    t: format.StorageType,
    data: []const u8,
    row_bytes: usize,
    rows: usize,
    acts: []const vecdot.Act,
    y: []f32,
    bias: ?[]const f32,
};

pub const Api = struct {
    /// Level name, for diagnostics.
    name: []const u8,

    /// y[t * rows + r] for rows [r0, r1).
    rows: *const fn (a: *const RowsArgs, r0: usize, r1: usize) void,
    /// Register tiled batch kernel over tiles [t0, t1) of `tile_rows` rows.
    tiles: *const fn (a: *const gemm.Args, scratch: []align(64) u8, t0: usize, t1: usize) void,
    tile_rows: usize,
    scratch_bytes: *const fn (t: format.StorageType, cols: usize) usize,

    /// Quantizes the activation vectors for the given kind of weight.
    prepare: *const fn (acts: []vecdot.Act, kind: vecdot.ActKind) void,

    /// Attention task for `pool.parallelFor` (ctx is an `attention_kernel.Job`).
    attn_task: pool_mod.TaskFn,

    /// out[t] = rmsnorm(x[t]) * w for `n` vectors of `dim`.
    rms_norm_rows: *const fn (out: []f32, x: []const f32, w: []const f32, eps: f32, n: usize, dim: usize) void,
    /// x[t] += y[t] over `n * dim` values.
    add_in_place: *const fn (x: []f32, y: []const f32) void,
    /// gate = silu(gate) * up
    swiglu: *const fn (gate: []f32, up: []const f32) void,
    /// Per token query/key preparation: optional per head RMS norm, then rotary embedding.
    /// q is [n][q_dim], k is [n][kv_dim]; cos and sin are [n][half].
    qk_rope: *const fn (a: *const QkRopeArgs) void,
};

pub const QkRopeArgs = struct {
    q: []f32,
    k: []f32,
    n: usize,
    q_dim: usize,
    kv_dim: usize,
    head_dim: usize,
    q_norm: ?[]const f32,
    k_norm: ?[]const f32,
    eps: f32,
    cos: []const f32,
    sin: []const f32,
    half: usize,
    style: config_mod.RopeStyle,
};
