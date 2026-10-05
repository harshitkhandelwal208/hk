//! Quantized matrix times activation batch, split across the thread pool.
//!
//! Every task owns a contiguous range of weight rows. A row is read from memory once and
//! reused for every token in the batch, so prompt processing costs one pass over the weights
//! per batch instead of one per token. Output layout is token major: `y[t * rows + r]`.

const std = @import("std");
const pool_mod = @import("../pool.zig");
const quant = @import("../quant.zig");
const weights = @import("weights.zig");
const kernels = @import("../kernels.zig");

const vecdot = quant.vecdot;
const Mat = weights.Mat;

/// Batches at least this long use the register tiled kernels, which unpack a tile of weights
/// once and amortize that over the whole batch.
pub var min_tiled_tokens: usize = 8;

/// Below this many weight bytes the cost of waking threads outweighs the work.
const min_parallel_bytes: usize = 96 * 1024;

/// One output of a fused call: y[t * mat.rows + r] = dot(row r of mat, acts[t]) + bias[r].
pub const Part = struct {
    mat: Mat,
    y: []f32,
    bias: ?[]const f32 = null,
};

const max_parts = 4;

const Job = struct {
    parts: [max_parts]Part,
    row_bytes: [max_parts]usize,
    /// First global row (or tile) of each part, plus the total as the last entry.
    starts: [max_parts + 1]usize,
    n_parts: usize,
    acts: []const vecdot.Act,
    pool: *const pool_mod.Pool,
};

fn rowsTask(job: *const Job, p: usize, r0: usize, r1: usize) void {
    const part = &job.parts[p];
    const args = kernels.RowsArgs{
        .t = part.mat.t,
        .data = part.mat.data,
        .row_bytes = job.row_bytes[p],
        .rows = part.mat.rows,
        .acts = job.acts,
        .y = part.y,
        .bias = part.bias,
    };
    kernels.get().rows(&args, r0, r1);
}

/// Splits a global range at part boundaries and hands each piece to `f`.
inline fn forPieces(job: *const Job, start: usize, end: usize, comptime f: anytype) void {
    var g = start;
    var p: usize = 0;
    while (g < end) {
        while (g >= job.starts[p + 1]) p += 1;
        const stop = @min(end, job.starts[p + 1]);
        f(job, p, g - job.starts[p], stop - job.starts[p]);
        g = stop;
    }
}

fn task(ctx: *anyopaque, start: usize, end: usize) void {
    const job: *const Job = @ptrCast(@alignCast(ctx));
    forPieces(job, start, end, rowsTask);
}

fn tilesTask(job: *const Job, p: usize, t0: usize, t1: usize) void {
    const part = &job.parts[p];
    const args = quant.gemm.Args{
        .t = part.mat.t,
        .data = part.mat.data,
        .rows = part.mat.rows,
        .cols = part.mat.cols,
        .row_bytes = job.row_bytes[p],
        .acts = job.acts,
        .y = part.y,
        .bias = part.bias,
    };
    kernels.get().tiles(&args, job.pool.myScratch(), t0, t1);
}

fn tiledTask(ctx: *anyopaque, start: usize, end: usize) void {
    const job: *const Job = @ptrCast(@alignCast(ctx));
    forPieces(job, start, end, tilesTask);
}

/// Computes several matrices against the same activations in one parallel region. Fusing the
/// projections that read the same input (q, k, v; gate, up) removes a synchronization point
/// and its idle tail per extra matrix, which matters when one token's work is microseconds.
pub fn runParts(pool: *pool_mod.Pool, parts: []const Part, acts: []const vecdot.Act, next: []const u8) void {
    pool.hintNext(next);
    std.debug.assert(parts.len >= 1 and parts.len <= max_parts);
    var job = Job{
        .parts = undefined,
        .row_bytes = undefined,
        .starts = undefined,
        .n_parts = parts.len,
        .acts = acts,
        .pool = pool,
    };
    var tiled = acts.len >= min_tiled_tokens;
    var total_bytes: usize = 0;
    var max_row_bytes: usize = 1;
    var scratch: usize = 0;
    for (parts, 0..) |pt, i| {
        std.debug.assert(pt.y.len >= acts.len * pt.mat.rows);
        job.parts[i] = pt;
        job.row_bytes[i] = pt.mat.rowBytes();
        total_bytes += pt.mat.rows * job.row_bytes[i];
        max_row_bytes = @max(max_row_bytes, job.row_bytes[i]);
        if (!quant.gemm.supported(pt.mat.t)) tiled = false;
        scratch = @max(scratch, kernels.get().scratch_bytes(pt.mat.t, pt.mat.cols));
    }
    if (tiled) pool.ensureScratch(scratch) catch {
        tiled = false;
    };

    var acc: usize = 0;
    for (parts, 0..) |pt, i| {
        job.starts[i] = acc;
        acc += if (tiled) (pt.mat.rows + kernels.get().tile_rows - 1) / kernels.get().tile_rows else pt.mat.rows;
    }
    job.starts[parts.len] = acc;
    // Unused entries must never be reached by `forPieces`.
    for (parts.len + 1..max_parts + 1) |i| job.starts[i] = std.math.maxInt(usize);

    if (tiled) {
        pool.parallelFor(acc, 1, &job, tiledTask);
        return;
    }
    if (total_bytes < min_parallel_bytes) {
        task(&job, 0, acc);
        return;
    }
    // Chunks of a few KiB of weights keep the tail short while scheduling stays cheap.
    const grain = @max(1, chunk_bytes / max_row_bytes);
    pool.parallelFor(acc, grain, &job, task);
}

/// Bytes of weights per scheduled chunk when decoding.
const chunk_bytes: usize = 16 * 1024;

/// y[t][r] = dot(row r of mat, acts[t].x) (+ bias[r]). Each act must already be prepared for
/// `vecdot.actKind(mat.t)`.
pub fn run(pool: *pool_mod.Pool, mat: Mat, acts: []const vecdot.Act, y: []f32, bias: ?[]const f32) void {
    runParts(pool, &.{.{ .mat = mat, .y = y, .bias = bias }}, acts, &.{});
}

test "matmul equals row by row dot, with bias and a batch" {
    const allocator = std.testing.allocator;
    var pool = try pool_mod.Pool.init(allocator, 4);
    defer pool.deinit();

    const rows = 300;
    const cols = 256;
    const n_tok = 13; // one 8 tile, one 4 tile, one single

    // f32 weights keep the expected value exact.
    const w = try allocator.alloc(f32, rows * cols);
    defer allocator.free(w);
    var prng = std.Random.DefaultPrng.init(5);
    for (w) |*v| v.* = prng.random().float(f32) - 0.5;

    const x = try allocator.alloc(f32, n_tok * cols);
    defer allocator.free(x);
    for (x) |*v| v.* = prng.random().float(f32) - 0.5;

    const bias = try allocator.alloc(f32, rows);
    defer allocator.free(bias);
    for (bias, 0..) |*b, i| b.* = @floatFromInt(i % 7);

    var acts: [n_tok]vecdot.Act = undefined;
    var a8: [n_tok][cols / 32]vecdot.BlockA8 = undefined;
    var qk: [n_tok][1]vecdot.BlockQ8K = undefined;
    for (&acts, 0..) |*a, t| {
        a.* = vecdot.Act.init(x[t * cols ..][0..cols], &a8[t], &qk[t]);
        a.prepare(.f32);
    }

    const m = Mat{ .data = std.mem.sliceAsBytes(w), .t = .f32, .rows = rows, .cols = cols };
    const y = try allocator.alloc(f32, n_tok * rows);
    defer allocator.free(y);
    run(&pool, m, &acts, y, bias);

    for (0..n_tok) |t| {
        for (0..rows) |r| {
            var want: f32 = bias[r];
            for (0..cols) |c| want += w[r * cols + c] * x[t * cols + c];
            try std.testing.expectApproxEqAbs(want, y[t * rows + r], 1e-3);
        }
    }
}
