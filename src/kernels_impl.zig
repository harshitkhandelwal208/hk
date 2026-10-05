//! Root of the kernel library built once per instruction set level. It exports one function,
//! `hk_kernels_<level>`, that returns the table of kernels compiled for that level (see
//! kernels/api.zig). The level's name comes from the build.

const std = @import("std");
const opts = @import("kernel_options");
const api = @import("kernels/api.zig");
const vecdot = @import("quant/vecdot.zig");
const gemm = @import("quant/gemm.zig");
const isa = @import("quant/isa.zig");
const ops = @import("engine/ops.zig");
const attention_kernel = @import("engine/attention_kernel.zig");

fn rows(a: *const api.RowsArgs, r0: usize, r1: usize) void {
    for (r0..r1) |r| {
        const row = a.data[r * a.row_bytes ..][0..a.row_bytes];
        const b: f32 = if (a.bias) |bs| bs[r] else 0;
        var t: usize = 0;
        // Eight, then four, tokens per pass share each unpacked weight block.
        while (t + 8 <= a.acts.len) : (t += 8) {
            const x = a.acts;
            const o = vecdot.dotRowN(8, a.t, row, .{ &x[t], &x[t + 1], &x[t + 2], &x[t + 3], &x[t + 4], &x[t + 5], &x[t + 6], &x[t + 7] });
            inline for (0..8) |k| a.y[(t + k) * a.rows + r] = o[k] + b;
        }
        while (t + 4 <= a.acts.len) : (t += 4) {
            const o = vecdot.dotRowN(4, a.t, row, .{ &a.acts[t], &a.acts[t + 1], &a.acts[t + 2], &a.acts[t + 3] });
            inline for (0..4) |k| a.y[(t + k) * a.rows + r] = o[k] + b;
        }
        while (t < a.acts.len) : (t += 1) {
            a.y[t * a.rows + r] = vecdot.dotRow(a.t, row, &a.acts[t]) + b;
        }
    }
}

fn tiles(a: *const gemm.Args, scratch: []align(64) u8, t0: usize, t1: usize) void {
    gemm.tiles(a, scratch, t0, t1);
}

fn prepare(acts: []vecdot.Act, kind: vecdot.ActKind) void {
    for (acts) |*a| a.prepare(kind);
}

fn rmsNormRows(out: []f32, x: []const f32, w: []const f32, eps: f32, n: usize, dim: usize) void {
    for (0..n) |t| ops.rmsNorm(out[t * dim ..][0..dim], x[t * dim ..][0..dim], w, eps);
}

fn addInPlace(x: []f32, y: []const f32) void {
    ops.addInPlace(x, y);
}

fn swiglu(gate: []f32, up: []const f32) void {
    ops.swiglu(gate, up);
}

fn qkRope(a: *const api.QkRopeArgs) void {
    for (0..a.n) |t| {
        const qt = a.q[t * a.q_dim ..][0..a.q_dim];
        const kt = a.k[t * a.kv_dim ..][0..a.kv_dim];
        if (a.q_norm) |w| ops.headNorm(qt, w, a.head_dim, a.eps);
        if (a.k_norm) |w| ops.headNorm(kt, w, a.head_dim, a.eps);
        const c = a.cos[t * a.half ..][0..a.half];
        const s = a.sin[t * a.half ..][0..a.half];
        ops.ropeApply(qt, a.head_dim, c, s, a.style);
        ops.ropeApply(kt, a.head_dim, c, s, a.style);
    }
}

const table = api.Api{
    .name = isa.name,
    .rows = rows,
    .tiles = tiles,
    .tile_rows = gemm.tile_rows,
    .scratch_bytes = gemm.scratchBytes,
    .prepare = prepare,
    .attn_task = attention_kernel.task,
    .rms_norm_rows = rmsNormRows,
    .add_in_place = addInPlace,
    .swiglu = swiglu,
    .qk_rope = qkRope,
};

fn getTable() callconv(.c) *const api.Api {
    return &table;
}

comptime {
    @export(&getTable, .{ .name = "hk_kernels_" ++ opts.variant });
}
