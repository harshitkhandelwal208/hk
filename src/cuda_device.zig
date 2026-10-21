//! Device code for the GPU backends, written once in Zig. Compiled for nvptx64 (NVIDIA: `build.zig` turns it into PTX
//! that the host embeds and the driver JIT-compiles for whatever GPU is present, so building HK needs no CUDA toolkit)
//! and for amdgcn (AMD RDNA: a code object the HIP runtime loads). Never imported by host code. The per-architecture
//! primitives (thread ids, barrier, shuffles, dp4a) live in `gpu_prims.zig`; the tensor core kernels exist for NVIDIA only.
//!
//! Weight formats are not re-implemented here. A matrix-vector kernel is instantiated per storage type at comptime
//! and calls the same exact block decoders the CPU engine is tested against (`quant/dequant.zig`), so a format that
//! the CPU decodes correctly decodes identically on the device. That is correctness first (plan §152); the kernels
//! are deliberately simple (one block of threads per output row) and are optimized only after they are right.

const std = @import("std");
const builtin = @import("builtin");
const format = @import("format.zig");
const blocks = @import("quant/blocks.zig");
const dequant = @import("quant/dequant.zig");

pub const panic = std.debug.no_panic;

// -- thread intrinsics -------------------------------------------------------------------------------------------

const gpu = @import("gpu_prims.zig");
const tid = gpu.tid;
const bid = gpu.bid;
const bidY = gpu.bidY;
const bdim = gpu.bdim;
const barSync = gpu.barSync;
const shflDown = gpu.shflDown;
const shflXor = gpu.shflXor;
const shflIdx = gpu.shflIdx;
const roundToInt = gpu.roundToInt;
const dp4a = gpu.dp4a;

/// One slot per warp, so up to 1024 threads.
var reduce_scratch: [32]f32 addrspace(.shared) = undefined;

/// Sum of `v` over every thread of the block. All threads get the result.
inline fn blockSum(v: f32) f32 {
    var s = v;
    var off: u32 = 16;
    while (off > 0) : (off >>= 1) s += shflDown(s, off);
    const lane = tid() & 31;
    const warp = tid() >> 5;
    if (lane == 0) reduce_scratch[warp] = s;
    barSync();
    var r: f32 = 0;
    if (warp == 0) {
        r = if (lane < ((bdim() + 31) >> 5)) reduce_scratch[lane] else 0;
        off = 16;
        while (off > 0) : (off >>= 1) r += shflDown(r, off);
        if (lane == 0) reduce_scratch[0] = r;
    }
    barSync();
    const total = reduce_scratch[0];
    barSync(); // the scratch may be reused by the next reduction
    return total;
}

/// Sum of `v` over the 32 lanes of a warp; every lane gets the result.
inline fn warpSum(v: f32) f32 {
    var s = v;
    var off: u32 = 16;
    while (off > 0) : (off >>= 1) s += shflDown(s, off);
    // shfl.down leaves the total in lane 0 only; broadcast it.
    return shflIdx(s, 0);
}

/// Maximum of `v` over every thread of the block. All threads get the result.
inline fn blockMax(v: f32) f32 {
    var m = v;
    var off: u32 = 16;
    while (off > 0) : (off >>= 1) m = @max(m, shflDown(m, off));
    const lane = tid() & 31;
    const warp = tid() >> 5;
    if (lane == 0) reduce_scratch[warp] = m;
    barSync();
    var r: f32 = -std.math.inf(f32);
    if (warp == 0) {
        r = if (lane < ((bdim() + 31) >> 5)) reduce_scratch[lane] else -std.math.inf(f32);
        off = 16;
        while (off > 0) : (off >>= 1) r = @max(r, shflDown(r, off));
        if (lane == 0) reduce_scratch[0] = r;
    }
    barSync();
    const total = reduce_scratch[0];
    barSync();
    return total;
}

/// e^x through the hardware 2^x instruction (NVPTX has no libm). Relative error is a few ulp, the same order as the
/// polynomial the CPU engine uses.
inline fn expf(x: f32) f32 {
    const y = x * 1.4426950408889634;
    return gpu.exp2(y);
}

inline fn globalId() u32 {
    return bid() * bdim() + tid();
}

// -- matrix product ---------------------------------------------------------------------------------------------

/// Decodes one block of `t` into `out`. Dense float types have one element per "block".
inline fn decodeBlock(comptime t: format.StorageType, src: []const u8, out: *[blocks.info(t).?.elems]f32) void {
    switch (t) {
        .f32 => out[0] = @bitCast(std.mem.readInt(u32, src[0..4], .little)),
        .f16 => out[0] = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, src[0..2], .little)))),
        .bf16 => out[0] = @bitCast(@as(u32, std.mem.readInt(u16, src[0..2], .little)) << 16),
        .q4_0 => dequant.q4_0(src, out),
        .q4_1 => dequant.q4_1(src, out),
        .q5_0 => dequant.q5_0(src, out),
        .q5_1 => dequant.q5_1(src, out),
        .q8_0 => dequant.q8_0(src, out),
        .iq4_nl => dequant.iq4_nl(src, out),
        .mxfp4 => dequant.mxfp4(src, out),
        .nvfp4 => dequant.nvfp4(src, out),
        .q2_k => dequant.q2_k(src, out),
        .q3_k => dequant.q3_k(src, out),
        .q4_k => dequant.q4_k(src, out),
        .q5_k => dequant.q5_k(src, out),
        .q6_k => dequant.q6_k(src, out),
        .iq2_xxs => dequant.iq2_xxs(src, out),
        .iq2_xs => dequant.iq2_xs(src, out),
        .iq2_s => dequant.iq2_s(src, out),
        .iq3_xxs => dequant.iq3_xxs(src, out),
        .iq3_s => dequant.iq3_s(src, out),
        .iq1_s => dequant.iq1_s(src, out),
        .iq1_m => dequant.iq1_m(src, out),
        .iq4_xs => dequant.iq4_xs(src, out),
        .tq1_0 => dequant.tq1_0(src, out),
        .tq2_0 => dequant.tq2_0(src, out),
        else => @compileError("no device decoder for " ++ @tagName(t)),
    }
}

/// Rows one thread block of the decode kernel computes: one warp per row.
const gemv_rows = 4;
/// Activation elements staged in shared memory per step of the decode kernel.
const gemv_chunk = 1024;
/// The staged chunk is padded by one float per 32 so that 32 lanes reading 32 consecutive blocks hit 32 different
/// banks instead of one.
var gemv_x: [gemv_chunk + gemv_chunk / 32]f32 addrspace(.shared) = undefined;

/// Single token matrix-vector product, the decode kernel. A warp owns one output row: its lanes take the row's weight
/// blocks in turn, decode them, and multiply against the activations from shared memory (staged once per block of
/// `gemv_rows` rows). The warp reduces with shuffles, so the only barriers are the ones around the staging.
/// `w` holds `k / elems` blocks per row, row after row.
inline fn gemvRow(comptime t: format.StorageType, w: [*]const u8, x: [*]const f32, y: [*]f32, k: u32, m: u32) void {
    const info = comptime blocks.info(t).?;
    const warp = tid() >> 5;
    const lane = tid() & 31;
    const row = bid() * gemv_rows + warp;
    const valid = row < m;
    const nb_row: usize = k / info.elems;
    const row_bytes = w + @as(usize, row) * nb_row * info.bytes;
    var acc: f32 = 0;

    var k0: usize = 0;
    while (k0 < k) : (k0 += gemv_chunk) {
        const len: usize = @min(@as(usize, gemv_chunk), k - k0);
        var i: usize = tid();
        while (i < len) : (i += bdim()) gemv_x[i + i / 32] = x[k0 + i];
        barSync();
        if (valid) {
            const first = k0 / info.elems;
            const nbc = len / info.elems;
            var b: usize = lane;
            while (b < nbc) : (b += 32) {
                var tmp: [info.elems]f32 = undefined;
                decodeBlock(t, row_bytes[(first + b) * info.bytes ..][0..info.bytes], &tmp);
                const base = b * info.elems;
                var e: usize = 0;
                while (e < info.elems) : (e += 1) acc += tmp[e] * gemv_x[base + e + (base + e) / 32];
            }
        }
        barSync(); // the staged chunk is overwritten by the next step
    }
    var off: u32 = 16;
    while (off > 0) : (off >>= 1) acc += shflDown(acc, off);
    if (lane == 0 and valid) y[row] = acc;
}

/// Output rows and tokens of one tile of the batched product, and the slice of the inner dimension staged per step.
/// A block has `tile_rows * tile_tokens` threads, one per output element of the tile.
const tile_rows = 16;
const tile_tokens = 8;
const tile_k = 256;
/// Row stride of the staged tiles: one extra float breaks the power of two stride that would put every thread of a
/// warp on the same shared memory bank.
const tile_stride = tile_k + 1;

var tile_w: [tile_rows * tile_stride]f32 addrspace(.shared) = undefined;
var tile_x: [tile_tokens * tile_stride]f32 addrspace(.shared) = undefined;

/// Decodes `unit` consecutive weights starting at `src` into `out`. Dense types decode element by element.
inline fn decodeUnit(comptime t: format.StorageType, comptime unit: usize, src: []const u8, out: *[unit]f32) void {
    const info = comptime blocks.info(t).?;
    if (comptime info.elems == unit) {
        decodeBlock(t, src[0..info.bytes], out);
    } else {
        var e: usize = 0;
        while (e < unit) : (e += 1) {
            var one: [1]f32 = undefined;
            decodeBlock(t, src[e * info.bytes ..][0..info.bytes], &one);
            out[e] = one[0];
        }
    }
}

/// y[token][row] = dot(W[row, :], x[token, :]) for a tile of `tile_rows` rows and `tile_tokens` tokens per block, for
/// batches. The activation tile is staged in shared memory once and reused by every row of the tile (global activation
/// traffic drops by `tile_rows`), and each weight block is decoded once into shared memory and reused by every token.
/// Grid x = row tiles, grid y = token tiles. `x` is [n][k] and `y` is [n][m].
inline fn gemmTile(comptime t: format.StorageType, w: [*]const u8, x: [*]const f32, y: [*]f32, k: u32, m: u32, n: u32) void {
    const info = comptime blocks.info(t).?;
    // Dense types are decoded 32 at a time so every format has work items of at least 32 weights.
    const unit = comptime if (info.elems < 32) 32 else info.elems;
    const unit_bytes = comptime unit / info.elems * info.bytes;
    const units_per_step = comptime tile_k / unit;
    const row0 = bid() * tile_rows;
    const tok0 = bidY() * tile_tokens;
    const row_stride: usize = (@as(usize, k) / info.elems) * info.bytes;
    const r = tid() / tile_tokens;
    const tk = tid() % tile_tokens;
    var acc: f32 = 0;

    var k0: u32 = 0;
    while (k0 < k) : (k0 += tile_k) {
        const len = @min(@as(u32, tile_k), k - k0);

        var idx: u32 = tid();
        while (idx < tile_tokens * tile_k) : (idx += bdim()) {
            const tt = idx / tile_k;
            const kk = idx % tile_k;
            const tok = tok0 + tt;
            tile_x[tt * tile_stride + kk] = if (tok < n and kk < len) x[@as(usize, tok) * k + k0 + kk] else 0;
        }

        idx = tid();
        while (idx < tile_rows * units_per_step) : (idx += bdim()) {
            const rr = idx / units_per_step;
            const off: u32 = (idx % units_per_step) * unit;
            const dst = tile_w[rr * tile_stride + off ..][0..unit];
            if (row0 + rr < m and off < len) {
                var tmp: [unit]f32 = undefined;
                const src = w + @as(usize, row0 + rr) * row_stride + (@as(usize, k0 + off) / info.elems) * info.bytes;
                decodeUnit(t, unit, src[0..unit_bytes], &tmp);
                var e: usize = 0;
                while (e < unit) : (e += 1) dst[e] = tmp[e];
            } else {
                var e: usize = 0;
                while (e < unit) : (e += 1) dst[e] = 0;
            }
        }
        barSync();

        var kk: u32 = 0;
        while (kk < len) : (kk += 1) acc += tile_w[r * tile_stride + kk] * tile_x[tk * tile_stride + kk];
        barSync(); // the tiles are overwritten by the next step
    }
    if (row0 + r < m and tok0 + tk < n) y[@as(usize, tok0 + tk) * m + row0 + r] = acc;
}

// -- tensor core matrix product (sm_80 and newer) --------------------------------------------------------------------

/// `mma.sync.m16n8k16` with f16 operands needs sm_80; the sm_70 module has no such kernels and the host falls back to
/// the tiled kernel above.
pub const has_tensor_cores = builtin.cpu.arch == .nvptx64 and builtin.cpu.has(.nvptx, .sm_80);

/// Output rows and tokens of one block of the tensor core kernel, and the weights per row decoded to f16 in one round.
const tc_rows = 64;
const tc_tokens = 128;
const tc_round = 256;
/// Activations staged per step inside a round.
const tc_step = 32;
/// Row strides of the staged tiles in halves. 8 extra halves put the eight rows a fragment load touches on distinct
/// groups of banks.
const tc_a_ld = tc_round + 8;
const tc_b_ld = tc_step + 8;
/// Threads per block: eight warps, a 2 x 4 grid of 32 x 32 sub-tiles.
pub const tc_threads = 256;

var tc_a: [tc_rows * tc_a_ld]f16 addrspace(.shared) = undefined;
var tc_b: [tc_tokens * tc_b_ld]f16 addrspace(.shared) = undefined;

/// A shared array of halves as an array of 32-bit words (two halves each).
inline fn wordsOf(p: anytype) [*]addrspace(.shared) u32 {
    return @ptrCast(@alignCast(p));
}

inline fn packHalves(lo: f32, hi: f32) u32 {
    const a: f16 = @floatCast(lo);
    const b: f16 = @floatCast(hi);
    return @as(u32, @as(u16, @bitCast(a))) | (@as(u32, @as(u16, @bitCast(b))) << 16);
}

/// d += a * b for one 16 x 8 x 16 tile, accumulating in f32. The operands are fragments in the layout PTX defines:
/// with g = lane / 4 and t = lane % 4, a0..a3 hold A[g][2t..2t+1], A[g+8][2t..], A[g][2t+8..], A[g+8][2t+8..] and b0, b1
/// hold B[2t..2t+1][g], B[2t+8..2t+9][g] as packed halves; d0..d3 are D[g][2t], D[g][2t+1], D[g+8][2t], D[g+8][2t+1].
inline fn mma16816(d: *[4]f32, a: [4]u32, b: [2]u32) void {
    var d0 = d[0];
    var d1 = d[1];
    var d2 = d[2];
    var d3 = d[3];
    asm volatile ("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%[d0], %[d1], %[d2], %[d3]}, {%[a0], %[a1], %[a2], %[a3]}, {%[b0], %[b1]}, {%[d0], %[d1], %[d2], %[d3]};"
        : [d0] "+f" (d0),
          [d1] "+f" (d1),
          [d2] "+f" (d2),
          [d3] "+f" (d3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
    );
    d.* = .{ d0, d1, d2, d3 };
}

/// y[token][row] = dot(W[row, :], x[token, :]) on tensor cores. A block owns `tc_rows` rows and `tc_tokens` tokens.
/// Per round of `tc_round` weights along k, the block decodes its rows to f16 in shared memory (every format goes
/// through the same block decoders as the other kernels), then walks the round in steps of `tc_step`, staging that
/// slice of the activations as f16 and multiplying with `mma.sync`. Operands are rounded to f16; products accumulate
/// in f32. Grid x = row tiles, grid y = token tiles.
inline fn gemmTensor(comptime t: format.StorageType, w: [*]const u8, x: [*]const f16, y: [*]f32, k: u32, m: u32, n: u32) void {
    const info = comptime blocks.info(t).?;
    const unit = comptime if (info.elems < 32) 32 else info.elems;
    const unit_bytes = comptime unit / info.elems * info.bytes;
    const units_per_row = comptime tc_round / unit;
    const row0 = bid() * tc_rows;
    const tok0 = bidY() * tc_tokens;
    const row_stride: usize = (@as(usize, k) / info.elems) * info.bytes;
    const warp = tid() >> 5;
    const lane = tid() & 31;
    const g = lane >> 2;
    const q = lane & 3;
    const wr = (warp & 1) * 32;
    const wc = (warp >> 1) * 32;

    var acc: [2][4][4]f32 = undefined;
    for (&acc) |*a| for (a) |*b| {
        b.* = .{ 0, 0, 0, 0 };
    };

    var k0: u32 = 0;
    while (k0 < k) : (k0 += tc_round) {
        const len = @min(@as(u32, tc_round), k - k0);

        // Decode this round of weights for all rows of the block.
        if (comptime t == .f16 or t == .bf16 or t == .f32) {
            // Dense weights need no block decoder: 8 weights per 16 byte (f16, bf16) or two 16 byte (f32) loads.
            const bytes_per: usize = comptime if (t == .f32) 4 else 2;
            var vi: u32 = tid();
            while (vi < tc_rows * (tc_round / 8)) : (vi += tc_threads) {
                const rr = vi / (tc_round / 8);
                const c8: u32 = (vi % (tc_round / 8)) * 8;
                var out: @Vector(4, u32) = .{ 0, 0, 0, 0 };
                if (row0 + rr < m and c8 < len) {
                    const src = w + @as(usize, row0 + rr) * row_stride + @as(usize, k0 + c8) * bytes_per;
                    if (comptime t == .f16) {
                        const v: *align(2) const @Vector(4, u32) = @ptrCast(@alignCast(src));
                        out = v.*;
                    } else if (comptime t == .bf16) {
                        const v: *align(2) const @Vector(4, u32) = @ptrCast(@alignCast(src));
                        const raw = v.*;
                        inline for (0..4) |i| {
                            const lo: f32 = @bitCast(raw[i] << 16);
                            const hi: f32 = @bitCast(raw[i] & 0xffff0000);
                            out[i] = packHalves(lo, hi);
                        }
                    } else {
                        const v: *align(4) const [2]@Vector(4, u32) = @ptrCast(@alignCast(src));
                        inline for (0..2) |h| {
                            inline for (0..2) |i| {
                                const lo: f32 = @bitCast(v[h][2 * i]);
                                const hi: f32 = @bitCast(v[h][2 * i + 1]);
                                out[2 * h + i] = packHalves(lo, hi);
                            }
                        }
                    }
                }
                const dst: *addrspace(.shared) align(4) @Vector(4, u32) = @ptrCast(wordsOf(&tc_a) + (rr * tc_a_ld + c8) / 2);
                dst.* = out;
            }
        } else {
            var idx: u32 = tid();
            while (idx < tc_rows * units_per_row) : (idx += tc_threads) {
                const rr = idx / units_per_row;
                const off: u32 = (idx % units_per_row) * unit;
                var tmp: [unit]f32 = undefined;
                if (row0 + rr < m and off < len) {
                    const src = w + @as(usize, row0 + rr) * row_stride + (@as(usize, k0 + off) / info.elems) * info.bytes;
                    decodeUnit(t, unit, src[0..unit_bytes], &tmp);
                } else {
                    for (&tmp) |*v| v.* = 0;
                }
                const words = wordsOf(&tc_a) + (rr * tc_a_ld + off) / 2;
                var e: usize = 0;
                while (e < unit) : (e += 2) words[e / 2] = packHalves(tmp[e], tmp[e + 1]);
            }
        }

        // Activations of tokens tok0.. for k in k0 + step .. + tc_step (already f16), `tc_vecs` copies of 16 bytes per
        // thread. They are fetched into registers while the previous step is multiplied, so the global load overlaps
        // the tensor core work; they are parked in shared memory once everyone has finished reading the old ones.
        const tc_vecs = tc_tokens * (tc_step / 8) / tc_threads;
        const stage = struct {
            inline fn fetch(xp: [*]const f16, tok0_: u32, n_: u32, k_: u32, k0_: u32, step_: u32, len_: u32, out: *[tc_vecs]@Vector(4, u32)) void {
                inline for (0..tc_vecs) |u| {
                    const p: u32 = tid() + @as(u32, u) * tc_threads;
                    const tok_ = tok0_ + p / (tc_step / 8);
                    const kq_: u32 = (p % (tc_step / 8)) * 8;
                    out[u] = .{ 0, 0, 0, 0 };
                    if (tok_ < n_ and step_ + kq_ < len_) {
                        const src: *align(2) const @Vector(4, u32) = @ptrCast(xp + @as(usize, tok_) * k_ + k0_ + step_ + kq_);
                        out[u] = src.*;
                    }
                }
            }
            inline fn park(out: *const [tc_vecs]@Vector(4, u32)) void {
                inline for (0..tc_vecs) |u| {
                    const p: u32 = tid() + @as(u32, u) * tc_threads;
                    const tt_ = p / (tc_step / 8);
                    const kq_: u32 = (p % (tc_step / 8)) * 8;
                    const dst: *addrspace(.shared) align(4) @Vector(4, u32) = @ptrCast(wordsOf(&tc_b) + (tt_ * tc_b_ld + kq_) / 2);
                    dst.* = out[u];
                }
            }
        };
        var v: [tc_vecs]@Vector(4, u32) = undefined;
        stage.fetch(x, tok0, n, k, k0, 0, len, &v);
        stage.park(&v);
        barSync(); // the decoded weights and the first activations are ready

        var step: u32 = 0;
        while (step < len) : (step += tc_step) {
            const more = step + tc_step < len;
            if (more) stage.fetch(x, tok0, n, k, k0, step + tc_step, len, &v);

            comptime var half_k = 0;
            inline while (half_k < tc_step) : (half_k += 16) {
                var af: [2][4]u32 = undefined;
                inline for (0..2) |mi| {
                    const r = wr + mi * 16 + g;
                    const kc = step + half_k + q * 2;
                    af[mi][0] = wordsOf(&tc_a)[(r * tc_a_ld + kc) / 2];
                    af[mi][1] = wordsOf(&tc_a)[((r + 8) * tc_a_ld + kc) / 2];
                    af[mi][2] = wordsOf(&tc_a)[(r * tc_a_ld + kc + 8) / 2];
                    af[mi][3] = wordsOf(&tc_a)[((r + 8) * tc_a_ld + kc + 8) / 2];
                }
                inline for (0..4) |ni| {
                    const c = wc + ni * 8 + g;
                    const kc = half_k + q * 2;
                    const bf = [2]u32{
                        wordsOf(&tc_b)[(c * tc_b_ld + kc) / 2],
                        wordsOf(&tc_b)[(c * tc_b_ld + kc + 8) / 2],
                    };
                    inline for (0..2) |mi| mma16816(&acc[mi][ni], af[mi], bf);
                }
            }
            barSync(); // everyone has read this step's activations
            if (more) stage.park(&v);
            barSync();
        }
    }

    inline for (0..2) |mi| inline for (0..4) |ni| {
        const r0 = row0 + wr + mi * 16 + g;
        const c0 = tok0 + wc + ni * 8 + q * 2;
        inline for (0..4) |i| {
            const r = r0 + (i / 2) * 8;
            const c = c0 + (i % 2);
            if (r < m and c < n) y[@as(usize, c) * m + r] = acc[mi][ni][i];
        }
    };
}

fn GemmTensorKernel(comptime t: format.StorageType) type {
    return struct {
        fn run(w: [*]const u8, x: [*]const f16, y: [*]f32, k: u32, m: u32, n: u32) callconv(.kernel) void {
            gemmTensor(t, w, x, y, k, m, n);
        }
    };
}

comptime {
    if (has_tensor_cores) {
        for (gemm_formats) |t| @export(&GemmTensorKernel(t).run, .{ .name = "gemm_tc_" ++ @tagName(t) });
    }
}

// -- integer tensor core matrix product (sm_80 and newer) ------------------------------------------------------------
//
// Quantized weights multiplied against activations quantized to 8 bits per 32 values, on `mma.sync.m16n8k32.s8` (or
// two `m16n8k16` where a format has a scale per 16 values): the products are exact in int32 and the scales are applied
// once per 32 values. This is the arithmetic the CPU engine uses (`vecdot`), and on Ampere the integer units run at four
// times the rate of f16 with f32 accumulation. Every 32 weights of a row become 32 signed bytes in shared memory plus a
// scale (and a minimum, for formats that have one): w = sc * q - mn, so a block's contribution to a token is
// d_x * (sc * sum(q * xq) - mn * sum(xq)).



/// Quantizes activations to int8 with one f32 scale per 32 values (`xs[b] = max|x| / 127`, `xq = round(x / scale)`) and
/// the sum of each block's quants (`xsum`, for formats with a minimum). One warp per block of 32; `x` is [n][k] with k a
/// multiple of 32, so block `b` is elements 32b .. 32b + 31.
fn quantizeActKernel(xq: [*]i8, xs: [*]f32, xsum: [*]f32, x: [*]const f32, blocks_total: u32) callconv(.kernel) void {
    const wid = globalId() >> 5;
    if (wid >= blocks_total) return; // the warp leaves together, so the shuffles below see all lanes
    const lane = tid() & 31;
    const v = x[@as(usize, wid) * 32 + lane];
    var amax = @abs(v);
    inline for (.{ 16, 8, 4, 2, 1 }) |m| amax = @max(amax, shflXor(amax, m));
    const inv: f32 = if (amax > 0) 127.0 / amax else 0;
    const q = std.math.clamp(roundToInt(v * inv), -127, 127);
    xq[@as(usize, wid) * 32 + lane] = @intCast(q);
    var sum: f32 = @floatFromInt(q);
    inline for (.{ 16, 8, 4, 2, 1 }) |m| sum += shflXor(sum, m);
    if (lane == 0) {
        xs[wid] = amax / 127.0;
        xsum[wid] = sum;
    }
}

inline fn mmaS8(d: *[4]i32, a: [4]u32, b: [2]u32) void {
    var d0 = d[0];
    var d1 = d[1];
    var d2 = d[2];
    var d3 = d[3];
    asm volatile ("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%[d0], %[d1], %[d2], %[d3]}, {%[a0], %[a1], %[a2], %[a3]}, {%[b0], %[b1]}, {%[d0], %[d1], %[d2], %[d3]};"
        : [d0] "+r" (d0),
          [d1] "+r" (d1),
          [d2] "+r" (d2),
          [d3] "+r" (d3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
    );
    d.* = .{ d0, d1, d2, d3 };
}

/// The k16 form: A rows g and g + 8 of 16 values, B 16 values for token g.
inline fn mmaS8k16(d: *[4]i32, a0: u32, a1: u32, b0: u32) void {
    var d0 = d[0];
    var d1 = d[1];
    var d2 = d[2];
    var d3 = d[3];
    asm volatile ("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%[d0], %[d1], %[d2], %[d3]}, {%[a0], %[a1]}, {%[b0]}, {%[d0], %[d1], %[d2], %[d3]};"
        : [d0] "+r" (d0),
          [d1] "+r" (d1),
          [d2] "+r" (d2),
          [d3] "+r" (d3),
        : [a0] "r" (a0),
          [a1] "r" (a1),
          [b0] "r" (b0),
    );
    d.* = .{ d0, d1, d2, d3 };
}

/// Blocks (rows, tokens) of the integer kernel and the k32 blocks staged per step.
const im_rows = 64;
const im_tokens = 128;
const im_blocks = 4;
/// Words per staged row: 4 blocks of 32 bytes plus 16 bytes of padding, which puts the eight rows a fragment load touches
/// on distinct banks.
const im_ld = im_blocks * 8 + 4;
var im_a: [im_rows * im_ld]u32 addrspace(.shared) = undefined;
var im_b: [im_tokens * im_ld]u32 addrspace(.shared) = undefined;
/// Scale per (row, block, half) and minimum per (row, block) of the weights; scale, sum of quants per (token, block).
var im_sw: [im_rows * im_blocks * 2]f32 addrspace(.shared) = undefined;
var im_sm: [im_rows * im_blocks]f32 addrspace(.shared) = undefined;
var im_sx: [im_tokens * im_blocks]f32 addrspace(.shared) = undefined;
var im_ss: [im_tokens * im_blocks]f32 addrspace(.shared) = undefined;

/// Storage types with an integer kernel.
pub const imma_formats = [_]format.StorageType{ .q8_0, .q4_0, .q4_1, .q5_0, .q5_1, .q4_k, .q5_k, .q6_k };

fn loadWord(p: usize) u32 {
    const ptr: *const u32 = @ptrFromInt(p);
    return ptr.*;
}

/// `nw` words of the byte stream that starts at `addr` (2-byte aligned), into `out[0..nw]`. Reads one word more than it
/// returns when the stream is not word aligned; weight buffers carry slack for that.
inline fn streamWords(addr: usize, comptime nw: usize, out: []u32) void {
    const base = addr & ~@as(usize, 3);
    if (addr & 2 == 0) {
        inline for (0..nw) |i| out[i] = loadWord(base + 4 * i);
    } else {
        var prev = loadWord(base);
        inline for (0..nw) |i| {
            const next = loadWord(base + 4 * (i + 1));
            out[i] = (prev >> 16) | (next << 16);
            prev = next;
        }
    }
}

/// Like `streamWords` for a stream that starts on a 16 byte boundary and is a whole number of 16 byte pieces: vector
/// loads, a quarter of the instructions (the K-quant block sizes 144 and 176 keep every block 16 byte aligned).
inline fn streamWords16(addr: usize, comptime nw: usize, out: []u32) void {
    inline for (0..nw / 4) |i| {
        const p: *align(16) const @Vector(4, u32) = @ptrFromInt(addr + 16 * i);
        const v = p.*;
        inline for (0..4) |j| out[4 * i + j] = v[j];
    }
}

const m0f: u32 = 0x0f0f0f0f;
inline fn signedBytes(x: u32, comptime bias: u32) u32 {
    // Per byte x - bias (x < 256 - 128), as a signed byte, without borrows between bytes.
    return ((x | 0x80808080) -% (bias * 0x01010101)) ^ 0x80808080;
}
inline fn spreadBits(bits: u32) u32 {
    // The low four bits of `bits`, one per byte.
    return (bits & 1) | (((bits >> 1) & 1) << 8) | (((bits >> 2) & 1) << 16) | (((bits >> 3) & 1) << 24);
}
inline fn f16lo(w: u32) f32 {
    return @floatCast(@as(f16, @bitCast(@as(u16, @truncate(w)))));
}
inline fn f16hi(w: u32) f32 {
    return @floatCast(@as(f16, @bitCast(@as(u16, @truncate(w >> 16)))));
}
inline fn byteOf(w: u32, i: u32) u32 {
    return (w >> @intCast(8 * (i & 3))) & 0xff;
}

const ImmaRaw = [20]u32;
const ImmaBlock = struct { q: [8]u32, sc: [2]f32, mn: f32 };

/// Reads what block `sub` (0-based among the 32-blocks of the quant block at `baddr`) needs, in registers.
inline fn immaFetch(comptime t: format.StorageType, baddr: usize, sub: u32) ImmaRaw {
    var raw: ImmaRaw = @splat(0);
    switch (t) {
        .q8_0 => streamWords(baddr, 9, raw[0..]),
        .q4_0, .q4_1 => streamWords(baddr, 5, raw[0..]),
        .q5_0, .q5_1 => streamWords(baddr, 6, raw[0..]),
        .q4_k => {
            streamWords16(baddr, 4, raw[0..]);
            streamWords16(baddr + 16 + 32 * (sub >> 1), 8, raw[4..]);
        },
        .q5_k => {
            streamWords16(baddr, 4, raw[0..]);
            streamWords16(baddr + 16, 8, raw[4..]);
            streamWords16(baddr + 48 + 32 * (sub >> 1), 8, raw[12..]);
        },
        .q6_k => {
            const h = sub >> 2;
            const i = sub & 3;
            streamWords(baddr + 64 * h + 32 * (i & 1), 8, raw[0..]);
            streamWords(baddr + 128 + 32 * h, 8, raw[8..]);
            streamWords(baddr + 192 + 8 * h + 2 * i, 1, raw[16..]);
            streamWords(baddr + 208, 1, raw[17..]);
        },
        else => @compileError("no integer kernel for " ++ @tagName(t)),
    }
    return raw;
}

/// The 32 signed quants of the block, its scale(s) and minimum, from what `immaFetch` read.
inline fn immaUnpack(comptime t: format.StorageType, raw: *const ImmaRaw, sub: u32) ImmaBlock {
    var b: ImmaBlock = .{ .q = @splat(0), .sc = .{ 0, 0 }, .mn = 0 };
    switch (t) {
        .q8_0 => {
            b.sc[0] = f16lo(raw[0]);
            inline for (0..8) |i| b.q[i] = (raw[i] >> 16) | (raw[i + 1] << 16);
        },
        .q4_0 => {
            b.sc[0] = f16lo(raw[0]);
            inline for (0..4) |i| {
                const w = (raw[i] >> 16) | (raw[i + 1] << 16);
                b.q[i] = signedBytes(w & m0f, 8);
                b.q[4 + i] = signedBytes((w >> 4) & m0f, 8);
            }
        },
        .q4_1 => {
            b.sc[0] = f16lo(raw[0]);
            b.mn = -f16hi(raw[0]);
            inline for (0..4) |i| {
                b.q[i] = raw[1 + i] & m0f;
                b.q[4 + i] = (raw[1 + i] >> 4) & m0f;
            }
        },
        .q5_0, .q5_1 => {
            const one = t == .q5_1;
            b.sc[0] = f16lo(raw[0]);
            const qh = if (one) raw[1] else (raw[0] >> 16) | (raw[1] << 16);
            if (one) b.mn = -f16hi(raw[0]);
            inline for (0..4) |i| {
                const w = if (one) raw[2 + i] else (raw[1 + i] >> 16) | (raw[2 + i] << 16);
                const lo = (w & m0f) | (spreadBits(qh >> (4 * i)) << 4);
                const hi = ((w >> 4) & m0f) | (spreadBits(qh >> (4 * i + 16)) << 4);
                b.q[i] = if (one) lo else signedBytes(lo, 16);
                b.q[4 + i] = if (one) hi else signedBytes(hi, 16);
            }
        },
        .q4_k, .q5_k => {
            const d = f16lo(raw[0]);
            const dmin = f16hi(raw[0]);
            // 6-bit scale and minimum of sub-block `sub` out of the 12 packed bytes in raw[1..4].
            const sb = struct {
                inline fn at(r: *const ImmaRaw, i: u32) u32 {
                    return byteOf(r[1 + i / 4], i);
                }
            };
            var sc: u32 = undefined;
            var mn: u32 = undefined;
            if (sub < 4) {
                sc = sb.at(raw, sub) & 63;
                mn = sb.at(raw, sub + 4) & 63;
            } else {
                sc = (sb.at(raw, sub + 4) & 15) | ((sb.at(raw, sub - 4) >> 6) << 4);
                mn = (sb.at(raw, sub + 4) >> 4) | ((sb.at(raw, sub) >> 6) << 4);
            }
            b.sc[0] = d * @as(f32, @floatFromInt(sc));
            b.mn = dmin * @as(f32, @floatFromInt(mn));
            const shift: u5 = @intCast(4 * (sub & 1));
            // Q4_K: qs at raw[4..12]. Q5_K: qh at raw[4..12], qs at raw[12..20], and bit `sub` of qh is the fifth bit.
            const qs_at: usize = if (t == .q5_k) 12 else 4;
            inline for (0..8) |i| {
                var v = (raw[qs_at + i] >> shift) & m0f;
                if (t == .q5_k) v |= ((raw[4 + i] >> @as(u5, @intCast(sub))) & 0x01010101) << 4;
                b.q[i] = v;
            }
        },
        .q6_k => {
            const i = sub & 3;
            const d = f16lo(raw[17]);
            const sa: i32 = @as(i8, @bitCast(@as(u8, @truncate(raw[16]))));
            const sbv: i32 = @as(i8, @bitCast(@as(u8, @truncate(raw[16] >> 8))));
            b.sc[0] = d * @as(f32, @floatFromInt(sa));
            b.sc[1] = d * @as(f32, @floatFromInt(sbv));
            const shift: u5 = @intCast(4 * (i >> 1));
            inline for (0..8) |w| {
                const v = ((raw[w] >> shift) & m0f) | (((raw[8 + w] >> @as(u5, @intCast(2 * i))) & 0x03030303) << 4);
                b.q[w] = signedBytes(v, 32);
            }
        },
        else => @compileError("no integer kernel for " ++ @tagName(t)),
    }
    return b;
}

/// y[token][row] = sum over 32-blocks of d_x * (sc * dot_int8(q, xq) - mn * sum(xq)): a block of 64 rows and 128 tokens,
/// eight warps in a 2 x 4 grid of 32 x 32 sub-tiles, `im_blocks` blocks of 32 along k per step. The next step's data is
/// fetched into registers while the current one is multiplied and parked in shared memory afterwards.
inline fn gemmImma(comptime t: format.StorageType, w: [*]const u8, xq: [*]const i8, xs: [*]const f32, xsum: [*]const f32, y: [*]f32, k: u32, m: u32, n: u32) void {
    const info = comptime blocks.info(t).?;
    const per_unit: u32 = comptime if (info.elems == 256) 8 else 1; // 32-blocks per quant block
    const has_min = comptime (t == .q4_1 or t == .q5_1 or t == .q4_k or t == .q5_k);
    const halves = comptime (t == .q6_k);
    const nb: u32 = k / 32;
    const row_bytes: usize = @as(usize, k / info.elems) * info.bytes;
    const row0 = bid() * im_rows;
    const tok0 = bidY() * im_tokens;
    const warp = tid() >> 5;
    const lane = tid() & 31;
    const g = lane >> 2;
    const q = lane & 3;
    const wr = (warp & 1) * 32;
    const wc = (warp >> 1) * 32;

    var acc: [2][4][4]f32 = undefined;
    for (&acc) |*a| for (a) |*b| {
        b.* = .{ 0, 0, 0, 0 };
    };

    // This thread stages one weight block (row tid / 4, block tid % 4 of the step), 4 sixteen byte pieces of the
    // activations and 2 scales and sums.
    const wrow = tid() / im_blocks;
    const wblk = tid() % im_blocks;
    const Fetched = struct { raw: ImmaRaw, valid: bool, act: [4]@Vector(4, u32), sx: [2]f32, ss: [2]f32 };
    const stage = struct {
        inline fn fetch(wp: [*]const u8, xqp: [*]const i8, xsp: [*]const f32, xsump: [*]const f32, b0: u32, nb_: u32, k_: u32, m_: u32, n_: u32, row0_: u32, tok0_: u32, wrow_: u32, wblk_: u32, rb: usize) Fetched {
            var f: Fetched = undefined;
            const bl = b0 + wblk_;
            f.valid = row0_ + wrow_ < m_ and bl < nb_;
            f.raw = @splat(0);
            if (f.valid) {
                const a = @intFromPtr(wp) + @as(usize, row0_ + wrow_) * rb + @as(usize, bl / per_unit) * info.bytes;
                f.raw = immaFetch(t, a, bl % per_unit);
            }
            inline for (0..4) |u| {
                const p: u32 = tid() + @as(u32, u) * tc_threads;
                const tk = p / 8;
                const v8 = p % 8;
                f.act[u] = .{ 0, 0, 0, 0 };
                if (tok0_ + tk < n_ and b0 + v8 / 2 < nb_) {
                    const src: *align(16) const @Vector(4, u32) = @ptrCast(@alignCast(xqp + @as(usize, tok0_ + tk) * k_ + @as(usize, b0) * 32 + v8 * 16));
                    f.act[u] = src.*;
                }
            }
            inline for (0..2) |u| {
                const p: u32 = tid() + @as(u32, u) * tc_threads;
                const tk = p / 4;
                const bi = p % 4;
                f.sx[u] = 0;
                f.ss[u] = 0;
                if (tok0_ + tk < n_ and b0 + bi < nb_) {
                    const idx = @as(usize, tok0_ + tk) * nb_ + b0 + bi;
                    f.sx[u] = xsp[idx];
                    if (has_min) f.ss[u] = xsump[idx];
                }
            }
            return f;
        }

        inline fn park(f: *const Fetched, b0: u32, wrow_: u32, wblk_: u32) void {
            var blk: ImmaBlock = .{ .q = @splat(0), .sc = .{ 0, 0 }, .mn = 0 };
            if (f.valid) blk = immaUnpack(t, &f.raw, (b0 + wblk_) % per_unit);
            const dst: [*]addrspace(.shared) u32 = @ptrCast(&im_a[wrow_ * im_ld + wblk_ * 8]);
            inline for (0..8) |i| dst[i] = blk.q[i];
            im_sw[(wrow_ * im_blocks + wblk_) * 2] = blk.sc[0];
            im_sw[(wrow_ * im_blocks + wblk_) * 2 + 1] = blk.sc[1];
            if (has_min) im_sm[wrow_ * im_blocks + wblk_] = blk.mn;
            inline for (0..4) |u| {
                const p: u32 = tid() + @as(u32, u) * tc_threads;
                const dstv: *addrspace(.shared) align(4) @Vector(4, u32) = @ptrCast(&im_b[(p / 8) * im_ld + (p % 8) * 4]);
                dstv.* = f.act[u];
            }
            inline for (0..2) |u| {
                const p: u32 = tid() + @as(u32, u) * tc_threads;
                im_sx[p] = f.sx[u];
                if (has_min) im_ss[p] = f.ss[u];
            }
        }
    };

    var f = stage.fetch(w, xq, xs, xsum, 0, nb, k, m, n, row0, tok0, wrow, wblk, row_bytes);
    stage.park(&f, 0, wrow, wblk);
    barSync();

    var b0: u32 = 0;
    while (b0 < nb) : (b0 += im_blocks) {
        const more = b0 + im_blocks < nb;
        if (more) f = stage.fetch(w, xq, xs, xsum, b0 + im_blocks, nb, k, m, n, row0, tok0, wrow, wblk, row_bytes);

        inline for (0..im_blocks) |blk| {
            var ci: [2][4][4]i32 = undefined;
            var cj: [2][4][4]i32 = undefined; // second 16 values, for formats with a scale per 16
            var af: [2][4]u32 = undefined;
            inline for (0..2) |mi| {
                const r = wr + mi * 16 + g;
                af[mi] = .{
                    im_a[r * im_ld + blk * 8 + q],
                    im_a[(r + 8) * im_ld + blk * 8 + q],
                    im_a[r * im_ld + blk * 8 + q + 4],
                    im_a[(r + 8) * im_ld + blk * 8 + q + 4],
                };
            }
            inline for (0..4) |ni| {
                const c = wc + ni * 8 + g;
                const bf = [2]u32{ im_b[c * im_ld + blk * 8 + q], im_b[c * im_ld + blk * 8 + q + 4] };
                inline for (0..2) |mi| {
                    ci[mi][ni] = .{ 0, 0, 0, 0 };
                    if (halves) {
                        cj[mi][ni] = .{ 0, 0, 0, 0 };
                        mmaS8k16(&ci[mi][ni], af[mi][0], af[mi][1], bf[0]);
                        mmaS8k16(&cj[mi][ni], af[mi][2], af[mi][3], bf[1]);
                    } else {
                        mmaS8(&ci[mi][ni], af[mi], bf);
                    }
                }
            }
            inline for (0..2) |mi| {
                const r = wr + mi * 16 + g;
                const swa = im_sw[(r * im_blocks + blk) * 2];
                const swb = im_sw[((r + 8) * im_blocks + blk) * 2];
                const swa2 = if (halves) im_sw[(r * im_blocks + blk) * 2 + 1] else 0;
                const swb2 = if (halves) im_sw[((r + 8) * im_blocks + blk) * 2 + 1] else 0;
                const mna = if (has_min) im_sm[r * im_blocks + blk] else 0;
                const mnb = if (has_min) im_sm[(r + 8) * im_blocks + blk] else 0;
                inline for (0..4) |ni| {
                    const c = wc + ni * 8 + q * 2;
                    inline for (0..2) |e| {
                        const sx = im_sx[(c + e) * im_blocks + blk];
                        const ss = if (has_min) im_ss[(c + e) * im_blocks + blk] else 0;
                        var va: f32 = swa * @as(f32, @floatFromInt(ci[mi][ni][e]));
                        var vb: f32 = swb * @as(f32, @floatFromInt(ci[mi][ni][2 + e]));
                        if (halves) {
                            va += swa2 * @as(f32, @floatFromInt(cj[mi][ni][e]));
                            vb += swb2 * @as(f32, @floatFromInt(cj[mi][ni][2 + e]));
                        }
                        if (has_min) {
                            va -= mna * ss;
                            vb -= mnb * ss;
                        }
                        acc[mi][ni][e] += sx * va;
                        acc[mi][ni][2 + e] += sx * vb;
                    }
                }
            }
        }
        barSync(); // everyone has read this step
        if (more) stage.park(&f, b0 + im_blocks, wrow, wblk);
        barSync();
    }

    inline for (0..2) |mi| inline for (0..4) |ni| {
        const r0 = row0 + wr + mi * 16 + g;
        const c0 = tok0 + wc + ni * 8 + q * 2;
        inline for (0..4) |i| {
            const r = r0 + (i / 2) * 8;
            const c = c0 + (i % 2);
            if (r < m and c < n) y[@as(usize, c) * m + r] = acc[mi][ni][i];
        }
    };
}


/// Quantized activations of one chunk of a decode product: int8 quants as words, scale and sum per 32.
var dq_x: [gemv_chunk / 4]u32 addrspace(.shared) = undefined;
var dq_s: [gemv_chunk / 32]f32 addrspace(.shared) = undefined;
var dq_m: [gemv_chunk / 32]f32 addrspace(.shared) = undefined;

/// Single token matrix-vector product with integer arithmetic: the same block decoding as the integer GEMM, then four
/// quants per `dp4a`. A warp owns one row; its lanes take the row's 32-blocks in turn. `xq`, `xs`, `xsum` are the
/// activations quantized by `quantize_act`.
inline fn gemvDp4a(comptime t: format.StorageType, w: [*]const u8, xq: [*]const i8, xs: [*]const f32, xsum: [*]const f32, y: [*]f32, k: u32, m: u32) void {
    const info = comptime blocks.info(t).?;
    const per_unit: u32 = comptime if (info.elems == 256) 8 else 1;
    const has_min = comptime (t == .q4_1 or t == .q5_1 or t == .q4_k or t == .q5_k);
    const halves = comptime (t == .q6_k);
    const warp = tid() >> 5;
    const lane = tid() & 31;
    const row = bid() * gemv_rows + warp;
    const valid = row < m;
    const row_bytes: usize = @as(usize, k / info.elems) * info.bytes;
    const nb: u32 = k / 32;
    var acc: f32 = 0;

    // No staging and no barrier: each lane reads its block's 32 quants of the activations (two 16 byte loads, cached
    // after the first row) next to the weights it fetches.
    if (valid) {
        var bl: u32 = lane;
        while (bl < nb) : (bl += 32) {
            const a = @intFromPtr(w) + @as(usize, row) * row_bytes + @as(usize, bl / per_unit) * info.bytes;
            const raw = immaFetch(t, a, bl % per_unit);
            const xv: *align(16) const [2]@Vector(4, u32) = @ptrCast(@alignCast(xq + @as(usize, bl) * 32));
            const sx = xs[bl];
            const blk = immaUnpack(t, &raw, bl % per_unit);
            var lo: i32 = 0;
            inline for (0..4) |u| lo = dp4a(blk.q[u], xv[0][u], lo);
            var hi: i32 = 0;
            inline for (0..4) |u| hi = dp4a(blk.q[4 + u], xv[1][u], hi);
            var v: f32 = undefined;
            if (halves) {
                v = blk.sc[0] * @as(f32, @floatFromInt(lo)) + blk.sc[1] * @as(f32, @floatFromInt(hi));
            } else {
                v = blk.sc[0] * @as(f32, @floatFromInt(lo + hi));
                if (has_min) v -= blk.mn * xsum[bl];
            }
            acc += sx * v;
        }
    }
    var off: u32 = 16;
    while (off > 0) : (off >>= 1) acc += shflDown(acc, off);
    if (lane == 0 and valid) y[row] = acc;
}

fn GemvDp4aKernel(comptime t: format.StorageType) type {
    return struct {
        fn run(w: [*]const u8, xq: [*]const i8, xs: [*]const f32, xsum: [*]const f32, y: [*]f32, k: u32, m: u32) callconv(.kernel) void {
            gemvDp4a(t, w, xq, xs, xsum, y, k, m);
        }
    };
}

fn GemmImmaKernel(comptime t: format.StorageType) type {
    return struct {
        fn run(w: [*]const u8, xq: [*]const i8, xs: [*]const f32, xsum: [*]const f32, y: [*]f32, k: u32, m: u32, n: u32) callconv(.kernel) void {
            gemmImma(t, w, xq, xs, xsum, y, k, m, n);
        }
    };
}

comptime {
    @export(&quantizeActKernel, .{ .name = "quantize_act" });
    if (has_tensor_cores) {
        for (imma_formats) |t| @export(&GemmImmaKernel(t).run, .{ .name = "gemm_imma_" ++ @tagName(t) });
    }
    for (imma_formats) |t| @export(&GemvDp4aKernel(t).run, .{ .name = "gemv_dp4a_" ++ @tagName(t) });
}

// -- tensor core prompt attention (sm_80 and newer) ---------------------------------------------------------------------

/// Positions of keys and values staged per step, and threads per block of the attention kernel.
const fa_positions = 32;
const fa_threads = 128;

fn pairShuffleMax(v: f32) f32 {
    var m = v;
    inline for (.{ 1, 2 }) |off| {
        const o = shflXor(m, @as(u32, off));
        m = @max(m, o);
    }
    return m;
}

fn quadShuffleSum(v: f32) f32 {
    var m = v;
    inline for (.{ 1, 2 }) |off| {
        const o = shflXor(m, @as(u32, off));
        m += o;
    }
    return m;
}

/// Causal attention for a prompt on tensor cores (flash attention). A block is four warps; a warp owns 16 query rows of
/// one head, a block 64 consecutive tokens of ONE sequence (the host launches this kernel only for such batches).
/// Keys and values are staged 32 positions at a time (values transposed, so both operands of the second product are
/// read as packed pairs). S = Q K^T and O += P V use `mma.sync` on f16 operands with f32 accumulation; the online
/// softmax works on the accumulator registers directly (a row lives in the four lanes of a quad, so its maximum and
/// sum are two shuffles) and O is rescaled in registers. Head sizes up to 128. Grid x = 64-token tiles, grid y = heads.
inline fn flashAttention(
    comptime hd: u32,
    q: [*]const f32,
    kptrs: [*]const u64,
    vptrs: [*]const u64,
    layer_off: u64,
    slots: [*]const u32,
    poss: [*]const u32,
    out: [*]f32,
    ctx: u32,
    n_tokens: u32,
    n_heads: u32,
    group: u32,
    scale: f32,
) void {
    const kld = hd + 8; // halves per staged key row
    const vld = fa_positions + 8; // halves per staged (transposed) value row
    const warp = tid() >> 5;
    const lane = tid() & 31;
    const g = lane >> 2;
    const qd = lane & 3;
    const h = bidY();
    const t0 = bid() * 64;
    const kvh = h / group;
    const slot = slots[t0];
    const pos0 = poss[t0];
    const t_end = @min(t0 + 64, n_tokens);
    const pos_last = poss[t_end - 1];
    const ck: [*]const f16 = @ptrFromInt(kptrs[slot] + layer_off);
    const cv: [*]const f16 = @ptrFromInt(vptrs[slot] + layer_off);
    const kbase = ck + @as(usize, kvh) * ctx * hd;
    const vbase = cv + @as(usize, kvh) * ctx * hd;

    const ra = t0 + warp * 16 + g; // the two query rows this lane holds results for
    const rb = ra + 8;
    const qpos_a: i64 = if (ra < n_tokens) @as(i64, pos0) + (ra - t0) else -1;
    const qpos_b: i64 = if (rb < n_tokens) @as(i64, pos0) + (rb - t0) else -1;
    const warp_last: i64 = @as(i64, pos0) + (t0 + warp * 16 + 15 - t0);

    // Queries, scaled, as A fragments for every 16-wide slice of the head.
    var qf: [hd / 16][4]u32 = undefined;
    inline for (0..hd / 16) |dk| {
        const c = dk * 16 + qd * 2;
        var v: [4][2]f32 = .{ .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } };
        if (ra < n_tokens) {
            const base = q + (@as(usize, ra) * n_heads + h) * hd;
            v[0] = .{ base[c] * scale, base[c + 1] * scale };
            v[2] = .{ base[c + 8] * scale, base[c + 9] * scale };
        }
        if (rb < n_tokens) {
            const base = q + (@as(usize, rb) * n_heads + h) * hd;
            v[1] = .{ base[c] * scale, base[c + 1] * scale };
            v[3] = .{ base[c + 8] * scale, base[c + 9] * scale };
        }
        inline for (0..4) |i| qf[dk][i] = packHalves(v[i][0], v[i][1]);
    }

    var o: [hd / 8][4]f32 = undefined;
    for (&o) |*a| a.* = .{ 0, 0, 0, 0 };
    const neg_inf = -std.math.inf(f32);
    var m_a: f32 = neg_inf;
    var m_b: f32 = neg_inf;
    var l_a: f32 = 0;
    var l_b: f32 = 0;

    var j0: u32 = 0;
    while (j0 <= pos_last) : (j0 += fa_positions) {
        barSync();
        // Stage keys and values of positions j0 .. j0 + 31; zero past the last position of the batch, where the cache
        // may hold anything (a probability of 0 times a NaN is NaN).
        var e: u32 = tid();
        while (e < fa_positions * (hd / 8)) : (e += fa_threads) {
            const row = e / (hd / 8);
            const c8 = (e % (hd / 8)) * 8;
            const j = j0 + row;
            var kv4: [4]u32 = .{ 0, 0, 0, 0 };
            var vv4: [4]u32 = .{ 0, 0, 0, 0 };
            if (j <= pos_last) {
                const ks: [*]const u32 = @ptrCast(@alignCast(kbase + @as(usize, j) * hd + c8));
                const vs: [*]const u32 = @ptrCast(@alignCast(vbase + @as(usize, j) * hd + c8));
                kv4 = .{ ks[0], ks[1], ks[2], ks[3] };
                vv4 = .{ vs[0], vs[1], vs[2], vs[3] };
            }
            const kd = wordsOf(&fa_k) + (row * kld + c8) / 2;
            inline for (0..4) |i| kd[i] = kv4[i];
            const vh: [*]addrspace(.shared) u16 = @ptrCast(&fa_vt);
            inline for (0..4) |i| {
                vh[(c8 + 2 * i) * vld + row] = @truncate(vv4[i]);
                vh[(c8 + 2 * i + 1) * vld + row] = @truncate(vv4[i] >> 16);
            }
        }
        barSync();
        if (@as(i64, j0) > warp_last) continue;

        var sacc: [4][4]f32 = undefined;
        for (&sacc) |*a| a.* = .{ 0, 0, 0, 0 };
        inline for (0..hd / 16) |dk| {
            inline for (0..4) |ni| {
                const row = ni * 8 + g;
                const kc = dk * 16 + qd * 2;
                const bf = [2]u32{ wordsOf(&fa_k)[(row * kld + kc) / 2], wordsOf(&fa_k)[(row * kld + kc + 8) / 2] };
                mma16816(&sacc[ni], qf[dk], bf);
            }
        }

        // Causal mask and online softmax of the two rows this lane holds columns of.
        var mx_a: f32 = neg_inf;
        var mx_b: f32 = neg_inf;
        inline for (0..4) |ni| {
            inline for (0..2) |c| {
                const col: i64 = @as(i64, j0) + ni * 8 + qd * 2 + c;
                if (col > qpos_a) sacc[ni][c] = neg_inf;
                if (col > qpos_b) sacc[ni][2 + c] = neg_inf;
                mx_a = @max(mx_a, sacc[ni][c]);
                mx_b = @max(mx_b, sacc[ni][2 + c]);
            }
        }
        mx_a = pairShuffleMax(mx_a);
        mx_b = pairShuffleMax(mx_b);
        const new_a = @max(m_a, mx_a);
        const new_b = @max(m_b, mx_b);
        // A row with nothing visible yet keeps m = -inf; its probabilities are 0 and nothing needs rescaling.
        const corr_a: f32 = if (new_a == neg_inf) 1 else expf(m_a - new_a);
        const corr_b: f32 = if (new_b == neg_inf) 1 else expf(m_b - new_b);
        m_a = new_a;
        m_b = new_b;
        l_a *= corr_a;
        l_b *= corr_b;
        inline for (0..hd / 8) |nt| {
            o[nt][0] *= corr_a;
            o[nt][1] *= corr_a;
            o[nt][2] *= corr_b;
            o[nt][3] *= corr_b;
        }
        inline for (0..4) |ni| {
            inline for (0..2) |c| {
                const pa: f32 = if (new_a == neg_inf) 0 else expf(sacc[ni][c] - new_a);
                const pb: f32 = if (new_b == neg_inf) 0 else expf(sacc[ni][2 + c] - new_b);
                sacc[ni][c] = pa;
                sacc[ni][2 + c] = pb;
                l_a += pa;
                l_b += pb;
            }
        }

        // O += P V: the accumulator layout of two neighbouring 8-column tiles is the A fragment of a 16-wide slice.
        inline for (0..2) |kc| {
            const pf = [4]u32{
                packHalves(sacc[2 * kc][0], sacc[2 * kc][1]),
                packHalves(sacc[2 * kc][2], sacc[2 * kc][3]),
                packHalves(sacc[2 * kc + 1][0], sacc[2 * kc + 1][1]),
                packHalves(sacc[2 * kc + 1][2], sacc[2 * kc + 1][3]),
            };
            inline for (0..hd / 8) |nt| {
                const row = nt * 8 + g;
                const kcol = kc * 16 + qd * 2;
                const bf = [2]u32{ wordsOf(&fa_vt)[(row * vld + kcol) / 2], wordsOf(&fa_vt)[(row * vld + kcol + 8) / 2] };
                mma16816(&o[nt], pf, bf);
            }
        }
    }

    const sum_a = quadShuffleSum(l_a);
    const sum_b = quadShuffleSum(l_b);
    const inv_a: f32 = if (sum_a > 0) 1.0 / sum_a else 0;
    const inv_b: f32 = if (sum_b > 0) 1.0 / sum_b else 0;
    inline for (0..hd / 8) |nt| {
        const c = nt * 8 + qd * 2;
        if (ra < n_tokens) {
            const dst = out + (@as(usize, ra) * n_heads + h) * hd + c;
            dst[0] = o[nt][0] * inv_a;
            dst[1] = o[nt][1] * inv_a;
        }
        if (rb < n_tokens) {
            const dst = out + (@as(usize, rb) * n_heads + h) * hd + c;
            dst[0] = o[nt][2] * inv_b;
            dst[1] = o[nt][3] * inv_b;
        }
    }
}

var fa_k: [fa_positions * (128 + 8)]f16 addrspace(.shared) = undefined;
var fa_vt: [128 * (fa_positions + 8)]f16 addrspace(.shared) = undefined;

fn FlashKernel(comptime hd: u32) type {
    return struct {
        fn run(
            q: [*]const f32,
            kptrs: [*]const u64,
            vptrs: [*]const u64,
            layer_off: u64,
            slots: [*]const u32,
            poss: [*]const u32,
            out: [*]f32,
            ctx: u32,
            n_tokens: u32,
            n_heads: u32,
            group: u32,
            scale: f32,
        ) callconv(.kernel) void {
            flashAttention(hd, q, kptrs, vptrs, layer_off, slots, poss, out, ctx, n_tokens, n_heads, group, scale);
        }
    };
}

comptime {
    if (has_tensor_cores) {
        @export(&FlashKernel(64).run, .{ .name = "attention_tc_64" });
        @export(&FlashKernel(128).run, .{ .name = "attention_tc_128" });
    }
}

/// Storage types the device multiplies, in the order of the CPU's `vecdot.supported`.
pub const gemm_formats = [_]format.StorageType{
    .f32,   .f16,   .bf16,     .q4_0,    .q4_1,    .q5_0, .q5_1, .q8_0, .iq4_nl, .mxfp4, .nvfp4,
    .q2_k,  .q3_k,  .q4_k,     .q5_k,    .q6_k,    .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s,
    .iq1_s, .iq1_m, .iq4_xs,   .tq1_0,   .tq2_0,
};

/// Staged activations of the dense matrix-vector kernel, 16 byte aligned for vector reads.
var dn_x: [gemv_chunk]f32 align(16) addrspace(.shared) = undefined;

/// Single token product for dense weights (f32, f16, bf16): a warp owns a row, each lane takes 8 consecutive weights per
/// step with one 16 byte load (two for f32), so a warp reads 512 contiguous bytes of the row per step instead of decoding
/// one weight at a time through the block decoders.
inline fn gemvDense(comptime t: format.StorageType, w: [*]const u8, x: [*]const f32, y: [*]f32, k: u32, m: u32) void {
    const bytes_per: usize = comptime if (t == .f32) 4 else 2;
    const warp = tid() >> 5;
    const lane = tid() & 31;
    const row = bid() * gemv_rows + warp;
    const valid = row < m;
    const row_ptr = w + @as(usize, row) * k * bytes_per;
    var acc: f32 = 0;
    if (valid) {
        // No staging and no barrier: the activations are re-read from cache by every row, and a warp keeps four 16 byte
        // loads per lane in flight before it starts on the arithmetic.
        var off: u32 = lane * 8;
        while (off < k) : (off += 4 * 256) {
            var wv: [4][8]f32 = undefined;
            inline for (0..4) |u| {
                const o = off + u * 256;
                if (o < k) {
                    const src = row_ptr + @as(usize, o) * bytes_per;
                    if (comptime t == .f32) {
                        const v: *align(4) const [2]@Vector(4, f32) = @ptrCast(@alignCast(src));
                        inline for (0..4) |j| {
                            wv[u][j] = v[0][j];
                            wv[u][4 + j] = v[1][j];
                        }
                    } else {
                        const v: *align(2) const @Vector(4, u32) = @ptrCast(@alignCast(src));
                        const raw = v.*;
                        inline for (0..4) |j| {
                            if (comptime t == .f16) {
                                const pair = halfPair(raw[j]);
                                wv[u][2 * j] = pair[0];
                                wv[u][2 * j + 1] = pair[1];
                            } else {
                                wv[u][2 * j] = @bitCast(raw[j] << 16);
                                wv[u][2 * j + 1] = @bitCast(raw[j] & 0xffff0000);
                            }
                        }
                    }
                } else {
                    wv[u] = @splat(0);
                }
            }
            inline for (0..4) |u| {
                const o = off + u * 256;
                if (o < k) {
                    const xv: *align(16) const [2]@Vector(4, f32) = @ptrCast(@alignCast(x + o));
                    inline for (0..4) |j| {
                        acc += wv[u][j] * xv[0][j];
                        acc += wv[u][4 + j] * xv[1][j];
                    }
                }
            }
        }
    }
    var off: u32 = 16;
    while (off > 0) : (off >>= 1) acc += shflDown(acc, off);
    if (lane == 0 and valid) y[row] = acc;
}

fn GemvDenseKernel(comptime t: format.StorageType) type {
    return struct {
        fn run(w: [*]const u8, x: [*]const f32, y: [*]f32, k: u32, m: u32, n: u32) callconv(.kernel) void {
            _ = n;
            gemvDense(t, w, x, y, k, m);
        }
    };
}

fn GemvKernel(comptime t: format.StorageType) type {
    return struct {
        fn run(w: [*]const u8, x: [*]const f32, y: [*]f32, k: u32, m: u32, n: u32) callconv(.kernel) void {
            _ = n;
            gemvRow(t, w, x, y, k, m);
        }
    };
}

fn GemmTileKernel(comptime t: format.StorageType) type {
    return struct {
        fn run(w: [*]const u8, x: [*]const f32, y: [*]f32, k: u32, m: u32, n: u32) callconv(.kernel) void {
            gemmTile(t, w, x, y, k, m, n);
        }
    };
}

comptime {
    for (gemm_formats) |t| {
        if (t == .f32 or t == .f16 or t == .bf16) {
            @export(&GemvDenseKernel(t).run, .{ .name = "gemv_" ++ @tagName(t) });
        } else {
            @export(&GemvKernel(t).run, .{ .name = "gemv_" ++ @tagName(t) });
        }
        @export(&GemmTileKernel(t).run, .{ .name = "gemm_" ++ @tagName(t) });
    }
}

// -- normalization, rotary embedding, cache, attention, activation ------------------------------------------------

/// out = x * w / rms(x) for each of `n` rows of `dim`; one block per row. `out` may alias `x`.
fn rmsNormKernel(out: [*]f32, x: [*]const f32, w: [*]const f32, dim: u32, eps: f32) callconv(.kernel) void {
    const xr = x + @as(usize, bid()) * dim;
    const outr = out + @as(usize, bid()) * dim;
    var ss: f32 = 0;
    var i: u32 = tid();
    while (i < dim) : (i += bdim()) ss += xr[i] * xr[i];
    const total = blockSum(ss);
    const inv = 1.0 / @sqrt(total / @as(f32, @floatFromInt(dim)) + eps);
    i = tid();
    while (i < dim) : (i += bdim()) outr[i] = xr[i] * inv * w[i];
}

/// In place RMS norm of every `head_dim` slice of `x` (Qwen3 q/k norm), one block per head, weight shared by heads.
fn headNormKernel(x: [*]f32, w: [*]const f32, head_dim: u32, eps: f32) callconv(.kernel) void {
    const hv = x + @as(usize, bid()) * head_dim;
    var ss: f32 = 0;
    var i: u32 = tid();
    while (i < head_dim) : (i += bdim()) ss += hv[i] * hv[i];
    const total = blockSum(ss);
    const inv = 1.0 / @sqrt(total / @as(f32, @floatFromInt(head_dim)) + eps);
    i = tid();
    while (i < head_dim) : (i += bdim()) hv[i] = hv[i] * inv * w[i];
}

/// Rotates the first `2 * half` dims of every head of `x` in place. One block per head; the block's token is
/// `bid / heads_per_row`, whose `half` cos/sin values start at `cos + token * half`. `neox` != 0: pairs (i, i + half);
/// else (2i, 2i + 1).
fn ropeKernel(x: [*]f32, cos: [*]const f32, sin: [*]const f32, head_dim: u32, half: u32, neox: u32, heads_per_row: u32) callconv(.kernel) void {
    const hv = x + @as(usize, bid()) * head_dim;
    const row = bid() / heads_per_row;
    const cr = cos + @as(usize, row) * half;
    const sr = sin + @as(usize, row) * half;
    var i: u32 = tid();
    while (i < half) : (i += bdim()) {
        const c = cr[i];
        const s = sr[i];
        if (neox != 0) {
            const a = hv[i];
            const b = hv[i + half];
            hv[i] = a * c - b * s;
            hv[i + half] = a * s + b * c;
        } else {
            const a = hv[2 * i];
            const b = hv[2 * i + 1];
            hv[2 * i] = a * c - b * s;
            hv[2 * i + 1] = a * s + b * c;
        }
    }
}

/// x[i] += b[i % len] for i < total: a bias added to every row.
fn addBiasKernel(x: [*]f32, b: [*]const f32, len: u32, total: u32) callconv(.kernel) void {
    const i = globalId();
    if (i >= total) return;
    x[i] += b[i % len];
}

/// Stores each token's keys and values (f32, `kv_dim` each) into the f16 cache of its slot at its position.
/// `kptrs`/`vptrs` hold the base address of each slot's cache; `layer_off` is the byte offset of this layer in it.
/// Cache layout per layer: [kv head][position][head_dim].
fn kvWriteKernel(
    k: [*]const f32,
    v: [*]const f32,
    kptrs: [*]const u64,
    vptrs: [*]const u64,
    layer_off: u64,
    slots: [*]const u32,
    poss: [*]const u32,
    ctx: u32,
    head_dim: u32,
    kv_dim: u32,
    total: u32,
) callconv(.kernel) void {
    const i = globalId();
    if (i >= total) return;
    const row = i / kv_dim;
    const j = i % kv_dim;
    const head = j / head_dim;
    const d = j % head_dim;
    const ck: [*]f16 = @ptrFromInt(kptrs[slots[row]] + layer_off);
    const cv: [*]f16 = @ptrFromInt(vptrs[slots[row]] + layer_off);
    const dst = (@as(usize, head) * ctx + poss[row]) * head_dim + d;
    ck[dst] = @floatCast(k[i]);
    cv[dst] = @floatCast(v[i]);
}

/// Largest head width the attention kernel supports (its query lives in shared memory).
const max_head_dim = 256;
/// Positions per attention tile (one per thread).
const attn_tile = 128;
var attn_query: [max_head_dim]f32 addrspace(.shared) = undefined;
var attn_probs: [attn_tile]f32 addrspace(.shared) = undefined;
var attn_scores: [attn_tile]f32 addrspace(.shared) = undefined;

/// Running softmax state of one (token, head) over a range of positions: the maximum score, the sum of exponentials
/// relative to it, and the weighted sum of values for the output dimensions this thread owns.
const AttnState = struct { m: f32, l: f32, acc0: f32, acc1: f32 };

/// Online softmax attention of one (token, head) over positions `lo .. hi` of the token's slot, computed tile by tile so
/// no per position scores are stored. All threads of the block must call it together (it uses barriers). Launch with
/// 128 threads (`attn_tile`); head widths up to 256.
inline fn attendRange(
    q: [*]const f32,
    kptrs: [*]const u64,
    vptrs: [*]const u64,
    layer_off: u64,
    slots: [*]const u32,
    row: u32,
    h: u32,
    ctx: u32,
    head_dim: u32,
    n_heads: u32,
    group: u32,
    scale: f32,
    lo: u32,
    hi: u32,
) AttnState {
    const kvh = h / group;
    const ck: [*]const f16 = @ptrFromInt(kptrs[slots[row]] + layer_off);
    const cv: [*]const f16 = @ptrFromInt(vptrs[slots[row]] + layer_off);
    const kbase = ck + @as(usize, kvh) * ctx * head_dim;
    const vbase = cv + @as(usize, kvh) * ctx * head_dim;
    const qh = q + (@as(usize, row) * n_heads + h) * head_dim;

    var d: u32 = tid();
    while (d < head_dim) : (d += bdim()) attn_query[d] = qh[d];
    barSync();

    const neg_inf = -std.math.inf(f32);
    var st = AttnState{ .m = neg_inf, .l = 0, .acc0 = 0, .acc1 = 0 };
    // Each thread owns output dimensions tid and tid + 128 (head widths up to 256).
    const d0 = tid();
    const d1 = tid() + attn_tile;
    const warp = tid() >> 5;
    const lane = tid() & 31;

    var p0: u32 = lo;
    while (p0 < hi) : (p0 += attn_tile) {
        const p = p0 + tid();
        // Scores of the tile: a warp works on one position at a time with its lanes spread over the head dimensions, so
        // the key row is read coalesced (a thread per position would read rows a whole head width apart).
        var qi: u32 = warp;
        while (qi < attn_tile) : (qi += attn_tile / 32) {
            const pp = p0 + qi;
            var sc: f32 = neg_inf;
            if (pp < hi) {
                const kr = kbase + @as(usize, pp) * head_dim;
                var part: f32 = 0;
                var j: u32 = lane;
                while (j < head_dim) : (j += 32) part += attn_query[j] * @as(f32, @floatCast(kr[j]));
                sc = warpSum(part) * scale;
            }
            if (lane == 0) attn_scores[qi] = sc;
        }
        barSync();
        const s: f32 = attn_scores[tid()];
        const m_new = @max(st.m, blockMax(s));
        const corr = expf(st.m - m_new); // 0 on the first tile, where m is -inf
        const pj = if (p < hi) expf(s - m_new) else 0;
        attn_probs[tid()] = pj;
        const tile_sum = blockSum(pj); // also publishes attn_probs to the block
        st.l = st.l * corr + tile_sum;
        st.acc0 *= corr;
        st.acc1 *= corr;
        const live = @min(@as(u32, attn_tile), hi - p0);
        var jj: u32 = 0;
        while (jj < live) : (jj += 1) {
            const pr = attn_probs[jj];
            const vr = vbase + @as(usize, p0 + jj) * head_dim;
            if (d0 < head_dim) st.acc0 += pr * @as(f32, @floatCast(vr[d0]));
            if (d1 < head_dim) st.acc1 += pr * @as(f32, @floatCast(vr[d1]));
        }
        barSync(); // attn_probs and attn_scores are rewritten by the next tile
        st.m = m_new;
    }
    return st;
}

/// Causal attention, one block per (token, query head): softmax(q . K^T * scale) V over positions 0..pos of the token's
/// slot. Used when there are enough (token, head) blocks to fill the device (prompt processing).
fn attentionKernel(
    q: [*]const f32,
    kptrs: [*]const u64,
    vptrs: [*]const u64,
    layer_off: u64,
    slots: [*]const u32,
    poss: [*]const u32,
    out: [*]f32,
    ctx: u32,
    head_dim: u32,
    n_heads: u32,
    group: u32,
    scale: f32,
) callconv(.kernel) void {
    const row = bid() / n_heads;
    const h = bid() % n_heads;
    const st = attendRange(q, kptrs, vptrs, layer_off, slots, row, h, ctx, head_dim, n_heads, group, scale, 0, poss[row] + 1);
    const inv = 1.0 / st.l;
    const o = out + (@as(usize, row) * n_heads + h) * head_dim;
    if (tid() < head_dim) o[tid()] = st.acc0 * inv;
    if (tid() + attn_tile < head_dim) o[tid() + attn_tile] = st.acc1 * inv;
}

/// Split-KV attention, first half: block (token, head, split) covers one contiguous slice of the token's positions and
/// writes its partial softmax state to `part` as [acc (head_dim), m, l]. For few (token, head) blocks, as in decode,
/// slicing the context across blocks is what keeps the device busy.
fn attentionPartKernel(
    q: [*]const f32,
    kptrs: [*]const u64,
    vptrs: [*]const u64,
    layer_off: u64,
    slots: [*]const u32,
    poss: [*]const u32,
    part: [*]f32,
    ctx: u32,
    head_dim: u32,
    n_heads: u32,
    group: u32,
    scale: f32,
    splits: u32,
) callconv(.kernel) void {
    const split = bid() % splits;
    const rh = bid() / splits;
    const row = rh / n_heads;
    const h = rh % n_heads;
    const n = poss[row] + 1;
    const chunk = (n + splits - 1) / splits;
    const lo = @min(n, split * chunk);
    const hi = @min(n, lo + chunk);
    const st = attendRange(q, kptrs, vptrs, layer_off, slots, row, h, ctx, head_dim, n_heads, group, scale, lo, hi);
    const o = part + @as(usize, bid()) * (head_dim + 2);
    if (tid() < head_dim) o[tid()] = st.acc0;
    if (tid() + attn_tile < head_dim) o[tid() + attn_tile] = st.acc1;
    if (tid() == 0) {
        o[head_dim] = st.m;
        o[head_dim + 1] = st.l;
    }
}

var warp_m: [4]f32 addrspace(.shared) = undefined;
var warp_l: [4]f32 addrspace(.shared) = undefined;
var warp_acc: [4][256]f32 addrspace(.shared) = undefined;


inline fn halfPair(w: u32) [2]f32 {
    return .{ @floatCast(@as(f16, @bitCast(@as(u16, @truncate(w))))), @floatCast(@as(f16, @bitCast(@as(u16, @truncate(w >> 16))))) };
}

/// Split-KV attention for decode, first half, with a warp per 32 positions: each lane scores one position by itself (a
/// serial dot product over the head width, no reductions), the warp does one softmax step for the 32 scores, and the
/// weighted sum of values has the lanes own slices of the head width. Compared with `attentionPartKernel` there is no
/// block wide barrier inside the loop: the four warps of a block walk their own tiles and are merged once at the end.
/// Writes the same [acc (head_dim), max, sum] state per (token, head, split). Head widths that are a multiple of 8, up to
/// 256.
fn attentionPartWarpKernel(
    q: [*]const f32,
    kptrs: [*]const u64,
    vptrs: [*]const u64,
    layer_off: u64,
    slots: [*]const u32,
    poss: [*]const u32,
    part: [*]f32,
    ctx: u32,
    head_dim: u32,
    n_heads: u32,
    group: u32,
    scale: f32,
    splits: u32,
) callconv(.kernel) void {
    const split = bid() % splits;
    const rh = bid() / splits;
    const row = rh / n_heads;
    const h = rh % n_heads;
    const npos = poss[row] + 1;
    const chunk = (npos + splits - 1) / splits;
    const lo = @min(npos, split * chunk);
    const hi = @min(npos, lo + chunk);
    const kvh = h / group;
    const ck: [*]const f16 = @ptrFromInt(kptrs[slots[row]] + layer_off);
    const cv: [*]const f16 = @ptrFromInt(vptrs[slots[row]] + layer_off);
    const kbase = ck + @as(usize, kvh) * ctx * head_dim;
    const vbase = cv + @as(usize, kvh) * ctx * head_dim;
    const qh = q + (@as(usize, row) * n_heads + h) * head_dim;

    var d: u32 = tid();
    while (d < head_dim) : (d += bdim()) attn_query[d] = qh[d] * scale;
    barSync();

    const warp = tid() >> 5;
    const lane = tid() & 31;
    const dpl = head_dim / 32; // dims of the value row each lane owns, contiguous: lane * dpl .. lane * dpl + dpl
    const neg_inf = -std.math.inf(f32);
    var m: f32 = neg_inf;
    var l: f32 = 0;
    var acc: [8]f32 = @splat(0);

    var p0: u32 = lo + warp * 32;
    while (p0 < hi) : (p0 += 128) {
        const p = p0 + lane;
        var sc: f32 = neg_inf;
        if (p < hi) {
            const kr: [*]const @Vector(4, u32) = @ptrCast(@alignCast(kbase + @as(usize, p) * head_dim));
            // Eight halves per load and two independent partial sums, so the loads and the arithmetic overlap.
            var s0: f32 = 0;
            var s1: f32 = 0;
            var i: u32 = 0;
            while (i < head_dim / 8) : (i += 1) {
                const kv = kr[i];
                const q4 = i * 8;
                inline for (0..4) |u| {
                    const pair = halfPair(kv[u]);
                    const t = attn_query[q4 + 2 * u] * pair[0] + attn_query[q4 + 2 * u + 1] * pair[1];
                    if (u % 2 == 0) s0 += t else s1 += t;
                }
            }
            sc = s0 + s1;
        }
        var mx = sc;
        inline for (.{ 16, 8, 4, 2, 1 }) |o| mx = @max(mx, shflXor(mx, o));
        const m_new = @max(m, mx);
        const corr = expf(m - m_new);
        const pj = if (p < hi) expf(sc - m_new) else 0;
        var ps = pj;
        inline for (.{ 16, 8, 4, 2, 1 }) |o| ps += shflXor(ps, o);
        l = l * corr + ps;
        inline for (0..8) |i| acc[i] *= corr;
        m = m_new;
        const live = @min(@as(u32, 32), hi - p0);
        var j: u32 = 0;
        if (dpl == 4) {
            // Four positions per step: the value loads of all four are in flight before the first is used.
            while (j + 4 <= live) : (j += 4) {
                var vv: [4]@Vector(2, u32) = undefined;
                var pw: [4]f32 = undefined;
                inline for (0..4) |u| {
                    pw[u] = shflIdx(pj, j + @as(u32, u));
                    const vr: *const @Vector(2, u32) = @ptrCast(@alignCast(vbase + @as(usize, p0 + j + @as(u32, u)) * head_dim + lane * 4));
                    vv[u] = vr.*;
                }
                inline for (0..4) |u| {
                    const a0 = halfPair(vv[u][0]);
                    const a1 = halfPair(vv[u][1]);
                    acc[0] += pw[u] * a0[0];
                    acc[1] += pw[u] * a0[1];
                    acc[2] += pw[u] * a1[0];
                    acc[3] += pw[u] * a1[1];
                }
            }
        }
        while (j < live) : (j += 1) {
            const pw = shflIdx(pj, j);
            const vr: [*]const u32 = @ptrCast(@alignCast(vbase + @as(usize, p0 + j) * head_dim + lane * dpl));
            inline for (0..4) |u| {
                if (u * 2 < dpl) {
                    const vv = halfPair(vr[u]);
                    acc[2 * u] += pw * vv[0];
                    acc[2 * u + 1] += pw * vv[1];
                }
            }
        }
    }

    if (lane == 0) {
        warp_m[warp] = m;
        warp_l[warp] = l;
    }
    inline for (0..8) |i| {
        if (i < dpl) warp_acc[warp][lane * dpl + i] = acc[i];
    }
    barSync();
    var big: f32 = neg_inf;
    for (warp_m) |wm| big = @max(big, wm);
    const o = part + @as(usize, bid()) * (head_dim + 2);
    var total: f32 = 0;
    if (big != neg_inf) {
        for (warp_m, warp_l) |wm, wl| total += expf(wm - big) * wl;
    }
    d = tid();
    while (d < head_dim) : (d += bdim()) {
        var v: f32 = 0;
        if (big != neg_inf) {
            for (warp_m, 0..) |wm, w| v += expf(wm - big) * warp_acc[w][d];
        }
        o[d] = v;
    }
    if (tid() == 0) {
        o[head_dim] = big;
        o[head_dim + 1] = total;
    }
}

/// Split-KV attention, second half: merges the `splits` partial states of each (token, head) exactly (rescaling each by
/// its maximum) and writes the normalized output. One block per (token, head).
fn attentionCombineKernel(part: [*]const f32, out: [*]f32, head_dim: u32, splits: u32) callconv(.kernel) void {
    const rh = bid();
    const base = part + @as(usize, rh) * splits * (head_dim + 2);
    var gm: f32 = -std.math.inf(f32);
    var s: u32 = 0;
    while (s < splits) : (s += 1) gm = @max(gm, base[s * (head_dim + 2) + head_dim]);
    var l: f32 = 0;
    s = 0;
    while (s < splits) : (s += 1) {
        const o = base + s * (head_dim + 2);
        l += o[head_dim + 1] * expf(o[head_dim] - gm);
    }
    var d: u32 = tid();
    while (d < head_dim) : (d += bdim()) {
        var acc: f32 = 0;
        s = 0;
        while (s < splits) : (s += 1) {
            const o = base + s * (head_dim + 2);
            acc += o[d] * expf(o[head_dim] - gm);
        }
        out[@as(usize, rh) * head_dim + d] = acc / l;
    }
}

/// gate[i] = silu(gate[i]) * up[i]
/// dst[i] = f16(src[i]) for pairs of floats; the matrix kernels on tensor cores take their activations as halves.
fn toHalfKernel(dst: [*]u32, src: [*]const f32, pairs: u32) callconv(.kernel) void {
    const i = globalId();
    if (i < pairs) dst[i] = packHalves(src[2 * i], src[2 * i + 1]);
}

fn swigluKernel(gate: [*]f32, up: [*]const f32, n: u32) callconv(.kernel) void {
    const i = globalId();
    if (i >= n) return;
    const g = gate[i];
    gate[i] = (g / (1.0 + expf(-g))) * up[i];
}

/// x[i] += y[i]
fn addKernel(x: [*]f32, y: [*]const f32, n: u32) callconv(.kernel) void {
    const i = globalId();
    if (i >= n) return;
    x[i] += y[i];
}

comptime {
    @export(&rmsNormKernel, .{ .name = "rms_norm" });
    @export(&headNormKernel, .{ .name = "head_norm" });
    @export(&ropeKernel, .{ .name = "rope" });
    @export(&addBiasKernel, .{ .name = "add_bias" });
    @export(&kvWriteKernel, .{ .name = "kv_write" });
    @export(&attentionKernel, .{ .name = "attention" });
    @export(&attentionPartKernel, .{ .name = "attention_part" });
    @export(&attentionPartWarpKernel, .{ .name = "attention_part_warp" });
    @export(&attentionCombineKernel, .{ .name = "attention_combine" });
    @export(&swigluKernel, .{ .name = "swiglu" });
    @export(&toHalfKernel, .{ .name = "to_half" });
    @export(&addKernel, .{ .name = "add" });
}
