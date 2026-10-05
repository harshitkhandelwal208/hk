const std = @import("std");
const hk = @import("hk");

// =========================================================================
// Benchmark Kernels: Baseline (Before) vs Optimized (After)
// =========================================================================

// --- 1. GEMM Baseline (Scalar, no zero-skip, no SIMD) ---
fn gemmF32_baseline(A: []const f32, B: []const f32, C: []f32, M: usize, K: usize, N: usize) void {
    @memset(C, 0.0);
    for (0..M) |m| {
        const c_row = C[m * N .. (m + 1) * N];
        for (0..K) |k| {
            const a_val = A[m * K + k];
            const b_row = B[k * N .. (k + 1) * N];
            for (0..N) |n| {
                c_row[n] += a_val * b_row[n];
            }
        }
    }
}

// --- 2. RoPE Baseline (Calculates pow, cos, sin per head) ---
fn rope_baseline(q: []f32, n_heads: usize, head_dim: usize, pos: usize, theta: f32) void {
    const half_dim = head_dim / 2;
    for (0..n_heads) |h| {
        const q_head = q[h * head_dim .. (h + 1) * head_dim];
        for (0..half_dim) |i| {
            const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
            const val = @as(f32, @floatFromInt(pos)) * freq;
            const cos_val = @cos(val);
            const sin_val = @sin(val);
            const v0 = q_head[2 * i];
            const v1 = q_head[2 * i + 1];
            q_head[2 * i] = v0 * cos_val - v1 * sin_val;
            q_head[2 * i + 1] = v0 * sin_val + v1 * cos_val;
        }
    }
}

// RoPE Optimized (Step-level stack cache)
fn rope_optimized(q: []f32, n_heads: usize, head_dim: usize, pos: usize, theta: f32) void {
    const half_dim = head_dim / 2;
    var cos_table: [256]f32 = undefined;
    var sin_table: [256]f32 = undefined;
    const table_len = @min(half_dim, 256);

    for (0..table_len) |i| {
        const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
        const val = @as(f32, @floatFromInt(pos)) * freq;
        cos_table[i] = @cos(val);
        sin_table[i] = @sin(val);
    }

    for (0..n_heads) |h| {
        const q_head = q[h * head_dim .. (h + 1) * head_dim];
        for (0..table_len) |i| {
            const cos_val = cos_table[i];
            const sin_val = sin_table[i];
            const v0 = q_head[2 * i];
            const v1 = q_head[2 * i + 1];
            q_head[2 * i] = v0 * cos_val - v1 * sin_val;
            q_head[2 * i + 1] = v0 * sin_val + v1 * cos_val;
        }
    }
}

// --- 3. FP8 E4M3 Dequantization Baseline (Dynamic arithmetic) ---
fn dequantizeFP8_baseline(in_bytes: []const u8, count: usize, out: []f32) void {
    const limit = @min(@min(in_bytes.len, count), out.len);
    for (0..limit) |i| {
        const b = in_bytes[i];
        const sign: f32 = if ((b & 0x80) != 0) -1.0 else 1.0;
        const exp = (b >> 3) & 0x0F;
        const mant = b & 0x07;
        if (exp == 0) {
            out[i] = sign * 0.015625 * (@as(f32, @floatFromInt(mant)) / 8.0);
        } else if (exp == 15 and mant == 7) {
            out[i] = std.math.nan(f32);
        } else {
            const exp_val = @as(f32, @floatFromInt(exp)) - 7.0;
            out[i] = sign * std.math.pow(f32, 2.0, exp_val) * (1.0 + @as(f32, @floatFromInt(mant)) / 8.0);
        }
    }
}

// --- 4. Softmax Baseline (Full exp without pruning) ---
fn softmax_baseline(logits: []const f32, out_probs: []f32) void {
    const len = @min(logits.len, out_probs.len);
    if (len == 0) return;
    var max_val: f32 = logits[0];
    for (1..len) |i| {
        if (logits[i] > max_val) max_val = logits[i];
    }
    var sum: f32 = 0.0;
    for (0..len) |i| {
        const e = @exp(logits[i] - max_val);
        out_probs[i] = e;
        sum += e;
    }
    const inv_sum = 1.0 / sum;
    for (0..len) |i| {
        out_probs[i] *= inv_sum;
    }
}

// --- 5. GEMV Baseline (Single row at a time) ---
fn gemvQ8_0_baseline(W_bytes: []const u8, x: []const f32, y: []f32, M: usize, K: usize) void {
    const safe_M = @min(M, y.len);
    const blocks_per_row = K / 32;
    const row_bytes_len = blocks_per_row * 34;
    for (0..safe_M) |r| {
        const row_bytes = W_bytes[r * row_bytes_len .. (r + 1) * row_bytes_len];
        var row_sum: f32 = 0.0;
        for (0..blocks_per_row) |b| {
            const block_bytes = row_bytes[b * 34 .. (b + 1) * 34];
            const d_bits = std.mem.readInt(u16, block_bytes[0..2], .little);
            const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
            const qs: [*]const i8 = @ptrCast(block_bytes[2..34].ptr);
            var dot: f32 = 0.0;
            for (0..32) |k| {
                dot += @as(f32, @floatFromInt(qs[k])) * x[b * 32 + k];
            }
            row_sum += dot * d;
        }
        y[r] = row_sum;
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = std.Options.debug_io;

    std.debug.print("\n" ++ &@as([78]u8, @splat('=')) ++ "\n", .{});
    std.debug.print("  EMPIRICAL KERNEL BENCHMARK: BASELINE (BEFORE) vs OPTIMIZED (AFTER)\n", .{});
    std.debug.print(&@as([78]u8, @splat('=')) ++ "\n", .{});

    // -------------------------------------------------------------
    // Benchmark 1: GEMM Matrix Multiply (M=64, K=512, N=512)
    // -------------------------------------------------------------
    {
        const M = 64;
        const K = 512;
        const N = 512;
        const A = try allocator.alloc(f32, M * K);
        defer allocator.free(A);
        const B = try allocator.alloc(f32, K * N);
        defer allocator.free(B);
        const C_base = try allocator.alloc(f32, M * N);
        defer allocator.free(C_base);
        const C_opt = try allocator.alloc(f32, M * N);
        defer allocator.free(C_opt);

        // 30% sparsity to mirror activation patterns
        for (0..M * K) |i| A[i] = if (i % 3 == 0) 0.0 else 1.0;
        for (0..K * N) |i| B[i] = 0.5;

        // Warmup
        gemmF32_baseline(A, B, C_base, M, K, N);
        hk.tensor_ops.gemmF32(A, B, C_opt, M, K, N);

        const iters = 10;
        const t0 = std.Io.Clock.Timestamp.now(io, .awake);
        for (0..iters) |_| gemmF32_baseline(A, B, C_base, M, K, N);
        const dur_base = @as(f64, @floatFromInt(t0.untilNow(io).raw.nanoseconds)) / (@as(f64, @floatFromInt(iters)) * 1e6);

        const t1 = std.Io.Clock.Timestamp.now(io, .awake);
        for (0..iters) |_| hk.tensor_ops.gemmF32(A, B, C_opt, M, K, N);
        const dur_opt = @as(f64, @floatFromInt(t1.untilNow(io).raw.nanoseconds)) / (@as(f64, @floatFromInt(iters)) * 1e6);

        const speedup = dur_base / dur_opt;
        std.debug.print("[1] GEMM (64x512x512):      Baseline = {d:.2} ms | Optimized = {d:.2} ms | Speedup = {d:.2}x\n", .{ dur_base, dur_opt, speedup });
    }

    // -------------------------------------------------------------
    // Benchmark 2: RoPE Kernel Time (32 heads, head_dim=64)
    // -------------------------------------------------------------
    {
        const n_heads = 32;
        const head_dim = 64;
        const q = try allocator.alloc(f32, n_heads * head_dim);
        defer allocator.free(q);
        @memset(q, 1.0);

        const iters = 2000;
        const t0 = std.Io.Clock.Timestamp.now(io, .awake);
        for (0..iters) |p| rope_baseline(q, n_heads, head_dim, p, 10000.0);
        const dur_base = @as(f64, @floatFromInt(t0.untilNow(io).raw.nanoseconds)) / (@as(f64, @floatFromInt(iters)) * 1e3);

        const t1 = std.Io.Clock.Timestamp.now(io, .awake);
        for (0..iters) |p| rope_optimized(q, n_heads, head_dim, p, 10000.0);
        const dur_opt = @as(f64, @floatFromInt(t1.untilNow(io).raw.nanoseconds)) / (@as(f64, @floatFromInt(iters)) * 1e3);

        const speedup = dur_base / dur_opt;
        std.debug.print("[2] RoPE (32 heads, d=64):   Baseline = {d:.2} us | Optimized = {d:.2} us | Speedup = {d:.2}x\n", .{ dur_base, dur_opt, speedup });
    }

    // -------------------------------------------------------------
    // Benchmark 3: FP8 E4M3 Dequantization (65,536 elements)
    // -------------------------------------------------------------
    {
        const count = 65536;
        const in_bytes = try allocator.alloc(u8, count);
        defer allocator.free(in_bytes);
        const out_base = try allocator.alloc(f32, count);
        defer allocator.free(out_base);
        const out_opt = try allocator.alloc(f32, count);
        defer allocator.free(out_opt);

        for (0..count) |i| in_bytes[i] = @intCast(i % 256);

        const iters = 100;
        const t0 = std.Io.Clock.Timestamp.now(io, .awake);
        for (0..iters) |_| dequantizeFP8_baseline(in_bytes, count, out_base);
        const dur_base = @as(f64, @floatFromInt(t0.untilNow(io).raw.nanoseconds)) / (@as(f64, @floatFromInt(iters)) * 1e6);

        const t1 = std.Io.Clock.Timestamp.now(io, .awake);
        for (0..iters) |_| hk.quantization.dequantizeFP8_E4M3(in_bytes, count, out_opt);
        const dur_opt = @as(f64, @floatFromInt(t1.untilNow(io).raw.nanoseconds)) / (@as(f64, @floatFromInt(iters)) * 1e6);

        const speedup = dur_base / dur_opt;
        std.debug.print("[3] FP8 Dequant (64K elem):  Baseline = {d:.2} ms | Optimized = {d:.2} ms | Speedup = {d:.2}x\n", .{ dur_base, dur_opt, speedup });
    }

    // -------------------------------------------------------------
    // Benchmark 4: Softmax Exponential Pruning (Long context seq=4096)
    // -------------------------------------------------------------
    {
        const seq_len = 4096;
        const logits = try allocator.alloc(f32, seq_len);
        defer allocator.free(logits);
        const probs_base = try allocator.alloc(f32, seq_len);
        defer allocator.free(probs_base);
        const probs_opt = try allocator.alloc(f32, seq_len);
        defer allocator.free(probs_opt);

        // Realistic attention logit decay (70% tail < -20)
        for (0..seq_len) |i| logits[i] = if (i > 1024) -50.0 else -@as(f32, @floatFromInt(i % 15));

        const iters = 2000;
        const t0 = std.Io.Clock.Timestamp.now(io, .awake);
        for (0..iters) |_| softmax_baseline(logits, probs_base);
        const dur_base = @as(f64, @floatFromInt(t0.untilNow(io).raw.nanoseconds)) / (@as(f64, @floatFromInt(iters)) * 1e3);

        const t1 = std.Io.Clock.Timestamp.now(io, .awake);
        for (0..iters) |_| hk.tensor_ops.softmaxF32(logits, probs_opt);
        const dur_opt = @as(f64, @floatFromInt(t1.untilNow(io).raw.nanoseconds)) / (@as(f64, @floatFromInt(iters)) * 1e3);

        const speedup = dur_base / dur_opt;
        std.debug.print("[4] Softmax (seq_len=4096):  Baseline = {d:.2} us | Optimized = {d:.2} us | Speedup = {d:.2}x\n", .{ dur_base, dur_opt, speedup });
    }

    // -------------------------------------------------------------
    // Benchmark 5: GEMV Q8_0 Register Tiling (M=128, K=512)
    // -------------------------------------------------------------
    {
        const M = 128;
        const K = 512;
        const num_blocks = M * (K / 32);
        const bytes_len = num_blocks * 34;
        const W_bytes = try allocator.alloc(u8, bytes_len);
        defer allocator.free(W_bytes);
        @memset(W_bytes, 1);
        const x = try allocator.alloc(f32, K);
        defer allocator.free(x);
        @memset(x, 0.5);
        const y_base = try allocator.alloc(f32, M);
        defer allocator.free(y_base);
        const y_opt = try allocator.alloc(f32, M);
        defer allocator.free(y_opt);

        const iters = 500;
        const t0 = std.Io.Clock.Timestamp.now(io, .awake);
        for (0..iters) |_| gemvQ8_0_baseline(W_bytes, x, y_base, M, K);
        const dur_base = @as(f64, @floatFromInt(t0.untilNow(io).raw.nanoseconds)) / (@as(f64, @floatFromInt(iters)) * 1e3);

        const t1 = std.Io.Clock.Timestamp.now(io, .awake);
        for (0..iters) |_| hk.tensor_ops.gemvQ8_0(W_bytes, x, null, y_opt, M, K);
        const dur_opt = @as(f64, @floatFromInt(t1.untilNow(io).raw.nanoseconds)) / (@as(f64, @floatFromInt(iters)) * 1e3);

        const speedup = dur_base / dur_opt;
        std.debug.print("[5] GEMV Q8_0 (128x512):     Baseline = {d:.2} us | Optimized = {d:.2} us | Speedup = {d:.2}x\n", .{ dur_base, dur_opt, speedup });
    }
    std.debug.print(&@as([78]u8, @splat('=')) ++ "\n\n", .{});
    return;
}
