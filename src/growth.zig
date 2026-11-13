const std = @import("std");

/// Lightweight fast XorShift64 pseudo-random number generator for noise generation
pub const Prng = struct {
    state: u64,

    pub fn init(seed: u64) Prng {
        return .{ .state = if (seed == 0) 0x853c49e6748fea9b else seed };
    }

    pub fn next(self: *Prng) u64 {
        var x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        return x;
    }

    /// Box-Muller transform for standard normal float32
    pub fn nextGaussian(self: *Prng, std_dev: f32) f32 {
        const u_val1 = @as(f32, @floatFromInt(self.next() & 0xFFFFFF)) / 16777216.0 + 1e-7;
        const u_val2 = @as(f32, @floatFromInt(self.next() & 0xFFFFFF)) / 16777216.0;
        const r = @sqrt(-2.0 * @log(u_val1));
        const theta = 2.0 * std.math.pi * u_val2;
        return r * @cos(theta) * std_dev;
    }
};

/// Native Zig implementation of Net2WiderNet:
/// Expands layer1 out_features from old_out to new_out, and layer2 in_features from old_out to new_out.
/// Preserves exact mathematical output: f_wider(x) == f_orig(x).
pub fn net2Wider(
    // Layer 1 inputs & outputs
    w_in_old: []const f32, // [old_out, in_f]
    b_in_old: ?[]const f32, // [old_out]
    w_in_new: []f32, // [new_out, in_f]
    b_in_new: ?[]f32, // [new_out]
    // Layer 2 inputs & outputs (outgoing)
    w_out_old: ?[]const f32, // [out_f, old_out]
    w_out_new: ?[]f32, // [out_f, new_out]
    // Dimensions
    old_out: usize,
    new_out: usize,
    in_f: usize,
    out_f: usize,
    noise_std: f32,
    seed: u64,
) !void {
    if (new_out < old_out) return error.InvalidDimension;
    const added = new_out - old_out;

    var prng = Prng.init(seed);

    // Compute mapping g(j) and replication counts c(k)
    // Allocator-free on-stack buffer if reasonably sized, or heap
    var g_buf: [4096]usize = undefined;
    var c_buf: [4096]usize = undefined;

    const g_allocated = if (new_out > 4096) try std.heap.page_allocator.alloc(usize, new_out) else null;
    defer if (g_allocated) |buf| std.heap.page_allocator.free(buf);
    const g = if (g_allocated) |buf| buf else g_buf[0..new_out];

    const c_allocated = if (old_out > 4096) try std.heap.page_allocator.alloc(usize, old_out) else null;
    defer if (c_allocated) |buf| std.heap.page_allocator.free(buf);
    const c = if (c_allocated) |buf| buf else c_buf[0..old_out];

    @memset(c, 0);

    // Original units map to themselves
    for (0..old_out) |i| {
        g[i] = i;
        c[i] += 1;
    }

    // Added units map to pseudo-randomly selected existing units
    for (0..added) |i| {
        const dst = old_out + i;
        const src = prng.next() % old_out;
        g[dst] = src;
        c[src] += 1;
    }

    // 1. Expand Layer 1 incoming weights: W_in_new[j, :] = W_in_old[g(j), :]
    for (0..new_out) |j| {
        const src = g[j];
        const src_row = w_in_old[src * in_f .. (src + 1) * in_f];
        const dst_row = w_in_new[j * in_f .. (j + 1) * in_f];
        @memcpy(dst_row, src_row);

        if (b_in_old) |b_old| {
            if (b_in_new) |b_new| {
                b_new[j] = b_old[src];
            }
        }
    }

    // 2. Expand Layer 2 outgoing weights: W_out_new[:, j] = (1 / c(g(j))) * W_out_old[:, g(j)] + noise
    if (w_out_old != null and w_out_new != null) {
        const old_out_mat = w_out_old.?;
        const new_out_mat = w_out_new.?;

        for (0..out_f) |r| {
            for (0..new_out) |j| {
                const src = g[j];
                const factor: f32 = 1.0 / @as(f32, @floatFromInt(c[src]));
                const base_val = old_out_mat[r * old_out + src] * factor;

                var final_val = base_val;
                if (noise_std > 0 and j >= old_out) {
                    final_val += prng.nextGaussian(noise_std);
                }

                new_out_mat[r * new_out + j] = final_val;
            }
        }
    }
}

/// Native Net2DeeperNet: initializes intermediate weight matrix of [dim, dim] to Identity (I)
/// and bias to 0, ensuring f(x) = x upon insertion.
pub fn net2Deeper(
    weights: []f32, // [dim, dim]
    bias: ?[]f32, // [dim]
    dim: usize,
) void {
    @memset(weights, 0.0);
    for (0..dim) |i| {
        weights[i * dim + i] = 1.0;
    }
    if (bias) |b| {
        @memset(b, 0.0);
    }
}

/// Native Net2WiderNet transformation for modern SwiGLU Transformer MLPs (LLaMA, SmolLM, Mistral, Qwen).
/// Mathematically preserves the exact output:
/// y = W_down * (SiLU(W_gate * x) * (W_up * x))
/// When zero_init is true, base channels are unchanged and new output channels are zeroed,
/// guaranteeing exact bitwise function preservation: ||f_wider(x) - f(x)||_inf == 0.0.
pub fn net2WiderSwiGLU(
    // Layer inputs
    w_gate_old: []const f32, // [old_inter, in_f]
    b_gate_old: ?[]const f32, // [old_inter]
    w_up_old: []const f32, // [old_inter, in_f]
    b_up_old: ?[]const f32, // [old_inter]
    w_down_old: []const f32, // [out_f, old_inter]
    b_down_old: ?[]const f32, // [out_f]
    // Layer outputs
    w_gate_new: []f32, // [new_inter, in_f]
    b_gate_new: ?[]f32, // [new_inter]
    w_up_new: []f32, // [new_inter, in_f]
    b_up_new: ?[]f32, // [new_inter]
    w_down_new: []f32, // [out_f, new_inter]
    b_down_new: ?[]f32, // [out_f]
    // Dimensions
    old_inter: usize,
    new_inter: usize,
    in_f: usize,
    out_f: usize,
    zero_init: bool,
    noise_std: f32,
    seed: u64,
) !void {
    if (new_inter < old_inter) return error.InvalidDimension;
    const added = new_inter - old_inter;

    var prng = Prng.init(seed);

    if (zero_init) {
        // --- Zero-Init Mode (Exact 0.00 Deviation & Anti-Catastrophic Forgetting) ---
        // 1. Copy base channels intact (factor = 1.0, NO division!)
        @memcpy(w_gate_new[0 .. old_inter * in_f], w_gate_old[0 .. old_inter * in_f]);
        @memcpy(w_up_new[0 .. old_inter * in_f], w_up_old[0 .. old_inter * in_f]);

        if (b_gate_old) |b_g_old| {
            if (b_gate_new) |b_g_new| @memcpy(b_g_new[0..old_inter], b_g_old[0..old_inter]);
        }
        if (b_up_old) |b_u_old| {
            if (b_up_new) |b_u_new| @memcpy(b_u_new[0..old_inter], b_u_old[0..old_inter]);
        }

        // 2. Initialize new incoming channels (Gaussian noise or random weights)
        for (old_inter..new_inter) |j| {
            for (0..in_f) |k| {
                w_gate_new[j * in_f + k] = prng.nextGaussian(0.02);
                w_up_new[j * in_f + k] = prng.nextGaussian(0.02);
            }
            if (b_gate_new) |b_g_new| b_g_new[j] = 0.0;
            if (b_up_new) |b_u_new| b_u_new[j] = 0.0;
        }

        // 3. Populate down_proj: base columns copied, new columns initialized to EXACT ZERO
        for (0..out_f) |r| {
            // copy base columns
            @memcpy(
                w_down_new[r * new_inter .. r * new_inter + old_inter],
                w_down_old[r * old_inter .. (r + 1) * old_inter],
            );
            // zero out newly expanded columns
            @memset(w_down_new[r * new_inter + old_inter .. (r + 1) * new_inter], 0.0);
        }

        if (b_down_old) |b_d_old| {
            if (b_down_new) |b_d_new| @memcpy(b_d_new[0..out_f], b_d_old[0..out_f]);
        }
    } else {
        // --- Replication Mode (Net2Wider standard capacity split) ---
        var g_buf: [8192]usize = undefined;
        var c_buf: [8192]usize = undefined;

        const g_allocated = if (new_inter > 8192) try std.heap.page_allocator.alloc(usize, new_inter) else null;
        defer if (g_allocated) |buf| std.heap.page_allocator.free(buf);
        const g = if (g_allocated) |buf| buf else g_buf[0..new_inter];

        const c_allocated = if (old_inter > 8192) try std.heap.page_allocator.alloc(usize, old_inter) else null;
        defer if (c_allocated) |buf| std.heap.page_allocator.free(buf);
        const c = if (c_allocated) |buf| buf else c_buf[0..old_inter];

        @memset(c, 0);

        for (0..old_inter) |i| {
            g[i] = i;
            c[i] += 1;
        }
        for (0..added) |i| {
            const dst = old_inter + i;
            const src = prng.next() % old_inter;
            g[dst] = src;
            c[src] += 1;
        }

        for (0..new_inter) |j| {
            const src = g[j];
            @memcpy(w_gate_new[j * in_f .. (j + 1) * in_f], w_gate_old[src * in_f .. (src + 1) * in_f]);
            @memcpy(w_up_new[j * in_f .. (j + 1) * in_f], w_up_old[src * in_f .. (src + 1) * in_f]);
            if (b_gate_old) |b_g_old| {
                if (b_gate_new) |b_g_new| b_g_new[j] = b_g_old[src];
            }
            if (b_up_old) |b_u_old| {
                if (b_up_new) |b_u_new| b_u_new[j] = b_u_old[src];
            }
        }

        for (0..out_f) |r| {
            for (0..new_inter) |j| {
                const src = g[j];
                const factor: f32 = 1.0 / @as(f32, @floatFromInt(c[src]));
                var val = w_down_old[r * old_inter + src] * factor;
                if (noise_std > 0 and j >= old_inter) {
                    val += prng.nextGaussian(noise_std);
                }
                w_down_new[r * new_inter + j] = val;
            }
        }

        if (b_down_old) |b_d_old| {
            if (b_down_new) |b_d_new| @memcpy(b_d_new[0..out_f], b_d_old[0..out_f]);
        }
    }
}

/// Native Dynamic Vocabulary Expansion:
/// Expands embedding table and LM head [old_vocab, hidden_dim] -> [new_vocab, hidden_dim].
/// Preserves exact weights for 0..old_vocab-1, ensuring f_new(token_i) == f_old(token_i).
pub fn expandVocab(
    embed_old: []const f32, // [old_vocab, hidden_dim]
    embed_new: []f32, // [new_vocab, hidden_dim]
    lm_head_old: ?[]const f32, // [old_vocab, hidden_dim]
    lm_head_new: ?[]f32, // [new_vocab, hidden_dim]
    old_vocab: usize,
    new_vocab: usize,
    hidden_dim: usize,
    seed: u64,
) !void {
    if (new_vocab < old_vocab) return error.InvalidDimension;
    var prng = Prng.init(seed);

    // 1. Copy base embedding rows (0..old_vocab-1) exactly
    const base_bytes = old_vocab * hidden_dim;
    @memcpy(embed_new[0..base_bytes], embed_old[0..base_bytes]);

    // Compute mean and std of base embeddings for matching new token distribution
    var sum: f32 = 0.0;
    for (0..base_bytes) |i| sum += embed_old[i];
    const mean = sum / @as(f32, @floatFromInt(base_bytes));

    var var_sum: f32 = 0.0;
    for (0..base_bytes) |i| {
        const diff = embed_old[i] - mean;
        var_sum += diff * diff;
    }
    const std_dev = @sqrt(var_sum / @as(f32, @floatFromInt(base_bytes)));

    // 2. Initialize new embedding slots (old_vocab..new_vocab-1) with matched statistics
    for (old_vocab..new_vocab) |v| {
        for (0..hidden_dim) |d| {
            const noise = prng.nextGaussian(0.02);
            embed_new[v * hidden_dim + d] = mean + noise * std_dev;
        }
    }

    // 3. If distinct LM head provided, copy base rows and initialize new rows
    if (lm_head_old != null and lm_head_new != null) {
        const head_old = lm_head_old.?;
        const head_new = lm_head_new.?;

        @memcpy(head_new[0..base_bytes], head_old[0..base_bytes]);
        for (old_vocab..new_vocab) |v| {
            for (0..hidden_dim) |d| {
                head_new[v * hidden_dim + d] = prng.nextGaussian(0.02);
            }
        }
    }
}

/// In-place Plasticity Isolation on 2D Matrix Rows:
/// Zeroes out gradients for base rows 0..cutoff-1 (e.g. embed_tokens, gate_proj, up_proj)
pub fn applyPlasticityMaskRows(
    grad_matrix: []f32,
    cutoff_rows: usize,
    cols: usize,
) void {
    const end_idx = @min(cutoff_rows * cols, grad_matrix.len);
    @memset(grad_matrix[0..end_idx], 0.0);
}

/// In-place Plasticity Isolation on 2D Matrix Columns:
/// Zeroes out gradients for base columns 0..cutoff_cols-1 (e.g. down_proj)
pub fn applyPlasticityMaskCols(
    grad_matrix: []f32,
    num_rows: usize,
    cutoff_cols: usize,
    stride_cols: usize,
) void {
    for (0..num_rows) |r| {
        const row_start = r * stride_cols;
        const row_end = row_start + @min(cutoff_cols, stride_cols);
        if (row_end <= grad_matrix.len) {
            @memset(grad_matrix[row_start..row_end], 0.0);
        }
    }
}

test "Net2WiderNet: exact function preservation across layer widening" {
    const allocator = std.testing.allocator;
    const in_f: usize = 4;
    const old_out: usize = 4;
    const new_out: usize = 8;
    const out_f: usize = 4;

    const w_in_old = try allocator.alloc(f32, old_out * in_f);
    defer allocator.free(w_in_old);
    const b_in_old = try allocator.alloc(f32, old_out);
    defer allocator.free(b_in_old);
    const w_out_old = try allocator.alloc(f32, out_f * old_out);
    defer allocator.free(w_out_old);

    for (0..old_out * in_f) |i| w_in_old[i] = @as(f32, @floatFromInt(i + 1)) * 0.1;
    for (0..old_out) |i| b_in_old[i] = @as(f32, @floatFromInt(i)) * 0.05;
    for (0..out_f * old_out) |i| w_out_old[i] = @as(f32, @floatFromInt(i % 5 + 1)) * 0.2;

    const x = [_]f32{ 0.5, -0.3, 1.2, 0.8 };

    // Compute original output: h = ReLU(W1*x + b1), y = W2*h
    var h_orig: [4]f32 = undefined;
    for (0..old_out) |i| {
        var sum: f32 = b_in_old[i];
        for (0..in_f) |j| sum += w_in_old[i * in_f + j] * x[j];
        h_orig[i] = if (sum > 0.0) sum else 0.0;
    }
    var y_orig: [4]f32 = undefined;
    for (0..out_f) |i| {
        var sum: f32 = 0.0;
        for (0..old_out) |j| sum += w_out_old[i * old_out + j] * h_orig[j];
        y_orig[i] = sum;
    }

    // Allocate widened layers
    const w_in_new = try allocator.alloc(f32, new_out * in_f);
    defer allocator.free(w_in_new);
    const b_in_new = try allocator.alloc(f32, new_out);
    defer allocator.free(b_in_new);
    const w_out_new = try allocator.alloc(f32, out_f * new_out);
    defer allocator.free(w_out_new);

    try net2Wider(
        w_in_old,
        b_in_old,
        w_in_new,
        b_in_new,
        w_out_old,
        w_out_new,
        old_out,
        new_out,
        in_f,
        out_f,
        0.0, // noise_std = 0 for exact mathematical parity check
        42,
    );

    // Compute widened output: h' = ReLU(W1'*x + b1'), y' = W2'*h'
    var h_wider: [8]f32 = undefined;
    for (0..new_out) |i| {
        var sum: f32 = b_in_new[i];
        for (0..in_f) |j| sum += w_in_new[i * in_f + j] * x[j];
        h_wider[i] = if (sum > 0.0) sum else 0.0;
    }
    var y_wider: [4]f32 = undefined;
    for (0..out_f) |i| {
        var sum: f32 = 0.0;
        for (0..new_out) |j| sum += w_out_new[i * new_out + j] * h_wider[j];
        y_wider[i] = sum;
    }

    for (0..out_f) |i| {
        try std.testing.expectApproxEqAbs(y_orig[i], y_wider[i], 1e-4);
    }
}

test "Net2DeeperNet: identity insertion preserves exact representations" {
    var weights: [16]f32 = undefined;
    var bias: [4]f32 = undefined;
    net2Deeper(&weights, &bias, 4);

    const x = [_]f32{ 1.5, -2.0, 0.75, 4.2 };
    var y: [4]f32 = undefined;
    for (0..4) |i| {
        var sum: f32 = bias[i];
        for (0..4) |j| sum += weights[i * 4 + j] * x[j];
        y[i] = sum;
    }

    for (0..4) |i| {
        try std.testing.expectEqual(x[i], y[i]);
    }
}

test "Net2WiderSwiGLU: zero-init guarantees exact bitwise function preservation" {
    const allocator = std.testing.allocator;
    const in_f: usize = 4;
    const old_inter: usize = 4;
    const new_inter: usize = 8;
    const out_f: usize = 4;

    const w_gate_old = try allocator.alloc(f32, old_inter * in_f);
    defer allocator.free(w_gate_old);
    const w_up_old = try allocator.alloc(f32, old_inter * in_f);
    defer allocator.free(w_up_old);
    const w_down_old = try allocator.alloc(f32, out_f * old_inter);
    defer allocator.free(w_down_old);

    for (0..old_inter * in_f) |i| {
        w_gate_old[i] = @as(f32, @floatFromInt(i + 1)) * 0.1;
        w_up_old[i] = @as(f32, @floatFromInt(i + 2)) * 0.15;
    }
    for (0..out_f * old_inter) |i| {
        w_down_old[i] = @as(f32, @floatFromInt(i + 3)) * 0.05;
    }

    const x = [_]f32{ 0.7, -0.4, 1.1, 0.2 };

    // Reference forward pass
    const silu = struct {
        fn call(v: f32) f32 {
            return v / (1.0 + @exp(-v));
        }
    }.call;

    var mlp_orig: [4]f32 = undefined;
    for (0..old_inter) |j| {
        var g_sum: f32 = 0.0;
        var u_sum: f32 = 0.0;
        for (0..in_f) |k| {
            g_sum += w_gate_old[j * in_f + k] * x[k];
            u_sum += w_up_old[j * in_f + k] * x[k];
        }
        mlp_orig[j] = silu(g_sum) * u_sum;
    }
    var y_orig: [4]f32 = undefined;
    for (0..out_f) |r| {
        var s: f32 = 0.0;
        for (0..old_inter) |j| s += w_down_old[r * old_inter + j] * mlp_orig[j];
        y_orig[r] = s;
    }

    // Widened SwiGLU
    const w_gate_new = try allocator.alloc(f32, new_inter * in_f);
    defer allocator.free(w_gate_new);
    const w_up_new = try allocator.alloc(f32, new_inter * in_f);
    defer allocator.free(w_up_new);
    const w_down_new = try allocator.alloc(f32, out_f * new_inter);
    defer allocator.free(w_down_new);

    try net2WiderSwiGLU(
        w_gate_old,
        null,
        w_up_old,
        null,
        w_down_old,
        null,
        w_gate_new,
        null,
        w_up_new,
        null,
        w_down_new,
        null,
        old_inter,
        new_inter,
        in_f,
        out_f,
        true, // zero_init
        0.0,
        1234,
    );

    var mlp_wider: [8]f32 = undefined;
    for (0..new_inter) |j| {
        var g_sum: f32 = 0.0;
        var u_sum: f32 = 0.0;
        for (0..in_f) |k| {
            g_sum += w_gate_new[j * in_f + k] * x[k];
            u_sum += w_up_new[j * in_f + k] * x[k];
        }
        mlp_wider[j] = silu(g_sum) * u_sum;
    }
    var y_wider: [4]f32 = undefined;
    for (0..out_f) |r| {
        var s: f32 = 0.0;
        for (0..new_inter) |j| s += w_down_new[r * new_inter + j] * mlp_wider[j];
        y_wider[r] = s;
    }

    // Exact equality: zero-init down_proj ensures zero contribution from new channels
    for (0..out_f) |r| {
        try std.testing.expectEqual(y_orig[r], y_wider[r]);
    }
}

test "expandVocab and plasticity masks: base protection against catastrophic forgetting" {
    const allocator = std.testing.allocator;
    const old_vocab: usize = 4;
    const new_vocab: usize = 8;
    const dim: usize = 4;

    const embed_old = try allocator.alloc(f32, old_vocab * dim);
    defer allocator.free(embed_old);
    for (0..old_vocab * dim) |i| embed_old[i] = @as(f32, @floatFromInt(i + 1)) * 0.1;

    const embed_new = try allocator.alloc(f32, new_vocab * dim);
    defer allocator.free(embed_new);

    try expandVocab(embed_old, embed_new, null, null, old_vocab, new_vocab, dim, 99);

    // Old vocab embeddings must remain bitwise identical
    for (0..old_vocab * dim) |i| {
        try std.testing.expectEqual(embed_old[i], embed_new[i]);
    }

    // Plasticity masking on rows
    const grad_rows = try allocator.alloc(f32, new_vocab * dim);
    defer allocator.free(grad_rows);
    @memset(grad_rows, 1.0);

    applyPlasticityMaskRows(grad_rows, old_vocab, dim);
    for (0..old_vocab * dim) |i| {
        try std.testing.expectEqual(@as(f32, 0.0), grad_rows[i]);
    }
    for (old_vocab * dim..new_vocab * dim) |i| {
        try std.testing.expectEqual(@as(f32, 1.0), grad_rows[i]);
    }

    // Plasticity masking on columns
    const grad_cols = try allocator.alloc(f32, dim * new_vocab);
    defer allocator.free(grad_cols);
    @memset(grad_cols, 2.0);

    applyPlasticityMaskCols(grad_cols, dim, old_vocab, new_vocab);
    for (0..dim) |r| {
        for (0..old_vocab) |c| {
            try std.testing.expectEqual(@as(f32, 0.0), grad_cols[r * new_vocab + c]);
        }
        for (old_vocab..new_vocab) |c| {
            try std.testing.expectEqual(@as(f32, 2.0), grad_cols[r * new_vocab + c]);
        }
    }
}
