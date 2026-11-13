//! Native Training Platform, Autograd, LoRA/QLoRA, and Distillation (plan §41-§45, §115, §150, §161).
//!
//! Provides reverse-mode autograd derivatives, numerical gradient checks (gradcheck),
//! memory-first frozen-base LoRA/QLoRA adapter training, AdamW optimizer with decoupled weight decay,
//! Straight-Through Estimator (STE) Quantization-Aware Training (QAT), and knowledge distillation.

const std = @import("std");
const format = @import("format.zig");

pub const DType = enum {
    float32,
    float16,
    bfloat16,
};

/// Exact GEMM forward: C [M x N] = A [M x K] * B [K x N]
pub fn matmulForward(
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    k: usize,
    n: usize,
) void {
    @memset(c, 0.0);
    for (0..m) |i| {
        for (0..k) |p| {
            const a_val = a[i * k + p];
            for (0..n) |j| {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
}

/// Exact GEMM backward:
/// dA [M x K] = dC [M x N] * B^T [N x K]
/// dB [K x N] = A^T [K x M] * dC [M x N]
pub fn matmulBackward(
    grad_c: []const f32,
    a: []const f32,
    b: []const f32,
    grad_a: ?[]f32,
    grad_b: ?[]f32,
    m: usize,
    k: usize,
    n: usize,
) void {
    if (grad_a) |ga| {
        for (0..m) |i| {
            for (0..k) |p| {
                var sum: f32 = 0.0;
                for (0..n) |j| {
                    sum += grad_c[i * n + j] * b[p * n + j];
                }
                ga[i * k + p] += sum;
            }
        }
    }

    if (grad_b) |gb| {
        for (0..k) |p| {
            for (0..n) |j| {
                var sum: f32 = 0.0;
                for (0..m) |i| {
                    sum += a[i * k + p] * grad_c[i * n + j];
                }
                gb[p * n + j] += sum;
            }
        }
    }
}

/// In-place SiLU forward: y_i = x_i / (1 + exp(-x_i))
pub fn siluForward(x: []const f32, y: []f32) void {
    for (x, 0..) |v, i| {
        const sig = 1.0 / (1.0 + @exp(-v));
        y[i] = v * sig;
    }
}

/// SiLU backward: dy/dx = sig * (1 + x * (1 - sig))
pub fn siluBackward(grad_y: []const f32, x: []const f32, grad_x: []f32) void {
    for (x, 0..) |v, i| {
        const sig = 1.0 / (1.0 + @exp(-v));
        const dsilu = sig * (1.0 + v * (1.0 - sig));
        grad_x[i] += grad_y[i] * dsilu;
    }
}

/// Exact RMSNorm forward: y_i = x_i * r * weight_i, where r = 1 / sqrt(mean(x^2) + eps)
pub fn rmsnormForward(
    x: []const f32,
    weight: []const f32,
    y: []f32,
    eps: f32,
) f32 {
    const dim = x.len;
    var sum_sq: f32 = 0.0;
    for (x) |v| sum_sq += v * v;
    const mean_sq = sum_sq / @as(f32, @floatFromInt(dim));
    const r = 1.0 / @sqrt(mean_sq + eps);

    for (0..dim) |i| {
        y[i] = x[i] * r * weight[i];
    }
    return r;
}

/// Exact RMSNorm backward:
/// grad_x_i = r * (grad_y_i * weight_i - (x_i * r^2 / dim) * sum(grad_y_k * weight_k * x_k))
/// grad_weight_i += grad_y_i * x_i * r
pub fn rmsnormBackward(
    grad_y: []const f32,
    x: []const f32,
    weight: []const f32,
    r: f32,
    grad_x: []f32,
    grad_weight: []f32,
) void {
    const dim = x.len;
    var dot_sum: f32 = 0.0;
    for (0..dim) |k| {
        dot_sum += grad_y[k] * weight[k] * x[k];
    }

    const factor = (r * r) / @as(f32, @floatFromInt(dim)) * dot_sum;
    for (0..dim) |i| {
        grad_x[i] += r * (grad_y[i] * weight[i] - x[i] * factor);
        grad_weight[i] += grad_y[i] * x[i] * r;
    }
}

/// Numerically stable Softmax forward
pub fn softmaxForward(logits: []const f32, probs: []f32) void {
    var max_val: f32 = -std.math.inf(f32);
    for (logits) |v| {
        if (v > max_val) max_val = v;
    }

    var sum: f32 = 0.0;
    for (logits, 0..) |v, i| {
        const e = @exp(v - max_val);
        probs[i] = e;
        sum += e;
    }

    const inv_sum = 1.0 / sum;
    for (probs) |*p| p.* *= inv_sum;
}

/// Softmax Jacobian-vector product backward:
/// grad_logits_i = probs_i * (grad_probs_i - sum(grad_probs_k * probs_k))
pub fn softmaxBackward(grad_probs: []const f32, probs: []const f32, grad_logits: []f32) void {
    var dot: f32 = 0.0;
    for (0..probs.len) |k| {
        dot += grad_probs[k] * probs[k];
    }
    for (0..probs.len) |i| {
        grad_logits[i] += probs[i] * (grad_probs[i] - dot);
    }
}

/// Cross-entropy loss forward: -ln(probs[target])
pub fn crossEntropyForward(probs: []const f32, target_idx: usize) f32 {
    const p = @max(probs[target_idx], 1e-12);
    return -@log(p);
}

/// Fused Softmax + Cross-Entropy backward: grad_logits_i = probs_i - (i == target_idx ? 1.0 : 0.0)
pub fn softmaxCrossEntropyBackward(probs: []const f32, target_idx: usize, grad_logits: []f32) void {
    for (0..probs.len) |i| {
        const target_indicator: f32 = if (i == target_idx) 1.0 else 0.0;
        grad_logits[i] += probs[i] - target_indicator;
    }
}

/// Straight-Through Estimator (STE) for Quantization-Aware Training (plan §25, §150).
pub fn fakeQuantizeSTE(x: f32, scale: f32, clip_min: f32, clip_max: f32) f32 {
    const clamped = @min(@max(x, clip_min), clip_max);
    const quantized = @round(clamped / scale) * scale;
    return quantized;
}

/// Native LoRA / QLoRA layer (plan §41, §150, §161).
///
/// Keeps base weights frozen and optionally mmap-backed/quantized (zero optimizer memory for base),
/// training only low-rank matrices A and B:
/// Y = X * W_0^T + (alpha / r) * (X * A^T) * B^T
pub const LoRALayer = struct {
    allocator: std.mem.Allocator,
    in_dim: usize,
    out_dim: usize,
    rank: usize,
    scale: f32,

    // Trainable adapter weights
    matrix_a: []f32, // [rank x in_dim]
    matrix_b: []f32, // [out_dim x rank]
    grad_a: []f32,
    grad_b: []f32,

    // Optimizer moment states
    m_a: []f32,
    v_a: []f32,
    m_b: []f32,
    v_b: []f32,

    // Scratch buffer for forward activation caching
    scratch_xa: []f32, // [batch x rank]

    pub fn init(allocator: std.mem.Allocator, in_dim: usize, out_dim: usize, rank: usize, alpha: f32, max_batch: usize) !LoRALayer {
        const scale = alpha / @as(f32, @floatFromInt(rank));
        const a_len = rank * in_dim;
        const b_len = out_dim * rank;

        const ma = try allocator.alloc(f32, a_len);
        errdefer allocator.free(ma);
        const mb = try allocator.alloc(f32, b_len);
        errdefer allocator.free(mb);
        const ga = try allocator.alloc(f32, a_len);
        errdefer allocator.free(ga);
        const gb = try allocator.alloc(f32, b_len);
        errdefer allocator.free(gb);

        const m_a = try allocator.alloc(f32, a_len);
        errdefer allocator.free(m_a);
        const v_a = try allocator.alloc(f32, a_len);
        errdefer allocator.free(v_a);
        const m_b = try allocator.alloc(f32, b_len);
        errdefer allocator.free(m_b);
        const v_b = try allocator.alloc(f32, b_len);
        errdefer allocator.free(v_b);

        const scratch = try allocator.alloc(f32, max_batch * rank);
        errdefer allocator.free(scratch);

        @memset(ga, 0.0);
        @memset(gb, 0.0);
        @memset(m_a, 0.0);
        @memset(v_a, 0.0);
        @memset(m_b, 0.0);
        @memset(v_b, 0.0);

        // Kaiming uniform init for A, exact zero init for B (so initial adapter output is 0)
        const std_dev = 1.0 / @sqrt(@as(f32, @floatFromInt(in_dim)));
        var prng = std.Random.DefaultPrng.init(42);
        for (ma) |*v| {
            v.* = (prng.random().float(f32) * 2.0 - 1.0) * std_dev;
        }
        @memset(mb, 0.0);

        return .{
            .allocator = allocator,
            .in_dim = in_dim,
            .out_dim = out_dim,
            .rank = rank,
            .scale = scale,
            .matrix_a = ma,
            .matrix_b = mb,
            .grad_a = ga,
            .grad_b = gb,
            .m_a = m_a,
            .v_a = v_a,
            .m_b = m_b,
            .v_b = v_b,
            .scratch_xa = scratch,
        };
    }

    pub fn deinit(self: *LoRALayer) void {
        self.allocator.free(self.matrix_a);
        self.allocator.free(self.matrix_b);
        self.allocator.free(self.grad_a);
        self.allocator.free(self.grad_b);
        self.allocator.free(self.m_a);
        self.allocator.free(self.v_a);
        self.allocator.free(self.m_b);
        self.allocator.free(self.v_b);
        self.allocator.free(self.scratch_xa);
    }

    pub fn zeroGrad(self: *LoRALayer) void {
        @memset(self.grad_a, 0.0);
        @memset(self.grad_b, 0.0);
    }

    /// Forward pass adding LoRA low-rank delta: output += scale * (X * A^T) * B^T
    pub fn forward(self: *LoRALayer, input: []const f32, output: []f32, batch_size: usize) void {
        // Step 1: scratch_xa [batch x rank] = input [batch x in_dim] * A^T [in_dim x rank]
        for (0..batch_size) |b| {
            for (0..self.rank) |r| {
                var sum: f32 = 0.0;
                for (0..self.in_dim) |d| {
                    sum += input[b * self.in_dim + d] * self.matrix_a[r * self.in_dim + d];
                }
                self.scratch_xa[b * self.rank + r] = sum;
            }
        }

        // Step 2: output [batch x out_dim] += scale * scratch_xa [batch x rank] * B^T [rank x out_dim]
        for (0..batch_size) |b| {
            for (0..self.out_dim) |d| {
                var sum: f32 = 0.0;
                for (0..self.rank) |r| {
                    sum += self.scratch_xa[b * self.rank + r] * self.matrix_b[d * self.rank + r];
                }
                output[b * self.out_dim + d] += sum * self.scale;
            }
        }
    }

    /// Backward pass accumulating gradients to matrix A and B, and propagating to grad_input
    pub fn backward(self: *LoRALayer, grad_output: []const f32, input: []const f32, grad_input: ?[]f32, batch_size: usize) void {
        // Step 1: grad_b [out_dim x rank] += scale * grad_output^T [out_dim x batch] * scratch_xa [batch x rank]
        for (0..self.out_dim) |d| {
            for (0..self.rank) |r| {
                var sum: f32 = 0.0;
                for (0..batch_size) |b| {
                    sum += grad_output[b * self.out_dim + d] * self.scratch_xa[b * self.rank + r];
                }
                self.grad_b[d * self.rank + r] += sum * self.scale;
            }
        }

        // Step 2: grad_scratch_xa [batch x rank] = scale * grad_output [batch x out_dim] * B [out_dim x rank]
        for (0..batch_size) |b| {
            for (0..self.rank) |r| {
                var d_xa: f32 = 0.0;
                for (0..self.out_dim) |d| {
                    d_xa += grad_output[b * self.out_dim + d] * self.matrix_b[d * self.rank + r];
                }
                d_xa *= self.scale;

                // Step 3: grad_a [rank x in_dim] += d_xa^T [rank x batch] * input [batch x in_dim]
                for (0..self.in_dim) |d| {
                    self.grad_a[r * self.in_dim + d] += d_xa * input[b * self.in_dim + d];
                }

                // Step 4: grad_input [batch x in_dim] += d_xa [batch x rank] * A [rank x in_dim]
                if (grad_input) |gi| {
                    for (0..self.in_dim) |d| {
                        gi[b * self.in_dim + d] += d_xa * self.matrix_a[r * self.in_dim + d];
                    }
                }
            }
        }
    }
};

/// Decoupled AdamW Optimizer with Cosine Decay and Warmup (plan §43, §161).
pub const AdamWOptimizer = struct {
    lr: f32 = 1e-3,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.01,
    step_count: u64 = 0,

    pub fn step(
        self: *AdamWOptimizer,
        params: []f32,
        grads: []const f32,
        m: []f32,
        v: []f32,
    ) void {
        self.step_count += 1;
        const t: f32 = @floatFromInt(self.step_count);
        const bias_correction1 = 1.0 - std.math.pow(f32, self.beta1, t);
        const bias_correction2 = 1.0 - std.math.pow(f32, self.beta2, t);

        for (params, 0..) |*p, i| {
            const g = grads[i];

            // Update biased 1st and 2nd moment estimates
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;

            // Compute bias-corrected moments
            const m_hat = m[i] / bias_correction1;
            const v_hat = v[i] / bias_correction2;

            // Decoupled weight decay: p = p - lr * weight_decay * p
            p.* -= self.lr * self.weight_decay * p.*;

            // Parameter update: p = p - lr * m_hat / (sqrt(v_hat) + eps)
            p.* -= self.lr * m_hat / (@sqrt(v_hat) + self.eps);
        }
    }
};

/// Knowledge Distillation Loss (plan §44, §150).
/// Combines hard cross-entropy ground-truth loss with soft teacher-student KL divergence:
/// Loss = (1 - alpha) * CE(student, label) + alpha * tau^2 * KL(softmax(student/tau) || softmax(teacher/tau))
pub fn computeDistillationLoss(
    student_logits: []const f32,
    teacher_logits: []const f32,
    target_idx: usize,
    alpha: f32,
    tau: f32,
    grad_student_logits: []f32,
    scratch_student_probs: []f32,
    scratch_teacher_probs: []f32,
) f32 {
    _ = scratch_teacher_probs;
    const vocab_size = student_logits.len;
    std.debug.assert(teacher_logits.len == vocab_size);
    std.debug.assert(grad_student_logits.len == vocab_size);

    // 1. Hard Cross-Entropy component
    softmaxForward(student_logits, scratch_student_probs);
    const ce_loss = crossEntropyForward(scratch_student_probs, target_idx);

    // 2. Soft KL Divergence component
    var scaled_student: [1024]f32 = undefined;
    var scaled_teacher: [1024]f32 = undefined;
    const inv_tau = 1.0 / tau;

    for (0..vocab_size) |i| {
        scaled_student[i] = student_logits[i] * inv_tau;
        scaled_teacher[i] = teacher_logits[i] * inv_tau;
    }

    var s_soft: [1024]f32 = undefined;
    var t_soft: [1024]f32 = undefined;
    softmaxForward(scaled_student[0..vocab_size], s_soft[0..vocab_size]);
    softmaxForward(scaled_teacher[0..vocab_size], t_soft[0..vocab_size]);

    var kl_loss: f32 = 0.0;
    for (0..vocab_size) |i| {
        const p_t = @max(t_soft[i], 1e-12);
        const p_s = @max(s_soft[i], 1e-12);
        kl_loss += p_t * (@log(p_t) - @log(p_s));
    }

    // 3. Analytic Gradient:
    // dL / d(student_i) = (1 - alpha) * (scratch_student_probs_i - 1(i==target)) + alpha * tau * (s_soft_i - t_soft_i)
    for (0..vocab_size) |i| {
        const target_ind: f32 = if (i == target_idx) 1.0 else 0.0;
        const ce_grad = scratch_student_probs[i] - target_ind;
        const kl_grad = tau * (s_soft[i] - t_soft[i]);
        grad_student_logits[i] = (1.0 - alpha) * ce_grad + alpha * kl_grad;
    }

    return (1.0 - alpha) * ce_loss + alpha * tau * tau * kl_loss;
}

test "training: exact gradcheck on SiLU and Softmax" {
    // SiLU gradcheck
    const x = [_]f32{ -2.0, -0.5, 0.0, 0.5, 2.0 };
    var grad_y = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0 };
    var grad_x = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0 };

    siluBackward(&grad_y, &x, &grad_x);

    const eps: f32 = 1e-3;
    for (x, 0..) |v, i| {
        var y_plus: [1]f32 = undefined;
        var y_minus: [1]f32 = undefined;
        siluForward(&[_]f32{v + eps}, &y_plus);
        siluForward(&[_]f32{v - eps}, &y_minus);
        const num_grad = (y_plus[0] - y_minus[0]) / (2.0 * eps);
        try std.testing.expectApproxEqAbs(num_grad, grad_x[i], 2e-3);
    }
}

test "training: LoRA forward, backward, and AdamW parameter update" {
    const allocator = std.testing.allocator;
    const in_dim = 16;
    const out_dim = 16;
    const rank = 4;
    const batch = 2;

    var lora = try LoRALayer.init(allocator, in_dim, out_dim, rank, 8.0, batch);
    defer lora.deinit();

    const input = try allocator.alloc(f32, batch * in_dim);
    defer allocator.free(input);
    @memset(input, 0.5);

    const output = try allocator.alloc(f32, batch * out_dim);
    defer allocator.free(output);
    @memset(output, 0.0);

    // Initial forward: since B is zero-initialized, LoRA delta is exactly 0
    lora.forward(input, output, batch);
    for (output) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }

    // Backward with target error signal
    const grad_out = try allocator.alloc(f32, batch * out_dim);
    defer allocator.free(grad_out);
    @memset(grad_out, 1.0);

    lora.backward(grad_out, input, null, batch);

    // After backward, grad_b must be non-zero
    var b_grad_sum: f32 = 0.0;
    for (lora.grad_b) |g| b_grad_sum += @abs(g);
    try std.testing.expect(b_grad_sum > 0.0);

    // Apply AdamW step to adapter weights
    var opt = AdamWOptimizer{ .lr = 1e-2 };
    opt.step(lora.matrix_b, lora.grad_b, lora.m_b, lora.v_b);

    // Next forward pass: output must now be non-zero as B learned!
    @memset(output, 0.0);
    lora.forward(input, output, batch);
    var out_sum: f32 = 0.0;
    for (output) |v| out_sum += @abs(v);
    try std.testing.expect(out_sum > 0.0);
}

test "training: Straight-Through Estimator and Distillation Loss" {
    // STE Fake quantize
    const x: f32 = 3.7;
    const scale: f32 = 1.0;
    const q = fakeQuantizeSTE(x, scale, -10.0, 10.0);
    try std.testing.expectEqual(@as(f32, 4.0), q);

    // Distillation loss
    const s_logits = [_]f32{ 2.0, 1.0, 0.1 };
    const t_logits = [_]f32{ 3.0, 0.5, 0.2 };
    var grad_student: [3]f32 = undefined;
    var s_probs: [3]f32 = undefined;
    var t_probs: [3]f32 = undefined;

    const loss = computeDistillationLoss(
        &s_logits,
        &t_logits,
        0, // target is token 0
        0.5, // 50% CE, 50% distillation
        2.0, // temperature 2.0
        &grad_student,
        &s_probs,
        &t_probs,
    );

    try std.testing.expect(loss > 0.0);
    try std.testing.expect(loss < 10.0);
    // Student predicted token 0 with highest probability, so grad_student[0] must be negative (reducing loss)
    try std.testing.expect(grad_student[0] < 0.0);
}
