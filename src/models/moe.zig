//! Mixture of Experts (MoE) Architecture & Routing Core (Plan §47, §150).
//!
//! Provides hardware-neutral, zero-alloc routing and dispatch for MoE architectures
//! including Mixtral (8x7B), DeepSeek-MoE (V2/V3), and Qwen-MoE:
//! 1. Top-K Gating Router with softmax normalization over chosen experts.
//! 2. Shared expert residual pathway (DeepSeek-MoE style).
//! 3. Deterministic expert dispatch with exact mathematical invariants.
//! 4. Bounded scratch buffer execution adhering to memory-first principles.

const std = @import("std");
const math = std.math;

pub const MoeRoutingType = enum {
    top_k_softmax,
    top_k_sigmoid,
};

pub const MoeConfig = struct {
    num_experts: usize,
    experts_per_tok: usize,
    num_shared_experts: usize = 0,
    routing_type: MoeRoutingType = .top_k_softmax,
    hidden_dim: usize,
    ffn_dim: usize,

    pub fn validate(self: MoeConfig) !void {
        if (self.num_experts == 0) return error.ZeroExperts;
        if (self.experts_per_tok == 0 or self.experts_per_tok > self.num_experts) {
            return error.InvalidExpertsPerTok;
        }
        if (self.hidden_dim == 0 or self.ffn_dim == 0) return error.InvalidDimension;
    }
};

pub const SelectedExpert = struct {
    index: usize,
    weight: f32,
};

/// Gating Router: computes routing logits over experts and selects Top-K experts.
pub const MoeRouter = struct {
    config: MoeConfig,

    pub fn init(cfg: MoeConfig) !MoeRouter {
        try cfg.validate();
        return .{ .config = cfg };
    }

    /// Routes a single token by selecting Top-K experts from raw gate logits.
    /// Uses bounded in-place selection without heap allocation.
    pub fn route(
        self: MoeRouter,
        gate_logits: []const f32,
        out_selected: []SelectedExpert,
    ) !void {
        std.debug.assert(gate_logits.len == self.config.num_experts);
        std.debug.assert(out_selected.len >= self.config.experts_per_tok);

        const k = self.config.experts_per_tok;

        // Initialize output buffer with lowest possible values
        for (0..k) |i| {
            out_selected[i] = .{ .index = 0, .weight = -math.inf(f32) };
        }

        // Top-K selection
        for (gate_logits, 0..) |logit, expert_idx| {
            // Find if logit is greater than the smallest currently selected
            var min_idx: usize = 0;
            var min_val: f32 = out_selected[0].weight;
            for (1..k) |j| {
                if (out_selected[j].weight < min_val) {
                    min_val = out_selected[j].weight;
                    min_idx = j;
                }
            }

            if (logit > min_val) {
                out_selected[min_idx] = .{ .index = expert_idx, .weight = logit };
            }
        }

        // Sort selected descending by logit
        var i: usize = 0;
        while (i < k) : (i += 1) {
            var j: usize = i + 1;
            while (j < k) : (j += 1) {
                if (out_selected[j].weight > out_selected[i].weight) {
                    const tmp = out_selected[i];
                    out_selected[i] = out_selected[j];
                    out_selected[j] = tmp;
                }
            }
        }

        // Apply softmax normalization over the chosen Top-K experts
        switch (self.config.routing_type) {
            .top_k_softmax => {
                var max_logit: f32 = out_selected[0].weight;
                for (1..k) |idx| {
                    if (out_selected[idx].weight > max_logit) max_logit = out_selected[idx].weight;
                }

                var sum_exp: f32 = 0.0;
                for (0..k) |idx| {
                    const e = @exp(out_selected[idx].weight - max_logit);
                    out_selected[idx].weight = e;
                    sum_exp += e;
                }

                const inv_sum = 1.0 / @max(sum_exp, 1e-12);
                for (0..k) |idx| {
                    out_selected[idx].weight *= inv_sum;
                }
            },
            .top_k_sigmoid => {
                var sum_sig: f32 = 0.0;
                for (0..k) |idx| {
                    const sig = 1.0 / (1.0 + @exp(-out_selected[idx].weight));
                    out_selected[idx].weight = sig;
                    sum_sig += sig;
                }
                const inv_sum = 1.0 / @max(sum_sig, 1e-12);
                for (0..k) |idx| {
                    out_selected[idx].weight *= inv_sum;
                }
            },
        }
    }
};

/// Dispatches hidden state through chosen experts and computes linear combination:
/// y = ∑_{k=1}^K w_k * Expert_k(x)  [+ SharedExpert(x)]
pub fn dispatchMoe(
    selected: []const SelectedExpert,
    expert_outputs: []const []const f32, // [K][hidden_dim]
    shared_expert_output: ?[]const f32, // optional [hidden_dim]
    out_combined: []f32,
) void {
    const dim = out_combined.len;

    // Start with shared expert residual if present
    if (shared_expert_output) |shared| {
        std.debug.assert(shared.len == dim);
        @memcpy(out_combined, shared);
    } else {
        @memset(out_combined, 0.0);
    }

    // Accumulate weighted expert contributions
    for (selected, expert_outputs) |sel, expert_out| {
        std.debug.assert(expert_out.len == dim);
        const w = sel.weight;
        for (0..dim) |d| {
            out_combined[d] += w * expert_out[d];
        }
    }
}

test "moe router top-k selection and softmax normalization" {
    const cfg = MoeConfig{
        .num_experts = 8,
        .experts_per_tok = 2,
        .hidden_dim = 64,
        .ffn_dim = 128,
    };
    const router = try MoeRouter.init(cfg);

    // Gate logits with clear winners: expert 3 (logit 5.0) and expert 6 (logit 3.0)
    const logits = [_]f32{ 0.1, -1.0, 0.5, 5.0, -0.2, 1.2, 3.0, -2.5 };
    var selected: [2]SelectedExpert = undefined;

    try router.route(&logits, &selected);

    // Selected must be 3 and 6
    try std.testing.expectEqual(@as(usize, 3), selected[0].index);
    try std.testing.expectEqual(@as(usize, 6), selected[1].index);

    // Normalization invariant: sum of weights == 1.0
    const sum_w = selected[0].weight + selected[1].weight;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum_w, 1e-6);

    // Expert 3 had higher logit (5.0 vs 3.0) -> higher softmax weight
    try std.testing.expect(selected[0].weight > selected[1].weight);
    // Softmax ratio: exp(5 - 5) / (exp(0) + exp(-2)) = 1 / (1 + 0.135335) ≈ 0.8808
    try std.testing.expectApproxEqAbs(@as(f32, 0.8808), selected[0].weight, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1192), selected[1].weight, 1e-3);
}

test "moe dispatch and shared expert combination" {
    var selected = [_]SelectedExpert{
        .{ .index = 1, .weight = 0.75 },
        .{ .index = 4, .weight = 0.25 },
    };

    const exp1 = [_]f32{ 2.0, 4.0, -1.0, 10.0 };
    const exp4 = [_]f32{ 6.0, -8.0, 3.0, 2.0 };
    const expert_outputs = [_][]const f32{ &exp1, &exp4 };

    // Case 1: Dense MoE without shared expert
    var combined: [4]f32 = undefined;
    dispatchMoe(&selected, &expert_outputs, null, &combined);

    // Expected:
    // d0: 0.75 * 2.0 + 0.25 * 6.0 = 1.5 + 1.5 = 3.0
    // d1: 0.75 * 4.0 + 0.25 * -8.0 = 3.0 - 2.0 = 1.0
    // d2: 0.75 * -1.0 + 0.25 * 3.0 = -0.75 + 0.75 = 0.0
    // d3: 0.75 * 10.0 + 0.25 * 2.0 = 7.5 + 0.5 = 8.0
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), combined[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), combined[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), combined[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), combined[3], 1e-6);

    // Case 2: DeepSeek-style MoE with shared expert residual
    const shared = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    dispatchMoe(&selected, &expert_outputs, &shared, &combined);

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), combined[0], 1e-6); // 3.0 + 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), combined[1], 1e-6); // 1.0 + 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), combined[2], 1e-6); // 0.0 + 3.0
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), combined[3], 1e-6); // 8.0 + 4.0
}
