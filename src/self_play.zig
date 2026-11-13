//! Native Self-Play Fine-Tuning (SPIN), Preference Optimization & Anti-Collapse Guards
//!
//! Implements Chen et al. Self-Play Fine-Tuning (SPIN) and DPO-style preference optimization:
//! 1. Preference pair formulation (winner vs. loser generation from self-play).
//! 2. Exact mathematical loss and analytical gradients for policy log-probabilities.
//! 3. Reference model snapshot comparison with exact KL divergence regularization.
//! 4. Anti-collapse distribution monitoring using Shannon entropy bounds.
//! 5. Automatic rollback safeguard when regression or entropy collapse occurs.

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const RegressionGovernor = @import("adaptive.zig").RegressionGovernor;
const LineageGraph = @import("lineage.zig").LineageGraph;

/// Preference sample consisting of prompt, winning output, and losing output.
pub const PreferencePair = struct {
    prompt_id: usize,
    /// Sum of log probabilities for winning completion under active model θ
    policy_win_logp: f32,
    /// Sum of log probabilities for losing completion under active model θ
    policy_lose_logp: f32,
    /// Sum of log probabilities for winning completion under reference model θ_ref
    ref_win_logp: f32,
    /// Sum of log probabilities for losing completion under reference model θ_ref
    ref_lose_logp: f32,
};

/// Result of evaluating SPIN objective over a batch of preference pairs.
pub const SpinEvalResult = struct {
    loss: f32,
    grad_policy_win: f32,
    grad_policy_lose: f32,
    implicit_margin: f32,
    accuracy: f32, // proportion where win margin > lose margin
};

/// Evaluates SPIN (Self-Play Fine-Tuning) loss and analytic gradients:
/// L_SPIN(θ) = -log σ( λ [ (log P_θ(y_w) - log P_ref(y_w)) - (log P_θ(y_l) - log P_ref(y_l)) ] )
pub fn computeSpinLoss(pair: PreferencePair, lambda_reg: f32) SpinEvalResult {
    const delta_win = pair.policy_win_logp - pair.ref_win_logp;
    const delta_lose = pair.policy_lose_logp - pair.ref_lose_logp;
    const margin = delta_win - delta_lose;
    const scaled_margin = lambda_reg * margin;

    // Stable sigmoid calculation: σ(z) = 1 / (1 + exp(-z))
    // -log σ(z) = log(1 + exp(-z)) = softplus(-z)
    const loss: f32 = if (scaled_margin > 0.0)
        @log(1.0 + @exp(-scaled_margin))
    else
        -scaled_margin + @log(1.0 + @exp(scaled_margin));

    const sig: f32 = 1.0 / (1.0 + @exp(-scaled_margin));
    // ∂L/∂(scaled_margin) = -(1 - σ(scaled_margin))
    const d_margin = -lambda_reg * (1.0 - sig);

    return .{
        .loss = loss,
        .grad_policy_win = d_margin, // ∂L/∂(policy_win_logp) = d_margin
        .grad_policy_lose = -d_margin, // ∂L/∂(policy_lose_logp) = -d_margin
        .implicit_margin = margin,
        .accuracy = if (margin > 0.0) 1.0 else 0.0,
    };
}

/// Evaluates Kullback-Leibler (KL) divergence D_KL(P_policy || P_ref) between two discrete probability distributions:
/// D_KL(P || Q) = ∑_i P(i) * ln(P(i) / (Q(i) + 1e-12))
pub fn computeKLDivergence(policy_probs: []const f32, ref_probs: []const f32) f32 {
    std.debug.assert(policy_probs.len == ref_probs.len);
    var kl: f32 = 0.0;
    const eps: f32 = 1e-12;

    for (policy_probs, ref_probs) |p, q| {
        if (p > 1e-9) {
            const denom = @max(q, eps);
            kl += p * (@log(p) - @log(denom));
        }
    }
    return @max(0.0, kl);
}

/// Computes Shannon entropy H(P) = -∑_i P(i) * ln(P(i) + 1e-12).
/// Used as an anti-collapse monitor to ensure generation diversity is preserved.
pub fn computeEntropy(probs: []const f32) f32 {
    var entropy: f32 = 0.0;
    const eps: f32 = 1e-12;

    for (probs) |p| {
        if (p > eps) {
            entropy -= p * @log(p);
        }
    }
    return @max(0.0, entropy);
}

/// Policy monitoring metrics and anti-collapse evaluation.
pub const AntiCollapseMonitor = struct {
    min_entropy_threshold: f32,
    max_kl_threshold: f32,

    pub fn init(min_entropy_threshold: f32, max_kl_threshold: f32) AntiCollapseMonitor {
        return .{
            .min_entropy_threshold = min_entropy_threshold,
            .max_kl_threshold = max_kl_threshold,
        };
    }

    pub fn check(self: AntiCollapseMonitor, policy_probs: []const f32, ref_probs: []const f32) !struct { entropy: f32, kl: f32 } {
        const ent = computeEntropy(policy_probs);
        if (ent < self.min_entropy_threshold) {
            return error.EntropyCollapseDetected;
        }

        const kl = computeKLDivergence(policy_probs, ref_probs);
        if (kl > self.max_kl_threshold) {
            return error.KLDriftExceeded;
        }

        return .{ .entropy = ent, .kl = kl };
    }
};

/// Self-Play Evolution Engine coordinating iterative refinement, SPIN loss updates,
/// reference model comparison, anti-collapse gating, and regression rollback.
pub const SelfPlayEngine = struct {
    allocator: Allocator,
    regression_gov: RegressionGovernor,
    monitor: AntiCollapseMonitor,
    lambda_reg: f32,
    current_generation: u32,
    best_loss: f32,

    pub fn init(
        allocator: Allocator,
        max_regression_ratio: f32,
        min_entropy: f32,
        max_kl: f32,
        lambda_reg: f32,
        baseline_loss: f32,
    ) SelfPlayEngine {
        return .{
            .allocator = allocator,
            .regression_gov = RegressionGovernor.init(max_regression_ratio, baseline_loss),
            .monitor = AntiCollapseMonitor.init(min_entropy, max_kl),
            .lambda_reg = lambda_reg,
            .current_generation = 0,
            .best_loss = baseline_loss,
        };
    }

    /// Evaluates a self-play iteration:
    /// 1. Computes mean SPIN loss over preference batch.
    /// 2. Asserts anti-collapse invariants (minimum entropy and bounded KL).
    /// 3. Checks regression governor against loss baseline.
    /// 4. Returns whether candidate generation is accepted or rolled back.
    pub fn evaluateGeneration(
        self: *SelfPlayEngine,
        pairs: []const PreferencePair,
        sample_policy_dist: []const f32,
        sample_ref_dist: []const f32,
    ) !struct { accepted: bool, mean_loss: f32, accuracy: f32 } {
        if (pairs.len == 0) return error.EmptyPreferenceBatch;

        // Step 1: Check anti-collapse monitor
        _ = try self.monitor.check(sample_policy_dist, sample_ref_dist);

        // Step 2: Accumulate SPIN loss & accuracy
        var total_loss: f32 = 0.0;
        var total_acc: f32 = 0.0;

        for (pairs) |pair| {
            const res = computeSpinLoss(pair, self.lambda_reg);
            total_loss += res.loss;
            total_acc += res.accuracy;
        }

        const mean_loss = total_loss / @as(f32, @floatFromInt(pairs.len));
        const accuracy = total_acc / @as(f32, @floatFromInt(pairs.len));

        // Step 3: Check regression governor
        if (self.regression_gov.shouldRollback(mean_loss)) {
            return .{
                .accepted = false,
                .mean_loss = mean_loss,
                .accuracy = accuracy,
            };
        }

        // Evolution step accepted
        self.current_generation += 1;
        if (mean_loss < self.best_loss) {
            self.best_loss = mean_loss;
        }
        return .{
            .accepted = true,
            .mean_loss = mean_loss,
            .accuracy = accuracy,
        };
    }
};

test "spin loss calculation and gradients" {
    // A pair where active policy prefers winning completion strongly over losing completion
    const pair = PreferencePair{
        .prompt_id = 42,
        .policy_win_logp = -1.5,
        .policy_lose_logp = -4.0,
        .ref_win_logp = -2.0,
        .ref_lose_logp = -2.1,
    };

    const res = computeSpinLoss(pair, 0.5);

    // Delta win = -1.5 - (-2.0) = +0.5
    // Delta lose = -4.0 - (-2.1) = -1.9
    // Margin = 0.5 - (-1.9) = 2.4 > 0
    try std.testing.expect(res.implicit_margin > 2.0);
    try std.testing.expectEqual(@as(f32, 1.0), res.accuracy);
    try std.testing.expect(res.loss < 0.35); // Small loss since policy already prefers winner
    try std.testing.expect(res.grad_policy_win < 0.0); // Negative gradient pulls policy_win_logp higher
    try std.testing.expect(res.grad_policy_lose > 0.0); // Positive gradient pushes policy_lose_logp lower
}

test "kl divergence and shannon entropy" {
    // Identical uniform distributions -> KL must be 0
    const p = [_]f32{ 0.25, 0.25, 0.25, 0.25 };
    const q = [_]f32{ 0.25, 0.25, 0.25, 0.25 };

    const kl_ident = computeKLDivergence(&p, &q);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), kl_ident, 1e-6);

    // Uniform entropy for 4 states = ln(4) ≈ 1.38629
    const ent_uniform = computeEntropy(&p);
    try std.testing.expectApproxEqAbs(@as(f32, 1.386294), ent_uniform, 1e-4);

    // Collapsed peaked distribution -> Low entropy, high KL
    const collapsed = [_]f32{ 0.999, 0.0003, 0.0003, 0.0004 };
    const ent_collapsed = computeEntropy(&collapsed);
    try std.testing.expect(ent_collapsed < 0.05);

    const kl_drift = computeKLDivergence(&collapsed, &q);
    try std.testing.expect(kl_drift > 1.0);
}

test "anti-collapse monitor detects distribution collapse" {
    const monitor = AntiCollapseMonitor.init(0.5, 2.0);

    const ref = [_]f32{ 0.25, 0.25, 0.25, 0.25 };
    const collapsed = [_]f32{ 0.999, 0.0003, 0.0003, 0.0004 };

    // Should return error.EntropyCollapseDetected
    const res = monitor.check(&collapsed, &ref);
    try std.testing.expectError(error.EntropyCollapseDetected, res);
}

test "self-play engine iteration and rollback" {
    const allocator = std.testing.allocator;
    var engine = SelfPlayEngine.init(allocator, 1.10, 0.20, 3.0, 0.5, 1.0);

    const ref_dist = [_]f32{ 0.5, 0.5 };
    const good_dist = [_]f32{ 0.6, 0.4 };

    const good_pairs = [_]PreferencePair{
        .{
            .prompt_id = 1,
            .policy_win_logp = -1.0,
            .policy_lose_logp = -3.0,
            .ref_win_logp = -2.0,
            .ref_lose_logp = -2.0,
        },
    };

    // Good step -> accepted
    const res1 = try engine.evaluateGeneration(&good_pairs, &good_dist, &ref_dist);
    try std.testing.expect(res1.accepted);
    try std.testing.expectEqual(@as(u32, 1), engine.current_generation);

    // Regressive step with severe loss
    const bad_pairs = [_]PreferencePair{
        .{
            .prompt_id = 2,
            .policy_win_logp = -5.0,
            .policy_lose_logp = -0.5,
            .ref_win_logp = -2.0,
            .ref_lose_logp = -2.0,
        },
    };

    const res2 = try engine.evaluateGeneration(&bad_pairs, &good_dist, &ref_dist);
    // Should be rejected by regression governor without advancing generation
    try std.testing.expect(!res2.accepted);
    try std.testing.expectEqual(@as(u32, 1), engine.current_generation);
}
