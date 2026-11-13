//! Speculative decoding engine (plan §51, §150 milestone 2.3).
//!
//! Implements draft/target verification batches, exact speculative rejection sampling
//! (Leviathan et al., 2023), dynamic draft length scheduling, and multi-device execution
//! topologies (same-device, CPU draft + GPU target, NPU draft + GPU target, small + large model).

const std = @import("std");

/// Device execution topologies supported by speculative decoding (plan §51).
pub const DeviceExecutionMode = enum {
    same_device,
    cpu_draft_gpu_target,
    npu_draft_gpu_target,
    small_model_large_model,
};

/// Configuration for speculative decoding.
pub const SpeculativeConfig = struct {
    max_draft_tokens: usize = 8,
    min_draft_tokens: usize = 1,
    initial_draft_tokens: usize = 4,
    target_acceptance_rate: f32 = 0.75,
    mode: DeviceExecutionMode = .same_device,
    dynamic_length: bool = true,
};

/// Candidate token proposed by the draft model.
pub const DraftCandidate = struct {
    token: u32,
    draft_prob: f32,
    target_prob: f32 = 0.0,
    accepted: bool = false,
};

/// Outcome of a speculative verification batch.
pub const VerificationResult = struct {
    pub const max_capacity = 16;

    accepted_tokens: [max_capacity]u32 = @splat(0),
    n_accepted: usize = 0,
    correction_token: ?u32 = null,
    total_emitted: usize = 0,
    all_accepted: bool = false,

    pub fn tokens(self: *const VerificationResult, out: []u32) usize {
        const count = @min(out.len, self.total_emitted);
        for (0..self.n_accepted) |i| {
            if (i < count) out[i] = self.accepted_tokens[i];
        }
        if (self.correction_token) |c| {
            if (self.n_accepted < count) {
                out[self.n_accepted] = c;
            }
        }
        return count;
    }
};

/// Telemetry and empirical performance metrics for speculative decoding.
pub const SpeculativeStats = struct {
    total_draft_proposed: u64 = 0,
    total_draft_accepted: u64 = 0,
    total_correction_tokens: u64 = 0,
    verification_steps: u64 = 0,
    ema_acceptance_rate: f32 = 0.75,

    pub fn acceptanceRate(self: *const SpeculativeStats) f32 {
        if (self.total_draft_proposed == 0) return 0.0;
        return @as(f32, @floatFromInt(self.total_draft_accepted)) /
            @as(f32, @floatFromInt(self.total_draft_proposed));
    }

    /// Effective acceleration factor: ratio of emitted tokens to verification passes.
    pub fn effectiveSpeedup(self: *const SpeculativeStats) f32 {
        if (self.verification_steps == 0) return 1.0;
        const total_emitted = self.total_draft_accepted + self.total_correction_tokens;
        return @as(f32, @floatFromInt(total_emitted)) /
            @as(f32, @floatFromInt(self.verification_steps));
    }
};

/// Adaptive draft length scheduler (plan §51).
pub const DynamicDraftScheduler = struct {
    current_draft_tokens: usize,
    min_tokens: usize,
    max_tokens: usize,
    ema_alpha: f32 = 0.25,

    pub fn init(cfg: SpeculativeConfig) DynamicDraftScheduler {
        return .{
            .current_draft_tokens = @min(cfg.initial_draft_tokens, cfg.max_draft_tokens),
            .min_tokens = cfg.min_draft_tokens,
            .max_tokens = cfg.max_draft_tokens,
        };
    }

    pub fn update(self: *DynamicDraftScheduler, stats: *SpeculativeStats, batch_accepted: usize, batch_proposed: usize) void {
        if (batch_proposed == 0) return;
        const batch_rate = @as(f32, @floatFromInt(batch_accepted)) / @as(f32, @floatFromInt(batch_proposed));
        stats.ema_acceptance_rate = (1.0 - self.ema_alpha) * stats.ema_acceptance_rate + self.ema_alpha * batch_rate;

        if (stats.ema_acceptance_rate > 0.80 and self.current_draft_tokens < self.max_tokens) {
            self.current_draft_tokens += 1;
        } else if (stats.ema_acceptance_rate < 0.50 and self.current_draft_tokens > self.min_tokens) {
            self.current_draft_tokens -= 1;
        }
    }
};

/// Memory-first speculative decoding verifier.
pub const SpeculativeVerifier = struct {
    allocator: std.mem.Allocator,
    cfg: SpeculativeConfig,
    vocab_size: usize,
    prng: std.Random.DefaultPrng,
    scheduler: DynamicDraftScheduler,
    stats: SpeculativeStats = .{},
    residual_buf: []f32,

    pub fn init(allocator: std.mem.Allocator, cfg: SpeculativeConfig, vocab_size: usize, seed: u64) !SpeculativeVerifier {
        std.debug.assert(vocab_size > 0);
        std.debug.assert(cfg.max_draft_tokens <= VerificationResult.max_capacity);
        const residual = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(residual);

        var s = seed;
        if (s == 0) s = 0x9E3779B97F4A7C15;

        return .{
            .allocator = allocator,
            .cfg = cfg,
            .vocab_size = vocab_size,
            .prng = std.Random.DefaultPrng.init(s),
            .scheduler = DynamicDraftScheduler.init(cfg),
            .residual_buf = residual,
        };
    }

    pub fn deinit(self: *SpeculativeVerifier) void {
        self.allocator.free(self.residual_buf);
    }

    /// Samples a token from an arbitrary probability distribution.
    pub fn sampleFromDistribution(self: *SpeculativeVerifier, probs: []const f32) ?u32 {
        std.debug.assert(probs.len == self.vocab_size);
        var sum: f32 = 0.0;
        for (probs) |p| {
            if (p > 0.0) sum += p;
        }
        if (sum <= 1e-12) return null;

        const r = self.prng.random().float(f32) * sum;
        var cum: f32 = 0.0;
        for (probs, 0..) |p, i| {
            if (p <= 0.0) continue;
            cum += p;
            if (cum >= r) return @intCast(i);
        }
        return @intCast(self.vocab_size - 1);
    }

    /// Verifies a draft sequence against target probability distributions.
    ///
    /// `draft_tokens`: slice of length K tokens proposed by draft model.
    /// `draft_probs`: slice of length K probability distributions over vocab (size K * vocab_size).
    /// `target_probs`: slice of length (K + 1) probability distributions over vocab (size (K + 1) * vocab_size).
    pub fn verifyBatch(
        self: *SpeculativeVerifier,
        draft_tokens: []const u32,
        draft_probs: []const f32,
        target_probs: []const f32,
    ) !VerificationResult {
        const k = draft_tokens.len;
        if (k > self.cfg.max_draft_tokens) return error.DraftExceedsCapacity;
        if (draft_probs.len != k * self.vocab_size) return error.InvalidDraftProbsLength;
        if (target_probs.len != (k + 1) * self.vocab_size) return error.InvalidTargetProbsLength;

        var result = VerificationResult{};
        var accepted_count: usize = 0;

        self.stats.verification_steps += 1;
        self.stats.total_draft_proposed += k;

        var reject_idx: ?usize = null;

        for (0..k) |i| {
            const tok = draft_tokens[i];
            if (tok >= self.vocab_size) return error.TokenOutOfBounds;

            const p_draft = draft_probs[i * self.vocab_size + tok];
            const p_target = target_probs[i * self.vocab_size + tok];

            const alpha = if (p_draft <= 1e-12)
                1.0
            else
                @min(1.0, p_target / p_draft);

            const r = self.prng.random().float(f32);
            if (r <= alpha) {
                // Token accepted
                result.accepted_tokens[accepted_count] = tok;
                accepted_count += 1;
            } else {
                // Token rejected: stop accepting further draft tokens
                reject_idx = i;
                break;
            }
        }

        result.n_accepted = accepted_count;
        self.stats.total_draft_accepted += accepted_count;

        if (reject_idx) |idx| {
            // Rejection sampling from residual distribution max(0, P_target - P_draft)
            const target_slice = target_probs[idx * self.vocab_size .. (idx + 1) * self.vocab_size];
            const draft_slice = draft_probs[idx * self.vocab_size .. (idx + 1) * self.vocab_size];

            var residual_sum: f32 = 0.0;
            for (0..self.vocab_size) |v| {
                const diff = target_slice[v] - draft_slice[v];
                const res = if (diff > 0.0) diff else 0.0;
                self.residual_buf[v] = res;
                residual_sum += res;
            }

            if (residual_sum > 1e-12) {
                const norm = 1.0 / residual_sum;
                for (self.residual_buf) |*v| v.* *= norm;
                result.correction_token = self.sampleFromDistribution(self.residual_buf);
            } else {
                // Fallback to direct sampling from target slice
                result.correction_token = self.sampleFromDistribution(target_slice);
            }

            result.all_accepted = false;
        } else {
            // All draft tokens were accepted! Sample bonus (K+1)-th token from target_probs[K]
            const bonus_slice = target_probs[k * self.vocab_size .. (k + 1) * self.vocab_size];
            result.correction_token = self.sampleFromDistribution(bonus_slice);
            result.all_accepted = true;
        }

        if (result.correction_token != null) {
            self.stats.total_correction_tokens += 1;
            result.total_emitted = accepted_count + 1;
        } else {
            result.total_emitted = accepted_count;
        }

        if (self.cfg.dynamic_length) {
            self.scheduler.update(&self.stats, accepted_count, k);
        }

        return result;
    }
};

test "speculative decoding: exact acceptance under identical distributions" {
    const allocator = std.testing.allocator;
    const vocab_size = 8;
    const k = 3;

    var verifier = try SpeculativeVerifier.init(allocator, .{
        .max_draft_tokens = 5,
        .initial_draft_tokens = 3,
        .min_draft_tokens = 1,
    }, vocab_size, 42);
    defer verifier.deinit();

    const draft_tokens = [_]u32{ 1, 3, 5 };

    // Equal probability distribution over vocab for draft
    var draft_probs: [k * vocab_size]f32 = undefined;
    for (0..k) |step| {
        for (0..vocab_size) |v| {
            draft_probs[step * vocab_size + v] = 1.0 / @as(f32, @floatFromInt(vocab_size));
        }
    }

    // Identical target probability distribution for 0..k
    var target_probs: [(k + 1) * vocab_size]f32 = undefined;
    for (0..k + 1) |step| {
        for (0..vocab_size) |v| {
            target_probs[step * vocab_size + v] = 1.0 / @as(f32, @floatFromInt(vocab_size));
        }
    }

    const res = try verifier.verifyBatch(&draft_tokens, &draft_probs, &target_probs);

    // When distributions match, alpha = min(1, p_target/p_draft) = 1.0, so all draft tokens MUST be accepted
    try std.testing.expectEqual(@as(usize, 3), res.n_accepted);
    try std.testing.expect(res.all_accepted);
    try std.testing.expectEqual(@as(u32, 1), res.accepted_tokens[0]);
    try std.testing.expectEqual(@as(u32, 3), res.accepted_tokens[1]);
    try std.testing.expectEqual(@as(u32, 5), res.accepted_tokens[2]);
    try std.testing.expect(res.correction_token != null);
    try std.testing.expectEqual(@as(usize, 4), res.total_emitted);
}

test "speculative decoding: rejection and residual resampling" {
    const allocator = std.testing.allocator;
    const vocab_size = 4;

    var verifier = try SpeculativeVerifier.init(allocator, .{
        .max_draft_tokens = 4,
        .initial_draft_tokens = 2,
    }, vocab_size, 12345);
    defer verifier.deinit();

    const draft_tokens = [_]u32{ 0, 1 };

    // Draft proposes token 0 with 100% certainty, but target has 0% for token 0
    var draft_probs = [_]f32{
        1.0, 0.0, 0.0, 0.0, // step 0
        0.0, 1.0, 0.0, 0.0, // step 1
    };

    var target_probs = [_]f32{
        0.0, 0.0, 1.0, 0.0, // step 0 (token 2 has 100%)
        0.0, 1.0, 0.0, 0.0, // step 1
        0.25, 0.25, 0.25, 0.25, // bonus step
    };

    const res = try verifier.verifyBatch(&draft_tokens, &draft_probs, &target_probs);

    // Step 0 must be rejected since p_target(0) / p_draft(0) = 0.0 / 1.0 = 0.0
    try std.testing.expectEqual(@as(usize, 0), res.n_accepted);
    try std.testing.expect(!res.all_accepted);
    try std.testing.expect(res.correction_token != null);
    // Correction token must be resampled from target residual, which is token 2
    try std.testing.expectEqual(@as(u32, 2), res.correction_token.?);
    try std.testing.expectEqual(@as(usize, 1), res.total_emitted);
}

test "speculative decoding: dynamic length scheduling" {
    var scheduler = DynamicDraftScheduler.init(.{
        .min_draft_tokens = 2,
        .max_draft_tokens = 6,
        .initial_draft_tokens = 3,
    });
    var stats = SpeculativeStats{};

    // Simulate repeated high acceptance rate (100%)
    for (0..10) |_| {
        scheduler.update(&stats, 3, 3);
    }
    // Should adapt up towards max_draft_tokens (6)
    try std.testing.expect(scheduler.current_draft_tokens >= 4);

    // Simulate repeated low acceptance rate (0%)
    for (0..10) |_| {
        scheduler.update(&stats, 0, 3);
    }
    // Should adapt down towards min_draft_tokens (2)
    try std.testing.expect(scheduler.current_draft_tokens <= 3);
}
