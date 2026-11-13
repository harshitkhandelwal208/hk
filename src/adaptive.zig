const std = @import("std");
const growth = @import("growth.zig");

pub const GrowthGovernor = struct {
    max_vram_mb: u64,
    max_growth_ratio: f32,

    pub fn init(max_vram_mb: u64, max_growth_ratio: f32) GrowthGovernor {
        return .{
            .max_vram_mb = max_vram_mb,
            .max_growth_ratio = max_growth_ratio,
        };
    }

    pub fn canGrow(
        self: GrowthGovernor,
        current_params: u64,
        added_params: u64,
        dtype_bytes: u32,
        out_reason: []u8,
    ) struct { approved: bool, reason: []const u8 } {
        const total_params = current_params + added_params;
        const cur_f = @as(f32, @floatFromInt(current_params));
        const tot_f = @as(f32, @floatFromInt(total_params));
        const ratio = if (cur_f > 0) tot_f / cur_f else 1.0;

        if (ratio > self.max_growth_ratio) {
            const msg = std.fmt.bufPrint(out_reason, "Growth ratio {d:.2}x exceeds max limit {d:.2}x", .{ ratio, self.max_growth_ratio }) catch "Growth ratio exceeds limit";
            return .{ .approved = false, .reason = msg };
        }

        const added_bytes = added_params * dtype_bytes;
        const added_mb = added_bytes / (1024 * 1024);
        if (added_mb > self.max_vram_mb) {
            const msg = std.fmt.bufPrint(out_reason, "Memory requirement {d} MB exceeds budget {d} MB", .{ added_mb, self.max_vram_mb }) catch "Memory exceeds budget";
            return .{ .approved = false, .reason = msg };
        }

        const msg = std.fmt.bufPrint(out_reason, "Approved", .{}) catch "Approved";
        return .{ .approved = true, .reason = msg };
    }
};

/// Expands vocabulary embeddings [V, D] -> [V', D] with exact function preservation on 0..V-1
pub fn expandVocabEmbeddings(
    old_embed: []const f32,
    old_vocab: usize,
    hidden_size: usize,
    new_vocab: usize,
    new_embed: []f32,
    init_std: f32,
    seed: u64,
) void {
    if (new_vocab <= old_vocab) return;

    // 1. Copy old embeddings exactly
    const old_elements = old_vocab * hidden_size;
    @memcpy(new_embed[0..old_elements], old_embed[0..old_elements]);

    // 2. Initialize newly added tokens with normal distribution
    var prng = growth.Prng.init(seed);
    const added_tokens = new_vocab - old_vocab;
    const added_elements = added_tokens * hidden_size;
    const new_slice = new_embed[old_elements .. old_elements + added_elements];

    for (0..added_elements) |i| {
        new_slice[i] = prng.nextGaussian(init_std);
    }
}

/// Initializes plasticity shields / capacity preservation gradient masks
pub fn initPlasticityMask(
    mask: []f32,
    total_units: usize,
    base_units: usize,
    decay_rate: f32,
) void {
    const n = @min(mask.len, total_units);
    for (0..n) |i| {
        if (i < base_units) {
            // Protect base units with lower learning rate factor (plasticity shield)
            mask[i] = decay_rate;
        } else {
            // Full plasticity for expanded units
            mask[i] = 1.0;
        }
    }
}

/// Continual learning regression governor (Plan §30, §85):
/// Monitors evaluation metrics after model updates or growth.
/// Enforces fail-closed rollback if performance regresses beyond threshold.
pub const RegressionGovernor = struct {
    baseline_loss: f32,
    max_regression_ratio: f32, // e.g. 1.05 (allows up to 5% regression)

    pub fn init(baseline_loss: f32, max_regression_ratio: f32) RegressionGovernor {
        return .{
            .baseline_loss = baseline_loss,
            .max_regression_ratio = max_regression_ratio,
        };
    }

    pub fn shouldRollback(self: RegressionGovernor, candidate_loss: f32) bool {
        // Fail-closed: non-finite loss immediately triggers rollback
        if (!std.math.isFinite(candidate_loss)) return true;
        return candidate_loss > (self.baseline_loss * self.max_regression_ratio);
    }
};

test "GrowthGovernor: budget validation and limits" {
    var reason_buf: [128]u8 = undefined;

    // 1. Valid growth within 2.0x ratio and within 4096 MB
    const gov = GrowthGovernor.init(4096, 2.0);
    const res1 = gov.canGrow(1_000_000, 500_000, 4, &reason_buf);
    try std.testing.expect(res1.approved);
    try std.testing.expectEqualStrings("Approved", res1.reason);

    // 2. Growth exceeding max_growth_ratio (1.0M + 1.5M = 2.5M, ratio = 2.5x > 2.0x)
    const res2 = gov.canGrow(1_000_000, 1_500_000, 4, &reason_buf);
    try std.testing.expect(!res2.approved);

    // 3. Growth exceeding max_vram_mb (10 MB budget, adding 20 MB)
    const tiny_gov = GrowthGovernor.init(10, 2.0);
    const res3 = tiny_gov.canGrow(10_000_000, 5_000_000, 4, &reason_buf); // 5M * 4 bytes = 20 MB > 10 MB
    try std.testing.expect(!res3.approved);
}

test "RegressionGovernor: rollback on regression detection" {
    const gov = RegressionGovernor.init(2.0, 1.05); // baseline loss = 2.0, max acceptable = 2.10

    // Small improvement or harmless variation
    try std.testing.expect(!gov.shouldRollback(1.95));
    try std.testing.expect(!gov.shouldRollback(2.05));

    // Severe regression triggers rollback
    try std.testing.expect(gov.shouldRollback(2.25));

    // NaN / Inf triggers fail-closed rollback
    try std.testing.expect(gov.shouldRollback(std.math.nan(f32)));
    try std.testing.expect(gov.shouldRollback(std.math.inf(f32)));
}

test "expandVocabEmbeddings: exact base preservation and initialization" {
    const allocator = std.testing.allocator;
    const old_vocab: usize = 3;
    const new_vocab: usize = 6;
    const hidden_size: usize = 4;

    const old_embed = try allocator.alloc(f32, old_vocab * hidden_size);
    defer allocator.free(old_embed);
    for (0..old_embed.len) |i| old_embed[i] = @as(f32, @floatFromInt(i + 1)) * 0.1;

    const new_embed = try allocator.alloc(f32, new_vocab * hidden_size);
    defer allocator.free(new_embed);

    expandVocabEmbeddings(old_embed, old_vocab, hidden_size, new_vocab, new_embed, 0.02, 42);

    // Base tokens must be exactly equal
    for (0..old_embed.len) |i| {
        try std.testing.expectEqual(old_embed[i], new_embed[i]);
    }

    // New tokens must be finite and initialized
    for (old_embed.len..new_embed.len) |i| {
        try std.testing.expect(std.math.isFinite(new_embed[i]));
    }
}

test "initPlasticityMask: protection of base units" {
    var mask: [8]f32 = undefined;
    initPlasticityMask(&mask, 8, 4, 0.1);

    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.1), mask[i], 1e-6);
    }
    for (4..8) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), mask[i], 1e-6);
    }
}
