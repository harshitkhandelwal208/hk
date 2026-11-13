//! Autonomous Growth & Model Evolution Engine (plan §36, §150 milestone 2.6).
//!
//! Orchestrates autonomous architectural mutation (Net2Wider, Net2Deeper, LoRA),
//! candidate evaluation, hardware governor validation, fail-closed regression rollback,
//! and automated lineage tracking.

const std = @import("std");
const adaptive = @import("adaptive.zig");
const growth = @import("growth.zig");
const lineage = @import("lineage.zig");

pub const EvolutionDecision = enum {
    promoted,
    rolled_back_regression,
    rejected_budget_exceeded,
};

pub const EvolutionCandidate = struct {
    name: []const u8,
    mutation: lineage.MutationType,
    eval_loss: f32,
    accuracy: f32,
    parameter_count: u64,
    vram_bytes: u64,
};

pub const EvolutionReport = struct {
    decision: EvolutionDecision,
    generation: u32,
    commit_id: ?[32]u8 = null,
    delta_loss: f32 = 0.0,
};

pub const EvolutionEngine = struct {
    allocator: std.mem.Allocator,
    growth_gov: adaptive.GrowthGovernor,
    regression_gov: adaptive.RegressionGovernor,
    graph: *lineage.LineageGraph,
    current_generation: u32 = 0,
    current_params: u64,
    baseline_loss: f32,
    head_commit: [32]u8,

    pub fn init(
        allocator: std.mem.Allocator,
        graph: *lineage.LineageGraph,
        initial_params: u64,
        max_vram_mb: u64,
        max_growth_ratio: f32,
        baseline_loss: f32,
        baseline_acc: f32,
        branch_name: []const u8,
    ) !EvolutionEngine {
        const root = try graph.createRootCommit(branch_name, baseline_loss, baseline_acc);
        const g_gov = adaptive.GrowthGovernor.init(max_vram_mb, max_growth_ratio);
        const r_gov = adaptive.RegressionGovernor.init(baseline_loss, 1.05);

        return .{
            .allocator = allocator,
            .growth_gov = g_gov,
            .regression_gov = r_gov,
            .graph = graph,
            .current_generation = 0,
            .current_params = initial_params,
            .baseline_loss = baseline_loss,
            .head_commit = root.id,
        };
    }

    /// Evaluates an autonomous candidate and either promotes or rolls back
    pub fn step(
        self: *EvolutionEngine,
        branch_name: []const u8,
        cand: EvolutionCandidate,
    ) !EvolutionReport {
        // 1. Hardware budget check via GrowthGovernor
        const added_params = if (cand.parameter_count > self.current_params)
            cand.parameter_count - self.current_params
        else
            0;
        var reason_buf: [128]u8 = undefined;
        const growth_check = self.growth_gov.canGrow(self.current_params, added_params, 4, &reason_buf);
        if (!growth_check.approved) {
            return .{
                .decision = .rejected_budget_exceeded,
                .generation = self.current_generation,
                .delta_loss = cand.eval_loss - self.baseline_loss,
            };
        }

        // 2. Performance regression check via RegressionGovernor
        if (self.regression_gov.shouldRollback(cand.eval_loss)) {
            return .{
                .decision = .rolled_back_regression,
                .generation = self.current_generation,
                .delta_loss = cand.eval_loss - self.baseline_loss,
            };
        }

        // 3. Candidate accepted! Create commit in lineage DAG
        const commit = try self.graph.createCommit(
            branch_name,
            self.head_commit,
            cand.mutation,
            cand.eval_loss,
            cand.accuracy,
        );

        const delta = cand.eval_loss - self.baseline_loss;
        self.head_commit = commit.id;
        self.current_generation = commit.generation;
        self.current_params = cand.parameter_count;
        self.baseline_loss = cand.eval_loss;

        return .{
            .decision = .promoted,
            .generation = commit.generation,
            .commit_id = commit.id,
            .delta_loss = delta,
        };
    }
};

test "evolution: autonomous candidate promotion and regression rollback" {
    const allocator = std.testing.allocator;
    var graph = lineage.LineageGraph.init(allocator);
    defer graph.deinit();

    var engine = try EvolutionEngine.init(
        allocator,
        &graph,
        1_000_000, // 1M initial params
        100, // 100 MB max VRAM
        2.0, // 2.0x max growth ratio
        2.5, // baseline loss 2.5
        0.60,
        "main",
    );

    // Candidate 1: Net2Wider improves loss from 2.5 to 2.2 within budget -> Promote!
    const cand1 = EvolutionCandidate{
        .name = "wider_1",
        .mutation = .net2wider,
        .eval_loss = 2.2,
        .accuracy = 0.68,
        .parameter_count = 1_500_000,
        .vram_bytes = 20_000_000,
    };
    const rep1 = try engine.step("main", cand1);
    try std.testing.expectEqual(EvolutionDecision.promoted, rep1.decision);
    try std.testing.expectEqual(@as(u32, 1), rep1.generation);
    try std.testing.expect(rep1.commit_id != null);

    // Candidate 2: Severe regression (loss increases from 2.2 to 3.5) -> Rollback!
    const cand2 = EvolutionCandidate{
        .name = "bad_lora",
        .mutation = .lora_finetune,
        .eval_loss = 3.5,
        .accuracy = 0.40,
        .parameter_count = 1_600_000,
        .vram_bytes = 22_000_000,
    };
    const rep2 = try engine.step("main", cand2);
    try std.testing.expectEqual(EvolutionDecision.rolled_back_regression, rep2.decision);
    try std.testing.expectEqual(@as(u32, 1), rep2.generation); // Unchanged generation!

    // Candidate 3: Exceeds maximum growth ratio budget (3.5M > 2.0x of 1.5M = 3.0M) -> Rejected budget!
    const cand3 = EvolutionCandidate{
        .name = "giant_wider",
        .mutation = .net2wider,
        .eval_loss = 1.8,
        .accuracy = 0.75,
        .parameter_count = 3_500_000,
        .vram_bytes = 60_000_000,
    };
    const rep3 = try engine.step("main", cand3);
    try std.testing.expectEqual(EvolutionDecision.rejected_budget_exceeded, rep3.decision);
}
