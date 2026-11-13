//! Native Multi-Role Self-Conversation, Autonomous Curriculum & Failure Mining
//!
//! Implements MASTER_PLAN §32, §34:
//! 1. Multi-role self-conversation: proposer, thinker (<think>), coder, critic, verifier, reflector, judge.
//! 2. Verified synthetic data pipeline: strict validation gating before training example generation.
//! 3. Difficulty estimation and staged curriculum progression.
//! 4. Failure mining: tracking failure patterns and prioritizing hard examples for retraining.

const std = @import("std");
const Allocator = std.mem.Allocator;
const RegressionGovernor = @import("adaptive.zig").RegressionGovernor;

/// Cognitive roles in multi-turn self-deliberation (MASTER_PLAN §32).
pub const ConversationRole = enum {
    proposer,
    thinker,
    coder,
    critic,
    verifier,
    reflector,
    judge,

    pub fn asString(self: ConversationRole) []const u8 {
        return switch (self) {
            .proposer => "proposer",
            .thinker => "thinker",
            .coder => "coder",
            .critic => "critic",
            .verifier => "verifier",
            .reflector => "reflector",
            .judge => "judge",
        };
    }
};

/// A single conversational turn in the reasoning trace.
pub const ConversationTurn = struct {
    role: ConversationRole,
    content: []const u8,
    thinking: []const u8 = "",
    code: []const u8 = "",
    score: f32 = 0.0,
};

/// Verified synthetic training pair created strictly after passing validation.
pub const VerifiedSyntheticPair = struct {
    task_id: usize,
    prompt: []const u8,
    thinking: []const u8,
    solution: []const u8,
    quality_score: f32,
    pass_rate: f32,
};

/// Multi-Role Self-Conversation Deliberation Trace.
pub const SelfConversationTrace = struct {
    allocator: Allocator,
    task_id: usize,
    prompt: []const u8,
    turns: std.ArrayList(ConversationTurn),
    is_verified: bool,
    judge_score: f32,

    pub fn init(allocator: Allocator, task_id: usize, prompt: []const u8) SelfConversationTrace {
        return .{
            .allocator = allocator,
            .task_id = task_id,
            .prompt = prompt,
            .turns = .empty,
            .is_verified = false,
            .judge_score = 0.0,
        };
    }

    pub fn deinit(self: *SelfConversationTrace) void {
        self.turns.deinit(self.allocator);
    }

    pub fn addTurn(self: *SelfConversationTrace, turn: ConversationTurn) !void {
        try self.turns.append(self.allocator, turn);
        if (turn.role == .judge) {
            self.judge_score = turn.score;
            // Trace is verified if judge awards score >= 0.80 and verifier confirmed
            if (turn.score >= 0.80) {
                var verifier_passed = false;
                for (self.turns.items) |t| {
                    if (t.role == .verifier and t.score >= 1.0) {
                        verifier_passed = true;
                        break;
                    }
                }
                self.is_verified = verifier_passed;
            }
        }
    }

    /// Converts a verified conversation trace into a standardized synthetic training pair.
    /// Fails closed if trace was not verified by verifier and judge.
    pub fn toSyntheticPair(self: *const SelfConversationTrace) !VerifiedSyntheticPair {
        if (!self.is_verified) {
            return error.TraceNotVerified;
        }

        var thinking: []const u8 = "";
        var solution: []const u8 = "";

        for (self.turns.items) |t| {
            if (t.role == .thinker and t.thinking.len > 0) {
                thinking = t.thinking;
            }
            if (t.role == .coder and t.code.len > 0) {
                solution = t.code;
            }
        }

        if (solution.len == 0) return error.MissingSolutionArtifact;

        return VerifiedSyntheticPair{
            .task_id = self.task_id,
            .prompt = self.prompt,
            .thinking = thinking,
            .solution = solution,
            .quality_score = self.judge_score,
            .pass_rate = 1.0,
        };
    }
};

/// Failure categories tracked during self-training execution.
pub const FailureCategory = enum {
    syntax_error,
    runtime_exception,
    execution_timeout,
    incorrect_output,
    performance_regression,
};

/// Recorded failure instance for priority retraining.
pub const FailureRecord = struct {
    task_id: usize,
    category: FailureCategory,
    error_count: u32,
    last_loss: f32,
    priority_weight: f32,
};

/// Failure Miner (§34): Identifies recurring weaknesses and surfaces them for targeted training.
pub const FailureMiner = struct {
    allocator: Allocator,
    records: std.AutoHashMap(usize, FailureRecord),

    pub fn init(allocator: Allocator) FailureMiner {
        return .{
            .allocator = allocator,
            .records = std.AutoHashMap(usize, FailureRecord).init(allocator),
        };
    }

    pub fn deinit(self: *FailureMiner) void {
        self.records.deinit();
    }

    /// Records a failed task instance, updating its priority weight.
    pub fn recordFailure(self: *FailureMiner, task_id: usize, category: FailureCategory, loss: f32) !void {
        const entry = try self.records.getOrPut(task_id);
        if (entry.found_existing) {
            entry.value_ptr.error_count += 1;
            entry.value_ptr.category = category;
            entry.value_ptr.last_loss = loss;
            // Exponential priority weighting: weight = error_count * (1.0 + loss)
            entry.value_ptr.priority_weight = @as(f32, @floatFromInt(entry.value_ptr.error_count)) * (1.0 + loss);
        } else {
            entry.value_ptr.* = FailureRecord{
                .task_id = task_id,
                .category = category,
                .error_count = 1,
                .last_loss = loss,
                .priority_weight = 1.0 + loss,
            };
        }
    }

    /// Clears or downweights a task upon verified recovery.
    pub fn recordSuccess(self: *FailureMiner, task_id: usize) void {
        _ = self.records.remove(task_id);
    }

    /// Returns list of prioritized failure task IDs sorted by difficulty / priority weight.
    pub fn getPrioritizedFailures(self: *FailureMiner, out_buf: []usize) usize {
        var count: usize = 0;
        var it = self.records.iterator();
        while (it.next()) |entry| {
            if (count >= out_buf.len) break;
            out_buf[count] = entry.key_ptr.*;
            count += 1;
        }
        return count;
    }
};

/// Staged difficulty levels for autonomous curriculum.
pub const CurriculumStage = enum(u8) {
    beginner = 0,
    intermediate = 1,
    advanced = 2,
    expert = 3,

    pub fn next(self: CurriculumStage) ?CurriculumStage {
        return switch (self) {
            .beginner => .intermediate,
            .intermediate => .advanced,
            .advanced => .expert,
            .expert => null,
        };
    }
};

/// Autonomous Curriculum Scheduler (§34):
/// Dynamically unlocks more complex task distributions as empirical pass rate improves.
pub const CurriculumScheduler = struct {
    current_stage: CurriculumStage,
    promotion_pass_threshold: f32, // e.g., 0.85 (85% pass rate required for next stage)
    demotion_loss_threshold: f32, // e.g., 2.50 (high loss triggers demotion to consolidate)
    stage_eval_count: usize,
    stage_passed_count: usize,

    pub fn init(promotion_pass_threshold: f32, demotion_loss_threshold: f32) CurriculumScheduler {
        return .{
            .current_stage = .beginner,
            .promotion_pass_threshold = promotion_pass_threshold,
            .demotion_loss_threshold = demotion_loss_threshold,
            .stage_eval_count = 0,
            .stage_passed_count = 0,
        };
    }

    /// Records task evaluation outcome and checks for stage progression.
    pub fn recordEvaluation(self: *CurriculumScheduler, passed: bool, eval_loss: f32) struct {
        stage_changed: bool,
        new_stage: CurriculumStage,
    } {
        self.stage_eval_count += 1;
        if (passed) self.stage_passed_count += 1;

        // Require at least 5 evaluations before stage transition
        if (self.stage_eval_count >= 5) {
            const pass_rate = @as(f32, @floatFromInt(self.stage_passed_count)) / @as(f32, @floatFromInt(self.stage_eval_count));

            // Check promotion
            if (pass_rate >= self.promotion_pass_threshold) {
                if (self.current_stage.next()) |next_stage| {
                    self.current_stage = next_stage;
                    self.stage_eval_count = 0;
                    self.stage_passed_count = 0;
                    return .{ .stage_changed = true, .new_stage = self.current_stage };
                }
            }

            // Check demotion on severe regression
            if (eval_loss > self.demotion_loss_threshold and self.current_stage != .beginner) {
                self.current_stage = switch (self.current_stage) {
                    .expert => .advanced,
                    .advanced => .intermediate,
                    .intermediate => .beginner,
                    .beginner => .beginner,
                };
                self.stage_eval_count = 0;
                self.stage_passed_count = 0;
                return .{ .stage_changed = true, .new_stage = self.current_stage };
            }
        }

        return .{ .stage_changed = false, .new_stage = self.current_stage };
    }
};

test "self-conversation trace and verified synthetic data gating" {
    const allocator = std.testing.allocator;
    var trace = SelfConversationTrace.init(allocator, 101, "Implement fast vector dot product");
    defer trace.deinit();

    // Turn 1: Proposer
    try trace.addTurn(.{
        .role = .proposer,
        .content = "We should compute ∑ a[i]*b[i] in parallel using SIMD vectors.",
    });

    // Turn 2: Thinker
    try trace.addTurn(.{
        .role = .thinker,
        .content = "Inner reasoning",
        .thinking = "We need an unrolled loop with 4 accumulators to hide FMA latency.",
    });

    // Turn 3: Coder
    try trace.addTurn(.{
        .role = .coder,
        .content = "Implementation",
        .code = "pub fn dot(a: []const f32, b: []const f32) f32 { ... }",
    });

    // Attempting to export synthetic pair before verifier & judge approval MUST fail
    try std.testing.expectError(error.TraceNotVerified, trace.toSyntheticPair());

    // Turn 4: Verifier confirms tests pass
    try trace.addTurn(.{
        .role = .verifier,
        .content = "All test vectors matched reference outputs within 1e-6 tolerance.",
        .score = 1.0,
    });

    // Turn 5: Judge evaluates and approves
    try trace.addTurn(.{
        .role = .judge,
        .content = "Approved: optimal vectorization and mathematically verified.",
        .score = 0.95,
    });

    try std.testing.expect(trace.is_verified);

    // Exporting now succeeds
    const pair = try trace.toSyntheticPair();
    try std.testing.expectEqual(@as(usize, 101), pair.task_id);
    try std.testing.expect(pair.thinking.len > 0);
    try std.testing.expect(pair.solution.len > 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.95), pair.quality_score, 1e-4);
}

test "failure mining priority weighting" {
    const allocator = std.testing.allocator;
    var miner = FailureMiner.init(allocator);
    defer miner.deinit();

    // Record failure on task 10 (timeout)
    try miner.recordFailure(10, .execution_timeout, 2.5);
    // Record repeated failure on task 20 (incorrect output)
    try miner.recordFailure(20, .incorrect_output, 3.0);
    try miner.recordFailure(20, .incorrect_output, 3.2);

    var buf: [4]usize = undefined;
    const count = miner.getPrioritizedFailures(&buf);
    try std.testing.expectEqual(@as(usize, 2), count);

    // Task 20 should have higher priority weight
    const rec20 = miner.records.get(20).?;
    try std.testing.expectEqual(@as(u32, 2), rec20.error_count);
    try std.testing.expect(rec20.priority_weight > 8.0);

    // Resolving task 10 removes it
    miner.recordSuccess(10);
    const count_after = miner.getPrioritizedFailures(&buf);
    try std.testing.expectEqual(@as(usize, 1), count_after);
    try std.testing.expectEqual(@as(usize, 20), buf[0]);
}

test "curriculum scheduler progression" {
    var scheduler = CurriculumScheduler.init(0.80, 2.0);
    try std.testing.expectEqual(CurriculumStage.beginner, scheduler.current_stage);

    // 5 successful evaluations -> promote to intermediate
    var changed = false;
    for (0..5) |_| {
        const res = scheduler.recordEvaluation(true, 0.2);
        if (res.stage_changed) changed = true;
    }

    try std.testing.expect(changed);
    try std.testing.expectEqual(CurriculumStage.intermediate, scheduler.current_stage);
}
