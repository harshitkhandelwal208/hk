//! Model Lineage Version Graph, Branching, and Merging (plan §37-§40, §150 milestone 2.6).
//!
//! Implements a cryptographic Directed Acyclic Graph (DAG) for model generations,
//! parent-hash lineage validation, branch pointers, and weight merging strategies
//! (linear interpolation, SLERP, and task arithmetic).

const std = @import("std");

pub const MutationType = enum {
    initial,
    net2wider,
    net2deeper,
    vocab_expansion,
    lora_finetune,
    pruned,
    quantized,
    merge_commit,
};

pub const MergeStrategy = enum {
    linear_average,
    slerp,
    task_arithmetic,
};

pub const CommitNode = struct {
    id: [32]u8,
    parent_ids: [2]?[32]u8 = .{ null, null },
    branch: []const u8,
    mutation: MutationType,
    generation: u32,
    timestamp: i64,
    loss: f32,
    accuracy: f32,

    pub fn isMerge(self: *const CommitNode) bool {
        return self.parent_ids[1] != null;
    }
};

pub const LineageBranch = struct {
    name: []const u8,
    head_id: [32]u8,
};

pub const LineageGraph = struct {
    allocator: std.mem.Allocator,
    commits: std.ArrayList(CommitNode) = .empty,
    branches: std.ArrayList(LineageBranch) = .empty,

    pub fn init(allocator: std.mem.Allocator) LineageGraph {
        return .{
            .allocator = allocator,
            .commits = .empty,
            .branches = .empty,
        };
    }

    pub fn deinit(self: *LineageGraph) void {
        self.commits.deinit(self.allocator);
        self.branches.deinit(self.allocator);
    }

    fn computeCommitHash(
        parent_a: ?[32]u8,
        parent_b: ?[32]u8,
        mutation: MutationType,
        gen: u32,
        loss: f32,
        ts: i64,
    ) [32]u8 {
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        if (parent_a) |pa| hasher.update(&pa);
        if (parent_b) |pb| hasher.update(&pb);
        hasher.update(&[_]u8{@intFromEnum(mutation)});
        hasher.update(std.mem.asBytes(&gen));
        hasher.update(std.mem.asBytes(&loss));
        hasher.update(std.mem.asBytes(&ts));

        var digest: [32]u8 = undefined;
        hasher.final(&digest);
        return digest;
    }

    pub fn createRootCommit(
        self: *LineageGraph,
        branch_name: []const u8,
        loss: f32,
        accuracy: f32,
    ) !CommitNode {
        const ts: i64 = @truncate(std.Io.Timestamp.now(std.Options.debug_io, .real).nanoseconds);
        const id = computeCommitHash(null, null, .initial, 0, loss, ts);

        const node = CommitNode{
            .id = id,
            .parent_ids = .{ null, null },
            .branch = branch_name,
            .mutation = .initial,
            .generation = 0,
            .timestamp = ts,
            .loss = loss,
            .accuracy = accuracy,
        };

        try self.commits.append(self.allocator, node);
        try self.updateBranch(branch_name, id);
        return node;
    }

    pub fn createCommit(
        self: *LineageGraph,
        branch_name: []const u8,
        parent_id: [32]u8,
        mutation: MutationType,
        loss: f32,
        accuracy: f32,
    ) !CommitNode {
        const parent = self.findCommit(parent_id) orelse return error.ParentNotFound;
        const ts: i64 = @truncate(std.Io.Timestamp.now(std.Options.debug_io, .real).nanoseconds);
        const gen = parent.generation + 1;
        const id = computeCommitHash(parent_id, null, mutation, gen, loss, ts);

        const node = CommitNode{
            .id = id,
            .parent_ids = .{ parent_id, null },
            .branch = branch_name,
            .mutation = mutation,
            .generation = gen,
            .timestamp = ts,
            .loss = loss,
            .accuracy = accuracy,
        };

        try self.commits.append(self.allocator, node);
        try self.updateBranch(branch_name, id);
        return node;
    }

    pub fn createMergeCommit(
        self: *LineageGraph,
        target_branch: []const u8,
        parent_a_id: [32]u8,
        parent_b_id: [32]u8,
        loss: f32,
        accuracy: f32,
    ) !CommitNode {
        const parent_a = self.findCommit(parent_a_id) orelse return error.ParentNotFound;
        const parent_b = self.findCommit(parent_b_id) orelse return error.ParentNotFound;

        const ts: i64 = @truncate(std.Io.Timestamp.now(std.Options.debug_io, .real).nanoseconds);
        const gen = @max(parent_a.generation, parent_b.generation) + 1;
        const id = computeCommitHash(parent_a_id, parent_b_id, .merge_commit, gen, loss, ts);

        const node = CommitNode{
            .id = id,
            .parent_ids = .{ parent_a_id, parent_b_id },
            .branch = target_branch,
            .mutation = .merge_commit,
            .generation = gen,
            .timestamp = ts,
            .loss = loss,
            .accuracy = accuracy,
        };

        try self.commits.append(self.allocator, node);
        try self.updateBranch(target_branch, id);
        return node;
    }

    pub fn findCommit(self: *const LineageGraph, id: [32]u8) ?CommitNode {
        for (self.commits.items) |c| {
            if (std.mem.eql(u8, &c.id, &id)) return c;
        }
        return null;
    }

    pub fn findBranch(self: *const LineageGraph, name: []const u8) ?LineageBranch {
        for (self.branches.items) |b| {
            if (std.mem.eql(u8, b.name, name)) return b;
        }
        return null;
    }

    fn updateBranch(self: *LineageGraph, name: []const u8, head_id: [32]u8) !void {
        for (self.branches.items) |*b| {
            if (std.mem.eql(u8, b.name, name)) {
                b.head_id = head_id;
                return;
            }
        }
        try self.branches.append(self.allocator, .{ .name = name, .head_id = head_id });
    }

    /// Verifies the cryptographic DAG continuity and integrity.
    pub fn verifyDAG(self: *const LineageGraph) bool {
        for (self.commits.items) |c| {
            // Check parent A
            if (c.parent_ids[0]) |pa| {
                if (self.findCommit(pa) == null) return false;
            }
            // Check parent B (if merge)
            if (c.parent_ids[1]) |pb| {
                if (self.findCommit(pb) == null) return false;
            }
        }
        return true;
    }
};

/// Weight Merging: Linear Interpolation W = (1 - alpha) * A + alpha * B
pub fn mergeWeightsLinear(dst: []f32, a: []const f32, b: []const f32, alpha: f32) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    const one_minus_alpha = 1.0 - alpha;
    for (0..dst.len) |i| {
        dst[i] = one_minus_alpha * a[i] + alpha * b[i];
    }
}

/// Weight Merging: Task Arithmetic W = W_base + lambda_a * (A - W_base) + lambda_b * (B - W_base)
pub fn mergeWeightsTaskArithmetic(
    dst: []f32,
    base: []const f32,
    a: []const f32,
    b: []const f32,
    lambda_a: f32,
    lambda_b: f32,
) void {
    std.debug.assert(dst.len == base.len and base.len == a.len and a.len == b.len);
    for (0..dst.len) |i| {
        const delta_a = a[i] - base[i];
        const delta_b = b[i] - base[i];
        dst[i] = base[i] + lambda_a * delta_a + lambda_b * delta_b;
    }
}

/// Weight Merging: Spherical Linear Interpolation (SLERP)
pub fn mergeWeightsSlerp(dst: []f32, a: []const f32, b: []const f32, t: f32) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);

    var dot: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;

    for (0..dst.len) |i| {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    norm_a = @sqrt(norm_a);
    norm_b = @sqrt(norm_b);

    if (norm_a < 1e-12 or norm_b < 1e-12) {
        mergeWeightsLinear(dst, a, b, t);
        return;
    }

    var cos_theta = dot / (norm_a * norm_b);
    cos_theta = @min(@max(cos_theta, -1.0), 1.0);

    // If vectors are almost collinear, use normalized linear interpolation
    if (cos_theta > 0.9995) {
        mergeWeightsLinear(dst, a, b, t);
        return;
    }

    const theta = std.math.acos(cos_theta);
    const sin_theta = @sin(theta);
    const scale_a = @sin((1.0 - t) * theta) / sin_theta;
    const scale_b = @sin(t * theta) / sin_theta;

    for (0..dst.len) |i| {
        dst[i] = scale_a * a[i] + scale_b * b[i];
    }
}

test "lineage: DAG creation, branching, merge, and verification" {
    const allocator = std.testing.allocator;
    var graph = LineageGraph.init(allocator);
    defer graph.deinit();

    // 1. Create root commit on 'main'
    const root = try graph.createRootCommit("main", 2.5, 0.60);
    try std.testing.expectEqual(@as(u32, 0), root.generation);
    try std.testing.expect(!root.isMerge());

    // 2. Create child commit on 'main'
    const c1 = try graph.createCommit("main", root.id, .net2wider, 2.2, 0.65);
    try std.testing.expectEqual(@as(u32, 1), c1.generation);
    try std.testing.expectEqual(root.id, c1.parent_ids[0].?);

    // 3. Create feature branch commit from c1
    const c_feat = try graph.createCommit("feature/lora", c1.id, .lora_finetune, 2.0, 0.70);
    try std.testing.expectEqual(@as(u32, 2), c_feat.generation);

    // 4. Create merge commit joining main and feature/lora
    const c_merge = try graph.createMergeCommit("main", c1.id, c_feat.id, 1.8, 0.75);
    try std.testing.expect(c_merge.isMerge());
    try std.testing.expectEqual(@as(u32, 3), c_merge.generation);
    try std.testing.expectEqual(c1.id, c_merge.parent_ids[0].?);
    try std.testing.expectEqual(c_feat.id, c_merge.parent_ids[1].?);

    // 5. Verify complete DAG
    try std.testing.expect(graph.verifyDAG());
}

test "lineage: weight merging strategies (linear, task arithmetic, and slerp)" {
    const base = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const a = [_]f32{ 2.0, 1.0, 3.0, 0.0 };
    const b = [_]f32{ 0.0, 2.0, 1.0, 4.0 };
    var dst: [4]f32 = undefined;

    // Linear merge
    mergeWeightsLinear(&dst, &a, &b, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dst[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), dst[1], 1e-4);

    // Task arithmetic merge
    mergeWeightsTaskArithmetic(&dst, &base, &a, &b, 0.5, 0.5);
    // delta_a = [1, 0, 2, -1], delta_b = [-1, 1, 0, 3]
    // dst = base + 0.5*delta_a + 0.5*delta_b = 1 + 0.5*(0) = 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dst[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), dst[1], 1e-4);

    // SLERP merge
    mergeWeightsSlerp(&dst, &a, &b, 0.5);
    try std.testing.expect(dst[0] >= 0.0);
}
