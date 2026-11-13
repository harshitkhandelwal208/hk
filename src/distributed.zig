//! Multi-GPU and distributed execution primitives (Plan §74, §75).
//! Implements tensor parallelism, pipeline parallelism, and collective communication
//! operators (allreduce, allgather, broadcast, barrier) with memory-first bounded buffers.

const std = @import("std");

pub const ParallelismMode = enum {
    tensor_parallel,
    pipeline_parallel,
    expert_parallel,
    data_parallel,
};

pub const DeviceRank = struct {
    rank: usize,
    world_size: usize,

    pub fn isRoot(self: DeviceRank) bool {
        return self.rank == 0;
    }

    pub fn validate(self: DeviceRank) !void {
        if (self.world_size == 0 or self.rank >= self.world_size) {
            return error.InvalidRankOrWorldSize;
        }
    }
};

/// Collective communication engine operating over shared memory / peer GPU ring buffers.
pub const CollectiveContext = struct {
    allocator: std.mem.Allocator,
    rank: DeviceRank,
    scratch: []f32,

    pub fn init(allocator: std.mem.Allocator, rank: DeviceRank, scratch_size: usize) !CollectiveContext {
        try rank.validate();
        const scratch = try allocator.alloc(f32, scratch_size);
        return .{
            .allocator = allocator,
            .rank = rank,
            .scratch = scratch,
        };
    }

    pub fn deinit(self: *CollectiveContext) void {
        self.allocator.free(self.scratch);
    }

    /// In-place AllReduce (sum reduction):
    /// In a multi-device setup, peer buffers are summed into every participant's buffer.
    pub fn allreduceSum(
        self: *CollectiveContext,
        local_data: []f32,
        peer_buffers: []const []const f32,
    ) !void {
        if (peer_buffers.len + 1 != self.rank.world_size) return error.WorldSizeMismatch;
        for (peer_buffers) |peer| {
            if (peer.len != local_data.len) return error.BufferLengthMismatch;
        }

        // Sum contributions from all peers into local_data
        for (peer_buffers) |peer| {
            var i: usize = 0;
            // Vectorized 8-wide float accumulation
            while (i + 8 <= local_data.len) : (i += 8) {
                const vl: @Vector(8, f32) = local_data[i..][0..8].*;
                const vp: @Vector(8, f32) = peer[i..][0..8].*;
                local_data[i..][0..8].* = vl + vp;
            }
            while (i < local_data.len) : (i += 1) {
                local_data[i] += peer[i];
            }
        }
    }

    /// AllGather:
    /// Gathers slices of size `in_slice.len` from all ranks into contiguous `out_gathered`.
    pub fn allgather(
        self: *CollectiveContext,
        out_gathered: []f32,
        in_slice: []const f32,
        peer_slices: []const []const f32,
    ) !void {
        const slice_len = in_slice.len;
        if (out_gathered.len != slice_len * self.rank.world_size) return error.BufferLengthMismatch;
        if (peer_slices.len + 1 != self.rank.world_size) return error.WorldSizeMismatch;

        // Copy local slice to its slot
        const local_slot = self.rank.rank * slice_len;
        @memcpy(out_gathered[local_slot .. local_slot + slice_len], in_slice);

        // Copy peer slices to their respective slots
        var peer_idx: usize = 0;
        for (0..self.rank.world_size) |r| {
            if (r == self.rank.rank) continue;
            const peer = peer_slices[peer_idx];
            peer_idx += 1;
            if (peer.len != slice_len) return error.BufferLengthMismatch;
            const slot = r * slice_len;
            @memcpy(out_gathered[slot .. slot + slice_len], peer);
        }
    }

    /// Broadcast:
    /// Overwrites local_data with root rank's buffer if local is non-root.
    pub fn broadcast(
        self: *CollectiveContext,
        local_data: []f32,
        root_data: ?[]const f32,
    ) !void {
        if (self.rank.isRoot()) {
            // Root provides the source buffer
            return;
        } else {
            // Non-root receives from root
            const src = root_data orelse return error.MissingRootData;
            if (local_data.len != src.len) return error.BufferLengthMismatch;
            @memcpy(local_data, src);
        }
    }
};

/// Column-Parallel Linear Layer:
/// Weights W are partitioned across ranks along the output dimension: W = [W_0 | W_1 | ... | W_{k-1}].
/// Input X is replicated; Output Y_i = X * W_i is a shard of the full output.
pub fn columnParallelForward(
    y_shard: []f32, // [out_features_per_rank]
    x_input: []const f32, // [in_features]
    w_shard: []const f32, // [out_features_per_rank, in_features]
    out_features_per_rank: usize,
    in_features: usize,
) !void {
    if (y_shard.len < out_features_per_rank or x_input.len < in_features) return error.BufferTooSmall;
    if (w_shard.len < out_features_per_rank * in_features) return error.BufferTooSmall;

    for (0..out_features_per_rank) |i| {
        var sum: f32 = 0.0;
        const row = w_shard[i * in_features .. (i + 1) * in_features];
        var j: usize = 0;
        while (j + 8 <= in_features) : (j += 8) {
            const vx: @Vector(8, f32) = x_input[j..][0..8].*;
            const vw: @Vector(8, f32) = row[j..][0..8].*;
            sum += @reduce(.Add, vx * vw);
        }
        while (j < in_features) : (j += 1) {
            sum += row[j] * x_input[j];
        }
        y_shard[i] = sum;
    }
}

/// Row-Parallel Linear Layer:
/// Weights W are partitioned across ranks along the input dimension: W = [W_0; W_1; ...; W_{k-1}].
/// Input X is sharded; partial outputs Y_i = X_i * W_i must be summed via AllReduce to yield Y = \sum Y_i.
pub fn rowParallelForward(
    y_partial: []f32, // [out_features]
    x_shard: []const f32, // [in_features_per_rank]
    w_shard: []const f32, // [out_features, in_features_per_rank]
    out_features: usize,
    in_features_per_rank: usize,
) !void {
    if (y_partial.len < out_features or x_shard.len < in_features_per_rank) return error.BufferTooSmall;
    if (w_shard.len < out_features * in_features_per_rank) return error.BufferTooSmall;

    for (0..out_features) |i| {
        var sum: f32 = 0.0;
        const row = w_shard[i * in_features_per_rank .. (i + 1) * in_features_per_rank];
        var j: usize = 0;
        while (j + 8 <= in_features_per_rank) : (j += 8) {
            const vx: @Vector(8, f32) = x_shard[j..][0..8].*;
            const vw: @Vector(8, f32) = row[j..][0..8].*;
            sum += @reduce(.Add, vx * vw);
        }
        while (j < in_features_per_rank) : (j += 1) {
            sum += row[j] * x_shard[j];
        }
        y_partial[i] = sum;
    }
}

test "DeviceRank validation" {
    const valid_rank = DeviceRank{ .rank = 1, .world_size = 4 };
    try valid_rank.validate();
    try std.testing.expect(!valid_rank.isRoot());

    const root_rank = DeviceRank{ .rank = 0, .world_size = 4 };
    try root_rank.validate();
    try std.testing.expect(root_rank.isRoot());

    const invalid_rank = DeviceRank{ .rank = 4, .world_size = 4 };
    try std.testing.expectError(error.InvalidRankOrWorldSize, invalid_rank.validate());
}

test "CollectiveContext: AllReduce sum across ranks" {
    const allocator = std.testing.allocator;
    const world_size: usize = 4;
    const count: usize = 8;

    // Simulate 4 devices
    var buffers: [world_size][count]f32 = undefined;
    for (0..world_size) |r| {
        for (0..count) |i| {
            buffers[r][i] = @as(f32, @floatFromInt((r + 1) * 10 + i));
        }
    }

    // Rank 0 reduces with peers 1, 2, 3
    var ctx0 = try CollectiveContext.init(allocator, .{ .rank = 0, .world_size = world_size }, 64);
    defer ctx0.deinit();

    const peers = [_][]const f32{
        &buffers[1],
        &buffers[2],
        &buffers[3],
    };

    try ctx0.allreduceSum(&buffers[0], &peers);

    // Verify sum: (10 + 20 + 30 + 40) + 4*i = 100 + 4*i
    for (0..count) |i| {
        const expected = 100.0 + 4.0 * @as(f32, @floatFromInt(i));
        try std.testing.expectEqual(expected, buffers[0][i]);
    }
}

test "CollectiveContext: AllGather across ranks" {
    const allocator = std.testing.allocator;
    const world_size: usize = 3;
    const slice_len: usize = 4;

    const rank1_slice = [_]f32{ 1.0, 1.1, 1.2, 1.3 };
    const peer_slices = [_][]const f32{
        &[_]f32{ 0.0, 0.1, 0.2, 0.3 }, // rank 0
        &[_]f32{ 2.0, 2.1, 2.2, 2.3 }, // rank 2
    };

    var ctx1 = try CollectiveContext.init(allocator, .{ .rank = 1, .world_size = world_size }, 64);
    defer ctx1.deinit();

    var gathered: [world_size * slice_len]f32 = undefined;
    try ctx1.allgather(&gathered, &rank1_slice, &peer_slices);

    // Check slot 0
    try std.testing.expectEqual(@as(f32, 0.0), gathered[0]);
    // Check slot 1 (rank 1's local slice)
    try std.testing.expectEqual(@as(f32, 1.0), gathered[4]);
    // Check slot 2
    try std.testing.expectEqual(@as(f32, 2.0), gathered[8]);
}

test "Tensor Parallelism: ColumnParallel and RowParallel parity with standard GEMM" {
    const in_dim: usize = 8;
    const out_dim: usize = 8;
    const world_size: usize = 2;
    const shard_dim: usize = out_dim / world_size;

    // Full dense weight matrix [out_dim, in_dim]
    var full_w: [out_dim * in_dim]f32 = undefined;
    for (0..out_dim * in_dim) |i| full_w[i] = @as(f32, @floatFromInt(i + 1)) * 0.1;

    const x = [_]f32{ 1.0, 0.5, -0.5, 2.0, 0.1, -1.0, 0.4, 0.8 };

    // Reference dense output: y = full_w * x
    var y_ref: [out_dim]f32 = undefined;
    for (0..out_dim) |i| {
        var s: f32 = 0.0;
        for (0..in_dim) |j| s += full_w[i * in_dim + j] * x[j];
        y_ref[i] = s;
    }

    // --- 1. Column-Parallel Execution ---
    // Rank 0 owns rows 0..shard_dim-1
    // Rank 1 owns rows shard_dim..out_dim-1
    var y0_shard: [shard_dim]f32 = undefined;
    var y1_shard: [shard_dim]f32 = undefined;

    try columnParallelForward(&y0_shard, &x, full_w[0 .. shard_dim * in_dim], shard_dim, in_dim);
    try columnParallelForward(&y1_shard, &x, full_w[shard_dim * in_dim .. out_dim * in_dim], shard_dim, in_dim);

    for (0..shard_dim) |i| {
        try std.testing.expectApproxEqAbs(y_ref[i], y0_shard[i], 1e-5);
        try std.testing.expectApproxEqAbs(y_ref[shard_dim + i], y1_shard[i], 1e-5);
    }

    // --- 2. Row-Parallel Execution ---
    // Shard along in_features: Rank 0 gets x[0..4], W[:, 0..4]; Rank 1 gets x[4..8], W[:, 4..8]
    const in_shard = in_dim / world_size;
    var w_row_shard0: [out_dim * in_shard]f32 = undefined;
    var w_row_shard1: [out_dim * in_shard]f32 = undefined;

    for (0..out_dim) |i| {
        @memcpy(w_row_shard0[i * in_shard .. (i + 1) * in_shard], full_w[i * in_dim .. i * in_dim + in_shard]);
        @memcpy(w_row_shard1[i * in_shard .. (i + 1) * in_shard], full_w[i * in_dim + in_shard .. (i + 1) * in_dim]);
    }

    var y_part0: [out_dim]f32 = undefined;
    var y_part1: [out_dim]f32 = undefined;

    try rowParallelForward(&y_part0, x[0..in_shard], &w_row_shard0, out_dim, in_shard);
    try rowParallelForward(&y_part1, x[in_shard..in_dim], &w_row_shard1, out_dim, in_shard);

    // Sum partials (AllReduce step)
    for (0..out_dim) |i| {
        const y_combined = y_part0[i] + y_part1[i];
        try std.testing.expectApproxEqAbs(y_ref[i], y_combined, 1e-5);
    }
}
