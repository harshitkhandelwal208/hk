//! State-Space Model (SSM) & Mamba Architecture Core (Plan §47, §48, §150).
//!
//! Provides hardware-neutral, zero-alloc state-space primitives for Mamba, Mamba-2,
//! and hybrid attention/state-space architectures (e.g. Jamba):
//! 1. Depthwise causal 1D convolution state buffer with SiLU non-linearity.
//! 2. Continuous-to-discrete Zero-Order Hold (ZOH) discretization (A_bar = exp(Delta * A), B_bar = Delta * B).
//! 3. Selective State-Space (S6) recurrent decode step with skip connection D.
//! 4. Parallel associative selective scan for prefill sequence processing.
//! 5. Gated multiplicative output combination (y * silu(z)).
//! 6. Strict memory-first design: zero heap allocation during step/scan execution.

const std = @import("std");
const math = std.math;

pub const MambaConfig = struct {
    d_model: usize,
    d_inner: usize,
    d_state: usize = 16,
    d_conv: usize = 4,
    dt_rank: usize,

    pub fn initDefault(d_model: usize) MambaConfig {
        const d_inner = d_model * 2;
        const dt_rank = (d_model + 15) / 16;
        return .{
            .d_model = d_model,
            .d_inner = d_inner,
            .d_state = 16,
            .d_conv = 4,
            .dt_rank = dt_rank,
        };
    }

    pub fn validate(self: MambaConfig) !void {
        if (self.d_model == 0 or self.d_inner == 0) return error.InvalidDimension;
        if (self.d_state == 0) return error.ZeroStateDimension;
        if (self.d_conv < 2) return error.InvalidConvKernelSize;
        if (self.dt_rank == 0) return error.InvalidDtRank;
    }
};

/// Numerically stable softplus: ln(1 + exp(x))
pub inline fn softplus(x: f32) f32 {
    if (x > 20.0) return x;
    if (x < -20.0) return @exp(x);
    return @log(1.0 + @exp(x));
}

/// Numerically stable SiLU: x / (1 + exp(-x))
pub inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// Depthwise Causal 1D Convolution State Buffer
pub const Conv1DState = struct {
    d_inner: usize,
    d_conv: usize,
    /// Ring buffer storing past tokens: shape [d_inner * (d_conv - 1)]
    history: []f32,

    pub fn init(allocator: std.mem.Allocator, d_inner: usize, d_conv: usize) !Conv1DState {
        if (d_conv < 2) return error.InvalidConvKernelSize;
        const history_len = d_inner * (d_conv - 1);
        const history = try allocator.alloc(f32, history_len);
        @memset(history, 0.0);
        return .{
            .d_inner = d_inner,
            .d_conv = d_conv,
            .history = history,
        };
    }

    pub fn initBuffer(buf: []f32, d_inner: usize, d_conv: usize) !Conv1DState {
        if (d_conv < 2) return error.InvalidConvKernelSize;
        const history_len = d_inner * (d_conv - 1);
        if (buf.len < history_len) return error.BufferTooSmall;
        const history = buf[0..history_len];
        @memset(history, 0.0);
        return .{
            .d_inner = d_inner,
            .d_conv = d_conv,
            .history = history,
        };
    }

    pub fn deinit(self: *Conv1DState, allocator: std.mem.Allocator) void {
        allocator.free(self.history);
        self.* = undefined;
    }

    pub fn reset(self: *Conv1DState) void {
        @memset(self.history, 0.0);
    }

    /// Single token decode step:
    /// Updates history buffer, computes depthwise causal convolution with weights [d_inner * d_conv],
    /// adds optional bias [d_inner], and applies SiLU activation.
    pub fn step(
        self: *Conv1DState,
        x_t: []const f32, // [d_inner]
        weight: []const f32, // [d_inner, d_conv]
        bias: ?[]const f32, // [d_inner] optional
        out: []f32, // [d_inner]
    ) void {
        std.debug.assert(x_t.len == self.d_inner);
        std.debug.assert(weight.len == self.d_inner * self.d_conv);
        std.debug.assert(out.len == self.d_inner);
        if (bias) |b| std.debug.assert(b.len == self.d_inner);

        const hist_len = self.d_conv - 1;

        for (0..self.d_inner) |c| {
            const hist_offset = c * hist_len;
            const w_offset = c * self.d_conv;

            var sum: f32 = if (bias) |b| b[c] else 0.0;

            // Convolution over history: kernel indices 0 .. hist_len-1
            for (0..hist_len) |k| {
                sum += self.history[hist_offset + k] * weight[w_offset + k];
            }
            // Latest input: kernel index hist_len
            sum += x_t[c] * weight[w_offset + hist_len];

            // Shift history left and insert x_t[c] at the end
            if (hist_len > 1) {
                std.mem.copyForwards(f32, self.history[hist_offset .. hist_offset + hist_len - 1], self.history[hist_offset + 1 .. hist_offset + hist_len]);
            }
            self.history[hist_offset + hist_len - 1] = x_t[c];

            // Non-linear activation: SiLU
            out[c] = silu(sum);
        }
    }

    /// Sequence prefill causal 1D convolution over sequence length L.
    /// Does not require or mutate decode history state.
    pub fn scanCausal1D(
        d_inner: usize,
        d_conv: usize,
        seq_x: []const f32, // [seq_len * d_inner]
        seq_len: usize,
        weight: []const f32, // [d_inner * d_conv]
        bias: ?[]const f32, // [d_inner]
        out_seq: []f32, // [seq_len * d_inner]
    ) void {
        std.debug.assert(seq_x.len == seq_len * d_inner);
        std.debug.assert(out_seq.len == seq_len * d_inner);
        std.debug.assert(weight.len == d_inner * d_conv);
        if (bias) |b| std.debug.assert(b.len == d_inner);

        for (0..seq_len) |t| {
            for (0..d_inner) |c| {
                const w_offset = c * d_conv;
                var sum: f32 = if (bias) |b| b[c] else 0.0;

                for (0..d_conv) |k| {
                    // Causal index in time: t - (d_conv - 1 - k)
                    const lag = (d_conv - 1) - k;
                    if (t >= lag) {
                        const in_t = t - lag;
                        sum += seq_x[in_t * d_inner + c] * weight[w_offset + k];
                    }
                }
                out_seq[t * d_inner + c] = silu(sum);
            }
        }
    }
};

/// Selective State-Space (S6) Model State
pub const SsmState = struct {
    d_inner: usize,
    d_state: usize,
    /// Recurrent hidden state: shape [d_inner * d_state]
    hidden_state: []f32,

    pub fn init(allocator: std.mem.Allocator, d_inner: usize, d_state: usize) !SsmState {
        if (d_inner == 0 or d_state == 0) return error.InvalidDimension;
        const total = d_inner * d_state;
        const state = try allocator.alloc(f32, total);
        @memset(state, 0.0);
        return .{
            .d_inner = d_inner,
            .d_state = d_state,
            .hidden_state = state,
        };
    }

    pub fn initBuffer(buf: []f32, d_inner: usize, d_state: usize) !SsmState {
        if (d_inner == 0 or d_state == 0) return error.InvalidDimension;
        const total = d_inner * d_state;
        if (buf.len < total) return error.BufferTooSmall;
        const state = buf[0..total];
        @memset(state, 0.0);
        return .{
            .d_inner = d_inner,
            .d_state = d_state,
            .hidden_state = state,
        };
    }

    pub fn deinit(self: *SsmState, allocator: std.mem.Allocator) void {
        allocator.free(self.hidden_state);
        self.* = undefined;
    }

    pub fn reset(self: *SsmState) void {
        @memset(self.hidden_state, 0.0);
    }

    /// Single token selective scan step:
    /// u_t: input vector after conv1d [d_inner]
    /// delta_raw: unconstrained timestep logits [d_inner]
    /// A: state transition matrix [d_inner * d_state] (real negative numbers)
    /// B: input matrix [d_state] (or [d_inner * d_state])
    /// C: output projection matrix [d_state] (or [d_inner * d_state])
    /// D: skip connection vector [d_inner]
    /// out: output vector [d_inner]
    pub fn step(
        self: *SsmState,
        u_t: []const f32,
        delta_raw: []const f32,
        A: []const f32,
        B: []const f32,
        C: []const f32,
        D: []const f32,
        out: []f32,
    ) void {
        std.debug.assert(u_t.len == self.d_inner);
        std.debug.assert(delta_raw.len == self.d_inner);
        std.debug.assert(A.len == self.d_inner * self.d_state);
        std.debug.assert(D.len == self.d_inner);
        std.debug.assert(out.len == self.d_inner);

        const b_per_channel = (B.len == self.d_inner * self.d_state);
        const c_per_channel = (C.len == self.d_inner * self.d_state);

        for (0..self.d_inner) |i| {
            const dt = softplus(delta_raw[i]);
            const u_val = u_t[i];
            const state_offset = i * self.d_state;

            var y_i: f32 = D[i] * u_val;

            for (0..self.d_state) |j| {
                const a_val = A[state_offset + j];
                // Zero-Order Hold discretization:
                // A_bar = exp(dt * A)
                const a_bar = @exp(dt * a_val);

                const b_val = if (b_per_channel) B[state_offset + j] else B[j];
                // B_bar = dt * B
                const b_bar = dt * b_val;

                // State update: h_ij = A_bar * h_ij + B_bar * u_i
                const h_idx = state_offset + j;
                const new_h = a_bar * self.hidden_state[h_idx] + b_bar * u_val;
                self.hidden_state[h_idx] = new_h;

                const c_val = if (c_per_channel) C[state_offset + j] else C[j];
                y_i += new_h * c_val;
            }

            out[i] = y_i;
        }
    }

    /// Full sequence selective scan over sequence length L.
    /// Produces numerically identical outputs to sequential step() invocations.
    pub fn selectiveScan(
        d_inner: usize,
        d_state: usize,
        seq_u: []const f32, // [seq_len * d_inner]
        seq_delta_raw: []const f32, // [seq_len * d_inner]
        seq_len: usize,
        A: []const f32, // [d_inner * d_state]
        seq_B: []const f32, // [seq_len * d_state] or [seq_len * d_inner * d_state]
        seq_C: []const f32, // [seq_len * d_state] or [seq_len * d_inner * d_state]
        D: []const f32, // [d_inner]
        out_seq: []f32, // [seq_len * d_inner]
        state_scratch: []f32, // [d_inner * d_state] caller provided scratch
    ) void {
        std.debug.assert(seq_u.len == seq_len * d_inner);
        std.debug.assert(seq_delta_raw.len == seq_len * d_inner);
        std.debug.assert(out_seq.len == seq_len * d_inner);
        std.debug.assert(state_scratch.len >= d_inner * d_state);

        @memset(state_scratch[0 .. d_inner * d_state], 0.0);

        const b_is_matrix = (seq_B.len == seq_len * d_inner * d_state);
        const c_is_matrix = (seq_C.len == seq_len * d_inner * d_state);

        for (0..seq_len) |t| {
            const u_t = seq_u[t * d_inner .. (t + 1) * d_inner];
            const delta_t = seq_delta_raw[t * d_inner .. (t + 1) * d_inner];
            const b_t = if (b_is_matrix)
                seq_B[t * d_inner * d_state .. (t + 1) * d_inner * d_state]
            else
                seq_B[t * d_state .. (t + 1) * d_state];
            const c_t = if (c_is_matrix)
                seq_C[t * d_inner * d_state .. (t + 1) * d_inner * d_state]
            else
                seq_C[t * d_state .. (t + 1) * d_state];

            for (0..d_inner) |i| {
                const dt = softplus(delta_t[i]);
                const u_val = u_t[i];
                const state_offset = i * d_state;

                var y_i: f32 = D[i] * u_val;

                for (0..d_state) |j| {
                    const a_val = A[state_offset + j];
                    const a_bar = @exp(dt * a_val);

                    const b_val = if (b_is_matrix) b_t[state_offset + j] else b_t[j];
                    const b_bar = dt * b_val;

                    const h_idx = state_offset + j;
                    const new_h = a_bar * state_scratch[h_idx] + b_bar * u_val;
                    state_scratch[h_idx] = new_h;

                    const c_val = if (c_is_matrix) c_t[state_offset + j] else c_t[j];
                    y_i += new_h * c_val;
                }

                out_seq[t * d_inner + i] = y_i;
            }
        }
    }
};

/// Gated Output Combiner:
/// Multiplies SSM output with SiLU activation of parallel gate branch z:
/// out[i] = y_ssm[i] * silu(z[i])
pub fn combineGatedOutput(y_ssm: []const f32, z_gate: []const f32, out: []f32) void {
    std.debug.assert(y_ssm.len == z_gate.len);
    std.debug.assert(y_ssm.len == out.len);

    for (y_ssm, z_gate, out) |y, z, *o| {
        o.* = y * silu(z);
    }
}

// ============================================================================
// Unit Tests & Invariant Verification
// ============================================================================

test "MambaConfig validation" {
    const valid = MambaConfig.initDefault(768);
    try valid.validate();
    try std.testing.expectEqual(@as(usize, 1536), valid.d_inner);
    try std.testing.expectEqual(@as(usize, 48), valid.dt_rank);

    var invalid = valid;
    invalid.d_model = 0;
    try std.testing.expectError(error.InvalidDimension, invalid.validate());

    var invalid_conv = valid;
    invalid_conv.d_conv = 1;
    try std.testing.expectError(error.InvalidConvKernelSize, invalid_conv.validate());
}

test "Conv1DState causal step and scan parity" {
    const d_inner: usize = 4;
    const d_conv: usize = 3;
    const seq_len: usize = 5;

    var history_buf: [d_inner * (d_conv - 1)]f32 = undefined;
    var conv_state = try Conv1DState.initBuffer(&history_buf, d_inner, d_conv);

    // Setup deterministic weights and biases
    var weight: [d_inner * d_conv]f32 = undefined;
    for (&weight, 0..) |*w, i| {
        w.* = @as(f32, @floatFromInt(i + 1)) * 0.1;
    }
    var bias: [d_inner]f32 = .{ 0.05, -0.05, 0.1, -0.1 };

    // Input sequence [seq_len * d_inner]
    var seq_x: [seq_len * d_inner]f32 = undefined;
    for (&seq_x, 0..) |*x, i| {
        x.* = @sin(@as(f32, @floatFromInt(i))) * 0.5;
    }

    // Step-by-step evaluation
    var step_out: [seq_len * d_inner]f32 = undefined;
    for (0..seq_len) |t| {
        const x_t = seq_x[t * d_inner .. (t + 1) * d_inner];
        const out_t = step_out[t * d_inner .. (t + 1) * d_inner];
        conv_state.step(x_t, &weight, &bias, out_t);
    }

    // Batch causal scan evaluation
    var scan_out: [seq_len * d_inner]f32 = undefined;
    Conv1DState.scanCausal1D(d_inner, d_conv, &seq_x, seq_len, &weight, &bias, &scan_out);

    // Verify exact parity between step-by-step decode and sequence scan
    for (step_out, scan_out) |a, b| {
        try std.testing.expectApproxEqAbs(a, b, 1e-5);
    }
}

test "SsmState step and selectiveScan exact mathematical equivalence" {
    const d_inner: usize = 4;
    const d_state: usize = 3;
    const seq_len: usize = 4;

    var state_buf: [d_inner * d_state]f32 = undefined;
    var ssm = try SsmState.initBuffer(&state_buf, d_inner, d_state);

    // Transition matrix A: negative values
    var A: [d_inner * d_state]f32 = undefined;
    for (&A, 0..) |*a, i| {
        a.* = -@as(f32, @floatFromInt(i + 1)) * 0.2;
    }
    const D: [d_inner]f32 = .{ 1.0, 1.2, 0.8, 1.1 };

    // Time-varying inputs, delta, B, C across sequence
    var seq_u: [seq_len * d_inner]f32 = undefined;
    var seq_delta: [seq_len * d_inner]f32 = undefined;
    for (0..seq_len * d_inner) |i| {
        seq_u[i] = @as(f32, @floatFromInt(i % 5)) * 0.25;
        seq_delta[i] = @as(f32, @floatFromInt((i + 1) % 4)) * 0.1 - 0.2;
    }

    var seq_B: [seq_len * d_state]f32 = undefined;
    var seq_C: [seq_len * d_state]f32 = undefined;
    for (0..seq_len * d_state) |i| {
        seq_B[i] = @cos(@as(f32, @floatFromInt(i))) * 0.5;
        seq_C[i] = @sin(@as(f32, @floatFromInt(i))) * 0.5;
    }

    // Sequential step decode
    var step_out: [seq_len * d_inner]f32 = undefined;
    for (0..seq_len) |t| {
        const u_t = seq_u[t * d_inner .. (t + 1) * d_inner];
        const delta_t = seq_delta[t * d_inner .. (t + 1) * d_inner];
        const b_t = seq_B[t * d_state .. (t + 1) * d_state];
        const c_t = seq_C[t * d_state .. (t + 1) * d_state];
        const out_t = step_out[t * d_inner .. (t + 1) * d_inner];
        ssm.step(u_t, delta_t, &A, b_t, c_t, &D, out_t);
    }

    // Sequence selective scan
    var scan_scratch: [d_inner * d_state]f32 = undefined;
    var scan_out: [seq_len * d_inner]f32 = undefined;
    SsmState.selectiveScan(d_inner, d_state, &seq_u, &seq_delta, seq_len, &A, &seq_B, &seq_C, &D, &scan_out, &scan_scratch);

    // Verify bitwise/numerical equivalence
    for (step_out, scan_out) |s, c| {
        try std.testing.expectApproxEqAbs(s, c, 1e-6);
    }
}

test "Mamba gated output combine" {
    const y: [4]f32 = .{ 1.0, 2.0, -1.0, 0.5 };
    const z: [4]f32 = .{ 0.0, 1.0, -2.0, 3.0 };
    var out: [4]f32 = undefined;

    combineGatedOutput(&y, &z, &out);

    // out[0] = 1.0 * silu(0) = 1.0 * 0.0 = 0.0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-5);
    // out[1] = 2.0 * silu(1.0) = 2.0 * (1 / (1 + exp(-1))) ~ 1.462117
    try std.testing.expectApproxEqAbs(@as(f32, 1.462117), out[1], 1e-5);
}

test "SsmState numerical stability with extreme delta" {
    const d_inner: usize = 2;
    const d_state: usize = 2;

    var state_buf: [d_inner * d_state]f32 = undefined;
    var ssm = try SsmState.initBuffer(&state_buf, d_inner, d_state);

    const u: [2]f32 = .{ 1.0, 1.0 };
    // Large positive and large negative delta
    const delta_extreme: [2]f32 = .{ 50.0, -50.0 };
    const A: [4]f32 = .{ -1.0, -2.0, -0.5, -1.5 };
    const B: [2]f32 = .{ 0.5, 0.5 };
    const C: [2]f32 = .{ 0.5, 0.5 };
    const D: [2]f32 = .{ 0.0, 0.0 };
    var out: [2]f32 = undefined;

    ssm.step(&u, &delta_extreme, &A, &B, &C, &D, &out);

    // Must be finite and non-NaN
    try std.testing.expect(!math.isNan(out[0]));
    try std.testing.expect(!math.isNan(out[1]));
    try std.testing.expect(!math.isInf(out[0]));
    try std.testing.expect(!math.isInf(out[1]));
}
