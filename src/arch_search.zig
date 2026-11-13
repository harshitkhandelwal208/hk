//! Constrained Neural Architecture Search & Hardware-Aware Model Design (Plan §113, §114, §151).
//!
//! Provides hardware-constrained multi-objective search over transformer architectures:
//! 1. Constrained search space: hidden_dim, n_layers, ffn_dim, n_heads, n_kv_heads, expert_count.
//! 2. Hardware constraints: VRAM ceiling (e.g. 6 GB), memory bandwidth (GB/s), compute (TFLOPS).
//! 3. Performance estimators: decode latency (memory bandwidth bound), prefill throughput (compute bound),
//!    parameter count, and peak memory footprint.
//! 4. Multi-objective Pareto Frontier analysis: computes non-dominated architectures balancing
//!    capacity against latency and memory footprint.
//! 5. Hardware-Aware queries (§114): optimal architecture under VRAM limit, optimal quantization format,
//!    and memory bandwidth saturation analysis.

const std = @import("std");
const config_mod = @import("engine/config.zig");
const Config = config_mod.Config;
const Arch = config_mod.Arch;
const planner = @import("engine/planner.zig");
const format = @import("format.zig");
const StorageType = format.StorageType;

pub const HardwareConstraints = struct {
    max_vram_bytes: usize, // e.g. 6 * 1024 * 1024 * 1024 for 6 GB
    memory_bandwidth_gbps: f32 = 288.0, // GB/s (e.g. typical RTX 3050 / mid-tier GPU)
    compute_tflops: f32 = 10.0, // FP16 TFLOPS
    target_context_tokens: usize = 2048,
};

pub const ArchitectureCandidate = struct {
    dim: usize,
    n_layers: usize,
    ffn_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    expert_count: usize = 1,
    experts_per_tok: usize = 1,
    storage: StorageType = .f16,

    pub fn toConfig(self: ArchitectureCandidate, vocab_size: usize) Config {
        return Config{
            .arch = if (self.expert_count > 1) .moe else .llama,
            .n_layers = self.n_layers,
            .dim = self.dim,
            .ffn_dim = self.ffn_dim,
            .n_heads = self.n_heads,
            .n_kv_heads = self.n_kv_heads,
            .head_dim = self.head_dim,
            .vocab = vocab_size,
            .ctx_train = 4096,
            .rms_eps = 1e-5,
            .rope_base = 10000.0,
            .rope_dim = self.head_dim,
            .rope_style = .norm,
            .rope_scaling = .{},
            .ffn_act = .silu,
            .qk_norm = false,
            .qkv_bias = false,
            .expert_count = if (self.expert_count > 1) self.expert_count else null,
            .experts_per_tok = if (self.expert_count > 1) self.experts_per_tok else null,
        };
    }

    /// Exact parameter count calculation based on architecture specification
    pub fn parameterCount(self: ArchitectureCandidate, vocab_size: usize) usize {
        const q_weights = self.dim * (self.n_heads * self.head_dim);
        const k_weights = self.dim * (self.n_kv_heads * self.head_dim);
        const v_weights = self.dim * (self.n_kv_heads * self.head_dim);
        const out_weights = (self.n_heads * self.head_dim) * self.dim;
        const attn_weights = q_weights + k_weights + v_weights + out_weights;

        const ffn_per_expert = 3 * self.dim * self.ffn_dim; // gate, up, down
        const ffn_total = self.expert_count * ffn_per_expert;

        const per_layer = attn_weights + ffn_total + 2 * self.dim; // norms
        const body_weights = self.n_layers * per_layer;
        const embed_weights = 2 * vocab_size * self.dim; // token_embd + output

        return body_weights + embed_weights;
    }
};

pub const EvaluationResult = struct {
    candidate: ArchitectureCandidate,
    parameter_count: usize,
    peak_vram_bytes: usize,
    fits_budget: bool,
    decode_tok_per_sec: f32,
    prefill_tok_per_sec: f32,
    score: f32, // Composite efficiency score: capacity * decode_speed / memory
};

/// Evaluates a candidate architecture against hardware constraints.
pub fn evaluateCandidate(
    cand: ArchitectureCandidate,
    constraints: HardwareConstraints,
    vocab_size: usize,
) EvaluationResult {
    const params = cand.parameterCount(vocab_size);

    // Bytes per parameter based on storage format
    const bytes_per_param: f32 = switch (cand.storage) {
        .f32 => 4.0,
        .f16, .bf16 => 2.0,
        .q8_0 => 1.0625, // 34 bytes / 32 elems
        .q4_0 => 0.5625, // 18 bytes / 32 elems
        .q4_k => 0.5625,
        else => 1.0,
    };

    const weights_bytes: usize = @intFromFloat(@as(f32, @floatFromInt(params)) * bytes_per_param);
    // KV Cache: 2 (K, V) * n_layers * n_kv_heads * head_dim * sizeof(f16) * context_len
    const kv_cache_bytes = 2 * cand.n_layers * cand.n_kv_heads * cand.head_dim * 2 * constraints.target_context_tokens;
    // Activation buffer estimate (prompt batching scratch)
    const activation_scratch = cand.dim * 4096 * 4;

    const peak_vram = weights_bytes + kv_cache_bytes + activation_scratch;
    const fits = peak_vram <= constraints.max_vram_bytes;

    // Decode throughput: memory bandwidth bound.
    // For MoE, active weights use experts_per_tok instead of all experts.
    const active_params = if (cand.expert_count > 1)
        params - (cand.expert_count - cand.experts_per_tok) * (3 * cand.dim * cand.ffn_dim * cand.n_layers)
    else
        params;
    const active_bytes: f32 = @as(f32, @floatFromInt(active_params)) * bytes_per_param + @as(f32, @floatFromInt(kv_cache_bytes));
    const decode_tok_per_sec = if (active_bytes > 0)
        (constraints.memory_bandwidth_gbps * 1e9) / active_bytes
    else
        0.0;

    // Prefill throughput: compute bound.
    // 2 * active_params FLOPs per token
    const flops_per_token: f32 = 2.0 * @as(f32, @floatFromInt(active_params));
    const prefill_tok_per_sec = if (flops_per_token > 0)
        (constraints.compute_tflops * 1e12) / flops_per_token
    else
        0.0;

    // Quality/Efficiency composite score: parameter count * log(decode_speed) / memory
    const score = if (fits and peak_vram > 0)
        (@as(f32, @floatFromInt(params)) / 1e6) * @log(decode_tok_per_sec + 1.0) / (@as(f32, @floatFromInt(peak_vram)) / 1e9)
    else
        0.0;

    return .{
        .candidate = cand,
        .parameter_count = params,
        .peak_vram_bytes = peak_vram,
        .fits_budget = fits,
        .decode_tok_per_sec = decode_tok_per_sec,
        .prefill_tok_per_sec = prefill_tok_per_sec,
        .score = score,
    };
}

/// Identifies whether candidate A dominates candidate B in Pareto terms:
/// A dominates B if A is no worse than B in all objectives (params, decode_speed, memory)
/// and strictly better in at least one.
pub fn paretoDominates(a: EvaluationResult, b: EvaluationResult) bool {
    if (!a.fits_budget and b.fits_budget) return false;
    if (a.fits_budget and !b.fits_budget) return true;

    const no_worse_params = a.parameter_count >= b.parameter_count;
    const no_worse_speed = a.decode_tok_per_sec >= b.decode_tok_per_sec;
    const no_worse_mem = a.peak_vram_bytes <= b.peak_vram_bytes;

    const strictly_better = (a.parameter_count > b.parameter_count) or
        (a.decode_tok_per_sec > b.decode_tok_per_sec) or
        (a.peak_vram_bytes < b.peak_vram_bytes);

    return no_worse_params and no_worse_speed and no_worse_mem and strictly_better;
}

/// Filters a slice of evaluated candidates to only the non-dominated Pareto frontier.
pub fn filterParetoFrontier(
    candidates: []const EvaluationResult,
    out_frontier: []EvaluationResult,
) usize {
    var count: usize = 0;
    for (candidates) |cand| {
        if (!cand.fits_budget) continue;
        var dominated = false;
        for (candidates) |other| {
            if (paretoDominates(other, cand)) {
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            if (count < out_frontier.len) {
                out_frontier[count] = cand;
                count += 1;
            }
        }
    }
    return count;
}

/// Hardware-Aware Design Query (§114): What architecture gives the best quality under VRAM budget?
pub fn findBestArchitectureUnderVram(
    candidates: []const ArchitectureCandidate,
    constraints: HardwareConstraints,
    vocab_size: usize,
) ?EvaluationResult {
    var best: ?EvaluationResult = null;
    for (candidates) |c| {
        const res = evaluateCandidate(c, constraints, vocab_size);
        if (res.fits_budget) {
            if (best == null or res.score > best.?.score) {
                best = res;
            }
        }
    }
    return best;
}

/// Hardware-Aware Design Query (§114): What quantization gives the best score on this target?
pub fn findOptimalQuantizationForTarget(
    base: ArchitectureCandidate,
    formats: []const StorageType,
    constraints: HardwareConstraints,
    vocab_size: usize,
) ?StorageType {
    var best_storage: ?StorageType = null;
    var best_score: f32 = -1.0;

    for (formats) |st| {
        var cand = base;
        cand.storage = st;
        const res = evaluateCandidate(cand, constraints, vocab_size);
        if (res.fits_budget and res.score > best_score) {
            best_score = res.score;
            best_storage = st;
        }
    }
    return best_storage;
}

/// Hardware-Aware Design Query (§114): What FFN width saturates memory bandwidth?
pub fn findSaturatingFfnWidth(
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    constraints: HardwareConstraints,
    target_tok_sec: f32,
    vocab_size: usize,
) usize {
    var low: usize = dim;
    var high: usize = dim * 8;
    var best_ffn: usize = dim * 4;

    while (low <= high) {
        const mid = (low + high) / 2;
        const cand = ArchitectureCandidate{
            .dim = dim,
            .n_layers = n_layers,
            .ffn_dim = mid,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .storage = .f16,
        };
        const res = evaluateCandidate(cand, constraints, vocab_size);
        if (res.fits_budget and res.decode_tok_per_sec >= target_tok_sec) {
            best_ffn = mid;
            low = mid + 64; // try wider FFN
        } else {
            if (mid < 64) break;
            high = mid - 64;
        }
    }
    return best_ffn;
}

test "architecture parameter count and VRAM evaluation" {
    const cand = ArchitectureCandidate{
        .dim = 512,
        .n_layers = 8,
        .ffn_dim = 1024,
        .n_heads = 8,
        .n_kv_heads = 4,
        .head_dim = 64,
        .storage = .f16,
    };
    const constraints = HardwareConstraints{
        .max_vram_bytes = 4 * 1024 * 1024 * 1024, // 4 GB
        .memory_bandwidth_gbps = 200.0,
        .compute_tflops = 10.0,
    };

    const res = evaluateCandidate(cand, constraints, 32000);
    try std.testing.expect(res.parameter_count > 0);
    try std.testing.expect(res.peak_vram_bytes > 0);
    try std.testing.expect(res.fits_budget);
    try std.testing.expect(res.decode_tok_per_sec > 0.0);
    try std.testing.expect(res.prefill_tok_per_sec > 0.0);
    try std.testing.expect(res.score > 0.0);
}

test "Pareto dominance and frontier extraction" {
    // Candidate A: Small & Fast
    const a = EvaluationResult{
        .candidate = .{ .dim = 256, .n_layers = 4, .ffn_dim = 512, .n_heads = 4, .n_kv_heads = 2, .head_dim = 64 },
        .parameter_count = 10_000_000,
        .peak_vram_bytes = 500_000_000,
        .fits_budget = true,
        .decode_tok_per_sec = 150.0,
        .prefill_tok_per_sec = 500.0,
        .score = 10.0,
    };

    // Candidate B: Large & High Capacity
    const b = EvaluationResult{
        .candidate = .{ .dim = 1024, .n_layers = 16, .ffn_dim = 2048, .n_heads = 16, .n_kv_heads = 4, .head_dim = 64 },
        .parameter_count = 200_000_000,
        .peak_vram_bytes = 2_000_000_000,
        .fits_budget = true,
        .decode_tok_per_sec = 40.0,
        .prefill_tok_per_sec = 100.0,
        .score = 25.0,
    };

    // Neither dominates the other (A is faster and uses less RAM, B has higher parameter capacity)
    try std.testing.expect(!paretoDominates(a, b));
    try std.testing.expect(!paretoDominates(b, a));

    // Candidate C: Dominated by A (fewer params, slower, and more RAM than A)
    const c = EvaluationResult{
        .candidate = a.candidate,
        .parameter_count = 8_000_000,
        .peak_vram_bytes = 600_000_000,
        .fits_budget = true,
        .decode_tok_per_sec = 120.0,
        .prefill_tok_per_sec = 400.0,
        .score = 6.0,
    };
    try std.testing.expect(paretoDominates(a, c));

    const set = [_]EvaluationResult{ a, b, c };
    var frontier: [3]EvaluationResult = undefined;
    const frontier_len = filterParetoFrontier(&set, &frontier);

    // Frontier must contain exactly A and B (C is dominated)
    try std.testing.expectEqual(@as(usize, 2), frontier_len);
}

test "Hardware-Aware Query: best architecture under 6GB VRAM (Plan §114)" {
    const candidates = [_]ArchitectureCandidate{
        // 1B class
        .{ .dim = 1024, .n_layers = 12, .ffn_dim = 2048, .n_heads = 16, .n_kv_heads = 4, .head_dim = 64, .storage = .f16 },
        // 3B class
        .{ .dim = 2048, .n_layers = 20, .ffn_dim = 4096, .n_heads = 32, .n_kv_heads = 8, .head_dim = 64, .storage = .f16 },
        // 7B class (won't fit in 6GB in FP16)
        .{ .dim = 4096, .n_layers = 32, .ffn_dim = 11008, .n_heads = 32, .n_kv_heads = 8, .head_dim = 128, .storage = .f16 },
    };

    const constraints = HardwareConstraints{
        .max_vram_bytes = 6 * 1024 * 1024 * 1024, // 6 GB
        .memory_bandwidth_gbps = 288.0,
        .compute_tflops = 15.0,
    };

    const best = findBestArchitectureUnderVram(&candidates, constraints, 32000).?;
    try std.testing.expect(best.fits_budget);
    try std.testing.expect(best.peak_vram_bytes <= constraints.max_vram_bytes);
    // Best under 6 GB should be the 3B or 1B model, definitely not the 7B model
    try std.testing.expect(best.candidate.dim < 4096);
}

test "Hardware-Aware Query: optimal quantization format (Plan §114)" {
    // 7B model candidate
    const base = ArchitectureCandidate{
        .dim = 4096,
        .n_layers = 32,
        .ffn_dim = 11008,
        .n_heads = 32,
        .n_kv_heads = 8,
        .head_dim = 128,
    };
    const formats = [_]StorageType{ .f16, .q8_0, .q4_0 };

    // Under tight 6GB VRAM, f16 exceeds budget (~14GB+), but Q4_0 (~4GB) fits!
    const constraints = HardwareConstraints{
        .max_vram_bytes = 6 * 1024 * 1024 * 1024, // 6 GB
        .memory_bandwidth_gbps = 288.0,
        .compute_tflops = 15.0,
    };

    const optimal = findOptimalQuantizationForTarget(base, &formats, constraints, 32000).?;
    try std.testing.expectEqual(StorageType.q4_0, optimal);
}

test "Hardware-Aware Query: saturating FFN width (Plan §114)" {
    const constraints = HardwareConstraints{
        .max_vram_bytes = 16 * 1024 * 1024 * 1024,
        .memory_bandwidth_gbps = 300.0,
        .compute_tflops = 20.0,
    };

    const ffn = findSaturatingFfnWidth(512, 8, 8, 2, 64, constraints, 50.0, 32000);
    try std.testing.expect(ffn >= 512);
    try std.testing.expect(ffn <= 512 * 8);
}
