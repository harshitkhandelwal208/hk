//! Multi-Model Execution and Memory Planner (Plan §46-§50, §73, §100, §150).
//! Validates multi-query, grouped-query, and multi-head attention (MQA/GQA/MHA),
//! RoPE scaling invariants, vocabulary mapping, and calculates memory plans.

const std = @import("std");
const config_mod = @import("config.zig");
const Config = config_mod.Config;
const Arch = config_mod.Arch;
const RopeScaling = config_mod.RopeScaling;
const registry = @import("../models/registry.zig");
const AttentionType = registry.AttentionType;

pub const PlannerError = error{
    InvalidHeadDivision,
    InvalidAttentionType,
    InvalidRopeScalingFactor,
    InvalidRopeFrequencies,
    InvalidContextLength,
    InvalidHeadDimension,
    VocabularyUnderflow,
    TokenIndexOutOfBounds,
};

pub const AttentionPlan = struct {
    attention_type: AttentionType,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    gqa_ratio: usize, // n_heads / n_kv_heads

    pub fn kvBytesPerToken(self: AttentionPlan, dtype_bytes: usize) usize {
        // 2 for K and V
        return 2 * self.n_kv_heads * self.head_dim * dtype_bytes;
    }
};

pub const LayerMemoryPlan = struct {
    q_bytes: usize,
    k_bytes: usize,
    v_bytes: usize,
    attn_out_bytes: usize,
    ffn_gate_bytes: usize,
    ffn_up_bytes: usize,
    ffn_down_bytes: usize,
    peak_activation_bytes: usize,
};

pub const ExecutionPlan = struct {
    config: Config,
    attention: AttentionPlan,
    kv_cache_total_bytes: usize,
    layer_activation: LayerMemoryPlan,
    total_weights_bytes: usize,
    peak_memory_bytes: usize,
};

/// Validates MHA/GQA/MQA attention parameters and computes head group ratios.
pub fn validateAttention(n_heads: usize, n_kv_heads: usize, head_dim: usize) PlannerError!AttentionPlan {
    if (n_heads == 0 or n_kv_heads == 0 or head_dim == 0) return error.InvalidHeadDivision;
    if (head_dim % 2 != 0 or head_dim > 256) return error.InvalidHeadDimension;
    if (n_heads % n_kv_heads != 0) return error.InvalidHeadDivision;

    const gqa_ratio = n_heads / n_kv_heads;
    const attn_type: AttentionType = if (n_kv_heads == 1)
        .mqa
    else if (n_kv_heads == n_heads)
        .mha
    else
        .gqa;

    return AttentionPlan{
        .attention_type = attn_type,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .gqa_ratio = gqa_ratio,
    };
}

/// Validates RoPE scaling factors and frequency thresholds.
pub fn validateRopeScaling(scaling: RopeScaling, orig_ctx: usize, rope_dim: usize) PlannerError!void {
    if (scaling.kind == .none) return;

    if (!(scaling.factor >= 1.0) or !std.math.isFinite(scaling.factor)) {
        return error.InvalidRopeScalingFactor;
    }

    if (scaling.kind != .linear and (orig_ctx == 0 or rope_dim == 0)) {
        return error.InvalidContextLength;
    }

    switch (scaling.kind) {
        .llama3 => {
            if (!(scaling.low_freq_factor > 0 and scaling.high_freq_factor > scaling.low_freq_factor)) {
                return error.InvalidRopeFrequencies;
            }
        },
        .yarn => {
            if (!(scaling.beta_fast > 0 and scaling.beta_slow > 0 and scaling.attn_factor > 0)) {
                return error.InvalidRopeFrequencies;
            }
        },
        else => {},
    }
}

/// Validates vocabulary size and token indexing bounds.
pub fn validateVocabulary(vocab_size: usize, tokens: []const u32) PlannerError!void {
    if (vocab_size == 0) return error.VocabularyUnderflow;
    for (tokens) |tok| {
        if (tok >= vocab_size) return error.TokenIndexOutOfBounds;
    }
}

/// Creates a complete memory and execution plan for a model configuration.
pub fn planExecution(cfg: Config, max_tokens: usize) PlannerError!ExecutionPlan {
    const attn_plan = try validateAttention(cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    try validateRopeScaling(cfg.rope_scaling, cfg.ctx_train, cfg.rope_dim);

    // KV Cache: 2 (K, V) * n_layers * n_kv_heads * head_dim * sizeof(f16) * max_tokens
    const kv_cache_bytes = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * @sizeOf(f16) * max_tokens;

    // Layer activation calculations (for prompt token batching)
    const q_bytes = cfg.qDim() * @sizeOf(f32);
    const k_bytes = cfg.kvDim() * @sizeOf(f32);
    const v_bytes = cfg.kvDim() * @sizeOf(f32);
    const attn_out_bytes = cfg.dim * @sizeOf(f32);

    const ffn_gate_bytes = cfg.ffn_dim * @sizeOf(f32);
    const ffn_up_bytes = cfg.ffn_dim * @sizeOf(f32);
    const ffn_down_bytes = cfg.dim * @sizeOf(f32);

    const peak_activation_bytes = @max(
        q_bytes + k_bytes + v_bytes + attn_out_bytes,
        ffn_gate_bytes + ffn_up_bytes + ffn_down_bytes,
    );

    // Weight estimate (approximate for planning, accounting for routed MoE experts)
    const num_experts = cfg.expert_count orelse 1;
    const qkv_weights = (cfg.dim * cfg.qDim() + 2 * cfg.dim * cfg.kvDim()) * @sizeOf(f16);
    const attn_out_weights = (cfg.qDim() * cfg.dim) * @sizeOf(f16);
    const mlp_weights = num_experts * (2 * cfg.dim * cfg.ffn_dim + cfg.ffn_dim * cfg.dim) * @sizeOf(f16);
    const layer_weights = qkv_weights + attn_out_weights + mlp_weights;
    const total_weights_bytes = cfg.n_layers * layer_weights + (cfg.vocab * cfg.dim * @sizeOf(f16));

    const peak_memory_bytes = total_weights_bytes + kv_cache_bytes + peak_activation_bytes * 2;

    return ExecutionPlan{
        .config = cfg,
        .attention = attn_plan,
        .kv_cache_total_bytes = kv_cache_bytes,
        .layer_activation = .{
            .q_bytes = q_bytes,
            .k_bytes = k_bytes,
            .v_bytes = v_bytes,
            .attn_out_bytes = attn_out_bytes,
            .ffn_gate_bytes = ffn_gate_bytes,
            .ffn_up_bytes = ffn_up_bytes,
            .ffn_down_bytes = ffn_down_bytes,
            .peak_activation_bytes = peak_activation_bytes,
        },
        .total_weights_bytes = total_weights_bytes,
        .peak_memory_bytes = peak_memory_bytes,
    };
}

test "validateAttention: MHA, GQA, and MQA" {
    // MHA: 32 heads, 32 kv_heads
    const mha = try validateAttention(32, 32, 128);
    try std.testing.expectEqual(AttentionType.mha, mha.attention_type);
    try std.testing.expectEqual(@as(usize, 1), mha.gqa_ratio);

    // GQA (LLaMA 3 / Gemma / Mistral): 32 heads, 8 kv_heads
    const gqa = try validateAttention(32, 8, 128);
    try std.testing.expectEqual(AttentionType.gqa, gqa.attention_type);
    try std.testing.expectEqual(@as(usize, 4), gqa.gqa_ratio);
    try std.testing.expectEqual(@as(usize, 2 * 8 * 128 * 2), gqa.kvBytesPerToken(2));

    // MQA: 32 heads, 1 kv_head
    const mqa = try validateAttention(32, 1, 128);
    try std.testing.expectEqual(AttentionType.mqa, mqa.attention_type);
    try std.testing.expectEqual(@as(usize, 32), mqa.gqa_ratio);

    // Error: heads not divisible by kv_heads
    try std.testing.expectError(error.InvalidHeadDivision, validateAttention(32, 7, 128));

    // Error: odd head_dim
    try std.testing.expectError(error.InvalidHeadDimension, validateAttention(32, 8, 127));
}

test "validateRopeScaling: linear, llama3, and yarn bounds" {
    // Valid linear
    try validateRopeScaling(.{ .kind = .linear, .factor = 2.0 }, 2048, 128);

    // Invalid factor < 1.0
    try std.testing.expectError(error.InvalidRopeScalingFactor, validateRopeScaling(.{ .kind = .linear, .factor = 0.5 }, 2048, 128));

    // Valid llama3
    try validateRopeScaling(.{ .kind = .llama3, .factor = 4.0, .low_freq_factor = 1.0, .high_freq_factor = 4.0 }, 2048, 128);

    // Invalid llama3 frequencies (high <= low)
    try std.testing.expectError(error.InvalidRopeFrequencies, validateRopeScaling(.{ .kind = .llama3, .factor = 4.0, .low_freq_factor = 4.0, .high_freq_factor = 1.0 }, 2048, 128));

    // Valid yarn
    try validateRopeScaling(.{ .kind = .yarn, .factor = 4.0, .beta_fast = 32.0, .beta_slow = 1.0, .attn_factor = 1.0 }, 2048, 128);
}

test "validateVocabulary: bounds checking" {
    const valid_tokens = [_]u32{ 0, 100, 32000 };
    try validateVocabulary(32064, &valid_tokens);

    const oob_tokens = [_]u32{ 0, 32064 };
    try std.testing.expectError(error.TokenIndexOutOfBounds, validateVocabulary(32064, &oob_tokens));
}

test "planExecution: dense vs MoE memory footprints" {
    const dense_cfg = Config{
        .arch = .llama,
        .n_layers = 4,
        .dim = 256,
        .ffn_dim = 512,
        .n_heads = 8,
        .n_kv_heads = 4,
        .head_dim = 32,
        .vocab = 1000,
        .ctx_train = 2048,
        .rms_eps = 1e-5,
        .rope_base = 10000.0,
        .rope_dim = 32,
        .rope_style = .norm,
        .rope_scaling = .{},
        .ffn_act = .silu,
        .qk_norm = false,
        .qkv_bias = false,
    };

    const dense_plan = try planExecution(dense_cfg, 512);
    try std.testing.expectEqual(AttentionType.gqa, dense_plan.attention.attention_type);
    try std.testing.expect(dense_plan.total_weights_bytes > 0);
    try std.testing.expect(dense_plan.peak_memory_bytes > dense_plan.total_weights_bytes);

    var moe_cfg = dense_cfg;
    moe_cfg.arch = .moe;
    moe_cfg.expert_count = 8;
    moe_cfg.experts_per_tok = 2;

    const moe_plan = try planExecution(moe_cfg, 512);
    // MoE weights should be strictly larger due to 8 experts in MLP
    try std.testing.expect(moe_plan.total_weights_bytes > dense_plan.total_weights_bytes);
    // KV cache footprint is identical because attention dimension and layers are the same
    try std.testing.expectEqual(dense_plan.kv_cache_total_bytes, moe_plan.kv_cache_total_bytes);
}
