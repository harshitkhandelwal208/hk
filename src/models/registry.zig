//! Model Architecture Registry (Plan §46, §47, §150).
//! Comprehensive registry for LLM families including LLaMA, Mistral, Gemma, Gemma 2,
//! Qwen 2, Qwen 3, Phi 3, DeepSeek, and their architectural parameter invariants.

const std = @import("std");
const config = @import("../engine/config.zig");
const Arch = config.Arch;
const RopeStyle = config.RopeStyle;
const FfnAct = config.FfnAct;

pub const AttentionType = enum {
    mha, // Multi-Head Attention (n_heads == n_kv_heads)
    gqa, // Grouped-Query Attention (1 < n_kv_heads < n_heads)
    mqa, // Multi-Query Attention (n_kv_heads == 1)
};

pub const NormalizationType = enum {
    rmsnorm,
    layernorm,
};

pub const VocabularyInfo = struct {
    default_vocab_size: usize,
    bos_id: ?u32 = null,
    eos_id: ?u32 = null,
    pad_id: ?u32 = null,
    unk_id: ?u32 = null,

    pub fn validateToken(self: VocabularyInfo, token: u32) bool {
        return token < self.default_vocab_size;
    }
};

pub const ArchitectureDescriptor = struct {
    name: []const u8,
    arch: Arch,
    canonical_family: []const u8,
    default_attention: AttentionType,
    ffn_activation: FfnAct,
    rope_style: RopeStyle,
    default_rope_base: f32,
    norm_type: NormalizationType,
    qk_norm: bool,
    qkv_bias: bool,
    supports_sliding_window: bool,
    default_sliding_window: ?usize = null,
    vocab_info: VocabularyInfo,

    pub fn matches(self: ArchitectureDescriptor, query: []const u8) bool {
        if (std.ascii.eqlIgnoreCase(self.name, query)) return true;
        if (std.ascii.eqlIgnoreCase(self.canonical_family, query)) return true;
        return false;
    }
};

pub const REGISTRY = [_]ArchitectureDescriptor{
    // LLaMA family (LLaMA 1, 2, 3, 3.1, 3.2, 3.3)
    .{
        .name = "llama",
        .arch = .llama,
        .canonical_family = "llama",
        .default_attention = .gqa,
        .ffn_activation = .silu,
        .rope_style = .norm,
        .default_rope_base = 500000.0,
        .norm_type = .rmsnorm,
        .qk_norm = false,
        .qkv_bias = false,
        .supports_sliding_window = false,
        .vocab_info = .{
            .default_vocab_size = 128256,
            .bos_id = 128000,
            .eos_id = 128001,
        },
    },
    // Mistral family (Mistral 7B, Mistral Nemo, Mixtral)
    .{
        .name = "mistral",
        .arch = .llama,
        .canonical_family = "mistral",
        .default_attention = .gqa,
        .ffn_activation = .silu,
        .rope_style = .norm,
        .default_rope_base = 10000.0,
        .norm_type = .rmsnorm,
        .qk_norm = false,
        .qkv_bias = false,
        .supports_sliding_window = true,
        .default_sliding_window = 4096,
        .vocab_info = .{
            .default_vocab_size = 32768,
            .bos_id = 1,
            .eos_id = 2,
        },
    },
    // Gemma family (Gemma 1, Gemma 2)
    .{
        .name = "gemma",
        .arch = .gemma,
        .canonical_family = "gemma",
        .default_attention = .gqa,
        .ffn_activation = .gelu_tanh,
        .rope_style = .neox,
        .default_rope_base = 10000.0,
        .norm_type = .rmsnorm,
        .qk_norm = true,
        .qkv_bias = false,
        .supports_sliding_window = true,
        .default_sliding_window = 4096,
        .vocab_info = .{
            .default_vocab_size = 256000,
            .bos_id = 2,
            .eos_id = 1,
            .pad_id = 0,
        },
    },
    .{
        .name = "gemma2",
        .arch = .gemma,
        .canonical_family = "gemma",
        .default_attention = .gqa,
        .ffn_activation = .gelu_tanh,
        .rope_style = .neox,
        .default_rope_base = 10000.0,
        .norm_type = .rmsnorm,
        .qk_norm = true,
        .qkv_bias = false,
        .supports_sliding_window = true,
        .default_sliding_window = 4096,
        .vocab_info = .{
            .default_vocab_size = 256000,
            .bos_id = 2,
            .eos_id = 1,
            .pad_id = 0,
        },
    },
    // Qwen 2 & 2.5
    .{
        .name = "qwen2",
        .arch = .qwen2,
        .canonical_family = "qwen",
        .default_attention = .gqa,
        .ffn_activation = .silu,
        .rope_style = .neox,
        .default_rope_base = 1000000.0,
        .norm_type = .rmsnorm,
        .qk_norm = false,
        .qkv_bias = true,
        .supports_sliding_window = true,
        .vocab_info = .{
            .default_vocab_size = 152064,
            .bos_id = 151643,
            .eos_id = 151645,
            .pad_id = 151643,
        },
    },
    // Qwen 3
    .{
        .name = "qwen3",
        .arch = .qwen3,
        .canonical_family = "qwen",
        .default_attention = .gqa,
        .ffn_activation = .silu,
        .rope_style = .neox,
        .default_rope_base = 1000000.0,
        .norm_type = .rmsnorm,
        .qk_norm = true,
        .qkv_bias = false,
        .supports_sliding_window = true,
        .vocab_info = .{
            .default_vocab_size = 152064,
            .bos_id = 151643,
            .eos_id = 151645,
            .pad_id = 151643,
        },
    },
    // Phi-3 & Phi-3.5
    .{
        .name = "phi3",
        .arch = .phi3,
        .canonical_family = "phi",
        .default_attention = .gqa,
        .ffn_activation = .silu,
        .rope_style = .neox,
        .default_rope_base = 10000.0,
        .norm_type = .rmsnorm,
        .qk_norm = false,
        .qkv_bias = true,
        .supports_sliding_window = true,
        .default_sliding_window = 2048,
        .vocab_info = .{
            .default_vocab_size = 32064,
            .bos_id = 1,
            .eos_id = 32000,
        },
    },
    // DeepSeek V2 / V3
    .{
        .name = "deepseek",
        .arch = .llama,
        .canonical_family = "deepseek",
        .default_attention = .gqa,
        .ffn_activation = .silu,
        .rope_style = .neox,
        .default_rope_base = 10000.0,
        .norm_type = .rmsnorm,
        .qk_norm = false,
        .qkv_bias = false,
        .supports_sliding_window = false,
        .vocab_info = .{
            .default_vocab_size = 102400,
            .bos_id = 0,
            .eos_id = 1,
        },
    },
    // Mixtral MoE family
    .{
        .name = "mixtral",
        .arch = .moe,
        .canonical_family = "moe",
        .default_attention = .gqa,
        .ffn_activation = .silu,
        .rope_style = .norm,
        .default_rope_base = 1000000.0,
        .norm_type = .rmsnorm,
        .qk_norm = false,
        .qkv_bias = false,
        .supports_sliding_window = true,
        .default_sliding_window = 4096,
        .vocab_info = .{
            .default_vocab_size = 32000,
            .bos_id = 1,
            .eos_id = 2,
        },
    },
    // DeepSeek MoE family
    .{
        .name = "deepseek_moe",
        .arch = .moe,
        .canonical_family = "moe",
        .default_attention = .gqa,
        .ffn_activation = .silu,
        .rope_style = .neox,
        .default_rope_base = 10000.0,
        .norm_type = .rmsnorm,
        .qk_norm = false,
        .qkv_bias = false,
        .supports_sliding_window = false,
        .vocab_info = .{
            .default_vocab_size = 102400,
            .bos_id = 0,
            .eos_id = 1,
        },
    },
    // Vision Transformer / CLIP
    .{
        .name = "clip",
        .arch = .clip_vision,
        .canonical_family = "vision",
        .default_attention = .mha,
        .ffn_activation = .gelu_tanh,
        .rope_style = .neox,
        .default_rope_base = 10000.0,
        .norm_type = .layernorm,
        .qk_norm = false,
        .qkv_bias = true,
        .supports_sliding_window = false,
        .vocab_info = .{
            .default_vocab_size = 49408,
            .bos_id = 49406,
            .eos_id = 49407,
        },
    },
    // Whisper Audio Transformer
    .{
        .name = "whisper",
        .arch = .whisper_audio,
        .canonical_family = "audio",
        .default_attention = .mha,
        .ffn_activation = .gelu_tanh,
        .rope_style = .neox,
        .default_rope_base = 10000.0,
        .norm_type = .layernorm,
        .qk_norm = false,
        .qkv_bias = true,
        .supports_sliding_window = false,
        .vocab_info = .{
            .default_vocab_size = 51865,
            .bos_id = 50258,
            .eos_id = 50257,
        },
    },
};

/// Lookup an architecture descriptor by name or family.
pub fn find(name: []const u8) ?ArchitectureDescriptor {
    for (REGISTRY) |desc| {
        if (desc.matches(name)) return desc;
    }
    return null;
}

test "ArchitectureRegistry: find LLaMA, Mistral, Gemma, Qwen, Phi, DeepSeek" {
    const llama_desc = find("llama").?;
    try std.testing.expectEqual(Arch.llama, llama_desc.arch);
    try std.testing.expectEqual(RopeStyle.norm, llama_desc.rope_style);
    try std.testing.expectEqual(false, llama_desc.qk_norm);
    try std.testing.expect(llama_desc.vocab_info.validateToken(100));
    try std.testing.expect(!llama_desc.vocab_info.validateToken(150000));

    const mistral_desc = find("mistral").?;
    try std.testing.expectEqual(Arch.llama, mistral_desc.arch);
    try std.testing.expect(mistral_desc.supports_sliding_window);
    try std.testing.expectEqual(@as(?usize, 4096), mistral_desc.default_sliding_window);

    const gemma_desc = find("gemma").?;
    try std.testing.expectEqual(Arch.gemma, gemma_desc.arch);
    try std.testing.expectEqual(FfnAct.gelu_tanh, gemma_desc.ffn_activation);
    try std.testing.expectEqual(true, gemma_desc.qk_norm);
    try std.testing.expectEqual(RopeStyle.neox, gemma_desc.rope_style);

    const qwen2_desc = find("qwen2").?;
    try std.testing.expectEqual(Arch.qwen2, qwen2_desc.arch);
    try std.testing.expectEqual(true, qwen2_desc.qkv_bias);

    const qwen3_desc = find("qwen3").?;
    try std.testing.expectEqual(Arch.qwen3, qwen3_desc.arch);
    try std.testing.expectEqual(true, qwen3_desc.qk_norm);

    const phi3_desc = find("phi3").?;
    try std.testing.expectEqual(Arch.phi3, phi3_desc.arch);

    const mixtral_desc = find("mixtral").?;
    try std.testing.expectEqual(Arch.moe, mixtral_desc.arch);
    try std.testing.expect(mixtral_desc.supports_sliding_window);

    const clip_desc = find("clip").?;
    try std.testing.expectEqual(Arch.clip_vision, clip_desc.arch);
    try std.testing.expectEqual(NormalizationType.layernorm, clip_desc.norm_type);

    const whisper_desc = find("whisper").?;
    try std.testing.expectEqual(Arch.whisper_audio, whisper_desc.arch);

    try std.testing.expect(find("nonexistent_model") == null);
}
