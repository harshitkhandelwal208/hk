//! Multimodal Vision & Audio Tensor Architecture Core (Plan §48, §87, §150).
//!
//! Provides hardware-neutral implementations for multimodal vision and audio encoders:
//! 1. Vision Transformer (ViT / CLIP) 2D patch extraction and linear embedding.
//! 2. 2D Spatial Positional Embedding addition for visual tokens.
//! 3. Vision-to-Language projection bridge (aligning visual tokens to LM hidden states).
//! 4. 1D Temporal Convolution feature extraction for audio models (Whisper).
//! 5. Audio-to-Decoder cross-attention projection adapter.

const std = @import("std");

pub const VisionPatchConfig = struct {
    image_height: usize,
    image_width: usize,
    channels: usize = 3,
    patch_size: usize,
    hidden_dim: usize,

    pub fn numPatchesY(self: VisionPatchConfig) usize {
        return self.image_height / self.patch_size;
    }

    pub fn numPatchesX(self: VisionPatchConfig) usize {
        return self.image_width / self.patch_size;
    }

    pub fn numPatches(self: VisionPatchConfig) usize {
        return self.numPatchesY() * self.numPatchesX();
    }

    pub fn patchDim(self: VisionPatchConfig) usize {
        return self.channels * self.patch_size * self.patch_size;
    }
};

/// Vision Patch Extractor & Linear Embedder
pub const VisionPatchEncoder = struct {
    config: VisionPatchConfig,

    pub fn init(cfg: VisionPatchConfig) VisionPatchEncoder {
        return .{ .config = cfg };
    }

    /// Extracts a single 2D spatial patch from CHW image into flat buffer.
    pub fn extractPatch(
        self: VisionPatchEncoder,
        image_chw: []const f32, // [channels * H * W]
        py: usize,
        px: usize,
        out_patch: []f32,
    ) void {
        const p = self.config.patch_size;
        const h = self.config.image_height;
        const w = self.config.image_width;
        const c = self.config.channels;
        std.debug.assert(out_patch.len >= self.config.patchDim());

        var out_idx: usize = 0;
        const start_y = py * p;
        const start_x = px * p;

        for (0..c) |ch| {
            const ch_offset = ch * h * w;
            for (0..p) |dy| {
                const row_offset = ch_offset + (start_y + dy) * w + start_x;
                for (0..p) |dx| {
                    out_patch[out_idx] = image_chw[row_offset + dx];
                    out_idx += 1;
                }
            }
        }
    }

    /// Projects patch features to hidden dimension and adds positional embedding:
    /// out_token = W_patch * patch + E_pos
    pub fn embedPatch(
        self: VisionPatchEncoder,
        patch: []const f32,
        patch_weights: []const f32, // [hidden_dim, patch_dim]
        pos_embed: []const f32, // [hidden_dim]
        out_token: []f32, // [hidden_dim]
    ) void {
        const hidden_dim = self.config.hidden_dim;
        const p_dim = self.config.patchDim();
        std.debug.assert(patch.len == p_dim);
        std.debug.assert(patch_weights.len == hidden_dim * p_dim);
        std.debug.assert(pos_embed.len == hidden_dim);
        std.debug.assert(out_token.len == hidden_dim);

        for (0..hidden_dim) |h_idx| {
            var sum: f32 = pos_embed[h_idx];
            const w_row = patch_weights[h_idx * p_dim .. (h_idx + 1) * p_dim];
            for (0..p_dim) |d| {
                sum += w_row[d] * patch[d];
            }
            out_token[h_idx] = sum;
        }
    }
};

/// Multimodal Vision-to-Language Projector
pub fn projectVisionToText(
    vision_tokens: []const f32, // [num_tokens, vision_dim]
    num_tokens: usize,
    vision_dim: usize,
    lm_dim: usize,
    proj_weights: []const f32, // [lm_dim, vision_dim]
    out_lm_tokens: []f32, // [num_tokens, lm_dim]
) void {
    std.debug.assert(proj_weights.len == lm_dim * vision_dim);
    std.debug.assert(out_lm_tokens.len >= num_tokens * lm_dim);

    for (0..num_tokens) |t| {
        const v_tok = vision_tokens[t * vision_dim .. (t + 1) * vision_dim];
        const out_tok = out_lm_tokens[t * lm_dim .. (t + 1) * lm_dim];

        for (0..lm_dim) |l| {
            var sum: f32 = 0.0;
            const w_row = proj_weights[l * vision_dim .. (l + 1) * vision_dim];
            for (0..vision_dim) |v| {
                sum += w_row[v] * v_tok[v];
            }
            out_tok[l] = sum;
        }
    }
}

pub const AudioMelConfig = struct {
    num_mel_bins: usize,
    time_steps: usize,
    hidden_dim: usize,
    conv_kernel_size: usize = 3,
    conv_stride: usize = 2,

    pub fn outputFrames(self: AudioMelConfig) usize {
        if (self.time_steps < self.conv_kernel_size) return 0;
        return (self.time_steps - self.conv_kernel_size) / self.conv_stride + 1;
    }
};

/// Audio 1D Temporal Convolution subsampler (Whisper front-end)
pub fn applyAudio1DConvSubsampling(
    mel_input: []const f32, // [num_mel_bins, time_steps]
    cfg: AudioMelConfig,
    weights: []const f32, // [hidden_dim, num_mel_bins, kernel_size]
    bias: ?[]const f32, // [hidden_dim]
    out_features: []f32, // [out_frames, hidden_dim]
) void {
    const out_frames = cfg.outputFrames();
    const k = cfg.conv_kernel_size;
    const stride = cfg.conv_stride;
    const mel_bins = cfg.num_mel_bins;
    const h_dim = cfg.hidden_dim;

    std.debug.assert(out_features.len >= out_frames * h_dim);

    for (0..out_frames) |f| {
        const t_start = f * stride;
        const out_frame = out_features[f * h_dim .. (f + 1) * h_dim];

        for (0..h_dim) |h| {
            var acc: f32 = if (bias) |b| b[h] else 0.0;
            const h_w = weights[h * mel_bins * k .. (h + 1) * mel_bins * k];

            for (0..mel_bins) |m| {
                const mel_row = mel_input[m * cfg.time_steps .. (m + 1) * cfg.time_steps];
                const w_bin = h_w[m * k .. (m + 1) * k];
                for (0..k) |ki| {
                    acc += mel_row[t_start + ki] * w_bin[ki];
                }
            }
            out_frame[h] = acc;
        }
    }
}

test "vision patch extraction and linear embedding" {
    // 1 channel, 4x4 image, patch size 2x2 -> 4 patches
    const cfg = VisionPatchConfig{
        .image_height = 4,
        .image_width = 4,
        .channels = 1,
        .patch_size = 2,
        .hidden_dim = 2,
    };
    const encoder = VisionPatchEncoder.init(cfg);
    try std.testing.expectEqual(@as(usize, 4), cfg.numPatches());
    try std.testing.expectEqual(@as(usize, 4), cfg.patchDim());

    // 4x4 image:
    // [ 1,  2,  3,  4 ]
    // [ 5,  6,  7,  8 ]
    // [ 9, 10, 11, 12 ]
    // [13, 14, 15, 16 ]
    const img = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    var patch00: [4]f32 = undefined;
    encoder.extractPatch(&img, 0, 0, &patch00);
    // Top-left 2x2: [1, 2, 5, 6]
    try std.testing.expectEqual(@as(f32, 1.0), patch00[0]);
    try std.testing.expectEqual(@as(f32, 2.0), patch00[1]);
    try std.testing.expectEqual(@as(f32, 5.0), patch00[2]);
    try std.testing.expectEqual(@as(f32, 6.0), patch00[3]);

    var patch11: [4]f32 = undefined;
    encoder.extractPatch(&img, 1, 1, &patch11);
    // Bottom-right 2x2: [11, 12, 15, 16]
    try std.testing.expectEqual(@as(f32, 11.0), patch11[0]);
    try std.testing.expectEqual(@as(f32, 12.0), patch11[1]);
    try std.testing.expectEqual(@as(f32, 15.0), patch11[2]);
    try std.testing.expectEqual(@as(f32, 16.0), patch11[3]);

    // Embed patch with weights and positional bias
    const weights = [_]f32{
        1.0, 0.0, 0.0, 0.0, // h0 picks patch[0]
        0.0, 0.0, 0.0, 1.0, // h1 picks patch[3]
    };
    const pos = [_]f32{ 0.5, -0.5 };
    var out_tok: [2]f32 = undefined;
    encoder.embedPatch(&patch00, &weights, &pos, &out_tok);

    // h0: 1.0 * 1.0 + 0.5 = 1.5
    // h1: 1.0 * 6.0 - 0.5 = 5.5
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), out_tok[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), out_tok[1], 1e-6);
}

test "vision to language projector" {
    // 2 visual tokens, vision_dim = 2, lm_dim = 3
    const v_tokens = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const proj = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
    };
    var lm_tokens: [6]f32 = undefined;
    projectVisionToText(&v_tokens, 2, 2, 3, &proj, &lm_tokens);

    // Token 0 [1, 2]:
    // l0 = 1*1 = 1
    // l1 = 1*2 = 2
    // l2 = 1+2 = 3
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), lm_tokens[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), lm_tokens[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), lm_tokens[2], 1e-6);

    // Token 1 [3, 4]:
    // l0 = 3, l1 = 4, l2 = 7
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), lm_tokens[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), lm_tokens[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), lm_tokens[5], 1e-6);
}

test "audio 1D convolution subsampling" {
    // 2 mel bins, 5 time steps, kernel = 3, stride = 2 -> (5 - 3)/2 + 1 = 2 frames
    const cfg = AudioMelConfig{
        .num_mel_bins = 2,
        .time_steps = 5,
        .hidden_dim = 1,
        .conv_kernel_size = 3,
        .conv_stride = 2,
    };
    try std.testing.expectEqual(@as(usize, 2), cfg.outputFrames());

    const mel = [_]f32{
        1.0, 2.0, 3.0, 4.0, 5.0, // mel 0
        2.0, 2.0, 2.0, 2.0, 2.0, // mel 1
    };
    // Kernel weights for 1 hidden unit:
    // mel 0 kernel: [1, 0, 0]
    // mel 1 kernel: [0, 1, 0]
    const w = [_]f32{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    };
    const bias = [_]f32{0.5};

    var out: [2]f32 = undefined;
    applyAudio1DConvSubsampling(&mel, cfg, &w, &bias, &out);

    // Frame 0 (t = 0):
    // mel0[0]*1 + mel1[1]*1 + 0.5 = 1.0 + 2.0 + 0.5 = 3.5
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), out[0], 1e-6);

    // Frame 1 (t = 2):
    // mel0[2]*1 + mel1[3]*1 + 0.5 = 3.0 + 2.0 + 0.5 = 5.5
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), out[1], 1e-6);
}
