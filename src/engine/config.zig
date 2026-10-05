//! Model hyperparameters, read from container metadata.
//!
//! Metadata keys follow the GGUF convention (`<arch>.block_count` and so on), so a model that
//! came from GGUF or from the safetensors importer is read the same way. Anything the engine
//! cannot honour is a load error that names the problem. Guessing a default for a missing
//! hyperparameter produces a model that runs and answers wrongly, which is worse than failing.

const std = @import("std");
const metadata = @import("../metadata.zig");

pub const Arch = enum {
    llama,
    qwen2,
    qwen3,

    pub fn name(self: Arch) []const u8 {
        return @tagName(self);
    }

    /// Architecture names as they appear in `general.architecture`.
    pub fn fromName(s: []const u8) ?Arch {
        const table = [_]struct { []const u8, Arch }{
            .{ "llama", .llama },
            // These families are llama compatible in the GGUF convention.
            .{ "mistral", .llama },
            .{ "qwen2", .qwen2 },
            .{ "qwen3", .qwen3 },
        };
        for (table) |e| if (std.mem.eql(u8, s, e[0])) return e[1];
        return null;
    }
};

/// How rotary position embedding pairs up the dimensions of a head.
pub const RopeStyle = enum {
    /// Adjacent pairs (2i, 2i+1). Used by llama style GGUF files.
    norm,
    /// Split halves (i, i + n/2). Used by Qwen, Gemma and most newer families.
    neox,
};

pub const FfnAct = enum { silu, gelu_tanh };

pub const RopeKind = enum { none, linear, llama3, yarn };

/// How rotary frequencies are stretched for contexts longer than the model was trained on.
/// The formulas follow Hugging Face Transformers and llama.cpp, and are checked against both.
pub const RopeScaling = struct {
    kind: RopeKind = .none,
    factor: f32 = 1.0,
    /// Context length the model was trained with before scaling (llama3 and yarn).
    orig_ctx: usize = 0,
    /// llama3: wavelengths shorter than orig_ctx / high are kept, longer than orig_ctx / low are
    /// divided by `factor`, and the band between is blended.
    low_freq_factor: f32 = 1.0,
    high_freq_factor: f32 = 4.0,
    /// yarn: rotation counts that bound the blended band.
    beta_fast: f32 = 32.0,
    beta_slow: f32 = 1.0,
    /// yarn: extra multiplier on the rotation magnitude, on top of the 0.1 ln(factor) + 1 term.
    attn_factor: f32 = 1.0,
};

pub const Config = struct {
    arch: Arch,
    n_layers: usize,
    dim: usize,
    ffn_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    /// Per head width of queries, keys and values.
    head_dim: usize,
    vocab: usize,
    ctx_train: usize,
    rms_eps: f32,
    rope_base: f32,
    /// Number of leading dimensions of each head that get rotated.
    rope_dim: usize,
    rope_style: RopeStyle,
    rope_scaling: RopeScaling,
    ffn_act: FfnAct,
    /// Qwen3 style RMS norm applied to each query and key head before RoPE.
    qk_norm: bool,
    /// Qwen2 style bias on the q, k and v projections.
    qkv_bias: bool,

    pub fn qDim(self: Config) usize {
        return self.n_heads * self.head_dim;
    }

    pub fn kvDim(self: Config) usize {
        return self.n_kv_heads * self.head_dim;
    }
};

pub const ConfigError = error{
    UnsupportedArchitecture,
    MissingKey,
    InvalidValue,
};

/// Human readable detail for the last failure, so callers can print something better than an
/// error code. Fixed size to keep loading allocation free on the failure path.
pub const Diag = struct {
    buf: [256]u8 = undefined,
    len: usize = 0,

    pub fn set(self: *Diag, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(&self.buf, fmt, args) catch self.buf[0..];
        self.len = s.len;
    }

    pub fn message(self: *const Diag) []const u8 {
        return self.buf[0..self.len];
    }
};

fn key(buf: []u8, arch_name: []const u8, comptime suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ arch_name, suffix }) catch unreachable;
}

fn needInt(meta: *const metadata.MetadataMap, arch_name: []const u8, comptime suffix: []const u8, diag: *Diag) ConfigError!usize {
    var kb: [96]u8 = undefined;
    const k = key(&kb, arch_name, suffix);
    const v = meta.getInt(k) orelse {
        diag.set("model metadata is missing '{s}'", .{k});
        return error.MissingKey;
    };
    if (v <= 0) {
        diag.set("metadata '{s}' must be positive, got {d}", .{ k, v });
        return error.InvalidValue;
    }
    return @intCast(v);
}

fn optInt(meta: *const metadata.MetadataMap, arch_name: []const u8, comptime suffix: []const u8) ?usize {
    var kb: [96]u8 = undefined;
    const v = meta.getInt(key(&kb, arch_name, suffix)) orelse return null;
    return if (v > 0) @intCast(v) else null;
}

fn optFloat(meta: *const metadata.MetadataMap, arch_name: []const u8, comptime suffix: []const u8) ?f64 {
    var kb: [96]u8 = undefined;
    const k = key(&kb, arch_name, suffix);
    const v = meta.get(k) orelse return null;
    return switch (v) {
        .val_float64 => |f| f,
        .val_int64 => |i| @floatFromInt(i),
        else => null,
    };
}

fn fullFloat(meta: *const metadata.MetadataMap, k: []const u8) ?f64 {
    return switch (meta.get(k) orelse return null) {
        .val_float64 => |f| f,
        .val_int64 => |i| @floatFromInt(i),
        else => null,
    };
}

/// Reads the rope scaling keys. An unknown type is an error: running with the wrong positions
/// gives plausible but worse output, which is the hardest kind of bug to notice.
fn parseRopeScaling(meta: *const metadata.MetadataMap, arch_name: []const u8, ctx_train: usize, rope_dim: usize, diag: *Diag) ConfigError!RopeScaling {
    var kb: [96]u8 = undefined;
    var r = RopeScaling{};
    const factor = optFloat(meta, arch_name, "rope.scaling.factor");
    const type_name = meta.getString(key(&kb, arch_name, "rope.scaling.type"));
    if (type_name) |st| {
        if (std.mem.eql(u8, st, "none")) {
            r.kind = .none;
        } else if (std.mem.eql(u8, st, "linear")) {
            r.kind = .linear;
        } else if (std.mem.eql(u8, st, "llama3")) {
            r.kind = .llama3;
        } else if (std.mem.eql(u8, st, "yarn")) {
            r.kind = .yarn;
        } else {
            diag.set("rope scaling type '{s}' is not supported (supported: none, linear, llama3, yarn)", .{st});
            return error.InvalidValue;
        }
    } else if (factor != null) {
        // Older files give a factor and no type, which has always meant linear.
        r.kind = .linear;
    }
    if (r.kind == .none) return r;

    const f = factor orelse {
        diag.set("rope scaling '{s}' needs '{s}.rope.scaling.factor'", .{ @tagName(r.kind), arch_name });
        return error.MissingKey;
    };
    if (!(f >= 1.0) or !std.math.isFinite(f)) {
        diag.set("rope scaling factor {d} must be at least 1", .{f});
        return error.InvalidValue;
    }
    r.factor = @floatCast(f);
    r.orig_ctx = optInt(meta, arch_name, "rope.scaling.original_context_length") orelse ctx_train;

    switch (r.kind) {
        .llama3 => {
            if (fullFloat(meta, "hk.rope.low_freq_factor")) |v| r.low_freq_factor = @floatCast(v);
            if (fullFloat(meta, "hk.rope.high_freq_factor")) |v| r.high_freq_factor = @floatCast(v);
            if (!(r.low_freq_factor > 0 and r.high_freq_factor > r.low_freq_factor)) {
                diag.set("llama3 rope scaling needs 0 < low_freq_factor < high_freq_factor, got {d} and {d}", .{ r.low_freq_factor, r.high_freq_factor });
                return error.InvalidValue;
            }
        },
        .yarn => {
            if (fullFloat(meta, "hk.rope.beta_fast")) |v| r.beta_fast = @floatCast(v);
            if (fullFloat(meta, "hk.rope.beta_slow")) |v| r.beta_slow = @floatCast(v);
            if (optFloat(meta, arch_name, "rope.scaling.attn_factor")) |v| r.attn_factor = @floatCast(v);
            if (!(r.beta_fast > 0 and r.beta_slow > 0 and r.attn_factor > 0)) {
                diag.set("yarn rope scaling needs positive beta_fast, beta_slow and attn_factor", .{});
                return error.InvalidValue;
            }
        },
        else => {},
    }
    if (r.kind != .linear and (r.orig_ctx == 0 or rope_dim == 0)) {
        diag.set("rope scaling needs the original context length", .{});
        return error.InvalidValue;
    }
    return r;
}

/// Reads and validates the configuration. `vocab_rows` is the row count of the token embedding
/// tensor, which is the authority on vocabulary size.
pub fn fromMetadata(meta: *const metadata.MetadataMap, vocab_rows: usize, diag: *Diag) ConfigError!Config {
    const arch_name = meta.getString("general.architecture") orelse {
        diag.set("model metadata has no 'general.architecture'", .{});
        return error.MissingKey;
    };
    const arch = Arch.fromName(arch_name) orelse {
        diag.set("architecture '{s}' is not supported by this build (supported: llama, mistral, qwen2, qwen3)", .{arch_name});
        return error.UnsupportedArchitecture;
    };

    const dim = try needInt(meta, arch_name, "embedding_length", diag);
    const n_heads = try needInt(meta, arch_name, "attention.head_count", diag);
    // A missing kv head count means plain multi head attention in the GGUF convention.
    const n_kv_heads = optInt(meta, arch_name, "attention.head_count_kv") orelse n_heads;
    if (n_heads % n_kv_heads != 0) {
        diag.set("head_count {d} is not a multiple of head_count_kv {d}", .{ n_heads, n_kv_heads });
        return error.InvalidValue;
    }

    // Qwen3 stores the head width explicitly because it is not dim / heads there.
    const head_dim = optInt(meta, arch_name, "attention.key_length") orelse blk: {
        if (dim % n_heads != 0) {
            diag.set("embedding_length {d} is not divisible by head_count {d} and no key_length is given", .{ dim, n_heads });
            return error.InvalidValue;
        }
        break :blk dim / n_heads;
    };
    if (optInt(meta, arch_name, "attention.value_length")) |vl| {
        if (vl != head_dim) {
            diag.set("value_length {d} differs from key_length {d}; not supported", .{ vl, head_dim });
            return error.InvalidValue;
        }
    }

    if (head_dim == 0 or head_dim > 256 or head_dim % 2 != 0) {
        diag.set("head width {d} is not supported (an even number up to 256)", .{head_dim});
        return error.InvalidValue;
    }
    const rope_dim = optInt(meta, arch_name, "rope.dimension_count") orelse head_dim;
    if (rope_dim > head_dim or rope_dim % 2 != 0) {
        diag.set("rope.dimension_count {d} is invalid for head width {d}", .{ rope_dim, head_dim });
        return error.InvalidValue;
    }

    // Sliding window attention changes which positions are visible. Running such a model with
    // full attention is wrong once the context passes the window, so refuse it instead.
    var swk: [96]u8 = undefined;
    if (meta.getInt(key(&swk, arch_name, "attention.sliding_window"))) |w| {
        if (w > 0) {
            diag.set("sliding window attention ({d}) is not supported yet", .{w});
            return error.InvalidValue;
        }
    }

    const rope_scaling = try parseRopeScaling(meta, arch_name, optInt(meta, arch_name, "context_length") orelse 2048, rope_dim, diag);

    return .{
        .arch = arch,
        .n_layers = try needInt(meta, arch_name, "block_count", diag),
        .dim = dim,
        .ffn_dim = try needInt(meta, arch_name, "feed_forward_length", diag),
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .vocab = vocab_rows,
        .ctx_train = optInt(meta, arch_name, "context_length") orelse 2048,
        .rms_eps = @floatCast(optFloat(meta, arch_name, "attention.layer_norm_rms_epsilon") orelse 1e-5),
        .rope_base = @floatCast(optFloat(meta, arch_name, "rope.freq_base") orelse 10000.0),
        .rope_dim = rope_dim,
        .rope_style = blk: {
            // A container can say how its weights are laid out. HF imports use the half-split
            // layout even for llama, whereas llama GGUF files are permuted for adjacent pairs.
            if (meta.getString("hk.rope_style")) |rs| {
                if (std.mem.eql(u8, rs, "neox")) break :blk .neox;
                if (std.mem.eql(u8, rs, "norm")) break :blk .norm;
                diag.set("hk.rope_style '{s}' is not 'neox' or 'norm'", .{rs});
                return error.InvalidValue;
            }
            break :blk switch (arch) {
                .llama => .norm,
                .qwen2, .qwen3 => .neox,
            };
        },
        .rope_scaling = rope_scaling,
        .ffn_act = .silu,
        .qk_norm = arch == .qwen3,
        .qkv_bias = arch == .qwen2,
    };
}
