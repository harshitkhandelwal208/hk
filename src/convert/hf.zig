//! Streaming conversion of a Hugging Face model (safetensors shards plus config and tokenizer
//! files) into an .hk container.
//!
//! The converter reads only headers up front: the safetensors JSON headers, `config.json` and
//! the tokenizer files. That is enough to fix the whole layout of the output. Weights are then
//! copied shard by shard, in file order, straight through a fixed buffer. They keep their stored
//! dtype (BF16 stays BF16) and nothing is transposed or rescaled, so conversion does no
//! arithmetic on weights, holds no tensor in memory, and cannot change a single bit of them.
//!
//! HF checkpoints store attention projections in the layout the half-split ("neox") rotary
//! embedding expects. Rather than permute rows the way GGUF converters do for llama, the
//! container records `hk.rope_style = neox` and the engine reads them as they are.

const std = @import("std");
const format = @import("../format.zig");
const metadata = @import("../metadata.zig");
const arrays = @import("../arrays.zig");
const source_mod = @import("source.zig");
const stream_writer = @import("stream_writer.zig");
const hf_tokenizer = @import("hf_tokenizer.zig");
const gguf = @import("gguf.zig");

pub const Options = gguf.Options;
pub const Diag = gguf.Diag;
const Source = source_mod.Source;

pub const Shard = struct {
    name: []const u8,
    src: Source,
};

pub const Inputs = struct {
    config_json: []const u8,
    tokenizer_json: ?[]const u8 = null,
    tokenizer_config_json: ?[]const u8 = null,
    generation_config_json: ?[]const u8 = null,
    shards: []const Shard,
    /// Used for `general.name`.
    model_name: []const u8 = "",
};

pub const ConvertError = error{
    UnsupportedArchitecture,
    UnsupportedConfig,
    BadConfig,
    BadSafetensors,
    UnknownTensor,
    UnsupportedDtype,
    MissingTokenizer,
} || gguf.ConvertError;

const Arch = enum { llama, qwen2, qwen3 };

fn fail(opts: Options, comptime fmt: []const u8, args: anytype) void {
    if (opts.diag) |d| d.set(fmt, args);
}

const Tensor = struct {
    hk_name: []const u8,
    dtype: format.StorageType,
    ndim: u8,
    shape: [format.MAX_DIMS]u64,
    begin: u64,
    end: u64,
    shard: usize,
};

const Header = struct {
    data_start: u64,
    value: std.json.Parsed(std.json.Value),
};

fn dtypeOf(name: []const u8) ?struct { t: format.StorageType, size: u64 } {
    const table = [_]struct { []const u8, format.StorageType, u64 }{
        .{ "F32", .f32, 4 },
        .{ "F16", .f16, 2 },
        .{ "BF16", .bf16, 2 },
        .{ "F64", .f64, 8 },
        .{ "I8", .int8, 1 },
        .{ "U8", .uint8, 1 },
        .{ "I16", .int16, 2 },
        .{ "I32", .int32, 4 },
        .{ "I64", .int64, 8 },
        .{ "BOOL", .bool, 1 },
        .{ "F8_E4M3", .fp8_e4m3, 1 },
        .{ "F8_E5M2", .fp8_e5m2, 1 },
    };
    for (table) |e| if (std.mem.eql(u8, e[0], name)) return .{ .t = e[1], .size = e[2] };
    return null;
}

/// Maps a Hugging Face tensor name to the container's name. Null means "not a weight": derived
/// buffers that the engine recomputes. Unknown names are an error, never silently dropped, so a
/// model with parts we do not understand is refused instead of converted wrong.
fn mapName(allocator: std.mem.Allocator, name: []const u8) !?[]const u8 {
    if (std.mem.eql(u8, name, "model.embed_tokens.weight")) return try allocator.dupe(u8, "token_embd.weight");
    if (std.mem.eql(u8, name, "model.norm.weight")) return try allocator.dupe(u8, "output_norm.weight");
    if (std.mem.eql(u8, name, "lm_head.weight")) return try allocator.dupe(u8, "output.weight");
    if (std.mem.endsWith(u8, name, "rotary_emb.inv_freq")) return null;

    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, name, prefix)) return error.UnknownTensor;
    const rest = name[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.UnknownTensor;
    const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return error.UnknownTensor;
    const tail = rest[dot + 1 ..];

    const map = [_]struct { []const u8, []const u8 }{
        .{ "self_attn.q_proj.weight", "attn_q.weight" },
        .{ "self_attn.q_proj.bias", "attn_q.bias" },
        .{ "self_attn.k_proj.weight", "attn_k.weight" },
        .{ "self_attn.k_proj.bias", "attn_k.bias" },
        .{ "self_attn.v_proj.weight", "attn_v.weight" },
        .{ "self_attn.v_proj.bias", "attn_v.bias" },
        .{ "self_attn.o_proj.weight", "attn_output.weight" },
        .{ "self_attn.q_norm.weight", "attn_q_norm.weight" },
        .{ "self_attn.k_norm.weight", "attn_k_norm.weight" },
        .{ "mlp.gate_proj.weight", "ffn_gate.weight" },
        .{ "mlp.up_proj.weight", "ffn_up.weight" },
        .{ "mlp.down_proj.weight", "ffn_down.weight" },
        .{ "input_layernorm.weight", "attn_norm.weight" },
        .{ "post_attention_layernorm.weight", "ffn_norm.weight" },
    };
    for (map) |e| {
        if (std.mem.eql(u8, tail, e[0])) return try std.fmt.allocPrint(allocator, "blk.{d}.{s}", .{ layer, e[1] });
    }
    if (std.mem.endsWith(u8, tail, "rotary_emb.inv_freq")) return null;
    return error.UnknownTensor;
}

fn objGet(v: std.json.Value, key: []const u8) ?std.json.Value {
    if (v != .object) return null;
    return v.object.get(key);
}

fn asInt(v: ?std.json.Value) ?i64 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| i,
        .float => |f| if (@floor(f) == f) @as(i64, @intFromFloat(f)) else null,
        else => null,
    };
}

fn asFloat(v: ?std.json.Value) ?f64 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

fn asString(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return switch (x) {
        .string => |s| s,
        else => null,
    };
}

fn asBool(v: ?std.json.Value) ?bool {
    const x = v orelse return null;
    return switch (x) {
        .bool => |b| b,
        else => null,
    };
}

/// An id that the config may give as one integer or as a list.
fn firstId(v: ?std.json.Value) ?i64 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| i,
        .array => |a| if (a.items.len > 0) asInt(a.items[0]) else null,
        else => null,
    };
}

fn collectIds(v: ?std.json.Value, out: *std.ArrayList(i32), allocator: std.mem.Allocator) !void {
    const x = v orelse return;
    switch (x) {
        .integer => |i| try addUnique(out, allocator, @intCast(i)),
        .array => |a| for (a.items) |it| if (asInt(it)) |i| try addUnique(out, allocator, @intCast(i)),
        else => {},
    }
}

fn addUnique(out: *std.ArrayList(i32), allocator: std.mem.Allocator, id: i32) !void {
    for (out.items) |e| if (e == id) return;
    try out.append(allocator, id);
}

/// Name of a special token in tokenizer_config.json, which is a string or an object with `content`.
fn tokenName(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return switch (x) {
        .string => |s| s,
        .object => |o| asString(o.get("content")),
        else => null,
    };
}

fn findToken(blob: []const u8, name: []const u8) ?u32 {
    const arr = arrays.StringArray.init(blob) catch return null;
    for (0..arr.count) |i| if (std.mem.eql(u8, arr.get(i), name)) return @intCast(i);
    return null;
}

pub fn convert(
    allocator: std.mem.Allocator,
    io: std.Io,
    inputs: Inputs,
    out_path: []const u8,
    opts: Options,
) ConvertError!void {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var cfg_parsed = std.json.parseFromSlice(std.json.Value, allocator, inputs.config_json, .{}) catch {
        fail(opts, "config.json is not valid JSON", .{});
        return error.BadConfig;
    };
    defer cfg_parsed.deinit();
    const cfg = cfg_parsed.value;

    // ---- architecture ------------------------------------------------------------------
    const arch_name = blk: {
        if (objGet(cfg, "architectures")) |a| if (a == .array and a.array.items.len > 0) if (asString(a.array.items[0])) |s| break :blk s;
        break :blk asString(objGet(cfg, "model_type")) orelse {
            fail(opts, "config.json names no architecture", .{});
            return error.BadConfig;
        };
    };
    const arch: Arch = if (std.mem.eql(u8, arch_name, "LlamaForCausalLM") or std.mem.eql(u8, arch_name, "MistralForCausalLM") or std.mem.eql(u8, arch_name, "llama") or std.mem.eql(u8, arch_name, "mistral"))
        .llama
    else if (std.mem.eql(u8, arch_name, "Qwen2ForCausalLM") or std.mem.eql(u8, arch_name, "qwen2"))
        .qwen2
    else if (std.mem.eql(u8, arch_name, "Qwen3ForCausalLM") or std.mem.eql(u8, arch_name, "qwen3"))
        .qwen3
    else {
        fail(opts, "architecture '{s}' is not supported (supported: Llama, Mistral, Qwen2, Qwen3)", .{arch_name});
        return error.UnsupportedArchitecture;
    };
    const arch_str = @tagName(arch);

    if (asString(objGet(cfg, "hidden_act"))) |act| if (!std.mem.eql(u8, act, "silu")) {
        fail(opts, "activation '{s}' is not supported (only silu)", .{act});
        return error.UnsupportedConfig;
    };
    if (asBool(objGet(cfg, "mlp_bias")) orelse false) {
        fail(opts, "mlp_bias is not supported", .{});
        return error.UnsupportedConfig;
    }
    if (arch == .llama and (asBool(objGet(cfg, "attention_bias")) orelse false)) {
        fail(opts, "attention_bias on a llama architecture is not supported", .{});
        return error.UnsupportedConfig;
    }
    if (asInt(objGet(cfg, "sliding_window"))) |w| {
        const used = asBool(objGet(cfg, "use_sliding_window")) orelse (arch == .llama);
        if (used and w > 0) {
            fail(opts, "sliding window attention ({d}) is not supported yet", .{w});
            return error.UnsupportedConfig;
        }
    }

    const hidden = asInt(objGet(cfg, "hidden_size")) orelse return badKey(opts, "hidden_size");
    const n_layers = asInt(objGet(cfg, "num_hidden_layers")) orelse return badKey(opts, "num_hidden_layers");
    const ffn = asInt(objGet(cfg, "intermediate_size")) orelse return badKey(opts, "intermediate_size");
    const heads = asInt(objGet(cfg, "num_attention_heads")) orelse return badKey(opts, "num_attention_heads");
    const kv_heads = asInt(objGet(cfg, "num_key_value_heads")) orelse heads;
    const head_dim = asInt(objGet(cfg, "head_dim")) orelse @divTrunc(hidden, heads);
    if (heads <= 0 or hidden <= 0 or head_dim <= 0) return badKey(opts, "head sizes");

    var meta = metadata.MetadataMap.init(allocator);
    defer meta.deinit();
    var kb: [96]u8 = undefined;
    const key = struct {
        fn f(buf: []u8, a: []const u8, suffix: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "{s}.{s}", .{ a, suffix }) catch unreachable;
        }
    }.f;

    try meta.setString("general.architecture", arch_str);
    if (inputs.model_name.len > 0) try meta.setString("general.name", inputs.model_name);
    try meta.setInt(key(&kb, arch_str, "block_count"), n_layers);
    try meta.setInt(key(&kb, arch_str, "context_length"), asInt(objGet(cfg, "max_position_embeddings")) orelse 4096);
    try meta.setInt(key(&kb, arch_str, "embedding_length"), hidden);
    try meta.setInt(key(&kb, arch_str, "feed_forward_length"), ffn);
    try meta.setInt(key(&kb, arch_str, "attention.head_count"), heads);
    try meta.setInt(key(&kb, arch_str, "attention.head_count_kv"), kv_heads);
    try meta.setFloat(key(&kb, arch_str, "attention.layer_norm_rms_epsilon"), asFloat(objGet(cfg, "rms_norm_eps")) orelse 1e-6);
    try meta.setFloat(key(&kb, arch_str, "rope.freq_base"), asFloat(objGet(cfg, "rope_theta")) orelse 10000.0);
    if (head_dim != @divTrunc(hidden, heads)) {
        try meta.setInt(key(&kb, arch_str, "attention.key_length"), head_dim);
        try meta.setInt(key(&kb, arch_str, "attention.value_length"), head_dim);
    }
    const partial = asFloat(objGet(cfg, "partial_rotary_factor")) orelse 1.0;
    try meta.setInt(key(&kb, arch_str, "rope.dimension_count"), @intFromFloat(@as(f64, @floatFromInt(head_dim)) * partial));
    if (asInt(objGet(cfg, "vocab_size"))) |v| try meta.setInt(key(&kb, arch_str, "vocab_size"), v);

    if (objGet(cfg, "rope_scaling")) |rs| if (rs == .object) {
        const ty = asString(rs.object.get("rope_type")) orelse asString(rs.object.get("type")) orelse "default";
        if (!std.mem.eql(u8, ty, "default")) {
            try meta.setString(key(&kb, arch_str, "rope.scaling.type"), ty);
            if (asFloat(rs.object.get("factor"))) |f| try meta.setFloat(key(&kb, arch_str, "rope.scaling.factor"), f);
        }
    };

    // HF weights are stored for the half-split rotary layout.
    try meta.setString("hk.rope_style", "neox");
    const tie = asBool(objGet(cfg, "tie_word_embeddings")) orelse false;

    // ---- tokenizer --------------------------------------------------------------------
    var parsed_tok: ?hf_tokenizer.Parsed = null;
    defer if (parsed_tok) |*p| p.deinit(allocator);
    const tj = inputs.tokenizer_json orelse {
        fail(opts, "the repository has no tokenizer.json", .{});
        return error.MissingTokenizer;
    };
    var tdiag = hf_tokenizer.Diag{};
    parsed_tok = hf_tokenizer.parse(allocator, tj, &tdiag) catch |e| {
        fail(opts, "tokenizer: {s}", .{tdiag.message()});
        return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.ArrayTooLarge => error.BadConfig,
            error.UnsupportedTokenizer => error.UnsupportedConfig,
            error.InvalidTokenizerJson => error.MissingTokenizer,
        };
    };
    const pt = parsed_tok.?;
    try meta.setString("tokenizer.ggml.model", "gpt2");
    // The container takes ownership of the blobs it is given, so hand it copies.
    try meta.set("tokenizer.ggml.tokens", .{ .val_bytes = try allocator.dupe(u8, pt.tokens_blob) });
    try meta.set("tokenizer.ggml.token_type", .{ .val_bytes = try allocator.dupe(u8, pt.types_blob) });
    try meta.set("tokenizer.ggml.merges", .{ .val_bytes = try allocator.dupe(u8, pt.merges_blob) });
    try meta.set("tokenizer.hk.pre_regex", .{ .val_bytes = try allocator.dupe(u8, pt.patterns_blob) });
    try meta.setBool("tokenizer.ggml.ignore_merges", pt.ignore_merges);

    var tcfg_parsed: ?std.json.Parsed(std.json.Value) = null;
    defer if (tcfg_parsed) |*p| p.deinit();
    if (inputs.tokenizer_config_json) |tc| tcfg_parsed = std.json.parseFromSlice(std.json.Value, allocator, tc, .{}) catch null;
    const tcfg: ?std.json.Value = if (tcfg_parsed) |p| p.value else null;

    const special = [_]struct { []const u8, []const u8, []const u8 }{
        .{ "bos_token", "bos_token_id", "tokenizer.ggml.bos_token_id" },
        .{ "eos_token", "eos_token_id", "tokenizer.ggml.eos_token_id" },
        .{ "pad_token", "pad_token_id", "tokenizer.ggml.padding_token_id" },
        .{ "unk_token", "unk_token_id", "tokenizer.ggml.unknown_token_id" },
    };
    for (special) |s| {
        var id: ?i64 = firstId(objGet(cfg, s[1]));
        if (id == null) if (tcfg) |c| if (tokenName(objGet(c, s[0]))) |n| if (findToken(pt.tokens_blob, n)) |f| {
            id = f;
        };
        if (id) |v| try meta.setInt(s[2], v);
    }
    if (tcfg) |c| {
        if (asBool(objGet(c, "add_bos_token"))) |b| try meta.setBool("tokenizer.ggml.add_bos_token", b);
        if (asBool(objGet(c, "add_eos_token"))) |b| try meta.setBool("tokenizer.ggml.add_eos_token", b);
        if (objGet(c, "chat_template")) |ct| switch (ct) {
            .string => |s| try meta.setString("tokenizer.chat_template", s),
            .array => |arr| for (arr.items) |item| {
                if (asString(objGet(item, "name"))) |nm| if (std.mem.eql(u8, nm, "default")) {
                    if (asString(objGet(item, "template"))) |t| try meta.setString("tokenizer.chat_template", t);
                };
            },
            else => {},
        };
    }

    // Tokens that end generation: everything the config files name as EOS.
    var eog: std.ArrayList(i32) = .empty;
    defer eog.deinit(allocator);
    try collectIds(objGet(cfg, "eos_token_id"), &eog, allocator);
    if (inputs.generation_config_json) |gj| {
        if (std.json.parseFromSlice(std.json.Value, allocator, gj, .{})) |gp| {
            var g = gp;
            defer g.deinit();
            try collectIds(objGet(g.value, "eos_token_id"), &eog, allocator);
        } else |_| {}
    }
    if (eog.items.len > 0) {
        const raw = try allocator.alloc(u8, eog.items.len * 4);
        defer allocator.free(raw);
        for (eog.items, 0..) |id, i| std.mem.writeInt(i32, raw[i * 4 ..][0..4], id, .little);
        try meta.set("tokenizer.hk.eog_ids", .{ .val_bytes = try arrays.fixedBlob(allocator, .i32, eog.items.len, raw) });
    }

    try meta.setString("hk.source.format", "safetensors");
    for (opts.extra_meta) |kv| try meta.setString(kv[0], kv[1]);

    // ---- safetensors headers -----------------------------------------------------------
    var tensors: std.ArrayList(Tensor) = .empty;
    defer tensors.deinit(allocator);
    const headers = try allocator.alloc(Header, inputs.shards.len);
    defer allocator.free(headers);
    var n_headers: usize = 0;
    defer for (headers[0..n_headers]) |*h| h.value.deinit();

    var has_lm_head = false;
    for (inputs.shards, 0..) |shard, si| {
        var len_bytes: [8]u8 = undefined;
        try shard.src.readExact(0, &len_bytes);
        const hlen = std.mem.readInt(u64, &len_bytes, .little);
        if (hlen == 0 or hlen > 256 << 20) {
            fail(opts, "shard '{s}' has an implausible header size ({d})", .{ shard.name, hlen });
            return error.BadSafetensors;
        }
        if (shard.src.len) |total| if (8 + hlen > total) {
            fail(opts, "shard '{s}' is truncated: its header alone is larger than the file", .{shard.name});
            return error.BadSafetensors;
        };
        const hbuf = try allocator.alloc(u8, @intCast(hlen));
        defer allocator.free(hbuf);
        try shard.src.readExact(8, hbuf);
        headers[si] = .{
            .data_start = 8 + hlen,
            .value = std.json.parseFromSlice(std.json.Value, allocator, hbuf, .{}) catch {
                fail(opts, "shard '{s}' has a malformed header", .{shard.name});
                return error.BadSafetensors;
            },
        };
        n_headers += 1;
        const root = headers[si].value.value;
        if (root != .object) return error.BadSafetensors;
        var it = root.object.iterator();
        while (it.next()) |kv| {
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
            const hf_name = kv.key_ptr.*;
            const mapped = mapName(arena, hf_name) catch {
                fail(opts, "tensor '{s}' is not part of a model this build understands. Refusing to drop it silently.", .{hf_name});
                return error.UnknownTensor;
            };
            const hk_name = mapped orelse continue;
            if (std.mem.eql(u8, hk_name, "output.weight")) has_lm_head = true;

            const t = kv.value_ptr.*;
            const dt_name = asString(objGet(t, "dtype")) orelse return error.BadSafetensors;
            const dt = dtypeOf(dt_name) orelse {
                fail(opts, "tensor '{s}' has dtype {s}, which HK cannot store", .{ hf_name, dt_name });
                return error.UnsupportedDtype;
            };
            const shape_v = objGet(t, "shape") orelse return error.BadSafetensors;
            const offs_v = objGet(t, "data_offsets") orelse return error.BadSafetensors;
            if (shape_v != .array or offs_v != .array or offs_v.array.items.len != 2) return error.BadSafetensors;
            if (shape_v.array.items.len == 0 or shape_v.array.items.len > format.MAX_DIMS) {
                fail(opts, "tensor '{s}' has {d} dimensions", .{ hf_name, shape_v.array.items.len });
                return error.BadSafetensors;
            }
            var shape: [format.MAX_DIMS]u64 = @splat(0);
            var numel: u64 = 1;
            for (shape_v.array.items, 0..) |d, i| {
                const dv = asInt(d) orelse return error.BadSafetensors;
                if (dv < 0) return error.BadSafetensors;
                shape[i] = @intCast(dv);
                numel = std.math.mul(u64, numel, shape[i]) catch return error.BadSafetensors;
            }
            const begin: u64 = @intCast(asInt(offs_v.array.items[0]) orelse return error.BadSafetensors);
            const end: u64 = @intCast(asInt(offs_v.array.items[1]) orelse return error.BadSafetensors);
            if (end < begin or end - begin != numel * dt.size) {
                fail(opts, "tensor '{s}': {d} bytes do not match shape and dtype ({d} expected)", .{ hf_name, end -| begin, numel * dt.size });
                return error.BadSafetensors;
            }
            if (shard.src.len) |total| if (headers[si].data_start + end > total) {
                fail(opts, "shard '{s}' is truncated: tensor '{s}' ends past the end of the file", .{ shard.name, hf_name });
                return error.BadSafetensors;
            };
            try tensors.append(allocator, .{
                .hk_name = hk_name,
                .dtype = dt.t,
                .ndim = @intCast(shape_v.array.items.len),
                .shape = shape,
                .begin = begin,
                .end = end,
                .shard = si,
            });
        }
    }
    if (!has_lm_head and !tie) {
        fail(opts, "the checkpoint has no lm_head.weight and tie_word_embeddings is not set", .{});
        return error.BadConfig;
    }

    // Copy order: by shard, then by position in the shard, so every shard is read front to back.
    std.mem.sort(Tensor, tensors.items, {}, struct {
        fn lt(_: void, a: Tensor, b: Tensor) bool {
            if (a.shard != b.shard) return a.shard < b.shard;
            return a.begin < b.begin;
        }
    }.lt);
    var total: u64 = 0;
    for (tensors.items, 0..) |t, i| {
        if (i > 0 and tensors.items[i - 1].shard == t.shard and t.begin < tensors.items[i - 1].end) {
            fail(opts, "tensors '{s}' and '{s}' overlap in the file", .{ tensors.items[i - 1].hk_name, t.hk_name });
            return error.BadSafetensors;
        }
        total += t.end - t.begin;
    }

    const specs = try allocator.alloc(stream_writer.TensorSpec, tensors.items.len);
    defer allocator.free(specs);
    for (specs, tensors.items) |*s, t| s.* = .{ .name = t.hk_name, .storage_type = t.dtype, .ndim = t.ndim, .shape = t.shape, .data_size = t.end - t.begin };

    var w = try stream_writer.StreamWriter.begin(allocator, io, out_path, &meta, specs, format.DEFAULT_ALIGNMENT_BYTES);
    defer w.deinit();

    // ---- copy ---------------------------------------------------------------------------
    var done: u64 = 0;
    var idx: usize = 0;
    for (inputs.shards, 0..) |shard, si| {
        var r = try source_mod.BufReader.init(allocator, shard.src, opts.chunk_bytes);
        defer r.deinit();
        try r.skip(headers[si].data_start);
        var rel: u64 = 0;
        while (idx < tensors.items.len and tensors.items[idx].shard == si) : (idx += 1) {
            const t = tensors.items[idx];
            if (t.begin > rel) {
                try r.skip(t.begin - rel);
                rel = t.begin;
            }
            const size = t.end - t.begin;
            var at: u64 = 0;
            while (at < size) {
                const n: usize = @intCast(@min(r.buf.len, size - at));
                try w.write(idx, at, try r.take(n));
                at += n;
                rel += n;
                done += n;
                if (opts.progress) |p| p(opts.progress_ctx, done, total);
            }
        }
    }
    if (opts.before_finish) |hook| try hook(opts.before_finish_ctx);
    try w.finish();
}

fn badKey(opts: Options, name: []const u8) ConvertError {
    fail(opts, "config.json is missing '{s}'", .{name});
    return error.BadConfig;
}

test "tensor names map into the container convention" {
    const a = std.testing.allocator;
    const cases = [_]struct { []const u8, ?[]const u8 }{
        .{ "model.embed_tokens.weight", "token_embd.weight" },
        .{ "model.layers.12.self_attn.q_proj.weight", "blk.12.attn_q.weight" },
        .{ "model.layers.0.self_attn.k_proj.bias", "blk.0.attn_k.bias" },
        .{ "model.layers.3.mlp.down_proj.weight", "blk.3.ffn_down.weight" },
        .{ "model.layers.3.post_attention_layernorm.weight", "blk.3.ffn_norm.weight" },
        .{ "model.layers.3.self_attn.q_norm.weight", "blk.3.attn_q_norm.weight" },
        .{ "lm_head.weight", "output.weight" },
        .{ "model.layers.0.self_attn.rotary_emb.inv_freq", null },
    };
    for (cases) |c| {
        const got = try mapName(a, c[0]);
        defer if (got) |g| a.free(g);
        if (c[1]) |want| {
            try std.testing.expectEqualStrings(want, got.?);
        } else try std.testing.expect(got == null);
    }
    try std.testing.expectError(error.UnknownTensor, mapName(a, "model.layers.0.self_attn.mystery.weight"));
    try std.testing.expectError(error.UnknownTensor, mapName(a, "vision_tower.x"));
}
