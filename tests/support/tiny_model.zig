//! Builds a tiny but complete llama style GGUF for tests: a real byte level BPE vocabulary with
//! a few merges, optional chat template, random weights from a fixed seed.

const std = @import("std");

pub const chatml =
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}" ++
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

pub const Options = struct {
    n_layers: usize = 2,
    dim: usize = 64,
    n_heads: usize = 4,
    n_kv: usize = 2,
    ffn: usize = 128,
    seed: u64 = 7,
    chat_template: ?[]const u8 = null,
    /// Store the weight matrices as Q8_0 instead of f32.
    q8_0: bool = false,
};

/// The GPT-2 byte to unicode table, so the vocabulary is a valid byte level BPE one.
fn byteTokens(allocator: std.mem.Allocator, out: *std.ArrayList([]const u8)) !void {
    var bs: [256]u16 = undefined;
    var cs: [256]u16 = undefined;
    var n: usize = 0;
    const ranges = [_][2]u16{ .{ 33, 126 }, .{ 161, 172 }, .{ 174, 255 } };
    for (ranges) |r| {
        var b = r[0];
        while (b <= r[1]) : (b += 1) {
            bs[n] = b;
            cs[n] = b;
            n += 1;
        }
    }
    var extra: u16 = 0;
    for (0..256) |b| {
        var found = false;
        for (bs[0..n]) |x| {
            if (x == b) found = true;
        }
        if (!found) {
            bs[n] = @intCast(b);
            cs[n] = 256 + extra;
            extra += 1;
            n += 1;
        }
    }
    var map: [256]u16 = undefined;
    for (0..n) |i| map[bs[i]] = cs[i];
    for (0..256) |b| {
        var buf: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(map[b], &buf);
        try out.append(allocator, try allocator.dupe(u8, buf[0..len]));
    }
}

const Writer = struct {
    list: std.ArrayList(u8) = .empty,
    allocator: std.mem.Allocator,

    fn int(self: *Writer, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try self.list.appendSlice(self.allocator, &b);
    }
    fn str(self: *Writer, s: []const u8) !void {
        try self.int(u64, s.len);
        try self.list.appendSlice(self.allocator, s);
    }
    fn f32v(self: *Writer, v: f32) !void {
        try self.int(u32, @bitCast(v));
    }
};

const T_U32 = 4;
const T_I32 = 5;
const T_F32 = 6;
const T_STR = 8;
const T_ARR = 9;

/// Writes the GGUF file at `path` (relative to the current directory) and returns nothing.
pub fn makeTinyGguf(allocator: std.mem.Allocator, io: std.Io, path: []const u8, o: Options) !void {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var toks: std.ArrayList([]const u8) = .empty;
    try byteTokens(arena, &toks);
    for ([_][]const u8{ "he", "ll", "hell", "hello", "<|im_start|>", "<|im_end|>", "<|eot|>" }) |t| try toks.append(arena, t);
    const merges = [_][]const u8{ "h e", "l l", "he ll", "hell o" };
    const vocab = toks.items.len;
    const hd = o.dim / o.n_heads;

    var kv = Writer{ .allocator = arena };
    var n_kv: u64 = 0;
    const kvu32 = struct {
        fn put(w: *Writer, n: *u64, key: []const u8, v: u32) !void {
            try w.str(key);
            try w.int(u32, T_U32);
            try w.int(u32, v);
            n.* += 1;
        }
        fn putF(w: *Writer, n: *u64, key: []const u8, v: f32) !void {
            try w.str(key);
            try w.int(u32, T_F32);
            try w.f32v(v);
            n.* += 1;
        }
        fn putS(w: *Writer, n: *u64, key: []const u8, v: []const u8) !void {
            try w.str(key);
            try w.int(u32, T_STR);
            try w.str(v);
            n.* += 1;
        }
    };
    try kvu32.putS(&kv, &n_kv, "general.architecture", "llama");
    try kvu32.put(&kv, &n_kv, "llama.block_count", @intCast(o.n_layers));
    try kvu32.put(&kv, &n_kv, "llama.context_length", 256);
    try kvu32.put(&kv, &n_kv, "llama.embedding_length", @intCast(o.dim));
    try kvu32.put(&kv, &n_kv, "llama.feed_forward_length", @intCast(o.ffn));
    try kvu32.put(&kv, &n_kv, "llama.attention.head_count", @intCast(o.n_heads));
    try kvu32.put(&kv, &n_kv, "llama.attention.head_count_kv", @intCast(o.n_kv));
    try kvu32.putF(&kv, &n_kv, "llama.attention.layer_norm_rms_epsilon", 1e-5);
    try kvu32.putF(&kv, &n_kv, "llama.rope.freq_base", 10000.0);
    try kvu32.put(&kv, &n_kv, "llama.rope.dimension_count", @intCast(hd));
    try kvu32.putS(&kv, &n_kv, "tokenizer.ggml.model", "gpt2");
    try kvu32.putS(&kv, &n_kv, "tokenizer.ggml.pre", "gpt-2");
    try kv.str("tokenizer.ggml.tokens");
    try kv.int(u32, T_ARR);
    try kv.int(u32, T_STR);
    try kv.int(u64, toks.items.len);
    for (toks.items) |t| try kv.str(t);
    n_kv += 1;
    try kv.str("tokenizer.ggml.merges");
    try kv.int(u32, T_ARR);
    try kv.int(u32, T_STR);
    try kv.int(u64, merges.len);
    for (merges) |m| try kv.str(m);
    n_kv += 1;
    try kv.str("tokenizer.ggml.token_type");
    try kv.int(u32, T_ARR);
    try kv.int(u32, T_I32);
    try kv.int(u64, vocab);
    for (0..vocab) |i| try kv.int(i32, if (i + 3 >= vocab) 3 else 1);
    n_kv += 1;
    if (o.chat_template) |ct| try kvu32.putS(&kv, &n_kv, "tokenizer.chat_template", ct);
    try kvu32.put(&kv, &n_kv, "tokenizer.ggml.eos_token_id", @intCast(vocab - 1));
    try kvu32.put(&kv, &n_kv, "tokenizer.ggml.bos_token_id", @intCast(vocab - 1));

    // Tensors: name, shape, and data generated in order.
    const Tensor = struct { name: []const u8, rows: usize, cols: usize, is_vec: bool };
    var tensors: std.ArrayList(Tensor) = .empty;
    try tensors.append(arena, .{ .name = "token_embd.weight", .rows = vocab, .cols = o.dim, .is_vec = false });
    try tensors.append(arena, .{ .name = "output_norm.weight", .rows = 1, .cols = o.dim, .is_vec = true });
    try tensors.append(arena, .{ .name = "output.weight", .rows = vocab, .cols = o.dim, .is_vec = false });
    for (0..o.n_layers) |i| {
        const nm = struct {
            fn f(a: std.mem.Allocator, layer: usize, suffix: []const u8) []const u8 {
                return std.fmt.allocPrint(a, "blk.{d}.{s}", .{ layer, suffix }) catch unreachable;
            }
        }.f;
        try tensors.append(arena, .{ .name = nm(arena, i, "attn_norm.weight"), .rows = 1, .cols = o.dim, .is_vec = true });
        try tensors.append(arena, .{ .name = nm(arena, i, "attn_q.weight"), .rows = o.n_heads * hd, .cols = o.dim, .is_vec = false });
        try tensors.append(arena, .{ .name = nm(arena, i, "attn_k.weight"), .rows = o.n_kv * hd, .cols = o.dim, .is_vec = false });
        try tensors.append(arena, .{ .name = nm(arena, i, "attn_v.weight"), .rows = o.n_kv * hd, .cols = o.dim, .is_vec = false });
        try tensors.append(arena, .{ .name = nm(arena, i, "attn_output.weight"), .rows = o.dim, .cols = o.n_heads * hd, .is_vec = false });
        try tensors.append(arena, .{ .name = nm(arena, i, "ffn_norm.weight"), .rows = 1, .cols = o.dim, .is_vec = true });
        try tensors.append(arena, .{ .name = nm(arena, i, "ffn_gate.weight"), .rows = o.ffn, .cols = o.dim, .is_vec = false });
        try tensors.append(arena, .{ .name = nm(arena, i, "ffn_up.weight"), .rows = o.ffn, .cols = o.dim, .is_vec = false });
        try tensors.append(arena, .{ .name = nm(arena, i, "ffn_down.weight"), .rows = o.dim, .cols = o.ffn, .is_vec = false });
    }

    var infos = Writer{ .allocator = arena };
    var data = Writer{ .allocator = arena };
    var prng = std.Random.DefaultPrng.init(o.seed);
    const rand = prng.random();
    for (tensors.items) |t| {
        try infos.str(t.name);
        if (t.is_vec) {
            try infos.int(u32, 1);
            try infos.int(u64, t.cols);
        } else {
            try infos.int(u32, 2);
            // GGUF lists the fastest varying dimension first.
            try infos.int(u64, t.cols);
            try infos.int(u64, t.rows);
        }
        const quant = o.q8_0 and !t.is_vec;
        try infos.int(u32, if (quant) 8 else 0); // GGML type: 0 = f32, 8 = Q8_0
        try infos.int(u64, data.list.items.len);
        const n = t.rows * t.cols;
        if (quant) {
            // Q8_0: blocks of 32 weights, an f16 scale and 32 signed bytes.
            var vals: [32]f32 = undefined;
            var b: usize = 0;
            while (b < n / 32) : (b += 1) {
                var amax: f32 = 0;
                for (&vals) |*x| {
                    x.* = rand.floatNorm(f32) * 0.08;
                    amax = @max(amax, @abs(x.*));
                }
                const d: f32 = amax / 127.0;
                const id: f32 = if (d != 0) 1.0 / d else 0.0;
                try data.int(u16, @bitCast(@as(f16, @floatCast(d))));
                for (vals) |x| try data.int(i8, @intFromFloat(@round(x * id)));
            }
        } else for (0..n) |_| {
            const v: f32 = if (t.is_vec) 0.5 + rand.float(f32) else rand.floatNorm(f32) * 0.08;
            try data.f32v(v);
        }
        while (data.list.items.len % 32 != 0) try data.list.append(arena, 0);
    }

    var file = Writer{ .allocator = arena };
    try file.list.appendSlice(arena, "GGUF");
    try file.int(u32, 3);
    try file.int(u64, tensors.items.len);
    try file.int(u64, n_kv);
    try file.list.appendSlice(arena, kv.list.items);
    try file.list.appendSlice(arena, infos.list.items);
    while (file.list.items.len % 32 != 0) try file.list.append(arena, 0);
    try file.list.appendSlice(arena, data.list.items);

    var f = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, file.list.items);
}


/// A tiny llama style Hugging Face repository in `dir`: config, tokenizer.json and safetensors
/// shards (with an index when there is more than one), laid out like a large repository.
pub fn makeTinyHf(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, shards: usize, o: Options) !void {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try std.Io.Dir.cwd().createDirPath(io, dir);

    var toks: std.ArrayList([]const u8) = .empty;
    try byteTokens(arena, &toks);
    for ([_][]const u8{ "he", "ll", "hell", "hello", "<|im_start|>", "<|im_end|>", "<|eot|>" }) |t| try toks.append(arena, t);
    const vocab = toks.items.len;
    const hd = o.dim / o.n_heads;

    const write = struct {
        fn file(io_: std.Io, a: std.mem.Allocator, d: []const u8, name: []const u8, bytes: []const u8) !void {
            const p = try std.fmt.allocPrint(a, "{s}/{s}", .{ d, name });
            var f = try std.Io.Dir.cwd().createFile(io_, p, .{});
            defer f.close(io_);
            try f.writeStreamingAll(io_, bytes);
        }
    };

    try write.file(io, arena, dir, "config.json", try std.fmt.allocPrint(arena,
        \\{{"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":{d},"intermediate_size":{d},"num_hidden_layers":{d},"num_attention_heads":{d},"num_key_value_heads":{d},"vocab_size":{d},"max_position_embeddings":256,"rms_norm_eps":1e-5,"rope_theta":10000.0,"bos_token_id":{d},"eos_token_id":{d},"tie_word_embeddings":false}}
    , .{ o.dim, o.ffn, o.n_layers, o.n_heads, o.n_kv, vocab, vocab - 1, vocab - 1 }));

    var tj: std.Io.Writer.Allocating = .init(arena);
    const w = &tj.writer;
    try w.writeAll("{\"model\":{\"type\":\"BPE\",\"vocab\":{");
    for (toks.items, 0..) |t, i| {
        if (i != 0) try w.writeAll(",");
        try w.print("{f}:{d}", .{ std.json.fmt(t, .{}), i });
    }
    try w.writeAll("},\"merges\":[\"h e\",\"l l\",\"he ll\",\"hell o\"]},\"added_tokens\":[");
    for (toks.items[vocab - 3 ..], 0..) |t, i| {
        if (i != 0) try w.writeAll(",");
        try w.print("{{\"id\":{d},\"content\":{f},\"special\":true}}", .{ vocab - 3 + i, std.json.fmt(t, .{}) });
    }
    try w.writeAll("],\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":true}}");
    try write.file(io, arena, dir, "tokenizer.json", tj.written());
    try write.file(io, arena, dir, "tokenizer_config.json", try std.fmt.allocPrint(arena, "{{\"chat_template\":{f},\"eos_token\":\"<|eot|>\",\"bos_token\":\"<|eot|>\"}}", .{std.json.fmt(chatml, .{})}));

    // Tensors, in the sorted name order the shards are dealt from.
    const Tensor = struct { name: []const u8, rows: usize, cols: usize, is_vec: bool };
    var tensors: std.ArrayList(Tensor) = .empty;
    try tensors.append(arena, .{ .name = "model.embed_tokens.weight", .rows = vocab, .cols = o.dim, .is_vec = false });
    try tensors.append(arena, .{ .name = "model.norm.weight", .rows = 1, .cols = o.dim, .is_vec = true });
    try tensors.append(arena, .{ .name = "lm_head.weight", .rows = vocab, .cols = o.dim, .is_vec = false });
    for (0..o.n_layers) |i| {
        const P = struct {
            fn n(a: std.mem.Allocator, layer: usize, suffix: []const u8) []const u8 {
                return std.fmt.allocPrint(a, "model.layers.{d}.{s}", .{ layer, suffix }) catch unreachable;
            }
        };
        try tensors.append(arena, .{ .name = P.n(arena, i, "input_layernorm.weight"), .rows = 1, .cols = o.dim, .is_vec = true });
        try tensors.append(arena, .{ .name = P.n(arena, i, "self_attn.q_proj.weight"), .rows = o.n_heads * hd, .cols = o.dim, .is_vec = false });
        try tensors.append(arena, .{ .name = P.n(arena, i, "self_attn.k_proj.weight"), .rows = o.n_kv * hd, .cols = o.dim, .is_vec = false });
        try tensors.append(arena, .{ .name = P.n(arena, i, "self_attn.v_proj.weight"), .rows = o.n_kv * hd, .cols = o.dim, .is_vec = false });
        try tensors.append(arena, .{ .name = P.n(arena, i, "self_attn.o_proj.weight"), .rows = o.dim, .cols = o.n_heads * hd, .is_vec = false });
        try tensors.append(arena, .{ .name = P.n(arena, i, "post_attention_layernorm.weight"), .rows = 1, .cols = o.dim, .is_vec = true });
        try tensors.append(arena, .{ .name = P.n(arena, i, "mlp.gate_proj.weight"), .rows = o.ffn, .cols = o.dim, .is_vec = false });
        try tensors.append(arena, .{ .name = P.n(arena, i, "mlp.up_proj.weight"), .rows = o.ffn, .cols = o.dim, .is_vec = false });
        try tensors.append(arena, .{ .name = P.n(arena, i, "mlp.down_proj.weight"), .rows = o.dim, .cols = o.ffn, .is_vec = false });
    }
    std.mem.sort(Tensor, tensors.items, {}, struct {
        fn lt(_: void, a: Tensor, b: Tensor) bool {
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.lt);

    var prng = std.Random.DefaultPrng.init(o.seed);
    const rand = prng.random();
    var weight_map: std.Io.Writer.Allocating = .init(arena);
    try weight_map.writer.writeAll("{\"metadata\":{},\"weight_map\":{");
    var first_entry = true;
    for (0..shards) |s| {
        var header: std.Io.Writer.Allocating = .init(arena);
        var data: std.ArrayList(u8) = .empty;
        try header.writer.writeAll("{");
        var first = true;
        var i = s;
        const fname = if (shards == 1) try arena.dupe(u8, "model.safetensors") else try std.fmt.allocPrint(arena, "model-{d:0>5}-of-{d:0>5}.safetensors", .{ s + 1, shards });
        while (i < tensors.items.len) : (i += shards) {
            const t = tensors.items[i];
            const start = data.items.len;
            for (0..t.rows * t.cols) |_| {
                const v: f32 = if (t.is_vec) 0.5 + rand.float(f32) else rand.floatNorm(f32) * 0.08;
                var b: [4]u8 = undefined;
                std.mem.writeInt(u32, &b, @bitCast(v), .little);
                try data.appendSlice(arena, &b);
            }
            if (!first) try header.writer.writeAll(",");
            first = false;
            if (t.is_vec) {
                try header.writer.print("{f}:{{\"dtype\":\"F32\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ std.json.fmt(t.name, .{}), t.cols, start, data.items.len });
            } else {
                try header.writer.print("{f}:{{\"dtype\":\"F32\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ std.json.fmt(t.name, .{}), t.rows, t.cols, start, data.items.len });
            }
            if (!first_entry) try weight_map.writer.writeAll(",");
            first_entry = false;
            try weight_map.writer.print("{f}:{f}", .{ std.json.fmt(t.name, .{}), std.json.fmt(fname, .{}) });
        }
        try header.writer.writeAll("}");
        while (header.written().len % 8 != 0) try header.writer.writeAll(" ");
        var file: std.ArrayList(u8) = .empty;
        var len_bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &len_bytes, header.written().len, .little);
        try file.appendSlice(arena, &len_bytes);
        try file.appendSlice(arena, header.written());
        try file.appendSlice(arena, data.items);
        try write.file(io, arena, dir, fname, file.items);
    }
    if (shards > 1) {
        try weight_map.writer.writeAll("}}");
        try write.file(io, arena, dir, "model.safetensors.index.json", weight_map.written());
    }
}
