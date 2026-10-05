//! Reads a Hugging Face `tokenizer.json` into the arrays the container stores.
//!
//! The file is walked as a token stream, never turned into a tree, and token strings that need
//! no unescaping stay slices of the input, so a 150,000 entry vocabulary costs one pass and a
//! couple of megabytes. Only the constructs that can be reproduced exactly are accepted: byte
//! level BPE with `Split`, `ByteLevel` and `Digits` pre-tokenizers. Anything else (SentencePiece
//! style normalizers, `Metaspace`, merged-with-previous splits) is refused with a message,
//! because converting it approximately would produce a model that tokenizes differently from
//! its reference.

const std = @import("std");
const arrays = @import("../arrays.zig");

pub const gpt2_pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

pub const Error = error{
    InvalidTokenizerJson,
    UnsupportedTokenizer,
    OutOfMemory,
    ArrayTooLarge,
};

pub const Diag = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn set(self: *Diag, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(&self.buf, fmt, args) catch self.buf[0..];
        self.len = s.len;
    }

    pub fn message(self: *const Diag) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const Parsed = struct {
    /// Token strings by id, as a ready to store blob.
    tokens_blob: []u8,
    /// i32 token types by id (1 normal, 3 control, 4 user defined, 5 unused), as a blob.
    types_blob: []u8,
    merges_blob: []u8,
    /// Pre-tokenizer patterns in application order, as a blob.
    patterns_blob: []u8,
    n_tokens: u32,
    ignore_merges: bool,

    pub fn deinit(self: *Parsed, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens_blob);
        allocator.free(self.types_blob);
        allocator.free(self.merges_blob);
        allocator.free(self.patterns_blob);
    }
};

const Entry = struct { id: u32, text: []const u8, special: bool = false, added: bool = false };

const Ctx = struct {
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    sc: *std.json.Scanner,
    diag: *Diag,
    entries: std.ArrayList(Entry) = .empty,
    merges: arrays.StringArrayBuilder,
    patterns: arrays.StringArrayBuilder,
    ignore_merges: bool = false,
    saw_model: bool = false,

    fn next(self: *Ctx) Error!std.json.Token {
        return self.sc.nextAlloc(self.arena, .alloc_if_needed) catch |e| switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            else => {
                self.diag.set("tokenizer.json is not valid JSON", .{});
                return error.InvalidTokenizerJson;
            },
        };
    }

    fn str(self: *Ctx) Error![]const u8 {
        return switch (try self.next()) {
            .string, .allocated_string => |s| s,
            else => {
                self.diag.set("tokenizer.json: expected a string", .{});
                return error.InvalidTokenizerJson;
            },
        };
    }

    fn int(self: *Ctx) Error!i64 {
        return switch (try self.next()) {
            .number, .allocated_number => |s| std.fmt.parseInt(i64, s, 10) catch {
                self.diag.set("tokenizer.json: expected an integer, found '{s}'", .{s});
                return error.InvalidTokenizerJson;
            },
            else => {
                self.diag.set("tokenizer.json: expected an integer", .{});
                return error.InvalidTokenizerJson;
            },
        };
    }

    fn boolean(self: *Ctx) Error!bool {
        return switch (try self.next()) {
            .true => true,
            .false => false,
            else => {
                self.diag.set("tokenizer.json: expected true or false", .{});
                return error.InvalidTokenizerJson;
            },
        };
    }

    fn skip(self: *Ctx) Error!void {
        self.sc.skipValue() catch return error.InvalidTokenizerJson;
    }

    fn isObjectBegin(t: std.json.Token) bool {
        return t == .object_begin;
    }

    fn unsupported(self: *Ctx, comptime fmt: []const u8, args: anytype) Error {
        self.diag.set(fmt, args);
        return error.UnsupportedTokenizer;
    }
};

pub fn parse(allocator: std.mem.Allocator, json_bytes: []const u8, diag: *Diag) Error!Parsed {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    var sc = std.json.Scanner.initCompleteInput(allocator, json_bytes);
    defer sc.deinit();

    var ctx = Ctx{
        .allocator = allocator,
        .arena = arena_state.allocator(),
        .sc = &sc,
        .diag = diag,
        .merges = arrays.StringArrayBuilder.init(allocator),
        .patterns = arrays.StringArrayBuilder.init(allocator),
    };
    defer ctx.entries.deinit(allocator);
    defer ctx.merges.deinit();
    defer ctx.patterns.deinit();

    if ((try ctx.next()) != .object_begin) {
        diag.set("tokenizer.json is not a JSON object", .{});
        return error.InvalidTokenizerJson;
    }
    while (true) {
        const key = switch (try ctx.next()) {
            .object_end => break,
            .string, .allocated_string => |s| s,
            else => return error.InvalidTokenizerJson,
        };
        if (std.mem.eql(u8, key, "model")) {
            try parseModel(&ctx);
        } else if (std.mem.eql(u8, key, "added_tokens")) {
            try parseAdded(&ctx);
        } else if (std.mem.eql(u8, key, "pre_tokenizer")) {
            try parsePre(&ctx);
        } else if (std.mem.eql(u8, key, "normalizer")) {
            try parseNormalizer(&ctx);
        } else {
            try ctx.skip();
        }
    }
    if (!ctx.saw_model) {
        diag.set("tokenizer.json has no 'model' section", .{});
        return error.InvalidTokenizerJson;
    }
    if (ctx.patterns.count() == 0) {
        // No pre-tokenizer means the whole text is one word. That is valid but unusual; keep it
        // explicit so the loader does not fall back to a guessed family.
        try ctx.patterns.append("[\\s\\S]+");
    }

    // Lay entries out by id. Holes become unused placeholder tokens so ids stay aligned with
    // the embedding matrix.
    var max_id: u32 = 0;
    for (ctx.entries.items) |e| max_id = @max(max_id, e.id);
    const n: usize = @as(usize, max_id) + 1;
    const slots = try allocator.alloc(?Entry, n);
    defer allocator.free(slots);
    @memset(slots, null);
    for (ctx.entries.items) |e| {
        // The model vocabulary wins for text; added_tokens only contribute their kind.
        if (slots[e.id]) |old| {
            var merged = old;
            if (e.added) {
                merged.special = e.special;
                merged.added = true;
            }
            slots[e.id] = merged;
        } else {
            slots[e.id] = e;
        }
    }

    var toks = arrays.StringArrayBuilder.init(allocator);
    defer toks.deinit();
    const types_raw = try allocator.alloc(u8, n * 4);
    defer allocator.free(types_raw);
    var pad_buf: [32]u8 = undefined;
    for (slots, 0..) |slot, i| {
        var ty: i32 = 1;
        if (slot) |e| {
            try toks.append(e.text);
            if (e.added) ty = if (e.special) 3 else 4;
        } else {
            try toks.append(std.fmt.bufPrint(&pad_buf, "[UNUSED_{d}]", .{i}) catch unreachable);
            ty = 5;
        }
        std.mem.writeInt(i32, types_raw[i * 4 ..][0..4], ty, .little);
    }

    const tokens_blob = try toks.finish();
    errdefer allocator.free(tokens_blob);
    const types_blob = try arrays.fixedBlob(allocator, .i32, n, types_raw);
    errdefer allocator.free(types_blob);
    const merges_blob = try ctx.merges.finish();
    errdefer allocator.free(merges_blob);
    const patterns_blob = try ctx.patterns.finish();
    return .{
        .tokens_blob = tokens_blob,
        .types_blob = types_blob,
        .merges_blob = merges_blob,
        .patterns_blob = patterns_blob,
        .n_tokens = @intCast(n),
        .ignore_merges = ctx.ignore_merges,
    };
}

fn parseModel(ctx: *Ctx) Error!void {
    if ((try ctx.next()) != .object_begin) return error.InvalidTokenizerJson;
    ctx.saw_model = true;
    var is_bpe = false;
    while (true) {
        const key = switch (try ctx.next()) {
            .object_end => break,
            .string, .allocated_string => |s| s,
            else => return error.InvalidTokenizerJson,
        };
        if (std.mem.eql(u8, key, "type")) {
            const t = try ctx.str();
            if (!std.mem.eql(u8, t, "BPE")) return ctx.unsupported("tokenizer model type '{s}' is not supported (only BPE)", .{t});
            is_bpe = true;
        } else if (std.mem.eql(u8, key, "vocab")) {
            if ((try ctx.next()) != .object_begin) return error.InvalidTokenizerJson;
            while (true) {
                const tok = switch (try ctx.next()) {
                    .object_end => break,
                    .string, .allocated_string => |s| s,
                    else => return error.InvalidTokenizerJson,
                };
                const id = try ctx.int();
                if (id < 0 or id > 1 << 24) {
                    ctx.diag.set("tokenizer.json: token id {d} is out of range", .{id});
                    return error.InvalidTokenizerJson;
                }
                try ctx.entries.append(ctx.allocator, .{ .id = @intCast(id), .text = tok });
            }
        } else if (std.mem.eql(u8, key, "merges")) {
            if ((try ctx.next()) != .array_begin) return error.InvalidTokenizerJson;
            var joined: std.ArrayList(u8) = .empty;
            defer joined.deinit(ctx.allocator);
            while (true) {
                const t = try ctx.next();
                switch (t) {
                    .array_end => break,
                    .string, .allocated_string => |s| try ctx.merges.append(s),
                    .array_begin => {
                        // Newer files store each merge as a two element array.
                        const a = try ctx.str();
                        joined.clearRetainingCapacity();
                        try joined.appendSlice(ctx.allocator, a);
                        try joined.append(ctx.allocator, ' ');
                        try joined.appendSlice(ctx.allocator, try ctx.str());
                        if ((try ctx.next()) != .array_end) return error.InvalidTokenizerJson;
                        try ctx.merges.append(joined.items);
                    },
                    else => return error.InvalidTokenizerJson,
                }
            }
        } else if (std.mem.eql(u8, key, "ignore_merges")) {
            ctx.ignore_merges = try ctx.boolean();
        } else if (std.mem.eql(u8, key, "byte_fallback")) {
            if (try ctx.boolean()) return ctx.unsupported("this tokenizer uses SentencePiece style byte fallback, which the safetensors importer does not support yet; use a GGUF file for this model", .{});
        } else if (std.mem.eql(u8, key, "continuing_subword_prefix") or std.mem.eql(u8, key, "end_of_word_suffix") or std.mem.eql(u8, key, "unk_token")) {
            // Must be absent for byte level BPE. A null or an empty string both mean "none".
            const t = try ctx.next();
            const empty = switch (t) {
                .null => true,
                .string, .allocated_string => |str| str.len == 0,
                else => false,
            };
            if (!empty) {
                if (std.mem.eql(u8, key, "unk_token")) continue; // an unk token name is harmless
                return ctx.unsupported("tokenizer option '{s}' is not supported", .{key});
            }
        } else {
            try ctx.skip();
        }
    }
    if (!is_bpe) return ctx.unsupported("tokenizer.json model has no type", .{});
}

fn parseAdded(ctx: *Ctx) Error!void {
    if ((try ctx.next()) != .array_begin) return error.InvalidTokenizerJson;
    while (true) {
        switch (try ctx.next()) {
            .array_end => break,
            .object_begin => {},
            else => return error.InvalidTokenizerJson,
        }
        var id: ?i64 = null;
        var content: ?[]const u8 = null;
        var special = false;
        while (true) {
            const key = switch (try ctx.next()) {
                .object_end => break,
                .string, .allocated_string => |s| s,
                else => return error.InvalidTokenizerJson,
            };
            if (std.mem.eql(u8, key, "id")) {
                id = try ctx.int();
            } else if (std.mem.eql(u8, key, "content")) {
                content = try ctx.str();
            } else if (std.mem.eql(u8, key, "special")) {
                special = try ctx.boolean();
            } else {
                try ctx.skip();
            }
        }
        const i = id orelse return error.InvalidTokenizerJson;
        const c = content orelse return error.InvalidTokenizerJson;
        if (i < 0 or i > 1 << 24) return error.InvalidTokenizerJson;
        try ctx.entries.append(ctx.allocator, .{ .id = @intCast(i), .text = c, .special = special, .added = true });
    }
}

fn parseNormalizer(ctx: *Ctx) Error!void {
    const t = try ctx.next();
    switch (t) {
        .null => return,
        .object_begin => {},
        else => return error.InvalidTokenizerJson,
    }
    // Byte level BPE tokenizers use no normalizer or NFC. NFC only changes text that is not
    // already normalized, which llama.cpp also leaves alone, so it is accepted.
    var ok = false;
    while (true) {
        const key = switch (try ctx.next()) {
            .object_end => break,
            .string, .allocated_string => |s| s,
            else => return error.InvalidTokenizerJson,
        };
        if (std.mem.eql(u8, key, "type")) {
            const ty = try ctx.str();
            if (std.mem.eql(u8, ty, "NFC")) {
                ok = true;
            } else {
                return ctx.unsupported("normalizer '{s}' is not supported", .{ty});
            }
        } else {
            try ctx.skip();
        }
    }
    if (!ok) return ctx.unsupported("this normalizer is not supported", .{});
}

fn parsePre(ctx: *Ctx) Error!void {
    const t = try ctx.next();
    switch (t) {
        .null => return,
        .object_begin => try parsePreObject(ctx),
        else => return error.InvalidTokenizerJson,
    }
}

/// Parses one pre-tokenizer object whose `{` has been consumed.
fn parsePreObject(ctx: *Ctx) Error!void {
    var kind: []const u8 = "";
    var regex: ?[]const u8 = null;
    var behavior: []const u8 = "Isolated";
    var invert = false;
    var use_regex = true;
    var add_prefix_space = false;
    var individual_digits = false;
    var string_pattern: ?[]const u8 = null;

    while (true) {
        const key = switch (try ctx.next()) {
            .object_end => break,
            .string, .allocated_string => |s| s,
            else => return error.InvalidTokenizerJson,
        };
        if (std.mem.eql(u8, key, "type")) {
            kind = try ctx.str();
        } else if (std.mem.eql(u8, key, "pretokenizers")) {
            if ((try ctx.next()) != .array_begin) return error.InvalidTokenizerJson;
            while (true) {
                switch (try ctx.next()) {
                    .array_end => break,
                    .object_begin => try parsePreObject(ctx),
                    else => return error.InvalidTokenizerJson,
                }
            }
        } else if (std.mem.eql(u8, key, "pattern")) {
            if ((try ctx.next()) != .object_begin) return error.InvalidTokenizerJson;
            while (true) {
                const pk = switch (try ctx.next()) {
                    .object_end => break,
                    .string, .allocated_string => |s| s,
                    else => return error.InvalidTokenizerJson,
                };
                if (std.mem.eql(u8, pk, "Regex")) regex = try ctx.str() else if (std.mem.eql(u8, pk, "String")) string_pattern = try ctx.str() else try ctx.skip();
            }
        } else if (std.mem.eql(u8, key, "behavior")) {
            behavior = try ctx.str();
        } else if (std.mem.eql(u8, key, "invert")) {
            invert = try ctx.boolean();
        } else if (std.mem.eql(u8, key, "use_regex")) {
            use_regex = try ctx.boolean();
        } else if (std.mem.eql(u8, key, "add_prefix_space")) {
            add_prefix_space = try ctx.boolean();
        } else if (std.mem.eql(u8, key, "individual_digits")) {
            individual_digits = try ctx.boolean();
        } else {
            try ctx.skip();
        }
    }

    if (std.mem.eql(u8, kind, "Sequence")) {
        // children were appended in order while they were parsed
    } else if (std.mem.eql(u8, kind, "Split")) {
        if (!std.mem.eql(u8, behavior, "Isolated") or invert) {
            return ctx.unsupported("pre-tokenizer Split with behavior '{s}' is not supported", .{behavior});
        }
        if (regex) |r| {
            try ctx.patterns.append(r);
        } else if (string_pattern) |sp| {
            var esc: std.ArrayList(u8) = .empty;
            defer esc.deinit(ctx.allocator);
            for (sp) |c| {
                if (std.mem.indexOfScalar(u8, "\\.+*?()|[]{}^$", c) != null) try esc.append(ctx.allocator, '\\');
                try esc.append(ctx.allocator, c);
            }
            try ctx.patterns.append(esc.items);
        } else return error.InvalidTokenizerJson;
    } else if (std.mem.eql(u8, kind, "ByteLevel")) {
        if (add_prefix_space) return ctx.unsupported("ByteLevel with add_prefix_space is not supported", .{});
        if (use_regex) try ctx.patterns.append(gpt2_pattern);
    } else if (std.mem.eql(u8, kind, "Digits")) {
        if (individual_digits) try ctx.patterns.append("[0-9]");
    } else {
        return ctx.unsupported("pre-tokenizer '{s}' is not supported", .{kind});
    }
}

fn expectBlob(blob: []const u8, want: []const []const u8) !void {
    const v = try arrays.StringArray.init(blob);
    try std.testing.expectEqual(@as(u32, @intCast(want.len)), v.count);
    for (want, 0..) |w, i| try std.testing.expectEqualStrings(w, v.get(i));
}

test "parses a byte level BPE tokenizer.json" {
    const a = std.testing.allocator;
    const json =
        \\{"version":"1.0","added_tokens":[{"id":3,"content":"<|end|>","special":true},{"id":4,"content":"<tag>","special":false}],
        \\ "normalizer":{"type":"NFC"},
        \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[
        \\   {"type":"Split","pattern":{"Regex":"\\p{N}"},"behavior":"Isolated","invert":false},
        \\   {"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]},
        \\ "model":{"type":"BPE","ignore_merges":true,"vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}}
    ;
    var diag = Diag{};
    var p = try parse(a, json, &diag);
    defer p.deinit(a);
    try std.testing.expectEqual(@as(u32, 5), p.n_tokens);
    try expectBlob(p.tokens_blob, &.{ "a", "b", "ab", "<|end|>", "<tag>" });
    try expectBlob(p.merges_blob, &.{"a b"});
    try expectBlob(p.patterns_blob, &.{"\\p{N}"});
    try std.testing.expect(p.ignore_merges);
    const types = try arrays.I32Array.init(p.types_blob);
    try std.testing.expectEqual(@as(i32, 1), types.get(0));
    try std.testing.expectEqual(@as(i32, 3), types.get(3));
    try std.testing.expectEqual(@as(i32, 4), types.get(4));
}

test "holes in the id space become unused placeholders" {
    const a = std.testing.allocator;
    const json =
        \\{"model":{"type":"BPE","vocab":{"a":0,"b":3},"merges":[["a","b"]]},"pre_tokenizer":null}
    ;
    var diag = Diag{};
    var p = try parse(a, json, &diag);
    defer p.deinit(a);
    try std.testing.expectEqual(@as(u32, 4), p.n_tokens);
    const toks = try arrays.StringArray.init(p.tokens_blob);
    try std.testing.expectEqualStrings("[UNUSED_1]", toks.get(1));
    try expectBlob(p.merges_blob, &.{"a b"});
}

test "empty subword prefix and suffix are the same as none" {
    const a = std.testing.allocator;
    var diag = Diag{};
    var p = try parse(a,
        \\{"model":{"type":"BPE","continuing_subword_prefix":"","end_of_word_suffix":"","unk_token":null,"vocab":{"a":0},"merges":[]}}
    , &diag);
    p.deinit(a);
    try std.testing.expectError(error.UnsupportedTokenizer, parse(a,
        \\{"model":{"type":"BPE","continuing_subword_prefix":"##","vocab":{},"merges":[]}}
    , &diag));
}

test "unsupported tokenizers are refused with a reason" {
    const a = std.testing.allocator;
    var diag = Diag{};
    try std.testing.expectError(error.UnsupportedTokenizer, parse(a,
        \\{"model":{"type":"Unigram","vocab":[]}}
    , &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "Unigram") != null);
    try std.testing.expectError(error.UnsupportedTokenizer, parse(a,
        \\{"model":{"type":"BPE","vocab":{},"merges":[],"byte_fallback":true}}
    , &diag));
    try std.testing.expectError(error.UnsupportedTokenizer, parse(a,
        \\{"pre_tokenizer":{"type":"Metaspace"},"model":{"type":"BPE","vocab":{},"merges":[]}}
    , &diag));
    try std.testing.expectError(error.InvalidTokenizerJson, parse(a, "not json", &diag));
}
