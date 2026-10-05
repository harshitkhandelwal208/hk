//! Tokenizers: byte level BPE (GPT-2 family, Llama 3, Qwen, ...) and SentencePiece style
//! (Llama 2, Mistral 7B, Gemma). Token ids match llama.cpp and Hugging Face for the same
//! vocabulary, which the parity tests check against both.
//!
//! The vocabulary is read in place from the container's metadata blobs. The only memory this
//! module owns is the hash index over those strings and the merge table.

const std = @import("std");
const metadata = @import("metadata.zig");
const arrays = @import("arrays.zig");
const regex = @import("tokenizer/regex.zig");
const pretok = @import("tokenizer/pretok.zig");
const uni = @import("tokenizer/unicode.zig");

pub const TokenType = enum(i32) {
    undefined = 0,
    normal = 1,
    unknown = 2,
    control = 3,
    user_defined = 4,
    unused = 5,
    byte = 6,
    _,
};

pub const Kind = enum { bpe, spm };

pub const Error = error{
    MissingVocabulary,
    UnsupportedTokenizer,
    UnsupportedPretokenizer,
    BadVocabulary,
    BadPattern,
    OutOfMemory,
};

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

pub const EncodeOptions = struct {
    /// Add BOS/EOS when the model asks for them.
    add_special: bool = true,
    /// Recognise control tokens such as `<|im_start|>` in the text and emit them as single ids.
    parse_special: bool = false,
};

const null_id: u32 = std.math.maxInt(u32);

// GPT-2 byte to unicode mapping: printable bytes map to themselves, the rest are shifted to
// code points from 256 up so every byte has a visible, unambiguous character.
const byte_cp: [256]u16 = blk: {
    var t: [256]u16 = undefined;
    var n: u16 = 0;
    for (0..256) |b| {
        const printable = (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174);
        if (printable) {
            t[b] = @intCast(b);
        } else {
            t[b] = 256 + n;
            n += 1;
        }
    }
    break :blk t;
};

/// Reverse of `byte_cp` for code points below 512, 0xFFFF where a code point is not a mapped byte.
const cp_byte: [512]u16 = blk: {
    var t: [512]u16 = @splat(0xFFFF);
    for (0..256) |b| t[byte_cp[b]] = @intCast(b);
    break :blk t;
};

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    kind: Kind,
    tokens: arrays.StringArray,
    types: ?arrays.I32Array,
    scores: ?arrays.F32Array,

    /// Open addressing hash of token strings: slot holds id + 1, 0 is empty.
    index: []u32,
    index_mask: usize,

    // BPE
    merge_keys: []u64 = &.{},
    merge_vals: []u64 = &.{}, // rank << 32 | merged id
    merge_mask: usize = 0,
    byte_tok: [256]u32 = @splat(null_id),
    pre: ?pretok.PreTokenizer = null,
    ignore_merges: bool = false,

    // SPM
    byte_fallback: [256]u32 = @splat(null_id),
    add_space_prefix: bool = false,

    bos: ?u32 = null,
    eos: ?u32 = null,
    unk: ?u32 = null,
    pad: ?u32 = null,
    /// Ids that end generation, when the container lists them.
    eog: ?arrays.I32Array = null,
    /// Arrays converted from JSON metadata, which the views above point into.
    json_blobs: std.ArrayList([]u8) = .empty,
    add_bos: bool = false,
    add_eos: bool = false,
    chat_template: ?[]const u8 = null,

    /// Special tokens grouped by first byte, longest first, for text partitioning.
    special_by_first: [256][]const u32 = @splat(&.{}),
    special_store: []u32 = &.{},

    // Reusable scratch so encoding does not allocate per call.
    cps: std.ArrayList(u32) = .empty,
    offs: std.ArrayList(u32) = .empty,
    segs: std.ArrayList(pretok.PreTokenizer.Segment) = .empty,
    segs_tmp: std.ArrayList(pretok.PreTokenizer.Segment) = .empty,
    rscratch: regex.Scratch,
    work: Work = .{},

    pub fn count(self: *const Tokenizer) usize {
        return self.tokens.count;
    }

    pub fn piece(self: *const Tokenizer, id: u32) []const u8 {
        return self.tokens.get(id);
    }

    pub fn tokenType(self: *const Tokenizer, id: u32) TokenType {
        if (self.types) |t| return @enumFromInt(t.get(id));
        return .normal;
    }

    fn hashOf(s: []const u8) u64 {
        return std.hash.Wyhash.hash(0, s);
    }

    pub fn find(self: *const Tokenizer, s: []const u8) ?u32 {
        var i: usize = @intCast(hashOf(s) & self.index_mask);
        while (true) {
            const e = self.index[i];
            if (e == 0) return null;
            if (std.mem.eql(u8, self.tokens.get(e - 1), s)) return e - 1;
            i = (i + 1) & self.index_mask;
        }
    }

    fn insert(self: *Tokenizer, id: u32) void {
        var i: usize = @intCast(hashOf(self.tokens.get(id)) & self.index_mask);
        while (self.index[i] != 0) i = (i + 1) & self.index_mask;
        self.index[i] = id + 1;
    }

    fn mergeSlot(self: *const Tokenizer, key: u64) usize {
        const h: u64 = (key *% 0x9E3779B97F4A7C15) >> 20;
        return @as(usize, @intCast(h)) & self.merge_mask;
    }

    fn mergeLookup(self: *const Tokenizer, l: u32, r: u32) ?u64 {
        const key = (@as(u64, l) << 32) | r | (1 << 63);
        var i = self.mergeSlot(key);
        while (true) {
            const k = self.merge_keys[i];
            if (k == 0) return null;
            if (k == key) return self.merge_vals[i];
            i = (i + 1) & self.merge_mask;
        }
    }

    fn mergeInsert(self: *Tokenizer, l: u32, r: u32, val: u64) void {
        const key = (@as(u64, l) << 32) | r | (1 << 63);
        var i = self.mergeSlot(key);
        while (self.merge_keys[i] != 0) {
            if (self.merge_keys[i] == key) return; // keep the earliest (lowest rank) merge
            i = (i + 1) & self.merge_mask;
        }
        self.merge_keys[i] = key;
        self.merge_vals[i] = val;
    }

    pub fn deinit(self: *Tokenizer) void {
        const a = self.allocator;
        a.free(self.index);
        if (self.merge_keys.len != 0) {
            a.free(self.merge_keys);
            a.free(self.merge_vals);
        }
        if (self.pre) |*p| p.deinit();
        for (self.json_blobs.items) |b| a.free(b);
        self.json_blobs.deinit(a);
        if (self.special_store.len != 0) a.free(self.special_store);
        self.cps.deinit(a);
        self.offs.deinit(a);
        self.segs.deinit(a);
        self.segs_tmp.deinit(a);
        self.rscratch.deinit();
        self.work.deinit(a);
    }

    /// Builds a tokenizer from container metadata. The returned value borrows the metadata
    /// blobs, so the container must outlive it.
    pub fn fromMetadata(allocator: std.mem.Allocator, meta: *const metadata.MetadataMap, diag: *Diag) Error!Tokenizer {
        const model = meta.getString("tokenizer.ggml.model") orelse {
            diag.set("model has no 'tokenizer.ggml.model'", .{});
            return error.MissingVocabulary;
        };
        const kind: Kind = if (std.mem.eql(u8, model, "gpt2"))
            .bpe
        else if (std.mem.eql(u8, model, "llama"))
            .spm
        else {
            diag.set("tokenizer model '{s}' is not supported (supported: gpt2, llama)", .{model});
            return error.UnsupportedTokenizer;
        };

        var owned: std.ArrayList([]u8) = .empty;
        var owned_moved = false;
        defer if (!owned_moved) {
            for (owned.items) |b| allocator.free(b);
            owned.deinit(allocator);
        };

        const tokens_blob = (try blobOf(allocator, meta, "tokenizer.ggml.tokens", &owned, diag)) orelse {
            diag.set("model has no 'tokenizer.ggml.tokens'", .{});
            return error.MissingVocabulary;
        };
        const tokens = arrays.StringArray.init(tokens_blob) catch {
            diag.set("'tokenizer.ggml.tokens' is malformed", .{});
            return error.BadVocabulary;
        };
        if (tokens.count == 0) {
            diag.set("vocabulary is empty", .{});
            return error.BadVocabulary;
        }

        const types: ?arrays.I32Array = if (try blobOf(allocator, meta, "tokenizer.ggml.token_type", &owned, diag)) |b|
            (arrays.I32Array.init(b) catch {
                diag.set("'tokenizer.ggml.token_type' is malformed", .{});
                return error.BadVocabulary;
            })
        else
            null;
        if (types) |t| if (t.count != tokens.count) {
            diag.set("token_type has {d} entries for {d} tokens", .{ t.count, tokens.count });
            return error.BadVocabulary;
        };
        const scores: ?arrays.F32Array = if (try blobOf(allocator, meta, "tokenizer.ggml.scores", &owned, diag)) |b|
            (arrays.F32Array.init(b) catch {
                diag.set("'tokenizer.ggml.scores' is malformed", .{});
                return error.BadVocabulary;
            })
        else
            null;
        if (kind == .spm and scores == null) {
            diag.set("a SentencePiece vocabulary needs 'tokenizer.ggml.scores'", .{});
            return error.BadVocabulary;
        }

        // Hash index with at most 50 percent load.
        var cap: usize = 16;
        while (cap < @as(usize, tokens.count) * 2) cap <<= 1;
        const index = try allocator.alloc(u32, cap);
        @memset(index, 0);

        var self = Tokenizer{
            .allocator = allocator,
            .kind = kind,
            .tokens = tokens,
            .types = types,
            .scores = scores,
            .index = index,
            .index_mask = cap - 1,
            .rscratch = regex.Scratch.init(allocator),
            .json_blobs = owned,
        };
        owned_moved = true;
        errdefer self.deinit();

        // First occurrence wins for duplicate strings, matching llama.cpp.
        for (0..tokens.count) |i| {
            if (self.find(tokens.get(i)) == null) self.insert(@intCast(i));
        }

        self.bos = optId(meta, "tokenizer.ggml.bos_token_id");
        self.eos = optId(meta, "tokenizer.ggml.eos_token_id");
        self.unk = optId(meta, "tokenizer.ggml.unknown_token_id");
        self.pad = optId(meta, "tokenizer.ggml.padding_token_id");
        self.chat_template = meta.getString("tokenizer.chat_template");
        if (try blobOf(allocator, meta, "tokenizer.hk.eog_ids", &self.json_blobs, diag)) |b| self.eog = arrays.I32Array.init(b) catch null;

        try self.indexSpecial();

        switch (kind) {
            .bpe => try self.initBpe(meta, diag),
            .spm => {
                self.add_bos = getBool(meta, "tokenizer.ggml.add_bos_token") orelse true;
                self.add_eos = getBool(meta, "tokenizer.ggml.add_eos_token") orelse false;
                self.add_space_prefix = getBool(meta, "tokenizer.ggml.add_space_prefix") orelse true;
                var hex: [6]u8 = "<0x00>".*;
                for (0..256) |b| {
                    _ = std.fmt.bufPrint(hex[3..5], "{X:0>2}", .{@as(u8, @intCast(b))}) catch unreachable;
                    if (self.find(&hex)) |id| self.byte_fallback[b] = id;
                }
            },
        }
        return self;
    }

    /// The array stored under `key`. Containers written by this project hold a compact blob that
    /// is used in place. Python tooling writes lists as JSON text; those are converted once into
    /// a blob that `owned` keeps alive. A JSON value that is not a flat list is an error.
    fn blobOf(a: std.mem.Allocator, meta: *const metadata.MetadataMap, key: []const u8, owned: *std.ArrayList([]u8), diag: *Diag) Error!?[]const u8 {
        const v = meta.get(key) orelse return null;
        switch (v) {
            .val_bytes => |b| return b,
            .val_json => |j| {
                const blob = arrays.fromJson(a, j) catch |e| {
                    if (e == error.OutOfMemory) return error.OutOfMemory;
                    diag.set("'{s}' is stored as JSON but is not a flat list of strings or numbers ({s})", .{ key, @errorName(e) });
                    return error.BadVocabulary;
                };
                owned.append(a, blob) catch {
                    a.free(blob);
                    return error.OutOfMemory;
                };
                return blob;
            },
            else => return null,
        }
    }

    fn optId(meta: *const metadata.MetadataMap, key: []const u8) ?u32 {
        const v = meta.getInt(key) orelse return null;
        return if (v >= 0) @intCast(v) else null;
    }

    fn getBool(meta: *const metadata.MetadataMap, key: []const u8) ?bool {
        const v = meta.get(key) orelse return null;
        return switch (v) {
            .val_bool => |b| b,
            else => null,
        };
    }

    /// Control and user defined tokens, bucketed by first byte, longest first.
    fn indexSpecial(self: *Tokenizer) Error!void {
        const a = self.allocator;
        var n: usize = 0;
        for (0..self.tokens.count) |i| {
            if (self.isSpecial(@intCast(i))) n += 1;
        }
        if (n == 0) return;
        const store = try a.alloc(u32, n);
        errdefer a.free(store);
        var k: usize = 0;
        for (0..self.tokens.count) |i| {
            if (self.isSpecial(@intCast(i))) {
                store[k] = @intCast(i);
                k += 1;
            }
        }
        const ctx = self;
        std.mem.sort(u32, store, ctx, struct {
            fn lt(t: *const Tokenizer, x: u32, y: u32) bool {
                const lx = t.tokens.get(x).len;
                const ly = t.tokens.get(y).len;
                if (lx != ly) return lx > ly;
                return x < y;
            }
        }.lt);
        // Bucket: since sorted by length, group by first byte keeping order.
        var counts: [256]usize = @splat(0);
        for (store) |id| counts[self.tokens.get(id)[0]] += 1;
        const grouped = try a.alloc(u32, n);
        errdefer a.free(grouped);
        var start: [256]usize = undefined;
        var run: usize = 0;
        for (0..256) |b| {
            start[b] = run;
            run += counts[b];
        }
        var fill = start;
        for (store) |id| {
            const b = self.tokens.get(id)[0];
            grouped[fill[b]] = id;
            fill[b] += 1;
        }
        a.free(store);
        self.special_store = grouped;
        for (0..256) |b| self.special_by_first[b] = grouped[start[b] .. start[b] + counts[b]];
    }

    fn isSpecial(self: *const Tokenizer, id: u32) bool {
        const t = self.tokenType(id);
        if (t != .control and t != .user_defined) return false;
        return self.tokens.get(id).len > 0;
    }

    fn initBpe(self: *Tokenizer, meta: *const metadata.MetadataMap, diag: *Diag) Error!void {
        const a = self.allocator;

        // Pre-tokenizer patterns: explicit list from the container wins over the family name.
        var family = pretok.defaultFamily();
        var custom: ?arrays.StringArray = null;
        if (try blobOf(a, meta, "tokenizer.hk.pre_regex", &self.json_blobs, diag)) |b| {
            custom = arrays.StringArray.init(b) catch {
                diag.set("'tokenizer.hk.pre_regex' is malformed", .{});
                return error.BadVocabulary;
            };
        } else if (meta.getString("tokenizer.ggml.pre")) |name| {
            family = pretok.lookup(name) orelse {
                diag.set("pre-tokenizer '{s}' is not known to this build; tokenization would be wrong", .{name});
                return error.UnsupportedPretokenizer;
            };
        }
        const patterns: []const []const u8 = blk: {
            if (custom) |c| {
                const list = try a.alloc([]const u8, c.count);
                for (list, 0..) |*p, i| p.* = c.get(i);
                break :blk list;
            }
            break :blk family.patterns;
        };
        defer if (custom != null) a.free(patterns);
        self.pre = pretok.PreTokenizer.init(a, patterns) catch {
            diag.set("a pre-tokenizer pattern could not be compiled", .{});
            return error.BadPattern;
        };

        self.ignore_merges = getBool(meta, "tokenizer.ggml.ignore_merges") orelse family.ignore_merges;
        self.add_bos = getBool(meta, "tokenizer.ggml.add_bos_token") orelse family.add_bos;
        self.add_eos = getBool(meta, "tokenizer.ggml.add_eos_token") orelse false;

        // Byte tokens: the vocabulary entry for each byte's mapped character.
        for (0..256) |b| {
            var buf: [4]u8 = undefined;
            const n = uni.encode(byte_cp[b], &buf);
            if (self.find(buf[0..n])) |id| self.byte_tok[b] = id;
        }

        // Merge table keyed by the ids of both halves.
        const merges_blob = (try blobOf(a, meta, "tokenizer.ggml.merges", &self.json_blobs, diag)) orelse {
            diag.set("model has no 'tokenizer.ggml.merges'", .{});
            return error.MissingVocabulary;
        };
        const merges = arrays.StringArray.init(merges_blob) catch {
            diag.set("'tokenizer.ggml.merges' is malformed", .{});
            return error.BadVocabulary;
        };
        var cap: usize = 16;
        while (cap < @as(usize, merges.count) * 2) cap <<= 1;
        self.merge_keys = try a.alloc(u64, cap);
        errdefer a.free(self.merge_keys);
        self.merge_vals = try a.alloc(u64, cap);
        errdefer a.free(self.merge_vals);
        @memset(self.merge_keys, 0);
        self.merge_mask = cap - 1;

        var cat: std.ArrayList(u8) = .empty;
        defer cat.deinit(a);
        for (0..merges.count) |rank| {
            const m = merges.get(rank);
            // "left right": the separator is the first space after the first character.
            const sp = std.mem.indexOfScalarPos(u8, m, 1, ' ') orelse continue;
            const l = self.find(m[0..sp]) orelse continue;
            const r = self.find(m[sp + 1 ..]) orelse continue;
            cat.clearRetainingCapacity();
            try cat.appendSlice(a, m[0..sp]);
            try cat.appendSlice(a, m[sp + 1 ..]);
            const merged = self.find(cat.items) orelse continue;
            self.mergeInsert(l, r, (@as(u64, @intCast(rank)) << 32) | merged);
        }
    }

    // -----------------------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------------------

    const Work = struct {
        sym_id: std.ArrayList(u32) = .empty,
        sym_off: std.ArrayList(u32) = .empty,
        sym_len: std.ArrayList(u32) = .empty,
        prev: std.ArrayList(i32) = .empty,
        next: std.ArrayList(i32) = .empty,
        heap: std.ArrayList(Bigram) = .empty,
        text: std.ArrayList(u8) = .empty,
        frag: std.ArrayList(Fragment) = .empty,
        frag2: std.ArrayList(Fragment) = .empty,

        fn deinit(self: *Work, a: std.mem.Allocator) void {
            self.sym_id.deinit(a);
            self.sym_off.deinit(a);
            self.sym_len.deinit(a);
            self.prev.deinit(a);
            self.next.deinit(a);
            self.heap.deinit(a);
            self.text.deinit(a);
            self.frag.deinit(a);
            self.frag2.deinit(a);
        }
    };

    const Bigram = struct {
        /// BPE: merge rank, lower first. SPM: negated score, so lower is better too.
        key: f64,
        left: u32,
        right: u32,
        /// Sizes or ids at insertion time, to detect stale entries.
        a: u32,
        b: u32,
        merged: u32,
    };

    const Fragment = struct {
        /// Raw text range, or a single token when `token != null_id`.
        start: u32,
        end: u32,
        token: u32,
    };

    fn heapLess(x: Bigram, y: Bigram) bool {
        if (x.key != y.key) return x.key < y.key;
        return x.left < y.left;
    }

    fn heapPush(self: *Tokenizer, b: Bigram) !void {
        const h = &self.work.heap;
        try h.append(self.allocator, b);
        var i = h.items.len - 1;
        while (i > 0) {
            const p = (i - 1) / 2;
            if (!heapLess(h.items[i], h.items[p])) break;
            std.mem.swap(Bigram, &h.items[i], &h.items[p]);
            i = p;
        }
    }

    fn heapPop(self: *Tokenizer) ?Bigram {
        const h = &self.work.heap;
        if (h.items.len == 0) return null;
        const top = h.items[0];
        const last = h.pop().?;
        if (h.items.len > 0) {
            h.items[0] = last;
            var i: usize = 0;
            while (true) {
                const l = 2 * i + 1;
                const r = l + 1;
                var m = i;
                if (l < h.items.len and heapLess(h.items[l], h.items[m])) m = l;
                if (r < h.items.len and heapLess(h.items[r], h.items[m])) m = r;
                if (m == i) break;
                std.mem.swap(Bigram, &h.items[i], &h.items[m]);
                i = m;
            }
        }
        return top;
    }

    /// Splits `text` into raw ranges and special tokens.
    fn partition(self: *Tokenizer, text: []const u8, parse_special: bool, out: *std.ArrayList(Fragment)) !void {
        const a = self.allocator;
        out.clearRetainingCapacity();
        if (text.len == 0) return;
        if (self.special_store.len == 0) {
            try out.append(a, .{ .start = 0, .end = @intCast(text.len), .token = null_id });
            return;
        }
        var raw_start: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            var matched: ?u32 = null;
            for (self.special_by_first[text[i]]) |id| {
                const t = self.tokens.get(id);
                if (std.mem.startsWith(u8, text[i..], t)) {
                    const ty = self.tokenType(id);
                    // Without parse_special, only user defined tokens are split out.
                    if (!parse_special and ty != .user_defined) continue;
                    matched = id;
                    break;
                }
            }
            if (matched) |id| {
                if (i > raw_start) try out.append(a, .{ .start = @intCast(raw_start), .end = @intCast(i), .token = null_id });
                try out.append(a, .{ .start = 0, .end = 0, .token = id });
                i += self.tokens.get(id).len;
                raw_start = i;
            } else {
                i += 1;
            }
        }
        if (raw_start < text.len) try out.append(a, .{ .start = @intCast(raw_start), .end = @intCast(text.len), .token = null_id });
    }

    /// Appends the token ids for `text` to `out`. Not thread safe: encoding uses scratch space
    /// owned by the tokenizer, so callers on several threads must serialize or keep one
    /// tokenizer per thread. Decoding only reads and may run anywhere.
    pub fn encode(self: *Tokenizer, text: []const u8, opts: EncodeOptions, out: *std.ArrayList(u32)) Error!void {
        const a = self.allocator;
        try self.partition(text, opts.parse_special, &self.work.frag);

        switch (self.kind) {
            .spm => {
                var prev_special = true;
                if (opts.add_special and self.add_bos) {
                    if (self.bos) |b| {
                        try out.append(a, b);
                        prev_special = true;
                    }
                }
                for (self.work.frag.items) |f| {
                    if (f.token != null_id) {
                        try out.append(a, f.token);
                        prev_special = true;
                        continue;
                    }
                    self.work.text.clearRetainingCapacity();
                    if (self.add_space_prefix and prev_special) try self.work.text.append(a, ' ');
                    try self.work.text.appendSlice(a, text[f.start..f.end]);
                    try self.encodeSpm(out);
                    prev_special = false;
                }
                if (opts.add_special and self.add_eos) {
                    if (self.eos) |e| try out.append(a, e);
                }
            },
            .bpe => {
                if (opts.add_special and self.add_bos) {
                    if (self.bos) |b| try out.append(a, b);
                }
                for (self.work.frag.items) |f| {
                    if (f.token != null_id) {
                        try out.append(a, f.token);
                        continue;
                    }
                    try self.encodeBpe(text[f.start..f.end], out);
                }
                if (opts.add_special and self.add_eos) {
                    if (self.eos) |e| try out.append(a, e);
                }
            },
        }
    }

    fn encodeBpe(self: *Tokenizer, text: []const u8, out: *std.ArrayList(u32)) !void {
        const a = self.allocator;
        // Decode to code points, remembering each one's byte offset.
        self.cps.clearRetainingCapacity();
        self.offs.clearRetainingCapacity();
        var i: usize = 0;
        while (i < text.len) {
            const d = uni.decode(text[i..]);
            try self.cps.append(a, d.cp);
            try self.offs.append(a, @intCast(i));
            i += d.len;
        }
        try self.offs.append(a, @intCast(text.len));

        try self.pre.?.split(self.cps.items, &self.rscratch, &self.segs, &self.segs_tmp);
        for (self.segs.items) |s| {
            const word = text[self.offs.items[s[0]]..self.offs.items[s[1]]];
            try self.bpeWord(word, out);
        }
    }

    fn bpeWord(self: *Tokenizer, word: []const u8, out: *std.ArrayList(u32)) !void {
        const a = self.allocator;
        const w = &self.work;

        if (self.ignore_merges) {
            // Whole word as one token when the vocabulary has it.
            w.text.clearRetainingCapacity();
            for (word) |b| {
                var buf: [4]u8 = undefined;
                const n = uni.encode(byte_cp[b], &buf);
                try w.text.appendSlice(a, buf[0..n]);
            }
            if (self.find(w.text.items)) |id| {
                try out.append(a, id);
                return;
            }
        }

        if (word.len == 0) return;
        w.sym_id.clearRetainingCapacity();
        w.prev.clearRetainingCapacity();
        w.next.clearRetainingCapacity();
        w.heap.clearRetainingCapacity();
        const n = word.len;
        for (word, 0..) |b, k| {
            try w.sym_id.append(a, self.byte_tok[b]);
            try w.prev.append(a, @as(i32, @intCast(k)) - 1);
            try w.next.append(a, if (k + 1 == n) -1 else @as(i32, @intCast(k)) + 1);
        }
        for (1..n) |k| try self.addBpeBigram(@intCast(k - 1), @intCast(k));

        while (self.heapPop()) |bg| {
            // Skip entries that no longer describe the current symbols.
            if (w.sym_id.items[bg.left] != bg.a or w.sym_id.items[bg.right] != bg.b) continue;
            if (w.next.items[bg.left] != @as(i32, @intCast(bg.right))) continue;
            w.sym_id.items[bg.left] = bg.merged;
            w.sym_id.items[bg.right] = null_id - 1; // dead
            const nx = w.next.items[bg.right];
            w.next.items[bg.left] = nx;
            if (nx >= 0) w.prev.items[@intCast(nx)] = @intCast(bg.left);
            const pv = w.prev.items[bg.left];
            if (pv >= 0) try self.addBpeBigram(@intCast(pv), bg.left);
            if (nx >= 0) try self.addBpeBigram(bg.left, @intCast(nx));
        }

        var k: i32 = 0;
        while (k >= 0) : (k = w.next.items[@intCast(k)]) {
            const id = w.sym_id.items[@intCast(k)];
            if (id == null_id) {
                // Byte token missing from the vocabulary: use unknown.
                if (self.unk) |u| try out.append(a, u);
            } else {
                try out.append(a, id);
            }
        }
    }

    fn addBpeBigram(self: *Tokenizer, left: u32, right: u32) !void {
        const w = &self.work;
        const l = w.sym_id.items[left];
        const r = w.sym_id.items[right];
        if (l == null_id or r == null_id or l == null_id - 1 or r == null_id - 1) return;
        const v = self.mergeLookup(l, r) orelse return;
        try self.heapPush(.{
            .key = @floatFromInt(v >> 32),
            .left = left,
            .right = right,
            .a = l,
            .b = r,
            .merged = @intCast(v & 0xFFFF_FFFF),
        });
    }

    fn encodeSpm(self: *Tokenizer, out: *std.ArrayList(u32)) !void {
        const a = self.allocator;
        const w = &self.work;

        // Escape spaces as U+2581 in place into a second buffer.
        var esc: std.ArrayList(u8) = .empty;
        defer esc.deinit(a);
        for (w.text.items) |c| {
            if (c == ' ') try esc.appendSlice(a, "\xe2\x96\x81") else try esc.append(a, c);
        }
        const text = esc.items;

        w.sym_off.clearRetainingCapacity();
        w.sym_len.clearRetainingCapacity();
        w.prev.clearRetainingCapacity();
        w.next.clearRetainingCapacity();
        w.heap.clearRetainingCapacity();
        var i: usize = 0;
        while (i < text.len) {
            const d = uni.decode(text[i..]);
            try w.sym_off.append(a, @intCast(i));
            try w.sym_len.append(a, d.len);
            i += d.len;
        }
        const n = w.sym_off.items.len;
        for (0..n) |k| {
            try w.prev.append(a, @as(i32, @intCast(k)) - 1);
            try w.next.append(a, if (k + 1 == n) -1 else @as(i32, @intCast(k)) + 1);
        }
        for (1..n) |k| try self.addSpmBigram(text, @intCast(k - 1), @intCast(k));

        while (self.heapPop()) |bg| {
            const ll = w.sym_len.items[bg.left];
            const rl = w.sym_len.items[bg.right];
            if (ll == 0 or rl == 0 or ll + rl != bg.a) continue;
            w.sym_len.items[bg.left] = ll + rl;
            w.sym_len.items[bg.right] = 0;
            const nx = w.next.items[bg.right];
            w.next.items[bg.left] = nx;
            if (nx >= 0) w.prev.items[@intCast(nx)] = @intCast(bg.left);
            const pv = w.prev.items[bg.left];
            if (pv >= 0) try self.addSpmBigram(text, @intCast(pv), bg.left);
            if (nx >= 0) try self.addSpmBigram(text, bg.left, @intCast(nx));
        }

        var k: i32 = 0;
        while (n > 0 and k >= 0) : (k = w.next.items[@intCast(k)]) {
            const o = w.sym_off.items[@intCast(k)];
            const len = w.sym_len.items[@intCast(k)];
            if (len == 0) continue;
            const s = text[o .. o + len];
            if (self.find(s)) |id| {
                try out.append(a, id);
            } else {
                // Not a token: fall back to one byte token per byte.
                for (s) |b| {
                    const id = self.byte_fallback[b];
                    if (id != null_id) {
                        try out.append(a, id);
                    } else if (self.unk) |u| {
                        try out.append(a, u);
                    }
                }
            }
        }
    }

    fn addSpmBigram(self: *Tokenizer, text: []const u8, left: u32, right: u32) !void {
        const w = &self.work;
        const lo = w.sym_off.items[left];
        const ll = w.sym_len.items[left];
        const rl = w.sym_len.items[right];
        if (ll == 0 or rl == 0) return;
        const joined = text[lo .. lo + ll + rl];
        const id = self.find(joined) orelse return;
        const score = self.scores.?.get(id);
        try self.heapPush(.{
            .key = -@as(f64, score),
            .left = left,
            .right = right,
            .a = ll + rl,
            .b = 0,
            .merged = id,
        });
    }

    // -----------------------------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------------------------

    /// Appends the bytes a token stands for. Control tokens are skipped unless `special`.
    pub fn decodeToken(self: *const Tokenizer, id: u32, special: bool, out: *std.ArrayList(u8)) !void {
        if (id >= self.tokens.count) return;
        const ty = self.tokenType(id);
        const s = self.tokens.get(id);
        if (ty == .control or ty == .unknown or ty == .unused) {
            if (special) try out.appendSlice(self.allocator, s);
            return;
        }
        switch (self.kind) {
            .spm => {
                if (ty == .byte or (s.len == 6 and std.mem.startsWith(u8, s, "<0x") and s[5] == '>')) {
                    if (std.fmt.parseInt(u8, s[3..5], 16)) |b| {
                        try out.append(self.allocator, b);
                        return;
                    } else |_| {}
                }
                var i: usize = 0;
                while (i < s.len) {
                    if (s.len - i >= 3 and s[i] == 0xE2 and s[i + 1] == 0x96 and s[i + 2] == 0x81) {
                        try out.append(self.allocator, ' ');
                        i += 3;
                    } else {
                        try out.append(self.allocator, s[i]);
                        i += 1;
                    }
                }
            },
            .bpe => {
                if (ty == .user_defined) {
                    try out.appendSlice(self.allocator, s);
                    return;
                }
                var i: usize = 0;
                while (i < s.len) {
                    const d = uni.decode(s[i..]);
                    if (d.cp < 512 and cp_byte[d.cp] != 0xFFFF) {
                        try out.append(self.allocator, @intCast(cp_byte[d.cp]));
                    } else {
                        try out.appendSlice(self.allocator, s[i .. i + d.len]);
                    }
                    i += d.len;
                }
            },
        }
    }

    pub fn decode(self: *const Tokenizer, ids: []const u32, special: bool, out: *std.ArrayList(u8)) !void {
        for (ids) |id| try self.decodeToken(id, special, out);
    }
};

// ---------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------

/// A tiny GPT-2 style vocabulary built in memory, enough to exercise merging and special tokens.
fn testVocab(a: std.mem.Allocator, meta: *metadata.MetadataMap) !void {
    var toks = arrays.StringArrayBuilder.init(a);
    defer toks.deinit();
    // All 256 byte tokens, then merged pieces, then specials.
    for (0..256) |b| {
        var buf: [4]u8 = undefined;
        const n = uni.encode(byte_cp[b], &buf);
        try toks.append(buf[0..n]);
    }
    const extra = [_][]const u8{ "he", "ll", "hell", "hello", "\xc4\xa0w", "<|im_start|>", "<|im_end|>" };
    for (extra) |e| try toks.append(e);
    const tb = try toks.finish();
    try meta.set("tokenizer.ggml.tokens", .{ .val_bytes = tb });

    var merges = arrays.StringArrayBuilder.init(a);
    defer merges.deinit();
    try merges.append("h e");
    try merges.append("l l");
    try merges.append("he ll");
    try merges.append("hell o");
    const mb = try merges.finish();
    try meta.set("tokenizer.ggml.merges", .{ .val_bytes = mb });

    var types: [256 + 7]u8 = undefined;
    var raw: [(256 + 7) * 4]u8 = undefined;
    for (0..256 + 7) |i| {
        const t: i32 = if (i == 256 + 5 or i == 256 + 6) 3 else 1;
        std.mem.writeInt(i32, raw[i * 4 ..][0..4], t, .little);
    }
    _ = &types;
    try meta.set("tokenizer.ggml.token_type", .{ .val_bytes = try arrays.fixedBlob(a, .i32, 256 + 7, &raw) });
    try meta.setString("tokenizer.ggml.model", "gpt2");
    try meta.setString("tokenizer.ggml.pre", "gpt-2");
}

test "bpe merges follow rank order and round trip" {
    const a = std.testing.allocator;
    var meta = metadata.MetadataMap.init(a);
    defer meta.deinit();
    try testVocab(a, &meta);
    var diag = Diag{};
    var tok = try Tokenizer.fromMetadata(a, &meta, &diag);
    defer tok.deinit();

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(a);
    try tok.encode("hello", .{}, &ids);
    // h e l l o: merges (h,e) rank0, (l,l) rank1, (he,ll) rank2, (hell,o) rank3 -> one token.
    try std.testing.expectEqual(@as(usize, 1), ids.items.len);
    try std.testing.expectEqualStrings("hello", tok.piece(ids.items[0]));

    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    try tok.decode(ids.items, false, &text);
    try std.testing.expectEqualStrings("hello", text.items);
}

/// Re-stores every array of `src` as JSON text, the way Python tooling writes lists.
fn jsonCopy(a: std.mem.Allocator, src: *const metadata.MetadataMap, dst: *metadata.MetadataMap) !void {
    for ([_][]const u8{ "tokenizer.ggml.tokens", "tokenizer.ggml.merges" }) |key| {
        const arr = try arrays.StringArray.init(src.get(key).?.val_bytes);
        var out: std.Io.Writer.Allocating = .init(a);
        defer out.deinit();
        var w: std.json.Stringify = .{ .writer = &out.writer };
        try w.beginArray();
        for (0..arr.count) |i| try w.write(arr.get(i));
        try w.endArray();
        try dst.setJson(key, out.written());
    }
    const types = try arrays.I32Array.init(src.get("tokenizer.ggml.token_type").?.val_bytes);
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    var w: std.json.Stringify = .{ .writer = &out.writer };
    try w.beginArray();
    for (0..types.count) |i| try w.write(types.get(i));
    try w.endArray();
    try dst.setJson("tokenizer.ggml.token_type", out.written());
    try dst.setString("tokenizer.ggml.model", "gpt2");
    try dst.setString("tokenizer.ggml.pre", "gpt-2");
}

test "a vocabulary stored as JSON text behaves the same as the compact layout" {
    const a = std.testing.allocator;
    var native = metadata.MetadataMap.init(a);
    defer native.deinit();
    try testVocab(a, &native);
    var as_json = metadata.MetadataMap.init(a);
    defer as_json.deinit();
    try jsonCopy(a, &native, &as_json);

    var d1 = Diag{};
    var t1 = try Tokenizer.fromMetadata(a, &native, &d1);
    defer t1.deinit();
    var d2 = Diag{};
    var t2 = try Tokenizer.fromMetadata(a, &as_json, &d2);
    defer t2.deinit();

    for ([_][]const u8{ "hello hello", "hell o w", "<|im_start|>hello<|im_end|>", "h\xc3\xa9llo" }) |text| {
        var x: std.ArrayList(u32) = .empty;
        defer x.deinit(a);
        var y: std.ArrayList(u32) = .empty;
        defer y.deinit(a);
        try t1.encode(text, .{ .add_special = false, .parse_special = true }, &x);
        try t2.encode(text, .{ .add_special = false, .parse_special = true }, &y);
        try std.testing.expectEqualSlices(u32, x.items, y.items);
    }
}

test "a JSON vocabulary that is not a flat list is refused with a reason" {
    const a = std.testing.allocator;
    var meta = metadata.MetadataMap.init(a);
    defer meta.deinit();
    try meta.setString("tokenizer.ggml.model", "gpt2");
    try meta.setJson("tokenizer.ggml.tokens", "[[\"a\"], [\"b\"]]");
    var diag = Diag{};
    try std.testing.expectError(error.BadVocabulary, Tokenizer.fromMetadata(a, &meta, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "flat list") != null);
}

test "bytes outside the merge table survive a round trip" {
    const a = std.testing.allocator;
    var meta = metadata.MetadataMap.init(a);
    defer meta.deinit();
    try testVocab(a, &meta);
    var diag = Diag{};
    var tok = try Tokenizer.fromMetadata(a, &meta, &diag);
    defer tok.deinit();

    const s = "héllo wörld \xff\xfe 日本語\n";
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(a);
    try tok.encode(s, .{}, &ids);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    try tok.decode(ids.items, false, &text);
    try std.testing.expectEqualStrings(s, text.items);
}

test "special tokens are only parsed on request" {
    const a = std.testing.allocator;
    var meta = metadata.MetadataMap.init(a);
    defer meta.deinit();
    try testVocab(a, &meta);
    var diag = Diag{};
    var tok = try Tokenizer.fromMetadata(a, &meta, &diag);
    defer tok.deinit();

    var plain: std.ArrayList(u32) = .empty;
    defer plain.deinit(a);
    try tok.encode("<|im_start|>hi", .{ .parse_special = false }, &plain);
    var parsed: std.ArrayList(u32) = .empty;
    defer parsed.deinit(a);
    try tok.encode("<|im_start|>hi", .{ .parse_special = true }, &parsed);
    try std.testing.expect(plain.items.len > parsed.items.len);
    try std.testing.expectEqualStrings("<|im_start|>", tok.piece(parsed.items[0]));
}

test "empty input" {
    const a = std.testing.allocator;
    var meta = metadata.MetadataMap.init(a);
    defer meta.deinit();
    try testVocab(a, &meta);
    var diag = Diag{};
    var tok = try Tokenizer.fromMetadata(a, &meta, &diag);
    defer tok.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(a);
    try tok.encode("", .{}, &ids);
    try std.testing.expectEqual(@as(usize, 0), ids.items.len);
}

test "missing and unsupported vocabularies are reported, not guessed" {
    const a = std.testing.allocator;
    var meta = metadata.MetadataMap.init(a);
    defer meta.deinit();
    var diag = Diag{};
    try std.testing.expectError(error.MissingVocabulary, Tokenizer.fromMetadata(a, &meta, &diag));
    try meta.setString("tokenizer.ggml.model", "bert");
    try std.testing.expectError(error.UnsupportedTokenizer, Tokenizer.fromMetadata(a, &meta, &diag));
}
