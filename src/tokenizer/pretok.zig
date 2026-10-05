//! Pre-tokenizer: splits text into "words" before byte pair merging.
//!
//! Each model family defines one or more regular expressions. They are applied in order, and
//! every one splits the pieces produced by the previous one, exactly as llama.cpp does, so token
//! ids match. Patterns come either from the container (`tokenizer.hk.pre_regex`, copied from a
//! Hugging Face tokenizer.json) or from the table below keyed by `tokenizer.ggml.pre`.

const std = @import("std");
const regex = @import("regex.zig");

pub const Family = struct {
    patterns: []const []const u8,
    /// If the whole pre-token is already a vocabulary entry, emit it without merging.
    ignore_merges: bool = false,
    add_bos: bool = false,
};

const contraction_i = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";
const gpt2_pat = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
const llama3_pat = contraction_i ++ "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const qwen2_pat = contraction_i ++ "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const qwen35_pat = contraction_i ++ "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const gpt4o_cased = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+" ++ contraction_i ++ "?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*" ++ contraction_i ++ "?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const tekken_pat = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

const default_family = Family{ .patterns = &.{
    "[\\p{P}\\$\\+<=>\\^~\\|]+",
    gpt2_pat,
    "\\p{N}+",
    "[0-9][0-9][0-9]",
} };

const gpt2_family = Family{ .patterns = &.{gpt2_pat} };
const llama3_family = Family{ .patterns = &.{llama3_pat}, .ignore_merges = true, .add_bos = true };
const qwen2_family = Family{ .patterns = &.{qwen2_pat} };
const smollm_family = Family{ .patterns = &.{ "\\p{N}", gpt2_pat } };

const Entry = struct { []const u8, Family };

/// `tokenizer.ggml.pre` values, as written by llama.cpp's converters.
const table = [_]Entry{
    .{ "default", default_family },
    .{ "llama3", llama3_family },
    .{ "llama-v3", llama3_family },
    .{ "llama-bpe", llama3_family },
    .{ "falcon3", llama3_family },
    .{ "dbrx", .{ .patterns = &.{llama3_pat} } },
    .{ "smaug-bpe", .{ .patterns = &.{llama3_pat} } },
    .{ "chatglm-bpe", .{ .patterns = &.{llama3_pat} } },
    .{ "glm4", .{ .patterns = &.{llama3_pat} } },
    .{ "gpt-2", gpt2_family },
    .{ "phi-2", gpt2_family },
    .{ "mpt", gpt2_family },
    .{ "olmo", gpt2_family },
    .{ "jais", gpt2_family },
    .{ "roberta-bpe", gpt2_family },
    .{ "trillion", gpt2_family },
    .{ "qwen2", qwen2_family },
    .{ "deepseek-r1-qwen", qwen2_family },
    .{ "stablelm2", qwen2_family },
    .{ "hunyuan", qwen2_family },
    .{ "megrez", qwen2_family },
    .{ "qwen35", .{ .patterns = &.{qwen35_pat} } },
    .{ "smollm", smollm_family },
    .{ "starcoder", smollm_family },
    .{ "refact", smollm_family },
    .{ "command-r", smollm_family },
    .{ "codeshell", smollm_family },
    .{ "exaone", smollm_family },
    .{ "falcon", .{ .patterns = &.{
        "[\\p{P}\\$\\+<=>\\^~\\|`]+",
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        "[0-9][0-9][0-9]",
    } } },
    .{ "gpt-4o", .{ .patterns = &.{gpt4o_cased} } },
    .{ "llama4", .{ .patterns = &.{gpt4o_cased} } },
    .{ "minimax-m2", .{ .patterns = &.{gpt4o_cased} } },
    .{ "tekken", .{ .patterns = &.{tekken_pat}, .ignore_merges = true, .add_bos = true } },
    .{ "deepseek-llm", .{ .patterns = &.{
        "[\r\n]",
        "\\s?[A-Za-z\u{00B5}\u{00C0}-\u{00D6}\u{00D8}-\u{00F6}\u{00F8}-\u{01BA}\u{01BC}-\u{01BF}\u{01C4}-\u{0293}\u{0295}-\u{02AF}\u{0370}-\u{0373}\u{0376}\u{0377}\u{037B}-\u{037D}\u{037F}\u{0386}\u{0388}-\u{038A}\u{038C}\u{038E}-\u{03A1}\u{03A3}-\u{03F5}\u{03F7}-\u{0481}\u{048A}-\u{052F}\u{0531}-\u{0556}\u{10A0}-\u{10C5}\u{1E00}-\u{1F15}\u{1F18}-\u{1F1D}\u{1F20}-\u{1F45}\u{1F48}-\u{1F4D}\u{1F50}-\u{1F57}\u{1F59}\u{1F5B}\u{1F5D}\u{1F5F}-\u{1F7D}\u{1F80}-\u{1FB4}\u{1FB6}-\u{1FBC}\u{1FBE}\u{1FC2}-\u{1FC4}\u{1FC6}-\u{1FCC}\u{1FD0}-\u{1FD3}\u{1FD6}-\u{1FDB}\u{1FE0}-\u{1FEC}\u{1FF2}-\u{1FF4}\u{1FF6}-\u{1FFC}\u{FF21}-\u{FF3A}\u{FF41}-\u{FF5A}]+",
        "\\s?[!-/:-~\u{FF01}-\u{FF0F}\u{FF1A}-\u{FF5E}\u{2018}-\u{201F}\u{3000}-\u{3002}]+",
        "\\s+$",
        "[\u{4E00}-\u{9FA5}\u{0800}-\u{4E00}\u{AC00}-\u{D7FF}]+",
        "\\p{N}+",
    } } },
    .{ "deepseek-coder", .{ .patterns = &.{
        "[\r\n]",
        "\\s?\\p{L}+",
        "\\s?\\p{P}+",
        "[\u{4E00}-\u{9FA5}\u{0800}-\u{4E00}\u{AC00}-\u{D7FF}]+",
        "\\p{N}",
    } } },
    .{ "deepseek-v3", .{ .patterns = &.{
        "\\p{N}{1,3}",
        "[\u{4E00}-\u{9FA5}\u{3040}-\u{309F}\u{30A0}-\u{30FF}]+",
        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
    } } },
};

pub fn lookup(name: []const u8) ?Family {
    for (table) |e| if (std.mem.eql(u8, e[0], name)) return e[1];
    return null;
}

pub fn defaultFamily() Family {
    return default_family;
}

/// Splits code point sequences into word segments using a list of compiled patterns.
pub const PreTokenizer = struct {
    allocator: std.mem.Allocator,
    regexes: []regex.Regex,

    pub fn init(allocator: std.mem.Allocator, patterns: []const []const u8) regex.Error!PreTokenizer {
        const res = try allocator.alloc(regex.Regex, patterns.len);
        var done: usize = 0;
        errdefer {
            for (res[0..done]) |*r| r.deinit();
            allocator.free(res);
        }
        for (patterns, 0..) |p, i| {
            res[i] = try regex.Regex.compile(allocator, p);
            done += 1;
        }
        return .{ .allocator = allocator, .regexes = res };
    }

    pub fn deinit(self: *PreTokenizer) void {
        for (self.regexes) |*r| r.deinit();
        self.allocator.free(self.regexes);
    }

    pub const Segment = [2]u32;

    /// Fills `out` with consecutive segments covering `cps` entirely. `tmp` is scratch space.
    pub fn split(
        self: *const PreTokenizer,
        cps: []const u32,
        sc: *regex.Scratch,
        out: *std.ArrayList(Segment),
        tmp: *std.ArrayList(Segment),
    ) std.mem.Allocator.Error!void {
        const a = self.allocator;
        out.clearRetainingCapacity();
        if (cps.len == 0) return;
        try out.append(a, .{ 0, @intCast(cps.len) });

        for (self.regexes) |*re| {
            tmp.clearRetainingCapacity();
            for (out.items) |seg| {
                var gap_start: u32 = seg[0];
                var pos: u32 = seg[0];
                // A match must stay inside its segment, so later patterns cannot see past the
                // boundary an earlier one drew.
                const window = cps[0..seg[1]];
                while (pos < seg[1]) {
                    const m = re.find(window, pos, sc) orelse break;
                    const ms: u32 = @intCast(m[0]);
                    const me: u32 = @intCast(m[1]);
                    if (me == ms) {
                        // Empty match: skip a character without cutting.
                        pos = ms + 1;
                        continue;
                    }
                    if (ms > gap_start) try tmp.append(a, .{ gap_start, ms });
                    try tmp.append(a, .{ ms, me });
                    gap_start = me;
                    pos = me;
                }
                if (gap_start < seg[1]) try tmp.append(a, .{ gap_start, seg[1] });
            }
            std.mem.swap(std.ArrayList(Segment), out, tmp);
        }
    }
};

test "families split like llama.cpp" {
    const a = std.testing.allocator;
    const fam = lookup("llama3").?;
    var pt = try PreTokenizer.init(a, fam.patterns);
    defer pt.deinit();
    var sc = regex.Scratch.init(a);
    defer sc.deinit();
    var out: std.ArrayList(PreTokenizer.Segment) = .empty;
    defer out.deinit(a);
    var tmp: std.ArrayList(PreTokenizer.Segment) = .empty;
    defer tmp.deinit(a);

    const text = "Hi, 12345!";
    var cps: [10]u32 = undefined;
    for (text, 0..) |c, i| cps[i] = c;
    try pt.split(&cps, &sc, &out, &tmp);
    // "Hi" "," " " is attached to nothing, digits in groups of three, "!" alone.
    var got: [16]u8 = undefined;
    var len: usize = 0;
    for (out.items) |s| {
        if (len > 0) {
            got[len] = '|';
            len += 1;
        }
        for (s[0]..s[1]) |i| {
            got[len] = text[i];
            len += 1;
        }
    }
    try std.testing.expectEqualStrings("Hi|,| |123|45|!", got[0..len]);
}

test "smollm splits digits individually before the gpt2 pattern" {
    const a = std.testing.allocator;
    const fam = lookup("smollm").?;
    var pt = try PreTokenizer.init(a, fam.patterns);
    defer pt.deinit();
    var sc = regex.Scratch.init(a);
    defer sc.deinit();
    var out: std.ArrayList(PreTokenizer.Segment) = .empty;
    defer out.deinit(a);
    var tmp: std.ArrayList(PreTokenizer.Segment) = .empty;
    defer tmp.deinit(a);
    var cps: [7]u32 = undefined;
    for ("ab 123x", 0..) |c, i| cps[i] = c;
    try pt.split(&cps, &sc, &out, &tmp);
    // ab, " " (space attaches to nothing since next is a digit), 1, 2, 3, x
    try std.testing.expectEqual(@as(usize, 6), out.items.len);
}
