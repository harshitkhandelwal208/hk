//! Unicode helpers for tokenization: general categories, whitespace, and UTF-8 decoding that
//! never fails (invalid bytes become U+FFFD of length one so byte offsets stay meaningful).

const std = @import("std");

pub const Cat = enum(u8) {
    Lu, Ll, Lt, Lm, Lo,
    Mn, Mc, Me,
    Nd, Nl, No,
    Pc, Pd, Ps, Pe, Pi, Pf, Po,
    Sm, Sc, Sk, So,
    Zs, Zl, Zp,
    Cc, Cf, Cs, Co,
};

/// Bit mask over `Cat`, used by character classes.
pub const CatSet = u32;

pub fn bit(c: Cat) CatSet {
    return @as(CatSet, 1) << @as(u5, @intCast(@intFromEnum(c)));
}

pub const letters: CatSet = bit(.Lu) | bit(.Ll) | bit(.Lt) | bit(.Lm) | bit(.Lo);
pub const marks: CatSet = bit(.Mn) | bit(.Mc) | bit(.Me);
pub const numbers: CatSet = bit(.Nd) | bit(.Nl) | bit(.No);
pub const punctuation: CatSet = bit(.Pc) | bit(.Pd) | bit(.Ps) | bit(.Pe) | bit(.Pi) | bit(.Pf) | bit(.Po);
pub const symbols: CatSet = bit(.Sm) | bit(.Sc) | bit(.Sk) | bit(.So);
pub const separators: CatSet = bit(.Zs) | bit(.Zl) | bit(.Zp);
pub const others: CatSet = bit(.Cc) | bit(.Cf) | bit(.Cs) | bit(.Co);

const table = @embedFile("unicode_cat.bin");
const entry_len = 9;
const n_entries = table.len / entry_len;

inline fn entryFirst(i: usize) u32 {
    return std.mem.readInt(u32, table[i * entry_len ..][0..4], .little);
}
inline fn entryLast(i: usize) u32 {
    return std.mem.readInt(u32, table[i * entry_len + 4 ..][0..4], .little);
}

/// General category of a code point, or null for unassigned code points.
pub fn category(cp: u32) ?Cat {
    var lo: usize = 0;
    var hi: usize = n_entries;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (cp < entryFirst(mid)) {
            hi = mid;
        } else if (cp > entryLast(mid)) {
            lo = mid + 1;
        } else {
            return @enumFromInt(table[mid * entry_len + 8]);
        }
    }
    return null;
}

pub fn inCats(cp: u32, set: CatSet) bool {
    const c = category(cp) orelse return false;
    return set & bit(c) != 0;
}

/// Unicode White_Space, which is what `\s` means in the tokenizer patterns.
pub fn isSpace(cp: u32) bool {
    return switch (cp) {
        0x09...0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

/// CJK ideographs, for the `\p{Han}` property.
pub fn isHan(cp: u32) bool {
    return switch (cp) {
        0x2E80...0x2E99, 0x2E9B...0x2EF3, 0x2F00...0x2FD5, 0x3005, 0x3007, 0x3021...0x3029, 0x3038...0x303B, 0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFA6D, 0xFA70...0xFAD9, 0x16FE2, 0x16FE3, 0x16FF0, 0x16FF1, 0x20000...0x2A6DF, 0x2A700...0x2B81D, 0x2B820...0x2CEAD, 0x2CEB0...0x2EBE0, 0x2EBF0...0x2EE5D, 0x2F800...0x2FA1D, 0x30000...0x3134A, 0x31350...0x323AF => true,
        else => false,
    };
}

pub const Decoded = struct { cp: u32, len: u3 };

/// Decodes one code point at the start of `s`. Invalid or truncated sequences yield U+FFFD
/// with length 1, so iterating always advances and byte offsets stay exact.
pub fn decode(s: []const u8) Decoded {
    const b0 = s[0];
    if (b0 < 0x80) return .{ .cp = b0, .len = 1 };
    const len = std.unicode.utf8ByteSequenceLength(b0) catch return .{ .cp = 0xFFFD, .len = 1 };
    if (len > s.len) return .{ .cp = 0xFFFD, .len = 1 };
    const cp = std.unicode.utf8Decode(s[0..len]) catch return .{ .cp = 0xFFFD, .len = 1 };
    return .{ .cp = cp, .len = @intCast(len) };
}

pub fn encode(cp: u32, out: *[4]u8) usize {
    return std.unicode.utf8Encode(@intCast(cp), out) catch {
        out[0] = 0xEF;
        out[1] = 0xBF;
        out[2] = 0xBD;
        return 3;
    };
}

/// Simple case folding sufficient for the `(?i:...)` groups in tokenizer patterns.
pub fn foldLower(cp: u32) u32 {
    if (cp >= 'A' and cp <= 'Z') return cp + 32;
    return cp;
}

test "categories" {
    try std.testing.expectEqual(Cat.Lu, category('A').?);
    try std.testing.expectEqual(Cat.Ll, category('a').?);
    try std.testing.expectEqual(Cat.Nd, category('7').?);
    try std.testing.expectEqual(Cat.Po, category('!').?);
    try std.testing.expectEqual(Cat.Lo, category(0x4E2D).?);
    try std.testing.expect(inCats(0x00E9, letters));
    try std.testing.expect(!inCats('1', letters));
    try std.testing.expect(isSpace(' ') and isSpace('\n') and isSpace(0x3000));
    try std.testing.expect(!isSpace('a'));
}

test "decode never fails and always advances" {
    const bad = [_]u8{ 0xFF, 'a', 0xE4, 0xB8 };
    var i: usize = 0;
    var count: usize = 0;
    while (i < bad.len) {
        const d = decode(bad[i..]);
        i += d.len;
        count += 1;
    }
    try std.testing.expectEqual(bad.len, i);
    try std.testing.expect(count >= 3);
    const ok = decode("\u{4E2D}x");
    try std.testing.expectEqual(@as(u32, 0x4E2D), ok.cp);
    try std.testing.expectEqual(@as(u3, 3), ok.len);
}
