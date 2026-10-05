//! Compact array values for container metadata.
//!
//! A vocabulary of 150,000 tokens as JSON is megabytes of text that has to be parsed on every
//! load. These blobs are read in place: opening one validates it once and then `get(i)` is two
//! offset loads. They are stored as ordinary `val_bytes` metadata values, so tools that do not
//! know about arrays still copy them intact.
//!
//! Layout, little endian:
//!     u8  kind      (see `Kind`)
//!     u8  reserved[3]
//!     u32 count
//!   strings:   u32 offsets[count + 1], then the string bytes; string i is data[off[i]..off[i+1]]
//!   f32 / i32: count * 4 bytes

const std = @import("std");

pub const Kind = enum(u8) {
    strings = 1,
    f32 = 2,
    i32 = 3,
};

const header_len = 8;

pub const Error = error{ Malformed, WrongKind };

fn checkHeader(blob: []const u8, want: Kind) Error!u32 {
    if (blob.len < header_len) return error.Malformed;
    const kind = std.enums.fromInt(Kind, blob[0]) orelse return error.Malformed;
    if (kind != want) return error.WrongKind;
    return std.mem.readInt(u32, blob[4..8], .little);
}

/// Read only view of a string array.
pub const StringArray = struct {
    count: u32,
    offsets: []const u8,
    data: []const u8,

    /// Validates the whole offset table up front so that `get` can never read out of bounds,
    /// even for a hostile file.
    pub fn init(blob: []const u8) Error!StringArray {
        const count = try checkHeader(blob, .strings);
        const table_len = (@as(usize, count) + 1) * 4;
        if (blob.len < header_len + table_len) return error.Malformed;
        const offsets = blob[header_len..][0..table_len];
        const data = blob[header_len + table_len ..];
        var prev: u32 = 0;
        for (0..@as(usize, count) + 1) |i| {
            const o = std.mem.readInt(u32, offsets[i * 4 ..][0..4], .little);
            if (o < prev or o > data.len) return error.Malformed;
            prev = o;
        }
        if (prev != data.len) return error.Malformed;
        return .{ .count = count, .offsets = offsets, .data = data };
    }

    pub fn get(self: StringArray, i: usize) []const u8 {
        std.debug.assert(i < self.count);
        const a = std.mem.readInt(u32, self.offsets[i * 4 ..][0..4], .little);
        const b = std.mem.readInt(u32, self.offsets[(i + 1) * 4 ..][0..4], .little);
        return self.data[a..b];
    }
};

pub const F32Array = struct {
    count: u32,
    bytes: []const u8,

    pub fn init(blob: []const u8) Error!F32Array {
        const count = try checkHeader(blob, .f32);
        if (blob.len != header_len + @as(usize, count) * 4) return error.Malformed;
        return .{ .count = count, .bytes = blob[header_len..] };
    }

    pub fn get(self: F32Array, i: usize) f32 {
        std.debug.assert(i < self.count);
        return @bitCast(std.mem.readInt(u32, self.bytes[i * 4 ..][0..4], .little));
    }
};

pub const I32Array = struct {
    count: u32,
    bytes: []const u8,

    pub fn init(blob: []const u8) Error!I32Array {
        const count = try checkHeader(blob, .i32);
        if (blob.len != header_len + @as(usize, count) * 4) return error.Malformed;
        return .{ .count = count, .bytes = blob[header_len..] };
    }

    pub fn get(self: I32Array, i: usize) i32 {
        std.debug.assert(i < self.count);
        return std.mem.readInt(i32, self.bytes[i * 4 ..][0..4], .little);
    }
};

/// Builds a string array blob incrementally. Strings are appended to one growing buffer, so
/// building a vocabulary costs its own size once, not one allocation per token.
pub const StringArrayBuilder = struct {
    allocator: std.mem.Allocator,
    offsets: std.ArrayList(u32) = .empty,
    data: std.ArrayList(u8) = .empty,

    pub fn init(allocator: std.mem.Allocator) StringArrayBuilder {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *StringArrayBuilder) void {
        self.offsets.deinit(self.allocator);
        self.data.deinit(self.allocator);
    }

    pub fn append(self: *StringArrayBuilder, s: []const u8) !void {
        if (self.offsets.items.len == 0) try self.offsets.append(self.allocator, 0);
        if (self.data.items.len + s.len > std.math.maxInt(u32)) return error.ArrayTooLarge;
        try self.data.appendSlice(self.allocator, s);
        try self.offsets.append(self.allocator, @intCast(self.data.items.len));
    }

    pub fn count(self: *const StringArrayBuilder) usize {
        return if (self.offsets.items.len == 0) 0 else self.offsets.items.len - 1;
    }

    /// Returns the finished blob. Caller owns it.
    pub fn finish(self: *StringArrayBuilder) ![]u8 {
        const n = self.count();
        if (self.offsets.items.len == 0) try self.offsets.append(self.allocator, 0);
        const table = self.offsets.items.len * 4;
        const out = try self.allocator.alloc(u8, header_len + table + self.data.items.len);
        errdefer self.allocator.free(out);
        out[0] = @intFromEnum(Kind.strings);
        @memset(out[1..4], 0);
        std.mem.writeInt(u32, out[4..8], @intCast(n), .little);
        for (self.offsets.items, 0..) |o, i| std.mem.writeInt(u32, out[header_len + i * 4 ..][0..4], o, .little);
        @memcpy(out[header_len + table ..], self.data.items);
        return out;
    }
};

/// Builds an f32 or i32 blob from raw little endian values.
pub fn fixedBlob(allocator: std.mem.Allocator, kind: Kind, count: usize, raw: []const u8) ![]u8 {
    std.debug.assert(kind != .strings and raw.len == count * 4);
    const out = try allocator.alloc(u8, header_len + raw.len);
    out[0] = @intFromEnum(kind);
    @memset(out[1..4], 0);
    std.mem.writeInt(u32, out[4..8], @intCast(count), .little);
    @memcpy(out[header_len..], raw);
    return out;
}

pub const JsonError = error{ NotAnArray, MixedOrNestedArray, BadNumber, ArrayTooLarge, OutOfMemory, InvalidJson };

/// Converts a JSON array of strings, of integers or of numbers into a blob. Python tooling stores
/// lists this way; the engine reads the compact layout. The input is scanned once and no value
/// tree is built, so the cost is the blob itself. Integers that fit 32 bits give an i32 array,
/// any other number list gives f32. Caller owns the result.
pub fn fromJson(allocator: std.mem.Allocator, json: []const u8) JsonError![]u8 {
    var scanner = std.json.Scanner.initCompleteInput(allocator, json);
    defer scanner.deinit();
    const next = struct {
        fn f(sc: *std.json.Scanner, a: std.mem.Allocator) JsonError!std.json.Token {
            return sc.nextAlloc(a, .alloc_if_needed) catch |e| switch (e) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidJson,
            };
        }
    }.f;

    if ((try next(&scanner, allocator)) != .array_begin) return error.NotAnArray;

    var strings = StringArrayBuilder.init(allocator);
    defer strings.deinit();
    var nums: std.ArrayList(f64) = .empty;
    defer nums.deinit(allocator);
    var all_int = true;
    var kind: ?Kind = null;
    while (true) {
        const tok = try next(&scanner, allocator);
        switch (tok) {
            .array_end => break,
            .string => |t| {
                if (kind != null and kind.? != .strings) return error.MixedOrNestedArray;
                kind = .strings;
                strings.append(t) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else error.ArrayTooLarge;
            },
            .allocated_string => |t| {
                defer allocator.free(t);
                if (kind != null and kind.? != .strings) return error.MixedOrNestedArray;
                kind = .strings;
                strings.append(t) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else error.ArrayTooLarge;
            },
            .number, .allocated_number => |t| {
                defer if (tok == .allocated_number) allocator.free(t);
                if (kind != null and kind.? == .strings) return error.MixedOrNestedArray;
                kind = .f32;
                const v = std.fmt.parseFloat(f64, t) catch return error.BadNumber;
                if (!std.math.isFinite(v)) return error.BadNumber;
                if (std.mem.indexOfAny(u8, t, ".eE") != null or v < std.math.minInt(i32) or v > std.math.maxInt(i32)) all_int = false;
                nums.append(allocator, v) catch return error.OutOfMemory;
            },
            else => return error.MixedOrNestedArray,
        }
        if (nums.items.len + strings.count() > std.math.maxInt(u32)) return error.ArrayTooLarge;
    }
    // Nothing may follow the closing bracket.
    if ((try next(&scanner, allocator)) != .end_of_document) return error.InvalidJson;

    if (kind == null or kind.? == .strings) return strings.finish() catch error.OutOfMemory;

    const raw = allocator.alloc(u8, nums.items.len * 4) catch return error.OutOfMemory;
    defer allocator.free(raw);
    for (nums.items, 0..) |v, i| {
        if (all_int) {
            std.mem.writeInt(i32, raw[i * 4 ..][0..4], @intFromFloat(v), .little);
        } else {
            std.mem.writeInt(u32, raw[i * 4 ..][0..4], @bitCast(@as(f32, @floatCast(v))), .little);
        }
    }
    return fixedBlob(allocator, if (all_int) .i32 else .f32, nums.items.len, raw) catch error.OutOfMemory;
}

test "json arrays become blobs" {
    const a = std.testing.allocator;
    {
        const blob = try fromJson(a, "[\"he\\u00e9llo\", \"\", \"a\\nb\"]");
        defer a.free(blob);
        const v = try StringArray.init(blob);
        try std.testing.expectEqual(@as(u32, 3), v.count);
        try std.testing.expectEqualStrings("he\u{e9}llo", v.get(0));
        try std.testing.expectEqualStrings("a\nb", v.get(2));
    }
    {
        const blob = try fromJson(a, " [1, -2, 3000000] ");
        defer a.free(blob);
        const v = try I32Array.init(blob);
        try std.testing.expectEqual(@as(i32, -2), v.get(1));
        try std.testing.expectEqual(@as(i32, 3000000), v.get(2));
    }
    {
        const blob = try fromJson(a, "[0.5, 1, -2.25e0]");
        defer a.free(blob);
        const v = try F32Array.init(blob);
        try std.testing.expectEqual(@as(f32, 0.5), v.get(0));
        try std.testing.expectEqual(@as(f32, -2.25), v.get(2));
    }
    {
        // Too large for i32 falls back to f32 instead of wrapping.
        const blob = try fromJson(a, "[1, 5000000000]");
        defer a.free(blob);
        const v = try F32Array.init(blob);
        try std.testing.expectEqual(@as(f32, 5000000000), v.get(1));
    }
    {
        const blob = try fromJson(a, "[]");
        defer a.free(blob);
        try std.testing.expectEqual(@as(u32, 0), (try StringArray.init(blob)).count);
    }
}

test "json arrays that are not flat lists of one kind are refused" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.NotAnArray, fromJson(a, "\"x\""));
    try std.testing.expectError(error.NotAnArray, fromJson(a, "{\"a\":1}"));
    try std.testing.expectError(error.MixedOrNestedArray, fromJson(a, "[\"a\", 1]"));
    try std.testing.expectError(error.MixedOrNestedArray, fromJson(a, "[1, \"a\"]"));
    try std.testing.expectError(error.MixedOrNestedArray, fromJson(a, "[[1]]"));
    try std.testing.expectError(error.MixedOrNestedArray, fromJson(a, "[true]"));
    try std.testing.expectError(error.MixedOrNestedArray, fromJson(a, "[null]"));
    try std.testing.expectError(error.InvalidJson, fromJson(a, "[\"a\""));
    try std.testing.expectError(error.InvalidJson, fromJson(a, "[\"a\"] x"));
    try std.testing.expectError(error.InvalidJson, fromJson(a, ""));
}

test "string array round trip" {
    const a = std.testing.allocator;
    var b = StringArrayBuilder.init(a);
    defer b.deinit();
    try b.append("hello");
    try b.append("");
    try b.append("wörld");
    const blob = try b.finish();
    defer a.free(blob);

    const v = try StringArray.init(blob);
    try std.testing.expectEqual(@as(u32, 3), v.count);
    try std.testing.expectEqualStrings("hello", v.get(0));
    try std.testing.expectEqualStrings("", v.get(1));
    try std.testing.expectEqualStrings("wörld", v.get(2));
}

test "empty string array is valid" {
    const a = std.testing.allocator;
    var b = StringArrayBuilder.init(a);
    defer b.deinit();
    const blob = try b.finish();
    defer a.free(blob);
    const v = try StringArray.init(blob);
    try std.testing.expectEqual(@as(u32, 0), v.count);
}

test "hostile blobs are rejected, not trusted" {
    const a = std.testing.allocator;
    var b = StringArrayBuilder.init(a);
    defer b.deinit();
    try b.append("abc");
    const blob = try b.finish();
    defer a.free(blob);

    // Truncated header and table.
    try std.testing.expectError(error.Malformed, StringArray.init(blob[0..4]));
    try std.testing.expectError(error.Malformed, StringArray.init(blob[0 .. blob.len - 5]));

    // Offset pointing past the data.
    const bad = try a.dupe(u8, blob);
    defer a.free(bad);
    std.mem.writeInt(u32, bad[header_len + 4 ..][0..4], 999, .little);
    try std.testing.expectError(error.Malformed, StringArray.init(bad));

    // Offsets that go backwards.
    var b2 = StringArrayBuilder.init(a);
    defer b2.deinit();
    try b2.append("ab");
    try b2.append("cd");
    const blob2 = try b2.finish();
    defer a.free(blob2);
    std.mem.writeInt(u32, blob2[header_len + 4 ..][0..4], 4, .little);
    std.mem.writeInt(u32, blob2[header_len + 8 ..][0..4], 2, .little);
    try std.testing.expectError(error.Malformed, StringArray.init(blob2));

    // Claimed count larger than the table that exists.
    const bad2 = try a.dupe(u8, blob);
    defer a.free(bad2);
    std.mem.writeInt(u32, bad2[4..8], 0xFFFF_FFF0, .little);
    try std.testing.expectError(error.Malformed, StringArray.init(bad2));

    // Wrong kind.
    try std.testing.expectError(error.WrongKind, F32Array.init(blob));
}

test "fixed arrays" {
    const a = std.testing.allocator;
    var raw: [8]u8 = undefined;
    std.mem.writeInt(u32, raw[0..4], @bitCast(@as(f32, 1.5)), .little);
    std.mem.writeInt(u32, raw[4..8], @bitCast(@as(f32, -2)), .little);
    const blob = try fixedBlob(a, .f32, 2, &raw);
    defer a.free(blob);
    const v = try F32Array.init(blob);
    try std.testing.expectEqual(@as(f32, 1.5), v.get(0));
    try std.testing.expectEqual(@as(f32, -2), v.get(1));
    try std.testing.expectError(error.Malformed, F32Array.init(blob[0 .. blob.len - 1]));
}
