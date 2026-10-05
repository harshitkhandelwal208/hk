//! Generates the Unicode general category table used by the tokenizer's regex engine.
//!
//!   zig run tools/gen_unicode_tables.zig -- UnicodeData.txt src/tokenizer/unicode_cat.bin
//!
//! UnicodeData.txt comes from https://www.unicode.org/Public/<version>/ucd/UnicodeData.txt. The
//! committed table was made from Unicode 16.0.0.
//!
//! Writes a sorted list of ranges, each 9 bytes little endian:
//!     u32 first_codepoint, u32 last_codepoint, u8 category
//! where category is an index into `categories` (two letter general category codes). Code
//! points that UnicodeData.txt does not list are unassigned (Cn) and match no category class.

const std = @import("std");

const categories = [_][]const u8{
    "Lu", "Ll", "Lt", "Lm", "Lo",
    "Mn", "Mc", "Me", "Nd", "Nl",
    "No", "Pc", "Pd", "Ps", "Pe",
    "Pi", "Pf", "Po", "Sm", "Sc",
    "Sk", "So", "Zs", "Zl", "Zp",
    "Cc", "Cf", "Cs", "Co",
};

fn categoryIndex(name: []const u8) ?u8 {
    for (categories, 0..) |c, i| {
        if (std.mem.eql(u8, c, name)) return @intCast(i);
    }
    return null;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer it.deinit();
    _ = it.next();
    const in_path = it.next() orelse return usage();
    const out_path = it.next() orelse return usage();

    const text = try std.Io.Dir.cwd().readFileAlloc(io, in_path, gpa, .limited(16 << 20));
    defer gpa.free(text);

    // 0xFF marks unassigned.
    const cat = try gpa.alloc(u8, 0x110000);
    defer gpa.free(cat);
    @memset(cat, 0xFF);

    var range_first: ?u32 = null;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var f = std.mem.splitScalar(u8, line, ';');
        const cp_s = f.next() orelse continue;
        const name = f.next() orelse continue;
        const gc = f.next() orelse continue;
        const cp = try std.fmt.parseInt(u32, cp_s, 16);
        const idx = categoryIndex(gc) orelse {
            std.debug.print("unknown general category '{s}' at U+{X}\n", .{ gc, cp });
            return error.UnknownCategory;
        };
        if (std.mem.endsWith(u8, name, ", First>")) {
            range_first = cp;
        } else if (std.mem.endsWith(u8, name, ", Last>")) {
            const first = range_first orelse return error.RangeWithoutStart;
            for (first..cp + 1) |c| cat[c] = idx;
            range_first = null;
        } else {
            cat[cp] = idx;
        }
    }

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    var n_ranges: usize = 0;
    var start: ?u32 = null;
    var prev: u8 = 0xFF;
    var cp: u32 = 0;
    while (cp <= 0x110000) : (cp += 1) {
        const c: u8 = if (cp == 0x110000) 0xFF else cat[cp];
        if (start != null and c != prev) {
            var rec: [9]u8 = undefined;
            std.mem.writeInt(u32, rec[0..4], start.?, .little);
            std.mem.writeInt(u32, rec[4..8], cp - 1, .little);
            rec[8] = prev;
            try out.appendSlice(gpa, &rec);
            n_ranges += 1;
            start = null;
        }
        if (c != 0xFF and start == null) {
            start = cp;
        }
        prev = c;
    }

    var f = try std.Io.Dir.cwd().createFile(io, out_path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, out.items);
    std.debug.print("{s}: {d} ranges, {d} bytes\n", .{ out_path, n_ranges, out.items.len });
}

fn usage() error{BadUsage} {
    std.debug.print("usage: gen_unicode_tables UnicodeData.txt out.bin\n", .{});
    return error.BadUsage;
}
