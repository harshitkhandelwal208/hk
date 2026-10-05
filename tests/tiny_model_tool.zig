//! Writes the tiny test model used by the end to end tests, for tests written in other languages.
//!
//!   hk-tiny-model out.gguf [--chat]

const std = @import("std");
const tiny = @import("support/tiny_model.zig");

pub fn main(init: std.process.Init) !void {
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer it.deinit();
    _ = it.next();
    const out = it.next() orelse {
        std.debug.print("usage: hk-tiny-model out.gguf [--chat]\n", .{});
        return error.BadUsage;
    };
    const chat = if (it.next()) |a| std.mem.eql(u8, a, "--chat") else false;
    try tiny.makeTinyGguf(init.gpa, init.io, out, .{ .chat_template = if (chat) tiny.chatml else null });
}
