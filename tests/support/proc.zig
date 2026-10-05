//! Runs the hk executable and collects what it printed. Output is kept as bytes: a model with
//! random weights emits arbitrary byte sequences, so it is never decoded as text.

const std = @import("std");

pub const Result = struct {
    code: i32,
    stdout: []u8,
    stderr: []u8,

    pub fn deinit(self: Result, gpa: std.mem.Allocator) void {
        gpa.free(self.stdout);
        gpa.free(self.stderr);
    }

    pub fn ok(self: Result) bool {
        return self.code == 0;
    }
};

/// Path of the hk executable under test, installed by the build before the tests run.
pub const exe_path = @import("test_options").hk_exe;

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    args: []const []const u8,
    stdin_bytes: ?[]const u8,
) !Result {
    return runEnv(gpa, io, null, args, stdin_bytes);
}

pub fn runEnv(
    gpa: std.mem.Allocator,
    io: std.Io,
    env: ?*const std.process.Environ.Map,
    args: []const []const u8,
    stdin_bytes: ?[]const u8,
) !Result {
    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    try argv.append(gpa, exe_path);
    try argv.appendSlice(gpa, args);

    var stdin_file: ?std.Io.File = null;
    defer if (stdin_file) |f| f.close(io);
    var stdin_path_buf: [64]u8 = undefined;
    var stdin_path: ?[]const u8 = null;
    defer if (stdin_path) |p| std.Io.Dir.cwd().deleteFile(io, p) catch {};
    if (stdin_bytes) |bytes| {
        const p = try std.fmt.bufPrint(&stdin_path_buf, ".hk-test-stdin-{d}", .{std.Thread.getCurrentId()});
        var wf = try std.Io.Dir.cwd().createFile(io, p, .{});
        try wf.writeStreamingAll(io, bytes);
        wf.close(io);
        stdin_path = p;
        stdin_file = try std.Io.Dir.cwd().openFile(io, p, .{});
    }

    var child = try std.process.spawn(io, .{
        .argv = argv.items,
        .environ_map = env,
        .stdin = if (stdin_file) |f| .{ .file = f } else .ignore,
        .stdout = .pipe,
        .stderr = .pipe,
    });
    defer child.kill(io);

    var buffer: std.Io.File.MultiReader.Buffer(2) = undefined;
    var multi: std.Io.File.MultiReader = undefined;
    multi.init(gpa, io, buffer.toStreams(), &.{ child.stdout.?, child.stderr.? });
    defer multi.deinit();
    while (multi.fill(64, .none)) |_| {} else |err| switch (err) {
        error.EndOfStream => {},
        else => |e| return e,
    }
    try multi.checkAnyError();
    const term = try child.wait(io);
    const out = try multi.toOwnedSlice(0);
    errdefer gpa.free(out);
    const errs = try multi.toOwnedSlice(1);
    const code: i32 = switch (term) {
        .exited => |c| c,
        else => -1,
    };
    return .{ .code = code, .stdout = out, .stderr = errs };
}
