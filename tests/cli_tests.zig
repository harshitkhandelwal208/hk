//! Argument handling and basic behaviour of `hk run`, `hk chat` and the small commands, run
//! against the built executable and a tiny generated model.

const std = @import("std");
const proc = @import("support/proc.zig");
const tiny = @import("support/tiny_model.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;

fn hk(args: []const []const u8, stdin: ?[]const u8) !proc.Result {
    return proc.run(gpa, io, args, stdin);
}

fn contains(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}

/// A scratch directory with the two tiny models of the tests, built once.
const Models = struct {
    tmp: std.testing.TmpDir,
    plain: []u8,
    templated: []u8,

    fn make() !Models {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        defer gpa.free(dir);
        var out: Models = .{ .tmp = tmp, .plain = undefined, .templated = undefined };
        const names = [_]struct { []const u8, ?[]const u8 }{ .{ "plain", null }, .{ "templated", tiny.chatml } };
        var made: [2][]u8 = undefined;
        for (names, 0..) |n, i| {
            const g = try std.fmt.allocPrint(gpa, "{s}/{s}.gguf", .{ dir, n[0] });
            defer gpa.free(g);
            try tiny.makeTinyGguf(gpa, io, g, .{ .chat_template = n[1] });
            made[i] = try std.fmt.allocPrint(gpa, "{s}/{s}.hk", .{ dir, n[0] });
            const r = try hk(&.{ "convert-gguf", g, made[i] }, null);
            defer r.deinit(gpa);
            if (!r.ok()) {
                std.debug.print("convert-gguf failed: {s}\n", .{r.stderr});
                return error.ConvertFailed;
            }
        }
        out.plain = made[0];
        out.templated = made[1];
        return out;
    }

    fn deinit(self: *Models) void {
        gpa.free(self.plain);
        gpa.free(self.templated);
        self.tmp.cleanup();
    }
};

test "bad run arguments fail before the model is touched" {
    const cases = [_]struct { args: []const []const u8, needle: []const u8 }{
        .{ .args = &.{"--bogus"}, .needle = "unknown argument '--bogus'" },
        .{ .args = &.{ "--temp", "warm" }, .needle = "not a valid value for --temp" },
        .{ .args = &.{ "--temp", "-1" }, .needle = "not a valid value for --temp" },
        .{ .args = &.{"--temp"}, .needle = "--temp needs a value" },
        .{ .args = &.{ "-n", "0" }, .needle = "not a valid value for -n" },
        .{ .args = &.{ "-n", "ten" }, .needle = "not a valid value for -n" },
        .{ .args = &.{ "--top-p", "1.5" }, .needle = "not a valid value for --top-p" },
        .{ .args = &.{ "--repeat-penalty", "0" }, .needle = "not a valid value for --repeat-penalty" },
        .{ .args = &.{ "--seed", "x" }, .needle = "not a valid value for --seed" },
        .{ .args = &.{ "a", "b" }, .needle = "more than one prompt" },
        .{ .args = &.{ "-p", "a", "b" }, .needle = "more than one prompt" },
    };
    for (cases) |c| {
        var argv: std.ArrayList([]const u8) = .empty;
        defer argv.deinit(gpa);
        try argv.appendSlice(gpa, &.{ "run", "/nonexistent/model.hk" });
        try argv.appendSlice(gpa, c.args);
        const r = try hk(argv.items, null);
        defer r.deinit(gpa);
        try std.testing.expect(r.code != 0);
        if (!contains(r.stderr, c.needle)) {
            std.debug.print("expected '{s}' in: {s}\n", .{ c.needle, r.stderr });
            return error.MissingMessage;
        }
        // If the error is about the flag, the flag was checked before the model path was.
        try std.testing.expect(!contains(r.stderr, "nonexistent"));
    }
}

test "chat rejects run only flags" {
    const flags = [_][]const []const u8{ &.{ "-p", "hi" }, &.{"--chat"}, &.{"hello"} };
    for (flags) |f| {
        var argv: std.ArrayList([]const u8) = .empty;
        defer argv.deinit(gpa);
        try argv.appendSlice(gpa, &.{ "chat", "/nonexistent/model.hk" });
        try argv.appendSlice(gpa, f);
        const r = try hk(argv.items, null);
        defer r.deinit(gpa);
        try std.testing.expect(r.code != 0 and contains(r.stderr, "unknown argument"));
    }
}

test "generation behaves" {
    var m = try Models.make();
    defer m.deinit();

    {
        const r = try hk(&.{ "run", m.plain }, null);
        defer r.deinit(gpa);
        try std.testing.expect(r.code != 0 and contains(r.stderr, "no prompt"));
    }

    // Greedy runs repeat exactly and every way of giving the prompt agrees. Text is on stdout,
    // statistics on stderr.
    {
        const a = try hk(&.{ "run", m.plain, "hello hello", "--temp", "0", "-n", "12" }, null);
        defer a.deinit(gpa);
        const b = try hk(&.{ "run", m.plain, "-p", "hello hello", "--temp", "0", "-n", "12" }, null);
        defer b.deinit(gpa);
        const c = try hk(&.{ "run", m.plain, "hello hello", "--temp", "0", "-n", "12" }, null);
        defer c.deinit(gpa);
        try std.testing.expect(a.ok());
        try std.testing.expectEqualSlices(u8, a.stdout, b.stdout);
        try std.testing.expectEqualSlices(u8, a.stdout, c.stdout);
        try std.testing.expect(std.mem.startsWith(u8, a.stdout, "hello hello"));
        try std.testing.expect(!contains(a.stdout, "generated") and contains(a.stderr, "generated"));
    }

    // The same seed repeats; a different seed does not.
    {
        const base = [_][]const u8{ "run", m.plain, "hello", "--temp", "1.5", "--top-k", "0", "--top-p", "1", "--min-p", "0", "-n", "24" };
        const x = try hk(&(base ++ [_][]const u8{ "--seed", "5" }), null);
        defer x.deinit(gpa);
        const y = try hk(&(base ++ [_][]const u8{ "--seed", "5" }), null);
        defer y.deinit(gpa);
        const z = try hk(&(base ++ [_][]const u8{ "--seed", "6" }), null);
        defer z.deinit(gpa);
        try std.testing.expectEqualSlices(u8, x.stdout, y.stdout);
        try std.testing.expect(!std.mem.eql(u8, x.stdout, z.stdout));
    }

    // --chat needs a template, and the template's own markers are fed in and echoed first.
    {
        const r = try hk(&.{ "run", m.plain, "hi", "--chat", "-n", "4" }, null);
        defer r.deinit(gpa);
        try std.testing.expect(r.code != 0 and contains(r.stderr, "no chat template"));
        const ok = try hk(&.{ "run", m.templated, "hi", "--chat", "--temp", "0", "-n", "4" }, null);
        defer ok.deinit(gpa);
        try std.testing.expect(ok.ok());
        try std.testing.expect(std.mem.startsWith(u8, ok.stdout, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"));
    }

    // Interactive chat answers and exits on an empty line.
    {
        const r = try hk(&.{ "chat", m.templated, "--temp", "0", "-n", "6" }, "hello\n\n");
        defer r.deinit(gpa);
        try std.testing.expect(r.ok());
        try std.testing.expect(contains(r.stderr, "generated"));
    }

    // A bad token id is rejected by name.
    {
        const r = try hk(&.{ "detokenize", m.plain, "1", "two", "3" }, null);
        defer r.deinit(gpa);
        try std.testing.expect(r.code != 0 and contains(r.stderr, "'two' is not a token id"));
    }
}

test "misuse exits non zero except bare help" {
    const cases = [_][]const []const u8{
        &.{},
        &.{"tokenize"},
        &.{ "tokenize", "/nonexistent.hk" },
        &.{"detokenize"},
        &.{"inspect"},
        &.{"convert-gguf"},
        &.{"no-such-command"},
    };
    for (cases, 0..) |args, i| {
        const r = try hk(args, null);
        defer r.deinit(gpa);
        if (i == 0) {
            try std.testing.expect(r.ok() and contains(r.stderr, "Usage"));
        } else {
            try std.testing.expect(r.code != 0);
        }
    }
}
