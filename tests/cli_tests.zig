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

// -- container integrity: `hk verify`, `hk repair`, sealed metadata edits -------------------------

fn exists(path: []const u8) bool {
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

fn fileSize(path: []const u8) !u64 {
    var f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    return (try f.stat(io)).size;
}

/// Applies `mask`/`value` to one byte: new = (old & ~clear) ^ flip.
fn editByte(path: []const u8, offset: u64, clear: u8, flip: u8) !void {
    var f = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_write });
    defer f.close(io);
    var b: [1]u8 = undefined;
    if (try f.readPositionalAll(io, &b, offset) != 1) return error.UnexpectedEof;
    b[0] = (b[0] & ~clear) ^ flip;
    try f.writePositionalAll(io, &b, offset);
}

fn tamper(path: []const u8) !void {
    try editByte(path, (try fileSize(path)) - 1, 0, 0xFF);
}

/// Clears header flag bit 9 (the SHA-256 seal) to make a file look like a pre-seal v1 container.
fn makeLegacy(path: []const u8) !void {
    try editByte(path, 9, 0x02, 0);
}

fn writeJournal(path: []const u8) !void {
    const journal = try std.fmt.allocPrint(gpa, "{s}.hkmeta-journal", .{path});
    defer gpa.free(journal);
    var f = try std.Io.Dir.cwd().createFile(io, journal, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, "HKMJ-torn");
}

fn journalExists(path: []const u8) !bool {
    const journal = try std.fmt.allocPrint(gpa, "{s}.hkmeta-journal", .{path});
    defer gpa.free(journal);
    return exists(journal);
}

test "converted models are sealed and verify cleanly" {
    var m = try Models.make();
    defer m.deinit();
    const r = try hk(&.{ "verify", m.plain }, null);
    defer r.deinit(gpa);
    try std.testing.expect(r.ok());
    try std.testing.expect(contains(r.stderr, "SHA-256 container checksum verified"));
}

test "verify exits non zero on a tampered file and metadata set refuses to bless it" {
    var m = try Models.make();
    defer m.deinit();
    try tamper(m.plain);

    const v = try hk(&.{ "verify", m.plain }, null);
    defer v.deinit(gpa);
    try std.testing.expect(!v.ok());
    try std.testing.expect(contains(v.stderr, "container checksum mismatch"));

    const set = try hk(&.{ "metadata", "set", m.plain, "k", "v" }, null);
    defer set.deinit(gpa);
    try std.testing.expect(!set.ok());

    // The refused edit must not have re-sealed the damage.
    const again = try hk(&.{ "verify", m.plain }, null);
    defer again.deinit(gpa);
    try std.testing.expect(!again.ok());
    try std.testing.expect(!(try journalExists(m.plain)));
}

test "a legacy unsealed file verifies with a warning and is sealed by its first edit" {
    var m = try Models.make();
    defer m.deinit();
    try makeLegacy(m.plain);

    const v = try hk(&.{ "verify", m.plain }, null);
    defer v.deinit(gpa);
    try std.testing.expect(v.ok());
    try std.testing.expect(contains(v.stderr, "legacy v1 file"));

    const set = try hk(&.{ "metadata", "set", m.plain, "owner", "me" }, null);
    defer set.deinit(gpa);
    try std.testing.expect(set.ok());

    const sealed = try hk(&.{ "verify", m.plain }, null);
    defer sealed.deinit(gpa);
    try std.testing.expect(sealed.ok());
    try std.testing.expect(contains(sealed.stderr, "SHA-256 container checksum verified"));
}

test "metadata set keeps the seal valid" {
    var m = try Models.make();
    defer m.deinit();
    const set = try hk(&.{ "metadata", "set", m.plain, "general.description", "a much longer description than before" }, null);
    defer set.deinit(gpa);
    try std.testing.expect(set.ok());
    const v = try hk(&.{ "verify", m.plain }, null);
    defer v.deinit(gpa);
    try std.testing.expect(v.ok());
    try std.testing.expect(contains(v.stderr, "SHA-256 container checksum verified"));
}

test "repair reports honestly in every state" {
    var m = try Models.make();
    defer m.deinit();

    // Healthy and nothing to do.
    {
        const r = try hk(&.{ "repair", m.plain }, null);
        defer r.deinit(gpa);
        try std.testing.expect(r.ok());
        try std.testing.expect(contains(r.stderr, "No interrupted edit found"));
        try std.testing.expect(contains(r.stderr, "REPAIR COMPLETE"));
    }

    // A journal is left behind: verify flags it, repair discards it only because the seal proves the file intact.
    try writeJournal(m.plain);
    {
        const v = try hk(&.{ "verify", m.plain }, null);
        defer v.deinit(gpa);
        try std.testing.expect(!v.ok());
        try std.testing.expect(contains(v.stderr, "hk repair"));
    }
    {
        const r = try hk(&.{ "repair", m.plain }, null);
        defer r.deinit(gpa);
        try std.testing.expect(r.ok());
        try std.testing.expect(contains(r.stderr, "Discarded an incomplete edit journal"));
        try std.testing.expect(!(try journalExists(m.plain)));
    }

    // Damaged file plus a torn journal: repair must refuse to guess and must keep the journal.
    try tamper(m.plain);
    try writeJournal(m.plain);
    {
        const r = try hk(&.{ "repair", m.plain }, null);
        defer r.deinit(gpa);
        try std.testing.expect(!r.ok());
        try std.testing.expect(contains(r.stderr, "cannot be applied safely"));
        try std.testing.expect(try journalExists(m.plain));
    }

    // Damaged file, no journal: repair says so and does not re-seal.
    {
        const j = try std.fmt.allocPrint(gpa, "{s}.hkmeta-journal", .{m.plain});
        defer gpa.free(j);
        try std.Io.Dir.cwd().deleteFile(io, j);
        const r = try hk(&.{ "repair", m.plain }, null);
        defer r.deinit(gpa);
        try std.testing.expect(!r.ok());
        try std.testing.expect(contains(r.stderr, "REPAIR INCOMPLETE"));
        const v = try hk(&.{ "verify", m.plain }, null);
        defer v.deinit(gpa);
        try std.testing.expect(!v.ok());
    }
}

// -- backend selection ---------------------------------------------------------------------------

test "backend flag: cpu runs, an unknown value and a missing backend are refused with a reason" {
    var m = try Models.make();
    defer m.deinit();

    {
        const r = try hk(&.{ "run", m.plain, "hi", "-n", "2", "--temp", "0", "--backend", "cpu" }, null);
        defer r.deinit(gpa);
        try std.testing.expect(r.ok());
        try std.testing.expect(!contains(r.stderr, "[gpu]"));
    }
    {
        const r = try hk(&.{ "run", m.plain, "hi", "--backend", "bogus" }, null);
        defer r.deinit(gpa);
        try std.testing.expect(!r.ok());
        try std.testing.expect(contains(r.stderr, "takes auto, cpu, vulkan, cuda or rocm"));
    }
    {
        // Naming a device means "use it or fail": either it ran on CUDA, or it failed saying CUDA was the problem.
        // It must never quietly run on the CPU.
        const r = try hk(&.{ "run", m.plain, "hi", "-n", "2", "--temp", "0", "--backend", "cuda" }, null);
        defer r.deinit(gpa);
        if (r.ok()) {
            try std.testing.expect(contains(r.stderr, "(CUDA)"));
        } else {
            try std.testing.expect(contains(r.stderr, "CUDA"));
        }
    }
    {
        // The same for ROCm: on a machine without an AMD GPU it must fail naming ROCm, never run on the CPU.
        const r = try hk(&.{ "run", m.plain, "hi", "-n", "2", "--temp", "0", "--backend", "rocm" }, null);
        defer r.deinit(gpa);
        if (r.ok()) {
            try std.testing.expect(contains(r.stderr, "(ROCm)"));
        } else {
            try std.testing.expect(contains(r.stderr, "ROCm"));
        }
    }
}

test "HK_BACKEND is validated and overridden by the flag" {
    var m = try Models.make();
    defer m.deinit();
    var env = std.process.Environ.Map.init(gpa);
    defer env.deinit();
    try env.put("HK_BACKEND", "nonsense");
    {
        const r = try proc.runEnv(gpa, io, &env, &.{ "run", m.plain, "hi", "-n", "2" }, null);
        defer r.deinit(gpa);
        try std.testing.expect(!r.ok());
        try std.testing.expect(contains(r.stderr, "HK_BACKEND takes"));
    }
    {
        const r = try proc.runEnv(gpa, io, &env, &.{ "run", m.plain, "hi", "-n", "2", "--temp", "0", "--backend", "cpu" }, null);
        defer r.deinit(gpa);
        try std.testing.expect(r.ok());
    }
}
