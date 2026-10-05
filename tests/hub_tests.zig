//! End to end tests of `hk pull`, `hk list` and `hk rm` against a mock hub that can inject
//! faults: wrong checksums, dropped and truncated connections, rate limits, redirects, auth.

const std = @import("std");
const proc = @import("support/proc.zig");
const tiny = @import("support/tiny_model.zig");
const Hub = @import("support/mock_hub.zig").Hub;

const gpa = std.testing.allocator;
const io = std.testing.io;

fn contains(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}

/// A scratch directory per test: files the hub serves, and the hk home the client caches into.
const Scratch = struct {
    tmp: std.testing.TmpDir,
    base: []u8,

    fn init() !Scratch {
        const tmp = std.testing.tmpDir(.{});
        const base = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        return .{ .tmp = tmp, .base = base };
    }

    fn deinit(self: *Scratch) void {
        gpa.free(self.base);
        self.tmp.cleanup();
    }

    fn path(self: *const Scratch, comptime fmt: []const u8, args: anytype) ![]u8 {
        const rest = try std.fmt.allocPrint(gpa, fmt, args);
        defer gpa.free(rest);
        return std.fmt.allocPrint(gpa, "{s}/{s}", .{ self.base, rest });
    }
};

/// Runs `hk` with the hub as its endpoint. `home` is where it caches models.
fn runHk(hub: *const Hub, home: []const u8, token: ?[]const u8, args: []const []const u8) !proc.Result {
    var env = std.process.Environ.Map.init(gpa);
    defer env.deinit();
    var url_buf: [64]u8 = undefined;
    try env.put("HF_ENDPOINT", hub.url(&url_buf));
    try env.put("HK_HOME", home);
    try env.put("HK_HTTP_BACKOFF_MS", "20");
    if (token) |t| try env.put("HF_TOKEN", t);
    return proc.runEnv(gpa, io, &env, args, null);
}

/// Number of files with `suffix` anywhere under `dir`, and the name of the last one.
fn countFiles(dir: []const u8, suffix: []const u8, last: ?*[]u8) !usize {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var walker = try d.walk(gpa);
    defer walker.deinit();
    var n: usize = 0;
    while (try walker.next(io)) |e| {
        if (e.kind == .file and std.mem.endsWith(u8, e.basename, suffix)) {
            n += 1;
            if (last) |l| {
                if (l.len != 0) gpa.free(l.*);
                l.* = try gpa.dupe(u8, e.path);
            }
        }
    }
    return n;
}

fn writeFile(path: []const u8, bytes: []const u8) !void {
    var f = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, bytes);
}

const Fixture = struct {
    s: Scratch,
    models: []u8,
    home: []u8,

    /// Two tiny GGUF files in `models`, like a repository with two quants.
    fn gguf() !Fixture {
        var s = try Scratch.init();
        errdefer s.deinit();
        const models = try s.path("files", .{});
        errdefer gpa.free(models);
        try std.Io.Dir.cwd().createDirPath(io, models);
        const a = try s.path("files/tiny-f32.gguf", .{});
        defer gpa.free(a);
        const b = try s.path("files/tiny-Q8_0.gguf", .{});
        defer gpa.free(b);
        try tiny.makeTinyGguf(gpa, io, a, .{});
        try tiny.makeTinyGguf(gpa, io, b, .{ .q8_0 = true });
        const home = try s.path("hkhome", .{});
        return .{ .s = s, .models = models, .home = home };
    }

    fn hf(shards: usize) !Fixture {
        var s = try Scratch.init();
        errdefer s.deinit();
        const models = try s.path("repo", .{});
        errdefer gpa.free(models);
        try tiny.makeTinyHf(gpa, io, models, shards, .{});
        const home = try s.path("hkhome", .{});
        return .{ .s = s, .models = models, .home = home };
    }

    fn deinit(self: *Fixture) void {
        gpa.free(self.models);
        gpa.free(self.home);
        self.s.deinit();
    }

    fn cached(self: *const Fixture) !usize {
        return countFiles(self.home, ".hk", null);
    }

    fn partials(self: *const Fixture) !usize {
        return countFiles(self.home, ".partial", null);
    }
};

fn fileSize(path: []const u8) !u64 {
    const st = try std.Io.Dir.cwd().statFile(io, path, .{});
    return st.size;
}

test "pull converts and verifies" {
    var fx = try Fixture.gguf();
    defer fx.deinit();
    const hub = try Hub.start(gpa, io, fx.models, "acme", "tiny-GGUF");
    defer hub.stop();

    const r = try runHk(hub, fx.home, null, &.{ "pull", "acme/tiny-GGUF:Q8_0" });
    defer r.deinit(gpa);
    try std.testing.expect(r.ok());
    try std.testing.expect(contains(r.stderr, "checksum verified"));
    var last: []u8 = &.{};
    defer if (last.len != 0) gpa.free(last);
    try std.testing.expectEqual(@as(usize, 1), try countFiles(fx.home, ".hk", &last));
    try std.testing.expect(std.mem.endsWith(u8, last, "tiny-Q8_0.hk"));
    try std.testing.expectEqual(@as(usize, 0), try fx.partials());

    // The tokenizer and weights came through: tokenizing works against the converted file.
    const full = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ fx.home, last });
    defer gpa.free(full);
    const t = try proc.run(gpa, io, &.{ "tokenize", full, "hello" }, null);
    defer t.deinit(gpa);
    try std.testing.expect(t.ok());
    try std.testing.expect(contains(t.stderr, "tokens"));
}

test "a second pull uses the cache" {
    var fx = try Fixture.gguf();
    defer fx.deinit();
    const hub = try Hub.start(gpa, io, fx.models, "acme", "tiny-GGUF");
    defer hub.stop();

    const first = try runHk(hub, fx.home, null, &.{ "pull", "acme/tiny-GGUF:Q8_0" });
    first.deinit(gpa);
    const downloads = hub.count("/resolve/");
    const r = try runHk(hub, fx.home, null, &.{ "pull", "acme/tiny-GGUF:Q8_0" });
    defer r.deinit(gpa);
    try std.testing.expect(r.ok() and contains(r.stderr, "already in the cache"));
    // Only the listing was fetched again, not the model.
    try std.testing.expectEqual(downloads, hub.count("/resolve/"));
}

test "a wrong checksum discards everything" {
    var fx = try Fixture.gguf();
    defer fx.deinit();
    const hub = try Hub.start(gpa, io, fx.models, "acme", "tiny-GGUF");
    defer hub.stop();
    hub.faults.wrong_sha = true;
    const r = try runHk(hub, fx.home, null, &.{ "pull", "acme/tiny-GGUF:Q8_0" });
    defer r.deinit(gpa);
    try std.testing.expect(r.code != 0 and contains(r.stderr, "checksum"));
    try std.testing.expectEqual(@as(usize, 0), try fx.cached());
    try std.testing.expectEqual(@as(usize, 0), try fx.partials());
}

test "a dropped connection resumes from the last byte" {
    var fx = try Fixture.gguf();
    defer fx.deinit();
    const p = try fx.s.path("files/tiny-Q8_0.gguf", .{});
    defer gpa.free(p);
    const size = try fileSize(p);
    const hub = try Hub.start(gpa, io, fx.models, "acme", "tiny-GGUF");
    defer hub.stop();
    hub.faults.drop_after = size / 3;
    const r = try runHk(hub, fx.home, null, &.{ "pull", "acme/tiny-GGUF:Q8_0" });
    defer r.deinit(gpa);
    try std.testing.expect(r.ok());
    try std.testing.expect(contains(r.stderr, "checksum verified"));
    // At least two file requests, one of them asking for a range that does not start at zero.
    try std.testing.expect(hub.count("/resolve/") >= 2);
    hub.mutex.lockUncancelable(io);
    defer hub.mutex.unlock(io);
    var resumed = false;
    for (hub.log.items) |e| {
        if (std.mem.indexOf(u8, e.path, "/resolve/") != null) {
            if (e.range) |rg| {
                if (!std.mem.eql(u8, rg, "bytes=0-")) resumed = true;
            }
        }
    }
    try std.testing.expect(resumed);
}

test "a truncated body fails cleanly" {
    var fx = try Fixture.gguf();
    defer fx.deinit();
    const hub = try Hub.start(gpa, io, fx.models, "acme", "tiny-GGUF");
    defer hub.stop();
    hub.faults.truncate = true;
    const r = try runHk(hub, fx.home, null, &.{ "pull", "acme/tiny-GGUF:Q8_0" });
    defer r.deinit(gpa);
    try std.testing.expect(r.code != 0);
    try std.testing.expectEqual(@as(usize, 0), try fx.cached());
    try std.testing.expectEqual(@as(usize, 0), try fx.partials());
}

test "missing repository, no matching file, invalid name" {
    var fx = try Fixture.gguf();
    defer fx.deinit();
    const hub = try Hub.start(gpa, io, fx.models, "acme", "tiny-GGUF");
    defer hub.stop();
    {
        const r = try runHk(hub, fx.home, null, &.{ "pull", "nobody/nothing" });
        defer r.deinit(gpa);
        try std.testing.expect(r.code != 0 and contains(r.stderr, "not found"));
    }
    {
        const r = try runHk(hub, fx.home, null, &.{ "pull", "acme/tiny-GGUF:Q9_9" });
        defer r.deinit(gpa);
        try std.testing.expect(r.code != 0 and contains(r.stderr, "no file"));
    }
    {
        const before = hub.total();
        const r = try runHk(hub, fx.home, null, &.{ "pull", "../etc/passwd" });
        defer r.deinit(gpa);
        try std.testing.expect(r.code != 0);
        // Rejected before any request was made.
        try std.testing.expectEqual(before, hub.total());
    }
}

test "unauthorized explains the token, and the token never follows a redirect" {
    var fx = try Fixture.gguf();
    defer fx.deinit();
    const hub = try Hub.start(gpa, io, fx.models, "acme", "tiny-GGUF");
    defer hub.stop();
    hub.faults.require_token = "s3cret";
    {
        const r = try runHk(hub, fx.home, null, &.{ "pull", "acme/tiny-GGUF:Q8_0" });
        defer r.deinit(gpa);
        try std.testing.expect(r.code != 0 and contains(r.stderr, "HF_TOKEN"));
    }
    {
        const r = try runHk(hub, fx.home, "s3cret", &.{ "pull", "acme/tiny-GGUF:Q8_0" });
        defer r.deinit(gpa);
        try std.testing.expect(r.ok());
    }

    // The mock answers file requests with a redirect to "localhost", a different host name than
    // 127.0.0.1, and records the Authorization header of every request it sees.
    var fx2 = try Fixture.gguf();
    defer fx2.deinit();
    const hub2 = try Hub.start(gpa, io, fx2.models, "acme", "tiny-GGUF");
    defer hub2.stop();
    hub2.faults.redirect_host = "localhost";
    const r = try runHk(hub2, fx2.home, "s3cret", &.{ "pull", "acme/tiny-GGUF:Q8_0" });
    defer r.deinit(gpa);
    try std.testing.expect(r.ok());
    hub2.mutex.lockUncancelable(io);
    defer hub2.mutex.unlock(io);
    var api: usize = 0;
    var second_hop: usize = 0;
    for (hub2.log.items) |e| {
        if (std.mem.indexOf(u8, e.path, "/api/") != null) {
            api += 1;
            try std.testing.expect(e.auth != null and std.mem.eql(u8, e.auth.?, "Bearer s3cret"));
        }
        if (std.mem.startsWith(u8, e.path, "/cdn/")) {
            second_hop += 1;
            try std.testing.expect(e.auth == null);
        }
    }
    try std.testing.expect(api > 0 and second_hop > 0);
}

test "a rate limit is retried and then given up on" {
    var fx = try Fixture.gguf();
    defer fx.deinit();
    const hub = try Hub.start(gpa, io, fx.models, "acme", "tiny-GGUF");
    defer hub.stop();
    hub.faults.status_path = "/api/models/acme/tiny-GGUF/revision/main";
    hub.faults.status_code = 429;
    const r = try runHk(hub, fx.home, null, &.{ "pull", "acme/tiny-GGUF:Q8_0" });
    defer r.deinit(gpa);
    try std.testing.expect(r.code != 0);
    try std.testing.expect(hub.count("/api/") > 1);
}

test "list and rm" {
    var fx = try Fixture.gguf();
    defer fx.deinit();
    const hub = try Hub.start(gpa, io, fx.models, "acme", "tiny-GGUF");
    defer hub.stop();
    {
        const r = try runHk(hub, fx.home, null, &.{ "pull", "acme/tiny-GGUF:Q8_0" });
        defer r.deinit(gpa);
        try std.testing.expect(r.ok());
    }
    {
        const r = try runHk(hub, fx.home, null, &.{"list"});
        defer r.deinit(gpa);
        try std.testing.expect(contains(r.stderr, "acme/tiny-GGUF") and contains(r.stderr, "tiny-Q8_0.hk"));
    }
    {
        const r = try runHk(hub, fx.home, null, &.{ "rm", "acme/tiny-GGUF" });
        defer r.deinit(gpa);
        try std.testing.expect(r.ok());
    }
    try std.testing.expectEqual(@as(usize, 0), try fx.cached());
}

fn mib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1048576.0;
}

test "a safetensors repository is converted while it streams" {
    var fx = try Fixture.hf(1);
    defer fx.deinit();
    const hub = try Hub.start(gpa, io, fx.models, "acme", "tiny-HF");
    defer hub.stop();
    const r = try runHk(hub, fx.home, null, &.{ "pull", "acme/tiny-HF" });
    defer r.deinit(gpa);
    try std.testing.expect(r.ok());
    try std.testing.expect(contains(r.stderr, "checksum verified"));
    const p = try fx.s.path("repo/model.safetensors", .{});
    defer gpa.free(p);
    var want_buf: [64]u8 = undefined;
    const want = try std.fmt.bufPrint(&want_buf, "downloaded {d:.1} MiB", .{mib(try fileSize(p))});
    try std.testing.expect(contains(r.stderr, want));
    var last: []u8 = &.{};
    defer if (last.len != 0) gpa.free(last);
    try std.testing.expectEqual(@as(usize, 1), try countFiles(fx.home, ".hk", &last));
    try std.testing.expectEqual(@as(usize, 0), try fx.partials());
    const full = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ fx.home, last });
    defer gpa.free(full);
    const t = try proc.run(gpa, io, &.{ "tokenize", full, "hello" }, null);
    defer t.deinit(gpa);
    try std.testing.expect(t.ok());
}

test "a sharded safetensors repository, and its failure modes" {
    var fx = try Fixture.hf(3);
    defer fx.deinit();
    const hub = try Hub.start(gpa, io, fx.models, "acme", "tiny-HF");
    defer hub.stop();

    {
        const r = try runHk(hub, fx.home, null, &.{ "pull", "acme/tiny-HF" });
        defer r.deinit(gpa);
        try std.testing.expect(r.ok());
        try std.testing.expect(contains(r.stderr, "checksum verified"));
        var total: u64 = 0;
        for (1..4) |i| {
            const p = try fx.s.path("repo/model-{d:0>5}-of-00003.safetensors", .{i});
            defer gpa.free(p);
            total += try fileSize(p);
        }
        var want_buf: [64]u8 = undefined;
        const want = try std.fmt.bufPrint(&want_buf, "downloaded {d:.1} MiB", .{mib(total)});
        try std.testing.expect(contains(r.stderr, want));
        try std.testing.expectEqual(@as(usize, 1), try fx.cached());
    }

    // Each failure mode pulls into a fresh cache.
    {
        hub.faults.no_sha = "model-00002-of-00003.safetensors";
        const home = try fx.s.path("h_nosha", .{});
        defer gpa.free(home);
        const r = try runHk(hub, home, null, &.{ "pull", "acme/tiny-HF" });
        defer r.deinit(gpa);
        try std.testing.expect(r.ok());
        try std.testing.expect(!contains(r.stderr, "checksum verified"));
        try std.testing.expect(contains(r.stderr, "no checksum available"));
        hub.faults.no_sha = null;
    }
    {
        hub.faults.wrong_sha = true;
        const home = try fx.s.path("h_wrong", .{});
        defer gpa.free(home);
        const r = try runHk(hub, home, null, &.{ "pull", "acme/tiny-HF" });
        defer r.deinit(gpa);
        try std.testing.expect(r.code != 0 and contains(r.stderr, "checksum"));
        try std.testing.expectEqual(@as(usize, 0), try countFiles(home, ".hk", null));
        try std.testing.expectEqual(@as(usize, 0), try countFiles(home, ".partial", null));
        hub.faults.wrong_sha = false;
    }
    {
        hub.faults.truncate = true;
        const home = try fx.s.path("h_trunc", .{});
        defer gpa.free(home);
        const r = try runHk(hub, home, null, &.{ "pull", "acme/tiny-HF" });
        defer r.deinit(gpa);
        try std.testing.expect(r.code != 0);
        try std.testing.expectEqual(@as(usize, 0), try countFiles(home, ".hk", null));
        try std.testing.expectEqual(@as(usize, 0), try countFiles(home, ".partial", null));
    }
}

test "a split gguf is refused with a clear message" {
    var s = try Scratch.init();
    defer s.deinit();
    const dir = try s.path("split", .{});
    defer gpa.free(dir);
    try std.Io.Dir.cwd().createDirPath(io, dir);
    for ([_][]const u8{ "m-Q4_K_M-00001-of-00002.gguf", "m-Q4_K_M-00002-of-00002.gguf" }) |n| {
        const p = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir, n });
        defer gpa.free(p);
        try writeFile(p, "GGUF");
    }
    const home = try s.path("hkhome", .{});
    defer gpa.free(home);
    const hub = try Hub.start(gpa, io, dir, "acme", "tiny-GGUF");
    defer hub.stop();
    const r = try runHk(hub, home, null, &.{ "pull", "acme/tiny-GGUF" });
    defer r.deinit(gpa);
    try std.testing.expect(r.code != 0 and contains(r.stderr, "split"));
    try std.testing.expectEqual(@as(usize, 0), hub.count("/resolve/"));
}
