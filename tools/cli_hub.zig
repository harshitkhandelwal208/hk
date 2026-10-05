//! Command line front end for the model hub: pull, search, list, rm.

const std = @import("std");
const hk = @import("hk");

const hub = hk.hub;

pub const Env = struct {
    map: *const std.process.Environ.Map,

    pub fn get(self: Env, key: []const u8) ?[]const u8 {
        return self.map.get(key);
    }

    /// Root of the model cache: $HK_HOME, else $XDG_CACHE_HOME/hk, else ~/.cache/hk.
    pub fn cacheDir(self: Env, allocator: std.mem.Allocator) ![]u8 {
        if (self.get("HK_HOME")) |h| return allocator.dupe(u8, h);
        if (self.get("XDG_CACHE_HOME")) |x| return std.fmt.allocPrint(allocator, "{s}/hk", .{x});
        if (self.get("HOME")) |h| return std.fmt.allocPrint(allocator, "{s}/.cache/hk", .{h});
        return error.NoHomeDirectory;
    }
};

/// Token from the environment, then from the file the Hugging Face tools write.
fn loadToken(allocator: std.mem.Allocator, io: std.Io, env: Env, session: *hub.http.Session) void {
    if (env.get("HF_TOKEN")) |t| return session.setToken(t) catch {};
    if (env.get("HUGGING_FACE_HUB_TOKEN")) |t| return session.setToken(t) catch {};
    const home = env.get("HOME") orelse return;
    const path = std.fmt.allocPrint(allocator, "{s}/.cache/huggingface/token", .{home}) catch return;
    defer allocator.free(path);
    var buf: [512]u8 = undefined;
    var f = std.Io.Dir.cwd().openFile(io, path, .{}) catch return;
    defer f.close(io);
    const n = f.readPositionalAll(io, &buf, 0) catch return;
    session.setToken(buf[0..n]) catch {};
}

pub fn newSession(allocator: std.mem.Allocator, io: std.Io, env: Env) hub.http.Session {
    var s = hub.http.Session.init(allocator, io);
    if (env.get("HF_ENDPOINT")) |e| s.endpoint = std.mem.trimEnd(u8, e, "/");
    if (env.get("HK_HTTP_BACKOFF_MS")) |v| s.backoff_ms = std.fmt.parseInt(u64, v, 10) catch s.backoff_ms;
    loadToken(allocator, io, env, &s);
    return s;
}

/// Splits `owner/name[:selector]`.
pub const Spec = struct {
    repo: []const u8,
    selector: ?[]const u8,

    pub fn parse(arg: []const u8) Spec {
        if (std.mem.indexOfScalar(u8, arg, ':')) |i| return .{ .repo = arg[0..i], .selector = arg[i + 1 ..] };
        return .{ .repo = arg, .selector = null };
    }

    /// True when `arg` could be a hub reference rather than a path.
    pub fn looksLikeRepo(arg: []const u8) bool {
        const spec = parse(arg);
        return hub.hf.validRepoId(spec.repo) and !std.mem.endsWith(u8, spec.repo, ".hk");
    }
};

const Progress = struct {
    io: std.Io,
    start_ns: i128,
    last_print_ns: i128 = 0,
    label: []const u8 = "",

    fn nowNs(self: *const Progress) i128 {
        return std.Io.Timestamp.now(self.io, .awake).nanoseconds;
    }

    fn callback(ctx: ?*anyopaque, done: u64, total: u64) void {
        const self: *Progress = @ptrCast(@alignCast(ctx.?));
        const now = self.nowNs();
        if (done < total and now - self.last_print_ns < 150 * std.time.ns_per_ms) return;
        self.last_print_ns = now;
        const secs = @max(@as(f64, @floatFromInt(now - self.start_ns)) / 1e9, 1e-3);
        const rate = @as(f64, @floatFromInt(done)) / secs;
        const pct = if (total == 0) 100.0 else @as(f64, @floatFromInt(done)) * 100.0 / @as(f64, @floatFromInt(total));
        const remaining = if (rate > 0) @as(f64, @floatFromInt(total - done)) / rate else 0;
        std.debug.print("\r  {d:5.1}%  {d:.0} / {d:.0} MiB  {d:.1} MiB/s  eta {d:.0}s   ", .{
            pct,
            @as(f64, @floatFromInt(done)) / 1048576.0,
            @as(f64, @floatFromInt(total)) / 1048576.0,
            rate / 1048576.0,
            remaining,
        });
    }

    fn announce(ctx: ?*anyopaque, repo: []const u8, file: hub.hf.File, commit: []const u8) void {
        _ = ctx;
        std.debug.print("{s}  {s}  ({d:.1} MiB, commit {s})\n", .{ repo, file.name, @as(f64, @floatFromInt(file.size)) / 1048576.0, commit[0..@min(commit.len, 8)] });
    }
};

/// Pulls `spec` into the cache and returns the path to the .hk file. Caller frees.
pub fn pullSpec(allocator: std.mem.Allocator, io: std.Io, env: Env, spec: Spec, force: bool, revision: []const u8) ![]u8 {
    const cache = try env.cacheDir(allocator);
    defer allocator.free(cache);
    var session = newSession(allocator, io, env);
    defer session.deinit();

    var prog = Progress{ .io = io, .start_ns = std.Io.Timestamp.now(io, .awake).nanoseconds };
    var diag = hub.pull.Diag{};
    const result = hub.pull.pull(allocator, io, &session, .{
        .repo = spec.repo,
        .revision = revision,
        .selector = spec.selector,
        .cache_dir = cache,
        .force = force,
        .progress = Progress.callback,
        .progress_ctx = &prog,
        .announce = Progress.announce,
    }, &diag) catch |err| {
        std.debug.print("\nError: {s}\n", .{if (diag.len > 0) diag.message() else @errorName(err)});
        return err;
    };
    if (result.cached) {
        std.debug.print("already in the cache: {s}\n", .{result.path});
    } else {
        std.debug.print("\n  downloaded {d:.1} MiB, {s}\n  saved {s}\n", .{
            @as(f64, @floatFromInt(result.bytes_downloaded)) / 1048576.0,
            if (result.verified) "checksum verified" else "no checksum available to verify",
            result.path,
        });
    }
    return result.path;
}

pub fn cmdPull(allocator: std.mem.Allocator, io: std.Io, env: Env, args: *std.process.Args.Iterator) !void {
    var target: ?[]const u8 = null;
    var force = false;
    var revision: []const u8 = "main";
    while (args.next()) |a| {
        if (std.mem.eql(u8, a, "--force") or std.mem.eql(u8, a, "-f")) {
            force = true;
        } else if (std.mem.eql(u8, a, "--revision")) {
            revision = args.next() orelse return usagePull();
        } else if (target == null) {
            target = a;
        } else return usagePull();
    }
    const t = target orelse return usagePull();
    const spec = Spec.parse(t);
    const path = try pullSpec(allocator, io, env, spec, force, revision);
    allocator.free(path);
}

fn usagePull() error{BadUsage} {
    std.debug.print(
        \\Usage: hk pull <owner/name>[:selector] [--force] [--revision REF]
        \\
        \\  hk pull Qwen/Qwen3-0.6B-GGUF
        \\  hk pull bartowski/SmolLM2-135M-Instruct-GGUF:Q4_K_M
        \\
        \\The selector is a file name or part of one (a quant name works). With no selector
        \\the best quant that is not huge is chosen. GGUF files are converted while they
        \\download, so only the .hk file is ever written.
        \\
    , .{});
    return error.BadUsage;
}

pub fn cmdSearch(allocator: std.mem.Allocator, io: std.Io, env: Env, args: *std.process.Args.Iterator) !void {
    var query: ?[]const u8 = null;
    var limit: usize = 15;
    var filter: ?[]const u8 = "gguf";
    while (args.next()) |a| {
        if (std.mem.eql(u8, a, "--limit")) {
            limit = std.fmt.parseInt(usize, args.next() orelse "15", 10) catch 15;
        } else if (std.mem.eql(u8, a, "--all")) {
            filter = null;
        } else if (query == null) {
            query = a;
        }
    }
    const q = query orelse {
        std.debug.print("Usage: hk search <query> [--limit N] [--all]\n", .{});
        return error.BadUsage;
    };
    var session = newSession(allocator, io, env);
    defer session.deinit();
    var res = hub.hf.search(allocator, &session, q, filter, limit) catch |err| {
        std.debug.print("Error: search failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer res.deinit();
    for (res.hits) |h| std.debug.print("{s:<60} {d:>10} downloads\n", .{ h.id, h.downloads });
    if (res.hits.len == 0) std.debug.print("no models found\n", .{});
}

pub fn cmdList(allocator: std.mem.Allocator, io: std.Io, env: Env) !void {
    const cache = try env.cacheDir(allocator);
    defer allocator.free(cache);
    const models_dir = try std.fmt.allocPrint(allocator, "{s}/models", .{cache});
    defer allocator.free(models_dir);
    var dir = std.Io.Dir.cwd().openDir(io, models_dir, .{ .iterate = true }) catch {
        std.debug.print("no models cached yet ({s})\n", .{models_dir});
        return;
    };
    defer dir.close(io);
    var total: u64 = 0;
    var it = dir.iterate();
    while (try it.next(io)) |repo_entry| {
        if (repo_entry.kind != .directory) continue;
        const sub_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ models_dir, repo_entry.name });
        defer allocator.free(sub_path);
        var sub = std.Io.Dir.cwd().openDir(io, sub_path, .{ .iterate = true }) catch continue;
        defer sub.close(io);
        var it2 = sub.iterate();
        while (try it2.next(io)) |f| {
            if (!std.mem.endsWith(u8, f.name, ".hk")) continue;
            const st = sub.statFile(io, f.name, .{}) catch continue;
            total += st.size;
            // owner--name  ->  owner/name
            var display: [256]u8 = undefined;
            const shown = blk: {
                const i = std.mem.indexOf(u8, repo_entry.name, "--") orelse break :blk repo_entry.name;
                break :blk std.fmt.bufPrint(&display, "{s}/{s}", .{ repo_entry.name[0..i], repo_entry.name[i + 2 ..] }) catch repo_entry.name;
            };
            std.debug.print("{s:<48} {s:<40} {d:>8.1} MiB\n", .{ shown, f.name, @as(f64, @floatFromInt(st.size)) / 1048576.0 });
        }
    }
    std.debug.print("total {d:.1} MiB in {s}\n", .{ @as(f64, @floatFromInt(total)) / 1048576.0, models_dir });
}

pub fn cmdRm(allocator: std.mem.Allocator, io: std.Io, env: Env, args: *std.process.Args.Iterator) !void {
    const target = args.next() orelse {
        std.debug.print("Usage: hk rm <owner/name>\n", .{});
        return error.BadUsage;
    };
    const spec = Spec.parse(target);
    if (!hub.hf.validRepoId(spec.repo)) {
        std.debug.print("Error: '{s}' is not a repository name\n", .{target});
        return error.BadUsage;
    }
    const cache = try env.cacheDir(allocator);
    defer allocator.free(cache);
    const dir = try hub.pull.repoDir(allocator, cache, spec.repo);
    defer allocator.free(dir);
    std.Io.Dir.cwd().deleteTree(io, dir) catch |err| {
        std.debug.print("Error: could not remove {s}: {s}\n", .{ dir, @errorName(err) });
        return err;
    };
    std.debug.print("removed {s}\n", .{dir});
}
