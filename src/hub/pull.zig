//! `hk pull`: fetch a model from the hub straight into an .hk file.
//!
//! A GGUF repository is never downloaded as a GGUF. The file is streamed from the network,
//! converted on the fly, and written once as .hk, so disk use is the size of the result and
//! memory use is a few megabytes whatever the model size. The download is checksummed while it
//! streams and compared with the hub's SHA-256 before the output is moved into place.

const std = @import("std");
const http = @import("http.zig");
const hf = @import("hf.zig");
const convert_gguf = @import("../convert/gguf.zig");
const convert_hf = @import("../convert/hf.zig");
const source_mod = @import("../convert/source.zig");

pub const Options = struct {
    repo: []const u8,
    revision: []const u8 = "main",
    /// Exact file name or a substring (such as a quant name) to pick among several files.
    selector: ?[]const u8 = null,
    /// Root of the model cache. Models go in `<cache_dir>/models/<owner>--<name>/`.
    cache_dir: []const u8,
    /// Download again even if the file is already in the cache.
    force: bool = false,
    progress: ?*const fn (ctx: ?*anyopaque, done: u64, total: u64) void = null,
    progress_ctx: ?*anyopaque = null,
    /// Called with a short description of what was chosen, before the transfer starts.
    announce: ?*const fn (ctx: ?*anyopaque, repo: []const u8, file: hf.File, commit: []const u8) void = null,
};

pub const Result = struct {
    /// Path of the .hk file. Caller frees.
    path: []u8,
    /// True when the file was already cached and nothing was downloaded.
    cached: bool,
    /// True when the download was checked against the hub's SHA-256.
    verified: bool,
    bytes_downloaded: u64,
};

pub const Diag = struct {
    buf: [512]u8 = undefined,
    len: usize = 0,

    pub fn set(self: *Diag, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(&self.buf, fmt, args) catch self.buf[0..];
        self.len = s.len;
    }

    pub fn message(self: *const Diag) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const Error = hf.Error || convert_hf.ConvertError || hf.SelectError ||
    std.Io.File.OpenError || std.Io.Dir.CreateDirPathError || std.Io.Dir.RenameError || std.Io.File.WritePositionalError || error{
    ChecksumMismatch,
    UnsupportedRepo,
    InvalidRepo,
};

/// `<cache_dir>/models/<owner>--<name>`. The repo id must already be validated.
pub fn repoDir(allocator: std.mem.Allocator, cache_dir: []const u8, repo: []const u8) ![]u8 {
    const slash = std.mem.indexOfScalar(u8, repo, '/') orelse return error.InvalidRepo;
    return std.fmt.allocPrint(allocator, "{s}/models/{s}--{s}", .{ cache_dir, repo[0..slash], repo[slash + 1 ..] });
}

fn outputName(allocator: std.mem.Allocator, file: []const u8) ![]u8 {
    const base = std.fs.path.basename(file);
    const stem = if (std.mem.lastIndexOfScalar(u8, base, '.')) |i| base[0..i] else base;
    return std.fmt.allocPrint(allocator, "{s}.hk", .{stem});
}

const VerifyCtx = struct {
    src: *http.HttpSource,
    expected_hex: ?[]const u8,
    verified: bool = false,
    mismatch: bool = false,
};

fn verifyHook(ctx_ptr: ?*anyopaque) error{VerificationFailed}!void {
    const ctx: *VerifyCtx = @ptrCast(@alignCast(ctx_ptr.?));
    const digest = ctx.src.finishHash() catch return error.VerificationFailed;
    const expected = ctx.expected_hex orelse return; // nothing to compare against
    const got = digest orelse return error.VerificationFailed;
    var hex: [64]u8 = undefined;
    _ = std.fmt.bufPrint(&hex, "{x}", .{&got}) catch unreachable;
    if (!std.ascii.eqlIgnoreCase(&hex, expected)) {
        ctx.mismatch = true;
        return error.VerificationFailed;
    }
    ctx.verified = true;
}

pub fn pull(
    allocator: std.mem.Allocator,
    io: std.Io,
    session: *http.Session,
    opts: Options,
    diag: *Diag,
) Error!Result {
    var repo = hf.modelInfo(allocator, session, opts.repo, opts.revision) catch |e| {
        switch (e) {
            error.NotFound => diag.set("repository '{s}' was not found (check the name; private repositories need a token)", .{opts.repo}),
            error.Unauthorized => diag.set("access to '{s}' was refused. Set HF_TOKEN to a token that can read it.", .{opts.repo}),
            error.InvalidRepo => diag.set("'{s}' is not a repository name (expected owner/name)", .{opts.repo}),
            else => diag.set("could not reach the hub: {s}", .{@errorName(e)}),
        }
        return e;
    };
    defer repo.deinit();

    var cands: std.ArrayList(hf.File) = .empty;
    defer cands.deinit(allocator);
    const choice = hf.select(allocator, repo.files, opts.selector, &cands) catch |e| {
        switch (e) {
            error.NoModelFiles => diag.set("'{s}' has no .hk, .gguf or .safetensors files", .{opts.repo}),
            error.SplitGgufOnly => diag.set("'{s}' only has GGUF files split into several pieces, which hk cannot pull yet. Look for a repository with a single-file GGUF or safetensors weights", .{opts.repo}),
            error.NoMatch => diag.set("no file in '{s}' matches '{s}'", .{ opts.repo, opts.selector orelse "" }),
            error.Ambiguous => {
                var w = std.Io.Writer.fixed(&diag.buf);
                w.print("'{s}' matches several files; pick one with a more specific name: ", .{opts.selector orelse ""}) catch {};
                for (cands.items, 0..) |c, i| w.print("{s}{s}", .{ if (i == 0) "" else ", ", c.name }) catch {};
                diag.len = w.end;
            },
            else => {},
        }
        return e;
    };

    const dir = try repoDir(allocator, opts.cache_dir, opts.repo);
    defer allocator.free(dir);
    // A safetensors checkpoint is many files that make one model, so name the result after the
    // repository. A GGUF or .hk file keeps its own name, which says which quant it is.
    const name = if (choice.kind == .safetensors)
        try std.fmt.allocPrint(allocator, "{s}.hk", .{repo.id[std.mem.indexOfScalar(u8, repo.id, '/').? + 1 ..]})
    else
        try outputName(allocator, choice.file.name);
    defer allocator.free(name);
    const out_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir, name });
    errdefer allocator.free(out_path);

    const cwd = std.Io.Dir.cwd();
    if (!opts.force) {
        if (cwd.statFile(io, out_path, .{})) |_| {
            return .{ .path = out_path, .cached = true, .verified = false, .bytes_downloaded = 0 };
        } else |_| {}
    }
    try cwd.createDirPath(io, dir);

    if (opts.announce) |f| f(opts.progress_ctx, opts.repo, choice.file, repo.commit);

    // Pin to the exact commit so the result is reproducible even if the branch moves on.
    const rev = if (repo.commit.len > 0) repo.commit else opts.revision;
    const url = try hf.resolveUrl(allocator, session, opts.repo, rev, choice.file.name);
    defer allocator.free(url);

    if (choice.kind == .safetensors) {
        const st = try pullSafetensors(allocator, io, session, opts, &repo, rev, out_path, diag);
        return .{ .path = out_path, .cached = false, .verified = st.verified, .bytes_downloaded = st.bytes_downloaded };
    }

    var src = http.HttpSource.open(allocator, session, url, if (choice.file.size > 0) choice.file.size else null) catch |e| {
        diag.set("could not download '{s}': {s}", .{ choice.file.name, @errorName(e) });
        return e;
    };
    defer src.close();

    var ctx = VerifyCtx{ .src = src, .expected_hex = choice.file.sha256 };
    const url_meta = [_][2][]const u8{
        .{ "hk.source.repo", opts.repo },
        .{ "hk.source.revision", repo.commit },
        .{ "hk.source.file", choice.file.name },
        .{ "hk.source.sha256", choice.file.sha256 orelse "" },
    };

    switch (choice.kind) {
        .gguf => {
            var cdiag = convert_gguf.Diag{};
            convert_gguf.convert(allocator, io, src.source(), out_path, .{
                .progress = opts.progress,
                .progress_ctx = opts.progress_ctx,
                .diag = &cdiag,
                .extra_meta = &url_meta,
                .before_finish = verifyHook,
                .before_finish_ctx = &ctx,
            }) catch |e| {
                if (e == error.VerificationFailed and ctx.mismatch) {
                    diag.set("the download of '{s}' does not match the hub's checksum. Nothing was kept.", .{choice.file.name});
                    return error.ChecksumMismatch;
                }
                if (cdiag.len > 0) diag.set("conversion of '{s}' failed: {s}", .{ choice.file.name, cdiag.message() }) else diag.set("conversion of '{s}' failed: {s}", .{ choice.file.name, @errorName(e) });
                return e;
            };
        },
        .hk => try copyHk(allocator, io, src, out_path, opts, &ctx, diag, choice.file.name),
        .safetensors => unreachable,
    }
    return .{ .path = out_path, .cached = false, .verified = ctx.verified, .bytes_downloaded = src.bytes_downloaded };
}

/// A native .hk file needs no conversion: stream it to disk and verify.
fn copyHk(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: *http.HttpSource,
    out_path: []const u8,
    opts: Options,
    ctx: *VerifyCtx,
    diag: *Diag,
    file_name: []const u8,
) Error!void {
    const cwd = std.Io.Dir.cwd();
    const partial = try std.fmt.allocPrint(allocator, "{s}.partial", .{out_path});
    defer allocator.free(partial);
    var f = try cwd.createFile(io, partial, .{});
    var closed = false;
    errdefer {
        if (!closed) f.close(io);
        cwd.deleteFile(io, partial) catch {};
    }
    const buf = try allocator.alloc(u8, 4 << 20);
    defer allocator.free(buf);
    var off: u64 = 0;
    const s = src.source();
    while (off < src.total) {
        const n = s.readAt(off, buf) catch |e| {
            diag.set("download of '{s}' failed: {s}", .{ file_name, @errorName(e) });
            return error.NetworkFailure;
        };
        if (n == 0) return error.NetworkFailure;
        try f.writePositionalAll(io, buf[0..n], off);
        off += n;
        if (opts.progress) |p| p(opts.progress_ctx, off, src.total);
    }
    verifyHook(ctx) catch {
        if (ctx.mismatch) {
            diag.set("the download of '{s}' does not match the hub's checksum. Nothing was kept.", .{file_name});
            return error.ChecksumMismatch;
        }
        return error.NetworkFailure;
    };
    f.close(io);
    closed = true;
    try cwd.rename(partial, cwd, out_path, io);
}

fn findFile(files: []const hf.File, name: []const u8) ?hf.File {
    for (files) |f| if (std.mem.eql(u8, f.name, name)) return f;
    return null;
}

/// Shard file names of a safetensors checkpoint: the files the index names, else the single
/// `model.safetensors`, else every `model-*.safetensors`.
fn shardNames(allocator: std.mem.Allocator, session: *http.Session, repo: *const hf.Repo, rev: []const u8, out: *std.ArrayList([]const u8), arena: std.mem.Allocator) !void {
    if (findFile(repo.files, "model.safetensors.index.json")) |_| {
        const url = try hf.resolveUrl(allocator, session, repo.id, rev, "model.safetensors.index.json");
        defer allocator.free(url);
        const body = try session.getAlloc(url, 64 << 20);
        defer allocator.free(body);
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return error.BadResponse;
        defer parsed.deinit();
        const wm = parsed.value.object.get("weight_map") orelse return error.BadResponse;
        if (wm != .object) return error.BadResponse;
        var it = wm.object.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.* != .string) continue;
            const n = kv.value_ptr.string;
            var seen = false;
            for (out.items) |e| if (std.mem.eql(u8, e, n)) {
                seen = true;
            };
            if (!seen) try out.append(allocator, try arena.dupe(u8, n));
        }
        std.mem.sort([]const u8, out.items, {}, struct {
            fn lt(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);
        return;
    }
    if (findFile(repo.files, "model.safetensors") != null) {
        try out.append(allocator, "model.safetensors");
        return;
    }
    for (repo.files) |f| {
        if (std.mem.startsWith(u8, f.name, "model-") and std.mem.endsWith(u8, f.name, ".safetensors")) try out.append(allocator, f.name);
    }
    std.mem.sort([]const u8, out.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
}

const ShardVerify = struct {
    sources: []*http.HttpSource,
    expected: []?[]const u8,
    mismatch: bool = false,
};

fn shardVerifyHook(ctx_ptr: ?*anyopaque) error{VerificationFailed}!void {
    const ctx: *ShardVerify = @ptrCast(@alignCast(ctx_ptr.?));
    for (ctx.sources, ctx.expected) |src, exp| {
        const digest = src.finishHash() catch return error.VerificationFailed;
        const want = exp orelse continue;
        const got = digest orelse return error.VerificationFailed;
        var hex: [64]u8 = undefined;
        _ = std.fmt.bufPrint(&hex, "{x}", .{&got}) catch unreachable;
        if (!std.ascii.eqlIgnoreCase(&hex, want)) {
            ctx.mismatch = true;
            return error.VerificationFailed;
        }
    }
}

const ShardResult = struct {
    bytes_downloaded: u64,
    /// True only when every shard had a hub checksum and all of them matched.
    verified: bool,
};

/// Converts a Hugging Face safetensors repository, streaming every shard.
fn pullSafetensors(
    allocator: std.mem.Allocator,
    io: std.Io,
    session: *http.Session,
    opts: Options,
    repo: *const hf.Repo,
    rev: []const u8,
    out_path: []const u8,
    diag: *Diag,
) Error!ShardResult {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Small files, fetched whole.
    const fetch = struct {
        fn get(a: std.mem.Allocator, sess: *http.Session, r: *const hf.Repo, revision: []const u8, name: []const u8, limit: usize) !?[]u8 {
            if (findFile(r.files, name) == null) return null;
            const u = try hf.resolveUrl(a, sess, r.id, revision, name);
            defer a.free(u);
            return try sess.getAlloc(u, limit);
        }
    }.get;
    const config = (try fetch(allocator, session, repo, rev, "config.json", 8 << 20)) orelse {
        diag.set("'{s}' has no config.json, so the model cannot be described", .{repo.id});
        return error.UnsupportedRepo;
    };
    defer allocator.free(config);
    const tokenizer_json = try fetch(allocator, session, repo, rev, "tokenizer.json", 256 << 20);
    defer if (tokenizer_json) |t| allocator.free(t);
    const tokenizer_config = try fetch(allocator, session, repo, rev, "tokenizer_config.json", 32 << 20);
    defer if (tokenizer_config) |t| allocator.free(t);
    const generation_config = try fetch(allocator, session, repo, rev, "generation_config.json", 1 << 20);
    defer if (generation_config) |t| allocator.free(t);

    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(allocator);
    try shardNames(allocator, session, repo, rev, &names, arena);
    if (names.items.len == 0) {
        diag.set("'{s}' has no safetensors weight files", .{repo.id});
        return error.UnsupportedRepo;
    }

    const sources = try allocator.alloc(*http.HttpSource, names.items.len);
    defer allocator.free(sources);
    const expected = try allocator.alloc(?[]const u8, names.items.len);
    defer allocator.free(expected);
    const shards = try allocator.alloc(convert_hf.Shard, names.items.len);
    defer allocator.free(shards);
    var opened: usize = 0;
    defer for (sources[0..opened]) |s| s.close();
    for (names.items, 0..) |n, i| {
        const f = findFile(repo.files, n) orelse {
            diag.set("the index names '{s}', which is not in the repository", .{n});
            return error.UnsupportedRepo;
        };
        const u = try hf.resolveUrl(allocator, session, repo.id, rev, n);
        defer allocator.free(u);
        sources[i] = http.HttpSource.openLazy(allocator, session, u, f.size) catch return error.NetworkFailure;
        opened += 1;
        expected[i] = f.sha256;
        shards[i] = .{ .name = n, .src = sources[i].source() };
    }

    var verify = ShardVerify{ .sources = sources, .expected = expected };
    const meta_extra = [_][2][]const u8{
        .{ "hk.source.repo", repo.id },
        .{ "hk.source.revision", repo.commit },
    };
    var cdiag = convert_gguf.Diag{};
    convert_hf.convert(allocator, io, .{
        .config_json = config,
        .tokenizer_json = tokenizer_json,
        .tokenizer_config_json = tokenizer_config,
        .generation_config_json = generation_config,
        .shards = shards,
        .model_name = repo.id,
    }, out_path, .{
        .progress = opts.progress,
        .progress_ctx = opts.progress_ctx,
        .diag = &cdiag,
        .extra_meta = &meta_extra,
        .before_finish = shardVerifyHook,
        .before_finish_ctx = &verify,
    }) catch |e| {
        if (e == error.VerificationFailed and verify.mismatch) {
            diag.set("a downloaded shard of '{s}' does not match the hub's checksum. Nothing was kept.", .{repo.id});
            return error.ChecksumMismatch;
        }
        if (cdiag.len > 0) diag.set("conversion of '{s}' failed: {s}", .{ repo.id, cdiag.message() }) else diag.set("conversion of '{s}' failed: {s}", .{ repo.id, @errorName(e) });
        return e;
    };

    var res = ShardResult{ .bytes_downloaded = 0, .verified = true };
    for (sources, expected) |src, exp| {
        res.bytes_downloaded += src.bytes_downloaded;
        if (exp == null) res.verified = false;
    }
    return res;
}
