//! Hugging Face Hub API: repository metadata, file listings with content hashes, and search.

const std = @import("std");
const http = @import("http.zig");

pub const File = struct {
    name: []const u8,
    size: u64,
    /// Lowercase hex SHA-256 of the file content, present for LFS files.
    sha256: ?[]const u8,
};

pub const Repo = struct {
    arena: std.heap.ArenaAllocator,
    id: []const u8,
    /// Commit the listing was taken at.
    commit: []const u8,
    files: []File,

    pub fn deinit(self: *Repo) void {
        self.arena.deinit();
    }
};

pub const Error = http.Error || error{ InvalidRepo, BadResponse };

/// Percent-encodes a path for use in a URL, keeping `/`.
pub fn encodePath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (path) |c| {
        const safe = std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.' or c == '~' or c == '/';
        if (safe) {
            try out.append(allocator, c);
        } else {
            try out.print(allocator, "%{X:0>2}", .{c});
        }
    }
    return out.toOwnedSlice(allocator);
}

pub fn validRepoId(id: []const u8) bool {
    // owner/name, both non empty, restricted characters.
    var parts = std.mem.splitScalar(u8, id, '/');
    const owner = parts.next() orelse return false;
    const name = parts.next() orelse return false;
    if (parts.next() != null) return false;
    if (owner.len == 0 or name.len == 0) return false;
    for (id) |c| {
        if (!(std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.' or c == '/')) return false;
    }
    return !std.mem.startsWith(u8, owner, ".") and !std.mem.startsWith(u8, name, ".");
}

/// File listing of `repo_id` at `revision`, including sizes and LFS hashes.
pub fn modelInfo(allocator: std.mem.Allocator, session: *http.Session, repo_id: []const u8, revision: []const u8) Error!Repo {
    if (!validRepoId(repo_id)) return error.InvalidRepo;
    const rev_enc = try encodePath(allocator, revision);
    defer allocator.free(rev_enc);
    const url = try std.fmt.allocPrint(allocator, "{s}/api/models/{s}/revision/{s}?blobs=true", .{ session.endpoint, repo_id, rev_enc });
    defer allocator.free(url);
    const body = try session.getAlloc(url, 64 << 20);
    defer allocator.free(body);

    const Json = struct {
        id: []const u8 = "",
        sha: []const u8 = "",
        siblings: []const struct {
            rfilename: []const u8,
            size: ?u64 = null,
            lfs: ?struct { sha256: []const u8 = "", oid: []const u8 = "", size: ?u64 = null } = null,
        } = &.{},
    };
    var parsed = std.json.parseFromSlice(Json, allocator, body, .{ .ignore_unknown_fields = true }) catch return error.BadResponse;
    defer parsed.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const files = try a.alloc(File, parsed.value.siblings.len);
    for (parsed.value.siblings, files) |s, *f| {
        // The hub reports the content hash as `sha256`; `oid` is the older spelling.
        const sha: ?[]const u8 = if (s.lfs) |l| blk: {
            if (l.sha256.len == 64) break :blk try a.dupe(u8, l.sha256);
            if (l.oid.len == 64) break :blk try a.dupe(u8, l.oid);
            break :blk null;
        } else null;
        f.* = .{
            .name = try a.dupe(u8, s.rfilename),
            .size = s.size orelse (if (s.lfs) |l| l.size orelse 0 else 0),
            .sha256 = sha,
        };
    }
    return .{
        .arena = arena,
        .id = try a.dupe(u8, repo_id),
        .commit = try a.dupe(u8, parsed.value.sha),
        .files = files,
    };
}

/// URL that serves `file` of `repo_id` at `revision` (usually a commit hash).
pub fn resolveUrl(allocator: std.mem.Allocator, session: *const http.Session, repo_id: []const u8, revision: []const u8, file: []const u8) ![]u8 {
    const f = try encodePath(allocator, file);
    defer allocator.free(f);
    const r = try encodePath(allocator, revision);
    defer allocator.free(r);
    return std.fmt.allocPrint(allocator, "{s}/{s}/resolve/{s}/{s}", .{ session.endpoint, repo_id, r, f });
}

pub const SearchHit = struct {
    id: []const u8,
    downloads: u64,
    likes: u64,
};

pub const SearchResult = struct {
    arena: std.heap.ArenaAllocator,
    hits: []SearchHit,

    pub fn deinit(self: *SearchResult) void {
        self.arena.deinit();
    }
};

/// Searches models. `filter` is a hub tag such as `gguf`.
pub fn search(allocator: std.mem.Allocator, session: *http.Session, query: []const u8, filter: ?[]const u8, limit: usize) Error!SearchResult {
    const q = try encodePath(allocator, query);
    defer allocator.free(q);
    const url = if (filter) |f|
        try std.fmt.allocPrint(allocator, "{s}/api/models?search={s}&filter={s}&sort=downloads&direction=-1&limit={d}", .{ session.endpoint, q, f, limit })
    else
        try std.fmt.allocPrint(allocator, "{s}/api/models?search={s}&sort=downloads&direction=-1&limit={d}", .{ session.endpoint, q, limit });
    defer allocator.free(url);
    const body = try session.getAlloc(url, 8 << 20);
    defer allocator.free(body);

    const Json = []const struct { id: []const u8 = "", downloads: u64 = 0, likes: u64 = 0 };
    var parsed = std.json.parseFromSlice(Json, allocator, body, .{ .ignore_unknown_fields = true }) catch return error.BadResponse;
    defer parsed.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const hits = try a.alloc(SearchHit, parsed.value.len);
    for (parsed.value, hits) |h, *o| o.* = .{ .id = try a.dupe(u8, h.id), .downloads = h.downloads, .likes = h.likes };
    return .{ .arena = arena, .hits = hits };
}

// ---------------------------------------------------------------------------------------
// File selection
// ---------------------------------------------------------------------------------------

pub const Kind = enum { hk, gguf, safetensors };

pub fn kindOf(name: []const u8) ?Kind {
    if (std.mem.endsWith(u8, name, ".hk")) return .hk;
    if (std.mem.endsWith(u8, name, ".gguf")) return .gguf;
    if (std.mem.endsWith(u8, name, ".safetensors")) return .safetensors;
    return null;
}

/// True for GGUF files that are one piece of a split model (`name-00001-of-00003.gguf`).
/// Safetensors shards use the same naming but are handled as a set, so this is GGUF only.
fn isSplitGguf(name: []const u8, kind: Kind) bool {
    if (kind != .gguf) return false;
    var lower_buf: [256]u8 = undefined;
    const lower = std.ascii.lowerString(&lower_buf, name[0..@min(name.len, lower_buf.len)]);
    return std.mem.indexOf(u8, lower, "-of-0") != null;
}

/// True for files that are not a model on their own: vision projectors.
fn isProjector(name: []const u8) bool {
    var lower_buf: [256]u8 = undefined;
    const lower = std.ascii.lowerString(&lower_buf, name[0..@min(name.len, lower_buf.len)]);
    return std.mem.indexOf(u8, lower, "mmproj") != null;
}

fn containsFold(hay: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > hay.len) return false;
    var i: usize = 0;
    while (i + needle.len <= hay.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(hay[i .. i + needle.len], needle)) return true;
    }
    return false;
}

/// Quant names in the order we prefer them when the user does not say. Q4_K_M is the usual
/// quality and size sweet spot.
const preferred_quants = [_][]const u8{ "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "Q4_0", "Q3_K_M", "IQ4_XS", "Q2_K" };

pub const Choice = struct {
    file: File,
    kind: Kind,
};

pub const SelectError = error{ NoModelFiles, NoMatch, Ambiguous, SplitGgufOnly };

/// Picks one model file from a listing. Native `.hk` files win over GGUF, which wins over
/// safetensors. `selector` is an exact file name or a substring such as a quant name. On
/// `Ambiguous`, `candidates` lists what matched so the caller can show it.
pub fn select(
    allocator: std.mem.Allocator,
    files: []const File,
    selector: ?[]const u8,
    candidates: *std.ArrayList(File),
) (SelectError || std.mem.Allocator.Error)!Choice {
    const kinds = [_]Kind{ .hk, .gguf, .safetensors };
    var any = false;
    var split_seen = false;
    for (kinds) |kind| {
        candidates.clearRetainingCapacity();
        for (files) |f| {
            const k = kindOf(f.name) orelse continue;
            if (k != kind or isProjector(f.name)) continue;
            if (isSplitGguf(f.name, k)) {
                split_seen = true;
                continue;
            }
            any = true;
            if (selector) |sel| {
                if (!(std.mem.eql(u8, f.name, sel) or containsFold(f.name, sel))) continue;
            }
            try candidates.append(allocator, f);
        }
        if (candidates.items.len == 0) continue;
        if (candidates.items.len == 1) return .{ .file = candidates.items[0], .kind = kind };

        // An exact name wins outright.
        if (selector) |sel| for (candidates.items) |f| {
            if (std.mem.eql(u8, f.name, sel)) return .{ .file = f, .kind = kind };
        };
        if (kind == .safetensors) return .{ .file = candidates.items[0], .kind = kind }; // shards: caller handles the set
        // No selector: take the best known quant.
        if (selector == null) {
            for (preferred_quants) |q| {
                for (candidates.items) |f| if (containsFold(f.name, q)) return .{ .file = f, .kind = kind };
            }
            // Otherwise the smallest, which is the safest default for a machine we know nothing about.
            var best = candidates.items[0];
            for (candidates.items[1..]) |f| if (f.size < best.size) {
                best = f;
            };
            return .{ .file = best, .kind = kind };
        }
        return error.Ambiguous;
    }
    if (!any) return if (split_seen) error.SplitGgufOnly else error.NoModelFiles;
    return error.NoMatch;
}

test "select prefers hk, then a sensible quant" {
    const a = std.testing.allocator;
    var cands: std.ArrayList(File) = .empty;
    defer cands.deinit(a);
    const files = [_]File{
        .{ .name = "m-Q8_0.gguf", .size = 8, .sha256 = null },
        .{ .name = "m-Q4_K_M.gguf", .size = 4, .sha256 = null },
        .{ .name = "m-Q2_K.gguf", .size = 2, .sha256 = null },
        .{ .name = "mmproj-m.gguf", .size = 1, .sha256 = null },
        .{ .name = "README.md", .size = 1, .sha256 = null },
    };
    const c = try select(a, &files, null, &cands);
    try std.testing.expectEqualStrings("m-Q4_K_M.gguf", c.file.name);
    const c2 = try select(a, &files, "q8_0", &cands);
    try std.testing.expectEqualStrings("m-Q8_0.gguf", c2.file.name);
    try std.testing.expectError(error.NoMatch, select(a, &files, "Q9_9", &cands));

    const with_hk = [_]File{ .{ .name = "m.hk", .size = 1, .sha256 = null }, .{ .name = "m-Q4_K_M.gguf", .size = 4, .sha256 = null } };
    const c3 = try select(a, &with_hk, null, &cands);
    try std.testing.expectEqual(Kind.hk, c3.kind);
    try std.testing.expectError(error.NoModelFiles, select(a, &[_]File{.{ .name = "README.md", .size = 1, .sha256 = null }}, null, &cands));
}

test "sharded safetensors are a model, split gguf is refused by name" {
    const a = std.testing.allocator;
    var cands: std.ArrayList(File) = .empty;
    defer cands.deinit(a);
    const shards = [_]File{
        .{ .name = "model-00001-of-00002.safetensors", .size = 1, .sha256 = null },
        .{ .name = "model-00002-of-00002.safetensors", .size = 1, .sha256 = null },
        .{ .name = "model.safetensors.index.json", .size = 1, .sha256 = null },
    };
    const c = try select(a, &shards, null, &cands);
    try std.testing.expectEqual(Kind.safetensors, c.kind);

    const split = [_]File{
        .{ .name = "m-Q4_K_M-00001-of-00002.gguf", .size = 1, .sha256 = null },
        .{ .name = "m-Q4_K_M-00002-of-00002.gguf", .size = 1, .sha256 = null },
    };
    try std.testing.expectError(error.SplitGgufOnly, select(a, &split, null, &cands));
    // A whole GGUF next to split ones is still found.
    const mixed = split ++ [_]File{.{ .name = "m-Q8_0.gguf", .size = 1, .sha256 = null }};
    try std.testing.expectEqualStrings("m-Q8_0.gguf", (try select(a, &mixed, null, &cands)).file.name);
}

test "repo ids are validated" {
    try std.testing.expect(validRepoId("Qwen/Qwen3-0.6B"));
    try std.testing.expect(!validRepoId("Qwen"));
    try std.testing.expect(!validRepoId("a/b/c"));
    try std.testing.expect(!validRepoId("../etc/passwd"));
    try std.testing.expect(!validRepoId("a/b c"));
}
