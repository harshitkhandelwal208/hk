//! HTTP access for the model hub, built on the standard library client.
//!
//! Two things matter here beyond fetching bytes:
//!
//!   * Credentials are only ever sent to the hub itself. Hub downloads redirect to a CDN with a
//!     signed URL, and that CDN must never see the token. Redirects are therefore followed by
//!     hand, and the Authorization header is attached per request, by host. (The client's own
//!     `privileged_headers` option is not used: in Zig 0.17.0 it is never written to the wire.)
//!   * `HttpSource` turns a remote file into the same `Source` the converters read from a local
//!     one. A sequential read costs one HTTP request for the whole file, a dropped connection
//!     resumes from the last good byte, and the SHA-256 of everything read is computed on the
//!     way so the download can be verified without a second pass.

const std = @import("std");
const source_mod = @import("../convert/source.zig");

const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Error = error{
    NotFound,
    Unauthorized,
    RateLimited,
    NetworkFailure,
    BadResponse,
    TooLarge,
    OutOfMemory,
};

pub const user_agent = "hk/1.2";

/// Host component of a URI as text. Unlike `HostName.fromUri` this accepts IP literals, which
/// matters for mirrors and local test servers.
fn rawHost(uri: std.Uri, buf: *[256]u8) ?[]const u8 {
    const comp = uri.host orelse return null;
    return comp.toRaw(buf) catch null;
}

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    client: std.http.Client,
    token: ?[]u8 = null,
    /// Base URL of the hub, without a trailing slash.
    endpoint: []const u8 = "https://huggingface.co",
    max_retries: u8 = 6,
    /// Base delay between retries; doubles each attempt up to 32 times this.
    backoff_ms: u64 = 1000,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) Session {
        return .{ .allocator = allocator, .io = io, .client = .{ .allocator = allocator, .io = io } };
    }

    pub fn setToken(self: *Session, token: []const u8) !void {
        if (self.token) |t| self.allocator.free(t);
        const trimmed = std.mem.trim(u8, token, " \r\n\t");
        self.token = if (trimmed.len == 0) null else try self.allocator.dupe(u8, trimmed);
    }

    pub fn deinit(self: *Session) void {
        if (self.token) |t| self.allocator.free(t);
        self.client.deinit();
    }

    fn backoff(self: *Session, attempt: u8) void {
        const ms = self.backoff_ms << @intCast(@min(attempt, 5));
        std.Io.sleep(self.io, .fromMilliseconds(@intCast(ms)), .awake) catch {};
    }

    /// GET a small resource fully into memory. Caller frees the result.
    pub fn getAlloc(self: *Session, url: []const u8, limit: usize) Error![]u8 {
        var attempt: u8 = 0;
        while (true) : (attempt += 1) {
            const result = self.getOnce(url, limit);
            if (result) |body| return body else |err| switch (err) {
                error.NetworkFailure, error.RateLimited => {
                    if (attempt + 1 >= self.max_retries) return err;
                    self.backoff(attempt);
                },
                else => return err,
            }
        }
    }

    fn getOnce(self: *Session, url: []const u8, limit: usize) Error![]u8 {
        const stream = Stream.open(self, url, 0, false) catch |e| return e;
        defer stream.close();
        const n_hint: usize = if (stream.content_length) |c| @intCast(@min(c, limit)) else 4096;
        var out = std.ArrayList(u8).initCapacity(self.allocator, n_hint) catch return error.OutOfMemory;
        errdefer out.deinit(self.allocator);
        var buf: [16 * 1024]u8 = undefined;
        while (true) {
            const n = stream.reader.readSliceShort(&buf) catch return error.NetworkFailure;
            if (n == 0) break;
            if (out.items.len + n > limit) return error.TooLarge;
            out.appendSlice(self.allocator, buf[0..n]) catch return error.OutOfMemory;
        }
        return out.toOwnedSlice(self.allocator) catch error.OutOfMemory;
    }

    /// The token goes to the configured hub and the official hosts, and nowhere else.
    fn isHubHost(self: *const Session, host: []const u8) bool {
        if (std.mem.eql(u8, host, "huggingface.co") or std.mem.endsWith(u8, host, ".huggingface.co")) return true;
        if (std.mem.eql(u8, host, "hf.co") or std.mem.endsWith(u8, host, ".hf.co")) return true;
        // A custom endpoint (a mirror or a test server): compare against its host.
        const ep = std.Uri.parse(self.endpoint) catch return false;
        var buf: [256]u8 = undefined;
        const eh = rawHost(ep, &buf) orelse return false;
        return std.mem.eql(u8, eh, host);
    }
};

/// An open HTTP response with its body reader. Heap allocated because the response points into it.
pub const Stream = struct {
    gpa: std.mem.Allocator,
    req: std.http.Client.Request,
    redirect_buf: [8192]u8 = undefined,
    transfer_buf: [16384]u8 = undefined,
    range_value: [48]u8 = undefined,
    extra: [2]std.http.Header = undefined,
    auth_value: ?[]u8 = null,
    reader: *std.Io.Reader = undefined,
    status: u16 = 0,
    content_length: ?u64 = null,

    /// Resolves a Location header against the URL it came from. Handles absolute URLs and
    /// absolute paths, which is what hubs and CDNs send.
    fn resolveLocation(gpa: std.mem.Allocator, base: []const u8, loc: []const u8) ![]u8 {
        if (std.mem.startsWith(u8, loc, "http://") or std.mem.startsWith(u8, loc, "https://")) return gpa.dupe(u8, loc);
        if (std.mem.startsWith(u8, loc, "/")) {
            const scheme_end = (std.mem.indexOf(u8, base, "://") orelse return error.BadResponse) + 3;
            const path_start = std.mem.indexOfScalarPos(u8, base, scheme_end, '/') orelse base.len;
            return std.fmt.allocPrint(gpa, "{s}{s}", .{ base[0..path_start], loc });
        }
        return error.BadResponse;
    }

    /// Opens `url`, requesting bytes from `start` to the end when `start > 0`. Returns after the
    /// response head has been checked, so a returned stream is positioned at byte `start`.
    ///
    /// Redirects are followed here rather than by the client so the credential policy is ours:
    /// the token is attached to a request only when its host is the hub, so it never reaches
    /// the CDN a download is redirected to.
    pub fn open(session: *Session, url: []const u8, start: u64, ranged: bool) Error!*Stream {
        const gpa = session.allocator;
        const self = gpa.create(Stream) catch return error.OutOfMemory;
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .req = undefined };

        var current = gpa.dupe(u8, url) catch return error.OutOfMemory;
        defer gpa.free(current);
        var req_live = false;
        errdefer {
            if (req_live) self.req.deinit();
            if (self.auth_value) |a| gpa.free(a);
        }

        var hops: u8 = 0;
        var resp: std.http.Client.Response = undefined;
        while (true) : (hops += 1) {
            if (hops > 8) return error.BadResponse;
            const uri = std.Uri.parse(current) catch return error.BadResponse;

            var n_extra: usize = 0;
            if (ranged or start > 0) {
                const v = std.fmt.bufPrint(&self.range_value, "bytes={d}-", .{start}) catch unreachable;
                self.extra[n_extra] = .{ .name = "Range", .value = v };
                n_extra += 1;
            }
            if (self.auth_value) |a| {
                gpa.free(a);
                self.auth_value = null;
            }
            const host_ok = blk: {
                var hb: [256]u8 = undefined;
                const host = rawHost(uri, &hb) orelse break :blk false;
                break :blk session.isHubHost(host);
            };
            if (session.token != null and host_ok) {
                self.auth_value = std.fmt.allocPrint(gpa, "Bearer {s}", .{session.token.?}) catch return error.OutOfMemory;
                self.extra[n_extra] = .{ .name = "Authorization", .value = self.auth_value.? };
                n_extra += 1;
            }

            self.req = session.client.request(.GET, uri, .{
                .headers = .{
                    .accept_encoding = .{ .override = "identity" },
                    .user_agent = .{ .override = user_agent },
                },
                .extra_headers = self.extra[0..n_extra],
                .redirect_behavior = .unhandled,
            }) catch return error.NetworkFailure;
            req_live = true;
            self.req.sendBodiless() catch return error.NetworkFailure;
            resp = self.req.receiveHead(&self.redirect_buf) catch return error.NetworkFailure;

            const st = @intFromEnum(resp.head.status);
            if (st == 301 or st == 302 or st == 303 or st == 307 or st == 308) {
                const loc = resp.head.location orelse return error.BadResponse;
                const next = resolveLocation(gpa, current, loc) catch return error.BadResponse;
                self.req.deinit();
                req_live = false;
                gpa.free(current);
                current = next;
                continue;
            }
            break;
        }

        self.status = @intFromEnum(resp.head.status);
        self.content_length = resp.head.content_length;
        switch (self.status) {
            200, 206 => {},
            401, 403 => return error.Unauthorized,
            404 => return error.NotFound,
            429 => return error.RateLimited,
            500...599 => return error.NetworkFailure,
            else => return error.BadResponse,
        }
        self.reader = resp.reader(&self.transfer_buf);
        if (start > 0 and self.status == 200) {
            // The server ignored the range and is sending the whole body. Skip ahead.
            self.reader.discardAll64(start) catch return error.NetworkFailure;
            if (self.content_length) |c| self.content_length = c - start;
        }
        return self;
    }

    pub fn close(self: *Stream) void {
        self.req.deinit();
        if (self.auth_value) |a| self.gpa.free(a);
        self.gpa.destroy(self);
    }
};

/// A remote file presented as a `Source`.
pub const HttpSource = struct {
    allocator: std.mem.Allocator,
    session: *Session,
    url: []u8,
    total: u64,
    stream: ?*Stream = null,
    /// Offset of the next byte the open stream will deliver.
    cursor: u64 = 0,
    hasher: Sha256 = Sha256.init(.{}),
    /// Number of leading bytes folded into `hasher`. The hash is only valid while every byte
    /// has been read in order from offset zero.
    hashed: u64 = 0,
    hash_valid: bool = true,
    bytes_downloaded: u64 = 0,
    /// The first bytes of the file, kept so a reader can look at the header again without a
    /// backward seek. Converters plan from the header, then read the file front to back; the
    /// replay must not disturb the stream or the running hash. Capped, so memory stays small.
    head: std.ArrayList(u8) = .empty,

    /// Opens the file and learns its size from the first response.
    pub fn open(allocator: std.mem.Allocator, session: *Session, url: []const u8, expected_len: ?u64) Error!*HttpSource {
        const self = allocator.create(HttpSource) catch return error.OutOfMemory;
        errdefer allocator.destroy(self);
        const url_copy = allocator.dupe(u8, url) catch return error.OutOfMemory;
        errdefer allocator.free(url_copy);
        self.* = .{ .allocator = allocator, .session = session, .url = url_copy, .total = 0 };

        const s = try self.openAt(0);
        self.total = s.content_length orelse expected_len orelse return error.BadResponse;
        if (expected_len) |e| if (e != self.total) return error.BadResponse;
        self.stream = s;
        return self;
    }

    /// Like `open` but makes no request until the first read, for files whose size is already
    /// known. Used for the later shards of a model, which would otherwise sit idle on an open
    /// connection while earlier shards are converted.
    pub fn openLazy(allocator: std.mem.Allocator, session: *Session, url: []const u8, total: u64) Error!*HttpSource {
        const self = allocator.create(HttpSource) catch return error.OutOfMemory;
        errdefer allocator.destroy(self);
        const url_copy = allocator.dupe(u8, url) catch return error.OutOfMemory;
        self.* = .{ .allocator = allocator, .session = session, .url = url_copy, .total = total };
        return self;
    }

    const head_limit: usize = 4 << 20;

    pub fn close(self: *HttpSource) void {
        self.head.deinit(self.allocator);
        if (self.stream) |s| s.close();
        self.allocator.free(self.url);
        self.allocator.destroy(self);
    }

    pub fn source(self: *HttpSource) source_mod.Source {
        return .{ .ctx = self, .readAtFn = read, .len = self.total };
    }

    fn openAt(self: *HttpSource, offset: u64) Error!*Stream {
        var attempt: u8 = 0;
        while (true) : (attempt += 1) {
            const r = Stream.open(self.session, self.url, offset, true);
            if (r) |s| return s else |err| switch (err) {
                error.NetworkFailure, error.RateLimited => {
                    if (attempt + 1 >= self.session.max_retries) return err;
                    self.session.backoff(attempt);
                },
                else => return err,
            }
        }
    }

    fn feed(self: *HttpSource, offset: u64, bytes: []const u8) void {
        if (offset == self.head.items.len and self.head.items.len < head_limit) {
            const room = head_limit - self.head.items.len;
            self.head.appendSlice(self.allocator, bytes[0..@min(bytes.len, room)]) catch {};
        }
        if (self.hash_valid and offset == self.hashed) {
            self.hasher.update(bytes);
            self.hashed += bytes.len;
        } else if (offset + bytes.len > self.hashed) {
            self.hash_valid = false;
        }
    }

    fn read(ctx: *anyopaque, offset: u64, buf: []u8) source_mod.ReadError!usize {
        const self: *HttpSource = @ptrCast(@alignCast(ctx));
        return self.readInner(offset, buf) catch |e| switch (e) {
            error.NotFound => error.NotFound,
            error.Unauthorized => error.Unauthorized,
            error.RateLimited => error.RateLimited,
            error.OutOfMemory, error.NetworkFailure => error.NetworkFailure,
            error.BadResponse, error.TooLarge => error.BadResponse,
        };
    }

    fn readInner(self: *HttpSource, offset: u64, buf: []u8) Error!usize {
        if (offset >= self.total or buf.len == 0) return 0;
        // Replay from the kept head: no network, no change to the stream or the hash.
        if (offset < self.head.items.len) {
            const n = @min(buf.len, self.head.items.len - @as(usize, @intCast(offset)));
            @memcpy(buf[0..n], self.head.items[@intCast(offset)..][0..n]);
            return n;
        }
        var failures: u8 = 0;
        while (true) {
            // Position the stream at `offset`: continue, skip a small gap, or reopen.
            if (self.stream != null and offset >= self.cursor and offset - self.cursor <= 8 << 20) {
                // continue / skip below
            } else {
                if (self.stream) |s| s.close();
                self.stream = null;
                self.stream = try self.openAt(offset);
                self.cursor = offset;
            }
            const s = self.stream.?;

            // Skip a small gap, hashing it so the digest stays contiguous.
            var skip_buf: [16 * 1024]u8 = undefined;
            var skip_failed = false;
            while (self.cursor < offset) {
                const want: usize = @intCast(@min(skip_buf.len, offset - self.cursor));
                const n = s.reader.readSliceShort(skip_buf[0..want]) catch 0;
                if (n == 0) {
                    skip_failed = true;
                    break;
                }
                self.feed(self.cursor, skip_buf[0..n]);
                self.cursor += n;
            }
            if (!skip_failed) {
                const want: usize = @intCast(@min(buf.len, self.total - offset));
                const n = s.reader.readSliceShort(buf[0..want]) catch 0;
                if (n > 0) {
                    self.feed(offset, buf[0..n]);
                    self.cursor = offset + n;
                    self.bytes_downloaded += n;
                    return n;
                }
            }
            // The connection died or ended early: reopen from the current position.
            failures += 1;
            if (failures >= self.session.max_retries) return error.NetworkFailure;
            if (self.stream) |st| st.close();
            self.stream = null;
            self.session.backoff(failures - 1);
        }
    }

    /// Reads whatever has not been read yet, then returns the SHA-256 of the whole file, or null
    /// if the file was not read strictly in order so the digest does not cover it.
    pub fn finishHash(self: *HttpSource) Error!?[32]u8 {
        var scratch: [64 * 1024]u8 = undefined;
        while (self.hash_valid and self.hashed < self.total) {
            const n = try self.readInner(self.hashed, &scratch);
            if (n == 0) return error.NetworkFailure;
        }
        if (!self.hash_valid) return null;
        var copy = self.hasher;
        var out: [32]u8 = undefined;
        copy.final(&out);
        return out;
    }
};
