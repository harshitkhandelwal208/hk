//! A small stand-in for the Hugging Face Hub that can misbehave on request.
//!
//! Serves /api/models/<owner>/<name>/revision/<rev> and /<owner>/<name>/resolve/<rev>/<file> from
//! a directory, with Range support, and injects faults selected by `faults`:
//!
//!   wrong_sha        report a SHA-256 that does not match the bytes
//!   drop_after       close the connection after this many body bytes, once per request offset
//!   truncate         serve half the bytes Content-Length promises
//!   status           answer one path with a fixed status (404, 401, 429, 500)
//!   require_token    answer 401 unless `Authorization: Bearer <token>` is present
//!   redirect_host    serve files from a second host name through a 302
//!   no_sha           leave the checksum out of the listing for one file
//!
//! The response is written by hand, byte by byte, because the faults are exactly the things a
//! well behaved HTTP server library refuses to do.

const std = @import("std");

pub const Faults = struct {
    wrong_sha: bool = false,
    drop_after: ?usize = null,
    truncate: bool = false,
    status_path: ?[]const u8 = null,
    status_code: u16 = 0,
    require_token: ?[]const u8 = null,
    redirect_host: ?[]const u8 = null,
    no_sha: ?[]const u8 = null,
};

pub const LogEntry = struct {
    path: []u8,
    range: ?[]u8,
    auth: ?[]u8,
};

pub const Hub = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    root: []const u8,
    owner: []const u8,
    name: []const u8,
    faults: Faults = .{},

    server: std.Io.net.Server = undefined,
    port: u16 = 0,
    thread: ?std.Thread = null,
    stopping: std.atomic.Value(bool) = .init(false),

    mutex: std.Io.Mutex = .init,
    log: std.ArrayList(LogEntry) = .empty,
    dropped: std.ArrayList(struct { name: []u8, start: usize }) = .empty,

    pub fn start(gpa: std.mem.Allocator, io: std.Io, root: []const u8, owner: []const u8, name: []const u8) !*Hub {
        const h = try gpa.create(Hub);
        h.* = .{ .gpa = gpa, .io = io, .root = root, .owner = owner, .name = name };
        const addr = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
        h.server = try addr.listen(io, .{});
        h.port = h.server.socket.address.getPort();
        h.thread = try std.Thread.spawn(.{}, acceptLoop, .{h});
        return h;
    }

    pub fn stop(h: *Hub) void {
        h.stopping.store(true, .release);
        // Wake the accept loop with a throwaway connection, then close the listener.
        if (std.Io.net.IpAddress.parse("127.0.0.1", h.port)) |addr| {
            if (addr.connect(h.io, .{ .mode = .stream })) |s| s.close(h.io) else |_| {}
        } else |_| {}
        if (h.thread) |t| t.join();
        h.server.deinit(h.io);
        for (h.log.items) |e| {
            h.gpa.free(e.path);
            if (e.range) |r| h.gpa.free(r);
            if (e.auth) |a| h.gpa.free(a);
        }
        h.log.deinit(h.gpa);
        for (h.dropped.items) |d| h.gpa.free(d.name);
        h.dropped.deinit(h.gpa);
        h.gpa.destroy(h);
    }

    pub fn url(h: *const Hub, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "http://127.0.0.1:{d}", .{h.port}) catch unreachable;
    }

    /// Number of logged requests whose path contains `needle`.
    pub fn count(h: *Hub, needle: []const u8) usize {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        var n: usize = 0;
        for (h.log.items) |e| {
            if (std.mem.indexOf(u8, e.path, needle) != null) n += 1;
        }
        return n;
    }

    pub fn total(h: *Hub) usize {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        return h.log.items.len;
    }

    fn acceptLoop(h: *Hub) void {
        while (!h.stopping.load(.acquire)) {
            const stream = h.server.accept(h.io) catch return;
            if (h.stopping.load(.acquire)) {
                stream.close(h.io);
                return;
            }
            const t = std.Thread.spawn(.{}, serve, .{ h, stream }) catch {
                stream.close(h.io);
                continue;
            };
            t.detach();
        }
    }

    fn serve(h: *Hub, stream: std.Io.net.Stream) void {
        defer stream.close(h.io);
        var rbuf: [8192]u8 = undefined;
        var rd = stream.reader(h.io, &rbuf);
        var wbuf: [4096]u8 = undefined;
        var wr = stream.writer(h.io, &wbuf);
        h.handle(&rd.interface, &wr.interface) catch {};
    }

    fn handle(h: *Hub, r: *std.Io.Reader, w: *std.Io.Writer) !void {
        // Read the request head (until the blank line).
        var head: std.ArrayList(u8) = .empty;
        defer head.deinit(h.gpa);
        while (true) {
            const b = try r.takeByte();
            try head.append(h.gpa, b);
            if (head.items.len >= 4 and std.mem.eql(u8, head.items[head.items.len - 4 ..], "\r\n\r\n")) break;
            if (head.items.len > 64 * 1024) return error.HeadTooLong;
        }
        var lines = std.mem.splitSequence(u8, head.items, "\r\n");
        const request_line = lines.next() orelse return;
        var parts = std.mem.splitScalar(u8, request_line, ' ');
        _ = parts.next();
        const target = parts.next() orelse return;
        const path = if (std.mem.indexOfScalar(u8, target, '?')) |q| target[0..q] else target;
        var range: ?[]const u8 = null;
        var auth: ?[]const u8 = null;
        while (lines.next()) |l| {
            if (l.len == 0) break;
            const colon = std.mem.indexOfScalar(u8, l, ':') orelse continue;
            const key = l[0..colon];
            const val = std.mem.trim(u8, l[colon + 1 ..], " ");
            if (std.ascii.eqlIgnoreCase(key, "range")) range = val;
            if (std.ascii.eqlIgnoreCase(key, "authorization")) auth = val;
        }

        {
            h.mutex.lockUncancelable(h.io);
            defer h.mutex.unlock(h.io);
            try h.log.append(h.gpa, .{
                .path = try h.gpa.dupe(u8, path),
                .range = if (range) |x| try h.gpa.dupe(u8, x) else null,
                .auth = if (auth) |x| try h.gpa.dupe(u8, x) else null,
            });
        }

        if (h.faults.status_path) |sp| {
            if (std.mem.eql(u8, sp, path)) return empty(w, h.faults.status_code);
        }
        if (h.faults.require_token) |tok| {
            var expect_buf: [128]u8 = undefined;
            const expect = try std.fmt.bufPrint(&expect_buf, "Bearer {s}", .{tok});
            if (auth == null or !std.mem.eql(u8, auth.?, expect)) return empty(w, 401);
        }

        var api_buf: [256]u8 = undefined;
        const api = try std.fmt.bufPrint(&api_buf, "/api/models/{s}/{s}/revision/", .{ h.owner, h.name });
        if (std.mem.startsWith(u8, path, api) and std.mem.indexOfScalar(u8, path[api.len..], '/') == null) {
            return h.listing(w);
        }
        var res_buf: [256]u8 = undefined;
        const res = try std.fmt.bufPrint(&res_buf, "/{s}/{s}/resolve/", .{ h.owner, h.name });
        if (std.mem.startsWith(u8, path, res)) {
            const rest = path[res.len..];
            const slash = std.mem.indexOfScalar(u8, rest, '/') orelse return empty(w, 404);
            const fname = rest[slash + 1 ..];
            if (h.faults.redirect_host) |host| {
                try w.print("HTTP/1.1 302 Found\r\nLocation: http://{s}:{d}/cdn/{s}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", .{ host, h.port, fname });
                return w.flush();
            }
            return h.file(w, fname, range);
        }
        if (std.mem.startsWith(u8, path, "/cdn/")) return h.file(w, path[5..], range);
        return empty(w, 404);
    }

    fn empty(w: *std.Io.Writer, code: u16) !void {
        try w.print("HTTP/1.1 {d} Status\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", .{code});
        try w.flush();
    }

    fn readFile(h: *Hub, name: []const u8) !?[]u8 {
        const p = try std.fmt.allocPrint(h.gpa, "{s}/{s}", .{ h.root, name });
        defer h.gpa.free(p);
        return std.Io.Dir.cwd().readFileAlloc(h.io, p, h.gpa, .limited(1 << 28)) catch return null;
    }

    fn listing(h: *Hub, w: *std.Io.Writer) !void {
        var dir = try std.Io.Dir.cwd().openDir(h.io, h.root, .{ .iterate = true });
        defer dir.close(h.io);
        var names: std.ArrayList([]u8) = .empty;
        defer {
            for (names.items) |n| h.gpa.free(n);
            names.deinit(h.gpa);
        }
        var it = dir.iterate();
        while (try it.next(h.io)) |e| {
            if (e.kind == .file) try names.append(h.gpa, try h.gpa.dupe(u8, e.name));
        }
        std.mem.sort([]u8, names.items, {}, struct {
            fn lt(_: void, a: []u8, b: []u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);

        var body: std.Io.Writer.Allocating = .init(h.gpa);
        defer body.deinit();
        const bw = &body.writer;
        try bw.print("{{\"id\":\"{s}/{s}\",\"sha\":\"c0ffeec0ffeec0ffeec0ffeec0ffeec0ffeec0ff\",\"siblings\":[", .{ h.owner, h.name });
        for (names.items, 0..) |n, i| {
            const data = (try h.readFile(n)) orelse continue;
            defer h.gpa.free(data);
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
            var hex: [64]u8 = undefined;
            if (h.faults.wrong_sha) {
                @memset(&hex, '0');
            } else {
                _ = std.fmt.bufPrint(&hex, "{x}", .{&digest}) catch unreachable;
            }
            if (i != 0) try bw.writeAll(",");
            const skip = if (h.faults.no_sha) |s| std.mem.eql(u8, s, n) else false;
            if (skip) {
                try bw.print("{{\"rfilename\":\"{s}\",\"size\":{d}}}", .{ n, data.len });
            } else {
                try bw.print("{{\"rfilename\":\"{s}\",\"size\":{d},\"lfs\":{{\"sha256\":\"{s}\",\"size\":{d}}}}}", .{ n, data.len, hex, data.len });
            }
        }
        try bw.writeAll("]}");
        try w.print("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{body.written().len});
        try w.writeAll(body.written());
        try w.flush();
    }

    fn file(h: *Hub, w: *std.Io.Writer, name: []const u8, range: ?[]const u8) !void {
        const data = (try h.readFile(name)) orelse return empty(w, 404);
        defer h.gpa.free(data);
        var from: usize = 0;
        if (range) |rg| {
            const eq = std.mem.indexOfScalar(u8, rg, '=') orelse return empty(w, 400);
            const dash = std.mem.indexOfScalar(u8, rg, '-') orelse return empty(w, 400);
            from = std.fmt.parseInt(usize, rg[eq + 1 .. dash], 10) catch return empty(w, 400);
        }
        if (from > data.len) from = data.len;
        var body = data[from..];
        const promised = body.len;
        if (h.faults.truncate) body = body[0 .. body.len / 2];
        if (h.faults.drop_after) |drop| {
            h.mutex.lockUncancelable(h.io);
            defer h.mutex.unlock(h.io);
            var seen = false;
            for (h.dropped.items) |d| {
                if (d.start == from and std.mem.eql(u8, d.name, name)) seen = true;
            }
            if (!seen and body.len > drop) {
                try h.dropped.append(h.gpa, .{ .name = try h.gpa.dupe(u8, name), .start = from });
                body = body[0..drop];
            }
        }
        if (range != null) {
            try w.print("HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {d}-{d}/{d}\r\n", .{ from, data.len - 1, data.len });
        } else {
            try w.writeAll("HTTP/1.1 200 OK\r\n");
        }
        try w.print("Content-Length: {d}\r\nContent-Type: application/octet-stream\r\nConnection: close\r\n\r\n", .{promised});
        try w.writeAll(body);
        try w.flush();
    }
};
