//! Small helpers for the server tests: a free port, a process wrapper that starts `hk serve`
//! and waits until it answers, and blocking JSON requests.

const std = @import("std");
const proc = @import("proc.zig");

pub const Response = struct {
    status: u16,
    body: []u8,

    pub fn deinit(self: Response, gpa: std.mem.Allocator) void {
        gpa.free(self.body);
    }

    /// The body parsed as JSON. The caller owns the result.
    pub fn json(self: Response, gpa: std.mem.Allocator) !std.json.Parsed(std.json.Value) {
        return std.json.parseFromSlice(std.json.Value, gpa, self.body, .{});
    }
};

pub fn freePort(io: std.Io) !u16 {
    const addr = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var srv = try addr.listen(io, .{});
    defer srv.deinit(io);
    return srv.socket.address.getPort();
}

pub fn request(
    gpa: std.mem.Allocator,
    io: std.Io,
    method: std.http.Method,
    url: []const u8,
    body: ?[]const u8,
    headers: []const std.http.Header,
) !Response {
    var client: std.http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();
    var aw: std.Io.Writer.Allocating = .init(gpa);
    errdefer aw.deinit();
    const res = try client.fetch(.{
        .location = .{ .url = url },
        .method = method,
        .payload = body,
        .extra_headers = headers,
        .headers = .{ .content_type = if (body != null) .{ .override = "application/json" } else .default },
        .response_writer = &aw.writer,
        .keep_alive = false,
    });
    return .{ .status = @intFromEnum(res.status), .body = try aw.toOwnedSlice() };
}

pub const Server = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    child: std.process.Child,
    port: u16,
    url: []u8,

    pub fn start(gpa: std.mem.Allocator, io: std.Io, model: []const u8, extra: []const []const u8) !Server {
        const port = try freePort(io);
        var port_buf: [8]u8 = undefined;
        const port_s = try std.fmt.bufPrint(&port_buf, "{d}", .{port});
        var argv: std.ArrayList([]const u8) = .empty;
        defer argv.deinit(gpa);
        try argv.appendSlice(gpa, &.{ proc.exe_path, "serve", model, "--port", port_s, "--threads", "2" });
        try argv.appendSlice(gpa, extra);
        var child = try std.process.spawn(io, .{ .argv = argv.items, .stdin = .ignore, .stdout = .ignore, .stderr = .ignore });
        errdefer child.kill(io);
        const url = try std.fmt.allocPrint(gpa, "http://127.0.0.1:{d}", .{port});
        errdefer gpa.free(url);

        const health = try std.fmt.allocPrint(gpa, "{s}/health", .{url});
        defer gpa.free(health);
        var tries: usize = 0;
        while (tries < 200) : (tries += 1) {
            if (request(gpa, io, .GET, health, null, &.{})) |r| {
                r.deinit(gpa);
                return .{ .gpa = gpa, .io = io, .child = child, .port = port, .url = url };
            } else |_| {
                std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {};
            }
        }
        return error.ServerDidNotStart;
    }

    pub fn stop(self: *Server) void {
        self.child.kill(self.io);
        self.gpa.free(self.url);
    }

    fn full(self: *const Server, path: []const u8) ![]u8 {
        return std.fmt.allocPrint(self.gpa, "{s}{s}", .{ self.url, path });
    }

    pub fn get(self: *const Server, path: []const u8) !Response {
        const u = try self.full(path);
        defer self.gpa.free(u);
        return request(self.gpa, self.io, .GET, u, null, &.{});
    }

    pub fn post(self: *const Server, path: []const u8, body: []const u8, headers: []const std.http.Header) !Response {
        const u = try self.full(path);
        defer self.gpa.free(u);
        return request(self.gpa, self.io, .POST, u, body, headers);
    }

    /// A chat request with greedy decoding; `extra` is raw JSON members appended to the object
    /// (for example `,"stop":["x"]`). Returns the assistant's text, or null when the status is
    /// not 200. The caller owns the text.
    pub fn chatText(self: *const Server, content: []const u8, max_tokens: usize, extra: []const u8) !?[]u8 {
        const body = try std.fmt.allocPrint(self.gpa, "{{\"messages\":[{{\"role\":\"user\",\"content\":\"{s}\"}}],\"temperature\":0,\"max_tokens\":{d}{s}}}", .{ content, max_tokens, extra });
        defer self.gpa.free(body);
        const r = try self.post("/v1/chat/completions", body, &.{});
        defer r.deinit(self.gpa);
        if (r.status != 200) return null;
        var parsed = try r.json(self.gpa);
        defer parsed.deinit();
        const msg = parsed.value.object.get("choices").?.array.items[0].object.get("message").?.object.get("content").?.string;
        return try self.gpa.dupe(u8, msg);
    }

    pub fn metric(self: *const Server, name: []const u8) !f64 {
        const r = try self.get("/metrics");
        defer r.deinit(self.gpa);
        var lines = std.mem.splitScalar(u8, r.body, '\n');
        while (lines.next()) |l| {
            if (l.len == 0 or l[0] == '#') continue;
            var it = std.mem.tokenizeScalar(u8, l, ' ');
            const k = it.next() orelse continue;
            const v = it.next() orelse continue;
            if (std.mem.eql(u8, k, name)) return try std.fmt.parseFloat(f64, v);
        }
        return error.MetricMissing;
    }
};
