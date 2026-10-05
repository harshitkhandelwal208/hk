//! OpenAI compatible HTTP API on top of the scheduler.
//!
//! Routes: /v1/chat/completions, /v1/completions, /v1/models, /health, /metrics, /tokenize,
//! /detokenize. Responses stream as server-sent events when asked. The server binds to
//! localhost unless told otherwise and applies hard limits to header size, body size and the
//! number of concurrent connections, because it parses untrusted input.

const std = @import("std");
const scheduler_mod = @import("scheduler.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const chat_mod = @import("../chat.zig");
const sampler_mod = @import("../sampler.zig");

const Scheduler = scheduler_mod.Scheduler;
const Request = scheduler_mod.Request;

pub const Config = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    api_key: ?[]const u8 = null,
    model_name: []const u8 = "hk",
    max_body_bytes: usize = 32 << 20,
    max_connections: u32 = 128,
    /// Largest reply a client may ask for. Requests asking for more are clamped, not refused.
    max_reply_tokens: u32 = 1 << 20,
};

pub const Server = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    sched: *Scheduler,
    tok: *tokenizer_mod.Tokenizer,
    template: ?*chat_mod.ChatTemplate,
    cfg: Config,
    started_ns: i128,
    /// `Tokenizer.encode` keeps scratch space inside the tokenizer, so it is not reentrant and
    /// connection threads take this lock around it.
    tok_mutex: std.Io.Mutex = .init,
    connections: std.atomic.Value(u32) = .init(0),
    counter: std.atomic.Value(u64) = .init(1),
    stopping: std.atomic.Value(bool) = .init(false),

    pub fn run(self: *Server) !void {
        const addr = std.Io.net.IpAddress.parse(self.cfg.host, self.cfg.port) catch {
            std.debug.print("Error: '{s}' is not a valid address\n", .{self.cfg.host});
            return error.BadAddress;
        };
        var listener = try addr.listen(self.io, .{ .reuse_address = true });
        defer listener.deinit(self.io);
        std.debug.print("hk server listening on http://{s}:{d}  (model: {s}, {d} slots)\n", .{ self.cfg.host, self.cfg.port, self.cfg.model_name, self.sched.slots.len });
        if (self.cfg.api_key == null and !isLoopback(self.cfg.host)) {
            std.debug.print("Warning: listening on a non-local address without --api-key. Anyone who can reach it can use the model.\n", .{});
        }

        while (!self.stopping.load(.acquire)) {
            const stream = listener.accept(self.io) catch |e| {
                if (e == error.SocketNotListening) break;
                continue;
            };
            if (self.connections.load(.acquire) >= self.cfg.max_connections) {
                // Over the limit: refuse politely and close.
                var wbuf: [512]u8 = undefined;
                var sw = stream.writer(self.io, &wbuf);
                sw.interface.writeAll("HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\nconnection: close\r\n\r\n") catch {};
                sw.interface.flush() catch {};
                stream.close(self.io);
                continue;
            }
            _ = self.connections.fetchAdd(1, .acq_rel);
            const t = std.Thread.spawn(.{ .stack_size = 1 << 20 }, connection, .{ self, stream }) catch {
                _ = self.connections.fetchSub(1, .acq_rel);
                stream.close(self.io);
                continue;
            };
            t.detach();
        }
    }
};

fn isLoopback(host: []const u8) bool {
    return std.mem.eql(u8, host, "127.0.0.1") or std.mem.eql(u8, host, "::1") or std.mem.eql(u8, host, "localhost");
}

fn connection(srv: *Server, stream: std.Io.net.Stream) void {
    defer {
        stream.close(srv.io);
        _ = srv.connections.fetchSub(1, .acq_rel);
    }
    var rbuf: [16 * 1024]u8 = undefined;
    var wbuf: [16 * 1024]u8 = undefined;
    var sr = stream.reader(srv.io, &rbuf);
    var sw = stream.writer(srv.io, &wbuf);
    var http = std.http.Server.init(&sr.interface, &sw.interface);
    while (true) {
        var req = http.receiveHead() catch return;
        const keep = handle(srv, &req) catch return;
        if (!keep) return;
    }
}

// ---------------------------------------------------------------------------------------
// Small response helpers
// ---------------------------------------------------------------------------------------

const cors = [_]std.http.Header{
    .{ .name = "access-control-allow-origin", .value = "*" },
    .{ .name = "access-control-allow-headers", .value = "authorization, content-type" },
    .{ .name = "access-control-allow-methods", .value = "GET, POST, OPTIONS" },
};

const json_headers = cors ++ [_]std.http.Header{.{ .name = "content-type", .value = "application/json" }};
const sse_headers = cors ++ [_]std.http.Header{
    .{ .name = "content-type", .value = "text/event-stream" },
    .{ .name = "cache-control", .value = "no-cache" },
};

/// Writes `s` as a JSON string. JSON text must be valid UTF-8, but model output is raw bytes
/// that may contain partial or invalid sequences, so those are replaced with U+FFFD instead of
/// producing a response no client could parse.
fn writeJsonString(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c >= 0x80) {
            const n = std.unicode.utf8ByteSequenceLength(c) catch 0;
            if (n > 1 and i + n <= s.len and std.unicode.utf8ValidateSlice(s[i .. i + n])) {
                try w.writeAll(s[i .. i + n]);
                i += n;
            } else {
                try w.writeAll("\\ufffd");
                i += 1;
            }
            continue;
        }
        switch (c) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            0x08 => try w.writeAll("\\b"),
            0x0C => try w.writeAll("\\f"),
            else => if (c < 0x20) try w.print("\\u{x:0>4}", .{c}) else try w.writeByte(c),
        }
        i += 1;
    }
    try w.writeByte('"');
}

fn respondJson(req: *std.http.Server.Request, status: std.http.Status, body: []const u8) !void {
    try req.respond(body, .{ .status = status, .extra_headers = &json_headers });
}

fn respondError(req: *std.http.Server.Request, a: std.mem.Allocator, status: std.http.Status, kind: []const u8, message: []const u8) !void {
    var aw = std.Io.Writer.Allocating.init(a);
    const w = &aw.writer;
    try w.writeAll("{\"error\":{\"message\":");
    try writeJsonString(w, message);
    try w.print(",\"type\":\"{s}\",\"code\":{d}}}}}", .{ kind, @intFromEnum(status) });
    try respondJson(req, status, aw.written());
}

fn authorized(srv: *Server, req: *std.http.Server.Request) bool {
    const key = srv.cfg.api_key orelse return true;
    var it = req.iterateHeaders();
    while (it.next()) |h| {
        if (!std.ascii.eqlIgnoreCase(h.name, "authorization")) continue;
        const prefix = "Bearer ";
        if (!std.mem.startsWith(u8, h.value, prefix)) return false;
        const given = h.value[prefix.len..];
        // Compare in constant time so the key cannot be guessed byte by byte.
        var diff: u8 = @intFromBool(given.len != key.len);
        const n = @min(given.len, key.len);
        for (0..n) |i| diff |= given[i] ^ key[i];
        return diff == 0;
    }
    return false;
}

/// How much of an oversized body is read and thrown away after the 413 has been sent.
const drain_limit: usize = 64 << 20;

/// Reads the request body. When it exceeds `limit`, `rest` is set to the body reader so the caller
/// can answer 413 and then drain what is still in flight: closing a connection that has unread
/// data makes the peer's stack send a reset, and on some systems (macOS) that reset wipes out the
/// response before the client has read it.
fn readBody(req: *std.http.Server.Request, a: std.mem.Allocator, limit: usize, rest: *?*std.Io.Reader) ![]u8 {
    const buf = try a.alloc(u8, 4096);
    const reader = try req.readerExpectContinue(buf);
    return reader.allocRemaining(a, .limited(limit)) catch |e| switch (e) {
        error.StreamTooLong => {
            rest.* = reader;
            return error.BodyTooLarge;
        },
        else => error.ReadFailed,
    };
}

// ---------------------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------------------

/// Handles one request. Returns whether the connection may be reused.
fn handle(srv: *Server, req: *std.http.Server.Request) !bool {
    var arena_state = std.heap.ArenaAllocator.init(srv.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    // The head strings point into the connection's read buffer, which the library reuses as soon
    // as the body is read. Copy what routing needs before that happens.
    const target = req.head.target;
    const path_view = if (std.mem.indexOfScalar(u8, target, '?')) |q| target[0..q] else target;
    const path = try a.dupe(u8, path_view);
    const method = req.head.method;

    if (method == .OPTIONS) {
        try req.respond("", .{ .status = .no_content, .extra_headers = &cors });
        return true;
    }
    if (std.mem.eql(u8, path, "/health") or std.mem.eql(u8, path, "/v1/health")) {
        try respondJson(req, .ok, "{\"status\":\"ok\"}");
        return true;
    }
    if (!authorized(srv, req)) {
        try respondError(req, a, .unauthorized, "invalid_request_error", "a valid API key is required");
        return true;
    }
    if (method == .GET and (std.mem.eql(u8, path, "/v1/models") or std.mem.eql(u8, path, "/models"))) {
        var aw = std.Io.Writer.Allocating.init(a);
        const w = &aw.writer;
        try w.writeAll("{\"object\":\"list\",\"data\":[{\"id\":");
        try writeJsonString(w, srv.cfg.model_name);
        try w.writeAll(",\"object\":\"model\",\"created\":0,\"owned_by\":\"hk\"}]}");
        try respondJson(req, .ok, aw.written());
        return true;
    }
    if (method == .GET and std.mem.eql(u8, path, "/metrics")) {
        try handleMetrics(srv, req, a);
        return true;
    }
    if (method == .POST) {
        var rest: ?*std.Io.Reader = null;
        const body = readBody(req, a, srv.cfg.max_body_bytes, &rest) catch |e| {
            const status: std.http.Status = if (e == error.BodyTooLarge) .payload_too_large else .bad_request;
            try respondError(req, a, status, "invalid_request_error", if (e == error.BodyTooLarge) "the request body is too large" else "could not read the request body");
            if (rest) |r| _ = r.discard(.limited(drain_limit)) catch {};
            return false;
        };
        if (std.mem.eql(u8, path, "/v1/chat/completions") or std.mem.eql(u8, path, "/chat/completions")) {
            try handleChat(srv, req, a, body);
            return true;
        }
        if (std.mem.eql(u8, path, "/v1/completions") or std.mem.eql(u8, path, "/completions")) {
            try handleCompletion(srv, req, a, body);
            return true;
        }
        if (std.mem.eql(u8, path, "/tokenize")) {
            try handleTokenize(srv, req, a, body);
            return true;
        }
        if (std.mem.eql(u8, path, "/detokenize")) {
            try handleDetokenize(srv, req, a, body);
            return true;
        }
    }
    try respondError(req, a, .not_found, "invalid_request_error", "unknown route");
    return true;
}

fn handleMetrics(srv: *Server, req: *std.http.Server.Request, a: std.mem.Allocator) !void {
    const st = &srv.sched.stats;
    var aw = std.Io.Writer.Allocating.init(a);
    const w = &aw.writer;
    const up = @as(f64, @floatFromInt(std.Io.Timestamp.now(srv.io, .awake).nanoseconds - srv.started_ns)) / 1e9;
    try w.print(
        \\# TYPE hk_requests_total counter
        \\hk_requests_total {d}
        \\# TYPE hk_prompt_tokens_total counter
        \\hk_prompt_tokens_total {d}
        \\# TYPE hk_cached_prompt_tokens_total counter
        \\hk_cached_prompt_tokens_total {d}
        \\# TYPE hk_generated_tokens_total counter
        \\hk_generated_tokens_total {d}
        \\# TYPE hk_slots_active gauge
        \\hk_slots_active {d}
        \\# TYPE hk_requests_queued gauge
        \\hk_requests_queued {d}
        \\# TYPE hk_uptime_seconds gauge
        \\hk_uptime_seconds {d:.1}
        \\
    , .{
        st.requests.load(.monotonic),
        st.prompt_tokens.load(.monotonic),
        st.cached_tokens.load(.monotonic),
        st.generated_tokens.load(.monotonic),
        st.active_slots.load(.monotonic),
        st.queued.load(.monotonic),
        up,
    });
    try req.respond(aw.written(), .{ .extra_headers = &(cors ++ [_]std.http.Header{.{ .name = "content-type", .value = "text/plain; version=0.0.4" }}) });
}

// ---------------------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------------------

fn getObj(v: std.json.Value) ?std.json.ObjectMap {
    return if (v == .object) v.object else null;
}

fn num(v: ?std.json.Value) ?f64 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

fn uint(v: ?std.json.Value) ?u64 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        .float => |f| if (f >= 0) @intFromFloat(f) else null,
        else => null,
    };
}

const Parsed = struct {
    params: sampler_mod.Params = .{},
    max_tokens: ?u32 = null,
    stream: bool = false,
    include_usage: bool = false,
    stop: []const []const u8 = &.{},
    want_logprob: bool = false,
};

fn parseCommon(srv: *Server, a: std.mem.Allocator, obj: std.json.ObjectMap) !Parsed {
    var p = Parsed{};
    if (num(obj.get("temperature"))) |t| p.params.temperature = @floatCast(@max(0, t));
    if (num(obj.get("top_p"))) |t| p.params.top_p = @floatCast(t);
    if (uint(obj.get("top_k"))) |t| p.params.top_k = @intCast(@min(t, std.math.maxInt(u32)));
    if (num(obj.get("min_p"))) |t| p.params.min_p = @floatCast(t);
    if (num(obj.get("frequency_penalty"))) |t| p.params.frequency_penalty = @floatCast(t);
    if (num(obj.get("presence_penalty"))) |t| p.params.presence_penalty = @floatCast(t);
    if (num(obj.get("repeat_penalty"))) |t| p.params.repeat_penalty = @floatCast(t);
    if (uint(obj.get("seed"))) |s| p.params.seed = s;
    const mt = uint(obj.get("max_completion_tokens")) orelse uint(obj.get("max_tokens"));
    if (mt) |m| p.max_tokens = @intCast(@min(m, srv.cfg.max_reply_tokens));
    if (obj.get("stream")) |s| if (s == .bool) {
        p.stream = s.bool;
    };
    if (obj.get("stream_options")) |so| if (getObj(so)) |o| if (o.get("include_usage")) |iu| if (iu == .bool) {
        p.include_usage = iu.bool;
    };
    if (obj.get("logprobs")) |lp| if (lp == .bool) {
        p.want_logprob = lp.bool;
    };
    if (obj.get("stop")) |s| {
        var list: std.ArrayList([]const u8) = .empty;
        switch (s) {
            .string => |t| try list.append(a, t),
            .array => |arr| for (arr.items) |it| if (it == .string) {
                if (list.items.len < 16) try list.append(a, it.string);
            },
            else => {},
        }
        p.stop = try list.toOwnedSlice(a);
    }
    if (obj.get("logit_bias")) |lb| if (getObj(lb)) |o| {
        var biases: std.ArrayList(sampler_mod.Bias) = .empty;
        var it = o.iterator();
        while (it.next()) |kv| {
            const id = std.fmt.parseInt(u32, kv.key_ptr.*, 10) catch continue;
            const b = num(kv.value_ptr.*) orelse continue;
            if (biases.items.len < 1024) try biases.append(a, .{ .token = id, .bias = @floatCast(b) });
        }
        p.params.logit_bias = try biases.toOwnedSlice(a);
    };
    return p;
}

// ---------------------------------------------------------------------------------------
// Generation plumbing shared by chat and completions
// ---------------------------------------------------------------------------------------

fn newId(srv: *Server, a: std.mem.Allocator, prefix: []const u8) ![]u8 {
    const n = srv.counter.fetchAdd(1, .monotonic);
    const t: u64 = @intCast(@divTrunc(std.Io.Timestamp.now(srv.io, .real).nanoseconds, std.time.ns_per_ms));
    return std.fmt.allocPrint(a, "{s}-{x}{x}", .{ prefix, t, n });
}

fn unixSeconds(srv: *Server) i64 {
    return @intCast(@divTrunc(std.Io.Timestamp.now(srv.io, .real).nanoseconds, std.time.ns_per_s));
}

/// Submits a prepared prompt and returns the live request. Maps scheduler refusals to HTTP.
fn submitRequest(srv: *Server, req: *std.http.Server.Request, a: std.mem.Allocator, prompt_ids: []const u32, p: Parsed) !?*Request {
    const ctx = srv.sched.slots[0].sess.kv.n_ctx;
    if (prompt_ids.len + 1 >= ctx) {
        var msg: [160]u8 = undefined;
        try respondError(req, a, .bad_request, "context_length_exceeded", std.fmt.bufPrint(&msg, "the prompt has {d} tokens but the context window is {d}", .{ prompt_ids.len, ctx }) catch "the prompt is too long");
        return null;
    }
    const room: u32 = @intCast(ctx - prompt_ids.len - 1);
    const r = try srv.allocator.create(Request);
    errdefer srv.allocator.destroy(r);
    r.* = .{
        .allocator = srv.allocator,
        .io = srv.io,
        .prompt = try srv.allocator.dupe(u32, prompt_ids),
        .max_tokens = @min(p.max_tokens orelse room, room),
        .params = p.params,
        .stop = p.stop,
        .want_logprob = p.want_logprob,
    };
    srv.sched.submit(r) catch |e| {
        r.deinit();
        srv.allocator.destroy(r);
        switch (e) {
            error.QueueFull => try respondError(req, a, .service_unavailable, "server_busy", "all slots are busy and the queue is full; retry shortly"),
            error.TooLong => try respondError(req, a, .bad_request, "context_length_exceeded", "the prompt does not fit the context window"),
            error.OutOfMemory => try respondError(req, a, .internal_server_error, "server_error", "out of memory"),
        }
        return null;
    };
    return r;
}

fn finishName(r: *const Request) []const u8 {
    return switch (r.finish) {
        .stop => "stop",
        .length => "length",
        else => "stop",
    };
}

const Collected = struct { text: []u8, request: *Request };

/// Waits for a request to finish and gathers its whole reply.
fn collect(srv: *Server, a: std.mem.Allocator, r: *Request) !Collected {
    var text: std.ArrayList(u8) = .empty;
    var events: std.ArrayList(scheduler_mod.Event) = .empty;
    while (true) {
        events.clearRetainingCapacity();
        const done = try r.wait(&events, a);
        for (events.items) |e| {
            try text.appendSlice(a, e.text);
            srv.allocator.free(e.text);
        }
        if (done) break;
    }
    return .{ .text = try text.toOwnedSlice(a), .request = r };
}

fn releaseRequest(srv: *Server, r: *Request) void {
    r.deinit();
    srv.allocator.destroy(r);
}

fn writeUsage(w: *std.Io.Writer, r: *const Request) !void {
    try w.print("\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_details\":{{\"cached_tokens\":{d}}}}}", .{
        r.prompt_tokens, r.completion_tokens, r.prompt_tokens + r.completion_tokens, r.cached_tokens,
    });
}

/// Streams a request as server-sent events. `chat` selects the chunk shape.
fn streamResponse(srv: *Server, req: *std.http.Server.Request, a: std.mem.Allocator, r: *Request, id: []const u8, chat: bool, include_usage: bool) !void {
    var sbuf: [4096]u8 = undefined;
    var bw = try req.respondStreaming(&sbuf, .{ .respond_options = .{ .extra_headers = &sse_headers } });
    const w = &bw.writer;
    const created = unixSeconds(srv);
    var broken = false;

    const writeChunk = struct {
        fn f(wr: *std.Io.Writer, id_: []const u8, created_: i64, model: []const u8, is_chat: bool, role: bool, delta: ?[]const u8, finish: ?[]const u8) !void {
            try wr.print("data: {{\"id\":\"{s}\",\"object\":\"{s}\",\"created\":{d},\"model\":", .{ id_, if (is_chat) "chat.completion.chunk" else "text_completion", created_ });
            try writeJsonString(wr, model);
            try wr.writeAll(",\"choices\":[{\"index\":0,");
            if (is_chat) {
                try wr.writeAll("\"delta\":{");
                var first = true;
                if (role) {
                    try wr.writeAll("\"role\":\"assistant\"");
                    first = false;
                }
                if (delta) |d| {
                    if (!first) try wr.writeAll(",");
                    try wr.writeAll("\"content\":");
                    try writeJsonString(wr, d);
                }
                try wr.writeAll("}");
            } else {
                try wr.writeAll("\"text\":");
                try writeJsonString(wr, delta orelse "");
            }
            if (finish) |fr| try wr.print(",\"finish_reason\":\"{s}\"", .{fr}) else try wr.writeAll(",\"finish_reason\":null");
            try wr.writeAll("}]}\n\n");
        }
    }.f;

    if (chat) writeChunk(w, id, created, srv.cfg.model_name, true, true, null, null) catch {
        broken = true;
    };
    if (!broken) bw.flush() catch {
        broken = true;
    };

    var events: std.ArrayList(scheduler_mod.Event) = .empty;
    while (true) {
        events.clearRetainingCapacity();
        const done = try r.wait(&events, a);
        for (events.items) |e| {
            if (!broken) {
                writeChunk(w, id, created, srv.cfg.model_name, chat, false, e.text, null) catch {
                    broken = true;
                };
                bw.flush() catch {
                    broken = true;
                };
            }
            srv.allocator.free(e.text);
            // A write failure means the client left. Stop generating for it.
            if (broken) r.cancel();
        }
        if (done) break;
    }
    if (!broken) {
        if (r.finish == .failed) {
            w.writeAll("data: {\"error\":{\"message\":") catch {};
            writeJsonString(w, r.failMessage()) catch {};
            w.writeAll("}}\n\n") catch {};
        } else {
            writeChunk(w, id, created, srv.cfg.model_name, chat, false, null, finishName(r)) catch {};
        }
        if (include_usage) {
            w.print("data: {{\"id\":\"{s}\",\"object\":\"{s}\",\"created\":{d},\"model\":", .{ id, if (chat) "chat.completion.chunk" else "text_completion", created }) catch {};
            writeJsonString(w, srv.cfg.model_name) catch {};
            w.writeAll(",\"choices\":[],") catch {};
            writeUsage(w, r) catch {};
            w.writeAll("}\n\n") catch {};
        }
        w.writeAll("data: [DONE]\n\n") catch {};
        bw.end() catch {};
    }
}

// ---------------------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------------------

/// Turns OpenAI style messages into what templates expect: text parts are joined into one
/// string, and tool call arguments (sent as a JSON string) become objects.
fn normalizeMessages(a: std.mem.Allocator, v: std.json.Value) !std.json.Value {
    if (v != .array) return error.BadMessages;
    for (v.array.items) |*m| {
        if (m.* != .object) return error.BadMessages;
        if (m.object.getPtr("content")) |c| {
            if (c.* == .array) {
                var joined: std.ArrayList(u8) = .empty;
                for (c.array.items) |part| {
                    const po = getObj(part) orelse return error.BadMessages;
                    const ty = if (po.get("type")) |t| (if (t == .string) t.string else "") else "";
                    if (std.mem.eql(u8, ty, "text")) {
                        if (po.get("text")) |t| if (t == .string) try joined.appendSlice(a, t.string);
                    } else {
                        return error.UnsupportedContent;
                    }
                }
                c.* = .{ .string = try joined.toOwnedSlice(a) };
            }
        }
        if (m.object.getPtr("tool_calls")) |tc| if (tc.* == .array) {
            for (tc.array.items) |*call| {
                const co = getObj(call.*) orelse continue;
                if (co.get("function")) |f| if (f == .object) {
                    if (f.object.getPtr("arguments")) |args| if (args.* == .string) {
                        if (std.json.parseFromSliceLeaky(std.json.Value, a, args.string, .{})) |parsed| args.* = parsed else |_| {}
                    };
                };
            }
        };
    }
    return v;
}

fn encodePrompt(srv: *Server, a: std.mem.Allocator, text: []const u8, template_rendered: bool) ![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    const tok = srv.tok;
    // A rendered template normally writes the BOS token itself; do not add a second one.
    var add_special = true;
    if (template_rendered) {
        if (tok.bos) |b| {
            if (std.mem.startsWith(u8, text, tok.piece(b))) add_special = false;
        }
    }
    srv.tok_mutex.lockUncancelable(srv.io);
    defer srv.tok_mutex.unlock(srv.io);
    try tok.encode(text, .{ .add_special = add_special, .parse_special = true }, &ids);
    return ids.toOwnedSlice(a);
}

fn handleChat(srv: *Server, req: *std.http.Server.Request, a: std.mem.Allocator, body: []const u8) !void {
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch {
        return respondError(req, a, .bad_request, "invalid_request_error", "the request body is not valid JSON");
    };
    const obj = getObj(parsed) orelse return respondError(req, a, .bad_request, "invalid_request_error", "the request body must be a JSON object");
    const msgs_v = obj.get("messages") orelse return respondError(req, a, .bad_request, "invalid_request_error", "'messages' is required");
    const template = srv.template orelse return respondError(req, a, .bad_request, "invalid_request_error", "this model has no chat template; use /v1/completions with a prompt");

    const messages = normalizeMessages(a, msgs_v) catch |e| {
        return respondError(req, a, .bad_request, "invalid_request_error", if (e == error.UnsupportedContent) "only text content is supported" else "'messages' must be an array of message objects");
    };
    var opts = chat_mod.template.RenderOptions{};
    if (obj.get("tools")) |t| if (t == .array and t.array.items.len > 0) {
        opts.tools_json = try std.json.Stringify.valueAlloc(a, t, .{});
    };
    if (obj.get("chat_template_kwargs")) |k| if (getObj(k)) |ko| {
        if (ko.get("enable_thinking")) |et| if (et == .bool) {
            opts.enable_thinking = et.bool;
        };
    };
    if (obj.get("enable_thinking")) |et| if (et == .bool) {
        opts.enable_thinking = et.bool;
    };

    var jd = chat_mod.jinja.Diag{};
    const prompt_text = template.renderJson(a, messages, opts, &jd) catch {
        var msg: [400]u8 = undefined;
        return respondError(req, a, .bad_request, "invalid_request_error", std.fmt.bufPrint(&msg, "the chat template rejected these messages: {s}", .{jd.message()}) catch "the chat template rejected these messages");
    };
    const ids = try encodePrompt(srv, a, prompt_text, true);
    const p = try parseCommon(srv, a, obj);
    const r = (try submitRequest(srv, req, a, ids, p)) orelse return;
    defer releaseRequest(srv, r);
    const id = try newId(srv, a, "chatcmpl");

    if (p.stream) return streamResponse(srv, req, a, r, id, true, p.include_usage);

    const got = try collect(srv, a, r);
    if (r.finish == .failed) return respondError(req, a, .internal_server_error, "server_error", r.failMessage());
    var aw = std.Io.Writer.Allocating.init(a);
    const w = &aw.writer;
    try w.print("{{\"id\":\"{s}\",\"object\":\"chat.completion\",\"created\":{d},\"model\":", .{ id, unixSeconds(srv) });
    try writeJsonString(w, srv.cfg.model_name);
    try w.writeAll(",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    try writeJsonString(w, got.text);
    try w.print("}},\"finish_reason\":\"{s}\"}}],", .{finishName(r)});
    try writeUsage(w, r);
    try w.writeAll("}");
    try respondJson(req, .ok, aw.written());
}

// ---------------------------------------------------------------------------------------
// Plain completions, tokenize, detokenize
// ---------------------------------------------------------------------------------------

fn handleCompletion(srv: *Server, req: *std.http.Server.Request, a: std.mem.Allocator, body: []const u8) !void {
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch {
        return respondError(req, a, .bad_request, "invalid_request_error", "the request body is not valid JSON");
    };
    const obj = getObj(parsed) orelse return respondError(req, a, .bad_request, "invalid_request_error", "the request body must be a JSON object");
    const prompt_v = obj.get("prompt") orelse return respondError(req, a, .bad_request, "invalid_request_error", "'prompt' is required");
    const prompt: []const u8 = switch (prompt_v) {
        .string => |s| s,
        .array => |arr| if (arr.items.len == 1 and arr.items[0] == .string) arr.items[0].string else return respondError(req, a, .bad_request, "invalid_request_error", "only a single prompt is supported"),
        else => return respondError(req, a, .bad_request, "invalid_request_error", "'prompt' must be a string"),
    };
    const ids = try encodePrompt(srv, a, prompt, false);
    if (ids.len == 0) return respondError(req, a, .bad_request, "invalid_request_error", "the prompt produced no tokens");
    const p = try parseCommon(srv, a, obj);
    const r = (try submitRequest(srv, req, a, ids, p)) orelse return;
    defer releaseRequest(srv, r);
    const id = try newId(srv, a, "cmpl");
    if (p.stream) return streamResponse(srv, req, a, r, id, false, p.include_usage);

    const got = try collect(srv, a, r);
    if (r.finish == .failed) return respondError(req, a, .internal_server_error, "server_error", r.failMessage());
    var aw = std.Io.Writer.Allocating.init(a);
    const w = &aw.writer;
    try w.print("{{\"id\":\"{s}\",\"object\":\"text_completion\",\"created\":{d},\"model\":", .{ id, unixSeconds(srv) });
    try writeJsonString(w, srv.cfg.model_name);
    try w.writeAll(",\"choices\":[{\"index\":0,\"text\":");
    try writeJsonString(w, got.text);
    try w.print(",\"finish_reason\":\"{s}\"}}],", .{finishName(r)});
    try writeUsage(w, r);
    try w.writeAll("}");
    try respondJson(req, .ok, aw.written());
}

fn handleTokenize(srv: *Server, req: *std.http.Server.Request, a: std.mem.Allocator, body: []const u8) !void {
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch return respondError(req, a, .bad_request, "invalid_request_error", "the request body is not valid JSON");
    const obj = getObj(parsed) orelse return respondError(req, a, .bad_request, "invalid_request_error", "the request body must be a JSON object");
    const content = if (obj.get("content")) |c| (if (c == .string) c.string else null) else null;
    const text = content orelse return respondError(req, a, .bad_request, "invalid_request_error", "'content' must be a string");
    const add_special = if (obj.get("add_special")) |v| (v == .bool and v.bool) else false;
    var ids: std.ArrayList(u32) = .empty;
    {
        srv.tok_mutex.lockUncancelable(srv.io);
        defer srv.tok_mutex.unlock(srv.io);
        try srv.tok.encode(text, .{ .add_special = add_special, .parse_special = true }, &ids);
    }
    var aw = std.Io.Writer.Allocating.init(a);
    const w = &aw.writer;
    try w.writeAll("{\"tokens\":[");
    for (ids.items, 0..) |t, i| {
        if (i > 0) try w.writeByte(',');
        try w.print("{d}", .{t});
    }
    try w.writeAll("]}");
    try respondJson(req, .ok, aw.written());
}

fn handleDetokenize(srv: *Server, req: *std.http.Server.Request, a: std.mem.Allocator, body: []const u8) !void {
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch return respondError(req, a, .bad_request, "invalid_request_error", "the request body is not valid JSON");
    const obj = getObj(parsed) orelse return respondError(req, a, .bad_request, "invalid_request_error", "the request body must be a JSON object");
    const arr = if (obj.get("tokens")) |t| (if (t == .array) t.array.items else null) else null;
    const items = arr orelse return respondError(req, a, .bad_request, "invalid_request_error", "'tokens' must be an array");
    var ids: std.ArrayList(u32) = .empty;
    for (items) |it| {
        const id = uint(it) orelse return respondError(req, a, .bad_request, "invalid_request_error", "'tokens' must contain non-negative integers");
        try ids.append(a, @intCast(@min(id, std.math.maxInt(u32))));
    }
    var text: std.ArrayList(u8) = .empty;
    try srv.tok.decode(ids.items, true, &text);
    var aw = std.Io.Writer.Allocating.init(a);
    const w = &aw.writer;
    try w.writeAll("{\"content\":");
    try writeJsonString(w, text.items);
    try w.writeAll("}");
    try respondJson(req, .ok, aw.written());
}

test "invalid utf8 is replaced, not passed through" {
    var buf: [128]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try writeJsonString(&w, "a\xff\xe4\xb8\"b\x01\xe4\xb8\xad");
    const out = w.buffered();
    try std.testing.expect(std.unicode.utf8ValidateSlice(out));
    try std.testing.expectEqualStrings("\"a\\ufffd\\ufffd\\ufffd\\\"b\\u0001\xe4\xb8\xad\"", out);
}
