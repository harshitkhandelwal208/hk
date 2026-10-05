//! End to end tests of `hk serve` on a tiny model with a chat template.

const std = @import("std");
const proc = @import("support/proc.zig");
const tiny = @import("support/tiny_model.zig");
const http = @import("support/http.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;

fn contains(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}

/// Builds the tiny model once per test and returns the path of the converted `.hk` file.
fn makeModel(tmp: *std.testing.TmpDir) ![]u8 {
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const g = try std.fmt.allocPrint(gpa, "{s}/tiny.gguf", .{dir});
    defer gpa.free(g);
    try tiny.makeTinyGguf(gpa, io, g, .{ .chat_template = tiny.chatml });
    const out = try std.fmt.allocPrint(gpa, "{s}/tiny.hk", .{dir});
    errdefer gpa.free(out);
    const r = try proc.run(gpa, io, &.{ "convert-gguf", g, out }, null);
    defer r.deinit(gpa);
    if (!r.ok()) return error.ConvertFailed;
    return out;
}

fn expectStatus(srv: *const http.Server, path: []const u8, body: []const u8, want: u16) !void {
    const r = try srv.post(path, body, &.{});
    defer r.deinit(gpa);
    if (r.status != want) {
        std.debug.print("POST {s} {s}: status {d}, wanted {d}: {s}\n", .{ path, body, r.status, want, r.body });
        return error.WrongStatus;
    }
}

/// The text of a non streaming greedy chat request, which must succeed.
fn chat(srv: *const http.Server, content: []const u8, max_tokens: usize) ![]u8 {
    return (try srv.chatText(content, max_tokens, "")) orelse error.ChatFailed;
}

const Worker = struct {
    srv: *const http.Server,
    prompt: []const u8,
    out: ?[]u8 = null,
    err: ?anyerror = null,

    fn run(w: *Worker) void {
        w.out = w.srv.chatText(w.prompt, 20, "") catch |e| {
            w.err = e;
            return;
        };
    }
};

const TokenizeWorker = struct {
    srv: *const http.Server,
    body: []const u8,
    out: ?[]u8 = null,
    err: ?anyerror = null,

    fn run(w: *TokenizeWorker) void {
        const r = w.srv.post("/tokenize", w.body, &.{}) catch |e| {
            w.err = e;
            return;
        };
        w.out = r.body;
    }
};

test "server end to end" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const model = try makeModel(&tmp);
    defer gpa.free(model);
    var srv = try http.Server.start(gpa, io, model, &.{ "--slots", "4", "--ctx", "512" });
    defer srv.stop();

    // Basics: health, models, one completion with consistent usage.
    {
        const h = try srv.get("/health");
        defer h.deinit(gpa);
        try std.testing.expectEqualStrings("{\"status\":\"ok\"}", h.body);
        const m = try srv.get("/v1/models");
        defer m.deinit(gpa);
        var pm = try m.json(gpa);
        defer pm.deinit();
        try std.testing.expectEqualStrings("tiny", pm.value.object.get("data").?.array.items[0].object.get("id").?.string);

        const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"temperature\":0,\"max_tokens\":20}";
        const r = try srv.post("/v1/chat/completions", body, &.{});
        defer r.deinit(gpa);
        try std.testing.expectEqual(@as(u16, 200), r.status);
        var p = try r.json(gpa);
        defer p.deinit();
        const root = p.value.object;
        try std.testing.expectEqualStrings("chat.completion", root.get("object").?.string);
        const choice = root.get("choices").?.array.items[0].object;
        try std.testing.expectEqualStrings("assistant", choice.get("message").?.object.get("role").?.string);
        const usage = root.get("usage").?.object;
        const completion = usage.get("completion_tokens").?.integer;
        try std.testing.expect(completion > 0);
        try std.testing.expectEqual(usage.get("prompt_tokens").?.integer + completion, usage.get("total_tokens").?.integer);
        const fin = choice.get("finish_reason").?.string;
        try std.testing.expect(std.mem.eql(u8, fin, "length") or std.mem.eql(u8, fin, "stop"));
    }

    // Greedy decoding repeats exactly.
    {
        const a = try chat(&srv, "hello", 20);
        defer gpa.free(a);
        const b = try chat(&srv, "hello", 20);
        defer gpa.free(b);
        try std.testing.expectEqualStrings(a, b);
    }

    // A streamed answer is the same text as the non streamed one, ends with [DONE], and reports usage.
    {
        const want = try chat(&srv, "hello world", 20);
        defer gpa.free(want);
        const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hello world\"}],\"temperature\":0,\"max_tokens\":20,\"stream\":true,\"stream_options\":{\"include_usage\":true}}";
        const r = try srv.post("/v1/chat/completions", body, &.{});
        defer r.deinit(gpa);
        try std.testing.expectEqual(@as(u16, 200), r.status);
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        var done = false;
        var saw_usage = false;
        var lines = std.mem.splitScalar(u8, r.body, '\n');
        while (lines.next()) |raw| {
            const line = std.mem.trim(u8, raw, " \r");
            if (!std.mem.startsWith(u8, line, "data: ")) continue;
            const payload = line[6..];
            if (std.mem.eql(u8, payload, "[DONE]")) {
                done = true;
                break;
            }
            var ev = try std.json.parseFromSlice(std.json.Value, gpa, payload, .{});
            defer ev.deinit();
            if (ev.value.object.get("usage")) |u| {
                if (u == .object) saw_usage = true;
            }
            for (ev.value.object.get("choices").?.array.items) |ch| {
                if (ch.object.get("delta").?.object.get("content")) |c| {
                    if (c == .string) try text.appendSlice(gpa, c.string);
                }
            }
        }
        try std.testing.expect(done and saw_usage);
        try std.testing.expectEqualStrings(want, text.items);
    }

    // Twelve requests over four slots, batched together, give the same answers as one at a time.
    {
        var prompts: [12][]u8 = undefined;
        var sequential: [12][]u8 = undefined;
        var made: usize = 0;
        defer for (0..made) |i| {
            gpa.free(prompts[i]);
            gpa.free(sequential[i]);
        };
        for (0..12) |i| {
            prompts[i] = try std.fmt.allocPrint(gpa, "prompt number {d} {s}", .{ i, "xxxxxxxxxxxx"[0..i] });
            sequential[i] = try chat(&srv, prompts[i], 20);
            made += 1;
        }
        var workers: [12]Worker = undefined;
        var threads: [12]std.Thread = undefined;
        for (0..12) |i| {
            workers[i] = .{ .srv = &srv, .prompt = prompts[i] };
            threads[i] = try std.Thread.spawn(.{}, Worker.run, .{&workers[i]});
        }
        for (threads) |t| t.join();
        for (0..12) |i| {
            defer if (workers[i].out) |o| gpa.free(o);
            try std.testing.expect(workers[i].err == null);
            try std.testing.expectEqualStrings(sequential[i], workers[i].out.?);
        }
        std.Io.sleep(io, .fromMilliseconds(300), .awake) catch {};
        try std.testing.expectEqual(@as(f64, 0), try srv.metric("hk_slots_active"));
    }

    // The prompt cache reuses a shared prefix.
    {
        const long = "shared system text " ** 6;
        const b1 = "{\"messages\":[{\"role\":\"system\",\"content\":\"" ++ long ++ "\"},{\"role\":\"user\",\"content\":\"one\"}],\"temperature\":0,\"max_tokens\":4}";
        const b2 = "{\"messages\":[{\"role\":\"system\",\"content\":\"" ++ long ++ "\"},{\"role\":\"user\",\"content\":\"two\"}],\"temperature\":0,\"max_tokens\":4}";
        const r1 = try srv.post("/v1/chat/completions", b1, &.{});
        r1.deinit(gpa);
        const r2 = try srv.post("/v1/chat/completions", b2, &.{});
        defer r2.deinit(gpa);
        var p = try r2.json(gpa);
        defer p.deinit();
        const cached = p.value.object.get("usage").?.object.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer;
        try std.testing.expect(cached >= 50);
    }

    // A stop string cuts the text and reports "stop"; max_tokens is respected.
    {
        const full = try chat(&srv, "hello", 30);
        defer gpa.free(full);
        try std.testing.expect(full.len > 8);
        const stop = full[5..8];
        const extra = try std.fmt.allocPrint(gpa, ",\"stop\":[{f}]", .{std.json.fmt(stop, .{})});
        defer gpa.free(extra);
        const body = try std.fmt.allocPrint(gpa, "{{\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}],\"temperature\":0,\"max_tokens\":30{s}}}", .{extra});
        defer gpa.free(body);
        const r = try srv.post("/v1/chat/completions", body, &.{});
        defer r.deinit(gpa);
        var p = try r.json(gpa);
        defer p.deinit();
        const ch = p.value.object.get("choices").?.array.items[0].object;
        const idx = std.mem.indexOf(u8, full, stop).?;
        try std.testing.expectEqualStrings(full[0..idx], ch.get("message").?.object.get("content").?.string);
        try std.testing.expectEqualStrings("stop", ch.get("finish_reason").?.string);

        const r3 = try srv.post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"temperature\":0,\"max_tokens\":3}", &.{});
        defer r3.deinit(gpa);
        var p3 = try r3.json(gpa);
        defer p3.deinit();
        try std.testing.expectEqual(@as(i64, 3), p3.value.object.get("usage").?.object.get("completion_tokens").?.integer);
        try std.testing.expectEqualStrings("length", p3.value.object.get("choices").?.array.items[0].object.get("finish_reason").?.string);
    }

    // Tokenizing and detokenizing round trips.
    {
        const t = try srv.post("/tokenize", "{\"content\":\"hello world\"}", &.{});
        defer t.deinit(gpa);
        var pt = try t.json(gpa);
        defer pt.deinit();
        var aw: std.Io.Writer.Allocating = .init(gpa);
        defer aw.deinit();
        try aw.writer.print("{{\"tokens\":{f}}}", .{std.json.fmt(pt.value.object.get("tokens").?, .{})});
        const d = try srv.post("/detokenize", aw.written(), &.{});
        defer d.deinit(gpa);
        var pd = try d.json(gpa);
        defer pd.deinit();
        try std.testing.expectEqualStrings("hello world", pd.value.object.get("content").?.string);
    }

    // Bad input is rejected cleanly and the server stays healthy.
    {
        try expectStatus(&srv, "/v1/chat/completions", "{not json", 400);
        try expectStatus(&srv, "/v1/chat/completions", "{\"nomessages\":1}", 400);
        try expectStatus(&srv, "/v1/chat/completions", "{\"messages\":\"x\"}", 400);
        try expectStatus(&srv, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"x\"}}]}]}", 400);
        const big = try gpa.alloc(u8, 5000);
        defer gpa.free(big);
        @memset(big, 'x');
        const body = try std.fmt.allocPrint(gpa, "{{\"messages\":[{{\"role\":\"user\",\"content\":\"{s}\"}}],\"temperature\":0,\"max_tokens\":20}}", .{big});
        defer gpa.free(body);
        const r = try srv.post("/v1/chat/completions", body, &.{});
        defer r.deinit(gpa);
        try std.testing.expectEqual(@as(u16, 400), r.status);
        try std.testing.expect(contains(r.body, "context_length_exceeded"));
        try expectStatus(&srv, "/nope", "{}", 404);
        try expectStatus(&srv, "/tokenize", "{\"content\":5}", 400);
        try expectStatus(&srv, "/detokenize", "{\"tokens\":[\"a\"]}", 400);
        const ok = try chat(&srv, "hello", 20);
        gpa.free(ok);
    }

    // A client that disconnects mid stream frees its slot.
    {
        const addr = try std.Io.net.IpAddress.parse("127.0.0.1", srv.port);
        const stream = try addr.connect(io, .{ .mode = .stream });
        var wbuf: [512]u8 = undefined;
        var w = stream.writer(io, &wbuf);
        const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"temperature\":0,\"max_tokens\":400,\"stream\":true}";
        try w.interface.print("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n{s}", .{ body.len, body });
        try w.interface.flush();
        var rbuf: [256]u8 = undefined;
        var rd = stream.reader(io, &rbuf);
        _ = rd.interface.takeByte() catch {};
        stream.close(io);
        var tries: usize = 0;
        while (tries < 50) : (tries += 1) {
            if ((try srv.metric("hk_slots_active")) == 0) break;
            std.Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
        }
        try std.testing.expectEqual(@as(f64, 0), try srv.metric("hk_slots_active"));
        const ok = try chat(&srv, "hello", 20);
        gpa.free(ok);
    }

    // Long, different texts tokenized from many threads at once. The tokenizer keeps scratch space
    // internally, so a missing lock shows up as corrupted ids here.
    {
        var bodies: [24][]u8 = undefined;
        var expected: [24][]u8 = undefined;
        var made: usize = 0;
        defer for (0..made) |i| {
            gpa.free(bodies[i]);
            gpa.free(expected[i]);
        };
        for (0..24) |i| {
            var text: std.ArrayList(u8) = .empty;
            defer text.deinit(gpa);
            for (0..120) |_| try text.print(gpa, "sample {d} h\u{e9}llo w\u{f6}rld, some text 123 ", .{i});
            bodies[i] = try std.fmt.allocPrint(gpa, "{{\"content\":{f}}}", .{std.json.fmt(text.items, .{})});
            const r = try srv.post("/tokenize", bodies[i], &.{});
            expected[i] = r.body;
            made += 1;
        }
        for (0..4) |_| {
            var workers: [24]TokenizeWorker = undefined;
            var threads: [24]std.Thread = undefined;
            for (0..24) |i| {
                workers[i] = .{ .srv = &srv, .body = bodies[i] };
                threads[i] = try std.Thread.spawn(.{}, TokenizeWorker.run, .{&workers[i]});
            }
            for (threads) |t| t.join();
            for (0..24) |i| {
                defer if (workers[i].out) |o| gpa.free(o);
                try std.testing.expect(workers[i].err == null);
                try std.testing.expectEqualStrings(expected[i], workers[i].out.?);
            }
        }
    }
}

test "an oversized body is refused" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const model = try makeModel(&tmp);
    defer gpa.free(model);
    var srv = try http.Server.start(gpa, io, model, &.{ "--max-body-mb", "1" });
    defer srv.stop();
    const big = try gpa.alloc(u8, 2 << 20);
    defer gpa.free(big);
    @memset(big, 'a');
    const body = try std.fmt.allocPrint(gpa, "{{\"messages\":[{{\"role\":\"user\",\"content\":\"{s}\"}}]}}", .{big});
    defer gpa.free(body);
    const r = try srv.post("/v1/chat/completions", body, &.{});
    defer r.deinit(gpa);
    try std.testing.expectEqual(@as(u16, 413), r.status);
    const ok = try chat(&srv, "hello", 20);
    gpa.free(ok);
}

test "the api key is required except on /health" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const model = try makeModel(&tmp);
    defer gpa.free(model);
    var srv = try http.Server.start(gpa, io, model, &.{ "--api-key", "s3cret" });
    defer srv.stop();
    const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2}";
    {
        const r = try srv.post("/v1/chat/completions", body, &.{});
        defer r.deinit(gpa);
        try std.testing.expectEqual(@as(u16, 401), r.status);
    }
    {
        const r = try srv.post("/v1/chat/completions", body, &.{.{ .name = "Authorization", .value = "Bearer wrong" }});
        defer r.deinit(gpa);
        try std.testing.expectEqual(@as(u16, 401), r.status);
    }
    {
        const r = try srv.post("/v1/chat/completions", body, &.{.{ .name = "Authorization", .value = "Bearer s3cret" }});
        defer r.deinit(gpa);
        try std.testing.expectEqual(@as(u16, 200), r.status);
    }
    {
        const r = try srv.get("/health");
        defer r.deinit(gpa);
        try std.testing.expectEqual(@as(u16, 200), r.status);
    }
}
