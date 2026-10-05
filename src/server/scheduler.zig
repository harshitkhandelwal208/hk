//! Continuous batching scheduler.
//!
//! One thread owns the model and runs a loop. Every step it gathers the next token of every
//! generating conversation plus a chunk of any prompt still being read, runs them through the
//! model in a single pass (the weights are streamed once for all of them), samples a token for
//! each conversation that is ready, and hands the text to the HTTP thread waiting on it.
//!
//! A new request goes to the idle slot whose cache already holds the longest prefix of its
//! prompt, so a chat that repeats a system prompt or earlier turns skips re-reading them.
//!
//! HTTP threads never touch the model. They submit a `Request`, wait on its condition variable,
//! and read events out of it.

const std = @import("std");
const model_mod = @import("../engine/model.zig");
const session_mod = @import("../engine/session.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const sampler_mod = @import("../sampler.zig");

pub const FinishReason = enum { stop, length, cancelled, failed };

pub const Event = struct {
    /// Decoded text, owned by the request's allocator.
    text: []u8,
    /// Log probability of the token this text came from, when requested.
    logprob: ?f32 = null,
};

pub const Request = struct {
    allocator: std.mem.Allocator,
    io: std.Io,

    // ---- set by the submitter before `submit` ----
    prompt: []u32,
    max_tokens: u32,
    params: sampler_mod.Params,
    stop: []const []const u8 = &.{},
    want_logprob: bool = false,

    // ---- shared state, guarded by `mutex` ----
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    events: std.ArrayList(Event) = .empty,
    done: bool = false,
    finish: FinishReason = .stop,
    fail_message: [160]u8 = undefined,
    fail_len: usize = 0,

    // ---- results, valid once `done` ----
    prompt_tokens: u32 = 0,
    cached_tokens: u32 = 0,
    completion_tokens: u32 = 0,

    cancelled: std.atomic.Value(bool) = .init(false),

    pub fn deinit(self: *Request) void {
        for (self.events.items) |e| self.allocator.free(e.text);
        self.events.deinit(self.allocator);
        self.allocator.free(self.prompt);
    }

    pub fn cancel(self: *Request) void {
        self.cancelled.store(true, .release);
    }

    /// Blocks until there are events or the request has finished, then moves the pending events
    /// into `out`. Returns true once the request is done and everything has been delivered.
    pub fn wait(self: *Request, out: *std.ArrayList(Event), a: std.mem.Allocator) !bool {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        while (self.events.items.len == 0 and !self.done) self.cond.waitUncancelable(self.io, &self.mutex);
        try out.appendSlice(a, self.events.items);
        self.events.clearRetainingCapacity();
        return self.done;
    }

    pub fn failMessage(self: *const Request) []const u8 {
        return self.fail_message[0..self.fail_len];
    }

    fn push(self: *Request, text: []const u8, logprob: ?f32) !void {
        const copy = try self.allocator.dupe(u8, text);
        errdefer self.allocator.free(copy);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        try self.events.append(self.allocator, .{ .text = copy, .logprob = logprob });
        self.cond.signal(self.io);
    }

    fn finishWith(self: *Request, reason: FinishReason) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.finish = reason;
        self.done = true;
        self.cond.broadcast(self.io);
    }

    fn fail(self: *Request, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(&self.fail_message, fmt, args) catch self.fail_message[0..];
        self.fail_len = s.len;
        self.finishWith(.failed);
    }
};

pub const Stats = struct {
    requests: std.atomic.Value(u64) = .init(0),
    prompt_tokens: std.atomic.Value(u64) = .init(0),
    cached_tokens: std.atomic.Value(u64) = .init(0),
    generated_tokens: std.atomic.Value(u64) = .init(0),
    active_slots: std.atomic.Value(u32) = .init(0),
    queued: std.atomic.Value(u32) = .init(0),
};

const Slot = struct {
    sess: session_mod.Session,
    sampler: sampler_mod.Sampler,
    req: ?*Request = null,
    /// Next prompt index to evaluate while reading the prompt.
    prompt_pos: usize = 0,
    /// Sampled token waiting to be evaluated.
    pending: ?u32 = null,
    generated: u32 = 0,
    /// UTF-8 bytes of the reply not yet released to the client.
    hold: std.ArrayList(u8) = .empty,
    /// Text of the whole reply, for stop string matching.
    reply: std.ArrayList(u8) = .empty,
    emitted: usize = 0,

    fn reset(self: *Slot, a: std.mem.Allocator) void {
        _ = a;
        self.req = null;
        self.prompt_pos = 0;
        self.pending = null;
        self.generated = 0;
        self.hold.clearRetainingCapacity();
        self.reply.clearRetainingCapacity();
        self.emitted = 0;
    }
};

pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    model: *model_mod.Model,
    tok: *tokenizer_mod.Tokenizer,
    slots: []Slot,
    queue: std.ArrayList(*Request) = .empty,
    max_queue: usize = 64,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    stopping: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    stats: Stats = .{},

    pub fn init(allocator: std.mem.Allocator, io: std.Io, model: *model_mod.Model, tok: *tokenizer_mod.Tokenizer, n_slots: usize, ctx_per_slot: usize) !Scheduler {
        const slots = try allocator.alloc(Slot, n_slots);
        var made: usize = 0;
        errdefer {
            for (slots[0..made]) |*s| {
                s.sess.deinit();
                s.sampler.deinit();
                s.hold.deinit(allocator);
                s.reply.deinit(allocator);
            }
            allocator.free(slots);
        }
        for (slots) |*s| {
            s.* = .{ .sess = try session_mod.Session.init(allocator, model, ctx_per_slot), .sampler = sampler_mod.Sampler.init(allocator, 0) };
            made += 1;
        }
        return .{ .allocator = allocator, .io = io, .model = model, .tok = tok, .slots = slots };
    }

    pub fn deinit(self: *Scheduler) void {
        self.stop();
        for (self.slots) |*s| {
            s.sess.deinit();
            s.sampler.deinit();
            s.hold.deinit(self.allocator);
            s.reply.deinit(self.allocator);
        }
        self.allocator.free(self.slots);
        self.queue.deinit(self.allocator);
    }

    pub fn start(self: *Scheduler) !void {
        self.thread = try std.Thread.spawn(.{}, run, .{self});
    }

    pub fn stop(self: *Scheduler) void {
        if (self.thread) |t| {
            self.stopping.store(true, .release);
            self.mutex.lockUncancelable(self.io);
            self.cond.broadcast(self.io);
            self.mutex.unlock(self.io);
            t.join();
            self.thread = null;
        }
    }

    /// Queues a request. The scheduler takes responsibility for finishing it.
    pub fn submit(self: *Scheduler, req: *Request) error{ QueueFull, OutOfMemory, TooLong }!void {
        // A prompt that cannot fit a slot could never run; refuse it now with a clear reason.
        if (req.prompt.len + 1 >= self.slots[0].sess.kv.n_ctx) return error.TooLong;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.queue.items.len >= self.max_queue) return error.QueueFull;
        try self.queue.append(self.allocator, req);
        _ = self.stats.requests.fetchAdd(1, .monotonic);
        self.stats.queued.store(@intCast(self.queue.items.len), .monotonic);
        self.cond.signal(self.io);
    }

    fn anyActive(self: *const Scheduler) bool {
        for (self.slots) |s| if (s.req != null) return true;
        return false;
    }

    fn run(self: *Scheduler) void {
        const a = self.allocator;
        var items: std.ArrayList(model_mod.Item) = .empty;
        defer items.deinit(a);
        const Want = struct { slot: usize, index: usize };
        var wanted: std.ArrayList(Want) = .empty;
        defer wanted.deinit(a);
        var picks: std.ArrayList(usize) = .empty;
        defer picks.deinit(a);

        while (!self.stopping.load(.acquire)) {
            // Sleep until there is something to do.
            self.mutex.lockUncancelable(self.io);
            while (self.queue.items.len == 0 and !self.anyActive() and !self.stopping.load(.acquire)) {
                self.cond.waitUncancelable(self.io, &self.mutex);
            }
            self.admitLocked();
            self.mutex.unlock(self.io);
            if (self.stopping.load(.acquire)) break;

            // Drop requests whose client went away.
            for (self.slots) |*s| if (s.req) |r| if (r.cancelled.load(.acquire)) self.finishSlot(s, .cancelled);
            self.stats.active_slots.store(@intCast(self.countActive()), .monotonic);

            items.clearRetainingCapacity();
            wanted.clearRetainingCapacity();
            const budget = self.model.n_batch;

            // One decode token for every generating slot, then prompt chunks with what is left.
            for (self.slots, 0..) |*s, si| {
                if (s.req == null) continue;
                if (s.pending) |tok| {
                    if (items.items.len >= budget) break;
                    items.append(a, .{ .kv = &s.sess.kv, .token = tok, .pos = @intCast(s.sess.cached.items.len) }) catch {
                        self.failAll("out of memory");
                        continue;
                    };
                    wanted.append(a, .{ .slot = si, .index = items.items.len - 1 }) catch {};
                }
            }
            var prefilling: usize = 0;
            for (self.slots) |s| if (s.req != null and s.pending == null) {
                prefilling += 1;
            };
            if (prefilling > 0) {
                const share = @max(1, (budget -| items.items.len) / prefilling);
                for (self.slots, 0..) |*s, si| {
                    const r = s.req orelse continue;
                    if (s.pending != null) continue;
                    const remaining = r.prompt.len - s.prompt_pos;
                    const n = @min(remaining, @min(share, budget -| items.items.len));
                    if (n == 0) continue;
                    for (0..n) |k| {
                        items.append(a, .{ .kv = &s.sess.kv, .token = r.prompt[s.prompt_pos + k], .pos = @intCast(s.prompt_pos + k) }) catch {
                            self.failAll("out of memory");
                            break;
                        };
                    }
                    // The last prompt token produces the first reply token.
                    if (n == remaining) wanted.append(a, .{ .slot = si, .index = items.items.len - 1 }) catch {};
                    s.prompt_pos += n;
                }
            }
            if (items.items.len == 0) continue;

            self.model.forwardItems(items.items) catch |e| {
                self.failAll(@errorName(e));
                continue;
            };
            // Every evaluated token joins its slot's cache, in order.
            for (items.items) |it| {
                for (self.slots) |*s| {
                    if (&s.sess.kv == it.kv) {
                        s.sess.cached.append(a, it.token) catch {};
                        break;
                    }
                }
            }

            if (wanted.items.len == 0) continue;
            picks.clearRetainingCapacity();
            for (wanted.items) |w| picks.append(a, w.index) catch {};
            const rows = self.model.logitsMany(picks.items) catch |e| {
                self.failAll(@errorName(e));
                continue;
            };
            const vocab = self.model.cfg.vocab;
            for (wanted.items, 0..) |w, k| {
                const s = &self.slots[w.slot];
                if (s.req == null) continue;
                self.advance(s, @constCast(rows[k * vocab ..][0..vocab]));
            }
            self.stats.active_slots.store(@intCast(self.countActive()), .monotonic);
        }
        self.failAll("server stopped");
    }

    fn countActive(self: *const Scheduler) usize {
        var n: usize = 0;
        for (self.slots) |s| if (s.req != null) {
            n += 1;
        };
        return n;
    }

    /// Moves queued requests onto idle slots, preferring the slot with the longest shared prefix.
    fn admitLocked(self: *Scheduler) void {
        while (self.queue.items.len > 0) {
            const req = self.queue.items[0];
            var best: ?usize = null;
            var best_len: usize = 0;
            for (self.slots, 0..) |*s, i| {
                if (s.req != null) continue;
                const cp = s.sess.commonPrefix(req.prompt);
                if (best == null or cp > best_len) {
                    best = i;
                    best_len = cp;
                }
            }
            const si = best orelse return;
            _ = self.queue.orderedRemove(0);
            self.stats.queued.store(@intCast(self.queue.items.len), .monotonic);
            const s = &self.slots[si];
            s.reset(self.allocator);
            s.req = req;
            // The last prompt token must be evaluated to get logits, so it is never "cached".
            var keep = best_len;
            if (keep >= req.prompt.len) keep = req.prompt.len - 1;
            s.sess.cached.shrinkRetainingCapacity(keep);
            s.sess.kv.truncate(keep);
            s.prompt_pos = keep;
            s.sampler.deinit();
            s.sampler = sampler_mod.Sampler.init(self.allocator, req.params.seed);
            req.prompt_tokens = @intCast(req.prompt.len);
            req.cached_tokens = @intCast(keep);
            _ = self.stats.prompt_tokens.fetchAdd(req.prompt.len - keep, .monotonic);
            _ = self.stats.cached_tokens.fetchAdd(keep, .monotonic);
        }
    }

    fn finishSlot(self: *Scheduler, s: *Slot, reason: FinishReason) void {
        const r = s.req orelse return;
        self.flush(s, true);
        r.completion_tokens = s.generated;
        r.finishWith(reason);
        s.reset(self.allocator);
        self.stats.active_slots.store(@intCast(self.countActive()), .monotonic);
    }

    fn failAll(self: *Scheduler, msg: []const u8) void {
        for (self.slots) |*s| {
            if (s.req) |r| {
                r.fail("{s}", .{msg});
                s.reset(self.allocator);
            }
        }
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        for (self.queue.items) |r| r.fail("{s}", .{msg});
        self.queue.clearRetainingCapacity();
    }

    fn isStop(self: *const Scheduler, id: u32) bool {
        const tok = self.tok;
        if (tok.eos) |e| if (id == e) return true;
        if (tok.eog) |list| for (0..list.count) |i| if (@as(u32, @intCast(list.get(i))) == id) return true;
        const ty = tok.tokenType(id);
        if (ty != .control and ty != .user_defined) return false;
        const p = tok.piece(id);
        const stops = [_][]const u8{ "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "<|end|>", "<end_of_turn>", "</s>" };
        for (stops) |st| if (std.mem.eql(u8, p, st)) return true;
        return false;
    }

    /// Samples the next token for `s` from `logits` and updates the slot.
    fn advance(self: *Scheduler, s: *Slot, logits: []f32) void {
        const r = s.req.?;
        const history = s.sess.cached.items;
        const id = s.sampler.sample(logits, history, r.params, r.want_logprob) catch {
            r.fail("sampling failed", .{});
            s.reset(self.allocator);
            return;
        };
        if (self.isStop(id)) return self.finishSlot(s, .stop);

        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(self.allocator);
        self.tok.decodeToken(id, false, &bytes) catch {};
        s.hold.appendSlice(self.allocator, bytes.items) catch {};
        s.reply.appendSlice(self.allocator, bytes.items) catch {};
        s.generated += 1;
        _ = self.stats.generated_tokens.fetchAdd(1, .monotonic);

        // Stop strings: cut the reply at the first match and end.
        if (r.stop.len > 0) {
            for (r.stop) |st| {
                if (st.len == 0) continue;
                if (std.mem.indexOf(u8, s.reply.items[s.emitted..], st)) |off| {
                    const cut = s.emitted + off;
                    s.reply.shrinkRetainingCapacity(cut);
                    // Release what precedes the stop string.
                    self.flushTo(s, cut);
                    return self.finishSlot(s, .stop);
                }
            }
        }
        self.flush(s, false);

        if (s.generated >= r.max_tokens) return self.finishSlot(s, .length);
        if (s.sess.cached.items.len + 1 >= s.sess.kv.n_ctx) return self.finishSlot(s, .length);
        s.pending = id;
    }

    /// Releases text that can no longer become part of a stop string. `final` releases all of it.
    fn flush(self: *Scheduler, s: *Slot, final: bool) void {
        const r = s.req orelse return;
        var upto = s.reply.items.len;
        if (!final) {
            // Hold back enough bytes to complete the longest stop string.
            var longest: usize = 0;
            for (r.stop) |st| longest = @max(longest, st.len);
            if (longest > 1) upto -|= longest - 1;
            upto = @max(upto, s.emitted);
            // And never split a UTF-8 sequence.
            while (upto > s.emitted and upto < s.reply.items.len and (s.reply.items[upto] & 0xC0) == 0x80) upto -= 1;
            // A sequence that is not complete yet stays held.
            if (upto == s.reply.items.len) {
                var i = upto;
                var back: usize = 0;
                while (i > s.emitted and back < 4) : (back += 1) {
                    i -= 1;
                    const c = s.reply.items[i];
                    if (c & 0xC0 != 0x80) {
                        const need = std.unicode.utf8ByteSequenceLength(c) catch 1;
                        if (upto - i < need) upto = i;
                        break;
                    }
                }
            }
        }
        self.flushTo(s, upto);
    }

    fn flushTo(self: *Scheduler, s: *Slot, upto: usize) void {
        _ = self;
        const r = s.req orelse return;
        if (upto <= s.emitted) return;
        r.push(s.reply.items[s.emitted..upto], if (r.want_logprob) s.sampler.last_logprob else null) catch {};
        s.emitted = upto;
    }
};
