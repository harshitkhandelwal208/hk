//! Token sampling.
//!
//! The chain follows llama.cpp's default order: penalties, top-k, top-p, min-p, temperature,
//! then a draw. Top-k uses a bounded heap, so a step over a 150,000 token vocabulary costs one
//! pass plus a sort of at most a few thousand candidates rather than a full sort. All scratch
//! space is kept between calls, so sampling does not allocate once warm.

const std = @import("std");
const ops = @import("engine/ops.zig");

pub const Bias = struct { token: u32, bias: f32 };

pub const Params = struct {
    /// 0 selects the most likely token (after penalties and bias) and ignores the rest.
    temperature: f32 = 0.8,
    /// 0 disables the limit, which is capped at `max_candidates` internally.
    top_k: u32 = 40,
    top_p: f32 = 0.95,
    min_p: f32 = 0.05,
    repeat_penalty: f32 = 1.0,
    /// How many of the most recent tokens the penalties look at.
    repeat_last_n: u32 = 64,
    frequency_penalty: f32 = 0.0,
    presence_penalty: f32 = 0.0,
    /// 0 seeds from the clock.
    seed: u64 = 0,
    logit_bias: []const Bias = &.{},
};

/// Upper bound on candidates considered when top-k is off or very large.
const max_candidates = 4096;

const Cand = struct { logit: f32, id: u32 };

pub const Sampler = struct {
    allocator: std.mem.Allocator,
    prng: std.Random.DefaultPrng,
    heap: std.ArrayList(Cand) = .empty,
    probs: std.ArrayList(f32) = .empty,
    counts: []u16 = &.{},
    touched: std.ArrayList(u32) = .empty,
    /// Natural log probability of the last token returned, under the penalised distribution at
    /// temperature 1. Only meaningful after `sample` with `want_logprob`.
    last_logprob: f32 = 0,

    pub fn init(allocator: std.mem.Allocator, seed: u64) Sampler {
        var s = seed;
        if (s == 0) {
            s = @intCast(std.Io.Timestamp.now(std.Options.debug_io, .real).nanoseconds & 0x7FFF_FFFF_FFFF_FFFF);
            if (s == 0) s = 0x9E3779B97F4A7C15;
        }
        return .{ .allocator = allocator, .prng = std.Random.DefaultPrng.init(s) };
    }

    pub fn deinit(self: *Sampler) void {
        self.heap.deinit(self.allocator);
        self.probs.deinit(self.allocator);
        self.touched.deinit(self.allocator);
        if (self.counts.len != 0) self.allocator.free(self.counts);
    }

    fn lessCand(a: Cand, b: Cand) bool {
        // Min-heap on logit; ties broken by id so results do not depend on scan order quirks.
        if (a.logit != b.logit) return a.logit < b.logit;
        return a.id > b.id;
    }

    fn siftDown(items: []Cand, start: usize) void {
        var i = start;
        while (true) {
            const l = 2 * i + 1;
            const r = l + 1;
            var m = i;
            if (l < items.len and lessCand(items[l], items[m])) m = l;
            if (r < items.len and lessCand(items[r], items[m])) m = r;
            if (m == i) return;
            std.mem.swap(Cand, &items[i], &items[m]);
            i = m;
        }
    }

    fn applyPenalties(self: *Sampler, logits: []f32, history: []const u32, p: Params) !void {
        const window = if (p.repeat_last_n == 0) history.len else @min(history.len, p.repeat_last_n);
        const recent = history[history.len - window ..];
        const active = p.repeat_penalty != 1.0 or p.frequency_penalty != 0 or p.presence_penalty != 0;
        if (active and recent.len > 0) {
            if (self.counts.len < logits.len) {
                if (self.counts.len != 0) self.allocator.free(self.counts);
                self.counts = &.{};
                self.counts = try self.allocator.alloc(u16, logits.len);
                @memset(self.counts, 0);
            }
            self.touched.clearRetainingCapacity();
            for (recent) |t| {
                if (t >= logits.len) continue;
                if (self.counts[t] == 0) try self.touched.append(self.allocator, t);
                self.counts[t] +|= 1;
            }
            for (self.touched.items) |t| {
                const c: f32 = @floatFromInt(self.counts[t]);
                var l = logits[t];
                if (p.repeat_penalty != 1.0) l = if (l > 0) l / p.repeat_penalty else l * p.repeat_penalty;
                l -= c * p.frequency_penalty + p.presence_penalty;
                logits[t] = l;
                self.counts[t] = 0;
            }
        }
        for (p.logit_bias) |b| if (b.token < logits.len) {
            logits[b.token] += b.bias;
        };
    }

    /// Draws the next token. `logits` is modified in place. `history` is the tokens so far, used
    /// by the repetition penalties.
    pub fn sample(self: *Sampler, logits: []f32, history: []const u32, p: Params, want_logprob: bool) !u32 {
        if (logits.len == 0) return 0;
        try self.applyPenalties(logits, history, p);

        var lse: f32 = 0;
        if (want_logprob) {
            var mx: f32 = -std.math.inf(f32);
            for (logits) |l| mx = @max(mx, l);
            var sum: f64 = 0;
            for (logits) |l| sum += @exp(@as(f64, l - mx));
            lse = mx + @as(f32, @floatCast(@log(sum)));
        }

        if (p.temperature <= 0) {
            var best: u32 = 0;
            for (logits, 0..) |l, i| if (l > logits[best]) {
                best = @intCast(i);
            };
            self.last_logprob = logits[best] - lse;
            return best;
        }

        // Top-k by bounded heap.
        const k: usize = blk: {
            const want: usize = if (p.top_k == 0) max_candidates else @min(p.top_k, max_candidates);
            break :blk @min(want, logits.len);
        };
        self.heap.clearRetainingCapacity();
        try self.heap.ensureTotalCapacity(self.allocator, k);
        for (logits, 0..) |l, i| {
            if (!std.math.isFinite(l)) continue;
            const c = Cand{ .logit = l, .id = @intCast(i) };
            if (self.heap.items.len < k) {
                self.heap.appendAssumeCapacity(c);
                if (self.heap.items.len == k) {
                    var j = k / 2;
                    while (j > 0) {
                        j -= 1;
                        siftDown(self.heap.items, j);
                    }
                }
            } else if (lessCand(self.heap.items[0], c)) {
                self.heap.items[0] = c;
                siftDown(self.heap.items, 0);
            }
        }
        const cands = self.heap.items;
        if (cands.len == 0) return 0;
        std.mem.sort(Cand, cands, {}, struct {
            fn gt(_: void, a: Cand, b: Cand) bool {
                if (a.logit != b.logit) return a.logit > b.logit;
                return a.id < b.id;
            }
        }.gt);

        // Probabilities at temperature 1 over the candidates, for top-p and min-p.
        try self.probs.resize(self.allocator, cands.len);
        const probs = self.probs.items;
        const max_logit = cands[0].logit;
        var total: f32 = 0;
        for (cands, 0..) |c, i| {
            probs[i] = @exp(c.logit - max_logit);
            total += probs[i];
        }
        for (probs) |*q| q.* /= total;

        var keep = cands.len;
        if (p.top_p > 0 and p.top_p < 1) {
            var cum: f32 = 0;
            for (probs, 0..) |q, i| {
                cum += q;
                if (cum >= p.top_p) {
                    keep = i + 1;
                    break;
                }
            }
        }
        if (p.min_p > 0) {
            const floor = probs[0] * p.min_p;
            var i: usize = 0;
            while (i < keep and probs[i] >= floor) i += 1;
            keep = @max(1, i);
        }

        // Temperature on what is left, then draw.
        var norm: f32 = 0;
        for (cands[0..keep], 0..) |c, i| {
            probs[i] = @exp((c.logit - max_logit) / p.temperature);
            norm += probs[i];
        }
        var r = self.prng.random().float(f32) * norm;
        var chosen = keep - 1;
        for (probs[0..keep], 0..) |q, i| {
            r -= q;
            if (r <= 0) {
                chosen = i;
                break;
            }
        }
        const id = cands[chosen].id;
        self.last_logprob = cands[chosen].logit - lse;
        return id;
    }
};

test "temperature zero is argmax and honours penalties" {
    const a = std.testing.allocator;
    var s = Sampler.init(a, 1);
    defer s.deinit();
    var logits = [_]f32{ 0.1, 2.0, 1.9, -1 };
    try std.testing.expectEqual(@as(u32, 1), try s.sample(&logits, &.{}, .{ .temperature = 0 }, false));
    // Token 1 was just produced; a strong repeat penalty hands the win to token 2.
    logits = .{ 0.1, 2.0, 1.9, -1 };
    try std.testing.expectEqual(@as(u32, 2), try s.sample(&logits, &.{1}, .{ .temperature = 0, .repeat_penalty = 2.0 }, false));
    // Bias can force a token.
    logits = .{ 0.1, 2.0, 1.9, -1 };
    try std.testing.expectEqual(@as(u32, 3), try s.sample(&logits, &.{}, .{ .temperature = 0, .logit_bias = &.{.{ .token = 3, .bias = 10 }} }, false));
}

test "top-k 1 is deterministic and sampling follows the distribution" {
    const a = std.testing.allocator;
    var s = Sampler.init(a, 42);
    defer s.deinit();
    var logits = [_]f32{ 1, 3, 2, 0 };
    for (0..20) |_| {
        logits = .{ 1, 3, 2, 0 };
        try std.testing.expectEqual(@as(u32, 1), try s.sample(&logits, &.{}, .{ .temperature = 1, .top_k = 1 }, false));
    }
    // Full distribution: frequencies approach softmax.
    var counts = [_]u32{ 0, 0, 0, 0 };
    const n = 20000;
    for (0..n) |_| {
        logits = .{ 1, 3, 2, 0 };
        const t = try s.sample(&logits, &.{}, .{ .temperature = 1, .top_k = 0, .top_p = 1, .min_p = 0 }, false);
        counts[t] += 1;
    }
    var z: f64 = 0;
    const raw = [_]f64{ 1, 3, 2, 0 };
    for (raw) |x| z += @exp(x);
    for (counts, raw) |c, x| {
        const want = @exp(x) / z;
        const got = @as(f64, @floatFromInt(c)) / n;
        try std.testing.expect(@abs(got - want) < 0.02);
    }
}

test "min-p and top-p cut the tail" {
    const a = std.testing.allocator;
    var s = Sampler.init(a, 7);
    defer s.deinit();
    for (0..500) |_| {
        // Token 0 dominates; with min-p 0.5 only it can survive.
        var logits = [_]f32{ 5, 1, 1, 1 };
        try std.testing.expectEqual(@as(u32, 0), try s.sample(&logits, &.{}, .{ .temperature = 1, .top_k = 0, .top_p = 1, .min_p = 0.5 }, false));
        logits = .{ 5, 1, 1, 1 };
        try std.testing.expectEqual(@as(u32, 0), try s.sample(&logits, &.{}, .{ .temperature = 1, .top_k = 0, .top_p = 0.5, .min_p = 0 }, false));
    }
}

test "log probability matches log softmax" {
    const a = std.testing.allocator;
    var s = Sampler.init(a, 3);
    defer s.deinit();
    var logits = [_]f32{ 1, 2, 3 };
    const id = try s.sample(&logits, &.{}, .{ .temperature = 0 }, true);
    try std.testing.expectEqual(@as(u32, 2), id);
    const z = @exp(@as(f64, 1)) + @exp(@as(f64, 2)) + @exp(@as(f64, 3));
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(3 - @log(z))), s.last_logprob, 1e-5);
}

test "seeded sampling repeats and handles non finite logits" {
    const a = std.testing.allocator;
    var s1 = Sampler.init(a, 99);
    defer s1.deinit();
    var s2 = Sampler.init(a, 99);
    defer s2.deinit();
    for (0..50) |_| {
        var l1 = [_]f32{ 1, 2, 3, 2, 1, std.math.nan(f32), -std.math.inf(f32) };
        var l2 = l1;
        const x = try s1.sample(&l1, &.{}, .{ .temperature = 0.9 }, false);
        const y = try s2.sample(&l2, &.{}, .{ .temperature = 0.9 }, false);
        try std.testing.expectEqual(x, y);
        try std.testing.expect(x < 5);
    }
}
