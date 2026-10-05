//! One conversation: a KV cache plus the token ids it currently holds.
//!
//! Asking the session to evaluate a prompt first compares it with what the cache already
//! contains and only runs the model on the part that differs. A chat turn therefore costs the
//! new user message and the reply, not the whole history again.

const std = @import("std");
const model_mod = @import("model.zig");

pub const Session = struct {
    allocator: std.mem.Allocator,
    model: *model_mod.Model,
    kv: model_mod.KvCache,
    /// Ids whose keys and values are in `kv`, in order.
    cached: std.ArrayList(u32) = .empty,
    /// Index, within the last forward pass, of the token whose logits `logits` returns.
    last_index: usize = 0,

    pub fn init(allocator: std.mem.Allocator, model: *model_mod.Model, n_ctx: usize) !Session {
        return .{
            .allocator = allocator,
            .model = model,
            .kv = try model_mod.KvCache.init(allocator, model.cfg, @min(n_ctx, model.n_ctx)),
        };
    }

    pub fn deinit(self: *Session) void {
        self.model.forget(&self.kv);
        self.kv.deinit();
        self.cached.deinit(self.allocator);
    }

    pub fn reset(self: *Session) void {
        self.kv.truncate(0);
        self.cached.clearRetainingCapacity();
    }

    pub fn length(self: *const Session) usize {
        return self.cached.items.len;
    }

    /// Length of the common prefix of `ids` and the cached tokens.
    pub fn commonPrefix(self: *const Session, ids: []const u32) usize {
        const n = @min(ids.len, self.cached.items.len);
        var i: usize = 0;
        while (i < n and ids[i] == self.cached.items[i]) i += 1;
        return i;
    }

    /// Makes the cache hold exactly `ids`, running the model only on tokens after the shared
    /// prefix. The model's last hidden state is left ready for `logits`. Returns how many tokens
    /// were reused from the cache.
    pub fn evaluate(self: *Session, ids: []const u32) !usize {
        if (ids.len == 0) return error.EmptyPrompt;
        if (ids.len > self.kv.n_ctx) return error.ContextFull;
        var keep = self.commonPrefix(ids);
        // The last token must be run again to have logits for it.
        if (keep == ids.len) keep -= 1;
        self.cached.shrinkRetainingCapacity(keep);
        self.kv.truncate(keep);

        const m = self.model;
        var pos = keep;
        var last_n: usize = 1;
        while (pos < ids.len) {
            const n = @min(m.n_batch, ids.len - pos);
            last_n = n;
            var items: [256]model_mod.Item = undefined;
            std.debug.assert(n <= items.len);
            for (0..n) |t| items[t] = .{ .kv = &self.kv, .token = ids[pos + t], .pos = @intCast(pos + t) };
            try m.forwardItems(items[0..n]);
            pos += n;
        }
        try self.cached.appendSlice(self.allocator, ids[keep..]);
        self.last_index = last_n - 1;
        return keep;
    }

    /// Appends one generated token to the cache and runs the model on it.
    pub fn step(self: *Session, token: u32) !void {
        const pos = self.cached.items.len;
        if (pos >= self.kv.n_ctx) return error.ContextFull;
        try self.model.forwardItems(&.{.{ .kv = &self.kv, .token = token, .pos = @intCast(pos) }});
        try self.cached.append(self.allocator, token);
        self.last_index = 0;
    }

    /// Next token logits for the last evaluated token. Valid until the next model call.
    pub fn logits(self: *Session) []const f32 {
        return self.model.logitsFor(self.last_index);
    }
};
