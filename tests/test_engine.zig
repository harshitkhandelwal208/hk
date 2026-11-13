//! Engine behaviour on a tiny random model built by the test itself, so these run anywhere.

const std = @import("std");
const hk = @import("hk");

const dim: usize = 32;
const n_heads: usize = 4;
const n_kv: usize = 2;
const head_dim: usize = dim / n_heads;
const ffn: usize = 64;
const vocab: usize = 40;
const n_layers: usize = 2;

fn addMat(w: *hk.HKWriter, a: std.mem.Allocator, bufs: *std.ArrayList([]f32), prng: *std.Random.DefaultPrng, name: []const u8, rows: usize, cols: usize) !void {
    const buf = try a.alloc(f32, rows * cols);
    try bufs.append(a, buf);
    for (buf) |*v| v.* = (prng.random().float(f32) - 0.5) * 0.3;
    try w.addTensor(.{ .name = name, .storage_type = .f32, .ndim = 2, .shape = .{ rows, cols, 0, 0, 0, 0, 0, 0 }, .data = std.mem.sliceAsBytes(buf) });
}

fn addVec(w: *hk.HKWriter, a: std.mem.Allocator, bufs: *std.ArrayList([]f32), prng: *std.Random.DefaultPrng, name: []const u8, n: usize) !void {
    const buf = try a.alloc(f32, n);
    try bufs.append(a, buf);
    for (buf) |*v| v.* = 0.5 + prng.random().float(f32);
    try w.addTensor(.{ .name = name, .storage_type = .f32, .ndim = 1, .shape = .{ n, 0, 0, 0, 0, 0, 0, 0 }, .data = std.mem.sliceAsBytes(buf) });
}

fn writeTinyArch(a: std.mem.Allocator, path: []const u8, arch_name: []const u8) !void {
    var w = hk.HKWriter.init(a);
    defer w.deinit();
    var bufs: std.ArrayList([]f32) = .empty;
    defer {
        for (bufs.items) |b| a.free(b);
        bufs.deinit(a);
    }
    var prng = std.Random.DefaultPrng.init(99);

    var kbuf: [64]u8 = undefined;
    const mk = struct {
        fn k(buf: []u8, prefix: []const u8, suffix: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "{s}.{s}", .{ prefix, suffix }) catch unreachable;
        }
    }.k;

    try w.addMetadataString("general.architecture", arch_name);
    try w.addMetadataInt(mk(&kbuf, arch_name, "block_count"), n_layers);
    try w.addMetadataInt(mk(&kbuf, arch_name, "context_length"), 64);
    try w.addMetadataInt(mk(&kbuf, arch_name, "embedding_length"), dim);
    try w.addMetadataInt(mk(&kbuf, arch_name, "feed_forward_length"), ffn);
    try w.addMetadataInt(mk(&kbuf, arch_name, "attention.head_count"), n_heads);
    try w.addMetadataInt(mk(&kbuf, arch_name, "attention.head_count_kv"), n_kv);
    try w.addMetadataFloat(mk(&kbuf, arch_name, "attention.layer_norm_rms_epsilon"), 1e-5);
    try w.addMetadataFloat(mk(&kbuf, arch_name, "rope.freq_base"), 10000.0);
    try w.addMetadataInt(mk(&kbuf, arch_name, "rope.dimension_count"), head_dim);

    try addMat(&w, a, &bufs, &prng, "token_embd.weight", vocab, dim);
    try addVec(&w, a, &bufs, &prng, "output_norm.weight", dim);
    try addMat(&w, a, &bufs, &prng, "output.weight", vocab, dim);

    const is_qwen2 = std.mem.eql(u8, arch_name, "qwen2");
    const is_gemma = std.mem.eql(u8, arch_name, "gemma");

    var name: [64]u8 = undefined;
    for (0..n_layers) |l| {
        const n = struct {
            fn f(buf: []u8, layer: usize, suffix: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ layer, suffix }) catch unreachable;
            }
        }.f;
        try addVec(&w, a, &bufs, &prng, try a.dupe(u8, n(&name, l, "attn_norm.weight")), dim);
        try addMat(&w, a, &bufs, &prng, try a.dupe(u8, n(&name, l, "attn_q.weight")), n_heads * head_dim, dim);
        try addMat(&w, a, &bufs, &prng, try a.dupe(u8, n(&name, l, "attn_k.weight")), n_kv * head_dim, dim);
        try addMat(&w, a, &bufs, &prng, try a.dupe(u8, n(&name, l, "attn_v.weight")), n_kv * head_dim, dim);
        try addMat(&w, a, &bufs, &prng, try a.dupe(u8, n(&name, l, "attn_output.weight")), dim, n_heads * head_dim);
        try addVec(&w, a, &bufs, &prng, try a.dupe(u8, n(&name, l, "ffn_norm.weight")), dim);
        try addMat(&w, a, &bufs, &prng, try a.dupe(u8, n(&name, l, "ffn_gate.weight")), ffn, dim);
        try addMat(&w, a, &bufs, &prng, try a.dupe(u8, n(&name, l, "ffn_up.weight")), ffn, dim);
        try addMat(&w, a, &bufs, &prng, try a.dupe(u8, n(&name, l, "ffn_down.weight")), dim, ffn);

        if (is_qwen2) {
            try addVec(&w, a, &bufs, &prng, try a.dupe(u8, n(&name, l, "attn_q.bias")), n_heads * head_dim);
            try addVec(&w, a, &bufs, &prng, try a.dupe(u8, n(&name, l, "attn_k.bias")), n_kv * head_dim);
            try addVec(&w, a, &bufs, &prng, try a.dupe(u8, n(&name, l, "attn_v.bias")), n_kv * head_dim);
        }
        if (is_gemma) {
            try addVec(&w, a, &bufs, &prng, try a.dupe(u8, n(&name, l, "attn_q_norm.weight")), head_dim);
            try addVec(&w, a, &bufs, &prng, try a.dupe(u8, n(&name, l, "attn_k_norm.weight")), head_dim);
        }
    }
    try w.writeToFile(path);
}

fn writeTiny(a: std.mem.Allocator, path: []const u8) !void {
    try writeTinyArch(a, path, "llama");
}

const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    reader: hk.HKReader,
    model: hk.engine.Model,
    path: []const u8,

    fn open() !*Fixture {
        const a = std.testing.allocator;
        const f = try a.create(Fixture);
        f.arena = std.heap.ArenaAllocator.init(a);
        errdefer {
            f.arena.deinit();
            a.destroy(f);
        }
        // Names are duplicated into the arena by writeTiny, so keep it alive for the writer.
        f.path = "test_engine_tiny.hk";
        try writeTiny(f.arena.allocator(), f.path);
        f.reader = try hk.HKReader.open(f.path, a);
        errdefer f.reader.deinit();
        var diag = hk.engine.Diag{};
        f.model = hk.engine.Model.init(a, &f.reader, .{ .n_threads = 3, .n_batch = 16 }, &diag) catch |e| {
            std.debug.print("load failed: {s}\n", .{diag.message()});
            return e;
        };
        return f;
    }

    fn close(self: *Fixture) void {
        const a = std.testing.allocator;
        self.model.deinit();
        self.reader.deinit();
        std.Io.Dir.cwd().deleteFile(std.Options.debug_io, self.path) catch {};
        self.arena.deinit();
        a.destroy(self);
    }
};

fn maxDiff(x: []const f32, y: []const f32) f32 {
    var m: f32 = 0;
    for (x, y) |p, q| m = @max(m, @abs(p - q));
    return m;
}

test "batched prefill equals token by token" {
    const f = try Fixture.open();
    defer f.close();
    const toks = [_]u32{ 3, 17, 5, 5, 29, 0, 11, 8 };

    try f.model.forward(&toks, 0);
    const batched = try std.testing.allocator.dupe(f32, f.model.logitsFor(toks.len - 1));
    defer std.testing.allocator.free(batched);

    // Same tokens one at a time into a fresh cache.
    var kv = try hk.engine.model.KvCache.init(std.testing.allocator, f.model.cfg, 64);
    defer kv.deinit();
    for (toks, 0..) |t, i| {
        try f.model.forwardItems(&.{.{ .kv = &kv, .token = t, .pos = @intCast(i) }});
    }
    const single = f.model.logitsFor(0);
    try std.testing.expect(maxDiff(batched, single) < 1e-4);
}

test "interleaving two sequences in one pass changes nothing" {
    const f = try Fixture.open();
    defer f.close();
    const a = std.testing.allocator;
    const seq_a = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const seq_b = [_]u32{ 9, 8, 7, 6, 5, 4 };

    var kv_a = try hk.engine.model.KvCache.init(a, f.model.cfg, 64);
    defer kv_a.deinit();
    var kv_b = try hk.engine.model.KvCache.init(a, f.model.cfg, 64);
    defer kv_b.deinit();

    // Reference: each sequence alone.
    var ref_a: [vocab]f32 = undefined;
    var ref_b: [vocab]f32 = undefined;
    var kv_ra = try hk.engine.model.KvCache.init(a, f.model.cfg, 64);
    defer kv_ra.deinit();
    var kv_rb = try hk.engine.model.KvCache.init(a, f.model.cfg, 64);
    defer kv_rb.deinit();
    for (seq_a, 0..) |t, i| try f.model.forwardItems(&.{.{ .kv = &kv_ra, .token = t, .pos = @intCast(i) }});
    @memcpy(&ref_a, f.model.logitsFor(0));
    for (seq_b, 0..) |t, i| try f.model.forwardItems(&.{.{ .kv = &kv_rb, .token = t, .pos = @intCast(i) }});
    @memcpy(&ref_b, f.model.logitsFor(0));

    // Interleaved: one token of each sequence per pass.
    var got_a: [vocab]f32 = undefined;
    var got_b: [vocab]f32 = undefined;
    for (seq_a, seq_b, 0..) |ta, tb, i| {
        try f.model.forwardItems(&.{
            .{ .kv = &kv_a, .token = ta, .pos = @intCast(i) },
            .{ .kv = &kv_b, .token = tb, .pos = @intCast(i) },
        });
        if (i == seq_a.len - 1) {
            @memcpy(&got_a, f.model.logitsFor(0));
            @memcpy(&got_b, f.model.logitsFor(1));
        }
    }
    try std.testing.expect(maxDiff(&ref_a, &got_a) < 1e-4);
    try std.testing.expect(maxDiff(&ref_b, &got_b) < 1e-4);
    // And the two sequences really are different, so the check above is not vacuous.
    try std.testing.expect(maxDiff(&got_a, &got_b) > 1e-3);
}

test "the same input always gives the same output" {
    const f = try Fixture.open();
    defer f.close();
    const toks = [_]u32{ 4, 5, 6, 7 };
    try f.model.forward(&toks, 0);
    const first = try std.testing.allocator.dupe(f32, f.model.logitsFor(3));
    defer std.testing.allocator.free(first);
    f.model.kv.truncate(0);
    try f.model.forward(&toks, 0);
    try std.testing.expectEqual(@as(f32, 0), maxDiff(first, f.model.logitsFor(3)));
}

test "failures are errors, not corruption" {
    const f = try Fixture.open();
    defer f.close();
    // Token outside the vocabulary.
    try std.testing.expectError(error.TokenOutOfRange, f.model.forward(&.{@as(u32, vocab)}, 0));
    // Past the context window (the file says 64 positions).
    var long: [8]u32 = @splat(1);
    try std.testing.expectError(error.ContextFull, f.model.forward(&long, 60));
    // A batch larger than the scratch space.
    var big: [17]u32 = @splat(1);
    try std.testing.expectError(error.BatchTooLarge, f.model.forward(&big, 0));
    // The model still works afterwards.
    try f.model.forward(&.{ 1, 2, 3 }, 0);
    for (f.model.logitsFor(2)) |l| try std.testing.expect(std.math.isFinite(l));
}

test "kv cache memory follows the context, not the maximum" {
    const f = try Fixture.open();
    defer f.close();
    try std.testing.expectEqual(@as(usize, 0), f.model.kv.bytes());
    try f.model.forward(&.{ 1, 2, 3 }, 0);
    const small = f.model.kv.bytes();
    try std.testing.expect(small > 0);
    f.model.kv.truncate(0);
    try std.testing.expectEqual(@as(usize, 0), f.model.kv.bytes());
}

test "a model that cannot be run is refused with a reason" {
    const a = std.testing.allocator;
    var w = hk.HKWriter.init(a);
    defer w.deinit();
    try w.addMetadataString("general.architecture", "mamba");
    const v = [_]f32{ 1, 2 };
    try w.addTensor(.{ .name = "token_embd.weight", .storage_type = .f32, .ndim = 2, .shape = .{ 1, 2, 0, 0, 0, 0, 0, 0 }, .data = std.mem.sliceAsBytes(&v) });
    const path = "test_engine_unsupported.hk";
    try w.writeToFile(path);
    defer std.Io.Dir.cwd().deleteFile(std.Options.debug_io, path) catch {};
    var r = try hk.HKReader.open(path, a);
    defer r.deinit();
    var diag = hk.engine.Diag{};
    try std.testing.expectError(error.UnsupportedArchitecture, hk.engine.Model.init(a, &r, .{}, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "mamba") != null);
}

test "logits for several items match one by one" {
    const f = try Fixture.open();
    defer f.close();
    const toks = [_]u32{ 3, 17, 5, 29, 0 };
    try f.model.forward(&toks, 0);
    var want: [3][vocab]f32 = undefined;
    const picks = [_]usize{ 4, 0, 2 };
    for (picks, 0..) |p, k| @memcpy(&want[k], f.model.logitsFor(p));
    const many = try f.model.logitsMany(&picks);
    try std.testing.expectEqual(@as(usize, 3 * vocab), many.len);
    for (0..3) |k| try std.testing.expect(maxDiff(&want[k], many[k * vocab ..][0..vocab]) < 1e-5);
}

test "Gemma architecture executes with gelu_tanh and qk_norm" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const path = "test_engine_gemma.hk";
    try writeTinyArch(arena.allocator(), path, "gemma");
    defer std.Io.Dir.cwd().deleteFile(std.Options.debug_io, path) catch {};

    var r = try hk.HKReader.open(path, a);
    defer r.deinit();
    var diag = hk.engine.Diag{};
    var model = try hk.engine.Model.init(a, &r, .{ .n_threads = 2, .n_batch = 8 }, &diag);
    defer model.deinit();

    try std.testing.expectEqual(hk.engine.config.Arch.gemma, model.cfg.arch);
    try std.testing.expectEqual(hk.engine.config.FfnAct.gelu_tanh, model.cfg.ffn_act);
    try std.testing.expect(model.cfg.qk_norm);

    const toks = [_]u32{ 1, 5, 12, 24 };
    try model.forward(&toks, 0);
    const logits = model.logitsFor(toks.len - 1);
    try std.testing.expectEqual(vocab, logits.len);
    for (logits) |v| try std.testing.expect(std.math.isFinite(v));
}

test "Qwen2 architecture executes with qkv_bias" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const path = "test_engine_qwen2.hk";
    try writeTinyArch(arena.allocator(), path, "qwen2");
    defer std.Io.Dir.cwd().deleteFile(std.Options.debug_io, path) catch {};

    var r = try hk.HKReader.open(path, a);
    defer r.deinit();
    var diag = hk.engine.Diag{};
    var model = try hk.engine.Model.init(a, &r, .{ .n_threads = 2, .n_batch = 8 }, &diag);
    defer model.deinit();

    try std.testing.expectEqual(hk.engine.config.Arch.qwen2, model.cfg.arch);
    try std.testing.expect(model.cfg.qkv_bias);

    const toks = [_]u32{ 2, 7, 19, 31 };
    try model.forward(&toks, 0);
    const logits = model.logitsFor(toks.len - 1);
    try std.testing.expectEqual(vocab, logits.len);
    for (logits) |v| try std.testing.expect(std.math.isFinite(v));
}
