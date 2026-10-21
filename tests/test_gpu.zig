//! A device engine must compute what the CPU engine computes. Each check runs for Vulkan, CUDA and ROCm and is skipped
//! (per backend) when the machine has no usable device of that kind.

const std = @import("std");
const hk = @import("hk");
const tiny = @import("support/tiny_model.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;
const Choice = hk.backend.select.Choice;

fn cosine(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * y;
        na += @as(f64, x) * x;
        nb += @as(f64, y) * y;
    }
    return dot / (@sqrt(na) * @sqrt(nb));
}

fn load(path: []const u8, gpu: hk.engine.model.GpuMode, backend: Choice, slots: usize, n_ctx: usize, reader: *hk.HKReader) !hk.engine.Model {
    return loadLayers(path, gpu, backend, slots, n_ctx, reader, null);
}

fn loadLayers(path: []const u8, gpu: hk.engine.model.GpuMode, backend: Choice, slots: usize, n_ctx: usize, reader: *hk.HKReader, layers: ?usize) !hk.engine.Model {
    var diag = hk.engine.Diag{};
    return hk.engine.Model.init(gpa, reader, .{ .n_ctx = n_ctx, .n_threads = 2, .gpu = gpu, .backend = backend, .gpu_slots = slots, .gpu_layers = layers }, &diag) catch |e| {
        if (gpu == .off) std.debug.print("load {s}: {s}: {s}\n", .{ path, @errorName(e), diag.message() });
        return e;
    };
}

fn makeModel(dir: []const u8, name: []const u8, o: tiny.Options) ![]u8 {
    const g = try std.fmt.allocPrint(gpa, "{s}/{s}.gguf", .{ dir, name });
    defer gpa.free(g);
    const h = try std.fmt.allocPrint(gpa, "{s}/{s}.hk", .{ dir, name });
    errdefer gpa.free(h);
    try tiny.makeTinyGguf(gpa, io, g, o);
    var cdiag = hk.convert.gguf.Diag{};
    try hk.convert.gguf.convertFile(gpa, io, g, h, .{ .diag = &cdiag });
    return h;
}

/// `min_cos` is how close the two engines must be: the CPU quantizes activations to 8 bits for
/// quantized weights and the GPU does not, so those agree to about three digits, not five.
fn check(dir: []const u8, name: []const u8, o: tiny.Options, min_cos: f64, backend: Choice) !void {
    const h = try makeModel(dir, name, o);
    defer gpa.free(h);
    var reader = try hk.HKReader.open(h, gpa);
    defer reader.deinit();
    var gpu_model = load(h, .on, backend, 1, 256, &reader) catch |e| switch (e) {
        error.GpuFailed => return error.SkipZigTest,
        else => return e,
    };
    defer gpu_model.deinit();
    // The device must really be the one asked for, not a quiet CPU run.
    try std.testing.expect(gpu_model.gpu != null);
    try std.testing.expectEqual(expectedKind(backend), gpu_model.gpu.?.kind());
    var cpu_model = try load(h, .off, .cpu, 1, 256, &reader);
    defer cpu_model.deinit();

    // A prompt batch (the batched kernels), then single steps (the decode kernels).
    const prompt = [_]u32{ 3, 17, 5, 5, 29, 0, 11, 8, 40, 41, 9 };
    try gpu_model.forward(&prompt, 0);
    try cpu_model.forward(&prompt, 0);
    const lg = try gpa.dupe(f32, gpu_model.logitsFor(prompt.len - 1));
    defer gpa.free(lg);
    const lc = cpu_model.logitsFor(prompt.len - 1);
    try std.testing.expect(cosine(lg, lc) > min_cos);

    var pos: usize = prompt.len;
    var tok: u32 = 7;
    for (0..6) |_| {
        try gpu_model.forward(&.{tok}, pos);
        try cpu_model.forward(&.{tok}, pos);
        const a = try gpa.dupe(f32, gpu_model.logitsFor(0));
        defer gpa.free(a);
        const b = cpu_model.logitsFor(0);
        try std.testing.expect(cosine(a, b) > min_cos);
        var best: usize = 0;
        for (b, 0..) |x, i| if (x > b[best]) {
            best = i;
        };
        tok = @intCast(best);
        pos += 1;
    }
}

fn expectedKind(backend: Choice) hk.backend.Kind {
    return switch (backend) {
        .vulkan => .vulkan,
        .cuda => .cuda,
        .rocm => .rocm,
        else => unreachable,
    };
}

fn checkAllVariants(backend: Choice) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    // Plain float weights, and Q8_0 weights, with grouped query attention.
    try check(dir, "f32", .{}, 0.9999, backend);
    try check(dir, "q8", .{ .q8_0 = true }, 0.99, backend);
    // Plain multi head attention and a wider model.
    try check(dir, "mha", .{ .n_heads = 4, .n_kv = 4, .dim = 128, .ffn = 256 }, 0.9999, backend);
    // Qwen2: q/k/v biases and NeoX rotary embedding. Qwen3: per head q/k norms and NeoX rotary embedding.
    try check(dir, "qwen2", .{ .arch = .qwen2 }, 0.9999, backend);
    try check(dir, "qwen2q8", .{ .arch = .qwen2, .q8_0 = true }, 0.99, backend);
    try check(dir, "qwen3", .{ .arch = .qwen3 }, 0.9999, backend);
    try check(dir, "qwen3q8", .{ .arch = .qwen3, .q8_0 = true }, 0.99, backend);
}

test "the Vulkan engine matches the CPU engine" {
    try checkAllVariants(.vulkan);
}

test "the CUDA engine matches the CPU engine" {
    try checkAllVariants(.cuda);
}

test "the ROCm engine matches the CPU engine" {
    try checkAllVariants(.rocm);
}

/// Two sequences share one pass, positions advance independently, a third sequence has no slot until one is released.
fn checkSlots(backend: Choice) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const h = try makeModel(dir, "slots", .{ .arch = .qwen3 });
    defer gpa.free(h);
    var reader = try hk.HKReader.open(h, gpa);
    defer reader.deinit();
    var dev = load(h, .on, backend, 2, 256, &reader) catch |e| switch (e) {
        error.GpuFailed => return error.SkipZigTest,
        else => return e,
    };
    defer dev.deinit();
    var cpu = try load(h, .off, .cpu, 1, 256, &reader);
    defer cpu.deinit();

    const Kv = hk.engine.KvCache;
    var a = try Kv.init(gpa, dev.cfg, dev.n_ctx);
    defer a.deinit();
    var b = try Kv.init(gpa, dev.cfg, dev.n_ctx);
    defer b.deinit();
    var c = try Kv.init(gpa, dev.cfg, dev.n_ctx);
    defer c.deinit();
    var ca = try Kv.init(gpa, cpu.cfg, cpu.n_ctx);
    defer ca.deinit();
    var cb = try Kv.init(gpa, cpu.cfg, cpu.n_ctx);
    defer cb.deinit();

    const Item = hk.engine.Item;
    // Pass 1: A and B each get a 3 token prompt in one batch. Pass 2: one new token each, at different positions.
    const p1 = [_]Item{
        .{ .kv = &a, .token = 3, .pos = 0 }, .{ .kv = &a, .token = 17, .pos = 1 }, .{ .kv = &a, .token = 5, .pos = 2 },
        .{ .kv = &b, .token = 29, .pos = 0 }, .{ .kv = &b, .token = 11, .pos = 1 },
    };
    const c1 = [_]Item{
        .{ .kv = &ca, .token = 3, .pos = 0 }, .{ .kv = &ca, .token = 17, .pos = 1 }, .{ .kv = &ca, .token = 5, .pos = 2 },
        .{ .kv = &cb, .token = 29, .pos = 0 }, .{ .kv = &cb, .token = 11, .pos = 1 },
    };
    try dev.forwardItems(&p1);
    try cpu.forwardItems(&c1);
    const want_idx = [_]usize{ 2, 4 };
    const got1 = try gpa.dupe(f32, try dev.logitsMany(&want_idx));
    defer gpa.free(got1);
    const ref1 = try gpa.dupe(f32, try cpu.logitsMany(&want_idx));
    defer gpa.free(ref1);
    const v = dev.cfg.vocab;
    try std.testing.expect(cosine(got1[0..v], ref1[0..v]) > 0.9999);
    try std.testing.expect(cosine(got1[v .. 2 * v], ref1[v .. 2 * v]) > 0.9999);

    const p2 = [_]Item{ .{ .kv = &a, .token = 8, .pos = 3 }, .{ .kv = &b, .token = 40, .pos = 2 } };
    const c2 = [_]Item{ .{ .kv = &ca, .token = 8, .pos = 3 }, .{ .kv = &cb, .token = 40, .pos = 2 } };
    try dev.forwardItems(&p2);
    try cpu.forwardItems(&c2);
    const idx2 = [_]usize{ 0, 1 };
    const got2 = try gpa.dupe(f32, try dev.logitsMany(&idx2));
    defer gpa.free(got2);
    const ref2 = try gpa.dupe(f32, try cpu.logitsMany(&idx2));
    defer gpa.free(ref2);
    try std.testing.expect(cosine(got2[0..v], ref2[0..v]) > 0.9999);
    try std.testing.expect(cosine(got2[v .. 2 * v], ref2[v .. 2 * v]) > 0.9999);

    // Both slots are taken: a third sequence is refused, not silently run against someone else's cache.
    const p3 = [_]Item{.{ .kv = &c, .token = 3, .pos = 0 }};
    try std.testing.expectError(error.GpuFailed, dev.forwardItems(&p3));
    // Releasing A frees its slot for C.
    dev.forget(&a);
    try dev.forwardItems(&p3);
}

test "the Vulkan engine keeps sequences apart and runs out of slots loudly" {
    try checkSlots(.vulkan);
}

test "the CUDA engine keeps sequences apart and runs out of slots loudly" {
    try checkSlots(.cuda);
}

test "the ROCm engine keeps sequences apart and runs out of slots loudly" {
    try checkSlots(.rocm);
}

/// A context long enough that decode slices attention across several blocks (more than 256 positions), checked token
/// by token against the CPU. Prompt processing runs the whole-context kernel; every decode step runs the split one.
fn checkLongContext(backend: Choice, head_dim: usize) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const h = try makeModel(dir, "longctx", .{ .arch = .qwen3, .ctx = 1024, .dim = 4 * head_dim, .ffn = 128 });
    defer gpa.free(h);
    var reader = try hk.HKReader.open(h, gpa);
    defer reader.deinit();
    var dev = load(h, .on, backend, 1, 1024, &reader) catch |e| switch (e) {
        error.GpuFailed => return error.SkipZigTest,
        else => return e,
    };
    defer dev.deinit();
    var cpu = try load(h, .off, .cpu, 1, 1024, &reader);
    defer cpu.deinit();

    // Fewer, longer slices than usual so each warp of the CUDA decode attention walks several tiles and rescales between them.
    switch (dev.gpu.?) {
        inline .cuda, .rocm => |e| e.slice_positions = 400,
        .vulkan => {},
    }
    const prompt_len = 700;
    var prompt: [prompt_len]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(3 + (i * 7919) % 200);
    var pos: usize = 0;
    while (pos < prompt_len) {
        const n = @min(dev.n_batch, prompt_len - pos);
        try dev.forward(prompt[pos..][0..n], pos);
        try cpu.forward(prompt[pos..][0..n], pos);
        pos += n;
    }
    const lg = try gpa.dupe(f32, dev.logitsFor((prompt_len - 1) % dev.n_batch));
    defer gpa.free(lg);
    try std.testing.expect(cosine(lg, cpu.logitsFor((prompt_len - 1) % cpu.n_batch)) > 0.9999);

    var tok: u32 = 9;
    for (0..4) |_| {
        try dev.forward(&.{tok}, pos);
        try cpu.forward(&.{tok}, pos);
        const a = try gpa.dupe(f32, dev.logitsFor(0));
        defer gpa.free(a);
        const b = cpu.logitsFor(0);
        try std.testing.expect(cosine(a, b) > 0.9999);
        var best: usize = 0;
        for (b, 0..) |x, i| if (x > b[best]) {
            best = i;
        };
        tok = @intCast(best);
        pos += 1;
    }
}

test "the Vulkan engine matches the CPU on a long context" {
    try checkLongContext(.vulkan, 16);
    try checkLongContext(.vulkan, 64);
}

test "the CUDA engine matches the CPU on a long context (split-KV attention)" {
    try checkLongContext(.cuda, 16);
    try checkLongContext(.cuda, 64);
    try checkLongContext(.cuda, 128);
}

test "the ROCm engine matches the CPU on a long context (split-KV attention)" {
    try checkLongContext(.rocm, 16);
    try checkLongContext(.rocm, 64);
    try checkLongContext(.rocm, 128);
}

/// Prompt attention has a matrix-unit kernel (Vulkan cooperative matrices, CUDA tensor cores) for head sizes 64 and 128
/// (the tiny models elsewhere use 16, which runs the per-token kernel). The prompt arrives in two chunks, so the second starts at a position that is not a
/// multiple of the kernel's 16 / 32 / 64 position tiles and attends over a cache the first one wrote.
fn checkPromptAttention(head_dim: usize, backend: Choice, q8: bool) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const h = try makeModel(dir, "hd", .{ .arch = .qwen3, .dim = 4 * head_dim, .n_heads = 4, .n_kv = 2, .ffn = 256, .n_layers = 2, .ctx = 512, .q8_0 = q8 });
    defer gpa.free(h);
    var reader = try hk.HKReader.open(h, gpa);
    defer reader.deinit();
    var dev = load(h, .on, backend, 1, 512, &reader) catch |e| switch (e) {
        error.GpuFailed => return error.SkipZigTest,
        else => return e,
    };
    defer dev.deinit();
    var cpu = try load(h, .off, .cpu, 1, 512, &reader);
    defer cpu.deinit();
    // On a device with matrix units the kernel must really be the one under test.
    switch (dev.gpu.?) {
        .vulkan => |e| if (std.mem.startsWith(u8, e.matmulPath(), "cooperative")) try std.testing.expect(e.promptAttentionIsCooperative()),
        .cuda => |e| if (hk.cuda.hasTensorCores()) try std.testing.expect(e.promptAttentionIsTensor()),
        .rocm => {},
    }

    var prompt: [190]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(3 + (i * 7919) % 200);
    const split = 101;
    try dev.forward(prompt[0..split], 0);
    try cpu.forward(prompt[0..split], 0);
    const a = try gpa.dupe(f32, dev.logitsFor(split - 1));
    defer gpa.free(a);
    // Quantized weights: both engines quantize the activations to 8 bits, in different places, so they agree to ~3 digits.
    const bar: f64 = if (q8) 0.999 else 0.9999;
    try std.testing.expect(cosine(a, cpu.logitsFor(split - 1)) > bar);
    try dev.forward(prompt[split..], split);
    try cpu.forward(prompt[split..], split);
    // Every row of the second chunk, not just the last: each query attends over a different number of positions.
    for (0..prompt.len - split) |i| {
        const row = try gpa.dupe(f32, dev.logitsFor(i));
        defer gpa.free(row);
        try std.testing.expect(cosine(row, cpu.logitsFor(i)) > bar);
    }
    // Then single steps: the decode kernels (split-KV attention, integer matrix-vector products, graph replay on CUDA).
    var pos: usize = prompt.len;
    var tok: u32 = 7;
    for (0..8) |_| {
        try dev.forward(&.{tok}, pos);
        try cpu.forward(&.{tok}, pos);
        const x = try gpa.dupe(f32, dev.logitsFor(0));
        defer gpa.free(x);
        const y = cpu.logitsFor(0);
        try std.testing.expect(cosine(x, y) > bar);
        var best: usize = 0;
        for (y, 0..) |v, i| if (v > y[best]) {
            best = i;
        };
        tok = @intCast(best);
        pos += 1;
    }
}

test "the Vulkan prompt attention matches the CPU at head size 64" {
    try checkPromptAttention(64, .vulkan, false);
}

test "the Vulkan prompt attention matches the CPU at head size 128" {
    try checkPromptAttention(128, .vulkan, false);
}

test "the CUDA prompt attention matches the CPU at head size 64" {
    try checkPromptAttention(64, .cuda, false);
}

test "the CUDA prompt attention matches the CPU at head size 128" {
    try checkPromptAttention(128, .cuda, false);
}

test "the ROCm prompt attention matches the CPU at head size 64" {
    try checkPromptAttention(64, .rocm, false);
}

test "the ROCm prompt attention matches the CPU at head size 128" {
    try checkPromptAttention(128, .rocm, false);
}

/// Partial offload: the first layers run on the device, the rest (and the embedding and output) on the CPU. The result
/// must be what the CPU alone computes, for a prompt and for single steps, and the device layers must not also hold a
/// host cache.
fn checkPartial(backend: Choice, device_layers: usize) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const h = try makeModel(dir, "partial", .{ .arch = .qwen3, .n_layers = 4 });
    defer gpa.free(h);
    var reader = try hk.HKReader.open(h, gpa);
    defer reader.deinit();
    var dev = loadLayers(h, .on, backend, 1, 256, &reader, device_layers) catch |e| switch (e) {
        error.GpuFailed => return error.SkipZigTest,
        else => return e,
    };
    defer dev.deinit();
    var cpu = try load(h, .off, .cpu, 1, 256, &reader);
    defer cpu.deinit();
    try std.testing.expect(dev.gpu != null);
    try std.testing.expectEqual(device_layers, dev.gpu_layers);
    try std.testing.expectEqual(device_layers, dev.gpu.?.layerCount());
    try std.testing.expect(dev.gpu.?.isPartial());

    const prompt = [_]u32{ 3, 17, 5, 5, 29, 0, 11, 8, 40, 41, 9, 12, 77, 3, 4, 90, 21, 33 };
    try dev.forward(&prompt, 0);
    try cpu.forward(&prompt, 0);
    const a = try gpa.dupe(f32, dev.logitsFor(prompt.len - 1));
    defer gpa.free(a);
    try std.testing.expect(cosine(a, cpu.logitsFor(prompt.len - 1)) > 0.9999);

    var pos: usize = prompt.len;
    var tok: u32 = 7;
    for (0..6) |_| {
        try dev.forward(&.{tok}, pos);
        try cpu.forward(&.{tok}, pos);
        const x = try gpa.dupe(f32, dev.logitsFor(0));
        defer gpa.free(x);
        const y = cpu.logitsFor(0);
        try std.testing.expect(cosine(x, y) > 0.9999);
        var best: usize = 0;
        for (y, 0..) |v, i| if (v > y[best]) {
            best = i;
        };
        tok = @intCast(best);
        pos += 1;
    }
    // The host cache covers only the layers the CPU runs.
    try std.testing.expect(dev.kv.bytes() > 0);
    try std.testing.expectEqual(cpu.kv.bytes() / 4 * (4 - device_layers), dev.kv.bytes());
}

test "the Vulkan engine runs the leading layers and the CPU the rest" {
    try checkPartial(.vulkan, 1);
    try checkPartial(.vulkan, 3);
}

test "the CUDA engine runs the leading layers and the CPU the rest" {
    try checkPartial(.cuda, 1);
    try checkPartial(.cuda, 3);
}

test "the ROCm engine runs the leading layers and the CPU the rest" {
    try checkPartial(.rocm, 1);
    try checkPartial(.rocm, 3);
}

test "the CUDA integer tensor core kernel on a Q8_0 model matches the CPU over a prompt" {
    try checkPromptAttention(64, .cuda, true);
    try checkPromptAttention(128, .vulkan, true);
}

/// A row that is not a whole number of 32 weights is refused by the device engines (the kernels read in groups of 32)
/// and the model runs on the CPU, with the reason.
fn checkOddShape(backend: Choice) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const h = try makeModel(dir, "odd", .{ .arch = .qwen3, .dim = 48, .n_heads = 4, .n_kv = 2, .ffn = 96, .n_layers = 1 });
    defer gpa.free(h);
    var reader = try hk.HKReader.open(h, gpa);
    defer reader.deinit();
    // Asked for explicitly: a refusal.
    if (load(h, .on, backend, 1, 128, &reader)) |m| {
        var model = m;
        model.deinit();
        return error.ShapeNotRefused;
    } else |e| {
        try std.testing.expectEqual(error.GpuFailed, e);
    }
    // Left to the planner: the CPU runs it.
    var model = try load(h, .auto, backend, 1, 128, &reader);
    defer model.deinit();
    try std.testing.expect(model.gpu == null);
    try model.forward(&.{ 3, 5, 7 }, 0);
}

test "a device engine refuses rows that are not a whole number of 32 weights" {
    try checkOddShape(.cuda);
    try checkOddShape(.vulkan);
    try checkOddShape(.rocm); // last: a machine without AMD hardware skips the test here, after the others ran
}

/// The server runs every request on a worker thread, not the one that loaded the model. A CUDA context is current per
/// thread, so an engine used from another thread must bind it first (it did not, and every request failed).
fn checkOtherThread(backend: Choice) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const h = try makeModel(dir, "thread", .{ .arch = .qwen3 });
    defer gpa.free(h);
    var reader = try hk.HKReader.open(h, gpa);
    defer reader.deinit();
    var dev = load(h, .on, backend, 1, 128, &reader) catch |e| switch (e) {
        error.GpuFailed => return error.SkipZigTest,
        else => return e,
    };
    defer dev.deinit();

    const Job = struct {
        model: *hk.engine.Model,
        ok: bool = false,
        fn run(self: *@This()) void {
            self.model.forward(&.{ 3, 17, 5, 5, 29 }, 0) catch return;
            const lg = self.model.logitsFor(4);
            self.ok = lg.len > 0 and std.math.isFinite(lg[0]);
        }
    };
    var job = Job{ .model = &dev };
    const th = try std.Thread.spawn(.{}, Job.run, .{&job});
    th.join();
    try std.testing.expect(job.ok);
}

test "a device engine works from a thread other than the one that loaded it" {
    try checkOtherThread(.cuda);
    try checkOtherThread(.vulkan);
    try checkOtherThread(.rocm);
}
