//! The GPU engine must compute what the CPU engine computes. Skipped when the machine has no
//! usable Vulkan device.

const std = @import("std");
const hk = @import("hk");
const tiny = @import("support/tiny_model.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;

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

fn load(path: []const u8, gpu: hk.engine.model.GpuMode, reader: *hk.HKReader) !hk.engine.Model {
    var diag = hk.engine.Diag{};
    return hk.engine.Model.init(gpa, reader, .{ .n_ctx = 256, .n_threads = 2, .gpu = gpu }, &diag) catch |e| {
        std.debug.print("load {s}: {s}: {s}\n", .{ path, @errorName(e), diag.message() });
        return e;
    };
}

/// `min_cos` is how close the two engines must be: the CPU quantizes activations to 8 bits for
/// quantized weights and the GPU does not, so those agree to about three digits, not five.
fn check(dir: []const u8, name: []const u8, o: tiny.Options, min_cos: f64) !void {
    const g = try std.fmt.allocPrint(gpa, "{s}/{s}.gguf", .{ dir, name });
    defer gpa.free(g);
    const h = try std.fmt.allocPrint(gpa, "{s}/{s}.hk", .{ dir, name });
    defer gpa.free(h);
    try tiny.makeTinyGguf(gpa, io, g, o);
    var cdiag = hk.convert.gguf.Diag{};
    try hk.convert.gguf.convertFile(gpa, io, g, h, .{ .diag = &cdiag });

    var reader = try hk.HKReader.open(h, gpa);
    defer reader.deinit();
    var gpu_model = load(h, .on, &reader) catch |e| switch (e) {
        error.GpuFailed => return error.SkipZigTest,
        else => return e,
    };
    defer gpu_model.deinit();
    var cpu_model = try load(h, .off, &reader);
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

test "the GPU engine matches the CPU engine" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    // Plain float weights, and Q8_0 weights, with grouped query attention.
    try check(dir, "f32", .{}, 0.9999);
    try check(dir, "q8", .{ .q8_0 = true }, 0.99);
    // Plain multi head attention and a wider model.
    try check(dir, "mha", .{ .n_heads = 4, .n_kv = 4, .dim = 128, .ffn = 256 }, 0.9999);
}
