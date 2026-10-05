//! Standalone GPU loading/compute benchmark: uploads every attention/MLP
//! weight matrix in a real .hk model to VRAM via the CUDA backend, then runs
//! one GEMV per matrix (a stand-in for one token's linear-layer work) to
//! prove the weights are both correctly placed on-device AND computable
//! there. Built only with -Dcuda=true.
//!
//! Scope note: this exercises matVec on the GPU only. Attention/RoPE/
//! softmax/KV-cache in TransformerEngine.forward() are still CPU-only -
//! this is NOT a full GPU inference path yet, just the loading + linear-
//! algebra half of it.

const std = @import("std");
const hk = @import("hk");
const cuda = hk.cuda;

const LayerWeightName = struct {
    suffix: []const u8,
};

const WEIGHT_SUFFIXES = [_][]const u8{
    "attn_q",     "attn_k",     "attn_v",     "attn_output",
    "ffn_gate",   "ffn_up",     "ffn_down",
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer it.deinit();
    _ = it.next(); // skip program name
    const path = it.next() orelse {
        std.debug.print("usage: hk_gpu_bench <file.hk>\n", .{});
        return error.MissingArgument;
    };

    if (!cuda.enabled) {
        std.debug.print("built without -Dcuda=true, nothing to benchmark\n", .{});
        return;
    }
    if (!cuda.isAvailable()) {
        std.debug.print("no CUDA device visible\n", .{});
        return error.NoCudaDevice;
    }

    var name_buf: [256]u8 = undefined;
    const dev_name = try cuda.getDeviceName(&name_buf);
    const total_mem = try cuda.totalMemBytes();
    const free_before = try cuda.freeMemBytes();
    std.debug.print("GPU: {s} | total {d} MiB | free {d} MiB\n", .{
        dev_name,
        @divTrunc(total_mem, 1024 * 1024),
        @divTrunc(free_before, 1024 * 1024),
    });

    var reader = try hk.reader.HKReader.open(path, allocator);
    defer reader.deinit();

    // Detect layer count the same way TransformerEngine.initFromReader does.
    var layer_count: usize = 0;
    var lname_buf: [64]u8 = undefined;
    while (true) : (layer_count += 1) {
        const test_name = std.fmt.bufPrint(&lname_buf, "blk.{}.attn_q.weight", .{layer_count}) catch break;
        if (reader.toc.find(test_name) == null) break;
    }
    if (layer_count == 0) return error.NoLayersFound;
    std.debug.print("layers: {d}\n", .{layer_count});

    const WeightRef = struct {
        dev: cuda.DeviceBuffer,
        storage_type: hk.format.StorageType,
        rows: usize,
        cols: usize,
        name: [96]u8,
        name_len: usize,
    };

    var weights = std.ArrayList(WeightRef).empty;
    defer {
        for (weights.items) |w| w.dev.free();
        weights.deinit(allocator);
    }

    var max_cols: usize = 0;
    var max_rows: usize = 0;
    var total_bytes: u64 = 0;
    var skipped: usize = 0;

    // Which tensor names are one of our per-layer matmul weights (kept resident
    // for the compute pass below) vs everything else in the container
    // (embeddings, per-layer norms, output head, output norm -- uploaded and
    // measured too, since llama.cpp's -ngl full offload moves those as well,
    // but not needed after that so freed immediately).
    const isMatmulWeight = struct {
        fn check(name: []const u8) bool {
            if (!std.mem.startsWith(u8, name, "blk.")) return false;
            for (WEIGHT_SUFFIXES) |suffix| {
                var buf: [16]u8 = undefined;
                const wanted = std.fmt.bufPrint(&buf, ".{s}.", .{suffix}) catch continue;
                if (std.mem.indexOf(u8, name, wanted) != null) return true;
            }
            return false;
        }
    }.check;

    var total_tensors: usize = 0;
    const io = std.Options.debug_io;
    const upload_start = std.Io.Clock.Timestamp.now(io, .awake);
    for (reader.toc.entries.items) |entry| {
        if (entry.storage_type != .q8_0 and entry.storage_type != .f32) {
            skipped += 1;
            continue;
        }
        const bytes = try reader.getTensorData(entry);
        const dev = try cuda.DeviceBuffer.upload(bytes);
        total_bytes += bytes.len;
        total_tensors += 1;

        if (!isMatmulWeight(entry.name)) {
            // Full-model transfer accounting only -- not one of the linear
            // layers the compute pass below exercises.
            dev.free();
            continue;
        }

        const rows: usize = @intCast(entry.shape[0]);
        const cols: usize = @intCast(entry.shape[1]);
        max_rows = @max(max_rows, rows);
        max_cols = @max(max_cols, cols);

        {
            var wref: WeightRef = .{
                .dev = dev,
                .storage_type = entry.storage_type,
                .rows = rows,
                .cols = cols,
                .name = undefined,
                .name_len = @min(entry.name.len, 96),
            };
            @memcpy(wref.name[0..wref.name_len], entry.name[0..wref.name_len]);
            try weights.append(allocator, wref);
        }
    }
    const upload_ns: i96 = upload_start.untilNow(io).raw.nanoseconds;
    const free_after_upload = try cuda.freeMemBytes();

    const upload_s = @as(f64, @floatFromInt(upload_ns)) / 1e9;
    const gb = @as(f64, @floatFromInt(total_bytes)) / (1024.0 * 1024.0 * 1024.0);
    std.debug.print(
        "\n== Full-model upload (host mmap -> VRAM), matches what llama.cpp's -ngl full offload moves ==\n" ++
            "  tensors uploaded : {d} total ({d} linear-layer weights kept resident, rest freed after transfer; skipped {d} non-f32/q8_0)\n" ++
            "  total size       : {d:.3} GB\n" ++
            "  wall time        : {d:.2} ms\n" ++
            "  throughput       : {d:.2} GB/s\n" ++
            "  VRAM resident now: {d} MiB (free {d} -> {d} MiB)\n",
        .{
            total_tensors,
            weights.items.len,
            skipped,
            gb,
            upload_s * 1000.0,
            gb / upload_s,
            @divTrunc(free_before - free_after_upload, 1024 * 1024),
            @divTrunc(free_before, 1024 * 1024),
            @divTrunc(free_after_upload, 1024 * 1024),
        },
    );

    // One shared random input vector and output scratch buffer, sized to the
    // largest K/M seen; kernels only read/write the first K/M elements of
    // each so this is safe to reuse across every matrix without re-uploading.
    var prng = std.Random.DefaultPrng.init(1234);
    const rand = prng.random();
    const host_x = try allocator.alloc(f32, max_cols);
    defer allocator.free(host_x);
    for (host_x) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    const d_x = try cuda.DeviceBuffer.upload(std.mem.sliceAsBytes(host_x));
    defer d_x.free();
    const d_y = try cuda.DeviceBuffer.allocUninit(max_rows * @sizeOf(f32));
    defer d_y.free();

    const compute_start = std.Io.Clock.Timestamp.now(io, .awake);
    for (weights.items) |w| {
        switch (w.storage_type) {
            .f32 => try cuda.gemvF32(w.dev, d_x, d_y, w.rows, w.cols),
            .q8_0 => try cuda.gemvQ8_0(w.dev, d_x, d_y, w.rows, w.cols),
            else => unreachable,
        }
    }
    try cuda.synchronize();
    const compute_ns: i96 = compute_start.untilNow(io).raw.nanoseconds;

    const host_y = try allocator.alloc(f32, max_rows);
    defer allocator.free(host_y);
    try d_y.download(std.mem.sliceAsBytes(host_y));
    var any_nonfinite = false;
    for (host_y) |v| {
        if (!std.math.isFinite(v)) any_nonfinite = true;
    }

    const compute_ms = @as(f64, @floatFromInt(compute_ns)) / 1e6;
    std.debug.print(
        "\n== One-token linear-layer GEMV pass, all layers, on GPU ==\n" ++
            "  matmuls run      : {d} (across {d} layers)\n" ++
            "  wall time        : {d:.3} ms\n" ++
            "  implied tok/s    : {d:.1}  (linear-algebra GEMV raw compute)\n" ++
            "  sample output ok : {} (finite)\n",
        .{
            weights.items.len,
            layer_count,
            compute_ms,
            1000.0 / compute_ms,
            !any_nonfinite,
        },
    );
}
