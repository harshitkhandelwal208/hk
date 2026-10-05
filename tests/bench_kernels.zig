//! Single thread kernel throughput for every quantized format, so a kernel change can be judged
//! without a whole model. Weights are built by repeating the valid blocks of the test fixtures.
//!
//!   hk-kernels [rows] [cols]     default 2048 rows of 4096 columns, 64 tokens for the tiled run
//!
//! Prints, per format: decode (one token) in GB/s of weights streamed and ns per row, and the
//! register tiled batch kernel in GMAC/s.

const std = @import("std");
const hk = @import("hk");
const quant = hk.quant;
const vecdot = quant.vecdot;
const gemm = quant.gemm;
const format = hk.format;

const Fixture = struct {
    block_elems: usize,
    block_bytes: usize,
    n_blocks: usize,
    raw: []const u8,

    fn parse(bytes: []const u8) Fixture {
        const rd = struct {
            fn at(b: []const u8, i: usize) u32 {
                return std.mem.readInt(u32, b[i * 4 ..][0..4], .little);
            }
        };
        const rows = rd.at(bytes, 1);
        const bpr = rd.at(bytes, 2);
        const bb = rd.at(bytes, 4);
        return .{
            .block_elems = rd.at(bytes, 3),
            .block_bytes = bb,
            .n_blocks = rows * bpr,
            .raw = bytes[20 .. 20 + rows * bpr * bb],
        };
    }
};

const Case = struct { name: []const u8, t: format.StorageType, fixture: ?[]const u8 };

const cases = [_]Case{
    .{ .name = "q8_0", .t = .q8_0, .fixture = @embedFile("fixtures/quants/q8_0.bin") },
    .{ .name = "q4_0", .t = .q4_0, .fixture = @embedFile("fixtures/quants/q4_0.bin") },
    .{ .name = "q4_1", .t = .q4_1, .fixture = @embedFile("fixtures/quants/q4_1.bin") },
    .{ .name = "q5_0", .t = .q5_0, .fixture = @embedFile("fixtures/quants/q5_0.bin") },
    .{ .name = "q5_1", .t = .q5_1, .fixture = @embedFile("fixtures/quants/q5_1.bin") },
    .{ .name = "iq4_nl", .t = .iq4_nl, .fixture = @embedFile("fixtures/quants/iq4_nl.bin") },
    .{ .name = "q2_k", .t = .q2_k, .fixture = @embedFile("fixtures/quants/q2_k.bin") },
    .{ .name = "q3_k", .t = .q3_k, .fixture = @embedFile("fixtures/quants/q3_k.bin") },
    .{ .name = "q4_k", .t = .q4_k, .fixture = @embedFile("fixtures/quants/q4_k.bin") },
    .{ .name = "q5_k", .t = .q5_k, .fixture = @embedFile("fixtures/quants/q5_k.bin") },
    .{ .name = "q6_k", .t = .q6_k, .fixture = @embedFile("fixtures/quants/q6_k.bin") },
    .{ .name = "iq4_xs", .t = .iq4_xs, .fixture = @embedFile("fixtures/quants/iq4_xs.bin") },
    .{ .name = "iq3_s", .t = .iq3_s, .fixture = @embedFile("fixtures/quants/iq3_s.bin") },
    .{ .name = "iq3_xxs", .t = .iq3_xxs, .fixture = @embedFile("fixtures/quants/iq3_xxs.bin") },
    .{ .name = "iq2_xxs", .t = .iq2_xxs, .fixture = @embedFile("fixtures/quants/iq2_xxs.bin") },
    .{ .name = "iq2_xs", .t = .iq2_xs, .fixture = @embedFile("fixtures/quants/iq2_xs.bin") },
    .{ .name = "iq2_s", .t = .iq2_s, .fixture = @embedFile("fixtures/quants/iq2_s.bin") },
    .{ .name = "iq1_s", .t = .iq1_s, .fixture = @embedFile("fixtures/quants/iq1_s.bin") },
    .{ .name = "iq1_m", .t = .iq1_m, .fixture = @embedFile("fixtures/quants/iq1_m.bin") },
    .{ .name = "tq1_0", .t = .tq1_0, .fixture = @embedFile("fixtures/quants/tq1_0.bin") },
    .{ .name = "tq2_0", .t = .tq2_0, .fixture = @embedFile("fixtures/quants/tq2_0.bin") },
    .{ .name = "mxfp4", .t = .mxfp4, .fixture = @embedFile("fixtures/quants/mxfp4.bin") },
    .{ .name = "bf16", .t = .bf16, .fixture = @embedFile("fixtures/quants/bf16.bin") },
    .{ .name = "f16", .t = .f16, .fixture = null },
    .{ .name = "f32", .t = .f32, .fixture = null },
};

inline fn now(io: std.Io) f64 {
    return @as(f64, @floatFromInt(std.Io.Timestamp.now(io, .awake).nanoseconds)) / 1e9;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer it.deinit();
    _ = it.next();
    const rows: usize = if (it.next()) |s| try std.fmt.parseInt(usize, s, 10) else 2048;
    const cols: usize = if (it.next()) |s| try std.fmt.parseInt(usize, s, 10) else 4096;
    const only: ?[]const u8 = it.next();
    const n_tok: usize = 64;

    std.debug.print("{s:8} {s:>10} {s:>10} {s:>12}   ({d} rows x {d} cols, 1 thread)\n", .{ "format", "decode GB/s", "ns/row", "tiled GMAC/s", rows, cols });

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    for (cases) |c| {
        if (only) |o| if (!std.mem.eql(u8, o, c.name)) continue;
        const info = quant.info(c.t).?;
        const row_bytes = (cols / info.elems) * info.bytes;
        const data = try allocator.alignedAlloc(u8, .@"64", rows * row_bytes);
        defer allocator.free(data);
        if (c.fixture) |fb| {
            const fx = Fixture.parse(fb);
            var blk: usize = 0;
            var off: usize = 0;
            while (off < data.len) : (blk += 1) {
                @memcpy(data[off..][0..info.bytes], fx.raw[(blk % fx.n_blocks) * info.bytes ..][0..info.bytes]);
                off += info.bytes;
            }
        } else if (c.t == .f32) {
            const f: []f32 = @alignCast(std.mem.bytesAsSlice(f32, data));
            for (f) |*v| v.* = rand.floatNorm(f32) * 0.05;
        } else {
            const f: []f16 = @alignCast(std.mem.bytesAsSlice(f16, data));
            for (f) |*v| v.* = @floatCast(rand.floatNorm(f32) * 0.05);
        }

        const xs = try allocator.alloc(f32, n_tok * cols);
        defer allocator.free(xs);
        for (xs) |*v| v.* = rand.floatNorm(f32);
        const a8 = try allocator.alloc(vecdot.BlockA8, n_tok * (cols / 32));
        defer allocator.free(a8);
        const qk = try allocator.alloc(vecdot.BlockQ8K, n_tok * (cols / 256));
        defer allocator.free(qk);
        const acts = try allocator.alloc(vecdot.Act, n_tok);
        defer allocator.free(acts);
        for (acts, 0..) |*a, k| {
            a.* = vecdot.Act.init(xs[k * cols ..][0..cols], a8[k * (cols / 32) ..], qk[k * (cols / 256) ..]);
            a.prepare(vecdot.actKind(c.t));
        }

        // Decode: one token against every row.
        var sink: f32 = 0;
        for (0..2) |_| for (0..rows) |r| {
            sink += vecdot.dotRow(c.t, data[r * row_bytes ..][0..row_bytes], &acts[0]);
        };
        const reps: usize = @max(2, 40_000_000 / (rows * row_bytes) + 1);
        const t0 = now(io);
        for (0..reps) |_| for (0..rows) |r| {
            sink += vecdot.dotRow(c.t, data[r * row_bytes ..][0..row_bytes], &acts[0]);
        };
        const dt = now(io) - t0;
        const gbs = @as(f64, @floatFromInt(reps * rows * row_bytes)) / dt / 1e9;
        const ns_row = dt / @as(f64, @floatFromInt(reps * rows)) * 1e9;

        // Tiled: the batch kernel.
        const y = try allocator.alloc(f32, n_tok * rows);
        defer allocator.free(y);
        const scratch = try allocator.alignedAlloc(u8, .@"64", gemm.scratchBytes(c.t, cols));
        defer allocator.free(scratch);
        const args = gemm.Args{ .t = c.t, .data = data, .rows = rows, .cols = cols, .row_bytes = row_bytes, .acts = acts, .y = y, .bias = null };
        const n_tiles = (rows + gemm.tile_rows - 1) / gemm.tile_rows;
        gemm.tiles(&args, scratch, 0, n_tiles);
        const t1 = now(io);
        const greps: usize = 3;
        for (0..greps) |_| gemm.tiles(&args, scratch, 0, n_tiles);
        const gt = now(io) - t1;
        const gmacs = @as(f64, @floatFromInt(greps * rows * cols * n_tok)) / gt / 1e9;

        std.mem.doNotOptimizeAway(sink);
        std.debug.print("{s:8} {d:10.2} {d:10.0} {d:12.1}\n", .{ c.name, gbs, ns_row, gmacs });
    }
}
