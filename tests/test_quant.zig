//! Every quantized format is decoded and compared against values produced by the reference
//! `gguf` Python package (see tools/gen_quant_fixtures.py). The fixtures hold random but valid
//! blocks, so these tests catch bit-level mistakes in the layouts, not just the happy path.

const std = @import("std");
const hk = @import("hk");
const quant = hk.quant;
const format = hk.format;

const Fixture = struct {
    ggml_type: u32,
    rows: u32,
    blocks_per_row: u32,
    block_elems: u32,
    block_bytes: u32,
    raw: []const u8,
    expected: []const u8,

    fn parse(bytes: []const u8) Fixture {
        const rd = struct {
            fn u32at(b: []const u8, i: usize) u32 {
                return std.mem.readInt(u32, b[i * 4 ..][0..4], .little);
            }
        };
        const rows = rd.u32at(bytes, 1);
        const bpr = rd.u32at(bytes, 2);
        const elems = rd.u32at(bytes, 3);
        const bb = rd.u32at(bytes, 4);
        const raw_len = rows * bpr * bb;
        const exp_len = rows * bpr * elems * 4;
        return .{
            .ggml_type = rd.u32at(bytes, 0),
            .rows = rows,
            .blocks_per_row = bpr,
            .block_elems = elems,
            .block_bytes = bb,
            .raw = bytes[20 .. 20 + raw_len],
            .expected = bytes[20 + raw_len .. 20 + raw_len + exp_len],
        };
    }
};

fn check(comptime name: []const u8, t: format.StorageType) !void {
    const fx = Fixture.parse(@embedFile("fixtures/quants/" ++ name ++ ".bin"));

    // The geometry we ship must match the geometry the reference uses.
    const i = quant.info(t).?;
    try std.testing.expectEqual(fx.block_elems, i.elems);
    try std.testing.expectEqual(fx.block_bytes, i.bytes);

    const row_bytes = fx.blocks_per_row * fx.block_bytes;
    const row_elems = fx.blocks_per_row * fx.block_elems;
    const allocator = std.testing.allocator;
    const out = try allocator.alloc(f32, row_elems);
    defer allocator.free(out);

    for (0..fx.rows) |r| {
        try quant.dequantizeRow(t, fx.raw[r * row_bytes ..][0..row_bytes], out);
        for (out, 0..) |got, k| {
            const want: f32 = @bitCast(std.mem.readInt(u32, fx.expected[(r * row_elems + k) * 4 ..][0..4], .little));
            const tol = 1e-5 * @max(1.0, @abs(want));
            if (@abs(got - want) > tol) {
                std.debug.print("{s} row {d} elem {d}: got {d} want {d}\n", .{ name, r, k, got, want });
                return error.DecodeMismatch;
            }
        }
    }
}

test "legacy quant decoders match the reference" {
    try check("q4_0", .q4_0);
    try check("q4_1", .q4_1);
    try check("q5_0", .q5_0);
    try check("q5_1", .q5_1);
    try check("q8_0", .q8_0);
    try check("bf16", .bf16);
}

test "k-quant decoders match the reference" {
    try check("q2_k", .q2_k);
    try check("q3_k", .q3_k);
    try check("q4_k", .q4_k);
    try check("q5_k", .q5_k);
    try check("q6_k", .q6_k);
}

test "iq decoders match the reference" {
    try check("iq2_xxs", .iq2_xxs);
    try check("iq2_xs", .iq2_xs);
    try check("iq2_s", .iq2_s);
    try check("iq3_xxs", .iq3_xxs);
    try check("iq3_s", .iq3_s);
    try check("iq1_s", .iq1_s);
    try check("iq1_m", .iq1_m);
    try check("iq4_nl", .iq4_nl);
    try check("iq4_xs", .iq4_xs);
}

test "ternary and fp4 decoders match the reference" {
    try check("tq1_0", .tq1_0);
    try check("tq2_0", .tq2_0);
    try check("mxfp4", .mxfp4);
    try check("nvfp4", .nvfp4);
}

test "decoding rejects wrong lengths instead of guessing" {
    var out: [32]f32 = undefined;
    const bytes: [34]u8 = @splat(0);
    try std.testing.expectError(error.BadLength, quant.dequantizeRow(.q8_0, bytes[0..33], &out));
    try std.testing.expectError(error.BadLength, quant.dequantizeRow(.q8_0, &bytes, out[0..31]));
    try std.testing.expectError(error.UnsupportedType, quant.dequantizeRow(.sparse_2_4, &bytes, &out));
}

// ---------------------------------------------------------------------------------------
// Dot product kernels
// ---------------------------------------------------------------------------------------

const vecdot = quant.vecdot;

/// Reference: decode the whole row in f64 and dot with the same activation.
fn referenceDot(t: format.StorageType, row: []const u8, x: []const f32, abs_sum: *f64) !f64 {
    const allocator = std.testing.allocator;
    const w = try allocator.alloc(f32, x.len);
    defer allocator.free(w);
    try quant.dequantizeRow(t, row, w);
    var s: f64 = 0;
    abs_sum.* = 0;
    for (w, x) |wv, xv| {
        s += @as(f64, wv) * xv;
        abs_sum.* += @abs(@as(f64, wv) * xv);
    }
    return s;
}

fn checkDot(comptime name: []const u8, t: format.StorageType, rel_tol: f64) !void {
    const fx = Fixture.parse(@embedFile("fixtures/quants/" ++ name ++ ".bin"));
    const row_bytes = fx.blocks_per_row * fx.block_bytes;
    const row_elems = fx.blocks_per_row * fx.block_elems;
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(0xD07);
    const rand = prng.random();
    const x = try allocator.alloc(f32, row_elems);
    defer allocator.free(x);
    const a8 = try allocator.alloc(vecdot.BlockA8, row_elems / 32);
    defer allocator.free(a8);
    const qk = try allocator.alloc(vecdot.BlockQ8K, @max(1, row_elems / 256));
    defer allocator.free(qk);

    for (0..8) |_| {
        for (x) |*v| v.* = rand.float(f32) * 2.0 - 1.0;
        var act = vecdot.Act.init(x, a8, qk);
        act.prepare(vecdot.actKind(t));
        for (0..fx.rows) |r| {
            const row = fx.raw[r * row_bytes ..][0..row_bytes];
            var abs_sum: f64 = 0;
            const want = try referenceDot(t, row, x, &abs_sum);
            const got = vecdot.dotRow(t, row, &act);
            // Allowed error scales with the magnitude of the terms, because rounding the
            // activation to 8 bits perturbs every product by a bounded relative amount.
            const tol = rel_tol * abs_sum + 1e-4;
            if (@abs(@as(f64, got) - want) > tol) {
                std.debug.print("{s}: got {d} want {d} tol {d}\n", .{ name, got, want, tol });
                return error.DotMismatch;
            }
        }
    }
}

test "integer dot kernels agree with decode then float dot" {
    // 8 bit activation quantization costs about 0.4 percent of the term magnitude.
    try checkDot("q4_0", .q4_0, 0.01);
    try checkDot("q4_1", .q4_1, 0.01);
    try checkDot("q5_0", .q5_0, 0.01);
    try checkDot("q5_1", .q5_1, 0.01);
    try checkDot("q8_0", .q8_0, 0.01);
    try checkDot("iq4_nl", .iq4_nl, 0.01);
    try checkDot("q2_k", .q2_k, 0.01);
    try checkDot("q3_k", .q3_k, 0.01);
    try checkDot("q4_k", .q4_k, 0.01);
    try checkDot("q5_k", .q5_k, 0.01);
    try checkDot("q6_k", .q6_k, 0.01);
    try checkDot("iq4_xs", .iq4_xs, 0.01);
    try checkDot("iq2_xxs", .iq2_xxs, 0.01);
    try checkDot("iq2_xs", .iq2_xs, 0.01);
    try checkDot("iq2_s", .iq2_s, 0.01);
    try checkDot("iq3_xxs", .iq3_xxs, 0.01);
    try checkDot("iq3_s", .iq3_s, 0.01);
    try checkDot("iq1_s", .iq1_s, 0.01);
    try checkDot("iq1_m", .iq1_m, 0.01);
    try checkDot("tq1_0", .tq1_0, 0.01);
    try checkDot("tq2_0", .tq2_0, 0.01);
    try checkDot("mxfp4", .mxfp4, 0.01);
}

test "exact fallback dot kernels match decode then float dot" {
    // No activation quantization on this path, so only float summation order differs.
    try checkDot("nvfp4", .nvfp4, 1e-5);
    try checkDot("bf16", .bf16, 1e-5);
}

test "every type the loaders accept has a dot kernel" {
    // Guards against adding a storage type to `supported` without handling it in `dotRow`.
    inline for (@typeInfo(format.StorageType).@"enum".fields) |f| {
        const t: format.StorageType = @enumFromInt(f.value);
        if (vecdot.supported(t)) try std.testing.expect(quant.info(t) != null);
    }
}

fn checkTilesN(comptime N: usize, comptime name: []const u8, t: format.StorageType) !void {
    const fx = Fixture.parse(@embedFile("fixtures/quants/" ++ name ++ ".bin"));
    const row_bytes = fx.blocks_per_row * fx.block_bytes;
    const row_elems = fx.blocks_per_row * fx.block_elems;
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(0x7117);
    const rand = prng.random();
    const xs = try allocator.alloc(f32, N * row_elems);
    defer allocator.free(xs);
    for (xs) |*v| v.* = rand.float(f32) * 2.0 - 1.0;
    const a8 = try allocator.alloc(vecdot.BlockA8, N * (row_elems / 32));
    defer allocator.free(a8);
    const qk = try allocator.alloc(vecdot.BlockQ8K, N * @max(1, row_elems / 256));
    defer allocator.free(qk);

    var acts: [N]vecdot.Act = undefined;
    for (&acts, 0..) |*a, k| {
        a.* = vecdot.Act.init(
            xs[k * row_elems ..][0..row_elems],
            a8[k * (row_elems / 32) ..],
            qk[k * @max(1, row_elems / 256) ..],
        );
        a.prepare(vecdot.actKind(t));
    }
    for (0..fx.rows) |r| {
        const row = fx.raw[r * row_bytes ..][0..row_bytes];
        var ptrs: [N]*const vecdot.Act = undefined;
        inline for (0..N) |k| ptrs[k] = &acts[k];
        const tiled = vecdot.dotRowN(N, t, row, ptrs);
        for (0..N) |k| {
            const single = vecdot.dotRow(t, row, &acts[k]);
            // Same arithmetic in the same order, so equal up to float contraction differences.
            if (@abs(tiled[k] - single) > 1e-4 * @max(1.0, @abs(single))) {
                std.debug.print("{s}: token {d} tiled {d} single {d}\n", .{ name, k, tiled[k], single });
                return error.TileMismatch;
            }
        }
    }
}

fn checkTiles(comptime name: []const u8, t: format.StorageType) !void {
    try checkTilesN(4, name, t);
    try checkTilesN(8, name, t);
}

test "four and eight token tiles equal single dots for every format" {
    try checkTiles("q4_0", .q4_0);
    try checkTiles("q4_1", .q4_1);
    try checkTiles("q5_0", .q5_0);
    try checkTiles("q5_1", .q5_1);
    try checkTiles("q8_0", .q8_0);
    try checkTiles("iq4_nl", .iq4_nl);
    try checkTiles("q2_k", .q2_k);
    try checkTiles("q3_k", .q3_k);
    try checkTiles("q4_k", .q4_k);
    try checkTiles("q5_k", .q5_k);
    try checkTiles("q6_k", .q6_k);
    try checkTiles("iq4_xs", .iq4_xs);
    try checkTiles("iq2_xxs", .iq2_xxs);
    try checkTiles("iq3_s", .iq3_s);
    try checkTiles("tq2_0", .tq2_0);
    try checkTiles("mxfp4", .mxfp4);
    try checkTiles("bf16", .bf16);
}

// ---------------------------------------------------------------------------------------
// Register tiled matrix multiply
// ---------------------------------------------------------------------------------------

const gemm = quant.gemm;

fn checkGemm(comptime name: []const u8, t: format.StorageType) !void {
    const fx = Fixture.parse(@embedFile("fixtures/quants/" ++ name ++ ".bin"));
    const row_bytes = fx.blocks_per_row * fx.block_bytes;
    const row_elems = fx.blocks_per_row * fx.block_elems;
    const allocator = std.testing.allocator;
    // Enough rows for several tiles and a ragged last one, tokens for every micro kernel width.
    const rows: usize = 37;
    const n_tok: usize = 11;

    const data = try allocator.alloc(u8, rows * row_bytes);
    defer allocator.free(data);
    for (0..rows) |r| @memcpy(data[r * row_bytes ..][0..row_bytes], fx.raw[(r % fx.rows) * row_bytes ..][0..row_bytes]);

    var prng = std.Random.DefaultPrng.init(0x6E33);
    const rand = prng.random();
    const xs = try allocator.alloc(f32, n_tok * row_elems);
    defer allocator.free(xs);
    for (xs) |*v| v.* = rand.float(f32) * 2.0 - 1.0;
    const a8 = try allocator.alloc(vecdot.BlockA8, n_tok * (row_elems / 32));
    defer allocator.free(a8);
    const qk = try allocator.alloc(vecdot.BlockQ8K, n_tok * @max(1, row_elems / 256));
    defer allocator.free(qk);
    var acts: [n_tok]vecdot.Act = undefined;
    for (&acts, 0..) |*a, k| {
        a.* = vecdot.Act.init(xs[k * row_elems ..][0..row_elems], a8[k * (row_elems / 32) ..], qk[k * @max(1, row_elems / 256) ..]);
        a.prepare(vecdot.actKind(t));
    }

    const bias = try allocator.alloc(f32, rows);
    defer allocator.free(bias);
    for (bias, 0..) |*b, i| b.* = @floatFromInt(i % 5);
    const y = try allocator.alloc(f32, n_tok * rows);
    defer allocator.free(y);
    @memset(y, std.math.nan(f32));

    const scratch = try allocator.alignedAlloc(u8, .@"64", gemm.scratchBytes(t, row_elems));
    defer allocator.free(scratch);
    const args = gemm.Args{ .t = t, .data = data, .rows = rows, .cols = row_elems, .row_bytes = row_bytes, .acts = &acts, .y = y, .bias = bias };
    gemm.tiles(&args, scratch, 0, (rows + gemm.tile_rows - 1) / gemm.tile_rows);

    for (0..rows) |r| {
        const row = data[r * row_bytes ..][0..row_bytes];
        for (0..n_tok) |k| {
            const want = vecdot.dotRow(t, row, &acts[k]) + bias[r];
            const got = y[k * rows + r];
            if (!(@abs(got - want) <= 1e-4 * @max(1.0, @abs(want)))) {
                std.debug.print("{s}: row {d} token {d} tiled {d} single {d}\n", .{ name, r, k, got, want });
                return error.GemmMismatch;
            }
        }
    }
}

test "tiled matmul equals single dots for every format" {
    try checkGemm("q4_0", .q4_0);
    try checkGemm("q4_1", .q4_1);
    try checkGemm("q5_0", .q5_0);
    try checkGemm("q5_1", .q5_1);
    try checkGemm("q8_0", .q8_0);
    try checkGemm("iq4_nl", .iq4_nl);
    try checkGemm("q2_k", .q2_k);
    try checkGemm("q3_k", .q3_k);
    try checkGemm("q4_k", .q4_k);
    try checkGemm("q5_k", .q5_k);
    try checkGemm("q6_k", .q6_k);
    try checkGemm("iq4_xs", .iq4_xs);
    try checkGemm("iq2_xxs", .iq2_xxs);
    try checkGemm("iq2_xs", .iq2_xs);
    try checkGemm("iq2_s", .iq2_s);
    try checkGemm("iq3_xxs", .iq3_xxs);
    try checkGemm("iq3_s", .iq3_s);
    try checkGemm("iq1_s", .iq1_s);
    try checkGemm("iq1_m", .iq1_m);
    try checkGemm("tq1_0", .tq1_0);
    try checkGemm("tq2_0", .tq2_0);
    try checkGemm("mxfp4", .mxfp4);
    try checkGemm("nvfp4", .nvfp4);
    try checkGemm("bf16", .bf16);
}

// ---------------------------------------------------------------------------------------
// Every instruction set level of the kernel table must agree with the portable definition
// ---------------------------------------------------------------------------------------

fn checkLevel(api: *const hk.kernels.Api, comptime name: []const u8, t: format.StorageType) !void {
    const fx = Fixture.parse(@embedFile("fixtures/quants/" ++ name ++ ".bin"));
    const row_bytes = fx.blocks_per_row * fx.block_bytes;
    const row_elems = fx.blocks_per_row * fx.block_elems;
    const allocator = std.testing.allocator;
    const rows: usize = 37;
    const n_tok: usize = 10;

    const data = try allocator.alloc(u8, rows * row_bytes);
    defer allocator.free(data);
    for (0..rows) |r| @memcpy(data[r * row_bytes ..][0..row_bytes], fx.raw[(r % fx.rows) * row_bytes ..][0..row_bytes]);

    var prng = std.Random.DefaultPrng.init(0x1E7E1);
    const rand = prng.random();
    const xs = try allocator.alloc(f32, n_tok * row_elems);
    defer allocator.free(xs);
    for (xs) |*v| v.* = rand.float(f32) * 2.0 - 1.0;
    const a8 = try allocator.alloc(vecdot.BlockA8, n_tok * (row_elems / 32));
    defer allocator.free(a8);
    const qk = try allocator.alloc(vecdot.BlockQ8K, n_tok * @max(1, row_elems / 256));
    defer allocator.free(qk);
    var acts: [n_tok]vecdot.Act = undefined;
    for (&acts, 0..) |*a, k| {
        a.* = vecdot.Act.init(xs[k * row_elems ..][0..row_elems], a8[k * (row_elems / 32) ..], qk[k * @max(1, row_elems / 256) ..]);
    }
    api.prepare(&acts, vecdot.actKind(t));

    const y_rows = try allocator.alloc(f32, n_tok * rows);
    defer allocator.free(y_rows);
    const y_tiles = try allocator.alloc(f32, n_tok * rows);
    defer allocator.free(y_tiles);

    const ra = hk.kernels.RowsArgs{ .t = t, .data = data, .row_bytes = row_bytes, .rows = rows, .acts = &acts, .y = y_rows, .bias = null };
    api.rows(&ra, 0, rows);

    const scratch = try allocator.alignedAlloc(u8, .@"64", api.scratch_bytes(t, row_elems));
    defer allocator.free(scratch);
    const ga = hk.quant.gemm.Args{ .t = t, .data = data, .rows = rows, .cols = row_elems, .row_bytes = row_bytes, .acts = &acts, .y = y_tiles, .bias = null };
    api.tiles(&ga, scratch, 0, (rows + api.tile_rows - 1) / api.tile_rows);

    for (0..rows) |r| {
        var abs_sum: f64 = 0;
        const row = data[r * row_bytes ..][0..row_bytes];
        for (0..n_tok) |k| {
            const want = try referenceDot(t, row, xs[k * row_elems ..][0..row_elems], &abs_sum);
            const tol = 0.01 * abs_sum + 1e-3;
            if (@abs(@as(f64, y_rows[k * rows + r]) - want) > tol or @abs(@as(f64, y_tiles[k * rows + r]) - want) > tol) {
                std.debug.print("level {s} {s} row {d} token {d}: rows {d} tiles {d} want {d}\n", .{ api.name, name, r, k, y_rows[k * rows + r], y_tiles[k * rows + r], want });
                return error.LevelMismatch;
            }
        }
    }
}

test "every compiled instruction set level the cpu supports computes the same thing" {
    var ran: usize = 0;
    for (hk.kernels.compiled_levels) |level| {
        const api = hk.kernels.byName(level) orelse continue;
        ran += 1;
        try checkLevel(api, "q8_0", .q8_0);
        try checkLevel(api, "q4_0", .q4_0);
        try checkLevel(api, "q4_1", .q4_1);
        try checkLevel(api, "q5_0", .q5_0);
        try checkLevel(api, "q5_1", .q5_1);
        try checkLevel(api, "iq4_nl", .iq4_nl);
        try checkLevel(api, "q2_k", .q2_k);
        try checkLevel(api, "q3_k", .q3_k);
        try checkLevel(api, "q4_k", .q4_k);
        try checkLevel(api, "q5_k", .q5_k);
        try checkLevel(api, "q6_k", .q6_k);
        try checkLevel(api, "iq4_xs", .iq4_xs);
        try checkLevel(api, "bf16", .bf16);
        try checkLevel(api, "iq3_s", .iq3_s);
    }
    try std.testing.expect(ran >= 1);
}

// ---------------------------------------------------------------------------------------
// Canonical unpacking must reproduce the exact decoders
// ---------------------------------------------------------------------------------------

fn checkUnpackK(comptime name: []const u8, comptime t: format.StorageType) !void {
    const fx = Fixture.parse(@embedFile("fixtures/quants/" ++ name ++ ".bin"));
    const row_bytes = fx.blocks_per_row * fx.block_bytes;
    var dec: [256]f32 = undefined;
    for (0..fx.rows) |r| {
        for (0..fx.blocks_per_row) |b| {
            const raw = fx.raw[r * row_bytes + b * fx.block_bytes ..][0..fx.block_bytes];
            try quant.dequantizeRow(t, raw, &dec);
            const kb = quant.unpack.unpackK(t, fx.raw[r * row_bytes ..][0..row_bytes], b);
            for (0..256) |j| {
                const v: i8 = @bitCast(kb.vals[j]);
                const got = kb.d * @as(f32, @floatFromInt(kb.sc[j / 16])) * @as(f32, @floatFromInt(if (std.mem.indexOfScalar(format.StorageType, &.{ .q2_k, .q4_k, .q5_k, .q3_k, .q6_k }, t) != null) @as(i32, kb.vals[j]) else @as(i32, v))) - kb.e * @as(f32, @floatFromInt(kb.mn[j / 16]));
                if (@abs(got - dec[j]) > 1e-5 * @max(1.0, @abs(dec[j]))) {
                    std.debug.print("{s} row {d} block {d} elem {d}: unpacked {d} decoded {d}\n", .{ name, r, b, j, got, dec[j] });
                    return error.UnpackMismatch;
                }
            }
        }
    }
}

test "unpacked K blocks equal the decoders" {
    try checkUnpackK("q2_k", .q2_k);
    try checkUnpackK("q3_k", .q3_k);
    try checkUnpackK("q4_k", .q4_k);
    try checkUnpackK("q5_k", .q5_k);
    try checkUnpackK("q6_k", .q6_k);
    try checkUnpackK("iq4_xs", .iq4_xs);
    try checkUnpackK("iq2_xxs", .iq2_xxs);
    try checkUnpackK("iq2_xs", .iq2_xs);
    try checkUnpackK("iq2_s", .iq2_s);
    try checkUnpackK("iq3_xxs", .iq3_xxs);
    try checkUnpackK("iq3_s", .iq3_s);
    try checkUnpackK("iq1_s", .iq1_s);
    try checkUnpackK("iq1_m", .iq1_m);
    try checkUnpackK("tq1_0", .tq1_0);
    try checkUnpackK("tq2_0", .tq2_0);
}
