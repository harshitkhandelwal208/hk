const std = @import("std");
const hk = @import("hk");

test "file header validation" {
    var header = hk.FileHeader{};
    try std.testing.expect(header.isValid());
    try std.testing.expectEqual(@sizeOf(hk.FileHeader), 128);
}

test "metadata serialization roundtrip" {
    const allocator = std.testing.allocator;
    var map = hk.MetadataMap.init(allocator);
    defer map.deinit();

    try map.setString("model_name", "HK-Transformer-v1");
    try map.setInt("hidden_dim", 768);
    try map.setFloat("learning_rate", 0.0003);
    try map.setBool("is_pruned", true);

    var writer = hk.buf.BufferWriter.init(allocator);
    defer writer.deinit();
    try map.serialize(&writer);

    var deserialized = try hk.MetadataMap.deserialize(writer.getBytes(), map.items.items.len, allocator);
    defer deserialized.deinit();

    const name_val = deserialized.get("model_name").?;
    try std.testing.expectEqualStrings("HK-Transformer-v1", name_val.val_string);

    const dim_val = deserialized.get("hidden_dim").?;
    try std.testing.expectEqual(768, dim_val.val_int64);

    const lr_val = deserialized.get("learning_rate").?;
    try std.testing.expectApproxEqAbs(@as(f64, 0.0003), lr_val.val_float64, 1e-6);

    const pruned_val = deserialized.get("is_pruned").?;
    try std.testing.expectEqual(true, pruned_val.val_bool);
}

test "nf4 quantization and precision recovery" {
    const original: [32]f32 = .{
        0.12,  -0.45, 0.88,  -0.02, 0.33,  -0.71, 0.95,  -0.34,
        0.05,  -0.19, 0.44,  -0.83, 0.27,  -0.55, 0.62,  -0.09,
        0.38,  -0.66, 0.77,  -0.15, 0.49,  -0.28, 0.81,  -0.42,
        0.18,  -0.37, 0.53,  -0.90, 0.22,  -0.61, 0.73,  -0.04,
    };

    var packed_bytes: [16]u8 = undefined;
    var residuals: [32]f32 = undefined;
    const scale = hk.quantization.quantizeBlockNF4(&original, &packed_bytes, &residuals);

    try std.testing.expect(scale > 0.0);

    // Dequantize base (approximate)
    var dequant_base: [32]f32 = undefined;
    hk.quantization.dequantizeBlockNF4(&packed_bytes, scale, 32, &dequant_base);

    // Dequantize with residual recovery (exact)
    var dequant_recovered: [32]f32 = undefined;
    for (0..32) |i| {
        dequant_recovered[i] = dequant_base[i] + residuals[i];
    }

    for (0..32) |i| {
        try std.testing.expectApproxEqAbs(original[i], dequant_recovered[i], 1e-6);
    }
}

test "dq8 quantization and precision recovery" {
    const original: [32]f32 = .{
        -1.5,  2.3, -0.4, 0.9, -3.1, 4.2, -0.1, 1.8,
        -2.2,  1.1, -1.9, 3.5, -0.8, 2.7, -3.8, 0.5,
        -1.2,  0.7, -2.5, 3.1, -0.3, 1.4, -2.9, 4.0,
        -0.6,  2.1, -1.7, 3.3, -2.0, 1.6, -0.9, 2.8,
    };

    var i8_data: [32]i8 = undefined;
    var residuals: [32]f32 = undefined;
    const scale = hk.quantization.quantizeBlockDQ8(&original, &i8_data, &residuals);

    try std.testing.expect(scale > 0.0);

    var dequant_base: [32]f32 = undefined;
    hk.quantization.dequantizeBlockDQ8(&i8_data, scale, 32, &dequant_base);

    var dequant_recovered: [32]f32 = undefined;
    for (0..32) |i| {
        dequant_recovered[i] = dequant_base[i] + residuals[i];
        try std.testing.expectApproxEqAbs(original[i], dequant_recovered[i], 1e-5);
    }
}

test "dqt ternary quantization roundtrip" {
    const original: [16]f32 = .{
        1.2, -1.5, 0.05, -0.02, 1.8, -1.9, 0.01, 0.0,
        -1.1, 1.4, -0.03, 0.04, 1.6, -1.7, 0.02, -0.01,
    };

    var packed_bytes: [4]u8 = undefined;
    const scale = hk.quantization.quantizeBlockDQT(&original, &packed_bytes, null);

    try std.testing.expect(scale > 0.0);

    var out: [16]f32 = undefined;
    hk.quantization.dequantizeBlockDQT(&packed_bytes, scale, 16, &out);

    // Large positive values become +scale, large negatives become -scale, zeros stay 0
    try std.testing.expectEqual(scale, out[0]);
    try std.testing.expectEqual(-scale, out[1]);
    try std.testing.expectEqual(@as(f32, 0.0), out[2]);
    try std.testing.expectEqual(@as(f32, 0.0), out[3]);
}

test "bitmask sparse encoding and decoding" {
    const allocator = std.testing.allocator;
    const original: [16]f32 = .{
        0.0, 1.5, 0.0, 0.0, -2.5, 0.0, 3.5, 0.0,
        0.0, 0.0, -4.5, 0.0, 0.0, 5.5, 0.0, 0.0,
    };

    const encoded = try hk.sparsity.encodeBitmaskF32(&original, allocator);
    defer allocator.free(encoded);

    var decoded: [16]f32 = undefined;
    try hk.sparsity.decodeBitmaskF32(encoded, 16, &decoded);

    for (0..16) |i| {
        try std.testing.expectEqual(original[i], decoded[i]);
    }
}

test "structured 2_4 sparse encoding and decoding" {
    const allocator = std.testing.allocator;
    const original: [8]f32 = .{
        1.5, 0.0, 2.5, 0.0,
        0.0, -3.5, 0.0, 4.5,
    };

    const encoded = try hk.sparsity.encodeStructured2_4_F32(&original, allocator);
    defer allocator.free(encoded);

    var decoded: [8]f32 = undefined;
    try hk.sparsity.decodeStructured2_4_F32(encoded, 8, &decoded);

    for (0..8) |i| {
        try std.testing.expectEqual(original[i], decoded[i]);
    }
}

test "tiling 16x16 pack and unpack roundtrip" {
    const allocator = std.testing.allocator;
    const M: usize = 16;
    const K: usize = 16;
    var original: [256]f32 = undefined;
    for (0..256) |i| {
        original[i] = @floatFromInt(i);
    }

    const tiled = try hk.tiling.packTilesF32(&original, M, K, .tile_16x16, allocator);
    defer allocator.free(tiled);

    var unpacked: [256]f32 = undefined;
    try hk.tiling.unpackTilesF32(tiled, M, K, .tile_16x16, &unpacked);

    for (0..256) |i| {
        try std.testing.expectEqual(original[i], unpacked[i]);
    }
}

test "full writer and reader roundtrip with advanced features" {
    const allocator = std.testing.allocator;
    const test_path = "test_model_advanced.hk";
    const io = std.Options.debug_io;
    const cwd = std.Io.Dir.cwd();
    defer cwd.deleteFile(io, test_path) catch {};

    var writer = hk.HKWriter.init(allocator);
    defer writer.deinit();

    try writer.addMetadataString("architecture", "vision_transformer");
    try writer.addMetadataInt("num_layers", 12);
    try writer.addMetadataFloat("pruning_ratio", 0.50);

    // 1. Dense F32 Tensor
    const weights1: [128]f32 = @as([128]f32, @splat(1.5));
    const weights1_bytes = std.mem.sliceAsBytes(&weights1);
    try writer.addTensor(.{
        .name = "encoder.layer0.attention.q_proj.weight",
        .storage_type = .f32,
        .tile_layout = .tile_16x16,
        .sparsity_type = .none,
        .ndim = 2,
        .shape = .{ 16, 8, 0, 0, 0, 0, 0, 0 },
        .data = weights1_bytes,
        .sparsity_ratio = 0.0,
    });

    // 2. 2:4 Structured Sparse Tensor
    const sparse_24: [8]f32 = .{ 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0 };
    const encoded_24 = try hk.sparsity.encodeStructured2_4_F32(&sparse_24, allocator);
    defer allocator.free(encoded_24);

    try writer.addTensor(.{
        .name = "encoder.layer0.mlp.dense.weight",
        .storage_type = .sparse_2_4,
        .tile_layout = .row_major,
        .sparsity_type = .structured_2_4,
        .ndim = 2,
        .shape = .{ 2, 4, 0, 0, 0, 0, 0, 0 },
        .data = encoded_24,
        .sparsity_ratio = 0.5,
    });

    // 3. Null Ref Tensor (pruned layer, 0 payload bytes)
    try writer.addTensor(.{
        .name = "encoder.layer0.pruned_head.weight",
        .storage_type = .null_ref,
        .tile_layout = .row_major,
        .sparsity_type = .physical_pruned,
        .ndim = 2,
        .shape = .{ 16, 16, 0, 0, 0, 0, 0, 0 },
        .data = &[_]u8{},
        .sparsity_ratio = 1.0,
    });

    // 4. Shared Ref Tensor (tied weights, refers to index 0)
    try writer.addTensor(.{
        .name = "encoder.layer0.attention.k_proj.weight",
        .storage_type = .shared_ref,
        .tile_layout = .tile_16x16,
        .sparsity_type = .none,
        .ndim = 2,
        .shape = .{ 16, 8, 0, 0, 0, 0, 0, 0 },
        .data = &[_]u8{},
        .shared_target_index = 0,
    });

    try writer.writeToFile(test_path);

    // Read back and verify
    var reader = try hk.HKReader.open(test_path, allocator);
    defer reader.deinit();

    try std.testing.expectEqual(4, reader.header.tensor_count);
    try std.testing.expect(reader.header.isValid());
    try std.testing.expectEqual(0, reader.header.tensor_data_offset % hk.format.ALIGNMENT_BYTES);

    // Verify 1: Dense F32
    const t0 = reader.toc.find("encoder.layer0.attention.q_proj.weight").?;
    try std.testing.expectEqual(0, t0.data_offset % hk.format.ALIGNMENT_BYTES);
    try std.testing.expectEqual(weights1_bytes.len, t0.data_size);

    // Verify 2: 2:4 Structured Sparse
    const t1 = reader.toc.find("encoder.layer0.mlp.dense.weight").?;
    var out_24: [8]f32 = undefined;
    try reader.dequantizeToF32(t1, false, &out_24);
    for (0..8) |i| {
        try std.testing.expectEqual(sparse_24[i], out_24[i]);
    }

    // Verify 3: Null Ref
    const t2 = reader.toc.find("encoder.layer0.pruned_head.weight").?;
    var out_null: [256]f32 = undefined;
    try reader.dequantizeToF32(t2, false, &out_null);
    for (0..256) |i| {
        try std.testing.expectEqual(@as(f32, 0.0), out_null[i]);
    }

    // Verify 4: Shared Ref
    const t3 = reader.toc.find("encoder.layer0.attention.k_proj.weight").?;
    try std.testing.expectEqual(t0.data_offset, t3.data_offset);
    try std.testing.expectEqual(t0.data_size, t3.data_size);
}

test "Appendix Region Append, Read, Lineage & Rollback" {
    const allocator = std.testing.allocator;
    const test_path = "test_appendix_model.hk";
    const io = std.Options.debug_io;
    const cwd = std.Io.Dir.cwd();
    defer cwd.deleteFile(io, test_path) catch {};

    // 1. Create a minimal base HK file
    var writer = hk.HKWriter.init(allocator);
    defer writer.deinit();

    const dummy_weights = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const dummy_bytes = std.mem.sliceAsBytes(&dummy_weights);
    try writer.addTensor(.{
        .name = "model.base.weight",
        .storage_type = .f32,
        .tile_layout = .row_major,
        .sparsity_type = .none,
        .ndim = 1,
        .shape = [_]u64{ 4, 0, 0, 0, 0, 0, 0, 0 },
        .data = dummy_bytes,
        .block_size = 32,
        .sparsity_ratio = 0.0,
    });
    try writer.writeToFile(test_path);

    // 2. Append Generation 1 (LoRA adapter)
    const lora_payload = "LORA_A_AND_B_ADAPTER_WEIGHTS_GEN1";
    var hasher1 = std.crypto.hash.sha2.Sha256.init(.{});
    hasher1.update(lora_payload);
    var gen1_hash: [32]u8 = undefined;
    hasher1.final(&gen1_hash);

    try hk.appendix.appendRecordToFile(allocator, test_path, .{
        .entry_type = .lora_adapter,
        .flags = hk.format.AppendixFlags.ACTIVE,
        .name = "adapter.generation1.lora",
        .target = "model.base.weight",
        .generation = 1,
        .timestamp = 1726000000,
        .parent_hash = @as([32]u8, @splat(0)),
        .metrics = .{ .loss = 0.12, .accuracy = 0.985, .pass_rate = 0.0, .custom = 0.0 },
        .data = lora_payload,
    });

    // 3. Append Generation 2 (Code Eval & Self-Play)
    const eval_payload = "{\"sample_id\": 42, \"pass_rate\": 1.0, \"tests\": 10}";
    try hk.appendix.appendRecordToFile(allocator, test_path, .{
        .entry_type = .code_eval,
        .flags = hk.format.AppendixFlags.ACTIVE,
        .name = "eval.generation2.execution",
        .target = "coding.solver",
        .generation = 2,
        .timestamp = 1726000100,
        .parent_hash = gen1_hash, // Chained from Gen 1
        .metrics = .{ .loss = 0.05, .accuracy = 0.992, .pass_rate = 1.0, .custom = 0.0 },
        .data = eval_payload,
    });

    // 4. Read back and verify via AppendixReader
    {
        var file_region = try hk.platform.mapOrReadFile(test_path, allocator);
        defer file_region.deinit(allocator);

        var app_reader = try hk.appendix.AppendixReader.init(allocator, file_region.bytes);
        defer app_reader.deinit();

        try std.testing.expectEqual(2, app_reader.records.items.len);
        try std.testing.expectEqual(hk.format.AppendixEntryType.lora_adapter, app_reader.records.items[0].entry_type);
        try std.testing.expectEqualStrings("adapter.generation1.lora", app_reader.records.items[0].name);
        try std.testing.expectEqual(1, app_reader.records.items[0].generation);
        try std.testing.expectEqualStrings(lora_payload, app_reader.records.items[0].data);

        try std.testing.expectEqual(hk.format.AppendixEntryType.code_eval, app_reader.records.items[1].entry_type);
        try std.testing.expectEqualStrings("eval.generation2.execution", app_reader.records.items[1].name);
        try std.testing.expectEqual(2, app_reader.records.items[1].generation);
        try std.testing.expectEqual(@as(f32, 1.0), app_reader.records.items[1].metrics.pass_rate);

        // Verify cryptographic SHA-256 lineage
        try std.testing.expect(app_reader.verifyLineage());
    }

    // 5. Test Rollback to Generation 1
    try hk.appendix.rollbackToFile(allocator, test_path, 1);

    {
        var rollback_region = try hk.platform.mapOrReadFile(test_path, allocator);
        defer rollback_region.deinit(allocator);

        var app_reader_rb1 = try hk.appendix.AppendixReader.init(allocator, rollback_region.bytes);
        defer app_reader_rb1.deinit();

        try std.testing.expectEqual(1, app_reader_rb1.records.items.len);
        try std.testing.expectEqual(1, app_reader_rb1.records.items[0].generation);
    }

    // 6. Test Rollback to Generation 0 (clears appendix)
    try hk.appendix.rollbackToFile(allocator, test_path, 0);

    {
        var reset_region = try hk.platform.mapOrReadFile(test_path, allocator);
        defer reset_region.deinit(allocator);

        var app_reader_rb0 = try hk.appendix.AppendixReader.init(allocator, reset_region.bytes);
        defer app_reader_rb0.deinit();

        try std.testing.expectEqual(0, app_reader_rb0.records.items.len);
    }
}

test "flexible alignment writing and reading (alignment = 1, 16, 64)" {
    const allocator = std.testing.allocator;
    const io = std.Options.debug_io;
    const cwd = std.Io.Dir.cwd();

    const alignments = [_]usize{ 1, 4, 16, 64 };
    for (alignments) |align_val| {
        var path_buf: [128]u8 = undefined;
        const test_path = try std.fmt.bufPrint(&path_buf, "test_align_{d}.hk", .{align_val});
        defer cwd.deleteFile(io, test_path) catch {};

        var writer = hk.HKWriter.init(allocator);
        defer writer.deinit();
        writer.setAlignment(align_val);

        try writer.addMetadataString("target_arch", "embedded_rv32");
        try writer.addMetadataInt("align_val", @intCast(align_val));

        const dummy_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        try writer.addTensor(.{
            .name = "embed.weight",
            .storage_type = .f32,
            .tile_layout = .row_major,
            .sparsity_type = .none,
            .ndim = 2,
            .shape = .{ 2, 4, 0, 0, 0, 0, 0, 0 },
            .data = std.mem.sliceAsBytes(&dummy_data),
        });

        try writer.writeToFile(test_path);

        // Read back
        var reader = try hk.HKReader.open(test_path, allocator);
        defer reader.deinit();

        try std.testing.expectEqual(@as(u16, @intCast(align_val)), reader.header.alignment);
        if (align_val < 128) {
            try std.testing.expect((reader.header.flags & hk.format.HeaderFlags.FLEXIBLE_ALIGNMENT) != 0);
        }

        const tensor_opt = reader.toc.find("embed.weight");
        try std.testing.expect(tensor_opt != null);
        const tensor = tensor_opt.?;
        const raw_bytes = try reader.getTensorData(tensor);
        const read_slice = std.mem.bytesAsSlice(f32, raw_bytes);
        for (0..dummy_data.len) |i| {
            try std.testing.expectEqual(dummy_data[i], read_slice[i]);
        }
    }
}

test "simd tensor ops: dot product and gemv" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    const b = [_]f32{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };

    const dot = hk.tensor_ops.dotProductF32(&a, &b);
    // sum(1..10) * 0.5 = 55 * 0.5 = 27.5
    try std.testing.expectApproxEqAbs(@as(f32, 27.5), dot, 1e-5);

    // GEMV: 2x4 matrix * 4-vector
    const w = [_]f32{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    };
    const x = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const bias = [_]f32{ 10.0, 20.0 };
    var y: [2]f32 = undefined;

    hk.tensor_ops.gemvF32(&w, &x, &bias, &y, 2, 4);
    // row 0: 1+2+3+4 + 10 = 20
    // row 1: 5+6+7+8 + 20 = 46
    try std.testing.expectEqual(@as(f32, 20.0), y[0]);
    try std.testing.expectEqual(@as(f32, 46.0), y[1]);
}

test "simd gemv q8_0" {
    var blocks: [4]hk.quantization.BlockQ8_0 = undefined;
    for (&blocks) |*b| {
        b.d = 1.0;
        @memset(&b.qs, 1);
    }
    const bytes = std.mem.sliceAsBytes(&blocks);
    var x: [64]f32 = @as([64]f32, @splat(1.0));
    var y: [2]f32 = undefined;
    hk.tensor_ops.gemvQ8_0(bytes, &x, null, &y, 2, 64);
    try std.testing.expectEqual(@as(f32, 64.0), y[0]);
    try std.testing.expectEqual(@as(f32, 64.0), y[1]);
}

test "simd gemv q4_0" {
    var blocks: [4]hk.quantization.BlockQ4_0 = undefined;
    for (&blocks) |*b| {
        b.d = 1.0;
        @memset(&b.qs, 0x88); // 8 is 0 in signed nibble
    }
    const bytes = std.mem.sliceAsBytes(&blocks);
    var x: [64]f32 = @as([64]f32, @splat(1.0));
    var y: [2]f32 = undefined;
    hk.tensor_ops.gemvQ4_0(bytes, &x, null, &y, 2, 64);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[1], 1e-4);
}

test "simd gemv q4_k" {
    var blocks: [2]hk.quantization.BlockQ4_K = undefined; // M=2, K=256
    for (&blocks) |*b| {
        b.d = 1.0;
        b.dmin = 0.0;
        @memset(&b.scales, 0);
        @memset(&b.qs, 0);
    }
    const bytes = std.mem.sliceAsBytes(&blocks);
    var x: [256]f32 = @as([256]f32, @splat(1.0));
    var y: [2]f32 = undefined;
    hk.tensor_ops.gemvQ4_K(bytes, &x, null, &y, 2, 256);
}



test "net2wider function preservation" {
    // Layer 1: [2, 3], Layer 2: [2, 2]
    // Input x: [3]
    const w1_old = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const b1_old = [_]f32{ 0.1, 0.2 };

    const w2_old = [_]f32{
        0.5, -0.5,
        1.0,  0.5,
    };
    const b2 = [_]f32{ 0.0, 0.0 };

    const x = [_]f32{ 1.0, 0.5, -1.0 };

    // Baseline forward pass:
    // h1[0] = 1*1 + 2*0.5 + 3*(-1) + 0.1 = 1 + 1 - 3 + 0.1 = -0.9
    // h1[1] = 4*1 + 5*0.5 + 6*(-1) + 0.2 = 4 + 2.5 - 6 + 0.2 = 0.7
    var h1_orig: [2]f32 = undefined;
    hk.tensor_ops.gemvF32(&w1_old, &x, &b1_old, &h1_orig, 2, 3);

    var y_orig: [2]f32 = undefined;
    hk.tensor_ops.gemvF32(&w2_old, &h1_orig, &b2, &y_orig, 2, 2);

    // Expand Layer 1 from 2 units to 4 units
    var w1_new: [4 * 3]f32 = undefined;
    var b1_new: [4]f32 = undefined;
    var w2_new: [2 * 4]f32 = undefined;

    try hk.growth.net2Wider(
        &w1_old,
        &b1_old,
        &w1_new,
        &b1_new,
        &w2_old,
        &w2_new,
        2,
        4,
        3,
        2,
        0.0, // zero noise for exact mathematical identity test
        42,
    );

    // Forward pass through wider network
    var h1_new: [4]f32 = undefined;
    hk.tensor_ops.gemvF32(&w1_new, &x, &b1_new, &h1_new, 4, 3);

    var y_new: [2]f32 = undefined;
    hk.tensor_ops.gemvF32(&w2_new, &h1_new, &b2, &y_new, 2, 4);

    // Check exact equivalence: y_orig == y_new
    try std.testing.expectApproxEqAbs(y_orig[0], y_new[0], 1e-5);
    try std.testing.expectApproxEqAbs(y_orig[1], y_new[1], 1e-5);
}

test "net2deeper identity initialization" {
    var w_new: [3 * 3]f32 = undefined;
    var b_new: [3]f32 = undefined;

    hk.growth.net2Deeper(&w_new, &b_new, 3);

    // Verify w_new is identity matrix: 1 on diagonal, 0 elsewhere
    for (0..3) |r| {
        for (0..3) |c| {
            const expected: f32 = if (r == c) 1.0 else 0.0;
            try std.testing.expectEqual(expected, w_new[r * 3 + c]);
        }
        try std.testing.expectEqual(@as(f32, 0.0), b_new[r]);
    }
}

test "Appendix cryptographic SHA-256 tamper detection" {
    const allocator = std.testing.allocator;
    const test_path = "test_tamper_appendix.hk";
    const io = std.Options.debug_io;
    const cwd = std.Io.Dir.cwd();
    defer cwd.deleteFile(io, test_path) catch {};

    var writer = hk.HKWriter.init(allocator);
    defer writer.deinit();

    const dummy_weights = [_]f32{ 1.0, 2.0 };
    const dummy_bytes = std.mem.sliceAsBytes(&dummy_weights);
    try writer.addTensor(.{
        .name = "model.base.weight",
        .storage_type = .f32,
        .tile_layout = .row_major,
        .sparsity_type = .none,
        .ndim = 1,
        .shape = [_]u64{ 2, 0, 0, 0, 0, 0, 0, 0 },
        .data = dummy_bytes,
        .block_size = 32,
        .sparsity_ratio = 0.0,
    });
    try writer.writeToFile(test_path);

    const p1 = "VALID_LORA_PAYLOAD_GEN1";
    var hasher1 = std.crypto.hash.sha2.Sha256.init(.{});
    hasher1.update(p1);
    var h1: [32]u8 = undefined;
    hasher1.final(&h1);

    try hk.appendix.appendRecordToFile(allocator, test_path, .{
        .entry_type = .lora_adapter,
        .flags = hk.format.AppendixFlags.ACTIVE,
        .name = "adapter.gen1",
        .target = "fc1",
        .generation = 1,
        .timestamp = 1000,
        .parent_hash = @as([32]u8, @splat(0)),
        .metrics = .{ .loss = 0.1, .accuracy = 0.90, .pass_rate = 1.0, .custom = 0.0 },
        .data = p1,
    });

    const p2 = "VALID_LORA_PAYLOAD_GEN2";
    var tampered_hash = h1;
    tampered_hash[0] ^= 0xFF; // Invert first byte

    try hk.appendix.appendRecordToFile(allocator, test_path, .{
        .entry_type = .lora_adapter,
        .flags = hk.format.AppendixFlags.ACTIVE,
        .name = "adapter.gen2",
        .target = "fc1",
        .generation = 2,
        .timestamp = 2000,
        .parent_hash = tampered_hash,
        .metrics = .{ .loss = 0.05, .accuracy = 0.95, .pass_rate = 1.0, .custom = 0.0 },
        .data = p2,
    });

    var file_region = try hk.platform.mapOrReadFile(test_path, allocator);
    defer file_region.deinit(allocator);

    var app_reader = try hk.appendix.AppendixReader.init(allocator, file_region.bytes);
    defer app_reader.deinit();

    try std.testing.expectEqual(2, app_reader.records.items.len);
    try std.testing.expect(!app_reader.verifyLineage());
}

test "Appendix all six entry types roundtrip" {
    const allocator = std.testing.allocator;
    const test_path = "test_all_types_appendix.hk";
    const io = std.Options.debug_io;
    const cwd = std.Io.Dir.cwd();
    defer cwd.deleteFile(io, test_path) catch {};

    var writer = hk.HKWriter.init(allocator);
    defer writer.deinit();

    const dummy_weights = [_]f32{ 0.5, -0.5 };
    const dummy_bytes = std.mem.sliceAsBytes(&dummy_weights);
    try writer.addTensor(.{
        .name = "base.weight",
        .storage_type = .f32,
        .tile_layout = .row_major,
        .sparsity_type = .none,
        .ndim = 1,
        .shape = [_]u64{ 2, 0, 0, 0, 0, 0, 0, 0 },
        .data = dummy_bytes,
        .block_size = 32,
        .sparsity_ratio = 0.0,
    });
    try writer.writeToFile(test_path);

    const types = [_]hk.format.AppendixEntryType{
        .lora_adapter,
        .delta_patch,
        .new_layer,
        .code_eval,
        .kv_cache_sink,
        .topology_head,
    };
    const names = [_][]const u8{
        "lora.adapter.01",
        "delta.patch.02",
        "new.layer.03",
        "code.eval.04",
        "kv.cache.sink.05",
        "topology.head.06",
    };

    for (types, names, 0..) |entry_type, name, idx| {
        const payload = try std.fmt.allocPrint(allocator, "PAYLOAD_TYPE_{d}", .{idx});
        defer allocator.free(payload);

        try hk.appendix.appendRecordToFile(allocator, test_path, .{
            .entry_type = entry_type,
            .flags = hk.format.AppendixFlags.ACTIVE,
            .name = name,
            .target = "module",
            .generation = @intCast(idx + 1),
            .timestamp = @intCast(1000 + idx * 100),
            .parent_hash = @as([32]u8, @splat(0)),
            .metrics = .{ .loss = 0.0, .accuracy = 0.0, .pass_rate = 0.0, .custom = 0.0 },
            .data = payload,
        });
    }

    var file_region = try hk.platform.mapOrReadFile(test_path, allocator);
    defer file_region.deinit(allocator);

    var app_reader = try hk.appendix.AppendixReader.init(allocator, file_region.bytes);
    defer app_reader.deinit();

    try std.testing.expectEqual(6, app_reader.records.items.len);
    for (0..6) |i| {
        try std.testing.expectEqual(types[i], app_reader.records.items[i].entry_type);
        try std.testing.expectEqualStrings(names[i], app_reader.records.items[i].name);
        try std.testing.expectEqual(@as(u32, @intCast(i + 1)), app_reader.records.items[i].generation);
    }
}

test "native SwiGLU Net2Wider exact function preservation" {
    const allocator = std.testing.allocator;

    const in_f: usize = 16;
    const old_inter: usize = 32;
    const new_inter: usize = 48; // +50% expansion
    const out_f: usize = 16;

    // Allocate and initialize base weights
    const w_gate_old = try allocator.alloc(f32, old_inter * in_f);
    defer allocator.free(w_gate_old);
    const w_up_old = try allocator.alloc(f32, old_inter * in_f);
    defer allocator.free(w_up_old);
    const w_down_old = try allocator.alloc(f32, out_f * old_inter);
    defer allocator.free(w_down_old);

    for (w_gate_old, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.05 - 0.4;
    for (w_up_old, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.06 - 0.3;
    for (w_down_old, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 19)) * 0.04 - 0.2;

    // Test input
    const x = try allocator.alloc(f32, in_f);
    defer allocator.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1)) * 0.1;

    // Run base SwiGLU forward pass
    const scratch_old = try allocator.alloc(f32, old_inter * 2);
    defer allocator.free(scratch_old);
    const out_old = try allocator.alloc(f32, out_f);
    defer allocator.free(out_old);

    try hk.tensor_ops.forwardSwiGLUF32(
        x,
        w_gate_old,
        null,
        w_up_old,
        null,
        w_down_old,
        null,
        scratch_old,
        out_old,
        in_f,
        old_inter,
        out_f,
    );

    // Allocate expanded weights
    const w_gate_new = try allocator.alloc(f32, new_inter * in_f);
    defer allocator.free(w_gate_new);
    const w_up_new = try allocator.alloc(f32, new_inter * in_f);
    defer allocator.free(w_up_new);
    const w_down_new = try allocator.alloc(f32, out_f * new_inter);
    defer allocator.free(w_down_new);

    // Apply native Net2WiderSwiGLU with zero_init = true
    try hk.growth.net2WiderSwiGLU(
        w_gate_old,
        null,
        w_up_old,
        null,
        w_down_old,
        null,
        w_gate_new,
        null,
        w_up_new,
        null,
        w_down_new,
        null,
        old_inter,
        new_inter,
        in_f,
        out_f,
        true, // zero_init!
        0.0,
        42,
    );

    // Run expanded SwiGLU forward pass
    const scratch_new = try allocator.alloc(f32, new_inter * 2);
    defer allocator.free(scratch_new);
    const out_new = try allocator.alloc(f32, out_f);
    defer allocator.free(out_new);

    try hk.tensor_ops.forwardSwiGLUF32(
        x,
        w_gate_new,
        null,
        w_up_new,
        null,
        w_down_new,
        null,
        scratch_new,
        out_new,
        in_f,
        new_inter,
        out_f,
    );

    // Verify bitwise/numerical exact function preservation
    var max_dev: f32 = 0.0;
    for (0..out_f) |i| {
        const diff = @abs(out_new[i] - out_old[i]);
        if (diff > max_dev) max_dev = diff;
        try std.testing.expectApproxEqAbs(out_old[i], out_new[i], 1e-6);
    }
    try std.testing.expect(max_dev < 1e-6);
}

test "native dynamic vocabulary expansion" {
    const allocator = std.testing.allocator;

    const old_vocab: usize = 20;
    const new_vocab: usize = 30; // +10 tokens
    const hidden_dim: usize = 8;

    const embed_old = try allocator.alloc(f32, old_vocab * hidden_dim);
    defer allocator.free(embed_old);
    for (embed_old, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1;

    const embed_new = try allocator.alloc(f32, new_vocab * hidden_dim);
    defer allocator.free(embed_new);

    try hk.growth.expandVocab(
        embed_old,
        embed_new,
        null,
        null,
        old_vocab,
        new_vocab,
        hidden_dim,
        123,
    );

    // Base tokens 0..old_vocab-1 must match bitwise exactly
    for (0..old_vocab * hidden_dim) |i| {
        try std.testing.expectEqual(embed_old[i], embed_new[i]);
    }

    // New tokens must be populated with non-zero initialized embeddings
    var non_zero = false;
    for (old_vocab * hidden_dim..new_vocab * hidden_dim) |i| {
        if (embed_new[i] != 0.0) non_zero = true;
    }
    try std.testing.expect(non_zero);
}

test "native plasticity isolation masking" {
    const allocator = std.testing.allocator;

    // Test row masking (e.g. embed_tokens, gate_proj)
    const rows: usize = 10;
    const cols: usize = 4;
    const cutoff_rows: usize = 6;

    const grad_matrix = try allocator.alloc(f32, rows * cols);
    defer allocator.free(grad_matrix);
    @memset(grad_matrix, 1.0);

    hk.growth.applyPlasticityMaskRows(grad_matrix, cutoff_rows, cols);

    // Rows 0..cutoff_rows-1 must be exactly 0.0
    for (0..cutoff_rows * cols) |i| {
        try std.testing.expectEqual(@as(f32, 0.0), grad_matrix[i]);
    }
    // Rows cutoff_rows..rows-1 must remain 1.0
    for (cutoff_rows * cols..rows * cols) |i| {
        try std.testing.expectEqual(@as(f32, 1.0), grad_matrix[i]);
    }

    // Test column masking (e.g. down_proj)
    const cutoff_cols: usize = 2;
    @memset(grad_matrix, 1.0);

    hk.growth.applyPlasticityMaskCols(grad_matrix, rows, cutoff_cols, cols);

    for (0..rows) |r| {
        for (0..cols) |c| {
            const val = grad_matrix[r * cols + c];
            if (c < cutoff_cols) {
                try std.testing.expectEqual(@as(f32, 0.0), val);
            } else {
                try std.testing.expectEqual(@as(f32, 1.0), val);
            }
        }
    }
}

test "native SIMD RMSNorm and SiLU activation" {
    const allocator = std.testing.allocator;

    const len: usize = 16;
    const x = try allocator.alloc(f32, len);
    defer allocator.free(x);
    const w = try allocator.alloc(f32, len);
    defer allocator.free(w);
    const out = try allocator.alloc(f32, len);
    defer allocator.free(out);

    for (0..len) |i| {
        x[i] = @as(f32, @floatFromInt(i + 1)) * 0.5;
        w[i] = 1.0;
    }

    // Test RMSNorm
    hk.tensor_ops.rmsNormF32(x, w, 1e-5, out);

    // Compute expected RMS
    var sum_sq: f32 = 0.0;
    for (x) |v| sum_sq += v * v;
    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(len)) + 1e-5);

    for (0..len) |i| {
        const expected = x[i] / rms;
        try std.testing.expectApproxEqAbs(expected, out[i], 1e-5);
    }

    // Test SiLU: f(0.0) = 0.0, f(2.0) = 2.0 / (1 + exp(-2))
    const silu_in = [_]f32{ 0.0, 2.0, -2.0 };
    var silu_out: [3]f32 = undefined;
    hk.tensor_ops.siluF32(&silu_in, &silu_out);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), silu_out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.761594), silu_out[1], 1e-4);
}

test "native expansion evaluator" {
    const diag = hk.expansion.evaluateExpansionNeed(
        4.5, // high loss
        0.1, // 90% error rate
        100,
        10,
        8, // 8 missing tokens
        128,
        2.0,
    );

    try std.testing.expect(diag.needs_expansion);
    try std.testing.expect(diag.needs_vocab_expansion);
    try std.testing.expectEqual(@as(usize, 8), diag.missing_token_count);
    try std.testing.expect(diag.needs_width_expansion);
    try std.testing.expect(diag.suggested_width_ratio >= 1.33);
}

test "native process sandbox execution" {
    const allocator = std.testing.allocator;
    var sandbox = hk.sandbox.NativeSandbox.initWithIo(allocator, std.testing.io, 5000);

    // Run simple command
    const argv = if (@import("builtin").os.tag == .windows)
        &[_][]const u8{ "cmd.exe", "/c", "echo", "NATIVE_ZIG_SANDBOX_OK" }
    else
        &[_][]const u8{ "echo", "NATIVE_ZIG_SANDBOX_OK" };

    var res = try sandbox.execute(argv);
    defer res.deinit(allocator);

    try std.testing.expect(res.success);
    try std.testing.expectEqual(@as(u8, 0), res.exit_code);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "NATIVE_ZIG_SANDBOX_OK") != null);
}

test "file header sharding fields" {
    var header = hk.FileHeader{
        .flags = hk.format.HeaderFlags.LITTLE_ENDIAN | hk.format.HeaderFlags.IS_SHARDED,
        .split_index = 2,
        .split_count = 4,
    };
    try std.testing.expect(header.isValid());
    try std.testing.expectEqual(header.split_index, 2);
    try std.testing.expectEqual(header.split_count, 4);
    try std.testing.expectEqual((header.flags & hk.format.HeaderFlags.IS_SHARDED) != 0, true);
    try std.testing.expectEqual(@sizeOf(hk.FileHeader), 128);
}

test "platform writeBytesAtOffset in-place test" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    const fname = "test_seek_tmp.bin";
    var f = try cwd.createFile(io, fname, .{});
    defer cwd.deleteFile(io, fname) catch {};
    try f.writeStreamingAll(io, "hello world 12345678");
    f.close(io);

    try hk.platform.writeBytesAtOffset(fname, "HELLO", 0, allocator);
    try hk.platform.writeBytesAtOffset(fname, "EARTH", 6, allocator);

    var region = try hk.platform.mapOrReadFile(fname, allocator);
    defer region.deinit(allocator);
    try std.testing.expectEqualStrings("HELLO EARTH 12345678", region.bytes);
}

test "patchFileMetadataInPlace roundtrip test" {
    const allocator = std.testing.allocator;
    const test_path = "test_metadata_patch.hk";
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    defer cwd.deleteFile(io, test_path) catch {};

    // 1. Create a model file
    var writer = hk.HKWriter.init(allocator);
    defer writer.deinit();
    try writer.addMetadataString("arch", "llama");
    try writer.addMetadataInt("layers", 32);

    const dummy_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try writer.addTensor(.{
        .name = "layer.0.weight",
        .storage_type = .f32,
        .ndim = 1,
        .shape = .{ 4, 0, 0, 0, 0, 0, 0, 0 },
        .data = std.mem.sliceAsBytes(&dummy_data),
    });
    try writer.writeToFile(test_path);

    // Get original tensor payload bytes
    var reader1 = try hk.HKReader.open(test_path, allocator);
    const orig_data_offset = reader1.header.tensor_data_offset;
    const t_entry1 = reader1.toc.find("layer.0.weight").?;
    const t_data1 = try allocator.dupe(u8, try reader1.getTensorData(t_entry1));
    defer allocator.free(t_data1);
    reader1.deinit();

    // 2. Patch metadata in-place
    try hk.metadata.patchFileMetadataInPlace(allocator, test_path, "tokenizer.chat_template", "{% for msg in messages %}{{ msg['content'] }}{% endfor %}");
    try hk.metadata.patchFileMetadataInPlace(allocator, test_path, "layers", "36");

    // 3. Re-open and verify metadata updated and tensor payload 100% identical!
    var reader2 = try hk.HKReader.open(test_path, allocator);
    defer reader2.deinit();

    try std.testing.expectEqual(orig_data_offset, reader2.header.tensor_data_offset);
    const t_entry2 = reader2.toc.find("layer.0.weight").?;
    const t_data2 = try reader2.getTensorData(t_entry2);
    try std.testing.expectEqualSlices(u8, t_data1, t_data2);

    const tmpl_val = reader2.metadata_map.get("tokenizer.chat_template").?;
    try std.testing.expectEqualStrings("{% for msg in messages %}{{ msg['content'] }}{% endfor %}", tmpl_val.val_string);

    const layers_val = reader2.metadata_map.get("layers").?;
    try std.testing.expectEqual(@as(i64, 36), layers_val.val_int64);
}

test "patchFileMetadataInPlace growing inside the padding keeps the TOC intact" {
    const allocator = std.testing.allocator;
    const test_path = "test_metadata_patch_grow.hk";
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    defer cwd.deleteFile(io, test_path) catch {};

    var writer = hk.HKWriter.init(allocator);
    defer writer.deinit();
    writer.setAlignment(4096);
    try writer.addMetadataString("name", "a");
    const dummy_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try writer.addTensor(.{
        .name = "w",
        .storage_type = .f32,
        .ndim = 1,
        .shape = .{ 4, 0, 0, 0, 0, 0, 0, 0 },
        .data = std.mem.sliceAsBytes(&dummy_data),
    });
    try writer.writeToFile(test_path);

    // A longer value shifts the TOC to the right; it still fits before the first payload.
    try hk.metadata.patchFileMetadataInPlace(allocator, test_path, "name", "a much longer name");

    var reader = try hk.HKReader.open(test_path, allocator);
    defer reader.deinit();
    try std.testing.expectEqualStrings("a much longer name", reader.metadata_map.get("name").?.val_string);
    const entry = reader.toc.find("w").?;
    try std.testing.expectEqualSlices(f32, &dummy_data, try reader.getRawF32(entry));
}

test "reader decodes GGUF block formats it has no dedicated branch for" {
    const allocator = std.testing.allocator;
    const test_path = "test_reader_q4_1.hk";
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    defer cwd.deleteFile(io, test_path) catch {};

    // One Q4_1 block: scale 1.0, minimum 2.0, every byte 0x21 (low nibble 1, high nibble 2).
    var block: [20]u8 = undefined;
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 1.0)), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(@as(f16, 2.0)), .little);
    @memset(block[4..], 0x21);

    var writer = hk.HKWriter.init(allocator);
    defer writer.deinit();
    try writer.addTensor(.{
        .name = "w",
        .storage_type = .q4_1,
        .ndim = 1,
        .shape = .{ 32, 0, 0, 0, 0, 0, 0, 0 },
        .data = &block,
    });
    try writer.writeToFile(test_path);

    var reader = try hk.HKReader.open(test_path, allocator);
    defer reader.deinit();
    var out: [32]f32 = undefined;
    try reader.dequantizeToF32(reader.toc.find("w").?, false, &out);
    for (out[0..16]) |v| try std.testing.expectEqual(@as(f32, 3.0), v);
    for (out[16..]) |v| try std.testing.expectEqual(@as(f32, 4.0), v);
}

test "k-quants q4_k super-block roundtrip" {
    var weights: [256]f32 = undefined;
    for (0..256) |i| {
        weights[i] = @as(f32, @floatFromInt(i)) / 256.0 - 0.5;
    }

    var block: hk.quantization.BlockQ4_K = .{};
    hk.quantization.quantizeSuperBlockQ4_K(&weights, &block);

    var dequant: [256]f32 = undefined;
    hk.quantization.dequantizeSuperBlockQ4_K(&block, 256, &dequant);

    // Q4_K has 16 quantization levels, check max absolute error is bounded (< 0.08)
    for (0..256) |i| {
        try std.testing.expectApproxEqAbs(weights[i], dequant[i], 0.08);
    }
}

test "k-quants q8_k super-block roundtrip" {
    var weights: [256]f32 = undefined;
    for (0..256) |i| {
        weights[i] = (@as(f32, @floatFromInt(i)) - 128.0) / 128.0;
    }

    var block: hk.quantization.BlockQ8_K = .{};
    hk.quantization.quantizeSuperBlockQ8_K(&weights, &block);

    var dequant: [256]f32 = undefined;
    hk.quantization.dequantizeSuperBlockQ8_K(&block, 256, &dequant);

    // Q8_K has 256 quantization levels, check max absolute error < 0.015
    for (0..256) |i| {
        try std.testing.expectApproxEqAbs(weights[i], dequant[i], 0.015);
    }
}

test "iq4_nl non-linear codebook roundtrip" {
    const vals: [16]f32 = .{
        -0.95, -0.75, -0.60, -0.45, -0.30, -0.20, -0.10, -0.04,
         0.04,  0.10,  0.20,  0.30,  0.45,  0.60,  0.75,  0.95,
    };
    var packed_bytes: [8]u8 = undefined;
    const scale = hk.quantization.quantizeBlockIQ4_NL(&vals, &packed_bytes);
    try std.testing.expect(scale > 0.0);

    var dequant: [16]f32 = undefined;
    hk.quantization.dequantizeBlockIQ4_NL(&packed_bytes, scale, 16, &dequant);

    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(vals[i], dequant[i], 0.1);
    }
}

test "microscaling mxfp4 and nvfp4 dequantization" {
    // 16 elements = 8 packed bytes
    const raw_packed = [_]u8{ 0x21, 0x43, 0x65, 0x87, 0x00, 0x11, 0x22, 0x33 };
    var out_mxfp4: [16]f32 = undefined;
    hk.quantization.dequantizeBlockMXFP4(&raw_packed, 127, 16, &out_mxfp4); // scale 2^(127-127) = 1.0

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out_mxfp4[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_mxfp4[1], 1e-4);

    var out_nvfp4: [16]f32 = undefined;
    hk.quantization.dequantizeBlockNVFP4(&raw_packed, 0x38, 16, &out_nvfp4); // FP8 scale 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out_nvfp4[0], 1e-4);
}

test "native safetensors parser and transcoder roundtrip" {
    const allocator = std.testing.allocator;

    const io = std.Options.debug_io;
    const cwd = std.Io.Dir.cwd();
    const tmp_st = "test_toy.safetensors";
    const tmp_hk = "test_toy_transcoded.hk";
    defer cwd.deleteFile(io, tmp_st) catch {};
    defer cwd.deleteFile(io, tmp_hk) catch {};

    // Create a synthetic .safetensors file
    const json_header = "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}";
    const header_len: u64 = json_header.len;

    const float_vals = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const raw_floats = std.mem.sliceAsBytes(&float_vals);

    var file = try cwd.createFile(io, tmp_st, .{});
    defer file.close(io);

    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, header_len, .little);
    try file.writeStreamingAll(io, &len_bytes);
    try file.writeStreamingAll(io, json_header);
    try file.writeStreamingAll(io, raw_floats);

    // Transcode to HK
    try hk.safetensors.transcodeSafeTensorsToHK(allocator, tmp_st, tmp_hk, .f32);

    // Read generated HK
    var reader = try hk.HKReader.open(tmp_hk, allocator);
    defer reader.deinit();

    const entry = reader.toc.find("weight").?;
    try std.testing.expectEqual(@as(u8, 2), entry.ndim);
    try std.testing.expectEqual(@as(u64, 2), entry.shape[0]);
    try std.testing.expectEqual(@as(u64, 2), entry.shape[1]);

    const data = try reader.getTensorData(entry);
    const floats_out: [*]const f32 = @ptrCast(@alignCast(data.ptr));
    try std.testing.expectEqual(@as(f32, 1.0), floats_out[0]);
    try std.testing.expectEqual(@as(f32, 2.0), floats_out[1]);
    try std.testing.expectEqual(@as(f32, 3.0), floats_out[2]);
    try std.testing.expectEqual(@as(f32, 4.0), floats_out[3]);
}

test "hardware capabilities detection and super-coalesced alignment" {
    const caps = hk.platform.detectHardwareCapabilities();
    // Verify alignment requirements: must be >= 4096 and exact multiple of 128 (NVIDIA Tensor Core requirement)
    try std.testing.expect(caps.optimal_page_alignment >= 4096);
    try std.testing.expect((caps.optimal_page_alignment % 128) == 0);
    try std.testing.expect((caps.dma_hugepage_alignment % 128) == 0);
    try std.testing.expectEqual(caps.dma_hugepage_alignment, 65536);

    var summary_buf: [256]u8 = undefined;
    const summary = caps.getSummary(&summary_buf);
    try std.testing.expect(summary.len > 0);
}

test "raw weight storage roundtrip with universal page alignment" {
    const io = std.Options.debug_io;
    const cwd = std.Io.Dir.cwd();
    const allocator = std.testing.allocator;
    const tmp_path = "test_raw_weights_universal.hk";
    defer cwd.deleteFile(io, tmp_path) catch {};

    var writer = hk.HKWriter.init(allocator);
    defer writer.deinit();

    // Configure for universal page alignment (4096) and raw weight storage
    writer.setAlignment(hk.format.UNIVERSAL_PAGE_ALIGNMENT_BYTES);
    writer.setRawWeightStorage(true);

    const f32_data = [_]f32{ 1.0, 2.5, -3.25, 4.125 };
    try writer.addTensor(.{
        .name = "raw_layer.f32",
        .storage_type = .f32,
        .ndim = 1,
        .shape = .{ 4, 0, 0, 0, 0, 0, 0, 0 },
        .data = std.mem.sliceAsBytes(&f32_data),
    });

    // BF16 data (represented as u16 raw bits)
    // 1.0 in BF16 is 0x3F80, 2.0 is 0x4000
    const bf16_data = [_]u16{ 0x3F80, 0x4000, 0xC000, 0x3E80 };
    try writer.addTensor(.{
        .name = "raw_layer.bf16",
        .storage_type = .bf16,
        .ndim = 1,
        .shape = .{ 4, 0, 0, 0, 0, 0, 0, 0 },
        .data = std.mem.sliceAsBytes(&bf16_data),
    });

    const i8_data = [_]i8{ 10, -20, 30, -40 };
    try writer.addTensor(.{
        .name = "raw_layer.int8",
        .storage_type = .int8,
        .ndim = 1,
        .shape = .{ 4, 0, 0, 0, 0, 0, 0, 0 },
        .data = std.mem.sliceAsBytes(&i8_data),
    });

    try writer.writeToFile(tmp_path);

    var reader = try hk.HKReader.open(tmp_path, allocator);
    defer reader.deinit();

    // Verify format properties
    try std.testing.expect(reader.isRawWeightStorage());
    try std.testing.expect(reader.isUniversalPageAligned());
    try std.testing.expect(reader.isTensorCoreAligned()); // NVIDIA Tensor Core 128-byte coalescing
    try std.testing.expectEqual(hk.format.UNIVERSAL_PAGE_ALIGNMENT_BYTES, reader.getAlignment());

    // Verify zero-copy raw typed retrieval
    const e_f32 = reader.toc.find("raw_layer.f32").?;
    const read_f32 = try reader.getRawF32(e_f32);
    try std.testing.expectEqual(4, read_f32.len);
    try std.testing.expectEqual(1.0, read_f32[0]);
    try std.testing.expectEqual(2.5, read_f32[1]);

    const e_bf16 = reader.toc.find("raw_layer.bf16").?;
    const read_bf16 = try reader.getRawBF16(e_bf16);
    try std.testing.expectEqual(4, read_bf16.len);
    try std.testing.expectEqual(@as(u16, 0x3F80), read_bf16[0]);
    // Check direct conversion to f32
    try std.testing.expectEqual(@as(f32, 1.0), hk.tensor_ops.bf16ToF32(read_bf16[0]));
    try std.testing.expectEqual(@as(f32, 2.0), hk.tensor_ops.bf16ToF32(read_bf16[1]));

    const e_i8 = reader.toc.find("raw_layer.int8").?;
    const read_i8 = try reader.getRawInt8(e_i8);
    try std.testing.expectEqual(4, read_i8.len);
    try std.testing.expectEqual(@as(i8, 10), read_i8[0]);
    try std.testing.expectEqual(@as(i8, -20), read_i8[1]);
}

test "raw weight storage split mode sharding" {
    const io = std.Options.debug_io;
    const cwd = std.Io.Dir.cwd();
    const allocator = std.testing.allocator;
    const shard0_path = "test_raw_shard0.hk";
    const shard1_path = "test_raw_shard1.hk";
    defer cwd.deleteFile(io, shard0_path) catch {};
    defer cwd.deleteFile(io, shard1_path) catch {};

    // Shard 0
    {
        var w0 = hk.HKWriter.init(allocator);
        defer w0.deinit();
        w0.setAlignment(hk.format.UNIVERSAL_PAGE_ALIGNMENT_BYTES);
        w0.setRawWeightStorage(true);
        w0.setSharding(0, 2);

        const d0 = [_]f32{ 1.0, 2.0 };
        try w0.addTensor(.{
            .name = "part0.weight",
            .storage_type = .f32,
            .ndim = 1,
            .shape = .{ 2, 0, 0, 0, 0, 0, 0, 0 },
            .data = std.mem.sliceAsBytes(&d0),
        });
        try w0.writeToFile(shard0_path);
    }

    // Shard 1
    {
        var w1 = hk.HKWriter.init(allocator);
        defer w1.deinit();
        w1.setAlignment(hk.format.UNIVERSAL_PAGE_ALIGNMENT_BYTES);
        w1.setRawWeightStorage(true);
        w1.setSharding(1, 2);

        const d1 = [_]f32{ 3.0, 4.0 };
        try w1.addTensor(.{
            .name = "part1.weight",
            .storage_type = .f32,
            .ndim = 1,
            .shape = .{ 2, 0, 0, 0, 0, 0, 0, 0 },
            .data = std.mem.sliceAsBytes(&d1),
        });
        try w1.writeToFile(shard1_path);
    }

    // Read and verify shards
    var r0 = try hk.HKReader.open(shard0_path, allocator);
    defer r0.deinit();
    try std.testing.expect(r0.isSharded());
    try std.testing.expect(r0.isRawWeightStorage());
    try std.testing.expect(r0.isTensorCoreAligned());
    try std.testing.expectEqual(@as(u16, 0), r0.getSplitIndex());
    try std.testing.expectEqual(@as(u16, 2), r0.getSplitCount());

    var r1 = try hk.HKReader.open(shard1_path, allocator);
    defer r1.deinit();
    try std.testing.expect(r1.isSharded());
    try std.testing.expect(r1.isRawWeightStorage());
    try std.testing.expect(r1.isTensorCoreAligned());
    try std.testing.expectEqual(@as(u16, 1), r1.getSplitIndex());
    try std.testing.expectEqual(@as(u16, 2), r1.getSplitCount());
}

test "raw BF16 and F16 GEMV mathematical correctness" {
    // 2x2 matrix:
    // [ 1.0, 2.0 ]
    // [ 3.0, 4.0 ]
    // BF16 bit patterns:
    // 1.0 = 0x3F80, 2.0 = 0x4000, 3.0 = 0x4040, 4.0 = 0x4080
    const w_bf16 = [_]u16{ 0x3F80, 0x4000, 0x4040, 0x4080 };
    const x = [_]f32{ 0.5, -1.0 };
    var y: [2]f32 = undefined;

    // y[0] = 1.0*0.5 + 2.0*(-1.0) = 0.5 - 2.0 = -1.5
    // y[1] = 3.0*0.5 + 4.0*(-1.0) = 1.5 - 4.0 = -2.5
    hk.tensor_ops.gemvBF16(&w_bf16, &x, null, &y, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), y[1], 1e-5);

    // FP16 test
    const w_f16 = [_]f16{ 1.0, 2.0, 3.0, 4.0 };
    hk.tensor_ops.gemvF16(&w_f16, &x, null, &y, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), y[1], 1e-5);
}





