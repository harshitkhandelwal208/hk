const std = @import("std");
const format = @import("format.zig");
const metadata = @import("metadata.zig");
const platform = @import("platform.zig");
const writer = @import("writer.zig");
const reader = @import("reader.zig");
const buf = @import("buf.zig");

pub const GGUF_MAGIC: u32 = 0x46554747; // 'G' 'G' 'U' 'F' in little-endian

pub const GGUFValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
    _,
};

pub const GGMLType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    q4_0_4_4 = 31,
    q4_0_4_8 = 32,
    q4_0_8_8 = 33,
    tq1_0 = 34,
    tq2_0 = 35,
    _,
};

pub fn ggmlTypeToStorageType(gt: GGMLType) ?format.StorageType {
    return switch (gt) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q5_0 => .q5_0,
        .q5_1 => .q5_1,
        .q8_0 => .q8_0,
        .q8_1 => .q8_1,
        .q2_k => .q2_k,
        .q3_k => .q3_k,
        .q4_k => .q4_k,
        .q5_k => .q5_k,
        .q6_k => .q6_k,
        .q8_k => .q8_k,
        .iq1_s => .iq1_s,
        .iq1_m => .iq1_m,
        .iq2_xxs => .iq2_xxs,
        .iq2_xs => .iq2_xs,
        .iq3_xxs => .iq3_xxs,
        .iq4_nl => .iq4_nl,
        .iq4_xs => .iq4_xs,
        .tq1_0 => .tq1_0,
        .tq2_0 => .tq2_0,
        .i8 => .int8,
        .i32 => .int32,
        .i64 => .int64,
        else => null,
    };
}

pub fn storageTypeToGGMLType(st: format.StorageType) ?GGMLType {
    return switch (st) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q5_0 => .q5_0,
        .q5_1 => .q5_1,
        .q8_0 => .q8_0,
        .q8_1 => .q8_1,
        .q2_k => .q2_k,
        .q3_k => .q3_k,
        .q4_k => .q4_k,
        .q5_k => .q5_k,
        .q6_k => .q6_k,
        .q8_k => .q8_k,
        .iq1_s => .iq1_s,
        .iq1_m => .iq1_m,
        .iq2_xxs => .iq2_xxs,
        .iq2_xs => .iq2_xs,
        .iq3_xxs => .iq3_xxs,
        .iq4_nl => .iq4_nl,
        .iq4_xs => .iq4_xs,
        .tq1_0 => .tq1_0,
        .tq2_0 => .tq2_0,
        .int8 => .i8,
        .int32 => .i32,
        .int64 => .i64,
        else => null,
    };
}

pub fn getGGMLTypeInfo(gt: GGMLType) struct { block_size: usize, type_size: usize } {
    return switch (gt) {
        .f32, .i32 => .{ .block_size = 1, .type_size = 4 },
        .f16, .bf16, .i16 => .{ .block_size = 1, .type_size = 2 },
        .i8 => .{ .block_size = 1, .type_size = 1 },
        .f64, .i64 => .{ .block_size = 1, .type_size = 8 },
        .q4_0 => .{ .block_size = 32, .type_size = 18 },
        .q4_1 => .{ .block_size = 32, .type_size = 20 },
        .q5_0 => .{ .block_size = 32, .type_size = 22 },
        .q5_1 => .{ .block_size = 32, .type_size = 24 },
        .q8_0 => .{ .block_size = 32, .type_size = 34 },
        .q8_1 => .{ .block_size = 32, .type_size = 36 },
        .q2_k => .{ .block_size = 256, .type_size = 84 },
        .q3_k => .{ .block_size = 256, .type_size = 110 },
        .q4_k => .{ .block_size = 256, .type_size = 144 },
        .q5_k => .{ .block_size = 256, .type_size = 176 },
        .q6_k => .{ .block_size = 256, .type_size = 210 },
        .q8_k => .{ .block_size = 256, .type_size = 292 },
        .iq1_s => .{ .block_size = 256, .type_size = 52 },
        .iq1_m => .{ .block_size = 256, .type_size = 56 },
        .iq2_xxs => .{ .block_size = 256, .type_size = 66 },
        .iq2_xs => .{ .block_size = 256, .type_size = 74 },
        .iq3_xxs => .{ .block_size = 256, .type_size = 98 },
        .iq4_nl => .{ .block_size = 32, .type_size = 18 },
        .iq4_xs => .{ .block_size = 256, .type_size = 136 },
        .tq1_0 => .{ .block_size = 256, .type_size = 48 },
        .tq2_0 => .{ .block_size = 256, .type_size = 80 },
        else => .{ .block_size = 1, .type_size = 4 },
    };
}

pub fn calculateGGMLTensorSize(gt: GGMLType, dims: []const u64) usize {
    if (dims.len == 0) return 0;
    var n_elements: usize = 1;
    for (dims) |d| {
        n_elements *= @intCast(d);
    }
    const info = getGGMLTypeInfo(gt);
    const n_blocks = (n_elements + info.block_size - 1) / info.block_size;
    return n_blocks * info.type_size;
}

pub const GGUFTensorInfo = struct {
    name: []const u8,
    n_dimensions: u32,
    dimensions: [format.MAX_DIMS]u64,
    ggml_type: GGMLType,
    offset: u64,
    size: usize,
};

pub const GGUFReader = struct {
    allocator: std.mem.Allocator,
    bytes: []const u8,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    alignment: usize = 32,
    tensor_data_offset: usize = 0,
    tensors: std.ArrayList(GGUFTensorInfo),
    metadata: metadata.MetadataMap,

    pub fn init(allocator: std.mem.Allocator, bytes: []const u8) !GGUFReader {
        if (bytes.len < 16) return error.FileTooSmall;

        const magic = std.mem.readInt(u32, bytes[0..4], .little);
        if (magic != GGUF_MAGIC) return error.InvalidMagic;

        const version = std.mem.readInt(u32, bytes[4..8], .little);
        if (version < 1 or version > 3) return error.UnsupportedGGUFVersion;

        var cursor: usize = 8;
        var tensor_count: u64 = 0;
        var kv_count: u64 = 0;

        if (version == 1) {
            tensor_count = std.mem.readInt(u32, bytes[cursor..][0..4], .little);
            cursor += 4;
            kv_count = std.mem.readInt(u32, bytes[cursor..][0..4], .little);
            cursor += 4;
        } else {
            tensor_count = std.mem.readInt(u64, bytes[cursor..][0..8], .little);
            cursor += 8;
            kv_count = std.mem.readInt(u64, bytes[cursor..][0..8], .little);
            cursor += 8;
        }

        var meta_map = metadata.MetadataMap.init(allocator);
        var alignment: usize = 32;

        // Parse Metadata KV Pairs
        for (0..kv_count) |_| {
            const key = try readGGUFString(bytes, &cursor, version);
            if (cursor + 4 > bytes.len) return error.UnexpectedEof;
            const val_type_raw = std.mem.readInt(u32, bytes[cursor..][0..4], .little);
            cursor += 4;
            const val_type: GGUFValueType = @enumFromInt(val_type_raw);

            switch (val_type) {
                .uint8 => {
                    if (cursor + 1 > bytes.len) return error.UnexpectedEof;
                    const v = bytes[cursor];
                    cursor += 1;
                    try meta_map.setInt(key, @intCast(v));
                },
                .int8 => {
                    if (cursor + 1 > bytes.len) return error.UnexpectedEof;
                    const v: i8 = @bitCast(bytes[cursor]);
                    cursor += 1;
                    try meta_map.setInt(key, @intCast(v));
                },
                .uint16 => {
                    if (cursor + 2 > bytes.len) return error.UnexpectedEof;
                    const v = std.mem.readInt(u16, bytes[cursor..][0..2], .little);
                    cursor += 2;
                    try meta_map.setInt(key, @intCast(v));
                },
                .int16 => {
                    if (cursor + 2 > bytes.len) return error.UnexpectedEof;
                    const v = std.mem.readInt(i16, bytes[cursor..][0..2], .little);
                    cursor += 2;
                    try meta_map.setInt(key, @intCast(v));
                },
                .uint32 => {
                    if (cursor + 4 > bytes.len) return error.UnexpectedEof;
                    const v = std.mem.readInt(u32, bytes[cursor..][0..4], .little);
                    cursor += 4;
                    if (std.mem.eql(u8, key, "general.alignment")) {
                        alignment = @intCast(v);
                    }
                    try meta_map.setInt(key, @intCast(v));
                },
                .int32 => {
                    if (cursor + 4 > bytes.len) return error.UnexpectedEof;
                    const v = std.mem.readInt(i32, bytes[cursor..][0..4], .little);
                    cursor += 4;
                    try meta_map.setInt(key, @intCast(v));
                },
                .float32 => {
                    if (cursor + 4 > bytes.len) return error.UnexpectedEof;
                    const bits = std.mem.readInt(u32, bytes[cursor..][0..4], .little);
                    const v: f32 = @bitCast(bits);
                    cursor += 4;
                    try meta_map.setFloat(key, @floatCast(v));
                },
                .bool => {
                    if (cursor + 1 > bytes.len) return error.UnexpectedEof;
                    const v = bytes[cursor] != 0;
                    cursor += 1;
                    try meta_map.setBool(key, v);
                },
                .string => {
                    const s = try readGGUFString(bytes, &cursor, version);
                    try meta_map.setString(key, s);
                },
                .uint64 => {
                    if (cursor + 8 > bytes.len) return error.UnexpectedEof;
                    const v = std.mem.readInt(u64, bytes[cursor..][0..8], .little);
                    cursor += 8;
                    try meta_map.setInt(key, @intCast(v));
                },
                .int64 => {
                    if (cursor + 8 > bytes.len) return error.UnexpectedEof;
                    const v = std.mem.readInt(i64, bytes[cursor..][0..8], .little);
                    cursor += 8;
                    try meta_map.setInt(key, v);
                },
                .float64 => {
                    if (cursor + 8 > bytes.len) return error.UnexpectedEof;
                    const bits = std.mem.readInt(u64, bytes[cursor..][0..8], .little);
                    const v: f64 = @bitCast(bits);
                    cursor += 8;
                    try meta_map.setFloat(key, v);
                },
                .array => {
                    if (cursor + 4 > bytes.len) return error.UnexpectedEof;
                    const item_type_raw = std.mem.readInt(u32, bytes[cursor..][0..4], .little);
                    cursor += 4;
                    const item_type: GGUFValueType = @enumFromInt(item_type_raw);

                    var arr_len: u64 = 0;
                    if (version == 1) {
                        if (cursor + 4 > bytes.len) return error.UnexpectedEof;
                        arr_len = std.mem.readInt(u32, bytes[cursor..][0..4], .little);
                        cursor += 4;
                    } else {
                        if (cursor + 8 > bytes.len) return error.UnexpectedEof;
                        arr_len = std.mem.readInt(u64, bytes[cursor..][0..8], .little);
                        cursor += 8;
                    }

                    // Skip array payload or read strings
                    for (0..arr_len) |_| {
                        switch (item_type) {
                            .uint8, .int8, .bool => cursor += 1,
                            .uint16, .int16 => cursor += 2,
                            .uint32, .int32, .float32 => cursor += 4,
                            .uint64, .int64, .float64 => cursor += 8,
                            .string => {
                                _ = try readGGUFString(bytes, &cursor, version);
                            },
                            .array => return error.NestedArraysNotSupported,
                            else => return error.UnsupportedItemType,
                        }
                    }
                },
                else => return error.UnsupportedValueType,
            }
        }

        // Parse Tensor TOC
        var tensors = std.ArrayList(GGUFTensorInfo).initCapacity(allocator, tensor_count) catch |err| return err;

        for (0..tensor_count) |_| {
            const name = try readGGUFString(bytes, &cursor, version);
            if (cursor + 4 > bytes.len) return error.UnexpectedEof;
            const n_dims = std.mem.readInt(u32, bytes[cursor..][0..4], .little);
            cursor += 4;

            if (n_dims > format.MAX_DIMS) return error.TooManyDimensions;

            var dims = @as([format.MAX_DIMS]u64, @splat(0));
            for (0..n_dims) |d| {
                if (cursor + 8 > bytes.len) return error.UnexpectedEof;
                dims[d] = std.mem.readInt(u64, bytes[cursor..][0..8], .little);
                cursor += 8;
            }

            if (cursor + 4 > bytes.len) return error.UnexpectedEof;
            const ggml_type_raw = std.mem.readInt(u32, bytes[cursor..][0..4], .little);
            cursor += 4;
            const ggml_type: GGMLType = @enumFromInt(ggml_type_raw);

            if (cursor + 8 > bytes.len) return error.UnexpectedEof;
            const offset = std.mem.readInt(u64, bytes[cursor..][0..8], .little);
            cursor += 8;

            const tensor_size = calculateGGMLTensorSize(ggml_type, dims[0..n_dims]);

            tensors.append(allocator, .{
                .name = name,
                .n_dimensions = n_dims,
                .dimensions = dims,
                .ggml_type = ggml_type,
                .offset = offset,
                .size = tensor_size,
            }) catch |err| return err;
        }

        const tensor_data_offset = platform.alignForward(cursor, alignment);

        return GGUFReader{
            .allocator = allocator,
            .bytes = bytes,
            .version = version,
            .tensor_count = tensor_count,
            .metadata_kv_count = kv_count,
            .alignment = alignment,
            .tensor_data_offset = tensor_data_offset,
            .tensors = tensors,
            .metadata = meta_map,
        };
    }

    pub fn deinit(self: *GGUFReader) void {
        self.metadata.deinit();
        self.tensors.deinit(self.allocator);
    }

    pub fn getTensorData(self: *const GGUFReader, tensor: *const GGUFTensorInfo) ![]const u8 {
        const start = self.tensor_data_offset + tensor.offset;
        const end = start + tensor.size;
        if (end > self.bytes.len) return error.TensorDataOutOfBounds;
        return self.bytes[start..end];
    }

    fn readGGUFString(bytes: []const u8, cursor: *usize, version: u32) ![]const u8 {
        var str_len: usize = 0;
        if (version == 1) {
            if (cursor.* + 4 > bytes.len) return error.UnexpectedEof;
            str_len = std.mem.readInt(u32, bytes[cursor.*..][0..4], .little);
            cursor.* += 4;
        } else {
            if (cursor.* + 8 > bytes.len) return error.UnexpectedEof;
            str_len = @intCast(std.mem.readInt(u64, bytes[cursor.*..][0..8], .little));
            cursor.* += 8;
        }

        if (cursor.* + str_len > bytes.len) return error.UnexpectedEof;
        const str = bytes[cursor.* .. cursor.* + str_len];
        cursor.* += str_len;
        return str;
    }
};

/// Transcodes/transplants any GGUF file into standard HK container with zero precision loss.
pub fn convertGGUFToHK(input_path: []const u8, output_path: []const u8, allocator: std.mem.Allocator) !void {
    var region = try platform.mapOrReadFile(input_path, allocator);
    defer region.deinit(allocator);

    var gguf_reader = try GGUFReader.init(allocator, region.bytes);
    defer gguf_reader.deinit();

    var hk_writer = writer.HKWriter.init(allocator);
    defer hk_writer.deinit();

    // Set standard HK 128-byte alignment for Tensor Cores
    hk_writer.setAlignment(128);

    // Transfer metadata
    for (gguf_reader.metadata.items.items) |item| {
        switch (item.value) {
            .val_string => |s| try hk_writer.addMetadataString(item.key, s),
            .val_int64 => |i| try hk_writer.addMetadataInt(item.key, i),
            .val_float64 => |f| try hk_writer.addMetadataFloat(item.key, f),
            .val_bool => |b| try hk_writer.addMetadataBool(item.key, b),
            .val_json => |j| try hk_writer.addMetadataString(item.key, j),
            .val_bytes => {},
        }
    }

    // Add source provenance metadata
    try hk_writer.addMetadataString("hk.transcoded_from", "gguf");
    try hk_writer.addMetadataInt("hk.gguf_version", gguf_reader.version);

    // Transfer tensors with zero-copy bitstream transplant for matching quants
    for (gguf_reader.tensors.items) |t| {
        const raw_bytes = try gguf_reader.getTensorData(&t);

        const st_opt = ggmlTypeToStorageType(t.ggml_type);
        const storage_type: format.StorageType = st_opt orelse .f32;

        // In GGUF, dimensions are innermost-first: [ne0, ne1, ne2, ne3].
        // In HK, dimensions are C row-major: [dim0, dim1, dim2, dim3].
        // Reverse them so dim0 corresponds to batch/outer dimension.
        var hk_shape = @as([format.MAX_DIMS]u64, @splat(0));
        const ndim: u8 = @intCast(t.n_dimensions);
        for (0..ndim) |i| {
            hk_shape[i] = t.dimensions[ndim - 1 - i];
        }

        try hk_writer.addTensor(.{
            .name = t.name,
            .storage_type = storage_type,
            .tile_layout = .row_major,
            .sparsity_type = .none,
            .ndim = ndim,
            .shape = hk_shape,
            .data = raw_bytes,
        });
    }

    try hk_writer.writeToFile(output_path);
}

/// Exports an HK model container directly to standard GGUF v3 format.
pub fn exportHKToGGUF(input_hk_path: []const u8, output_gguf_path: []const u8, allocator: std.mem.Allocator) !void {
    var hk_reader = try reader.HKReader.open(input_hk_path, allocator);
    defer hk_reader.deinit();

    const io = std.Options.debug_io;
    const cwd = std.Io.Dir.cwd();

    var file = try cwd.createFile(io, output_gguf_path, .{});
    defer file.close(io);

    var bw = buf.BufferWriter.init(allocator);
    defer bw.deinit();

    // 1. GGUF Magic & Version 3
    try bw.writeU32(GGUF_MAGIC);
    try bw.writeU32(3);

    const tensor_count: u64 = hk_reader.toc.entries.items.len;
    // Count valid exportable metadata (plus general.alignment)
    const kv_count: u64 = hk_reader.metadata_map.items.items.len + 1;

    try bw.writeU64(tensor_count);
    try bw.writeU64(kv_count);

    // 2. Metadata KV Pairs
    // Always write general.alignment = 32
    try writeGGUFString(&bw, "general.alignment");
    try bw.writeU32(@intFromEnum(GGUFValueType.uint32));
    try bw.writeU32(32);

    for (hk_reader.metadata_map.items.items) |item| {
        try writeGGUFString(&bw, item.key);
        switch (item.value) {
            .val_string => |s| {
                try bw.writeU32(@intFromEnum(GGUFValueType.string));
                try writeGGUFString(&bw, s);
            },
            .val_int64 => |i| {
                try bw.writeU32(@intFromEnum(GGUFValueType.int64));
                try bw.writeI64(i);
            },
            .val_float64 => |f| {
                try bw.writeU32(@intFromEnum(GGUFValueType.float64));
                const bits: u64 = @bitCast(f);
                try bw.writeU64(bits);
            },
            .val_bool => |b| {
                try bw.writeU32(@intFromEnum(GGUFValueType.bool));
                try bw.writeU8(if (b) 1 else 0);
            },
            .val_json => |j| {
                try bw.writeU32(@intFromEnum(GGUFValueType.string));
                try writeGGUFString(&bw, j);
            },
            .val_bytes => |bytes| {
                try bw.writeU32(@intFromEnum(GGUFValueType.string));
                try writeGGUFString(&bw, bytes);
            },
        }
    }

    // 3. Tensor TOC calculation
    // Measure header + metadata size so far
    var current_offset: u64 = 0;
    const header_meta_len = bw.getBytes().len;

    // Calculate TOC size
    var toc_writer = buf.BufferWriter.init(allocator);
    defer toc_writer.deinit();

    for (hk_reader.toc.entries.items) |t| {
        const ggml_t: GGMLType = storageTypeToGGMLType(t.storage_type) orelse .f32;
        try writeGGUFString(&toc_writer, t.name);
        try toc_writer.writeU32(@intCast(t.ndim));
        // Reverse dimensions back to GGUF order
        for (0..t.ndim) |i| {
            const d = t.shape[t.ndim - 1 - i];
            try toc_writer.writeU64(d);
        }
        try toc_writer.writeU32(@intFromEnum(ggml_t));
        try toc_writer.writeU64(current_offset);

        // Align each tensor payload to 32 bytes
        const aligned_data_size = platform.alignForward(t.data_size, 32);
        current_offset += aligned_data_size;
    }

    const total_header_size = header_meta_len + toc_writer.getBytes().len;
    const aligned_data_start = platform.alignForward(total_header_size, 32);
    const initial_padding = aligned_data_start - total_header_size;

    // Write header and metadata
    try file.writeStreamingAll(io, bw.getBytes());
    // Write TOC
    try file.writeStreamingAll(io, toc_writer.getBytes());

    // Write initial padding
    if (initial_padding > 0) {
        const pad = @as([32]u8, @splat(0));
        try file.writeStreamingAll(io, pad[0..initial_padding]);
    }

    // Write tensor payloads directly from HK container
    for (hk_reader.toc.entries.items) |t| {
        const tensor_bytes = try hk_reader.getTensorData(t);
        try file.writeStreamingAll(io, tensor_bytes);

        // Pad to 32-byte alignment if needed
        const rem = t.data_size % 32;
        if (rem != 0) {
            const pad_len = 32 - rem;
            const pad = @as([32]u8, @splat(0));
            try file.writeStreamingAll(io, pad[0..pad_len]);
        }
    }
}

fn writeGGUFString(bw: *buf.BufferWriter, str: []const u8) !void {
    try bw.writeU64(@intCast(str.len));
    try bw.writeBytes(str);
}
