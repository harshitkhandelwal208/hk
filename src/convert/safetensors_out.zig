//! Writes the tensors of an `.hk` container as a safetensors file.
//!
//! Tensor names and shapes are those stored in the container (a model converted from GGUF keeps
//! its GGUF names). Float tensors keep their type. Quantized and sparse tensors are decoded and
//! stored as F16, because safetensors has no block formats; that conversion is lossy for anything
//! quantized below 16 bits, and it is the one place memory use is not flat: the largest tensor is
//! decoded to f32 in memory first.

const std = @import("std");
const format = @import("../format.zig");
const reader_mod = @import("../reader.zig");

pub const Error = error{ OutOfMemory, UnsupportedTensor } || std.Io.File.OpenError || std.Io.File.WritePositionalError || std.Io.Writer.Error || anyerror;

fn dtypeName(t: format.StorageType) ?[]const u8 {
    return switch (t) {
        .f32 => "F32",
        .f16 => "F16",
        .bf16 => "BF16",
        .f64 => "F64",
        .int8 => "I8",
        .uint8 => "U8",
        .int16 => "I16",
        .uint16 => "U16",
        .int32 => "I32",
        .uint32 => "U32",
        .int64 => "I64",
        .uint64 => "U64",
        .bool => "BOOL",
        else => null,
    };
}

fn elemBytes(t: format.StorageType) usize {
    return switch (t) {
        .f32, .int32, .uint32 => 4,
        .f16, .bf16, .int16, .uint16 => 2,
        .f64, .int64, .uint64 => 8,
        else => 1,
    };
}

pub fn exportFile(allocator: std.mem.Allocator, io: std.Io, reader: *const reader_mod.HKReader, out_path: []const u8) !void {
    const entries = reader.toc.entries.items;

    // Layout: every tensor's byte length in the output, then the header that lists them.
    var header: std.Io.Writer.Allocating = .init(allocator);
    defer header.deinit();
    const hw = &header.writer;
    try hw.writeAll("{\"__metadata__\":{\"format\":\"pt\",\"source\":\"hk\"}");
    var offset: u64 = 0;
    for (entries) |e| {
        var n: u64 = 1;
        for (0..e.ndim) |i| n *= e.shape[i];
        const native = dtypeName(e.storage_type);
        const dt: []const u8 = native orelse "F16";
        const size: u64 = if (native != null) n * elemBytes(e.storage_type) else n * 2;
        try hw.print(",{f}:{{\"dtype\":\"{s}\",\"shape\":[", .{ std.json.fmt(e.name, .{}), dt });
        for (0..e.ndim) |i| {
            if (i != 0) try hw.writeAll(",");
            try hw.print("{d}", .{e.shape[i]});
        }
        try hw.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + size });
        offset += size;
    }
    try hw.writeAll("}");
    while (header.written().len % 8 != 0) try hw.writeAll(" ");

    var file = try std.Io.Dir.cwd().createFile(io, out_path, .{});
    defer file.close(io);
    var wbuf: [1 << 16]u8 = undefined;
    var fw = file.writer(io, &wbuf);
    const w = &fw.interface;
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, header.written().len, .little);
    try w.writeAll(&len_bytes);
    try w.writeAll(header.written());

    for (entries) |e| {
        var n: u64 = 1;
        for (0..e.ndim) |i| n *= e.shape[i];
        if (dtypeName(e.storage_type) != null) {
            try w.writeAll(try reader.getRawTensorBytes(e));
            continue;
        }
        // Decode to f32, then narrow to f16 in pieces.
        const f = try allocator.alloc(f32, @intCast(n));
        defer allocator.free(f);
        try reader.dequantizeToF32(e, true, f);
        var piece: [8192]u16 = undefined;
        var done: usize = 0;
        while (done < f.len) {
            const k = @min(piece.len, f.len - done);
            for (0..k) |i| piece[i] = @bitCast(@as(f16, @floatCast(f[done + i])));
            try w.writeAll(std.mem.sliceAsBytes(piece[0..k]));
            done += k;
        }
    }
    try w.flush();
}
