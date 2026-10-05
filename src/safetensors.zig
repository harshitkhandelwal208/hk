const std = @import("std");
const format = @import("format.zig");
const writer_mod = @import("writer.zig");
const platform = @import("platform.zig");
const quantization = @import("quantization.zig");

pub const SafeTensorMeta = struct {
    dtype: []const u8,
    shape: []const u64,
    data_offsets: [2]u64,
};

pub const SafeTensorsReader = struct {
    allocator: std.mem.Allocator,
    mmap_region: platform.MmapRegion,
    header_len: u64,
    data_start: u64,

    pub fn open(path: []const u8, allocator: std.mem.Allocator) !SafeTensorsReader {
        var region = try platform.mapOrReadFile(path, allocator);
        errdefer region.deinit(allocator);

        if (region.bytes.len < 8) return error.FileTooSmall;

        const header_len = std.mem.readInt(u64, region.bytes[0..8], .little);
        const data_start = 8 + header_len;
        if (data_start > region.bytes.len) return error.HeaderOutOfBounds;

        return .{
            .allocator = allocator,
            .mmap_region = region,
            .header_len = header_len,
            .data_start = data_start,
        };
    }

    pub fn deinit(self: *SafeTensorsReader) void {
        self.mmap_region.deinit(self.allocator);
    }

    pub fn getHeaderJson(self: *const SafeTensorsReader) []const u8 {
        return self.mmap_region.bytes[8 .. 8 + self.header_len];
    }

    pub fn getRawTensorBytes(self: *const SafeTensorsReader, start_off: u64, end_off: u64) ![]const u8 {
        const abs_start = self.data_start + start_off;
        const abs_end = self.data_start + end_off;
        if (abs_end > self.mmap_region.bytes.len or abs_start > abs_end) {
            return error.TensorOutOfBounds;
        }
        return self.mmap_region.bytes[abs_start..abs_end];
    }
};

/// Transcodes a .safetensors model directly to an .hk container natively in Zig
pub fn transcodeSafeTensorsToHK(
    allocator: std.mem.Allocator,
    in_path: []const u8,
    out_path: []const u8,
    target_storage: format.StorageType,
) !void {
    var st_reader = try SafeTensorsReader.open(in_path, allocator);
    defer st_reader.deinit();

    const header_json = st_reader.getHeaderJson();

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_json, .{});
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidSafeTensorsHeader;

    var writer = writer_mod.HKWriter.init(allocator);
    defer writer.deinit();

    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        const tensor_name = entry.key_ptr.*;
        if (std.mem.eql(u8, tensor_name, "__metadata__")) {
            // Store string metadata into HK
            if (entry.value_ptr.* == .object) {
                var meta_it = entry.value_ptr.object.iterator();
                while (meta_it.next()) |m_entry| {
                    if (m_entry.value_ptr.* == .string) {
                        try writer.addMetadataString(m_entry.key_ptr.*, m_entry.value_ptr.string);
                    }
                }
            }
            continue;
        }

        if (entry.value_ptr.* != .object) continue;
        const t_obj = entry.value_ptr.object;

        const dtype_val = t_obj.get("dtype") orelse continue;
        const dtype_str = if (dtype_val == .string) dtype_val.string else continue;

        const shape_val = t_obj.get("shape") orelse continue;
        if (shape_val != .array) continue;

        const offsets_val = t_obj.get("data_offsets") orelse continue;
        if (offsets_val != .array or offsets_val.array.items.len < 2) continue;

        const start_off: u64 = switch (offsets_val.array.items[0]) {
            .integer => |i| @intCast(i),
            else => 0,
        };
        const end_off: u64 = switch (offsets_val.array.items[1]) {
            .integer => |i| @intCast(i),
            else => 0,
        };

        const raw_bytes = try st_reader.getRawTensorBytes(start_off, end_off);

        // Extract shape
        var shape_arr = @as([format.MAX_DIMS]u64, @splat(0));
        const ndim: u8 = @min(@as(u8, @intCast(shape_val.array.items.len)), format.MAX_DIMS);
        var total_elements: usize = 1;
        for (0..ndim) |i| {
            shape_arr[i] = switch (shape_val.array.items[i]) {
                .integer => |val| @intCast(val),
                else => 1,
            };
            total_elements *= @intCast(shape_arr[i]);
        }

        // Handle Float32 vs Float16/BFloat16 conversion
        var f32_buf: ?[]f32 = null;
        defer if (f32_buf) |b| allocator.free(b);

        var final_data: []const u8 = raw_bytes;
        var final_storage: format.StorageType = .f32;

        if (std.mem.eql(u8, dtype_str, "F32")) {
            final_data = raw_bytes;
            final_storage = .f32;
        } else if (std.mem.eql(u8, dtype_str, "F16")) {
            const num_f16 = raw_bytes.len / 2;
            const converted = try allocator.alloc(f32, num_f16);
            f32_buf = converted;
            for (0..num_f16) |i| {
                const u_val = std.mem.readInt(u16, raw_bytes[i * 2 .. (i + 1) * 2][0..2], .little);
                converted[i] = @floatCast(@as(f16, @bitCast(u_val)));
            }
            final_data = std.mem.sliceAsBytes(converted);
            final_storage = .f32;
        } else if (std.mem.eql(u8, dtype_str, "BF16")) {
            const num_bf16 = raw_bytes.len / 2;
            const converted = try allocator.alloc(f32, num_bf16);
            f32_buf = converted;
            for (0..num_bf16) |i| {
                const u_val = std.mem.readInt(u16, raw_bytes[i * 2 .. (i + 1) * 2][0..2], .little);
                const bits: u32 = @as(u32, u_val) << 16;
                converted[i] = @bitCast(bits);
            }
            final_data = std.mem.sliceAsBytes(converted);
            final_storage = .f32;
        }

        // On-the-fly quantization if requested
        var quant_buf: ?[]u8 = null;
        defer if (quant_buf) |qb| allocator.free(qb);

        var aligned_f32_buf: ?[]f32 = null;
        defer if (aligned_f32_buf) |ab| allocator.free(ab);

        if ((target_storage == .q8_0 or target_storage == .q4_0) and total_elements % 32 == 0 and final_storage == .f32) {
            const num_floats = final_data.len / @sizeOf(f32);
            const aligned_f32: []const f32 = if (std.mem.isAligned(@intFromPtr(final_data.ptr), @alignOf(f32))) blk: {
                const ptr: [*]const f32 = @ptrCast(@alignCast(final_data.ptr));
                break :blk ptr[0..num_floats];
            } else blk: {
                const copy = try allocator.alloc(f32, num_floats);
                aligned_f32_buf = copy;
                for (0..num_floats) |i| {
                    const u = std.mem.readInt(u32, final_data[i * 4 .. (i + 1) * 4][0..4], .little);
                    copy[i] = @bitCast(u);
                }
                break :blk copy;
            };

            if (target_storage == .q8_0) {
                const num_blocks = total_elements / 32;
                const q_bytes = try allocator.alloc(u8, num_blocks * @sizeOf(quantization.BlockQ8_0));
                quant_buf = q_bytes;
                const blocks: [*]quantization.BlockQ8_0 = @ptrCast(@alignCast(q_bytes.ptr));

                for (0..num_blocks) |b| {
                    quantization.quantizeBlockQ8_0(aligned_f32[b * 32 .. (b + 1) * 32], &blocks[b]);
                }

                final_data = q_bytes;
                final_storage = .q8_0;
            } else if (target_storage == .q4_0) {
                const num_blocks = total_elements / 32;
                const q_bytes = try allocator.alloc(u8, num_blocks * @sizeOf(quantization.BlockQ4_0));
                quant_buf = q_bytes;
                const blocks: [*]quantization.BlockQ4_0 = @ptrCast(@alignCast(q_bytes.ptr));

                for (0..num_blocks) |b| {
                    quantization.quantizeBlockQ4_0(aligned_f32[b * 32 .. (b + 1) * 32], &blocks[b]);
                }

                final_data = q_bytes;
                final_storage = .q4_0;
            }
        }

        try writer.addTensor(.{
            .name = tensor_name,
            .storage_type = final_storage,
            .tile_layout = .row_major,
            .sparsity_type = .none,
            .ndim = ndim,
            .shape = shape_arr,
            .data = final_data,
        });
    }

    try writer.writeToFile(out_path);
}
