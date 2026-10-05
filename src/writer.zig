const std = @import("std");
const format = @import("format.zig");
const metadata = @import("metadata.zig");
const tensor_toc = @import("tensor_toc.zig");
const platform = @import("platform.zig");
const buf = @import("buf.zig");

pub const TensorPayload = struct {
    name: []const u8,
    storage_type: format.StorageType,
    tile_layout: format.TileLayout = .row_major,
    sparsity_type: format.SparsityType = .none,
    ndim: u8,
    shape: [format.MAX_DIMS]u64,
    data: []const u8 = &[_]u8{},
    residual: ?[]const u8 = null,
    scales: ?[]const u8 = null,
    block_size: u16 = 32,
    sparsity_ratio: f32 = 0.0,
    shared_target_index: ?usize = null,
};

pub const HKWriter = struct {
    allocator: std.mem.Allocator,
    meta_map: metadata.MetadataMap,
    tensors: std.ArrayList(TensorPayload) = .empty,
    alignment: usize = format.DEFAULT_ALIGNMENT_BYTES,
    split_index: u16 = 0,
    split_count: u16 = 1,
    is_sharded: bool = false,
    raw_weight_storage: bool = false,

    pub fn init(allocator: std.mem.Allocator) HKWriter {
        return .{
            .allocator = allocator,
            .meta_map = metadata.MetadataMap.init(allocator),
            .tensors = .empty,
            .alignment = format.DEFAULT_ALIGNMENT_BYTES,
            .split_index = 0,
            .split_count = 1,
            .is_sharded = false,
            .raw_weight_storage = false,
        };
    }

    pub fn setAlignment(self: *HKWriter, align_bytes: usize) void {
        self.alignment = if (align_bytes == 0) 1 else align_bytes;
    }

    pub fn setRawWeightStorage(self: *HKWriter, enabled: bool) void {
        self.raw_weight_storage = enabled;
    }

    pub fn setSharding(self: *HKWriter, split_index: u16, split_count: u16) void {
        self.split_index = split_index;
        self.split_count = split_count;
        self.is_sharded = (split_count > 1);
    }

    pub fn deinit(self: *HKWriter) void {
        self.meta_map.deinit();
        self.tensors.deinit(self.allocator);
    }

    pub fn addMetadataString(self: *HKWriter, key: []const u8, val: []const u8) !void {
        try self.meta_map.setString(key, val);
    }

    pub fn addMetadataInt(self: *HKWriter, key: []const u8, val: i64) !void {
        try self.meta_map.setInt(key, val);
    }

    pub fn addMetadataFloat(self: *HKWriter, key: []const u8, val: f64) !void {
        try self.meta_map.setFloat(key, val);
    }

    pub fn addMetadataBool(self: *HKWriter, key: []const u8, val: bool) !void {
        try self.meta_map.setBool(key, val);
    }

    pub fn addMetadataJson(self: *HKWriter, key: []const u8, val: []const u8) !void {
        try self.meta_map.setJson(key, val);
    }

    pub fn addTensor(self: *HKWriter, payload: TensorPayload) !void {
        try self.tensors.append(self.allocator, payload);
    }

    /// Writes the complete .hk file to the specified path with 128-byte alignment
    pub fn writeToFile(self: *HKWriter, path: []const u8) !void {
        const io = std.Options.debug_io;
        const cwd = std.Io.Dir.cwd();

        var file = try cwd.createFile(io, path, .{});
        defer file.close(io);

        var meta_writer = buf.BufferWriter.init(self.allocator);
        defer meta_writer.deinit();
        try self.meta_map.serialize(&meta_writer);
        const meta_bytes = meta_writer.getBytes();

        // Estimate TOC entries
        var toc = tensor_toc.TensorTOC.init(self.allocator);
        defer toc.deinit();

        for (self.tensors.items) |t| {
            try toc.add(.{
                .name = t.name,
                .storage_type = t.storage_type,
                .tile_layout = t.tile_layout,
                .sparsity_type = t.sparsity_type,
                .ndim = t.ndim,
                .shape = t.shape,
                .data_offset = 0, // calculated below
                .data_size = t.data.len,
                .residual_offset = 0,
                .residual_size = if (t.residual) |r| r.len else 0,
                .scale_offset = 0,
                .scale_size = if (t.scales) |s| s.len else 0,
                .block_size = t.block_size,
                .sparsity_ratio = t.sparsity_ratio,
            });
        }

        // Measure TOC serialized size
        var dummy_toc_writer = buf.BufferWriter.init(self.allocator);
        defer dummy_toc_writer.deinit();
        try toc.serialize(&dummy_toc_writer);
        const estimated_toc_size = dummy_toc_writer.getBytes().len;

        const header_size = @sizeOf(format.FileHeader); // 128 bytes
        const meta_offset = header_size;
        const meta_size = meta_bytes.len;

        const toc_offset = meta_offset + meta_size;
        const toc_size = estimated_toc_size;

        // Data payload starts at next aligned offset
        const raw_data_offset = toc_offset + toc_size;
        const tensor_data_offset = platform.alignForward(raw_data_offset, self.alignment);

        // Assign actual payload offsets
        var current_offset = tensor_data_offset;
        for (0..self.tensors.items.len) |i| {
            const t = self.tensors.items[i];

            if (t.storage_type == .null_ref) {
                toc.entries.items[i].data_offset = 0;
                toc.entries.items[i].data_size = 0;
                continue;
            }

            if (t.storage_type == .shared_ref and t.shared_target_index != null) {
                const target_idx = t.shared_target_index.?;
                if (target_idx < i) {
                    toc.entries.items[i].data_offset = toc.entries.items[target_idx].data_offset;
                    toc.entries.items[i].data_size = toc.entries.items[target_idx].data_size;
                    toc.entries.items[i].scale_offset = toc.entries.items[target_idx].scale_offset;
                    toc.entries.items[i].scale_size = toc.entries.items[target_idx].scale_size;
                    continue;
                }
            }

            current_offset = platform.alignForward(current_offset, self.alignment);
            toc.entries.items[i].data_offset = current_offset;
            current_offset += t.data.len;

            if (t.scales) |s| {
                current_offset = platform.alignForward(current_offset, self.alignment);
                toc.entries.items[i].scale_offset = current_offset;
                current_offset += s.len;
            }

            if (t.residual) |r| {
                current_offset = platform.alignForward(current_offset, self.alignment);
                toc.entries.items[i].residual_offset = current_offset;
                current_offset += r.len;
            }
        }

        // Re-serialize TOC with final offsets
        var final_toc_writer = buf.BufferWriter.init(self.allocator);
        defer final_toc_writer.deinit();
        try toc.serialize(&final_toc_writer);
        const final_toc_bytes = final_toc_writer.getBytes();

        // Prepare Header
        var flags: u32 = format.HeaderFlags.LITTLE_ENDIAN;
        if ((self.alignment % format.DEFAULT_ALIGNMENT_BYTES) == 0) {
            // Preserves NVIDIA Tensor Core 128-byte coalescing alignment
            flags |= format.HeaderFlags.TILE_ALIGNED;
        } else {
            flags |= format.HeaderFlags.FLEXIBLE_ALIGNMENT;
        }
        if (self.alignment >= format.UNIVERSAL_PAGE_ALIGNMENT_BYTES) {
            // Super-coalesced multi-device page alignment (AMD ROCm, Intel NPU, Apple Metal)
            flags |= format.HeaderFlags.UNIVERSAL_PAGE_ALIGNED;
        }
        if (self.raw_weight_storage) {
            flags |= format.HeaderFlags.RAW_WEIGHT_STORAGE;
        }
        if (self.is_sharded or self.split_count > 1) {
            flags |= format.HeaderFlags.IS_SHARDED;
        }
        var header = format.FileHeader{
            .flags = flags,
            .alignment = @intCast(self.alignment),
            .split_index = self.split_index,
            .split_count = self.split_count,
            .tensor_count = self.tensors.items.len,
            .metadata_kv_count = self.meta_map.items.items.len,
            .metadata_offset = meta_offset,
            .metadata_size = meta_size,
            .tensor_toc_offset = toc_offset,
            .tensor_toc_size = final_toc_bytes.len,
            .tensor_data_offset = tensor_data_offset,
            .appendix_offset = 0,
        };

        // Write Header
        const header_bytes = std.mem.asBytes(&header);
        try file.writeStreamingAll(io, header_bytes);

        // Write Metadata
        if (meta_bytes.len > 0) {
            try file.writeStreamingAll(io, meta_bytes);
        }

        // Write TOC
        try file.writeStreamingAll(io, final_toc_bytes);

        // Pad up to tensor_data_offset
        const current_file_pos = header_size + meta_size + final_toc_bytes.len;
        if (tensor_data_offset > current_file_pos) {
            try writeZeroPad(&file, io, tensor_data_offset - current_file_pos);
        }

        // Write Tensor Payloads
        var write_pos = tensor_data_offset;
        for (0..self.tensors.items.len) |i| {
            const t = self.tensors.items[i];
            const entry = toc.entries.items[i];

            if (t.storage_type == .null_ref or t.storage_type == .shared_ref) {
                continue;
            }

            // Pad to data_offset
            if (entry.data_offset > write_pos) {
                try writeZeroPad(&file, io, entry.data_offset - write_pos);
                write_pos = entry.data_offset;
            }
            if (t.data.len > 0) {
                try file.writeStreamingAll(io, t.data);
                write_pos += t.data.len;
            }

            if (t.scales) |s| {
                if (entry.scale_offset > write_pos) {
                    try writeZeroPad(&file, io, entry.scale_offset - write_pos);
                    write_pos = entry.scale_offset;
                }
                try file.writeStreamingAll(io, s);
                write_pos += s.len;
            }

            if (t.residual) |r| {
                if (entry.residual_offset > write_pos) {
                    try writeZeroPad(&file, io, entry.residual_offset - write_pos);
                    write_pos = entry.residual_offset;
                }
                try file.writeStreamingAll(io, r);
                write_pos += r.len;
            }
        }
    }
};

fn writeZeroPad(file: anytype, io: anytype, pad_len: usize) !void {
    const ZERO_PAD: [4096]u8 = @as([4096]u8, @splat(0));
    var rem = pad_len;
    while (rem > 0) {
        const chunk = @min(rem, ZERO_PAD.len);
        try file.writeStreamingAll(io, ZERO_PAD[0..chunk]);
        rem -= chunk;
    }
}
