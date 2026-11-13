const std = @import("std");
const format = @import("format.zig");
const buf = @import("buf.zig");

pub const TensorTOC = struct {
    entries: std.ArrayList(format.TensorEntry) = .empty,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) TensorTOC {
        return .{
            .entries = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TensorTOC) void {
        for (self.entries.items) |e| {
            self.allocator.free(e.name.ptr[0 .. e.name.len + 1]);
        }
        self.entries.deinit(self.allocator);
    }

    pub fn add(self: *TensorTOC, entry: format.TensorEntry) !void {
        var e = entry;
        e.name = try self.allocator.dupeSentinel(u8, entry.name, 0);
        errdefer self.allocator.free(e.name.ptr[0 .. e.name.len + 1]);
        try self.entries.append(self.allocator, e);
    }

    pub fn find(self: *const TensorTOC, name: []const u8) ?format.TensorEntry {
        for (self.entries.items) |entry| {
            if (std.mem.eql(u8, entry.name, name)) {
                return entry;
            }
        }
        return null;
    }

    pub fn serialize(self: *const TensorTOC, writer: *buf.BufferWriter) !void {
        try self.serializeV2(writer, false);
    }

    pub fn serializeV2(self: *const TensorTOC, writer: *buf.BufferWriter, include_v2_fields: bool) !void {
        for (self.entries.items) |e| {
            const nlen: u16 = @intCast(e.name.len);
            try writer.writeU16(nlen);
            try writer.writeBytes(e.name);

            try writer.writeU8(@intFromEnum(e.storage_type));
            try writer.writeU8(@intFromEnum(e.tile_layout));
            try writer.writeU8(@intFromEnum(e.sparsity_type));
            try writer.writeU8(e.ndim);

            for (0..e.ndim) |d| {
                try writer.writeU64(e.shape[d]);
            }

            try writer.writeU64(e.data_offset);
            try writer.writeU64(e.data_size);
            try writer.writeU64(e.residual_offset);
            try writer.writeU64(e.residual_size);
            try writer.writeU64(e.scale_offset);
            try writer.writeU64(e.scale_size);
            try writer.writeU16(e.block_size);

            const sbits: u32 = @bitCast(e.sparsity_ratio);
            try writer.writeU32(sbits);

            if (include_v2_fields) {
                try writer.writeU16(e.backend_layout_id);
                try writer.writeBytes(&e.tensor_hash);
            }
        }
    }

    pub fn deserialize(buffer: []const u8, count: usize, allocator: std.mem.Allocator) !TensorTOC {
        return deserializeV2(buffer, count, allocator, false);
    }

    pub fn deserializeV2(buffer: []const u8, count: usize, allocator: std.mem.Allocator, has_v2_fields: bool) !TensorTOC {
        var toc = TensorTOC.init(allocator);
        errdefer toc.deinit();
        var reader = buf.BufferReader.init(buffer);

        var i: usize = 0;
        while (i < count) : (i += 1) {
            const nlen = try reader.readU16();
            const name_raw = try reader.readBytes(nlen);
            const name = try allocator.dupeSentinel(u8, name_raw, 0);
            errdefer allocator.free(name);

            const stype_b = try reader.readU8();
            const layout_b = try reader.readU8();
            const sparse_b = try reader.readU8();
            const ndim = try reader.readU8();

            if (ndim > format.MAX_DIMS) return error.TooManyDimensions;

            var shape = @as([format.MAX_DIMS]u64, @splat(0));
            for (0..ndim) |d| {
                shape[d] = try reader.readU64();
            }

            const data_offset = try reader.readU64();
            const data_size = try reader.readU64();
            const residual_offset = try reader.readU64();
            const residual_size = try reader.readU64();
            const scale_offset = try reader.readU64();
            const scale_size = try reader.readU64();
            const block_size = try reader.readU16();
            const sbits = try reader.readU32();
            const sparsity_ratio: f32 = @bitCast(sbits);

            var backend_layout_id: u16 = 0;
            var tensor_hash: [32]u8 = @splat(0);
            if (has_v2_fields) {
                backend_layout_id = try reader.readU16();
                const h = try reader.readBytes(32);
                @memcpy(&tensor_hash, h);
            }

            try toc.entries.append(allocator, .{
                .name = name,
                .storage_type = @enumFromInt(stype_b),
                .tile_layout = @enumFromInt(layout_b),
                .sparsity_type = @enumFromInt(sparse_b),
                .ndim = ndim,
                .shape = shape,
                .data_offset = data_offset,
                .data_size = data_size,
                .residual_offset = residual_offset,
                .residual_size = residual_size,
                .scale_offset = scale_offset,
                .scale_size = scale_size,
                .block_size = block_size,
                .sparsity_ratio = sparsity_ratio,
                .backend_layout_id = backend_layout_id,
                .tensor_hash = tensor_hash,
            });
        }
        return toc;
    }
};
