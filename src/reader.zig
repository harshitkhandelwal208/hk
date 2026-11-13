const std = @import("std");
const format = @import("format.zig");
const metadata = @import("metadata.zig");
const tensor_toc = @import("tensor_toc.zig");
const platform = @import("platform.zig");
const nf4 = @import("nf4.zig");
const research = @import("quant/research.zig");
const sparsity = @import("sparsity.zig");
const tiling = @import("tiling.zig");
const quant = @import("quant.zig");
const integrity = @import("integrity.zig");

pub const HKReader = struct {
    allocator: std.mem.Allocator,
    mmap_region: platform.MmapRegion,
    header: format.FileHeader,
    metadata_map: metadata.MetadataMap,
    toc: tensor_toc.TensorTOC,

    pub fn open(path: []const u8, allocator: std.mem.Allocator) !HKReader {
        var region = try platform.mapOrReadFile(path, allocator);
        errdefer region.deinit(allocator);

        if (region.bytes.len < @sizeOf(format.FileHeader)) {
            return error.FileTooSmall;
        }

        const header_ptr: *const format.FileHeader = @ptrCast(@alignCast(region.bytes.ptr));
        const header = header_ptr.*;

        if (!header.isValid()) {
            return error.InvalidMagicOrVersion;
        }

        try validateHeaderLayout(&header, region.bytes.len);

        // Deserialize Metadata
        const meta_start: usize = @intCast(header.metadata_offset);
        const meta_end = try checkedEnd(header.metadata_offset, header.metadata_size, region.bytes.len, error.MetadataOutOfBounds);
        const meta_buf = region.bytes[meta_start..meta_end];
        var meta_map = try metadata.MetadataMap.deserialize(meta_buf, header.metadata_kv_count, allocator);
        errdefer meta_map.deinit();

        // Deserialize TOC
        const toc_start: usize = @intCast(header.tensor_toc_offset);
        const toc_end = try checkedEnd(header.tensor_toc_offset, header.tensor_toc_size, region.bytes.len, error.TOCOutOfBounds);
        const toc_buf = region.bytes[toc_start..toc_end];
        const has_v2 = (header.flags & format.HeaderFlags.HAS_V2_EXTENSIONS != 0) or
            (header.flags & format.HeaderFlags.HAS_PER_TENSOR_HASH != 0) or
            header.version_major >= 2;
        var toc = try tensor_toc.TensorTOC.deserializeV2(toc_buf, header.tensor_count, allocator, has_v2);
        errdefer toc.deinit();
        try validateTensorRanges(&toc, region.bytes.len);

        return HKReader{
            .allocator = allocator,
            .mmap_region = region,
            .header = header,
            .metadata_map = meta_map,
            .toc = toc,
        };
    }

    pub fn deinit(self: *HKReader) void {
        self.toc.deinit();
        self.metadata_map.deinit();
        self.mmap_region.deinit(self.allocator);
    }

    pub fn isSharded(self: *const HKReader) bool {
        return (self.header.flags & format.HeaderFlags.IS_SHARDED) != 0 or self.header.split_count > 1;
    }

    pub fn isRawWeightStorage(self: *const HKReader) bool {
        return (self.header.flags & format.HeaderFlags.RAW_WEIGHT_STORAGE) != 0;
    }

    pub fn isUniversalPageAligned(self: *const HKReader) bool {
        return (self.header.flags & format.HeaderFlags.UNIVERSAL_PAGE_ALIGNED) != 0;
    }

    pub fn isTensorCoreAligned(self: *const HKReader) bool {
        return (self.header.flags & format.HeaderFlags.TILE_ALIGNED) != 0;
    }

    pub fn getAlignment(self: *const HKReader) usize {
        return self.header.alignment;
    }

    pub fn getSplitIndex(self: *const HKReader) u16 {
        return self.header.split_index;
    }

    pub fn getSplitCount(self: *const HKReader) u16 {
        return self.header.split_count;
    }

    /// Checks the optional container SHA-256 over the existing mapping. This faults every page of the file
    /// into the process (RSS grows to the file size); prefer `integrity.verifyFile` for large models.
    ///
    /// Legacy v1 files report `.legacy_missing` rather than failing to open.
    pub fn verifyIntegrity(self: *const HKReader) !integrity.Verification {
        return integrity.verify(self.mmap_region.bytes);
    }

    /// Zero-copy raw byte slice for raw tensor storage
    pub fn getRawTensorBytes(self: *const HKReader, entry: format.TensorEntry) ![]const u8 {
        return self.getTensorData(entry);
    }

    pub fn getRawF32(self: *const HKReader, entry: format.TensorEntry) ![]const f32 {
        const bytes = try self.getTensorData(entry);
        if ((bytes.len % @sizeOf(f32)) != 0) return error.InvalidByteLength;
        if (!std.mem.isAligned(@intFromPtr(bytes.ptr), @alignOf(f32))) return error.UnalignedData;
        const count = bytes.len / @sizeOf(f32);
        return @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0..count];
    }

    pub fn getRawF16(self: *const HKReader, entry: format.TensorEntry) ![]const f16 {
        const bytes = try self.getTensorData(entry);
        if ((bytes.len % @sizeOf(f16)) != 0) return error.InvalidByteLength;
        if (!std.mem.isAligned(@intFromPtr(bytes.ptr), @alignOf(f16))) return error.UnalignedData;
        const count = bytes.len / @sizeOf(f16);
        return @as([*]const f16, @ptrCast(@alignCast(bytes.ptr)))[0..count];
    }

    pub fn getRawBF16(self: *const HKReader, entry: format.TensorEntry) ![]const u16 {
        const bytes = try self.getTensorData(entry);
        if ((bytes.len % @sizeOf(u16)) != 0) return error.InvalidByteLength;
        if (!std.mem.isAligned(@intFromPtr(bytes.ptr), @alignOf(u16))) return error.UnalignedData;
        const count = bytes.len / @sizeOf(u16);
        return @as([*]const u16, @ptrCast(@alignCast(bytes.ptr)))[0..count];
    }

    pub fn getRawInt8(self: *const HKReader, entry: format.TensorEntry) ![]const i8 {
        const bytes = try self.getTensorData(entry);
        return @as([*]const i8, @ptrCast(bytes.ptr))[0..bytes.len];
    }

    /// Zero-copy pointer to tensor payload directly inside mapped file buffer
    pub fn getTensorData(self: *const HKReader, entry: format.TensorEntry) ![]const u8 {
        if (entry.storage_type == .null_ref) {
            return &[_]u8{};
        }
        const end = entry.data_offset + entry.data_size;
        if (end > self.mmap_region.bytes.len) return error.TensorOutOfBounds;
        return self.mmap_region.bytes[entry.data_offset..end];
    }

    /// Zero-copy pointer to quantization scales
    pub fn getTensorScales(self: *const HKReader, entry: format.TensorEntry) ?[]const u8 {
        if (entry.scale_size == 0) return null;
        const end = entry.scale_offset + entry.scale_size;
        if (end > self.mmap_region.bytes.len) return null;
        return self.mmap_region.bytes[entry.scale_offset..end];
    }

    /// Zero-copy pointer to residual precision recovery buffer
    pub fn getTensorResidual(self: *const HKReader, entry: format.TensorEntry) ?[]const u8 {
        if (entry.residual_size == 0) return null;
        const end = entry.residual_offset + entry.residual_size;
        if (end > self.mmap_region.bytes.len) return null;
        return self.mmap_region.bytes[entry.residual_offset..end];
    }

    /// Zero-copy pointer to raw appendix region if present
    pub fn getAppendixBytes(self: *const HKReader) ?[]const u8 {
        if (self.header.appendix_offset == 0 or self.header.appendix_offset >= self.mmap_region.bytes.len) return null;
        return self.mmap_region.bytes[self.header.appendix_offset..];
    }


    /// Dequantizes and reconstructs tensor elements to f32.
    /// Handles all storage formats, sparsity modes (bitmask, 2:4, BSR), null refs,
    /// and applies dual-mode precision recovery and untiling when applicable.
    pub fn dequantizeToF32(
        self: *const HKReader,
        entry: format.TensorEntry,
        with_residual: bool,
        out: []f32,
    ) !void {
        // 1. Null Reference: 0 bytes stored, expand to zeros
        if (entry.storage_type == .null_ref) {
            @memset(out, 0.0);
            return;
        }

        // 2. Sparsity Decoding
        if (entry.storage_type == .sparse_2_4 or (entry.storage_type == .f32 and entry.sparsity_type == .structured_2_4)) {
            const data_bytes = try self.getTensorData(entry);
            try sparsity.decodeStructured2_4_F32(data_bytes, out.len, out);
            return;
        }

        if (entry.storage_type == .sparse_f16 or (entry.storage_type == .f32 and entry.sparsity_type == .bitmask)) {
            const data_bytes = try self.getTensorData(entry);
            try sparsity.decodeBitmaskF32(data_bytes, out.len, out);
            return;
        }

        if (entry.sparsity_type == .csr and entry.ndim == 2) {
            const data_bytes = try self.getTensorData(entry);
            try sparsity.decodeBSR_F32(data_bytes, @intCast(entry.shape[0]), @intCast(entry.shape[1]), out);
            return;
        }

        // 3. Dense & Quantized Types
        const data_bytes = try self.getTensorData(entry);

        if (entry.storage_type == .f32 or entry.storage_type == .shared_ref) {
            const float_slice: []const f32 = @as([*]const f32, @ptrCast(@alignCast(data_bytes.ptr)))[0..@min(out.len, data_bytes.len / 4)];

            if (entry.tile_layout != .row_major and entry.ndim == 2) {
                try tiling.unpackTilesF32(
                    float_slice,
                    @intCast(entry.shape[0]),
                    @intCast(entry.shape[1]),
                    entry.tile_layout,
                    out,
                );
            } else {
                @memcpy(out[0..float_slice.len], float_slice);
            }
            return;
        }

        if (entry.storage_type == .fp8_e4m3) {
            research.dequantizeFP8_E4M3(data_bytes, out.len, out);
            return;
        }

        if (entry.storage_type == .int8) {
            const i8_slice: []const i8 = @as([*]const i8, @ptrCast(data_bytes.ptr))[0..out.len];
            for (0..out.len) |i| {
                out[i] = @floatFromInt(i8_slice[i]);
            }
            return;
        }

        if (entry.storage_type == .uint8) {
            for (0..out.len) |i| {
                out[i] = @floatFromInt(data_bytes[i]);
            }
            return;
        }

        if (entry.storage_type == .int16) {
            const s = @as([*]const i16, @ptrCast(@alignCast(data_bytes.ptr)))[0..out.len];
            for (0..out.len) |i| {
                out[i] = @floatFromInt(s[i]);
            }
            return;
        }

        if (entry.storage_type == .uint16) {
            const s = @as([*]const u16, @ptrCast(@alignCast(data_bytes.ptr)))[0..out.len];
            for (0..out.len) |i| {
                out[i] = @floatFromInt(s[i]);
            }
            return;
        }

        if (entry.storage_type == .int32) {
            const s = @as([*]const i32, @ptrCast(@alignCast(data_bytes.ptr)))[0..out.len];
            for (0..out.len) |i| {
                out[i] = @floatFromInt(s[i]);
            }
            return;
        }

        if (entry.storage_type == .uint32) {
            const s = @as([*]const u32, @ptrCast(@alignCast(data_bytes.ptr)))[0..out.len];
            for (0..out.len) |i| {
                out[i] = @floatFromInt(s[i]);
            }
            return;
        }

        if (entry.storage_type == .int64) {
            const s = @as([*]const i64, @ptrCast(@alignCast(data_bytes.ptr)))[0..out.len];
            for (0..out.len) |i| {
                out[i] = @floatFromInt(s[i]);
            }
            return;
        }

        if (entry.storage_type == .uint64) {
            const s = @as([*]const u64, @ptrCast(@alignCast(data_bytes.ptr)))[0..out.len];
            for (0..out.len) |i| {
                out[i] = @floatFromInt(s[i]);
            }
            return;
        }

        if (entry.storage_type == .f64) {
            const s = @as([*]const f64, @ptrCast(@alignCast(data_bytes.ptr)))[0..out.len];
            for (0..out.len) |i| {
                out[i] = @floatCast(s[i]);
            }
            return;
        }

        if (entry.storage_type == .bool) {
            for (0..out.len) |i| {
                out[i] = if (data_bytes[i] != 0) 1.0 else 0.0;
            }
            return;
        }

        if (entry.storage_type == .dq4) {
            const scale_bytes = self.getTensorScales(entry) orelse return error.MissingScales;
            const scales: []const f32 = @as([*]const f32, @ptrCast(@alignCast(scale_bytes.ptr)))[0..(scale_bytes.len / 4)];

            const bsize: usize = entry.block_size;
            const num_blocks = (out.len + bsize - 1) / bsize;

            for (0..num_blocks) |b| {
                const start = b * bsize;
                const end = @min(start + bsize, out.len);
                const count = end - start;
                const block_scale = if (b < scales.len) scales[b] else 1.0;
                const packed_block = data_bytes[(b * (bsize / 2))..];

                research.dequantizeBlockNF4(packed_block, block_scale, count, out[start..end]);
            }

            // Apply residual recovery if requested
            if (with_residual) {
                if (self.getTensorResidual(entry)) |res_bytes| {
                    const res_slice: []const f32 = @as([*]const f32, @ptrCast(@alignCast(res_bytes.ptr)))[0..out.len];
                    for (0..out.len) |i| {
                        out[i] += res_slice[i];
                    }
                }
            }
            return;
        }

        if (entry.storage_type == .dq8) {
            const scale_bytes = self.getTensorScales(entry) orelse return error.MissingScales;
            const scales: []const f32 = @as([*]const f32, @ptrCast(@alignCast(scale_bytes.ptr)))[0..(scale_bytes.len / 4)];

            const bsize: usize = entry.block_size;
            const num_blocks = (out.len + bsize - 1) / bsize;
            const i8_data: []const i8 = @as([*]const i8, @ptrCast(data_bytes.ptr))[0..out.len];

            for (0..num_blocks) |b| {
                const start = b * bsize;
                const end = @min(start + bsize, out.len);
                const count = end - start;
                const block_scale = if (b < scales.len) scales[b] else 1.0;

                research.dequantizeBlockDQ8(i8_data[start..end], block_scale, count, out[start..end]);
            }

            if (with_residual) {
                if (self.getTensorResidual(entry)) |res_bytes| {
                    const res_slice: []const f32 = @as([*]const f32, @ptrCast(@alignCast(res_bytes.ptr)))[0..out.len];
                    for (0..out.len) |i| {
                        out[i] += res_slice[i];
                    }
                }
            }
            return;
        }

        if (entry.storage_type == .dqt) {
            const scale_bytes = self.getTensorScales(entry) orelse return error.MissingScales;
            const scales: []const f32 = @as([*]const f32, @ptrCast(@alignCast(scale_bytes.ptr)))[0..(scale_bytes.len / 4)];

            const bsize: usize = entry.block_size;
            const num_blocks = (out.len + bsize - 1) / bsize;

            for (0..num_blocks) |b| {
                const start = b * bsize;
                const end = @min(start + bsize, out.len);
                const count = end - start;
                const block_scale = if (b < scales.len) scales[b] else 1.0;
                const packed_block = data_bytes[(b * (bsize / 4))..];

                research.dequantizeBlockDQT(packed_block, block_scale, count, out[start..end]);
            }

            if (with_residual) {
                if (self.getTensorResidual(entry)) |res_bytes| {
                    const res_slice: []const f32 = @as([*]const f32, @ptrCast(@alignCast(res_bytes.ptr)))[0..out.len];
                    for (0..out.len) |i| {
                        out[i] += res_slice[i];
                    }
                }
            }
            return;
        }

        if (entry.storage_type == .iq4_nl and entry.scale_size != 0) {
            const scale_bytes = self.getTensorScales(entry) orelse return error.MissingScales;
            const scales: []const f32 = @as([*]const f32, @ptrCast(@alignCast(scale_bytes.ptr)))[0..(scale_bytes.len / 4)];
            const bsize: usize = entry.block_size;
            const num_blocks = (out.len + bsize - 1) / bsize;

            for (0..num_blocks) |b| {
                const start = b * bsize;
                const end = @min(start + bsize, out.len);
                const count = end - start;
                const block_scale = if (b < scales.len) scales[b] else 1.0;
                const packed_block = data_bytes[(b * (bsize / 2))..];
                research.dequantizeBlockIQ4_NL(packed_block, block_scale, count, out[start..end]);
            }
            return;
        }

        if (entry.storage_type == .mxfp4 and entry.scale_size != 0) {
            const scale_bytes = self.getTensorScales(entry) orelse return error.MissingScales;
            const bsize: usize = 32;
            const num_blocks = (out.len + bsize - 1) / bsize;

            for (0..num_blocks) |b| {
                const start = b * bsize;
                const end = @min(start + bsize, out.len);
                const count = end - start;
                const e8m0_scale = if (b < scale_bytes.len) scale_bytes[b] else 127;
                const packed_block = data_bytes[(b * 16)..];
                research.dequantizeBlockMXFP4(packed_block, e8m0_scale, count, out[start..end]);
            }
            return;
        }

        if (entry.storage_type == .nvfp4 and entry.scale_size != 0) {
            const scale_bytes = self.getTensorScales(entry) orelse return error.MissingScales;
            const bsize: usize = 16;
            const num_blocks = (out.len + bsize - 1) / bsize;

            for (0..num_blocks) |b| {
                const start = b * bsize;
                const end = @min(start + bsize, out.len);
                const count = end - start;
                const fp8_scale = if (b < scale_bytes.len) scale_bytes[b] else 0x38;
                const packed_block = data_bytes[(b * 8)..];
                research.dequantizeBlockNVFP4(packed_block, fp8_scale, count, out[start..end]);
            }
            return;
        }

        // Everything else that GGUF defines as a block format (Q4_1, Q5_0, Q5_1, the I-quants, IQ4_XS,
        // the ternary types) is decoded by the same routines the engine uses.
        if (quant.blocks.info(entry.storage_type)) |bi| {
            if (out.len % bi.elems != 0) return error.UnsupportedStorageType;
            const need = out.len / bi.elems * bi.bytes;
            if (data_bytes.len < need) return error.UnsupportedStorageType;
            quant.dequant.dequantizeRow(entry.storage_type, data_bytes[0..need], out) catch return error.UnsupportedStorageType;
            return;
        }

        return error.UnsupportedStorageType;
    }
};

fn checkedEnd(offset: u64, size: u64, file_len: usize, comptime bounds_error: anyerror) !usize {
    const end = std.math.add(u64, offset, size) catch return bounds_error;
    if (end > file_len) return bounds_error;
    return @intCast(end);
}

fn validateHeaderLayout(header: *const format.FileHeader, file_len: usize) !void {
    if (header.alignment == 0) return error.InvalidAlignment;
    if (header.metadata_offset < @sizeOf(format.FileHeader)) return error.MetadataOutOfBounds;
    _ = try checkedEnd(header.metadata_offset, header.metadata_size, file_len, error.MetadataOutOfBounds);
    _ = try checkedEnd(header.tensor_toc_offset, header.tensor_toc_size, file_len, error.TOCOutOfBounds);
    if (header.tensor_data_offset > file_len) return error.TensorDataOutOfBounds;
    if (header.appendix_offset != 0 and header.appendix_offset >= file_len) return error.AppendixOutOfBounds;
}

fn validateTensorRanges(toc: *const tensor_toc.TensorTOC, file_len: usize) !void {
    for (toc.entries.items) |entry| {
        if (entry.ndim > format.MAX_DIMS) return error.InvalidTensorDimensions;
        if (entry.storage_type == .null_ref) {
            if (entry.data_offset != 0 or entry.data_size != 0) return error.InvalidNullReference;
            continue;
        }
        _ = try checkedEnd(entry.data_offset, entry.data_size, file_len, error.TensorOutOfBounds);
        if (entry.scale_size != 0) {
            _ = try checkedEnd(entry.scale_offset, entry.scale_size, file_len, error.TensorScaleOutOfBounds);
        }
        if (entry.residual_size != 0) {
            _ = try checkedEnd(entry.residual_offset, entry.residual_size, file_len, error.TensorResidualOutOfBounds);
        }
    }
}
