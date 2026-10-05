const std = @import("std");
const format = @import("format.zig");
const buf = @import("buf.zig");

pub const StandardKeys = struct {
    // General & Provenance
    pub const GENERAL_ARCHITECTURE = "general.architecture";
    pub const GENERAL_NAME = "general.name";
    pub const GENERAL_AUTHOR = "general.author";
    pub const GENERAL_VERSION = "general.version";
    pub const GENERAL_BASE_MODEL = "general.base_model_name";
    pub const GENERAL_LICENSE = "general.license";
    pub const GENERAL_DESCRIPTION = "general.description";

    // Attention
    pub const ATTN_HEAD_COUNT = "attention.head_count";
    pub const ATTN_HEAD_COUNT_KV = "attention.head_count_kv";
    pub const ATTN_KEY_LENGTH_MLA = "attention.key_length_mla";
    pub const ATTN_VALUE_LENGTH_MLA = "attention.value_length_mla";
    pub const ATTN_KEY_LENGTH_SWA = "attention.key_length_swa";
    pub const ATTN_LOGIT_SOFTCAPPING = "attention.attn_logit_softcapping";
    pub const ATTN_LAYER_NORM_RMS_EPS = "attention.layer_norm_rms_epsilon";

    // RoPE
    pub const ROPE_DIMENSION_COUNT = "rope.dimension_count";
    pub const ROPE_FREQ_BASE = "rope.freq_base";
    pub const ROPE_SCALE_TYPE = "rope.scale_type";
    pub const ROPE_YARN_EXT_FACTOR = "rope.yarn_ext_factor";
    pub const ROPE_YARN_ATTN_FACTOR = "rope.yarn_attn_factor";

    // MoE
    pub const MOE_EXPERT_COUNT = "moe.expert_count";
    pub const MOE_EXPERT_USED_COUNT = "moe.expert_used_count";
    pub const MOE_EXPERT_SHARED_COUNT = "moe.expert_shared_count";
    pub const MOE_EXPERT_WEIGHTS_SCALE = "moe.expert_weights_scale";

    // SSM
    pub const SSM_CONV_KERNEL = "ssm.conv_kernel";
    pub const SSM_INNER_SIZE = "ssm.inner_size";
    pub const SSM_STATE_SIZE = "ssm.state_size";

    // Tokenizer
    pub const TOKENIZER_MODEL = "tokenizer.ggml.model";
    pub const TOKENIZER_PRE = "tokenizer.ggml.pre";
    pub const TOKENIZER_TOKENS = "tokenizer.tokens";
    pub const TOKENIZER_SCORES = "tokenizer.scores";
    pub const TOKENIZER_TOKEN_TYPE = "tokenizer.token_type";
    pub const TOKENIZER_MERGES = "tokenizer.merges";
    pub const TOKENIZER_CHAT_TEMPLATE = "tokenizer.chat_template";
    pub const TOKENIZER_PRECOMPILED_CHARSMAP = "tokenizer.ggml.precompiled_charsmap";

    // Sampling
    pub const SAMPLING_TEMP = "sampling.temp";
    pub const SAMPLING_TOP_P = "sampling.top_p";
    pub const SAMPLING_TOP_K = "sampling.top_k";
    pub const SAMPLING_MIN_P = "sampling.min_p";
    pub const SAMPLING_XTC_PROB = "sampling.xtc_probability";

    // Quantization
    pub const QUANT_RECIPE = "quantization.recipe";
    pub const QUANT_VERSION = "quantization.version";
    pub const QUANT_IMATRIX_DATASET = "quantization.imatrix_dataset";
};

pub const MetadataValue = union(format.MetadataValueType) {
    val_string: []const u8,
    val_int64: i64,
    val_float64: f64,
    val_bool: bool,
    val_json: []const u8,
    val_bytes: []const u8,
};

pub const MetadataItem = struct {
    key: []const u8,
    value: MetadataValue,
};

pub const MetadataMap = struct {
    items: std.ArrayList(MetadataItem) = .empty,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) MetadataMap {
        return .{
            .items = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MetadataMap) void {
        for (self.items.items) |item| {
            self.allocator.free(item.key);
            switch (item.value) {
                .val_string => |s| self.allocator.free(s),
                .val_json => |j| self.allocator.free(j),
                .val_bytes => |b| self.allocator.free(b),
                else => {},
            }
        }
        self.items.deinit(self.allocator);
    }

    pub fn set(self: *MetadataMap, key: []const u8, value: MetadataValue) !void {
        for (self.items.items) |*item| {
            if (std.mem.eql(u8, item.key, key)) {
                switch (item.value) {
                    .val_string => |s| self.allocator.free(s),
                    .val_json => |j| self.allocator.free(j),
                    .val_bytes => |b| self.allocator.free(b),
                    else => {},
                }
                item.value = value;
                return;
            }
        }
        const k = try self.allocator.dupe(u8, key);
        try self.items.append(self.allocator, .{
            .key = k,
            .value = value,
        });
    }

    pub fn setString(self: *MetadataMap, key: []const u8, val: []const u8) !void {
        const v = try self.allocator.dupe(u8, val);
        errdefer self.allocator.free(v);
        try self.set(key, .{ .val_string = v });
    }

    pub fn setJson(self: *MetadataMap, key: []const u8, json_str: []const u8) !void {
        const v = try self.allocator.dupe(u8, json_str);
        errdefer self.allocator.free(v);
        try self.set(key, .{ .val_json = v });
    }

    pub fn setInt(self: *MetadataMap, key: []const u8, val: i64) !void {
        try self.set(key, .{ .val_int64 = val });
    }

    pub fn setFloat(self: *MetadataMap, key: []const u8, val: f64) !void {
        try self.set(key, .{ .val_float64 = val });
    }

    pub fn setBool(self: *MetadataMap, key: []const u8, val: bool) !void {
        try self.set(key, .{ .val_bool = val });
    }

    pub fn get(self: *const MetadataMap, key: []const u8) ?MetadataValue {
        for (self.items.items) |item| {
            if (std.mem.eql(u8, item.key, key)) {
                return item.value;
            }
        }
        return null;
    }

    pub fn getInt(self: *const MetadataMap, key: []const u8) ?i64 {
        const val = self.get(key) orelse return null;
        return switch (val) {
            .val_int64 => |v| v,
            else => null,
        };
    }

    pub fn getFloat(self: *const MetadataMap, key: []const u8) ?f64 {
        const val = self.get(key) orelse return null;
        return switch (val) {
            .val_float64 => |v| v,
            else => null,
        };
    }

    pub fn getString(self: *const MetadataMap, key: []const u8) ?[]const u8 {
        const val = self.get(key) orelse return null;
        return switch (val) {
            .val_string => |v| v,
            .val_json => |j| j,
            else => null,
        };
    }

    pub fn find(self: *const MetadataMap, query_key: []const u8) ?MetadataValue {
        for (self.items.items) |item| {
            if (std.mem.eql(u8, item.key, query_key)) {
                return item.value;
            }
        }
        for (self.items.items) |item| {
            if (std.mem.endsWith(u8, item.key, query_key)) {
                const prefix_len = item.key.len - query_key.len;
                if (prefix_len == 0 or item.key[prefix_len - 1] == '.') {
                    return item.value;
                }
            }
        }
        return null;
    }

    pub fn findInt(self: *const MetadataMap, query_key: []const u8) ?i64 {
        const val = self.find(query_key) orelse return null;
        return switch (val) {
            .val_int64 => |v| v,
            else => null,
        };
    }

    pub fn findFloat(self: *const MetadataMap, query_key: []const u8) ?f64 {
        const val = self.find(query_key) orelse return null;
        return switch (val) {
            .val_float64 => |v| v,
            .val_int64 => |v| @floatFromInt(v),
            else => null,
        };
    }

    pub fn findString(self: *const MetadataMap, query_key: []const u8) ?[]const u8 {
        const val = self.find(query_key) orelse return null;
        return switch (val) {
            .val_string => |v| v,
            .val_json => |j| j,
            else => null,
        };
    }

    /// Serializes metadata map into a byte array
    pub fn serialize(self: *const MetadataMap, writer: *buf.BufferWriter) !void {
        for (self.items.items) |item| {
            // Write key len (u16)
            const klen: u16 = @intCast(item.key.len);
            try writer.writeU16(klen);
            try writer.writeBytes(item.key);

            // Write type tag (u8)
            const tag: u8 = @intFromEnum(item.value);
            try writer.writeU8(tag);

            switch (item.value) {
                .val_string => |s| {
                    const vlen: u32 = @intCast(s.len);
                    try writer.writeU32(vlen);
                    try writer.writeBytes(s);
                },
                .val_json => |j| {
                    const vlen: u32 = @intCast(j.len);
                    try writer.writeU32(vlen);
                    try writer.writeBytes(j);
                },
                .val_bytes => |b| {
                    const vlen: u32 = @intCast(b.len);
                    try writer.writeU32(vlen);
                    try writer.writeBytes(b);
                },
                .val_int64 => |v| {
                    try writer.writeU32(8);
                    try writer.writeI64(v);
                },
                .val_float64 => |f| {
                    try writer.writeU32(8);
                    try writer.writeF64(f);
                },
                .val_bool => |b| {
                    try writer.writeU32(1);
                    try writer.writeU8(if (b) 1 else 0);
                },
            }
        }
    }

    /// Deserializes metadata from a byte buffer
    pub fn deserialize(buffer: []const u8, kv_count: usize, allocator: std.mem.Allocator) !MetadataMap {
        var map = MetadataMap.init(allocator);
        errdefer map.deinit();
        var reader = buf.BufferReader.init(buffer);

        var i: usize = 0;
        while (i < kv_count) : (i += 1) {
            const klen = try reader.readU16();
            const key_raw = try reader.readBytes(klen);
            const key = try allocator.dupe(u8, key_raw);
            errdefer allocator.free(key);

            const tag_byte = try reader.readU8();
            const tag: format.MetadataValueType = @enumFromInt(tag_byte);

            const vlen = try reader.readU32();
            const val: MetadataValue = switch (tag) {
                .val_string => blk: {
                    const s = try reader.readBytes(vlen);
                    break :blk .{ .val_string = try allocator.dupe(u8, s) };
                },
                .val_json => blk: {
                    const s = try reader.readBytes(vlen);
                    break :blk .{ .val_json = try allocator.dupe(u8, s) };
                },
                .val_bytes => blk: {
                    const b = try reader.readBytes(vlen);
                    break :blk .{ .val_bytes = try allocator.dupe(u8, b) };
                },
                .val_int64 => blk: {
                    const v = try reader.readI64();
                    break :blk .{ .val_int64 = v };
                },
                .val_float64 => blk: {
                    const f = try reader.readF64();
                    break :blk .{ .val_float64 = f };
                },
                .val_bool => blk: {
                    const b = (try reader.readU8()) != 0;
                    break :blk .{ .val_bool = b };
                },
            };

            try map.items.append(allocator, .{
                .key = key,
                .value = val,
            });
        }
        return map;
    }
};

const platform = @import("platform.zig");

/// Updates or inserts a key-value pair directly in the file's metadata table in-place
/// without rewriting the multi-gigabyte tensor payload.
pub fn patchFileMetadataInPlace(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    key: []const u8,
    val_str: []const u8,
) !void {
    var region = try platform.mapOrReadFile(file_path, allocator);
    defer region.deinit(allocator);

    if (region.bytes.len < @sizeOf(format.FileHeader)) {
        return error.FileTooSmall;
    }

    const header_ptr: *const format.FileHeader = @ptrCast(@alignCast(region.bytes.ptr));
    if (!header_ptr.isValid()) {
        return error.InvalidMagicOrVersion;
    }

    var header = header_ptr.*;

    // Deserialize existing metadata
    const meta_end = header.metadata_offset + header.metadata_size;
    if (meta_end > region.bytes.len) return error.MetadataOutOfBounds;
    const meta_buf = region.bytes[header.metadata_offset..meta_end];
    var meta_map = try MetadataMap.deserialize(meta_buf, header.metadata_kv_count, allocator);
    defer meta_map.deinit();

    // Parse value type and set into meta_map
    var is_json = false;
    if (std.mem.startsWith(u8, val_str, "{") or std.mem.startsWith(u8, val_str, "[")) {
        if (std.json.parseFromSlice(std.json.Value, allocator, val_str, .{})) |parsed| {
            parsed.deinit();
            is_json = true;
        } else |_| {}
    }

    if (is_json) {
        try meta_map.setJson(key, val_str);
    } else if (std.mem.eql(u8, val_str, "true")) {
        try meta_map.setBool(key, true);
    } else if (std.mem.eql(u8, val_str, "false")) {
        try meta_map.setBool(key, false);
    } else if (std.fmt.parseInt(i64, val_str, 10)) |iv| {
        try meta_map.setInt(key, iv);
    } else |_| if (std.fmt.parseFloat(f64, val_str)) |fv| {
        try meta_map.setFloat(key, fv);
    } else |_| {
        try meta_map.setString(key, val_str);
    }

    // Serialize new metadata
    var meta_writer = buf.BufferWriter.init(allocator);
    defer meta_writer.deinit();
    try meta_map.serialize(&meta_writer);
    const new_meta_bytes = meta_writer.getBytes();

    // Copy existing TOC
    const toc_end = header.tensor_toc_offset + header.tensor_toc_size;
    if (toc_end > region.bytes.len) return error.TOCOutOfBounds;
    // Copied out of the mapping: the metadata below is written over the front of the old TOC when
    // it grows, and a view into the mapped file would change under us.
    const toc_bytes = try allocator.dupe(u8, region.bytes[header.tensor_toc_offset..toc_end]);
    defer allocator.free(toc_bytes);

    const header_size = @sizeOf(format.FileHeader);
    const needed_space = new_meta_bytes.len + toc_bytes.len;
    const available_space = header.tensor_data_offset - header_size;

    if (needed_space <= available_space) {
        // Fits before tensor payloads!
        // Overwrite header, metadata, TOC in-place.
        const new_meta_offset = header_size;
        const new_toc_offset = new_meta_offset + new_meta_bytes.len;

        header.metadata_kv_count = meta_map.items.items.len;
        header.metadata_offset = new_meta_offset;
        header.metadata_size = new_meta_bytes.len;
        header.tensor_toc_offset = new_toc_offset;
        header.tensor_toc_size = toc_bytes.len;

        // Write header
        const header_bytes = std.mem.asBytes(&header);
        try platform.writeBytesAtOffset(file_path, header_bytes, 0, allocator);

        // Write new metadata
        try platform.writeBytesAtOffset(file_path, new_meta_bytes, new_meta_offset, allocator);

        // Write TOC
        try platform.writeBytesAtOffset(file_path, toc_bytes, new_toc_offset, allocator);

        // Zero out padding between end of TOC and tensor_data_offset
        const pad_start = new_toc_offset + toc_bytes.len;
        if (header.tensor_data_offset > pad_start) {
            const pad_len = header.tensor_data_offset - pad_start;
            const zeros = try allocator.alloc(u8, pad_len);
            defer allocator.free(zeros);
            @memset(zeros, 0);
            try platform.writeBytesAtOffset(file_path, zeros, pad_start, allocator);
        }
    } else {
        // If metadata exceeds pre-tensor space, append metadata to EOF. Appendix records run
        // to end of file, so trailing metadata bytes would be misparsed as a record.
        if (header.appendix_offset != 0) return error.MetadataExceedsPaddingWithAppendix;
        const new_meta_offset = region.bytes.len;
        header.metadata_kv_count = meta_map.items.items.len;
        header.metadata_offset = new_meta_offset;
        header.metadata_size = new_meta_bytes.len;

        // Write new metadata at EOF
        try platform.writeBytesAtOffset(file_path, new_meta_bytes, new_meta_offset, allocator);

        // Update header at offset 0
        const header_bytes = std.mem.asBytes(&header);
        try platform.writeBytesAtOffset(file_path, header_bytes, 0, allocator);
    }
}
