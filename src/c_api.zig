const std = @import("std");
const format = @import("format.zig");
const reader_mod = @import("reader.zig");
const writer_mod = @import("writer.zig");
const appendix_mod = @import("appendix.zig");
const nf4 = @import("nf4.zig");
const quantization = @import("quantization.zig");
const sparsity = @import("sparsity.zig");
const tiling = @import("tiling.zig");
const tensor_ops_mod = @import("tensor_ops.zig");
const growth_mod = @import("growth.zig");
const tokenizer_mod = @import("tokenizer.zig");
const engine_mod = @import("engine.zig");
const sampler_mod = @import("sampler.zig");
const safetensors_mod = @import("safetensors.zig");
const hf_mapper_mod = @import("hf_mapper.zig");
const context_mod = @import("context.zig");
const pipeline_mod = @import("pipeline.zig");
const adaptive_mod = @import("adaptive.zig");
const platform_mod = @import("platform.zig");

pub const hk_reader_t = opaque {};

const ReaderWrapper = struct {
    reader: reader_mod.HKReader,
    appendix_reader: ?appendix_mod.AppendixReader = null,
    /// NUL terminated copies of every appendix record's name and target (name at 2i, target at
    /// 2i + 1). The records in the file are length prefixed, so the C side cannot read them as
    /// C strings directly.
    appendix_cstrs: std.ArrayList([:0]u8) = .empty,
    allocator: std.mem.Allocator,
};

pub export fn hk_open(path_c: [*:0]const u8) ?*hk_reader_t {
    const path = std.mem.sliceTo(path_c, 0);
    const allocator = std.heap.page_allocator;

    const wrapper = allocator.create(ReaderWrapper) catch return null;
    wrapper.allocator = allocator;
    wrapper.reader = reader_mod.HKReader.open(path, allocator) catch {
        allocator.destroy(wrapper);
        return null;
    };
    wrapper.appendix_cstrs = .empty;
    wrapper.appendix_reader = appendix_mod.AppendixReader.init(allocator, wrapper.reader.mmap_region.bytes) catch null;
    if (wrapper.appendix_reader) |ar| {
        for (ar.records.items) |rec| {
            for ([_][]const u8{ rec.name, rec.target }) |text| {
                const copy = allocator.dupeSentinel(u8, text, 0) catch {
                    hk_close(@ptrCast(wrapper));
                    return null;
                };
                wrapper.appendix_cstrs.append(allocator, copy) catch {
                    allocator.free(copy);
                    hk_close(@ptrCast(wrapper));
                    return null;
                };
            }
        }
    }

    return @ptrCast(wrapper);
}

pub export fn hk_close(reader_ptr: ?*hk_reader_t) void {
    if (reader_ptr == null) return;
    const wrapper: *ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (wrapper.appendix_reader) |*ar| {
        ar.deinit();
    }
    for (wrapper.appendix_cstrs.items) |c| wrapper.allocator.free(c);
    wrapper.appendix_cstrs.deinit(wrapper.allocator);
    wrapper.reader.deinit();
    wrapper.allocator.destroy(wrapper);
}

pub export fn hk_get_tensor_count(reader_ptr: ?*const hk_reader_t) u64 {
    if (reader_ptr == null) return 0;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    return wrapper.reader.toc.entries.items.len;
}

pub const C_TensorInfo = extern struct {
    name: [*:0]const u8,
    storage_type: u8,
    tile_layout: u8,
    sparsity_type: u8,
    ndim: u8,
    shape: [8]u64,
    data_offset: u64,
    data_size: u64,
    residual_offset: u64,
    residual_size: u64,
    scale_offset: u64,
    scale_size: u64,
    block_size: u16,
    sparsity_ratio: f32,
};

pub export fn hk_get_tensor_info(reader_ptr: ?*const hk_reader_t, index: u64, out_info: ?*C_TensorInfo) c_int {
    if (reader_ptr == null or out_info == null) return -1;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (index >= wrapper.reader.toc.entries.items.len) return -1;

    const entry = wrapper.reader.toc.entries.items[index];
    out_info.?.* = .{
        .name = @ptrCast(entry.name.ptr),
        .storage_type = @intFromEnum(entry.storage_type),
        .tile_layout = @intFromEnum(entry.tile_layout),
        .sparsity_type = @intFromEnum(entry.sparsity_type),
        .ndim = entry.ndim,
        .shape = entry.shape,
        .data_offset = entry.data_offset,
        .data_size = entry.data_size,
        .residual_offset = entry.residual_offset,
        .residual_size = entry.residual_size,
        .scale_offset = entry.scale_offset,
        .scale_size = entry.scale_size,
        .block_size = entry.block_size,
        .sparsity_ratio = entry.sparsity_ratio,
    };
    return 0;
}

pub export fn hk_get_all_tensor_infos(reader_ptr: ?*const hk_reader_t, out_infos: ?[*]C_TensorInfo, max_count: u64) u64 {
    if (reader_ptr == null or out_infos == null) return 0;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    const total: u64 = @intCast(wrapper.reader.toc.entries.items.len);
    const count = @min(total, max_count);
    for (0..count) |i| {
        const entry = wrapper.reader.toc.entries.items[i];
        out_infos.?[i] = .{
            .name = @ptrCast(entry.name.ptr),
            .storage_type = @intFromEnum(entry.storage_type),
            .tile_layout = @intFromEnum(entry.tile_layout),
            .sparsity_type = @intFromEnum(entry.sparsity_type),
            .ndim = entry.ndim,
            .shape = entry.shape,
            .data_offset = entry.data_offset,
            .data_size = entry.data_size,
            .residual_offset = entry.residual_offset,
            .residual_size = entry.residual_size,
            .scale_offset = entry.scale_offset,
            .scale_size = entry.scale_size,
            .block_size = entry.block_size,
            .sparsity_ratio = entry.sparsity_ratio,
        };
    }
    return count;
}

pub export fn hk_get_tensor_data(reader_ptr: ?*const hk_reader_t, index: u64, out_size: ?*u64) ?*const anyopaque {
    if (reader_ptr == null) return null;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (index >= wrapper.reader.toc.entries.items.len) return null;

    const entry = wrapper.reader.toc.entries.items[index];
    const data = wrapper.reader.getTensorData(entry) catch return null;
    if (out_size) |sz| {
        sz.* = data.len;
    }
    return @ptrCast(data.ptr);
}

pub export fn hk_get_tensor_residual(reader_ptr: ?*const hk_reader_t, index: u64, out_size: ?*u64) ?*const anyopaque {
    if (reader_ptr == null) return null;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (index >= wrapper.reader.toc.entries.items.len) return null;

    const entry = wrapper.reader.toc.entries.items[index];
    const res = wrapper.reader.getTensorResidual(entry) orelse return null;
    if (out_size) |sz| {
        sz.* = res.len;
    }
    return @ptrCast(res.ptr);
}

pub export fn hk_get_tensor_scales(reader_ptr: ?*const hk_reader_t, index: u64, out_size: ?*u64) ?*const anyopaque {
    if (reader_ptr == null) return null;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (index >= wrapper.reader.toc.entries.items.len) return null;

    const entry = wrapper.reader.toc.entries.items[index];
    const sc = wrapper.reader.getTensorScales(entry) orelse return null;
    if (out_size) |sz| {
        sz.* = sc.len;
    }
    return @ptrCast(sc.ptr);
}

pub export fn hk_dequantize_f32(
    reader_ptr: ?*const hk_reader_t,
    index: u64,
    with_residual: c_int,
    out_buf: ?[*]f32,
    count: u64,
) c_int {
    if (reader_ptr == null or out_buf == null) return -1;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (index >= wrapper.reader.toc.entries.items.len) return -1;

    const entry = wrapper.reader.toc.entries.items[index];
    const out_slice = out_buf.?[0..count];
    wrapper.reader.dequantizeToF32(entry, with_residual != 0, out_slice) catch return -1;
    return 0;
}

pub export fn hk_get_metadata_string(reader_ptr: ?*const hk_reader_t, key_c: [*:0]const u8) ?[*:0]const u8 {
    if (reader_ptr == null) return null;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    const key = std.mem.sliceTo(key_c, 0);

    const val = wrapper.reader.metadata_map.get(key) orelse return null;
    return switch (val) {
        .val_string => |s| @ptrCast(s.ptr),
        .val_json => |j| @ptrCast(j.ptr),
        else => null,
    };
}

pub export fn hk_get_metadata_int(reader_ptr: ?*const hk_reader_t, key_c: [*:0]const u8, out_val: ?*i64) c_int {
    if (reader_ptr == null or out_val == null) return -1;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    const key = std.mem.sliceTo(key_c, 0);

    const val = wrapper.reader.metadata_map.get(key) orelse return -1;
    switch (val) {
        .val_int64 => |v| {
            out_val.?.* = v;
            return 0;
        },
        else => return -1,
    }
}

pub export fn hk_get_metadata_float(reader_ptr: ?*const hk_reader_t, key_c: [*:0]const u8, out_val: ?*f64) c_int {
    if (reader_ptr == null or out_val == null) return -1;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    const key = std.mem.sliceTo(key_c, 0);

    const val = wrapper.reader.metadata_map.get(key) orelse return -1;
    switch (val) {
        .val_float64 => |f| {
            out_val.?.* = f;
            return 0;
        },
        else => return -1,
    }
}

pub export fn hk_get_metadata_bool(reader_ptr: ?*const hk_reader_t, key_c: [*:0]const u8, out_val: ?*c_int) c_int {
    if (reader_ptr == null or out_val == null) return -1;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    const key = std.mem.sliceTo(key_c, 0);

    const val = wrapper.reader.metadata_map.get(key) orelse return -1;
    switch (val) {
        .val_bool => |b| {
            out_val.?.* = if (b) 1 else 0;
            return 0;
        },
        else => return -1,
    }
}

pub export fn hk_quantize_block_nf4(
    block: [*]const f32,
    count: u32,
    packed_out: [*]u8,
    residual_out: ?[*]f32,
) f32 {
    const block_slice = block[0..count];
    const packed_len = (count + 1) / 2;
    const packed_slice = packed_out[0..packed_len];
    const res_slice: ?[]f32 = if (residual_out) |r| r[0..count] else null;
    return quantization.quantizeBlockNF4(block_slice, packed_slice, res_slice);
}

pub export fn hk_quantize_block_dq8(
    block: [*]const f32,
    count: u32,
    out_i8: [*]i8,
    residual_out: ?[*]f32,
) f32 {
    const block_slice = block[0..count];
    const i8_slice = out_i8[0..count];
    const res_slice: ?[]f32 = if (residual_out) |r| r[0..count] else null;
    return quantization.quantizeBlockDQ8(block_slice, i8_slice, res_slice);
}

pub export fn hk_quantize_block_dqt(
    block: [*]const f32,
    count: u32,
    packed_out: [*]u8,
    residual_out: ?[*]f32,
) f32 {
    const block_slice = block[0..count];
    const packed_len = (count + 3) / 4;
    const packed_slice = packed_out[0..packed_len];
    const res_slice: ?[]f32 = if (residual_out) |r| r[0..count] else null;
    return quantization.quantizeBlockDQT(block_slice, packed_slice, res_slice);
}

pub export fn hk_quantize_block_q4_k(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ4_K) c_int {
    if (count != quantization.QK_K) return -1;
    quantization.quantizeSuperBlockQ4_K(weights[0..quantization.QK_K], out_block);
    return 0;
}

pub export fn hk_quantize_block_q4_0(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ4_0) c_int {
    if (count != quantization.QK4_0) return -1;
    quantization.quantizeBlockQ4_0(weights[0..quantization.QK4_0], out_block);
    return 0;
}

pub export fn hk_dequantize_block_q4_0(block: *const quantization.BlockQ4_0, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeBlockQ4_0(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_quantize_block_q8_0(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ8_0) c_int {
    if (count != quantization.QK8_0) return -1;
    quantization.quantizeBlockQ8_0(weights[0..quantization.QK8_0], out_block);
    return 0;
}

pub export fn hk_dequantize_block_q8_0(block: *const quantization.BlockQ8_0, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeBlockQ8_0(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_quantize_block_q5_k(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ5_K) c_int {
    if (count != quantization.QK_K) return -1;
    quantization.quantizeSuperBlockQ5_K(weights[0..quantization.QK_K], out_block);
    return 0;
}

pub export fn hk_dequantize_block_q5_k(block: *const quantization.BlockQ5_K, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeSuperBlockQ5_K(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_quantize_block_q3_k(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ3_K) c_int {
    if (count != quantization.QK_K) return -1;
    quantization.quantizeSuperBlockQ3_K(weights[0..quantization.QK_K], out_block);
    return 0;
}

pub export fn hk_dequantize_block_q3_k(block: *const quantization.BlockQ3_K, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeSuperBlockQ3_K(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_quantize_block_q6_k(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ6_K) c_int {
    if (count != quantization.QK_K) return -1;
    quantization.quantizeSuperBlockQ6_K(weights[0..quantization.QK_K], out_block);
    return 0;
}

pub export fn hk_quantize_block_q2_k(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ2_K) c_int {
    if (count != quantization.QK_K) return -1;
    quantization.quantizeSuperBlockQ2_K(weights[0..quantization.QK_K], out_block);
    return 0;
}

pub export fn hk_dequantize_block_q4_k(block: *const quantization.BlockQ4_K, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeSuperBlockQ4_K(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_quantize_block_q8_k(weights: [*]const f32, count: u32, out_block: *quantization.BlockQ8_K) c_int {
    if (count != quantization.QK_K) return -1;
    quantization.quantizeSuperBlockQ8_K(weights[0..quantization.QK_K], out_block);
    return 0;
}

pub export fn hk_dequantize_block_q8_k(block: *const quantization.BlockQ8_K, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeSuperBlockQ8_K(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_dequantize_block_q6_k(block: *const quantization.BlockQ6_K, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeSuperBlockQ6_K(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_dequantize_block_q2_k(block: *const quantization.BlockQ2_K, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeSuperBlockQ2_K(block, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_dequantize_block_iq4_nl(packed_in: [*]const u8, scale: f32, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeBlockIQ4_NL(packed_in[0..(count + 1) / 2], scale, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_dequantize_block_mxfp4(packed_in: [*]const u8, scale_e8m0: u8, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeBlockMXFP4(packed_in[0..(count + 1) / 2], scale_e8m0, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_dequantize_block_nvfp4(packed_in: [*]const u8, scale_fp8: u8, count: u32, out_f32: [*]f32) c_int {
    quantization.dequantizeBlockNVFP4(packed_in[0..(count + 1) / 2], scale_fp8, count, out_f32[0..count]);
    return 0;
}

pub export fn hk_quantize_tensor_q4_0(weights: [*]const f32, count: u64, out_bytes: [*]u8) c_int {
    if (count % quantization.QK4_0 != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK4_0);
    const blocks: [*]quantization.BlockQ4_0 = @ptrCast(@alignCast(out_bytes));
    for (0..num_blocks) |b| {
        quantization.quantizeBlockQ4_0(weights[b * quantization.QK4_0 .. (b + 1) * quantization.QK4_0], &blocks[b]);
    }
    return 0;
}

pub export fn hk_dequantize_tensor_q4_0(in_bytes: [*]const u8, count: u64, out_f32: [*]f32) c_int {
    if (count % quantization.QK4_0 != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK4_0);
    const blocks: [*]const quantization.BlockQ4_0 = @ptrCast(@alignCast(in_bytes));
    for (0..num_blocks) |b| {
        quantization.dequantizeBlockQ4_0(&blocks[b], quantization.QK4_0, out_f32[b * quantization.QK4_0 .. (b + 1) * quantization.QK4_0]);
    }
    return 0;
}

pub export fn hk_quantize_tensor_q8_0(weights: [*]const f32, count: u64, out_bytes: [*]u8) c_int {
    if (count % quantization.QK8_0 != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK8_0);
    const blocks: [*]quantization.BlockQ8_0 = @ptrCast(@alignCast(out_bytes));
    for (0..num_blocks) |b| {
        quantization.quantizeBlockQ8_0(weights[b * quantization.QK8_0 .. (b + 1) * quantization.QK8_0], &blocks[b]);
    }
    return 0;
}

pub export fn hk_dequantize_tensor_q8_0(in_bytes: [*]const u8, count: u64, out_f32: [*]f32) c_int {
    if (count % quantization.QK8_0 != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK8_0);
    const blocks: [*]const quantization.BlockQ8_0 = @ptrCast(@alignCast(in_bytes));
    for (0..num_blocks) |b| {
        quantization.dequantizeBlockQ8_0(&blocks[b], quantization.QK8_0, out_f32[b * quantization.QK8_0 .. (b + 1) * quantization.QK8_0]);
    }
    return 0;
}

pub export fn hk_quantize_tensor_q4_k(weights: [*]const f32, count: u64, out_bytes: [*]u8) c_int {
    if (count % quantization.QK_K != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK_K);
    const blocks: [*]quantization.BlockQ4_K = @ptrCast(@alignCast(out_bytes));
    for (0..num_blocks) |b| {
        quantization.quantizeSuperBlockQ4_K(weights[b * quantization.QK_K .. (b + 1) * quantization.QK_K], &blocks[b]);
    }
    return 0;
}

pub export fn hk_dequantize_tensor_q4_k(in_bytes: [*]const u8, count: u64, out_f32: [*]f32) c_int {
    if (count % quantization.QK_K != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK_K);
    const blocks: [*]const quantization.BlockQ4_K = @ptrCast(@alignCast(in_bytes));
    for (0..num_blocks) |b| {
        quantization.dequantizeSuperBlockQ4_K(&blocks[b], quantization.QK_K, out_f32[b * quantization.QK_K .. (b + 1) * quantization.QK_K]);
    }
    return 0;
}

pub export fn hk_quantize_tensor_q8_k(weights: [*]const f32, count: u64, out_bytes: [*]u8) c_int {
    if (count % quantization.QK_K != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK_K);
    const blocks: [*]quantization.BlockQ8_K = @ptrCast(@alignCast(out_bytes));
    for (0..num_blocks) |b| {
        quantization.quantizeSuperBlockQ8_K(weights[b * quantization.QK_K .. (b + 1) * quantization.QK_K], &blocks[b]);
    }
    return 0;
}

pub export fn hk_dequantize_tensor_q8_k(in_bytes: [*]const u8, count: u64, out_f32: [*]f32) c_int {
    if (count % quantization.QK_K != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK_K);
    const blocks: [*]const quantization.BlockQ8_K = @ptrCast(@alignCast(in_bytes));
    for (0..num_blocks) |b| {
        quantization.dequantizeSuperBlockQ8_K(&blocks[b], quantization.QK_K, out_f32[b * quantization.QK_K .. (b + 1) * quantization.QK_K]);
    }
    return 0;
}

pub export fn hk_quantize_tensor_q6_k(weights: [*]const f32, count: u64, out_bytes: [*]u8) c_int {
    if (count % quantization.QK_K != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK_K);
    const blocks: [*]quantization.BlockQ6_K = @ptrCast(@alignCast(out_bytes));
    for (0..num_blocks) |b| {
        quantization.quantizeSuperBlockQ6_K(weights[b * quantization.QK_K .. (b + 1) * quantization.QK_K], &blocks[b]);
    }
    return 0;
}

pub export fn hk_dequantize_tensor_q6_k(in_bytes: [*]const u8, count: u64, out_f32: [*]f32) c_int {
    if (count % quantization.QK_K != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK_K);
    const blocks: [*]const quantization.BlockQ6_K = @ptrCast(@alignCast(in_bytes));
    for (0..num_blocks) |b| {
        quantization.dequantizeSuperBlockQ6_K(&blocks[b], quantization.QK_K, out_f32[b * quantization.QK_K .. (b + 1) * quantization.QK_K]);
    }
    return 0;
}

pub export fn hk_quantize_tensor_q5_k(weights: [*]const f32, count: u64, out_bytes: [*]u8) c_int {
    if (count % quantization.QK_K != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK_K);
    const blocks: [*]quantization.BlockQ5_K = @ptrCast(@alignCast(out_bytes));
    for (0..num_blocks) |b| {
        quantization.quantizeSuperBlockQ5_K(weights[b * quantization.QK_K .. (b + 1) * quantization.QK_K], &blocks[b]);
    }
    return 0;
}

pub export fn hk_dequantize_tensor_q5_k(in_bytes: [*]const u8, count: u64, out_f32: [*]f32) c_int {
    if (count % quantization.QK_K != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK_K);
    const blocks: [*]const quantization.BlockQ5_K = @ptrCast(@alignCast(in_bytes));
    for (0..num_blocks) |b| {
        quantization.dequantizeSuperBlockQ5_K(&blocks[b], quantization.QK_K, out_f32[b * quantization.QK_K .. (b + 1) * quantization.QK_K]);
    }
    return 0;
}

pub export fn hk_quantize_tensor_q3_k(weights: [*]const f32, count: u64, out_bytes: [*]u8) c_int {
    if (count % quantization.QK_K != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK_K);
    const blocks: [*]quantization.BlockQ3_K = @ptrCast(@alignCast(out_bytes));
    for (0..num_blocks) |b| {
        quantization.quantizeSuperBlockQ3_K(weights[b * quantization.QK_K .. (b + 1) * quantization.QK_K], &blocks[b]);
    }
    return 0;
}

pub export fn hk_dequantize_tensor_q3_k(in_bytes: [*]const u8, count: u64, out_f32: [*]f32) c_int {
    if (count % quantization.QK_K != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK_K);
    const blocks: [*]const quantization.BlockQ3_K = @ptrCast(@alignCast(in_bytes));
    for (0..num_blocks) |b| {
        quantization.dequantizeSuperBlockQ3_K(&blocks[b], quantization.QK_K, out_f32[b * quantization.QK_K .. (b + 1) * quantization.QK_K]);
    }
    return 0;
}

pub export fn hk_quantize_tensor_q2_k(weights: [*]const f32, count: u64, out_bytes: [*]u8) c_int {
    if (count % quantization.QK_K != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK_K);
    const blocks: [*]quantization.BlockQ2_K = @ptrCast(@alignCast(out_bytes));
    for (0..num_blocks) |b| {
        quantization.quantizeSuperBlockQ2_K(weights[b * quantization.QK_K .. (b + 1) * quantization.QK_K], &blocks[b]);
    }
    return 0;
}

pub export fn hk_dequantize_tensor_q2_k(in_bytes: [*]const u8, count: u64, out_f32: [*]f32) c_int {
    if (count % quantization.QK_K != 0) return -1;
    const num_blocks: usize = @intCast(count / quantization.QK_K);
    const blocks: [*]const quantization.BlockQ2_K = @ptrCast(@alignCast(in_bytes));
    for (0..num_blocks) |b| {
        quantization.dequantizeSuperBlockQ2_K(&blocks[b], quantization.QK_K, out_f32[b * quantization.QK_K .. (b + 1) * quantization.QK_K]);
    }
    return 0;
}

pub export fn hk_quantize_tensor_nf4(
    weights: [*]const f32,
    count: u64,
    block_size: u32,
    packed_out: [*]u8,
    scales_out: [*]f32,
    residual_out: ?[*]f32,
) c_int {
    if (block_size == 0 or count % block_size != 0) return -1;
    const num_blocks: usize = @intCast(count / block_size);
    const bytes_per_block: usize = (block_size + 1) / 2;

    for (0..num_blocks) |b| {
        const in_b = weights[b * block_size .. (b + 1) * block_size];
        const out_p = packed_out[b * bytes_per_block .. (b + 1) * bytes_per_block];
        const res_p: ?[]f32 = if (residual_out) |r| r[b * block_size .. (b + 1) * block_size] else null;
        scales_out[b] = quantization.quantizeBlockNF4(in_b, out_p, res_p);
    }
    return 0;
}

pub export fn hk_dequantize_tensor_nf4(
    packed_in: [*]const u8,
    scales_in: [*]const f32,
    count: u64,
    block_size: u32,
    out_f32: [*]f32,
    residual_in: ?[*]const f32,
) c_int {
    if (block_size == 0 or count % block_size != 0) return -1;
    const num_blocks: usize = @intCast(count / block_size);
    const bytes_per_block: usize = (block_size + 1) / 2;

    for (0..num_blocks) |b| {
        const in_p = packed_in[b * bytes_per_block .. (b + 1) * bytes_per_block];
        const out_b = out_f32[b * block_size .. (b + 1) * block_size];
        const scale = scales_in[b];
        quantization.dequantizeBlockNF4(in_p, scale, block_size, out_b);
        if (residual_in) |r| {
            const res_b = r[b * block_size .. (b + 1) * block_size];
            for (0..block_size) |i| {
                out_b[i] += res_b[i];
            }
        }
    }
    return 0;
}

pub export fn hk_pack_2_4(
    dense_in: [*]const f32,
    count: u64,
    out_bytes: [*]u8,
) c_int {
    if (count % 4 != 0) return -1;
    const num_groups: usize = @intCast(count / 4);
    const meta_len = (num_groups + 1) / 2;
    const val_offset = (meta_len + 3) & ~@as(usize, 3);

    @memset(out_bytes[0..val_offset], 0);
    const val_out: [*]f32 = @ptrCast(@alignCast(out_bytes + val_offset));

    for (0..num_groups) |g| {
        const base = g * 4;
        const block = dense_in[base..][0..4];

        var idx0: u2 = 0;
        var idx1: u2 = 1;
        var found: usize = 0;
        for (0..4) |pos| {
            if (block[pos] != 0.0) {
                if (found == 0) {
                    idx0 = @intCast(pos);
                    found += 1;
                } else if (found == 1) {
                    idx1 = @intCast(pos);
                    found += 1;
                    break;
                }
            }
        }

        val_out[g * 2 + 0] = block[idx0];
        val_out[g * 2 + 1] = block[idx1];

        const nibble: u4 = (@as(u4, idx0) & 0x03) | ((@as(u4, idx1) & 0x03) << 2);
        const meta_idx = g / 2;
        if (g % 2 == 0) {
            out_bytes[meta_idx] |= @as(u8, nibble);
        } else {
            out_bytes[meta_idx] |= (@as(u8, nibble) << 4);
        }
    }
    return 0;
}

pub export fn hk_unpack_2_4(
    payload: [*]const u8,
    payload_len: u64,
    count: u64,
    out_buf: [*]f32,
) c_int {
    const payload_slice = payload[0..payload_len];
    const out_slice = out_buf[0..count];
    sparsity.decodeStructured2_4_F32(payload_slice, count, out_slice) catch return -1;
    return 0;
}

pub export fn hk_tile_16x16_pack(
    in_row_major: [*]const f32,
    M: u64,
    K: u64,
    out_tiled: [*]f32,
) c_int {
    const pad_m = (16 - (M % 16)) % 16;
    const pad_k = (16 - (K % 16)) % 16;
    const padded_m = M + pad_m;
    const padded_k = K + pad_k;

    const num_tiles_m = padded_m / 16;
    const num_tiles_k = padded_k / 16;
    const total_elements = num_tiles_m * num_tiles_k * 16 * 16;
    @memset(out_tiled[0..total_elements], 0.0);

    var tile_idx: usize = 0;
    for (0..num_tiles_m) |tm| {
        for (0..num_tiles_k) |tk| {
            const tile_start = tile_idx * 256;
            for (0..16) |r| {
                const global_r = tm * 16 + r;
                for (0..16) |c| {
                    const global_c = tk * 16 + c;
                    const val = if (global_r < M and global_c < K)
                        in_row_major[global_r * K + global_c]
                    else
                        0.0;
                    out_tiled[tile_start + (r * 16 + c)] = val;
                }
            }
            tile_idx += 1;
        }
    }
    return 0;
}

pub export fn hk_tile_16x16_unpack(
    in_tiled: [*]const f32,
    M: u64,
    K: u64,
    out_row_major: [*]f32,
) c_int {
    const pad_m = (16 - (M % 16)) % 16;
    const pad_k = (16 - (K % 16)) % 16;
    const padded_m = M + pad_m;
    const padded_k = K + pad_k;

    const num_tiles_m = padded_m / 16;
    const num_tiles_k = padded_k / 16;

    var tile_idx: usize = 0;
    for (0..num_tiles_m) |tm| {
        for (0..num_tiles_k) |tk| {
            const tile_start = tile_idx * 256;
            for (0..16) |r| {
                const global_r = tm * 16 + r;
                for (0..16) |c| {
                    const global_c = tk * 16 + c;
                    if (global_r < M and global_c < K) {
                        out_row_major[global_r * K + global_c] = in_tiled[tile_start + (r * 16 + c)];
                    }
                }
            }
            tile_idx += 1;
        }
    }
    return 0;
}

pub const C_AppendixEntry = extern struct {
    entry_type: u8,
    flags: u8,
    generation: u32,
    timestamp: u64,
    parent_hash: [32]u8,
    metric_loss: f32,
    metric_acc: f32,
    metric_pass: f32,
    metric_custom: f32,
    name: [*:0]const u8,
    target: [*:0]const u8,
    data: ?*const anyopaque,
    data_size: u64,
};

pub export fn hk_appendix_get_count(reader_ptr: ?*const hk_reader_t) u64 {
    if (reader_ptr == null) return 0;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (wrapper.appendix_reader) |ar| {
        return ar.records.items.len;
    }
    return 0;
}

pub export fn hk_appendix_get_entry(reader_ptr: ?*const hk_reader_t, index: u64, out_entry: ?*C_AppendixEntry) c_int {
    if (reader_ptr == null or out_entry == null) return -1;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (wrapper.appendix_reader == null) return -1;
    const records = wrapper.appendix_reader.?.records.items;
    if (index >= records.len) return -1;
    const rec = records[index];

    out_entry.?.* = .{
        .entry_type = @intFromEnum(rec.entry_type),
        .flags = rec.flags,
        .generation = rec.generation,
        .timestamp = rec.timestamp,
        .parent_hash = rec.parent_hash,
        .metric_loss = rec.metrics.loss,
        .metric_acc = rec.metrics.accuracy,
        .metric_pass = rec.metrics.pass_rate,
        .metric_custom = rec.metrics.custom,
        .name = wrapper.appendix_cstrs.items[index * 2].ptr,
        .target = wrapper.appendix_cstrs.items[index * 2 + 1].ptr,
        .data = if (rec.data.len > 0) @ptrCast(rec.data.ptr) else null,
        .data_size = rec.data.len,
    };
    return 0;
}

pub export fn hk_appendix_append(
    file_path_c: [*:0]const u8,
    entry_type: u8,
    flags: u8,
    name_c: [*:0]const u8,
    target_c: [*:0]const u8,
    generation: u32,
    parent_hash_ptr: ?[*]const u8,
    metric_loss: f32,
    metric_acc: f32,
    metric_pass: f32,
    metric_custom: f32,
    data_ptr: ?[*]const u8,
    data_size: u64,
) c_int {
    const file_path = std.mem.sliceTo(file_path_c, 0);
    const name = std.mem.sliceTo(name_c, 0);
    const target = std.mem.sliceTo(target_c, 0);
    const allocator = std.heap.page_allocator;

    var ph: [32]u8 = @as([32]u8, @splat(0));
    if (parent_hash_ptr != null) {
        @memcpy(&ph, parent_hash_ptr.?[0..32]);
    }

    const payload: []const u8 = if (data_ptr != null and data_size > 0)
        data_ptr.?[0..@intCast(data_size)]
    else
        &[_]u8{};

    const io = std.Options.debug_io;
    const cur_time_ns = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const cur_time_sec: u64 = @intCast(@max(0, @divTrunc(cur_time_ns, 1_000_000_000)));

    const record = format.AppendixRecord{
        .entry_type = @enumFromInt(entry_type),
        .flags = flags,
        .name = name,
        .target = target,
        .generation = generation,
        .timestamp = cur_time_sec,
        .parent_hash = ph,
        .metrics = .{
            .loss = metric_loss,
            .accuracy = metric_acc,
            .pass_rate = metric_pass,
            .custom = metric_custom,
        },
        .data = payload,
    };

    appendix_mod.appendRecordToFile(allocator, file_path, record) catch return -1;
    return 0;
}

pub export fn hk_appendix_rollback(file_path_c: [*:0]const u8, target_generation: u32) c_int {
    const file_path = std.mem.sliceTo(file_path_c, 0);
    const allocator = std.heap.page_allocator;
    appendix_mod.rollbackToFile(allocator, file_path, target_generation) catch return -1;
    return 0;
}

pub export fn hk_dot_product_f32(a: [*]const f32, b: [*]const f32, count: u64) f32 {
    return tensor_ops_mod.dotProductF32(a[0..@intCast(count)], b[0..@intCast(count)]);
}

pub export fn hk_gemv_f32(
    W: [*]const f32,
    x: [*]const f32,
    bias: ?[*]const f32,
    y: [*]f32,
    M: u64,
    K: u64,
) void {
    const bias_slice: ?[]const f32 = if (bias) |b| b[0..@intCast(M)] else null;
    tensor_ops_mod.gemvF32(W[0..@intCast(M * K)], x[0..@intCast(K)], bias_slice, y[0..@intCast(M)], @intCast(M), @intCast(K));
}

pub export fn hk_gemm_f32(
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    M: u64,
    K: u64,
    N: u64,
) void {
    tensor_ops_mod.gemmF32(A[0..@intCast(M * K)], B[0..@intCast(K * N)], C[0..@intCast(M * N)], @intCast(M), @intCast(K), @intCast(N));
}

pub export fn hk_fused_gemv_nf4(
    packed_W: [*]const u8,
    scales: [*]const f32,
    x: [*]const f32,
    bias: ?[*]const f32,
    y: [*]f32,
    M: u64,
    K: u64,
    block_size: u32,
) void {
    const k_bytes = (K + 1) / 2;
    const num_blocks = M * ((K + block_size - 1) / block_size);
    const bias_slice: ?[]const f32 = if (bias) |b| b[0..@intCast(M)] else null;
    tensor_ops_mod.fusedGemvNF4(
        packed_W[0..@intCast(M * k_bytes)],
        scales[0..@intCast(num_blocks)],
        x[0..@intCast(K)],
        bias_slice,
        y[0..@intCast(M)],
        @intCast(M),
        @intCast(K),
        @intCast(block_size),
    );
}

pub export fn hk_fused_gemv_dq8(
    W_i8: [*]const i8,
    scales: [*]const f32,
    x: [*]const f32,
    bias: ?[*]const f32,
    y: [*]f32,
    M: u64,
    K: u64,
    block_size: u32,
) void {
    const num_blocks = M * ((K + block_size - 1) / block_size);
    const bias_slice: ?[]const f32 = if (bias) |b| b[0..@intCast(M)] else null;
    tensor_ops_mod.fusedGemvDQ8(
        W_i8[0..@intCast(M * K)],
        scales[0..@intCast(num_blocks)],
        x[0..@intCast(K)],
        bias_slice,
        y[0..@intCast(M)],
        @intCast(M),
        @intCast(K),
        @intCast(block_size),
    );
}

pub export fn hk_net2wider(
    w_in_old: [*]const f32,
    b_in_old: ?[*]const f32,
    w_in_new: [*]f32,
    b_in_new: ?[*]f32,
    w_out_old: ?[*]const f32,
    w_out_new: ?[*]f32,
    old_out: u64,
    new_out: u64,
    in_f: u64,
    out_f: u64,
    noise_std: f32,
    seed: u64,
) c_int {
    const b_in_old_slice: ?[]const f32 = if (b_in_old) |b| b[0..@intCast(old_out)] else null;
    const b_in_new_slice: ?[]f32 = if (b_in_new) |b| b[0..@intCast(new_out)] else null;
    const w_out_old_slice: ?[]const f32 = if (w_out_old) |w| w[0..@intCast(out_f * old_out)] else null;
    const w_out_new_slice: ?[]f32 = if (w_out_new) |w| w[0..@intCast(out_f * new_out)] else null;

    growth_mod.net2Wider(
        w_in_old[0..@intCast(old_out * in_f)],
        b_in_old_slice,
        w_in_new[0..@intCast(new_out * in_f)],
        b_in_new_slice,
        w_out_old_slice,
        w_out_new_slice,
        @intCast(old_out),
        @intCast(new_out),
        @intCast(in_f),
        @intCast(out_f),
        noise_std,
        seed,
    ) catch return -1;
    return 0;
}

pub export fn hk_net2deeper(weights: [*]f32, bias: ?[*]f32, dim: u64) void {
    const bias_slice: ?[]f32 = if (bias) |b| b[0..@intCast(dim)] else null;
    growth_mod.net2Deeper(weights[0..@intCast(dim * dim)], bias_slice, @intCast(dim));
}

pub export fn hk_net2wider_swiglu(
    w_gate_old: [*]const f32,
    b_gate_old: ?[*]const f32,
    w_up_old: [*]const f32,
    b_up_old: ?[*]const f32,
    w_down_old: [*]const f32,
    b_down_old: ?[*]const f32,
    w_gate_new: [*]f32,
    b_gate_new: ?[*]f32,
    w_up_new: [*]f32,
    b_up_new: ?[*]f32,
    w_down_new: [*]f32,
    b_down_new: ?[*]f32,
    old_inter: u64,
    new_inter: u64,
    in_f: u64,
    out_f: u64,
    zero_init: u8,
    noise_std: f32,
    seed: u64,
) c_int {
    const bg_old: ?[]const f32 = if (b_gate_old) |b| b[0..@intCast(old_inter)] else null;
    const bu_old: ?[]const f32 = if (b_up_old) |b| b[0..@intCast(old_inter)] else null;
    const bd_old: ?[]const f32 = if (b_down_old) |b| b[0..@intCast(out_f)] else null;

    const bg_new: ?[]f32 = if (b_gate_new) |b| b[0..@intCast(new_inter)] else null;
    const bu_new: ?[]f32 = if (b_up_new) |b| b[0..@intCast(new_inter)] else null;
    const bd_new: ?[]f32 = if (b_down_new) |b| b[0..@intCast(out_f)] else null;

    growth_mod.net2WiderSwiGLU(
        w_gate_old[0..@intCast(old_inter * in_f)],
        bg_old,
        w_up_old[0..@intCast(old_inter * in_f)],
        bu_old,
        w_down_old[0..@intCast(out_f * old_inter)],
        bd_old,
        w_gate_new[0..@intCast(new_inter * in_f)],
        bg_new,
        w_up_new[0..@intCast(new_inter * in_f)],
        bu_new,
        w_down_new[0..@intCast(out_f * new_inter)],
        bd_new,
        @intCast(old_inter),
        @intCast(new_inter),
        @intCast(in_f),
        @intCast(out_f),
        (zero_init != 0),
        noise_std,
        seed,
    ) catch return -1;
    return 0;
}

pub export fn hk_expand_vocab(
    embed_old: [*]const f32,
    embed_new: [*]f32,
    lm_head_old: ?[*]const f32,
    lm_head_new: ?[*]f32,
    old_vocab: u64,
    new_vocab: u64,
    hidden_dim: u64,
    seed: u64,
) c_int {
    const head_old: ?[]const f32 = if (lm_head_old) |h| h[0..@intCast(old_vocab * hidden_dim)] else null;
    const head_new: ?[]f32 = if (lm_head_new) |h| h[0..@intCast(new_vocab * hidden_dim)] else null;

    growth_mod.expandVocab(
        embed_old[0..@intCast(old_vocab * hidden_dim)],
        embed_new[0..@intCast(new_vocab * hidden_dim)],
        head_old,
        head_new,
        @intCast(old_vocab),
        @intCast(new_vocab),
        @intCast(hidden_dim),
        seed,
    ) catch return -1;
    return 0;
}

pub export fn hk_plasticity_mask_rows(grad: [*]f32, total_len: u64, cutoff_rows: u64, cols: u64) void {
    growth_mod.applyPlasticityMaskRows(grad[0..@intCast(total_len)], @intCast(cutoff_rows), @intCast(cols));
}

pub export fn hk_plasticity_mask_cols(grad: [*]f32, total_len: u64, num_rows: u64, cutoff_cols: u64, stride_cols: u64) void {
    growth_mod.applyPlasticityMaskCols(grad[0..@intCast(total_len)], @intCast(num_rows), @intCast(cutoff_cols), @intCast(stride_cols));
}

pub export fn hk_forward_swiglu(
    x: [*]const f32,
    w_gate: [*]const f32,
    b_gate: ?[*]const f32,
    w_up: [*]const f32,
    b_up: ?[*]const f32,
    w_down: [*]const f32,
    b_down: ?[*]const f32,
    inter_buf: [*]f32,
    out: [*]f32,
    in_f: u64,
    inter_f: u64,
    out_f: u64,
) c_int {
    const bg: ?[]const f32 = if (b_gate) |b| b[0..@intCast(inter_f)] else null;
    const bu: ?[]const f32 = if (b_up) |b| b[0..@intCast(inter_f)] else null;
    const bd: ?[]const f32 = if (b_down) |b| b[0..@intCast(out_f)] else null;

    tensor_ops_mod.forwardSwiGLUF32(
        x[0..@intCast(in_f)],
        w_gate[0..@intCast(inter_f * in_f)],
        bg,
        w_up[0..@intCast(inter_f * in_f)],
        bu,
        w_down[0..@intCast(out_f * inter_f)],
        bd,
        inter_buf[0..@intCast(inter_f * 2)],
        out[0..@intCast(out_f)],
        @intCast(in_f),
        @intCast(inter_f),
        @intCast(out_f),
    ) catch return -1;
    return 0;
}

pub export fn hk_forward_rmsnorm(
    x: [*]const f32,
    weight: [*]const f32,
    eps: f32,
    out: [*]f32,
    len: u64,
) void {
    tensor_ops_mod.rmsNormF32(
        x[0..@intCast(len)],
        weight[0..@intCast(len)],
        eps,
        out[0..@intCast(len)],
    );
}

pub export fn hk_forward_silu(x: [*]const f32, out: [*]f32, len: u64) void {
    tensor_ops_mod.siluF32(x[0..@intCast(len)], out[0..@intCast(len)]);
}

pub const hk_writer_t = opaque {};

const WriterWrapper = struct {
    arena: std.heap.ArenaAllocator,
    writer: writer_mod.HKWriter,
};

pub export fn hk_writer_create(alignment: u64) ?*hk_writer_t {
    const parent_allocator = std.heap.page_allocator;
    const wrapper = parent_allocator.create(WriterWrapper) catch return null;
    wrapper.arena = std.heap.ArenaAllocator.init(parent_allocator);
    wrapper.writer = writer_mod.HKWriter.init(wrapper.arena.allocator());
    if (alignment > 0) {
        wrapper.writer.setAlignment(@intCast(alignment));
    }
    return @ptrCast(wrapper);
}

pub export fn hk_writer_destroy(writer_ptr: ?*hk_writer_t) void {
    if (writer_ptr == null) return;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    wrapper.arena.deinit();
    std.heap.page_allocator.destroy(wrapper);
}

pub export fn hk_writer_set_raw_storage(writer_ptr: ?*hk_writer_t, enabled: c_int) void {
    if (writer_ptr == null) return;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    wrapper.writer.setRawWeightStorage(enabled != 0);
}

pub export fn hk_writer_add_metadata_string(writer_ptr: ?*hk_writer_t, key_c: [*:0]const u8, val_c: [*:0]const u8) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    const key = std.mem.sliceTo(key_c, 0);
    const val = std.mem.sliceTo(val_c, 0);
    wrapper.writer.addMetadataString(key, val) catch return -1;
    return 0;
}

pub export fn hk_writer_add_metadata_int(writer_ptr: ?*hk_writer_t, key_c: [*:0]const u8, val: i64) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    const key = std.mem.sliceTo(key_c, 0);
    wrapper.writer.addMetadataInt(key, val) catch return -1;
    return 0;
}

pub export fn hk_writer_add_metadata_float(writer_ptr: ?*hk_writer_t, key_c: [*:0]const u8, val: f64) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    const key = std.mem.sliceTo(key_c, 0);
    wrapper.writer.addMetadataFloat(key, val) catch return -1;
    return 0;
}

pub export fn hk_writer_add_metadata_bool(writer_ptr: ?*hk_writer_t, key_c: [*:0]const u8, val: c_int) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    const key = std.mem.sliceTo(key_c, 0);
    wrapper.writer.addMetadataBool(key, val != 0) catch return -1;
    return 0;
}

pub export fn hk_writer_add_metadata_json(writer_ptr: ?*hk_writer_t, key_c: [*:0]const u8, val_c: [*:0]const u8) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    const key = std.mem.sliceTo(key_c, 0);
    const val = std.mem.sliceTo(val_c, 0);
    wrapper.writer.addMetadataJson(key, val) catch return -1;
    return 0;
}

/// Converts a raw byte to an enum value only if it names a declared tag, so untrusted
/// input can never produce an invalid enum (undefined behavior in safe builds).
fn safeEnum(comptime E: type, val: u8) ?E {
    inline for (@typeInfo(E).@"enum".fields) |f| {
        if (val == f.value) return @enumFromInt(val);
    }
    return null;
}

fn safeStorageType(val: u8) ?format.StorageType {
    return safeEnum(format.StorageType, val);
}

fn safeTileLayout(val: u8) ?format.TileLayout {
    return safeEnum(format.TileLayout, val);
}

fn safeSparsityType(val: u8) ?format.SparsityType {
    return safeEnum(format.SparsityType, val);
}

pub export fn hk_writer_add_tensor(
    writer_ptr: ?*hk_writer_t,
    name_c: [*:0]const u8,
    storage_type_raw: u8,
    tile_layout_raw: u8,
    sparsity_type_raw: u8,
    ndim: u8,
    shape_ptr: [*]const u64,
    data_ptr: [*]const u8,
    data_len: u64,
    sparsity_ratio: f32,
) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    const name = std.mem.sliceTo(name_c, 0);
    const alloc = wrapper.arena.allocator();

    const name_dup = alloc.dupe(u8, name) catch return -1;
    const data_dup = alloc.alloc(u8, @intCast(data_len)) catch return -1;
    @memcpy(data_dup, data_ptr[0..@intCast(data_len)]);

    var shape_arr: [format.MAX_DIMS]u64 = @as([format.MAX_DIMS]u64, @splat(0));
    const count = @min(@as(usize, ndim), format.MAX_DIMS);
    for (0..count) |i| {
        shape_arr[i] = shape_ptr[i];
    }

    const st = safeStorageType(storage_type_raw) orelse return -2;
    const tl = safeTileLayout(tile_layout_raw) orelse return -3;
    const sp = safeSparsityType(sparsity_type_raw) orelse return -4;

    const payload = writer_mod.TensorPayload{
        .name = name_dup,
        .storage_type = st,
        .tile_layout = tl,
        .sparsity_type = sp,
        .ndim = ndim,
        .shape = shape_arr,
        .data = data_dup,
        .sparsity_ratio = sparsity_ratio,
    };

    wrapper.writer.addTensor(payload) catch return -1;
    return 0;
}

pub export fn hk_writer_write_to_file(writer_ptr: ?*hk_writer_t, path_c: [*:0]const u8) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    const path = std.mem.sliceTo(path_c, 0);
    wrapper.writer.writeToFile(path) catch return -1;
    return 0;
}

pub export fn hk_writer_set_sharding(writer_ptr: ?*hk_writer_t, split_index: u16, split_count: u16) c_int {
    if (writer_ptr == null) return -1;
    const wrapper: *WriterWrapper = @ptrCast(@alignCast(writer_ptr));
    wrapper.writer.setSharding(split_index, split_count);
    return 0;
}

pub export fn hk_reader_is_sharded(reader_ptr: ?*const hk_reader_t) c_int {
    if (reader_ptr == null) return 0;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    return if (wrapper.reader.isSharded()) 1 else 0;
}

pub export fn hk_reader_get_split_index(reader_ptr: ?*const hk_reader_t) u16 {
    if (reader_ptr == null) return 0;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    return wrapper.reader.getSplitIndex();
}

pub export fn hk_reader_get_split_count(reader_ptr: ?*const hk_reader_t) u16 {
    if (reader_ptr == null) return 1;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    return wrapper.reader.getSplitCount();
}

pub export fn hk_metadata_patch_in_place(file_path_c: [*:0]const u8, key_c: [*:0]const u8, val_c: [*:0]const u8) c_int {
    const allocator = std.heap.page_allocator;
    const file_path = std.mem.sliceTo(file_path_c, 0);
    const key = std.mem.sliceTo(key_c, 0);
    const val = std.mem.sliceTo(val_c, 0);
    const metadata_mod = @import("metadata.zig");
    metadata_mod.patchFileMetadataInPlace(allocator, file_path, key, val) catch return -1;
    return 0;
}

pub export fn hk_convert_gguf(input_path_c: [*:0]const u8, output_path_c: [*:0]const u8) c_int {
    const allocator = std.heap.page_allocator;
    const input_path = std.mem.sliceTo(input_path_c, 0);
    const output_path = std.mem.sliceTo(output_path_c, 0);
    const gguf_mod = @import("gguf.zig");
    gguf_mod.convertGGUFToHK(input_path, output_path, allocator) catch return -1;
    return 0;
}

pub export fn hk_export_gguf(input_path_c: [*:0]const u8, output_path_c: [*:0]const u8) c_int {
    const allocator = std.heap.page_allocator;
    const input_path = std.mem.sliceTo(input_path_c, 0);
    const output_path = std.mem.sliceTo(output_path_c, 0);
    const gguf_mod = @import("gguf.zig");
    gguf_mod.exportHKToGGUF(input_path, output_path, allocator) catch return -1;
    return 0;
}

pub export fn hk_rope_permute_hf_to_gguf(in: [*]const f32, out: [*]f32, total_len: u64, n_heads: u64, head_dim: u64) void {
    const len: usize = @intCast(total_len);
    tensor_ops_mod.ropePermuteHFToGGUF(in[0..len], out[0..len], @intCast(n_heads), @intCast(head_dim));
}

pub export fn hk_rope_unpermute_gguf_to_hf(in: [*]const f32, out: [*]f32, total_len: u64, n_heads: u64, head_dim: u64) void {
    const len: usize = @intCast(total_len);
    tensor_ops_mod.ropeUnpermuteGGUFToHF(in[0..len], out[0..len], @intCast(n_heads), @intCast(head_dim));
}

pub export fn hk_layernorm_offset_f32(data: [*]f32, len: u64, offset: f32) void {
    tensor_ops_mod.layerNormOffsetF32(data[0..@intCast(len)], offset);
}

pub export fn hk_gemv_q8_0(W: [*]const u8, x: [*]const f32, bias: ?[*]const f32, y: [*]f32, M: u64, K: u64) void {
    const m: usize = @intCast(M);
    const k: usize = @intCast(K);
    const w_bytes_len = m * (k / 32) * @sizeOf(quantization.BlockQ8_0);
    const bias_slice: ?[]const f32 = if (bias) |b| b[0..m] else null;
    tensor_ops_mod.gemvQ8_0(W[0..w_bytes_len], x[0..k], bias_slice, y[0..m], m, k);
}

pub export fn hk_gemv_q4_0(W: [*]const u8, x: [*]const f32, bias: ?[*]const f32, y: [*]f32, M: u64, K: u64) void {
    const m: usize = @intCast(M);
    const k: usize = @intCast(K);
    const w_bytes_len = m * (k / 32) * @sizeOf(quantization.BlockQ4_0);
    const bias_slice: ?[]const f32 = if (bias) |b| b[0..m] else null;
    tensor_ops_mod.gemvQ4_0(W[0..w_bytes_len], x[0..k], bias_slice, y[0..m], m, k);
}

pub export fn hk_gemv_q4_k(W: [*]const u8, x: [*]const f32, bias: ?[*]const f32, y: [*]f32, M: u64, K: u64) void {
    const m: usize = @intCast(M);
    const k: usize = @intCast(K);
    const w_bytes_len = m * (k / 256) * @sizeOf(quantization.BlockQ4_K);
    const bias_slice: ?[]const f32 = if (bias) |b| b[0..m] else null;
    tensor_ops_mod.gemvQ4_K(W[0..w_bytes_len], x[0..k], bias_slice, y[0..m], m, k);
}

// ---------------------------------------------------------------------------
// Native Tokenizer C ABI
// ---------------------------------------------------------------------------
pub const hk_tokenizer_t = opaque {};

/// The tokenizer borrows its vocabulary from the container, so the wrapper owns the reader too.
const TokenizerWrapper = struct {
    reader: reader_mod.HKReader,
    tok: tokenizer_mod.Tokenizer,
    allocator: std.mem.Allocator,
};

pub export fn hk_tokenizer_load_from_file(file_path_c: [*:0]const u8) ?*hk_tokenizer_t {
    const path = std.mem.sliceTo(file_path_c, 0);
    const allocator = std.heap.page_allocator;

    const wrapper = allocator.create(TokenizerWrapper) catch return null;
    wrapper.allocator = allocator;
    wrapper.reader = reader_mod.HKReader.open(path, allocator) catch {
        allocator.destroy(wrapper);
        return null;
    };
    var diag = tokenizer_mod.Diag{};
    wrapper.tok = tokenizer_mod.Tokenizer.fromMetadata(allocator, &wrapper.reader.metadata_map, &diag) catch {
        wrapper.reader.deinit();
        allocator.destroy(wrapper);
        return null;
    };
    return @ptrCast(wrapper);
}

pub export fn hk_tokenizer_free(tok_ptr: ?*hk_tokenizer_t) void {
    if (tok_ptr == null) return;
    const wrapper: *TokenizerWrapper = @ptrCast(@alignCast(tok_ptr));
    wrapper.tok.deinit();
    wrapper.reader.deinit();
    wrapper.allocator.destroy(wrapper);
}

pub export fn hk_tokenizer_get_vocab_size(tok_ptr: ?*const hk_tokenizer_t) u32 {
    if (tok_ptr == null) return 0;
    const wrapper: *const TokenizerWrapper = @ptrCast(@alignCast(tok_ptr));
    return @intCast(wrapper.tok.count());
}

/// Encodes `text`. Returns the number of ids written, or 0 on failure. `add_special` asks for
/// BOS/EOS where the model wants them; `parse_special` turns text like `<|im_start|>` into
/// control token ids.
pub export fn hk_tokenizer_encode(
    tok_ptr: ?*hk_tokenizer_t,
    text_c: [*:0]const u8,
    add_special: c_int,
    parse_special: c_int,
    out_ids: [*]u32,
    max_ids: u32,
) u32 {
    if (tok_ptr == null) return 0;
    const wrapper: *TokenizerWrapper = @ptrCast(@alignCast(tok_ptr));
    const text = std.mem.sliceTo(text_c, 0);

    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(wrapper.allocator);

    wrapper.tok.encode(text, .{ .add_special = add_special != 0, .parse_special = parse_special != 0 }, &tokens) catch return 0;

    const count: u32 = @min(@as(u32, @intCast(tokens.items.len)), max_ids);
    @memcpy(out_ids[0..count], tokens.items[0..count]);
    return count;
}

pub export fn hk_tokenizer_decode(
    tok_ptr: ?*const hk_tokenizer_t,
    ids_ptr: [*]const u32,
    num_ids: u32,
    show_special: c_int,
    out_buf: [*]u8,
    max_len: u32,
) u32 {
    if (tok_ptr == null) return 0;
    const wrapper: *const TokenizerWrapper = @ptrCast(@alignCast(tok_ptr));

    var text_list: std.ArrayList(u8) = .empty;
    defer text_list.deinit(wrapper.allocator);

    wrapper.tok.decode(ids_ptr[0..num_ids], show_special != 0, &text_list) catch return 0;

    const count: u32 = @min(@as(u32, @intCast(text_list.items.len)), max_len);
    @memcpy(out_buf[0..count], text_list.items[0..count]);
    return count;
}

// ---------------------------------------------------------------------------
// Native Inference Engine C ABI
// ---------------------------------------------------------------------------
pub const hk_engine_t = opaque {};

const EngineWrapper = struct {
    reader: reader_mod.HKReader,
    model: engine_mod.Model,
    allocator: std.mem.Allocator,
};

/// Loads a model for token by token decoding. Returns null when the file cannot be opened or the
/// architecture is not supported; use `hk_engine_last_error` for the reason.
pub export fn hk_engine_load_from_file(file_path_c: [*:0]const u8) ?*hk_engine_t {
    const path = std.mem.sliceTo(file_path_c, 0);
    const allocator = std.heap.page_allocator;
    engine_error_len = 0;

    const wrapper = allocator.create(EngineWrapper) catch return null;
    wrapper.allocator = allocator;
    wrapper.reader = reader_mod.HKReader.open(path, allocator) catch |e| {
        setEngineError("could not open '{s}': {s}", .{ path, @errorName(e) });
        allocator.destroy(wrapper);
        return null;
    };
    var diag = engine_mod.Diag{};
    wrapper.model = engine_mod.Model.init(allocator, &wrapper.reader, .{}, &diag) catch |e| {
        setEngineError("{s}: {s}", .{ @errorName(e), diag.message() });
        wrapper.reader.deinit();
        allocator.destroy(wrapper);
        return null;
    };
    return @ptrCast(wrapper);
}

var engine_error_buf: [512]u8 = undefined;
var engine_error_len: usize = 0;

fn setEngineError(comptime fmt: []const u8, args: anytype) void {
    const out = std.fmt.bufPrint(&engine_error_buf, fmt, args) catch engine_error_buf[0..0];
    engine_error_len = out.len;
}

/// Copies the reason the last `hk_engine_load_from_file` failed into `out` (NUL terminated) and
/// returns its length, or 0 when there was no failure. Not thread safe.
pub export fn hk_engine_last_error(out: [*]u8, cap: u32) u32 {
    if (cap == 0) return 0;
    const n: usize = @min(engine_error_len, cap - 1);
    @memcpy(out[0..n], engine_error_buf[0..n]);
    out[n] = 0;
    return @intCast(n);
}

pub export fn hk_engine_free(engine_ptr: ?*hk_engine_t) void {
    if (engine_ptr == null) return;
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(engine_ptr));
    wrapper.model.deinit();
    wrapper.reader.deinit();
    wrapper.allocator.destroy(wrapper);
}

pub export fn hk_engine_get_vocab_size(engine_ptr: ?*const hk_engine_t) u32 {
    if (engine_ptr == null) return 0;
    const wrapper: *const EngineWrapper = @ptrCast(@alignCast(engine_ptr));
    return @intCast(wrapper.model.cfg.vocab);
}

/// Number of positions the engine can hold before `hk_engine_forward` reports a full context.
pub export fn hk_engine_get_context_size(engine_ptr: ?*const hk_engine_t) u32 {
    if (engine_ptr == null) return 0;
    const wrapper: *const EngineWrapper = @ptrCast(@alignCast(engine_ptr));
    return @intCast(wrapper.model.n_ctx);
}

pub export fn hk_engine_reset_cache(engine_ptr: ?*hk_engine_t) void {
    if (engine_ptr == null) return;
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(engine_ptr));
    wrapper.model.kv.truncate(0);
}

/// Feeds `n` tokens that start at position `pos` and writes the logits of the last one to
/// `out_logits` (vocabulary sized). Returns 0 on success, -1 for a null argument, -2 when the
/// context window is full, -3 when `n` exceeds the batch limit, -4 on any other failure.
pub export fn hk_engine_forward_tokens(
    engine_ptr: ?*hk_engine_t,
    tokens: ?[*]const u32,
    n: u32,
    pos: u32,
    out_logits: ?[*]f32,
) c_int {
    if (engine_ptr == null or tokens == null or out_logits == null or n == 0) return -1;
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(engine_ptr));
    const m = &wrapper.model;
    var done: usize = 0;
    while (done < n) {
        const take = @min(m.n_batch, n - done);
        m.forward(tokens.?[done..][0..take], pos + done) catch |e| return switch (e) {
            error.ContextFull => -2,
            error.BatchTooLarge => -3,
            else => -4,
        };
        done += take;
        if (done == n) {
            const logits = m.logitsFor(take - 1);
            @memcpy(out_logits.?[0..logits.len], logits);
        }
    }
    return 0;
}

pub export fn hk_engine_forward(
    engine_ptr: ?*hk_engine_t,
    token: u32,
    pos: u32,
    out_logits: [*]f32,
) c_int {
    const t = [1]u32{token};
    return hk_engine_forward_tokens(engine_ptr, &t, 1, pos, out_logits);
}

// ---------------------------------------------------------------------------
// Native Sampler C ABI
// ---------------------------------------------------------------------------
/// Draws one token from `logits`, which is modified in place. A temperature of 0 is greedy.
/// Out-of-range values are clamped instead of trusted. Returns 0 if the sampler could not
/// allocate its scratch space.
pub export fn hk_sample_token(
    logits_ptr: [*]f32,
    vocab_size: u64,
    temp: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    rep_pen: f32,
    history_ptr: ?[*]const u32,
    history_len: u32,
    seed: u64,
) u32 {
    const allocator = std.heap.page_allocator;
    const logits = logits_ptr[0..@intCast(vocab_size)];
    const history: []const u32 = if (history_ptr) |h| h[0..history_len] else &[_]u32{};

    var sampler = sampler_mod.Sampler.init(allocator, seed);
    defer sampler.deinit();
    return sampler.sample(logits, history, .{
        .temperature = if (std.math.isFinite(temp)) @max(temp, 0) else 0,
        .top_k = top_k,
        .top_p = top_p,
        .min_p = min_p,
        .repeat_penalty = if (rep_pen > 0 and std.math.isFinite(rep_pen)) rep_pen else 1.0,
        .repeat_last_n = 0,
        .seed = seed,
    }, false) catch 0;
}

// ---------------------------------------------------------------------------
// Native SafeTensors Transcoder C ABI
// ---------------------------------------------------------------------------
pub export fn hk_convert_safetensors(
    input_path_c: [*:0]const u8,
    output_path_c: [*:0]const u8,
    storage_type_raw: u8,
) c_int {
    const allocator = std.heap.page_allocator;
    const input_path = std.mem.sliceTo(input_path_c, 0);
    const output_path = std.mem.sliceTo(output_path_c, 0);
    const target_st = safeStorageType(storage_type_raw) orelse format.StorageType.f32;

    safetensors_mod.transcodeSafeTensorsToHK(allocator, input_path, output_path, target_st) catch return -1;
    return 0;
}

// ---------------------------------------------------------------------------
// Native Hugging Face Architecture Mapper C ABI
// ---------------------------------------------------------------------------
pub export fn hk_hf_detect_architecture(
    json_config_c: [*:0]const u8,
    out_arch: [*]u8,
    max_len: usize,
) c_int {
    const json_str = std.mem.sliceTo(json_config_c, 0);
    const arch = hf_mapper_mod.detectArchitectureFromJson(json_str);
    if (arch.len + 1 > max_len) return -1;
    @memcpy(out_arch[0..arch.len], arch);
    out_arch[arch.len] = 0;
    return 0;
}

pub export fn hk_hf_map_tensor_name(
    tensor_name_c: [*:0]const u8,
    arch_c: [*:0]const u8,
    to_hk: bool,
    out_name: [*]u8,
    max_len: usize,
) c_int {
    const tensor_name = std.mem.sliceTo(tensor_name_c, 0);
    const arch = std.mem.sliceTo(arch_c, 0);
    const out_slice = out_name[0..max_len];

    const mapped = if (to_hk)
        hf_mapper_mod.mapTensorNameToHk(tensor_name, arch, out_slice) catch return -1
    else
        hf_mapper_mod.mapTensorNameToHf(tensor_name, arch, out_slice) catch return -1;

    if (mapped.len >= max_len) return -1;
    out_name[mapped.len] = 0;
    return 0;
}

// ---------------------------------------------------------------------------
// Native Context Window Management C ABI
// ---------------------------------------------------------------------------
pub export fn hk_context_truncate(
    in_tokens: [*]const u32,
    in_len: usize,
    max_tokens: usize,
    strategy: c_int,
    head_ratio: f32,
    out_tokens: [*]u32,
    out_len: *usize,
) c_int {
    const tokens = in_tokens[0..in_len];
    const strat = context_mod.TruncationStrategy.fromInt(strategy);
    const out_buf = out_tokens[0..max_tokens];
    const written = context_mod.truncateTokens(tokens, max_tokens, strat, head_ratio, out_buf);
    out_len.* = written;
    return 0;
}

// ---------------------------------------------------------------------------
// Native Adaptive Autonomous Framework C ABI
// ---------------------------------------------------------------------------
pub export fn hk_governor_can_grow(
    current_params: u64,
    added_params: u64,
    max_growth_ratio: f32,
    max_vram_mb: u64,
    dtype_bytes: u32,
    out_reason: ?[*]u8,
    max_reason_len: usize,
) c_int {
    const gov = adaptive_mod.GrowthGovernor.init(max_vram_mb, max_growth_ratio);
    if (out_reason) |reason_ptr| {
        const reason_buf = reason_ptr[0..max_reason_len];
        const res = gov.canGrow(current_params, added_params, dtype_bytes, reason_buf);
        const len = @min(res.reason.len, max_reason_len - 1);
        reason_ptr[len] = 0;
        return if (res.approved) 1 else 0;
    } else {
        var dummy: [64]u8 = undefined;
        const res = gov.canGrow(current_params, added_params, dtype_bytes, &dummy);
        return if (res.approved) 1 else 0;
    }
}

pub export fn hk_governor_can_grow_batch(
    current_params: [*]const u64,
    added_params: [*]const u64,
    n: usize,
    max_growth_ratio: f32,
    max_vram_mb: u64,
    dtype_bytes: u32,
    out_results: [*]u8,
) c_int {
    const gov = adaptive_mod.GrowthGovernor.init(max_vram_mb, max_growth_ratio);
    var dummy: [64]u8 = undefined;
    for (0..n) |i| {
        const res = gov.canGrow(current_params[i], added_params[i], dtype_bytes, &dummy);
        out_results[i] = if (res.approved) 1 else 0;
    }
    return 0;
}

pub export fn hk_expand_vocab_embeddings(
    old_embed: [*]const f32,
    old_vocab: usize,
    hidden_size: usize,
    new_vocab: usize,
    new_embed: [*]f32,
    init_std: f32,
    seed: u64,
) c_int {
    const old_slice = old_embed[0 .. old_vocab * hidden_size];
    const new_slice = new_embed[0 .. new_vocab * hidden_size];
    adaptive_mod.expandVocabEmbeddings(old_slice, old_vocab, hidden_size, new_vocab, new_slice, init_std, seed);
    return 0;
}

pub export fn hk_init_plasticity_mask(
    mask: [*]f32,
    total_units: usize,
    base_units: usize,
    decay_rate: f32,
) c_int {
    const mask_slice = mask[0..total_units];
    adaptive_mod.initPlasticityMask(mask_slice, total_units, base_units, decay_rate);
    return 0;
}

pub const C_HardwareCapabilities = extern struct {
    vendor: u8,
    has_avx2: u8,
    has_avx512f: u8,
    has_avx512vnni: u8,
    has_avx_vnni: u8,
    has_amx: u8,
    has_arm_neon: u8,
    has_arm_sve: u8,
    is_apple_silicon: u8,
    has_rocm_ready: u8,
    has_npu_ready: u8,
    reserved: [5]u8 = @as([5]u8, @splat(0)),
    optimal_page_alignment: u64,
    dma_hugepage_alignment: u64,
};

pub export fn hk_detect_hardware(out_caps: ?*C_HardwareCapabilities) void {
    if (out_caps == null) return;
    const caps = platform_mod.detectHardwareCapabilities();
    out_caps.?.* = .{
        .vendor = @intFromEnum(caps.vendor),
        .has_avx2 = if (caps.has_avx2) 1 else 0,
        .has_avx512f = if (caps.has_avx512f) 1 else 0,
        .has_avx512vnni = if (caps.has_avx512vnni) 1 else 0,
        .has_avx_vnni = if (caps.has_avx_vnni) 1 else 0,
        .has_amx = if (caps.has_amx) 1 else 0,
        .has_arm_neon = if (caps.has_arm_neon) 1 else 0,
        .has_arm_sve = if (caps.has_arm_sve) 1 else 0,
        .is_apple_silicon = if (caps.is_apple_silicon) 1 else 0,
        .has_rocm_ready = if (caps.has_rocm_ready) 1 else 0,
        .has_npu_ready = if (caps.has_npu_ready) 1 else 0,
        .reserved = @as([5]u8, @splat(0)),
        .optimal_page_alignment = @intCast(caps.optimal_page_alignment),
        .dma_hugepage_alignment = @intCast(caps.dma_hugepage_alignment),
    };
}

pub export fn hk_get_optimal_alignment() usize {
    const caps = platform_mod.detectHardwareCapabilities();
    return caps.optimal_page_alignment;
}

pub export fn hk_is_raw_storage(reader_ptr: ?*const hk_reader_t) c_int {
    if (reader_ptr == null) return 0;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    return if (wrapper.reader.isRawWeightStorage()) 1 else 0;
}

pub export fn hk_is_universal_page_aligned(reader_ptr: ?*const hk_reader_t) c_int {
    if (reader_ptr == null) return 0;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    return if (wrapper.reader.isUniversalPageAligned()) 1 else 0;
}

pub export fn hk_get_file_alignment(reader_ptr: ?*const hk_reader_t) u32 {
    if (reader_ptr == null) return 128;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    return @intCast(wrapper.reader.getAlignment());
}

pub export fn hk_get_tensor_raw_ptr(reader_ptr: ?*const hk_reader_t, index: u64, out_size: ?*u64) ?*const anyopaque {
    if (reader_ptr == null) return null;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    if (index >= wrapper.reader.toc.entries.items.len) return null;
    const entry = wrapper.reader.toc.entries.items[index];
    const data_bytes = wrapper.reader.getRawTensorBytes(entry) catch return null;
    if (out_size) |sz| {
        sz.* = data_bytes.len;
    }
    return @ptrCast(data_bytes.ptr);
}

pub export fn hk_get_raw_buffer(reader_ptr: ?*const hk_reader_t, out_size: ?*u64) ?*const anyopaque {
    if (reader_ptr == null) return null;
    const wrapper: *const ReaderWrapper = @ptrCast(@alignCast(reader_ptr));
    const bytes = wrapper.reader.mmap_region.bytes;
    if (out_size) |sz| {
        sz.* = bytes.len;
    }
    return @ptrCast(bytes.ptr);
}

pub export fn hk_gemv_bf16(
    w_bf16: [*]const u16,
    x: [*]const f32,
    bias: ?[*]const f32,
    y: [*]f32,
    m: usize,
    k: usize,
) void {
    const b_slice: ?[]const f32 = if (bias) |b| b[0..m] else null;
    tensor_ops_mod.gemvBF16(w_bf16[0 .. m * k], x[0..k], b_slice, y[0..m], m, k);
}

pub export fn hk_gemv_f16(
    w_f16: [*]const f16,
    x: [*]const f32,
    bias: ?[*]const f32,
    y: [*]f32,
    m: usize,
    k: usize,
) void {
    const b_slice: ?[]const f32 = if (bias) |b| b[0..m] else null;
    tensor_ops_mod.gemvF16(w_f16[0 .. m * k], x[0..k], b_slice, y[0..m], m, k);
}

pub export fn hk_gemv_int8(
    w_i8: [*]const i8,
    x: [*]const f32,
    scale_w: f32,
    bias: ?[*]const f32,
    y: [*]f32,
    m: usize,
    k: usize,
) void {
    const b_slice: ?[]const f32 = if (bias) |b| b[0..m] else null;
    tensor_ops_mod.gemvInt8Scaled(w_i8[0 .. m * k], x[0..k], scale_w, b_slice, y[0..m], m, k);
}

pub export fn hk_dot_bf16(a: [*]const u16, b: [*]const f32, len: usize) f32 {
    return tensor_ops_mod.dotProductBF16(a[0..len], b[0..len]);
}

pub export fn hk_dot_f16(a: [*]const f16, b: [*]const f32, len: usize) f32 {
    return tensor_ops_mod.dotProductF16(a[0..len], b[0..len]);
}

pub export fn hk_dot_int8(a: [*]const i8, b: [*]const i8, len: usize) i32 {
    return tensor_ops_mod.dotProductInt8(a[0..len], b[0..len]);
}





