const std = @import("std");

/// Decodes bitmask-compressed tensor elements to dense FP32.
/// Input payload layout:
///   [bitmask: ceil(count / 8) bytes] [padding to 4 bytes] [non_zero_values: nnz * 4 bytes (f32)]
pub fn decodeBitmaskF32(
    payload: []const u8,
    count: usize,
    out: []f32,
) !void {
    const mask_bytes_len = (count + 7) / 8;
    const val_offset = (mask_bytes_len + 3) & ~@as(usize, 3);
    if (payload.len < val_offset) return error.PayloadTooShort;

    const mask = payload[0..mask_bytes_len];
    const val_bytes = payload[val_offset..];
    const num_floats = val_bytes.len / 4;

    var val_idx: usize = 0;
    for (0..count) |i| {
        const byte_idx = i / 8;
        const bit_idx: u3 = @intCast(i % 8);
        const is_non_zero = ((mask[byte_idx] >> bit_idx) & 1) != 0;

        if (is_non_zero) {
            if (val_idx >= num_floats) return error.BitmaskValueCountMismatch;
            const u32_bits = std.mem.readInt(u32, val_bytes[val_idx * 4 ..][0..4], .little);
            out[i] = @bitCast(u32_bits);
            val_idx += 1;
        } else {
            out[i] = 0.0;
        }
    }
}

/// Encodes dense FP32 elements into bitmask-compressed format.
pub fn encodeBitmaskF32(
    dense_in: []const f32,
    allocator: std.mem.Allocator,
) ![]u8 {
    const count = dense_in.len;
    const mask_bytes_len = (count + 7) / 8;
    const val_offset = (mask_bytes_len + 3) & ~@as(usize, 3);

    var nnz: usize = 0;
    for (dense_in) |x| {
        if (x != 0.0) nnz += 1;
    }

    const total_bytes = val_offset + (nnz * @sizeOf(f32));
    const out_buf = try allocator.alloc(u8, total_bytes);
    errdefer allocator.free(out_buf);

    @memset(out_buf[0..val_offset], 0);

    var val_idx: usize = 0;
    for (0..count) |i| {
        if (dense_in[i] != 0.0) {
            const byte_idx = i / 8;
            const bit_idx: u3 = @intCast(i % 8);
            out_buf[byte_idx] |= (@as(u8, 1) << bit_idx);

            const bits: u32 = @bitCast(dense_in[i]);
            std.mem.writeInt(u32, out_buf[val_offset + val_idx * 4 ..][0..4], bits, .little);
            val_idx += 1;
        }
    }

    return out_buf;
}

/// Unpacks NVIDIA Ampere 2:4 structured sparse payload into dense FP32.
/// Input payload format:
///   [meta: ceil(count / 8) bytes] [padding to 4-byte boundary] [values: (count / 2) * 4 bytes (f32)]
pub fn decodeStructured2_4_F32(
    payload: []const u8,
    count: usize,
    out: []f32,
) !void {
    if (count % 4 != 0) return error.InvalidStructuredCount;

    const num_groups = count / 4;
    const meta_len = (num_groups + 1) / 2;
    const val_offset = (meta_len + 3) & ~@as(usize, 3);
    const expected_val_bytes = (count / 2) * @sizeOf(f32);

    if (payload.len < val_offset + expected_val_bytes) return error.PayloadTooShort;

    const meta = payload[0..meta_len];
    const val_bytes = payload[val_offset .. val_offset + expected_val_bytes];

    @memset(out, 0.0);
    const is_aligned = std.mem.isAligned(@intFromPtr(val_bytes.ptr), @alignOf(f32));
    const val_ptr: ?[*]const f32 = if (is_aligned) @ptrCast(@alignCast(val_bytes.ptr)) else null;
    const full_pairs = num_groups / 2;

    const readVal = struct {
        inline fn get(p: ?[*]const f32, bytes: []const u8, idx: usize) f32 {
            if (p) |ptr| return ptr[idx];
            const u = std.mem.readInt(u32, bytes[idx * 4 .. (idx + 1) * 4][0..4], .little);
            return @bitCast(u);
        }
    }.get;

    for (0..full_pairs) |byte_idx| {
        const meta_byte = meta[byte_idx];
        const nibble0: u4 = @truncate(meta_byte & 0x0F);
        const nibble1: u4 = @truncate((meta_byte >> 4) & 0x0F);

        const g0 = byte_idx * 2;
        const g1 = g0 + 1;

        const base0 = g0 * 4;
        const base1 = g1 * 4;

        const idx0_0: usize = (nibble0 & 0x03);
        const idx0_1: usize = ((nibble0 >> 2) & 0x03);
        out[base0 + idx0_0] = readVal(val_ptr, val_bytes, g0 * 2 + 0);
        out[base0 + idx0_1] = readVal(val_ptr, val_bytes, g0 * 2 + 1);

        const idx1_0: usize = (nibble1 & 0x03);
        const idx1_1: usize = ((nibble1 >> 2) & 0x03);
        out[base1 + idx1_0] = readVal(val_ptr, val_bytes, g1 * 2 + 0);
        out[base1 + idx1_1] = readVal(val_ptr, val_bytes, g1 * 2 + 1);
    }

    if (num_groups % 2 != 0) {
        const g = num_groups - 1;
        const meta_byte = meta[g / 2];
        const nibble0: u4 = @truncate(meta_byte & 0x0F);
        const idx0: usize = (nibble0 & 0x03);
        const idx1: usize = ((nibble0 >> 2) & 0x03);
        const base = g * 4;
        out[base + idx0] = readVal(val_ptr, val_bytes, g * 2 + 0);
        out[base + idx1] = readVal(val_ptr, val_bytes, g * 2 + 1);
    }
}

/// Packs dense 2:4 structured sparse FP32 elements into Ampere 2:4 hardware payload.
pub fn encodeStructured2_4_F32(
    dense_in: []const f32,
    allocator: std.mem.Allocator,
) ![]u8 {
    if (dense_in.len % 4 != 0) return error.InvalidStructuredCount;

    const count = dense_in.len;
    const num_groups = count / 4;
    const meta_len = (num_groups + 1) / 2;
    const val_offset = (meta_len + 3) & ~@as(usize, 3);
    const val_bytes_len = (count / 2) * @sizeOf(f32);
    const total_len = val_offset + val_bytes_len;

    const out_buf = try allocator.alloc(u8, total_len);
    errdefer allocator.free(out_buf);

    @memset(out_buf[0..val_offset], 0);

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

        if (found == 1) {
            idx1 = if (idx0 == 0) 1 else 0;
        }

        const bits0: u32 = @bitCast(block[idx0]);
        const bits1: u32 = @bitCast(block[idx1]);
        std.mem.writeInt(u32, out_buf[val_offset + (g * 2 + 0) * 4 ..][0..4], bits0, .little);
        std.mem.writeInt(u32, out_buf[val_offset + (g * 2 + 1) * 4 ..][0..4], bits1, .little);

        const nibble: u4 = (@as(u4, idx0) & 0x03) | ((@as(u4, idx1) & 0x03) << 2);
        const meta_idx = g / 2;
        if (g % 2 == 0) {
            out_buf[meta_idx] |= @as(u8, nibble);
        } else {
            out_buf[meta_idx] |= (@as(u8, nibble) << 4);
        }
    }

    return out_buf;
}

/// Unpacks Block Sparse Row (BSR) payload into dense matrix [M, K].
pub fn decodeBSR_F32(
    payload: []const u8,
    M: usize,
    K: usize,
    out: []f32,
) !void {
    if (payload.len < 12) return error.PayloadTooShort;

    const block_h = std.mem.readInt(u16, payload[0..2], .little);
    const block_w = std.mem.readInt(u16, payload[2..4], .little);
    const num_row_ptrs = std.mem.readInt(u32, payload[4..8], .little);
    const num_col_indices = std.mem.readInt(u32, payload[8..12], .little);

    const bh: usize = block_h;
    const bw: usize = block_w;
    if (bh == 0 or bw == 0) return error.InvalidBlockDimensions;

    var pos: usize = 12;
    const row_ptrs_bytes = num_row_ptrs * @sizeOf(u32);
    if (pos + row_ptrs_bytes > payload.len) return error.PayloadTooShort;
    const row_ptrs_raw = payload[pos .. pos + row_ptrs_bytes];
    pos += row_ptrs_bytes;

    const col_indices_bytes = num_col_indices * @sizeOf(u32);
    if (pos + col_indices_bytes > payload.len) return error.PayloadTooShort;
    const col_indices_raw = payload[pos .. pos + col_indices_bytes];
    pos += col_indices_bytes;

    const num_blocks: usize = num_col_indices;
    const values_bytes = num_blocks * bh * bw * @sizeOf(f32);
    if (pos + values_bytes > payload.len) return error.PayloadTooShort;
    const values_raw = payload[pos .. pos + values_bytes];

    @memset(out, 0.0);

    const num_block_rows = M / bh;
    for (0..num_block_rows) |br| {
        if (br + 1 >= num_row_ptrs) break;
        const start_b = std.mem.readInt(u32, row_ptrs_raw[br * 4 ..][0..4], .little);
        const end_b = std.mem.readInt(u32, row_ptrs_raw[(br + 1) * 4 ..][0..4], .little);

        var b = start_b;
        while (b < end_b) : (b += 1) {
            if (b >= num_col_indices) break;
            const bc: usize = std.mem.readInt(u32, col_indices_raw[b * 4 ..][0..4], .little);
            const block_start_byte = b * bh * bw * 4;

            for (0..bh) |r| {
                const global_r = br * bh + r;
                if (global_r >= M) continue;
                for (0..bw) |c| {
                    const global_c = bc * bw + c;
                    if (global_c >= K) continue;
                    const elem_byte = block_start_byte + (r * bw + c) * 4;
                    const bits = std.mem.readInt(u32, values_raw[elem_byte..][0..4], .little);
                    out[global_r * K + global_c] = @bitCast(bits);
                }
            }
        }
    }
}

/// Encodes a dense matrix [M, K] into Block Sparse Row (BSR) payload.
/// Format matches decodeBSR_F32:
///   Header (12 bytes):
///     [0..2]: block_h (u16 little-endian)
///     [2..4]: block_w (u16 little-endian)
///     [4..8]: num_row_ptrs (u32 little-endian, = M / bh + 1)
///     [8..12]: num_col_indices (u32 little-endian, = nnzb)
///   row_ptrs: num_row_ptrs * 4 bytes
///   col_indices: num_col_indices * 4 bytes
///   values: num_col_indices * bh * bw * 4 bytes (f32 little-endian)
pub fn encodeBSR_F32(
    dense_in: []const f32,
    M: usize,
    K: usize,
    bh: usize,
    bw: usize,
    allocator: std.mem.Allocator,
) ![]u8 {
    if (bh == 0 or bw == 0 or M % bh != 0 or K % bw != 0) return error.InvalidBlockDimensions;
    if (dense_in.len < M * K) return error.BufferTooSmall;

    const num_block_rows = M / bh;
    const num_block_cols = K / bw;
    const block_size = bh * bw;

    // First pass: identify non-zero blocks and compute nnzb
    var nnzb: usize = 0;
    for (0..num_block_rows) |br| {
        for (0..num_block_cols) |bc| {
            var is_non_zero = false;
            for (0..bh) |r| {
                const row_offset = (br * bh + r) * K;
                for (0..bw) |c| {
                    if (dense_in[row_offset + bc * bw + c] != 0.0) {
                        is_non_zero = true;
                        break;
                    }
                }
                if (is_non_zero) break;
            }
            if (is_non_zero) nnzb += 1;
        }
    }

    const num_row_ptrs = num_block_rows + 1;
    const header_len: usize = 12;
    const row_ptrs_len: usize = num_row_ptrs * @sizeOf(u32);
    const col_indices_len: usize = nnzb * @sizeOf(u32);
    const values_len: usize = nnzb * block_size * @sizeOf(f32);
    const total_bytes = header_len + row_ptrs_len + col_indices_len + values_len;

    const out_buf = try allocator.alloc(u8, total_bytes);
    errdefer allocator.free(out_buf);

    // Write header
    std.mem.writeInt(u16, out_buf[0..2], @intCast(bh), .little);
    std.mem.writeInt(u16, out_buf[2..4], @intCast(bw), .little);
    std.mem.writeInt(u32, out_buf[4..8], @intCast(num_row_ptrs), .little);
    std.mem.writeInt(u32, out_buf[8..12], @intCast(nnzb), .little);

    const row_ptrs_offset = header_len;
    const col_indices_offset = row_ptrs_offset + row_ptrs_len;
    const values_offset = col_indices_offset + col_indices_len;

    var cur_b: usize = 0;
    for (0..num_block_rows) |br| {
        std.mem.writeInt(u32, out_buf[row_ptrs_offset + br * 4 ..][0..4], @intCast(cur_b), .little);

        for (0..num_block_cols) |bc| {
            var is_non_zero = false;
            for (0..bh) |r| {
                const row_offset = (br * bh + r) * K;
                for (0..bw) |c| {
                    if (dense_in[row_offset + bc * bw + c] != 0.0) {
                        is_non_zero = true;
                        break;
                    }
                }
                if (is_non_zero) break;
            }

            if (is_non_zero) {
                // Record column index
                std.mem.writeInt(u32, out_buf[col_indices_offset + cur_b * 4 ..][0..4], @intCast(bc), .little);

                // Copy block values
                const block_val_base = values_offset + cur_b * block_size * 4;
                for (0..bh) |r| {
                    const row_offset = (br * bh + r) * K;
                    for (0..bw) |c| {
                        const val = dense_in[row_offset + bc * bw + c];
                        const bits: u32 = @bitCast(val);
                        const val_idx = (r * bw + c) * 4;
                        std.mem.writeInt(u32, out_buf[block_val_base + val_idx ..][0..4], bits, .little);
                    }
                }
                cur_b += 1;
            }
        }
    }
    std.mem.writeInt(u32, out_buf[row_ptrs_offset + num_block_rows * 4 ..][0..4], @intCast(cur_b), .little);

    return out_buf;
}

/// Matrix-vector multiplication with 2:4 structured sparse weight matrix:
/// y = W * x
/// where W is [M, K] stored in Ampere 2:4 structured sparse payload.
/// Memory-first: zero heap allocations; streams row-by-row directly from payload.
pub fn gemvStructured2_4_F32(
    y: []f32,
    payload: []const u8,
    x: []const f32,
    M: usize,
    K: usize,
) !void {
    if (K % 4 != 0) return error.InvalidStructuredCount;
    if (y.len < M or x.len < K) return error.BufferTooSmall;

    const count = M * K;
    const num_groups = count / 4;
    const meta_len = (num_groups + 1) / 2;
    const val_offset = (meta_len + 3) & ~@as(usize, 3);
    const expected_val_bytes = (count / 2) * @sizeOf(f32);

    if (payload.len < val_offset + expected_val_bytes) return error.PayloadTooShort;

    const meta = payload[0..meta_len];
    const val_bytes = payload[val_offset .. val_offset + expected_val_bytes];

    const is_aligned = std.mem.isAligned(@intFromPtr(val_bytes.ptr), @alignOf(f32));
    const val_ptr: ?[*]const f32 = if (is_aligned) @ptrCast(@alignCast(val_bytes.ptr)) else null;

    const readVal = struct {
        inline fn get(p: ?[*]const f32, bytes: []const u8, idx: usize) f32 {
            if (p) |ptr| return ptr[idx];
            const u = std.mem.readInt(u32, bytes[idx * 4 .. (idx + 1) * 4][0..4], .little);
            return @bitCast(u);
        }
    }.get;

    const groups_per_row = K / 4;

    for (0..M) |m| {
        var sum: f32 = 0.0;
        const g_row_start = m * groups_per_row;

        var k_grp: usize = 0;
        while (k_grp + 1 < groups_per_row) : (k_grp += 2) {
            const g0 = g_row_start + k_grp;
            const g1 = g0 + 1;
            const meta_byte = meta[g0 / 2];

            const nibble0: u4 = @truncate(meta_byte & 0x0F);
            const nibble1: u4 = @truncate((meta_byte >> 4) & 0x0F);

            const idx0_0: usize = (nibble0 & 0x03);
            const idx0_1: usize = ((nibble0 >> 2) & 0x03);
            const v0_0 = readVal(val_ptr, val_bytes, g0 * 2 + 0);
            const v0_1 = readVal(val_ptr, val_bytes, g0 * 2 + 1);

            const k_base0 = k_grp * 4;
            sum += v0_0 * x[k_base0 + idx0_0] + v0_1 * x[k_base0 + idx0_1];

            const idx1_0: usize = (nibble1 & 0x03);
            const idx1_1: usize = ((nibble1 >> 2) & 0x03);
            const v1_0 = readVal(val_ptr, val_bytes, g1 * 2 + 0);
            const v1_1 = readVal(val_ptr, val_bytes, g1 * 2 + 1);

            const k_base1 = (k_grp + 1) * 4;
            sum += v1_0 * x[k_base1 + idx1_0] + v1_1 * x[k_base1 + idx1_1];
        }

        if (k_grp < groups_per_row) {
            const g = g_row_start + k_grp;
            const meta_byte = meta[g / 2];
            const nibble: u4 = if (g % 2 == 0) @truncate(meta_byte & 0x0F) else @truncate((meta_byte >> 4) & 0x0F);
            const idx0: usize = (nibble & 0x03);
            const idx1: usize = ((nibble >> 2) & 0x03);
            const v0 = readVal(val_ptr, val_bytes, g * 2 + 0);
            const v1 = readVal(val_ptr, val_bytes, g * 2 + 1);
            const k_base = k_grp * 4;
            sum += v0 * x[k_base + idx0] + v1 * x[k_base + idx1];
        }

        y[m] = sum;
    }
}

/// Matrix-matrix multiplication with 2:4 structured sparse weight matrix:
/// Y[t, :] = W * X[t, :]
/// Memory-first: streaming over tokens with zero temporary allocations.
pub fn gemmStructured2_4_F32(
    Y: []f32,
    payload: []const u8,
    X: []const f32,
    M: usize,
    N: usize,
    K: usize,
) !void {
    if (Y.len < N * M or X.len < N * K) return error.BufferTooSmall;
    for (0..N) |t| {
        const y_row = Y[t * M .. (t + 1) * M];
        const x_row = X[t * K .. (t + 1) * K];
        try gemvStructured2_4_F32(y_row, payload, x_row, M, K);
    }
}

/// Matrix-vector multiplication for Block Sparse Row (BSR) weights:
/// y = W * x
/// Memory-first: streams directly from BSR payload without temporary memory.
pub fn gemvBSR_F32(
    y: []f32,
    payload: []const u8,
    x: []const f32,
    M: usize,
    K: usize,
) !void {
    if (payload.len < 12) return error.PayloadTooShort;

    const block_h = std.mem.readInt(u16, payload[0..2], .little);
    const block_w = std.mem.readInt(u16, payload[2..4], .little);
    const num_row_ptrs = std.mem.readInt(u32, payload[4..8], .little);
    const num_col_indices = std.mem.readInt(u32, payload[8..12], .little);

    const bh: usize = block_h;
    const bw: usize = block_w;
    if (bh == 0 or bw == 0 or M % bh != 0 or K % bw != 0) return error.InvalidBlockDimensions;
    if (y.len < M or x.len < K) return error.BufferTooSmall;

    var pos: usize = 12;
    const row_ptrs_bytes = num_row_ptrs * @sizeOf(u32);
    if (pos + row_ptrs_bytes > payload.len) return error.PayloadTooShort;
    const row_ptrs_raw = payload[pos .. pos + row_ptrs_bytes];
    pos += row_ptrs_bytes;

    const col_indices_bytes = num_col_indices * @sizeOf(u32);
    if (pos + col_indices_bytes > payload.len) return error.PayloadTooShort;
    const col_indices_raw = payload[pos .. pos + col_indices_bytes];
    pos += col_indices_bytes;

    const num_blocks: usize = num_col_indices;
    const values_bytes = num_blocks * bh * bw * @sizeOf(f32);
    if (pos + values_bytes > payload.len) return error.PayloadTooShort;
    const values_raw = payload[pos .. pos + values_bytes];

    const is_aligned = std.mem.isAligned(@intFromPtr(values_raw.ptr), @alignOf(f32));
    const val_ptr: ?[*]const f32 = if (is_aligned) @ptrCast(@alignCast(values_raw.ptr)) else null;

    const readVal = struct {
        inline fn get(p: ?[*]const f32, bytes: []const u8, idx: usize) f32 {
            if (p) |ptr| return ptr[idx];
            const u = std.mem.readInt(u32, bytes[idx * 4 .. (idx + 1) * 4][0..4], .little);
            return @bitCast(u);
        }
    }.get;

    @memset(y[0..M], 0.0);

    const num_block_rows = M / bh;
    for (0..num_block_rows) |br| {
        if (br + 1 >= num_row_ptrs) break;
        const start_b = std.mem.readInt(u32, row_ptrs_raw[br * 4 ..][0..4], .little);
        const end_b = std.mem.readInt(u32, row_ptrs_raw[(br + 1) * 4 ..][0..4], .little);

        var b = start_b;
        while (b < end_b) : (b += 1) {
            if (b >= num_col_indices) break;
            const bc: usize = std.mem.readInt(u32, col_indices_raw[b * 4 ..][0..4], .little);
            const block_start_elem = b * bh * bw;

            for (0..bh) |r| {
                const global_r = br * bh + r;
                if (global_r >= M) continue;

                var row_sum: f32 = 0.0;
                for (0..bw) |c| {
                    const global_c = bc * bw + c;
                    if (global_c >= K) continue;
                    const elem_idx = block_start_elem + r * bw + c;
                    const v = readVal(val_ptr, values_raw, elem_idx);
                    row_sum += v * x[global_c];
                }
                y[global_r] += row_sum;
            }
        }
    }
}

/// Multi-token matrix-matrix multiplication for Block Sparse Row (BSR) weights:
/// Y[t, :] = W * X[t, :]
/// Memory-first: zero temporary allocations.
pub fn gemmBSR_F32(
    Y: []f32,
    payload: []const u8,
    X: []const f32,
    M: usize,
    N: usize,
    K: usize,
) !void {
    if (Y.len < N * M or X.len < N * K) return error.BufferTooSmall;
    for (0..N) |t| {
        const y_row = Y[t * M .. (t + 1) * M];
        const x_row = X[t * K .. (t + 1) * K];
        try gemvBSR_F32(y_row, payload, x_row, M, K);
    }
}

test "2:4 structured sparse: GEMV & GEMM numerical parity vs dense reference" {
    const allocator = std.testing.allocator;
    const M: usize = 16;
    const K: usize = 32;
    const N: usize = 4;

    var dense_W = try allocator.alloc(f32, M * K);
    defer allocator.free(dense_W);
    @memset(dense_W, 0.0);

    // Create 2:4 structured sparse matrix: 2 non-zeros per group of 4
    for (0..M) |m| {
        for (0..K / 4) |g| {
            const base = (m * (K / 4) + g) * 4;
            dense_W[base + 0] = @as(f32, @floatFromInt(m + 1)) * 0.5;
            dense_W[base + 2] = @as(f32, @floatFromInt(g + 1)) * 1.5;
            // base + 1 and base + 3 remain 0.0
        }
    }

    var x = try allocator.alloc(f32, K);
    defer allocator.free(x);
    for (0..K) |k| {
        x[k] = @as(f32, @floatFromInt(k % 7)) * 0.25;
    }

    var X = try allocator.alloc(f32, N * K);
    defer allocator.free(X);
    for (0..N) |t| {
        for (0..K) |k| {
            X[t * K + k] = @as(f32, @floatFromInt((t + 1) * (k + 1))) * 0.1;
        }
    }

    // Dense reference for GEMV
    var y_ref = try allocator.alloc(f32, M);
    defer allocator.free(y_ref);
    for (0..M) |m| {
        var s: f32 = 0.0;
        for (0..K) |k| {
            s += dense_W[m * K + k] * x[k];
        }
        y_ref[m] = s;
    }

    // Dense reference for GEMM
    var Y_ref = try allocator.alloc(f32, N * M);
    defer allocator.free(Y_ref);
    for (0..N) |t| {
        for (0..M) |m| {
            var s: f32 = 0.0;
            for (0..K) |k| {
                s += dense_W[m * K + k] * X[t * K + k];
            }
            Y_ref[t * M + m] = s;
        }
    }

    // Encode to 2:4 structured sparse payload
    const payload = try encodeStructured2_4_F32(dense_W, allocator);
    defer allocator.free(payload);

    // Run gemvStructured2_4_F32
    const y_sparse = try allocator.alloc(f32, M);
    defer allocator.free(y_sparse);
    try gemvStructured2_4_F32(y_sparse, payload, x, M, K);

    for (0..M) |m| {
        try std.testing.expectApproxEqAbs(y_ref[m], y_sparse[m], 1e-4);
    }

    // Run gemmStructured2_4_F32
    const Y_sparse = try allocator.alloc(f32, N * M);
    defer allocator.free(Y_sparse);
    try gemmStructured2_4_F32(Y_sparse, payload, X, M, N, K);

    for (0..N * M) |i| {
        try std.testing.expectApproxEqAbs(Y_ref[i], Y_sparse[i], 1e-4);
    }
}

test "BSR sparse: GEMV & GEMM numerical parity vs dense reference across block shapes" {
    const allocator = std.testing.allocator;
    const M: usize = 32;
    const K: usize = 32;
    const N: usize = 2;

    inline for (.{ .{ 4, 4 }, .{ 8, 8 }, .{ 16, 16 } }) |dims| {
        const bh = dims[0];
        const bw = dims[1];

        var dense_W = try allocator.alloc(f32, M * K);
        defer allocator.free(dense_W);
        @memset(dense_W, 0.0);

        // Populate every alternate block
        const nbr = M / bh;
        const nbc = K / bw;
        for (0..nbr) |br| {
            for (0..nbc) |bc| {
                if ((br + bc) % 2 == 0) {
                    for (0..bh) |r| {
                        for (0..bw) |c| {
                            const idx = (br * bh + r) * K + bc * bw + c;
                            dense_W[idx] = @as(f32, @floatFromInt(r + c + 1)) * 0.5;
                        }
                    }
                }
            }
        }

        var x = try allocator.alloc(f32, K);
        defer allocator.free(x);
        for (0..K) |k| x[k] = @as(f32, @floatFromInt(k + 1)) * 0.1;

        var X = try allocator.alloc(f32, N * K);
        defer allocator.free(X);
        for (0..N * K) |i| X[i] = @as(f32, @floatFromInt(i % 13)) * 0.2;

        // Dense reference GEMV
        var y_ref = try allocator.alloc(f32, M);
        defer allocator.free(y_ref);
        for (0..M) |m| {
            var s: f32 = 0.0;
            for (0..K) |k| s += dense_W[m * K + k] * x[k];
            y_ref[m] = s;
        }

        // Encode BSR
        const bsr_payload = try encodeBSR_F32(dense_W, M, K, bh, bw, allocator);
        defer allocator.free(bsr_payload);

        // Verify roundtrip decode
        const decoded_W = try allocator.alloc(f32, M * K);
        defer allocator.free(decoded_W);
        try decodeBSR_F32(bsr_payload, M, K, decoded_W);
        for (0..M * K) |i| {
            try std.testing.expectEqual(dense_W[i], decoded_W[i]);
        }

        // Run gemvBSR_F32
        const y_bsr = try allocator.alloc(f32, M);
        defer allocator.free(y_bsr);
        try gemvBSR_F32(y_bsr, bsr_payload, x, M, K);

        for (0..M) |m| {
            try std.testing.expectApproxEqAbs(y_ref[m], y_bsr[m], 1e-4);
        }

        // Run gemmBSR_F32
        var Y_ref = try allocator.alloc(f32, N * M);
        defer allocator.free(Y_ref);
        for (0..N) |t| {
            for (0..M) |m| {
                var s: f32 = 0.0;
                for (0..K) |k| s += dense_W[m * K + k] * X[t * K + k];
                Y_ref[t * M + m] = s;
            }
        }

        const Y_bsr = try allocator.alloc(f32, N * M);
        defer allocator.free(Y_bsr);
        try gemmBSR_F32(Y_bsr, bsr_payload, X, M, N, K);

        for (0..N * M) |i| {
            try std.testing.expectApproxEqAbs(Y_ref[i], Y_bsr[i], 1e-4);
        }
    }
}

test "sparsity guards: invalid dimensions and truncated payloads fail cleanly" {
    const allocator = std.testing.allocator;
    var y: [16]f32 = undefined;
    var x: [16]f32 = undefined;

    // K not divisible by 4
    try std.testing.expectError(error.InvalidStructuredCount, gemvStructured2_4_F32(&y, "", &x, 16, 15));

    // Buffer too small
    var y_small: [4]f32 = undefined;
    try std.testing.expectError(error.BufferTooSmall, gemvStructured2_4_F32(&y_small, "", &x, 16, 16));

    // BSR invalid block dimensions
    try std.testing.expectError(error.InvalidBlockDimensions, encodeBSR_F32(&x, 16, 16, 0, 4, allocator));
    try std.testing.expectError(error.InvalidBlockDimensions, encodeBSR_F32(&x, 16, 16, 5, 4, allocator));

    // Truncated BSR payload
    const short_payload = [_]u8{ 4, 0, 4, 0, 2, 0, 0, 0 }; // only 8 bytes instead of >= 12
    try std.testing.expectError(error.PayloadTooShort, gemvBSR_F32(&y, &short_payload, &x, 16, 16));
}
