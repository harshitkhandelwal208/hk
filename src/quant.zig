//! Quantization formats: block geometry, codebooks and exact decoders.
pub const blocks = @import("quant/blocks.zig");
pub const tables = @import("quant/tables.zig");
pub const dequant = @import("quant/dequant.zig");
pub const isa = @import("quant/isa.zig");
pub const unpack = @import("quant/unpack.zig");
pub const vecdot = @import("quant/vecdot.zig");
pub const gemm = @import("quant/gemm.zig");

pub const info = blocks.info;
pub const byteSize = blocks.byteSize;
pub const dequantizeRow = dequant.dequantizeRow;
