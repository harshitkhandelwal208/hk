const std = @import("std");
pub const format = @import("format.zig");
pub const buf = @import("buf.zig");
pub const metadata = @import("metadata.zig");
pub const tensor_toc = @import("tensor_toc.zig");
pub const nf4 = @import("nf4.zig");
pub const quantization = @import("quantization.zig");
pub const quant = @import("quant.zig");
pub const pool = @import("pool.zig");
pub const kernels = @import("kernels.zig");
pub const vk = @import("vk/vk.zig");
pub const gpu = @import("vk/engine.zig");
pub const cores = @import("cores.zig");
/// Instruction set levels built into this binary, comma separated.
pub const build_options_kernel_levels = @import("build_options").kernel_levels;
pub const arrays = @import("arrays.zig");
pub const convert = @import("convert.zig");
pub const hub = @import("hub.zig");
pub const chat = @import("chat.zig");
pub const server = @import("server.zig");
pub const engine = @import("engine.zig");
pub const sparsity = @import("sparsity.zig");
pub const tiling = @import("tiling.zig");
pub const platform = @import("platform.zig");
pub const writer = @import("writer.zig");
pub const reader = @import("reader.zig");
pub const appendix = @import("appendix.zig");
pub const tensor_ops = @import("tensor_ops.zig");
pub const growth = @import("growth.zig");
pub const sandbox = @import("sandbox.zig");
pub const expansion = @import("expansion.zig");
pub const gguf = @import("gguf.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const sampler = @import("sampler.zig");
pub const safetensors = @import("safetensors.zig");
pub const hf_mapper = @import("hf_mapper.zig");
pub const context = @import("context.zig");
pub const pipeline = @import("pipeline.zig");
pub const adaptive = @import("adaptive.zig");
pub const c_api = @import("c_api.zig");
pub const cuda = @import("cuda.zig");

// Re-export key structs
pub const FileHeader = format.FileHeader;
pub const StorageType = format.StorageType;
pub const TileLayout = format.TileLayout;
pub const SparsityType = format.SparsityType;
pub const TensorEntry = format.TensorEntry;
pub const MetadataMap = metadata.MetadataMap;
pub const TensorTOC = tensor_toc.TensorTOC;
pub const HKWriter = writer.HKWriter;
pub const HKReader = reader.HKReader;
pub const TensorPayload = writer.TensorPayload;

// Force export of C ABI functions into dynamic library
comptime {
    _ = c_api;
}

/// Pulls the test blocks of `T` and of every module it re-exports into the test binary. This
/// walks one level of modules itself: each public module declaration is a namespace whose own declarations are
/// referenced, which is what makes their `test` blocks reachable.
fn refModule(comptime T: type) void {
    inline for (comptime std.meta.declarations(T)) |d| {
        const decl = @field(T, d.name);
        if (@TypeOf(decl) == type) {
            switch (@typeInfo(decl)) {
                .@"struct" => refModule(decl),
                else => {},
            }
        }
    }
}

test {
    std.testing.refAllDecls(@This());
    refModule(@This());
}
