//! The ROCm engine: the whole network on one AMD device, through the HIP runtime. The engine itself is
//! `gpu/engine.zig`, shared with the NVIDIA backend; this file instantiates it for `hip.zig`.

const gpu = @import("../gpu/engine.zig");
const Impl = gpu.Make(@import("../hip.zig"));

pub const Error = gpu.Error;
pub const Options = gpu.Options;
pub const supports = gpu.supports;
pub const Engine = Impl.Engine;
pub const layerBytes = Impl.layerBytes;
pub const bytesNeeded = Impl.bytesNeeded;
