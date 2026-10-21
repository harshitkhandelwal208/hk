//! The CUDA engine: the whole network on one NVIDIA device, through the CUDA driver API. The engine itself is
//! `gpu/engine.zig`, shared with the AMD backend; this file instantiates it for `cuda.zig`.

const gpu = @import("../gpu/engine.zig");
const Impl = gpu.Make(@import("../cuda.zig"));

pub const Error = gpu.Error;
pub const Options = gpu.Options;
pub const supports = gpu.supports;
pub const Engine = Impl.Engine;
pub const layerBytes = Impl.layerBytes;
pub const bytesNeeded = Impl.bytesNeeded;
