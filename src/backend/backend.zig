//! The seam between the model and the device that runs the network.
//!
//! A backend executes the *whole* network for a batch of tokens (embedding to final hidden state) and reads
//! logits back, which is the granularity the Vulkan engine already had and the one CUDA will use. A per-operator
//! graph would force every backend through the same op boundaries and re-introduce host round trips between them;
//! whole-network executors keep each device free to fuse and keep activations resident.
//!
//! The set of backends is closed and known at build time, so this is a tagged union with `inline else` dispatch
//! rather than a vtable: no indirect call, and a backend that is not built for the target is simply absent.
//!
//! Weights are views into the mapped container on the CPU; a device backend uploads them once (a deliberate copy
//! to device memory, reported by `bytesNeeded`). The host `KvCache` is only an identity key for a device
//! backend: the device owns the real cache for the sequence while it holds a slot.

const std = @import("std");
const kv_mod = @import("../engine/kv.zig");
const vk_engine = @import("../vk/engine.zig");
const cuda_engine = @import("../cuda/engine.zig");
const rocm_engine = @import("../rocm/engine.zig");
const cuda = @import("../cuda.zig");
const hip = @import("../hip.zig");
const vk = @import("../vk/vk.zig");
const weights_mod = @import("../engine/weights.zig");
const config_mod = @import("../engine/config.zig");

pub const caps = @import("caps.zig");
pub const select = @import("select.zig");

pub const Kind = enum { vulkan, cuda, rocm };

pub const Error = vk_engine.Error || cuda_engine.Error || rocm_engine.Error;

pub const Item = kv_mod.Item;

/// What running only leading layers costs on a device, from the engine's own accounting: the fixed part is what one
/// layer leaves of `bytesNeeded` after its weights, and the cache cost per position comes from asking for one more.
fn partialNeed(
    w: *const weights_mod.Weights,
    cfg: config_mod.Config,
    opts: anytype,
    comptime bytesNeeded: anytype,
    comptime layerBytes: anytype,
) ?select.Partial {
    if (w.layers.len < 2 or w.layers.len > select.max_layers) return null;
    var o = opts;
    o.layers = 1;
    o.n_ctx = 0;
    const one = bytesNeeded(w, cfg, o);
    o.n_ctx = 1;
    const one_ctx = bytesNeeded(w, cfg, o) - one;
    var part = select.Partial{ .layers = w.layers.len, .fixed = one -| layerBytes(w, 0), .per_layer_ctx = one_ctx };
    for (0..w.layers.len) |i| part.layer_bytes[i] = layerBytes(w, i);
    return part;
}

/// Probes the Vulkan device for the planner: usability, free memory, whether the model's formats have kernels, and
/// what the model needs (linear in the context, so two engine queries give the exact slope).
pub fn probeVulkan(
    allocator: std.mem.Allocator,
    w: *const weights_mod.Weights,
    cfg: config_mod.Config,
    opts: vk_engine.Options,
) select.Candidate {
    var cand = select.Candidate{
        .target = .vulkan,
        .need = .{ .base = 0, .per_ctx = 0 },
        .unsupported = caps.firstUnsupported(.vulkan, w),
    };
    var probe = vk.Context.init(allocator, opts.device) catch |e| {
        cand.unusable = switch (e) {
            error.NoVulkan => "no Vulkan driver found",
            error.NoDevice => "no Vulkan device found",
            error.MissingFeature => "the Vulkan device lacks a feature the kernels need",
            else => "the Vulkan device could not be opened",
        };
        return cand;
    };
    defer probe.deinit();
    if (probe.caps.is_cpu) {
        cand.unusable = "only a software Vulkan device exists";
        return cand;
    }
    cand.free_bytes = probe.freeDeviceBytes();
    var o = opts;
    o.n_ctx = 0;
    const base = vk_engine.Engine.bytesNeeded(w, cfg, o);
    o.n_ctx = 1;
    cand.need = .{ .base = base, .per_ctx = vk_engine.Engine.bytesNeeded(w, cfg, o) - base };
    cand.need.partial = partialNeed(w, cfg, o, vk_engine.Engine.bytesNeeded, vk_engine.Engine.layerBytes);
    return cand;
}

/// Probes CUDA for the planner. See `probeVulkan`; the CUDA engine needs no device index yet.
pub fn probeCuda(
    w: *const weights_mod.Weights,
    cfg: config_mod.Config,
    opts: cuda_engine.Options,
) select.Candidate {
    var cand = select.Candidate{
        .target = .cuda,
        .need = .{ .base = 0, .per_ctx = 0 },
        .unsupported = caps.firstUnsupported(.cuda, w),
    };
    if (!cuda.enabled) {
        cand.unusable = "this build has no CUDA backend";
        return cand;
    }
    if (!cuda.isAvailable()) {
        cand.unusable = "no CUDA driver or device found";
        return cand;
    }
    _ = cuda.deviceKernel("rms_norm") catch {
        cand.unusable = "the CUDA device code could not be loaded by the driver";
        return cand;
    };
    const free = cuda.freeMemBytes() catch {
        cand.unusable = "the CUDA device could not be queried";
        return cand;
    };
    cand.free_bytes = @intCast(@max(free, 0));
    var o = opts;
    o.n_ctx = 0;
    const base = cuda_engine.bytesNeeded(w, cfg, o);
    o.n_ctx = 1;
    cand.need = .{ .base = base, .per_ctx = cuda_engine.bytesNeeded(w, cfg, o) - base };
    cand.need.partial = partialNeed(w, cfg, o, cuda_engine.bytesNeeded, cuda_engine.layerBytes);
    return cand;
}

/// Probes ROCm (HIP) for the planner. See `probeVulkan`.
pub fn probeRocm(
    w: *const weights_mod.Weights,
    cfg: config_mod.Config,
    opts: rocm_engine.Options,
) select.Candidate {
    var cand = select.Candidate{
        .target = .rocm,
        .need = .{ .base = 0, .per_ctx = 0 },
        .unsupported = caps.firstUnsupported(.rocm, w),
    };
    if (!hip.enabled) {
        cand.unusable = "this build has no ROCm backend";
        return cand;
    }
    if (!hip.isAvailable()) {
        cand.unusable = hip.unavailableReason();
        return cand;
    }
    _ = hip.deviceKernel("rms_norm") catch {
        cand.unusable = "the ROCm device code could not be loaded by the runtime";
        return cand;
    };
    const free = hip.freeMemBytes() catch {
        cand.unusable = "the ROCm device could not be queried";
        return cand;
    };
    cand.free_bytes = @intCast(@max(free, 0));
    var o = opts;
    o.n_ctx = 0;
    const base = rocm_engine.bytesNeeded(w, cfg, o);
    o.n_ctx = 1;
    cand.need = .{ .base = base, .per_ctx = rocm_engine.bytesNeeded(w, cfg, o) - base };
    cand.need.partial = partialNeed(w, cfg, o, rocm_engine.bytesNeeded, rocm_engine.layerBytes);
    return cand;
}

pub const Device = union(Kind) {
    vulkan: *vk_engine.Engine,
    cuda: *cuda_engine.Engine,
    rocm: *rocm_engine.Engine,

    pub fn kind(self: Device) Kind {
        return std.meta.activeTag(self);
    }

    /// Name of the compute API, for messages ("running on X (Vulkan)").
    pub fn apiName(self: Device) []const u8 {
        return switch (self) {
            .vulkan => "Vulkan",
            .cuda => "CUDA",
            .rocm => "ROCm",
        };
    }

    pub fn deviceName(self: Device) []const u8 {
        return switch (self) {
            inline else => |e| e.deviceName(),
        };
    }

    /// Most logit rows `logits` can return at once.
    pub fn maxRows(self: Device) usize {
        return switch (self) {
            inline else => |e| @as(usize, e.max_rows),
        };
    }

    /// Runs one forward pass over `items` (tokens of one or more sequences).
    pub fn forward(self: Device, items: []const Item) Error!void {
        return switch (self) {
            inline else => |e| e.forward(items),
        };
    }

    /// Partial offload: runs the device's leading layers over the hidden states in `hidden` ([n][dim], embedded by the
    /// CPU) and leaves the result in `hidden`.
    pub fn forwardPartial(self: Device, items: []const Item, hidden: []f32) Error!void {
        return switch (self) {
            inline else => |e| e.forwardPartial(items, hidden),
        };
    }

    /// Whether this device runs only the leading layers (partial offload).
    pub fn isPartial(self: Device) bool {
        return switch (self) {
            inline else => |e| e.partial,
        };
    }

    /// Layers the device runs.
    pub fn layerCount(self: Device) usize {
        return switch (self) {
            inline else => |e| e.layers.len,
        };
    }

    /// Logits of the given items of the last pass, one vocabulary row each, valid until the next call.
    pub fn logits(self: Device, indices: []const usize) Error![]const f32 {
        return switch (self) {
            inline else => |e| e.logits(indices),
        };
    }

    /// Gives back whatever device state a sequence's cache held.
    pub fn release(self: Device, kv: *const kv_mod.KvCache) void {
        switch (self) {
            inline else => |e| e.release(kv),
        }
    }

    /// Frees the device resources and the engine object itself (which `allocator` created).
    pub fn destroy(self: Device, allocator: std.mem.Allocator) void {
        switch (self) {
            inline else => |e| {
                e.deinit();
                allocator.destroy(e);
            },
        }
    }
};
