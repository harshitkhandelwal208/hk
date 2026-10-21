//! The HIP runtime for AMD GPUs, loaded at run time (`libamdhip64.so` / `amdhip64.dll`). Nothing is linked and no ROCm
//! toolkit is needed to build HK: the device code is Zig (`cuda_device.zig`, compiled for amdgcn by `build.zig`) and the
//! host side is the handful of runtime calls below.
//!
//! This module has the same surface as `cuda.zig` as far as the GPU engine is concerned (`engine_gpu.zig` is written
//! once and instantiated for either), which is why the stream, event and graph types keep their `Cuda` names. HIP's
//! module API mirrors the CUDA driver API, so the code follows `cuda.zig` call for call.
//!
//! Scope, stated plainly: wave32 GPUs only (RDNA 1 to 4; the kernels assume 32 lanes), no matrix cores (WMMA kernels are
//! later work, so prompts use the portable tiled kernel), no graph capture. Status of this backend is in
//! docs/plan/STATUS.md: it compiles and its planner logic is unit tested, but it has not been run on AMD hardware
//! unless that file says otherwise.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const format = @import("format.zig");
const envvar = @import("envvar.zig");
const Loader = @import("dynlib.zig").Loader;

pub const enabled = build_options.rocm;

/// Graph capture is not implemented for HIP; the engine launches every kernel directly.
pub const graphs_supported = false;

pub const HipError = error{
    HipNotCompiledIn,
    OutOfDeviceMemory,
    HipCallFailed,
    NoHipDevice,
    DriverNotLoaded,
    KernelNotFound,
    UnsupportedStorageType,
    GraphsUnsupported,
};

const HipResult = c_int;
const hip_error_out_of_memory: HipResult = 2;
const HipModule = ?*anyopaque;
const HipFunction = ?*anyopaque;
const HipStream = ?*anyopaque;
const HipEvent = ?*anyopaque;

const Api = struct {
    hipInit: *const fn (flags: c_uint) callconv(.c) HipResult,
    hipGetDeviceCount: *const fn (count: *c_int) callconv(.c) HipResult,
    hipSetDevice: *const fn (device: c_int) callconv(.c) HipResult,
    hipDeviceGetName: *const fn (name: [*]u8, len: c_int, dev: c_int) callconv(.c) HipResult,
    hipDeviceTotalMem: *const fn (bytes: *usize, dev: c_int) callconv(.c) HipResult,
    hipMemGetInfo: *const fn (free: *usize, total: *usize) callconv(.c) HipResult,
    hipDeviceSynchronize: *const fn () callconv(.c) HipResult,
    hipMalloc: *const fn (ptr: *?*anyopaque, bytes: usize) callconv(.c) HipResult,
    hipFree: *const fn (ptr: ?*anyopaque) callconv(.c) HipResult,
    hipMemcpyHtoD: *const fn (dst: ?*anyopaque, src: ?*const anyopaque, bytes: usize) callconv(.c) HipResult,
    hipMemcpyDtoH: *const fn (dst: ?*anyopaque, src: ?*anyopaque, bytes: usize) callconv(.c) HipResult,
    hipStreamCreate: *const fn (stream: *HipStream) callconv(.c) HipResult,
    hipStreamDestroy: *const fn (stream: HipStream) callconv(.c) HipResult,
    hipStreamSynchronize: *const fn (stream: HipStream) callconv(.c) HipResult,
    hipEventCreate: *const fn (event: *HipEvent) callconv(.c) HipResult,
    hipEventDestroy: *const fn (event: HipEvent) callconv(.c) HipResult,
    hipEventRecord: *const fn (event: HipEvent, stream: HipStream) callconv(.c) HipResult,
    hipEventSynchronize: *const fn (event: HipEvent) callconv(.c) HipResult,
    hipEventElapsedTime: *const fn (ms: *f32, start: HipEvent, stop: HipEvent) callconv(.c) HipResult,
    hipModuleLoadData: *const fn (module: *HipModule, image: ?*const anyopaque) callconv(.c) HipResult,
    hipModuleGetFunction: *const fn (f: *HipFunction, module: HipModule, name: [*:0]const u8) callconv(.c) HipResult,
    hipModuleLaunchKernel: *const fn (
        f: HipFunction,
        gx: c_uint,
        gy: c_uint,
        gz: c_uint,
        bx: c_uint,
        by: c_uint,
        bz: c_uint,
        shared_bytes: c_uint,
        stream: HipStream,
        params: ?[*]const ?*anyopaque,
        extra: ?[*]const ?*anyopaque,
    ) callconv(.c) HipResult,
};

/// The AMD code object built by `build.zig` from `cuda_device.zig` for the generic gfx11 target (RDNA3 and newer
/// share one binary; the runtime refuses it on a GPU of another family, which `init` reports as an unusable device).
const DEVICE_CODE_OBJECT: []const u8 = if (enabled) @embedFile("rocm_device_co") else "";

const State = struct {
    lib: ?*anyopaque = null,
    api: Api = undefined,
    ready: bool = false,
    attempted: bool = false,
    device: c_int = 0,
    module: HipModule = null,
    fns: std.StringHashMapUnmanaged(HipFunction) = .empty,
    /// Why the backend is not usable, for the planner's "rejected because" line.
    why_not: []const u8 = "",
};

const Mutex = struct {
    state: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    fn lock(self: *Mutex) void {
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }
    fn unlock(self: *Mutex) void {
        self.state.store(0, .release);
    }
};

var state: State = .{};
var mutex: Mutex = .{};
/// HIP keeps the current device per thread; a server worker thread has not selected one.
threadlocal var thread_bound: bool = false;

fn lib_names() []const [*:0]const u8 {
    return if (builtin.os.tag == .windows)
        &.{ "amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll" }
    else
        &.{ "libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5" };
}

fn proc(comptime T: type, lib: *anyopaque, name: [*:0]const u8) ?T {
    const p = Loader.lookup(lib, name) orelse return null;
    return @ptrCast(@alignCast(p));
}

fn fail(msg: []const u8, err: HipError) HipError {
    state.why_not = msg;
    return err;
}

fn ensure() HipError!*State {
    if (!enabled) return HipError.HipNotCompiledIn;
    if (thread_bound and state.ready) return &state;
    mutex.lock();
    defer mutex.unlock();
    if (state.ready) {
        if (state.api.hipSetDevice(state.device) != 0) return HipError.HipCallFailed;
        thread_bound = true;
        return &state;
    }
    if (state.attempted) return HipError.DriverNotLoaded;
    state.attempted = true;

    var lib: ?*anyopaque = null;
    for (lib_names()) |n| {
        lib = Loader.open(n);
        if (lib != null) break;
    }
    const handle = lib orelse return fail("no ROCm: the HIP runtime (libamdhip64) is not installed", HipError.DriverNotLoaded);

    var api: Api = undefined;
    const info = @typeInfo(Api).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(api, name) = proc(T, handle, name) orelse return fail("ROCm: the HIP runtime is missing " ++ name, HipError.DriverNotLoaded);
    }
    if (api.hipInit(0) != 0) return fail("ROCm: hipInit failed (is the amdgpu driver loaded and /dev/kfd readable? the user needs to be in the `render` group)", HipError.NoHipDevice);
    var count: c_int = 0;
    if (api.hipGetDeviceCount(&count) != 0 or count <= 0) return fail("ROCm: the HIP runtime sees no AMD GPU (for an unsupported RDNA3 iGPU, try HSA_OVERRIDE_GFX_VERSION=11.0.0)", HipError.NoHipDevice);

    var dev: c_int = 0;
    if (envvar.get("HK_HIP_DEVICE")) |v| dev = std.fmt.parseInt(c_int, v, 10) catch 0;
    if (dev < 0 or dev >= count) dev = 0;
    if (api.hipSetDevice(dev) != 0) return fail("ROCm: hipSetDevice failed", HipError.HipCallFailed);

    var module: HipModule = null;
    if (api.hipModuleLoadData(&module, DEVICE_CODE_OBJECT.ptr) != 0 or module == null) {
        return fail("ROCm: the GPU cannot run the bundled kernels (they target RDNA3 and newer, gfx11 and gfx12)", HipError.NoHipDevice);
    }

    state.lib = handle;
    state.api = api;
    state.device = dev;
    state.module = module;
    state.ready = true;
    thread_bound = true;
    return &state;
}

/// Why `isAvailable` is false, or "" when it is true.
pub fn unavailableReason() []const u8 {
    if (!enabled) return "HK was built without the ROCm backend (-Drocm=false)";
    _ = ensure() catch return state.why_not;
    return "";
}

pub fn deviceCount() i32 {
    const s = ensure() catch return 0;
    var count: c_int = 0;
    if (s.api.hipGetDeviceCount(&count) != 0) return 0;
    return @intCast(count);
}

pub fn isAvailable() bool {
    return deviceCount() > 0;
}

pub fn getDeviceName(buf: []u8) ![]const u8 {
    const s = try ensure();
    if (s.api.hipDeviceGetName(buf.ptr, @intCast(@min(buf.len, std.math.maxInt(c_int))), s.device) != 0) return HipError.HipCallFailed;
    return std.mem.sliceTo(buf, 0);
}

pub fn totalMemBytes() !i64 {
    const s = try ensure();
    var bytes: usize = 0;
    if (s.api.hipDeviceTotalMem(&bytes, s.device) != 0) return HipError.HipCallFailed;
    return @intCast(bytes);
}

pub fn freeMemBytes() !i64 {
    const s = try ensure();
    var free_b: usize = 0;
    var total_b: usize = 0;
    if (s.api.hipMemGetInfo(&free_b, &total_b) != 0) return HipError.HipCallFailed;
    return @intCast(free_b);
}

pub fn synchronize() !void {
    const s = try ensure();
    if (s.api.hipDeviceSynchronize() != 0) return HipError.HipCallFailed;
}

// -- streams, events, graphs (the engine's names) ------------------------------------------------------------------

pub const CudaStream = struct {
    handle: HipStream,

    pub fn create() !CudaStream {
        const s = try ensure();
        var h: HipStream = null;
        if (s.api.hipStreamCreate(&h) != 0 or h == null) return HipError.HipCallFailed;
        return .{ .handle = h };
    }
    pub fn destroy(self: *CudaStream) void {
        const s = ensure() catch return;
        _ = s.api.hipStreamDestroy(self.handle);
    }
    pub fn synchronize(self: *const CudaStream) !void {
        const s = try ensure();
        if (s.api.hipStreamSynchronize(self.handle) != 0) return HipError.HipCallFailed;
    }
};

pub const CudaEvent = struct {
    handle: HipEvent,

    pub fn create() !CudaEvent {
        const s = try ensure();
        var h: HipEvent = null;
        if (s.api.hipEventCreate(&h) != 0 or h == null) return HipError.HipCallFailed;
        return .{ .handle = h };
    }
    pub fn destroy(self: *CudaEvent) void {
        const s = ensure() catch return;
        _ = s.api.hipEventDestroy(self.handle);
    }
    pub fn record(self: *const CudaEvent, stream: ?CudaStream) !void {
        const s = try ensure();
        if (s.api.hipEventRecord(self.handle, if (stream) |st| st.handle else null) != 0) return HipError.HipCallFailed;
    }
    pub fn synchronize(self: *const CudaEvent) !void {
        const s = try ensure();
        if (s.api.hipEventSynchronize(self.handle) != 0) return HipError.HipCallFailed;
    }
    pub fn elapsedMs(start: *const CudaEvent, end: *const CudaEvent) !f32 {
        const s = try ensure();
        var ms: f32 = 0;
        if (s.api.hipEventElapsedTime(&ms, start.handle, end.handle) != 0) return HipError.HipCallFailed;
        return ms;
    }
};

/// Graph capture is not implemented (see `graphs_supported`); these exist so the engine compiles against either runtime.
pub const CudaGraphExec = struct {
    pub fn launch(_: *const CudaGraphExec, _: ?CudaStream) !void {
        return HipError.GraphsUnsupported;
    }
    pub fn destroy(_: *CudaGraphExec) void {}
};

pub const CudaGraph = struct {
    pub fn beginCapture(_: CudaStream) !void {
        return HipError.GraphsUnsupported;
    }
    pub fn endCapture(_: CudaStream) !CudaGraph {
        return HipError.GraphsUnsupported;
    }
    pub fn instantiate(_: *const CudaGraph) !CudaGraphExec {
        return HipError.GraphsUnsupported;
    }
    pub fn destroy(_: *CudaGraph) void {}
};

// -- device memory ---------------------------------------------------------------------------------------------------

/// A device allocation with an exact lifetime (model weights, KV cache), as in `cuda.zig`.
pub const DeviceMem = struct {
    addr: u64,
    len: usize,

    pub fn at(self: DeviceMem, offset: usize) u64 {
        std.debug.assert(offset <= self.len);
        return self.addr + offset;
    }
};

pub fn deviceAlloc(nbytes: usize) !DeviceMem {
    const s = try ensure();
    var p: ?*anyopaque = null;
    // 16 bytes of slack, as on CUDA: kernels that read whole words around unaligned blocks may touch the bytes after the last one.
    const r = s.api.hipMalloc(&p, @max(nbytes, 1) + 16);
    if (r == hip_error_out_of_memory) return HipError.OutOfDeviceMemory;
    if (r != 0 or p == null) return HipError.HipCallFailed;
    return .{ .addr = @intFromPtr(p.?), .len = nbytes };
}

pub fn deviceFree(m: DeviceMem) void {
    const s = ensure() catch return;
    _ = s.api.hipFree(@ptrFromInt(m.addr));
}

pub fn copyToDevice(dst: u64, src: []const u8) !void {
    if (src.len == 0) return;
    const s = try ensure();
    if (s.api.hipMemcpyHtoD(@ptrFromInt(dst), src.ptr, src.len) != 0) return HipError.HipCallFailed;
}

pub fn copyFromDevice(dst: []u8, src: u64) !void {
    if (dst.len == 0) return;
    const s = try ensure();
    if (s.api.hipMemcpyDtoH(dst.ptr, @ptrFromInt(src), dst.len) != 0) return HipError.HipCallFailed;
}

// -- kernels ---------------------------------------------------------------------------------------------------------

pub const Kernel = struct { f: HipFunction, name: []const u8 };

pub fn deviceKernel(name: []const u8) !Kernel {
    const s = try ensure();
    mutex.lock();
    defer mutex.unlock();
    if (s.fns.getEntry(name)) |e| return .{ .f = e.value_ptr.*, .name = e.key_ptr.* };
    var buf: [64]u8 = undefined;
    if (name.len >= buf.len) return HipError.KernelNotFound;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    var f: HipFunction = null;
    if (s.api.hipModuleGetFunction(&f, s.module, @ptrCast(&buf)) != 0 or f == null) return HipError.KernelNotFound;
    const key = std.heap.page_allocator.dupe(u8, name) catch return error.OutOfMemory;
    s.fns.put(std.heap.page_allocator, key, f) catch return error.OutOfMemory;
    return .{ .f = f, .name = key };
}

/// Threads per block of the matrix kernels (see `cuda.zig`; the device code is the same).
pub const gemm_block_threads: c_uint = 128;
pub const gemv_rows: usize = 4;
pub const gemm_tile_rows: usize = 16;
pub const gemm_tile_tokens: usize = 8;
pub const tensor_min_tokens: usize = 16;
pub const tensor_tile_rows: usize = 64;
pub const tensor_tile_tokens: usize = 128;
pub const tensor_block_threads: c_uint = 256;

pub fn gemmGrid(m: usize, n: usize) [2]u32 {
    if (n <= 1) return .{ @intCast((m + gemv_rows - 1) / gemv_rows), 1 };
    return .{ @intCast((m + gemm_tile_rows - 1) / gemm_tile_rows), @intCast((n + gemm_tile_tokens - 1) / gemm_tile_tokens) };
}

/// No matrix-core kernels on AMD yet.
pub fn hasTensorCores() bool {
    return false;
}
pub fn tensorKernel(_: format.StorageType) ?Kernel {
    return null;
}
pub fn immaKernel(_: format.StorageType) ?Kernel {
    return null;
}

/// The single-token integer kernel for `fmt` (activations quantized by `quantize_act`, products with the 4-way int8 dot
/// product), or null when the format has none. `HK_NO_DP4A=1` turns these kernels off.
pub fn dp4aKernel(fmt: format.StorageType) ?Kernel {
    if (envvar.isSet("HK_NO_DP4A")) return null;
    var name_buf: [48]u8 = undefined;
    const name = std.fmt.bufPrint(&name_buf, "gemv_dp4a_{s}", .{@tagName(fmt)}) catch return null;
    return deviceKernel(name) catch null;
}

pub fn gemmKernel(fmt: format.StorageType, n: usize) !Kernel {
    var name_buf: [48]u8 = undefined;
    const name = std.fmt.bufPrint(&name_buf, "{s}_{s}", .{ if (n <= 1) "gemv" else "gemm", @tagName(fmt) }) catch return HipError.UnsupportedStorageType;
    return deviceKernel(name) catch |e| switch (e) {
        error.KernelNotFound => HipError.UnsupportedStorageType,
        else => e,
    };
}

pub fn envDisabled(name: [*:0]const u8) bool {
    return envvar.isSet(name);
}

var launch_stream: HipStream = null;

pub fn setLaunchStream(stream: ?CudaStream) void {
    launch_stream = if (stream) |st| st.handle else null;
}

pub fn launchKernel(k: Kernel, grid: u32, block: u32, args: anytype) !void {
    return launchKernel2D(k, grid, 1, block, args);
}

/// Launches `k` with a two dimensional grid. `args` is a tuple whose fields match the kernel's parameters exactly in
/// size: device pointers as `u64`, counts as `u32`, scalars as `f32`.
pub fn launchKernel2D(k: Kernel, grid_x: u32, grid_y: u32, block: u32, args: anytype) !void {
    const s = try ensure();
    var storage = args;
    const n = args.len;
    var ptrs: [n]?*anyopaque = undefined;
    inline for (0..n) |i| ptrs[i] = @ptrCast(&storage[i]);
    if (profiling.on) try profiling.begin(s);
    if (s.api.hipModuleLaunchKernel(k.f, grid_x, grid_y, 1, block, 1, 1, 0, launch_stream, &ptrs, null) != 0) return HipError.HipCallFailed;
    if (profiling.on) try profiling.end(s, k.name);
}

/// Per kernel GPU time, as in `cuda.zig`: synchronizes after every launch, so it says where the time goes, not how fast it is.
pub const profiling = struct {
    const max_kernels = 96;
    pub var on = false;
    var start: HipEvent = null;
    var stop: HipEvent = null;
    var names: [max_kernels][]const u8 = undefined;
    var ms: [max_kernels]f64 = @splat(0);
    var calls: [max_kernels]u64 = @splat(0);
    var count: usize = 0;

    pub fn enable() !void {
        const s = try ensure();
        if (on) return;
        if (s.api.hipEventCreate(&start) != 0 or s.api.hipEventCreate(&stop) != 0) return HipError.HipCallFailed;
        on = true;
    }

    pub fn reset() void {
        ms = @splat(0);
        calls = @splat(0);
    }

    fn begin(s: *State) !void {
        if (s.api.hipEventRecord(start, null) != 0) return HipError.HipCallFailed;
    }

    fn end(s: *State, name: []const u8) !void {
        if (s.api.hipEventRecord(stop, null) != 0 or s.api.hipEventSynchronize(stop) != 0) return HipError.HipCallFailed;
        var t: f32 = 0;
        if (s.api.hipEventElapsedTime(&t, start, stop) != 0) return HipError.HipCallFailed;
        var i: usize = 0;
        while (i < count and !std.mem.eql(u8, names[i], name)) : (i += 1) {}
        if (i == count) {
            if (count == max_kernels) return;
            names[count] = name;
            count += 1;
        }
        ms[i] += t;
        calls[i] += 1;
    }

    pub fn report(w: *std.Io.Writer) std.Io.Writer.Error!void {
        var total: f64 = 0;
        for (ms[0..count]) |v| total += v;
        for (names[0..count], ms[0..count], calls[0..count]) |nm, t, c| {
            const mean_us = if (c == 0) 0 else t * 1000.0 / @as(f64, @floatFromInt(c));
            try w.print("  {s:<18} {d:>7} calls {d:>9.2} ms  {d:>8.1} us/call  {d:>5.1}%\n", .{ nm, c, t, mean_us, if (total > 0) 100.0 * t / total else 0 });
        }
        try w.print("  total {d:.2} ms\n", .{total});
    }
};
