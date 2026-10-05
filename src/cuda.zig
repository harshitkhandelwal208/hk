//! Production-grade CUDA GPU Subsystem for HK.
//!
//! Architected on native CUDA Driver API & PyTorch CUDA runtime principles:
//! 1. Zero-dependency runtime dynamic loading (nvcuda.dll / libcuda.so)
//! 2. Embedded PTX kernel compilation (no build-time nvcc required)
//! 3. Asynchronous CUDA Streams & High-precision Event Timing
//! 4. Binned Block Caching Allocator (zero malloc latency)
//! 5. Hardware CUDA Graph Capture & Replay
//! 6. Optimized & Fused Tensor Operators (GEMV, RMSNorm, RoPE, GQA, SwiGLU)

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const format = @import("format.zig");

pub const enabled = build_options.cuda;

pub const CudaError = error{
    CudaNotCompiledIn,
    CudaCallFailed,
    NoCudaDevice,
    UnsupportedStorageType,
    DriverNotLoaded,
    KernelNotFound,
};

// ---------------------------------------------------------------------
// Platform Dynamic Library Loader
// ---------------------------------------------------------------------
const Loader = struct {
    const is_windows = builtin.os.tag == .windows;

    extern "kernel32" fn LoadLibraryA(lpLibFileName: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn GetProcAddress(hModule: ?*anyopaque, lpProcName: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn FreeLibrary(hLibModule: ?*anyopaque) callconv(.winapi) c_int;

    extern "c" fn dlopen(filename: [*:0]const u8, flags: c_int) ?*anyopaque;
    extern "c" fn dlsym(handle: ?*anyopaque, symbol: [*:0]const u8) ?*anyopaque;
    extern "c" fn dlclose(handle: ?*anyopaque) c_int;

    pub fn open(path: [*:0]const u8) ?*anyopaque {
        if (is_windows) {
            return LoadLibraryA(path);
        } else {
            return dlopen(path, 1); // RTLD_LAZY
        }
    }

    pub fn lookup(handle: ?*anyopaque, name: [*:0]const u8) ?*anyopaque {
        if (handle == null) return null;
        if (is_windows) {
            return GetProcAddress(handle, name);
        } else {
            return dlsym(handle, name);
        }
    }

    pub fn close(handle: ?*anyopaque) void {
        if (handle == null) return;
        if (is_windows) {
            _ = FreeLibrary(handle);
        } else {
            _ = dlclose(handle);
        }
    }
};

// ---------------------------------------------------------------------
// CUDA Driver API Function Pointer Types
// ---------------------------------------------------------------------
const CuResult = c_int;
const CuDevice = c_int;
const CuContext = ?*anyopaque;
const CuModule = ?*anyopaque;
const CuFunction = ?*anyopaque;
const CuStream = ?*anyopaque;
const CuEvent = ?*anyopaque;
const CuGraph = ?*anyopaque;
const CuGraphExec = ?*anyopaque;
const CuDeviceptr = u64;

const DriverApi = struct {
    cuInit: *const fn (flags: c_uint) callconv(.c) CuResult,
    cuDeviceGetCount: *const fn (count: *c_int) callconv(.c) CuResult,
    cuDeviceGet: *const fn (device: *CuDevice, ordinal: c_int) callconv(.c) CuResult,
    cuDeviceGetName: *const fn (name: [*]u8, len: c_int, dev: CuDevice) callconv(.c) CuResult,
    cuDeviceTotalMem_v2: *const fn (bytes: *usize, dev: CuDevice) callconv(.c) CuResult,
    cuMemGetInfo_v2: *const fn (free: *usize, total: *usize) callconv(.c) CuResult,
    cuCtxCreate_v2: *const fn (pctx: *CuContext, flags: c_uint, dev: CuDevice) callconv(.c) CuResult,
    cuCtxDestroy_v2: *const fn (ctx: CuContext) callconv(.c) CuResult,
    cuCtxSynchronize: *const fn () callconv(.c) CuResult,
    cuMemAlloc_v2: *const fn (dptr: *CuDeviceptr, bytesize: usize) callconv(.c) CuResult,
    cuMemFree_v2: *const fn (dptr: CuDeviceptr) callconv(.c) CuResult,
    cuMemcpyHtoD_v2: *const fn (dstDevice: CuDeviceptr, srcHost: ?*const anyopaque, ByteCount: usize) callconv(.c) CuResult,
    cuMemcpyDtoH_v2: *const fn (dstHost: ?*anyopaque, srcDevice: CuDeviceptr, ByteCount: usize) callconv(.c) CuResult,
    cuMemcpyHtoDAsync_v2: *const fn (dstDevice: CuDeviceptr, srcHost: ?*const anyopaque, ByteCount: usize, hStream: CuStream) callconv(.c) CuResult,
    cuMemcpyDtoHAsync_v2: *const fn (dstHost: ?*anyopaque, srcDevice: CuDeviceptr, ByteCount: usize, hStream: CuStream) callconv(.c) CuResult,
    cuStreamCreate: *const fn (phStream: *CuStream, Flags: c_uint) callconv(.c) CuResult,
    cuStreamDestroy_v2: *const fn (hStream: CuStream) callconv(.c) CuResult,
    cuStreamSynchronize: *const fn (hStream: CuStream) callconv(.c) CuResult,
    cuEventCreate: *const fn (phEvent: *CuEvent, Flags: c_uint) callconv(.c) CuResult,
    cuEventDestroy_v2: *const fn (hEvent: CuEvent) callconv(.c) CuResult,
    cuEventRecord: *const fn (hEvent: CuEvent, hStream: CuStream) callconv(.c) CuResult,
    cuEventSynchronize: *const fn (hEvent: CuEvent) callconv(.c) CuResult,
    cuEventElapsedTime: *const fn (pMilliseconds: *f32, hStart: CuEvent, hEnd: CuEvent) callconv(.c) CuResult,
    cuModuleLoadData: *const fn (module: *CuModule, image: ?*const anyopaque) callconv(.c) CuResult,
    cuModuleGetFunction: *const fn (hfunc: *CuFunction, hmod: CuModule, name: [*:0]const u8) callconv(.c) CuResult,
    cuLaunchKernel: *const fn (
        f: CuFunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: CuStream,
        kernelParams: ?[*]const ?*anyopaque,
        extra: ?[*]const ?*anyopaque,
    ) callconv(.c) CuResult,
    cuGraphCreate: *const fn (phGraph: *CuGraph, flags: c_uint) callconv(.c) CuResult,
    cuStreamBeginCapture_v2: *const fn (hStream: CuStream, mode: c_int) callconv(.c) CuResult,
    cuStreamEndCapture: *const fn (hStream: CuStream, phGraph: *CuGraph) callconv(.c) CuResult,
    cuGraphInstantiate_v2: *const fn (phGraphExec: *CuGraphExec, hGraph: CuGraph, phErrorNode: ?*anyopaque, pLogBuffer: ?*anyopaque, bufferSize: usize) callconv(.c) CuResult,
    cuGraphLaunch: *const fn (hGraphExec: CuGraphExec, hStream: CuStream) callconv(.c) CuResult,
    cuGraphDestroy: *const fn (hGraph: CuGraph) callconv(.c) CuResult,
    cuGraphExecDestroy: *const fn (hGraphExec: CuGraphExec) callconv(.c) CuResult,
};

// ---------------------------------------------------------------------
// Embedded PTX Kernel Module
// ---------------------------------------------------------------------
const PTX_SOURCE: [:0]const u8 = @embedFile("cuda/hk_cuda.ptx");

const DriverState = struct {
    lib_handle: ?*anyopaque = null,
    api: ?DriverApi = null,
    initialized: bool = false,
    init_attempted: bool = false,
    device: CuDevice = 0,
    context: CuContext = null,
    module: CuModule = null,

    // Kernel Function Pointers
    fn_gemv_f32: CuFunction = null,
    fn_gemv_q8_0: CuFunction = null,
    fn_gemv_q4_0: CuFunction = null,
    fn_rmsnorm: CuFunction = null,
    fn_head_rmsnorm: CuFunction = null,
    fn_rope: CuFunction = null,
    fn_kv_cache_update: CuFunction = null,
    fn_gqa_attention: CuFunction = null,
    fn_swiglu: CuFunction = null,
    fn_add_residual: CuFunction = null,
};

const Mutex = struct {
    state: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

    pub fn lock(self: *Mutex) void {
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }

    pub fn unlock(self: *Mutex) void {
        self.state.store(0, .release);
    }
};

var global_state: DriverState = .{};
var global_mutex: Mutex = .{};

fn ensureDriver() !*DriverState {
    if (!enabled) return CudaError.CudaNotCompiledIn;
    global_mutex.lock();
    defer global_mutex.unlock();

    if (global_state.initialized) return &global_state;
    if (global_state.init_attempted) return CudaError.DriverNotLoaded;
    global_state.init_attempted = true;

    const lib_name: [*:0]const u8 = if (builtin.os.tag == .windows) "nvcuda.dll" else "libcuda.so.1";
    const handle = Loader.open(lib_name) orelse {
        // Fallback on Linux for alternative soname
        const alt_handle = if (builtin.os.tag != .windows) Loader.open("libcuda.so") else null;
        if (alt_handle == null) return CudaError.DriverNotLoaded;
        return initFromHandle(alt_handle.?);
    };

    return initFromHandle(handle);
}

fn getProc(comptime T: type, handle: *anyopaque, name: [*:0]const u8) ?T {
    const p = Loader.lookup(handle, name) orelse return null;
    return @ptrCast(@alignCast(p));
}

fn initFromHandle(handle: *anyopaque) !*DriverState {
    const api = DriverApi{
        .cuInit = getProc(*const fn (c_uint) callconv(.c) CuResult, handle, "cuInit") orelse return CudaError.DriverNotLoaded,
        .cuDeviceGetCount = getProc(*const fn (*c_int) callconv(.c) CuResult, handle, "cuDeviceGetCount") orelse return CudaError.DriverNotLoaded,
        .cuDeviceGet = getProc(*const fn (*CuDevice, c_int) callconv(.c) CuResult, handle, "cuDeviceGet") orelse return CudaError.DriverNotLoaded,
        .cuDeviceGetName = getProc(*const fn ([*]u8, c_int, CuDevice) callconv(.c) CuResult, handle, "cuDeviceGetName") orelse return CudaError.DriverNotLoaded,
        .cuDeviceTotalMem_v2 = getProc(*const fn (*usize, CuDevice) callconv(.c) CuResult, handle, "cuDeviceTotalMem_v2") orelse return CudaError.DriverNotLoaded,
        .cuMemGetInfo_v2 = getProc(*const fn (*usize, *usize) callconv(.c) CuResult, handle, "cuMemGetInfo_v2") orelse return CudaError.DriverNotLoaded,
        .cuCtxCreate_v2 = getProc(*const fn (*CuContext, c_uint, CuDevice) callconv(.c) CuResult, handle, "cuCtxCreate_v2") orelse return CudaError.DriverNotLoaded,
        .cuCtxDestroy_v2 = getProc(*const fn (CuContext) callconv(.c) CuResult, handle, "cuCtxDestroy_v2") orelse return CudaError.DriverNotLoaded,
        .cuCtxSynchronize = getProc(*const fn () callconv(.c) CuResult, handle, "cuCtxSynchronize") orelse return CudaError.DriverNotLoaded,
        .cuMemAlloc_v2 = getProc(*const fn (*CuDeviceptr, usize) callconv(.c) CuResult, handle, "cuMemAlloc_v2") orelse return CudaError.DriverNotLoaded,
        .cuMemFree_v2 = getProc(*const fn (CuDeviceptr) callconv(.c) CuResult, handle, "cuMemFree_v2") orelse return CudaError.DriverNotLoaded,
        .cuMemcpyHtoD_v2 = getProc(*const fn (CuDeviceptr, ?*const anyopaque, usize) callconv(.c) CuResult, handle, "cuMemcpyHtoD_v2") orelse return CudaError.DriverNotLoaded,
        .cuMemcpyDtoH_v2 = getProc(*const fn (?*anyopaque, CuDeviceptr, usize) callconv(.c) CuResult, handle, "cuMemcpyDtoH_v2") orelse return CudaError.DriverNotLoaded,
        .cuMemcpyHtoDAsync_v2 = getProc(*const fn (CuDeviceptr, ?*const anyopaque, usize, CuStream) callconv(.c) CuResult, handle, "cuMemcpyHtoDAsync_v2") orelse return CudaError.DriverNotLoaded,
        .cuMemcpyDtoHAsync_v2 = getProc(*const fn (?*anyopaque, CuDeviceptr, usize, CuStream) callconv(.c) CuResult, handle, "cuMemcpyDtoHAsync_v2") orelse return CudaError.DriverNotLoaded,
        .cuStreamCreate = getProc(*const fn (*CuStream, c_uint) callconv(.c) CuResult, handle, "cuStreamCreate") orelse return CudaError.DriverNotLoaded,
        .cuStreamDestroy_v2 = getProc(*const fn (CuStream) callconv(.c) CuResult, handle, "cuStreamDestroy_v2") orelse return CudaError.DriverNotLoaded,
        .cuStreamSynchronize = getProc(*const fn (CuStream) callconv(.c) CuResult, handle, "cuStreamSynchronize") orelse return CudaError.DriverNotLoaded,
        .cuEventCreate = getProc(*const fn (*CuEvent, c_uint) callconv(.c) CuResult, handle, "cuEventCreate") orelse return CudaError.DriverNotLoaded,
        .cuEventDestroy_v2 = getProc(*const fn (CuEvent) callconv(.c) CuResult, handle, "cuEventDestroy_v2") orelse return CudaError.DriverNotLoaded,
        .cuEventRecord = getProc(*const fn (CuEvent, CuStream) callconv(.c) CuResult, handle, "cuEventRecord") orelse return CudaError.DriverNotLoaded,
        .cuEventSynchronize = getProc(*const fn (CuEvent) callconv(.c) CuResult, handle, "cuEventSynchronize") orelse return CudaError.DriverNotLoaded,
        .cuEventElapsedTime = getProc(*const fn (*f32, CuEvent, CuEvent) callconv(.c) CuResult, handle, "cuEventElapsedTime") orelse return CudaError.DriverNotLoaded,
        .cuModuleLoadData = getProc(*const fn (*CuModule, ?*const anyopaque) callconv(.c) CuResult, handle, "cuModuleLoadData") orelse return CudaError.DriverNotLoaded,
        .cuModuleGetFunction = getProc(*const fn (*CuFunction, CuModule, [*:0]const u8) callconv(.c) CuResult, handle, "cuModuleGetFunction") orelse return CudaError.DriverNotLoaded,
        .cuLaunchKernel = getProc(*const fn (CuFunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CuStream, ?[*]const ?*anyopaque, ?[*]const ?*anyopaque) callconv(.c) CuResult, handle, "cuLaunchKernel") orelse return CudaError.DriverNotLoaded,
        .cuGraphCreate = getProc(*const fn (*CuGraph, c_uint) callconv(.c) CuResult, handle, "cuGraphCreate") orelse return CudaError.DriverNotLoaded,
        .cuStreamBeginCapture_v2 = getProc(*const fn (CuStream, c_int) callconv(.c) CuResult, handle, "cuStreamBeginCapture_v2") orelse return CudaError.DriverNotLoaded,
        .cuStreamEndCapture = getProc(*const fn (CuStream, *CuGraph) callconv(.c) CuResult, handle, "cuStreamEndCapture") orelse return CudaError.DriverNotLoaded,
        .cuGraphInstantiate_v2 = getProc(*const fn (*CuGraphExec, CuGraph, ?*anyopaque, ?*anyopaque, usize) callconv(.c) CuResult, handle, "cuGraphInstantiate_v2") orelse return CudaError.DriverNotLoaded,
        .cuGraphLaunch = getProc(*const fn (CuGraphExec, CuStream) callconv(.c) CuResult, handle, "cuGraphLaunch") orelse return CudaError.DriverNotLoaded,
        .cuGraphDestroy = getProc(*const fn (CuGraph) callconv(.c) CuResult, handle, "cuGraphDestroy") orelse return CudaError.DriverNotLoaded,
        .cuGraphExecDestroy = getProc(*const fn (CuGraphExec) callconv(.c) CuResult, handle, "cuGraphExecDestroy") orelse return CudaError.DriverNotLoaded,
    };

    if (api.cuInit(0) != 0) return CudaError.DriverNotLoaded;

    var count: c_int = 0;
    if (api.cuDeviceGetCount(&count) != 0 or count <= 0) return CudaError.NoCudaDevice;

    var dev: CuDevice = 0;
    if (api.cuDeviceGet(&dev, 0) != 0) return CudaError.NoCudaDevice;

    var ctx: CuContext = null;
    if (api.cuCtxCreate_v2(&ctx, 0, dev) != 0 or ctx == null) return CudaError.CudaCallFailed;

    var mod: CuModule = null;
    if (api.cuModuleLoadData(&mod, PTX_SOURCE.ptr) != 0 or mod == null) return CudaError.CudaCallFailed;

    global_state.lib_handle = handle;
    global_state.api = api;
    global_state.device = dev;
    global_state.context = ctx;
    global_state.module = mod;

    _ = api.cuModuleGetFunction(&global_state.fn_gemv_f32, mod, "gemvF32KernelOpt");
    _ = api.cuModuleGetFunction(&global_state.fn_gemv_q8_0, mod, "gemvQ8_0KernelOpt");
    _ = api.cuModuleGetFunction(&global_state.fn_gemv_q4_0, mod, "gemvQ4_0KernelOpt");
    _ = api.cuModuleGetFunction(&global_state.fn_rmsnorm, mod, "rmsNormKernel");
    _ = api.cuModuleGetFunction(&global_state.fn_head_rmsnorm, mod, "headRmsNormKernel");
    _ = api.cuModuleGetFunction(&global_state.fn_rope, mod, "ropeKernel");
    _ = api.cuModuleGetFunction(&global_state.fn_kv_cache_update, mod, "kvCacheUpdateKernel");
    _ = api.cuModuleGetFunction(&global_state.fn_gqa_attention, mod, "gqaAttentionKernel");
    _ = api.cuModuleGetFunction(&global_state.fn_swiglu, mod, "swigluKernel");
    _ = api.cuModuleGetFunction(&global_state.fn_add_residual, mod, "addResidualKernel");

    global_state.initialized = true;
    return &global_state;
}

// ---------------------------------------------------------------------
// Device Diagnostics & Queries
// ---------------------------------------------------------------------
pub fn deviceCount() i32 {
    const s = ensureDriver() catch return 0;
    var count: c_int = 0;
    if (s.api.?.cuDeviceGetCount(&count) != 0) return 0;
    return @intCast(count);
}

pub fn isAvailable() bool {
    return deviceCount() > 0;
}

pub fn getDeviceName(buf: []u8) ![]const u8 {
    const s = try ensureDriver();
    const len: c_int = @intCast(@min(buf.len, std.math.maxInt(c_int)));
    if (s.api.?.cuDeviceGetName(buf.ptr, len, s.device) != 0) return CudaError.CudaCallFailed;
    return std_mem_sliceTo(buf, 0);
}

fn std_mem_sliceTo(buf: []u8, sentinel: u8) []const u8 {
    var i: usize = 0;
    while (i < buf.len and buf[i] != sentinel) : (i += 1) {}
    return buf[0..i];
}

pub fn totalMemBytes() !i64 {
    const s = try ensureDriver();
    var bytes: usize = 0;
    if (s.api.?.cuDeviceTotalMem_v2(&bytes, s.device) != 0) return CudaError.CudaCallFailed;
    return @intCast(bytes);
}

pub fn freeMemBytes() !i64 {
    const s = try ensureDriver();
    var free_b: usize = 0;
    var total_b: usize = 0;
    if (s.api.?.cuMemGetInfo_v2(&free_b, &total_b) != 0) return CudaError.CudaCallFailed;
    return @intCast(free_b);
}

// ---------------------------------------------------------------------
// PyTorch-style Binned Caching Allocator
// ---------------------------------------------------------------------
pub const CudaCachingAllocator = struct {
    pub const Block = struct {
        ptr: CuDeviceptr,
        size: usize,
    };

    var mutex: Mutex = .{};
    var small_blocks: [128]?Block = @as([128]?Block, @splat(null));
    var small_count: usize = 0;
    var large_blocks: [64]?Block = @as([64]?Block, @splat(null));
    var large_count: usize = 0;

    pub const SMALL_THRESHOLD: usize = 1024 * 1024; // 1 MB

    pub fn alloc(nbytes: usize) !CuDeviceptr {
        const s = try ensureDriver();
        mutex.lock();
        defer mutex.unlock();

        if (nbytes <= SMALL_THRESHOLD) {
            var best_idx: ?usize = null;
            var best_diff: usize = std.math.maxInt(usize);
            for (0..small_count) |i| {
                if (small_blocks[i]) |b| {
                    if (b.size >= nbytes and (b.size - nbytes) < best_diff) {
                        best_diff = b.size - nbytes;
                        best_idx = i;
                    }
                }
            }
            if (best_idx) |idx| {
                const blk = small_blocks[idx].?;
                small_blocks[idx] = small_blocks[small_count - 1];
                small_blocks[small_count - 1] = null;
                small_count -= 1;
                return blk.ptr;
            }
        } else {
            var best_idx: ?usize = null;
            var best_diff: usize = std.math.maxInt(usize);
            for (0..large_count) |i| {
                if (large_blocks[i]) |b| {
                    if (b.size >= nbytes and (b.size - nbytes) < best_diff) {
                        best_diff = b.size - nbytes;
                        best_idx = i;
                    }
                }
            }
            if (best_idx) |idx| {
                const blk = large_blocks[idx].?;
                large_blocks[idx] = large_blocks[large_count - 1];
                large_blocks[large_count - 1] = null;
                large_count -= 1;
                return blk.ptr;
            }
        }

        var dev_ptr: CuDeviceptr = 0;
        if (s.api.?.cuMemAlloc_v2(&dev_ptr, nbytes) != 0 or dev_ptr == 0) {
            return CudaError.CudaCallFailed;
        }
        return dev_ptr;
    }

    pub fn free(ptr: CuDeviceptr, size: usize) void {
        const s = ensureDriver() catch return;
        mutex.lock();
        defer mutex.unlock();

        if (size <= SMALL_THRESHOLD and small_count < small_blocks.len) {
            small_blocks[small_count] = .{ .ptr = ptr, .size = size };
            small_count += 1;
            return;
        } else if (size > SMALL_THRESHOLD and large_count < large_blocks.len) {
            large_blocks[large_count] = .{ .ptr = ptr, .size = size };
            large_count += 1;
            return;
        }

        _ = s.api.?.cuMemFree_v2(ptr);
    }

    pub fn emptyCache() void {
        const s = ensureDriver() catch return;
        mutex.lock();
        defer mutex.unlock();

        for (0..small_count) |i| {
            if (small_blocks[i]) |b| {
                _ = s.api.?.cuMemFree_v2(b.ptr);
                small_blocks[i] = null;
            }
        }
        small_count = 0;

        for (0..large_count) |i| {
            if (large_blocks[i]) |b| {
                _ = s.api.?.cuMemFree_v2(b.ptr);
                large_blocks[i] = null;
            }
        }
        large_count = 0;
    }
};

pub fn emptyCache() void {
    CudaCachingAllocator.emptyCache();
}

// ---------------------------------------------------------------------
// CUDA Streams & Events
// ---------------------------------------------------------------------
pub const CudaStream = struct {
    handle: CuStream,

    pub fn create() !CudaStream {
        const s = try ensureDriver();
        var h: CuStream = null;
        if (s.api.?.cuStreamCreate(&h, 0) != 0 or h == null) return CudaError.CudaCallFailed;
        return CudaStream{ .handle = h };
    }

    pub fn destroy(self: *CudaStream) void {
        const s = ensureDriver() catch return;
        _ = s.api.?.cuStreamDestroy_v2(self.handle);
    }

    pub fn synchronize(self: *const CudaStream) !void {
        const s = try ensureDriver();
        if (s.api.?.cuStreamSynchronize(self.handle) != 0) return CudaError.CudaCallFailed;
    }
};

pub const CudaEvent = struct {
    handle: CuEvent,

    pub fn create() !CudaEvent {
        const s = try ensureDriver();
        var h: CuEvent = null;
        if (s.api.?.cuEventCreate(&h, 0) != 0 or h == null) return CudaError.CudaCallFailed;
        return CudaEvent{ .handle = h };
    }

    pub fn destroy(self: *CudaEvent) void {
        const s = ensureDriver() catch return;
        _ = s.api.?.cuEventDestroy_v2(self.handle);
    }

    pub fn record(self: *const CudaEvent, stream: ?CudaStream) !void {
        const s = try ensureDriver();
        const s_handle = if (stream) |st| st.handle else null;
        if (s.api.?.cuEventRecord(self.handle, s_handle) != 0) return CudaError.CudaCallFailed;
    }

    pub fn synchronize(self: *const CudaEvent) !void {
        const s = try ensureDriver();
        if (s.api.?.cuEventSynchronize(self.handle) != 0) return CudaError.CudaCallFailed;
    }

    pub fn elapsedMs(start: *const CudaEvent, end: *const CudaEvent) !f32 {
        const s = try ensureDriver();
        var ms: f32 = 0.0;
        if (s.api.?.cuEventElapsedTime(&ms, start.handle, end.handle) != 0) return CudaError.CudaCallFailed;
        return ms;
    }
};

// ---------------------------------------------------------------------
// CUDA Hardware Graphs (Capture & Replay)
// ---------------------------------------------------------------------
pub const CudaGraph = struct {
    handle: CuGraph,

    pub fn beginCapture(stream: CudaStream) !void {
        const s = try ensureDriver();
        if (s.api.?.cuStreamBeginCapture_v2(stream.handle, 0) != 0) return CudaError.CudaCallFailed;
    }

    pub fn endCapture(stream: CudaStream) !CudaGraph {
        const s = try ensureDriver();
        var h: CuGraph = null;
        if (s.api.?.cuStreamEndCapture(stream.handle, &h) != 0 or h == null) return CudaError.CudaCallFailed;
        return CudaGraph{ .handle = h };
    }

    pub fn instantiate(self: *const CudaGraph) !CudaGraphExec {
        const s = try ensureDriver();
        var exec_h: CuGraphExec = null;
        if (s.api.?.cuGraphInstantiate_v2(&exec_h, self.handle, null, null, 0) != 0 or exec_h == null) {
            return CudaError.CudaCallFailed;
        }
        return CudaGraphExec{ .handle = exec_h };
    }

    pub fn destroy(self: *CudaGraph) void {
        const s = ensureDriver() catch return;
        _ = s.api.?.cuGraphDestroy(self.handle);
    }
};

pub const CudaGraphExec = struct {
    handle: CuGraphExec,

    pub fn launch(self: *const CudaGraphExec, stream: ?CudaStream) !void {
        const s = try ensureDriver();
        const s_handle = if (stream) |st| st.handle else null;
        if (s.api.?.cuGraphLaunch(self.handle, s_handle) != 0) return CudaError.CudaCallFailed;
    }

    pub fn destroy(self: *CudaGraphExec) void {
        const s = ensureDriver() catch return;
        _ = s.api.?.cuGraphExecDestroy(self.handle);
    }
};

// ---------------------------------------------------------------------
// Owning Handle to a Device Allocation
// ---------------------------------------------------------------------
pub const DeviceBuffer = struct {
    ptr: *anyopaque,
    len: usize,

    pub fn upload(host_bytes: []const u8) !DeviceBuffer {
        const s = try ensureDriver();
        const dptr = try CudaCachingAllocator.alloc(host_bytes.len);
        if (s.api.?.cuMemcpyHtoD_v2(dptr, host_bytes.ptr, host_bytes.len) != 0) {
            CudaCachingAllocator.free(dptr, host_bytes.len);
            return CudaError.CudaCallFailed;
        }
        return DeviceBuffer{ .ptr = @ptrFromInt(dptr), .len = host_bytes.len };
    }

    pub fn allocUninit(nbytes: usize) !DeviceBuffer {
        const dptr = try CudaCachingAllocator.alloc(nbytes);
        return DeviceBuffer{ .ptr = @ptrFromInt(dptr), .len = nbytes };
    }

    pub fn copyFromHost(self: DeviceBuffer, host_bytes: []const u8) !void {
        const s = try ensureDriver();
        const copy_len = @min(self.len, host_bytes.len);
        const dptr: CuDeviceptr = @intFromPtr(self.ptr);
        if (s.api.?.cuMemcpyHtoD_v2(dptr, host_bytes.ptr, copy_len) != 0) return CudaError.CudaCallFailed;
    }

    pub fn download(self: DeviceBuffer, host_out: []u8) !void {
        const s = try ensureDriver();
        const copy_len = @min(self.len, host_out.len);
        const dptr: CuDeviceptr = @intFromPtr(self.ptr);
        if (s.api.?.cuMemcpyDtoH_v2(host_out.ptr, dptr, copy_len) != 0) return CudaError.CudaCallFailed;
    }

    pub fn free(self: DeviceBuffer) void {
        const dptr: CuDeviceptr = @intFromPtr(self.ptr);
        CudaCachingAllocator.free(dptr, self.len);
    }
};

pub fn synchronize() !void {
    const s = try ensureDriver();
    if (s.api.?.cuCtxSynchronize() != 0) return CudaError.CudaCallFailed;
}

pub fn uploadTo(dev_ptr: *anyopaque, host_bytes: []const u8) !void {
    const s = try ensureDriver();
    const dptr: CuDeviceptr = @intFromPtr(dev_ptr);
    if (s.api.?.cuMemcpyHtoD_v2(dptr, host_bytes.ptr, host_bytes.len) != 0) return CudaError.CudaCallFailed;
}

pub fn downloadFrom(host_bytes: []u8, dev_ptr: *const anyopaque) !void {
    const s = try ensureDriver();
    const dptr: CuDeviceptr = @intFromPtr(dev_ptr);
    if (s.api.?.cuMemcpyDtoH_v2(host_bytes.ptr, dptr, host_bytes.len) != 0) return CudaError.CudaCallFailed;
}

// ---------------------------------------------------------------------
// Unified Pointer Coercion Helpers
// ---------------------------------------------------------------------
inline fn toDevicePtr(val: anytype) CuDeviceptr {
    const T = @TypeOf(val);
    if (T == DeviceBuffer) return @intFromPtr(val.ptr);
    if (T == ?DeviceBuffer) return if (val) |b| @intFromPtr(b.ptr) else 0;
    if (T == *anyopaque or T == *const anyopaque) return @intFromPtr(val);
    if (T == ?*anyopaque or T == ?*const anyopaque) return if (val) |p| @intFromPtr(p) else 0;
    if (T == CuDeviceptr) return val;
    return @intFromPtr(@as(*const anyopaque, @ptrCast(val)));
}

// ---------------------------------------------------------------------
// Kernel Dispatches (Accepts DeviceBuffer or Raw Device Pointers)
// ---------------------------------------------------------------------
pub fn gemvF32(w: anytype, x: anytype, y: anytype, m: usize, k: usize) !void {
    const s = try ensureDriver();
    const fn_ptr = s.fn_gemv_f32 orelse return CudaError.KernelNotFound;

    var d_w = toDevicePtr(w);
    var d_x = toDevicePtr(x);
    var d_y = toDevicePtr(y);
    var k_val: c_int = @intCast(k);

    var params = [_]?*anyopaque{
        @ptrCast(&d_w),
        @ptrCast(&d_x),
        @ptrCast(&d_y),
        @ptrCast(&k_val),
    };

    const res = s.api.?.cuLaunchKernel(
        fn_ptr,
        @intCast(m), 1, 1,
        256, 1, 1,
        0,
        null,
        &params,
        null,
    );
    if (res != 0) return CudaError.CudaCallFailed;
}

pub fn gemvQ8_0(w: anytype, x: anytype, y: anytype, m: usize, k: usize) !void {
    const s = try ensureDriver();
    const fn_ptr = s.fn_gemv_q8_0 orelse return CudaError.KernelNotFound;

    var d_w = toDevicePtr(w);
    var d_x = toDevicePtr(x);
    var d_y = toDevicePtr(y);
    var blocks_per_row: c_int = @intCast(k / 32);

    var params = [_]?*anyopaque{
        @ptrCast(&d_w),
        @ptrCast(&d_x),
        @ptrCast(&d_y),
        @ptrCast(&blocks_per_row),
    };

    const res = s.api.?.cuLaunchKernel(
        fn_ptr,
        @intCast(m), 1, 1,
        256, 1, 1,
        0,
        null,
        &params,
        null,
    );
    if (res != 0) return CudaError.CudaCallFailed;
}

pub fn gemvQ4_0(w: anytype, x: anytype, y: anytype, m: usize, k: usize) !void {
    const s = try ensureDriver();
    const fn_ptr = s.fn_gemv_q4_0 orelse return CudaError.KernelNotFound;

    var d_w = toDevicePtr(w);
    var d_x = toDevicePtr(x);
    var d_y = toDevicePtr(y);
    var blocks_per_row: c_int = @intCast(k / 32);

    var params = [_]?*anyopaque{
        @ptrCast(&d_w),
        @ptrCast(&d_x),
        @ptrCast(&d_y),
        @ptrCast(&blocks_per_row),
    };

    const res = s.api.?.cuLaunchKernel(
        fn_ptr,
        @intCast(m), 1, 1,
        256, 1, 1,
        0,
        null,
        &params,
        null,
    );
    if (res != 0) return CudaError.CudaCallFailed;
}

pub fn matVec(w: anytype, st: format.StorageType, x: anytype, y: anytype, m: usize, k: usize) !void {
    switch (st) {
        .f32 => try gemvF32(w, x, y, m, k),
        .q8_0 => try gemvQ8_0(w, x, y, m, k),
        .q4_0 => try gemvQ4_0(w, x, y, m, k),
        else => return CudaError.UnsupportedStorageType,
    }
}

pub fn rmsNorm(x: anytype, weight: anytype, out: anytype, dim: usize, eps: f32) !void {
    const s = try ensureDriver();
    const fn_ptr = s.fn_rmsnorm orelse return CudaError.KernelNotFound;

    var d_x = toDevicePtr(x);
    var d_w = toDevicePtr(weight);
    var d_out = toDevicePtr(out);
    var dim_val: c_int = @intCast(dim);
    var eps_val: f32 = eps;

    var params = [_]?*anyopaque{
        @ptrCast(&d_x),
        @ptrCast(&d_w),
        @ptrCast(&d_out),
        @ptrCast(&dim_val),
        @ptrCast(&eps_val),
    };

    const res = s.api.?.cuLaunchKernel(
        fn_ptr,
        1, 1, 1,
        256, 1, 1,
        0,
        null,
        &params,
        null,
    );
    if (res != 0) return CudaError.CudaCallFailed;
}

pub fn headRmsNorm(x: anytype, weight: anytype, n_heads: usize, head_dim: usize, weight_len: usize, eps: f32) !void {
    const s = try ensureDriver();
    const fn_ptr = s.fn_head_rmsnorm orelse return CudaError.KernelNotFound;

    var d_x = toDevicePtr(x);
    var d_w = toDevicePtr(weight);
    var heads_val: c_int = @intCast(n_heads);
    var hdim_val: c_int = @intCast(head_dim);
    var wlen_val: c_int = @intCast(weight_len);
    var eps_val: f32 = eps;

    var params = [_]?*anyopaque{
        @ptrCast(&d_x),
        @ptrCast(&d_w),
        @ptrCast(&heads_val),
        @ptrCast(&hdim_val),
        @ptrCast(&wlen_val),
        @ptrCast(&eps_val),
    };

    const res = s.api.?.cuLaunchKernel(
        fn_ptr,
        @intCast(n_heads), 1, 1,
        128, 1, 1,
        0,
        null,
        &params,
        null,
    );
    if (res != 0) return CudaError.CudaCallFailed;
}

pub fn rope(q: anytype, k: anytype, pos: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, rope_theta: f32) !void {
    const s = try ensureDriver();
    const fn_ptr = s.fn_rope orelse return CudaError.KernelNotFound;

    var d_q = toDevicePtr(q);
    var d_k = toDevicePtr(k);
    var pos_val: c_int = @intCast(pos);
    var nh_val: c_int = @intCast(n_heads);
    var nkv_val: c_int = @intCast(n_kv_heads);
    var hdim_val: c_int = @intCast(head_dim);
    var theta_val: f32 = rope_theta;

    var params = [_]?*anyopaque{
        @ptrCast(&d_q),
        @ptrCast(&d_k),
        @ptrCast(&pos_val),
        @ptrCast(&nh_val),
        @ptrCast(&nkv_val),
        @ptrCast(&hdim_val),
        @ptrCast(&theta_val),
    };

    const max_heads = @max(n_heads, n_kv_heads);
    const half_dim = @max(head_dim / 2, 1);

    const res = s.api.?.cuLaunchKernel(
        fn_ptr,
        @intCast(max_heads), 1, 1,
        @intCast(half_dim), 1, 1,
        0,
        null,
        &params,
        null,
    );
    if (res != 0) return CudaError.CudaCallFailed;
}

pub fn kvCacheUpdate(key_cache: anytype, val_cache: anytype, k: anytype, v: anytype, layer: usize, pos: usize, max_seq_len: usize, kv_dim: usize) !void {
    const s = try ensureDriver();
    const fn_ptr = s.fn_kv_cache_update orelse return CudaError.KernelNotFound;

    var d_kc = toDevicePtr(key_cache);
    var d_vc = toDevicePtr(val_cache);
    var d_k = toDevicePtr(k);
    var d_v = toDevicePtr(v);
    var layer_val: c_int = @intCast(layer);
    var pos_val: c_int = @intCast(pos);
    var max_seq_val: c_int = @intCast(max_seq_len);
    var kv_dim_val: c_int = @intCast(kv_dim);

    var params = [_]?*anyopaque{
        @ptrCast(&d_kc),
        @ptrCast(&d_vc),
        @ptrCast(&d_k),
        @ptrCast(&d_v),
        @ptrCast(&layer_val),
        @ptrCast(&pos_val),
        @ptrCast(&max_seq_val),
        @ptrCast(&kv_dim_val),
    };

    const res = s.api.?.cuLaunchKernel(
        fn_ptr,
        1, 1, 1,
        256, 1, 1,
        0,
        null,
        &params,
        null,
    );
    if (res != 0) return CudaError.CudaCallFailed;
}

pub fn gqaAttention(q: anytype, key_cache: anytype, val_cache: anytype, out: anytype, layer: usize, pos: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize, kv_dim: usize) !void {
    const s = try ensureDriver();
    const fn_ptr = s.fn_gqa_attention orelse return CudaError.KernelNotFound;

    var d_q = toDevicePtr(q);
    var d_kc = toDevicePtr(key_cache);
    var d_vc = toDevicePtr(val_cache);
    var d_out = toDevicePtr(out);
    var layer_val: c_int = @intCast(layer);
    var pos_val: c_int = @intCast(pos);
    var nh_val: c_int = @intCast(n_heads);
    var nkv_val: c_int = @intCast(n_kv_heads);
    var hdim_val: c_int = @intCast(head_dim);
    var max_seq_val: c_int = @intCast(max_seq_len);
    var kv_dim_val: c_int = @intCast(kv_dim);

    var params = [_]?*anyopaque{
        @ptrCast(&d_q),
        @ptrCast(&d_kc),
        @ptrCast(&d_vc),
        @ptrCast(&d_out),
        @ptrCast(&layer_val),
        @ptrCast(&pos_val),
        @ptrCast(&nh_val),
        @ptrCast(&nkv_val),
        @ptrCast(&hdim_val),
        @ptrCast(&max_seq_val),
        @ptrCast(&kv_dim_val),
    };

    const max_t = @min(pos + 1, max_seq_len);
    const shmem_bytes: c_uint = @intCast(max_t * @sizeOf(f32));

    const res = s.api.?.cuLaunchKernel(
        fn_ptr,
        @intCast(n_heads), 1, 1,
        128, 1, 1,
        shmem_bytes,
        null,
        &params,
        null,
    );
    if (res != 0) return CudaError.CudaCallFailed;
}

pub fn swiglu(gate: anytype, up: anytype, hidden_dim: usize) !void {
    const s = try ensureDriver();
    const fn_ptr = s.fn_swiglu orelse return CudaError.KernelNotFound;

    var d_gate = toDevicePtr(gate);
    var d_up = toDevicePtr(up);
    var hdim_val: c_int = @intCast(hidden_dim);

    var params = [_]?*anyopaque{
        @ptrCast(&d_gate),
        @ptrCast(&d_up),
        @ptrCast(&hdim_val),
    };

    const threads: c_uint = 256;
    const blocks: c_uint = @intCast((hidden_dim + threads - 1) / threads);

    const res = s.api.?.cuLaunchKernel(
        fn_ptr,
        blocks, 1, 1,
        threads, 1, 1,
        0,
        null,
        &params,
        null,
    );
    if (res != 0) return CudaError.CudaCallFailed;
}

pub fn addResidual(x: anytype, res: anytype, dim: usize) !void {
    const s = try ensureDriver();
    const fn_ptr = s.fn_add_residual orelse return CudaError.KernelNotFound;

    var d_x = toDevicePtr(x);
    var d_res = toDevicePtr(res);
    var dim_val: c_int = @intCast(dim);

    var params = [_]?*anyopaque{
        @ptrCast(&d_x),
        @ptrCast(&d_res),
        @ptrCast(&dim_val),
    };

    const threads: c_uint = 256;
    const blocks: c_uint = @intCast((dim + threads - 1) / threads);

    const r = s.api.?.cuLaunchKernel(
        fn_ptr,
        blocks, 1, 1,
        threads, 1, 1,
        0,
        null,
        &params,
        null,
    );
    if (r != 0) return CudaError.CudaCallFailed;
}

pub fn fusedQkNormRope(q: anytype, k: anytype, wq: anytype, wk: anytype, pos: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, w_len_q: usize, w_len_k: usize, eps: f32, rope_theta: f32) !void {
    if (toDevicePtr(wq) != 0) {
        try headRmsNorm(q, wq, n_heads, head_dim, w_len_q, eps);
    }
    if (toDevicePtr(wk) != 0) {
        try headRmsNorm(k, wk, n_kv_heads, head_dim, w_len_k, eps);
    }
    try rope(q, k, pos, n_heads, n_kv_heads, head_dim, rope_theta);
}

pub fn fusedSwiGluResidual(gate: anytype, up: anytype, res: anytype, hidden_dim: usize, dim: usize) !void {
    try swiglu(gate, up, hidden_dim);
    if (toDevicePtr(res) != 0 and dim > 0) {
        try addResidual(gate, res, dim);
    }
}
