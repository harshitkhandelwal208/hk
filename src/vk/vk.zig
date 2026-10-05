//! A small Vulkan compute runtime: find a GPU, allocate buffers, build compute pipelines from
//! SPIR-V, record dispatches and run them. Nothing here knows about neural networks.
//!
//! The Vulkan loader is opened at run time (libvulkan.so.1, vulkan-1.dll, libvulkan.dylib or
//! MoltenVK), so hk has no build or run time dependency on a Vulkan SDK and starts normally on a
//! machine without a GPU or driver; `Context.init` then fails with `error.NoVulkan` and the CPU
//! engine carries on.

const std = @import("std");
const builtin = @import("builtin");
pub const c = @import("c.zig");

pub const Error = error{
    NoVulkan,
    NoDevice,
    NoComputeQueue,
    MissingFeature,
    OutOfDeviceMemory,
    OutOfHostMemory,
    VulkanFailed,
};

fn check(r: c.VkResult) Error!void {
    if (r == c.VK_SUCCESS) return;
    return switch (r) {
        c.VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        c.VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        else => error.VulkanFailed,
    };
}

pub fn makeApiVersion(variant: u32, major: u32, minor: u32, patch: u32) u32 {
    return (variant << 29) | (major << 22) | (minor << 12) | patch;
}

/// std.DynLib has no Windows implementation in Zig 0.16, so Windows goes through kernel32 directly.
const DynLib = if (builtin.os.tag == .windows) struct {
    const win = struct {
        extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.winapi) ?std.os.windows.HMODULE;
        extern "kernel32" fn GetProcAddress(module: std.os.windows.HMODULE, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
        extern "kernel32" fn FreeLibrary(module: std.os.windows.HMODULE) callconv(.winapi) i32;
    };
    handle: std.os.windows.HMODULE,

    fn open(name: []const u8) error{NotFound}!@This() {
        var buf: [256:0]u8 = undefined;
        if (name.len >= buf.len) return error.NotFound;
        @memcpy(buf[0..name.len], name);
        buf[name.len] = 0;
        return .{ .handle = win.LoadLibraryA(&buf) orelse return error.NotFound };
    }

    fn lookup(self: *@This(), comptime T: type, name: [:0]const u8) ?T {
        const p = win.GetProcAddress(self.handle, name.ptr) orelse return null;
        return @ptrCast(p);
    }

    fn close(self: *@This()) void {
        _ = win.FreeLibrary(self.handle);
    }
} else std.DynLib;

const loader_names = switch (builtin.os.tag) {
    .windows => &[_][]const u8{"vulkan-1.dll"},
    .macos, .ios => &[_][]const u8{ "libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib" },
    else => &[_][]const u8{ "libvulkan.so.1", "libvulkan.so" },
};

/// Device level functions, loaded once.
const Fns = struct {
    destroyDevice: std.meta.Child(c.PFN_vkDestroyDevice),
    getDeviceQueue: std.meta.Child(c.PFN_vkGetDeviceQueue),
    createBuffer: std.meta.Child(c.PFN_vkCreateBuffer),
    destroyBuffer: std.meta.Child(c.PFN_vkDestroyBuffer),
    getBufferMemoryRequirements: std.meta.Child(c.PFN_vkGetBufferMemoryRequirements),
    allocateMemory: std.meta.Child(c.PFN_vkAllocateMemory),
    freeMemory: std.meta.Child(c.PFN_vkFreeMemory),
    bindBufferMemory: std.meta.Child(c.PFN_vkBindBufferMemory),
    mapMemory: std.meta.Child(c.PFN_vkMapMemory),
    unmapMemory: std.meta.Child(c.PFN_vkUnmapMemory),
    flushMappedMemoryRanges: std.meta.Child(c.PFN_vkFlushMappedMemoryRanges),
    invalidateMappedMemoryRanges: std.meta.Child(c.PFN_vkInvalidateMappedMemoryRanges),
    createShaderModule: std.meta.Child(c.PFN_vkCreateShaderModule),
    destroyShaderModule: std.meta.Child(c.PFN_vkDestroyShaderModule),
    createDescriptorSetLayout: std.meta.Child(c.PFN_vkCreateDescriptorSetLayout),
    destroyDescriptorSetLayout: std.meta.Child(c.PFN_vkDestroyDescriptorSetLayout),
    createPipelineLayout: std.meta.Child(c.PFN_vkCreatePipelineLayout),
    destroyPipelineLayout: std.meta.Child(c.PFN_vkDestroyPipelineLayout),
    createComputePipelines: std.meta.Child(c.PFN_vkCreateComputePipelines),
    destroyPipeline: std.meta.Child(c.PFN_vkDestroyPipeline),
    createDescriptorPool: std.meta.Child(c.PFN_vkCreateDescriptorPool),
    destroyDescriptorPool: std.meta.Child(c.PFN_vkDestroyDescriptorPool),
    allocateDescriptorSets: std.meta.Child(c.PFN_vkAllocateDescriptorSets),
    updateDescriptorSets: std.meta.Child(c.PFN_vkUpdateDescriptorSets),
    createCommandPool: std.meta.Child(c.PFN_vkCreateCommandPool),
    destroyCommandPool: std.meta.Child(c.PFN_vkDestroyCommandPool),
    allocateCommandBuffers: std.meta.Child(c.PFN_vkAllocateCommandBuffers),
    resetCommandBuffer: std.meta.Child(c.PFN_vkResetCommandBuffer),
    beginCommandBuffer: std.meta.Child(c.PFN_vkBeginCommandBuffer),
    endCommandBuffer: std.meta.Child(c.PFN_vkEndCommandBuffer),
    cmdBindPipeline: std.meta.Child(c.PFN_vkCmdBindPipeline),
    cmdBindDescriptorSets: std.meta.Child(c.PFN_vkCmdBindDescriptorSets),
    cmdPushConstants: std.meta.Child(c.PFN_vkCmdPushConstants),
    cmdDispatch: std.meta.Child(c.PFN_vkCmdDispatch),
    cmdPipelineBarrier: std.meta.Child(c.PFN_vkCmdPipelineBarrier),
    cmdCopyBuffer: std.meta.Child(c.PFN_vkCmdCopyBuffer),
    queueSubmit: std.meta.Child(c.PFN_vkQueueSubmit),
    queueWaitIdle: std.meta.Child(c.PFN_vkQueueWaitIdle),
    createFence: std.meta.Child(c.PFN_vkCreateFence),
    destroyFence: std.meta.Child(c.PFN_vkDestroyFence),
    waitForFences: std.meta.Child(c.PFN_vkWaitForFences),
    resetFences: std.meta.Child(c.PFN_vkResetFences),
    deviceWaitIdle: std.meta.Child(c.PFN_vkDeviceWaitIdle),
    createQueryPool: std.meta.Child(c.PFN_vkCreateQueryPool),
    destroyQueryPool: std.meta.Child(c.PFN_vkDestroyQueryPool),
    cmdWriteTimestamp: std.meta.Child(c.PFN_vkCmdWriteTimestamp),
    cmdResetQueryPool: std.meta.Child(c.PFN_vkCmdResetQueryPool),
    getQueryPoolResults: std.meta.Child(c.PFN_vkGetQueryPoolResults),
};

/// What the chosen device can do, for shaders and for deciding how much to offload.
pub const Caps = struct {
    name: [256]u8 = undefined,
    name_len: usize = 0,
    is_discrete: bool = false,
    is_cpu: bool = false,
    vendor_id: u32 = 0,
    subgroup_size: u32 = 32,
    max_workgroup_size: u32 = 128,
    max_shared_bytes: u32 = 16384,
    max_alloc: u64 = 0,
    /// Nanoseconds per timestamp tick.
    timestamp_period: f32 = 1.0,
    /// Memory that lives on the device (VRAM, or the shared pool on integrated parts), in bytes.
    device_local_bytes: u64 = 0,

    pub fn deviceName(self: *const Caps) []const u8 {
        return self.name[0..self.name_len];
    }
};

pub const Buffer = struct {
    buf: c.VkBuffer = null,
    mem: c.VkDeviceMemory = null,
    size: u64 = 0,
    /// Set for host visible buffers.
    mapped: ?[*]u8 = null,
};

pub const MemKind = enum {
    /// Fast memory on the device; the CPU cannot touch it directly.
    device,
    /// Memory the CPU writes and the device reads (staging, small parameter blocks).
    upload,
    /// Memory the device writes and the CPU reads (logits).
    readback,
};

pub const Binding = struct { buf: Buffer, offset: u64 = 0, size: u64 = c.VK_WHOLE_SIZE };

pub const Pipeline = struct {
    pipe: c.VkPipeline = null,
    layout: c.VkPipelineLayout = null,
    dsl: c.VkDescriptorSetLayout = null,
    n_bindings: u32 = 0,
    push_bytes: u32 = 0,
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    lib: DynLib,
    instance: c.VkInstance,
    phys: c.VkPhysicalDevice,
    device: c.VkDevice,
    queue: c.VkQueue,
    queue_family: u32,
    f: Fns,
    mem_props: c.VkPhysicalDeviceMemoryProperties,
    caps: Caps,
    cmd_pool: c.VkCommandPool,
    desc_pool: c.VkDescriptorPool,
    fence: c.VkFence,
    xfer_cb: c.VkCommandBuffer,
    /// Host visible staging memory for uploads, created on first use.
    stage: ?Buffer = null,
    destroyInstance: std.meta.Child(c.PFN_vkDestroyInstance),
    getMemoryBudget: ?std.meta.Child(c.PFN_vkGetPhysicalDeviceMemoryProperties2),

    /// Opens the loader, picks a device and creates the compute queue. `pick` selects the n-th
    /// device when several are present (HK_VK_DEVICE in the engine); otherwise a discrete GPU
    /// wins over an integrated one and software rasterizers are skipped.
    pub fn init(allocator: std.mem.Allocator, pick: ?usize) !Context {
        var lib: DynLib = undefined;
        var opened = false;
        for (loader_names) |name| {
            if (DynLib.open(name)) |l| {
                lib = l;
                opened = true;
                break;
            } else |_| {}
        }
        if (!opened) return error.NoVulkan;
        errdefer lib.close();

        const get_inst = lib.lookup(std.meta.Child(c.PFN_vkGetInstanceProcAddr), "vkGetInstanceProcAddr") orelse return error.NoVulkan;
        const create_instance: std.meta.Child(c.PFN_vkCreateInstance) = @ptrCast(get_inst(null, "vkCreateInstance") orelse return error.NoVulkan);

        var app = std.mem.zeroes(c.VkApplicationInfo);
        app.sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "hk";
        app.apiVersion = makeApiVersion(0, 1, 2, 0);
        var ici = std.mem.zeroes(c.VkInstanceCreateInfo);
        ici.sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        var instance: c.VkInstance = null;
        check(create_instance(&ici, null, &instance)) catch return error.NoVulkan;
        errdefer {
            const d: std.meta.Child(c.PFN_vkDestroyInstance) = @ptrCast(get_inst(instance, "vkDestroyInstance").?);
            d(instance, null);
        }

        const enumerate: std.meta.Child(c.PFN_vkEnumeratePhysicalDevices) = @ptrCast(get_inst(instance, "vkEnumeratePhysicalDevices") orelse return error.NoVulkan);
        const get_props: std.meta.Child(c.PFN_vkGetPhysicalDeviceProperties) = @ptrCast(get_inst(instance, "vkGetPhysicalDeviceProperties").?);
        const get_props2: std.meta.Child(c.PFN_vkGetPhysicalDeviceProperties2) = @ptrCast(get_inst(instance, "vkGetPhysicalDeviceProperties2").?);
        const get_feat2: std.meta.Child(c.PFN_vkGetPhysicalDeviceFeatures2) = @ptrCast(get_inst(instance, "vkGetPhysicalDeviceFeatures2").?);
        const get_mem: std.meta.Child(c.PFN_vkGetPhysicalDeviceMemoryProperties) = @ptrCast(get_inst(instance, "vkGetPhysicalDeviceMemoryProperties").?);
        const get_qfp: std.meta.Child(c.PFN_vkGetPhysicalDeviceQueueFamilyProperties) = @ptrCast(get_inst(instance, "vkGetPhysicalDeviceQueueFamilyProperties").?);
        const enum_ext: std.meta.Child(c.PFN_vkEnumerateDeviceExtensionProperties) = @ptrCast(get_inst(instance, "vkEnumerateDeviceExtensionProperties").?);
        const create_device: std.meta.Child(c.PFN_vkCreateDevice) = @ptrCast(get_inst(instance, "vkCreateDevice").?);
        const get_dev_proc: std.meta.Child(c.PFN_vkGetDeviceProcAddr) = @ptrCast(get_inst(instance, "vkGetDeviceProcAddr").?);
        const destroy_instance: std.meta.Child(c.PFN_vkDestroyInstance) = @ptrCast(get_inst(instance, "vkDestroyInstance").?);

        var n_dev: u32 = 0;
        try check(enumerate(instance, &n_dev, null));
        if (n_dev == 0) return error.NoDevice;
        const devs = try allocator.alloc(c.VkPhysicalDevice, n_dev);
        defer allocator.free(devs);
        try check(enumerate(instance, &n_dev, devs.ptr));

        // Rank devices: discrete first, then integrated, then anything that is not a CPU.
        var best: ?usize = null;
        var best_score: i32 = -1;
        for (devs[0..n_dev], 0..) |d, i| {
            var p: c.VkPhysicalDeviceProperties = undefined;
            get_props(d, &p);
            if (pick) |want| {
                if (want == i) best = i;
                continue;
            }
            const score: i32 = switch (p.deviceType) {
                c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => 3,
                c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => 2,
                c.VK_PHYSICAL_DEVICE_TYPE_CPU => -1,
                else => 1,
            };
            if (score > best_score) {
                best_score = score;
                best = i;
            }
        }
        if (best == null or (pick == null and best_score < 0)) return error.NoDevice;
        const phys = devs[best.?];

        // The shaders need 8 and 16 bit storage access and subgroup arithmetic.
        var f12 = std.mem.zeroes(c.VkPhysicalDeviceVulkan12Features);
        f12.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        var f11 = std.mem.zeroes(c.VkPhysicalDeviceVulkan11Features);
        f11.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        f11.pNext = &f12;
        var feats = std.mem.zeroes(c.VkPhysicalDeviceFeatures2);
        feats.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        feats.pNext = &f11;
        get_feat2(phys, &feats);
        if (f11.storageBuffer16BitAccess == 0 or f12.storageBuffer8BitAccess == 0 or f12.shaderInt8 == 0 or f12.shaderFloat16 == 0) {
            return error.MissingFeature;
        }

        var sg = std.mem.zeroes(c.VkPhysicalDeviceVulkan11Properties);
        sg.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
        var props2 = std.mem.zeroes(c.VkPhysicalDeviceProperties2);
        props2.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &sg;
        get_props2(phys, &props2);
        if ((sg.subgroupSupportedOperations & c.VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) == 0) return error.MissingFeature;

        var caps = Caps{};
        const nm = std.mem.sliceTo(&props2.properties.deviceName, 0);
        caps.name_len = @min(nm.len, caps.name.len);
        @memcpy(caps.name[0..caps.name_len], nm[0..caps.name_len]);
        caps.is_discrete = props2.properties.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        caps.is_cpu = props2.properties.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_CPU;
        caps.vendor_id = props2.properties.vendorID;
        caps.subgroup_size = sg.subgroupSize;
        caps.max_workgroup_size = props2.properties.limits.maxComputeWorkGroupSize[0];
        caps.max_shared_bytes = props2.properties.limits.maxComputeSharedMemorySize;
        caps.max_alloc = 1 << 30;
        caps.timestamp_period = props2.properties.limits.timestampPeriod;

        var mem_props: c.VkPhysicalDeviceMemoryProperties = undefined;
        get_mem(phys, &mem_props);
        for (0..mem_props.memoryHeapCount) |i| {
            if (mem_props.memoryHeaps[i].flags & c.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0) {
                caps.device_local_bytes = @max(caps.device_local_bytes, mem_props.memoryHeaps[i].size);
            }
        }

        // A compute capable queue family.
        var n_q: u32 = 0;
        get_qfp(phys, &n_q, null);
        const qprops = try allocator.alloc(c.VkQueueFamilyProperties, n_q);
        defer allocator.free(qprops);
        get_qfp(phys, &n_q, qprops.ptr);
        var family: ?u32 = null;
        for (qprops[0..n_q], 0..) |q, i| {
            if (q.queueFlags & c.VK_QUEUE_COMPUTE_BIT != 0) {
                family = @intCast(i);
                // Prefer a family without graphics, which is the dedicated compute queue.
                if (q.queueFlags & c.VK_QUEUE_GRAPHICS_BIT == 0) break;
            }
        }
        const qf = family orelse return error.NoComputeQueue;

        const prio: f32 = 1.0;
        var qci = std.mem.zeroes(c.VkDeviceQueueCreateInfo);
        qci.sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = qf;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        // Memory budget, when the driver reports it, tells how much VRAM is really free.
        var has_budget = false;
        var n_ext: u32 = 0;
        _ = enum_ext(phys, null, &n_ext, null);
        if (n_ext > 0) {
            const exts = try allocator.alloc(c.VkExtensionProperties, n_ext);
            defer allocator.free(exts);
            _ = enum_ext(phys, null, &n_ext, exts.ptr);
            for (exts[0..n_ext]) |e| {
                if (std.mem.eql(u8, std.mem.sliceTo(&e.extensionName, 0), "VK_EXT_memory_budget")) has_budget = true;
            }
        }
        const dev_exts = [_][*:0]const u8{"VK_EXT_memory_budget"};

        // Enable exactly the features that were checked.
        var en12 = std.mem.zeroes(c.VkPhysicalDeviceVulkan12Features);
        en12.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        en12.storageBuffer8BitAccess = 1;
        en12.shaderInt8 = 1;
        en12.shaderFloat16 = 1;
        var en11 = std.mem.zeroes(c.VkPhysicalDeviceVulkan11Features);
        en11.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        en11.storageBuffer16BitAccess = 1;
        en11.pNext = &en12;
        var en2 = std.mem.zeroes(c.VkPhysicalDeviceFeatures2);
        en2.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        en2.pNext = &en11;
        en2.features.shaderInt16 = feats.features.shaderInt16;
        en2.features.shaderInt64 = feats.features.shaderInt64;

        var dci = std.mem.zeroes(c.VkDeviceCreateInfo);
        dci.sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.pNext = &en2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        if (has_budget) {
            dci.enabledExtensionCount = 1;
            dci.ppEnabledExtensionNames = &dev_exts;
        }
        var device: c.VkDevice = null;
        try check(create_device(phys, &dci, null, &device));
        errdefer {
            const d: std.meta.Child(c.PFN_vkDestroyDevice) = @ptrCast(get_dev_proc(device, "vkDestroyDevice").?);
            d(device, null);
        }

        var fns: Fns = undefined;
        inline for (@typeInfo(Fns).@"struct".fields) |fld| {
            const name = "vk" ++ [_]u8{std.ascii.toUpper(fld.name[0])} ++ fld.name[1..];
            @field(fns, fld.name) = @ptrCast(get_dev_proc(device, name) orelse return error.NoVulkan);
        }

        var queue: c.VkQueue = null;
        fns.getDeviceQueue(device, qf, 0, &queue);

        var cpi = std.mem.zeroes(c.VkCommandPoolCreateInfo);
        cpi.sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi.queueFamilyIndex = qf;
        cpi.flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        var cmd_pool: c.VkCommandPool = null;
        try check(fns.createCommandPool(device, &cpi, null, &cmd_pool));

        // One pool for every descriptor set the engine will ever need: sets are created once per
        // distinct (kernel, buffers) pair when a model loads.
        const pool_sizes = [_]c.VkDescriptorPoolSize{.{ .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 << 17 }};
        var dpi = std.mem.zeroes(c.VkDescriptorPoolCreateInfo);
        dpi.sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = 1 << 14;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &pool_sizes;
        var desc_pool: c.VkDescriptorPool = null;
        try check(fns.createDescriptorPool(device, &dpi, null, &desc_pool));

        var fci = std.mem.zeroes(c.VkFenceCreateInfo);
        fci.sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        var fence: c.VkFence = null;
        try check(fns.createFence(device, &fci, null, &fence));

        var xfer: c.VkCommandBuffer = null;
        {
            var ai = std.mem.zeroes(c.VkCommandBufferAllocateInfo);
            ai.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = cmd_pool;
            ai.level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            try check(fns.allocateCommandBuffers(device, &ai, &xfer));
        }

        return .{
            .allocator = allocator,
            .lib = lib,
            .instance = instance,
            .phys = phys,
            .device = device,
            .queue = queue,
            .queue_family = qf,
            .f = fns,
            .mem_props = mem_props,
            .caps = caps,
            .cmd_pool = cmd_pool,
            .desc_pool = desc_pool,
            .fence = fence,
            .xfer_cb = xfer,
            .destroyInstance = destroy_instance,
            .getMemoryBudget = if (has_budget) @ptrCast(get_inst(instance, "vkGetPhysicalDeviceMemoryProperties2")) else null,
        };
    }

    pub fn deinit(self: *Context) void {
        _ = self.f.deviceWaitIdle(self.device);
        if (self.stage) |b| self.destroyBuffer(b);
        self.f.destroyFence(self.device, self.fence, null);
        self.f.destroyDescriptorPool(self.device, self.desc_pool, null);
        self.f.destroyCommandPool(self.device, self.cmd_pool, null);
        self.f.destroyDevice(self.device, null);
        self.destroyInstance(self.instance, null);
        self.lib.close();
    }

    /// Bytes of device memory the driver says are still available, or the heap size when it does
    /// not say.
    pub fn freeDeviceBytes(self: *Context) u64 {
        if (self.getMemoryBudget) |f| {
            var budget = std.mem.zeroes(c.VkPhysicalDeviceMemoryBudgetPropertiesEXT);
            budget.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
            var p2 = std.mem.zeroes(c.VkPhysicalDeviceMemoryProperties2);
            p2.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
            p2.pNext = &budget;
            f(self.phys, &p2);
            var best: u64 = 0;
            for (0..self.mem_props.memoryHeapCount) |i| {
                if (self.mem_props.memoryHeaps[i].flags & c.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0) {
                    const b = budget.heapBudget[i];
                    const u = budget.heapUsage[i];
                    best = @max(best, if (b > u) b - u else 0);
                }
            }
            if (best != 0) return best;
        }
        return self.caps.device_local_bytes;
    }

    fn findMemoryType(self: *const Context, type_bits: u32, want: u32, avoid: u32) ?u32 {
        for (0..self.mem_props.memoryTypeCount) |i| {
            const flags = self.mem_props.memoryTypes[i].propertyFlags;
            if (type_bits & (@as(u32, 1) << @intCast(i)) != 0 and flags & want == want and flags & avoid == 0) return @intCast(i);
        }
        return null;
    }

    pub fn createBuffer(self: *Context, size: u64, kind: MemKind) !Buffer {
        var bci = std.mem.zeroes(c.VkBufferCreateInfo);
        bci.sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = @max(size, 16);
        bci.usage = c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = c.VK_SHARING_MODE_EXCLUSIVE;
        var buf: c.VkBuffer = null;
        try check(self.f.createBuffer(self.device, &bci, null, &buf));
        errdefer self.f.destroyBuffer(self.device, buf, null);

        var req: c.VkMemoryRequirements = undefined;
        self.f.getBufferMemoryRequirements(self.device, buf, &req);

        const host = c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const type_index: u32 = switch (kind) {
            .device => self.findMemoryType(req.memoryTypeBits, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0) orelse
                self.findMemoryType(req.memoryTypeBits, 0, 0) orelse return error.VulkanFailed,
            // Upload memory should not be the small BAR window of a discrete card when avoidable.
            .upload => self.findMemoryType(req.memoryTypeBits, host, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) orelse
                self.findMemoryType(req.memoryTypeBits, host, 0) orelse return error.VulkanFailed,
            .readback => self.findMemoryType(req.memoryTypeBits, host | c.VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0) orelse
                self.findMemoryType(req.memoryTypeBits, host, 0) orelse return error.VulkanFailed,
        };
        var mai = std.mem.zeroes(c.VkMemoryAllocateInfo);
        mai.sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type_index;
        var mem: c.VkDeviceMemory = null;
        try check(self.f.allocateMemory(self.device, &mai, null, &mem));
        errdefer self.f.freeMemory(self.device, mem, null);
        try check(self.f.bindBufferMemory(self.device, buf, mem, 0));

        var out = Buffer{ .buf = buf, .mem = mem, .size = size };
        if (kind != .device) {
            var p: ?*anyopaque = null;
            try check(self.f.mapMemory(self.device, mem, 0, c.VK_WHOLE_SIZE, 0, &p));
            out.mapped = @ptrCast(p);
        }
        return out;
    }

    pub fn destroyBuffer(self: *Context, b: Buffer) void {
        if (b.mapped != null) self.f.unmapMemory(self.device, b.mem);
        self.f.destroyBuffer(self.device, b.buf, null);
        self.f.freeMemory(self.device, b.mem, null);
    }

    /// Creates a compute pipeline from SPIR-V with `n_bindings` storage buffers in set 0 and
    /// `push_bytes` of push constants.
    pub fn createPipeline(self: *Context, spirv: []const u8, n_bindings: u32, push_bytes: u32) !Pipeline {
        std.debug.assert(spirv.len % 4 == 0);
        var smci = std.mem.zeroes(c.VkShaderModuleCreateInfo);
        smci.sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spirv.len;
        // SPIR-V must be 4 byte aligned; embedded data may not be.
        const words = try self.allocator.alloc(u32, spirv.len / 4);
        defer self.allocator.free(words);
        @memcpy(std.mem.sliceAsBytes(words), spirv);
        smci.pCode = words.ptr;
        var module: c.VkShaderModule = null;
        try check(self.f.createShaderModule(self.device, &smci, null, &module));
        defer self.f.destroyShaderModule(self.device, module, null);

        var bindings: [16]c.VkDescriptorSetLayoutBinding = undefined;
        for (0..n_bindings) |i| bindings[i] = .{
            .binding = @intCast(i),
            .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = null,
        };
        var dslci = std.mem.zeroes(c.VkDescriptorSetLayoutCreateInfo);
        dslci.sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = n_bindings;
        dslci.pBindings = &bindings;
        var dsl: c.VkDescriptorSetLayout = null;
        try check(self.f.createDescriptorSetLayout(self.device, &dslci, null, &dsl));

        const pcr = c.VkPushConstantRange{ .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = push_bytes };
        var plci = std.mem.zeroes(c.VkPipelineLayoutCreateInfo);
        plci.sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        if (push_bytes != 0) {
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pcr;
        }
        var layout: c.VkPipelineLayout = null;
        try check(self.f.createPipelineLayout(self.device, &plci, null, &layout));

        var stage = std.mem.zeroes(c.VkPipelineShaderStageCreateInfo);
        stage.sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = c.VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";
        var cpci = std.mem.zeroes(c.VkComputePipelineCreateInfo);
        cpci.sType = c.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = layout;
        var pipe: c.VkPipeline = null;
        try check(self.f.createComputePipelines(self.device, null, 1, &cpci, null, &pipe));
        return .{ .pipe = pipe, .layout = layout, .dsl = dsl, .n_bindings = n_bindings, .push_bytes = push_bytes };
    }

    pub fn destroyPipeline(self: *Context, p: Pipeline) void {
        self.f.destroyPipeline(self.device, p.pipe, null);
        self.f.destroyPipelineLayout(self.device, p.layout, null);
        self.f.destroyDescriptorSetLayout(self.device, p.dsl, null);
    }

    /// A descriptor set binding `buffers` (in order) to the pipeline's storage buffer slots.
    pub fn makeSet(self: *Context, p: Pipeline, buffers: []const Binding) !c.VkDescriptorSet {
        std.debug.assert(buffers.len == p.n_bindings);
        var ai = std.mem.zeroes(c.VkDescriptorSetAllocateInfo);
        ai.sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = self.desc_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &p.dsl;
        var set: c.VkDescriptorSet = null;
        try check(self.f.allocateDescriptorSets(self.device, &ai, &set));
        var infos: [16]c.VkDescriptorBufferInfo = undefined;
        var writes: [16]c.VkWriteDescriptorSet = undefined;
        for (buffers, 0..) |b, i| {
            infos[i] = .{ .buffer = b.buf.buf, .offset = b.offset, .range = b.size };
            writes[i] = std.mem.zeroes(c.VkWriteDescriptorSet);
            writes[i].sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = @intCast(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        self.f.updateDescriptorSets(self.device, @intCast(buffers.len), &writes, 0, null);
        return set;
    }

    pub fn newCommandBuffer(self: *Context) !c.VkCommandBuffer {
        var ai = std.mem.zeroes(c.VkCommandBufferAllocateInfo);
        ai.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = self.cmd_pool;
        ai.level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        var cb: c.VkCommandBuffer = null;
        try check(self.f.allocateCommandBuffers(self.device, &ai, &cb));
        return cb;
    }

    pub fn begin(self: *Context, cb: c.VkCommandBuffer, one_time: bool) !void {
        var bi = std.mem.zeroes(c.VkCommandBufferBeginInfo);
        bi.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (one_time) bi.flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        try check(self.f.beginCommandBuffer(cb, &bi));
    }

    pub fn end(self: *Context, cb: c.VkCommandBuffer) !void {
        try check(self.f.endCommandBuffer(cb));
    }

    pub fn dispatch(self: *Context, cb: c.VkCommandBuffer, p: Pipeline, set: c.VkDescriptorSet, push: []const u8, x: u32, y: u32, z: u32) void {
        self.f.cmdBindPipeline(cb, c.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
        self.f.cmdBindDescriptorSets(cb, c.VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, null);
        if (push.len != 0) self.f.cmdPushConstants(cb, p.layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @intCast(push.len), push.ptr);
        self.f.cmdDispatch(cb, x, y, z);
    }

    /// Makes writes of the previous dispatches visible to the next ones.
    pub fn barrier(self: *Context, cb: c.VkCommandBuffer) void {
        var mb = std.mem.zeroes(c.VkMemoryBarrier);
        mb.sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = c.VK_ACCESS_SHADER_READ_BIT | c.VK_ACCESS_SHADER_WRITE_BIT;
        self.f.cmdPipelineBarrier(cb, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, null, 0, null);
    }

    pub fn copy(self: *Context, cb: c.VkCommandBuffer, src: Buffer, dst: Buffer, src_off: u64, dst_off: u64, size: u64) void {
        const region = c.VkBufferCopy{ .srcOffset = src_off, .dstOffset = dst_off, .size = size };
        self.f.cmdCopyBuffer(cb, src.buf, dst.buf, 1, &region);
    }

    pub fn createTimestampPool(self: *Context, count: u32) !c.VkQueryPool {
        var qi = std.mem.zeroes(c.VkQueryPoolCreateInfo);
        qi.sType = c.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qi.queryType = c.VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = count;
        var pool: c.VkQueryPool = null;
        try check(self.f.createQueryPool(self.device, &qi, null, &pool));
        return pool;
    }

    /// Runs a recorded command buffer and waits for it.
    pub fn submitAndWait(self: *Context, cb: c.VkCommandBuffer) !void {
        var si = std.mem.zeroes(c.VkSubmitInfo);
        si.sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        try check(self.f.resetFences(self.device, 1, &self.fence));
        try check(self.f.queueSubmit(self.queue, 1, &si, self.fence));
        try check(self.f.waitForFences(self.device, 1, &self.fence, 1, std.math.maxInt(u64)));
    }

    /// Copies `data` into a device buffer through a staging buffer, in pieces of at most 32 MiB.
    pub fn upload(self: *Context, dst: Buffer, offset: u64, data: []const u8) !void {
        const chunk: u64 = 32 << 20;
        if (self.stage == null) self.stage = try self.createBuffer(chunk, .upload);
        const stage = self.stage.?;
        const cb = self.xfer_cb;
        var done: u64 = 0;
        while (done < data.len) {
            const n = @min(chunk, data.len - done);
            @memcpy(stage.mapped.?[0..n], data[done..][0..n]);
            try self.begin(cb, true);
            self.copy(cb, stage, dst, 0, offset + done, n);
            try self.end(cb);
            try self.submitAndWait(cb);
            done += n;
        }
    }
};

test "a compute shader runs on the device, or there is no device" {
    var ctx = Context.init(std.testing.allocator, null) catch |e| switch (e) {
        error.NoVulkan, error.NoDevice, error.MissingFeature, error.NoComputeQueue => return error.SkipZigTest,
        else => return e,
    };
    defer ctx.deinit();
    const n: u32 = 1000;
    const a = try ctx.createBuffer(n * 4, .upload);
    defer ctx.destroyBuffer(a);
    const b = try ctx.createBuffer(n * 4, .upload);
    defer ctx.destroyBuffer(b);
    const out = try ctx.createBuffer(n * 4, .readback);
    defer ctx.destroyBuffer(out);
    const af: [*]f32 = @ptrCast(@alignCast(a.mapped.?));
    const bf: [*]f32 = @ptrCast(@alignCast(b.mapped.?));
    for (0..n) |i| {
        af[i] = @floatFromInt(i);
        bf[i] = 0.5;
    }
    const pipe = try ctx.createPipeline(@embedFile("spv/vadd.spv"), 3, 4);
    defer ctx.destroyPipeline(pipe);
    const set = try ctx.makeSet(pipe, &.{ .{ .buf = a }, .{ .buf = b }, .{ .buf = out } });
    const cb = ctx.xfer_cb;
    try ctx.begin(cb, true);
    ctx.dispatch(cb, pipe, set, std.mem.asBytes(&n), (n + 63) / 64, 1, 1);
    try ctx.end(cb);
    try ctx.submitAndWait(cb);
    const of: [*]const f32 = @ptrCast(@alignCast(out.mapped.?));
    for (0..n) |i| try std.testing.expectEqual(@as(f32, @floatFromInt(i)) + 0.5, of[i]);
}
