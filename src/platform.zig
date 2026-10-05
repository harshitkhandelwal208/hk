const std = @import("std");

pub const PAGE_SIZE: usize = if (@import("builtin").os.tag == .windows)
    4096
else if (@hasDecl(std.posix, "page_size_min"))
    std.posix.page_size_min
else
    4096;

pub const MmapRegion = struct {
    bytes: []align(PAGE_SIZE) u8,
    is_mmap: bool = false,
    file_handle: ?std.os.windows.HANDLE = null,
    mapping_handle: ?std.os.windows.HANDLE = null,

    pub fn deinit(self: *MmapRegion, allocator: std.mem.Allocator) void {
        if (@import("builtin").os.tag == .windows) {
            if (self.is_mmap) {
                const win = struct {
                    extern "kernel32" fn UnmapViewOfFile(lpBaseAddress: ?*const anyopaque) callconv(.winapi) i32;
                    extern "kernel32" fn CloseHandle(hObject: std.os.windows.HANDLE) callconv(.winapi) i32;
                };
                if (self.bytes.len > 0) {
                    _ = win.UnmapViewOfFile(self.bytes.ptr);
                }
                if (self.mapping_handle) |h| _ = win.CloseHandle(h);
                if (self.file_handle) |h| _ = win.CloseHandle(h);
                return;
            }
        } else {
            if (self.is_mmap) {
                if (self.bytes.len > 0) {
                    std.posix.munmap(@alignCast(self.bytes));
                }
                return;
            }
        }
        if (self.bytes.len > 0) {
            allocator.free(self.bytes);
        }
    }
};

pub const CpuVendor = enum(u8) {
    intel = 0,
    amd = 1,
    arm = 2,
    apple = 3,
    unknown = 4,
};

pub const HardwareCapabilities = struct {
    vendor: CpuVendor = .unknown,
    has_avx2: bool = false,
    has_avx512f: bool = false,
    has_avx512vnni: bool = false,
    has_avx_vnni: bool = false,
    has_amx: bool = false,
    has_arm_neon: bool = false,
    has_arm_sve: bool = false,
    is_apple_silicon: bool = false,
    has_rocm_ready: bool = false,
    has_npu_ready: bool = false,
    optimal_page_alignment: usize = 4096, // 4KB default (ROCm / Intel NPU / x86_64)
    dma_hugepage_alignment: usize = 65536, // 64KB Direct DMA / Windows page allocation granularity

    pub fn getSummary(self: HardwareCapabilities, buf_out: []u8) []const u8 {
        const vendor_name = switch (self.vendor) {
            .intel => "Intel",
            .amd => "AMD",
            .arm => "ARM",
            .apple => "Apple Silicon",
            .unknown => "Generic",
        };
        return std.fmt.bufPrint(buf_out, "Vendor: {s}, AVX2: {}, AVX-512: {}, VNNI: {}, NEON: {}, PageAlign: {}B", .{
            vendor_name,
            self.has_avx2,
            self.has_avx512f,
            self.has_avx512vnni or self.has_avx_vnni,
            self.has_arm_neon,
            self.optimal_page_alignment,
        }) catch "HardwareCapabilities";
    }
};

/// Detect runtime hardware capabilities of current host machine
pub fn detectHardwareCapabilities() HardwareCapabilities {
    var caps = HardwareCapabilities{};
    const arch = @import("builtin").cpu.arch;
    const os_tag = @import("builtin").os.tag;

    if (arch == .x86_64) {
        // x86_64 CPU feature detection via cpuid
        // On x86_64, safe cpuid
        const info0 = cpuidSafe(0, 0);
        if (info0.max_leaf >= 1) {
            // Check vendor string: GenuineIntel or AuthenticAMD
            // ebx, edx, ecx
            var vendor_str: [12]u8 = undefined;
            @memcpy(vendor_str[0..4], std.mem.asBytes(&info0.ebx));
            @memcpy(vendor_str[4..8], std.mem.asBytes(&info0.edx));
            @memcpy(vendor_str[8..12], std.mem.asBytes(&info0.ecx));

            if (std.mem.eql(u8, &vendor_str, "GenuineIntel")) {
                caps.vendor = .intel;
            } else if (std.mem.eql(u8, &vendor_str, "AuthenticAMD")) {
                caps.vendor = .amd;
            }

            const info1 = cpuidSafe(1, 0);
            _ = info1;

            if (info0.max_leaf >= 7) {
                const info7 = cpuidSafe(7, 0);
                caps.has_avx2 = (info7.ebx & (1 << 5)) != 0;
                caps.has_avx512f = (info7.ebx & (1 << 16)) != 0;
                caps.has_avx512vnni = (info7.ecx & (1 << 11)) != 0;
                caps.has_amx = (info7.edx & (1 << 24)) != 0; // AMX-TILE

                const info7_1 = cpuidSafe(7, 1);
                caps.has_avx_vnni = (info7_1.eax & (1 << 4)) != 0;
            }
        }
        caps.optimal_page_alignment = 4096;
    } else if (arch == .aarch64 or arch == .arm) {
        caps.vendor = .arm;
        caps.has_arm_neon = true; // Standard on ARMv8-A
        if (os_tag == .macos) {
            caps.vendor = .apple;
            caps.is_apple_silicon = true;
            // Apple Silicon Metal requires 16KB (16384 bytes) page alignment for zero-copy GPU buffers
            caps.optimal_page_alignment = 16384;
        } else {
            caps.optimal_page_alignment = 4096;
        }
    }

    return caps;
}

const CpuidResult = struct {
    max_leaf: u32 = 0,
    eax: u32 = 0,
    ebx: u32 = 0,
    ecx: u32 = 0,
    edx: u32 = 0,
};

fn cpuidSafe(leaf: u32, subleaf: u32) CpuidResult {
    const arch = @import("builtin").cpu.arch;
    if (arch != .x86_64) return .{};

    var eax: u32 = 0;
    var ebx: u32 = 0;
    var ecx: u32 = 0;
    var edx: u32 = 0;

    asm volatile (
        \\cpuid
        : [eax] "={eax}" (eax),
          [ebx] "={ebx}" (ebx),
          [ecx] "={ecx}" (ecx),
          [edx] "={edx}" (edx),
        : [leaf] "{eax}" (leaf),
          [subleaf] "{ecx}" (subleaf),
    );

    return .{
        .max_leaf = eax,
        .eax = eax,
        .ebx = ebx,
        .ecx = ecx,
        .edx = edx,
    };
}

/// Aligns an offset forward to the next multiple of alignment
pub fn alignForward(offset: usize, alignment: usize) usize {
    const rem = offset % alignment;
    if (rem == 0) return offset;
    return offset + (alignment - rem);
}

pub fn isAligned(offset: usize, alignment: usize) bool {
    return (offset % alignment) == 0;
}

var empty_aligned_page: [PAGE_SIZE]u8 align(PAGE_SIZE) = undefined;

/// Reads or maps an entire file into memory with page alignment
pub fn mapOrReadFile(path: []const u8, allocator: std.mem.Allocator) !MmapRegion {
    if (@import("builtin").os.tag == .windows) {
        const win = struct {
            extern "kernel32" fn CreateFileA(
                lpFileName: [*:0]const u8,
                dwDesiredAccess: u32,
                dwShareMode: u32,
                lpSecurityAttributes: ?*anyopaque,
                dwCreationDisposition: u32,
                dwFlagsAndAttributes: u32,
                hTemplateFile: ?*anyopaque,
            ) callconv(.winapi) std.os.windows.HANDLE;

            extern "kernel32" fn GetFileSizeEx(
                hFile: std.os.windows.HANDLE,
                lpFileSize: *i64,
            ) callconv(.winapi) i32;

            extern "kernel32" fn CreateFileMappingA(
                hFile: std.os.windows.HANDLE,
                lpFileMappingAttributes: ?*anyopaque,
                flProtect: u32,
                dwMaximumSizeHigh: u32,
                dwMaximumSizeLow: u32,
                lpName: ?[*:0]const u8,
            ) callconv(.winapi) ?std.os.windows.HANDLE;

            extern "kernel32" fn MapViewOfFile(
                hFileMappingObject: std.os.windows.HANDLE,
                dwDesiredAccess: u32,
                dwFileOffsetHigh: u32,
                dwFileOffsetLow: u32,
                dwNumberOfBytesToMap: usize,
            ) callconv(.winapi) ?*anyopaque;

            extern "kernel32" fn CloseHandle(hObject: std.os.windows.HANDLE) callconv(.winapi) i32;
        };

        const path_z = allocator.dupeSentinel(u8, path, 0) catch null;
        if (path_z) |pz| {
            defer allocator.free(pz);
            // GENERIC_READ = 0x80000000, FILE_SHARE_READ = 1, FILE_SHARE_WRITE = 2, OPEN_EXISTING = 3, FILE_ATTRIBUTE_NORMAL = 0x80
            const file_handle = win.CreateFileA(pz.ptr, 0x80000000, 1 | 2, null, 3, 0x80, null);
            if (file_handle != std.os.windows.INVALID_HANDLE_VALUE) {
                var file_size: i64 = 0;
                if (win.GetFileSizeEx(file_handle, &file_size) != 0 and file_size >= 0) {
                    const size: usize = @intCast(file_size);
                    if (size == 0) {
                        return MmapRegion{
                            .bytes = empty_aligned_page[0..0],
                            .is_mmap = true,
                            .file_handle = file_handle,
                            .mapping_handle = null,
                        };
                    }
                    // PAGE_READONLY = 0x02
                    const map_handle = win.CreateFileMappingA(file_handle, null, 0x02, 0, 0, null);
                    if (map_handle != null and map_handle.? != std.os.windows.INVALID_HANDLE_VALUE) {
                        // FILE_MAP_READ = 0x0004
                        const view_ptr = win.MapViewOfFile(map_handle.?, 0x0004, 0, 0, 0);
                        if (view_ptr != null) {
                            const raw_ptr: [*]align(PAGE_SIZE) u8 = @ptrCast(@alignCast(view_ptr.?));
                            return MmapRegion{
                                .bytes = raw_ptr[0..size],
                                .is_mmap = true,
                                .file_handle = file_handle,
                                .mapping_handle = map_handle,
                            };
                        }
                        _ = win.CloseHandle(map_handle.?);
                    }
                }
                _ = win.CloseHandle(file_handle);
            }
        }
    }

    const io = std.Options.debug_io;
    const cwd = std.Io.Dir.cwd();

    var file = try cwd.openFile(io, path, .{});
    defer file.close(io);

    const st = try file.stat(io);
    const size: usize = @intCast(st.size);
    if (size == 0) {
        return MmapRegion{
            .bytes = empty_aligned_page[0..0],
            .is_mmap = false,
        };
    }

    // POSIX zero-copy path: real mmap(2), matching the CreateFileMapping/MapViewOfFile
    // path already implemented above for Windows. std.posix.MAP/PROT are stubbed to
    // `void` on non-POSIX targets, so this must not be typechecked when building for
    // Windows even though that branch is unreachable there at runtime.
    if (@import("builtin").os.tag != .windows) {
        if (std.posix.mmap(
            null,
            size,
            .{ .READ = true },
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        )) |mapped| {
            return MmapRegion{
                .bytes = mapped,
                .is_mmap = true,
            };
        } else |_| {
            // Fall through to a buffered read (e.g. non-regular file, or mmap
            // unsupported/denied on this filesystem).
        }
    }

    const aligned_slice = try allocator.alignedAlloc(u8, .fromByteUnits(PAGE_SIZE), size);
    errdefer allocator.free(aligned_slice);

    var total_read: usize = 0;
    while (total_read < size) {
        var slices = [_][]u8{aligned_slice[total_read..]};
        const chunk_read = try file.readStreaming(io, &slices);
        if (chunk_read == 0) return error.UnexpectedEof;
        total_read += chunk_read;
    }

    return MmapRegion{
        .bytes = aligned_slice,
        .is_mmap = false,
    };
}

/// Writes bytes into an existing file at a specific offset without rewriting other bytes
pub fn writeBytesAtOffset(path: []const u8, data: []const u8, offset: u64, allocator: std.mem.Allocator) !void {
    if (data.len == 0) return;
    if (@import("builtin").os.tag == .windows) {
        const win = struct {
            extern "kernel32" fn CreateFileA(
                lpFileName: [*:0]const u8,
                dwDesiredAccess: u32,
                dwShareMode: u32,
                lpSecurityAttributes: ?*anyopaque,
                dwCreationDisposition: u32,
                dwFlagsAndAttributes: u32,
                hTemplateFile: ?*anyopaque,
            ) callconv(.winapi) std.os.windows.HANDLE;

            extern "kernel32" fn WriteFile(
                hFile: std.os.windows.HANDLE,
                lpBuffer: [*]const u8,
                nNumberOfBytesToWrite: u32,
                lpNumberOfBytesWritten: ?*u32,
                lpOverlapped: ?*anyopaque,
            ) callconv(.winapi) i32;

            extern "kernel32" fn SetFilePointer(
                hFile: std.os.windows.HANDLE,
                lDistanceToMove: i32,
                lpDistanceToMoveHigh: ?*i32,
                dwMoveMethod: u32,
            ) callconv(.winapi) u32;

            extern "kernel32" fn CloseHandle(hObject: std.os.windows.HANDLE) callconv(.winapi) i32;
        };

        const path_z = try allocator.dupeSentinel(u8, path, 0);
        defer allocator.free(path_z);

        const handle = win.CreateFileA(path_z.ptr, 0x40000000 | 0x80000000, 1 | 2, null, 3, 0x80, null);
        if (handle == std.os.windows.INVALID_HANDLE_VALUE) return error.FileNotFound;
        defer _ = win.CloseHandle(handle);

        var high: i32 = @intCast((offset >> 32) & 0xFFFFFFFF);
        const low: i32 = @intCast(offset & 0xFFFFFFFF);
        _ = win.SetFilePointer(handle, low, &high, 0);

        var total_written: usize = 0;
        while (total_written < data.len) {
            var written: u32 = 0;
            const chunk_len: u32 = @intCast(@min(data.len - total_written, 1024 * 1024 * 64));
            const ok = win.WriteFile(handle, data[total_written..].ptr, chunk_len, &written, null);
            if (ok == 0) return error.WriteFailed;
            total_written += written;
        }
    } else {
        const io = std.Options.debug_io;
        const cwd = std.Io.Dir.cwd();
        var file = try cwd.openFile(io, path, .{ .mode = .read_write });
        defer file.close(io);
        try file.writePositionalAll(io, data, offset);
    }
}

/// Truncates a file to the specified size in-place (O(1) filesystem metadata update)
pub fn truncateFile(path: []const u8, new_size: u64, allocator: std.mem.Allocator) !void {
    if (@import("builtin").os.tag == .windows) {
        const win = struct {
            extern "kernel32" fn CreateFileA(
                lpFileName: [*:0]const u8,
                dwDesiredAccess: u32,
                dwShareMode: u32,
                lpSecurityAttributes: ?*anyopaque,
                dwCreationDisposition: u32,
                dwFlagsAndAttributes: u32,
                hTemplateFile: ?*anyopaque,
            ) callconv(.winapi) std.os.windows.HANDLE;

            extern "kernel32" fn SetFilePointer(
                hFile: std.os.windows.HANDLE,
                lDistanceToMove: i32,
                lpDistanceToMoveHigh: ?*i32,
                dwMoveMethod: u32,
            ) callconv(.winapi) u32;

            extern "kernel32" fn SetEndOfFile(hFile: std.os.windows.HANDLE) callconv(.winapi) i32;
            extern "kernel32" fn CloseHandle(hObject: std.os.windows.HANDLE) callconv(.winapi) i32;
        };

        const path_z = try allocator.dupeSentinel(u8, path, 0);
        defer allocator.free(path_z);

        const handle = win.CreateFileA(path_z.ptr, 0x40000000 | 0x80000000, 1 | 2, null, 3, 0x80, null);
        if (handle == std.os.windows.INVALID_HANDLE_VALUE) return error.FileNotFound;
        defer _ = win.CloseHandle(handle);

        var high: i32 = @intCast((new_size >> 32) & 0xFFFFFFFF);
        const low: i32 = @intCast(new_size & 0xFFFFFFFF);
        _ = win.SetFilePointer(handle, low, &high, 0);
        if (win.SetEndOfFile(handle) == 0) return error.WriteFailed;
    } else {
        const io = std.Options.debug_io;
        const cwd = std.Io.Dir.cwd();
        var file = try cwd.openFile(io, path, .{ .mode = .read_write });
        defer file.close(io);
        const rc = std.posix.system.ftruncate(file.handle, @intCast(new_size));
        switch (std.posix.errno(rc)) {
            .SUCCESS => {},
            else => return error.WriteFailed,
        }
    }
}

