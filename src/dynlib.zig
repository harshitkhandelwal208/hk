//! Opening a shared library at run time and looking up symbols in it: `dlopen` on POSIX, `LoadLibrary` on Windows.
//! The GPU runtimes (CUDA, HIP) are never linked; HK finds them when it starts, so one binary runs with or without them.

const builtin = @import("builtin");

pub const Loader = struct {
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
