//! How many threads to run compute on: the number of fast physical cores, not logical CPUs.
//!
//! The kernels are limited by memory bandwidth and by synchronization, so a second hardware
//! thread on a core adds contention and little work, and slow efficiency cores in a hybrid
//! processor drag every barrier. Counting is best effort and always falls back to the number of
//! logical CPUs the process may use, never to zero.

const std = @import("std");
const builtin = @import("builtin");

/// Physical performance cores usable by this process, at least 1.
pub fn physicalCores(io: std.Io) usize {
    const logical = std.Thread.getCpuCount() catch 1;
    const n: usize = switch (builtin.os.tag) {
        .linux => linuxCores(io) orelse logical,
        .macos, .ios => macCores() orelse logical,
        .windows => windowsCores() orelse logical,
        else => logical,
    };
    return std.math.clamp(n, 1, @max(1, logical));
}

fn linuxCores(io: std.Io) ?usize {
    var seen: [256][64]u8 = undefined;
    var seen_len: [256]usize = undefined;
    var n_seen: usize = 0;
    var max_capacity: usize = 0;
    var caps: [1024]usize = @splat(0);

    const max_cpus = 1024;
    for (0..max_cpus) |cpu| {
        var buf: [96]u8 = undefined;
        const p = std.fmt.bufPrint(&buf, "/sys/devices/system/cpu/cpu{d}/cpu_capacity", .{cpu}) catch break;
        var small: [32]u8 = undefined;
        if (readSmall(io, p, &small)) |text| {
            const v = std.fmt.parseInt(usize, std.mem.trim(u8, text, " \n"), 10) catch 0;
            caps[cpu] = v;
            max_capacity = @max(max_capacity, v);
        }
    }

    var any = false;
    for (0..max_cpus) |cpu| {
        var buf: [96]u8 = undefined;
        const p = std.fmt.bufPrint(&buf, "/sys/devices/system/cpu/cpu{d}/topology/thread_siblings_list", .{cpu}) catch break;
        var small: [64]u8 = undefined;
        const text = readSmall(io, p, &small) orelse {
            // Offline or absent CPUs leave gaps; the first absent index past some CPUs ends the scan.
            if (any and cpu > 0 and !cpuDirExists(io, cpu)) break;
            continue;
        };
        any = true;
        // On big.LITTLE parts count only the fast cores.
        if (max_capacity != 0 and caps[cpu] * 10 < max_capacity * 8) continue;
        const key = std.mem.trim(u8, text, " \n");
        var dup = false;
        for (0..n_seen) |i| {
            if (std.mem.eql(u8, seen[i][0..seen_len[i]], key)) {
                dup = true;
                break;
            }
        }
        if (!dup and n_seen < seen.len and key.len <= seen[0].len) {
            @memcpy(seen[n_seen][0..key.len], key);
            seen_len[n_seen] = key.len;
            n_seen += 1;
        }
    }
    return if (n_seen > 0) n_seen else null;
}

fn cpuDirExists(io: std.Io, cpu: usize) bool {
    var buf: [64]u8 = undefined;
    const p = std.fmt.bufPrint(&buf, "/sys/devices/system/cpu/cpu{d}", .{cpu}) catch return false;
    var d = std.Io.Dir.cwd().openDir(io, p, .{}) catch return false;
    d.close(io);
    return true;
}

/// Reads a small text file whose size the OS does not report (sysfs, procfs).
fn readSmall(io: std.Io, path: []const u8, buf: []u8) ?[]const u8 {
    var f = std.Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer f.close(io);
    var n: usize = 0;
    while (n < buf.len) {
        const got = f.readStreaming(io, &.{buf[n..]}) catch break;
        if (got == 0) break;
        n += got;
    }
    return buf[0..n];
}

fn macCores() ?usize {
    if (builtin.os.tag != .macos and builtin.os.tag != .ios) return null;
    // Performance cores first (Apple Silicon), then all physical cores.
    inline for (.{ "hw.perflevel0.physicalcpu", "hw.physicalcpu" }) |name| {
        var v: c_int = 0;
        var len: usize = @sizeOf(c_int);
        if (std.c.sysctlbyname(name, &v, &len, null, 0) == 0 and v > 0) return @intCast(v);
    }
    return null;
}

fn windowsCores() ?usize {
    if (builtin.os.tag != .windows) return null;
    const win = std.os.windows;
    const Info = extern struct {
        mask: usize,
        relationship: u32,
        pad: [16]u8,
    };
    const k32 = struct {
        extern "kernel32" fn GetLogicalProcessorInformation(buf: ?*Info, len: *u32) callconv(.winapi) c_int;
    };
    _ = win;
    var len: u32 = 0;
    _ = k32.GetLogicalProcessorInformation(null, &len);
    if (len == 0) return null;
    var buf: [256]Info = undefined;
    if (len > @sizeOf(@TypeOf(buf))) return null;
    if (k32.GetLogicalProcessorInformation(&buf[0], &len) == 0) return null;
    var cores: usize = 0;
    for (buf[0 .. len / @sizeOf(Info)]) |i| {
        if (i.relationship == 0) cores += 1; // RelationProcessorCore
    }
    return if (cores > 0) cores else null;
}

test "core count is at least one and at most the logical count" {
    const n = physicalCores(std.testing.io);
    try std.testing.expect(n >= 1);
    try std.testing.expect(n <= (std.Thread.getCpuCount() catch 1));
}
