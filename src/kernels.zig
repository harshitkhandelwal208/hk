//! Picks the kernel table for the processor the program is running on.
//!
//! The kernels are compiled several times, once per instruction set level (kernels/variants.zig).
//! `get()` detects the processor once and returns the best level it supports, so one binary is
//! fast on every machine of its architecture. Setting HK_KERNELS=<level> forces a level, which
//! is how the slower paths are tested and measured on a machine that could run a faster one.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const api_mod = @import("kernels/api.zig");
const variants = @import("kernels/variants.zig");

pub const Api = api_mod.Api;
pub const RowsArgs = api_mod.RowsArgs;
pub const QkRopeArgs = api_mod.QkRopeArgs;

const levels = variants.forArch(builtin.cpu.arch);

/// Levels compiled into this build, named in the build option `kernel_levels` (comma separated).
fn compiledIn(comptime name: []const u8) bool {
    var it = std.mem.tokenizeScalar(u8, build_options.kernel_levels, ',');
    while (it.next()) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

fn table(comptime name: []const u8) *const Api {
    const f = @extern(*const fn () callconv(.c) *const Api, .{ .name = variants.symbol(name) });
    return f();
}

fn supports(comptime v: variants.Variant, features: std.Target.Cpu.Feature.Set) bool {
    const Feature = switch (builtin.cpu.arch) {
        .x86_64, .x86 => std.Target.x86.Feature,
        .aarch64 => std.Target.aarch64.Feature,
        else => return v.require.len == 0,
    };
    inline for (v.require) |req| {
        const f = @field(Feature, req);
        const ok = switch (builtin.cpu.arch) {
            .x86_64, .x86 => std.Target.x86.featureSetHas(features, f),
            .aarch64 => std.Target.aarch64.featureSetHas(features, f),
            else => false,
        };
        if (!ok) return false;
    }
    return true;
}

var chosen: ?*const Api = null;

/// The kernel table in use. Detection runs on the first call; call it from one thread first (the
/// model does, in `init`) so the choice is made before any worker runs.
pub fn get() *const Api {
    if (chosen) |c| return c;
    const c = select();
    chosen = c;
    return c;
}

fn forced() ?[]const u8 {
    if (builtin.os.tag == .windows or !builtin.link_libc) return null;
    const p = std.c.getenv("HK_KERNELS") orelse return null;
    return std.mem.sliceTo(p, 0);
}

fn select() *const Api {
    if (forced()) |want| {
        inline for (levels) |v| {
            if (comptime compiledIn(v.name)) {
                if (std.mem.eql(u8, want, v.name)) return table(v.name);
            }
        }
    }
    const target = std.zig.system.resolveTargetQuery(std.Options.debug_io, .{}) catch null;
    inline for (levels) |v| {
        if (comptime compiledIn(v.name)) {
            if (target) |t| {
                if (supports(v, t.cpu.features)) return table(v.name);
            } else if (comptime v.require.len == 0) {
                return table(v.name);
            }
        }
    }
    @panic("no kernel level compiled in for this processor; build with -Dkernels=generic or all");
}

/// Names of the levels compiled into this build, best first.
pub const compiled_levels: []const []const u8 = blk: {
    var out: []const []const u8 = &.{};
    for (levels) |v| {
        if (compiledIn(v.name)) out = out ++ [_][]const u8{v.name};
    }
    break :blk out;
};

/// The level for `name`, if it is compiled in and the processor can run it.
pub fn byName(name: []const u8) ?*const Api {
    const target = std.zig.system.resolveTargetQuery(std.Options.debug_io, .{}) catch return null;
    inline for (levels) |v| {
        if (comptime compiledIn(v.name)) {
            if (std.mem.eql(u8, name, v.name) and supports(v, target.cpu.features)) return table(v.name);
        }
    }
    return null;
}
