//! The instruction set levels the hot kernels are built for.
//!
//! The build compiles the kernel module once per level that exists for the target architecture,
//! and the engine picks the best level the processor supports when it starts. A release binary
//! therefore runs on every machine of its architecture and is fast on each. Levels are listed
//! best first. The last entry of an architecture needs nothing special from the CPU.

const std = @import("std");

pub const Variant = struct {
    name: []const u8,
    arch: std.Target.Cpu.Arch,
    /// CPU model the level is compiled for, as `zig -mcpu` names it.
    model: []const u8,
    /// Features added on top of the model.
    add: []const []const u8 = &.{},
    /// Features the processor must have for the level to be usable.
    require: []const []const u8 = &.{},
};

pub const all = [_]Variant{
    .{
        .name = "avx512",
        .arch = .x86_64,
        .model = "x86_64_v4",
        .add = &.{"avx512vnni"},
        .require = &.{ "avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512vnni" },
    },
    .{
        .name = "avx2vnni",
        .arch = .x86_64,
        .model = "x86_64_v3",
        .add = &.{"avxvnni"},
        .require = &.{ "avx2", "fma", "f16c", "avxvnni" },
    },
    .{ .name = "avx2", .arch = .x86_64, .model = "x86_64_v3", .require = &.{ "avx2", "fma", "f16c" } },
    .{ .name = "generic", .arch = .x86_64, .model = "x86_64" },

    .{ .name = "dotprod", .arch = .aarch64, .model = "generic", .add = &.{"dotprod"}, .require = &.{"dotprod"} },
    .{ .name = "neon", .arch = .aarch64, .model = "generic" },

    // Other architectures get one portable level.
    .{ .name = "generic", .arch = .riscv64, .model = "baseline" },
    .{ .name = "generic", .arch = .x86, .model = "baseline" },
    .{ .name = "generic", .arch = .arm, .model = "baseline" },
    .{ .name = "generic", .arch = .powerpc64le, .model = "baseline" },
    .{ .name = "generic", .arch = .loongarch64, .model = "baseline" },
    .{ .name = "generic", .arch = .wasm32, .model = "baseline" },
};

/// Levels for one architecture, best first. Comptime so the engine can switch over them.
pub fn forArch(comptime arch: std.Target.Cpu.Arch) []const Variant {
    comptime {
        var n: usize = 0;
        for (all) |v| {
            if (v.arch == arch) n += 1;
        }
        var out: [n]Variant = undefined;
        var i: usize = 0;
        for (all) |v| {
            if (v.arch == arch) {
                out[i] = v;
                i += 1;
            }
        }
        const final = out;
        return &final;
    }
}

/// Symbol under which a level exports its function table.
pub fn symbol(comptime name: []const u8) []const u8 {
    return "hk_kernels_" ++ name;
}
