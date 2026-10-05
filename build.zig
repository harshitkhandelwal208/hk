const std = @import("std");
const variants = @import("src/kernels/variants.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const default_cuda = target.result.os.tag != .macos;
    const cuda_enabled = b.option(
        bool,
        "cuda",
        "Enable the CUDA GPU backend (dynamically loads CUDA driver API at runtime)",
    ) orelse default_cuda;

    // Instruction set levels of the hot kernels. Every level is compiled and the engine picks
    // the best one the CPU supports when it starts. Debug builds default to the portable level
    // only, which keeps test builds fast.
    const kernel_sel = b.option(
        []const u8,
        "kernels",
        "Kernel instruction set levels to build: all, fallback, or a comma separated list (default: all, fallback for Debug)",
    ) orelse if (optimize == .Debug) "fallback" else "all";

    const build_opts = b.addOptions();
    build_opts.addOption(bool, "cuda", cuda_enabled);
    const build_opts_mod = build_opts.createModule();

    // Core HK Module
    const hk_mod = b.addModule("hk", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_opts_mod },
        },
    });

    addKernelLevels(b, hk_mod, build_opts, target, optimize, kernel_sel);

    if (cuda_enabled) {
        if (target.result.os.tag != .windows) {
            hk_mod.linkSystemLibrary("dl", .{});
        }
    }

    // Shared Library (DLL / .so / .dylib) for C ABI bindings (Python ctypes, C#, etc.)
    // Reuses hk_mod directly (rather than recompiling src/root.zig into a second
    // module) so the CUDA object file / library links above apply here too.
    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "hk",
        .root_module = hk_mod,
    });
    b.installArtifact(lib);

    // Optional JNI glue for the Java binding: `zig build jni -Djdk=/usr/lib/jvm/java-21-openjdk`.
    if (b.option([]const u8, "jdk", "JDK directory (the one with include/jni.h) for the `jni` step")) |jdk| {
        const jni_mod = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
        jni_mod.addCSourceFile(.{ .file = b.path("bindings/java/jni/hk_jni.c"), .flags = &.{"-std=c11"} });
        jni_mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{jdk}) });
        jni_mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include/{s}", .{ jdk, switch (target.result.os.tag) {
            .windows => "win32",
            .macos => "darwin",
            else => "linux",
        } }) });
        const jni_lib = b.addLibrary(.{ .linkage = .dynamic, .name = "hkjni", .root_module = jni_mod });
        jni_lib.root_module.linkLibrary(lib);
        const jni_step = b.step("jni", "Build the JNI glue library for the Java binding (needs -Djdk=<dir>)");
        jni_step.dependOn(&b.addInstallArtifact(jni_lib, .{}).step);
    }

    // HK CLI Tool
    const cli_mod = b.createModule(.{
        .root_source_file = b.path("tools/hk_cli.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "hk", .module = hk_mod },
        },
    });
    const cli_exe = b.addExecutable(.{
        .name = "hk",
        .root_module = cli_mod,
    });
    b.installArtifact(cli_exe);

    // Developer probe: runs token ids through the engine and dumps logits for comparison.
    const probe_mod = b.createModule(.{
        .root_source_file = b.path("tools/hk_probe.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "hk", .module = hk_mod }},
    });
    b.installArtifact(b.addExecutable(.{ .name = "hk-probe", .root_module = probe_mod }));

    // Tests
    const tests_mod = b.createModule(.{
        .root_source_file = b.path("tests/roundtrip_tests.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "hk", .module = hk_mod },
        },
    });
    const tests = b.addTest(.{
        .name = "hk-tests",
        .root_module = tests_mod,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run HK library unit tests");
    test_step.dependOn(&run_tests.step);

    // Unit tests that live next to the code in src/.
    const unit_tests = b.addTest(.{ .name = "hk-unit-tests", .root_module = hk_mod });
    test_step.dependOn(&b.addRunArtifact(unit_tests).step);

    // Additional test suites, each its own binary so one failure cannot hide another.
    const extra_suites = [_]struct { file: []const u8, name: []const u8 }{
        .{ .file = "tests/test_quant.zig", .name = "hk-quant-tests" },
        .{ .file = "tests/test_engine.zig", .name = "hk-engine-tests" },
        .{ .file = "tests/test_gpu.zig", .name = "hk-gpu-tests" },
    };
    for (extra_suites) |suite| {
        const mod = b.createModule(.{
            .root_source_file = b.path(suite.file),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "hk", .module = hk_mod }},
        });
        const t = b.addTest(.{ .name = suite.name, .root_module = mod });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // End to end tests of the command line and the server, run against the installed executable.
    {
        const test_opts = b.addOptions();
        test_opts.addOption([]const u8, "hk_exe", b.getInstallPath(.bin, cli_exe.out_filename));
        const e2e_step = b.step("test-e2e", "Run the command line and server end to end tests (builds and installs hk first)");
        const suites = [_]struct { file: []const u8, name: []const u8 }{
            .{ .file = "tests/cli_tests.zig", .name = "hk-cli-tests" },
            .{ .file = "tests/server_tests.zig", .name = "hk-server-tests" },
            .{ .file = "tests/hub_tests.zig", .name = "hk-hub-tests" },
        };
        for (suites) |suite| {
            const mod = b.createModule(.{
                .root_source_file = b.path(suite.file),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "test_options", .module = test_opts.createModule() }},
            });
            const t = b.addTest(.{ .name = suite.name, .root_module = mod });
            const run = b.addRunArtifact(t);
            run.step.dependOn(b.getInstallStep());
            e2e_step.dependOn(&run.step);
        }
    }

    // The tiny test model, for tests written in other languages.
    const tiny_mod = b.createModule(.{
        .root_source_file = b.path("tests/tiny_model_tool.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(b.addExecutable(.{ .name = "hk-tiny-model", .root_module = tiny_mod }));

    // Side by side comparison with llama.cpp.
    const compare_mod = b.createModule(.{
        .root_source_file = b.path("tools/hk_compare.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(b.addExecutable(.{ .name = "hk-compare", .root_module = compare_mod }));

    // Single thread kernel throughput per format.
    const kbench_mod = b.createModule(.{
        .root_source_file = b.path("tests/bench_kernels.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "hk", .module = hk_mod }},
    });
    b.installArtifact(b.addExecutable(.{ .name = "hk-kernels", .root_module = kbench_mod }));

    // Benchmarks
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("tests/test_bench_compare.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "hk", .module = hk_mod },
        },
    });
    const bench_exe = b.addExecutable(.{
        .name = "hk-bench-compare",
        .root_module = bench_mod,
    });
    b.installArtifact(bench_exe);
    const run_bench = b.addRunArtifact(bench_exe);
    const bench_step = b.step("bench", "Run kernel optimization benchmarks");
    bench_step.dependOn(&run_bench.step);

    if (cuda_enabled) {
        const gpu_bench_mod = b.createModule(.{
            .root_source_file = b.path("tools/hk_gpu_bench.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "hk", .module = hk_mod },
            },
        });
        const gpu_bench_exe = b.addExecutable(.{
            .name = "hk-gpu-bench",
            .root_module = gpu_bench_mod,
        });
        b.installArtifact(gpu_bench_exe);

        const cuda_tests_mod = b.createModule(.{
            .root_source_file = b.path("tests/test_cuda.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "hk", .module = hk_mod },
            },
        });
        const cuda_tests = b.addTest(.{
            .name = "hk-cuda-tests",
            .root_module = cuda_tests_mod,
        });
        const run_cuda_tests = b.addRunArtifact(cuda_tests);
        const cuda_test_step = b.step("test-cuda", "Run CUDA GPU backend correctness tests");
        cuda_test_step.dependOn(&run_cuda_tests.step);
        test_step.dependOn(&run_cuda_tests.step);
    }
}

fn selected(sel: []const u8, name: []const u8, is_fallback: bool) bool {
    if (std.mem.eql(u8, sel, "all")) return true;
    if (is_fallback) return true;
    var it = std.mem.tokenizeScalar(u8, sel, ',');
    while (it.next()) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

/// Builds the kernel library once per selected level and links them into `hk_mod`.
fn addKernelLevels(
    b: *std.Build,
    hk_mod: *std.Build.Module,
    build_opts: *std.Build.Step.Options,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    sel: []const u8,
) void {
    const arch = target.result.cpu.arch;
    var last: ?usize = null;
    for (variants.all, 0..) |v, i| {
        if (v.arch == arch) last = i;
    }
    const fallback = last orelse std.debug.panic("no kernel level is defined for {s}", .{@tagName(arch)});

    var names: std.ArrayList(u8) = .empty;
    for (variants.all, 0..) |v, i| {
        if (v.arch != arch) continue;
        if (!selected(sel, v.name, i == fallback)) continue;

        var q = target.query;
        q.cpu_arch = arch;
        q.cpu_model = .{ .explicit = std.Target.Cpu.Arch.parseCpuModel(arch, v.model) catch
            std.debug.panic("unknown cpu model {s}", .{v.model}) };
        q.cpu_features_add = std.Target.Cpu.Feature.Set.empty;
        for (v.add) |fname| {
            const list = arch.allFeaturesList();
            const idx = for (list, 0..) |f, fi| {
                if (std.mem.eql(u8, f.name, fname)) break fi;
            } else std.debug.panic("unknown cpu feature {s}", .{fname});
            q.cpu_features_add.addFeature(@intCast(idx));
        }

        const opts = b.addOptions();
        opts.addOption([]const u8, "variant", v.name);
        const mod = b.createModule(.{
            .root_source_file = b.path("src/kernels_impl.zig"),
            .target = b.resolveTargetQuery(q),
            .optimize = optimize,
            .pic = true,
            .imports = &.{.{ .name = "kernel_options", .module = opts.createModule() }},
        });
        const lib = b.addLibrary(.{
            .linkage = .static,
            .name = b.fmt("hk_kernels_{s}", .{v.name}),
            .root_module = mod,
        });
        hk_mod.linkLibrary(lib);

        if (names.items.len != 0) names.append(b.allocator, ',') catch @panic("OOM");
        names.appendSlice(b.allocator, v.name) catch @panic("OOM");
    }
    build_opts.addOption([]const u8, "kernel_levels", names.items);
}
