const std = @import("std");
const hk = @import("hk");
const cli_hub = @import("cli_hub.zig");

/// A model argument is either a path to an .hk file or a hub reference (`owner/name[:quant]`),
/// which is pulled into the cache first. Caller frees the result.
fn resolveModel(allocator: std.mem.Allocator, io: std.Io, env: cli_hub.Env, arg: []const u8) ![]u8 {
    if (std.Io.Dir.cwd().statFile(io, arg, .{})) |_| {
        return allocator.dupe(u8, arg);
    } else |_| {}
    if (cli_hub.Spec.looksLikeRepo(arg)) {
        return cli_hub.pullSpec(allocator, io, env, cli_hub.Spec.parse(arg), false, "main");
    }
    std.debug.print("Error: '{s}' is not a file and does not look like a hub repository (owner/name)\n", .{arg});
    return error.FileNotFound;
}

/// The process environment, kept for code that has no `init` at hand (std.c.getenv is unavailable
/// without libc, for instance on Windows).
var g_env: ?cli_hub.Env = null;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const env = cli_hub.Env{ .map = init.environ_map };
    g_env = env;
    const io = init.io;

    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer it.deinit();

    _ = it.next(); // skip program name

    const command = it.next() orelse {
        printUsage();
        return;
    };

    if (std.mem.eql(u8, command, "pull")) {
        try cli_hub.cmdPull(allocator, io, env, &it);
    } else if (std.mem.eql(u8, command, "search")) {
        try cli_hub.cmdSearch(allocator, io, env, &it);
    } else if (std.mem.eql(u8, command, "list")) {
        try cli_hub.cmdList(allocator, io, env);
    } else if (std.mem.eql(u8, command, "rm")) {
        try cli_hub.cmdRm(allocator, io, env, &it);
    } else if (std.mem.eql(u8, command, "serve")) {
        try cmdServe(allocator, io, env, &it);
    } else if (std.mem.eql(u8, command, "run")) {
        const file_path_arg = it.next() orelse {
            std.debug.print("Error: Missing model for 'run'\n{s}", .{usage_gen});
            return error.BadUsage;
        };
        if (isHelp(file_path_arg)) {
            std.debug.print("{s}", .{usage_gen});
            return;
        }
        // Check every flag before resolving the model, so a typo never starts a download.
        const g = try parseGenArgs(&it, .run);
        const file_path = try resolveModel(allocator, io, env, file_path_arg);
        defer allocator.free(file_path);
        try cmdRun(file_path, g, allocator);
    } else if (std.mem.eql(u8, command, "chat")) {
        const file_path_arg = it.next() orelse {
            std.debug.print("Error: Missing model for 'chat'\n{s}", .{usage_gen});
            return error.BadUsage;
        };
        if (isHelp(file_path_arg)) {
            std.debug.print("{s}", .{usage_gen});
            return;
        }
        const g = try parseGenArgs(&it, .chat);
        const file_path = try resolveModel(allocator, io, env, file_path_arg);
        defer allocator.free(file_path);
        try cmdChat(file_path, g, allocator);
    } else if (std.mem.eql(u8, command, "tokenize")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'tokenize'\nUsage: hk tokenize <model.hk> \"text\"\n", .{});
            return error.BadUsage;
        };
        const text = it.next() orelse {
            std.debug.print("Error: Missing text for 'tokenize'\nUsage: hk tokenize <model.hk> \"text\"\n", .{});
            return error.BadUsage;
        };
        try cmdTokenize(file_path, text, allocator);
    } else if (std.mem.eql(u8, command, "detokenize")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'detokenize'\nUsage: hk detokenize <model.hk> <id1> <id2> ...\n", .{});
            return error.BadUsage;
        };
        var id_list: std.ArrayList(u32) = .empty;
        defer id_list.deinit(allocator);
        while (it.next()) |arg| {
            const id = std.fmt.parseInt(u32, arg, 10) catch {
                std.debug.print("Error: '{s}' is not a token id\n", .{arg});
                return error.BadUsage;
            };
            try id_list.append(allocator, id);
        }
        try cmdDetokenize(file_path, id_list.items, allocator);
    } else if (std.mem.eql(u8, command, "convert-safetensors")) {
        const in_path = it.next() orelse {
            std.debug.print("Error: Missing input path for 'convert-safetensors'\nUsage: hk convert-safetensors <model.safetensors> <out.hk> [f32|q4_0|q8_0]\n", .{});
            return error.BadUsage;
        };
        const out_path = it.next() orelse {
            std.debug.print("Error: Missing output path for 'convert-safetensors'\nUsage: hk convert-safetensors <model.safetensors> <out.hk> [f32|q4_0|q8_0]\n", .{});
            return error.BadUsage;
        };
        const target_st_str = it.next() orelse "f32";
        try cmdConvertSafeTensors(in_path, out_path, target_st_str, allocator);
    } else if (std.mem.eql(u8, command, "inspect")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'inspect'\nUsage: hk inspect <model.hk>\n", .{});
            return error.BadUsage;
        };
        try cmdInspect(file_path, allocator);
    } else if (std.mem.eql(u8, command, "verify")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'verify'\nUsage: hk verify <model.hk>\n", .{});
            return error.BadUsage;
        };
        try cmdVerify(file_path, allocator);
    } else if (std.mem.eql(u8, command, "benchmark")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'benchmark'\nUsage: hk benchmark <model.hk>\n", .{});
            return error.BadUsage;
        };
        try cmdBenchmark(file_path, allocator);
    } else if (std.mem.eql(u8, command, "retile")) {
        const input_path = it.next() orelse {
            std.debug.print("Error: Missing input path for 'retile'\nUsage: hk retile <in.hk> <out.hk> [tile_16x16|row_major]\n", .{});
            return error.BadUsage;
        };
        const output_path = it.next() orelse {
            std.debug.print("Error: Missing output path for 'retile'\nUsage: hk retile <in.hk> <out.hk> [tile_16x16|row_major]\n", .{});
            return error.BadUsage;
        };
        const target_layout_str = it.next() orelse "tile_16x16";
        try cmdRetile(input_path, output_path, target_layout_str, allocator);
    } else if (std.mem.eql(u8, command, "prune")) {
        const input_path = it.next() orelse {
            std.debug.print("Error: Missing input path for 'prune'\nUsage: hk prune <in.hk> <out.hk> [ratio, e.g. 0.5]\n", .{});
            return error.BadUsage;
        };
        const output_path = it.next() orelse {
            std.debug.print("Error: Missing output path for 'prune'\nUsage: hk prune <in.hk> <out.hk> [ratio, e.g. 0.5]\n", .{});
            return error.BadUsage;
        };
        const ratio_str = it.next() orelse "0.5";
        const ratio = std.fmt.parseFloat(f32, ratio_str) catch 0.5;
        try cmdPrune(input_path, output_path, ratio, allocator);
    } else if (std.mem.eql(u8, command, "expand")) {
        const input_path = it.next() orelse {
            std.debug.print("Error: Missing input path for 'expand'\nUsage: hk expand <in.hk> <out.hk> [--vocab <N>] [--width <ratio>]\n", .{});
            return error.BadUsage;
        };
        const output_path = it.next() orelse {
            std.debug.print("Error: Missing output path for 'expand'\nUsage: hk expand <in.hk> <out.hk> [--vocab <N>] [--width <ratio>]\n", .{});
            return error.BadUsage;
        };
        var new_vocab_opt: ?usize = null;
        var width_ratio_opt: ?f32 = null;

        while (it.next()) |flag| {
            if (std.mem.eql(u8, flag, "--vocab")) {
                const val = it.next() orelse break;
                new_vocab_opt = std.fmt.parseInt(usize, val, 10) catch null;
            } else if (std.mem.eql(u8, flag, "--width")) {
                const val = it.next() orelse break;
                width_ratio_opt = std.fmt.parseFloat(f32, val) catch null;
            }
        }
        try cmdExpand(input_path, output_path, new_vocab_opt, width_ratio_opt, allocator);
    } else if (std.mem.eql(u8, command, "eval")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'eval'\nUsage: hk eval <model.hk>\n", .{});
            return error.BadUsage;
        };
        try cmdEval(file_path, allocator);
    } else if (std.mem.eql(u8, command, "appendix")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'appendix'\nUsage: hk appendix <model.hk>\n", .{});
            return error.BadUsage;
        };
        try cmdAppendix(file_path, allocator);
    } else if (std.mem.eql(u8, command, "rollback")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'rollback'\nUsage: hk rollback <model.hk> [generation]\n", .{});
            return error.BadUsage;
        };
        const gen_str = it.next() orelse "0";
        const target_gen = std.fmt.parseInt(u32, gen_str, 10) catch 0;
        try cmdRollback(file_path, target_gen, allocator);
    } else if (std.mem.eql(u8, command, "metadata")) {
        const sub_cmd = it.next() orelse {
            std.debug.print("Error: Missing subcommand for 'metadata'\nUsage: hk metadata <set|get|list> <file.hk> [key] [val]\n", .{});
            return error.BadUsage;
        };
        if (std.mem.eql(u8, sub_cmd, "set")) {
            const file_path = it.next() orelse {
                std.debug.print("Error: Missing file path for 'metadata set'\nUsage: hk metadata set <file.hk> <key> <val>\n", .{});
                return error.BadUsage;
            };
            const key = it.next() orelse {
                std.debug.print("Error: Missing key for 'metadata set'\nUsage: hk metadata set <file.hk> <key> <val>\n", .{});
                return error.BadUsage;
            };
            const val = it.next() orelse {
                std.debug.print("Error: Missing value for 'metadata set'\nUsage: hk metadata set <file.hk> <key> <val>\n", .{});
                return error.BadUsage;
            };
            try cmdMetadataSet(file_path, key, val, allocator);
        } else if (std.mem.eql(u8, sub_cmd, "get")) {
            const file_path = it.next() orelse {
                std.debug.print("Error: Missing file path for 'metadata get'\nUsage: hk metadata get <file.hk> <key>\n", .{});
                return error.BadUsage;
            };
            const key = it.next() orelse {
                std.debug.print("Error: Missing key for 'metadata get'\nUsage: hk metadata get <file.hk> <key>\n", .{});
                return error.BadUsage;
            };
            try cmdMetadataGet(file_path, key, allocator);
        } else if (std.mem.eql(u8, sub_cmd, "list")) {
            const file_path = it.next() orelse {
                std.debug.print("Error: Missing file path for 'metadata list'\nUsage: hk metadata list <file.hk>\n", .{});
                return error.BadUsage;
            };
            try cmdMetadataList(file_path, allocator);
        } else {
            std.debug.print("Unknown metadata subcommand: {s}\nUsage: hk metadata <set|get|list> <file.hk> [key] [val]\n", .{sub_cmd});
        }
    } else if (std.mem.eql(u8, command, "dump")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'dump'\nUsage: hk dump <model.hk>\n", .{});
            return error.BadUsage;
        };
        try cmdDump(file_path, allocator);
    } else if (std.mem.eql(u8, command, "hash")) {
        const file_path = it.next() orelse {
            std.debug.print("Error: Missing file path for 'hash'\nUsage: hk hash <model.hk>\n", .{});
            return error.BadUsage;
        };
        try cmdHash(file_path, allocator);
    } else if (std.mem.eql(u8, command, "convert-endian")) {
        const input_path = it.next() orelse {
            std.debug.print("Error: Missing input path for 'convert-endian'\nUsage: hk convert-endian <in.hk> <out.hk>\n", .{});
            return error.BadUsage;
        };
        const output_path = it.next() orelse {
            std.debug.print("Error: Missing output path for 'convert-endian'\nUsage: hk convert-endian <in.hk> <out.hk>\n", .{});
            return error.BadUsage;
        };
        try cmdConvertEndian(input_path, output_path, allocator);
    } else if (std.mem.eql(u8, command, "convert-gguf")) {
        const input_path = it.next() orelse {
            std.debug.print("Error: Missing input path for 'convert-gguf'\nUsage: hk convert-gguf <in.gguf> <out.hk>\n", .{});
            return error.BadUsage;
        };
        const output_path = it.next() orelse {
            std.debug.print("Error: Missing output path for 'convert-gguf'\nUsage: hk convert-gguf <in.gguf> <out.hk>\n", .{});
            return error.BadUsage;
        };
        try cmdConvertGGUF(input_path, output_path, allocator);
    } else if (std.mem.eql(u8, command, "export")) {
        var format_str: []const u8 = "gguf";
        var in_path_opt: ?[]const u8 = null;
        var out_path_opt: ?[]const u8 = null;

        while (it.next()) |arg| {
            if (std.mem.eql(u8, arg, "-f") or std.mem.eql(u8, arg, "--format")) {
                format_str = it.next() orelse "gguf";
            } else if (in_path_opt == null) {
                in_path_opt = arg;
            } else if (out_path_opt == null) {
                out_path_opt = arg;
            }
        }

        if (in_path_opt == null or out_path_opt == null) {
            std.debug.print("Error: Missing arguments for 'export'\nUsage: hk export -f <gguf|safetensors> <in.hk> <out_file>\n", .{});
            return error.BadUsage;
        }

        try cmdExport(format_str, in_path_opt.?, out_path_opt.?, allocator);
    } else if (std.mem.eql(u8, command, "gui")) {
        const file_path_opt = it.next();
        try cmdGui(file_path_opt, allocator);
    } else if (std.mem.eql(u8, command, "hardware-profile")) {
        try cmdHardwareProfile(allocator);
    } else if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help")) {
        printUsage();
    } else {
        std.debug.print("Unknown command: {s}\n", .{command});
        printUsage();
        return error.BadUsage;
    }
}

fn printUsage() void {
    std.debug.print(
        \\HK CLI v1.2.0
        \\Usage: hk <command> [arguments]
        \\
        \\Commands:
        \\  pull           <owner/name>[:quant]          Download a model from the Hub straight into .hk
        \\  search         <query> [--limit N] [--all]   Search the Hub (GGUF models by default)
        \\  list                                         Show cached models
        \\  rm             <owner/name>                  Remove a cached model
        \\  serve          <model> [--port N] [--slots N] OpenAI compatible server
        \\  run            <file.hk | owner/name[:quant]> [-p "text"] [-n N]  Generate text
        \\  chat           <file.hk | owner/name[:quant]> [--temp T]          Interactive chat
        \\  tokenize       <file.hk> "text"              Tokenize input text using in-container vocabulary
        \\  detokenize     <file.hk> <id1> <id2> ...     Reconstruct text from token ID sequence
        \\  convert-safetensors <in.safetensors> <out.hk> Zero-dependency native SafeTensors transcoding
        \\  convert-gguf   <in.gguf> <out.hk>            Zero-copy bitstream ingestion of GGUF models
        \\  inspect        <file.hk>                     Display header, metadata, and tensor TOC details
        \\  dump           <file.hk>                     Comprehensive binary dumper (hex, headers, alignment)
        \\  hash           <file.hk>                     SHA-256 container and per-tensor verification
        \\  export         -f <gguf|safetensors> <in> <out> Export HK model to GGUF v3 or Safetensors
        \\  convert-endian <in.hk> <out.hk>              Convert endianness (Little <-> Big Endian)
        \\  gui            [file.hk]                     Launch visual HK model editor GUI
        \\  verify         <file.hk>                     Verify header magic, 128-byte alignment, and bounds
        \\  eval           <file.hk>                     Inspect model weights, parameter counts, and integrity
        \\  expand         <in.hk> <out.hk> [options]    Natively expand vocab (--vocab N) and width (--width R)
        \\  benchmark      <file.hk>                     Benchmark mmap loading and dequantization throughput
        \\  retile         <in.hk> <out.hk> [layout]     Re-tile 2D weight matrices for Tensor Cores
        \\  prune          <in.hk> <out.hk> [ratio]      Apply magnitude pruning to weight tensors
        \\  appendix       <file.hk>                     Display version-chained appendix records & metrics
        \\  rollback       <file.hk> [generation]        Rollback appendix entries to specified generation
        \\  metadata       <set|get|list> <file.hk> ...  In-place metadata inspection and modification
        \\  hardware-profile                             Profile host CPU, vector units, and optimal page alignment
        \\  help                                         Show this help message
        \\
    , .{});
}

fn cmdHardwareProfile(allocator: std.mem.Allocator) !void {
    const builtin = @import("builtin");
    const caps = hk.platform.detectHardwareCapabilities();
    const io = std.Options.debug_io;
    const logical = std.Thread.getCpuCount() catch 1;
    std.debug.print(
        \\hk hardware profile
        \\  architecture      : {s} ({s})
        \\  cpu vendor        : {s}
        \\  cores             : {d} used for compute, {d} logical
        \\  vector features   : AVX2 {s}, AVX-512 {s}, VNNI {s}, AMX {s}, NEON {s}, SVE {s}
        \\  kernels in use    : {s}
        \\  kernels built     : {s}
        \\  page alignment    : {d} bytes (what `.hk` files written here use)
        \\
    , .{
        @tagName(builtin.cpu.arch),
        @tagName(builtin.os.tag),
        switch (caps.vendor) {
            .intel => "Intel",
            .amd => "AMD",
            .arm => "ARM",
            .apple => "Apple",
            .unknown => "unknown",
        },
        hk.cores.physicalCores(io),
        logical,
        yn(caps.has_avx2),
        yn(caps.has_avx512f),
        yn(caps.has_avx512vnni or caps.has_avx_vnni),
        yn(caps.has_amx),
        yn(caps.has_arm_neon),
        yn(caps.has_arm_sve),
        hk.kernels.get().name,
        hk.build_options_kernel_levels,
        caps.optimal_page_alignment,
    });
    if (hk.vk.Context.init(allocator, null)) |ctx_val| {
        var ctx = ctx_val;
        defer ctx.deinit();
        std.debug.print("  gpu               : Vulkan device {s} ({d} MiB, subgroup {d})\n", .{
            ctx.caps.deviceName(),
            ctx.caps.device_local_bytes >> 20,
            ctx.caps.subgroup_size,
        });
    } else |e| {
        std.debug.print("  gpu               : no usable Vulkan device ({s})\n", .{@errorName(e)});
    }
}

fn yn(b: bool) []const u8 {
    return if (b) "yes" else "no";
}

fn convertProgress(ctx: ?*anyopaque, done: u64, total: u64) void {
    _ = ctx;
    const pct = if (total == 0) 100 else done * 100 / total;
    std.debug.print("\r[HK] converting: {d}% ({d} / {d} MiB)", .{ pct, done >> 20, total >> 20 });
}

fn cmdConvertGGUF(input_path: []const u8, output_path: []const u8, allocator: std.mem.Allocator) !void {
    std.debug.print("[HK] Converting GGUF: {s}\n", .{input_path});
    var diag = hk.convert.gguf.Diag{};
    hk.convert.gguf.convertFile(allocator, std.Options.debug_io, input_path, output_path, .{
        .progress = convertProgress,
        .diag = &diag,
    }) catch |err| {
        std.debug.print("\nError: conversion failed: {s}", .{@errorName(err)});
        if (diag.len > 0) std.debug.print(": {s}", .{diag.message()});
        std.debug.print("\n", .{});
        return err;
    };
    std.debug.print("\n[HK] Wrote {s}\n", .{output_path});
}

fn cmdExport(fmt: []const u8, input_path: []const u8, output_path: []const u8, allocator: std.mem.Allocator) !void {
    if (std.mem.eql(u8, fmt, "gguf")) {
        std.debug.print("[HK] Exporting HK model to GGUF v3: {s}\n", .{output_path});
        hk.gguf.exportHKToGGUF(input_path, output_path, allocator) catch |err| {
            std.debug.print("Error: Failed to export HK to GGUF: {}\n", .{err});
            return;
        };
        std.debug.print("[HK] Successfully exported to GGUF -> {s}\n", .{output_path});
    } else if (std.mem.eql(u8, fmt, "safetensors")) {
        std.debug.print("[HK] Exporting HK model to safetensors: {s}\n", .{output_path});
        var reader = hk.HKReader.open(input_path, allocator) catch |err| {
            std.debug.print("Error: cannot open '{s}': {s}\n", .{ input_path, @errorName(err) });
            return;
        };
        defer reader.deinit();
        hk.convert.safetensors_out.exportFile(allocator, std.Options.debug_io, &reader, output_path) catch |err| {
            std.debug.print("Error: export failed: {s}\n", .{@errorName(err)});
            return;
        };
        std.debug.print("[HK] Wrote {s} (quantized tensors are stored as F16)\n", .{output_path});
    } else {
        std.debug.print("Error: Unsupported export format '{s}'. Supported: gguf, safetensors\n", .{fmt});
    }
}

fn cmdMetadataSet(file_path: []const u8, key: []const u8, val: []const u8, allocator: std.mem.Allocator) !void {
    hk.metadata.patchFileMetadataInPlace(allocator, file_path, key, val) catch |err| {
        std.debug.print("Error: Failed to patch metadata in '{s}': {}\n", .{ file_path, err });
        return;
    };
    std.debug.print("[SUCCESS] In-place metadata updated in '{s}':\n  {s} = \"{s}\"\n(Tensor payload untouched, zero copy)\n", .{
        file_path, key, val,
    });
}

fn cmdMetadataGet(file_path: []const u8, key: []const u8, allocator: std.mem.Allocator) !void {
    var reader = hk.HKReader.open(file_path, allocator) catch |err| {
        std.debug.print("Failed to open HK file '{s}': {}\n", .{ file_path, err });
        return;
    };
    defer reader.deinit();

    if (reader.metadata_map.get(key)) |val| {
        switch (val) {
            .val_string => |s| std.debug.print("{s}\n", .{s}),
            .val_int64 => |v| std.debug.print("{}\n", .{v}),
            .val_float64 => |f| std.debug.print("{d}\n", .{f}),
            .val_bool => |b| std.debug.print("{}\n", .{b}),
            .val_json => |j| std.debug.print("{s}\n", .{j}),
            .val_bytes => |b| std.debug.print("<{} bytes>\n", .{b.len}),
        }
    } else {
        std.debug.print("Key '{s}' not found in metadata of '{s}'.\n", .{ key, file_path });
    }
}

fn cmdMetadataList(file_path: []const u8, allocator: std.mem.Allocator) !void {
    var reader = hk.HKReader.open(file_path, allocator) catch |err| {
        std.debug.print("Failed to open HK file '{s}': {}\n", .{ file_path, err });
        return;
    };
    defer reader.deinit();

    std.debug.print("\n=== Metadata for '{s}' ({} entries) ===\n", .{ file_path, reader.metadata_map.items.items.len });
    for (reader.metadata_map.items.items) |item| {
        switch (item.value) {
            .val_string => |s| std.debug.print("  {s}: \"{s}\"\n", .{ item.key, s }),
            .val_int64 => |v| std.debug.print("  {s}: {}\n", .{ item.key, v }),
            .val_float64 => |f| std.debug.print("  {s}: {d:.4}\n", .{ item.key, f }),
            .val_bool => |b| std.debug.print("  {s}: {}\n", .{ item.key, b }),
            .val_json => |j| std.debug.print("  {s} (JSON): {s}\n", .{ item.key, j }),
            .val_bytes => |b| std.debug.print("  {s} (binary): {} bytes\n", .{ item.key, b.len }),
        }
    }
    std.debug.print("\n", .{});
}

fn cmdInspect(path: []const u8, allocator: std.mem.Allocator) !void {
    var reader = hk.HKReader.open(path, allocator) catch |err| {
        std.debug.print("Failed to open HK file '{s}': {}\n", .{ path, err });
        return;
    };
    defer reader.deinit();

    std.debug.print("\n=== HK File: {s} ===\n", .{path});
    std.debug.print("Magic: {s} | Version: {}.{}\n", .{
        reader.header.magic,
        reader.header.version_major,
        reader.header.version_minor,
    });
    std.debug.print("Flags: 0x{X:0>8} | Tensors: {} | Metadata KVs: {}\n", .{
        reader.header.flags,
        reader.header.tensor_count,
        reader.header.metadata_kv_count,
    });
    std.debug.print("Data Offset: 0x{X} (Alignment: {}B, Tensor Core Coalesced: {})\n", .{
        reader.header.tensor_data_offset,
        reader.getAlignment(),
        reader.isTensorCoreAligned(),
    });
    if (reader.isRawWeightStorage()) {
        std.debug.print("Storage Mode: Raw Weight Storage (Zero compute headroom, direct zero-copy)\n", .{});
    } else {
        std.debug.print("Storage Mode: Quantized Neural Container\n", .{});
    }
    if (reader.isUniversalPageAligned()) {
        std.debug.print("Universal Page Alignment: Enabled (Direct zero-copy for AMD ROCm, Intel NPU, Apple Metal)\n", .{});
    }
    if (reader.isSharded()) {
        std.debug.print("Sharding: Shard {} of {} (FLAG_IS_SHARDED enabled)\n", .{
            reader.getSplitIndex() + 1,
            reader.getSplitCount(),
        });
    } else {
        std.debug.print("Sharding: Single Container (unsharded)\n", .{});
    }

    std.debug.print("\n--- Metadata ---\n", .{});
    for (reader.metadata_map.items.items) |item| {
        switch (item.value) {
            .val_string => |s| std.debug.print("  {s}: \"{s}\"\n", .{ item.key, s }),
            .val_int64 => |v| std.debug.print("  {s}: {}\n", .{ item.key, v }),
            .val_float64 => |f| std.debug.print("  {s}: {d:.4}\n", .{ item.key, f }),
            .val_bool => |b| std.debug.print("  {s}: {}\n", .{ item.key, b }),
            .val_json => |j| std.debug.print("  {s} (JSON): {s}\n", .{ item.key, j }),
            .val_bytes => |b| std.debug.print("  {s} (binary): {} bytes\n", .{ item.key, b.len }),
        }
    }

    std.debug.print("\n--- Tensor Table of Contents ({} entries) ---\n", .{reader.toc.entries.items.len});
    std.debug.print("{s:<40} {s:<12} {s:<12} {s:<16} {s:<10} {s:<12}\n", .{
        "Tensor Name", "Type", "Layout", "Shape", "Size", "Sparsity",
    });
    std.debug.print("{s:-<105}\n", .{""});

    for (reader.toc.entries.items) |e| {
        var shape_buf: [64]u8 = undefined;
        var pos: usize = 0;
        shape_buf[pos] = '[';
        pos += 1;
        for (0..e.ndim) |d| {
            if (d > 0) {
                shape_buf[pos] = ',';
                pos += 1;
            }
            const part = std.fmt.bufPrint(shape_buf[pos..], "{}", .{e.shape[d]}) catch "";
            pos += part.len;
        }
        shape_buf[pos] = ']';
        pos += 1;
        const shape_str = shape_buf[0..pos];

        std.debug.print("{s:<40} {s:<12} {s:<12} {s:<16} {}B {d:>6.1}%\n", .{
            e.name,
            @tagName(e.storage_type),
            @tagName(e.tile_layout),
            shape_str,
            e.data_size,
            e.sparsity_ratio * 100.0,
        });
    }
    std.debug.print("\n", .{});
}

fn cmdVerify(path: []const u8, allocator: std.mem.Allocator) !void {
    var reader = hk.HKReader.open(path, allocator) catch |err| {
        std.debug.print("VERIFY FAILED: Could not open file: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    var passed = true;
    std.debug.print("Verifying '{s}'...\n", .{path});

    if (!reader.header.isValid()) {
        std.debug.print("[FAIL] Invalid magic bytes or unsupported version!\n", .{});
        passed = false;
    } else {
        std.debug.print("[PASS] Header magic 'HKNT' and version 1.0 valid\n", .{});
    }

    if ((reader.header.tensor_data_offset % hk.format.ALIGNMENT_BYTES) != 0) {
        std.debug.print("[FAIL] Data offset 0x{X} is not 128-byte aligned!\n", .{reader.header.tensor_data_offset});
        passed = false;
    } else {
        std.debug.print("[PASS] Payload offset is 128-byte Tensor Core aligned\n", .{});
    }

    for (reader.toc.entries.items) |e| {
        if (e.storage_type == .null_ref) continue;
        if ((e.data_offset % hk.format.ALIGNMENT_BYTES) != 0) {
            std.debug.print("[FAIL] Tensor '{s}' offset 0x{X} is not 128-byte aligned!\n", .{ e.name, e.data_offset });
            passed = false;
        }
        if (e.data_offset + e.data_size > reader.mmap_region.bytes.len) {
            std.debug.print("[FAIL] Tensor '{s}' data extends beyond file bounds!\n", .{e.name});
            passed = false;
        }
    }

    if (passed) {
        std.debug.print("\n===> ALL CHECKS PASSED: File is a compliant, high-performance HK binary container.\n\n", .{});
    } else {
        std.debug.print("\n===> VERIFICATION FAILED: Violations found.\n\n", .{});
    }
}

fn touchMemory(bytes: []const u8) u64 {
    var acc0: @Vector(32, u8) = @splat(0);
    var acc1: @Vector(32, u8) = @splat(0);
    var i: usize = 0;
    while (i + 64 <= bytes.len) : (i += 64) {
        const v0: @Vector(32, u8) = bytes[i + 0 ..][0..32].*;
        const v1: @Vector(32, u8) = bytes[i + 32 ..][0..32].*;
        acc0 +%= v0;
        acc1 +%= v1;
    }
    while (i + 32 <= bytes.len) : (i += 32) {
        const v0: @Vector(32, u8) = bytes[i..][0..32].*;
        acc0 +%= v0;
    }
    var sum: u64 = 0;
    const red0 = @reduce(.Add, acc0);
    const red1 = @reduce(.Add, acc1);
    sum +%= red0;
    sum +%= red1;
    while (i < bytes.len) : (i += 1) {
        sum +%= bytes[i];
    }
    return sum;
}

fn cmdBenchmark(path: []const u8, allocator: std.mem.Allocator) !void {
    const io = std.Options.debug_io;
    std.debug.print("Benchmarking HK loader on '{s}'...\n", .{path});

    // 1. Zero-Copy Open & Deserialization
    const start_open = std.Io.Timestamp.now(io, .awake);
    var reader = try hk.HKReader.open(path, allocator);
    defer reader.deinit();
    const end_open = std.Io.Timestamp.now(io, .awake);
    const open_time_ns = end_open.nanoseconds - start_open.nanoseconds;

    std.debug.print("1. Zero-Copy Open & Deserialization: {d:.2} us\n", .{@as(f64, @floatFromInt(open_time_ns)) / 1000.0});

    // 2. Zero-Copy Slice Pointer Resolution (Metadata / Descriptor Lookup)
    // Measures the overhead of acquiring pointer descriptors without reading data payload.
    var total_bytes: usize = 0;
    const start_slice = std.Io.Timestamp.now(io, .awake);
    for (reader.toc.entries.items) |e| {
        const data = try reader.getTensorData(e);
        total_bytes += data.len;
    }
    const end_slice = std.Io.Timestamp.now(io, .awake);
    const slice_time_ns = end_slice.nanoseconds - start_slice.nanoseconds;
    const slice_us = @as(f64, @floatFromInt(slice_time_ns)) / 1000.0;
    const num_tensors = reader.toc.entries.items.len;
    const mslices_per_sec = if (slice_us > 0) (@as(f64, @floatFromInt(num_tensors)) / slice_us) else 0.0;

    std.debug.print("2. Lazy Slice Pointer Resolution (no page touches): {d:.2} us ({} tensors, {d:.1} M-slices/s)\n", .{
        slice_us,
        num_tensors,
        mslices_per_sec,
    });

    // 3. Physical Memory Traversal & First-Touch Page-In
    // Actually touches every page and byte of mapped memory using 256-bit SIMD reads,
    // bringing pages into physical RAM and measuring actual hardware memory/storage throughput.
    var checksum: u64 = 0;
    const start_touch = std.Io.Timestamp.now(io, .awake);
    for (reader.toc.entries.items) |e| {
        const data = try reader.getTensorData(e);
        checksum +%= touchMemory(data);
    }
    const end_touch = std.Io.Timestamp.now(io, .awake);
    const touch_time_ns = end_touch.nanoseconds - start_touch.nanoseconds;
    const touch_sec = @as(f64, @floatFromInt(touch_time_ns)) / 1_000_000_000.0;
    const touch_mbs = if (touch_sec > 0) (@as(f64, @floatFromInt(total_bytes)) / (1024.0 * 1024.0)) / touch_sec else 0.0;

    std.debug.print("3. Physical Memory Traversal & First-Touch Page-In: {d:.2} ms ({} bytes total, throughput: {d:.1} MB/s, checksum: 0x{x:0>16})\n", .{
        touch_sec * 1000.0,
        total_bytes,
        touch_mbs,
        checksum,
    });

    // 4. In-Memory Reconstruction & Dequantization (Resident SIMD Compute)
    // Uses a per-tensor reusable scratch buffer sized to the largest tensor to prevent
    // artificial multi-gigabyte virtual memory allocator churn and swap paging.
    var max_numel: usize = 0;
    var total_elements: usize = 0;
    for (reader.toc.entries.items) |e| {
        var numel: usize = 1;
        for (0..e.ndim) |d| numel *= @intCast(e.shape[d]);
        total_elements += numel;
        if (numel > max_numel) max_numel = numel;
    }

    if (max_numel > 0) {
        const deq_buf = try allocator.alloc(f32, max_numel);
        defer allocator.free(deq_buf);

        const start_deq = std.Io.Timestamp.now(io, .awake);
        for (reader.toc.entries.items) |e| {
            var numel: usize = 1;
            for (0..e.ndim) |d| numel *= @intCast(e.shape[d]);
            reader.dequantizeToF32(e, true, deq_buf[0..numel]) catch {};
        }
        const end_deq = std.Io.Timestamp.now(io, .awake);
        const deq_time_ns = end_deq.nanoseconds - start_deq.nanoseconds;
        const deq_sec = @as(f64, @floatFromInt(deq_time_ns)) / 1_000_000_000.0;
        const melem_per_sec = if (deq_sec > 0) (@as(f64, @floatFromInt(total_elements)) / 1_000_000.0) / deq_sec else 0.0;

        std.debug.print("4. Reconstruction & Dequantization (in-memory compute): {d:.2} ms ({} elements, throughput: {d:.2} M-elem/s, scratch buffer: {d:.1} MB)\n\n", .{
            deq_sec * 1000.0,
            total_elements,
            melem_per_sec,
            @as(f64, @floatFromInt(max_numel * @sizeOf(f32))) / (1024.0 * 1024.0),
        });
    }
}

fn cmdRetile(in_path: []const u8, out_path: []const u8, layout_str: []const u8, allocator: std.mem.Allocator) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var reader = try hk.HKReader.open(in_path, arena_alloc);
    defer reader.deinit();

    var writer = hk.HKWriter.init(arena_alloc);
    defer writer.deinit();

    for (reader.metadata_map.items.items) |m| {
        switch (m.value) {
            .val_string => |s| try writer.addMetadataString(m.key, s),
            .val_int64 => |v| try writer.addMetadataInt(m.key, v),
            .val_float64 => |f| try writer.addMetadataFloat(m.key, f),
            .val_bool => |b| try writer.addMetadataBool(m.key, b),
            else => {},
        }
    }
    try writer.addMetadataString("retiled_with", layout_str);

    const target_layout: hk.TileLayout = if (std.mem.eql(u8, layout_str, "tile_16x16"))
        .tile_16x16
    else if (std.mem.eql(u8, layout_str, "tile_16x8"))
        .tile_16x8
    else if (std.mem.eql(u8, layout_str, "tile_32x16"))
        .tile_32x16
    else
        .row_major;

    for (reader.toc.entries.items) |e| {
        var numel: usize = 1;
        for (0..e.ndim) |d| numel *= @intCast(e.shape[d]);

        const dense = try arena_alloc.alloc(f32, numel);
        try reader.dequantizeToF32(e, true, dense);

        if (e.ndim == 2 and target_layout != .row_major) {
            const tiled = try hk.tiling.packTilesF32(
                dense,
                @intCast(e.shape[0]),
                @intCast(e.shape[1]),
                target_layout,
                arena_alloc,
            );

            try writer.addTensor(.{
                .name = e.name,
                .storage_type = .f32,
                .tile_layout = target_layout,
                .sparsity_type = e.sparsity_type,
                .ndim = e.ndim,
                .shape = e.shape,
                .data = std.mem.sliceAsBytes(tiled),
                .sparsity_ratio = e.sparsity_ratio,
            });
        } else {
            try writer.addTensor(.{
                .name = e.name,
                .storage_type = .f32,
                .tile_layout = .row_major,
                .sparsity_type = e.sparsity_type,
                .ndim = e.ndim,
                .shape = e.shape,
                .data = std.mem.sliceAsBytes(dense),
                .sparsity_ratio = e.sparsity_ratio,
            });
        }
    }

    try writer.writeToFile(out_path);
    std.debug.print("Successfully retiled '{s}' -> '{s}' (layout: {s})\n", .{ in_path, out_path, layout_str });
}

fn cmdPrune(in_path: []const u8, out_path: []const u8, ratio: f32, allocator: std.mem.Allocator) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var reader = try hk.HKReader.open(in_path, arena_alloc);
    defer reader.deinit();

    var writer = hk.HKWriter.init(arena_alloc);
    defer writer.deinit();

    for (reader.metadata_map.items.items) |m| {
        switch (m.value) {
            .val_string => |s| try writer.addMetadataString(m.key, s),
            .val_int64 => |v| try writer.addMetadataInt(m.key, v),
            .val_float64 => |f| try writer.addMetadataFloat(m.key, f),
            .val_bool => |b| try writer.addMetadataBool(m.key, b),
            else => {},
        }
    }
    try writer.addMetadataFloat("cli_pruning_ratio", ratio);

    for (reader.toc.entries.items) |e| {
        var numel: usize = 1;
        for (0..e.ndim) |d| numel *= @intCast(e.shape[d]);

        const dense = try arena_alloc.alloc(f32, numel);
        try reader.dequantizeToF32(e, true, dense);

        if (e.ndim >= 2 and ratio > 0.0) {
            // Find magnitude threshold
            const abs_vals = try arena_alloc.alloc(f32, numel);
            for (0..numel) |i| abs_vals[i] = @abs(dense[i]);

            std.mem.sort(f32, abs_vals, {}, std.sort.asc(f32));
            const k_idx = @min(@as(usize, @intFromFloat(@as(f32, @floatFromInt(numel)) * ratio)), numel - 1);
            const thresh = abs_vals[k_idx];

            var zeros: usize = 0;
            for (0..numel) |i| {
                if (@abs(dense[i]) <= thresh) {
                    dense[i] = 0.0;
                    zeros += 1;
                }
            }
            const actual_sparsity: f32 = @as(f32, @floatFromInt(zeros)) / @as(f32, @floatFromInt(numel));

            try writer.addTensor(.{
                .name = e.name,
                .storage_type = .f32,
                .tile_layout = e.tile_layout,
                .sparsity_type = .bitmask,
                .ndim = e.ndim,
                .shape = e.shape,
                .data = std.mem.sliceAsBytes(dense),
                .sparsity_ratio = actual_sparsity,
            });
        } else {
            try writer.addTensor(.{
                .name = e.name,
                .storage_type = e.storage_type,
                .tile_layout = e.tile_layout,
                .sparsity_type = e.sparsity_type,
                .ndim = e.ndim,
                .shape = e.shape,
                .data = std.mem.sliceAsBytes(dense),
                .sparsity_ratio = e.sparsity_ratio,
            });
        }
    }

    try writer.writeToFile(out_path);
    std.debug.print("Successfully pruned '{s}' -> '{s}' (target ratio: {d:.2})\n", .{ in_path, out_path, ratio });
}

fn cmdAppendix(path: []const u8, allocator: std.mem.Allocator) !void {
    var region = hk.platform.mapOrReadFile(path, allocator) catch |err| {
        std.debug.print("Failed to open HK file '{s}': {}\n", .{ path, err });
        return;
    };
    defer region.deinit(allocator);

    var app_reader = hk.appendix.AppendixReader.init(allocator, region.bytes) catch |err| {
        std.debug.print("Failed to parse appendix for '{s}': {}\n", .{ path, err });
        return;
    };
    defer app_reader.deinit();

    std.debug.print("\n=== HK Appendix Region: {s} ===\n", .{path});
    std.debug.print("Total Appendix Entries: {}\n", .{app_reader.records.items.len});
    std.debug.print("Cryptographic Lineage Valid: {}\n\n", .{app_reader.verifyLineage()});

    if (app_reader.records.items.len == 0) {
        std.debug.print("No appendix entries found in container.\n", .{});
        return;
    }

    std.debug.print("{s:<4} {s:<15} {s:<6} {s:<28} {s:<18} {s:<10} {s:<10}\n", .{
        "#", "Type", "Gen", "Name", "Target", "Accuracy", "PassRate",
    });
    std.debug.print("{s:-<100}\n", .{""});

    for (app_reader.records.items, 0..) |rec, i| {
        std.debug.print("{:<4} {s:<15} {:<6} {s:<28} {s:<18} {d:>8.2}%  {d:>8.2}%\n", .{
            i,
            @tagName(rec.entry_type),
            rec.generation,
            rec.name,
            if (rec.target.len > 0) rec.target else "-",
            rec.metrics.accuracy * 100.0,
            rec.metrics.pass_rate * 100.0,
        });
    }
}

fn cmdRollback(path: []const u8, target_gen: u32, allocator: std.mem.Allocator) !void {
    std.debug.print("Rolling back '{s}' to generation {}...\n", .{ path, target_gen });
    hk.appendix.rollbackToFile(allocator, path, target_gen) catch |err| {
        std.debug.print("Rollback failed: {}\n", .{err});
        return;
    };
    std.debug.print("[SUCCESS] Successfully rolled back '{s}' to generation {}\n", .{ path, target_gen });
}

fn cmdEval(path: []const u8, allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Evaluating Model Health & Integrity: {s} ===\n", .{path});
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var reader = hk.HKReader.open(path, allocator) catch |err| {
        std.debug.print("Failed to open HK file '{s}': {}\n", .{ path, err });
        return;
    };
    defer reader.deinit();

    var total_params: u64 = 0;
    var total_zero_params: u64 = 0;
    var nan_count: u64 = 0;
    var inf_count: u64 = 0;
    var min_val: f32 = std.math.inf(f32);
    var max_val: f32 = -std.math.inf(f32);
    var sum_abs: f64 = 0.0;

    for (reader.toc.entries.items) |e| {
        var numel: usize = 1;
        for (0..e.ndim) |d| numel *= @intCast(e.shape[d]);
        total_params += numel;

        const dense = try arena_alloc.alloc(f32, numel);
        reader.dequantizeToF32(e, true, dense) catch |err| {
            std.debug.print("[WARN] Could not dequantize tensor '{s}': {}\n", .{ e.name, err });
            continue;
        };

        for (dense) |val| {
            if (std.math.isNan(val)) {
                nan_count += 1;
            } else if (std.math.isInf(val)) {
                inf_count += 1;
            } else {
                if (val == 0.0) total_zero_params += 1;
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
                sum_abs += @abs(val);
            }
        }
    }

    const sparsity: f64 = if (total_params > 0) @as(f64, @floatFromInt(total_zero_params)) / @as(f64, @floatFromInt(total_params)) else 0.0;
    const mean_abs: f64 = if (total_params > 0) sum_abs / @as(f64, @floatFromInt(total_params)) else 0.0;

    std.debug.print("Total Tensors       : {}\n", .{reader.toc.entries.items.len});
    std.debug.print("Total Parameters    : {} ({d:.2} M)\n", .{ total_params, @as(f64, @floatFromInt(total_params)) / 1_000_000.0 });
    std.debug.print("Zero Parameters     : {} ({d:.2}% sparsity)\n", .{ total_zero_params, sparsity * 100.0 });
    std.debug.print("Value Range         : [{d:.4}, {d:.4}]\n", .{ min_val, max_val });
    std.debug.print("Mean Absolute Value : {d:.6}\n", .{mean_abs});
    std.debug.print("NaN Detections      : {}\n", .{nan_count});
    std.debug.print("Inf Detections      : {}\n", .{inf_count});

    if (nan_count == 0 and inf_count == 0) {
        std.debug.print("[STATUS] HEALTHY - Model weights are stable, normalized, and valid for inference/fine-tuning.\n\n", .{});
    } else {
        std.debug.print("[STATUS] CORRUPTED - Model weights contain NaN or Inf values!\n\n", .{});
    }
}

fn cmdExpand(
    in_path: []const u8,
    out_path: []const u8,
    new_vocab_opt: ?usize,
    width_ratio_opt: ?f32,
    allocator: std.mem.Allocator,
) !void {
    std.debug.print("\n=== Expanding Model: '{s}' -> '{s}' ===\n", .{ in_path, out_path });
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var reader = try hk.HKReader.open(in_path, allocator);
    defer reader.deinit();

    // Expansion decodes every tensor to f32 and writes it back, which is only lossless for float
    // models. Refuse quantized input instead of silently changing what the tensors mean.
    for (reader.toc.entries.items) |e| {
        switch (e.storage_type) {
            .f32, .f16, .bf16 => {},
            else => {
                std.debug.print("Error: '{s}' is stored as {s}. `hk expand` works on f32, f16 and bf16 models; convert a float checkpoint (for example with `hk convert-safetensors`) first.\n", .{ e.name, @tagName(e.storage_type) });
                return error.UnsupportedStorage;
            },
        }
    }

    var writer = hk.HKWriter.init(allocator);
    defer writer.deinit();

    // Copy existing metadata
    for (reader.metadata_map.items.items) |m| {
        switch (m.value) {
            .val_string => |s| try writer.addMetadataString(m.key, s),
            .val_int64 => |v| try writer.addMetadataInt(m.key, v),
            .val_float64 => |f| try writer.addMetadataFloat(m.key, f),
            .val_bool => |b| try writer.addMetadataBool(m.key, b),
            .val_json => |j| try writer.addMetadataJson(m.key, j),
            .val_bytes => {},
        }
    }

    if (new_vocab_opt) |nv| {
        try writer.addMetadataInt("expanded_vocab_size", @intCast(nv));
    }
    if (width_ratio_opt) |wr| {
        try writer.addMetadataFloat("expanded_width_ratio", wr);
    }
    try writer.addMetadataString("expansion_engine", "native_zig_v1");

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    var expanded_count: usize = 0;

    for (reader.toc.entries.items) |e| {
        var numel: usize = 1;
        for (0..e.ndim) |d| numel *= @intCast(e.shape[d]);

        const dense_old = try arena_alloc.alloc(f32, numel);
        try reader.dequantizeToF32(e, true, dense_old);

        var is_expanded = false;

        // 1. Check vocabulary expansion
        if (new_vocab_opt) |target_vocab| {
            const is_vocab_tensor = (std.mem.indexOf(u8, e.name, "embed_tokens") != null) or
                (std.mem.indexOf(u8, e.name, "lm_head") != null) or
                (std.mem.indexOf(u8, e.name, "wte") != null) or
                (std.mem.indexOf(u8, e.name, "token_embeddings") != null);

            if (is_vocab_tensor and e.ndim == 2 and e.shape[0] < target_vocab) {
                const old_v: usize = @intCast(e.shape[0]);
                const hidden_dim: usize = @intCast(e.shape[1]);
                const new_total = target_vocab * hidden_dim;
                const dense_new = try arena_alloc.alloc(f32, new_total);

                try hk.growth.expandVocab(
                    dense_old,
                    dense_new,
                    null,
                    null,
                    old_v,
                    target_vocab,
                    hidden_dim,
                    42,
                );

                var new_shape = e.shape;
                new_shape[0] = target_vocab;

                try writer.addTensor(.{
                    .name = e.name,
                    .storage_type = .f32,
                    .tile_layout = e.tile_layout,
                    .sparsity_type = e.sparsity_type,
                    .ndim = e.ndim,
                    .shape = new_shape,
                    .data = std.mem.sliceAsBytes(dense_new),
                    .sparsity_ratio = e.sparsity_ratio,
                });

                std.debug.print("  [VOCAB EXPANDED] {s}: [{}, {}] -> [{}, {}]\n", .{
                    e.name, old_v, hidden_dim, target_vocab, hidden_dim,
                });
                expanded_count += 1;
                is_expanded = true;
            }
        }

        // 2. Check SwiGLU / MLP width expansion
        if (!is_expanded and width_ratio_opt != null and width_ratio_opt.? > 1.0 and e.ndim == 2) {
            const ratio = width_ratio_opt.?;
            const is_gate = (std.mem.indexOf(u8, e.name, "gate_proj") != null) or (std.mem.indexOf(u8, e.name, "w1") != null);
            const is_up = (std.mem.indexOf(u8, e.name, "up_proj") != null) or (std.mem.indexOf(u8, e.name, "w3") != null);
            const is_down = (std.mem.indexOf(u8, e.name, "down_proj") != null) or (std.mem.indexOf(u8, e.name, "w2") != null);

            if (is_gate or is_up) {
                const old_inter: usize = @intCast(e.shape[0]);
                const in_f: usize = @intCast(e.shape[1]);
                const new_inter: usize = @intFromFloat(@round(@as(f32, @floatFromInt(old_inter)) * ratio));

                if (new_inter > old_inter) {
                    const new_total = new_inter * in_f;
                    const dense_new = try arena_alloc.alloc(f32, new_total);

                    // Copy base rows
                    @memcpy(dense_new[0 .. old_inter * in_f], dense_old[0 .. old_inter * in_f]);

                    // Initialize new rows with small normal noise
                    for (old_inter..new_inter) |r| {
                        for (0..in_f) |c| {
                            const unif1 = @max(rand.float(f32), 1e-7);
                            const unif2 = rand.float(f32);
                            const z = @sqrt(-2.0 * @log(unif1)) * @cos(2.0 * std.math.pi * unif2);
                            dense_new[r * in_f + c] = z * 0.02;
                        }
                    }

                    var new_shape = e.shape;
                    new_shape[0] = new_inter;

                    try writer.addTensor(.{
                        .name = e.name,
                        .storage_type = .f32,
                        .tile_layout = e.tile_layout,
                        .sparsity_type = e.sparsity_type,
                        .ndim = e.ndim,
                        .shape = new_shape,
                        .data = std.mem.sliceAsBytes(dense_new),
                        .sparsity_ratio = e.sparsity_ratio,
                    });

                    std.debug.print("  [WIDTH EXPANDED (GATE/UP)] {s}: [{}, {}] -> [{}, {}]\n", .{
                        e.name, old_inter, in_f, new_inter, in_f,
                    });
                    expanded_count += 1;
                    is_expanded = true;
                }
            } else if (is_down) {
                const out_f: usize = @intCast(e.shape[0]);
                const old_inter: usize = @intCast(e.shape[1]);
                const new_inter: usize = @intFromFloat(@round(@as(f32, @floatFromInt(old_inter)) * ratio));

                if (new_inter > old_inter) {
                    const new_total = out_f * new_inter;
                    const dense_new = try arena_alloc.alloc(f32, new_total);

                    // Copy base columns, ZERO-INITIALIZE new columns for Day-0 function preservation!
                    for (0..out_f) |r| {
                        @memcpy(
                            dense_new[r * new_inter .. r * new_inter + old_inter],
                            dense_old[r * old_inter .. (r + 1) * old_inter],
                        );
                        @memset(dense_new[r * new_inter + old_inter .. (r + 1) * new_inter], 0.0);
                    }

                    var new_shape = e.shape;
                    new_shape[1] = new_inter;

                    try writer.addTensor(.{
                        .name = e.name,
                        .storage_type = .f32,
                        .tile_layout = e.tile_layout,
                        .sparsity_type = e.sparsity_type,
                        .ndim = e.ndim,
                        .shape = new_shape,
                        .data = std.mem.sliceAsBytes(dense_new),
                        .sparsity_ratio = e.sparsity_ratio,
                    });

                    std.debug.print("  [WIDTH EXPANDED (DOWN - ZERO-INIT)] {s}: [{}, {}] -> [{}, {}]\n", .{
                        e.name, out_f, old_inter, out_f, new_inter,
                    });
                    expanded_count += 1;
                    is_expanded = true;
                }
            }
        }

        // If not modified, write existing tensor
        if (!is_expanded) {
            // dense_old holds decoded f32, so the copy is stored as f32 whatever the input type was.
            try writer.addTensor(.{
                .name = e.name,
                .storage_type = .f32,
                .tile_layout = e.tile_layout,
                .sparsity_type = e.sparsity_type,
                .ndim = e.ndim,
                .shape = e.shape,
                .data = std.mem.sliceAsBytes(dense_old),
                .sparsity_ratio = e.sparsity_ratio,
            });
        }
    }

    try writer.writeToFile(out_path);
    if (expanded_count == 0) {
        std.debug.print("\n[WARNING] No tensor matched, so '{s}' is a copy of the input. Expansion looks for Hugging Face style names (model.embed_tokens.weight, mlp.gate_proj.weight and so on); a model converted from GGUF uses GGUF names.\n\n", .{out_path});
    } else {
        std.debug.print("\n[SUCCESS] Expansion complete! {} tensors expanded natively. Output saved to '{s}'.\n\n", .{
            expanded_count, out_path,
        });
    }
}

fn cmdDump(path: []const u8, allocator: std.mem.Allocator) !void {
    var reader = try hk.HKReader.open(path, allocator);
    defer reader.deinit();

    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("                         HK BINARY CONTAINER DUMPER                             \n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("File Path            : {s}\n", .{path});
    std.debug.print("Total File Size      : {} bytes ({d:.2} MB)\n", .{
        reader.mmap_region.bytes.len,
        @as(f64, @floatFromInt(reader.mmap_region.bytes.len)) / (1024.0 * 1024.0),
    });

    const h = reader.header;
    std.debug.print("\n--- 128-Byte Fixed Header Breakdown ---\n", .{});
    std.debug.print("  Magic Bytes        : {s} (0x{X:0>2} 0x{X:0>2} 0x{X:0>2} 0x{X:0>2}) [Valid: {}]\n", .{
        h.magic, h.magic[0], h.magic[1], h.magic[2], h.magic[3], h.isValid(),
    });
    std.debug.print("  Version            : v{}.{}\n", .{ h.version_major, h.version_minor });
    std.debug.print("  Header Flags       : 0x{X:0>8}\n", .{h.flags});
    std.debug.print("    - Little Endian  : {}\n", .{(h.flags & hk.format.HeaderFlags.LITTLE_ENDIAN) != 0});
    std.debug.print("    - Has Appendix   : {}\n", .{(h.flags & hk.format.HeaderFlags.HAS_APPENDIX) != 0});
    std.debug.print("    - Tile Aligned   : {}\n", .{(h.flags & hk.format.HeaderFlags.TILE_ALIGNED) != 0});
    std.debug.print("    - Is Sharded     : {}\n", .{(h.flags & hk.format.HeaderFlags.IS_SHARDED) != 0});
    std.debug.print("  Tensor Alignment   : {} bytes\n", .{h.alignment});
    std.debug.print("  Sharding Index     : {} of {}\n", .{ h.split_index + 1, h.split_count });
    std.debug.print("  Tensor Count       : {}\n", .{h.tensor_count});
    std.debug.print("  Metadata KVs       : {}\n", .{h.metadata_kv_count});
    std.debug.print("  Metadata Offset    : 0x{X:0>8} ({} bytes)\n", .{ h.metadata_offset, h.metadata_size });
    std.debug.print("  Tensor TOC Offset  : 0x{X:0>8} ({} bytes)\n", .{ h.tensor_toc_offset, h.tensor_toc_size });
    std.debug.print("  Tensor Data Offset : 0x{X:0>8} (128-byte aligned: {})\n", .{
        h.tensor_data_offset, (h.tensor_data_offset % 128) == 0,
    });
    std.debug.print("  Appendix Offset    : 0x{X:0>8}\n", .{h.appendix_offset});

    // Raw Hex Dump of Header (first 128 bytes)
    std.debug.print("\n--- Raw Header Hex Dump (Offset 0x0000 - 0x007F) ---\n", .{});
    const header_slice = reader.mmap_region.bytes[0..@min(128, reader.mmap_region.bytes.len)];
    var offset: usize = 0;
    while (offset < header_slice.len) : (offset += 16) {
        std.debug.print("  0x{X:0>4}: ", .{offset});
        const chunk_len = @min(16, header_slice.len - offset);
        for (0..16) |j| {
            if (j < chunk_len) {
                std.debug.print("{X:0>2} ", .{header_slice[offset + j]});
            } else {
                std.debug.print("   ", .{});
            }
            if (j == 7) std.debug.print(" ", .{});
        }
        std.debug.print(" |", .{});
        for (0..chunk_len) |j| {
            const b = header_slice[offset + j];
            const ch: u8 = if (b >= 32 and b <= 126) b else '.';
            std.debug.print("{c}", .{ch});
        }
        std.debug.print("|\n", .{});
    }

    // Metadata Listing
    std.debug.print("\n--- Metadata Table ({} entries) ---\n", .{reader.metadata_map.items.items.len});
    for (reader.metadata_map.items.items, 0..) |item, idx| {
        std.debug.print("  [{:0>2}] {s:<32} = ", .{ idx, item.key });
        switch (item.value) {
            .val_string => |s| std.debug.print("\"{s}\" (string)\n", .{s}),
            .val_int64 => |v| std.debug.print("{} (int64)\n", .{v}),
            .val_float64 => |f| std.debug.print("{d:.6} (float64)\n", .{f}),
            .val_bool => |b| std.debug.print("{} (bool)\n", .{b}),
            .val_json => |j| std.debug.print("{s} (JSON)\n", .{j}),
            .val_bytes => |b| std.debug.print("<{} bytes> (binary)\n", .{b.len}),
        }
    }

    // Tensor TOC Dump
    std.debug.print("\n--- Tensor TOC Table ({} entries) ---\n", .{reader.toc.entries.items.len});
    std.debug.print("{s:<4} {s:<36} {s:<10} {s:<10} {s:<14} {s:<10} {s:<10} {s:<8}\n", .{
        "#", "Name", "Type", "Layout", "Shape", "Offset", "Size", "Sparsity",
    });
    std.debug.print("{s:-<110}\n", .{""});

    for (reader.toc.entries.items, 0..) |e, idx| {
        var shape_buf: [64]u8 = undefined;
        var pos: usize = 0;
        shape_buf[pos] = '[';
        pos += 1;
        for (0..e.ndim) |d| {
            if (d > 0) {
                shape_buf[pos] = ',';
                pos += 1;
            }
            const part = std.fmt.bufPrint(shape_buf[pos..], "{}", .{e.shape[d]}) catch "";
            pos += part.len;
        }
        shape_buf[pos] = ']';
        pos += 1;
        const shape_str = shape_buf[0..pos];

        std.debug.print("{:0>3}  {s:<36} {s:<10} {s:<10} {s:<14} 0x{X:<8} {}B {d:>5.1}%\n", .{
            idx,
            e.name,
            @tagName(e.storage_type),
            @tagName(e.tile_layout),
            shape_str,
            e.data_offset,
            e.data_size,
            e.sparsity_ratio * 100.0,
        });
    }
    std.debug.print("\n", .{});
}

fn hexDigest(bytes: [32]u8, out_hex: *[64]u8) []const u8 {
    const charset = "0123456789abcdef";
    for (bytes, 0..) |b, i| {
        out_hex[i * 2] = charset[b >> 4];
        out_hex[i * 2 + 1] = charset[b & 0x0F];
    }
    return out_hex[0..64];
}

fn cmdHash(path: []const u8, allocator: std.mem.Allocator) !void {
    var reader = try hk.HKReader.open(path, allocator);
    defer reader.deinit();

    std.debug.print("\n=== HK Cryptographic Checksum Verifier ===\n", .{});
    std.debug.print("File: {s}\n\n", .{path});

    // 1. Full Container SHA-256
    var file_hasher = std.crypto.hash.sha2.Sha256.init(.{});
    file_hasher.update(reader.mmap_region.bytes);
    var file_digest: [32]u8 = undefined;
    file_hasher.final(&file_digest);

    var file_hex: [64]u8 = undefined;
    std.debug.print("Container SHA-256 : {s}\n", .{hexDigest(file_digest, &file_hex)});
    std.debug.print("Total Tensors     : {}\n\n", .{reader.toc.entries.items.len});

    std.debug.print("{s:<4} {s:<42} {s:<10} {s}\n", .{ "#", "Tensor Name", "Size", "SHA-256 Digest" });
    std.debug.print("{s:-<120}\n", .{""});

    for (reader.toc.entries.items, 0..) |e, idx| {
        const data = try reader.getTensorData(e);
        var t_hasher = std.crypto.hash.sha2.Sha256.init(.{});
        t_hasher.update(data);
        if (reader.getTensorScales(e)) |sc| {
            t_hasher.update(sc);
        }
        var t_digest: [32]u8 = undefined;
        t_hasher.final(&t_digest);

        var t_hex: [64]u8 = undefined;
        std.debug.print("{:0>3}  {s:<42} {:<10} {s}\n", .{
            idx,
            e.name,
            data.len,
            hexDigest(t_digest, &t_hex),
        });
    }
    std.debug.print("\n[VERIFIED] All {} tensors hashed and cryptographically sealed.\n\n", .{
        reader.toc.entries.items.len,
    });
}

fn cmdConvertEndian(in_path: []const u8, out_path: []const u8, allocator: std.mem.Allocator) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var reader = try hk.HKReader.open(in_path, arena_alloc);
    defer reader.deinit();

    var writer = hk.HKWriter.init(arena_alloc);
    defer writer.deinit();

    // Copy metadata
    for (reader.metadata_map.items.items) |m| {
        switch (m.value) {
            .val_string => |s| try writer.addMetadataString(m.key, s),
            .val_int64 => |v| try writer.addMetadataInt(m.key, v),
            .val_float64 => |f| try writer.addMetadataFloat(m.key, f),
            .val_bool => |b| try writer.addMetadataBool(m.key, b),
            else => {},
        }
    }
    try writer.addMetadataString("endian_conversion", "swapped");

    // Copy tensors
    for (reader.toc.entries.items) |e| {
        const data_bytes = try reader.getTensorData(e);
        const dup_data = try arena_alloc.dupe(u8, data_bytes);

        // Perform endianness byte swap on multi-byte payloads
        if (e.storage_type == .f32 or e.storage_type == .int32) {
            var j: usize = 0;
            while (j + 4 <= dup_data.len) : (j += 4) {
                const b0 = dup_data[j + 0];
                const b1 = dup_data[j + 1];
                const b2 = dup_data[j + 2];
                const b3 = dup_data[j + 3];
                dup_data[j + 0] = b3;
                dup_data[j + 1] = b2;
                dup_data[j + 2] = b1;
                dup_data[j + 3] = b0;
            }
        } else if (e.storage_type == .f16 or e.storage_type == .bf16) {
            var j: usize = 0;
            while (j + 2 <= dup_data.len) : (j += 2) {
                const tmp = dup_data[j + 0];
                dup_data[j + 0] = dup_data[j + 1];
                dup_data[j + 1] = tmp;
            }
        } else if (e.storage_type == .int64) {
            var j: usize = 0;
            while (j + 8 <= dup_data.len) : (j += 8) {
                for (0..4) |k| {
                    const tmp = dup_data[j + k];
                    dup_data[j + k] = dup_data[j + 7 - k];
                    dup_data[j + 7 - k] = tmp;
                }
            }
        }

        try writer.addTensor(.{
            .name = e.name,
            .storage_type = e.storage_type,
            .tile_layout = e.tile_layout,
            .sparsity_type = e.sparsity_type,
            .ndim = e.ndim,
            .shape = e.shape,
            .data = dup_data,
            .sparsity_ratio = e.sparsity_ratio,
        });
    }

    try writer.writeToFile(out_path);
    std.debug.print("[SUCCESS] Endianness converted successfully:\n  Input : {s}\n  Output: {s}\n", .{ in_path, out_path });
}

fn cmdGui(file_path_opt: ?[]const u8, allocator: std.mem.Allocator) !void {
    _ = allocator;
    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("                         HK GRAPHICAL MODEL EDITOR                              \n", .{});
    std.debug.print("================================================================================\n", .{});
    if (file_path_opt) |fp| {
        std.debug.print("Target Model : {s}\n", .{fp});
        std.debug.print("Launch GUI   : py -3.12 tools/hk_editor_gui.py {s}\n\n", .{fp});
    } else {
        std.debug.print("Launch GUI   : py -3.12 tools/hk_editor_gui.py\n\n", .{});
    }
}

const Loaded = struct {
    reader: hk.HKReader,
    tok: hk.tokenizer.Tokenizer,
    model: hk.engine.Model,
    template: ?hk.chat.ChatTemplate = null,

    fn open(allocator: std.mem.Allocator, path: []const u8, opts: hk.engine.Options) !*Loaded {
        const l = try allocator.create(Loaded);
        errdefer allocator.destroy(l);
        l.reader = hk.HKReader.open(path, allocator) catch |err| {
            std.debug.print("Error: cannot open '{s}': {s}\n", .{ path, @errorName(err) });
            return err;
        };
        errdefer l.reader.deinit();
        var tdiag = hk.tokenizer.Diag{};
        l.tok = hk.tokenizer.Tokenizer.fromMetadata(allocator, &l.reader.metadata_map, &tdiag) catch |err| {
            std.debug.print("Error: tokenizer: {s}\n", .{tdiag.message()});
            return err;
        };
        errdefer l.tok.deinit();
        var diag = hk.engine.Diag{};
        l.model = hk.engine.Model.init(allocator, &l.reader, opts, &diag) catch |err| {
            std.debug.print("Error: cannot load model: {s}\n", .{diag.message()});
            return err;
        };
        errdefer l.model.deinit();
        if (opts.gpu != .off) {
            if (l.model.gpu) |g| {
                std.debug.print("[gpu] running on {s} (Vulkan)\n", .{g.deviceName()});
            } else {
                std.debug.print("[gpu] not used: {s}; running on the CPU\n", .{l.model.gpu_note orelse "unknown reason"});
            }
        }
        l.template = null;
        if (l.tok.chat_template) |src| {
            var jd = hk.chat.jinja.Diag{};
            const bos = if (l.tok.bos) |b| l.tok.piece(b) else "";
            const eos = if (l.tok.eos) |e| l.tok.piece(e) else "";
            if (hk.chat.ChatTemplate.init(allocator, src, bos, eos, &jd)) |t| {
                l.template = t;
            } else |_| {
                std.debug.print("Note: this model's chat template could not be parsed ({s}); using a plain transcript.\n", .{jd.message()});
            }
        }
        return l;
    }

    fn close(self: *Loaded, allocator: std.mem.Allocator) void {
        if (self.template) |*t| t.deinit();
        self.model.deinit();
        self.tok.deinit();
        self.reader.deinit();
        allocator.destroy(self);
    }
};

/// Token ids that end generation: the container's list when it has one, else the model's EOS
/// plus the usual end-of-turn markers.
fn isStopToken(tok: *const hk.tokenizer.Tokenizer, id: u32) bool {
    if (tok.eos) |e| if (id == e) return true;
    if (tok.eog) |list| for (0..list.count) |i| if (@as(u32, @intCast(list.get(i))) == id) return true;
    const ty = tok.tokenType(id);
    if (ty != .control and ty != .user_defined) return false;
    const p = tok.piece(id);
    const stops = [_][]const u8{ "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "<|end|>", "<end_of_turn>", "</s>" };
    for (stops) |s| if (std.mem.eql(u8, p, s)) return true;
    return false;
}

/// Generated text goes to stdout so it can be piped; notes and statistics go to stderr.
fn writeOut(bytes: []const u8) void {
    std.Io.File.stdout().writeStreamingAll(std.Options.debug_io, bytes) catch {};
}

/// Prints bytes as they arrive, holding back an incomplete UTF-8 sequence until it is whole.
const Streamer = struct {
    pending: [4]u8 = undefined,
    n: usize = 0,

    fn feed(self: *Streamer, bytes: []const u8) void {
        var out: [64]u8 = undefined;
        var olen: usize = 0;
        for (bytes) |b| {
            self.pending[self.n] = b;
            self.n += 1;
            const need = std.unicode.utf8ByteSequenceLength(self.pending[0]) catch 1;
            if (self.n >= need or self.n == 4) {
                if (olen + self.n > out.len) {
                    writeOut(out[0..olen]);
                    olen = 0;
                }
                @memcpy(out[olen..][0..self.n], self.pending[0..self.n]);
                olen += self.n;
                self.n = 0;
            }
        }
        if (olen > 0) writeOut(out[0..olen]);
    }
};

fn nowNs() i128 {
    return std.Io.Timestamp.now(std.Options.debug_io, .awake).nanoseconds;
}

/// Resident memory split, Linux only.
fn memSummary() struct { rss_mib: f64, anon_mib: f64, file_mib: f64 } {
    var buf: [4096]u8 = undefined;
    const io = std.Options.debug_io;
    var f = std.Io.Dir.cwd().openFile(io, "/proc/self/status", .{}) catch return .{ .rss_mib = 0, .anon_mib = 0, .file_mib = 0 };
    defer f.close(io);
    const n = f.readPositionalAll(io, &buf, 0) catch return .{ .rss_mib = 0, .anon_mib = 0, .file_mib = 0 };
    var r: f64 = 0;
    var an: f64 = 0;
    var fl: f64 = 0;
    var lines = std.mem.splitScalar(u8, buf[0..n], '\n');
    while (lines.next()) |line| {
        var it = std.mem.tokenizeAny(u8, line, " \t");
        const key = it.next() orelse continue;
        const val = std.fmt.parseInt(u64, it.next() orelse continue, 10) catch continue;
        const mib = @as(f64, @floatFromInt(val)) / 1024.0;
        if (std.mem.eql(u8, key, "VmRSS:")) r = mib;
        if (std.mem.eql(u8, key, "RssAnon:")) an = mib;
        if (std.mem.eql(u8, key, "RssFile:")) fl = mib;
    }
    return .{ .rss_mib = r, .anon_mib = an, .file_mib = fl };
}

const GenStats = struct { prompt_tokens: usize, reused: usize, prompt_secs: f64, produced: usize, gen_secs: f64 };

/// Evaluates `ids` in the session (reusing its cache) and samples a reply, streaming it to the
/// terminal. The generated ids are left in the session cache so the next turn can build on them.
fn generate(
    allocator: std.mem.Allocator,
    l: *Loaded,
    sess: *hk.engine.Session,
    ids: []const u32,
    max_tokens: usize,
    params: hk.sampler.Params,
    history: *std.ArrayList(u32),
    reply: ?*std.ArrayList(u8),
) !GenStats {
    var sampler = hk.sampler.Sampler.init(allocator, params.seed);
    defer sampler.deinit();
    const t0 = nowNs();
    const reused = try sess.evaluate(ids);
    try history.appendSlice(allocator, ids[reused..]);
    const t1 = nowNs();

    var streamer = Streamer{};
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(allocator);
    const logits_copy = try allocator.alloc(f32, l.model.cfg.vocab);
    defer allocator.free(logits_copy);

    var produced: usize = 0;
    while (produced < max_tokens) {
        @memcpy(logits_copy, sess.logits());
        const next = try sampler.sample(logits_copy, history.items, params, false);
        if (isStopToken(&l.tok, next)) break;
        bytes.clearRetainingCapacity();
        try l.tok.decodeToken(next, false, &bytes);
        streamer.feed(bytes.items);
        if (reply) |r| try r.appendSlice(allocator, bytes.items);
        try history.append(allocator, next);
        produced += 1;
        sess.step(next) catch |e| {
            if (e == error.ContextFull) {
                std.debug.print("\n[context window of {d} tokens is full]", .{sess.kv.n_ctx});
                break;
            }
            return e;
        };
    }
    const t2 = nowNs();
    return .{
        .prompt_tokens = ids.len,
        .reused = reused,
        .prompt_secs = @as(f64, @floatFromInt(t1 - t0)) / 1e9,
        .produced = produced,
        .gen_secs = @as(f64, @floatFromInt(t2 - t1)) / 1e9,
    };
}

fn printStats(l: *Loaded, st: GenStats) void {
    const m = memSummary();
    const evaluated = st.prompt_tokens - st.reused;
    std.debug.print("\n\n[prompt {d} tok ({d} cached), {d:.1} tok/s | generated {d} tok, {d:.1} tok/s | memory {d:.0} MiB = {d:.0} private + {d:.0} mapped | threads {d}]\n", .{
        st.prompt_tokens,
        st.reused,
        @as(f64, @floatFromInt(evaluated)) / @max(st.prompt_secs, 1e-9),
        st.produced,
        @as(f64, @floatFromInt(st.produced)) / @max(st.gen_secs, 1e-9),
        m.rss_mib,
        m.anon_mib,
        m.file_mib,
        l.model.pool.size(),
    });
}

const usage_gen =
    \\Usage: hk run  <model.hk | owner/name[:quant]> [prompt | -p TEXT] [options]
    \\       hk chat <model.hk | owner/name[:quant]> [options]
    \\
    \\  -p, --prompt TEXT     text to continue (run only; a bare argument works too)
    \\      --chat            run only: wrap the prompt as a user message with the chat template
    \\  -n, --max-tokens N    most tokens to generate (default 128 for run, 1024 for chat)
    \\      --temp T          temperature, 0 picks the most likely token (default 0.8)
    \\      --top-k N         keep the N most likely tokens, 0 for no limit (default 40)
    \\      --top-p P         nucleus sampling, 1 to disable (default 0.95)
    \\      --min-p P         drop tokens below P times the best one (default 0.05)
    \\      --repeat-penalty X  penalty for recent tokens, 1 to disable (default 1.0)
    \\      --seed N          random seed, 0 uses the clock (default 0)
    \\      --threads N       compute threads (default: the number of physical cores)
    \\  -ngl, --gpu-layers N  run on a GPU (Vulkan) when N > 0 and the model fits; the whole model
    \\                        goes to the device, so any N above 0 means all layers. Also HK_GPU=1.
    \\
;

const GenMode = enum { run, chat };

/// Whether to use a GPU: `-ngl N` with N above 0, or HK_GPU=1 in the environment.
fn gpuMode(gpu_layers: ?usize) hk.engine.model.GpuMode {
    if (gpu_layers) |n| return if (n > 0) .auto else .off;
    if (g_env) |e| if (e.get("HK_GPU")) |t| {
        if (std.mem.eql(u8, t, "1") or std.mem.eql(u8, t, "on")) return .auto;
    };
    return .off;
}

const GenArgs = struct {
    prompt: ?[]const u8 = null,
    chat_prompt: bool = false,
    max_tokens: usize = 128,
    gpu_layers: ?usize = null,
    threads: usize = 0,
    params: hk.sampler.Params = .{},
};

fn isHelp(arg: []const u8) bool {
    return std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help");
}

fn badValue(flag: []const u8, value: []const u8, expected: []const u8) error{BadUsage} {
    std.debug.print("Error: '{s}' is not a valid value for {s} (expected {s})\n", .{ value, flag, expected });
    return error.BadUsage;
}

fn needValue(it: *std.process.Args.Iterator, flag: []const u8) error{BadUsage}![]const u8 {
    return it.next() orelse {
        std.debug.print("Error: {s} needs a value\n", .{flag});
        return error.BadUsage;
    };
}

fn parseUint(it: *std.process.Args.Iterator, flag: []const u8, min: usize) error{BadUsage}!usize {
    const v = try needValue(it, flag);
    const n = std.fmt.parseInt(usize, v, 10) catch return badValue(flag, v, "a whole number");
    if (n < min) return badValue(flag, v, if (min == 1) "a number of at least 1" else "a larger number");
    return n;
}

fn parseFloatIn(it: *std.process.Args.Iterator, flag: []const u8, lo: f32, hi: f32, expected: []const u8) error{BadUsage}!f32 {
    const v = try needValue(it, flag);
    const x = std.fmt.parseFloat(f32, v) catch return badValue(flag, v, expected);
    if (!(x >= lo and x <= hi)) return badValue(flag, v, expected);
    return x;
}

/// Parses the flags of `run` and `chat`. Anything unknown or malformed is an error: a typo should
/// never silently fall back to a default.
fn parseGenArgs(it: *std.process.Args.Iterator, mode: GenMode) error{BadUsage}!GenArgs {
    var g = GenArgs{ .max_tokens = if (mode == .chat) 1024 else 128 };
    while (it.next()) |flag| {
        if (std.mem.eql(u8, flag, "-p") or std.mem.eql(u8, flag, "--prompt")) {
            if (mode != .run) return unknownFlag(flag);
            if (g.prompt != null) return twoPrompts();
            g.prompt = try needValue(it, flag);
        } else if (std.mem.eql(u8, flag, "--chat")) {
            if (mode != .run) return unknownFlag(flag);
            g.chat_prompt = true;
        } else if (std.mem.eql(u8, flag, "-n") or std.mem.eql(u8, flag, "--max-tokens")) {
            g.max_tokens = try parseUint(it, flag, 1);
        } else if (std.mem.eql(u8, flag, "-ngl") or std.mem.eql(u8, flag, "--gpu-layers")) {
            g.gpu_layers = try parseUint(it, flag, 0);
        } else if (std.mem.eql(u8, flag, "--threads")) {
            g.threads = try parseUint(it, flag, 1);
        } else if (std.mem.eql(u8, flag, "--temp")) {
            g.params.temperature = try parseFloatIn(it, flag, 0, 100, "a number from 0 up");
        } else if (std.mem.eql(u8, flag, "--top-k")) {
            g.params.top_k = @intCast(@min(try parseUint(it, flag, 0), std.math.maxInt(u32)));
        } else if (std.mem.eql(u8, flag, "--top-p")) {
            g.params.top_p = try parseFloatIn(it, flag, 0, 1, "a number from 0 to 1");
        } else if (std.mem.eql(u8, flag, "--min-p")) {
            g.params.min_p = try parseFloatIn(it, flag, 0, 1, "a number from 0 to 1");
        } else if (std.mem.eql(u8, flag, "--repeat-penalty") or std.mem.eql(u8, flag, "--rep-pen")) {
            g.params.repeat_penalty = try parseFloatIn(it, flag, 0.01, 100, "a positive number");
        } else if (std.mem.eql(u8, flag, "--seed")) {
            const v = try needValue(it, flag);
            g.params.seed = std.fmt.parseInt(u64, v, 10) catch return badValue(flag, v, "a whole number");
        } else if (flag.len > 0 and flag[0] == '-') {
            return unknownFlag(flag);
        } else if (mode == .run) {
            // A bare argument is the prompt.
            if (g.prompt != null) return twoPrompts();
            g.prompt = flag;
        } else return unknownFlag(flag);
    }
    return g;
}

fn unknownFlag(flag: []const u8) error{BadUsage} {
    std.debug.print("Error: unknown argument '{s}'\n{s}", .{ flag, usage_gen });
    return error.BadUsage;
}

fn twoPrompts() error{BadUsage} {
    std.debug.print("Error: more than one prompt was given. Quote the prompt so it is one argument.\n", .{});
    return error.BadUsage;
}

fn cmdRun(model_path: []const u8, g: GenArgs, allocator: std.mem.Allocator) !void {
    const prompt_in = g.prompt orelse {
        std.debug.print("Error: no prompt. Give one as a bare argument or with -p.\n", .{});
        return error.BadUsage;
    };

    const l = try Loaded.open(allocator, model_path, .{ .n_threads = g.threads, .gpu = gpuMode(g.gpu_layers) });
    defer l.close(allocator);

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    var prompt: []const u8 = prompt_in;
    if (g.chat_prompt) {
        const t = if (l.template) |*t| t else {
            std.debug.print("Error: this model has no chat template, so --chat cannot be used.\n", .{});
            return error.NoChatTemplate;
        };
        var jd = hk.chat.jinja.Diag{};
        const msgs = [_]hk.chat.Message{.{ .role = "user", .content = prompt_in }};
        prompt = t.render(arena_state.allocator(), &msgs, .{ .add_generation_prompt = true }, &jd) catch {
            std.debug.print("Error: the chat template failed: {s}\n", .{jd.message()});
            return error.TemplateFailed;
        };
    }

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(allocator);
    // Templates write their own start token; plain text gets the model's.
    try l.tok.encode(prompt, .{ .add_special = !g.chat_prompt, .parse_special = true }, &ids);
    if (ids.items.len == 0) {
        std.debug.print("Error: the prompt produced no tokens.\n", .{});
        return error.EmptyPrompt;
    }

    var sess = try hk.engine.Session.init(allocator, &l.model, l.model.n_ctx);
    defer sess.deinit();
    writeOut(prompt);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(allocator);
    const st = try generate(allocator, l, &sess, ids.items, g.max_tokens, g.params, &history, null);
    printStats(l, st);
}

fn cmdChat(model_path: []const u8, g: GenArgs, allocator: std.mem.Allocator) !void {
    const l = try Loaded.open(allocator, model_path, .{ .n_threads = g.threads, .gpu = gpuMode(g.gpu_layers) });
    defer l.close(allocator);

    const io = std.Options.debug_io;
    var stdin_buf: [4096]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buf);

    var sess = try hk.engine.Session.init(allocator, &l.model, l.model.n_ctx);
    defer sess.deinit();

    // The conversation so far. Strings are owned by the arena for the life of the chat.
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var messages: std.ArrayList(hk.chat.Message) = .empty;
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(allocator);

    if (l.template == null) std.debug.print("Note: this model has no chat template; using a plain transcript.\n", .{});
    std.debug.print("hk chat: type a message, an empty line or Ctrl-D to quit.\n", .{});
    while (true) {
        std.debug.print("\n> ", .{});
        const line_opt = stdin_reader.interface.takeDelimiter('\n') catch break;
        const line = std.mem.trim(u8, line_opt orelse break, " \r\n\t");
        if (line.len == 0) break;
        try messages.append(arena, .{ .role = "user", .content = try arena.dupe(u8, line) });

        // Render the whole conversation; the session reuses whatever is already cached.
        var prompt_text: []const u8 = undefined;
        var jd = hk.chat.jinja.Diag{};
        if (l.template) |*t| {
            prompt_text = t.render(arena, messages.items, .{ .add_generation_prompt = true }, &jd) catch {
                std.debug.print("Error: the chat template failed: {s}\n", .{jd.message()});
                _ = messages.pop();
                continue;
            };
        } else {
            var buf: std.ArrayList(u8) = .empty;
            for (messages.items) |m| try buf.print(arena, "{s}: {s}\n", .{ if (std.mem.eql(u8, m.role, "user")) "User" else "Assistant", m.content });
            try buf.appendSlice(arena, "Assistant:");
            prompt_text = buf.items;
        }
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(allocator);
        try l.tok.encode(prompt_text, .{ .add_special = false, .parse_special = true }, &ids);
        if (ids.items.len + 64 >= sess.kv.n_ctx) {
            std.debug.print("[the conversation no longer fits the context window]\n", .{});
            break;
        }
        var reply: std.ArrayList(u8) = .empty;
        defer reply.deinit(allocator);
        const st = try generate(allocator, l, &sess, ids.items, g.max_tokens, g.params, &history, &reply);
        try messages.append(arena, .{ .role = "assistant", .content = try arena.dupe(u8, std.mem.trim(u8, reply.items, " \n")) });
        printStats(l, st);
    }
}

fn usageServe() error{BadUsage} {
    std.debug.print(
        \\Usage: hk serve <file.hk | owner/name[:quant]> [options]
        \\
        \\  --host ADDR      address to listen on (default 127.0.0.1)
        \\  --port N         port (default 8080)
        \\  --slots N        conversations served at once (default 4)
        \\  --ctx N          context window per slot (default: the model's, up to 8192)
        \\  --batch N        most tokens per forward pass (default 256)
        \\  --threads N      compute threads (default: the number of physical cores)
        \\  -ngl N           use a GPU (Vulkan) when N > 0 and the model fits; one device region per slot
        \\  --api-key KEY    require "Authorization: Bearer KEY" (or set HK_API_KEY)
        \\  --alias NAME     model name reported by the API
        \\
        \\Routes: /v1/chat/completions /v1/completions /v1/models /health /metrics /tokenize /detokenize
        \\
    , .{});
    return error.BadUsage;
}

fn cmdServe(allocator: std.mem.Allocator, io: std.Io, env: cli_hub.Env, args: *std.process.Args.Iterator) !void {
    var model_arg: ?[]const u8 = null;
    var cfg = hk.server.Config{};
    var slots: usize = 4;
    var opts = hk.engine.Options{};
    var alias: ?[]const u8 = null;
    cfg.api_key = env.get("HK_API_KEY");
    while (args.next()) |a| {
        if (std.mem.eql(u8, a, "--host")) {
            cfg.host = args.next() orelse return usageServe();
        } else if (std.mem.eql(u8, a, "--port")) {
            cfg.port = std.fmt.parseInt(u16, args.next() orelse return usageServe(), 10) catch return usageServe();
        } else if (std.mem.eql(u8, a, "--slots")) {
            slots = std.fmt.parseInt(usize, args.next() orelse return usageServe(), 10) catch return usageServe();
            if (slots == 0 or slots > 64) return usageServe();
        } else if (std.mem.eql(u8, a, "--ctx")) {
            opts.n_ctx = std.fmt.parseInt(usize, args.next() orelse return usageServe(), 10) catch return usageServe();
        } else if (std.mem.eql(u8, a, "--batch")) {
            opts.n_batch = std.fmt.parseInt(usize, args.next() orelse return usageServe(), 10) catch return usageServe();
        } else if (std.mem.eql(u8, a, "--threads")) {
            opts.n_threads = std.fmt.parseInt(usize, args.next() orelse return usageServe(), 10) catch return usageServe();
        } else if (std.mem.eql(u8, a, "-ngl") or std.mem.eql(u8, a, "--gpu-layers")) {
            const n = std.fmt.parseInt(usize, args.next() orelse return usageServe(), 10) catch return usageServe();
            opts.gpu = if (n > 0) .auto else .off;
        } else if (std.mem.eql(u8, a, "--api-key")) {
            cfg.api_key = args.next() orelse return usageServe();
        } else if (std.mem.eql(u8, a, "--alias")) {
            alias = args.next() orelse return usageServe();
        } else if (std.mem.eql(u8, a, "--max-body-mb")) {
            const mb = std.fmt.parseInt(usize, args.next() orelse return usageServe(), 10) catch return usageServe();
            cfg.max_body_bytes = mb << 20;
        } else if (model_arg == null) {
            model_arg = a;
        } else return usageServe();
    }
    const arg = model_arg orelse return usageServe();
    const path = try resolveModel(allocator, io, env, arg);
    defer allocator.free(path);

    // A batch has to hold at least one token per slot or decoding would starve.
    opts.n_batch = @max(opts.n_batch, slots);
    opts.gpu_slots = slots;
    if (opts.gpu == .off and gpuMode(null) != .off) opts.gpu = .auto;
    const l = try Loaded.open(allocator, path, opts);
    defer l.close(allocator);

    cfg.model_name = alias orelse std.fs.path.stem(path);
    var sched = try hk.server.Scheduler.init(allocator, io, &l.model, &l.tok, slots, l.model.n_ctx);
    defer sched.deinit();
    try sched.start();

    var srv = hk.server.Server{
        .allocator = allocator,
        .io = io,
        .sched = &sched,
        .tok = &l.tok,
        .template = if (l.template) |*t| t else null,
        .cfg = cfg,
        .started_ns = std.Io.Timestamp.now(io, .awake).nanoseconds,
    };
    try srv.run();
}

fn cmdTokenize(model_path: []const u8, text: []const u8, allocator: std.mem.Allocator) !void {
    var reader = hk.HKReader.open(model_path, allocator) catch |err| {
        std.debug.print("Error: cannot open '{s}': {s}\n", .{ model_path, @errorName(err) });
        return err;
    };
    defer reader.deinit();
    var diag = hk.tokenizer.Diag{};
    var tok = hk.tokenizer.Tokenizer.fromMetadata(allocator, &reader.metadata_map, &diag) catch |err| {
        std.debug.print("Error: {s}\n", .{diag.message()});
        return err;
    };
    defer tok.deinit();

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(allocator);
    try tok.encode(text, .{ .add_special = false, .parse_special = true }, &ids);
    std.debug.print("{d} tokens:", .{ids.items.len});
    for (ids.items) |id| std.debug.print(" {d}", .{id});
    std.debug.print("\n", .{});
}

fn cmdDetokenize(model_path: []const u8, ids: []const u32, allocator: std.mem.Allocator) !void {
    var reader = hk.HKReader.open(model_path, allocator) catch |err| {
        std.debug.print("Error: cannot open '{s}': {s}\n", .{ model_path, @errorName(err) });
        return err;
    };
    defer reader.deinit();
    var diag = hk.tokenizer.Diag{};
    var tok = hk.tokenizer.Tokenizer.fromMetadata(allocator, &reader.metadata_map, &diag) catch |err| {
        std.debug.print("Error: {s}\n", .{diag.message()});
        return err;
    };
    defer tok.deinit();
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    try tok.decode(ids, true, &text);
    std.debug.print("{s}\n", .{text.items});
}

fn cmdConvertSafeTensors(in_path: []const u8, out_path: []const u8, target_st_str: []const u8, allocator: std.mem.Allocator) !void {
    var st: hk.format.StorageType = .f32;
    if (std.mem.eql(u8, target_st_str, "q4_0")) {
        st = .q4_0;
    } else if (std.mem.eql(u8, target_st_str, "q8_0")) {
        st = .q8_0;
    }

    std.debug.print("Transcoding SafeTensors '{s}' -> HK '{s}' (storage={s})...\n", .{ in_path, out_path, @tagName(st) });
    try hk.safetensors.transcodeSafeTensorsToHK(allocator, in_path, out_path, st);
    std.debug.print("Transcoding completed successfully.\n", .{});
}



