//! Compares hk with llama.cpp on the same models, on this machine, back to back.
//!
//!   hk-compare --llama-bin DIR --pair model.hk:model.gguf [--pair ...] [options]
//!
//! Speed comes from each tool's own benchmark (hk-probe --bench and llama-bench) with the same
//! prompt and generation lengths and the same thread count. The two tools run alternately, so a
//! slow moment on the machine (another process, a hot CPU) hits both. Both are summarized the
//! same way, by the median of the repeats, with the range next to it. Memory comes from polling
//! /proc while a real generation runs: peak resident memory and peak private (anonymous) memory,
//! which is what the process costs on top of the model file the OS can drop from its cache.
//!
//! Options: --threads N (default: half the logical CPUs), --prompt-tokens N (512),
//! --gen-tokens N (64), --repeats N (5), --out FILE (also write the markdown there),
//! --hk-bin DIR (default: next to this program).
//!
//! Run it on an idle machine and look at the ranges before trusting a number.

const std = @import("std");
const builtin = @import("builtin");

const Allocator = std.mem.Allocator;

const Opts = struct {
    llama_bin: []const u8 = "",
    hk_bin: []const u8 = "",
    pairs: std.ArrayList([]const u8) = .empty,
    threads: usize = 0,
    prompt_tokens: usize = 512,
    gen_tokens: usize = 64,
    repeats: usize = 5,
    out: ?[]const u8 = null,
};

const Stat = struct {
    median: f64,
    min: f64,
    max: f64,

    fn of(xs: []f64) Stat {
        std.mem.sort(f64, xs, {}, std.sort.asc(f64));
        const n = xs.len;
        const med = if (n % 2 == 1) xs[n / 2] else (xs[n / 2 - 1] + xs[n / 2]) / 2.0;
        return .{ .median = med, .min = xs[0], .max = xs[n - 1] };
    }
};

const Mem = struct { rss_mib: f64 = 0, anon_mib: f64 = 0, ok: bool = false };

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var opts = Opts{};
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, arena);
    const self_path = it.next() orelse "hk-compare";
    while (it.next()) |a| {
        if (std.mem.eql(u8, a, "--llama-bin")) {
            opts.llama_bin = it.next() orelse return usage();
        } else if (std.mem.eql(u8, a, "--hk-bin")) {
            opts.hk_bin = it.next() orelse return usage();
        } else if (std.mem.eql(u8, a, "--pair")) {
            try opts.pairs.append(arena, it.next() orelse return usage());
        } else if (std.mem.eql(u8, a, "--threads")) {
            opts.threads = try std.fmt.parseInt(usize, it.next() orelse return usage(), 10);
        } else if (std.mem.eql(u8, a, "--prompt-tokens")) {
            opts.prompt_tokens = try std.fmt.parseInt(usize, it.next() orelse return usage(), 10);
        } else if (std.mem.eql(u8, a, "--gen-tokens")) {
            opts.gen_tokens = try std.fmt.parseInt(usize, it.next() orelse return usage(), 10);
        } else if (std.mem.eql(u8, a, "--repeats")) {
            opts.repeats = try std.fmt.parseInt(usize, it.next() orelse return usage(), 10);
        } else if (std.mem.eql(u8, a, "--out")) {
            opts.out = it.next() orelse return usage();
        } else return usage();
    }
    if (opts.llama_bin.len == 0 or opts.pairs.items.len == 0) return usage();
    if (opts.hk_bin.len == 0) opts.hk_bin = std.fs.path.dirname(self_path) orelse ".";
    if (opts.threads == 0) opts.threads = @max(1, (std.Thread.getCpuCount() catch 2) / 2);
    if (opts.repeats == 0) opts.repeats = 1;

    const hk = try std.fs.path.join(arena, &.{ opts.hk_bin, "hk" });
    const probe = try std.fs.path.join(arena, &.{ opts.hk_bin, "hk-probe" });
    const llama_bench = try std.fs.path.join(arena, &.{ opts.llama_bin, "llama-bench" });
    const llama_cli = try std.fs.path.join(arena, &.{ opts.llama_bin, "llama-completion" });

    var env = std.process.Environ.Map.init(arena);
    {
        var src = init.environ_map.iterator();
        while (src.next()) |e| try env.put(e.key_ptr.*, e.value_ptr.*);
    }
    try env.put("LD_LIBRARY_PATH", opts.llama_bin);
    var threads_buf: [16]u8 = undefined;
    try env.put("HK_THREADS", try std.fmt.bufPrint(&threads_buf, "{d}", .{opts.threads}));

    const Row = struct {
        name: []const u8,
        hk_pp: Stat,
        hk_tg: Stat,
        ll_pp: Stat,
        ll_tg: Stat,
        hk_mem: Mem,
        ll_mem: Mem,
    };
    var rows: std.ArrayList(Row) = .empty;

    for (opts.pairs.items) |pair| {
        const colon = std.mem.lastIndexOfScalar(u8, pair, ':') orelse return usage();
        const hk_model = pair[0..colon];
        const gguf = pair[colon + 1 ..];
        const name = std.fs.path.stem(hk_model);
        std.debug.print("== {s}\n", .{name});

        var p_s = std.fmt.allocPrint(arena, "{d}", .{opts.prompt_tokens}) catch unreachable;
        var g_s = std.fmt.allocPrint(arena, "{d}", .{opts.gen_tokens}) catch unreachable;
        var t_s = std.fmt.allocPrint(arena, "{d}", .{opts.threads}) catch unreachable;

        // One untimed pass so both tools start with a warm page cache.
        _ = std.process.run(gpa, io, .{ .argv = &.{ probe, hk_model, "--bench", "64", "8" }, .environ_map = &env }) catch {};
        _ = std.process.run(gpa, io, .{ .argv = &.{ llama_bench, "-m", gguf, "-t", t_s, "-p", "64", "-n", "8", "-r", "1" }, .environ_map = &env }) catch {};

        const hk_pp = try arena.alloc(f64, opts.repeats);
        const hk_tg = try arena.alloc(f64, opts.repeats);
        const ll_pp = try arena.alloc(f64, opts.repeats);
        const ll_tg = try arena.alloc(f64, opts.repeats);
        for (0..opts.repeats) |r| {
            const h = try hkSpeed(gpa, io, &env, probe, hk_model, p_s, g_s);
            hk_pp[r] = h[0];
            hk_tg[r] = h[1];
            const l = try llamaSpeed(gpa, io, &env, llama_bench, gguf, t_s, p_s, g_s);
            ll_pp[r] = l[0];
            ll_tg[r] = l[1];
            std.debug.print("  run {d}: hk {d:.0}/{d:.1}  llama.cpp {d:.0}/{d:.1}\n", .{ r + 1, h[0], h[1], l[0], l[1] });
        }
        _ = &p_s;
        _ = &g_s;
        _ = &t_s;

        const prompt = "Write a short story about a robot who learns to paint.";
        const hk_mem = try memoryRun(gpa, io, &env, &.{ hk, "run", hk_model, prompt, "--temp", "0", "-n", g_s, "--threads", t_s });
        const ll_mem = try memoryRun(gpa, io, &env, &.{ llama_cli, "-m", gguf, "-p", prompt, "-n", g_s, "-t", t_s, "--temp", "0", "--no-display-prompt", "-no-cnv", "-c", "1024" });
        if (!hk_mem.ok or !ll_mem.ok) std.debug.print("warning: a generation run failed; memory numbers are not valid\n", .{});

        try rows.append(arena, .{
            .name = try arena.dupe(u8, name),
            .hk_pp = Stat.of(hk_pp),
            .hk_tg = Stat.of(hk_tg),
            .ll_pp = Stat.of(ll_pp),
            .ll_tg = Stat.of(ll_tg),
            .hk_mem = hk_mem,
            .ll_mem = ll_mem,
        });
    }

    var aw: std.Io.Writer.Allocating = .init(arena);
    const w = &aw.writer;
    try w.print("Machine: {s}, {d} threads, CPU only. Kernel level: see hk-probe.\n", .{ cpuName(gpa, io) catch "unknown CPU", opts.threads });
    try w.print("Prompt {d} tokens, generate {d}; median of {d} alternating runs of each tool, range in brackets.\n\n", .{ opts.prompt_tokens, opts.gen_tokens, opts.repeats });
    try w.writeAll("| Model | Prefill hk | Prefill llama.cpp | hk / llama.cpp | Decode hk | Decode llama.cpp | hk / llama.cpp |\n");
    try w.writeAll("|:---|---:|---:|---:|---:|---:|---:|\n");
    for (rows.items) |r| {
        try w.print("| {s} | {d:.0} [{d:.0}-{d:.0}] | {d:.0} [{d:.0}-{d:.0}] | {d:.0}% | {d:.1} [{d:.1}-{d:.1}] | {d:.1} [{d:.1}-{d:.1}] | {d:.0}% |\n", .{
            r.name,
            r.hk_pp.median,
            r.hk_pp.min,
            r.hk_pp.max,
            r.ll_pp.median,
            r.ll_pp.min,
            r.ll_pp.max,
            100.0 * r.hk_pp.median / r.ll_pp.median,
            r.hk_tg.median,
            r.hk_tg.min,
            r.hk_tg.max,
            r.ll_tg.median,
            r.ll_tg.min,
            r.ll_tg.max,
            100.0 * r.hk_tg.median / r.ll_tg.median,
        });
    }
    try w.writeAll("\nMemory during a greedy generation (MiB). Private is memory the process owns; the rest of resident memory is the model file mapped from disk. llama.cpp is run with a 1024 token context, because by default it reserves the model's whole training context up front. hk allocates its cache as it fills.\n\n");
    try w.writeAll("| Model | hk peak resident | hk private | llama.cpp peak resident | llama.cpp private |\n|:---|---:|---:|---:|---:|\n");
    for (rows.items) |r| {
        try w.print("| {s} | {d:.0} | {d:.0} | {d:.0} | {d:.0} |\n", .{ r.name, r.hk_mem.rss_mib, r.hk_mem.anon_mib, r.ll_mem.rss_mib, r.ll_mem.anon_mib });
    }
    const text = aw.written();
    std.debug.print("\n{s}", .{text});
    if (opts.out) |path| {
        var f = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer f.close(io);
        try f.writeStreamingAll(io, text);
    }
}

fn usage() error{BadUsage} {
    std.debug.print("usage: hk-compare --llama-bin DIR --pair model.hk:model.gguf [--pair ...] [--threads N] [--prompt-tokens N] [--gen-tokens N] [--repeats N] [--hk-bin DIR] [--out FILE]\n", .{});
    return error.BadUsage;
}

/// Reads a small file whose size the OS does not report (/proc) into `buf`.
fn slurp(io: std.Io, path: []const u8, buf: []u8) ![]const u8 {
    var f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    var n: usize = 0;
    while (n < buf.len) {
        const got = f.readStreaming(io, &.{buf[n..]}) catch |e| switch (e) {
            error.EndOfStream => break,
            else => return e,
        };
        if (got == 0) break;
        n += got;
    }
    return buf[0..n];
}

fn cpuName(gpa: Allocator, io: std.Io) ![]const u8 {
    const big = try gpa.alloc(u8, 1 << 20);
    const text = try slurp(io, "/proc/cpuinfo", big);
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |l| {
        if (std.mem.startsWith(u8, l, "model name")) {
            const c = std.mem.indexOfScalar(u8, l, ':') orelse continue;
            return std.mem.trim(u8, l[c + 1 ..], " ");
        }
    }
    return "unknown CPU";
}

/// Number following the first occurrence of `key` in `text`.
fn numberAfter(text: []const u8, key: []const u8) ?f64 {
    const i = std.mem.indexOf(u8, text, key) orelse return null;
    var j = i + key.len;
    while (j < text.len and (text[j] == ' ')) j += 1;
    var e = j;
    while (e < text.len and (std.ascii.isDigit(text[e]) or text[e] == '.')) e += 1;
    return std.fmt.parseFloat(f64, text[j..e]) catch null;
}

/// {prefill tok/s, decode tok/s} from one `hk-probe --bench` run.
fn hkSpeed(gpa: Allocator, io: std.Io, env: *const std.process.Environ.Map, probe: []const u8, model: []const u8, p: []const u8, g: []const u8) ![2]f64 {
    const r = try std.process.run(gpa, io, .{ .argv = &.{ probe, model, "--bench", p, g }, .environ_map = env });
    defer gpa.free(r.stdout);
    defer gpa.free(r.stderr);
    const mark = std.mem.indexOf(u8, r.stderr, "tok/s | decode") orelse {
        std.debug.print("could not read hk-probe output:\n{s}\n", .{r.stderr});
        return error.BadOutput;
    };
    var line_start = mark;
    while (line_start > 0 and r.stderr[line_start - 1] != '\n') line_start -= 1;
    var line_end = mark;
    while (line_end < r.stderr.len and r.stderr[line_end] != '\n') line_end += 1;
    const line = r.stderr[line_start..line_end];
    const bar = std.mem.indexOf(u8, line, "| decode") orelse return error.BadOutput;
    const pp = numberAfter(line[0..bar], " = ") orelse return error.BadOutput;
    const tg = numberAfter(line[bar..], " = ") orelse return error.BadOutput;
    return .{ pp, tg };
}

/// {prefill tok/s, decode tok/s} from one `llama-bench` run (CSV output).
fn llamaSpeed(gpa: Allocator, io: std.Io, env: *const std.process.Environ.Map, bench: []const u8, gguf: []const u8, t: []const u8, p: []const u8, g: []const u8) ![2]f64 {
    const r = try std.process.run(gpa, io, .{ .argv = &.{ bench, "-m", gguf, "-t", t, "-p", p, "-n", g, "-r", "1", "-o", "csv" }, .environ_map = env });
    defer gpa.free(r.stdout);
    defer gpa.free(r.stderr);
    var lines = std.mem.splitScalar(u8, r.stdout, '\n');
    const header = lines.next() orelse return error.BadOutput;
    var col_p: usize = 0;
    var col_g: usize = 0;
    var col_ts: usize = 0;
    var ci: usize = 0;
    var hf = std.mem.splitScalar(u8, header, ',');
    while (hf.next()) |h| : (ci += 1) {
        const name = std.mem.trim(u8, h, "\" \r");
        if (std.mem.eql(u8, name, "n_prompt")) col_p = ci;
        if (std.mem.eql(u8, name, "n_gen")) col_g = ci;
        if (std.mem.eql(u8, name, "avg_ts")) col_ts = ci;
    }
    var pp: ?f64 = null;
    var tg: ?f64 = null;
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var fields: [64][]const u8 = undefined;
        var n: usize = 0;
        var fi = std.mem.splitScalar(u8, line, ',');
        while (fi.next()) |f| : (n += 1) {
            if (n < fields.len) fields[n] = std.mem.trim(u8, f, "\" \r");
        }
        if (n <= @max(col_p, @max(col_g, col_ts))) continue;
        const np = std.fmt.parseInt(usize, fields[col_p], 10) catch continue;
        const ng = std.fmt.parseInt(usize, fields[col_g], 10) catch continue;
        const ts = std.fmt.parseFloat(f64, fields[col_ts]) catch continue;
        if (np > 0 and ng == 0) pp = ts;
        if (ng > 0 and np == 0) tg = ts;
    }
    if (pp == null or tg == null) {
        std.debug.print("could not read llama-bench output:\n{s}\n{s}\n", .{ r.stdout, r.stderr });
        return error.BadOutput;
    }
    return .{ pp.?, tg.? };
}

fn statusKiB(text: []const u8, key: []const u8) usize {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |l| {
        if (std.mem.startsWith(u8, l, key) and l.len > key.len and l[key.len] == ':') {
            var it = std.mem.tokenizeAny(u8, l[key.len + 1 ..], " \tkB");
            if (it.next()) |v| return std.fmt.parseInt(usize, v, 10) catch 0;
        }
    }
    return 0;
}

/// Runs a generation and returns the peak resident and private memory seen by polling /proc.
fn memoryRun(gpa: Allocator, io: std.Io, env: *const std.process.Environ.Map, argv: []const []const u8) !Mem {
    if (builtin.os.tag != .linux) return .{};
    var child = try std.process.spawn(io, .{ .argv = argv, .environ_map = env, .stdin = .ignore, .stdout = .ignore, .stderr = .ignore });
    const pid = child.id orelse return .{};
    var path_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "/proc/{d}/status", .{pid});

    const Poll = struct {
        fn run(gpa_: Allocator, io_: std.Io, path_: []const u8, stop: *std.atomic.Value(bool), peak_rss: *usize, peak_anon: *usize) void {
            while (!stop.load(.acquire)) {
                var buf: [1 << 15]u8 = undefined;
                if (slurp(io_, path_, &buf)) |text| {
                    peak_rss.* = @max(peak_rss.*, statusKiB(text, "VmRSS"));
                    peak_anon.* = @max(peak_anon.*, statusKiB(text, "RssAnon"));
                } else |_| {}
                _ = gpa_;
                std.Io.sleep(io_, .fromMilliseconds(5), .awake) catch {};
            }
        }
    };
    var stop = std.atomic.Value(bool).init(false);
    var rss: usize = 0;
    var anon: usize = 0;
    const th = try std.Thread.spawn(.{}, Poll.run, .{ gpa, io, path, &stop, &rss, &anon });
    const term = try child.wait(io);
    stop.store(true, .release);
    th.join();
    const ok = switch (term) {
        .exited => |c| c == 0,
        else => false,
    };
    return .{ .rss_mib = @as(f64, @floatFromInt(rss)) / 1024.0, .anon_mib = @as(f64, @floatFromInt(anon)) / 1024.0, .ok = ok };
}
