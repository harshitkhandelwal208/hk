//! Developer tool: run token ids through the engine and dump logits for external comparison.
//!
//!   hk-probe model.hk 1,2,3 [out.f32]     logits of the last token, raw little endian f32 to out.f32
//!   hk-probe model.hk --ppl ids.u32 [ctx] mean negative log likelihood over a token file
//!
//! Exists so correctness can be checked against Transformers and llama.cpp on the same ids.
const std = @import("std");
const hk = @import("hk");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer it.deinit();
    _ = it.next();
    const path = it.next() orelse return usage();
    const arg2 = it.next() orelse return usage();

    if (std.mem.eql(u8, path, "--render")) {
        // hk-probe --render template.jinja context.json  (no model needed)
        return renderTemplate(allocator, io, arg2, it.next() orelse return usage());
    }

    var reader = try hk.reader.HKReader.open(path, allocator);
    defer reader.deinit();

    if (std.mem.eql(u8, arg2, "--tok")) {
        // hk-probe model.hk --tok in.txt out.u32 [special]
        const in_path = it.next() orelse return usage();
        const out_path = it.next() orelse return usage();
        const parse_special = if (it.next()) |x| std.mem.eql(u8, x, "special") else false;
        var tdiag = hk.tokenizer.Diag{};
        var tok = hk.tokenizer.Tokenizer.fromMetadata(allocator, &reader.metadata_map, &tdiag) catch |err| {
            std.debug.print("tokenizer: {s}: {s}\n", .{ @errorName(err), tdiag.message() });
            return err;
        };
        defer tok.deinit();
        const cwd = std.Io.Dir.cwd();
        const text = try cwd.readFileAlloc(io, in_path, allocator, .limited(1 << 30));
        defer allocator.free(text);
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(allocator);
        const t0 = nowNs(io);
        try tok.encode(text, .{ .add_special = false, .parse_special = parse_special }, &ids);
        const t1 = nowNs(io);
        var f = try cwd.createFile(io, out_path, .{});
        defer f.close(io);
        try f.writeStreamingAll(io, std.mem.sliceAsBytes(ids.items));
        const secs = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
        std.debug.print("{d} tokens from {d} bytes in {d:.3}s ({d:.1} MB/s)\n", .{ ids.items.len, text.len, secs, @as(f64, @floatFromInt(text.len)) / 1e6 / secs });
        return;
    }

    var diag = hk.engine.Diag{};
    if (init.environ_map.get("HK_TILE_MIN")) |v| hk.engine.matmul.min_tiled_tokens = std.fmt.parseInt(usize, v, 10) catch 8;
    var n_threads: usize = 0;
    if (init.environ_map.get("HK_THREADS")) |v| n_threads = std.fmt.parseInt(usize, v, 10) catch 0;
    const want_gpu = if (init.environ_map.get("HK_GPU")) |v| std.mem.eql(u8, v, "1") else false;
    gpu_profile = init.environ_map.get("HK_GPU_PROFILE") != null;
    cuda_events = init.environ_map.get("HK_CUDA_EVENTS") != null;
    // HK_BACKEND=vulkan|cuda picks the device (and implies HK_GPU=1); cpu forces the CPU.
    var backend: hk.backend.select.Choice = .auto;
    var gpu_mode: hk.engine.model.GpuMode = if (want_gpu) .on else .off;
    if (init.environ_map.get("HK_BACKEND")) |v| {
        backend = std.meta.stringToEnum(hk.backend.select.Choice, v) orelse {
            std.debug.print("HK_BACKEND must be auto, cpu, vulkan or cuda\n", .{});
            return error.BadUsage;
        };
        gpu_mode = if (backend == .cpu) .off else .on;
    }
    // HK_BATCH=N sets the most tokens per pass (1 makes `--ppl` run every token through the decode kernels).
    const n_batch: usize = if (init.environ_map.get("HK_BATCH")) |v| std.fmt.parseInt(usize, v, 10) catch 256 else 256;
    // HK_GPU_LAYERS=N runs the first N layers on the device and the rest on the CPU (partial offload).
    const gpu_layers: ?usize = if (init.environ_map.get("HK_GPU_LAYERS")) |v| std.fmt.parseInt(usize, v, 10) catch null else null;
    var model = hk.engine.Model.init(allocator, &reader, .{ .n_ctx = 4096, .n_threads = n_threads, .n_batch = n_batch, .gpu = gpu_mode, .backend = backend, .gpu_layers = gpu_layers }, &diag) catch |err| {
        std.debug.print("load failed: {s}: {s}\n", .{ @errorName(err), diag.message() });
        return err;
    };
    defer model.deinit();

    std.debug.print("kernels {s} (built: {s}){s}\n", .{ hk.kernels.get().name, hk.build_options_kernel_levels, if (model.gpu_layers != 0) "; first layers on the GPU (partial offload)" else if (model.gpu != null) "; running on the GPU" else "" });
    if (vkEngine(&model)) |e| std.debug.print("vulkan prompt matmul: {s}\n", .{e.matmulPath()});
    if (std.mem.eql(u8, arg2, "--bench")) {
        const n_prompt: usize = if (it.next()) |x| try std.fmt.parseInt(usize, x, 10) else 512;
        const n_gen: usize = if (it.next()) |x| try std.fmt.parseInt(usize, x, 10) else 64;
        return bench(allocator, io, &model, n_prompt, n_gen);
    }
    if (std.mem.eql(u8, arg2, "--multi")) {
        const n_seq: usize = if (it.next()) |x| try std.fmt.parseInt(usize, x, 10) else 8;
        const n_gen: usize = if (it.next()) |x| try std.fmt.parseInt(usize, x, 10) else 64;
        return multi(allocator, io, &model, n_seq, n_gen);
    }
    if (std.mem.eql(u8, arg2, "--ppl")) {
        const ids_path = it.next() orelse return usage();
        const ctx_len: usize = if (it.next()) |s| try std.fmt.parseInt(usize, s, 10) else 512;
        const bos: ?u32 = if (init.environ_map.get("HK_PPL_BOS")) |v| std.fmt.parseInt(u32, v, 10) catch null else null;
        return perplexity(allocator, io, &model, ids_path, ctx_len, bos);
    }

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(allocator);
    var parts = std.mem.splitScalar(u8, arg2, ',');
    while (parts.next()) |p| try ids.append(allocator, try std.fmt.parseInt(u32, p, 10));

    var pos: usize = 0;
    while (pos < ids.items.len) {
        const n = @min(model.n_batch, ids.items.len - pos);
        try model.forward(ids.items[pos..][0..n], pos);
        pos += n;
        if (pos == ids.items.len) {
            const logits = model.logitsFor(n - 1);
            if (it.next()) |out_path| {
                const cwd = std.Io.Dir.cwd();
                var f = try cwd.createFile(io, out_path, .{});
                defer f.close(io);
                try f.writeStreamingAll(io, std.mem.sliceAsBytes(logits));
            }
            var best: usize = 0;
            for (logits, 0..) |l, i| if (l > logits[best]) {
                best = i;
            };
            std.debug.print("argmax {d} logit {d:.4}\n", .{ best, logits[best] });
        }
    }
}

fn perplexity(allocator: std.mem.Allocator, io: std.Io, model: *hk.engine.Model, ids_path: []const u8, ctx_len: usize, bos: ?u32) !void {
    const cwd = std.Io.Dir.cwd();
    const bytes = try cwd.readFileAlloc(io, ids_path, allocator, .limited(1 << 30));
    defer allocator.free(bytes);
    const ids_in: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, bytes));
    // Models whose vocabulary adds a beginning-of-sequence token: llama-perplexity puts it at the start of every window
    // (replacing the first token), so HK_PPL_BOS=<id> does the same.
    const ids = try allocator.dupe(u32, ids_in);
    defer allocator.free(ids);
    if (bos) |b| {
        var w: usize = 0;
        while (w + ctx_len <= ids.len) : (w += ctx_len) ids[w] = b;
    }

    var nll: f64 = 0;
    var count: usize = 0;
    // Same protocol as llama-perplexity so the numbers are directly comparable: only full
    // windows, and within a window the first half is context and only the second half is
    // scored.
    const n_chunks = ids.len / ctx_len;
    for (0..n_chunks) |c| {
        const win = ids[c * ctx_len ..][0..ctx_len];
        var pos: usize = 0;
        while (pos < win.len) {
            const n = @min(model.n_batch, win.len - pos);
            try model.forward(win[pos..][0..n], pos);
            for (0..n) |t| {
                const gi = pos + t;
                if (gi < ctx_len / 2 or gi + 1 >= win.len) continue;
                const logits = model.logitsFor(t);
                var mx: f32 = -std.math.inf(f32);
                for (logits) |l| mx = @max(mx, l);
                var sum: f64 = 0;
                for (logits) |l| sum += @exp(@as(f64, l - mx));
                const lp = @as(f64, logits[win[gi + 1]] - mx) - @log(sum);
                nll -= lp;
                count += 1;
            }
            pos += n;
        }
    }
    std.debug.print("tokens {d} mean_nll {d:.5} ppl {d:.4}\n", .{ count, nll / @as(f64, @floatFromInt(count)), @exp(nll / @as(f64, @floatFromInt(count))) });
}

fn nowNs(io: std.Io) i128 {
    return std.Io.Timestamp.now(io, .awake).nanoseconds;
}

/// Prefill and decode throughput with synthetic tokens. Decode feeds the previous argmax back
/// in, exactly like generation, so it includes the output projection every step.
var gpu_profile = false;
/// Per kernel GPU timing for CUDA (synchronizes after every launch, so it distorts speed).
var cuda_events = false;

fn bench(allocator: std.mem.Allocator, io: std.Io, model: *hk.engine.Model, n_prompt: usize, n_gen: usize) !void {
    const prompt = try allocator.alloc(u32, n_prompt);
    defer allocator.free(prompt);
    for (prompt, 0..) |*t, i| t.* = @intCast(1 + (i * 7919) % (model.cfg.vocab - 1));

    model.cpu.profile.enabled = true;
    const cuda_profile = cuda_events and isCuda(model);
    if (vkEngine(model)) |g| if (gpu_profile) g.enableProfiling() catch {};
    if (gpu_profile) if (model.gpu) |dev| switch (dev) {
        .cuda => |e| e.enablePassTiming() catch {},
        .vulkan => {},
    };
    if (cuda_profile) hk.cuda.profiling.enable() catch |e| std.debug.print("cuda profiling unavailable: {s}\n", .{@errorName(e)});
    printMem(io, "loaded");
    const t0 = nowNs(io);
    var pos: usize = 0;
    while (pos < n_prompt) {
        const n = @min(model.n_batch, n_prompt - pos);
        try model.forward(prompt[pos..][0..n], pos);
        pos += n;
    }
    // A prefill ends with the first prediction. Device passes run asynchronously, so without reading the logits the
    // clock would stop while the GPU is still busy and that time would be charged to the first decode steps.
    _ = model.logitsFor((n_prompt - 1) % model.n_batch);
    const t1 = nowNs(io);
    printProfile("prefill", &model.cpu.profile);
    if (vkEngine(model)) |g| if (g.tq != null) {
        const names = [_][]const u8{ "embed", "norm", "mmv", "mm", "rope", "attn", "swiglu", "add", "qact" };
        std.debug.print("gpu kernels, prefill (ms per pass of the whole prompt):", .{});
        for (names, 0..) |nm, i| std.debug.print(" {s} {d:.2}", .{ nm, g.kind_ns[i] / 1e6 });
        std.debug.print("\n", .{});
    };
    if (cuda_profile) {
        std.debug.print("cuda kernels, prefill:\n", .{});
        printCudaProfile();
        hk.cuda.profiling.reset();
    }
    model.cpu.profile = .{ .enabled = true };
    if (vkEngine(model)) |g| {
        g.kind_ns = @splat(0);
        g.prof_record = 0;
        g.prof_wait = 0;
        g.prof_calls = 0;
    }
    var tok: u32 = prompt[n_prompt - 1];
    for (0..n_gen) |_| {
        try model.forward(&.{tok}, pos);
        pos += 1;
        const logits = model.logitsFor(0);
        var best: usize = 0;
        for (logits, 0..) |l, i| if (l > logits[best]) {
            best = i;
        };
        tok = @intCast(best);
    }
    const t2 = nowNs(io);
    printProfile("decode ", &model.cpu.profile);
    if (cuda_profile) {
        std.debug.print("cuda kernels, decode:\n", .{});
        printCudaProfile();
    }
    if (gpu_profile) if (model.gpu) |dev| switch (dev) {
        .cuda => |e| std.debug.print("cuda host time: {d} passes, issuing launches {d:.1} ms, waiting for results {d:.1} ms; gpu time of the passes read back by logits {d:.1} ms\n", .{ e.prof_passes, e.prof_launch_s * 1e3, e.prof_wait_s * 1e3, e.prof_gpu_ms }),
        .vulkan => {},
    };
    if (vkEngine(model)) |g| if (g.tq != null) {
        const names = [_][]const u8{ "embed", "norm", "mmv", "mm", "rope", "attn", "swiglu", "add", "qact" };
        std.debug.print("gpu kernels (ms per token):", .{});
        for (names, 0..) |nm, i| std.debug.print(" {s} {d:.2}", .{ nm, g.kind_ns[i] / 1e6 / @as(f64, @floatFromInt(@max(1, n_gen))) });
        std.debug.print("\n", .{});
    };
    if (vkEngine(model)) |g| std.debug.print("gpu: {d} passes, record {d:.1} ms, device {d:.1} ms\n", .{ g.prof_calls, g.prof_record * 1e3, g.prof_wait * 1e3 });
    printMem(io, "after run");
    const pf = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const dc = @as(f64, @floatFromInt(t2 - t1)) / 1e9;
    std.debug.print("prefill {d} tok in {d:.3}s = {d:.1} tok/s | decode {d} tok in {d:.3}s = {d:.1} tok/s | threads {d}\n", .{
        n_prompt, pf, @as(f64, @floatFromInt(n_prompt)) / pf, n_gen, dc, @as(f64, @floatFromInt(n_gen)) / dc, model.cpu.pool.size(),
    });
}

/// Decode with `n_seq` independent sequences stepped together, the way the server does. Reports
/// aggregate tokens per second and the share of time spent in the output projection.
fn multi(allocator: std.mem.Allocator, io: std.Io, model: *hk.engine.Model, n_seq: usize, n_gen: usize) !void {
    const caches = try allocator.alloc(hk.engine.KvCache, n_seq);
    defer allocator.free(caches);
    var made: usize = 0;
    defer for (caches[0..made]) |*c| c.deinit();
    for (caches) |*c| {
        c.* = try hk.engine.KvCache.init(allocator, model.cfg, 512);
        made += 1;
    }
    const items = try allocator.alloc(hk.engine.model.Item, n_seq);
    defer allocator.free(items);
    const picks = try allocator.alloc(usize, n_seq);
    defer allocator.free(picks);
    for (picks, 0..) |*p, i| p.* = i;
    const toks = try allocator.alloc(u32, n_seq);
    defer allocator.free(toks);
    for (toks, 0..) |*t, i| t.* = @intCast(1 + (i * 7919) % (model.cfg.vocab - 1));

    var t_fwd: i128 = 0;
    var t_log: i128 = 0;
    model.cpu.profile = .{ .enabled = true };
    const t0 = nowNs(io);
    for (0..n_gen) |step| {
        for (items, 0..) |*it, i| it.* = .{ .kv = &caches[i], .token = toks[i], .pos = @intCast(step) };
        const a = nowNs(io);
        try model.forwardItems(items);
        const b = nowNs(io);
        const rows = try model.logitsMany(picks);
        const c = nowNs(io);
        t_fwd += b - a;
        t_log += c - b;
        const vocab = model.cfg.vocab;
        for (toks, 0..) |*t, i| {
            const row = rows[i * vocab ..][0..vocab];
            var best: usize = 0;
            for (row, 0..) |l, j| if (l > row[best]) {
                best = j;
            };
            t.* = @intCast(best);
        }
    }
    const t1 = nowNs(io);
    const secs = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    printProfile("multi  ", &model.cpu.profile);
    const ms = struct {
        fn f(x: i128) f64 {
            return @as(f64, @floatFromInt(x)) / 1e6;
        }
    }.f;
    std.debug.print("seqs {d}: {d:.1} agg tok/s | per step {d:.2} ms = forward {d:.2} + logits {d:.2}\n", .{
        n_seq, @as(f64, @floatFromInt(n_seq * n_gen)) / secs, ms(t1 - t0) / @as(f64, @floatFromInt(n_gen)), ms(t_fwd) / @as(f64, @floatFromInt(n_gen)), ms(t_log) / @as(f64, @floatFromInt(n_gen)),
    });
}

/// Resident memory split into anonymous (the engine's own cost) and file backed (weights mapped
/// from the container, shared with the page cache). Prints the total only where the platform
/// cannot split it, and nothing where it cannot be read.
fn printMem(io: std.Io, label: []const u8) void {
    _ = io;
    const m = hk.platform.selfMemory();
    if (!m.ok) return;
    if (!m.has_split) {
        std.debug.print("mem[{s}] rss {d:.1} MiB\n", .{ label, m.rss_mib });
        return;
    }
    std.debug.print("mem[{s}] rss {d:.1} MiB = anon {d:.1} + file {d:.1}\n", .{ label, m.rss_mib, m.anon_mib, m.file_mib });
}

fn printProfile(label: []const u8, p: *const hk.engine.model.Profile) void {
    const ms = struct {
        fn f(x: i128) f64 {
            return @as(f64, @floatFromInt(x)) / 1e6;
        }
    }.f;
    std.debug.print("{s} ms: norm {d:.1} prepare {d:.1} qkv {d:.1} rope {d:.1} kv {d:.1} attn {d:.1} wo {d:.1} ffn_in {d:.1} act {d:.1} ffn_down {d:.1}\n", .{
        label, ms(p.norm), ms(p.prepare), ms(p.qkv), ms(p.rope), ms(p.kv), ms(p.attn), ms(p.wo), ms(p.ffn_in), ms(p.act), ms(p.ffn_down),
    });
}

/// Renders a chat template for the parity tests against Python's jinja2. Prints the output to
/// stdout, or `RAISED: message` / `ERROR: message` and a non-zero exit when rendering fails.
fn renderTemplate(allocator: std.mem.Allocator, io: std.Io, tpl_path: []const u8, ctx_path: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    const src = try cwd.readFileAlloc(io, tpl_path, allocator, .limited(16 << 20));
    defer allocator.free(src);
    const ctx_json = try cwd.readFileAlloc(io, ctx_path, allocator, .limited(64 << 20));
    defer allocator.free(ctx_json);

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    var diag = hk.chat.jinja.Diag{};
    var tpl = hk.chat.jinja.Template.parse(allocator, src, &diag) catch {
        std.debug.print("ERROR: {s}\n", .{diag.message()});
        std.process.exit(2);
    };
    defer tpl.deinit();
    const parsed = try std.json.parseFromSlice(std.json.Value, a, ctx_json, .{});
    const ctx_v = try hk.chat.jinja.fromJson(a, parsed.value);
    const out = hk.chat.jinja.render(a, &tpl, ctx_v.dict, &diag) catch |e| {
        if (e == error.TemplateRaised) std.debug.print("RAISED: {s}\n", .{diag.message()}) else std.debug.print("ERROR: {s}\n", .{diag.message()});
        std.process.exit(3);
    };
    var f = std.Io.File.stdout();
    try f.writeStreamingAll(io, out);
}

fn usage() void {
    std.debug.print("usage: hk-probe model.hk id,id,... [out.f32]\n       hk-probe model.hk --ppl ids.u32 [ctx]\n", .{});
}

/// The Vulkan engine behind the model's device, for the profiling counters only it has.
fn vkEngine(model: *hk.engine.Model) ?*hk.gpu.Engine {
    const dev = model.gpu orelse return null;
    return switch (dev) {
        .vulkan => |e| e,
        .cuda => null,
    };
}

fn isCuda(model: *hk.engine.Model) bool {
    const dev = model.gpu orelse return false;
    return dev.kind() == .cuda;
}

fn printCudaProfile() void {
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    hk.cuda.profiling.report(&w) catch {};
    std.debug.print("{s}", .{w.buffered()});
}
