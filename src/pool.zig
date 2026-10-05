//! A small persistent thread pool for data parallel loops.
//!
//! Workers are started once and reused for every matrix multiply. A dispatch costs a few
//! atomic operations when workers are still spinning from the previous call, which is the
//! common case during token generation where one call follows the next within microseconds.
//! After a short spin the workers park on a futex so an idle process uses no CPU.
//!
//! Work is handed out in fixed size chunks through one shared counter, so a slow chunk (a
//! cache miss storm, a descheduled core) delays only itself instead of a whole static slice.

const std = @import("std");

pub const TaskFn = *const fn (ctx: *anyopaque, start: usize, end: usize) void;

const cache_line = 64;

/// Most bytes of a hint that are prefetched, across all threads.
const max_hint: usize = 1 << 20;

/// Spin iterations before a worker parks. About 100 microseconds, long enough to bridge the
/// gap between consecutive operations of one token without keeping an idle process hot.
const spin_before_park: u32 = 4000;

const Shared = struct {
    generation: std.atomic.Value(u32) align(cache_line) = .init(0),
    next: std.atomic.Value(usize) align(cache_line) = .init(0),
    remaining: std.atomic.Value(u32) align(cache_line) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
    /// Workers currently asleep on the futex. A dispatch only pays for a wake syscall when this
    /// is not zero, which during token generation it almost never is.
    parked: std.atomic.Value(u32) align(cache_line) = .init(0),

    /// Memory the next dispatch will read. Threads that run out of chunks early touch their
    /// stripe of it while they wait, so the fetch overlaps the stragglers instead of following them.
    hint_ptr: [*]const u8 = undefined,
    hint_len: usize = 0,
    n_threads: usize = 1,

    func: TaskFn = undefined,
    ctx: *anyopaque = undefined,
    total: usize = 0,
    grain: usize = 1,
};

/// Index of the calling thread inside its pool: 0 for the thread that calls `parallelFor`,
/// 1.. for workers. Lets a task find its own scratch buffer without any locking.
threadlocal var tls_worker_id: usize = 0;

pub fn workerId() usize {
    return tls_worker_id;
}

pub const Scratch = []align(64) u8;

pub const Pool = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    shared: *Shared,
    threads: []std.Thread,
    /// One buffer per participating thread, grown by `ensureScratch`.
    scratch: []Scratch = &.{},
    pending_hint: []const u8 = &.{},

    /// `n_threads` counts the calling thread, so 1 means everything runs inline.
    pub fn init(allocator: std.mem.Allocator, n_threads: usize) !Pool {
        const shared = try allocator.create(Shared);
        errdefer allocator.destroy(shared);
        shared.* = .{};

        const n_workers = if (n_threads > 1) n_threads - 1 else 0;
        const threads = try allocator.alloc(std.Thread, n_workers);
        errdefer allocator.free(threads);

        const io = std.Options.debug_io;
        var spawned: usize = 0;
        errdefer {
            // Stop and join whichever workers did start so nothing outlives the shared state.
            shared.stop.store(true, .release);
            _ = shared.generation.fetchAdd(1, .release);
            io.futexWake(u32, &shared.generation.raw, std.math.maxInt(u32));
            for (threads[0..spawned]) |t| t.join();
        }
        for (0..n_workers) |i| {
            threads[i] = try std.Thread.spawn(.{}, worker, .{ shared, io, i + 1 });
            spawned += 1;
        }
        return .{ .allocator = allocator, .io = io, .shared = shared, .threads = threads };
    }

    /// Makes every thread's scratch buffer at least `bytes` long. Call between dispatches, from
    /// the thread that dispatches. Contents are not preserved.
    pub fn ensureScratch(self: *Pool, bytes: usize) error{OutOfMemory}!void {
        if (self.scratch.len == 0) {
            self.scratch = try self.allocator.alloc(Scratch, self.threads.len + 1);
            for (self.scratch) |*b| b.* = &.{};
        }
        for (self.scratch) |*b| {
            if (b.len >= bytes) continue;
            if (b.len != 0) self.allocator.free(b.*);
            b.* = &.{};
            b.* = try self.allocator.alignedAlloc(u8, .@"64", bytes);
        }
    }

    /// The calling thread's scratch buffer. Only valid inside a task after `ensureScratch`.
    pub fn myScratch(self: *const Pool) Scratch {
        return self.scratch[tls_worker_id];
    }

    pub fn deinit(self: *Pool) void {
        for (self.scratch) |b| if (b.len != 0) self.allocator.free(b);
        if (self.scratch.len != 0) self.allocator.free(self.scratch);
        self.shared.stop.store(true, .release);
        _ = self.shared.generation.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.shared.generation.raw, std.math.maxInt(u32));
        for (self.threads) |t| t.join();
        self.allocator.free(self.threads);
        self.allocator.destroy(self.shared);
    }

    /// Number of threads that take part in a `parallelFor`, including the caller.
    /// Tells the pool what the dispatch after the current one will read.
    pub fn hintNext(self: *Pool, mem: []const u8) void {
        self.pending_hint = mem[0..@min(mem.len, max_hint)];
    }

    pub fn size(self: *const Pool) usize {
        return self.threads.len + 1;
    }

    /// Runs `func(ctx, start, end)` over `[0, total)` split into chunks of `grain` items.
    /// Returns when every chunk has finished. Not reentrant: call from one thread at a time.
    pub fn parallelFor(self: *Pool, total: usize, grain: usize, ctx: *anyopaque, func: TaskFn) void {
        if (total == 0) return;
        const g = @max(grain, 1);
        // Not worth waking anyone for a single chunk.
        if (self.threads.len == 0 or total <= g) {
            func(ctx, 0, total);
            return;
        }
        const s = self.shared;
        s.func = func;
        s.ctx = ctx;
        s.total = total;
        s.grain = g;
        s.n_threads = self.threads.len + 1;
        s.hint_ptr = self.pending_hint.ptr;
        s.hint_len = self.pending_hint.len;
        self.pending_hint = &.{};
        s.next.store(0, .monotonic);
        s.remaining.store(@intCast(self.threads.len), .monotonic);
        _ = s.generation.fetchAdd(1, .seq_cst);
        if (s.parked.load(.seq_cst) != 0) self.io.futexWake(u32, &s.generation.raw, std.math.maxInt(u32));

        runChunks(s);
        prefetchStripe(s, 0);

        var spins: u32 = 0;
        while (s.remaining.load(.acquire) != 0) {
            std.atomic.spinLoopHint();
            spins += 1;
            if (spins > 1 << 20) {
                // A worker was descheduled for a long time. Yielding keeps the machine
                // responsive without changing the result.
                std.Thread.yield() catch {};
                spins = 0;
            }
        }
    }
};

fn prefetchStripe(s: *Shared, id: usize) void {
    const len = s.hint_len;
    if (len == 0) return;
    const per = (len / s.n_threads + 63) & ~@as(usize, 63);
    const lo = @min(len, id * per);
    const hi = @min(len, lo + per);
    var off = lo;
    while (off < hi) : (off += 64) @prefetch(s.hint_ptr + off, .{ .rw = .read, .locality = 3, .cache = .data });
}

fn runChunks(s: *Shared) void {
    while (true) {
        const start = s.next.fetchAdd(s.grain, .monotonic);
        if (start >= s.total) return;
        s.func(s.ctx, start, @min(start + s.grain, s.total));
    }
}

fn worker(s: *Shared, io: std.Io, id: usize) void {
    tls_worker_id = id;
    // Start from the initial generation, not the current one. A dispatch can land before this
    // thread is scheduled; reading the live value here would make it skip that job and leave
    // the caller waiting on it forever.
    var seen: u32 = 0;
    while (true) {
        var spins: u32 = 0;
        var gen = s.generation.load(.acquire);
        while (gen == seen) {
            if (spins < spin_before_park) {
                std.atomic.spinLoopHint();
                spins += 1;
            } else {
                _ = s.parked.fetchAdd(1, .seq_cst);
                if (s.generation.load(.seq_cst) == seen) io.futexWaitUncancelable(u32, &s.generation.raw, seen);
                _ = s.parked.fetchSub(1, .seq_cst);
            }
            gen = s.generation.load(.acquire);
        }
        seen = gen;
        if (s.stop.load(.acquire)) return;
        runChunks(s);
        _ = s.remaining.fetchSub(1, .release);
        prefetchStripe(s, id);
    }
}

fn sumTask(ctx: *anyopaque, start: usize, end: usize) void {
    const total: *std.atomic.Value(u64) = @ptrCast(@alignCast(ctx));
    var local: u64 = 0;
    for (start..end) |i| local += i;
    _ = total.fetchAdd(local, .monotonic);
}

test "parallelFor covers every index exactly once" {
    var pool = try Pool.init(std.testing.allocator, 4);
    defer pool.deinit();
    var total = std.atomic.Value(u64).init(0);
    const n: usize = 100_003;
    pool.parallelFor(n, 64, &total, sumTask);
    try std.testing.expectEqual(@as(u64, n * (n - 1) / 2), total.load(.monotonic));
}

test "repeated dispatch stays correct" {
    var pool = try Pool.init(std.testing.allocator, 6);
    defer pool.deinit();
    for (0..2000) |round| {
        var total = std.atomic.Value(u64).init(0);
        const n: usize = 1000 + round;
        pool.parallelFor(n, 17, &total, sumTask);
        try std.testing.expectEqual(@as(u64, n * (n - 1) / 2), total.load(.monotonic));
    }
}

test "single thread pool runs inline" {
    var pool = try Pool.init(std.testing.allocator, 1);
    defer pool.deinit();
    var total = std.atomic.Value(u64).init(0);
    pool.parallelFor(10, 1, &total, sumTask);
    try std.testing.expectEqual(@as(u64, 45), total.load(.monotonic));
}

test "workers park and wake after idling" {
    var pool = try Pool.init(std.testing.allocator, 3);
    defer pool.deinit();
    var total = std.atomic.Value(u64).init(0);
    pool.parallelFor(500, 8, &total, sumTask);
    // Let workers exhaust their spin budget and park, then dispatch again.
    var spin: u64 = 0;
    for (0..5_000_000) |i| spin +%= i;
    std.mem.doNotOptimizeAway(spin);
    total.store(0, .monotonic);
    pool.parallelFor(500, 8, &total, sumTask);
    try std.testing.expectEqual(@as(u64, 500 * 499 / 2), total.load(.monotonic));
}
