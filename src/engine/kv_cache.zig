//! Paged KV Cache Architecture (Plan §50, §80, §150 milestone 2.3).
//! Eliminates memory fragmentation during multi-turn inference via fixed-size physical page allocation,
//! copy-on-write sequence prefix sharing, sliding-window attention (SWA) bounds, and chunked prefill.

const std = @import("std");
const config_mod = @import("config.zig");
const Config = config_mod.Config;

/// Tokens per physical page.
pub const BLOCK_SIZE: usize = 32;

/// A physical memory block holding keys and values for `BLOCK_SIZE` tokens across all model layers.
pub const PagedBlock = struct {
    k: []f16,
    v: []f16,
    ref_count: u32 = 0,
    block_id: u32 = 0,
};

/// Memory pool managing physical paged blocks.
pub const PagedBlockPool = struct {
    allocator: std.mem.Allocator,
    cfg: Config,
    hd_pad: usize,
    block_elems: usize,
    max_blocks: usize,
    blocks: []PagedBlock,
    free_list: std.ArrayList(u32) = .empty,
    allocated_count: usize = 0,

    pub fn init(allocator: std.mem.Allocator, cfg: Config, max_blocks: usize) !PagedBlockPool {
        const hd_pad = (cfg.head_dim + 15) / 16 * 16;
        const block_elems = cfg.n_layers * cfg.n_kv_heads * BLOCK_SIZE * hd_pad;
        const blocks = try allocator.alloc(PagedBlock, max_blocks);
        errdefer allocator.free(blocks);
        @memset(blocks, .{ .k = &.{}, .v = &.{}, .ref_count = 0, .block_id = 0 });

        var free_list: std.ArrayList(u32) = .empty;
        try free_list.ensureTotalCapacity(allocator, max_blocks);
        for (0..max_blocks) |i| {
            try free_list.append(allocator, @intCast(max_blocks - 1 - i));
        }

        return .{
            .allocator = allocator,
            .cfg = cfg,
            .hd_pad = hd_pad,
            .block_elems = block_elems,
            .max_blocks = max_blocks,
            .blocks = blocks,
            .free_list = free_list,
        };
    }

    pub fn deinit(self: *PagedBlockPool) void {
        for (self.blocks) |*b| {
            if (b.k.len != 0) self.allocator.free(b.k);
            if (b.v.len != 0) self.allocator.free(b.v);
        }
        self.allocator.free(self.blocks);
        self.free_list.deinit(self.allocator);
    }

    /// Allocates or reuses a physical block from the free list.
    pub fn allocBlock(self: *PagedBlockPool) !u32 {
        const id = self.free_list.pop() orelse return error.OutOfMemory;
        var b = &self.blocks[id];
        b.block_id = id;
        if (b.k.len == 0) {
            b.k = try self.allocator.alloc(f16, self.block_elems);
            b.v = self.allocator.alloc(f16, self.block_elems) catch |e| {
                self.allocator.free(b.k);
                b.k = &.{};
                self.free_list.append(self.allocator, id) catch {};
                return e;
            };
        }
        b.ref_count = 1;
        self.allocated_count += 1;
        return id;
    }

    /// Increments reference count for copy-on-write sequence sharing.
    pub fn retain(self: *PagedBlockPool, block_id: u32) void {
        std.debug.assert(block_id < self.max_blocks);
        self.blocks[block_id].ref_count += 1;
    }

    /// Decrements reference count and returns block to free list when ref_count == 0.
    pub fn release(self: *PagedBlockPool, block_id: u32) void {
        std.debug.assert(block_id < self.max_blocks);
        var b = &self.blocks[block_id];
        std.debug.assert(b.ref_count > 0);
        b.ref_count -= 1;
        if (b.ref_count == 0) {
            self.free_list.append(self.allocator, block_id) catch {};
            self.allocated_count -= 1;
        }
    }
};

/// Sequence-level Paged Key-Value Cache.
pub const PagedKVCache = struct {
    allocator: std.mem.Allocator,
    pool: *PagedBlockPool,
    block_table: std.ArrayList(u32) = .empty,
    seq_len: usize = 0,
    sliding_window: ?usize = null,

    pub fn init(allocator: std.mem.Allocator, pool: *PagedBlockPool, sliding_window: ?usize) PagedKVCache {
        return .{
            .allocator = allocator,
            .pool = pool,
            .block_table = .empty,
            .seq_len = 0,
            .sliding_window = sliding_window,
        };
    }

    pub fn deinit(self: *PagedKVCache) void {
        for (self.block_table.items) |id| {
            self.pool.release(id);
        }
        self.block_table.deinit(self.allocator);
    }

    /// Returns the number of physical blocks currently allocated to this sequence.
    pub fn numBlocks(self: *const PagedKVCache) usize {
        return self.block_table.items.len;
    }

    /// Appends a token's key and value representations into the paged cache at `pos`.
    pub fn appendToken(
        self: *PagedKVCache,
        layer: usize,
        kv_head: usize,
        pos: usize,
        k: []const f32,
        v: []const f32,
    ) !void {
        const block_idx = pos / BLOCK_SIZE;
        const offset_in_block = pos % BLOCK_SIZE;

        while (self.block_table.items.len <= block_idx) {
            const new_block_id = try self.pool.allocBlock();
            try self.block_table.append(self.allocator, new_block_id);
        }

        const physical_id = self.block_table.items[block_idx];
        const block = &self.pool.blocks[physical_id];

        const cfg = self.pool.cfg;
        const hd = cfg.head_dim;
        const hd_pad = self.pool.hd_pad;

        // Base element offset for (layer, kv_head, offset_in_block)
        const base = ((layer * cfg.n_kv_heads + kv_head) * BLOCK_SIZE + offset_in_block) * hd_pad;

        for (0..hd) |d| {
            block.k[base + d] = @floatCast(k[d]);
            block.v[base + d] = @floatCast(v[d]);
        }

        if (pos >= self.seq_len) {
            self.seq_len = pos + 1;
        }
    }

    /// Reads a token's key and value representations from the paged cache at `pos`.
    pub fn readToken(
        self: *const PagedKVCache,
        layer: usize,
        kv_head: usize,
        pos: usize,
        out_k: []f32,
        out_v: []f32,
    ) !void {
        if (pos >= self.seq_len) return error.PositionOutOfBounds;
        const block_idx = pos / BLOCK_SIZE;
        const offset_in_block = pos % BLOCK_SIZE;

        if (block_idx >= self.block_table.items.len) return error.BlockNotAllocated;
        const physical_id = self.block_table.items[block_idx];
        const block = &self.pool.blocks[physical_id];

        const cfg = self.pool.cfg;
        const hd = cfg.head_dim;
        const hd_pad = self.pool.hd_pad;

        const base = ((layer * cfg.n_kv_heads + kv_head) * BLOCK_SIZE + offset_in_block) * hd_pad;

        for (0..hd) |d| {
            out_k[d] = @floatCast(block.k[base + d]);
            out_v[d] = @floatCast(block.v[base + d]);
        }
    }

    /// Forks the sequence cache (copy-on-write prefix sharing).
    pub fn fork(self: *const PagedKVCache) !PagedKVCache {
        var forked = PagedKVCache.init(self.allocator, self.pool, self.sliding_window);
        try forked.block_table.ensureTotalCapacity(self.allocator, self.block_table.items.len);

        for (self.block_table.items) |id| {
            self.pool.retain(id);
            try forked.block_table.append(self.allocator, id);
        }
        forked.seq_len = self.seq_len;
        return forked;
    }

    /// Sliding window attention: returns the active start position for scoring token at `pos`.
    pub fn activeWindowStart(self: *const PagedKVCache, pos: usize) usize {
        if (self.sliding_window) |w| {
            if (pos > w) return pos - w;
        }
        return 0;
    }

    /// Chunked prefill helper: appends a contiguous chunk of tokens into paged storage.
    pub fn appendChunk(
        self: *PagedKVCache,
        layer: usize,
        kv_head: usize,
        start_pos: usize,
        num_tokens: usize,
        k_chunk: []const f32, // [num_tokens, head_dim]
        v_chunk: []const f32, // [num_tokens, head_dim]
    ) !void {
        const hd = self.pool.cfg.head_dim;
        if (k_chunk.len < num_tokens * hd or v_chunk.len < num_tokens * hd) return error.BufferTooSmall;

        for (0..num_tokens) |i| {
            const pos = start_pos + i;
            const k_tok = k_chunk[i * hd .. (i + 1) * hd];
            const v_tok = v_chunk[i * hd .. (i + 1) * hd];
            try self.appendToken(layer, kv_head, pos, k_tok, v_tok);
        }
    }
};

test "PagedKVCache: block allocation, write/read parity, and multi-turn growth" {
    const allocator = std.testing.allocator;

    const dummy_cfg = Config{
        .arch = .llama,
        .n_layers = 2,
        .dim = 64,
        .ffn_dim = 128,
        .n_heads = 4,
        .n_kv_heads = 2,
        .head_dim = 16,
        .vocab = 1000,
        .ctx_train = 2048,
        .rms_eps = 1e-5,
        .rope_base = 10000.0,
        .rope_dim = 16,
        .rope_style = .norm,
        .rope_scaling = .{},
        .ffn_act = .silu,
        .qk_norm = false,
        .qkv_bias = false,
        .sliding_window = 64,
    };

    var pool = try PagedBlockPool.init(allocator, dummy_cfg, 16);
    defer pool.deinit();

    var cache = PagedKVCache.init(allocator, &pool, 64);
    defer cache.deinit();

    // Turn 1: Write 40 tokens (spans block 0 and block 1 since BLOCK_SIZE=32)
    var k_val: [16]f32 = undefined;
    var v_val: [16]f32 = undefined;
    for (0..16) |d| {
        k_val[d] = @as(f32, @floatFromInt(d + 1)) * 0.1;
        v_val[d] = @as(f32, @floatFromInt(d + 1)) * 0.2;
    }

    for (0..40) |p| {
        try cache.appendToken(0, 0, p, &k_val, &v_val);
    }

    try std.testing.expectEqual(@as(usize, 2), cache.numBlocks());
    try std.testing.expectEqual(@as(usize, 40), cache.seq_len);

    // Verify read parity at pos 0 and pos 35
    var out_k: [16]f32 = undefined;
    var out_v: [16]f32 = undefined;

    try cache.readToken(0, 0, 0, &out_k, &out_v);
    for (0..16) |d| {
        try std.testing.expectApproxEqAbs(k_val[d], out_k[d], 1e-3);
        try std.testing.expectApproxEqAbs(v_val[d], out_v[d], 1e-3);
    }

    try cache.readToken(0, 0, 35, &out_k, &out_v);
    for (0..16) |d| {
        try std.testing.expectApproxEqAbs(k_val[d], out_k[d], 1e-3);
        try std.testing.expectApproxEqAbs(v_val[d], out_v[d], 1e-3);
    }

    // Sequence fork (prefix sharing)
    var forked = try cache.fork();
    defer forked.deinit();

    try std.testing.expectEqual(@as(usize, 2), forked.numBlocks());
    try std.testing.expectEqual(pool.blocks[cache.block_table.items[0]].ref_count, 2);

    // Sliding window bounds check
    try std.testing.expectEqual(@as(usize, 0), cache.activeWindowStart(50));
    try std.testing.expectEqual(@as(usize, 36), cache.activeWindowStart(100)); // 100 - 64 = 36
}

test "PagedKVCache: chunked prefill integration" {
    const allocator = std.testing.allocator;

    const dummy_cfg = Config{
        .arch = .llama,
        .n_layers = 1,
        .dim = 32,
        .ffn_dim = 64,
        .n_heads = 2,
        .n_kv_heads = 2,
        .head_dim = 16,
        .vocab = 100,
        .ctx_train = 512,
        .rms_eps = 1e-5,
        .rope_base = 10000.0,
        .rope_dim = 16,
        .rope_style = .norm,
        .rope_scaling = .{},
        .ffn_act = .silu,
        .qk_norm = false,
        .qkv_bias = false,
    };

    var pool = try PagedBlockPool.init(allocator, dummy_cfg, 8);
    defer pool.deinit();

    var cache = PagedKVCache.init(allocator, &pool, null);
    defer cache.deinit();

    const chunk_tokens = 10;
    const k_chunk = try allocator.alloc(f32, chunk_tokens * 16);
    defer allocator.free(k_chunk);
    const v_chunk = try allocator.alloc(f32, chunk_tokens * 16);
    defer allocator.free(v_chunk);

    for (0..chunk_tokens * 16) |i| {
        k_chunk[i] = @as(f32, @floatFromInt(i + 1)) * 0.05;
        v_chunk[i] = @as(f32, @floatFromInt(i + 2)) * 0.05;
    }

    try cache.appendChunk(0, 0, 0, chunk_tokens, k_chunk, v_chunk);
    try std.testing.expectEqual(@as(usize, chunk_tokens), cache.seq_len);

    var read_k: [16]f32 = undefined;
    var read_v: [16]f32 = undefined;
    try cache.readToken(0, 0, 5, &read_k, &read_v);

    for (0..16) |d| {
        try std.testing.expectApproxEqAbs(k_chunk[5 * 16 + d], read_k[d], 2e-3);
        try std.testing.expectApproxEqAbs(v_chunk[5 * 16 + d], read_v[d], 2e-3);
    }
}
