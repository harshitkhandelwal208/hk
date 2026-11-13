//! KV cache for one sequence.
//!
//! Keys and values are stored in f16 and grow in segments of `seg_len` positions as the context
//! fills, so resident memory follows the real context length and never the configured maximum,
//! whatever the kernel's transparent huge page setting is. A model serves many sequences by
//! giving each its own `KvCache`; the weights are shared.
//!
//! Layout inside a segment, per kv head: keys are stored transposed in tiles of `key_tile`
//! positions, [tile][dim][position in tile], so the attention kernel can score sixteen
//! positions per vector without a horizontal sum and reads a tile as one contiguous run;
//! values stay [position][dim], which is what the weighted sum wants.

const std = @import("std");
const config_mod = @import("config.zig");

/// Positions per segment. Must be a multiple of the attention tile so a tile never spans two.
pub const seg_len = 256;

/// Key positions per transposed tile; the attention kernel's tile.
pub const key_tile = 64;

/// One token of one sequence in a forward pass. A pass may mix several sequences (each with its
/// own cache) so a server can decode one token for every active session in a single sweep over
/// the weights. Items that share a cache must appear in increasing position order.
pub const Item = struct {
    kv: *KvCache,
    token: u32,
    pos: u32,
};

pub const KvCache = struct {
    allocator: std.mem.Allocator,
    cfg: config_mod.Config,
    n_ctx: usize,
    n_segs: usize,
    /// Head width rounded up to whole 16 lane vectors; the padding is zero.
    hd_pad: usize,
    /// Number of segments allocated so far (per layer).
    allocated: usize = 0,
    /// Layers below this index live on a device (partial offload) and have no host cache.
    first_layer: usize = 0,
    /// Segment (layer, s) holds positions [s * seg_len, (s + 1) * seg_len) as
    /// [kv_head][pos][head_dim]. Empty until the context first reaches it.
    k_segs: [][]f16,
    v_segs: [][]f16,

    pub fn init(allocator: std.mem.Allocator, cfg: config_mod.Config, n_ctx: usize) !KvCache {
        return initFrom(allocator, cfg, n_ctx, 0);
    }

    /// A cache for layers `first_layer ..` only: the layers before it run on a device that keeps its own cache.
    pub fn initFrom(allocator: std.mem.Allocator, cfg: config_mod.Config, n_ctx: usize, first_layer: usize) !KvCache {
        std.debug.assert(first_layer <= cfg.n_layers);
        const n_segs = (n_ctx + seg_len - 1) / seg_len;
        const k = try allocator.alloc([]f16, cfg.n_layers * n_segs);
        errdefer allocator.free(k);
        const v = try allocator.alloc([]f16, cfg.n_layers * n_segs);
        @memset(k, &.{});
        @memset(v, &.{});
        return .{
            .allocator = allocator,
            .cfg = cfg,
            .n_ctx = n_ctx,
            .n_segs = n_segs,
            .hd_pad = (cfg.head_dim + 15) / 16 * 16,
            .first_layer = first_layer,
            .k_segs = k,
            .v_segs = v,
        };
    }

    pub fn deinit(self: *KvCache) void {
        for (self.k_segs) |sg| if (sg.len != 0) self.allocator.free(sg);
        for (self.v_segs) |sg| if (sg.len != 0) self.allocator.free(sg);
        self.allocator.free(self.k_segs);
        self.allocator.free(self.v_segs);
    }

    /// Bytes allocated so far. Grows with the context, never past `n_ctx`.
    pub fn bytes(self: *const KvCache) usize {
        var n: usize = 0;
        for (self.k_segs) |sg| n += sg.len;
        for (self.v_segs) |sg| n += sg.len;
        return n * @sizeOf(f16);
    }

    /// Makes sure every layer has segments covering positions below `end`.
    pub fn ensure(self: *KvCache, end: usize) !void {
        const cfg = self.cfg;
        const need = (end + seg_len - 1) / seg_len;
        const seg_elems = cfg.n_kv_heads * seg_len * self.hd_pad;
        var s: usize = self.allocated;
        while (s < need) : (s += 1) {
            for (self.first_layer..cfg.n_layers) |l| {
                const i = l * self.n_segs + s;
                self.k_segs[i] = try self.allocator.alloc(f16, seg_elems);
                self.v_segs[i] = self.allocator.alloc(f16, seg_elems) catch |e| {
                    self.allocator.free(self.k_segs[i]);
                    self.k_segs[i] = &.{};
                    return e;
                };
                if (self.hd_pad != cfg.head_dim) {
                    @memset(self.k_segs[i], 0);
                    @memset(self.v_segs[i], 0);
                }
            }
            self.allocated = s + 1;
        }
    }

    /// Start of the key tile at position `p0` (a multiple of `key_tile`) of head `g`:
    /// element (d, j) is `ptr[d * key_tile + j]` for the positions p0 + j of the tile.
    pub inline fn kTile(self: *const KvCache, layer: usize, g: usize, p0: usize) [*]const f16 {
        const seg = self.k_segs[layer * self.n_segs + p0 / seg_len];
        return seg.ptr + g * self.hd_pad * seg_len + (p0 % seg_len) * self.hd_pad;
    }

    /// Value rows from position `p0` on, [position][hd_pad], for head `g`.
    pub inline fn vRows(self: *const KvCache, layer: usize, g: usize, p0: usize) [*]const f16 {
        const seg = self.v_segs[layer * self.n_segs + p0 / seg_len];
        return seg.ptr + (g * seg_len + p0 % seg_len) * self.hd_pad;
    }

    /// Stores the key and value of head `g` at `pos`. Both come in as f32.
    pub fn write(self: *const KvCache, layer: usize, g: usize, pos: usize, k: []const f32, v: []const f32) void {
        const hd = self.cfg.head_dim;
        const kseg = self.k_segs[layer * self.n_segs + pos / seg_len];
        const vseg = self.v_segs[layer * self.n_segs + pos / seg_len];
        const j = pos % seg_len;
        const kbase = g * self.hd_pad * seg_len + (j / key_tile) * key_tile * self.hd_pad + j % key_tile;
        for (0..hd) |d| kseg[kbase + d * key_tile] = @floatCast(k[d]);
        const vbase = (g * seg_len + j) * self.hd_pad;
        for (0..hd) |d| vseg[vbase + d] = @floatCast(v[d]);
    }

    /// Frees everything above `keep` positions, for a sequence that is rewound.
    pub fn truncate(self: *KvCache, keep: usize) void {
        const need = (keep + seg_len - 1) / seg_len;
        var s = self.allocated;
        while (s > need) {
            s -= 1;
            for (self.first_layer..self.cfg.n_layers) |l| {
                const i = l * self.n_segs + s;
                if (self.k_segs[i].len != 0) self.allocator.free(self.k_segs[i]);
                if (self.v_segs[i].len != 0) self.allocator.free(self.v_segs[i]);
                self.k_segs[i] = &.{};
                self.v_segs[i] = &.{};
            }
        }
        self.allocated = @min(self.allocated, need);
    }
};

/// Fixed-size block/page tokens for paged attention and prefix caching (plan §80, §150 milestone 2.3).
pub const block_tokens = 32;

/// A physical page holding keys and values for `block_tokens` across all layers.
pub const Page = struct {
    k: []f16,
    v: []f16,
    ref_count: u32 = 0,
};

/// Logical-to-physical block table for one sequence.
pub const BlockTable = struct {
    allocator: std.mem.Allocator,
    page_ids: std.ArrayList(u32) = .empty,

    pub fn init(allocator: std.mem.Allocator) BlockTable {
        return .{ .allocator = allocator, .page_ids = .empty };
    }

    pub fn deinit(self: *BlockTable) void {
        self.page_ids.deinit(self.allocator);
    }

    pub fn numPages(self: *const BlockTable) usize {
        return self.page_ids.items.len;
    }
};

/// Global or engine-level Paged KV Cache Manager (plan §80).
pub const PagedKvManager = struct {
    allocator: std.mem.Allocator,
    cfg: config_mod.Config,
    hd_pad: usize,
    page_elems: usize,
    max_pages: usize,
    pages: []Page,
    free_list: std.ArrayList(u32) = .empty,
    allocated_pages: usize = 0,

    pub fn init(allocator: std.mem.Allocator, cfg: config_mod.Config, max_pages: usize) !PagedKvManager {
        const hd_pad = (cfg.head_dim + 15) / 16 * 16;
        const page_elems = cfg.n_layers * cfg.n_kv_heads * block_tokens * hd_pad;
        const pages = try allocator.alloc(Page, max_pages);
        errdefer allocator.free(pages);
        @memset(pages, .{ .k = &.{}, .v = &.{}, .ref_count = 0 });

        var free_list: std.ArrayList(u32) = .empty;
        try free_list.ensureTotalCapacity(allocator, max_pages);
        for (0..max_pages) |i| {
            try free_list.append(allocator, @intCast(max_pages - 1 - i));
        }

        return .{
            .allocator = allocator,
            .cfg = cfg,
            .hd_pad = hd_pad,
            .page_elems = page_elems,
            .max_pages = max_pages,
            .pages = pages,
            .free_list = free_list,
        };
    }

    pub fn deinit(self: *PagedKvManager) void {
        for (self.pages) |*p| {
            if (p.k.len != 0) self.allocator.free(p.k);
            if (p.v.len != 0) self.allocator.free(p.v);
        }
        self.allocator.free(self.pages);
        self.free_list.deinit(self.allocator);
    }

    /// Allocates a new physical page from the pool.
    pub fn allocPage(self: *PagedKvManager) !u32 {
        const id = self.free_list.pop() orelse return error.OutOfMemory;
        var p = &self.pages[id];
        if (p.k.len == 0) {
            p.k = try self.allocator.alloc(f16, self.page_elems);
            p.v = self.allocator.alloc(f16, self.page_elems) catch |e| {
                self.allocator.free(p.k);
                p.k = &.{};
                self.free_list.append(self.allocator, id) catch {};
                return e;
            };
        }
        p.ref_count = 1;
        self.allocated_pages += 1;
        return id;
    }

    /// Increments reference count for prefix caching / shared prompt fork.
    pub fn retain(self: *PagedKvManager, page_id: u32) void {
        std.debug.assert(page_id < self.max_pages);
        self.pages[page_id].ref_count += 1;
    }

    /// Releases a page; when reference count reaches zero it returns to the free pool.
    pub fn release(self: *PagedKvManager, page_id: u32) void {
        std.debug.assert(page_id < self.max_pages);
        var p = &self.pages[page_id];
        if (p.ref_count > 1) {
            p.ref_count -= 1;
            return;
        }
        p.ref_count = 0;
        self.free_list.append(self.allocator, page_id) catch {};
        self.allocated_pages -|= 1;
    }

    /// Forks a sequence's block table, sharing all prefix pages (copy-on-write).
    pub fn forkSequence(self: *PagedKvManager, src: *const BlockTable, dst: *BlockTable) !void {
        try dst.page_ids.resize(dst.allocator, src.page_ids.items.len);
        for (src.page_ids.items, 0..) |pid, i| {
            self.retain(pid);
            dst.page_ids.items[i] = pid;
        }
    }

    /// Frees all pages belonging to a sequence.
    pub fn freeSequence(self: *PagedKvManager, bt: *BlockTable) void {
        for (bt.page_ids.items) |pid| {
            self.release(pid);
        }
        bt.page_ids.clearRetainingCapacity();
    }

    /// Writes key and value vectors into a sequence's paged cache.
    pub fn write(self: *PagedKvManager, bt: *BlockTable, layer: usize, head: usize, pos: usize, k: []const f32, v: []const f32) !void {
        const block_idx = pos / block_tokens;
        const block_offset = pos % block_tokens;
        while (bt.page_ids.items.len <= block_idx) {
            const pid = try self.allocPage();
            try bt.page_ids.append(bt.allocator, pid);
        }
        const pid = bt.page_ids.items[block_idx];
        const page = &self.pages[pid];
        const hd = self.cfg.head_dim;
        const hdp = self.hd_pad;
        const base = (layer * self.cfg.n_kv_heads + head) * block_tokens * hdp + block_offset * hdp;
        for (0..hd) |d| {
            page.k[base + d] = @floatCast(k[d]);
            page.v[base + d] = @floatCast(v[d]);
        }
    }

    /// Reads key and value vectors from a sequence's paged cache.
    pub fn read(self: *const PagedKvManager, bt: *const BlockTable, layer: usize, head: usize, pos: usize, k_out: []f32, v_out: []f32) bool {
        const block_idx = pos / block_tokens;
        const block_offset = pos % block_tokens;
        if (block_idx >= bt.page_ids.items.len) return false;
        const pid = bt.page_ids.items[block_idx];
        const page = &self.pages[pid];
        if (page.k.len == 0 or page.ref_count == 0) return false;
        const hd = self.cfg.head_dim;
        const hdp = self.hd_pad;
        const base = (layer * self.cfg.n_kv_heads + head) * block_tokens * hdp + block_offset * hdp;
        for (0..hd) |d| {
            k_out[d] = @floatCast(page.k[base + d]);
            v_out[d] = @floatCast(page.v[base + d]);
        }
        return true;
    }

    /// Total memory currently allocated for paged KV storage in bytes.
    pub fn residentBytes(self: *const PagedKvManager) usize {
        return self.allocated_pages * self.page_elems * 2 * @sizeOf(f16);
    }
};

test "PagedKvManager allocation, prefix sharing, and release" {
    const allocator = std.testing.allocator;
    const cfg = config_mod.Config{
        .arch = .llama,
        .n_layers = 2,
        .dim = 64,
        .ffn_dim = 128,
        .n_heads = 4,
        .n_kv_heads = 2,
        .head_dim = 16,
        .vocab = 100,
        .ctx_train = 2048,
        .rms_eps = 1e-5,
        .rope_base = 10000.0,
        .rope_dim = 16,
        .rope_style = .neox,
        .rope_scaling = .{},
        .ffn_act = .silu,
        .qk_norm = false,
        .qkv_bias = false,
    };

    var mgr = try PagedKvManager.init(allocator, cfg, 16);
    defer mgr.deinit();

    var seq1 = BlockTable.init(allocator);
    defer seq1.deinit();

    const k_in = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
    const v_in = [_]f32{ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5 };

    try mgr.write(&seq1, 0, 0, 0, &k_in, &v_in);
    try std.testing.expectEqual(@as(usize, 1), seq1.numPages());
    try std.testing.expectEqual(@as(usize, 1), mgr.allocated_pages);

    var k_out: [16]f32 = undefined;
    var v_out: [16]f32 = undefined;
    try std.testing.expect(mgr.read(&seq1, 0, 0, 0, &k_out, &v_out));
    try std.testing.expectEqualSlices(f32, &k_in, &k_out);
    try std.testing.expectEqualSlices(f32, &v_in, &v_out);

    // Fork sequence (prefix sharing)
    var seq2 = BlockTable.init(allocator);
    defer seq2.deinit();
    try mgr.forkSequence(&seq1, &seq2);
    try std.testing.expectEqual(@as(usize, 1), seq2.numPages());
    try std.testing.expectEqual(@as(u32, 2), mgr.pages[seq1.page_ids.items[0]].ref_count);

    // Free seq1; page stays alive for seq2
    mgr.freeSequence(&seq1);
    try std.testing.expectEqual(@as(u32, 1), mgr.pages[seq2.page_ids.items[0]].ref_count);
    try std.testing.expectEqual(@as(usize, 1), mgr.allocated_pages);

    // Free seq2; page returns to pool
    mgr.freeSequence(&seq2);
    try std.testing.expectEqual(@as(usize, 0), mgr.allocated_pages);
}
