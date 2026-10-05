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
    /// Segment (layer, s) holds positions [s * seg_len, (s + 1) * seg_len) as
    /// [kv_head][pos][head_dim]. Empty until the context first reaches it.
    k_segs: [][]f16,
    v_segs: [][]f16,

    pub fn init(allocator: std.mem.Allocator, cfg: config_mod.Config, n_ctx: usize) !KvCache {
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
            for (0..cfg.n_layers) |l| {
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
            for (0..self.cfg.n_layers) |l| {
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
