//! Streaming GGUF to HK conversion.
//!
//! The header is parsed through a fixed buffer, the tensor table is sized from the quant
//! geometry, and tensor data is copied in fixed size chunks straight from the source to its
//! final offset in the output. Nothing is mapped or decoded and no step needs more than one
//! chunk of tensor data in memory, so converting a 70 GB file costs the same as a 70 MB one.
//!
//! Anything that cannot be represented faithfully is an error that names the culprit. In
//! particular, unknown tensor types are refused rather than reinterpreted, and metadata
//! arrays are converted to compact blobs instead of being dropped (the vocabulary lives in
//! one).

const std = @import("std");
const format = @import("../format.zig");
const metadata = @import("../metadata.zig");
const quant = @import("../quant.zig");
const arrays = @import("../arrays.zig");
const source_mod = @import("source.zig");
const stream_writer = @import("stream_writer.zig");

const Source = source_mod.Source;

pub const GGUF_MAGIC: u32 = 0x46554747;

const ValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
    _,
};

/// GGML tensor type ids and the HK storage type each maps to. Types not listed here have no
/// HK equivalent and make conversion fail.
pub fn storageTypeFor(ggml_type: u32) ?format.StorageType {
    return switch (ggml_type) {
        0 => .f32,
        1 => .f16,
        2 => .q4_0,
        3 => .q4_1,
        6 => .q5_0,
        7 => .q5_1,
        8 => .q8_0,
        10 => .q2_k,
        11 => .q3_k,
        12 => .q4_k,
        13 => .q5_k,
        14 => .q6_k,
        15 => .q8_k,
        16 => .iq2_xxs,
        17 => .iq2_xs,
        18 => .iq3_xxs,
        19 => .iq1_s,
        20 => .iq4_nl,
        21 => .iq3_s,
        22 => .iq2_s,
        23 => .iq4_xs,
        24 => .int8,
        25 => .int16,
        26 => .int32,
        27 => .int64,
        28 => .f64,
        29 => .iq1_m,
        30 => .bf16,
        34 => .tq1_0,
        35 => .tq2_0,
        39 => .mxfp4,
        40 => .nvfp4,
        else => null,
    };
}

pub const Options = struct {
    /// Bytes moved per read/write. 4 MiB keeps syscall overhead low and memory small.
    chunk_bytes: usize = 4 * 1024 * 1024,
    progress: ?*const fn (ctx: ?*anyopaque, done: u64, total: u64) void = null,
    progress_ctx: ?*anyopaque = null,
    /// Where to put a human readable reason when conversion fails.
    diag: ?*Diag = null,
    /// Extra key/value strings added to the output metadata (provenance, for example).
    extra_meta: []const [2][]const u8 = &.{},
    /// Called after every byte has been copied and before the output is renamed into place. A
    /// failure here (a checksum mismatch, say) discards the output.
    before_finish: ?*const fn (ctx: ?*anyopaque) error{VerificationFailed}!void = null,
    before_finish_ctx: ?*anyopaque = null,
};

pub const Diag = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn set(self: *Diag, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(&self.buf, fmt, args) catch self.buf[0..];
        self.len = s.len;
    }

    pub fn message(self: *const Diag) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const ConvertError = error{
    NotGguf,
    UnsupportedGgufVersion,
    UnsupportedTensorType,
    UnsupportedMetadata,
    BadTensorLayout,
    BadHeader,
    ArrayTooLarge,
    VerificationFailed,
} || source_mod.ReadError || stream_writer.StreamWriter.Error || std.mem.Allocator.Error;

const TensorInfo = struct {
    name: []const u8,
    storage: format.StorageType,
    ndim: u8,
    shape: [format.MAX_DIMS]u64,
    offset: u64,
    size: u64,
};

fn fail(opts: Options, comptime fmt: []const u8, args: anytype) void {
    if (opts.diag) |d| d.set(fmt, args);
}

pub fn convert(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: Source,
    out_path: []const u8,
    opts: Options,
) ConvertError!void {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var r = try source_mod.BufReader.init(allocator, src, opts.chunk_bytes);
    defer r.deinit();

    if ((r.readInt(u32) catch return error.NotGguf) != GGUF_MAGIC) return error.NotGguf;
    const version = try r.readInt(u32);
    if (version < 2 or version > 3) {
        fail(opts, "GGUF version {d} is not supported (versions 2 and 3 are)", .{version});
        return error.UnsupportedGgufVersion;
    }
    const tensor_count = try r.readInt(u64);
    const kv_count = try r.readInt(u64);
    // Sanity limits so a corrupt header cannot make us allocate absurd tables.
    if (tensor_count > 1 << 20 or kv_count > 1 << 20) {
        fail(opts, "implausible header: {d} tensors, {d} metadata entries", .{ tensor_count, kv_count });
        return error.BadHeader;
    }

    var meta = metadata.MetadataMap.init(allocator);
    defer meta.deinit();
    var alignment: u64 = 32;

    for (0..kv_count) |_| {
        const key = try readString(&r, arena);
        const vt: ValueType = @enumFromInt(try r.readInt(u32));
        if (vt == .array) {
            try readArray(&r, allocator, arena, &meta, key, opts);
            _ = arena_state.reset(.retain_capacity);
            continue;
        }
        switch (vt) {
            .uint8 => try meta.setInt(key, try r.readInt(u8)),
            .int8 => try meta.setInt(key, try r.readInt(i8)),
            .uint16 => try meta.setInt(key, try r.readInt(u16)),
            .int16 => try meta.setInt(key, try r.readInt(i16)),
            .uint32 => {
                const v = try r.readInt(u32);
                if (std.mem.eql(u8, key, "general.alignment")) alignment = v;
                try meta.setInt(key, v);
            },
            .int32 => try meta.setInt(key, try r.readInt(i32)),
            .float32 => try meta.setFloat(key, @as(f32, @bitCast(try r.readInt(u32)))),
            .bool => try meta.setBool(key, (try r.readInt(u8)) != 0),
            .string => try meta.setString(key, try readString(&r, arena)),
            .uint64 => {
                const v = try r.readInt(u64);
                if (v > std.math.maxInt(i64)) {
                    fail(opts, "metadata '{s}' does not fit in a signed 64 bit integer", .{key});
                    return error.UnsupportedMetadata;
                }
                try meta.setInt(key, @intCast(v));
            },
            .int64 => try meta.setInt(key, try r.readInt(i64)),
            .float64 => try meta.setFloat(key, @as(f64, @bitCast(try r.readInt(u64)))),
            else => {
                fail(opts, "metadata '{s}' has unknown value type {d}", .{ key, @intFromEnum(vt) });
                return error.UnsupportedMetadata;
            },
        }
        _ = arena_state.reset(.retain_capacity);
    }
    if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) {
        fail(opts, "general.alignment {d} is not a power of two", .{alignment});
        return error.BadHeader;
    }

    const tensors = try allocator.alloc(TensorInfo, tensor_count);
    defer allocator.free(tensors);
    var names = std.heap.ArenaAllocator.init(allocator);
    defer names.deinit();

    for (tensors) |*t| {
        t.name = try names.allocator().dupe(u8, try readString(&r, arena));
        const n_dims = try r.readInt(u32);
        if (n_dims == 0 or n_dims > format.MAX_DIMS) {
            fail(opts, "tensor '{s}' has {d} dimensions", .{ t.name, n_dims });
            return error.BadTensorLayout;
        }
        var dims: [format.MAX_DIMS]u64 = @splat(0);
        var n_elems: u64 = 1;
        for (0..n_dims) |i| {
            dims[i] = try r.readInt(u64);
            n_elems = std.math.mul(u64, n_elems, dims[i]) catch {
                fail(opts, "tensor '{s}' element count overflows", .{t.name});
                return error.BadTensorLayout;
            };
        }
        const ggml_type = try r.readInt(u32);
        t.offset = try r.readInt(u64);
        t.storage = storageTypeFor(ggml_type) orelse {
            fail(opts, "tensor '{s}' has GGML type {d}, which HK cannot store. Refusing to guess.", .{ t.name, ggml_type });
            return error.UnsupportedTensorType;
        };
        // GGUF lists dimensions innermost first; HK is row major, outermost first.
        t.ndim = @intCast(n_dims);
        t.shape = @splat(0);
        for (0..n_dims) |i| t.shape[i] = dims[n_dims - 1 - i];
        t.size = quant.byteSize(t.storage, n_elems) catch {
            fail(opts, "tensor '{s}' ({s}) has {d} elements, not a whole number of {s} blocks", .{ t.name, @tagName(t.storage), n_elems, @tagName(t.storage) });
            return error.BadTensorLayout;
        };
    }
    _ = arena_state.reset(.retain_capacity);

    const data_start = std.mem.alignForward(u64, r.pos(), alignment);

    // Tensors are copied in source order so a streaming source is read front to back.
    const order = try allocator.alloc(usize, tensor_count);
    defer allocator.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, tensors, struct {
        fn lt(ts: []TensorInfo, a: usize, b: usize) bool {
            return ts[a].offset < ts[b].offset;
        }
    }.lt);

    var prev_end: u64 = 0;
    var total: u64 = 0;
    for (order) |i| {
        const t = tensors[i];
        if (t.offset < prev_end or t.offset % alignment != 0) {
            fail(opts, "tensor '{s}' at offset {d} overlaps its predecessor or is misaligned", .{ t.name, t.offset });
            return error.BadTensorLayout;
        }
        prev_end = t.offset + t.size;
        total += t.size;
        if (src.len) |len| {
            if (data_start + prev_end > len) {
                fail(opts, "tensor '{s}' ends at byte {d} but the file has {d} bytes (truncated?)", .{ t.name, data_start + prev_end, len });
                return error.BadTensorLayout;
            }
        }
    }

    try meta.setString("hk.source.format", "gguf");
    try meta.setInt("hk.source.gguf_version", version);
    for (opts.extra_meta) |kv| try meta.setString(kv[0], kv[1]);

    const specs = try allocator.alloc(stream_writer.TensorSpec, tensor_count);
    defer allocator.free(specs);
    for (specs, tensors) |*s, t| s.* = .{ .name = t.name, .storage_type = t.storage, .ndim = t.ndim, .shape = t.shape, .data_size = t.size };

    var w = try stream_writer.StreamWriter.begin(allocator, io, out_path, &meta, specs, format.DEFAULT_ALIGNMENT_BYTES);
    defer w.deinit();

    // Copy tensor data through the same sequential reader, so a streaming source is only ever
    // read forward: padding and gaps are skipped, never re-read.
    try r.skip(data_start - r.pos());
    var rel: u64 = 0;
    var done: u64 = 0;
    for (order) |i| {
        const t = tensors[i];
        if (t.offset > rel) {
            try r.skip(t.offset - rel);
            rel = t.offset;
        }
        var at: u64 = 0;
        while (at < t.size) {
            const n: usize = @intCast(@min(r.buf.len, t.size - at));
            try w.write(i, at, try r.take(n));
            at += n;
            rel += n;
            done += n;
            if (opts.progress) |p| p(opts.progress_ctx, done, total);
        }
    }
    if (opts.before_finish) |hook| try hook(opts.before_finish_ctx);
    try w.finish();
}

fn readString(r: *source_mod.BufReader, arena: std.mem.Allocator) (source_mod.ReadError || ConvertError)![]const u8 {
    const len = try r.readInt(u64);
    if (len > 64 * 1024 * 1024) return error.BadHeader;
    const n: usize = @intCast(len);
    if (n <= r.buf.len) {
        // Copy out: the next read may recycle the buffer.
        return arena.dupe(u8, try r.take(n));
    }
    const out = try arena.alloc(u8, n);
    var done: usize = 0;
    while (done < n) {
        const k = @min(r.buf.len, n - done);
        @memcpy(out[done..][0..k], try r.take(k));
        done += k;
    }
    return out;
}

fn readArray(
    r: *source_mod.BufReader,
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    meta: *metadata.MetadataMap,
    key: []const u8,
    opts: Options,
) ConvertError!void {
    const et: ValueType = @enumFromInt(try r.readInt(u32));
    const count = try r.readInt(u64);
    if (count > 1 << 28) {
        fail(opts, "metadata array '{s}' claims {d} elements", .{ key, count });
        return error.BadHeader;
    }
    switch (et) {
        .string => {
            var b = arrays.StringArrayBuilder.init(allocator);
            defer b.deinit();
            for (0..count) |_| {
                const s = try readString(r, arena);
                try b.append(s);
            }
            const blob = try b.finish();
            errdefer allocator.free(blob);
            try meta.set(key, .{ .val_bytes = blob });
        },
        .float32 => {
            const raw = try allocator.alloc(u8, count * 4);
            defer allocator.free(raw);
            var done: usize = 0;
            while (done < raw.len) {
                const k = @min(r.buf.len, raw.len - done);
                @memcpy(raw[done..][0..k], try r.take(k));
                done += k;
            }
            const blob = try arrays.fixedBlob(allocator, .f32, count, raw);
            errdefer allocator.free(blob);
            try meta.set(key, .{ .val_bytes = blob });
        },
        .uint8, .int8, .uint16, .int16, .int32, .uint32, .bool => {
            const raw = try allocator.alloc(u8, count * 4);
            defer allocator.free(raw);
            for (0..count) |i| {
                const v: i64 = switch (et) {
                    .uint8 => try r.readInt(u8),
                    .int8 => try r.readInt(i8),
                    .uint16 => try r.readInt(u16),
                    .int16 => try r.readInt(i16),
                    .int32 => try r.readInt(i32),
                    .uint32 => try r.readInt(u32),
                    .bool => try r.readInt(u8),
                    else => unreachable,
                };
                if (v > std.math.maxInt(i32) or v < std.math.minInt(i32)) {
                    fail(opts, "metadata array '{s}' has a value outside the signed 32 bit range", .{key});
                    return error.UnsupportedMetadata;
                }
                std.mem.writeInt(i32, raw[i * 4 ..][0..4], @intCast(v), .little);
            }
            const blob = try arrays.fixedBlob(allocator, .i32, count, raw);
            errdefer allocator.free(blob);
            try meta.set(key, .{ .val_bytes = blob });
        },
        else => {
            fail(opts, "metadata array '{s}' has element type {d}, which HK cannot store yet", .{ key, @intFromEnum(et) });
            return error.UnsupportedMetadata;
        },
    }
}

/// Convenience wrapper for a local file.
pub fn convertFile(allocator: std.mem.Allocator, io: std.Io, in_path: []const u8, out_path: []const u8, opts: Options) !void {
    var fs = try source_mod.FileSource.open(io, in_path);
    defer fs.close();
    try convert(allocator, io, fs.source(), out_path, opts);
}
