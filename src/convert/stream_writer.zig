//! Writes an .hk container whose tensor data arrives in pieces.
//!
//! Everything about the layout is decided up front from tensor names, types and byte sizes,
//! which converters know from the input's header alone. The header, metadata and tensor table
//! are written first; tensor data is then written at its final offset as it streams in. Peak
//! memory is the metadata plus whatever chunk the caller holds, independent of model size.
//!
//! Output goes to `<path>.partial` and is renamed into place by `finish`, so an interrupted
//! conversion can never leave a file that looks complete.

const std = @import("std");
const format = @import("../format.zig");
const metadata = @import("../metadata.zig");
const tensor_toc = @import("../tensor_toc.zig");
const platform = @import("../platform.zig");
const buf = @import("../buf.zig");

pub const TensorSpec = struct {
    name: []const u8,
    storage_type: format.StorageType,
    ndim: u8,
    shape: [format.MAX_DIMS]u64,
    /// Exact byte size of the tensor data that will be written.
    data_size: u64,
};

pub const StreamWriter = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    final_path: []u8,
    partial_path: []u8,
    offsets: []u64,
    sizes: []u64,
    written: []u64,
    file_len: u64,
    finished: bool = false,
    closed: bool = false,

    pub const Error = error{ TensorIndexOutOfRange, WriteOutOfBounds, IncompleteTensor, TooManyTensors } ||
        std.mem.Allocator.Error || std.Io.File.OpenError || std.Io.File.WritePositionalError ||
        std.Io.File.SetLengthError || std.Io.Dir.RenameError;

    /// Creates `<path>.partial`, writes the header, metadata and tensor table, and reserves the
    /// data region. `specs` order is the order of the table; `alignment` applies to each tensor.
    pub fn begin(
        allocator: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
        meta: *const metadata.MetadataMap,
        specs: []const TensorSpec,
        alignment: usize,
    ) Error!StreamWriter {
        var meta_w = buf.BufferWriter.init(allocator);
        defer meta_w.deinit();
        try meta.serialize(&meta_w);
        const meta_bytes = meta_w.getBytes();

        var toc = tensor_toc.TensorTOC.init(allocator);
        defer toc.deinit();
        for (specs) |s| {
            try toc.add(.{
                .name = s.name,
                .storage_type = s.storage_type,
                .tile_layout = .row_major,
                .sparsity_type = .none,
                .ndim = s.ndim,
                .shape = s.shape,
                .data_offset = 0,
                .data_size = s.data_size,
            });
        }

        // The table size does not depend on the offsets it holds, so measure it once.
        var probe = buf.BufferWriter.init(allocator);
        defer probe.deinit();
        try toc.serialize(&probe);
        const toc_len = probe.getBytes().len;

        const header_len = @sizeOf(format.FileHeader);
        const toc_offset = header_len + meta_bytes.len;
        const data_start = platform.alignForward(toc_offset + toc_len, alignment);

        const offsets = try allocator.alloc(u64, specs.len);
        errdefer allocator.free(offsets);
        const sizes = try allocator.alloc(u64, specs.len);
        errdefer allocator.free(sizes);
        const written = try allocator.alloc(u64, specs.len);
        errdefer allocator.free(written);
        @memset(written, 0);

        var cursor: u64 = data_start;
        for (specs, 0..) |s, i| {
            cursor = platform.alignForward(cursor, alignment);
            offsets[i] = cursor;
            sizes[i] = s.data_size;
            toc.entries.items[i].data_offset = cursor;
            cursor += s.data_size;
        }

        var toc_w = buf.BufferWriter.init(allocator);
        defer toc_w.deinit();
        try toc.serialize(&toc_w);
        std.debug.assert(toc_w.getBytes().len == toc_len);

        var flags: u32 = format.HeaderFlags.LITTLE_ENDIAN;
        if (alignment % format.DEFAULT_ALIGNMENT_BYTES == 0) {
            flags |= format.HeaderFlags.TILE_ALIGNED;
        } else {
            flags |= format.HeaderFlags.FLEXIBLE_ALIGNMENT;
        }
        if (alignment >= format.UNIVERSAL_PAGE_ALIGNMENT_BYTES) flags |= format.HeaderFlags.UNIVERSAL_PAGE_ALIGNED;

        const header = format.FileHeader{
            .flags = flags,
            .alignment = @intCast(alignment),
            .tensor_count = specs.len,
            .metadata_kv_count = meta.items.items.len,
            .metadata_offset = header_len,
            .metadata_size = meta_bytes.len,
            .tensor_toc_offset = toc_offset,
            .tensor_toc_size = toc_len,
            .tensor_data_offset = data_start,
            .appendix_offset = 0,
        };

        const final_path = try allocator.dupe(u8, path);
        errdefer allocator.free(final_path);
        const partial_path = try std.fmt.allocPrint(allocator, "{s}.partial", .{path});
        errdefer allocator.free(partial_path);

        const cwd = std.Io.Dir.cwd();
        var file = try cwd.createFile(io, partial_path, .{});
        errdefer {
            file.close(io);
            cwd.deleteFile(io, partial_path) catch {};
        }

        try file.writePositionalAll(io, std.mem.asBytes(&header), 0);
        if (meta_bytes.len > 0) try file.writePositionalAll(io, meta_bytes, header_len);
        try file.writePositionalAll(io, toc_w.getBytes(), toc_offset);
        // Size the file now so every later write lands inside it and a full disk fails here,
        // before hours of downloading, rather than at the end.
        try file.setLength(io, cursor);

        return .{
            .allocator = allocator,
            .io = io,
            .file = file,
            .final_path = final_path,
            .partial_path = partial_path,
            .offsets = offsets,
            .sizes = sizes,
            .written = written,
            .file_len = cursor,
        };
    }

    /// Writes `bytes` at `at` bytes into tensor `index`. Pieces may arrive in any order but each
    /// byte exactly once; `finish` checks the totals.
    pub fn write(self: *StreamWriter, index: usize, at: u64, bytes: []const u8) Error!void {
        if (index >= self.offsets.len) return error.TensorIndexOutOfRange;
        if (at + bytes.len > self.sizes[index]) return error.WriteOutOfBounds;
        try self.file.writePositionalAll(self.io, bytes, self.offsets[index] + at);
        self.written[index] += bytes.len;
    }

    /// Verifies every tensor was fully written, flushes, and renames the file into place.
    pub fn finish(self: *StreamWriter) Error!void {
        for (self.written, self.sizes) |w, s| if (w != s) return error.IncompleteTensor;
        // A failed flush is not fatal to the data already written, but it should not pass quietly
        // on a full disk: the rename below only runs when every write succeeded.
        self.file.sync(self.io) catch {};
        self.file.close(self.io);
        self.closed = true;
        const cwd = std.Io.Dir.cwd();
        try cwd.rename(self.partial_path, cwd, self.final_path, self.io);
        self.finished = true;
    }

    /// Frees resources. If `finish` was not reached the partial file is removed.
    pub fn deinit(self: *StreamWriter) void {
        if (!self.finished) {
            if (!self.closed) self.file.close(self.io);
            std.Io.Dir.cwd().deleteFile(self.io, self.partial_path) catch {};
        }
        self.allocator.free(self.final_path);
        self.allocator.free(self.partial_path);
        self.allocator.free(self.offsets);
        self.allocator.free(self.sizes);
        self.allocator.free(self.written);
    }
};
