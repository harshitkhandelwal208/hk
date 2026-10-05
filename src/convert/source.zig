//! Where converter input comes from. A converter only ever asks for "these bytes at this
//! offset", so the same code converts a local file or an HTTP resource fetched with range
//! requests, and never needs the whole input in memory.

const std = @import("std");

/// Errors a source can report. Implementations map their own failures onto these so callers
/// can handle them without knowing whether the bytes come from a disk or a network.
pub const ReadError = error{
    ReadFailed,
    UnexpectedEndOfInput,
    NotFound,
    Unauthorized,
    RateLimited,
    NetworkFailure,
    BadResponse,
};

pub const Source = struct {
    ctx: *anyopaque,
    readAtFn: *const fn (ctx: *anyopaque, offset: u64, buf: []u8) ReadError!usize,
    /// Total size when known up front.
    len: ?u64 = null,

    /// Reads up to `buf.len` bytes. Returns fewer only at the end of the data.
    pub fn readAt(self: Source, offset: u64, buf: []u8) ReadError!usize {
        return self.readAtFn(self.ctx, offset, buf);
    }

    pub fn readExact(self: Source, offset: u64, buf: []u8) ReadError!void {
        var done: usize = 0;
        while (done < buf.len) {
            const n = try self.readAt(offset + done, buf[done..]);
            if (n == 0) return error.UnexpectedEndOfInput;
            done += n;
        }
    }
};

pub const FileSource = struct {
    file: std.Io.File,
    io: std.Io,
    size: u64,

    pub fn open(io: std.Io, path: []const u8) !FileSource {
        const cwd = std.Io.Dir.cwd();
        const file = try cwd.openFile(io, path, .{});
        errdefer file.close(io);
        const st = try file.stat(io);
        return .{ .file = file, .io = io, .size = st.size };
    }

    pub fn close(self: *FileSource) void {
        self.file.close(self.io);
    }

    pub fn source(self: *FileSource) Source {
        return .{ .ctx = self, .readAtFn = read, .len = self.size };
    }

    fn read(ctx: *anyopaque, offset: u64, buf: []u8) ReadError!usize {
        const self: *FileSource = @ptrCast(@alignCast(ctx));
        return self.file.readPositionalAll(self.io, buf, offset) catch return error.ReadFailed;
    }
};

/// Sequential reader with a fixed buffer. Metadata is parsed through this so a multi megabyte
/// header costs one buffer, not a copy of the header.
pub const BufReader = struct {
    src: Source,
    /// Offset in the source of `buf[0]`.
    base: u64 = 0,
    buf: []u8,
    start: usize = 0,
    end: usize = 0,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, src: Source, buffer_len: usize) !BufReader {
        return .{ .src = src, .buf = try allocator.alloc(u8, buffer_len), .allocator = allocator };
    }

    pub fn deinit(self: *BufReader) void {
        self.allocator.free(self.buf);
    }

    /// Source offset of the next unread byte.
    pub fn pos(self: *const BufReader) u64 {
        return self.base + self.start;
    }

    fn refill(self: *BufReader, need: usize) ReadError!void {
        std.debug.assert(need <= self.buf.len);
        const have = self.end - self.start;
        std.mem.copyForwards(u8, self.buf[0..have], self.buf[self.start..self.end]);
        self.base += self.start;
        self.start = 0;
        self.end = have;
        while (self.end < need) {
            const n = try self.src.readAt(self.base + self.end, self.buf[self.end..]);
            if (n == 0) return error.UnexpectedEndOfInput;
            self.end += n;
        }
    }

    /// Next `n` bytes, valid until the next call. `n` must not exceed the buffer size.
    pub fn take(self: *BufReader, n: usize) ReadError![]const u8 {
        if (self.end - self.start < n) try self.refill(n);
        const s = self.buf[self.start..][0..n];
        self.start += n;
        return s;
    }

    pub fn readInt(self: *BufReader, comptime T: type) ReadError!T {
        const b = try self.take(@sizeOf(T));
        return std.mem.readInt(T, b[0..@sizeOf(T)], .little);
    }

    pub fn skip(self: *BufReader, n: u64) ReadError!void {
        var left = n;
        while (left > 0) {
            const k: usize = @intCast(@min(left, self.buf.len));
            _ = try self.take(k);
            left -= k;
        }
    }
};

test "BufReader reads across refills" {
    const data = "0123456789abcdefghij";
    const Mem = struct {
        bytes: []const u8,
        fn read(ctx: *anyopaque, offset: u64, buf: []u8) ReadError!usize {
            const m: *@This() = @ptrCast(@alignCast(ctx));
            if (offset >= m.bytes.len) return 0;
            // Return short reads on purpose to exercise the loop.
            const n = @min(@min(buf.len, 3), m.bytes.len - offset);
            @memcpy(buf[0..n], m.bytes[offset..][0..n]);
            return n;
        }
    };
    var m = Mem{ .bytes = data };
    const src = Source{ .ctx = &m, .readAtFn = Mem.read, .len = data.len };
    var r = try BufReader.init(std.testing.allocator, src, 8);
    defer r.deinit();
    try std.testing.expectEqualStrings("0123", try r.take(4));
    try std.testing.expectEqual(@as(u64, 4), r.pos());
    try r.skip(10);
    try std.testing.expectEqualStrings("efgh", try r.take(4));
    try std.testing.expectError(error.UnexpectedEndOfInput, r.take(8));
}
