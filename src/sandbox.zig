const std = @import("std");

pub const SandboxResult = struct {
    success: bool,
    exit_code: u8,
    stdout: []const u8,
    stderr: []const u8,
    timed_out: bool,
    elapsed_ms: u64,

    pub fn deinit(self: *SandboxResult, allocator: std.mem.Allocator) void {
        if (self.stdout.len > 0) allocator.free(self.stdout);
        if (self.stderr.len > 0) allocator.free(self.stderr);
    }
};

/// Native Zig process execution sandbox with timeouts and resource protection.
pub const NativeSandbox = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    timeout_ms: u64,

    max_output_bytes: usize = 16 * 1024 * 1024, // 16 MiB default max output

    pub fn init(allocator: std.mem.Allocator, timeout_ms: u64) NativeSandbox {
        return .{
            .allocator = allocator,
            .io = std.Options.debug_io,
            .timeout_ms = timeout_ms,
        };
    }

    pub fn initWithIo(allocator: std.mem.Allocator, io: std.Io, timeout_ms: u64) NativeSandbox {
        return .{
            .allocator = allocator,
            .io = io,
            .timeout_ms = timeout_ms,
        };
    }

    /// Executes an isolated child process collecting stdout and stderr.
    /// Hard-enforces timeout (killing hanging processes) and output buffer limits.
    pub fn execute(self: *NativeSandbox, argv: []const []const u8) !SandboxResult {
        const start_time = std.Io.Timestamp.now(self.io, .awake);

        const timeout_opt: std.Io.Timeout = if (self.timeout_ms > 0)
            .{ .duration = .{ .clock = .awake, .raw = .{ .nanoseconds = @as(i96, self.timeout_ms) * 1_000_000 } } }
        else
            .none;

        const run_res = std.process.run(self.allocator, self.io, .{
            .argv = argv,
            .expand_arg0 = .expand,
            .timeout = timeout_opt,
            .stdout_limit = std.Io.Limit.limited(self.max_output_bytes),
            .stderr_limit = std.Io.Limit.limited(self.max_output_bytes),
        }) catch |err| {
            const end_time = std.Io.Timestamp.now(self.io, .awake);
            const elapsed_ns = end_time.nanoseconds - start_time.nanoseconds;
            const elapsed = @as(u64, @intCast(@max(0, @divTrunc(elapsed_ns, 1_000_000))));

            if (err == error.Timeout) {
                const err_msg = try self.allocator.dupe(u8, "Process execution timed out and was terminated");
                return SandboxResult{
                    .success = false,
                    .exit_code = 124, // POSIX timeout exit code
                    .stdout = &[_]u8{},
                    .stderr = err_msg,
                    .timed_out = true,
                    .elapsed_ms = elapsed,
                };
            } else if (err == error.StreamTooLong) {
                const err_msg = try self.allocator.dupe(u8, "Process output exceeded maximum memory buffer limit");
                return SandboxResult{
                    .success = false,
                    .exit_code = 137, // Killed due to resource ceiling
                    .stdout = &[_]u8{},
                    .stderr = err_msg,
                    .timed_out = false,
                    .elapsed_ms = elapsed,
                };
            }

            const err_msg = try std.fmt.allocPrint(self.allocator, "Process execution error: {}", .{err});
            return SandboxResult{
                .success = false,
                .exit_code = 1,
                .stdout = &[_]u8{},
                .stderr = err_msg,
                .timed_out = false,
                .elapsed_ms = elapsed,
            };
        };

        const end_time = std.Io.Timestamp.now(self.io, .awake);
        const elapsed_ns = end_time.nanoseconds - start_time.nanoseconds;
        const elapsed = @as(u64, @intCast(@max(0, @divTrunc(elapsed_ns, 1_000_000))));

        var exit_code: u8 = 0;
        var success = false;
        switch (run_res.term) {
            .exited => |code| {
                exit_code = code;
                success = (code == 0);
            },
            else => {
                exit_code = 1;
                success = false;
            },
        }

        const timed_out = (elapsed >= self.timeout_ms);
        if (timed_out) {
            success = false;
        }

        return SandboxResult{
            .success = success,
            .exit_code = exit_code,
            .stdout = run_res.stdout,
            .stderr = run_res.stderr,
            .timed_out = timed_out,
            .elapsed_ms = elapsed,
        };
    }
};

test "sandbox successful execution" {
    const allocator = std.testing.allocator;
    var sb = NativeSandbox.initWithIo(allocator, std.testing.io, 5000);
    const argv = [_][]const u8{ "echo", "hk_sandbox_ok" };
    var res = try sb.execute(&argv);
    defer res.deinit(allocator);

    try std.testing.expect(res.success);
    try std.testing.expectEqual(@as(u8, 0), res.exit_code);
    try std.testing.expect(!res.timed_out);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "hk_sandbox_ok") != null);
}

test "sandbox timeout enforcement kills hanging process" {
    const allocator = std.testing.allocator;
    // Set 50ms timeout on a command that sleeps for 1 second
    var sb = NativeSandbox.initWithIo(allocator, std.testing.io, 50);
    const argv = [_][]const u8{ "sleep", "1" };
    var res = try sb.execute(&argv);
    defer res.deinit(allocator);

    try std.testing.expect(!res.success);
    try std.testing.expect(res.timed_out);
    try std.testing.expectEqual(@as(u8, 124), res.exit_code);
}

test "sandbox output limit enforcement" {
    const allocator = std.testing.allocator;
    // Set 16-byte limit on output
    var sb = NativeSandbox.initWithIo(allocator, std.testing.io, 5000);
    sb.max_output_bytes = 16;
    const argv = [_][]const u8{ "echo", "this_is_a_very_long_string_exceeding_the_sixteen_byte_limit" };
    var res = try sb.execute(&argv);
    defer res.deinit(allocator);

    try std.testing.expect(!res.success);
    try std.testing.expectEqual(@as(u8, 137), res.exit_code);
    try std.testing.expect(std.mem.indexOf(u8, res.stderr, "exceeded") != null);
}
