//! Environment switches for the device backends (`HK_NO_COOPMAT`, `HK_NO_TENSOR_CORES`).
//!
//! Reading the environment through libc keeps these modules free of an `Io`/`Init` plumbing they otherwise have no
//! use for. Builds without libc (some cross targets) and Windows have no such switch: it reads as unset there.

const std = @import("std");
const builtin = @import("builtin");

/// True when the variable is set (to anything).
pub fn isSet(name: [*:0]const u8) bool {
    if (builtin.os.tag == .windows or !builtin.link_libc) return false;
    return std.c.getenv(name) != null;
}

/// The variable's value, or null when it is unset (or this build has no way to read it, as for `isSet`).
pub fn get(name: [*:0]const u8) ?[]const u8 {
    if (builtin.os.tag == .windows or !builtin.link_libc) return null;
    const p = std.c.getenv(name) orelse return null;
    return std.mem.span(p);
}
