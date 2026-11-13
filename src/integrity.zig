//! Container-wide integrity sealing for `.hk` files.
//!
//! The stored digest is SHA-256 over the complete container with its own
//! checksum fields cleared and the seal flag forced on. That makes the seal a
//! single 128-byte header write: the digest is computed over the bytes the file
//! will have, so no intermediate "flag set, digest zero" state can reach disk.
//! File-based entry points stream the container in fixed-size chunks, so peak
//! memory is one chunk regardless of model size (a mapping that fails must not
//! turn into an O(model) read).
const std = @import("std");
const format = @import("format.zig");

pub const Verification = enum { legacy_missing, valid, invalid };

pub const Error = error{ FileTooSmall, InvalidMagicOrVersion, ChecksumMismatch } ||
    std.mem.Allocator.Error || std.Io.File.OpenError || std.Io.File.ReadPositionalError ||
    std.Io.File.WritePositionalError || std.Io.File.StatError;

const header_len = @sizeOf(format.FileHeader);
const checksum_offset = @offsetOf(format.FileHeader, "checksum");
const checksum_len = @sizeOf(u64);
const digest_offset = @offsetOf(format.FileHeader, "reserved1");
pub const digest_len = 32;

/// Streaming read granularity. Large enough to keep SHA-256 throughput-bound, small enough to be noise in RSS.
pub const chunk_len: usize = 1 << 20;

comptime {
    std.debug.assert(header_len == 128);
    std.debug.assert(checksum_offset + checksum_len <= header_len);
    std.debug.assert(digest_offset + digest_len <= header_len);
}

/// The header exactly as it is hashed: seal flag on, checksum and digest slots zero.
fn canonicalHeader(header: format.FileHeader) [header_len]u8 {
    var h = header;
    h.flags |= format.HeaderFlags.HAS_SHA256_CHECKSUM;
    h.checksum = 0;
    @memset(h.reserved1[0..digest_len], 0);
    return std.mem.toBytes(h);
}

/// Computes the canonical container digest of an in-memory (usually mapped) container.
pub fn digest(bytes: []const u8) Error![digest_len]u8 {
    if (bytes.len < header_len) return error.FileTooSmall;
    const header = std.mem.bytesToValue(format.FileHeader, bytes[0..header_len]);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(&canonicalHeader(header));
    hasher.update(bytes[header_len..]);
    var result: [digest_len]u8 = undefined;
    hasher.final(&result);
    return result;
}

/// Reads the header and the canonical digest of the file at `path` in `chunk_len` pieces.
fn digestFile(path: []const u8, allocator: std.mem.Allocator) Error!struct { header: format.FileHeader, digest: [digest_len]u8 } {
    const io = std.Options.debug_io;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const size = (try file.stat(io)).size;
    if (size < header_len) return error.FileTooSmall;
    var header_bytes: [header_len]u8 = undefined;
    if ((try file.readPositionalAll(io, &header_bytes, 0)) != header_len) return error.FileTooSmall;
    const header = std.mem.bytesToValue(format.FileHeader, &header_bytes);
    if (!header.isValid()) return error.InvalidMagicOrVersion;

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(&canonicalHeader(header));

    const buf = try allocator.alloc(u8, chunk_len);
    defer allocator.free(buf);
    var offset: u64 = header_len;
    while (offset < size) {
        const want: usize = @intCast(@min(size - offset, buf.len));
        const got = try file.readPositionalAll(io, buf[0..want], offset);
        if (got != want) return error.FileTooSmall; // file shrank under us: never hash a partial read as the whole
        hasher.update(buf[0..want]);
        offset += want;
    }
    var result: [digest_len]u8 = undefined;
    hasher.final(&result);
    return .{ .header = header, .digest = result };
}

fn matches(header: format.FileHeader, computed: [digest_len]u8) bool {
    const compact = std.mem.readInt(u64, computed[0..checksum_len], .little);
    return header.checksum == compact and std.mem.eql(u8, header.reserved1[0..digest_len], &computed);
}

pub fn verify(bytes: []const u8) Error!Verification {
    if (bytes.len < header_len) return error.FileTooSmall;
    const header = std.mem.bytesToValue(format.FileHeader, bytes[0..header_len]);
    if (!header.isValid()) return error.InvalidMagicOrVersion;
    if ((header.flags & format.HeaderFlags.HAS_SHA256_CHECKSUM) == 0) return .legacy_missing;
    return if (matches(header, try digest(bytes))) .valid else .invalid;
}

/// Same verdict as `verify`, for a path, in bounded memory.
pub fn verifyFile(path: []const u8, allocator: std.mem.Allocator) Error!Verification {
    const d = try digestFile(path, allocator);
    if ((d.header.flags & format.HeaderFlags.HAS_SHA256_CHECKSUM) == 0) return .legacy_missing;
    return if (matches(d.header, d.digest)) .valid else .invalid;
}

/// Refuses to build on a container whose existing seal is wrong. Re-sealing after a mutation
/// would otherwise bless whatever corruption was already there. Legacy (unsealed) files pass:
/// they made no claim, and the next seal records their current contents.
pub fn requireIntactSeal(path: []const u8, allocator: std.mem.Allocator) Error!void {
    if (try verifyFile(path, allocator) == .invalid) return error.ChecksumMismatch;
}

/// Enables and refreshes the container seal after a mutation: one streamed pass to hash, then
/// one 128-byte header write that commits the flag, compact checksum and digest together.
pub fn sealFile(path: []const u8, allocator: std.mem.Allocator) Error!void {
    const d = try digestFile(path, allocator);
    var sealed = d.header;
    sealed.flags |= format.HeaderFlags.HAS_SHA256_CHECKSUM;
    sealed.checksum = std.mem.readInt(u64, d.digest[0..checksum_len], .little);
    @memcpy(sealed.reserved1[0..digest_len], &d.digest);

    const io = std.Options.debug_io;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_write });
    defer file.close(io);
    try file.writePositionalAll(io, std.mem.asBytes(&sealed), 0);
}

const Ed25519 = std.crypto.sign.Ed25519;

/// Signed container manifest for cryptographic publisher verification (§4, §150 milestone 2.4).
pub const SignedManifest = struct {
    model_digest: [digest_len]u8,
    publisher_pk: [32]u8,
    timestamp: i64,
    signature: [64]u8,

    pub fn serialize(self: *const SignedManifest) [136]u8 {
        var buf: [136]u8 = undefined;
        @memcpy(buf[0..32], &self.model_digest);
        @memcpy(buf[32..64], &self.publisher_pk);
        std.mem.writeInt(i64, buf[64..72], self.timestamp, .little);
        @memcpy(buf[72..136], &self.signature);
        return buf;
    }

    pub fn deserialize(bytes: *const [136]u8) SignedManifest {
        var m: SignedManifest = undefined;
        @memcpy(&m.model_digest, bytes[0..32]);
        @memcpy(&m.publisher_pk, bytes[32..64]);
        m.timestamp = std.mem.readInt(i64, bytes[64..72], .little);
        @memcpy(&m.signature, bytes[72..136]);
        return m;
    }
};

fn manifestPayload(model_digest: *const [digest_len]u8, timestamp: i64) [40]u8 {
    var msg: [40]u8 = undefined;
    @memcpy(msg[0..32], model_digest);
    std.mem.writeInt(i64, msg[32..40], timestamp, .little);
    return msg;
}

/// Signs a model digest with the publisher's Ed25519 keypair.
pub fn signManifest(model_digest: [digest_len]u8, key_pair: Ed25519.KeyPair, timestamp: i64) !SignedManifest {
    const msg = manifestPayload(&model_digest, timestamp);
    const sig = try key_pair.sign(&msg, null);
    return .{
        .model_digest = model_digest,
        .publisher_pk = key_pair.public_key.toBytes(),
        .timestamp = timestamp,
        .signature = sig.toBytes(),
    };
}

/// Cryptographically verifies that the manifest was signed by the publisher holding `publisher_pk`.
pub fn verifyPublisher(manifest: *const SignedManifest) bool {
    const pk = Ed25519.PublicKey.fromBytes(manifest.publisher_pk) catch return false;
    const sig = Ed25519.Signature.fromBytes(manifest.signature);
    const msg = manifestPayload(&manifest.model_digest, manifest.timestamp);
    sig.verify(&msg, pk) catch return false;
    return true;
}

/// Verifies container bytes: checksum must match `manifest.model_digest` AND publisher signature must be valid.
pub fn verifyContainerWithManifest(bytes: []const u8, manifest: *const SignedManifest) Error!bool {
    if (!verifyPublisher(manifest)) return false;
    const computed = try digest(bytes);
    return std.mem.eql(u8, &computed, &manifest.model_digest);
}

/// Verifies file on disk: checksum must match `manifest.model_digest` AND publisher signature must be valid.
pub fn verifyFileWithManifest(path: []const u8, allocator: std.mem.Allocator, manifest: *const SignedManifest) Error!bool {
    if (!verifyPublisher(manifest)) return false;
    const d = try digestFile(path, allocator);
    return std.mem.eql(u8, &d.digest, &manifest.model_digest);
}

test "SignedManifest: generation, publisher verification, and tampering detection" {
    const io = std.Options.debug_io;
    const kp = Ed25519.KeyPair.generate(io);
    const fake_digest: [32]u8 = @splat(0x42);
    const timestamp: i64 = 1731500000;

    const manifest = try signManifest(fake_digest, kp, timestamp);
    try std.testing.expect(verifyPublisher(&manifest));

    // Serialization roundtrip
    const bytes = manifest.serialize();
    const deserialized = SignedManifest.deserialize(&bytes);
    try std.testing.expect(verifyPublisher(&deserialized));

    // Tampered digest fails verification
    var tampered = manifest;
    tampered.model_digest[0] ^= 0xFF;
    try std.testing.expect(!verifyPublisher(&tampered));

    // Tampered timestamp fails verification
    var tampered_time = manifest;
    tampered_time.timestamp += 1;
    try std.testing.expect(!verifyPublisher(&tampered_time));

    // Tampered publisher public key fails verification
    var tampered_pk = manifest;
    tampered_pk.publisher_pk[0] ^= 0xFF;
    try std.testing.expect(!verifyPublisher(&tampered_pk));
}
