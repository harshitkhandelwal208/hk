//! Universal Model Registry & External Reference Resolver (plan §94, §150 milestone 2.4).
//!
//! Provides canonical model identity, versioning, verified publisher manifests,
//! cryptographic container checksums, external shard resolution, and hardware target mappings.

const std = @import("std");
const format = @import("format.zig");
const integrity = @import("integrity.zig");

/// Canonical model identity (namespace, name, version, and optional VCS commit).
pub const ModelIdentity = struct {
    namespace: []const u8,
    name: []const u8,
    version: []const u8,
    commit: ?[]const u8 = null,

    pub fn formatUri(self: ModelIdentity, buf: []u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "{s}/{s}:{s}", .{ self.namespace, self.name, self.version });
    }

    pub fn matches(self: ModelIdentity, namespace: []const u8, name: []const u8, version: ?[]const u8) bool {
        if (!std.mem.eql(u8, self.namespace, namespace)) return false;
        if (!std.mem.eql(u8, self.name, name)) return false;
        if (version) |v| {
            return std.mem.eql(u8, self.version, v);
        }
        return true;
    }
};

/// High-level model metadata.
pub const ModelMetadata = struct {
    architecture: []const u8,
    storage_type: format.StorageType,
    context_length: usize,
    parameter_count: u64 = 0,
    vocab_size: usize = 0,
};

/// Provenance, lineage, and publisher signatures.
pub const ModelProvenance = struct {
    checksum: [32]u8, // SHA-256 container digest
    publisher_id: ?[]const u8 = null,
    signature: ?[64]u8 = null, // Ed25519 signature
    lineage_parent: ?[]const u8 = null,
};

/// Supported execution hardware and minimum memory requirements.
pub const HardwareCompatibility = struct {
    cpu: bool = true,
    cuda: bool = false,
    rocm: bool = false,
    vulkan: bool = false,
    metal: bool = false,
    min_ram_bytes: u64 = 0,
    min_vram_bytes: u64 = 0,
};

/// External shard reference for distributed / sharded model storage (plan §3, §150).
pub const ExternalShard = struct {
    shard_index: u16,
    total_shards: u16,
    uri: []const u8,
    checksum: [32]u8 = @splat(0),
};

/// Complete entry in the universal model registry.
pub const RegistryEntry = struct {
    identity: ModelIdentity,
    metadata: ModelMetadata,
    provenance: ModelProvenance,
    hardware: HardwareCompatibility = .{},
    shards: []const ExternalShard = &.{},
};

/// Memory-first Universal Model Registry.
pub const ModelRegistry = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayList(RegistryEntry) = .empty,

    pub fn init(allocator: std.mem.Allocator) ModelRegistry {
        return .{
            .allocator = allocator,
            .entries = .empty,
        };
    }

    pub fn deinit(self: *ModelRegistry) void {
        self.entries.deinit(self.allocator);
    }

    /// Registers a new model in the local registry.
    pub fn register(self: *ModelRegistry, entry: RegistryEntry) !void {
        // Enforce uniqueness of identity (namespace + name + version)
        for (self.entries.items) |existing| {
            if (existing.identity.matches(entry.identity.namespace, entry.identity.name, entry.identity.version)) {
                return error.ModelAlreadyRegistered;
            }
        }
        try self.entries.append(self.allocator, entry);
    }

    /// Finds a registered model by namespace, name, and optional version.
    pub fn find(self: *const ModelRegistry, namespace: []const u8, name: []const u8, version: ?[]const u8) ?RegistryEntry {
        for (self.entries.items) |entry| {
            if (entry.identity.matches(namespace, name, version)) {
                return entry;
            }
        }
        return null;
    }

    /// Resolves an external shard reference against a base container directory.
    /// Defends against directory traversal attacks (e.g. "../../../etc/passwd").
    pub fn resolveExternalReference(
        base_dir: []const u8,
        shard_uri: []const u8,
        out_buf: []u8,
    ) ![]const u8 {
        // Strip optional file:// prefix
        var clean_uri = shard_uri;
        if (std.mem.startsWith(u8, clean_uri, "file://")) {
            clean_uri = clean_uri["file://".len..];
        }

        // Hostile input check: prevent directory traversal
        if (std.mem.indexOf(u8, clean_uri, "..") != null) {
            return error.InvalidShardPathTraversal;
        }

        if (std.fs.path.isAbsolute(clean_uri)) {
            if (clean_uri.len > out_buf.len) return error.BufferTooSmall;
            @memcpy(out_buf[0..clean_uri.len], clean_uri);
            return out_buf[0..clean_uri.len];
        }

        return std.fmt.bufPrint(out_buf, "{s}/{s}", .{ base_dir, clean_uri });
    }

    /// Verifies the cryptographic container integrity of a registered model on disk.
    pub fn verifyEntry(
        self: *const ModelRegistry,
        entry: *const RegistryEntry,
        file_path: []const u8,
        publisher_public_key: ?[32]u8,
    ) !bool {
        _ = self;
        const io = std.Options.debug_io;
        const cwd = std.Io.Dir.cwd();

        var file = try cwd.openFile(io, file_path, .{});
        defer file.close(io);

        // Stream SHA-256 over entire file
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        var chunk: [65536]u8 = undefined;

        while (true) {
            const n = try file.readStreaming(io, &chunk);
            if (n == 0) break;
            hasher.update(chunk[0..n]);
        }

        var digest: [32]u8 = undefined;
        hasher.final(&digest);

        // Checksum verification
        if (!std.mem.eql(u8, &digest, &entry.provenance.checksum)) {
            return false;
        }

        // Cryptographic publisher verification if signature and public key are provided
        if (entry.provenance.signature) |sig| {
            if (publisher_public_key) |pubkey| {
                const verified = integrity.verifyPublisher(&digest, sig, pubkey);
                if (!verified) return false;
            }
        }

        return true;
    }
};

test "model registry: registration and identity lookup" {
    const allocator = std.testing.allocator;
    var registry = ModelRegistry.init(allocator);
    defer registry.deinit();

    const entry1 = RegistryEntry{
        .identity = .{
            .namespace = "hk-hub",
            .name = "llama-3-8b",
            .version = "1.0.0",
        },
        .metadata = .{
            .architecture = "llama",
            .storage_type = .q4_k,
            .context_length = 8192,
            .parameter_count = 8_000_000_000,
        },
        .provenance = .{
            .checksum = @splat(0xAA),
            .publisher_id = "hk-official",
        },
        .hardware = .{
            .cpu = true,
            .cuda = true,
            .rocm = true,
            .vulkan = true,
        },
    };

    try registry.register(entry1);

    // Duplicate registration should fail
    try std.testing.expectError(error.ModelAlreadyRegistered, registry.register(entry1));

    // Lookup matching entry
    const found = registry.find("hk-hub", "llama-3-8b", "1.0.0");
    try std.testing.expect(found != null);
    try std.testing.expectEqualStrings("llama", found.?.metadata.architecture);
    try std.testing.expectEqual(format.StorageType.q4_k, found.?.metadata.storage_type);

    // Lookup non-existent
    const missing = registry.find("hk-hub", "missing-model", null);
    try std.testing.expect(missing == null);
}

test "model registry: external shard resolution and traversal safety" {
    var buf_path: [256]u8 = undefined;

    // Normal relative path resolution
    const path1 = try ModelRegistry.resolveExternalReference("/models/llama", "shard-00000.hk", &buf_path);
    try std.testing.expectEqualStrings("/models/llama/shard-00000.hk", path1);

    // file:// URI resolution
    const path2 = try ModelRegistry.resolveExternalReference("/models/llama", "file://shard-00001.hk", &buf_path);
    try std.testing.expectEqualStrings("/models/llama/shard-00001.hk", path2);

    // Directory traversal attempt must be rejected fail-closed
    const traversal_err = ModelRegistry.resolveExternalReference("/models/llama", "../../etc/shadow", &buf_path);
    try std.testing.expectError(error.InvalidShardPathTraversal, traversal_err);
}
