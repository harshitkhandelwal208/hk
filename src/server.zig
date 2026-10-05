//! HTTP server: OpenAI compatible API with continuous batching.
pub const scheduler = @import("server/scheduler.zig");
pub const api = @import("server/api.zig");
pub const Scheduler = scheduler.Scheduler;
pub const Server = api.Server;
pub const Config = api.Config;
