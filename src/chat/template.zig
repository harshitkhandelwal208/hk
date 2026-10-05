//! Chat prompt formatting using the template stored with the model.

const std = @import("std");
const jinja = @import("jinja.zig");

pub const Message = struct {
    role: []const u8,
    content: []const u8,
};

pub const RenderOptions = struct {
    add_generation_prompt: bool = true,
    /// JSON array of tool definitions, passed to the template as `tools`.
    tools_json: ?[]const u8 = null,
    /// Passed as `enable_thinking` for templates that support a reasoning switch.
    enable_thinking: ?bool = null,
    /// Additional template variables as a JSON object.
    extra_json: ?[]const u8 = null,
};

pub const ChatTemplate = struct {
    allocator: std.mem.Allocator,
    tpl: jinja.Template,
    bos_token: []const u8,
    eos_token: []const u8,

    /// `source` is the Jinja text from the model. `bos` and `eos` are the token strings the
    /// template may reference.
    pub fn init(allocator: std.mem.Allocator, source: []const u8, bos: []const u8, eos: []const u8, diag: *jinja.Diag) jinja.Error!ChatTemplate {
        return .{ .allocator = allocator, .tpl = try jinja.Template.parse(allocator, source, diag), .bos_token = bos, .eos_token = eos };
    }

    pub fn deinit(self: *ChatTemplate) void {
        self.tpl.deinit();
    }

    /// Renders a conversation given as plain messages. The result lives in `arena`.
    pub fn render(self: *const ChatTemplate, arena: std.mem.Allocator, messages: []const Message, opts: RenderOptions, diag: *jinja.Diag) jinja.Error![]u8 {
        const list = try jinja.newList(arena);
        for (messages) |m| {
            const d = try jinja.newDict(arena);
            try d.put(arena, "role", .{ .string = m.role });
            try d.put(arena, "content", .{ .string = m.content });
            try list.append(arena, .{ .dict = d });
        }
        return self.renderValue(arena, .{ .list = list }, opts, diag);
    }

    /// Renders a conversation given as an OpenAI style JSON array of messages, which may carry
    /// tool calls and content parts that plain `Message` cannot express.
    pub fn renderJson(self: *const ChatTemplate, arena: std.mem.Allocator, messages_json: std.json.Value, opts: RenderOptions, diag: *jinja.Diag) jinja.Error![]u8 {
        return self.renderValue(arena, try jinja.fromJson(arena, messages_json), opts, diag);
    }

    fn renderValue(self: *const ChatTemplate, arena: std.mem.Allocator, messages: jinja.Value, opts: RenderOptions, diag: *jinja.Diag) jinja.Error![]u8 {
        const ctx = try jinja.newDict(arena);
        try ctx.put(arena, "messages", messages);
        try ctx.put(arena, "bos_token", .{ .string = self.bos_token });
        try ctx.put(arena, "eos_token", .{ .string = self.eos_token });
        try ctx.put(arena, "add_generation_prompt", .{ .boolean = opts.add_generation_prompt });
        if (opts.enable_thinking) |t| try ctx.put(arena, "enable_thinking", .{ .boolean = t });
        if (opts.tools_json) |tj| {
            const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, tj, .{}) catch {
                diag.set("tools is not valid JSON", .{});
                return error.RenderError;
            };
            try ctx.put(arena, "tools", try jinja.fromJson(arena, parsed));
        }
        if (opts.extra_json) |ej| {
            const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, ej, .{}) catch {
                diag.set("extra template variables are not valid JSON", .{});
                return error.RenderError;
            };
            if (parsed == .object) {
                var it = parsed.object.iterator();
                while (it.next()) |kv| try ctx.put(arena, kv.key_ptr.*, try jinja.fromJson(arena, kv.value_ptr.*));
            }
        }
        return jinja.render(arena, &self.tpl, ctx, diag);
    }
};

test "renders a ChatML style template" {
    const a = std.testing.allocator;
    var diag = jinja.Diag{};
    var t = try ChatTemplate.init(a,
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        "", "", &diag);
    defer t.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const out = try t.render(arena.allocator(), &.{ .{ .role = "system", .content = "Be brief." }, .{ .role = "user", .content = "Hi" } }, .{}, &diag);
    try std.testing.expectEqualStrings("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", out);
}
