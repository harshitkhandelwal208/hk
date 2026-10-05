//! A Jinja2 subset interpreter for chat templates.
//!
//! Chat templates are small Jinja programs stored with the model. Rendering one correctly
//! matters: a wrong prompt format silently degrades every answer. This implements what real
//! templates use: `for`/`if`/`set`/`macro`, `namespace`, filters, tests, the string and dict
//! methods templates rely on, and whitespace control (`{%-`, `-%}`, trim_blocks and
//! lstrip_blocks as Hugging Face configures them). Constructs it does not know are reported
//! with the line they appear on, never skipped, so an unsupported template fails loudly.
//!
//! Templates are parsed once into a tree; each render uses an arena that is dropped afterwards.

const std = @import("std");

pub const Error = error{
    SyntaxError,
    RenderError,
    OutOfMemory,
    /// The template called `raise_exception`; the message is in `Diag`.
    TemplateRaised,
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

// ---------------------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------------------

pub const List = std.ArrayList(Value);

pub const Pair = struct { key: []const u8, val: Value };

/// Insertion ordered mapping, which is what Python dicts (and so Jinja) give.
pub const Dict = struct {
    items: std.ArrayList(Pair) = .empty,

    pub fn get(self: *const Dict, key: []const u8) ?Value {
        for (self.items.items) |p| if (std.mem.eql(u8, p.key, key)) return p.val;
        return null;
    }

    pub fn put(self: *Dict, a: std.mem.Allocator, key: []const u8, val: Value) !void {
        for (self.items.items) |*p| if (std.mem.eql(u8, p.key, key)) {
            p.val = val;
            return;
        };
        try self.items.append(a, .{ .key = key, .val = val });
    }
};


pub const Value = union(enum) {
    undefined,
    none,
    boolean: bool,
    int: i64,
    float: f64,
    string: []const u8,
    list: *List,
    dict: *Dict,
    macro: *const MacroDef,

    pub fn truthy(self: Value) bool {
        return switch (self) {
            .undefined, .none => false,
            .boolean => |b| b,
            .int => |i| i != 0,
            .float => |f| f != 0,
            .string => |s| s.len > 0,
            .list => |l| l.items.len > 0,
            .dict => |d| d.items.items.len > 0,
            .macro => true,
        };
    }

    pub fn typeName(self: Value) []const u8 {
        return switch (self) {
            .undefined => "undefined",
            .none => "none",
            .boolean => "boolean",
            .int => "integer",
            .float => "float",
            .string => "string",
            .list => "list",
            .dict => "mapping",
            .macro => "macro",
        };
    }
};


pub fn newList(a: std.mem.Allocator) !*List {
    const l = try a.create(List);
    l.* = .empty;
    return l;
}

pub fn newDict(a: std.mem.Allocator) !*Dict {
    const d = try a.create(Dict);
    d.* = .{};
    return d;
}

// ---------------------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------------------

const TokKind = enum { ident, string, int, float, op, eof };

const Tok = struct {
    kind: TokKind,
    text: []const u8,
    line: u32,
};

const Lexer = struct {
    src: []const u8,
    i: usize = 0,
    line: u32,
    toks: std.ArrayList(Tok) = .empty,

    fn lexExpr(self: *Lexer, a: std.mem.Allocator, diag: *Diag) Error!void {
        const s = self.src;
        while (self.i < s.len) {
            const c = s[self.i];
            if (c == '\n') {
                self.line += 1;
                self.i += 1;
            } else if (c == ' ' or c == '\t' or c == '\r') {
                self.i += 1;
            } else if (std.ascii.isAlphabetic(c) or c == '_') {
                const st = self.i;
                while (self.i < s.len and (std.ascii.isAlphanumeric(s[self.i]) or s[self.i] == '_')) self.i += 1;
                try self.toks.append(a, .{ .kind = .ident, .text = s[st..self.i], .line = self.line });
            } else if (std.ascii.isDigit(c)) {
                const st = self.i;
                var is_float = false;
                while (self.i < s.len and std.ascii.isDigit(s[self.i])) self.i += 1;
                if (self.i + 1 < s.len and s[self.i] == '.' and std.ascii.isDigit(s[self.i + 1])) {
                    is_float = true;
                    self.i += 1;
                    while (self.i < s.len and std.ascii.isDigit(s[self.i])) self.i += 1;
                }
                try self.toks.append(a, .{ .kind = if (is_float) .float else .int, .text = s[st..self.i], .line = self.line });
            } else if (c == '"' or c == '\'') {
                const q = c;
                self.i += 1;
                const st = self.i;
                while (self.i < s.len and s[self.i] != q) {
                    if (s[self.i] == '\\') self.i += 1;
                    if (self.i < s.len and s[self.i] == '\n') self.line += 1;
                    self.i += 1;
                }
                if (self.i >= s.len) {
                    diag.set("line {d}: unterminated string", .{self.line});
                    return error.SyntaxError;
                }
                const raw = s[st..self.i];
                self.i += 1;
                try self.toks.append(a, .{ .kind = .string, .text = raw, .line = self.line });
            } else {
                // Operators, longest match first.
                const ops3 = [_][]const u8{"//=" , "**="};
                _ = ops3;
                const two = if (self.i + 1 < s.len) s[self.i .. self.i + 2] else "";
                const ops2 = [_][]const u8{ "==", "!=", "<=", ">=", "//", "**" };
                var matched = false;
                for (ops2) |o| if (std.mem.eql(u8, two, o)) {
                    try self.toks.append(a, .{ .kind = .op, .text = o, .line = self.line });
                    self.i += 2;
                    matched = true;
                    break;
                };
                if (!matched) {
                    if (std.mem.indexOfScalar(u8, "+-*/%<>=()[]{},.:|~!", c) == null) {
                        diag.set("line {d}: unexpected character '{c}'", .{ self.line, c });
                        return error.SyntaxError;
                    }
                    try self.toks.append(a, .{ .kind = .op, .text = s[self.i .. self.i + 1], .line = self.line });
                    self.i += 1;
                }
            }
        }
        try self.toks.append(a, .{ .kind = .eof, .text = "", .line = self.line });
    }
};

fn lexExpression(a: std.mem.Allocator, src: []const u8, line: u32, diag: *Diag) Error![]Tok {
    var lx = Lexer{ .src = src, .line = line };
    try lx.lexExpr(a, diag);
    return lx.toks.toOwnedSlice(a);
}

// ---------------------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------------------

pub const KwArg = struct { name: []const u8, v: *Expr };
pub const DictItem = struct { k: *Expr, v: *Expr };
pub const Branch = struct { cond: ?*Expr, body: []const Node };

pub const Expr = union(enum) {
    literal: Value,
    name: []const u8,
    list: []const *Expr,
    dict: []const DictItem,
    attr: struct { obj: *Expr, name: []const u8 },
    index: struct { obj: *Expr, idx: *Expr },
    slice: struct { obj: *Expr, lo: ?*Expr, hi: ?*Expr, step: ?*Expr },
    call: struct { callee: *Expr, args: []const *Expr, kwargs: []const KwArg },
    filter: struct { obj: *Expr, name: []const u8, args: []const *Expr, kwargs: []const KwArg },
    is_test: struct { obj: *Expr, name: []const u8, args: []const *Expr, negate: bool },
    unary: struct { op: u8, e: *Expr }, // '-', 'n' (not)
    binary: struct { op: []const u8, l: *Expr, r: *Expr },
    cond: struct { c: *Expr, t: *Expr, f: ?*Expr },
};

pub const MacroDef = struct {
    name: []const u8,
    params: []const Param,
    body: []const Node,
    pub const Param = struct { name: []const u8, default: ?*Expr };
};

pub const Node = union(enum) {
    text: []const u8,
    output: *Expr,
    if_: struct { branches: []const Branch },
    for_: struct { targets: []const []const u8, iter: *Expr, filter: ?*Expr, body: []const Node, else_body: []const Node },
    set: struct { target: *Expr, value: *Expr },
    set_block: struct { name: []const u8, body: []const Node },
    macro: *const MacroDef,
    brk,
    cont,
    line: u32,
};

// ---------------------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------------------

const Parser = struct {
    a: std.mem.Allocator,
    toks: []const Tok,
    i: usize = 0,
    diag: *Diag,

    fn peek(self: *Parser) Tok {
        return self.toks[self.i];
    }

    fn advance(self: *Parser) Tok {
        const t = self.toks[self.i];
        if (self.i + 1 < self.toks.len) self.i += 1;
        return t;
    }

    fn isOp(self: *Parser, op: []const u8) bool {
        const t = self.peek();
        return t.kind == .op and std.mem.eql(u8, t.text, op);
    }

    fn isIdent(self: *Parser, name: []const u8) bool {
        const t = self.peek();
        return t.kind == .ident and std.mem.eql(u8, t.text, name);
    }

    fn eatOp(self: *Parser, op: []const u8) bool {
        if (self.isOp(op)) {
            _ = self.advance();
            return true;
        }
        return false;
    }

    fn eatIdent(self: *Parser, name: []const u8) bool {
        if (self.isIdent(name)) {
            _ = self.advance();
            return true;
        }
        return false;
    }

    fn fail(self: *Parser, comptime fmt: []const u8, args: anytype) Error {
        self.diag.set("line {d}: " ++ fmt, .{self.peek().line} ++ args);
        return error.SyntaxError;
    }

    fn expectOp(self: *Parser, op: []const u8) Error!void {
        if (!self.eatOp(op)) return self.fail("expected '{s}', found '{s}'", .{ op, self.peek().text });
    }

    fn mk(self: *Parser, e: Expr) Error!*Expr {
        const p = try self.a.create(Expr);
        p.* = e;
        return p;
    }

    // expression := conditional
    fn parseExpr(self: *Parser) Error!*Expr {
        return self.parseConditional();
    }

    fn parseConditional(self: *Parser) Error!*Expr {
        const t = try self.parseOr();
        if (self.isIdent("if")) {
            _ = self.advance();
            const c = try self.parseOr();
            var f: ?*Expr = null;
            if (self.eatIdent("else")) f = try self.parseConditional();
            return self.mk(.{ .cond = .{ .c = c, .t = t, .f = f } });
        }
        return t;
    }

    fn parseOr(self: *Parser) Error!*Expr {
        var l = try self.parseAnd();
        while (self.isIdent("or")) {
            _ = self.advance();
            const r = try self.parseAnd();
            l = try self.mk(.{ .binary = .{ .op = "or", .l = l, .r = r } });
        }
        return l;
    }

    fn parseAnd(self: *Parser) Error!*Expr {
        var l = try self.parseNot();
        while (self.isIdent("and")) {
            _ = self.advance();
            const r = try self.parseNot();
            l = try self.mk(.{ .binary = .{ .op = "and", .l = l, .r = r } });
        }
        return l;
    }

    fn parseNot(self: *Parser) Error!*Expr {
        if (self.isIdent("not")) {
            _ = self.advance();
            const e = try self.parseNot();
            return self.mk(.{ .unary = .{ .op = 'n', .e = e } });
        }
        return self.parseCompare();
    }

    fn parseCompare(self: *Parser) Error!*Expr {
        var l = try self.parseConcat();
        while (true) {
            const t = self.peek();
            var op: ?[]const u8 = null;
            if (t.kind == .op) {
                for ([_][]const u8{ "==", "!=", "<=", ">=", "<", ">" }) |o| if (std.mem.eql(u8, t.text, o)) {
                    op = o;
                };
            } else if (t.kind == .ident) {
                if (std.mem.eql(u8, t.text, "in")) {
                    op = "in";
                } else if (std.mem.eql(u8, t.text, "not") and self.i + 1 < self.toks.len and self.toks[self.i + 1].kind == .ident and std.mem.eql(u8, self.toks[self.i + 1].text, "in")) {
                    _ = self.advance();
                    op = "not in";
                } else if (std.mem.eql(u8, t.text, "is")) {
                    _ = self.advance();
                    const neg = self.eatIdent("not");
                    const name_t = self.advance();
                    if (name_t.kind != .ident) return self.fail("expected a test name after 'is'", .{});
                    var args: std.ArrayList(*Expr) = .empty;
                    if (self.eatOp("(")) {
                        while (!self.isOp(")")) {
                            try args.append(self.a, try self.parseExpr());
                            if (!self.eatOp(",")) break;
                        }
                        try self.expectOp(")");
                    } else if (self.peek().kind != .eof and !self.isOp(")") and !self.isOp("]") and !self.isOp(",") and !self.isIdent("and") and !self.isIdent("or") and !self.isIdent("else") and !self.isIdent("if") and !self.isOp("}") and !self.isOp(":") and !self.isOp("|") and self.peek().kind != .op) {
                        // "x is divisibleby 3": a bare argument
                        try args.append(self.a, try self.parseConcat());
                    }
                    l = try self.mk(.{ .is_test = .{ .obj = l, .name = name_t.text, .args = try args.toOwnedSlice(self.a), .negate = neg } });
                    continue;
                }
            }
            const o = op orelse return l;
            if (!std.mem.eql(u8, o, "not in")) _ = self.advance() else _ = self.advance();
            const r = try self.parseConcat();
            l = try self.mk(.{ .binary = .{ .op = o, .l = l, .r = r } });
        }
    }

    fn parseConcat(self: *Parser) Error!*Expr {
        var l = try self.parseAdd();
        while (self.isOp("~")) {
            _ = self.advance();
            const r = try self.parseAdd();
            l = try self.mk(.{ .binary = .{ .op = "~", .l = l, .r = r } });
        }
        return l;
    }

    fn parseAdd(self: *Parser) Error!*Expr {
        var l = try self.parseMul();
        while (self.isOp("+") or self.isOp("-")) {
            const op = self.advance().text;
            const r = try self.parseMul();
            l = try self.mk(.{ .binary = .{ .op = op, .l = l, .r = r } });
        }
        return l;
    }

    fn parseMul(self: *Parser) Error!*Expr {
        var l = try self.parseUnary();
        while (self.isOp("*") or self.isOp("/") or self.isOp("//") or self.isOp("%")) {
            const op = self.advance().text;
            const r = try self.parseUnary();
            l = try self.mk(.{ .binary = .{ .op = op, .l = l, .r = r } });
        }
        return l;
    }

    fn parseUnary(self: *Parser) Error!*Expr {
        if (self.isOp("-")) {
            _ = self.advance();
            const e = try self.parseUnary();
            return self.mk(.{ .unary = .{ .op = '-', .e = e } });
        }
        if (self.isOp("+")) {
            _ = self.advance();
            return self.parseUnary();
        }
        return self.parsePostfix();
    }

    fn parseCallArgs(self: *Parser, args: *std.ArrayList(*Expr), kwargs: *std.ArrayList(KwArg)) Error!void {
        // current token is just after '('
        while (!self.isOp(")")) {
            if (self.peek().kind == .ident and self.i + 1 < self.toks.len and self.toks[self.i + 1].kind == .op and std.mem.eql(u8, self.toks[self.i + 1].text, "=")) {
                const name = self.advance().text;
                _ = self.advance();
                try kwargs.append(self.a, .{ .name = name, .v = try self.parseExpr() });
            } else {
                try args.append(self.a, try self.parseExpr());
            }
            if (!self.eatOp(",")) break;
        }
        try self.expectOp(")");
    }

    fn parsePostfix(self: *Parser) Error!*Expr {
        var e = try self.parsePrimary();
        while (true) {
            if (self.isOp(".")) {
                _ = self.advance();
                const n = self.advance();
                if (n.kind != .ident and n.kind != .int) return self.fail("expected an attribute name after '.'", .{});
                e = try self.mk(.{ .attr = .{ .obj = e, .name = n.text } });
            } else if (self.isOp("[")) {
                _ = self.advance();
                var lo: ?*Expr = null;
                if (!self.isOp(":")) lo = try self.parseExpr();
                if (self.isOp(":")) {
                    _ = self.advance();
                    var hi: ?*Expr = null;
                    var step: ?*Expr = null;
                    if (!self.isOp("]") and !self.isOp(":")) hi = try self.parseExpr();
                    if (self.eatOp(":")) {
                        if (!self.isOp("]")) step = try self.parseExpr();
                    }
                    try self.expectOp("]");
                    e = try self.mk(.{ .slice = .{ .obj = e, .lo = lo, .hi = hi, .step = step } });
                } else {
                    try self.expectOp("]");
                    e = try self.mk(.{ .index = .{ .obj = e, .idx = lo.? } });
                }
            } else if (self.isOp("(")) {
                _ = self.advance();
                var args: std.ArrayList(*Expr) = .empty;
                var kwargs: std.ArrayList(KwArg) = .empty;
                try self.parseCallArgs(&args, &kwargs);
                e = try self.mk(.{ .call = .{ .callee = e, .args = try args.toOwnedSlice(self.a), .kwargs = try kwargs.toOwnedSlice(self.a) } });
            } else if (self.isOp("|")) {
                _ = self.advance();
                const n = self.advance();
                if (n.kind != .ident) return self.fail("expected a filter name after '|'", .{});
                var args: std.ArrayList(*Expr) = .empty;
                var kwargs: std.ArrayList(KwArg) = .empty;
                if (self.eatOp("(")) try self.parseCallArgs(&args, &kwargs);
                e = try self.mk(.{ .filter = .{ .obj = e, .name = n.text, .args = try args.toOwnedSlice(self.a), .kwargs = try kwargs.toOwnedSlice(self.a) } });
            } else break;
        }
        return e;
    }

    fn unescape(self: *Parser, raw: []const u8) Error![]const u8 {
        if (std.mem.indexOfScalar(u8, raw, '\\') == null) return raw;
        var out: std.ArrayList(u8) = .empty;
        var i: usize = 0;
        while (i < raw.len) : (i += 1) {
            if (raw[i] != '\\' or i + 1 >= raw.len) {
                try out.append(self.a, raw[i]);
                continue;
            }
            i += 1;
            switch (raw[i]) {
                'n' => try out.append(self.a, '\n'),
                't' => try out.append(self.a, '\t'),
                'r' => try out.append(self.a, '\r'),
                '\\' => try out.append(self.a, '\\'),
                '\'' => try out.append(self.a, '\''),
                '"' => try out.append(self.a, '"'),
                else => {
                    try out.append(self.a, '\\');
                    try out.append(self.a, raw[i]);
                },
            }
        }
        return out.toOwnedSlice(self.a);
    }

    fn parsePrimary(self: *Parser) Error!*Expr {
        const t = self.advance();
        switch (t.kind) {
            .string => return self.mk(.{ .literal = .{ .string = try self.unescape(t.text) } }),
            .int => return self.mk(.{ .literal = .{ .int = std.fmt.parseInt(i64, t.text, 10) catch return self.fail("bad integer '{s}'", .{t.text}) } }),
            .float => return self.mk(.{ .literal = .{ .float = std.fmt.parseFloat(f64, t.text) catch return self.fail("bad number '{s}'", .{t.text}) } }),
            .ident => {
                if (std.mem.eql(u8, t.text, "true") or std.mem.eql(u8, t.text, "True")) return self.mk(.{ .literal = .{ .boolean = true } });
                if (std.mem.eql(u8, t.text, "false") or std.mem.eql(u8, t.text, "False")) return self.mk(.{ .literal = .{ .boolean = false } });
                if (std.mem.eql(u8, t.text, "none") or std.mem.eql(u8, t.text, "None")) return self.mk(.{ .literal = .none });
                return self.mk(.{ .name = t.text });
            },
            .op => {
                if (std.mem.eql(u8, t.text, "(")) {
                    const e = try self.parseExpr();
                    if (self.isOp(",")) {
                        // tuple
                        var items: std.ArrayList(*Expr) = .empty;
                        try items.append(self.a, e);
                        while (self.eatOp(",")) {
                            if (self.isOp(")")) break;
                            try items.append(self.a, try self.parseExpr());
                        }
                        try self.expectOp(")");
                        return self.mk(.{ .list = try items.toOwnedSlice(self.a) });
                    }
                    try self.expectOp(")");
                    return e;
                }
                if (std.mem.eql(u8, t.text, "[")) {
                    var items: std.ArrayList(*Expr) = .empty;
                    while (!self.isOp("]")) {
                        try items.append(self.a, try self.parseExpr());
                        if (!self.eatOp(",")) break;
                    }
                    try self.expectOp("]");
                    return self.mk(.{ .list = try items.toOwnedSlice(self.a) });
                }
                if (std.mem.eql(u8, t.text, "{")) {
                    var items: std.ArrayList(DictItem) = .empty;
                    while (!self.isOp("}")) {
                        const k = try self.parseExpr();
                        try self.expectOp(":");
                        const v = try self.parseExpr();
                        try items.append(self.a, .{ .k = k, .v = v });
                        if (!self.eatOp(",")) break;
                    }
                    try self.expectOp("}");
                    return self.mk(.{ .dict = try items.toOwnedSlice(self.a) });
                }
                self.i -= 1;
                return self.fail("unexpected '{s}'", .{t.text});
            },
            .eof => {
                self.i -= 1;
                return self.fail("the expression ended unexpectedly", .{});
            },
        }
    }
};

// ---------------------------------------------------------------------------------------
// Template: tokenizing into text and tags, then parsing statements
// ---------------------------------------------------------------------------------------

const Piece = union(enum) {
    text: []const u8,
    expr: struct { src: []const u8, line: u32 },
    stmt: struct { src: []const u8, line: u32 },
};

/// Splits the source into text, `{{ }}` and `{% %}` pieces, applying whitespace control:
/// `-` markers, trim_blocks (newline after a block tag) and lstrip_blocks (indentation before
/// one), which Hugging Face enables for chat templates.
fn splitPieces(a: std.mem.Allocator, src: []const u8, diag: *Diag) Error![]Piece {
    var out: std.ArrayList(Piece) = .empty;
    var i: usize = 0;
    var line: u32 = 1;
    var text_start: usize = 0;
    var strip_next_ws = false; // set by `-%}` / `-}}`

    while (i < src.len) {
        const open = std.mem.indexOfPos(u8, src, i, "{") orelse break;
        if (open + 1 >= src.len) break;
        const kind = src[open + 1];
        if (kind != '{' and kind != '%' and kind != '#') {
            i = open + 1;
            continue;
        }
        var text = src[text_start..open];
        if (strip_next_ws) {
            text = std.mem.trimStart(u8, text, " \t\r\n");
            strip_next_ws = false;
        }
        // Tag opening with `-` strips whitespace before it.
        const left_strip = open + 2 < src.len and src[open + 2] == '-';
        if (left_strip) text = std.mem.trimEnd(u8, text, " \t\r\n");
        // lstrip_blocks: for block tags and comments, drop indentation back to the line start.
        if (!left_strip and (kind == '%' or kind == '#')) {
            var k = text.len;
            while (k > 0 and (text[k - 1] == ' ' or text[k - 1] == '\t')) k -= 1;
            if (k == 0 or text[k - 1] == '\n') {
                // Only if the tag starts its line (modulo indentation) and text start is the
                // beginning of the source or a line.
                if (k > 0 or text_start == 0 or src[text_start - 1] == '\n' or true) text = text[0..k];
            }
        }
        if (text.len > 0) try out.append(a, .{ .text = text });
        line += @intCast(std.mem.count(u8, src[text_start..open], "\n"));

        const close_seq: []const u8 = switch (kind) {
            '{' => "}}",
            '%' => "%}",
            else => "#}",
        };
        // Find the closing sequence, skipping string literals.
        var j = open + 2;
        var in_str: u8 = 0;
        var found: ?usize = null;
        while (j + 1 < src.len + 1 and j < src.len) : (j += 1) {
            const ch = src[j];
            if (in_str != 0) {
                if (ch == '\\') {
                    j += 1;
                } else if (ch == in_str) in_str = 0;
                continue;
            }
            if (kind != '#' and (ch == '"' or ch == '\'')) {
                in_str = ch;
                continue;
            }
            if (j + 1 < src.len and src[j] == close_seq[0] and src[j + 1] == close_seq[1]) {
                found = j;
                break;
            }
        }
        const close = found orelse {
            diag.set("line {d}: unterminated '{s}' tag", .{ line, switch (kind) {
                '{' => "{{",
                '%' => "{%",
                else => "{#",
            } });
            return error.SyntaxError;
        };
        var inner = src[open + 2 .. close];
        if (left_strip) inner = inner[1..];
        var right_strip = false;
        if (inner.len > 0 and inner[inner.len - 1] == '-') {
            right_strip = true;
            inner = inner[0 .. inner.len - 1];
        }
        switch (kind) {
            '{' => try out.append(a, .{ .expr = .{ .src = inner, .line = line } }),
            '%' => try out.append(a, .{ .stmt = .{ .src = std.mem.trim(u8, inner, " \t\r\n"), .line = line } }),
            else => {},
        }
        line += @intCast(std.mem.count(u8, src[open..close], "\n"));
        i = close + 2;
        text_start = i;
        if (right_strip) strip_next_ws = true;
        // trim_blocks: a newline right after a block tag (or comment) is removed.
        if ((kind == '%' or kind == '#') and !right_strip and i < src.len) {
            if (src[i] == '\n') {
                text_start += 1;
                i += 1;
                line += 1;
            } else if (i + 1 < src.len and src[i] == '\r' and src[i + 1] == '\n') {
                text_start += 2;
                i += 2;
                line += 1;
            }
        }
    }
    var tail = src[text_start..];
    if (strip_next_ws) tail = std.mem.trimStart(u8, tail, " \t\r\n");
    if (tail.len > 0) try out.append(a, .{ .text = tail });
    return out.toOwnedSlice(a);
}

pub const Template = struct {
    arena: std.heap.ArenaAllocator,
    body: []const Node,

    pub fn deinit(self: *Template) void {
        self.arena.deinit();
    }

    pub fn parse(allocator: std.mem.Allocator, source: []const u8, diag: *Diag) Error!Template {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const a = arena.allocator();
        const src = try a.dupe(u8, source);
        const pieces = try splitPieces(a, src, diag);
        var pos: usize = 0;
        const body = try parseBlock(a, pieces, &pos, diag, &.{});
        if (pos < pieces.len) {
            diag.set("line {d}: unexpected '{s}'", .{ pieces[pos].stmt.line, pieces[pos].stmt.src });
            return error.SyntaxError;
        }
        return .{ .arena = arena, .body = body };
    }
};

fn firstWord(s: []const u8) []const u8 {
    var i: usize = 0;
    while (i < s.len and (std.ascii.isAlphanumeric(s[i]) or s[i] == '_')) i += 1;
    return s[0..i];
}

/// Parses nodes until a statement whose keyword is in `stops` (left unconsumed) or the end.
fn parseBlock(a: std.mem.Allocator, pieces: []const Piece, pos: *usize, diag: *Diag, stops: []const []const u8) Error![]const Node {
    var nodes: std.ArrayList(Node) = .empty;
    while (pos.* < pieces.len) {
        const p = pieces[pos.*];
        switch (p) {
            .text => |t| {
                try nodes.append(a, .{ .text = t });
                pos.* += 1;
            },
            .expr => |e| {
                const toks = try lexExpression(a, e.src, e.line, diag);
                var ps = Parser{ .a = a, .toks = toks, .diag = diag };
                const ex = try ps.parseExpr();
                if (ps.peek().kind != .eof) return ps.fail("unexpected '{s}' in expression", .{ps.peek().text});
                try nodes.append(a, .{ .line = e.line });
                try nodes.append(a, .{ .output = ex });
                pos.* += 1;
            },
            .stmt => |s| {
                const kw = firstWord(s.src);
                for (stops) |st| if (std.mem.eql(u8, kw, st)) return nodes.toOwnedSlice(a);
                try nodes.append(a, .{ .line = s.line });
                try parseStatement(a, pieces, pos, diag, &nodes);
            },
        }
    }
    return nodes.toOwnedSlice(a);
}

fn parseStatement(a: std.mem.Allocator, pieces: []const Piece, pos: *usize, diag: *Diag, nodes: *std.ArrayList(Node)) Error!void {
    const s = pieces[pos.*].stmt;
    const kw = firstWord(s.src);
    const rest = std.mem.trim(u8, s.src[kw.len..], " \t\r\n");
    pos.* += 1;

    if (std.mem.eql(u8, kw, "if")) {
        var branches: std.ArrayList(Branch) = .empty;
        var cond_src: ?[]const u8 = rest;
        var cond_line = s.line;
        while (true) {
            const body = try parseBlock(a, pieces, pos, diag, &.{ "elif", "else", "endif" });
            var cond: ?*Expr = null;
            if (cond_src) |cs| cond = try parseExprSrc(a, cs, cond_line, diag);
            try branches.append(a, .{ .cond = cond, .body = body });
            if (pos.* >= pieces.len) {
                diag.set("line {d}: 'if' is never closed with 'endif'", .{s.line});
                return error.SyntaxError;
            }
            const nx = pieces[pos.*].stmt;
            const nkw = firstWord(nx.src);
            pos.* += 1;
            if (std.mem.eql(u8, nkw, "endif")) break;
            if (std.mem.eql(u8, nkw, "elif")) {
                cond_src = std.mem.trim(u8, nx.src[nkw.len..], " \t\r\n");
                cond_line = nx.line;
            } else {
                cond_src = null;
            }
        }
        try nodes.append(a, .{ .if_ = .{ .branches = try branches.toOwnedSlice(a) } });
    } else if (std.mem.eql(u8, kw, "for")) {
        // for a, b in expr [if cond]
        const in_idx = std.mem.indexOf(u8, rest, " in ") orelse {
            diag.set("line {d}: malformed 'for'", .{s.line});
            return error.SyntaxError;
        };
        var targets: std.ArrayList([]const u8) = .empty;
        var it = std.mem.tokenizeAny(u8, rest[0..in_idx], ", \t");
        while (it.next()) |t| try targets.append(a, t);
        var iter_src = std.mem.trim(u8, rest[in_idx + 4 ..], " \t\r\n");
        var filter_src: ?[]const u8 = null;
        // `for x in xs if cond` (rare): split on the trailing ` if `.
        if (std.mem.lastIndexOf(u8, iter_src, " if ")) |k| {
            filter_src = std.mem.trim(u8, iter_src[k + 4 ..], " \t\r\n");
            iter_src = std.mem.trim(u8, iter_src[0..k], " \t\r\n");
        }
        const iter = try parseExprSrc(a, iter_src, s.line, diag);
        const filter: ?*Expr = if (filter_src) |fs| try parseExprSrc(a, fs, s.line, diag) else null;
        const body = try parseBlock(a, pieces, pos, diag, &.{ "else", "endfor" });
        var else_body: []const Node = &.{};
        if (pos.* < pieces.len and std.mem.eql(u8, firstWord(pieces[pos.*].stmt.src), "else")) {
            pos.* += 1;
            else_body = try parseBlock(a, pieces, pos, diag, &.{"endfor"});
        }
        if (pos.* >= pieces.len) {
            diag.set("line {d}: 'for' is never closed with 'endfor'", .{s.line});
            return error.SyntaxError;
        }
        pos.* += 1; // endfor
        try nodes.append(a, .{ .for_ = .{ .targets = try targets.toOwnedSlice(a), .iter = iter, .filter = filter, .body = body, .else_body = else_body } });
    } else if (std.mem.eql(u8, kw, "set")) {
        // set x = expr | set ns.attr = expr | set x (block form)
        if (std.mem.indexOfScalar(u8, rest, '=')) |eq| {
            // Not "==", "!=", "<=", ">=".
            const lhs = std.mem.trim(u8, rest[0..eq], " \t");
            const rhs = std.mem.trim(u8, rest[eq + 1 ..], " \t\r\n");
            const target = try parseExprSrc(a, lhs, s.line, diag);
            const value = try parseExprSrc(a, rhs, s.line, diag);
            try nodes.append(a, .{ .set = .{ .target = target, .value = value } });
        } else {
            const body = try parseBlock(a, pieces, pos, diag, &.{"endset"});
            if (pos.* >= pieces.len) {
                diag.set("line {d}: 'set' block is never closed", .{s.line});
                return error.SyntaxError;
            }
            pos.* += 1;
            try nodes.append(a, .{ .set_block = .{ .name = rest, .body = body } });
        }
    } else if (std.mem.eql(u8, kw, "macro")) {
        // macro name(a, b=1)
        const paren = std.mem.indexOfScalar(u8, rest, '(') orelse {
            diag.set("line {d}: malformed 'macro'", .{s.line});
            return error.SyntaxError;
        };
        const name = std.mem.trim(u8, rest[0..paren], " \t");
        const toks = try lexExpression(a, rest[paren..], s.line, diag);
        var ps = Parser{ .a = a, .toks = toks, .diag = diag };
        try ps.expectOp("(");
        var params: std.ArrayList(MacroDef.Param) = .empty;
        while (!ps.isOp(")")) {
            const pn = ps.advance();
            if (pn.kind != .ident) return ps.fail("expected a parameter name", .{});
            var def: ?*Expr = null;
            if (ps.eatOp("=")) def = try ps.parseExpr();
            try params.append(a, .{ .name = pn.text, .default = def });
            if (!ps.eatOp(",")) break;
        }
        try ps.expectOp(")");
        const body = try parseBlock(a, pieces, pos, diag, &.{"endmacro"});
        if (pos.* >= pieces.len) {
            diag.set("line {d}: 'macro' is never closed with 'endmacro'", .{s.line});
            return error.SyntaxError;
        }
        pos.* += 1;
        const def = try a.create(MacroDef);
        def.* = .{ .name = name, .params = try params.toOwnedSlice(a), .body = body };
        try nodes.append(a, .{ .macro = def });
    } else if (std.mem.eql(u8, kw, "break")) {
        try nodes.append(a, .brk);
    } else if (std.mem.eql(u8, kw, "continue")) {
        try nodes.append(a, .cont);
    } else if (std.mem.eql(u8, kw, "generation") or std.mem.eql(u8, kw, "endgeneration")) {
        // Training-time markers in some templates: they render nothing themselves.
    } else {
        diag.set("line {d}: unsupported statement '{s}'", .{ s.line, kw });
        return error.SyntaxError;
    }
}

fn parseExprSrc(a: std.mem.Allocator, src: []const u8, line: u32, diag: *Diag) Error!*Expr {
    const toks = try lexExpression(a, src, line, diag);
    var ps = Parser{ .a = a, .toks = toks, .diag = diag };
    const e = try ps.parseExpr();
    if (ps.peek().kind != .eof) return ps.fail("unexpected '{s}'", .{ps.peek().text});
    return e;
}

// ---------------------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------------------

const Scope = struct {
    vars: Dict = .{},
};

pub const Renderer = struct {
    a: std.mem.Allocator,
    diag: *Diag,
    out: std.ArrayList(u8) = .empty,
    scopes: std.ArrayList(*Scope) = .empty,
    line: u32 = 0,
    depth: u32 = 0,
    /// Set by `break` / `continue` and consumed by the innermost loop.
    signal: enum { none, brk, cont } = .none,
    /// Timestamp text returned by `strftime_now`; fixed so output is reproducible in tests.
    now_text: []const u8 = "2026-01-01",

    fn fail(self: *Renderer, comptime fmt: []const u8, args: anytype) Error {
        self.diag.set("line {d}: " ++ fmt, .{self.line} ++ args);
        return error.RenderError;
    }

    fn lookup(self: *Renderer, name: []const u8) Value {
        var i = self.scopes.items.len;
        while (i > 0) {
            i -= 1;
            if (self.scopes.items[i].vars.get(name)) |v| return v;
        }
        return .undefined;
    }

    fn setVar(self: *Renderer, name: []const u8, v: Value) !void {
        try self.scopes.items[self.scopes.items.len - 1].vars.put(self.a, name, v);
    }

    fn pushScope(self: *Renderer) !void {
        const s = try self.a.create(Scope);
        s.* = .{};
        try self.scopes.append(self.a, s);
    }

    fn popScope(self: *Renderer) void {
        _ = self.scopes.pop();
    }

    fn mkString(self: *Renderer, s: []const u8) Value {
        _ = self;
        return .{ .string = s };
    }

    // ---- node execution ----

    pub fn execNodes(self: *Renderer, nodes: []const Node) Error!void {
        for (nodes) |n| {
            try self.exec(n);
            if (self.signal != .none) return;
        }
    }

    fn exec(self: *Renderer, n: Node) Error!void {
        switch (n) {
            .line => |l| self.line = l,
            .text => |t| try self.out.appendSlice(self.a, t),
            .output => |e| {
                const v = try self.eval(e);
                try self.writeValue(v);
            },
            .if_ => |i| {
                for (i.branches) |b| {
                    if (b.cond) |c| {
                        if (!(try self.eval(c)).truthy()) continue;
                    }
                    try self.execNodes(b.body);
                    break;
                }
            },
            .for_ => |f| try self.execFor(f.targets, f.iter, f.filter, f.body, f.else_body),
            .set => |s| try self.execSet(s.target, s.value),
            .set_block => |s| {
                const saved = self.out;
                self.out = .empty;
                try self.execNodes(s.body);
                const text = try self.out.toOwnedSlice(self.a);
                self.out = saved;
                try self.setVar(s.name, .{ .string = text });
            },
            .macro => |m| try self.setVar(m.name, .{ .macro = m }),
            .brk => self.signal = .brk,
            .cont => self.signal = .cont,
        }
    }

    fn execSet(self: *Renderer, target: *Expr, value: *Expr) Error!void {
        const v = try self.eval(value);
        switch (target.*) {
            .name => |nm| try self.setVar(nm, v),
            .attr => |at| {
                const obj = try self.eval(at.obj);
                if (obj != .dict) return self.fail("cannot assign attribute '{s}' on a {s}", .{ at.name, obj.typeName() });
                try obj.dict.put(self.a, at.name, v);
            },
            .list => |items| {
                // a, b = expr
                if (v != .list or v.list.items.len != items.len) return self.fail("cannot unpack into {d} names", .{items.len});
                for (items, 0..) |it, k| {
                    if (it.* != .name) return self.fail("can only unpack into plain names", .{});
                    try self.setVar(it.name, v.list.items[k]);
                }
            },
            else => return self.fail("this assignment target is not supported", .{}),
        }
    }

    fn execFor(self: *Renderer, targets: []const []const u8, iter_e: *Expr, filter: ?*Expr, body: []const Node, else_body: []const Node) Error!void {
        const iterable = try self.eval(iter_e);
        var items: std.ArrayList(Value) = .empty;
        switch (iterable) {
            .list => |l| try items.appendSlice(self.a, l.items),
            .dict => |d| for (d.items.items) |p| try items.append(self.a, .{ .string = p.key }),
            .string => |s| {
                var i: usize = 0;
                while (i < s.len) {
                    const n = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
                    try items.append(self.a, .{ .string = s[i..@min(s.len, i + n)] });
                    i += n;
                }
            },
            .undefined, .none => {},
            else => return self.fail("cannot iterate over a {s}", .{iterable.typeName()}),
        }
        if (filter) |fe| {
            // The filter runs before iteration, so `loop` describes the filtered sequence.
            try self.pushScope();
            defer self.popScope();
            var kept: std.ArrayList(Value) = .empty;
            for (items.items) |item| {
                if (targets.len == 1) {
                    try self.setVar(targets[0], item);
                } else if (item == .list and item.list.items.len == targets.len) {
                    for (targets, 0..) |t, k| try self.setVar(t, item.list.items[k]);
                }
                if ((try self.eval(fe)).truthy()) try kept.append(self.a, item);
            }
            items = kept;
        }
        if (items.items.len == 0) {
            try self.execNodes(else_body);
            return;
        }
        try self.pushScope();
        defer self.popScope();
        const loop = try newDict(self.a);
        try self.setVar("loop", .{ .dict = loop });
        const n = items.items.len;
        for (items.items, 0..) |item, idx| {
            if (targets.len == 1) {
                try self.setVar(targets[0], item);
            } else if (item == .list and item.list.items.len == targets.len) {
                for (targets, 0..) |t, k| try self.setVar(t, item.list.items[k]);
            } else {
                return self.fail("cannot unpack {s} into {d} names", .{ item.typeName(), targets.len });
            }
            try loop.put(self.a, "index", .{ .int = @intCast(idx + 1) });
            try loop.put(self.a, "index0", .{ .int = @intCast(idx) });
            try loop.put(self.a, "revindex", .{ .int = @intCast(n - idx) });
            try loop.put(self.a, "revindex0", .{ .int = @intCast(n - idx - 1) });
            try loop.put(self.a, "first", .{ .boolean = idx == 0 });
            try loop.put(self.a, "last", .{ .boolean = idx + 1 == n });
            try loop.put(self.a, "length", .{ .int = @intCast(n) });
            if (idx > 0) try loop.put(self.a, "previtem", items.items[idx - 1]) else try loop.put(self.a, "previtem", .undefined);
            if (idx + 1 < n) try loop.put(self.a, "nextitem", items.items[idx + 1]) else try loop.put(self.a, "nextitem", .undefined);
            try self.execNodes(body);
            switch (self.signal) {
                .brk => {
                    self.signal = .none;
                    break;
                },
                .cont => self.signal = .none,
                .none => {},
            }
        }
    }

    // ---- values ----

    fn writeValue(self: *Renderer, v: Value) Error!void {
        switch (v) {
            .undefined, .none => if (v == .none) try self.out.appendSlice(self.a, "None"),
            .string => |s| try self.out.appendSlice(self.a, s),
            .boolean => |b| try self.out.appendSlice(self.a, if (b) "True" else "False"),
            .int => |i| try self.out.print(self.a, "{d}", .{i}),
            .float => |f| try self.writeFloat(f),
            .list, .dict => try self.writePython(v),
            .macro => {},
        }
    }

    fn writeFloat(self: *Renderer, f: f64) !void {
        if (@floor(f) == f and @abs(f) < 1e15) {
            try self.out.print(self.a, "{d:.1}", .{f});
        } else {
            try self.out.print(self.a, "{d}", .{f});
        }
    }

    /// Python `repr` style rendering, which is what `{{ some_list }}` produces.
    fn writePython(self: *Renderer, v: Value) Error!void {
        switch (v) {
            .string => |s| {
                try self.out.append(self.a, '\'');
                try self.out.appendSlice(self.a, s);
                try self.out.append(self.a, '\'');
            },
            .list => |l| {
                try self.out.append(self.a, '[');
                for (l.items, 0..) |it, i| {
                    if (i > 0) try self.out.appendSlice(self.a, ", ");
                    try self.writePython(it);
                }
                try self.out.append(self.a, ']');
            },
            .dict => |d| {
                try self.out.append(self.a, '{');
                for (d.items.items, 0..) |p, i| {
                    if (i > 0) try self.out.appendSlice(self.a, ", ");
                    try self.out.print(self.a, "'{s}': ", .{p.key});
                    try self.writePython(p.val);
                }
                try self.out.append(self.a, '}');
            },
            else => try self.writeValue(v),
        }
    }

    fn toStr(self: *Renderer, v: Value) Error![]const u8 {
        switch (v) {
            .string => |s| return s,
            else => {
                const saved = self.out;
                self.out = .empty;
                try self.writeValue(v);
                const s = try self.out.toOwnedSlice(self.a);
                self.out = saved;
                return s;
            },
        }
    }

    fn valuesEqual(self: *Renderer, x: Value, y: Value) bool {
        _ = self;
        return eqValue(x, y);
    }

    // ---- expressions ----

    fn eval(self: *Renderer, e: *const Expr) Error!Value {
        switch (e.*) {
            .literal => |v| return v,
            .name => |n| {
                const v = self.lookup(n);
                return v;
            },
            .list => |items| {
                const l = try newList(self.a);
                for (items) |it| try l.append(self.a, try self.eval(it));
                return .{ .list = l };
            },
            .dict => |items| {
                const d = try newDict(self.a);
                for (items) |kv| {
                    const k = try self.eval(kv.k);
                    try d.put(self.a, try self.toStr(k), try self.eval(kv.v));
                }
                return .{ .dict = d };
            },
            .attr => |at| {
                const o = try self.eval(at.obj);
                return self.getAttr(o, at.name);
            },
            .index => |ix| {
                const o = try self.eval(ix.obj);
                const k = try self.eval(ix.idx);
                return self.getItem(o, k);
            },
            .slice => |sl| return self.evalSlice(sl),
            .call => |c| return self.evalCall(c.callee, c.args, c.kwargs),
            .filter => |f| {
                const o = try self.eval(f.obj);
                return self.applyFilter(f.name, o, f.args, f.kwargs);
            },
            .is_test => |t| {
                const o = try self.eval(t.obj);
                const r = try self.applyTest(t.name, o, t.args);
                return .{ .boolean = r != t.negate };
            },
            .unary => |u| {
                const v = try self.eval(u.e);
                if (u.op == 'n') return .{ .boolean = !v.truthy() };
                return switch (v) {
                    .int => |i| .{ .int = -i },
                    .float => |f| .{ .float = -f },
                    else => self.fail("cannot negate a {s}", .{v.typeName()}),
                };
            },
            .binary => |b| return self.evalBinary(b.op, b.l, b.r),
            .cond => |c| {
                if ((try self.eval(c.c)).truthy()) return self.eval(c.t);
                if (c.f) |f| return self.eval(f);
                return .undefined;
            },
        }
    }

    fn getAttr(self: *Renderer, o: Value, name: []const u8) Error!Value {
        switch (o) {
            .dict => |d| {
                if (d.get(name)) |v| return v;
                return .undefined;
            },
            .undefined => return self.fail("'{s}' is undefined", .{name}),
            // Attribute access on other types yields a bound method marker handled in calls.
            else => return .undefined,
        }
    }

    fn getItem(self: *Renderer, o: Value, k: Value) Error!Value {
        switch (o) {
            .dict => |d| {
                const key = try self.toStr(k);
                return d.get(key) orelse .undefined;
            },
            .list => |l| {
                if (k != .int) return .undefined;
                var i = k.int;
                if (i < 0) i += @intCast(l.items.len);
                if (i < 0 or i >= l.items.len) return .undefined;
                return l.items[@intCast(i)];
            },
            .string => |s| {
                if (k != .int) return .undefined;
                var i = k.int;
                if (i < 0) i += @intCast(s.len);
                if (i < 0 or i >= s.len) return .undefined;
                return .{ .string = s[@intCast(i)..][0..1] };
            },
            .undefined => return self.fail("cannot index an undefined value", .{}),
            else => return .undefined,
        }
    }

    fn evalSlice(self: *Renderer, sl: anytype) Error!Value {
        const o = try self.eval(sl.obj);
        const lo_v: ?Value = if (sl.lo) |x| try self.eval(x) else null;
        const hi_v: ?Value = if (sl.hi) |x| try self.eval(x) else null;
        const st_v: ?Value = if (sl.step) |x| try self.eval(x) else null;
        const step: i64 = if (st_v) |v| (if (v == .int) v.int else 1) else 1;
        if (step == 0) return self.fail("slice step cannot be zero", .{});
        const len: i64 = switch (o) {
            .list => |l| @intCast(l.items.len),
            .string => |s| @intCast(s.len),
            else => return self.fail("cannot slice a {s}", .{o.typeName()}),
        };
        const norm = struct {
            fn f(v: ?Value, default: i64, n: i64) i64 {
                const x = if (v) |vv| (if (vv == .int) vv.int else default) else default;
                var r = x;
                if (r < 0) r += n;
                return r;
            }
        }.f;
        var lo: i64 = undefined;
        var hi: i64 = undefined;
        if (step > 0) {
            lo = std.math.clamp(norm(lo_v, 0, len), 0, len);
            hi = std.math.clamp(norm(hi_v, len, len), 0, len);
        } else {
            lo = std.math.clamp(norm(lo_v, len - 1, len), -1, len - 1);
            hi = std.math.clamp(norm(hi_v, -1, len), -1, len - 1);
            if (hi_v == null) hi = -1;
        }
        switch (o) {
            .list => |l| {
                const out = try newList(self.a);
                var i = lo;
                while ((step > 0 and i < hi) or (step < 0 and i > hi)) : (i += step) try out.append(self.a, l.items[@intCast(i)]);
                return .{ .list = out };
            },
            .string => |s| {
                if (step == 1) return .{ .string = if (lo < hi) s[@intCast(lo)..@intCast(hi)] else "" };
                var buf: std.ArrayList(u8) = .empty;
                var i = lo;
                while ((step > 0 and i < hi) or (step < 0 and i > hi)) : (i += step) try buf.append(self.a, s[@intCast(i)]);
                return .{ .string = try buf.toOwnedSlice(self.a) };
            },
            else => unreachable,
        }
    }

    fn arith(self: *Renderer, op: []const u8, x: Value, y: Value) Error!Value {
        if (std.mem.eql(u8, op, "+")) {
            if (x == .string and y == .string) return .{ .string = try std.mem.concat(self.a, u8, &.{ x.string, y.string }) };
            if (x == .list and y == .list) {
                const l = try newList(self.a);
                try l.appendSlice(self.a, x.list.items);
                try l.appendSlice(self.a, y.list.items);
                return .{ .list = l };
            }
        }
        if (std.mem.eql(u8, op, "*") and x == .string and y == .int) {
            var buf: std.ArrayList(u8) = .empty;
            for (0..@intCast(@max(y.int, 0))) |_| try buf.appendSlice(self.a, x.string);
            return .{ .string = try buf.toOwnedSlice(self.a) };
        }
        const both_int = x == .int and y == .int;
        if ((x == .int or x == .float) and (y == .int or y == .float)) {
            const xf: f64 = if (x == .int) @floatFromInt(x.int) else x.float;
            const yf: f64 = if (y == .int) @floatFromInt(y.int) else y.float;
            if (std.mem.eql(u8, op, "+")) return if (both_int) .{ .int = x.int + y.int } else .{ .float = xf + yf };
            if (std.mem.eql(u8, op, "-")) return if (both_int) .{ .int = x.int - y.int } else .{ .float = xf - yf };
            if (std.mem.eql(u8, op, "*")) return if (both_int) .{ .int = x.int * y.int } else .{ .float = xf * yf };
            if (std.mem.eql(u8, op, "/")) {
                if (yf == 0) return self.fail("division by zero", .{});
                return .{ .float = xf / yf };
            }
            if (std.mem.eql(u8, op, "//")) {
                if (yf == 0) return self.fail("division by zero", .{});
                if (both_int) return .{ .int = @divFloor(x.int, y.int) };
                return .{ .float = @floor(xf / yf) };
            }
            if (std.mem.eql(u8, op, "%")) {
                if (yf == 0) return self.fail("division by zero", .{});
                if (both_int) return .{ .int = @mod(x.int, y.int) };
                return .{ .float = xf - yf * @floor(xf / yf) };
            }
        }
        return self.fail("unsupported operands for '{s}': {s} and {s}", .{ op, x.typeName(), y.typeName() });
    }

    fn compare(self: *Renderer, op: []const u8, x: Value, y: Value) Error!bool {
        if (std.mem.eql(u8, op, "==")) return self.valuesEqual(x, y);
        if (std.mem.eql(u8, op, "!=")) return !self.valuesEqual(x, y);
        if ((x == .int or x == .float) and (y == .int or y == .float)) {
            const xf: f64 = if (x == .int) @floatFromInt(x.int) else x.float;
            const yf: f64 = if (y == .int) @floatFromInt(y.int) else y.float;
            if (std.mem.eql(u8, op, "<")) return xf < yf;
            if (std.mem.eql(u8, op, ">")) return xf > yf;
            if (std.mem.eql(u8, op, "<=")) return xf <= yf;
            return xf >= yf;
        }
        if (x == .string and y == .string) {
            const ord = std.mem.order(u8, x.string, y.string);
            if (std.mem.eql(u8, op, "<")) return ord == .lt;
            if (std.mem.eql(u8, op, ">")) return ord == .gt;
            if (std.mem.eql(u8, op, "<=")) return ord != .gt;
            return ord != .lt;
        }
        return self.fail("cannot compare {s} with {s}", .{ x.typeName(), y.typeName() });
    }

    fn contains(self: *Renderer, hay: Value, needle: Value) Error!bool {
        switch (hay) {
            .string => |s| {
                if (needle != .string) return false;
                return std.mem.indexOf(u8, s, needle.string) != null;
            },
            .list => |l| {
                for (l.items) |it| if (eqValue(it, needle)) return true;
                return false;
            },
            .dict => |d| {
                if (needle != .string) return false;
                return d.get(needle.string) != null;
            },
            .undefined, .none => return false,
            else => return self.fail("'in' needs a string, list or mapping", .{}),
        }
    }

    fn evalBinary(self: *Renderer, op: []const u8, le: *Expr, re: *Expr) Error!Value {
        if (std.mem.eql(u8, op, "and")) {
            const l = try self.eval(le);
            if (!l.truthy()) return l;
            return self.eval(re);
        }
        if (std.mem.eql(u8, op, "or")) {
            const l = try self.eval(le);
            if (l.truthy()) return l;
            return self.eval(re);
        }
        const l = try self.eval(le);
        const r = try self.eval(re);
        if (std.mem.eql(u8, op, "~")) return .{ .string = try std.mem.concat(self.a, u8, &.{ try self.toStr(l), try self.toStr(r) }) };
        if (std.mem.eql(u8, op, "in")) return .{ .boolean = try self.contains(r, l) };
        if (std.mem.eql(u8, op, "not in")) return .{ .boolean = !(try self.contains(r, l)) };
        if (std.mem.eql(u8, op, "==") or std.mem.eql(u8, op, "!=") or std.mem.eql(u8, op, "<") or std.mem.eql(u8, op, ">") or std.mem.eql(u8, op, "<=") or std.mem.eql(u8, op, ">=")) {
            return .{ .boolean = try self.compare(op, l, r) };
        }
        return self.arith(op, l, r);
    }

    // ---- calls ----

    fn evalArgs(self: *Renderer, args: []const *Expr) Error![]Value {
        const out = try self.a.alloc(Value, args.len);
        for (args, 0..) |x, i| out[i] = try self.eval(x);
        return out;
    }

    fn kwarg(self: *Renderer, kwargs: anytype, name: []const u8) Error!?Value {
        for (kwargs) |k| if (std.mem.eql(u8, k.name, name)) return try self.eval(k.v);
        return null;
    }

    fn evalCall(self: *Renderer, callee: *Expr, args: []const *Expr, kwargs: anytype) Error!Value {
        // Method call on an object: obj.method(...)
        if (callee.* == .attr) {
            const at = callee.attr;
            const obj = try self.eval(at.obj);
            // A dict entry that is a macro or namespace attribute is not a method call.
            return self.callMethod(obj, at.name, args, kwargs);
        }
        if (callee.* == .name) {
            const n = callee.name;
            if (std.mem.eql(u8, n, "namespace")) {
                const d = try newDict(self.a);
                for (kwargs) |k| try d.put(self.a, k.name, try self.eval(k.v));
                return .{ .dict = d };
            }
            if (std.mem.eql(u8, n, "raise_exception")) {
                const vals = try self.evalArgs(args);
                const msg = if (vals.len > 0) try self.toStr(vals[0]) else "template raised an exception";
                self.diag.set("{s}", .{msg});
                return error.TemplateRaised;
            }
            if (std.mem.eql(u8, n, "strftime_now")) return .{ .string = self.now_text };
            if (std.mem.eql(u8, n, "range")) {
                const vals = try self.evalArgs(args);
                var lo: i64 = 0;
                var hi: i64 = 0;
                var step: i64 = 1;
                if (vals.len == 1 and vals[0] == .int) hi = vals[0].int else if (vals.len >= 2 and vals[0] == .int and vals[1] == .int) {
                    lo = vals[0].int;
                    hi = vals[1].int;
                    if (vals.len == 3 and vals[2] == .int) step = vals[2].int;
                } else return self.fail("range needs integers", .{});
                if (step == 0) return self.fail("range step cannot be zero", .{});
                const l = try newList(self.a);
                var i = lo;
                while ((step > 0 and i < hi) or (step < 0 and i > hi)) : (i += step) try l.append(self.a, .{ .int = i });
                return .{ .list = l };
            }
            if (std.mem.eql(u8, n, "dict")) {
                const d = try newDict(self.a);
                for (kwargs) |k| try d.put(self.a, k.name, try self.eval(k.v));
                return .{ .dict = d };
            }
            const v = self.lookup(n);
            if (v == .macro) return self.callMacro(v.macro, args, kwargs);
            return self.fail("'{s}' is not callable", .{n});
        }
        const callee_v = try self.eval(callee);
        if (callee_v == .macro) return self.callMacro(callee_v.macro, args, kwargs);
        return self.fail("this expression is not callable", .{});
    }

    fn callMacro(self: *Renderer, m: *const MacroDef, args: []const *Expr, kwargs: anytype) Error!Value {
        if (self.depth > 64) return self.fail("macro recursion is too deep", .{});
        self.depth += 1;
        defer self.depth -= 1;
        const vals = try self.evalArgs(args);
        try self.pushScope();
        defer self.popScope();
        for (m.params, 0..) |p, i| {
            var v: Value = .undefined;
            if (i < vals.len) {
                v = vals[i];
            } else if (try self.kwarg(kwargs, p.name)) |kv| {
                v = kv;
            } else if (p.default) |d| {
                v = try self.eval(d);
            }
            try self.setVar(p.name, v);
        }
        const saved = self.out;
        self.out = .empty;
        try self.execNodes(m.body);
        const text = try self.out.toOwnedSlice(self.a);
        self.out = saved;
        return .{ .string = text };
    }

    fn callMethod(self: *Renderer, obj: Value, name: []const u8, args: []const *Expr, kwargs: anytype) Error!Value {
        const vals = try self.evalArgs(args);
        switch (obj) {
            .string => |s| return self.stringMethod(s, name, vals),
            .dict => |d| {
                if (std.mem.eql(u8, name, "get")) {
                    if (vals.len == 0 or vals[0] != .string) return .none;
                    return d.get(vals[0].string) orelse (if (vals.len > 1) vals[1] else .none);
                }
                if (std.mem.eql(u8, name, "items")) {
                    const l = try newList(self.a);
                    for (d.items.items) |p| {
                        const pair = try newList(self.a);
                        try pair.append(self.a, .{ .string = p.key });
                        try pair.append(self.a, p.val);
                        try l.append(self.a, .{ .list = pair });
                    }
                    return .{ .list = l };
                }
                if (std.mem.eql(u8, name, "keys")) {
                    const l = try newList(self.a);
                    for (d.items.items) |p| try l.append(self.a, .{ .string = p.key });
                    return .{ .list = l };
                }
                if (std.mem.eql(u8, name, "values")) {
                    const l = try newList(self.a);
                    for (d.items.items) |p| try l.append(self.a, p.val);
                    return .{ .list = l };
                }
                if (std.mem.eql(u8, name, "update")) {
                    if (vals.len > 0 and vals[0] == .dict) for (vals[0].dict.items.items) |p| try d.put(self.a, p.key, p.val);
                    return .none;
                }
                if (std.mem.eql(u8, name, "pop")) {
                    if (vals.len > 0 and vals[0] == .string) {
                        for (d.items.items, 0..) |p, i| if (std.mem.eql(u8, p.key, vals[0].string)) {
                            _ = d.items.orderedRemove(i);
                            return p.val;
                        };
                    }
                    return if (vals.len > 1) vals[1] else .none;
                }
                // A macro stored in a dict is not supported; unknown method.
            },
            .list => |l| {
                if (std.mem.eql(u8, name, "append")) {
                    if (vals.len > 0) try l.append(self.a, vals[0]);
                    return .none;
                }
                if (std.mem.eql(u8, name, "pop")) {
                    if (l.items.len == 0) return self.fail("pop from an empty list", .{});
                    if (vals.len > 0 and vals[0] == .int) {
                        var i = vals[0].int;
                        if (i < 0) i += @intCast(l.items.len);
                        if (i < 0 or i >= l.items.len) return self.fail("pop index out of range", .{});
                        return l.orderedRemove(@intCast(i));
                    }
                    return l.pop().?;
                }
                if (std.mem.eql(u8, name, "extend")) {
                    if (vals.len > 0 and vals[0] == .list) try l.appendSlice(self.a, vals[0].list.items);
                    return .none;
                }
            },
            else => {},
        }
        _ = kwargs;
        return self.fail("'{s}' has no method '{s}'", .{ obj.typeName(), name });
    }

    fn stringMethod(self: *Renderer, s: []const u8, name: []const u8, vals: []const Value) Error!Value {
        const a = self.a;
        const arg_s: []const u8 = if (vals.len > 0 and vals[0] == .string) vals[0].string else "";
        if (std.mem.eql(u8, name, "strip")) return .{ .string = if (vals.len > 0) std.mem.trim(u8, s, arg_s) else std.mem.trim(u8, s, " \t\r\n") };
        if (std.mem.eql(u8, name, "lstrip")) return .{ .string = if (vals.len > 0) std.mem.trimStart(u8, s, arg_s) else std.mem.trimStart(u8, s, " \t\r\n") };
        if (std.mem.eql(u8, name, "rstrip")) return .{ .string = if (vals.len > 0) std.mem.trimEnd(u8, s, arg_s) else std.mem.trimEnd(u8, s, " \t\r\n") };
        if (std.mem.eql(u8, name, "startswith")) return .{ .boolean = std.mem.startsWith(u8, s, arg_s) };
        if (std.mem.eql(u8, name, "endswith")) return .{ .boolean = std.mem.endsWith(u8, s, arg_s) };
        if (std.mem.eql(u8, name, "lower")) return .{ .string = try std.ascii.allocLowerString(a, s) };
        if (std.mem.eql(u8, name, "upper")) return .{ .string = try std.ascii.allocUpperString(a, s) };
        if (std.mem.eql(u8, name, "title") or std.mem.eql(u8, name, "capitalize")) {
            const out = try a.dupe(u8, s);
            var new_word = true;
            for (out) |*c| {
                if (std.ascii.isAlphabetic(c.*)) {
                    c.* = if (new_word) std.ascii.toUpper(c.*) else std.ascii.toLower(c.*);
                    new_word = std.mem.eql(u8, name, "capitalize") and false;
                } else new_word = !std.mem.eql(u8, name, "capitalize");
            }
            return .{ .string = out };
        }
        if (std.mem.eql(u8, name, "replace")) {
            if (vals.len < 2 or vals[0] != .string or vals[1] != .string) return self.fail("replace needs two strings", .{});
            return .{ .string = try std.mem.replaceOwned(u8, a, s, vals[0].string, vals[1].string) };
        }
        if (std.mem.eql(u8, name, "split")) {
            const l = try newList(a);
            if (vals.len == 0 or vals[0] == .none) {
                var it = std.mem.tokenizeAny(u8, s, " \t\r\n");
                while (it.next()) |p| try l.append(a, .{ .string = p });
            } else {
                var max: i64 = -1;
                if (vals.len > 1 and vals[1] == .int) max = vals[1].int;
                var it = std.mem.splitSequence(u8, s, arg_s);
                var n: i64 = 0;
                while (it.next()) |p| {
                    if (max >= 0 and n == max) {
                        // keep the remainder intact
                        const consumed = @intFromPtr(p.ptr) - @intFromPtr(s.ptr);
                        try l.append(a, .{ .string = s[consumed..] });
                        break;
                    }
                    try l.append(a, .{ .string = p });
                    n += 1;
                }
            }
            return .{ .list = l };
        }
        if (std.mem.eql(u8, name, "join")) {
            if (vals.len == 0 or vals[0] != .list) return self.fail("join needs a list", .{});
            var buf: std.ArrayList(u8) = .empty;
            for (vals[0].list.items, 0..) |it, i| {
                if (i > 0) try buf.appendSlice(a, s);
                try buf.appendSlice(a, try self.toStr(it));
            }
            return .{ .string = try buf.toOwnedSlice(a) };
        }
        if (std.mem.eql(u8, name, "find")) {
            const i = std.mem.indexOf(u8, s, arg_s);
            return .{ .int = if (i) |k| @intCast(k) else -1 };
        }
        if (std.mem.eql(u8, name, "count")) return .{ .int = @intCast(std.mem.count(u8, s, arg_s)) };
        if (std.mem.eql(u8, name, "format")) {
            // Positional "{}" substitution only.
            var buf: std.ArrayList(u8) = .empty;
            var idx: usize = 0;
            var i: usize = 0;
            while (i < s.len) : (i += 1) {
                if (s[i] == '{' and i + 1 < s.len and s[i + 1] == '}') {
                    if (idx < vals.len) try buf.appendSlice(a, try self.toStr(vals[idx]));
                    idx += 1;
                    i += 1;
                } else try buf.append(a, s[i]);
            }
            return .{ .string = try buf.toOwnedSlice(a) };
        }
        return self.fail("strings have no method '{s}'", .{name});
    }

    // ---- filters ----

    fn applyFilter(self: *Renderer, name: []const u8, o: Value, args: []const *Expr, kwargs: anytype) Error!Value {
        const a = self.a;
        const vals = try self.evalArgs(args);
        const eq = std.mem.eql;
        if (eq(u8, name, "trim")) return .{ .string = std.mem.trim(u8, try self.toStr(o), " \t\r\n") };
        if (eq(u8, name, "length") or eq(u8, name, "count")) return switch (o) {
            .string => |s| .{ .int = @intCast(std.unicode.utf8CountCodepoints(s) catch s.len) },
            .list => |l| .{ .int = @intCast(l.items.len) },
            .dict => |d| .{ .int = @intCast(d.items.items.len) },
            .undefined, .none => .{ .int = 0 },
            else => self.fail("'{s}' has no length", .{o.typeName()}),
        };
        if (eq(u8, name, "lower")) return .{ .string = try std.ascii.allocLowerString(a, try self.toStr(o)) };
        if (eq(u8, name, "upper")) return .{ .string = try std.ascii.allocUpperString(a, try self.toStr(o)) };
        if (eq(u8, name, "string")) return .{ .string = try self.toStr(o) };
        if (eq(u8, name, "safe") or eq(u8, name, "forceescape")) return o;
        if (eq(u8, name, "capitalize") or eq(u8, name, "title")) return self.stringMethod(try self.toStr(o), name, &.{});
        if (eq(u8, name, "int")) return switch (o) {
            .int => o,
            .float => |f| .{ .int = @intFromFloat(f) },
            .string => |s| .{ .int = std.fmt.parseInt(i64, std.mem.trim(u8, s, " "), 10) catch 0 },
            .boolean => |b| .{ .int = @intFromBool(b) },
            else => .{ .int = 0 },
        };
        if (eq(u8, name, "float")) return switch (o) {
            .int => |i| .{ .float = @floatFromInt(i) },
            .float => o,
            .string => |s| .{ .float = std.fmt.parseFloat(f64, std.mem.trim(u8, s, " ")) catch 0 },
            else => .{ .float = 0 },
        };
        if (eq(u8, name, "abs")) return switch (o) {
            .int => |i| .{ .int = @intCast(@abs(i)) },
            .float => |f| .{ .float = @abs(f) },
            else => o,
        };
        if (eq(u8, name, "list")) return switch (o) {
            .list => o,
            .string => |s| blk: {
                const l = try newList(a);
                var i: usize = 0;
                while (i < s.len) {
                    const n = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
                    try l.append(a, .{ .string = s[i..@min(s.len, i + n)] });
                    i += n;
                }
                break :blk .{ .list = l };
            },
            .dict => |d| blk: {
                const l = try newList(a);
                for (d.items.items) |p| try l.append(a, .{ .string = p.key });
                break :blk .{ .list = l };
            },
            else => .{ .list = try newList(a) },
        };
        if (eq(u8, name, "first")) return switch (o) {
            .list => |l| if (l.items.len > 0) l.items[0] else .undefined,
            .string => |s| if (s.len > 0) Value{ .string = s[0..1] } else .undefined,
            else => .undefined,
        };
        if (eq(u8, name, "last")) return switch (o) {
            .list => |l| if (l.items.len > 0) l.items[l.items.len - 1] else .undefined,
            .string => |s| if (s.len > 0) Value{ .string = s[s.len - 1 ..] } else .undefined,
            else => .undefined,
        };
        if (eq(u8, name, "reverse")) return switch (o) {
            .list => |l| blk: {
                const out = try newList(a);
                var i = l.items.len;
                while (i > 0) {
                    i -= 1;
                    try out.append(a, l.items[i]);
                }
                break :blk .{ .list = out };
            },
            .string => |s| blk: {
                const out = try a.dupe(u8, s);
                std.mem.reverse(u8, out);
                break :blk .{ .string = out };
            },
            else => o,
        };
        if (eq(u8, name, "join")) {
            const sep: []const u8 = if (vals.len > 0 and vals[0] == .string) vals[0].string else "";
            if (o != .list) return self.fail("join needs a list", .{});
            var buf: std.ArrayList(u8) = .empty;
            for (o.list.items, 0..) |it, i| {
                if (i > 0) try buf.appendSlice(a, sep);
                try buf.appendSlice(a, try self.toStr(it));
            }
            return .{ .string = try buf.toOwnedSlice(a) };
        }
        if (eq(u8, name, "replace")) {
            if (vals.len < 2) return self.fail("replace needs two arguments", .{});
            return .{ .string = try std.mem.replaceOwned(u8, a, try self.toStr(o), try self.toStr(vals[0]), try self.toStr(vals[1])) };
        }
        if (eq(u8, name, "default") or eq(u8, name, "d")) {
            var boolean = false;
            if (vals.len > 1 and vals[1] == .boolean) boolean = vals[1].boolean;
            if (try self.kwarg(kwargs, "boolean")) |b| boolean = b.truthy();
            const dflt: Value = if (vals.len > 0) vals[0] else (try self.kwarg(kwargs, "default_value")) orelse .{ .string = "" };
            if (o == .undefined) return dflt;
            if (boolean and !o.truthy()) return dflt;
            return o;
        }
        if (eq(u8, name, "tojson")) {
            var indent: ?usize = null;
            if (try self.kwarg(kwargs, "indent")) |iv| if (iv == .int and iv.int >= 0) {
                indent = @intCast(iv.int);
            };
            if (vals.len > 0 and vals[0] == .int) indent = @intCast(vals[0].int);
            var buf: std.ArrayList(u8) = .empty;
            try self.writeJson(&buf, o, indent, 0);
            return .{ .string = try buf.toOwnedSlice(a) };
        }
        if (eq(u8, name, "items")) {
            if (o != .dict) return self.fail("items needs a mapping", .{});
            return self.callMethod(o, "items", &.{}, &[0]KwArg{});
        }
        if (eq(u8, name, "map")) return self.filterMap(o, vals, kwargs);
        if (eq(u8, name, "selectattr") or eq(u8, name, "rejectattr")) return self.filterSelectAttr(o, vals, eq(u8, name, "rejectattr"));
        if (eq(u8, name, "select") or eq(u8, name, "reject")) {
            if (o != .list) return self.fail("{s} needs a list", .{name});
            const out = try newList(a);
            for (o.list.items) |it| {
                var keep = it.truthy();
                if (vals.len > 0 and vals[0] == .string) keep = try self.testValue(vals[0].string, it, vals[1..]);
                if (keep != eq(u8, name, "reject")) try out.append(a, it);
            }
            return .{ .list = out };
        }
        if (eq(u8, name, "sort")) {
            if (o != .list) return self.fail("sort needs a list", .{});
            const copy = try newList(a);
            try copy.appendSlice(a, o.list.items);
            std.mem.sort(Value, copy.items, {}, lessValue);
            return .{ .list = copy };
        }
        if (eq(u8, name, "unique")) {
            if (o != .list) return self.fail("unique needs a list", .{});
            const out = try newList(a);
            for (o.list.items) |it| {
                var seen = false;
                for (out.items) |e| if (eqValue(e, it)) {
                    seen = true;
                };
                if (!seen) try out.append(a, it);
            }
            return .{ .list = out };
        }
        if (eq(u8, name, "indent")) {
            const width: usize = if (vals.len > 0 and vals[0] == .int) @intCast(vals[0].int) else 4;
            const s = try self.toStr(o);
            var buf: std.ArrayList(u8) = .empty;
            var it = std.mem.splitScalar(u8, s, '\n');
            var first = true;
            while (it.next()) |ln| {
                if (!first) try buf.append(a, '\n');
                if (!first and ln.len > 0) try buf.appendNTimes(a, ' ', width);
                try buf.appendSlice(a, ln);
                first = false;
            }
            return .{ .string = try buf.toOwnedSlice(a) };
        }
        if (eq(u8, name, "e") or eq(u8, name, "escape")) {
            const s = try self.toStr(o);
            var buf: std.ArrayList(u8) = .empty;
            for (s) |c| switch (c) {
                '&' => try buf.appendSlice(a, "&amp;"),
                '<' => try buf.appendSlice(a, "&lt;"),
                '>' => try buf.appendSlice(a, "&gt;"),
                '"' => try buf.appendSlice(a, "&#34;"),
                '\'' => try buf.appendSlice(a, "&#39;"),
                else => try buf.append(a, c),
            };
            return .{ .string = try buf.toOwnedSlice(a) };
        }
        if (eq(u8, name, "sum")) {
            if (o != .list) return self.fail("sum needs a list", .{});
            var acc: Value = .{ .int = 0 };
            for (o.list.items) |it| acc = try self.arith("+", acc, it);
            return acc;
        }
        if (eq(u8, name, "min") or eq(u8, name, "max")) {
            if (o != .list or o.list.items.len == 0) return .undefined;
            var best = o.list.items[0];
            for (o.list.items[1..]) |it| {
                const lt = try self.compare("<", it, best);
                if (lt == eq(u8, name, "min")) best = it;
            }
            return best;
        }
        return self.fail("unknown filter '{s}'", .{name});
    }

    fn lessValue(_: void, x: Value, y: Value) bool {
        if (x == .string and y == .string) return std.mem.order(u8, x.string, y.string) == .lt;
        const xf: f64 = switch (x) {
            .int => |i| @floatFromInt(i),
            .float => |f| f,
            else => 0,
        };
        const yf: f64 = switch (y) {
            .int => |i| @floatFromInt(i),
            .float => |f| f,
            else => 0,
        };
        return xf < yf;
    }

    fn filterMap(self: *Renderer, o: Value, vals: []const Value, kwargs: anytype) Error!Value {
        if (o != .list) return self.fail("map needs a list", .{});
        const out = try newList(self.a);
        if (try self.kwarg(kwargs, "attribute")) |av| {
            const key = try self.toStr(av);
            for (o.list.items) |it| try out.append(self.a, if (it == .dict) (it.dict.get(key) orelse .undefined) else .undefined);
            return .{ .list = out };
        }
        if (vals.len > 0 and vals[0] == .string) {
            // map('filtername')
            const fname = vals[0].string;
            for (o.list.items) |it| try out.append(self.a, try self.applyFilterValue(fname, it));
            return .{ .list = out };
        }
        return self.fail("map needs attribute= or a filter name", .{});
    }

    fn applyFilterValue(self: *Renderer, name: []const u8, v: Value) Error!Value {
        const lit = try self.a.create(Expr);
        lit.* = .{ .literal = v };
        return self.applyFilter(name, v, &.{}, &[0]KwArg{});
    }

    fn filterSelectAttr(self: *Renderer, o: Value, vals: []const Value, reject: bool) Error!Value {
        if (o != .list) return self.fail("selectattr needs a list", .{});
        if (vals.len == 0 or vals[0] != .string) return self.fail("selectattr needs an attribute name", .{});
        const out = try newList(self.a);
        for (o.list.items) |it| {
            const attr: Value = if (it == .dict) (it.dict.get(vals[0].string) orelse .undefined) else .undefined;
            var keep = attr.truthy();
            if (vals.len > 1 and vals[1] == .string) keep = try self.testValue(vals[1].string, attr, vals[2..]);
            if (keep != reject) try out.append(self.a, it);
        }
        return .{ .list = out };
    }

    fn writeJsonString(self: *Renderer, buf: *std.ArrayList(u8), s: []const u8) !void {
        try buf.append(self.a, '"');
        for (s) |c| switch (c) {
            '"' => try buf.appendSlice(self.a, "\\\""),
            '\\' => try buf.appendSlice(self.a, "\\\\"),
            '\n' => try buf.appendSlice(self.a, "\\n"),
            '\r' => try buf.appendSlice(self.a, "\\r"),
            '\t' => try buf.appendSlice(self.a, "\\t"),
            0x08 => try buf.appendSlice(self.a, "\\b"),
            0x0C => try buf.appendSlice(self.a, "\\f"),
            else => if (c < 0x20) try buf.print(self.a, "\\u{x:0>4}", .{c}) else try buf.append(self.a, c),
        };
        try buf.append(self.a, '"');
    }

    /// JSON as Python's json.dumps(ensure_ascii=False) writes it: ", " and ": " separators, or
    /// newlines and indentation when an indent is requested.
    fn writeJson(self: *Renderer, buf: *std.ArrayList(u8), v: Value, indent: ?usize, level: usize) Error!void {
        const a = self.a;
        switch (v) {
            .undefined, .none => try buf.appendSlice(a, "null"),
            .boolean => |b| try buf.appendSlice(a, if (b) "true" else "false"),
            .int => |i| try buf.print(a, "{d}", .{i}),
            .float => |f| if (@floor(f) == f and @abs(f) < 1e15) try buf.print(a, "{d:.1}", .{f}) else try buf.print(a, "{d}", .{f}),
            .string => |s| try self.writeJsonString(buf, s),
            .macro => try buf.appendSlice(a, "null"),
            .list => |l| {
                if (l.items.len == 0) return buf.appendSlice(a, "[]");
                try buf.append(a, '[');
                for (l.items, 0..) |it, i| {
                    if (i > 0) try buf.appendSlice(a, if (indent != null) "," else ", ");
                    if (indent) |w| {
                        try buf.append(a, '\n');
                        try buf.appendNTimes(a, ' ', w * (level + 1));
                    }
                    try self.writeJson(buf, it, indent, level + 1);
                }
                if (indent) |w| {
                    try buf.append(a, '\n');
                    try buf.appendNTimes(a, ' ', w * level);
                }
                try buf.append(a, ']');
            },
            .dict => |d| {
                if (d.items.items.len == 0) return buf.appendSlice(a, "{}");
                try buf.append(a, '{');
                for (d.items.items, 0..) |p, i| {
                    if (i > 0) try buf.appendSlice(a, if (indent != null) "," else ", ");
                    if (indent) |w| {
                        try buf.append(a, '\n');
                        try buf.appendNTimes(a, ' ', w * (level + 1));
                    }
                    try self.writeJsonString(buf, p.key);
                    try buf.appendSlice(a, ": ");
                    try self.writeJson(buf, p.val, indent, level + 1);
                }
                if (indent) |w| {
                    try buf.append(a, '\n');
                    try buf.appendNTimes(a, ' ', w * level);
                }
                try buf.append(a, '}');
            },
        }
    }

    // ---- tests ----

    fn applyTest(self: *Renderer, name: []const u8, o: Value, args: []const *Expr) Error!bool {
        const vals = try self.evalArgs(args);
        return self.testValue(name, o, vals);
    }

    fn testValue(self: *Renderer, name: []const u8, o: Value, vals: []const Value) Error!bool {
        const eq = std.mem.eql;
        if (eq(u8, name, "defined")) return o != .undefined;
        if (eq(u8, name, "undefined")) return o == .undefined;
        if (eq(u8, name, "none")) return o == .none;
        if (eq(u8, name, "string")) return o == .string;
        if (eq(u8, name, "number")) return o == .int or o == .float;
        if (eq(u8, name, "integer")) return o == .int;
        if (eq(u8, name, "float")) return o == .float;
        if (eq(u8, name, "boolean")) return o == .boolean;
        if (eq(u8, name, "mapping")) return o == .dict;
        if (eq(u8, name, "iterable")) return o == .list or o == .dict or o == .string;
        if (eq(u8, name, "sequence")) return o == .list or o == .string;
        if (eq(u8, name, "callable")) return o == .macro;
        if (eq(u8, name, "true")) return o == .boolean and o.boolean;
        if (eq(u8, name, "false")) return o == .boolean and !o.boolean;
        if (eq(u8, name, "odd")) return o == .int and @mod(o.int, 2) == 1;
        if (eq(u8, name, "even")) return o == .int and @mod(o.int, 2) == 0;
        if (eq(u8, name, "divisibleby")) return o == .int and vals.len > 0 and vals[0] == .int and vals[0].int != 0 and @mod(o.int, vals[0].int) == 0;
        if (eq(u8, name, "eq") or eq(u8, name, "equalto") or eq(u8, name, "==")) return vals.len > 0 and eqValue(o, vals[0]);
        if (eq(u8, name, "ne")) return vals.len > 0 and !eqValue(o, vals[0]);
        if (eq(u8, name, "lt")) return vals.len > 0 and try self.compare("<", o, vals[0]);
        if (eq(u8, name, "gt")) return vals.len > 0 and try self.compare(">", o, vals[0]);
        if (eq(u8, name, "le")) return vals.len > 0 and try self.compare("<=", o, vals[0]);
        if (eq(u8, name, "ge")) return vals.len > 0 and try self.compare(">=", o, vals[0]);
        if (eq(u8, name, "in")) return vals.len > 0 and try self.contains(vals[0], o);
        if (eq(u8, name, "startingwith")) return o == .string and vals.len > 0 and vals[0] == .string and std.mem.startsWith(u8, o.string, vals[0].string);
        if (eq(u8, name, "endingwith")) return o == .string and vals.len > 0 and vals[0] == .string and std.mem.endsWith(u8, o.string, vals[0].string);
        return self.fail("unknown test '{s}'", .{name});
    }
};

fn eqValue(x: Value, y: Value) bool {
    switch (x) {
        .undefined => return y == .undefined,
        .none => return y == .none or y == .undefined,
        .boolean => |b| return (y == .boolean and y.boolean == b) or (y == .int and y.int == @intFromBool(b)),
        .int => |i| return (y == .int and y.int == i) or (y == .float and y.float == @as(f64, @floatFromInt(i))) or (y == .boolean and @intFromBool(y.boolean) == i),
        .float => |f| return (y == .float and y.float == f) or (y == .int and @as(f64, @floatFromInt(y.int)) == f),
        .string => |s| return y == .string and std.mem.eql(u8, s, y.string),
        .list => |l| {
            if (y != .list or y.list.items.len != l.items.len) return false;
            for (l.items, y.list.items) |p, q| if (!eqValue(p, q)) return false;
            return true;
        },
        .dict => |d| {
            if (y != .dict or y.dict.items.items.len != d.items.items.len) return false;
            for (d.items.items) |p| {
                const o = y.dict.get(p.key) orelse return false;
                if (!eqValue(p.val, o)) return false;
            }
            return true;
        },
        .macro => return y == .macro and y.macro == x.macro,
    }
}

/// Converts parsed JSON into template values. Object key order is preserved.
pub fn fromJson(a: std.mem.Allocator, j: std.json.Value) Error!Value {
    switch (j) {
        .null => return .none,
        .bool => |b| return .{ .boolean = b },
        .integer => |i| return .{ .int = i },
        .float => |f| return .{ .float = f },
        .number_string => |s| return .{ .float = std.fmt.parseFloat(f64, s) catch 0 },
        .string => |s| return .{ .string = try a.dupe(u8, s) },
        .array => |arr| {
            const l = try newList(a);
            for (arr.items) |it| try l.append(a, try fromJson(a, it));
            return .{ .list = l };
        },
        .object => |o| {
            const d = try newDict(a);
            var it = o.iterator();
            while (it.next()) |kv| try d.put(a, try a.dupe(u8, kv.key_ptr.*), try fromJson(a, kv.value_ptr.*));
            return .{ .dict = d };
        },
    }
}

/// Renders `tpl` with `context` (a dict of variables). The result lives in `arena`.
pub fn render(arena: std.mem.Allocator, tpl: *const Template, context: *Dict, diag: *Diag) Error![]u8 {
    var r = Renderer{ .a = arena, .diag = diag };
    try r.pushScope();
    // Globals are the context itself, so assignments at top level do not mutate it.
    for (context.items.items) |p| try r.setVar(p.key, p.val);
    try r.pushScope();
    try r.execNodes(tpl.body);
    return r.out.toOwnedSlice(arena);
}

// ---------------------------------------------------------------------------------------
// Tests (the parity suite against Python's jinja2 lives in tests/test_chat_templates.py)
// ---------------------------------------------------------------------------------------

fn renderStr(src: []const u8, ctx: *Dict, arena: std.mem.Allocator) ![]u8 {
    var diag = Diag{};
    var tpl = try Template.parse(std.testing.allocator, src, &diag);
    defer tpl.deinit();
    return render(arena, &tpl, ctx, &diag);
}

test "text, output, whitespace control" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const ctx = try newDict(a);
    try ctx.put(a, "name", .{ .string = "world" });
    try std.testing.expectEqualStrings("Hello world!", try renderStr("Hello {{ name }}!", ctx, a));
    try std.testing.expectEqualStrings("a b", try renderStr("a {{- ' ' -}} b", ctx, a));
    try std.testing.expectEqualStrings("x\ny", try renderStr("x\n{% if true %}\ny{% endif %}", ctx, a));
}

test "for loops with loop variables, namespace and set" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const ctx = try newDict(a);
    const items = try newList(a);
    for ([_][]const u8{ "a", "b", "c" }) |s| try items.append(a, .{ .string = s });
    try ctx.put(a, "items", .{ .list = items });
    const src = "{% set ns = namespace(n=0) %}{% for x in items %}{{ loop.index }}{{ x }}{% if not loop.last %},{% endif %}{% set ns.n = ns.n + 1 %}{% endfor %}={{ ns.n }}";
    try std.testing.expectEqualStrings("1a,2b,3c=3", try renderStr(src, ctx, a));
}

test "filters, tests and string methods" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const ctx = try newDict(a);
    try ctx.put(a, "s", .{ .string = "  Hi  " });
    try std.testing.expectEqualStrings("Hi", try renderStr("{{ s | trim }}", ctx, a));
    try std.testing.expectEqualStrings("hi", try renderStr("{{ s.strip().lower() }}", ctx, a));
    try std.testing.expectEqualStrings("yes", try renderStr("{% if s is string and s is defined %}yes{% endif %}", ctx, a));
    try std.testing.expectEqualStrings("[1, 2]", try renderStr("{{ [1, 2] | tojson }}", ctx, a));
    try std.testing.expectEqualStrings("ab", try renderStr("{{ 'ab' if true else 'cd' }}", ctx, a));
    try std.testing.expectEqualStrings("6", try renderStr("{{ 2 * 3 }}", ctx, a));
    try std.testing.expectEqualStrings("cba", try renderStr("{{ 'abc'[::-1] }}", ctx, a));
}

test "macros" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const ctx = try newDict(a);
    try std.testing.expectEqualStrings("<b>x</b>", try renderStr("{% macro bold(t) %}<b>{{ t }}</b>{% endmacro %}{{ bold('x') }}", ctx, a));
}

test "errors name the line and never pass silently" {
    var diag = Diag{};
    try std.testing.expectError(error.SyntaxError, Template.parse(std.testing.allocator, "a\n{% if x %}", &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "line 2") != null);
    try std.testing.expectError(error.SyntaxError, Template.parse(std.testing.allocator, "{% frobnicate %}", &diag));
    try std.testing.expectError(error.SyntaxError, Template.parse(std.testing.allocator, "{{ 1 + }}", &diag));

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const ctx = try newDict(a);
    var d2 = Diag{};
    var tpl = try Template.parse(std.testing.allocator, "{{ raise_exception('bad roles') }}", &d2);
    defer tpl.deinit();
    try std.testing.expectError(error.TemplateRaised, render(a, &tpl, ctx, &d2));
    try std.testing.expectEqualStrings("bad roles", d2.message());
    var tpl2 = try Template.parse(std.testing.allocator, "{{ nope | nosuchfilter }}", &d2);
    defer tpl2.deinit();
    try std.testing.expectError(error.RenderError, render(a, &tpl2, ctx, &d2));
}

test "loop filters and loop controls" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const ctx = try newDict(a);
    try std.testing.expectEqualStrings("1:2,2:4", try renderStr("{% for x in [1,2,3,4] if x is even %}{{ loop.index }}:{{ x }}{% if not loop.last %},{% endif %}{% endfor %}", ctx, a));
    try std.testing.expectEqualStrings("12", try renderStr("{% for x in [1,2,3,4] %}{% if x == 3 %}{% break %}{% endif %}{{ x }}{% endfor %}", ctx, a));
    try std.testing.expectEqualStrings("134", try renderStr("{% for x in [1,2,3,4] %}{% if x == 2 %}{% continue %}{% endif %}{{ x }}{% endfor %}", ctx, a));
}
