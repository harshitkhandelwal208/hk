//! A small backtracking regular expression engine, enough for tokenizer pre-splitting patterns.
//!
//! Supported: literals, `.`, classes with ranges and negation, `\s \S \d \D`, `\p{..}` and
//! `\P{..}` general categories (plus `Han`), alternation, grouping (`(...)`, `(?:...)`),
//! greedy and lazy `? * + {n} {n,} {n,m}`, lookahead `(?=...)` and `(?!...)`, `^ $ \b`, and the
//! inline flag group `(?i:...)` for ASCII case folding. Matching works on Unicode code points
//! and returns leftmost-first (PCRE style) matches.
//!
//! Patterns compile to a flat instruction list and run on a VM with an explicit backtrack
//! stack, so matching depth is limited by memory, not the call stack. A step budget turns
//! pathological patterns into a clean "no match" instead of a hang.

const std = @import("std");
const uni = @import("unicode.zig");

pub const Error = error{ InvalidPattern, UnsupportedPattern, OutOfMemory };

const Range = struct { lo: u32, hi: u32 };

const Set = struct {
    negate: bool = false,
    fold: bool = false,
    ranges: []const Range = &.{},
    cats: uni.CatSet = 0,
    /// Matches when the category is NOT in this set (for `\P{..}` and `\D` inside classes).
    not_cats: uni.CatSet = 0,
    has_not_cats: bool = false,
    space: bool = false,
    not_space: bool = false,
    han: bool = false,

    fn matchesRaw(self: *const Set, cp: u32) bool {
        for (self.ranges) |r| if (cp >= r.lo and cp <= r.hi) return true;
        if (self.cats != 0 and uni.inCats(cp, self.cats)) return true;
        if (self.has_not_cats) {
            const c = uni.category(cp);
            if (c == null or self.not_cats & uni.bit(c.?) == 0) return true;
        }
        if (self.space and uni.isSpace(cp)) return true;
        if (self.not_space and !uni.isSpace(cp)) return true;
        if (self.han and uni.isHan(cp)) return true;
        return false;
    }

    fn matches(self: *const Set, cp: u32) bool {
        var m = self.matchesRaw(cp);
        if (!m and self.fold) {
            const lower = uni.foldLower(cp);
            const upper = if (cp >= 'a' and cp <= 'z') cp - 32 else cp;
            m = self.matchesRaw(lower) or self.matchesRaw(upper);
        }
        return m != self.negate;
    }
};

const Op = enum { char, char_fold, set, any, split, jmp, look_pos, look_neg, bol, eol, word_b, match };

const Inst = struct {
    op: Op,
    x: u32 = 0,
    y: u32 = 0,
    cp: u32 = 0,
};

// ---------------------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------------------

const Node = union(enum) {
    empty,
    char: struct { cp: u32, fold: bool },
    set: u32,
    any,
    seq: []Node,
    alt: []Node,
    repeat: struct { node: *Node, min: u32, max: ?u32, lazy: bool },
    look: struct { node: *Node, neg: bool },
    bol,
    eol,
    word_b,
};

const max_repeat = 1000;

const Parser = struct {
    arena: std.mem.Allocator,
    src: []const u8,
    i: usize = 0,
    icase: bool = false,
    sets: std.ArrayList(Set) = .empty,

    fn peek(self: *Parser) ?u8 {
        return if (self.i < self.src.len) self.src[self.i] else null;
    }

    fn next(self: *Parser) ?u8 {
        const c = self.peek() orelse return null;
        self.i += 1;
        return c;
    }

    fn eat(self: *Parser, c: u8) bool {
        if (self.peek() == c) {
            self.i += 1;
            return true;
        }
        return false;
    }

    fn nextCp(self: *Parser) ?u32 {
        if (self.i >= self.src.len) return null;
        const d = uni.decode(self.src[self.i..]);
        self.i += d.len;
        return d.cp;
    }

    fn parseAlt(self: *Parser) Error!Node {
        var branches: std.ArrayList(Node) = .empty;
        try branches.append(self.arena, try self.parseSeq());
        while (self.eat('|')) try branches.append(self.arena, try self.parseSeq());
        if (branches.items.len == 1) return branches.items[0];
        return .{ .alt = try branches.toOwnedSlice(self.arena) };
    }

    fn parseSeq(self: *Parser) Error!Node {
        var items: std.ArrayList(Node) = .empty;
        while (self.peek()) |c| {
            if (c == '|' or c == ')') break;
            var atom = try self.parseAtom();
            atom = try self.parseQuant(atom);
            try items.append(self.arena, atom);
        }
        if (items.items.len == 0) return .empty;
        if (items.items.len == 1) return items.items[0];
        return .{ .seq = try items.toOwnedSlice(self.arena) };
    }

    fn parseNumber(self: *Parser) ?u32 {
        var v: u32 = 0;
        var any = false;
        while (self.peek()) |c| {
            if (c < '0' or c > '9') break;
            v = v * 10 + (c - '0');
            any = true;
            self.i += 1;
            if (v > max_repeat) return null;
        }
        return if (any) v else null;
    }

    fn parseQuant(self: *Parser, atom: Node) Error!Node {
        var node = atom;
        while (true) {
            var min: u32 = 0;
            var max: ?u32 = null;
            const c = self.peek() orelse return node;
            switch (c) {
                '?' => {
                    self.i += 1;
                    min = 0;
                    max = 1;
                },
                '*' => {
                    self.i += 1;
                },
                '+' => {
                    self.i += 1;
                    min = 1;
                },
                '{' => {
                    const save = self.i;
                    self.i += 1;
                    const lo = self.parseNumber() orelse {
                        // Not a quantifier: a literal brace.
                        self.i = save;
                        return node;
                    };
                    min = lo;
                    if (self.eat(',')) {
                        max = self.parseNumber();
                    } else {
                        max = lo;
                    }
                    if (!self.eat('}')) return error.InvalidPattern;
                    if (max != null and max.? < min) return error.InvalidPattern;
                },
                else => return node,
            }
            const lazy = self.eat('?');
            const inner = try self.arena.create(Node);
            inner.* = node;
            node = .{ .repeat = .{ .node = inner, .min = min, .max = max, .lazy = lazy } };
        }
    }

    fn parseAtom(self: *Parser) Error!Node {
        const c = self.peek().?;
        switch (c) {
            '(' => {
                self.i += 1;
                var look: ?bool = null; // null: none, true: negative
                const saved_icase = self.icase;
                if (self.eat('?')) {
                    if (self.eat(':')) {
                        // non capturing
                    } else if (self.eat('=')) {
                        look = false;
                    } else if (self.eat('!')) {
                        look = true;
                    } else if (self.eat('i')) {
                        if (!self.eat(':')) return error.UnsupportedPattern;
                        self.icase = true;
                    } else return error.UnsupportedPattern;
                }
                const inner = try self.parseAlt();
                if (!self.eat(')')) return error.InvalidPattern;
                self.icase = saved_icase;
                if (look) |neg| {
                    const p = try self.arena.create(Node);
                    p.* = inner;
                    return .{ .look = .{ .node = p, .neg = neg } };
                }
                return inner;
            },
            '[' => {
                self.i += 1;
                return .{ .set = try self.parseClass() };
            },
            '.' => {
                self.i += 1;
                return .any;
            },
            '^' => {
                self.i += 1;
                return .bol;
            },
            '$' => {
                self.i += 1;
                return .eol;
            },
            '\\' => {
                self.i += 1;
                return try self.parseEscapeAtom();
            },
            else => {
                const cp = self.nextCp().?;
                return .{ .char = .{ .cp = if (self.icase) uni.foldLower(cp) else cp, .fold = self.icase } };
            },
        }
    }

    fn addSet(self: *Parser, s: Set) Error!Node {
        try self.sets.append(self.arena, s);
        return .{ .set = @intCast(self.sets.items.len - 1) };
    }

    fn parseCatName(self: *Parser) Error!struct { cats: uni.CatSet, han: bool } {
        if (!self.eat('{')) return error.InvalidPattern;
        const start = self.i;
        while (self.peek()) |c| : (self.i += 1) if (c == '}') break;
        const name = self.src[start..self.i];
        if (!self.eat('}')) return error.InvalidPattern;
        if (std.mem.eql(u8, name, "Han")) return .{ .cats = 0, .han = true };
        if (std.mem.eql(u8, name, "L")) return .{ .cats = uni.letters, .han = false };
        if (std.mem.eql(u8, name, "M")) return .{ .cats = uni.marks, .han = false };
        if (std.mem.eql(u8, name, "N")) return .{ .cats = uni.numbers, .han = false };
        if (std.mem.eql(u8, name, "P")) return .{ .cats = uni.punctuation, .han = false };
        if (std.mem.eql(u8, name, "S")) return .{ .cats = uni.symbols, .han = false };
        if (std.mem.eql(u8, name, "Z")) return .{ .cats = uni.separators, .han = false };
        if (std.mem.eql(u8, name, "C")) return .{ .cats = uni.others, .han = false };
        if (std.meta.stringToEnum(uni.Cat, name)) |c| return .{ .cats = uni.bit(c), .han = false };
        return error.UnsupportedPattern;
    }

    fn escapedLiteral(c: u8) ?u32 {
        return switch (c) {
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            'f' => 0x0C,
            'v' => 0x0B,
            '0' => 0,
            else => null,
        };
    }

    fn parseEscapeAtom(self: *Parser) Error!Node {
        const c = self.next() orelse return error.InvalidPattern;
        switch (c) {
            's' => return self.addSet(.{ .space = true }),
            'S' => return self.addSet(.{ .not_space = true }),
            'd' => return self.addSet(.{ .cats = uni.bit(.Nd) }),
            'D' => return self.addSet(.{ .has_not_cats = true, .not_cats = uni.bit(.Nd) }),
            'p', 'P' => {
                const p = try self.parseCatName();
                var s = Set{};
                if (c == 'p') {
                    s.cats = p.cats;
                    s.han = p.han;
                } else {
                    if (p.han) return error.UnsupportedPattern;
                    s.has_not_cats = true;
                    s.not_cats = p.cats;
                }
                return self.addSet(s);
            },
            'b' => return .word_b,
            else => {
                if (escapedLiteral(c)) |cp| return .{ .char = .{ .cp = cp, .fold = false } };
                // Any other escaped character is that character.
                self.i -= 1;
                const cp = self.nextCp().?;
                return .{ .char = .{ .cp = if (self.icase) uni.foldLower(cp) else cp, .fold = self.icase } };
            },
        }
    }

    fn parseClass(self: *Parser) Error!u32 {
        var set = Set{ .fold = self.icase };
        if (self.eat('^')) set.negate = true;
        var ranges: std.ArrayList(Range) = .empty;
        var first = true;
        while (true) {
            const c = self.peek() orelse return error.InvalidPattern;
            if (c == ']' and !first) {
                self.i += 1;
                break;
            }
            first = false;
            var lo: u32 = undefined;
            if (c == '\\') {
                self.i += 1;
                const e = self.next() orelse return error.InvalidPattern;
                switch (e) {
                    's' => {
                        set.space = true;
                        continue;
                    },
                    'S' => {
                        set.not_space = true;
                        continue;
                    },
                    'd' => {
                        set.cats |= uni.bit(.Nd);
                        continue;
                    },
                    'p', 'P' => {
                        const p = try self.parseCatName();
                        if (e == 'p') {
                            set.cats |= p.cats;
                            set.han = set.han or p.han;
                        } else {
                            // Class members are unioned; a negated category inside a class is
                            // expressed as "not in these categories".
                            if (set.has_not_cats) return error.UnsupportedPattern;
                            set.has_not_cats = true;
                            set.not_cats = p.cats;
                        }
                        continue;
                    },
                    else => {
                        if (escapedLiteral(e)) |cp| {
                            lo = cp;
                        } else {
                            self.i -= 1;
                            lo = self.nextCp().?;
                        }
                    },
                }
            } else {
                lo = self.nextCp().?;
            }
            var hi = lo;
            // A dash starts a range unless it is the last character of the class.
            if (self.peek() == '-' and self.i + 1 < self.src.len and self.src[self.i + 1] != ']') {
                self.i += 1;
                if (self.peek() == '\\') {
                    self.i += 1;
                    const e = self.next() orelse return error.InvalidPattern;
                    if (escapedLiteral(e)) |cp| {
                        hi = cp;
                    } else {
                        self.i -= 1;
                        hi = self.nextCp().?;
                    }
                } else {
                    hi = self.nextCp().?;
                }
                if (hi < lo) return error.InvalidPattern;
            }
            try ranges.append(self.arena, .{ .lo = lo, .hi = hi });
        }
        set.ranges = try ranges.toOwnedSlice(self.arena);
        try self.sets.append(self.arena, set);
        return @intCast(self.sets.items.len - 1);
    }
};

// ---------------------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------------------

const max_program = 1 << 16;

const Compiler = struct {
    prog: std.ArrayList(Inst) = .empty,
    allocator: std.mem.Allocator,

    fn emit(self: *Compiler, inst: Inst) Error!u32 {
        if (self.prog.items.len >= max_program) return error.UnsupportedPattern;
        try self.prog.append(self.allocator, inst);
        return @intCast(self.prog.items.len - 1);
    }

    fn here(self: *const Compiler) u32 {
        return @intCast(self.prog.items.len);
    }

    fn compile(self: *Compiler, node: Node) Error!void {
        switch (node) {
            .empty => {},
            .char => |c| _ = try self.emit(.{ .op = if (c.fold) .char_fold else .char, .cp = c.cp }),
            .set => |s| _ = try self.emit(.{ .op = .set, .x = s }),
            .any => _ = try self.emit(.{ .op = .any }),
            .bol => _ = try self.emit(.{ .op = .bol }),
            .eol => _ = try self.emit(.{ .op = .eol }),
            .word_b => _ = try self.emit(.{ .op = .word_b }),
            .seq => |items| for (items) |n| try self.compile(n),
            .alt => |branches| {
                var jumps: std.ArrayList(u32) = .empty;
                defer jumps.deinit(self.allocator);
                for (branches, 0..) |b, k| {
                    if (k + 1 < branches.len) {
                        const split = try self.emit(.{ .op = .split });
                        self.prog.items[split].x = self.here();
                        try self.compile(b);
                        try jumps.append(self.allocator, try self.emit(.{ .op = .jmp }));
                        self.prog.items[split].y = self.here();
                    } else {
                        try self.compile(b);
                    }
                }
                for (jumps.items) |j| self.prog.items[j].x = self.here();
            },
            .look => |l| {
                const at = try self.emit(.{ .op = if (l.neg) .look_neg else .look_pos });
                try self.compile(l.node.*);
                _ = try self.emit(.{ .op = .match });
                self.prog.items[at].x = self.here();
            },
            .repeat => |r| {
                for (0..r.min) |_| try self.compile(r.node.*);
                if (r.max) |mx| {
                    // (mx - min) nested optional copies: x? nested so the earlier ones gate later.
                    var splits: std.ArrayList(u32) = .empty;
                    defer splits.deinit(self.allocator);
                    for (0..mx - r.min) |_| {
                        const split = try self.emit(.{ .op = .split });
                        try splits.append(self.allocator, split);
                        const body = self.here();
                        try self.compile(r.node.*);
                        if (r.lazy) {
                            self.prog.items[split].y = body;
                        } else {
                            self.prog.items[split].x = body;
                        }
                    }
                    const end = self.here();
                    for (splits.items) |s| {
                        if (r.lazy) self.prog.items[s].x = end else self.prog.items[s].y = end;
                    }
                } else {
                    // loop: split body, end; body; jmp loop
                    const split = try self.emit(.{ .op = .split });
                    const body = self.here();
                    try self.compile(r.node.*);
                    _ = try self.emit(.{ .op = .jmp, .x = split });
                    const end = self.here();
                    if (r.lazy) {
                        self.prog.items[split].x = end;
                        self.prog.items[split].y = body;
                    } else {
                        self.prog.items[split].x = body;
                        self.prog.items[split].y = end;
                    }
                }
            },
        }
    }
};

// ---------------------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------------------

const Frame = struct { pc: u32, pos: u32 };

/// Reusable matcher state. One per thread; holds the backtrack stack so repeated matching does
/// not allocate.
pub const Scratch = struct {
    allocator: std.mem.Allocator,
    stack: std.ArrayList(Frame) = .empty,
    steps: u64 = 0,

    pub fn init(allocator: std.mem.Allocator) Scratch {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Scratch) void {
        self.stack.deinit(self.allocator);
    }
};

/// Hard cap on backtrack depth (8 bytes per frame, so 64 MiB). A nullable loop such as `(a*)*`
/// would otherwise grow the stack until the step budget ends it.
const max_frames: usize = 8 * 1024 * 1024;

/// Work allowed per match attempt before giving up. Generous for real patterns, a hard stop
/// for pathological ones.
const step_budget: u64 = 50_000_000;

pub const Regex = struct {
    allocator: std.mem.Allocator,
    prog: []Inst,
    sets: []Set,
    ranges: [][]const Range,

    pub fn compile(allocator: std.mem.Allocator, pattern: []const u8) Error!Regex {
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        var p = Parser{ .arena = arena, .src = pattern };
        const root = try p.parseAlt();
        if (p.i != pattern.len) return error.InvalidPattern;

        var c = Compiler{ .allocator = allocator };
        errdefer c.prog.deinit(allocator);
        try c.compile(root);
        _ = try c.emit(.{ .op = .match });

        // Sets and their range slices live in the arena; copy them out.
        const sets = try allocator.alloc(Set, p.sets.items.len);
        errdefer allocator.free(sets);
        const range_store = try allocator.alloc([]const Range, p.sets.items.len);
        errdefer allocator.free(range_store);
        var done: usize = 0;
        errdefer for (range_store[0..done]) |r| allocator.free(r);
        for (p.sets.items, 0..) |s, i| {
            const copy = try allocator.dupe(Range, s.ranges);
            range_store[i] = copy;
            done += 1;
            sets[i] = s;
            sets[i].ranges = copy;
        }
        return .{ .allocator = allocator, .prog = try c.prog.toOwnedSlice(allocator), .sets = sets, .ranges = range_store };
    }

    pub fn deinit(self: *Regex) void {
        for (self.ranges) |r| self.allocator.free(r);
        self.allocator.free(self.ranges);
        self.allocator.free(self.sets);
        self.allocator.free(self.prog);
    }

    fn isWord(cp: u32) bool {
        return cp == '_' or uni.inCats(cp, uni.letters | uni.numbers);
    }

    fn run(self: *const Regex, cps: []const u32, start_pc: u32, start_pos: u32, sc: *Scratch) ?u32 {
        const base = sc.stack.items.len;
        var pc = start_pc;
        var pos = start_pos;
        const n: u32 = @intCast(cps.len);
        while (true) {
            sc.steps += 1;
            var ok = sc.steps < step_budget;
            if (ok) {
                const inst = self.prog[pc];
                switch (inst.op) {
                    .char => {
                        if (pos < n and cps[pos] == inst.cp) {
                            pos += 1;
                            pc += 1;
                        } else ok = false;
                    },
                    .char_fold => {
                        if (pos < n and uni.foldLower(cps[pos]) == inst.cp) {
                            pos += 1;
                            pc += 1;
                        } else ok = false;
                    },
                    .set => {
                        if (pos < n and self.sets[inst.x].matches(cps[pos])) {
                            pos += 1;
                            pc += 1;
                        } else ok = false;
                    },
                    .any => {
                        if (pos < n and cps[pos] != '\n') {
                            pos += 1;
                            pc += 1;
                        } else ok = false;
                    },
                    .bol => {
                        if (pos == 0) pc += 1 else ok = false;
                    },
                    .eol => {
                        if (pos == n) pc += 1 else ok = false;
                    },
                    .word_b => {
                        const before = pos > 0 and isWord(cps[pos - 1]);
                        const after = pos < n and isWord(cps[pos]);
                        if (before != after) pc += 1 else ok = false;
                    },
                    .jmp => pc = inst.x,
                    .split => {
                        if (sc.stack.items.len >= max_frames) {
                            sc.stack.shrinkRetainingCapacity(base);
                            return null;
                        }
                        sc.stack.append(sc.allocator, .{ .pc = inst.y, .pos = pos }) catch {
                            sc.stack.shrinkRetainingCapacity(base);
                            return null;
                        };
                        pc = inst.x;
                    },
                    .look_pos, .look_neg => {
                        const matched = self.run(cps, pc + 1, pos, sc) != null;
                        if (matched == (inst.op == .look_pos)) pc = inst.x else ok = false;
                    },
                    .match => {
                        sc.stack.shrinkRetainingCapacity(base);
                        return pos;
                    },
                }
            }
            if (!ok) {
                if (sc.steps >= step_budget or sc.stack.items.len == base) {
                    sc.stack.shrinkRetainingCapacity(base);
                    return null;
                }
                const f = sc.stack.pop().?;
                pc = f.pc;
                pos = f.pos;
            }
        }
    }

    /// Match anchored at `start`. Returns the end index (exclusive) of the leftmost-first match.
    pub fn matchAt(self: *const Regex, cps: []const u32, start: usize, sc: *Scratch) ?usize {
        sc.steps = 0;
        const r = self.run(cps, 0, @intCast(start), sc) orelse return null;
        return r;
    }

    /// First match starting at or after `from`. Returns [start, end).
    pub fn find(self: *const Regex, cps: []const u32, from: usize, sc: *Scratch) ?[2]usize {
        var s = from;
        while (s <= cps.len) : (s += 1) {
            if (self.matchAt(cps, s, sc)) |e| return .{ s, e };
        }
        return null;
    }
};

// ---------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------

fn toCps(allocator: std.mem.Allocator, s: []const u8) ![]u32 {
    var list: std.ArrayList(u32) = .empty;
    var i: usize = 0;
    while (i < s.len) {
        const d = uni.decode(s[i..]);
        try list.append(allocator, d.cp);
        i += d.len;
    }
    return list.toOwnedSlice(allocator);
}

/// Returns every non overlapping match as a string, joined with '|' for easy comparison.
fn allMatches(allocator: std.mem.Allocator, pattern: []const u8, text: []const u8) ![]u8 {
    var re = try Regex.compile(allocator, pattern);
    defer re.deinit();
    const cps = try toCps(allocator, text);
    defer allocator.free(cps);
    var sc = Scratch.init(allocator);
    defer sc.deinit();
    var out: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    while (re.find(cps, pos, &sc)) |m| {
        if (out.items.len > 0) try out.append(allocator, '|');
        for (cps[m[0]..m[1]]) |cp| {
            var buf: [4]u8 = undefined;
            const n = uni.encode(cp, &buf);
            try out.appendSlice(allocator, buf[0..n]);
        }
        pos = if (m[1] > m[0]) m[1] else m[1] + 1;
    }
    return out.toOwnedSlice(allocator);
}

fn expectMatches(pattern: []const u8, text: []const u8, want: []const u8) !void {
    const got = try allMatches(std.testing.allocator, pattern, text);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings(want, got);
}

test "literals, classes, quantifiers" {
    try expectMatches("a+", "caaab a", "aaa|a");
    try expectMatches("[a-c]{2}", "abcabc", "ab|ca|bc");
    try expectMatches("x?y", "xy y", "xy|y");
    try expectMatches("a|bc", "abcbc", "a|bc|bc");
    try expectMatches("[^a-c]+", "abxyzc", "xyz");
}

test "unicode categories and negated classes" {
    try expectMatches("\\p{L}+", "héllo wörld 42", "héllo|wörld");
    try expectMatches("\\p{N}+", "a12b345", "12|345");
    try expectMatches("[^\\s\\p{L}\\p{N}]+", "ab!?  cd.", "!?|.");
    try expectMatches("\\p{Han}+", "ab中文cd", "中文");
}

test "lookahead" {
    // Whitespace run, leaving one space to attach to the next word.
    try expectMatches("\\s+(?!\\S)", "a   b", "  ");
    try expectMatches("a(?=b)", "ab ac", "a");
    try expectMatches("a(?!b)", "ab ac", "a");
}

test "case insensitive group" {
    try expectMatches("(?i:'s|'t)", "It'S he'T", "'S|'T");
}

test "bounded repeat expansion" {
    try expectMatches("\\p{N}{1,3}", "123456", "123|456");
    try expectMatches("\\d{1,3}(?=(?:\\d{3})*\\b)", "1234567", "1|234|567");
}

test "gpt2 pretokenizer pattern" {
    const gpt2 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
    try expectMatches(gpt2, "Hello world's 123 !!", "Hello| world|'s| 123| !!");
}

test "llama3 pretokenizer pattern" {
    const llama3 = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    try expectMatches(llama3, "Hello, world!\n\n12345", "Hello|,| world|!\n\n|123|45");
}

test "invalid patterns are errors" {
    try std.testing.expectError(error.InvalidPattern, Regex.compile(std.testing.allocator, "(abc"));
    try std.testing.expectError(error.InvalidPattern, Regex.compile(std.testing.allocator, "[abc"));
    try std.testing.expectError(error.UnsupportedPattern, Regex.compile(std.testing.allocator, "\\p{Nope}"));
}

test "pathological backtracking hits the budget instead of hanging" {
    var re = try Regex.compile(std.testing.allocator, "(a*)*b");
    defer re.deinit();
    var cps: [64]u32 = @splat('a');
    var sc = Scratch.init(std.testing.allocator);
    defer sc.deinit();
    try std.testing.expect(re.matchAt(&cps, 0, &sc) == null);
}
