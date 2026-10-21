//! The handful of per-architecture primitives the device kernels are written against (`cuda_device.zig`, which is
//! compiled to PTX for NVIDIA and to an HSA code object for AMD from the same source).
//!
//! Everything else in the kernels is plain Zig. The primitives are the things Zig has no portable spelling for: the
//! thread and block ids, the block barrier, lane shuffles, the hardware 2^x, round-to-nearest-even conversion and the
//! 4-way int8 dot product. The NVIDIA bodies are the inline PTX the kernels always used; the AMD bodies call the
//! LLVM `amdgcn` intrinsics, which the compiler lowers to the right instruction for the target (`ds_bpermute_b32`,
//! `v_dot4_i32_iu8`, `s_barrier`, ...).
//!
//! AMD support is for RDNA3 and newer (gfx11): they run 32-lane waves, which is the warp size every kernel assumes
//! (`warp_size`). GCN and CDNA run 64-lane waves and are not supported by these kernels.

const std = @import("std");
const builtin = @import("builtin");

const nvidia = builtin.cpu.arch == .nvptx64;
const amd = builtin.cpu.arch == .amdgcn;

comptime {
    if (!nvidia and !amd) @compileError("gpu_prims.zig is device code: build it for nvptx64 or amdgcn only");
}

/// Lanes that execute in lock step. 32 on NVIDIA and on RDNA.
pub const warp_size = 32;

// The LLVM intrinsics are only declared on the architecture that has them.
const amdgcn = if (amd) struct {
    extern fn @"llvm.amdgcn.s.barrier"() void;
    extern fn @"llvm.amdgcn.ds.bpermute"(addr: i32, data: i32) i32;
    extern fn @"llvm.amdgcn.sdot4"(a: i32, b: i32, c: i32, clamp: bool) i32;
    extern fn @"llvm.amdgcn.exp2.f32"(x: f32) f32;
    extern fn @"llvm.rint.f32"(x: f32) f32;
} else struct {};

/// Thread index within the block, x dimension.
pub inline fn tid() u32 {
    if (amd) return @workItemId(0);
    return asm volatile ("mov.u32 %[r], %%tid.x;"
        : [r] "=r" (-> u32),
    );
}

/// Block index in the grid, x dimension.
pub inline fn bid() u32 {
    if (amd) return @workGroupId(0);
    return asm volatile ("mov.u32 %[r], %%ctaid.x;"
        : [r] "=r" (-> u32),
    );
}

/// Block index in the grid, y dimension.
pub inline fn bidY() u32 {
    if (amd) return @workGroupId(1);
    return asm volatile ("mov.u32 %[r], %%ctaid.y;"
        : [r] "=r" (-> u32),
    );
}

/// Threads per block, x dimension.
pub inline fn bdim() u32 {
    if (amd) return @workGroupSize(0);
    return asm volatile ("mov.u32 %[r], %%ntid.x;"
        : [r] "=r" (-> u32),
    );
}

/// Block-wide barrier. Shared (LDS) and global writes made before it are visible to the whole block after it.
pub inline fn barSync() void {
    if (amd) {
        // Wait for this wave's outstanding LDS, global loads and stores first: unlike `bar.sync`, `s_barrier` does not.
        asm volatile (
            \\s_waitcnt vmcnt(0) lgkmcnt(0)
            \\s_waitcnt_vscnt null, 0x0
            \\s_barrier
            ::: .{ .memory = true });
        return;
    }
    asm volatile ("bar.sync 0;" ::: .{ .memory = true });
}

inline fn lane() u32 {
    return tid() & (warp_size - 1);
}

/// Bit pattern of `v` read from lane `src` (0..31) of the warp.
inline fn readLane(v: u32, src: u32) u32 {
    return @bitCast(amdgcn.@"llvm.amdgcn.ds.bpermute"(@intCast(src * 4), @bitCast(v)));
}

/// The value of `v` in lane `lane + off`; a lane whose source is past the end of the warp keeps its own value.
pub inline fn shflDown(v: f32, off: u32) f32 {
    if (amd) {
        const src = lane() + off;
        return @bitCast(readLane(@bitCast(v), if (src < warp_size) src else lane()));
    }
    return asm volatile ("shfl.sync.down.b32 %[r], %[v], %[o], 31, 0xffffffff;"
        : [r] "=f" (-> f32),
        : [v] "f" (v),
          [o] "r" (off),
    );
}

/// The value of `v` in lane `lane ^ mask`.
pub inline fn shflXor(v: f32, mask: u32) f32 {
    if (amd) return @bitCast(readLane(@bitCast(v), lane() ^ mask));
    return asm volatile ("shfl.sync.bfly.b32 %[r], %[v], %[o], 31, 0xffffffff;"
        : [r] "=f" (-> f32),
        : [v] "f" (v),
          [o] "r" (mask),
    );
}

/// The value of `v` in lane `src` (0..31), the same for every lane.
pub inline fn shflIdx(v: f32, src: u32) f32 {
    if (amd) return @bitCast(readLane(@bitCast(v), src));
    return asm volatile ("shfl.sync.idx.b32 %[r], %[v], %[s], 31, 0xffffffff;"
        : [r] "=f" (-> f32),
        : [v] "f" (v),
          [s] "r" (src),
    );
}

/// 2^x through the hardware instruction. Relative error is a few ulp.
pub inline fn exp2(x: f32) f32 {
    if (amd) return amdgcn.@"llvm.amdgcn.exp2.f32"(x);
    return asm volatile ("ex2.approx.ftz.f32 %[r], %[v];"
        : [r] "=f" (-> f32),
        : [v] "f" (x),
    );
}

/// `v` rounded to the nearest integer, ties to even.
pub inline fn roundToInt(v: f32) i32 {
    if (amd) return @intFromFloat(amdgcn.@"llvm.rint.f32"(v));
    return asm volatile ("cvt.rni.s32.f32 %[r], %[v];"
        : [r] "=r" (-> i32),
        : [v] "f" (v),
    );
}

/// `c` plus the dot product of the four signed bytes of `a` with the four signed bytes of `b`.
pub inline fn dp4a(a: u32, b: u32, c: i32) i32 {
    if (amd) return amdgcn.@"llvm.amdgcn.sdot4"(@bitCast(a), @bitCast(b), c, false);
    return asm volatile ("dp4a.s32.s32 %[d], %[a], %[b], %[c];"
        : [d] "=r" (-> i32),
        : [a] "r" (a),
          [b] "r" (b),
          [c] "r" (c),
    );
}
