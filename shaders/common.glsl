// Weight decoding shared by the matrix kernels and the embedding lookup.
//
// `w` is the weight tensor in a layout made for the GPU (see gpuLayout in src/vk/engine.zig):
// the GGML block layouts, with every block starting on a 4 byte boundary and small scales
// widened to f32 where that costs little (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL). Reads are
// word loads.
//
// Every format defines
//   ELEMS   weights per block
//   BYTES   bytes per block (in this layout)
//   dq(o,i) the i-th weight of the block at byte offset o            (used by the batch kernel)
//   dotblock(o, xo)  sum of the block's weights times x[xo .. xo+ELEMS)   (used by the decode kernel)
// Select the format with -DTYPE_<NAME> when compiling.

uint b8(uint o) { return (w32[o >> 2] >> ((o & 3u) * 8u)) & 255u; }
uint b16(uint o) { return (w32[o >> 2] >> ((o & 2u) * 8u)) & 65535u; }
float f32at(uint o) { return uintBitsToFloat(w32[o >> 2]); }
float h16(uint o) { return unpackHalf2x16(b16(o)).x; }
int s8(uint o) { return int(w32[o >> 2] << (24u - (o & 3u) * 8u)) >> 24; }

const int kvalues_iq4nl[16] = int[16](-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

// Four signed bytes of a word.
ivec4 sb4(uint u) {
    return ivec4(int(u << 24) >> 24, int(u << 16) >> 24, int(u << 8) >> 24, int(u) >> 24);
}
// Low and high nibbles of the four bytes of a word, as unsigned values.
uvec4 lo4(uint u) { return uvec4(u & 15u, (u >> 8) & 15u, (u >> 16) & 15u, (u >> 24) & 15u); }
uvec4 hi4(uint u) { return uvec4((u >> 4) & 15u, (u >> 12) & 15u, (u >> 20) & 15u, (u >> 28) & 15u); }
vec4 xv(uint xo, uint k) { return x4[(xo >> 2) + k]; }

// Scale and min of sub block j of a Q4_K / Q5_K block (12 packed bytes at byte offset s).
void sm4(uint s, uint j, out uint sc, out uint mn) {
    if (j < 4u) {
        sc = b8(s + j) & 63u;
        mn = b8(s + j + 4u) & 63u;
    } else {
        sc = (b8(s + j + 4u) & 15u) | ((b8(s + j - 4u) >> 6) << 4);
        mn = (b8(s + j + 4u) >> 4) | ((b8(s + j) >> 6) << 4);
    }
}

#if defined(TYPE_F32)
#define ELEMS 8u
#define BYTES 32u
#define NPARTS 1u
float dq(uint o, uint i) { return f32at(o + 4u * i); }
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    return dot(uintBitsToFloat(uvec4(w32[wb], w32[wb + 1u], w32[wb + 2u], w32[wb + 3u])), xv(xo, 0u)) +
           dot(uintBitsToFloat(uvec4(w32[wb + 4u], w32[wb + 5u], w32[wb + 6u], w32[wb + 7u])), xv(xo, 1u));
}

#elif defined(TYPE_F16)
#define ELEMS 8u
#define BYTES 16u
#define NPARTS 1u
float dq(uint o, uint i) { return h16(o + 2u * i); }
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    uvec4 u = uvec4(w32[wb], w32[wb + 1u], w32[wb + 2u], w32[wb + 3u]);
    vec2 a = unpackHalf2x16(u.x), b = unpackHalf2x16(u.y), c = unpackHalf2x16(u.z), d = unpackHalf2x16(u.w);
    return dot(vec4(a, b), xv(xo, 0u)) + dot(vec4(c, d), xv(xo, 1u));
}

#elif defined(TYPE_BF16)
#define ELEMS 8u
#define BYTES 16u
#define NPARTS 1u
float dq(uint o, uint i) { return uintBitsToFloat(b16(o + 2u * i) << 16); }
vec2 bf2(uint u) { return vec2(uintBitsToFloat(u << 16), uintBitsToFloat(u & 0xffff0000u)); }
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    return dot(vec4(bf2(w32[wb]), bf2(w32[wb + 1u])), xv(xo, 0u)) + dot(vec4(bf2(w32[wb + 2u]), bf2(w32[wb + 3u])), xv(xo, 1u));
}

#elif defined(TYPE_Q8_0)
// [f32 d][32 x i8]
#define ELEMS 32u
#define BYTES 36u
#define NPARTS 1u
float dq(uint o, uint i) { return f32at(o) * float(s8(o + 4u + i)); }
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    vec4 s = vec4(0.0);
    for (uint k = 0u; k < 8u; k++) s += vec4(sb4(w32[wb + 1u + k])) * xv(xo, k);
    return uintBitsToFloat(w32[wb]) * (s.x + s.y + s.z + s.w);
}

#elif defined(TYPE_Q4_0)
// [f32 d][16 bytes of nibbles]
#define ELEMS 32u
#define BYTES 20u
#define NPARTS 1u
float dq(uint o, uint i) {
    uint q = b8(o + 4u + (i & 15u));
    return f32at(o) * float(int(i < 16u ? (q & 15u) : (q >> 4)) - 8);
}
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    vec4 s = vec4(0.0);
    for (uint k = 0u; k < 4u; k++) {
        uint u = w32[wb + 1u + k];
        s += (vec4(lo4(u)) - 8.0) * xv(xo, k) + (vec4(hi4(u)) - 8.0) * xv(xo, k + 4u);
    }
    return uintBitsToFloat(w32[wb]) * (s.x + s.y + s.z + s.w);
}

#elif defined(TYPE_Q4_1)
// [f32 d][f32 m][16 bytes of nibbles]
#define ELEMS 32u
#define BYTES 24u
#define NPARTS 1u
float dq(uint o, uint i) {
    uint q = b8(o + 8u + (i & 15u));
    return float(i < 16u ? (q & 15u) : (q >> 4)) * f32at(o) + f32at(o + 4u);
}
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    vec4 s = vec4(0.0);
    vec4 sx = vec4(0.0);
    for (uint k = 0u; k < 4u; k++) {
        uint u = w32[wb + 2u + k];
        vec4 xa = xv(xo, k);
        vec4 xb = xv(xo, k + 4u);
        s += vec4(lo4(u)) * xa + vec4(hi4(u)) * xb;
        sx += xa + xb;
    }
    return uintBitsToFloat(w32[wb]) * (s.x + s.y + s.z + s.w) + uintBitsToFloat(w32[wb + 1u]) * (sx.x + sx.y + sx.z + sx.w);
}

#elif defined(TYPE_Q5_0)
// [f32 d][u32 qh][16 bytes of nibbles]
#define ELEMS 32u
#define BYTES 24u
#define NPARTS 1u
float dq(uint o, uint i) {
    uint qh = w32[(o >> 2) + 1u];
    uint q = b8(o + 8u + (i & 15u));
    uint x = (i < 16u ? (q & 15u) : (q >> 4)) | (((qh >> i) & 1u) << 4);
    return f32at(o) * float(int(x) - 16);
}
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    uint qh = w32[wb + 1u];
    vec4 s = vec4(0.0);
    for (uint k = 0u; k < 4u; k++) {
        uint u = w32[wb + 2u + k];
        uvec4 bl = (uvec4(qh) >> uvec4(4u * k, 4u * k + 1u, 4u * k + 2u, 4u * k + 3u)) & 1u;
        uvec4 bh = (uvec4(qh) >> uvec4(16u + 4u * k, 17u + 4u * k, 18u + 4u * k, 19u + 4u * k)) & 1u;
        s += (vec4(lo4(u) | (bl << 4)) - 16.0) * xv(xo, k) + (vec4(hi4(u) | (bh << 4)) - 16.0) * xv(xo, k + 4u);
    }
    return uintBitsToFloat(w32[wb]) * (s.x + s.y + s.z + s.w);
}

#elif defined(TYPE_Q5_1)
// [f32 d][f32 m][u32 qh][16 bytes of nibbles]
#define ELEMS 32u
#define BYTES 28u
#define NPARTS 1u
float dq(uint o, uint i) {
    uint qh = w32[(o >> 2) + 2u];
    uint q = b8(o + 12u + (i & 15u));
    uint x = (i < 16u ? (q & 15u) : (q >> 4)) | (((qh >> i) & 1u) << 4);
    return float(x) * f32at(o) + f32at(o + 4u);
}
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    uint qh = w32[wb + 2u];
    vec4 s = vec4(0.0);
    vec4 sx = vec4(0.0);
    for (uint k = 0u; k < 4u; k++) {
        uint u = w32[wb + 3u + k];
        uvec4 bl = (uvec4(qh) >> uvec4(4u * k, 4u * k + 1u, 4u * k + 2u, 4u * k + 3u)) & 1u;
        uvec4 bh = (uvec4(qh) >> uvec4(16u + 4u * k, 17u + 4u * k, 18u + 4u * k, 19u + 4u * k)) & 1u;
        vec4 xa = xv(xo, k);
        vec4 xb = xv(xo, k + 4u);
        s += vec4(lo4(u) | (bl << 4)) * xa + vec4(hi4(u) | (bh << 4)) * xb;
        sx += xa + xb;
    }
    return uintBitsToFloat(w32[wb]) * (s.x + s.y + s.z + s.w) + uintBitsToFloat(w32[wb + 1u]) * (sx.x + sx.y + sx.z + sx.w);
}

#elif defined(TYPE_IQ4_NL)
// [f32 d][16 bytes of nibbles]
#define ELEMS 32u
#define BYTES 20u
#define NPARTS 1u
float dq(uint o, uint i) {
    uint q = b8(o + 4u + (i & 15u));
    return f32at(o) * float(kvalues_iq4nl[i < 16u ? (q & 15u) : (q >> 4)]);
}
vec4 kv4(uvec4 n) { return vec4(float(kvalues_iq4nl[n.x]), float(kvalues_iq4nl[n.y]), float(kvalues_iq4nl[n.z]), float(kvalues_iq4nl[n.w])); }
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    vec4 s = vec4(0.0);
    for (uint k = 0u; k < 4u; k++) {
        uint u = w32[wb + 1u + k];
        s += kv4(lo4(u)) * xv(xo, k) + kv4(hi4(u)) * xv(xo, k + 4u);
    }
    return uintBitsToFloat(w32[wb]) * (s.x + s.y + s.z + s.w);
}

#elif defined(TYPE_Q2_K)
#define ELEMS 256u
#define BYTES 84u
#define NPARTS 4u
float dq(uint o, uint i) {
    uint g = i >> 4;
    uint n = g >> 3;
    uint s = (g & 7u) >> 1;
    uint half_ = g & 1u;
    uint q = (b8(o + 16u + 32u * n + 16u * half_ + (i & 15u)) >> (2u * s)) & 3u;
    uint sc = b8(o + g);
    return h16(o + 80u) * float(sc & 15u) * float(q) - h16(o + 82u) * float(sc >> 4);
}
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    float d = h16(o + 80u);
    float dmin = h16(o + 82u);
    float acc = 0.0;
    float mins = 0.0;
    uint n = part >> 1;
    {
        for (uint s = 2u * (part & 1u); s < 2u * (part & 1u) + 2u; s++) {
            for (uint half_ = 0u; half_ < 2u; half_++) {
                uint g = n * 8u + s * 2u + half_;
                uint sc = b8(o + g);
                vec4 sq = vec4(0.0);
                vec4 sx = vec4(0.0);
                for (uint k = 0u; k < 4u; k++) {
                    uint u = w32[wb + 4u + 8u * n + 4u * half_ + k];
                    uvec4 qv = (uvec4(u >> (2u * s)) >> uvec4(0u, 8u, 16u, 24u)) & 3u;
                    vec4 xx = xv(xo, g * 4u + k);
                    sq += vec4(qv) * xx;
                    sx += xx;
                }
                acc += float(sc & 15u) * (sq.x + sq.y + sq.z + sq.w);
                mins += float(sc >> 4) * (sx.x + sx.y + sx.z + sx.w);
            }
        }
    }
    return d * acc - dmin * mins;
}

#elif defined(TYPE_Q3_K)
// 112 bytes: 110 used, 2 spare
#define ELEMS 256u
#define BYTES 112u
#define NPARTS 4u
int q3scale(uint o, uint g) {
    uint a0 = w32[(o >> 2) + 24u];
    uint a1 = w32[(o >> 2) + 25u];
    uint tmp = w32[(o >> 2) + 26u];
    uint k1 = 0x03030303u;
    uint k2 = 0x0f0f0f0fu;
    uint r;
    switch (g >> 2) {
        case 0u: r = (a0 & k2) | (((tmp >> 0) & k1) << 4); break;
        case 1u: r = (a1 & k2) | (((tmp >> 2) & k1) << 4); break;
        case 2u: r = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4); break;
        default: r = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4); break;
    }
    return int((r >> (8u * (g & 3u))) & 255u) - 32;
}
float dq(uint o, uint i) {
    uint g = i >> 4;
    uint n = g >> 3;
    uint s = (g & 7u) >> 1;
    uint half_ = g & 1u;
    uint l = i & 15u;
    uint lo = (b8(o + 32u + 32u * n + 16u * half_ + l) >> (2u * s)) & 3u;
    uint hm = b8(o + 16u * half_ + l);
    int hi = (hm & (1u << (4u * n + s))) != 0u ? 0 : 4;
    return h16(o + 108u) * float(q3scale(o, g)) * float(int(lo) - hi);
}
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    float acc = 0.0;
    uint n = part >> 1;
    {
        for (uint s = 2u * (part & 1u); s < 2u * (part & 1u) + 2u; s++) {
            for (uint half_ = 0u; half_ < 2u; half_++) {
                uint g = n * 8u + s * 2u + half_;
                vec4 sv = vec4(0.0);
                for (uint k = 0u; k < 4u; k++) {
                    uint u = w32[wb + 8u + 8u * n + 4u * half_ + k];
                    uint hm = w32[wb + 4u * half_ + k];
                    uvec4 lo = (uvec4(u >> (2u * s)) >> uvec4(0u, 8u, 16u, 24u)) & 3u;
                    uvec4 hb = (uvec4(hm >> (4u * n + s)) >> uvec4(0u, 8u, 16u, 24u)) & 1u;
                    sv += (vec4(lo) - vec4(4u - 4u * hb)) * xv(xo, g * 4u + k);
                }
                acc += float(q3scale(o, g)) * (sv.x + sv.y + sv.z + sv.w);
            }
        }
    }
    return h16(o + 108u) * acc;
}

#elif defined(TYPE_Q4_K)
#define ELEMS 256u
#define BYTES 144u
#define NPARTS 4u
float dq(uint o, uint i) {
    uint c = i >> 6;
    uint half_ = (i >> 5) & 1u;
    uint l = i & 31u;
    uint sc, mn;
    sm4(o + 4u, 2u * c + half_, sc, mn);
    uint qb = b8(o + 16u + 32u * c + l);
    uint nib = half_ == 0u ? (qb & 15u) : (qb >> 4);
    return h16(o) * float(sc) * float(nib) - h16(o + 2u) * float(mn);
}
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    float d = h16(o);
    float dmin = h16(o + 2u);
    float acc = 0.0;
    float mins = 0.0;
    {
        uint c = part;
        uint sc0, mn0, sc1, mn1;
        sm4(o + 4u, 2u * c, sc0, mn0);
        sm4(o + 4u, 2u * c + 1u, sc1, mn1);
        vec4 sl = vec4(0.0), sh = vec4(0.0), xl = vec4(0.0), xh = vec4(0.0);
        for (uint k = 0u; k < 8u; k++) {
            uint u = w32[wb + 4u + 8u * c + k];
            vec4 xa = xv(xo, 16u * c + k);
            vec4 xb = xv(xo, 16u * c + 8u + k);
            sl += vec4(lo4(u)) * xa;
            sh += vec4(hi4(u)) * xb;
            xl += xa;
            xh += xb;
        }
        acc += float(sc0) * (sl.x + sl.y + sl.z + sl.w) + float(sc1) * (sh.x + sh.y + sh.z + sh.w);
        mins += float(mn0) * (xl.x + xl.y + xl.z + xl.w) + float(mn1) * (xh.x + xh.y + xh.z + xh.w);
    }
    return d * acc - dmin * mins;
}

#elif defined(TYPE_Q5_K)
#define ELEMS 256u
#define BYTES 176u
#define NPARTS 4u
float dq(uint o, uint i) {
    uint c = i >> 6;
    uint half_ = (i >> 5) & 1u;
    uint l = i & 31u;
    uint sc, mn;
    sm4(o + 4u, 2u * c + half_, sc, mn);
    uint qb = b8(o + 48u + 32u * c + l);
    uint nib = half_ == 0u ? (qb & 15u) : (qb >> 4);
    uint hb = (b8(o + 16u + l) >> (2u * c + half_)) & 1u;
    return h16(o) * float(sc) * float(nib | (hb << 4)) - h16(o + 2u) * float(mn);
}
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    float d = h16(o);
    float dmin = h16(o + 2u);
    float acc = 0.0;
    float mins = 0.0;
    {
        uint c = part;
        uint sc0, mn0, sc1, mn1;
        sm4(o + 4u, 2u * c, sc0, mn0);
        sm4(o + 4u, 2u * c + 1u, sc1, mn1);
        vec4 sl = vec4(0.0), sh = vec4(0.0), xl = vec4(0.0), xh = vec4(0.0);
        for (uint k = 0u; k < 8u; k++) {
            uint u = w32[wb + 12u + 8u * c + k];
            uint hq = w32[wb + 4u + k];
            uvec4 hl = (uvec4(hq >> (2u * c)) >> uvec4(0u, 8u, 16u, 24u)) & 1u;
            uvec4 hh = (uvec4(hq >> (2u * c + 1u)) >> uvec4(0u, 8u, 16u, 24u)) & 1u;
            vec4 xa = xv(xo, 16u * c + k);
            vec4 xb = xv(xo, 16u * c + 8u + k);
            sl += vec4(lo4(u) | (hl << 4)) * xa;
            sh += vec4(hi4(u) | (hh << 4)) * xb;
            xl += xa;
            xh += xb;
        }
        acc += float(sc0) * (sl.x + sl.y + sl.z + sl.w) + float(sc1) * (sh.x + sh.y + sh.z + sh.w);
        mins += float(mn0) * (xl.x + xl.y + xl.z + xl.w) + float(mn1) * (xh.x + xh.y + xh.z + xh.w);
    }
    return d * acc - dmin * mins;
}

#elif defined(TYPE_Q6_K)
// 212 bytes: 210 used, 2 spare
#define ELEMS 256u
#define BYTES 212u
#define NPARTS 8u
float dq(uint o, uint i) {
    uint n = i >> 7;
    uint r = i & 127u;
    uint part = r >> 5;
    uint l = r & 31u;
    uint is = l >> 4;
    uint qlb = b8(o + 64u * n + l + ((part & 1u) != 0u ? 32u : 0u));
    uint lo = part < 2u ? (qlb & 15u) : (qlb >> 4);
    uint hi = (b8(o + 128u + 32u * n + l) >> (2u * part)) & 3u;
    int q = int(lo | (hi << 4)) - 32;
    int sc = s8(o + 192u + 8u * n + is + 2u * part);
    return h16(o + 208u) * float(sc * q);
}
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    float acc = 0.0;
    uint n = part >> 2;
    {
        uint part_ = part & 3u;
        {
            // 32 weights: two scales (16 each).
            vec4 s0 = vec4(0.0), s1 = vec4(0.0);
            for (uint k = 0u; k < 8u; k++) {
                uint ql = w32[wb + 16u * n + 8u * (part_ & 1u) + k];
                uint qh = w32[wb + 32u + 8u * n + k];
                uvec4 lo = part_ < 2u ? lo4(ql) : hi4(ql);
                uvec4 hi = ((uvec4(qh >> (2u * part_)) >> uvec4(0u, 8u, 16u, 24u)) & 3u) << 4;
                vec4 q = vec4(lo | hi) - 32.0;
                vec4 xx = xv(xo, 32u * n + 8u * part_ + k);
                if (k < 4u) s0 += q * xx; else s1 += q * xx;
            }
            uint sbase = o + 192u + 8u * n + 2u * part_;
            acc += float(s8(sbase)) * (s0.x + s0.y + s0.z + s0.w) + float(s8(sbase + 1u)) * (s1.x + s1.y + s1.z + s1.w);
        }
    }
    return h16(o + 208u) * acc;
}

#elif defined(TYPE_IQ4_XS)
#define ELEMS 256u
#define BYTES 136u
#define NPARTS 8u
float dq(uint o, uint i) {
    uint ib = i >> 5;
    uint j = i & 31u;
    uint sh = b16(o + 2u);
    uint sl = b8(o + 4u + (ib >> 1));
    int ls = int(((sl >> (4u * (ib & 1u))) & 15u) | (((sh >> (2u * ib)) & 3u) << 4)) - 32;
    uint q = b8(o + 8u + 16u * ib + (j & 15u));
    uint nib = j < 16u ? (q & 15u) : (q >> 4);
    return h16(o) * float(ls) * float(kvalues_iq4nl[nib]);
}
vec4 kv4(uvec4 n) { return vec4(float(kvalues_iq4nl[n.x]), float(kvalues_iq4nl[n.y]), float(kvalues_iq4nl[n.z]), float(kvalues_iq4nl[n.w])); }
float dotpart(uint o, uint xo, uint part) {
    uint wb = o >> 2;
    uint sh = b16(o + 2u);
    float acc = 0.0;
    {
        uint ib = part;
        uint sl = b8(o + 4u + (ib >> 1));
        int ls = int(((sl >> (4u * (ib & 1u))) & 15u) | (((sh >> (2u * ib)) & 3u) << 4)) - 32;
        vec4 s = vec4(0.0);
        for (uint k = 0u; k < 4u; k++) {
            uint u = w32[wb + 2u + 4u * ib + k];
            s += kv4(lo4(u)) * xv(xo, 8u * ib + k) + kv4(hi4(u)) * xv(xo, 8u * ib + 4u + k);
        }
        acc += float(ls) * (s.x + s.y + s.z + s.w);
    }
    return h16(o) * acc;
}

#else
#error "select a weight format with -DTYPE_<NAME>"
#endif
