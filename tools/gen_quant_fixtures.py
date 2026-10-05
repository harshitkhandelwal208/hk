#!/usr/bin/env python3
"""Generate quantization test fixtures from the reference `gguf` decoder.

Each fixture is a small binary file holding random but valid quantized blocks plus the
values the reference decoder produces for them. The Zig tests decode the same bytes and
compare, so our kernels are checked against an implementation we did not write.

Layout of tests/fixtures/quants/<type>.bin (all little endian):
    u32 ggml_type, u32 rows, u32 blocks_per_row, u32 block_elems, u32 block_bytes
    rows * blocks_per_row * block_bytes   raw quantized bytes
    rows * blocks_per_row * block_elems   f32 expected values

Run: python tools/gen_quant_fixtures.py  (needs `pip install gguf numpy`)
"""
import struct
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf import quants
from gguf.constants import GGML_QUANT_SIZES

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "quants"
ROWS = 4
LEGACY_BLOCKS = 8   # 32 element blocks per row
SUPER_BLOCKS = 2    # 256 element super blocks per row
MAX_ABS = 40.0      # keep decoded magnitudes tame so float comparisons stay meaningful

TYPES = [
    T.Q4_0, T.Q4_1, T.Q5_0, T.Q5_1, T.Q8_0,
    T.Q2_K, T.Q3_K, T.Q4_K, T.Q5_K, T.Q6_K,
    T.IQ2_XXS, T.IQ2_XS, T.IQ2_S, T.IQ3_XXS, T.IQ3_S,
    T.IQ1_S, T.IQ1_M, T.IQ4_NL, T.IQ4_XS,
    T.TQ1_0, T.TQ2_0, T.MXFP4, T.NVFP4, T.BF16,
]


def valid_block(rng, qtype, block_bytes, block_elems):
    """Draw random bytes until the reference decodes them to finite, bounded values."""
    for _ in range(100000):
        raw = rng.integers(0, 256, size=(1, block_bytes), dtype=np.uint8)
        vals = quants.dequantize(raw, qtype)
        if np.all(np.isfinite(vals)) and np.max(np.abs(vals)) < MAX_ABS:
            return raw[0]
    raise RuntimeError(f"could not draw a tame block for {qtype.name}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261005)
    for qtype in TYPES:
        block_elems, block_bytes = GGML_QUANT_SIZES[qtype]
        per_row = LEGACY_BLOCKS if block_elems <= 64 else SUPER_BLOCKS
        raw = np.stack([
            np.stack([valid_block(rng, qtype, block_bytes, block_elems) for _ in range(per_row)])
            for _ in range(ROWS)
        ])  # rows x blocks x bytes
        expected = quants.dequantize(raw.reshape(ROWS, per_row * block_bytes), qtype).astype("<f4")
        assert expected.shape == (ROWS, per_row * block_elems), expected.shape
        path = OUT / f"{qtype.name.lower()}.bin"
        with open(path, "wb") as f:
            f.write(struct.pack("<5I", int(qtype), ROWS, per_row, block_elems, block_bytes))
            f.write(raw.tobytes())
            f.write(expected.tobytes())
        print(f"{path.name}: {ROWS}x{per_row} blocks, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
