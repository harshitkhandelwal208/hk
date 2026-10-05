#!/usr/bin/env python3
"""Extract the IQ codebook grids from the reference `gguf` package into raw binary tables.

The Zig side embeds these files with @embedFile (src/quant/tables.zig). Re-run only when the
reference tables change, which they should not: the grids are part of the file format.

Run: python tools/gen_quant_tables.py  (needs `pip install gguf numpy`)
"""
from pathlib import Path

import numpy as np
from gguf import quants

OUT = Path(__file__).resolve().parent.parent / "src" / "quant" / "tables"


def dump(name, arr, dtype):
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(arr).astype(dtype).tobytes()
    (OUT / f"{name}.bin").write_bytes(data)
    print(f"{name}.bin: {len(data)} bytes")


def grid(cls):
    cls.init_grid()
    return cls.grid.reshape(cls.grid_shape)


def main():
    # Sign patterns shared by IQ2_XXS, IQ2_XS and IQ3_XXS (7 bit index to 8 sign bits).
    dump("ksigns_iq2xs", np.frombuffer(quants.IQ2_XXS.ksigns, dtype=np.uint8), np.uint8)
    # Codebooks. Entries are small non negative integers except the IQ1 grid, which is {-1, 0, 1}.
    dump("iq2xxs_grid", grid(quants.IQ2_XXS), np.uint8)   # 256 x 8
    dump("iq2xs_grid", grid(quants.IQ2_XS), np.uint8)     # 512 x 8
    dump("iq2s_grid", grid(quants.IQ2_S), np.uint8)       # 1024 x 8
    dump("iq3xxs_grid", grid(quants.IQ3_XXS), np.uint8)   # 256 x 4
    dump("iq3s_grid", grid(quants.IQ3_S), np.uint8)       # 512 x 4
    dump("iq1s_grid", grid(quants.IQ1_S), np.int8)        # 2048 x 8


if __name__ == "__main__":
    main()
