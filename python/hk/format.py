"""
HK Neural Tensor Format: Binary Specification Constants and Format Utilities.
128-byte aligned, zero-copy, cross-platform neural format specification.
"""


from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List, Union
from enum import IntEnum
import os
from ._lazy import lazy

np = lazy("numpy")
torch = lazy("torch")

MAGIC = b"HKNT"
VERSION_MAJOR = 1
VERSION_MINOR = 0
DEFAULT_ALIGNMENT_BYTES = 128
ALIGNMENT_BYTES = 128
UNIVERSAL_PAGE_ALIGNMENT_BYTES = 4096
APPLE_SILICON_ALIGNMENT_BYTES = 16384
DIRECT_DMA_ALIGNMENT_BYTES = 65536
HEADER_SIZE = 128

# Storage Types
STORAGE_F32 = 0x00
STORAGE_F16 = 0x01
STORAGE_BF16 = 0x02
STORAGE_FP8_E4M3 = 0x03
STORAGE_FP8_E5M2 = 0x04
STORAGE_INT8 = 0x05
STORAGE_INT32 = 0x06
STORAGE_INT64 = 0x07
STORAGE_UINT8 = 0x08
STORAGE_BOOL = 0x09
STORAGE_INT16 = 0x0A
STORAGE_UINT16 = 0x0B
STORAGE_UINT32 = 0x0C
STORAGE_UINT64 = 0x0D
STORAGE_F64 = 0x0E
STORAGE_DQ4 = 0x10
STORAGE_DQ8 = 0x11
STORAGE_DQ6 = 0x12
STORAGE_DQ12 = 0x13
STORAGE_DQT = 0x14
STORAGE_Q4_0 = 0x15
STORAGE_Q8_0 = 0x16
STORAGE_Q4_1 = 0x17
STORAGE_Q5_0 = 0x18
STORAGE_Q5_1 = 0x19
STORAGE_Q8_1 = 0x1A
STORAGE_SPARSE_F16 = 0x20
STORAGE_SPARSE_DQ8 = 0x21
STORAGE_SPARSE_2_4 = 0x22
STORAGE_SPARSE_DQ4_2_4 = 0x23
STORAGE_NULL_REF = 0x30
STORAGE_SHARED_REF = 0x31
STORAGE_LORA_REF = 0x32

# Advanced K-Quants (256-element super-blocks)
STORAGE_Q2_K = 0x40
STORAGE_Q3_K = 0x41
STORAGE_Q4_K = 0x42
STORAGE_Q5_K = 0x43
STORAGE_Q6_K = 0x44
STORAGE_Q8_K = 0x45

# Advanced I-Quants (Importance Matrix & Non-Linear Vector Quantization)
STORAGE_IQ1_S = 0x50
STORAGE_IQ1_M = 0x51
STORAGE_IQ2_XXS = 0x52
STORAGE_IQ2_XS = 0x53
STORAGE_IQ3_XXS = 0x54
STORAGE_IQ4_NL = 0x55
STORAGE_IQ4_XS = 0x56

# Microscaling & Ternary Formats
STORAGE_TQ1_0 = 0x60
STORAGE_TQ2_0 = 0x61
STORAGE_MXFP4 = 0x62
STORAGE_NVFP4 = 0x63


class StorageType(IntEnum):
    F32 = STORAGE_F32
    F16 = STORAGE_F16
    BF16 = STORAGE_BF16
    FP8_E4M3 = STORAGE_FP8_E4M3
    FP8_E5M2 = STORAGE_FP8_E5M2
    INT8 = STORAGE_INT8
    INT32 = STORAGE_INT32
    INT64 = STORAGE_INT64
    UINT8 = STORAGE_UINT8
    BOOL = STORAGE_BOOL
    INT16 = STORAGE_INT16
    UINT16 = STORAGE_UINT16
    UINT32 = STORAGE_UINT32
    UINT64 = STORAGE_UINT64
    F64 = STORAGE_F64
    DQ4 = STORAGE_DQ4
    DQ8 = STORAGE_DQ8
    DQ6 = STORAGE_DQ6
    DQ12 = STORAGE_DQ12
    DQT = STORAGE_DQT
    Q4_0 = STORAGE_Q4_0
    Q8_0 = STORAGE_Q8_0
    Q4_1 = STORAGE_Q4_1
    Q5_0 = STORAGE_Q5_0
    Q5_1 = STORAGE_Q5_1
    Q8_1 = STORAGE_Q8_1
    SPARSE_F16 = STORAGE_SPARSE_F16
    SPARSE_DQ8 = STORAGE_SPARSE_DQ8
    SPARSE_2_4 = STORAGE_SPARSE_2_4
    SPARSE_DQ4_2_4 = STORAGE_SPARSE_DQ4_2_4
    NULL_REF = STORAGE_NULL_REF
    SHARED_REF = STORAGE_SHARED_REF
    LORA_REF = STORAGE_LORA_REF
    Q2_K = STORAGE_Q2_K
    Q3_K = STORAGE_Q3_K
    Q4_K = STORAGE_Q4_K
    Q5_K = STORAGE_Q5_K
    Q6_K = STORAGE_Q6_K
    Q8_K = STORAGE_Q8_K
    IQ1_S = STORAGE_IQ1_S
    IQ1_M = STORAGE_IQ1_M
    IQ2_XXS = STORAGE_IQ2_XXS
    IQ2_XS = STORAGE_IQ2_XS
    IQ3_XXS = STORAGE_IQ3_XXS
    IQ4_NL = STORAGE_IQ4_NL
    IQ4_XS = STORAGE_IQ4_XS
    TQ1_0 = STORAGE_TQ1_0
    TQ2_0 = STORAGE_TQ2_0
    MXFP4 = STORAGE_MXFP4
    NVFP4 = STORAGE_NVFP4


# Tile Layouts
TILE_ROW_MAJOR = 0x00
TILE_COL_MAJOR = 0x01
TILE_16X16 = 0x02
TILE_16X8 = 0x03
TILE_32X16 = 0x04
TILE_BLOCK_SPARSE_2_4 = 0x05
TILE_32X32 = 0x06
TILE_64X64 = 0x07


class TileLayout(IntEnum):
    ROW_MAJOR = TILE_ROW_MAJOR
    COL_MAJOR = TILE_COL_MAJOR
    TILE_16X16 = TILE_16X16
    TILE_16X8 = TILE_16X8
    TILE_32X16 = TILE_32X16
    BLOCK_SPARSE_2_4 = TILE_BLOCK_SPARSE_2_4
    TILE_32X32 = TILE_32X32
    TILE_64X64 = TILE_64X64


# Sparsity Types
SPARSITY_NONE = 0x00
SPARSITY_BITMASK = 0x01
SPARSITY_CSR = 0x02
SPARSITY_2_4 = 0x03
SPARSITY_PHYSICAL_PRUNED = 0x04
SPARSITY_BSR = 0x05


class SparsityType(IntEnum):
    NONE = SPARSITY_NONE
    BITMASK = SPARSITY_BITMASK
    CSR = SPARSITY_CSR
    SPARSE_2_4 = SPARSITY_2_4
    PHYSICAL_PRUNED = SPARSITY_PHYSICAL_PRUNED
    BSR = SPARSITY_BSR


# Header Flags
FLAG_NONE = 0x00000000
FLAG_MMAP_COW = 0x00000001
FLAG_HAS_APPENDIX = 0x00000002
FLAG_ENCRYPTED = 0x00000004
FLAG_STRICT_128B = 0x00000008
FLAG_IS_SHARDED = 0x00000040
FLAG_RAW_WEIGHT_STORAGE = 0x00000080
FLAG_UNIVERSAL_PAGE_ALIGNED = 0x00000100

def _torch_maps() -> Dict[str, Dict[Any, Any]]:
    """The torch dtype tables. Built on first use so that this module imports without torch."""
    TORCH_DTYPE_TO_STORAGE = {
        torch.float32: STORAGE_F32,
        torch.float16: STORAGE_F16,
        torch.bfloat16: STORAGE_BF16,
        torch.int8: STORAGE_INT8,
        torch.int16: STORAGE_INT16,
        torch.int32: STORAGE_INT32,
        torch.int64: STORAGE_INT64,
        torch.uint8: STORAGE_UINT8,
        torch.float64: STORAGE_F64,
        torch.bool: STORAGE_BOOL,
    }

    STORAGE_TO_TORCH_DTYPE = {
        STORAGE_F32: torch.float32,
        STORAGE_F16: torch.float16,
        STORAGE_BF16: torch.bfloat16,
        STORAGE_INT8: torch.int8,
        STORAGE_INT16: torch.int16,
        STORAGE_INT32: torch.int32,
        STORAGE_INT64: torch.int64,
        STORAGE_UINT8: torch.uint8,
        STORAGE_F64: torch.float64,
        STORAGE_BOOL: torch.bool,
    }

    return {"TORCH_DTYPE_TO_STORAGE": TORCH_DTYPE_TO_STORAGE, "STORAGE_TO_TORCH_DTYPE": STORAGE_TO_TORCH_DTYPE}


def _numpy_maps() -> Dict[str, Dict[Any, Any]]:
    """The NumPy dtype tables, built on first use."""
    NUMPY_DTYPE_TO_STORAGE = {
        np.dtype('float32'): STORAGE_F32,
        np.dtype('float16'): STORAGE_F16,
        np.dtype('int8'): STORAGE_INT8,
        np.dtype('int16'): STORAGE_INT16,
        np.dtype('int32'): STORAGE_INT32,
        np.dtype('int64'): STORAGE_INT64,
        np.dtype('uint8'): STORAGE_UINT8,
        np.dtype('uint16'): STORAGE_UINT16,
        np.dtype('uint32'): STORAGE_UINT32,
        np.dtype('uint64'): STORAGE_UINT64,
        np.dtype('float64'): STORAGE_F64,
        np.dtype('bool'): STORAGE_BOOL,
    }

    STORAGE_TO_NUMPY_DTYPE = {
        STORAGE_F32: np.float32,
        STORAGE_F16: np.float16,
        STORAGE_BF16: np.uint16,
        STORAGE_INT8: np.int8,
        STORAGE_INT16: np.int16,
        STORAGE_INT32: np.int32,
        STORAGE_INT64: np.int64,
        STORAGE_UINT8: np.uint8,
        STORAGE_UINT16: np.uint16,
        STORAGE_UINT32: np.uint32,
        STORAGE_UINT64: np.uint64,
        STORAGE_F64: np.float64,
        STORAGE_BOOL: np.bool_,
    }
    return {"NUMPY_DTYPE_TO_STORAGE": NUMPY_DTYPE_TO_STORAGE, "STORAGE_TO_NUMPY_DTYPE": STORAGE_TO_NUMPY_DTYPE}


_DTYPE_MAP_CACHE: Dict[str, Dict[Any, Any]] = {}


def __getattr__(name: str) -> Any:
    if name in ("TORCH_DTYPE_TO_STORAGE", "STORAGE_TO_TORCH_DTYPE", "NUMPY_DTYPE_TO_STORAGE", "STORAGE_TO_NUMPY_DTYPE"):
        if name not in _DTYPE_MAP_CACHE:
            _DTYPE_MAP_CACHE.update(_torch_maps() if "TORCH" in name else _numpy_maps())
        return _DTYPE_MAP_CACHE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def align_forward(offset: int, alignment: int = 128) -> int:
    """Aligns byte offset forward to the next multiple of alignment."""
    return (offset + alignment - 1) & ~(alignment - 1)


def tile_matrix_16x16(mat: torch.Tensor) -> torch.Tensor:
    """Tiles a 2D matrix into 16x16 tensor core layout."""
    M, K = mat.shape
    pad_m = (16 - (M % 16)) % 16
    pad_k = (16 - (K % 16)) % 16
    if pad_m > 0 or pad_k > 0:
        mat = torch.nn.functional.pad(mat, (0, pad_k, 0, pad_m))
    padded_m, padded_k = mat.shape
    grid_m = padded_m // 16
    grid_k = padded_k // 16
    return mat.view(grid_m, 16, grid_k, 16).permute(0, 2, 1, 3).contiguous()


def untile_matrix_16x16(tiled: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
    """Restores tiled 16x16 layout back to standard row-major dense matrix."""
    M, K = original_shape
    grid_m, grid_k, tm, tk = tiled.shape
    permuted = tiled.permute(0, 2, 1, 3).contiguous()
    dense = permuted.view(grid_m * tm, grid_k * tk)
    return dense[:M, :K]
