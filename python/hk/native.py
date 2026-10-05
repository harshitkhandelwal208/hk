"""
HK Native C ABI Bridge
Direct ctypes dispatch to native Zig SIMD compute engine and file format container.
"""

import ctypes
import json
import os
import platform
import struct
import sys
import types
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List, Sequence
import numpy as np
import torch

def _library_names() -> List[str]:
    """File names to look for, for this operating system and CPU only."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    arm = "arm" in machine or "aarch64" in machine
    if "windows" in system or sys.platform == "win32":
        return ["hk.dll", "libhk.dll"]
    if "darwin" in system:
        return ["libhk-macos-arm64.dylib" if arm else "libhk-macos-x86_64.dylib", "libhk.dylib"]
    if "linux" in system:
        return ["libhk-linux-aarch64.so" if arm else "libhk-linux-x86_64.so", "libhk.so"]
    return []


def _search_dirs() -> List[Path]:
    here = Path(__file__).resolve().parent
    dirs = []
    override = os.environ.get("HK_LIB_DIR")
    if override:
        dirs.append(Path(override))
    dirs += [
        here.parent.parent / "zig-out" / "lib",
        Path(os.getcwd()) / "zig-out" / "lib",
        here.parent.parent / "zig-out" / "bin",
        Path(os.getcwd()) / "zig-out" / "bin",
        here,
        here / "lib",
        Path(os.getcwd()),
    ]
    return dirs


def _load_native() -> Tuple[Optional[ctypes.CDLL], str, str]:
    """Loads the first library that works. Returns (library, path, reason it is missing)."""
    names = _library_names()
    if not names:
        return None, "", f"no native library is built for {platform.system()} {platform.machine()}"
    tried = []
    for d in _search_dirs():
        for name in names:
            candidate = d / name
            if not candidate.is_file():
                continue
            try:
                return ctypes.CDLL(str(candidate)), str(candidate), ""
            except OSError as e:
                tried.append(f"{candidate}: {e}")
    if tried:
        return None, "", "found a library but could not load it: " + "; ".join(tried)
    return None, "", "no native library found (build one with `zig build -Doptimize=ReleaseFast`, or set HK_LIB_DIR)"


_LIB, _LIB_PATH, _LIB_ERROR = _load_native()


def native_unavailable_reason() -> str:
    """Why the native library is not in use, or an empty string when it is."""
    return _LIB_ERROR


# Struct definitions matching C_TensorInfo
class C_TensorInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("storage_type", ctypes.c_uint8),
        ("tile_layout", ctypes.c_uint8),
        ("sparsity_type", ctypes.c_uint8),
        ("ndim", ctypes.c_uint8),
        ("shape", ctypes.c_uint64 * 8),
        ("data_offset", ctypes.c_uint64),
        ("data_size", ctypes.c_uint64),
        ("residual_offset", ctypes.c_uint64),
        ("residual_size", ctypes.c_uint64),
        ("scale_offset", ctypes.c_uint64),
        ("scale_size", ctypes.c_uint64),
        ("block_size", ctypes.c_uint16),
        ("sparsity_ratio", ctypes.c_float),
    ]

class C_HardwareCapabilities(ctypes.Structure):
    _fields_ = [
        ("vendor", ctypes.c_uint8),
        ("has_avx2", ctypes.c_uint8),
        ("has_avx512f", ctypes.c_uint8),
        ("has_avx512vnni", ctypes.c_uint8),
        ("has_avx_vnni", ctypes.c_uint8),
        ("has_amx", ctypes.c_uint8),
        ("has_arm_neon", ctypes.c_uint8),
        ("has_arm_sve", ctypes.c_uint8),
        ("is_apple_silicon", ctypes.c_uint8),
        ("has_rocm_ready", ctypes.c_uint8),
        ("has_npu_ready", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 5),
        ("optimal_page_alignment", ctypes.c_uint64),
        ("dma_hugepage_alignment", ctypes.c_uint64),
    ]

# Setup function signatures if library loaded
if _LIB is not None:
    if hasattr(_LIB, "hk_detect_hardware"):
        _LIB.hk_detect_hardware.argtypes = [ctypes.POINTER(C_HardwareCapabilities)]
        _LIB.hk_detect_hardware.restype = None
    if hasattr(_LIB, "hk_get_optimal_alignment"):
        _LIB.hk_get_optimal_alignment.argtypes = []
        _LIB.hk_get_optimal_alignment.restype = ctypes.c_size_t
    if hasattr(_LIB, "hk_is_raw_storage"):
        _LIB.hk_is_raw_storage.argtypes = [ctypes.c_void_p]
        _LIB.hk_is_raw_storage.restype = ctypes.c_int
    if hasattr(_LIB, "hk_is_universal_page_aligned"):
        _LIB.hk_is_universal_page_aligned.argtypes = [ctypes.c_void_p]
        _LIB.hk_is_universal_page_aligned.restype = ctypes.c_int
    if hasattr(_LIB, "hk_get_file_alignment"):
        _LIB.hk_get_file_alignment.argtypes = [ctypes.c_void_p]
        _LIB.hk_get_file_alignment.restype = ctypes.c_uint32
    if hasattr(_LIB, "hk_get_tensor_raw_ptr"):
        _LIB.hk_get_tensor_raw_ptr.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        _LIB.hk_get_tensor_raw_ptr.restype = ctypes.c_void_p
    if hasattr(_LIB, "hk_get_raw_buffer"):
        _LIB.hk_get_raw_buffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        _LIB.hk_get_raw_buffer.restype = ctypes.c_void_p
    if hasattr(_LIB, "hk_gemv_bf16"):
        _LIB.hk_gemv_bf16.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t]
        _LIB.hk_gemv_bf16.restype = None
    if hasattr(_LIB, "hk_gemv_f16"):
        _LIB.hk_gemv_f16.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t]
        _LIB.hk_gemv_f16.restype = None
    if hasattr(_LIB, "hk_gemv_int8"):
        _LIB.hk_gemv_int8.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t]
        _LIB.hk_gemv_int8.restype = None
    if hasattr(_LIB, "hk_dot_bf16"):
        _LIB.hk_dot_bf16.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        _LIB.hk_dot_bf16.restype = ctypes.c_float
    if hasattr(_LIB, "hk_dot_f16"):
        _LIB.hk_dot_f16.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        _LIB.hk_dot_f16.restype = ctypes.c_float
    if hasattr(_LIB, "hk_dot_int8"):
        _LIB.hk_dot_int8.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        _LIB.hk_dot_int8.restype = ctypes.c_int32
    # Reader API
    _LIB.hk_open.argtypes = [ctypes.c_char_p]
    _LIB.hk_open.restype = ctypes.c_void_p

    _LIB.hk_close.argtypes = [ctypes.c_void_p]
    _LIB.hk_close.restype = None

    _LIB.hk_get_tensor_count.argtypes = [ctypes.c_void_p]
    _LIB.hk_get_tensor_count.restype = ctypes.c_uint64

    _LIB.hk_get_tensor_info.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(C_TensorInfo)]
    _LIB.hk_get_tensor_info.restype = ctypes.c_int

    if hasattr(_LIB, "hk_get_all_tensor_infos"):
        _LIB.hk_get_all_tensor_infos.argtypes = [ctypes.c_void_p, ctypes.POINTER(C_TensorInfo), ctypes.c_uint64]
        _LIB.hk_get_all_tensor_infos.restype = ctypes.c_uint64

    _LIB.hk_get_tensor_data.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    _LIB.hk_get_tensor_data.restype = ctypes.c_void_p

    _LIB.hk_get_tensor_residual.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    _LIB.hk_get_tensor_residual.restype = ctypes.c_void_p

    _LIB.hk_get_tensor_scales.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    _LIB.hk_get_tensor_scales.restype = ctypes.c_void_p

    _LIB.hk_dequantize_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    _LIB.hk_dequantize_f32.restype = ctypes.c_int

    _LIB.hk_get_metadata_string.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    _LIB.hk_get_metadata_string.restype = ctypes.c_char_p

    _LIB.hk_get_metadata_int.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64)]
    _LIB.hk_get_metadata_int.restype = ctypes.c_int

    if hasattr(_LIB, "hk_reader_is_sharded"):
        _LIB.hk_reader_is_sharded.argtypes = [ctypes.c_void_p]
        _LIB.hk_reader_is_sharded.restype = ctypes.c_int
        _LIB.hk_reader_get_split_index.argtypes = [ctypes.c_void_p]
        _LIB.hk_reader_get_split_index.restype = ctypes.c_uint16
        _LIB.hk_reader_get_split_count.argtypes = [ctypes.c_void_p]
        _LIB.hk_reader_get_split_count.restype = ctypes.c_uint16

    if hasattr(_LIB, "hk_metadata_patch_in_place"):
        _LIB.hk_metadata_patch_in_place.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        _LIB.hk_metadata_patch_in_place.restype = ctypes.c_int

    # SIMD Compute API
    _LIB.hk_dot_product_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    _LIB.hk_dot_product_f32.restype = ctypes.c_float

    _LIB.hk_gemv_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    _LIB.hk_gemv_f32.restype = None

    _LIB.hk_gemm_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    _LIB.hk_gemm_f32.restype = None

    _LIB.hk_fused_gemv_nf4.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint32,
    ]
    _LIB.hk_fused_gemv_nf4.restype = None

    _LIB.hk_fused_gemv_dq8.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint32,
    ]
    _LIB.hk_fused_gemv_dq8.restype = None

    # Net2Net Growth API
    _LIB.hk_net2wider.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_float,
        ctypes.c_uint64,
    ]
    _LIB.hk_net2wider.restype = ctypes.c_int

    _LIB.hk_net2deeper.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    _LIB.hk_net2deeper.restype = None

    # Ampere 2:4 Structured Sparsity API
    _LIB.hk_pack_2_4.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    _LIB.hk_pack_2_4.restype = ctypes.c_int

    _LIB.hk_unpack_2_4.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_float),
    ]
    _LIB.hk_unpack_2_4.restype = ctypes.c_int

    # SwiGLU Net2WiderNet API
    _LIB.hk_net2wider_swiglu.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_uint64,
    ]
    _LIB.hk_net2wider_swiglu.restype = ctypes.c_int

    # Vocab Expansion API
    _LIB.hk_expand_vocab.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    _LIB.hk_expand_vocab.restype = ctypes.c_int

    # Plasticity Masking API
    _LIB.hk_plasticity_mask_rows.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    _LIB.hk_plasticity_mask_rows.restype = None

    _LIB.hk_plasticity_mask_cols.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    _LIB.hk_plasticity_mask_cols.restype = None

    # SIMD Activations & Norms
    _LIB.hk_forward_swiglu.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    _LIB.hk_forward_swiglu.restype = ctypes.c_int

    _LIB.hk_forward_rmsnorm.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    _LIB.hk_forward_rmsnorm.restype = None

    _LIB.hk_forward_silu.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    _LIB.hk_forward_silu.restype = None

    # Writer API
    _LIB.hk_writer_create.argtypes = [ctypes.c_uint64]
    _LIB.hk_writer_create.restype = ctypes.c_void_p

    _LIB.hk_writer_destroy.argtypes = [ctypes.c_void_p]
    _LIB.hk_writer_destroy.restype = None

    _LIB.hk_writer_add_metadata_string.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    _LIB.hk_writer_add_metadata_string.restype = ctypes.c_int

    _LIB.hk_writer_add_metadata_int.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int64]
    _LIB.hk_writer_add_metadata_int.restype = ctypes.c_int

    _LIB.hk_writer_add_metadata_float.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
    _LIB.hk_writer_add_metadata_float.restype = ctypes.c_int

    _LIB.hk_writer_add_metadata_bool.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    _LIB.hk_writer_add_metadata_bool.restype = ctypes.c_int

    if hasattr(_LIB, "hk_writer_add_metadata_json"):
        _LIB.hk_writer_add_metadata_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        _LIB.hk_writer_add_metadata_json.restype = ctypes.c_int

    _LIB.hk_writer_add_tensor.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.c_float,
    ]
    _LIB.hk_writer_add_tensor.restype = ctypes.c_int

    _LIB.hk_writer_write_to_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    _LIB.hk_writer_write_to_file.restype = ctypes.c_int

    if hasattr(_LIB, "hk_writer_set_sharding"):
        _LIB.hk_writer_set_sharding.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16]
        _LIB.hk_writer_set_sharding.restype = None

    if hasattr(_LIB, "hk_writer_set_raw_storage"):
        _LIB.hk_writer_set_raw_storage.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _LIB.hk_writer_set_raw_storage.restype = None

    if hasattr(_LIB, "hk_quantize_block_q4_0"):
        _LIB.hk_quantize_block_q4_0.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_void_p]
        _LIB.hk_quantize_block_q4_0.restype = ctypes.c_int

    if hasattr(_LIB, "hk_dequantize_block_q4_0"):
        _LIB.hk_dequantize_block_q4_0.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        _LIB.hk_dequantize_block_q4_0.restype = ctypes.c_int

    if hasattr(_LIB, "hk_quantize_block_q8_0"):
        _LIB.hk_quantize_block_q8_0.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_void_p]
        _LIB.hk_quantize_block_q8_0.restype = ctypes.c_int

    if hasattr(_LIB, "hk_dequantize_block_q8_0"):
        _LIB.hk_dequantize_block_q8_0.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        _LIB.hk_dequantize_block_q8_0.restype = ctypes.c_int

    if hasattr(_LIB, "hk_quantize_block_q5_k"):
        _LIB.hk_quantize_block_q5_k.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_void_p]
        _LIB.hk_quantize_block_q5_k.restype = ctypes.c_int

    if hasattr(_LIB, "hk_dequantize_block_q5_k"):
        _LIB.hk_dequantize_block_q5_k.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        _LIB.hk_dequantize_block_q5_k.restype = ctypes.c_int

    if hasattr(_LIB, "hk_quantize_block_q3_k"):
        _LIB.hk_quantize_block_q3_k.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_void_p]
        _LIB.hk_quantize_block_q3_k.restype = ctypes.c_int

    if hasattr(_LIB, "hk_dequantize_block_q3_k"):
        _LIB.hk_dequantize_block_q3_k.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        _LIB.hk_dequantize_block_q3_k.restype = ctypes.c_int

    if hasattr(_LIB, "hk_quantize_block_q6_k"):
        _LIB.hk_quantize_block_q6_k.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_void_p]
        _LIB.hk_quantize_block_q6_k.restype = ctypes.c_int

    if hasattr(_LIB, "hk_quantize_block_q2_k"):
        _LIB.hk_quantize_block_q2_k.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_void_p]
        _LIB.hk_quantize_block_q2_k.restype = ctypes.c_int

    if hasattr(_LIB, "hk_quantize_block_q4_k"):
        _LIB.hk_quantize_block_q4_k.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_void_p]
        _LIB.hk_quantize_block_q4_k.restype = ctypes.c_int

    if hasattr(_LIB, "hk_dequantize_block_q4_k"):
        _LIB.hk_dequantize_block_q4_k.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        _LIB.hk_dequantize_block_q4_k.restype = ctypes.c_int

    if hasattr(_LIB, "hk_quantize_block_q8_k"):
        _LIB.hk_quantize_block_q8_k.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_void_p]
        _LIB.hk_quantize_block_q8_k.restype = ctypes.c_int

    if hasattr(_LIB, "hk_dequantize_block_q8_k"):
        _LIB.hk_dequantize_block_q8_k.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        _LIB.hk_dequantize_block_q8_k.restype = ctypes.c_int

    if hasattr(_LIB, "hk_dequantize_block_q6_k"):
        _LIB.hk_dequantize_block_q6_k.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        _LIB.hk_dequantize_block_q6_k.restype = ctypes.c_int

    if hasattr(_LIB, "hk_dequantize_block_q2_k"):
        _LIB.hk_dequantize_block_q2_k.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        _LIB.hk_dequantize_block_q2_k.restype = ctypes.c_int

    if hasattr(_LIB, "hk_dequantize_block_iq4_nl"):
        _LIB.hk_dequantize_block_iq4_nl.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        _LIB.hk_dequantize_block_iq4_nl.restype = ctypes.c_int

    if hasattr(_LIB, "hk_dequantize_block_mxfp4"):
        _LIB.hk_dequantize_block_mxfp4.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        _LIB.hk_dequantize_block_mxfp4.restype = ctypes.c_int

    if hasattr(_LIB, "hk_dequantize_block_nvfp4"):
        _LIB.hk_dequantize_block_nvfp4.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        _LIB.hk_dequantize_block_nvfp4.restype = ctypes.c_int

    for fn in [
        "hk_quantize_tensor_q4_0", "hk_dequantize_tensor_q4_0",
        "hk_quantize_tensor_q8_0", "hk_dequantize_tensor_q8_0",
        "hk_quantize_tensor_q4_k", "hk_dequantize_tensor_q4_k",
        "hk_quantize_tensor_q8_k", "hk_dequantize_tensor_q8_k",
        "hk_quantize_tensor_q6_k", "hk_dequantize_tensor_q6_k",
        "hk_quantize_tensor_q5_k", "hk_dequantize_tensor_q5_k",
        "hk_quantize_tensor_q3_k", "hk_dequantize_tensor_q3_k",
        "hk_quantize_tensor_q2_k", "hk_dequantize_tensor_q2_k",
    ]:
        if hasattr(_LIB, fn):
            func = getattr(_LIB, fn)
            if "dequantize" in fn:
                func.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_float)]
            else:
                func.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.c_char_p]
            func.restype = ctypes.c_int

    if hasattr(_LIB, "hk_quantize_tensor_nf4"):
        _LIB.hk_quantize_tensor_nf4.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        _LIB.hk_quantize_tensor_nf4.restype = ctypes.c_int

    if hasattr(_LIB, "hk_dequantize_tensor_nf4"):
        _LIB.hk_dequantize_tensor_nf4.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        _LIB.hk_dequantize_tensor_nf4.restype = ctypes.c_int

    if hasattr(_LIB, "hk_convert_gguf"):
        _LIB.hk_convert_gguf.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        _LIB.hk_convert_gguf.restype = ctypes.c_int

    if hasattr(_LIB, "hk_export_gguf"):
        _LIB.hk_export_gguf.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        _LIB.hk_export_gguf.restype = ctypes.c_int

    if hasattr(_LIB, "hk_rope_permute_hf_to_gguf"):
        _LIB.hk_rope_permute_hf_to_gguf.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        _LIB.hk_rope_permute_hf_to_gguf.restype = None

    if hasattr(_LIB, "hk_rope_unpermute_gguf_to_hf"):
        _LIB.hk_rope_unpermute_gguf_to_hf.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        _LIB.hk_rope_unpermute_gguf_to_hf.restype = None

    if hasattr(_LIB, "hk_layernorm_offset_f32"):
        _LIB.hk_layernorm_offset_f32.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.c_float]
        _LIB.hk_layernorm_offset_f32.restype = None

    if hasattr(_LIB, "hk_gemv_q8_0"):
        _LIB.hk_gemv_q8_0.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.c_uint64]
        _LIB.hk_gemv_q8_0.restype = None

    if hasattr(_LIB, "hk_gemv_q4_0"):
        _LIB.hk_gemv_q4_0.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.c_uint64]
        _LIB.hk_gemv_q4_0.restype = None

    if hasattr(_LIB, "hk_gemv_q4_k"):
        _LIB.hk_gemv_q4_k.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.c_uint64]
        _LIB.hk_gemv_q4_k.restype = None

    if hasattr(_LIB, "hk_tokenizer_load_from_file"):
        _LIB.hk_tokenizer_load_from_file.argtypes = [ctypes.c_char_p]
        _LIB.hk_tokenizer_load_from_file.restype = ctypes.c_void_p

        _LIB.hk_tokenizer_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
        ]
        _LIB.hk_tokenizer_encode.restype = ctypes.c_uint32

        _LIB.hk_tokenizer_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        _LIB.hk_tokenizer_decode.restype = ctypes.c_uint32

        _LIB.hk_tokenizer_get_vocab_size.argtypes = [ctypes.c_void_p]
        _LIB.hk_tokenizer_get_vocab_size.restype = ctypes.c_uint32

        _LIB.hk_tokenizer_free.argtypes = [ctypes.c_void_p]
        _LIB.hk_tokenizer_free.restype = None

    if hasattr(_LIB, "hk_engine_load_from_file"):
        _LIB.hk_engine_load_from_file.argtypes = [ctypes.c_char_p]
        _LIB.hk_engine_load_from_file.restype = ctypes.c_void_p

        _LIB.hk_engine_forward.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
        ]
        _LIB.hk_engine_forward.restype = ctypes.c_int

        _LIB.hk_engine_reset_cache.argtypes = [ctypes.c_void_p]
        _LIB.hk_engine_reset_cache.restype = None

        _LIB.hk_engine_get_vocab_size.argtypes = [ctypes.c_void_p]
        _LIB.hk_engine_get_vocab_size.restype = ctypes.c_uint32

        _LIB.hk_engine_free.argtypes = [ctypes.c_void_p]
        _LIB.hk_engine_free.restype = None

        if hasattr(_LIB, "hk_engine_forward_tokens"):
            _LIB.hk_engine_forward_tokens.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_float),
            ]
            _LIB.hk_engine_forward_tokens.restype = ctypes.c_int
            _LIB.hk_engine_get_context_size.argtypes = [ctypes.c_void_p]
            _LIB.hk_engine_get_context_size.restype = ctypes.c_uint32
            _LIB.hk_engine_last_error.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
            _LIB.hk_engine_last_error.restype = ctypes.c_uint32

    if hasattr(_LIB, "hk_sample_token"):
        _LIB.hk_sample_token.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64,
            ctypes.c_float,
            ctypes.c_uint32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.c_uint64,
        ]
        _LIB.hk_sample_token.restype = ctypes.c_uint32

    if hasattr(_LIB, "hk_convert_safetensors"):
        _LIB.hk_convert_safetensors.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint8]
        _LIB.hk_convert_safetensors.restype = ctypes.c_int

    if hasattr(_LIB, "hk_hf_detect_architecture"):
        _LIB.hk_hf_detect_architecture.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        _LIB.hk_hf_detect_architecture.restype = ctypes.c_int

    if hasattr(_LIB, "hk_hf_map_tensor_name"):
        _LIB.hk_hf_map_tensor_name.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_char_p, ctypes.c_size_t]
        _LIB.hk_hf_map_tensor_name.restype = ctypes.c_int

    if hasattr(_LIB, "hk_context_truncate"):
        _LIB.hk_context_truncate.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        _LIB.hk_context_truncate.restype = ctypes.c_int

    if hasattr(_LIB, "hk_governor_can_grow"):
        _LIB.hk_governor_can_grow.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        _LIB.hk_governor_can_grow.restype = ctypes.c_int

    if hasattr(_LIB, "hk_governor_can_grow_batch"):
        _LIB.hk_governor_can_grow_batch.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        _LIB.hk_governor_can_grow_batch.restype = ctypes.c_int

    if hasattr(_LIB, "hk_expand_vocab_embeddings"):
        _LIB.hk_expand_vocab_embeddings.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.c_uint64,
        ]
        _LIB.hk_expand_vocab_embeddings.restype = ctypes.c_int

    if hasattr(_LIB, "hk_init_plasticity_mask"):
        _LIB.hk_init_plasticity_mask.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        _LIB.hk_init_plasticity_mask.restype = ctypes.c_int


def is_native_available() -> bool:
    return _LIB is not None


# High-level Python functions wrapping native SIMD compute

def native_dot(a: np.ndarray, b: np.ndarray) -> float:
    """SIMD dot product in Zig."""
    if _LIB is None:
        return float(np.dot(a, b))
    a_c = np.ascontiguousarray(a, dtype=np.float32)
    b_c = np.ascontiguousarray(b, dtype=np.float32)
    n = a_c.size
    return float(_LIB.hk_dot_product_f32(
        a_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
    ))


def native_gemv(W: np.ndarray, x: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
    """SIMD matrix-vector product y = W * x + b in Zig."""
    if _LIB is None:
        y = np.matmul(W, x)
        if bias is not None:
            y += bias
        return y.astype(np.float32)

    W_c = np.ascontiguousarray(W, dtype=np.float32)
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    M, K = W_c.shape
    y = np.empty(M, dtype=np.float32)

    bias_ptr = None
    if bias is not None:
        b_c = np.ascontiguousarray(bias, dtype=np.float32)
        bias_ptr = b_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    _LIB.hk_gemv_f32(
        W_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bias_ptr,
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M,
        K,
    )
    return y


def native_gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """SIMD matrix multiplication C = A * B in Zig."""
    if _LIB is None:
        return np.matmul(A, B).astype(np.float32)

    A_c = np.ascontiguousarray(A, dtype=np.float32)
    B_c = np.ascontiguousarray(B, dtype=np.float32)
    M, K = A_c.shape
    K2, N = B_c.shape
    assert K == K2, f"Dimension mismatch: {K} vs {K2}"
    C = np.empty((M, N), dtype=np.float32)

    _LIB.hk_gemm_f32(
        A_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M,
        K,
        N,
    )
    return C


def native_fused_gemv_nf4(
    packed_W: np.ndarray,
    scales: np.ndarray,
    x: np.ndarray,
    M: int,
    K: int,
    bias: Optional[np.ndarray] = None,
    block_size: int = 32,
) -> np.ndarray:
    """Native Zig fused NF4 dequantization and GEMV without intermediate float weight buffer."""
    if _LIB is None:
        raise RuntimeError("Native HK library not available for fused NF4 GEMV.")

    pW_c = np.ascontiguousarray(packed_W, dtype=np.uint8)
    sc_c = np.ascontiguousarray(scales, dtype=np.float32)
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    y = np.empty(M, dtype=np.float32)

    bias_ptr = None
    if bias is not None:
        b_c = np.ascontiguousarray(bias, dtype=np.float32)
        bias_ptr = b_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    _LIB.hk_fused_gemv_nf4(
        pW_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        sc_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bias_ptr,
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M,
        K,
        block_size,
    )
    return y


def native_net2wider(
    w_in_old: np.ndarray,
    b_in_old: Optional[np.ndarray],
    new_out: int,
    w_out_old: Optional[np.ndarray] = None,
    noise_std: float = 0.0,
    seed: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Native Net2WiderNet function-preserving width expansion in Zig."""
    if _LIB is None:
        raise RuntimeError("Native HK library not available for Net2WiderNet.")

    w1_c = np.ascontiguousarray(w_in_old, dtype=np.float32)
    old_out, in_f = w1_c.shape

    w1_new = np.empty((new_out, in_f), dtype=np.float32)
    b1_new = np.empty(new_out, dtype=np.float32) if b_in_old is not None else None

    b1_old_ptr = None
    b1_new_ptr = None
    if b_in_old is not None:
        b1_c = np.ascontiguousarray(b_in_old, dtype=np.float32)
        b1_old_ptr = b1_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b1_new_ptr = b1_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    w_out_old_ptr = None
    w_out_new = None
    w_out_new_ptr = None
    out_f = 0
    if w_out_old is not None:
        w2_c = np.ascontiguousarray(w_out_old, dtype=np.float32)
        out_f = w2_c.shape[0]
        w_out_new = np.empty((out_f, new_out), dtype=np.float32)
        w_out_old_ptr = w2_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w_out_new_ptr = w_out_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    err = _LIB.hk_net2wider(
        w1_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b1_old_ptr,
        w1_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b1_new_ptr,
        w_out_old_ptr,
        w_out_new_ptr,
        old_out,
        new_out,
        in_f,
        out_f,
        noise_std,
        seed,
    )
    if err != 0:
        raise RuntimeError(f"Native hk_net2wider failed with code {err}")

    return w1_new, b1_new, w_out_new


def native_net2deeper(dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Native Net2DeeperNet identity depth expansion in Zig."""
    if _LIB is None:
        weights = np.eye(dim, dtype=np.float32)
        bias = np.zeros(dim, dtype=np.float32)
        return weights, bias

    weights = np.empty((dim, dim), dtype=np.float32)
    bias = np.empty(dim, dtype=np.float32)
    _LIB.hk_net2deeper(
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dim,
    )
    return weights, bias


def native_pack_2_4(dense_tensor: torch.Tensor) -> Optional[bytes]:
    """Native Ampere 2:4 structured sparsity packing in Zig."""
    if _LIB is None:
        return None
    flat = dense_tensor.detach().cpu().to(torch.float32).contiguous().numpy().flatten()
    n = flat.size
    pad_n = (4 - (n % 4)) % 4
    if pad_n > 0:
        flat = np.pad(flat, (0, pad_n))
        n = flat.size
    num_groups = n // 4
    meta_len = (num_groups + 1) // 2
    val_offset = (meta_len + 3) & ~3
    total_len = val_offset + (n // 2) * 4
    out_buf = bytearray(total_len)
    c_out = (ctypes.c_uint8 * total_len).from_buffer(out_buf)
    err = _LIB.hk_pack_2_4(
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
        c_out,
    )
    if err != 0:
        return None
    return bytes(out_buf)


def native_unpack_2_4(packed_data: Any, target_shape: Tuple[int, ...]) -> Optional[torch.Tensor]:
    """Native Ampere 2:4 structured sparsity unpacking directly into PyTorch float tensor in Zig."""
    if _LIB is None:
        return None
    n_elements = int(np.prod(target_shape))
    pad_n = (4 - (n_elements % 4)) % 4
    total_elements = n_elements + pad_n

    if isinstance(packed_data, (bytes, bytearray, memoryview)):
        np_packed = np.frombuffer(packed_data, dtype=np.uint8)
    elif isinstance(packed_data, np.ndarray):
        np_packed = np.ascontiguousarray(packed_data, dtype=np.uint8)
    else:
        np_packed = np.frombuffer(packed_data, dtype=np.uint8)

    out_tensor = torch.empty(total_elements, dtype=torch.float32)
    err = _LIB.hk_unpack_2_4(
        np_packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        np_packed.nbytes,
        total_elements,
        ctypes.cast(out_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)),
    )
    if err != 0:
        return None
    if pad_n > 0:
        out_tensor = out_tensor[:n_elements]
    return out_tensor.view(*target_shape)


def native_quantize_q4_k(weights: np.ndarray) -> bytes:
    """Quantizes 256 float32 weights to a Q4_K super-block (144 bytes)."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_block_q4_k"):
        raise RuntimeError("Native Q4_K quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    assert w_c.size == 256
    buf = bytearray(144)
    c_buf = (ctypes.c_uint8 * 144).from_buffer(buf)
    err = _LIB.hk_quantize_block_q4_k(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 256, ctypes.byref(c_buf))
    if err != 0:
        raise RuntimeError(f"Native Q4_K quantization failed: {err}")
    return bytes(buf)


def native_dequantize_q4_k(block_bytes: bytes, count: int = 256) -> np.ndarray:
    """Dequantizes a Q4_K super-block (144 bytes) to float32 weights."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_block_q4_k"):
        raise RuntimeError("Native Q4_K dequantizer not available")
    buf = (ctypes.c_uint8 * len(block_bytes)).from_buffer_copy(block_bytes)
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_block_q4_k(ctypes.byref(buf), count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native Q4_K dequantization failed: {err}")
    return out


def native_quantize_q8_k(weights: np.ndarray) -> bytes:
    """Quantizes 256 float32 weights to a Q8_K super-block (292 bytes)."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_block_q8_k"):
        raise RuntimeError("Native Q8_K quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    assert w_c.size == 256
    buf = bytearray(292)
    c_buf = (ctypes.c_uint8 * 292).from_buffer(buf)
    err = _LIB.hk_quantize_block_q8_k(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 256, ctypes.byref(c_buf))
    if err != 0:
        raise RuntimeError(f"Native Q8_K quantization failed: {err}")
    return bytes(buf)


def native_dequantize_q8_k(block_bytes: bytes, count: int = 256) -> np.ndarray:
    """Dequantizes a Q8_K super-block (292 bytes) to float32 weights."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_block_q8_k"):
        raise RuntimeError("Native Q8_K dequantizer not available")
    buf = (ctypes.c_uint8 * len(block_bytes)).from_buffer_copy(block_bytes)
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_block_q8_k(ctypes.byref(buf), count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native Q8_K dequantization failed: {err}")
    return out


def native_dequantize_q6_k(block_bytes: bytes, count: int = 256) -> np.ndarray:
    """Dequantizes a Q6_K super-block (210 bytes) to float32 weights."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_block_q6_k"):
        raise RuntimeError("Native Q6_K dequantizer not available")
    buf = (ctypes.c_uint8 * len(block_bytes)).from_buffer_copy(block_bytes)
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_block_q6_k(ctypes.byref(buf), count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native Q6_K dequantization failed: {err}")
    return out


def native_dequantize_q2_k(block_bytes: bytes, count: int = 256) -> np.ndarray:
    """Dequantizes a Q2_K super-block (84 bytes) to float32 weights."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_block_q2_k"):
        raise RuntimeError("Native Q2_K dequantizer not available")
    buf = (ctypes.c_uint8 * len(block_bytes)).from_buffer_copy(block_bytes)
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_block_q2_k(ctypes.byref(buf), count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native Q2_K dequantization failed: {err}")
    return out


def native_quantize_q4_0(weights: np.ndarray) -> bytes:
    """Quantizes 32 float32 weights to a Q4_0 block (18 bytes)."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_block_q4_0"):
        raise RuntimeError("Native Q4_0 quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    assert w_c.size == 32
    buf = bytearray(18)
    c_buf = (ctypes.c_uint8 * 18).from_buffer(buf)
    err = _LIB.hk_quantize_block_q4_0(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 32, ctypes.byref(c_buf))
    if err != 0:
        raise RuntimeError(f"Native Q4_0 quantization failed: {err}")
    return bytes(buf)


def native_dequantize_q4_0(block_bytes: bytes, count: int = 32) -> np.ndarray:
    """Dequantizes a Q4_0 block (18 bytes) to float32 weights."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_block_q4_0"):
        raise RuntimeError("Native Q4_0 dequantizer not available")
    buf = (ctypes.c_uint8 * len(block_bytes)).from_buffer_copy(block_bytes)
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_block_q4_0(ctypes.byref(buf), count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native Q4_0 dequantization failed: {err}")
    return out


def native_quantize_q8_0(weights: np.ndarray) -> bytes:
    """Quantizes 32 float32 weights to a Q8_0 block (34 bytes)."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_block_q8_0"):
        raise RuntimeError("Native Q8_0 quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    assert w_c.size == 32
    buf = bytearray(34)
    c_buf = (ctypes.c_uint8 * 34).from_buffer(buf)
    err = _LIB.hk_quantize_block_q8_0(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 32, ctypes.byref(c_buf))
    if err != 0:
        raise RuntimeError(f"Native Q8_0 quantization failed: {err}")
    return bytes(buf)


def native_dequantize_q8_0(block_bytes: bytes, count: int = 32) -> np.ndarray:
    """Dequantizes a Q8_0 block (34 bytes) to float32 weights."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_block_q8_0"):
        raise RuntimeError("Native Q8_0 dequantizer not available")
    buf = (ctypes.c_uint8 * len(block_bytes)).from_buffer_copy(block_bytes)
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_block_q8_0(ctypes.byref(buf), count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native Q8_0 dequantization failed: {err}")
    return out


def native_quantize_q5_k(weights: np.ndarray) -> bytes:
    """Quantizes 256 float32 weights to a Q5_K super-block (176 bytes)."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_block_q5_k"):
        raise RuntimeError("Native Q5_K quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    assert w_c.size == 256
    buf = bytearray(176)
    c_buf = (ctypes.c_uint8 * 176).from_buffer(buf)
    err = _LIB.hk_quantize_block_q5_k(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 256, ctypes.byref(c_buf))
    if err != 0:
        raise RuntimeError(f"Native Q5_K quantization failed: {err}")
    return bytes(buf)


def native_dequantize_q5_k(block_bytes: bytes, count: int = 256) -> np.ndarray:
    """Dequantizes a Q5_K super-block (176 bytes) to float32 weights."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_block_q5_k"):
        raise RuntimeError("Native Q5_K dequantizer not available")
    buf = (ctypes.c_uint8 * len(block_bytes)).from_buffer_copy(block_bytes)
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_block_q5_k(ctypes.byref(buf), count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native Q5_K dequantization failed: {err}")
    return out


def native_quantize_q3_k(weights: np.ndarray) -> bytes:
    """Quantizes 256 float32 weights to a Q3_K super-block (110 bytes)."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_block_q3_k"):
        raise RuntimeError("Native Q3_K quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    assert w_c.size == 256
    buf = bytearray(110)
    c_buf = (ctypes.c_uint8 * 110).from_buffer(buf)
    err = _LIB.hk_quantize_block_q3_k(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 256, ctypes.byref(c_buf))
    if err != 0:
        raise RuntimeError(f"Native Q3_K quantization failed: {err}")
    return bytes(buf)


def native_dequantize_q3_k(block_bytes: bytes, count: int = 256) -> np.ndarray:
    """Dequantizes a Q3_K super-block (110 bytes) to float32 weights."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_block_q3_k"):
        raise RuntimeError("Native Q3_K dequantizer not available")
    buf = (ctypes.c_uint8 * len(block_bytes)).from_buffer_copy(block_bytes)
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_block_q3_k(ctypes.byref(buf), count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native Q3_K dequantization failed: {err}")
    return out


def native_quantize_q6_k(weights: np.ndarray) -> bytes:
    """Quantizes 256 float32 weights to a Q6_K super-block (210 bytes)."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_block_q6_k"):
        raise RuntimeError("Native Q6_K quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    assert w_c.size == 256
    buf = bytearray(210)
    c_buf = (ctypes.c_uint8 * 210).from_buffer(buf)
    err = _LIB.hk_quantize_block_q6_k(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 256, ctypes.byref(c_buf))
    if err != 0:
        raise RuntimeError(f"Native Q6_K quantization failed: {err}")
    return bytes(buf)


def native_quantize_q2_k(weights: np.ndarray) -> bytes:
    """Quantizes 256 float32 weights to a Q2_K super-block (84 bytes)."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_block_q2_k"):
        raise RuntimeError("Native Q2_K quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    assert w_c.size == 256
    buf = bytearray(84)
    c_buf = (ctypes.c_uint8 * 84).from_buffer(buf)
    err = _LIB.hk_quantize_block_q2_k(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 256, ctypes.byref(c_buf))
    if err != 0:
        raise RuntimeError(f"Native Q2_K quantization failed: {err}")
    return bytes(buf)


def native_quantize_tensor_q4_0(weights: np.ndarray) -> bytes:
    """Batch quantizes a full float32 array to Q4_0 using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_tensor_q4_0"):
        raise RuntimeError("Native tensor Q4_0 quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    n = w_c.size
    assert n % 32 == 0
    num_blocks = n // 32
    buf = bytearray(num_blocks * 18)
    c_buf = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
    err = _LIB.hk_quantize_tensor_q4_0(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, ctypes.cast(c_buf, ctypes.c_char_p))
    if err != 0:
        raise RuntimeError(f"Native tensor Q4_0 quantize failed: {err}")
    return bytes(buf)


def native_dequantize_tensor_q4_0(packed_bytes: bytes, count: int) -> np.ndarray:
    """Batch dequantizes a full Q4_0 byte buffer to float32 using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_tensor_q4_0"):
        raise RuntimeError("Native tensor Q4_0 dequantizer not available")
    assert count % 32 == 0
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_tensor_q4_0(packed_bytes, count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native tensor Q4_0 dequantize failed: {err}")
    return out


def native_quantize_tensor_q8_0(weights: np.ndarray) -> bytes:
    """Batch quantizes a full float32 array to Q8_0 using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_tensor_q8_0"):
        raise RuntimeError("Native tensor Q8_0 quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    n = w_c.size
    assert n % 32 == 0
    num_blocks = n // 32
    buf = bytearray(num_blocks * 34)
    c_buf = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
    err = _LIB.hk_quantize_tensor_q8_0(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, ctypes.cast(c_buf, ctypes.c_char_p))
    if err != 0:
        raise RuntimeError(f"Native tensor Q8_0 quantize failed: {err}")
    return bytes(buf)


def native_dequantize_tensor_q8_0(packed_bytes: bytes, count: int) -> np.ndarray:
    """Batch dequantizes a full Q8_0 byte buffer to float32 using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_tensor_q8_0"):
        raise RuntimeError("Native tensor Q8_0 dequantizer not available")
    assert count % 32 == 0
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_tensor_q8_0(packed_bytes, count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native tensor Q8_0 dequantize failed: {err}")
    return out


def native_quantize_tensor_q4_k(weights: np.ndarray) -> bytes:
    """Batch quantizes a full float32 array to Q4_K using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_tensor_q4_k"):
        raise RuntimeError("Native tensor Q4_K quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    n = w_c.size
    assert n % 256 == 0
    num_blocks = n // 256
    buf = bytearray(num_blocks * 144)
    c_buf = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
    err = _LIB.hk_quantize_tensor_q4_k(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, ctypes.cast(c_buf, ctypes.c_char_p))
    if err != 0:
        raise RuntimeError(f"Native tensor Q4_K quantize failed: {err}")
    return bytes(buf)


def native_dequantize_tensor_q4_k(packed_bytes: bytes, count: int) -> np.ndarray:
    """Batch dequantizes a full Q4_K byte buffer to float32 using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_tensor_q4_k"):
        raise RuntimeError("Native tensor Q4_K dequantizer not available")
    assert count % 256 == 0
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_tensor_q4_k(packed_bytes, count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native tensor Q4_K dequantize failed: {err}")
    return out


def native_quantize_tensor_q8_k(weights: np.ndarray) -> bytes:
    """Batch quantizes a full float32 array to Q8_K using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_tensor_q8_k"):
        raise RuntimeError("Native tensor Q8_K quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    n = w_c.size
    assert n % 256 == 0
    num_blocks = n // 256
    buf = bytearray(num_blocks * 292)
    c_buf = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
    err = _LIB.hk_quantize_tensor_q8_k(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, ctypes.cast(c_buf, ctypes.c_char_p))
    if err != 0:
        raise RuntimeError(f"Native tensor Q8_K quantize failed: {err}")
    return bytes(buf)


def native_dequantize_tensor_q8_k(packed_bytes: bytes, count: int) -> np.ndarray:
    """Batch dequantizes a full Q8_K byte buffer to float32 using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_tensor_q8_k"):
        raise RuntimeError("Native tensor Q8_K dequantizer not available")
    assert count % 256 == 0
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_tensor_q8_k(packed_bytes, count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native tensor Q8_K dequantize failed: {err}")
    return out


def native_quantize_tensor_q6_k(weights: np.ndarray) -> bytes:
    """Batch quantizes a full float32 array to Q6_K using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_tensor_q6_k"):
        raise RuntimeError("Native tensor Q6_K quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    n = w_c.size
    assert n % 256 == 0
    num_blocks = n // 256
    buf = bytearray(num_blocks * 210)
    c_buf = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
    err = _LIB.hk_quantize_tensor_q6_k(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, ctypes.cast(c_buf, ctypes.c_char_p))
    if err != 0:
        raise RuntimeError(f"Native tensor Q6_K quantize failed: {err}")
    return bytes(buf)


def native_dequantize_tensor_q6_k(packed_bytes: bytes, count: int) -> np.ndarray:
    """Batch dequantizes a full Q6_K byte buffer to float32 using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_tensor_q6_k"):
        raise RuntimeError("Native tensor Q6_K dequantizer not available")
    assert count % 256 == 0
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_tensor_q6_k(packed_bytes, count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native tensor Q6_K dequantize failed: {err}")
    return out


def native_quantize_tensor_q5_k(weights: np.ndarray) -> bytes:
    """Batch quantizes a full float32 array to Q5_K using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_tensor_q5_k"):
        raise RuntimeError("Native tensor Q5_K quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    n = w_c.size
    assert n % 256 == 0
    num_blocks = n // 256
    buf = bytearray(num_blocks * 176)
    c_buf = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
    err = _LIB.hk_quantize_tensor_q5_k(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, ctypes.cast(c_buf, ctypes.c_char_p))
    if err != 0:
        raise RuntimeError(f"Native tensor Q5_K quantize failed: {err}")
    return bytes(buf)


def native_dequantize_tensor_q5_k(packed_bytes: bytes, count: int) -> np.ndarray:
    """Batch dequantizes a full Q5_K byte buffer to float32 using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_tensor_q5_k"):
        raise RuntimeError("Native tensor Q5_K dequantizer not available")
    assert count % 256 == 0
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_tensor_q5_k(packed_bytes, count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native tensor Q5_K dequantize failed: {err}")
    return out


def native_quantize_tensor_q3_k(weights: np.ndarray) -> bytes:
    """Batch quantizes a full float32 array to Q3_K using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_tensor_q3_k"):
        raise RuntimeError("Native tensor Q3_K quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    n = w_c.size
    assert n % 256 == 0
    num_blocks = n // 256
    buf = bytearray(num_blocks * 110)
    c_buf = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
    err = _LIB.hk_quantize_tensor_q3_k(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, ctypes.cast(c_buf, ctypes.c_char_p))
    if err != 0:
        raise RuntimeError(f"Native tensor Q3_K quantize failed: {err}")
    return bytes(buf)


def native_dequantize_tensor_q3_k(packed_bytes: bytes, count: int) -> np.ndarray:
    """Batch dequantizes a full Q3_K byte buffer to float32 using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_tensor_q3_k"):
        raise RuntimeError("Native tensor Q3_K dequantizer not available")
    assert count % 256 == 0
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_tensor_q3_k(packed_bytes, count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native tensor Q3_K dequantize failed: {err}")
    return out


def native_quantize_tensor_q2_k(weights: np.ndarray) -> bytes:
    """Batch quantizes a full float32 array to Q2_K using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_tensor_q2_k"):
        raise RuntimeError("Native tensor Q2_K quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    n = w_c.size
    assert n % 256 == 0
    num_blocks = n // 256
    buf = bytearray(num_blocks * 84)
    c_buf = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
    err = _LIB.hk_quantize_tensor_q2_k(w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, ctypes.cast(c_buf, ctypes.c_char_p))
    if err != 0:
        raise RuntimeError(f"Native tensor Q2_K quantize failed: {err}")
    return bytes(buf)


def native_dequantize_tensor_q2_k(packed_bytes: bytes, count: int) -> np.ndarray:
    """Batch dequantizes a full Q2_K byte buffer to float32 using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_tensor_q2_k"):
        raise RuntimeError("Native tensor Q2_K dequantizer not available")
    assert count % 256 == 0
    out = np.empty(count, dtype=np.float32)
    err = _LIB.hk_dequantize_tensor_q2_k(packed_bytes, count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if err != 0:
        raise RuntimeError(f"Native tensor Q2_K dequantize failed: {err}")
    return out


def native_quantize_tensor_nf4(
    weights: np.ndarray,
    block_size: int = 32,
    compute_residual: bool = False,
) -> Tuple[bytes, bytes, Optional[bytes]]:
    """Batch quantizes a full float32 array to NF4 dual-mode using native Zig SIMD without large memory allocations."""
    if _LIB is None or not hasattr(_LIB, "hk_quantize_tensor_nf4"):
        raise RuntimeError("Native tensor NF4 quantizer not available")
    w_c = np.ascontiguousarray(weights, dtype=np.float32)
    n = w_c.size
    assert n % block_size == 0
    num_blocks = n // block_size
    bytes_per_block = (block_size + 1) // 2

    packed_buf = bytearray(num_blocks * bytes_per_block)
    c_packed = (ctypes.c_uint8 * len(packed_buf)).from_buffer(packed_buf)

    scales = np.empty(num_blocks, dtype=np.float32)
    c_scales = scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    c_res = None
    res = None
    if compute_residual:
        res = np.empty(n, dtype=np.float32)
        c_res = res.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    err = _LIB.hk_quantize_tensor_nf4(
        w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
        block_size,
        c_packed,
        c_scales,
        c_res,
    )
    if err != 0:
        raise RuntimeError(f"Native tensor NF4 quantize failed: {err}")
    res_bytes = res.tobytes() if compute_residual and res is not None else None
    return bytes(packed_buf), scales.tobytes(), res_bytes


def native_dequantize_tensor_nf4(
    packed_bytes: bytes,
    scales_bytes: bytes,
    count: int,
    block_size: int = 32,
    residual_bytes: Optional[bytes] = None,
) -> np.ndarray:
    """Batch dequantizes a full NF4 byte buffer to float32 using native Zig SIMD."""
    if _LIB is None or not hasattr(_LIB, "hk_dequantize_tensor_nf4"):
        raise RuntimeError("Native tensor NF4 dequantizer not available")
    assert count % block_size == 0
    out = np.empty(count, dtype=np.float32)

    scales = np.ascontiguousarray(np.frombuffer(scales_bytes, dtype=np.float32))
    c_scales = scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    c_res = None
    if residual_bytes is not None and len(residual_bytes) > 0:
        res = np.ascontiguousarray(np.frombuffer(residual_bytes, dtype=np.float32))
        c_res = res.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    err = _LIB.hk_dequantize_tensor_nf4(
        packed_bytes,
        c_scales,
        count,
        block_size,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        c_res,
    )
    if err != 0:
        raise RuntimeError(f"Native tensor NF4 dequantize failed: {err}")
    return out


def native_net2wider_swiglu(
    w_gate_old: np.ndarray,
    w_up_old: np.ndarray,
    w_down_old: np.ndarray,
    new_intermediate_size: int,
    b_gate_old: Optional[np.ndarray] = None,
    b_up_old: Optional[np.ndarray] = None,
    b_down_old: Optional[np.ndarray] = None,
    zero_init: bool = True,
    noise_std: float = 0.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Native Zig Net2WiderNet transformation for modern SwiGLU transformer MLP layers."""
    if _LIB is None:
        raise RuntimeError("Native HK library (hk.dll) is not available")

    w_g_c = np.ascontiguousarray(w_gate_old, dtype=np.float32)
    w_u_c = np.ascontiguousarray(w_up_old, dtype=np.float32)
    w_d_c = np.ascontiguousarray(w_down_old, dtype=np.float32)

    old_inter, in_f = w_g_c.shape
    out_f = w_d_c.shape[0]

    w_g_new = np.empty((new_intermediate_size, in_f), dtype=np.float32)
    w_u_new = np.empty((new_intermediate_size, in_f), dtype=np.float32)
    w_d_new = np.empty((out_f, new_intermediate_size), dtype=np.float32)

    b_g_new = np.empty(new_intermediate_size, dtype=np.float32) if b_gate_old is not None else None
    b_u_new = np.empty(new_intermediate_size, dtype=np.float32) if b_up_old is not None else None
    b_d_new = np.empty(out_f, dtype=np.float32) if b_down_old is not None else None

    b_g_old_ptr = np.ascontiguousarray(b_gate_old, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if b_gate_old is not None else None
    b_g_new_ptr = b_g_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if b_g_new is not None else None

    b_u_old_ptr = np.ascontiguousarray(b_up_old, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if b_up_old is not None else None
    b_u_new_ptr = b_u_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if b_u_new is not None else None

    b_d_old_ptr = np.ascontiguousarray(b_down_old, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if b_down_old is not None else None
    b_d_new_ptr = b_d_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if b_d_new is not None else None

    err = _LIB.hk_net2wider_swiglu(
        w_g_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_g_old_ptr,
        w_u_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_u_old_ptr,
        w_d_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_d_old_ptr,
        w_g_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_g_new_ptr,
        w_u_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_u_new_ptr,
        w_d_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_d_new_ptr,
        old_inter,
        new_intermediate_size,
        in_f,
        out_f,
        1 if zero_init else 0,
        noise_std,
        seed,
    )
    if err != 0:
        raise RuntimeError(f"Native hk_net2wider_swiglu failed with code {err}")

    return w_g_new, w_u_new, w_d_new, b_g_new, b_u_new, b_d_new


def native_expand_vocab(
    embed_old: np.ndarray,
    new_vocab_size: int,
    lm_head_old: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Native dynamic vocabulary expansion in Zig."""
    if _LIB is None:
        raise RuntimeError("Native HK library (hk.dll) is not available")

    e_c = np.ascontiguousarray(embed_old, dtype=np.float32)
    old_vocab, hidden_dim = e_c.shape

    e_new = np.empty((new_vocab_size, hidden_dim), dtype=np.float32)
    h_new = np.empty((new_vocab_size, hidden_dim), dtype=np.float32) if lm_head_old is not None else None

    h_old_ptr = np.ascontiguousarray(lm_head_old, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if lm_head_old is not None else None
    h_new_ptr = h_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if h_new is not None else None

    err = _LIB.hk_expand_vocab(
        e_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        e_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h_old_ptr,
        h_new_ptr,
        old_vocab,
        new_vocab_size,
        hidden_dim,
        seed,
    )
    if err != 0:
        raise RuntimeError(f"Native hk_expand_vocab failed with code {err}")

    return e_new, h_new


def native_plasticity_mask_rows(grad: torch.Tensor, cutoff_rows: int) -> None:
    """In-place gradient plasticity zeroing for preserved base rows in Zig."""
    if _LIB is None or not grad.is_floating_point():
        grad[:cutoff_rows].zero_()
        return
    c_grad = grad.detach().contiguous()
    cols = grad.numel() // grad.shape[0]
    _LIB.hk_plasticity_mask_rows(
        ctypes.cast(c_grad.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        grad.numel(),
        cutoff_rows,
        cols,
    )


def native_plasticity_mask_cols(grad: torch.Tensor, cutoff_cols: int) -> None:
    """In-place gradient plasticity zeroing for preserved base columns in Zig."""
    if _LIB is None or not grad.is_floating_point():
        grad[:, :cutoff_cols].zero_()
        return
    c_grad = grad.detach().contiguous()
    num_rows = grad.shape[0]
    stride_cols = grad.shape[1]
    _LIB.hk_plasticity_mask_cols(
        ctypes.cast(c_grad.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        grad.numel(),
        num_rows,
        cutoff_cols,
        stride_cols,
    )


def native_forward_silu(x: torch.Tensor) -> torch.Tensor:
    """Native SIMD SiLU activation in Zig."""
    if _LIB is None or not x.is_contiguous() or x.dtype != torch.float32:
        return torch.nn.functional.silu(x)
    out = torch.empty_like(x)
    _LIB.hk_forward_silu(
        ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        x.numel(),
    )
    return out


def native_forward_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Native SIMD RMSNorm in Zig."""
    if _LIB is None or not x.is_contiguous() or x.dtype != torch.float32:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * weight
    out = torch.empty_like(x)
    _LIB.hk_forward_rmsnorm(
        ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(weight.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(eps),
        ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        x.numel(),
    )
    return out


class NativeHKWriter:
    """High-level Python wrapper around native Zig HKWriter for 128-byte aligned container creation."""
    def __init__(self, alignment: int = 128):
        if _LIB is None:
            raise RuntimeError("Native HK library (hk.dll) is not available")
        self.ptr = _LIB.hk_writer_create(ctypes.c_uint64(alignment))
        if not self.ptr:
            raise RuntimeError("Failed to create native HKWriter")
        self._pinned_refs: List[Any] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if hasattr(self, "_pinned_refs"):
            self._pinned_refs.clear()
        if hasattr(self, "ptr") and self.ptr:
            _LIB.hk_writer_destroy(self.ptr)
            self.ptr = None

    def __del__(self):
        self.close()

    def set_sharding(self, split_index: int, split_count: int):
        if _LIB is not None and hasattr(_LIB, "hk_writer_set_sharding"):
            _LIB.hk_writer_set_sharding(self.ptr, ctypes.c_uint16(split_index), ctypes.c_uint16(split_count))

    def set_raw_storage(self, enabled: bool = True):
        if _LIB is not None and hasattr(_LIB, "hk_writer_set_raw_storage"):
            _LIB.hk_writer_set_raw_storage(self.ptr, ctypes.c_int(1 if enabled else 0))

    def add_metadata_string(self, key: str, value: str):
        err = _LIB.hk_writer_add_metadata_string(self.ptr, key.encode("utf-8"), value.encode("utf-8"))
        if err != 0:
            raise RuntimeError(f"Failed to add metadata string {key}")

    def add_metadata_int(self, key: str, value: int):
        err = _LIB.hk_writer_add_metadata_int(self.ptr, key.encode("utf-8"), ctypes.c_int64(value))
        if err != 0:
            raise RuntimeError(f"Failed to add metadata int {key}")

    def add_metadata_float(self, key: str, value: float):
        err = _LIB.hk_writer_add_metadata_float(self.ptr, key.encode("utf-8"), ctypes.c_double(value))
        if err != 0:
            raise RuntimeError(f"Failed to add metadata float {key}")

    def add_metadata_bool(self, key: str, value: bool):
        err = _LIB.hk_writer_add_metadata_bool(self.ptr, key.encode("utf-8"), ctypes.c_int(1 if value else 0))
        if err != 0:
            raise RuntimeError(f"Failed to add metadata bool {key}")

    def add_metadata_json(self, key: str, value: str):
        if _LIB is not None and hasattr(_LIB, "hk_writer_add_metadata_json"):
            err = _LIB.hk_writer_add_metadata_json(self.ptr, key.encode("utf-8"), value.encode("utf-8"))
            if err != 0:
                raise RuntimeError(f"Failed to add metadata json {key}")
        else:
            self.add_metadata_string(key, value)

    def add_tensor(
        self,
        name: str,
        tensor_data: Union[bytes, bytearray, np.ndarray, torch.Tensor],
        shape: Tuple[int, ...],
        storage_type: int = 0, # STORAGE_F32 = 0x00
        tile_layout: int = 0,
        sparsity_type: int = 0,
        sparsity_ratio: float = 0.0,
    ):
        if isinstance(tensor_data, torch.Tensor):
            t = tensor_data.detach().cpu().contiguous()
            self._pinned_refs.append(t)
            data_len = t.numel() * t.element_size()
            if data_len == 0:
                c_data_ptr = ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_uint8))
            else:
                c_data_ptr = ctypes.cast(ctypes.c_void_p(t.data_ptr()), ctypes.POINTER(ctypes.c_uint8))
        elif isinstance(tensor_data, np.ndarray):
            arr = np.ascontiguousarray(tensor_data)
            self._pinned_refs.append(arr)
            data_len = arr.nbytes
            if data_len == 0:
                c_data_ptr = ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_uint8))
            else:
                c_data_ptr = ctypes.cast(ctypes.c_void_p(arr.ctypes.data), ctypes.POINTER(ctypes.c_uint8))
        elif isinstance(tensor_data, (bytes, bytearray, memoryview)):
            data_len = len(tensor_data)
            if data_len == 0:
                c_data_ptr = ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_uint8))
            else:
                c_buf = (ctypes.c_uint8 * data_len).from_buffer_copy(tensor_data)
                self._pinned_refs.append(c_buf)
                c_data_ptr = ctypes.cast(c_buf, ctypes.POINTER(ctypes.c_uint8))
        else:
            raise TypeError(f"Unsupported tensor data type: {type(tensor_data)}")

        ndim = len(shape)
        c_shape = (ctypes.c_uint64 * 8)(*([shape[i] if i < ndim else 0 for i in range(8)]))

        err = _LIB.hk_writer_add_tensor(
            self.ptr,
            name.encode("utf-8"),
            ctypes.c_uint8(storage_type),
            ctypes.c_uint8(tile_layout),
            ctypes.c_uint8(sparsity_type),
            ctypes.c_uint8(ndim),
            c_shape,
            c_data_ptr,
            ctypes.c_uint64(data_len),
            ctypes.c_float(sparsity_ratio),
        )
        if err != 0:
            raise RuntimeError(f"Failed to add tensor {name}")

    def write_to_file(self, path: str):
        err = _LIB.hk_writer_write_to_file(self.ptr, path.encode("utf-8"))
        if err != 0:
            raise RuntimeError(f"Failed to write HK file {path}")
        if hasattr(self, "_pinned_refs"):
            self._pinned_refs.clear()


class NativeHKReader:
    """High-level Python wrapper around native Zig HKReader for zero-copy memory-mapped loading."""
    def __init__(self, file_path: str):
        if _LIB is None:
            raise RuntimeError("Native HK library (hk.dll) is not available")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"HK file not found: {file_path}")
        self.file_path = file_path
        self.ptr = _LIB.hk_open(file_path.encode("utf-8"))
        if not self.ptr:
            raise RuntimeError(f"Failed to open HK file: {file_path}")

        self.tensor_count = int(_LIB.hk_get_tensor_count(self.ptr))
        self.tensors: Dict[str, Dict[str, Any]] = {}
        if hasattr(_LIB, "hk_get_all_tensor_infos") and self.tensor_count > 0:
            InfoArray = C_TensorInfo * self.tensor_count
            info_arr = InfoArray()
            n = _LIB.hk_get_all_tensor_infos(self.ptr, info_arr, self.tensor_count)
            for i in range(n):
                info = info_arr[i]
                name = info.name.decode("utf-8") if info.name else f"tensor_{i}"
                shape = tuple(info.shape[d] for d in range(info.ndim))
                self.tensors[name] = {
                    "index": i,
                    "name": name,
                    "storage_type": info.storage_type,
                    "tile_layout": info.tile_layout,
                    "sparsity_type": info.sparsity_type,
                    "ndim": info.ndim,
                    "shape": shape,
                    "data_offset": info.data_offset,
                    "data_size": info.data_size,
                    "residual_size": info.residual_size,
                    "scale_size": info.scale_size,
                    "block_size": info.block_size,
                    "sparsity_ratio": info.sparsity_ratio,
                }
        else:
            for i in range(self.tensor_count):
                info = C_TensorInfo()
                if _LIB.hk_get_tensor_info(self.ptr, i, ctypes.byref(info)) == 0:
                    name = info.name.decode("utf-8") if info.name else f"tensor_{i}"
                    shape = tuple(info.shape[d] for d in range(info.ndim))
                    self.tensors[name] = {
                        "index": i,
                        "name": name,
                        "storage_type": info.storage_type,
                        "tile_layout": info.tile_layout,
                        "sparsity_type": info.sparsity_type,
                        "ndim": info.ndim,
                        "shape": shape,
                        "data_offset": info.data_offset,
                        "data_size": info.data_size,
                        "residual_size": info.residual_size,
                        "scale_size": info.scale_size,
                        "block_size": info.block_size,
                        "sparsity_ratio": info.sparsity_ratio,
                    }

        self.metadata: Dict[str, Any] = {}
        self.alignment: int = 128
        self.header: Any = types.SimpleNamespace(alignment=128)
        try:
            mmap_buf = None
            if _LIB is not None and hasattr(_LIB, "hk_get_raw_buffer"):
                raw_buf_size = ctypes.c_uint64(0)
                raw_ptr = _LIB.hk_get_raw_buffer(self.ptr, ctypes.byref(raw_buf_size))
                if raw_ptr and raw_buf_size.value >= 128:
                    mmap_buf = (ctypes.c_char * raw_buf_size.value).from_address(raw_ptr)

            if mmap_buf is not None:
                header_bytes = bytes(mmap_buf[:128])
            else:
                with open(file_path, "rb") as f:
                    header_bytes = f.read(128)

            if len(header_bytes) >= 128:
                (
                    magic,
                    ver_maj,
                    ver_min,
                    flags,
                    align,
                    split_index,
                    t_count,
                    kv_count,
                    meta_off,
                    meta_size,
                    toc_off,
                    toc_size,
                    data_off,
                    app_off,
                    chk,
                    split_count,
                    _,
                ) = struct.unpack_from("<4s H H I H H Q Q Q Q Q Q Q Q Q H 38s", header_bytes, 0)
                self.alignment = align
                self.split_index = split_index
                self.split_count = split_count
                self.is_sharded = bool(flags & 0x00000040 or split_count > 1)
                self.is_raw_storage = bool(flags & 0x00000080)
                self.is_universal_page_aligned = bool(flags & 0x00000100)
                self.is_tensor_core_aligned = bool((align % 128) == 0)
                self.header = types.SimpleNamespace(
                    magic=magic,
                    version_major=ver_maj,
                    version_minor=ver_min,
                    flags=flags,
                    alignment=align,
                    split_index=split_index,
                    split_count=split_count,
                    tensor_count=t_count,
                    metadata_kv_count=kv_count,
                    metadata_offset=meta_off,
                    metadata_size=meta_size,
                    tensor_toc_offset=toc_off,
                    tensor_toc_size=toc_size,
                    tensor_data_offset=data_off,
                    appendix_offset=app_off,
                    checksum=chk,
                )
                if magic == b"HKNT" and meta_size > 0 and meta_off > 0:
                    if mmap_buf is not None and (meta_off + meta_size) <= len(mmap_buf):
                        m_data = bytes(mmap_buf[meta_off : meta_off + meta_size])
                    else:
                        with open(file_path, "rb") as f:
                            f.seek(meta_off)
                            m_data = f.read(meta_size)
                    m_pos = 0
                    for _ in range(kv_count):
                        if m_pos + 2 > len(m_data):
                            break
                        klen = struct.unpack_from("<H", m_data, m_pos)[0]
                        m_pos += 2
                        if m_pos + klen > len(m_data):
                            break
                        k = m_data[m_pos : m_pos + klen].decode("utf-8", errors="ignore")
                        m_pos += klen
                        if m_pos + 1 > len(m_data):
                            break
                        tag = m_data[m_pos]
                        m_pos += 1
                        if m_pos + 4 > len(m_data):
                            break
                        vlen = struct.unpack_from("<I", m_data, m_pos)[0]
                        m_pos += 4
                        if m_pos + vlen > len(m_data):
                            break
                        raw_val = m_data[m_pos : m_pos + vlen]
                        m_pos += vlen
                        if tag == 0x01:
                            self.metadata[k] = raw_val.decode("utf-8", errors="ignore")
                        elif tag == 0x02:
                            self.metadata[k] = struct.unpack("<q", raw_val)[0]
                        elif tag == 0x03:
                            self.metadata[k] = struct.unpack("<d", raw_val)[0]
                        elif tag == 0x04:
                            self.metadata[k] = bool(raw_val[0])
                        elif tag == 0x05:
                            try:
                                self.metadata[k] = json.loads(raw_val.decode("utf-8"))
                            except Exception:
                                self.metadata[k] = raw_val.decode("utf-8", errors="ignore")
                        else:
                            self.metadata[k] = raw_val.decode("utf-8", errors="ignore")
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if hasattr(self, "ptr") and self.ptr:
            _LIB.hk_close(self.ptr)
            self.ptr = None

    def __del__(self):
        self.close()

    def dequantize(self, name: str, with_residual: bool = True) -> np.ndarray:
        if name not in self.tensors:
            raise KeyError(f"Tensor {name} not found")
        meta = self.tensors[name]
        idx = meta["index"]
        shape = meta["shape"]
        numel = int(np.prod(shape))
        out = np.empty(numel, dtype=np.float32)
        err = _LIB.hk_dequantize_f32(
            self.ptr,
            idx,
            1 if with_residual else 0,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            numel,
        )
        if err != 0:
            raise RuntimeError(f"Failed to dequantize tensor {name}")
        return out.reshape(shape)

    def get_raw_data(self, name: str) -> bytes:
        if name not in self.tensors:
            raise KeyError(f"Tensor {name} not found")
        meta = self.tensors[name]
        idx = meta["index"]
        size = ctypes.c_uint64(0)
        ptr = _LIB.hk_get_tensor_data(self.ptr, idx, ctypes.byref(size))
        if not ptr or size.value == 0:
            return b""
        buf = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * size.value))
        return bytes(buf.contents)

    def get_tensor_raw(self, name: str) -> np.ndarray:
        """Returns a zero-copy NumPy array mapped directly to raw tensor bytes without decoding overhead."""
        if name not in self.tensors:
            raise KeyError(f"Tensor {name} not found")
        meta = self.tensors[name]
        idx = meta["index"]
        shape = meta["shape"]
        stype = meta["storage_type"]
        size = ctypes.c_uint64(0)

        if _LIB is not None and hasattr(_LIB, "hk_get_tensor_raw_ptr"):
            ptr = _LIB.hk_get_tensor_raw_ptr(self.ptr, idx, ctypes.byref(size))
        else:
            ptr = _LIB.hk_get_tensor_data(self.ptr, idx, ctypes.byref(size))

        if not ptr or size.value == 0:
            return np.empty(shape, dtype=np.float32)

        from .format import STORAGE_TO_NUMPY_DTYPE
        np_dtype = STORAGE_TO_NUMPY_DTYPE.get(stype, np.float32)
        if np_dtype == np.float16:
            ptr_typed = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint16))
            numel = int(np.prod(shape)) if shape else (size.value // 2)
            arr = np.ctypeslib.as_array(ptr_typed, shape=(numel,)).view(np.float16)
        else:
            ptr_typed = ctypes.cast(ptr, ctypes.POINTER(np.ctypeslib.as_ctypes_type(np_dtype)))
            numel = int(np.prod(shape)) if shape else (size.value // np_dtype.itemsize)
            arr = np.ctypeslib.as_array(ptr_typed, shape=(numel,))
        return arr.reshape(shape) if shape else arr

    def get_metadata_string(self, key: str) -> Optional[str]:
        val = _LIB.hk_get_metadata_string(self.ptr, key.encode("utf-8"))
        return val.decode("utf-8") if val else None

    def get_metadata_int(self, key: str) -> Optional[int]:
        out = ctypes.c_int64(0)
        err = _LIB.hk_get_metadata_int(self.ptr, key.encode("utf-8"), ctypes.byref(out))
        return out.value if err == 0 else None


def native_metadata_patch_in_place(file_path: str, key: str, val: str) -> bool:
    """Invokes native Zig in-place metadata patching."""
    if not is_native_available() or _LIB is None or not hasattr(_LIB, "hk_metadata_patch_in_place"):
        return False
    ret = _LIB.hk_metadata_patch_in_place(file_path.encode("utf-8"), key.encode("utf-8"), val.encode("utf-8"))
    return ret == 0


def native_convert_gguf(input_path: str, output_path: str) -> int:
    """Ingests a GGUF file and writes an HK model container using native Zig zero-copy transplant."""
    if not is_native_available() or _LIB is None or not hasattr(_LIB, "hk_convert_gguf"):
        return -1
    return _LIB.hk_convert_gguf(input_path.encode("utf-8"), output_path.encode("utf-8"))


def native_export_gguf(input_path: str, output_path: str) -> int:
    """Exports an HK container to standard GGUF v3 format using native Zig."""
    if not is_native_available() or _LIB is None or not hasattr(_LIB, "hk_export_gguf"):
        return -1
    return _LIB.hk_export_gguf(input_path.encode("utf-8"), output_path.encode("utf-8"))


def native_rope_permute_hf_to_gguf(weight: np.ndarray, n_heads: int, head_dim: int) -> np.ndarray:
    """Permutes weights from Hugging Face half-split RoPE layout to GGUF interleaved layout."""
    arr = np.ascontiguousarray(weight, dtype=np.float32)
    out = np.empty_like(arr)
    if is_native_available() and _LIB is not None and hasattr(_LIB, "hk_rope_permute_hf_to_gguf"):
        _LIB.hk_rope_permute_hf_to_gguf(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            arr.size,
            n_heads,
            head_dim,
        )
        return out
    # Python fallback
    half = head_dim // 2
    reshaped_in = arr.reshape(-1, n_heads, head_dim)
    reshaped_out = out.reshape(-1, n_heads, head_dim)
    for b in range(reshaped_in.shape[0]):
        for h in range(n_heads):
            for i in range(half):
                reshaped_out[b, h, 2 * i] = reshaped_in[b, h, i]
                reshaped_out[b, h, 2 * i + 1] = reshaped_in[b, h, half + i]
    return out


def native_rope_unpermute_gguf_to_hf(weight: np.ndarray, n_heads: int, head_dim: int) -> np.ndarray:
    """Unpermutes weights from GGUF interleaved layout to Hugging Face half-split RoPE layout."""
    arr = np.ascontiguousarray(weight, dtype=np.float32)
    out = np.empty_like(arr)
    if is_native_available() and _LIB is not None and hasattr(_LIB, "hk_rope_unpermute_gguf_to_hf"):
        _LIB.hk_rope_unpermute_gguf_to_hf(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            arr.size,
            n_heads,
            head_dim,
        )
        return out
    # Python fallback
    half = head_dim // 2
    reshaped_in = arr.reshape(-1, n_heads, head_dim)
    reshaped_out = out.reshape(-1, n_heads, head_dim)
    for b in range(reshaped_in.shape[0]):
        for h in range(n_heads):
            for i in range(half):
                reshaped_out[b, h, i] = reshaped_in[b, h, 2 * i]
                reshaped_out[b, h, half + i] = reshaped_in[b, h, 2 * i + 1]
    return out


def native_layernorm_offset(data: np.ndarray, offset: float) -> np.ndarray:
    """Applies additive offset (+1.0 or -1.0) to LayerNorm/RMSNorm weights."""
    arr = np.array(data, dtype=np.float32, copy=True)
    if is_native_available() and _LIB is not None and hasattr(_LIB, "hk_layernorm_offset_f32"):
        _LIB.hk_layernorm_offset_f32(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            arr.size,
            ctypes.c_float(offset),
        )
        return arr
    return arr + offset


def native_gemv_q8_0(W_bytes: bytes, x: np.ndarray, bias: Optional[np.ndarray], M: int, K: int) -> np.ndarray:
    """Fast packed-weight GEMV for Q8_0 quantized weights without full FP32 decompression."""
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty(M, dtype=np.float32)
    bias_c = np.ascontiguousarray(bias, dtype=np.float32) if bias is not None else None
    bias_ptr = bias_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if bias_c is not None else None
    if is_native_available() and _LIB is not None and hasattr(_LIB, "hk_gemv_q8_0"):
        w_raw = bytes(W_bytes)
        _LIB.hk_gemv_q8_0(
            w_raw,
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            bias_ptr,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint64(M),
            ctypes.c_uint64(K),
        )
        return out
    raise RuntimeError("Native Zig GEMV kernel required for Q8_0")


def native_gemv_q4_0(W_bytes: bytes, x: np.ndarray, bias: Optional[np.ndarray], M: int, K: int) -> np.ndarray:
    """Fast packed-weight GEMV for Q4_0 quantized weights without full FP32 decompression."""
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty(M, dtype=np.float32)
    bias_c = np.ascontiguousarray(bias, dtype=np.float32) if bias is not None else None
    bias_ptr = bias_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if bias_c is not None else None
    if is_native_available() and _LIB is not None and hasattr(_LIB, "hk_gemv_q4_0"):
        w_raw = bytes(W_bytes)
        _LIB.hk_gemv_q4_0(
            w_raw,
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            bias_ptr,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint64(M),
            ctypes.c_uint64(K),
        )
        return out
    raise RuntimeError("Native Zig GEMV kernel required for Q4_0")


def native_gemv_q4_k(W_bytes: bytes, x: np.ndarray, bias: Optional[np.ndarray], M: int, K: int) -> np.ndarray:
    """Fast packed-weight GEMV for Q4_K quantized weights without full FP32 decompression."""
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty(M, dtype=np.float32)
    bias_c = np.ascontiguousarray(bias, dtype=np.float32) if bias is not None else None
    bias_ptr = bias_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if bias_c is not None else None
    if is_native_available() and _LIB is not None and hasattr(_LIB, "hk_gemv_q4_k"):
        w_raw = bytes(W_bytes)
        _LIB.hk_gemv_q4_k(
            w_raw,
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            bias_ptr,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint64(M),
            ctypes.c_uint64(K),
        )
        return out
    raise RuntimeError("Native Zig GEMV kernel required for Q4_K")


class NativeHKTokenizer:
    """High-performance native BPE and SentencePiece Tokenizer powered by compiled Zig engine."""

    def __init__(self, file_path: Union[str, Path]):
        if not is_native_available() or not hasattr(_LIB, "hk_tokenizer_load_from_file"):
            raise RuntimeError("Native HK shared library (libhk) required for NativeHKTokenizer")
        p = str(file_path).encode("utf-8")
        self._ptr = _LIB.hk_tokenizer_load_from_file(p)
        if not self._ptr:
            raise RuntimeError(f"Failed to load native tokenizer from {file_path}")

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr and _LIB is not None:
            _LIB.hk_tokenizer_free(self._ptr)
            self._ptr = None

    @property
    def vocab_size(self) -> int:
        if not self._ptr:
            return 0
        return int(_LIB.hk_tokenizer_get_vocab_size(self._ptr))

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        if not self._ptr:
            return []
        text_bytes = text.encode("utf-8")
        max_ids = max(len(text_bytes) * 2 + 16, 256)
        out_arr = (ctypes.c_uint32 * max_ids)()
        count = _LIB.hk_tokenizer_encode(
            self._ptr,
            text_bytes,
            1 if add_bos else 0,
            1 if add_eos else 0,
            out_arr,
            max_ids,
        )
        return [int(out_arr[i]) for i in range(count)]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if not self._ptr or not token_ids:
            return ""
        arr = (ctypes.c_uint32 * len(token_ids))(*token_ids)
        max_len = len(token_ids) * 64 + 256
        buf = ctypes.create_string_buffer(max_len)
        written = _LIB.hk_tokenizer_decode(
            self._ptr,
            arr,
            len(token_ids),
            1 if skip_special_tokens else 0,
            buf,
            max_len,
        )
        return buf.raw[:written].decode("utf-8", errors="replace")


_ENGINE_ERRORS = {
    -1: "invalid argument",
    -2: "the context window is full",
    -3: "too many tokens in one call",
    -4: "the forward pass failed",
}


class NativeHKEngine:
    """Native transformer engine (Llama, Qwen2 and Qwen3 style models) over the compiled Zig library.

    Weights are memory mapped from the file, so opening a model costs very little memory.
    Positions are explicit: pass the position of the first token you feed, and call
    `reset_cache()` before starting a new conversation.
    """

    def __init__(self, file_path: Union[str, Path]):
        if not is_native_available() or not hasattr(_LIB, "hk_engine_load_from_file"):
            raise RuntimeError("Native HK shared library (libhk) required for NativeHKEngine: " + native_unavailable_reason())
        self._ptr = _LIB.hk_engine_load_from_file(str(file_path).encode("utf-8"))
        if not self._ptr:
            reason = ""
            if hasattr(_LIB, "hk_engine_last_error"):
                buf = ctypes.create_string_buffer(512)
                n = _LIB.hk_engine_last_error(buf, 512)
                reason = buf.raw[:n].decode("utf-8", "replace")
            raise RuntimeError(f"Failed to load native engine from {file_path}" + (f": {reason}" if reason else ""))
        self._vocab_size = int(_LIB.hk_engine_get_vocab_size(self._ptr))

    def close(self):
        if getattr(self, "_ptr", None) and _LIB is not None:
            _LIB.hk_engine_free(self._ptr)
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def context_size(self) -> int:
        if not self._ptr:
            raise RuntimeError("Engine not loaded")
        return int(_LIB.hk_engine_get_context_size(self._ptr))

    def reset_cache(self):
        if self._ptr:
            _LIB.hk_engine_reset_cache(self._ptr)

    def forward(self, tokens: Sequence[int], pos: int = 0) -> np.ndarray:
        """Feeds `tokens` starting at position `pos` and returns the logits after the last one."""
        if not self._ptr:
            raise RuntimeError("Engine not loaded")
        arr = np.ascontiguousarray(tokens, dtype=np.uint32)
        if arr.size == 0:
            raise ValueError("forward needs at least one token")
        if int(arr.max()) >= self._vocab_size:
            raise ValueError(f"token id {int(arr.max())} is outside the vocabulary of {self._vocab_size}")
        logits = np.empty(self._vocab_size, dtype=np.float32)
        ret = _LIB.hk_engine_forward_tokens(
            self._ptr,
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(arr.size),
            ctypes.c_uint32(pos),
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if ret != 0:
            raise RuntimeError(f"Engine forward failed: {_ENGINE_ERRORS.get(ret, ret)}")
        return logits

    def forward_step(self, token: int, pos: int) -> np.ndarray:
        return self.forward([token], pos)


def convert_safetensors_to_hk(input_path: Union[str, Path], output_path: Union[str, Path], storage_type: int = 0) -> None:
    """Natively converts a .safetensors model directly into an .hk container via Zig."""
    if not is_native_available() or not hasattr(_LIB, "hk_convert_safetensors"):
        raise RuntimeError("Native HK shared library required for convert_safetensors_to_hk")
    in_b = str(input_path).encode("utf-8")
    out_b = str(output_path).encode("utf-8")
    ret = _LIB.hk_convert_safetensors(in_b, out_b, ctypes.c_uint8(storage_type))
    if ret != 0:
        raise RuntimeError(f"SafeTensors transcoding failed with exit code {ret}")


def native_hf_detect_architecture(json_config_str: str) -> Optional[str]:
    """Detects canonical architecture from HF JSON config using native Zig engine."""
    if not is_native_available() or not hasattr(_LIB, "hk_hf_detect_architecture"):
        return None
    out_buf = ctypes.create_string_buffer(128)
    ret = _LIB.hk_hf_detect_architecture(json_config_str.encode("utf-8"), out_buf, 128)
    if ret == 0:
        return out_buf.value.decode("utf-8")
    return None


def native_hf_map_tensor_name(name: str, arch: str = "llama", to_hk: bool = True) -> Optional[str]:
    """Maps tensor name between HF and HK conventions using native Zig engine."""
    if not is_native_available() or not hasattr(_LIB, "hk_hf_map_tensor_name"):
        return None
    out_buf = ctypes.create_string_buffer(512)
    ret = _LIB.hk_hf_map_tensor_name(name.encode("utf-8"), arch.encode("utf-8"), to_hk, out_buf, 512)
    if ret == 0:
        return out_buf.value.decode("utf-8")
    return None


def native_context_truncate(
    tokens: Union[List[int], np.ndarray],
    max_tokens: int,
    strategy: int,
    head_ratio: float,
    out: Optional[np.ndarray] = None,
) -> Optional[Union[List[int], np.ndarray]]:
    """Truncates token sequence using native Zig ContextWindowManager with zero-copy buffer passing."""
    if not is_native_available() or not hasattr(_LIB, "hk_context_truncate"):
        return None
    is_py_list = isinstance(tokens, list)
    if is_py_list:
        arr = np.array(tokens, dtype=np.uint32)
    elif isinstance(tokens, np.ndarray):
        arr = np.ascontiguousarray(tokens, dtype=np.uint32)
    else:
        arr = np.ascontiguousarray(tokens.detach().cpu().numpy(), dtype=np.uint32)

    n = arr.size
    if out is not None and isinstance(out, np.ndarray) and out.size >= max_tokens:
        out_arr = out[:max_tokens]
    else:
        out_arr = np.empty(max_tokens, dtype=np.uint32)
    out_len = ctypes.c_size_t(0)
    ret = _LIB.hk_context_truncate(
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_size_t(n),
        ctypes.c_size_t(max_tokens),
        ctypes.c_int(strategy),
        ctypes.c_float(head_ratio),
        out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(out_len),
    )
    if ret == 0:
        actual = out_arr[:out_len.value]
        return actual.tolist() if is_py_list else actual
    return None


_GOVERNOR_REASON_BUF = ctypes.create_string_buffer(256)

def native_governor_can_grow(
    current_params: int,
    added_params: int,
    max_growth_ratio: float = 2.0,
    max_vram_mb: int = 4096,
    dtype_bytes: int = 4,
) -> Optional[Tuple[bool, str]]:
    """Evaluates growth request constraints using native Zig GrowthGovernor."""
    if not is_native_available() or not hasattr(_LIB, "hk_governor_can_grow"):
        return None
    approved = _LIB.hk_governor_can_grow(
        ctypes.c_uint64(current_params),
        ctypes.c_uint64(added_params),
        ctypes.c_float(max_growth_ratio),
        ctypes.c_uint64(max_vram_mb),
        ctypes.c_uint32(dtype_bytes),
        _GOVERNOR_REASON_BUF,
        256,
    )
    if approved == 1:
        return True, "Approved"
    return False, _GOVERNOR_REASON_BUF.value.decode("utf-8", errors="replace")


def native_governor_can_grow_batch(
    current_params: np.ndarray,
    added_params: np.ndarray,
    max_growth_ratio: float = 2.0,
    max_vram_mb: int = 4096,
    dtype_bytes: int = 4,
) -> Optional[np.ndarray]:
    """Evaluates growth requests in bulk with a single C-ABI boundary crossing."""
    if not is_native_available() or not hasattr(_LIB, "hk_governor_can_grow_batch"):
        return None
    n = len(current_params)
    cur_u64 = np.ascontiguousarray(current_params, dtype=np.uint64)
    add_u64 = np.ascontiguousarray(added_params, dtype=np.uint64)
    c_cur = cur_u64.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    c_add = add_u64.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    out_res = np.empty(n, dtype=np.uint8)
    c_out = out_res.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    ret = _LIB.hk_governor_can_grow_batch(
        c_cur,
        c_add,
        ctypes.c_size_t(n),
        ctypes.c_float(max_growth_ratio),
        ctypes.c_uint64(max_vram_mb),
        ctypes.c_uint32(dtype_bytes),
        c_out,
    )
    if ret == 0:
        return out_res.astype(bool)
    return None


def native_expand_vocab_embeddings(
    old_embed: np.ndarray,
    new_vocab: int,
    init_std: float = 0.02,
    seed: int = 42,
) -> Optional[np.ndarray]:
    """Expands embedding tensor capacity natively with exact function preservation."""
    if not is_native_available() or not hasattr(_LIB, "hk_expand_vocab_embeddings"):
        return None
    old_vocab, hidden_size = old_embed.shape
    if new_vocab <= old_vocab:
        return old_embed
    new_embed = np.empty((new_vocab, hidden_size), dtype=np.float32)
    c_old = old_embed.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_new = new_embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ret = _LIB.hk_expand_vocab_embeddings(
        c_old,
        ctypes.c_size_t(old_vocab),
        ctypes.c_size_t(hidden_size),
        ctypes.c_size_t(new_vocab),
        c_new,
        ctypes.c_float(init_std),
        ctypes.c_uint64(seed),
    )
    if ret == 0:
        return new_embed
    return None


def native_detect_hardware() -> Dict[str, Any]:
    """Detects host CPU vendor, vector extensions, and optimal zero-copy page alignment."""
    if not is_native_available() or not hasattr(_LIB, "hk_detect_hardware"):
        # Fallback Python detection
        import platform as _plat
        mach = _plat.machine().lower()
        sys_name = _plat.system().lower()
        is_apple = ("darwin" in sys_name and ("arm" in mach or "aarch64" in mach))
        return {
            "vendor": "apple" if is_apple else ("arm" if ("arm" in mach or "aarch64" in mach) else "x86_64"),
            "has_avx2": "x86" in mach or "amd64" in mach,
            "has_avx512f": False,
            "has_avx512vnni": False,
            "has_avx_vnni": False,
            "has_amx": False,
            "has_arm_neon": "arm" in mach or "aarch64" in mach,
            "has_arm_sve": False,
            "is_apple_silicon": is_apple,
            "has_rocm_ready": True,
            "has_npu_ready": True,
            "optimal_page_alignment": 16384 if is_apple else 4096,
            "dma_hugepage_alignment": 65536,
        }

    caps = C_HardwareCapabilities()
    _LIB.hk_detect_hardware(ctypes.byref(caps))
    vendors = {0: "intel", 1: "amd", 2: "arm", 3: "apple", 4: "generic"}
    return {
        "vendor": vendors.get(caps.vendor, "unknown"),
        "has_avx2": bool(caps.has_avx2),
        "has_avx512f": bool(caps.has_avx512f),
        "has_avx512vnni": bool(caps.has_avx512vnni),
        "has_avx_vnni": bool(caps.has_avx_vnni),
        "has_amx": bool(caps.has_amx),
        "has_arm_neon": bool(caps.has_arm_neon),
        "has_arm_sve": bool(caps.has_arm_sve),
        "is_apple_silicon": bool(caps.is_apple_silicon),
        "has_rocm_ready": bool(caps.has_rocm_ready),
        "has_npu_ready": bool(caps.has_npu_ready),
        "optimal_page_alignment": int(caps.optimal_page_alignment),
        "dma_hugepage_alignment": int(caps.dma_hugepage_alignment),
    }


def native_get_optimal_alignment() -> int:
    """Returns optimal page alignment for zero-copy memory mapping on current hardware."""
    if is_native_available() and hasattr(_LIB, "hk_get_optimal_alignment"):
        return int(_LIB.hk_get_optimal_alignment())
    import platform as _plat
    if "darwin" in _plat.system().lower() and ("arm" in _plat.machine().lower() or "aarch64" in _plat.machine().lower()):
        return 16384
    return 4096


def native_gemv_bf16(
    w: np.ndarray,
    x: np.ndarray,
    bias: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fast Matrix-Vector Multiplication with raw BF16 weights: y = W_bf16 * x + bias."""
    M, K = w.shape
    y = np.empty(M, dtype=np.float32)
    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    b_ptr = None
    if bias is not None:
        b_f32 = np.ascontiguousarray(bias, dtype=np.float32)
        b_ptr = b_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if is_native_available() and hasattr(_LIB, "hk_gemv_bf16"):
        w_u16 = np.ascontiguousarray(w.view(np.uint16) if w.dtype != np.uint16 else w)
        _LIB.hk_gemv_bf16(
            w_u16.ctypes.data_as(ctypes.c_void_p),
            x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b_ptr,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(M),
            ctypes.c_size_t(K),
        )
    else:
        # High-performance numpy fallback
        if w.dtype == np.uint16:
            w_f32 = (w.astype(np.uint32) << 16).view(np.float32)
        else:
            w_f32 = w.astype(np.float32)
        y = np.dot(w_f32, x_f32)
        if bias is not None:
            y += bias
    return y


def native_gemv_f32(
    w: np.ndarray,
    x: np.ndarray,
    bias: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fast Matrix-Vector Multiplication with raw FP32 weights: y = W_f32 * x + bias."""
    M, K = w.shape
    y = np.empty(M, dtype=np.float32)
    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    b_ptr = None
    if bias is not None:
        b_f32 = np.ascontiguousarray(bias, dtype=np.float32)
        b_ptr = b_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if is_native_available() and hasattr(_LIB, "hk_gemv_f32"):
        w_f32 = np.ascontiguousarray(w, dtype=np.float32)
        _LIB.hk_gemv_f32(
            w_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b_ptr,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint64(M),
            ctypes.c_uint64(K),
        )
    else:
        w_f32 = np.ascontiguousarray(w, dtype=np.float32)
        y = np.dot(w_f32, x_f32)
        if bias is not None:
            y += bias
    return y


def native_gemv_f16(
    w: np.ndarray,
    x: np.ndarray,
    bias: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fast Matrix-Vector Multiplication with raw FP16 weights: y = W_f16 * x + bias."""
    M, K = w.shape
    y = np.empty(M, dtype=np.float32)
    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    b_ptr = None
    if bias is not None:
        b_f32 = np.ascontiguousarray(bias, dtype=np.float32)
        b_ptr = b_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if is_native_available() and hasattr(_LIB, "hk_gemv_f16"):
        w_f16 = np.ascontiguousarray(w, dtype=np.float16)
        _LIB.hk_gemv_f16(
            w_f16.ctypes.data_as(ctypes.c_void_p),
            x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b_ptr,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(M),
            ctypes.c_size_t(K),
        )
    else:
        w_f32 = w.astype(np.float32)
        y = np.dot(w_f32, x_f32)
        if bias is not None:
            y += bias
    return y


def native_gemv_int8(
    w: np.ndarray,
    x: np.ndarray,
    scale: float = 1.0,
    bias: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fast Matrix-Vector Multiplication with raw INT8 weights: y = (W_i8 * x) * scale + bias."""
    M, K = w.shape
    y = np.empty(M, dtype=np.float32)
    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    b_ptr = None
    if bias is not None:
        b_f32 = np.ascontiguousarray(bias, dtype=np.float32)
        b_ptr = b_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if is_native_available() and hasattr(_LIB, "hk_gemv_int8"):
        w_i8 = np.ascontiguousarray(w, dtype=np.int8)
        _LIB.hk_gemv_int8(
            w_i8.ctypes.data_as(ctypes.c_void_p),
            x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(scale),
            b_ptr,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(M),
            ctypes.c_size_t(K),
        )
    else:
        y = np.dot(w.astype(np.float32), x_f32) * scale
        if bias is not None:
            y += bias
    return y







