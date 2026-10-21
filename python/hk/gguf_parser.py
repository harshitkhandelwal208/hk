"""
HK Framework - Pure Python Zero-Dependency GGUF v1/v2/v3 Binary Parser & Ingestion Engine.
Enables transparent inspection, tensor mapping, and conversion without requiring external dependencies.
"""


from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, BinaryIO
import os
import struct
from ._lazy import lazy

np = lazy("numpy")

# GGUF Constants
GGUF_MAGIC = 0x46554747  # 'GGUF'

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30
GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35

# Block size and byte size mapping
GGML_TYPE_INFO = {
    GGML_TYPE_F32: (1, 4),
    GGML_TYPE_F16: (1, 2),
    GGML_TYPE_BF16: (1, 2),
    GGML_TYPE_Q4_0: (32, 18),
    GGML_TYPE_Q4_1: (32, 20),
    GGML_TYPE_Q5_0: (32, 22),
    GGML_TYPE_Q5_1: (32, 24),
    GGML_TYPE_Q8_0: (32, 34),
    GGML_TYPE_Q8_1: (32, 36),
    GGML_TYPE_Q2_K: (256, 84),
    GGML_TYPE_Q3_K: (256, 110),
    GGML_TYPE_Q4_K: (256, 144),
    GGML_TYPE_Q5_K: (256, 176),
    GGML_TYPE_Q6_K: (256, 210),
    GGML_TYPE_Q8_K: (256, 292),
    GGML_TYPE_IQ1_S: (256, 52),
    GGML_TYPE_IQ1_M: (256, 56),
    GGML_TYPE_IQ2_XXS: (256, 66),
    GGML_TYPE_IQ2_XS: (256, 74),
    GGML_TYPE_IQ3_XXS: (256, 98),
    GGML_TYPE_IQ4_NL: (32, 18),
    GGML_TYPE_IQ4_XS: (256, 136),
    GGML_TYPE_I8: (1, 1),
    GGML_TYPE_I16: (1, 2),
    GGML_TYPE_I32: (1, 4),
    GGML_TYPE_I64: (1, 8),
    GGML_TYPE_F64: (1, 8),
    GGML_TYPE_TQ1_0: (256, 48),
    GGML_TYPE_TQ2_0: (256, 80),
}


class GGUFTensorDescriptor:
    def __init__(self, name: str, shape: List[int], ggml_type: int, offset: int, size: int):
        self.name = name
        self.shape = shape  # C-order (dim0, dim1, ...)
        self.ggml_shape = list(reversed(shape))  # GGUF/GGML order (ne0, ne1, ...)
        self.ggml_type = ggml_type
        self.offset = offset
        self.size = size

    def __repr__(self) -> str:
        return f"<GGUFTensor {self.name} shape={self.shape} type={self.ggml_type} size={self.size}>"


class GGUFReaderLight:
    """
    Pure Python zero-dependency GGUF reader and validator.
    Parses metadata and tensor TOC from GGUF v1, v2, and v3 binary streams.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.version = 0
        self.alignment = 32
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, GGUFTensorDescriptor] = {}
        self.tensor_data_offset = 0
        self._parse()

    def _read_str(self, f: BinaryIO, is_v1: bool) -> str:
        if is_v1:
            (length,) = struct.unpack("<I", f.read(4))
        else:
            (length,) = struct.unpack("<Q", f.read(8))
        raw = f.read(length)
        return raw.decode("utf-8", errors="replace")

    def _read_val(self, f: BinaryIO, val_type: int, is_v1: bool) -> Any:
        if val_type == GGUF_TYPE_UINT8:
            return struct.unpack("<B", f.read(1))[0]
        elif val_type == GGUF_TYPE_INT8:
            return struct.unpack("<b", f.read(1))[0]
        elif val_type == GGUF_TYPE_UINT16:
            return struct.unpack("<H", f.read(2))[0]
        elif val_type == GGUF_TYPE_INT16:
            return struct.unpack("<h", f.read(2))[0]
        elif val_type == GGUF_TYPE_UINT32:
            return struct.unpack("<I", f.read(4))[0]
        elif val_type == GGUF_TYPE_INT32:
            return struct.unpack("<i", f.read(4))[0]
        elif val_type == GGUF_TYPE_FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif val_type == GGUF_TYPE_BOOL:
            return struct.unpack("<?", f.read(1))[0]
        elif val_type == GGUF_TYPE_STRING:
            return self._read_str(f, is_v1)
        elif val_type == GGUF_TYPE_UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        elif val_type == GGUF_TYPE_INT64:
            return struct.unpack("<q", f.read(8))[0]
        elif val_type == GGUF_TYPE_FLOAT64:
            return struct.unpack("<d", f.read(8))[0]
        elif val_type == GGUF_TYPE_ARRAY:
            (item_type,) = struct.unpack("<I", f.read(4))
            if is_v1:
                (arr_len,) = struct.unpack("<I", f.read(4))
            else:
                (arr_len,) = struct.unpack("<Q", f.read(8))
            items = []
            for _ in range(arr_len):
                items.append(self._read_val(f, item_type, is_v1))
            return items
        else:
            raise ValueError(f"Unknown GGUF value type {val_type}")

    def _parse(self):
        with open(self.file_path, "rb") as f:
            magic, version = struct.unpack("<II", f.read(8))
            if magic != GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF magic 0x{magic:08X} in {self.file_path}")
            if version not in (1, 2, 3):
                raise ValueError(f"Unsupported GGUF version {version}")

            self.version = version
            is_v1 = (version == 1)

            if is_v1:
                tensor_count, kv_count = struct.unpack("<II", f.read(8))
            else:
                tensor_count, kv_count = struct.unpack("<QQ", f.read(16))

            # 1. Parse Metadata Key-Values
            for _ in range(kv_count):
                key = self._read_str(f, is_v1)
                (val_type,) = struct.unpack("<I", f.read(4))
                val = self._read_val(f, val_type, is_v1)
                self.metadata[key] = val
                if key == "general.alignment":
                    self.alignment = int(val)

            # 2. Parse Tensor TOC
            for _ in range(tensor_count):
                t_name = self._read_str(f, is_v1)
                (n_dims,) = struct.unpack("<I", f.read(4))
                dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
                (ggml_type, offset) = struct.unpack("<IQ", f.read(12))

                # Compute byte size
                block_size, type_size = GGML_TYPE_INFO.get(ggml_type, (1, 4))
                num_elements = 1
                for d in dims:
                    num_elements *= d
                n_blocks = (num_elements + block_size - 1) // block_size
                tensor_size = n_blocks * type_size

                # Store shape in C row-major order (reversed from GGUF)
                c_shape = list(reversed(dims))
                desc = GGUFTensorDescriptor(
                    name=t_name,
                    shape=c_shape,
                    ggml_type=ggml_type,
                    offset=offset,
                    size=tensor_size,
                )
                self.tensors[t_name] = desc

            # Aligned start of tensor payloads
            current_pos = f.tell()
            rem = current_pos % self.alignment
            self.tensor_data_offset = current_pos if rem == 0 else current_pos + (self.alignment - rem)

    def read_tensor_bytes(self, tensor_name: str) -> bytes:
        """Reads raw binary bytes for a tensor directly from the GGUF file."""
        desc = self.tensors.get(tensor_name)
        if desc is None:
            raise KeyError(f"Tensor '{tensor_name}' not found in GGUF file")

        with open(self.file_path, "rb") as f:
            f.seek(self.tensor_data_offset + desc.offset)
            return f.read(desc.size)

    def convert_to_hk(self, output_hk_path: str):
        """Transcodes/transplants this GGUF file into an HK model container."""
        from .native import is_native_available, native_convert_gguf
        if not is_native_available():
            raise RuntimeError("Native HK engine (hk.dll / libhk.so) is required for high-performance GGUF transcoding.")

        ret = native_convert_gguf(self.file_path, output_hk_path)
        if ret != 0:
            raise RuntimeError(f"Native GGUF conversion failed with error code: {ret}")


def convert_gguf_to_hk(input_gguf: str, output_hk: str):
    """Converts a GGUF file to HK container format with zero-copy bitstream transplant in native Zig."""
    from .native import is_native_available, native_convert_gguf
    if is_native_available():
        ret = native_convert_gguf(input_gguf, output_hk)
        if ret == 0:
            return
        raise RuntimeError(f"Native GGUF conversion failed with error code: {ret}")
    reader = GGUFReaderLight(input_gguf)
    reader.convert_to_hk(output_hk)


def export_hk_to_gguf(input_hk: str, output_gguf: str):
    """Exports an HK container model to standard GGUF v3 format."""
    from .native import is_native_available, native_export_gguf
    if is_native_available():
        ret = native_export_gguf(input_hk, output_gguf)
        if ret == 0:
            return
    raise NotImplementedError("Native Zig library required for export_hk_to_gguf")

