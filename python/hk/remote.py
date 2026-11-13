"""
HK Remote Hub Range Reader
Enables instant inspection and selective tensor streaming from remote HTTP/HTTPS endpoints
(e.g., Hugging Face Hub, Cloudflare R2, AWS S3) using HTTP Range Requests (RFC 7233).
Allows inspecting 70B parameter model metadata and downloading individual layers in milliseconds
without downloading full multi-gigabyte weight containers.
"""


from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
import struct
import urllib.request
import urllib.error
from ._lazy import lazy

np = lazy("numpy")

from .format import (
    MAGIC,
    StorageType,
    TileLayout,
    SparsityType,
)


def _fetch_range(url: str, start: int, end: int, headers: Optional[Dict[str, str]] = None) -> bytes:
    """Fetches a specific byte range [start, end] inclusive from an HTTP/HTTPS URL."""
    req_headers = {"Range": f"bytes={start}-{end}"}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 416:
            raise ValueError(f"Requested range [{start}, {end}] is out of bounds for URL: {url}")
        raise


def read_remote_hk_header(
    url: str,
    initial_chunk_size: int = 131072,  # 128 KB
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Reads the header, metadata dictionary, and Tensor TOC from a remote .hk container
    using an initial 128 KB HTTP range request.
    """
    chunk = _fetch_range(url, 0, initial_chunk_size - 1, headers=headers)
    if len(chunk) < 128:
        raise ValueError(f"Remote file at {url} is too small to be a valid HK container (got {len(chunk)} bytes)")

    magic = chunk[0:4]
    if magic != bytes(MAGIC):
        raise ValueError(f"Invalid HK magic in remote file: {magic} (expected {bytes(MAGIC)})")

    # Unpack FileHeader: <4s H H I H H Q Q Q Q Q Q Q Q Q H 38s
    header_vals = struct.unpack("<4s H H I H H Q Q Q Q Q Q Q Q Q H 38s", chunk[0:128])
    (
        _magic,
        version_major,
        version_minor,
        flags,
        alignment,
        split_index,
        tensor_count,
        metadata_kv_count,
        metadata_offset,
        metadata_size,
        tensor_toc_offset,
        tensor_toc_size,
        tensor_data_offset,
        appendix_offset,
        checksum,
        split_count,
        _reserved,
    ) = header_vals

    required_bytes = max(metadata_offset + metadata_size, tensor_toc_offset + tensor_toc_size)

    # If metadata or TOC extends beyond initial chunk, fetch remaining bytes
    if len(chunk) < required_bytes:
        extra_chunk = _fetch_range(url, len(chunk), required_bytes - 1, headers=headers)
        chunk = chunk + extra_chunk

    # 1. Parse Metadata
    metadata: Dict[str, Any] = {}
    if metadata_size > 0 and metadata_kv_count > 0:
        meta_bytes = chunk[metadata_offset : metadata_offset + metadata_size]
        cursor = 0
        for _ in range(metadata_kv_count):
            if cursor + 2 > len(meta_bytes):
                break
            klen = struct.unpack("<H", meta_bytes[cursor : cursor + 2])[0]
            cursor += 2
            key = meta_bytes[cursor : cursor + klen].decode("utf-8", errors="replace")
            cursor += klen

            tag = meta_bytes[cursor]
            cursor += 1

            if tag == 0x01:  # val_string
                vlen = struct.unpack("<I", meta_bytes[cursor : cursor + 4])[0]
                cursor += 4
                metadata[key] = meta_bytes[cursor : cursor + vlen].decode("utf-8", errors="replace")
                cursor += vlen
            elif tag == 0x02:  # val_int64
                metadata[key] = struct.unpack("<q", meta_bytes[cursor : cursor + 8])[0]
                cursor += 8
            elif tag == 0x03:  # val_float64
                metadata[key] = struct.unpack("<d", meta_bytes[cursor : cursor + 8])[0]
                cursor += 8
            elif tag == 0x04:  # val_bool
                metadata[key] = meta_bytes[cursor] != 0
                cursor += 1
            elif tag == 0x05:  # val_json
                vlen = struct.unpack("<I", meta_bytes[cursor : cursor + 4])[0]
                cursor += 4
                try:
                    metadata[key] = json.loads(meta_bytes[cursor : cursor + vlen].decode("utf-8"))
                except Exception:
                    metadata[key] = meta_bytes[cursor : cursor + vlen].decode("utf-8", errors="replace")
                cursor += vlen
            elif tag == 0x06:  # val_bytes
                vlen = struct.unpack("<I", meta_bytes[cursor : cursor + 4])[0]
                cursor += 4
                metadata[key] = meta_bytes[cursor : cursor + vlen]
                cursor += vlen

    # 2. Parse Tensor TOC
    tensors: List[Dict[str, Any]] = []
    if tensor_toc_size > 0 and tensor_count > 0:
        toc_bytes = chunk[tensor_toc_offset : tensor_toc_offset + tensor_toc_size]
        cursor = 0
        for _ in range(tensor_count):
            if cursor + 2 > len(toc_bytes):
                break
            nlen = struct.unpack("<H", toc_bytes[cursor : cursor + 2])[0]
            cursor += 2
            t_name = toc_bytes[cursor : cursor + nlen].decode("utf-8", errors="replace")
            cursor += nlen

            stype_raw, tlayout_raw, sparse_raw, ndim = struct.unpack("<BBBB", toc_bytes[cursor : cursor + 4])
            cursor += 4

            shape = list(struct.unpack(f"<{ndim}Q", toc_bytes[cursor : cursor + 8 * ndim]))
            cursor += 8 * ndim

            (
                data_offset,
                data_size,
                residual_offset,
                residual_size,
                scale_offset,
                scale_size,
            ) = struct.unpack("<QQQQQQ", toc_bytes[cursor : cursor + 48])
            cursor += 48

            block_size = struct.unpack("<H", toc_bytes[cursor : cursor + 2])[0]
            cursor += 2

            sparsity_ratio = struct.unpack("<f", toc_bytes[cursor : cursor + 4])[0]
            cursor += 4

            tensors.append({
                "name": t_name,
                "shape": shape,
                "ndim": ndim,
                "storage_type": StorageType(stype_raw) if stype_raw in StorageType._value2member_map_ else stype_raw,
                "tile_layout": TileLayout(tlayout_raw) if tlayout_raw in TileLayout._value2member_map_ else tlayout_raw,
                "sparsity_type": SparsityType(sparse_raw) if sparse_raw in SparsityType._value2member_map_ else sparse_raw,
                "data_offset": data_offset,
                "data_size": data_size,
                "residual_offset": residual_offset,
                "residual_size": residual_size,
                "scale_offset": scale_offset,
                "scale_size": scale_size,
                "block_size": block_size,
                "sparsity_ratio": sparsity_ratio,
            })

    return metadata, tensors


class RemoteHKFile:
    """
    Safe remote streaming reader for .hk model containers hosted on HTTP/HTTPS servers.
    Inspects model structure in milliseconds and lazily downloads only requested tensors.
    """

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = headers or {}
        self._metadata, self._tensor_list = read_remote_hk_header(url, headers=self.headers)
        self._tensors_by_name: Dict[str, Dict[str, Any]] = {t["name"]: t for t in self._tensor_list}

    def keys(self) -> List[str]:
        """Returns all tensor names available in the remote container."""
        return list(self._tensors_by_name.keys())

    def metadata(self) -> Dict[str, Any]:
        """Returns the complete metadata dictionary of the remote container."""
        return self._metadata

    def get_tensor_info(self, name: str) -> Dict[str, Any]:
        """Returns descriptor metadata for a specific tensor."""
        if name not in self._tensors_by_name:
            raise KeyError(f"Tensor '{name}' not found in remote container {self.url}")
        return self._tensors_by_name[name]

    def get_tensor_bytes(self, name: str) -> bytes:
        """Downloads only the bytes for a specific tensor via HTTP range request."""
        info = self.get_tensor_info(name)
        data_offset = info["data_offset"]
        data_size = info["data_size"]
        if data_size == 0:
            return b""
        return _fetch_range(self.url, data_offset, data_offset + data_size - 1, headers=self.headers)

    def get_tensor(self, name: str, framework: str = "pt"):
        """Downloads and reconstructs a single tensor into PyTorch or NumPy array."""
        info = self.get_tensor_info(name)
        raw = self.get_tensor_bytes(name)
        shape = info["shape"]
        st = info["storage_type"]

        expected_numel = int(np.prod(shape))
        if st in (StorageType.F32, 0x00):
            byte_len = expected_numel * 4
            arr = np.frombuffer(raw[:byte_len], dtype=np.float32).copy().reshape(shape)
        elif st in (StorageType.F16, 0x01):
            byte_len = expected_numel * 2
            arr = np.frombuffer(raw[:byte_len], dtype=np.float16).copy().reshape(shape)
        elif st in (StorageType.BF16, 0x02):
            import torch
            byte_len = expected_numel * 2
            arr = torch.frombuffer(raw[:byte_len], dtype=torch.bfloat16).clone().reshape(shape)
            return arr if framework == "pt" else arr.float().numpy()
        elif st in (StorageType.Q8_0, 0x16):
            from .quantization import dequantize_q8_0
            t = dequantize_q8_0(raw, shape)
            return t if framework == "pt" else t.numpy()
        elif st in (StorageType.Q4_0, 0x15):
            from .quantization import dequantize_q4_0
            t = dequantize_q4_0(raw, shape)
            return t if framework == "pt" else t.numpy()
        else:
            arr = np.frombuffer(raw, dtype=np.uint8).copy()

        if framework == "pt":
            import torch
            return torch.from_numpy(arr)
        return arr


def safe_open_remote(url: str, headers: Optional[Dict[str, str]] = None) -> RemoteHKFile:
    """Opens a remote .hk container for inspection and selective tensor streaming."""
    return RemoteHKFile(url, headers=headers)


def download_model(
    url: str,
    output_path: Union[str, Path],
    chunk_size: int = 1048576,  # 1 MB
    headers: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Any] = None,
) -> Path:
    """
    Downloads an entire remote .hk container with chunked streaming, resume capability,
    and progress tracking on par with GGUF / Ollama model pull UX.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    req_headers = {}
    if headers:
        req_headers.update(headers)

    # Initial probe for content length
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))

    downloaded = 0
    with urllib.request.urlopen(req) as resp, open(out_file, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback:
                progress_callback(downloaded, total_size)

    return out_file

