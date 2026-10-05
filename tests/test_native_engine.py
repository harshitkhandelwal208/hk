"""
Tests for native Zig engine, tokenizer, and SafeTensors transcoder via ctypes C-ABI bridge.
"""

import json
import struct
import tempfile
from pathlib import Path
import numpy as np
import torch
import pytest

import hk
from hk.native import (
    is_native_available,
    NativeHKTokenizer,
    NativeHKEngine,
    convert_safetensors_to_hk,
)


def _create_minimal_safetensors(file_path: Path, tensors_dict: dict):
    header = {}
    data_bytes = bytearray()
    offset = 0

    for name, tensor in tensors_dict.items():
        arr = np.ascontiguousarray(tensor, dtype=np.float32)
        raw = arr.tobytes()
        length = len(raw)
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + length],
        }
        data_bytes.extend(raw)
        offset += length

    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)

    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        f.write(data_bytes)


@pytest.mark.skipif(not is_native_available(), reason="Native library not compiled")
def test_native_safetensors_transcoder():
    with tempfile.TemporaryDirectory() as tmpdir:
        st_file = Path(tmpdir) / "model.safetensors"
        hk_file = Path(tmpdir) / "model.hk"

        # Create dummy weights
        w1 = np.random.randn(4, 8).astype(np.float32)
        w2 = np.random.randn(8).astype(np.float32)
        _create_minimal_safetensors(st_file, {"layer1.weight": w1, "layer1.bias": w2})

        # Transcode using native Zig transcoder (storage_type 0 = dense uncompressed)
        convert_safetensors_to_hk(st_file, hk_file, storage_type=0)
        assert hk_file.exists()
        assert hk_file.stat().st_size > 0

        # Verify reading with hk.safe_open
        with hk.safe_open(str(hk_file), framework="numpy") as f:
            keys = f.keys()
            assert "layer1.weight" in keys
            assert "layer1.bias" in keys
            np.testing.assert_allclose(f.get_tensor("layer1.weight"), w1, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(f.get_tensor("layer1.bias"), w2, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not is_native_available(), reason="Native library not compiled")
def test_native_tokenizer_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        hk_file = Path(tmpdir) / "tok_model.hk"

        tokens = ["<unk>", "<s>", "</s>", "hello", "world", " ", "h", "e", "l", "o", "w", "r", "d"]
        scores = [0.0] * len(tokens)
        types = [2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        merges = ["h e", "he l", "hel l", "hell o", "w o", "wo r", "wor l", "worl d"]

        meta = {
            "general.architecture": "llama",
            "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.scores": scores,
            "tokenizer.ggml.token_type": types,
            "tokenizer.ggml.merges": merges,
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.unknown_token_id": 0,
        }

        # Save model containing metadata and a dummy tensor
        dummy_tensor = torch.zeros((2, 2), dtype=torch.float32)
        hk.save_file({"dummy": dummy_tensor}, str(hk_file), metadata=meta)

        # Initialize native tokenizer. It holds the file mapped, so close it before the temporary
        # directory goes away (Windows cannot delete a mapped file).
        with NativeHKTokenizer(hk_file) as tok:
            assert tok.vocab_size == len(tokens)

            # Test encoding
            ids = tok.encode("hello", add_bos=True, add_eos=False)
            assert len(ids) > 0
            assert ids[0] == 1  # BOS

            # Test decoding
            text = tok.decode(ids, skip_special_tokens=True)
            assert "hello" in text


@pytest.mark.skipif(not is_native_available(), reason="Native library not compiled")
def test_native_inference_engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        hk_file = Path(tmpdir) / "tiny_transformer.hk"

        dim = 8
        hidden = 16
        vocab = 10

        meta = {
            "general.architecture": "llama",
            "llama.block_count": 1,
            "llama.embedding_length": dim,
            "llama.feed_forward_length": hidden,
            "llama.attention.head_count": 2,
            "llama.attention.head_count_kv": 2,
            "llama.context_length": 32,
            "llama.vocab_size": vocab,
            "llama.rope.dimension_count": 4,
        }

        # Build dummy weights
        tensors = {
            "token_embd.weight": torch.randn(vocab, dim, dtype=torch.float32) * 0.1,
            "blk.0.attn_norm.weight": torch.ones(dim, dtype=torch.float32),
            "blk.0.attn_q.weight": torch.randn(dim, dim, dtype=torch.float32) * 0.1,
            "blk.0.attn_k.weight": torch.randn(dim, dim, dtype=torch.float32) * 0.1,
            "blk.0.attn_v.weight": torch.randn(dim, dim, dtype=torch.float32) * 0.1,
            "blk.0.attn_output.weight": torch.randn(dim, dim, dtype=torch.float32) * 0.1,
            "blk.0.ffn_norm.weight": torch.ones(dim, dtype=torch.float32),
            "blk.0.ffn_gate.weight": torch.randn(hidden, dim, dtype=torch.float32) * 0.1,
            "blk.0.ffn_up.weight": torch.randn(hidden, dim, dtype=torch.float32) * 0.1,
            "blk.0.ffn_down.weight": torch.randn(dim, hidden, dtype=torch.float32) * 0.1,
            "output_norm.weight": torch.ones(dim, dtype=torch.float32),
            "output.weight": torch.randn(vocab, dim, dtype=torch.float32) * 0.1,
        }

        hk.save_file(tensors, str(hk_file), metadata=meta)

        # Load native engine
        engine = NativeHKEngine(hk_file)
        assert engine.vocab_size == vocab

        # Forward step 0
        logits0 = engine.forward_step(token=1, pos=0)
        assert isinstance(logits0, np.ndarray)
        assert logits0.shape == (vocab,)
        assert not np.isnan(logits0).any()
        assert not np.isinf(logits0).any()

        # Forward step 1
        logits1 = engine.forward_step(token=2, pos=1)
        assert isinstance(logits1, np.ndarray)
        assert logits1.shape == (vocab,)
        assert not np.isnan(logits1).any()
        assert not np.isinf(logits1).any()

        # Cache reset
        engine.reset_cache()
        engine.close()
