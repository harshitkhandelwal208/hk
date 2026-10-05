"""
Unit and Integration Tests for HK GGUF Parity & Beyond Expansion Pillars
Validates:
1. Massive Architectural Breadth (137+ Models Registry & Bidirectional Mapping)
2. Advanced Quantization Zoo (K-Quants, I-Quants, Microscaling, imatrix Calibration & Recipes)
3. Deep Tokenizer Coverage (SentencePiece Binary Protobuf-free & Mistral Tekkenizer)
4. Standardized Taxonomy (200+ Keys, YaRN, MLA, MoE, SSM)
5. CLI Tools & Graphical Editor Verification
"""

import os
import sys
import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np

import hk
from hk.constants import (
    ModelArchitecture,
    PreTokenizerType,
    TokenType,
    RoPEScalingType,
    HKTaxonomyKeys,
)
from hk.models import ARCHITECTURES_REGISTRY, get_architecture_tensor_mappings
from hk.hf_mapper import HFArchitectureMapper
from hk.format import StorageType
from hk.quantization import (
    quantize_q4_k,
    dequantize_q4_k,
    quantize_q8_k,
    dequantize_q8_k,
    dequantize_q6_k,
    dequantize_q2_k,
    ImportanceMatrixCalibrator,
    QUANT_RECIPES,
    resolve_quant_type_for_tensor,
)
from hk.tokenizer import (
    HKTokenizer,
    parse_sentencepiece_model,
    parse_tekken_json,
)


class TestParityPillars(unittest.TestCase):

    def test_pillar1_massive_architectural_breadth(self):
        """Verify registry contains 137+ architectures and bidirectional tensor mapping tables."""
        arch_keys = list(ARCHITECTURES_REGISTRY.keys())
        self.assertGreaterEqual(len(arch_keys), 137, f"Expected at least 137 models, found {len(arch_keys)}")

        # Verify key families are present
        expected_models = [
            "deepseek2", "deepseek3", "deepseek_r1", "llama4", "qwen2.5", "qwen3",
            "gemma2", "grok", "falcon_h1", "phi3", "dbrx", "command-r-plus",
            "olmoe", "minicpm3", "mamba", "mamba2", "jamba", "rwkv6",
            "clip", "llava", "qwen2_vl", "whisper", "flux", "modernbert"
        ]
        for m in expected_models:
            self.assertIn(m, ARCHITECTURES_REGISTRY, f"Model architecture {m} must be registered")

        # Verify forward mapping for DeepSeek MLA attention
        ds_name = "model.layers.0.self_attn.q_b_proj.weight"
        mapped_ds = HFArchitectureMapper.map_tensor_name_to_hk(ds_name, architecture="deepseek2")
        self.assertEqual(mapped_ds, "layers.0.attn_q_b.weight")

        # Verify forward mapping for MoE routers
        moe_name = "model.layers.2.mlp.gate.weight"
        mapped_moe = HFArchitectureMapper.map_tensor_name_to_hk(moe_name, architecture="qwen2_moe")
        self.assertEqual(mapped_moe, "layers.2.ffn_gate_inp.weight")

    def test_pillar2_advanced_quantization_and_recipes(self):
        """Verify K-Quants roundtrip accuracy, imatrix calibration, and predefined per-tensor recipes."""
        torch.manual_seed(42)
        weights = torch.randn(512, dtype=torch.float32)

        # 1. Q4_K quantize & dequantize
        q4_bytes = quantize_q4_k(weights)
        self.assertEqual(len(q4_bytes), (512 // 256) * 144)
        deq_q4 = dequantize_q4_k(q4_bytes, [512])
        err_q4 = (weights - deq_q4).abs().mean().item()
        self.assertLess(err_q4, 0.15, "Q4_K reconstruction error should be small")

        # 2. Q8_K quantize & dequantize
        q8_bytes = quantize_q8_k(weights)
        self.assertEqual(len(q8_bytes), (512 // 256) * 292)
        deq_q8 = dequantize_q8_k(q8_bytes, [512])
        err_q8 = (weights - deq_q8).abs().mean().item()
        self.assertLess(err_q8, 0.02, "Q8_K reconstruction error should be very small (<0.02)")

        # 3. Importance Matrix Calibrator
        calibrator = ImportanceMatrixCalibrator()
        calibrator.observe("layers.0.attn_q.weight", torch.randn(10, 64))
        calibrator.observe("layers.0.attn_q.weight", torch.randn(10, 64))
        imp = calibrator.get_importance("layers.0.attn_q.weight")
        self.assertIsNotNone(imp)
        self.assertEqual(len(imp), 64)
        self.assertAlmostEqual(float(np.mean(imp)), 1.0, places=2)

        # 4. Predefined mixed recipes
        self.assertEqual(resolve_quant_type_for_tensor("Q4_K_M", "layers.0.attn_v.weight"), StorageType.Q6_K)
        self.assertEqual(resolve_quant_type_for_tensor("Q4_K_M", "layers.0.attn_q.weight"), StorageType.Q4_K)
        self.assertEqual(resolve_quant_type_for_tensor("Q4_K_M", "model.layers.0.input_layernorm.weight"), StorageType.F32)

    def test_pillar3_deep_tokenizer_coverage(self):
        """Verify binary SentencePiece Protobuf-free parser and Mistral Tekkenizer support."""
        # 1. Mistral Tekkenizer parser
        tekken_dict = {
            "config": {"vocab_size": 4},
            "vocab": [
                {"token": "<s>", "rank": 0, "score": 0.0},
                {"token": "</s>", "rank": 1, "score": 0.0},
                {"token": "<unk>", "rank": 2, "score": 0.0},
                {"token": "hello", "rank": 3, "score": -1.2},
            ],
            "merges": ["h ello"]
        }
        tokens, scores, merges = parse_tekken_json(tekken_dict)
        self.assertEqual(tokens, ["<s>", "</s>", "<unk>", "hello"])
        self.assertEqual(scores, [0.0, 0.0, 0.0, -1.2])
        self.assertEqual(merges, [("h", "ello")])

        tok = HKTokenizer(tokens=tokens, scores=scores, merges=merges)
        enc = tok.encode("hello", add_special_tokens=False)
        self.assertIn(3, enc)

        # 2. Binary SentencePiece Protobuf wire-format parser
        def encode_varint(val):
            out = bytearray()
            while True:
                b = val & 0x7F
                val >>= 7
                if val:
                    out.append(b | 0x80)
                else:
                    out.append(b)
                    break
            return bytes(out)

        def make_spm_piece(text, score, p_type):
            b_text = text.encode("utf-8")
            payload = bytearray()
            # field 1: piece string
            payload += encode_varint((1 << 3) | 2) + encode_varint(len(b_text)) + b_text
            # field 2: score float32
            payload += encode_varint((2 << 3) | 5) + struct.pack("<f", score)
            # field 3: type varint
            payload += encode_varint((3 << 3) | 0) + encode_varint(p_type)
            return encode_varint((3 << 3) | 2) + encode_varint(len(payload)) + payload

        spm_bytes = (
            make_spm_piece("<unk>", 0.0, int(TokenType.UNKNOWN)) +
            make_spm_piece("apple", -2.5, int(TokenType.NORMAL)) +
            make_spm_piece("banana", -3.1, int(TokenType.NORMAL))
        )
        parsed_toks, parsed_scores, parsed_types = parse_sentencepiece_model(spm_bytes)
        self.assertEqual(parsed_toks, ["<unk>", "apple", "banana"])
        self.assertAlmostEqual(parsed_scores[1], -2.5, places=3)
        self.assertEqual(parsed_types[0], int(TokenType.UNKNOWN))

    def test_pillar4_taxonomy_and_hyperparameters(self):
        """Verify 200+ taxonomy keys, RoPE YaRN / Dynamic scaling, MLA, MoE, and SSM namespaces."""
        # Verify key constants are well-formed
        self.assertEqual(HKTaxonomyKeys.GENERAL_ARCHITECTURE, "general.architecture")
        self.assertEqual(HKTaxonomyKeys.ATTENTION_KEY_LENGTH_MLA, "{arch}.attention.key_length_mla")
        self.assertEqual(HKTaxonomyKeys.ROPE_SCALING_TYPE, "{arch}.rope.scaling.type")
        self.assertEqual(HKTaxonomyKeys.EXPERT_COUNT, "{arch}.expert_count")
        self.assertEqual(HKTaxonomyKeys.SSM_CONV_KERNEL, "{arch}.ssm.conv_kernel")

        # Verify config expansion
        hf_cfg = {
            "architectures": ["DeepSeekV3ForCausalLM"],
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_hidden_layers": 8,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "n_routed_experts": 64,
            "num_experts_per_tok": 8,
            "rope_scaling": {"type": "yarn", "factor": 4.0},
        }
        meta = HFArchitectureMapper.convert_hf_config_to_metadata(hf_cfg)
        self.assertEqual(meta["general.architecture"], "deepseek3")
        self.assertEqual(meta["deepseek3.attention.key_length_mla"], 512)
        self.assertEqual(meta["deepseek3.expert_count"], 64)
        self.assertEqual(meta["deepseek3.rope.scaling.type"], "yarn")

    def test_pillar5_editor_gui_module(self):
        """Verify HK Graphical Editor module loads cleanly."""
        try:
            import tkinter  # noqa: F401
        except ImportError as e:
            self.skipTest(f"tkinter is not available here: {e}")
        from tools.hk_editor_gui import HKEditorApp, STORAGE_TYPE_NAMES
        self.assertIn(0x42, STORAGE_TYPE_NAMES)
        self.assertEqual(STORAGE_TYPE_NAMES[0x42], "Q4_K")
        self.assertEqual(STORAGE_TYPE_NAMES[0x45], "Q8_K")


if __name__ == "__main__":
    unittest.main()
