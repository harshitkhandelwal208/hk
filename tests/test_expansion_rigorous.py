"""
Rigorous & Exhaustive Test Suite for HK Dynamic Architecture Expansion:
1. Multi-Layer SwiGLU Function Preservation (exact numerical identity)
2. Vocabulary Expansion Invariance (existing token logits identical)
3. Tied-Weight Memory Address Preservation after Vocabulary Resize
4. Plasticity Isolation (Zero gradients on base weights, non-zero on expanded capacity)
5. GrowthGovernor Hardware Memory Ceiling & Boundary Enforcement
6. Day-0 Loss Continuity (exact 0.00 loss jump on validation data)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local python directory is on path
sys.path.insert(0, os.path.abspath("python"))

from hk import HKConfig, HKForCausalLM
from hk.adaptive.growth import (
    GrowthGovernor,
    net2wider_linear,
    net2wider_swiglu,
    net2deeper_linear,
    ModularResidualBlock,
    expand_vocab,
    expand_model_width,
    protect_base_capacity,
)


def test_multilayer_swiglu_preservation():
    print("\n--- 1. Testing Multi-Layer SwiGLU Net2Wider Function Preservation ---")
    torch.manual_seed(42)

    class ThreeLayerSwiGLU(nn.Module):
        def __init__(self, hidden_dim=64, inter_dim=128):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "gate_proj": nn.Linear(hidden_dim, inter_dim, bias=False),
                    "up_proj": nn.Linear(hidden_dim, inter_dim, bias=False),
                    "down_proj": nn.Linear(inter_dim, hidden_dim, bias=False),
                })
                for _ in range(3)
            ])

        def forward(self, x):
            for l in self.layers:
                # SwiGLU MLP: down_proj(SiLU(gate(x)) * up(x))
                mlp = l["down_proj"](F.silu(l["gate_proj"](x)) * l["up_proj"](x))
                x = x + mlp
            return x

    model = ThreeLayerSwiGLU(hidden_dim=64, inter_dim=128)
    model.eval()

    x = torch.randn(8, 16, 64)
    with torch.no_grad():
        out_before = model(x)

    # Widen all 3 SwiGLU layers from 128 -> 192 (1.5x capacity)
    for idx, l in enumerate(model.layers):
        w_gate, w_up, w_down = net2wider_swiglu(
            l["gate_proj"], l["up_proj"], l["down_proj"],
            new_intermediate_size=192,
            noise_std=0.0,
            seed=100 + idx,
        )
        l["gate_proj"] = w_gate
        l["up_proj"] = w_up
        l["down_proj"] = w_down

    with torch.no_grad():
        out_after = model(x)

    max_diff = torch.max(torch.abs(out_after - out_before)).item()
    print(f"  Max absolute deviation across 3 stacked SwiGLU layers: {max_diff:.2e}")
    assert max_diff < 1e-6, f"Function not preserved across SwiGLU stack: {max_diff}"
    print("  [PASS] Multi-layer SwiGLU Net2Wider preserved exact function output.")


def test_vocab_expansion_logit_invariance():
    print("\n--- 2. Testing Vocabulary Expansion Invariance on Existing Tokens ---")
    torch.manual_seed(42)

    config = HKConfig(
        model_type="causal_lm",
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        tie_word_embeddings=True,
    )
    # float64: float32 logits differ by BLAS summation order when the output width
    # changes (100 -> 150 rows), which would mask real regressions behind a loose bound.
    model = HKForCausalLM(config).double()
    model.eval()

    # Input using only existing tokens (0..99)
    input_ids = torch.randint(0, 100, (4, 16))
    with torch.no_grad():
        logits_before = model(input_ids).logits

    # Expand vocabulary from 100 -> 150 (+50 new language tokens)
    model.expand_vocab(new_vocab_size=150)
    assert model.config.vocab_size == 150
    assert model.embed_tokens.weight.shape[0] == 150
    assert model.lm_head.weight.shape[0] == 150

    with torch.no_grad():
        logits_after = model(input_ids).logits

    # Logits for the original 100 tokens must be unchanged (float64 rounding only)
    logits_after_old_slice = logits_after[..., :100]
    max_diff = torch.max(torch.abs(logits_after_old_slice - logits_before)).item()
    print(f"  Logit difference for original 100 tokens: {max_diff:.2e}")
    assert max_diff < 1e-10, f"Original token logits changed: {max_diff}"
    print("  [PASS] Existing token logits are numerically identical (< 1e-5) after vocabulary expansion.")


def test_tied_weights_integrity():
    print("\n--- 3. Testing Tied Weights Memory Integrity During Expansion ---")
    config = HKConfig(
        vocab_size=50,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        tie_word_embeddings=True,
    )
    model = HKForCausalLM(config)
    assert model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()

    # Expand vocab
    model.expand_vocab(new_vocab_size=80)
    assert model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr(), (
        "Tied weights must maintain identical data_ptr after vocabulary expansion!"
    )
    print("  [PASS] embed_tokens and lm_head maintain tied memory sharing after expansion.")


def test_plasticity_isolation_gradient_masking():
    print("\n--- 4. Testing Plasticity Isolation & Gradient Masking ---")
    torch.manual_seed(42)

    config = HKConfig(
        vocab_size=50,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        tie_word_embeddings=False,
    )
    model = HKForCausalLM(config)
    model.train()

    old_v = 50
    old_inter = 64

    # 1. Expand vocab and width
    model.expand_vocab(new_vocab_size=75) # +25 tokens
    model.expand_width(expansion_ratio=1.5) # 64 -> 96 (+32 channels)

    # 2. Activate plasticity protection
    hooks = model.enable_continual_learning(protect_base=True)
    assert len(hooks) > 0, "Hooks must be registered for plasticity protection"

    # 3. Perform a forward pass on new tokens
    # Pass tokens with some new indices (>= 50)
    input_ids = torch.tensor([[55, 60, 65, 70]])
    labels = torch.tensor([[60, 65, 70, 74]])

    out = model(input_ids, labels=labels)
    out.loss.backward()

    # 4. Verify gradient masking on base capacity:
    # Embedding table: grad for 0..49 must be strictly zero!
    embed_grad = model.embed_tokens.weight.grad
    assert embed_grad is not None
    base_embed_grad_norm = torch.norm(embed_grad[:old_v, :]).item()
    new_embed_grad_norm = torch.norm(embed_grad[old_v:, :]).item()
    print(f"  Base vocabulary grad norm: {base_embed_grad_norm:.2e} (expected: 0.00)")
    print(f"  New vocabulary grad norm:  {new_embed_grad_norm:.2e} (expected: > 0.00)")
    assert base_embed_grad_norm == 0.0, "Base vocabulary received non-zero gradients!"
    assert new_embed_grad_norm > 0.0, "New vocabulary did not receive gradients!"

    # MLP layers: grad for old intermediate channels must be strictly zero!
    fc1_grad = model.layers[0].mlp_fc1.weight.grad
    base_fc1_grad_norm = torch.norm(fc1_grad[:old_inter, :]).item()
    new_fc1_grad_norm = torch.norm(fc1_grad[old_inter:, :]).item()
    print(f"  Base intermediate channel grad norm: {base_fc1_grad_norm:.2e} (expected: 0.00)")
    print(f"  New intermediate channel grad norm:  {new_fc1_grad_norm:.2e} (expected: > 0.00)")
    assert base_fc1_grad_norm == 0.0, "Base intermediate channels received non-zero gradients!"
    assert new_fc1_grad_norm > 0.0, "New intermediate channels did not receive gradients!"

    print("  [PASS] Plasticity isolation verified: base weights are shielded from catastrophic forgetting.")


def test_growth_governor_boundaries():
    print("\n--- 5. Testing GrowthGovernor Memory Ceilings & Boundary Rules ---")
    governor = GrowthGovernor(max_vram_mb=500, max_growth_ratio=2.0)

    # Case A: Modest growth (1.25x, ~20MB) -> Approved
    ok, reason = governor.can_grow(current_params=10_000_000, additional_params=2_500_000, dtype_bytes=4)
    assert ok, f"Expected approval, got {reason}"
    print(f"  [PASS] 1.25x growth approved: {reason}")

    # Case B: Exceeds max growth ratio (2.5x > 2.0x) -> Rejected
    ok, reason = governor.can_grow(current_params=10_000_000, additional_params=15_000_000, dtype_bytes=4)
    assert not ok, "Expected rejection for ratio"
    print(f"  [PASS] 2.5x growth correctly rejected: {reason}")

    # Case C: Exceeds memory budget (600MB > 500MB) -> Rejected
    ok, reason = governor.can_grow(current_params=100_000_000, additional_params=200_000_000, dtype_bytes=4)
    assert not ok, "Expected rejection for VRAM ceiling"
    print(f"  [PASS] Excessive memory allocation correctly rejected: {reason}")


def test_day_zero_loss_continuity():
    print("\n--- 6. Testing Day-0 Loss Continuity on Native Language Data ---")
    torch.manual_seed(42)

    config = HKConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    model = HKForCausalLM(config)
    model.eval()

    # Validation batch
    input_ids = torch.randint(0, 100, (4, 16))
    labels = input_ids.clone()

    with torch.no_grad():
        loss_before = model(input_ids, labels=labels).loss.item()

    # Expand width across model
    model.expand_width(expansion_ratio=1.5, noise_std=0.0)

    with torch.no_grad():
        loss_after = model(input_ids, labels=labels).loss.item()

    loss_diff = abs(loss_after - loss_before)
    print(f"  Loss before expansion: {loss_before:.6f}")
    print(f"  Loss after expansion:  {loss_after:.6f}")
    print(f"  Loss deviation:        {loss_diff:.2e}")
    assert loss_diff < 1e-5, f"Day-0 loss jump detected: {loss_diff}"
    print("  [PASS] Day-0 loss continuity confirmed (0.00 loss jump).")


def main():
    print("=" * 80)
    print("RUNNING RIGOROUS ARCHITECTURE EXPANSION VERIFICATION SUITE")
    print("=" * 80)

    tests = [
        test_multilayer_swiglu_preservation,
        test_vocab_expansion_logit_invariance,
        test_tied_weights_integrity,
        test_plasticity_isolation_gradient_masking,
        test_growth_governor_boundaries,
        test_day_zero_loss_continuity,
    ]

    passed = 0
    failed = 0

    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"EXPANSION SUITE RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 80)

    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
