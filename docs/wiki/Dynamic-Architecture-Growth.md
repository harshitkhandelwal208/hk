# Dynamic Architecture Growth (Net2Net)

> **Status:** growth is an offline/training-time feature. It exists in two independent implementations: PyTorch routines in `python/hk/adaptive/growth.py` (for `nn.Module` models) and native Zig routines in `src/growth.zig` driving `hk expand` (for `.hk` files). The inference engine runs the resulting model like any other; it has no growth logic. "Function preserving" below is a property of the construction, checked in the repo's tests on small models, not a promise about task quality after you continue training.

---

## What is guaranteed, and what is not

Net2Net-style widening copies the original weights and adds new units arranged so that the new units contribute nothing initially:

- **Wider MLP (SwiGLU)**: `src/growth.zig` (`net2WiderSwiGLU`) has two modes. *Zero-init*: existing rows are copied, new `gate`/`up` rows are added, and the matching new `down_proj` columns are **zero**, so the extra units contribute nothing. *Classic Net2Net*: new units replicate randomly chosen existing units and the outgoing `down_proj` weights of each replicated unit are divided by its replication count, so the sum is unchanged. With zero noise both give the original outputs up to floating-point rounding. `hk expand --width` uses the zero-init construction.
- **Deeper**: `net2deeper_linear` / `ModularResidualBlock` insert a block whose residual branch starts at zero, so the stack is initially an identity on the added layer.
- **Larger vocabulary**: existing token rows are copied unchanged; new rows get small random values. Logits for the existing tokens are unchanged, but new logits now take part in the softmax, so output *probabilities* shift slightly unless the new rows are initialized to produce very low logits.

What this does not promise: that the grown model trains well, that new capacity gets used, or that it fits your hardware. Training behavior is yours to evaluate.

---

## Python API (`hk.adaptive.growth`)

```python
from hk.adaptive.growth import (
    net2wider_swiglu, net2wider_linear, net2deeper_linear, ModularResidualBlock,
    expand_vocab, expand_model_width, protect_base_capacity, GrowthGovernor,
)

mlp = model.model.layers[0].mlp
mlp.gate_proj, mlp.up_proj, mlp.down_proj = net2wider_swiglu(
    mlp.gate_proj, mlp.up_proj, mlp.down_proj,
    new_intermediate_size=2048, noise_std=0.0,
)

# Whole-model helpers take an nn.Module with `layers` / `embed_tokens` / `lm_head` attributes
model = expand_model_width(model, expansion_ratio=1.33)
model = expand_vocab(model, new_vocab_size=32500, init_std=0.02)
```

Notes:

- `expand_vocab` and `expand_model_width` take the **model** and look up `embed_tokens`, `lm_head` and `layers` (directly or under `.model`); they raise `AttributeError` for other layouts.
- When the native library is built, the wider-SwiGLU math is done by it (`native_net2wider_swiglu`); otherwise PyTorch code is used.
- These work on PyTorch modules, not on the engine's `.hk` runtime. To run a grown model with the engine, save it to `.hk` and make sure its tensor names and metadata match a supported architecture (see [Compatibility](Compatibility)).

### Plasticity isolation

`protect_base_capacity(model, old_intermediate_sizes, old_vocab_size)` registers gradient hooks that zero gradients for the original neurons and vocabulary rows, so training only updates the new capacity. This stops the *original parameters* moving, which limits forgetting of what they computed; it does not prevent the new units from changing the model's behavior, so forgetting is reduced, not eliminated.

### `GrowthGovernor`

```python
gov = GrowthGovernor(max_vram_mb=6144, max_growth_ratio=1.5)
ok, reason = gov.can_grow(current_params=135_000_000, additional_params=25_000_000, dtype_bytes=2)
```

A simple budget check: rejects if `(current + additional) / current > max_growth_ratio`, or if the added parameters' bytes exceed `max_vram_mb`. If `max_vram_mb` is omitted it uses 80% of free CUDA memory when PyTorch reports CUDA, else a fixed 4096 MB. It estimates only the *added* parameter bytes; it does not model optimizer state, activations or fragmentation, so treat approval as a coarse filter. It runs in the native library when available and in Python otherwise; the decision is a handful of arithmetic operations, and no speed claim is made.

---

## CLI: `hk expand`

```bash
hk expand in.hk out.hk --width 1.33 --vocab 32000
```

- Works on f32, f16 and bf16 `.hk` files; any other storage type is refused with an error. Tensors are decoded to f32 and the output is written as **f32** (so the file grows when the input was f16/bf16).
- `--vocab N` grows tensors whose name contains `embed_tokens`, `lm_head`, `wte` or `token_embeddings` and whose first dimension is smaller than `N`.
- `--width R` (R > 1) scales the intermediate size of tensors named `gate_proj`/`w1`, `up_proj`/`w3` (new rows: Gaussian noise, std 0.02) and `down_proj`/`w2` (new columns: zero).
- It matches Hugging Face-style tensor names. A model converted from GGUF uses GGUF names (`blk.N.ffn_gate.weight`, ...), so nothing matches and the tool prints a warning that the output is just a copy. Hyperparameters in the metadata (hidden/intermediate size, vocabulary) are **not** updated; only `expanded_vocab_size` / `expanded_width_ratio` markers are added, so a grown file is meant as input to your training code rather than as a drop-in for the engine.
- Fixed random seed (42), so results are reproducible.
