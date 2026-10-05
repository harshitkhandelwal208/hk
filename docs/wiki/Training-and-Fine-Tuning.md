# Training and Fine-Tuning (`HKTrainer`)

> **Status: experimental, Python/PyTorch only.** Training lives in `python/hk/trainer.py` and `python/hk/modeling.py`. It is **not** part of the Zig engine, which is inference-only. Two things to know before you start:
>
> 1. `HKForCausalLM` is **HK's own small transformer** (learned token embedding, multi-head attention without grouped-query or RoPE, `LayerNorm`, a GELU two-layer MLP `mlp_fc1`/`mlp_fc2`). It is *not* a loader for Llama, Qwen, Mistral or other Hugging Face architectures, and it does not read GGUF files. You can pretrain or fine-tune models of this architecture; you cannot point it at an arbitrary downloaded LLM. (`from_pretrained` reads `.hk` or `.safetensors` files whose tensor names match this architecture.)
> 2. The models this produces are **not** loadable by the `hk run` inference engine unless you export weights with matching names for a supported architecture. They are for experiments and research on the Python side.
>
> The trainer's test coverage is a small-model smoke test (`tests/` Python suites). No convergence, speed or memory results are claimed on this page.

---

## What `HKTrainer` does

A plain PyTorch training loop (AdamW, linear warmup, gradient accumulation, grad clipping) with optional extras:

| Feature | Flag | What it does |
| :--- | :--- | :--- |
| Full fine-tuning | `use_qlora=False` | All parameters train. |
| QLoRA-style adapters | `use_qlora=True`, `lora_rank`, `lora_alpha` | `enable_qlora` freezes the model, replaces linear layers with `HKQuantizedLinear` (base weights quantized to Q4_0 by default) and attaches trainable LoRA A/B matrices. Only the adapters get gradients. |
| Plasticity isolation | `protect_base_capacity=True` | Gradient hooks zero the updates to pre-expansion neurons and vocabulary rows (see [Dynamic Architecture Growth](Dynamic-Architecture-Growth)). |
| Plateau growth | `enable_adaptive_growth`, `growth_patience`, `growth_width_factor` | When loss does not improve by >1e-4 for `growth_patience` checks, widens every MLP by `growth_width_factor` (`model.grow_width`, new units seeded with 1e-5 noise) and migrates AdamW moment estimates to the new shapes. Function preservation holds up to that small noise. Without an `eval_dataset` the training loss is the plateau signal. |
| Self-play / sandbox flags | `enable_self_play`, `enable_sandbox_eval` | Currently **only construct** a `SPINLoss` / `CodeSandbox` object on the trainer (`trainer.spin_loss_fn`, `trainer.sandbox`). The built-in training loop does not call them; they are there for custom loops. `spin_lambda` is not read. See [Autonomous Self-Training](Autonomous-Self-Training). |
| Async evaluation | (with `eval_dataset`) | Validation runs on a background thread over a snapshot of the weights. |

```python
from hk import HKConfig, HKForCausalLM
from hk.trainer import HKTrainer, HKTrainingArguments

config = HKConfig(vocab_size=8192, hidden_size=256, num_hidden_layers=4,
                  num_attention_heads=4, intermediate_size=1024)
model = HKForCausalLM(config)

args = HKTrainingArguments(output_dir="./out", learning_rate=5e-4, batch_size=4,
                           num_train_epochs=3, use_qlora=False)
trainer = HKTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)
result = trainer.train()        # returns global_step, final_loss, eval_metrics, generation, output_path
```

Datasets yield `(input_ids, labels)`, a dict with `input_ids` (and optionally `labels`), or bare `input_ids`. Loss comes from the model's `labels=` argument. For instruction tuning, mask the prompt tokens in `labels` with `-100` yourself; the trainer does no chat templating or masking. If you pass no dataset, the trainer trains on one random dummy batch (a smoke-test mode, not useful training).

---

## Checkpoints and the appendix

`save_model` calls `model.save_pretrained(path, alignment=...)` and then appends one `lora_adapter` record to the file's appendix (see [In-Container Version Lineage](In-Container-Version-Lineage)) containing:

- **the `torch.save` bytes of every parameter with `requires_grad`**, plus loss/accuracy/pass-rate metrics;
- or the marker `HK_BASE_WEIGHTS_SAVED` if nothing is trainable.

Consequences you should know:

- With QLoRA the trainable set is just the adapters, so records are small.
- With **full fine-tuning every parameter is trainable**, so each appendix record is a complete copy of the model's weights. FFT checkpoints are therefore not smaller than ordinary checkpoints; there is no delta compression.
- The appendix write is wrapped in a broad `try/except` that ignores failures, so a failed append does not stop training and does not warn. Verify with `hk appendix <file>` when it matters.
- Appended adapters are not merged or applied on load; reading them back is up to your code.

---

## Pretraining and sharding

Define a model through `HKConfig`, train with `HKTrainer`, and save with `save_pretrained`. For large checkpoints `hk.torch.save_sharded_file` / `hk.raw.save_sharded_raw` produce shard files (see [Storage and Sparsity](Storage-and-Sparsity)). Multi-GPU data/tensor parallelism is not implemented in the trainer; use PyTorch's own tooling around it if you need that.

---

## Arguments (`HKTrainingArguments`)

| Argument | Default | Description |
| :--- | :--- | :--- |
| `output_dir` | `"./output"` | Where checkpoints (`checkpoint-<step>.hk`, final `model.hk`) are written. |
| `learning_rate` | `5e-4` | Peak AdamW learning rate. |
| `batch_size` | `4` | Per-step batch size. |
| `num_train_epochs` | `3` | Epochs. |
| `warmup_steps` | `10` | Linear warmup steps (constant LR afterwards; no decay schedule). |
| `weight_decay` | `0.01` | AdamW weight decay. |
| `logging_steps` | `5` | Interval for averaging loss and dispatching async eval. |
| `save_steps` | `50` | Checkpoint interval in optimizer steps (`0` disables). |
| `gradient_accumulation_steps` | `1` | Micro-batches per optimizer step. |
| `max_grad_norm` | `1.0` | Gradient clipping (`0` disables). |
| `enable_adaptive_growth` | `False` | Plateau-triggered MLP widening. |
| `growth_patience` | `5` | Stagnant checks before growing. |
| `growth_width_factor` | `1.25` | Intermediate-size multiplier (rounded up to a multiple of 4). |
| `protect_base_capacity` | `False` | Gradient masks on pre-expansion capacity. |
| `use_qlora` | `False` | Quantize base weights and train low-rank adapters only. |
| `lora_rank` / `lora_alpha` | `8` / `16.0` | LoRA hyperparameters. |
| `enable_self_play` / `spin_lambda` | `False` / `0.1` | Creates `trainer.spin_loss_fn` (fixed beta 0.1); not used by `train()`. `spin_lambda` is unused. |
| `enable_sandbox_eval` / `sandbox_timeout_ms` | `False` / `2000` | Creates `trainer.sandbox` (a `CodeSandbox`); not used by `train()`. |
| `alignment` | `128` | Payload alignment of saved checkpoints (see [Raw Storage](Raw-Storage-and-Super-Coalescing)). |
