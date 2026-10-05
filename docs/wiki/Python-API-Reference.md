# Python API Reference

> **Status:** the Python package (`hknt` on PyPI as named in `pyproject.toml`, import name `hk`, source in `python/hk/`) is a **companion layer** around the same container format and, when available, the compiled Zig library. The fast inference path is the Zig CLI/server (`hk run`, `hk serve`); the Python side is for converting, inspecting and editing `.hk` files, small-model experiments and training research.
>
> Requirements: Python ≥ 3.9, `torch`, `numpy`. Native functions need the shared library built by `zig build -Doptimize=ReleaseFast` (searched for next to the package or in `HK_LIB_DIR`; `hk.is_native_available()` tells you). Without it, some features fall back to Python or raise `RuntimeError`. CI installs the package and runs the Python tests on Linux/macOS/Windows with Python 3.10–3.12; the reference below was written from the source, and the Python suite was not re-run on the machine used for the Zig work.
>
> The Python HTTP/CLI test scripts were replaced by Zig tests (`zig build test-e2e`); the Python package keeps its own pytest suite for the library code.

---

## 1. Top level (`hk`)

### Models and config
- `HKConfig` / `AutoConfig`: hyperparameters (`vocab_size`, `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, ...). `from_pretrained(path)` reads them from `.hk` metadata; `save_pretrained(dir)` writes a config.
- `HKForCausalLM(config)`: **HK's own** small transformer: token embedding, multi-head attention (no GQA, no RoPE), `LayerNorm`, GELU MLP (`mlp_fc1`, `mlp_fc2`), optional tied embeddings. It is not an adapter for Llama/Qwen/Mistral checkpoints.
  - `from_pretrained(path, config=None, device="cpu", torch_dtype=None, device_map=None)` loads `.hk` or `.safetensors` whose tensor names match this architecture. `device_map="auto"` plans CPU/GPU placement per layer from free memory.
  - `forward(input_ids, attention_mask=None, labels=None) -> ModelOutput` (`.logits`, `.loss`).
  - `generate(input_ids, max_new_tokens=20, temperature=1.0, top_k=50)`: **greedy** decoding (it takes the argmax; temperature/top-k do not introduce sampling). For a single sequence on CPU from a `.hk` file it first tries the native engine (`NativeHKEngine`) and silently falls back to PyTorch on any error. Use the CLI for real sampling.
  - `save_pretrained(path, alignment=...)`, `grow_width(layer, new_size, noise_std)`, `enable_qlora(rank, alpha, ...)`, `enable_continual_learning(protect_base=True)`.
- `AutoModel` (aliases `AutoModelForCausalLM`, `AutoModelForSequenceClassification`): picks the class from the config. Same architecture limits as above.
- Other heads/pipelines: `HKForSequenceClassification`, `HKForHandwritingRecognition`, `pipeline(...)`, `UniversalPipeline`/`CompositePipeline` (`composite.py`).

### Tokenizers
- `HKTokenizer`, `AutoTokenizer.from_pretrained(path)`: tokenizer built from the `.hk` metadata (pure-Python implementation).
- `NativeHKTokenizer`: wrapper over the Zig tokenizer (`hk_tokenizer_*` C functions), same one the engine uses.

### Native inference
- `NativeHKEngine(path)`: loads a model into the Zig engine (Llama, Qwen2 and Qwen3 style architectures; same support list as [Compatibility](Compatibility)). `forward(tokens, pos=0) -> np.ndarray` returns the logits after the last token; `forward_step(token, pos)`; `reset_cache()`; `vocab_size`, `context_size`; use as a context manager. Positions are explicit. This gives you logits, not a sampler or chat loop.

### Conversion
- `convert_safetensors_to_hk` (native), `convert_hf_checkpoint`, `HFArchitectureMapper` and helpers in `hf_mapper.py`, `convert_gguf_to_hk` / `export_hk_to_gguf` / `GGUFReaderLight` in `gguf_parser.py`. The Zig CLI (`hk convert-gguf`, `hk convert-safetensors`, `hk export`) is the primary, tested conversion path; see [CLI Reference](CLI-Reference).

---

## 2. Tensor files (`hk.torch`, `hk.numpy`, JAX/Flax)

A `safetensors`-style API over `.hk` files.

- `save_file(tensors, filename, metadata=None, split_index=0, split_count=1)`: writes tensors; tensors that share a `data_ptr` are stored once as `shared_ref`.
- `load_file(filename, device="cpu", with_residual=True) -> dict`: loads all tensors, following shards/index manifests automatically; dequantizes quantized tensors to float.
- `save_model(model, filename, metadata=None)` / `load_model(model, filename, strict=True)`.
- `save_sharded_file(tensors, filename_pattern, max_shard_size=..., ...)` / `load_sharded_file(...)`: multi-file checkpoints with a `*.hk.index.json` manifest.
- `safe_open(filename, framework="pt")`: lazy reader (`keys()`, `metadata()`, `get_tensor(name)`, `get_slice(name)`).
- `metadata_set(filename, key, value)`: in-place metadata edit (same limits as `hk metadata set`, see [Storage and Sparsity](Storage-and-Sparsity)).
- `hk.remote`: `read_remote_hk_header`, `safe_open_remote` (reads a remote file's header/tensors over HTTP range requests).

## 3. Raw store (`hk.raw`)

- `save_raw(filename, tensors, metadata=None, alignment=4096, split_index=0, split_count=1)`: filename first. Raw (unquantized) storage.
- `load_raw(filename, as_torch=True, device=None) -> dict` backed by the mapped file.
- `HKRawWeightStore(path)`: context manager with `keys()`, `__getitem__`, `get_numpy(name)`, `metadata()`, `hardware_profile()`, `gemv(name, x, bias=None)`, and attributes `alignment`, `is_universal_page_aligned`, `is_tensor_core_aligned`, `is_raw_storage`, `is_sharded`.
- `save_sharded_raw(base_path, shards, metadata=None, alignment=4096)` (shards = list of dicts) / `load_sharded_raw(paths)`.
- `to_amd_rocm`, `to_intel_npu`, `to_apple_metal`, `to_nvidia_tensor_core`: only return a contiguous array/tensor; no device interaction.

See [Raw Storage](Raw-Storage-and-Super-Coalescing).

## 4. Quantization and pruning (`hk.quantization`, `hk.pruning`)

- `quantize_nf4_dual_mode` / `dequantize_nf4_dual_mode`, `quantize_dq8_dual_mode`, `quantize_dqt`, `quantize_q4_k`/`dequantize_q4_k`, `quantize_q8_k`, `dequantize_q6_k`, `dequantize_q2_k`, `ImportanceMatrixCalibrator`, `QUANT_RECIPES`, `resolve_quant_type_for_tensor`.
- `make_2_4_sparse(tensor, scale_correction=True) -> (tensor, 0.5)`, `pack_2_4(tensor) -> bytes`, `unpack_2_4(bytes, shape)` (the last two need the native library).
- Pruning of `nn.Module`s: `prune_unstructured_magnitude`, `prune_wanda`, `prune_structured_2_4`, `prune_block_sparse`, `prune_structured_l2`, `fine_tune_recovery`, `LayerSparsitySchedule`.

For producing GGUF-style quantized models that the engine can run, prefer the Zig tools or llama.cpp's quantizer; the Python quantizers cover a subset of formats. See [Compatibility](Compatibility).

## 5. Training (`hk.trainer`)

`HKTrainingArguments` and `HKTrainer(model, args, train_dataset, eval_dataset, compute_metrics)` with `train()`, `save_model(path)`. Full details and defaults: [Training and Fine-Tuning](Training-and-Fine-Tuning).

## 6. Adaptive tooling (`hk.adaptive`)

- Growth: `GrowthGovernor`, `net2wider_linear`, `net2wider_swiglu`, `net2deeper_linear`, `expand_vocab(model, new_vocab_size, init_std)`, `expand_model_width(model, expansion_ratio, ...)`, `protect_base_capacity(model, ...)`. See [Dynamic Architecture Growth](Dynamic-Architecture-Growth).
- Appendix: `AppendixManager(path)` with `get_records()`, `append_lora_checkpoint(...)`, `rollback(generation)`, `verify()`; module functions `read_appendix`, `append_record`, `rollback_appendix`, `verify_lineage`, `compute_parent_hash`. See [In-Container Version Lineage](In-Container-Version-Lineage).
- Self-training: `SelfTrainingPipeline`, `SelfTrainingCurriculum`, `SelfConversationalEngine`, `ExpansionEvaluator`, `CodeSandbox`, `SPINLoss`. See [Self-Training Toolkit](Autonomous-Self-Training); note the sandbox is not a security boundary by default.

## 7. Misc

- `hk.benchmark.benchmark_model` / `compare_models`: timing helpers for PyTorch-side models (not the Zig engine; for engine-vs-llama.cpp comparisons use `zig build hk-compare`, see [Benchmarks and Performance](Benchmarks-and-Performance)).
- `hk.offload`: `HardwareMemoryInspector`, `DynamicOffloadPlanner`, `AutoDeviceDispatcher`, `DynamicOOMGuard` for placing layers across CPU/GPU memory in PyTorch.
- `hk.cli` / `hk.gui.launch_gui`: Python entry points; the `hk` console script installed by the wheel points at `hk.cli:main`.
