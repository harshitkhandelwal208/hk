# Coming from Another Tool

## From llama.cpp, Ollama and LM Studio

hk reads the same GGUF files and runs the same kinds of models from a terminal or behind an OpenAI compatible server. The table maps what you do today to what you do here.

| Action | llama.cpp / Ollama | hk |
|:---|:---|:---|
| Download a model | `ollama pull llama3`, or download a GGUF by hand | `hk pull owner/name[:quant]` |
| One prompt | `llama-cli -m m.gguf -p "..." -n 128` | `hk run m.hk "..." -n 128` |
| Chat | `llama-cli -m m.gguf -cnv`, `ollama run` | `hk chat m.hk` |
| Server | `llama-server -m m.gguf --port 8080` | `hk serve m.hk --port 8080` |
| Use a GGUF you already have | pass it | `hk convert-gguf m.gguf m.hk`, once |
| Threads | `-t N` | `--threads N` (default: physical cores) |
| Offload to a GPU | `-ngl 99` | `-ngl 99` (Vulkan, whole model only) |
| Context size | `-c 4096` | `--ctx 4096` (server) |
| Sampling | `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--seed` | the same names |
| Inspect a file | `gguf-dump` | `hk inspect m.hk` |
| Perplexity | `llama-perplexity` | `hk-probe m.hk --ppl ids.u32` (developer tool) |

**What is the same.** The converted file holds the original quantized blocks byte for byte, so the numbers a model produces are the numbers GGML would produce, up to float summation order. The server speaks the OpenAI API, so clients written for llama-server or Ollama's OpenAI endpoint work.

**What is different.**

- The container is `.hk`, not GGUF. `hk export -f gguf in.hk out.gguf` writes one back.
- hk runs fewer architectures: Llama, Qwen2 and Qwen3 style dense models. If you run Gemma, Phi, mixture of experts or vision models, stay with llama.cpp. See [Compatibility](Compatibility.md).
- hk has no Metal, ROCm or CUDA backend. The GPU path is Vulkan and runs the whole model on one device. If your model is larger than your VRAM, llama.cpp's partial offload is the better tool today.
- Memory: a running model costs far less private memory than in llama.cpp (see [Benchmarks and Performance](Benchmarks-and-Performance.md)), because the weights stay in the mapped file and the KV cache grows as the context fills.
- Speed: see the same page for a measured comparison per format, including where hk is behind.

## From Hugging Face Transformers and safetensors (Python)

The Python package has a safetensors compatible API for storing and loading tensors in `.hk` files:

```python
from hk.torch import save_file, load_file, safe_open

save_file(tensors, "weights.hk")          # same call shape as safetensors.torch.save_file
tensors = load_file("weights.hk")          # memory mapped
with safe_open("weights.hk", framework="pt") as f:
    part = f.get_slice("model.layers.0.mlp.gate_proj.weight")[0:128, :]
```

To turn a Hugging Face checkpoint into something the native engine can run, convert it:

```bash
hk pull Qwen/Qwen3-0.6B                                 # converts as it downloads
hk convert-safetensors model.safetensors model.hk       # or one local file
```

**Not a replacement for Transformers.** The Python `AutoModelForCausalLM` of this package builds hk's own small transformer (LayerNorm, GELU MLP, full attention); it does not load Llama or Qwen weights into the Hugging Face model classes. For Llama family inference, use the native engine (`hk run`, `hk serve`, or `hk.native.NativeHKEngine`). For training, see [Training and Fine-Tuning](Training-and-Fine-Tuning.md), which is experimental.

## From `torch.save`

```python
import hk.torch as hkt

hkt.save_model(model, "model.hk")                  # no pickle, tied weights stored once
hkt.load_model(model, "model.hk", strict=True)
```

## NumPy, JAX and Flax

```python
import hk.numpy as hknp
hknp.save_file({"features": array}, "data.hk")
loaded = hknp.load_file("data.hk")

import hk.jax as hkjax
hkjax.save_file(params, "flax_weights.hk")
```

## From a fine-tuning stack (PEFT, Unsloth)

`HKTrainer` can run full fine-tuning and a LoRA style adapter mode, and can widen a layer when the loss plateaus. It is a research feature: it works in its unit tests, and it has not been compared with PEFT, Unsloth or TRL on real models. If you need a proven fine-tuning recipe today, use those. See [Training and Fine-Tuning](Training-and-Fine-Tuning.md) for what is tested.
