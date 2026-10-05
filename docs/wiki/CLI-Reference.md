# CLI Reference

```bash
hk <command> [arguments]
hk help
```

`hk` is one executable. It needs no Python, no GPU toolkit and no configuration. Commands that take a model accept a `.hk` file or a Hub reference (`owner/name[:quant]`); a reference is pulled first if it is not cached yet.

## Running models

### `hk run <model> [prompt | -p TEXT] [options]`

Generates text. Text goes to standard output, statistics to standard error, so the output can be piped.

| Option | Meaning | Default |
|:---|:---|:---|
| `-p, --prompt TEXT` | text to continue (a bare argument works too; quote it) | none |
| `--chat` | wrap the prompt as one user message using the model's chat template | off |
| `-n, --max-tokens N` | most tokens to generate | 128 |
| `--temp T` | temperature; `0` takes the most likely token | 0.8 |
| `--top-k N` | keep the N most likely tokens, `0` for no limit | 40 |
| `--top-p P` | nucleus sampling, `1` disables | 0.95 |
| `--min-p P` | drop tokens below P times the best one | 0.05 |
| `--repeat-penalty X` | penalty for recent tokens, `1` disables | 1.0 |
| `--seed N` | random seed, `0` uses the clock | 0 |
| `--threads N` | compute threads | physical cores |
| `-ngl, --gpu-layers N` | run on a GPU (Vulkan) when N > 0; also `HK_GPU=1` | CPU |

Every flag is checked before the model is opened, and an unknown or malformed one is an error, never a silent default. With `--temp 0` the output is deterministic.

### `hk chat <model> [options]`

An interactive session with the model's chat template and history kept across turns. Takes the sampling options of `run` (not `-p` or `--chat`) and `-n` (default 1024). An empty line ends the session.

### `hk serve <model> [options]`

An OpenAI compatible HTTP server. See [Inference and Serving](Inference-and-Serving.md).

## Getting models

| Command | |
|:---|:---|
| `hk pull <owner/name>[:quant] [--force] [--revision REF]` | download and convert, see [Downloading Models](Downloading-Models.md) |
| `hk search <query> [--limit N] [--all]` | search the Hub |
| `hk list` | show cached models |
| `hk rm <owner/name>` | delete a cached model |
| `hk convert-gguf <in.gguf> <out.hk>` | convert a GGUF file; block formats are carried over byte for byte |
| `hk convert-safetensors <in.safetensors> <out.hk> [f32\|q4_0\|q8_0]` | convert one safetensors file, optionally quantizing |
| `hk export -f gguf <in.hk> <out.gguf>` | write a GGUF v3 file |
| `hk export -f safetensors <in.hk> <out.safetensors>` | write safetensors; float tensors keep their type, quantized ones are decoded and stored as F16 |

Tensor names are kept as they are in the container, so a model converted from GGUF exports with GGUF names.

## Looking at a container

| Command | |
|:---|:---|
| `hk inspect <file.hk>` | header, metadata and the tensor table |
| `hk dump <file.hk>` | the header field by field, with offsets |
| `hk verify <file.hk>` | check the magic, version and the alignment of the payload |
| `hk hash <file.hk>` | SHA-256 of the container and of every tensor |
| `hk eval <file.hk>` | parameter count, value range, NaN and Inf count |
| `hk tokenize <file.hk> "text"` | token ids using the vocabulary stored in the file |
| `hk detokenize <file.hk> <id> ...` | text from token ids |
| `hk metadata list <file.hk>` / `get` / `set <key> <value>` | read or change metadata in place; the weights are not rewritten |
| `hk hardware-profile` | what the processor offers, which kernels are in use, whether a GPU is usable |
| `hk benchmark <file.hk>` | time to open the file, resolve tensors, touch every page, decode a tensor |

## Editing containers (research features)

| Command | |
|:---|:---|
| `hk expand <in> <out> [--vocab N] [--width R]` | grow the vocabulary and the MLP width of a float model; see [Dynamic Architecture Growth](Dynamic-Architecture-Growth.md) |
| `hk prune <in> <out> [ratio]` | magnitude pruning |
| `hk retile <in> <out> [tile_16x16\|row_major]` | change the layout of 2-D tensors |
| `hk appendix <file.hk>` | list the version history stored in the file |
| `hk rollback <file.hk> [generation]` | cut the version history back; this truncates the file |
| `hk convert-endian <in> <out>` | byte swap a container |
| `hk gui [file.hk]` | the Tk model editor (needs Python and Tk) |

`hk expand` refuses quantized models, and says so when no tensor name matched (it looks for Hugging Face style names).

## Environment

| Variable | Effect |
|:---|:---|
| `HK_HOME` | model cache directory |
| `HF_TOKEN`, `HF_ENDPOINT`, `HK_HTTP_BACKOFF_MS` | downloads, see [Downloading Models](Downloading-Models.md) |
| `HK_GPU=1` | use the GPU when possible, without `-ngl` |
| `HK_KERNELS=<level>` | force an instruction set level (for example `avx2` or `generic`); useful for testing and measuring |
| `HK_THREADS=N` | thread count for the developer tools |
| `HK_API_KEY` | API key for `hk serve` |

## Exit status

Zero on success, non-zero on any error. Messages for the user go to standard error.
