# Inference and Serving

## The engine

`hk run`, `hk chat` and `hk serve` use the same engine, written for this project. It is a dense decoder transformer (see [Compatibility](Compatibility.md)) with these properties:

- **Weights stay in the file.** They are memory mapped and read in place; the quantized formats are used as they are, with integer dot products against activations quantized to 8 bits per block of 32 (legacy formats) or 256 (K and IQ formats), the same scheme GGML uses.
- **Prompts are processed in batches.** Up to 256 tokens go through the network at once, with a register tiled matrix multiply, so a long prompt is far faster than the same tokens one by one.
- **The KV cache is f16** and grows in segments of 256 positions, so the memory it uses follows the real context length, not the configured maximum.
- **Attention is exact.** One pass with an online softmax, keys stored transposed in tiles so sixteen positions are scored per vector. At long contexts one token's attention is split over several threads.
- **Everything is allocated up front.** A forward pass allocates nothing.
- **Deterministic.** With `--temp 0` the same prompt gives the same text, run after run and on every instruction set level.

## Command line

```bash
hk run model.hk "Explain gravity in one sentence." --temp 0 -n 64
hk run model.hk "What is 2+2?" --chat           # wrap the prompt with the chat template
hk chat model.hk                                # interactive, keeps the conversation
```

Statistics are printed to standard error after each answer:

```
[prompt 21 tok (0 cached), 412.0 tok/s | generated 64 tok, 38.9 tok/s | memory 663 MiB = 52 private + 611 mapped | threads 6]
```

"Private" is memory the process owns; "mapped" is the model file in the page cache, which the operating system can drop. Within one `hk chat` session the text of earlier turns is not evaluated again: only the new message is.

## Server

```bash
hk serve model.hk --port 8080 --slots 4 --ctx 4096
```

| Option | Meaning | Default |
|:---|:---|:---|
| `--host ADDR` | address to listen on | 127.0.0.1 |
| `--port N` | port | 8080 |
| `--slots N` | conversations served at once (1 to 64) | 4 |
| `--ctx N` | context window per slot | the model's, up to 8192 |
| `--batch N` | most tokens per forward pass | 256 |
| `--threads N` | compute threads | physical cores |
| `-ngl N` | use the GPU (a device region per slot) | CPU |
| `--api-key KEY` | require `Authorization: Bearer KEY` (or set `HK_API_KEY`) | none |
| `--alias NAME` | model name reported by the API | file name |
| `--max-body-mb N` | largest request body | 32 |

### Routes

| Route | |
|:---|:---|
| `POST /v1/chat/completions` | chat, with or without streaming (`"stream": true`, `stream_options.include_usage`) |
| `POST /v1/completions` | plain completion |
| `GET /v1/models` | the loaded model |
| `GET /health` | `{"status":"ok"}`; open even when an API key is set |
| `GET /metrics` | Prometheus style counters (`hk_slots_active` and others) |
| `POST /tokenize`, `POST /detokenize` | the model's tokenizer |

Request fields understood: `messages`, `prompt`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `presence_penalty`, `frequency_penalty`, `repeat_penalty`, `logit_bias`, `stream`, `stream_options`, `tools` (rendered through the template). Errors come back as OpenAI style JSON with the right status: 400 for malformed input and for a prompt longer than the context (`context_length_exceeded`), 401 for a missing or wrong key, 413 for an oversized body, 503 when every slot is busy and the queue is full.

### How concurrency works

A scheduler thread owns the engine. Each step it takes one token from every active conversation, runs them as one batch, so the weights are streamed once for all of them, and samples a token for each. New prompts are evaluated in chunks between decode steps. A prompt cache reuses the longest shared prefix of a finished conversation, which is why a long system prompt costs nothing the second time. A client that disconnects mid stream frees its slot.

This is tested end to end (`tests/server_tests.zig`): a stream equals the non streamed answer, twelve concurrent requests over four slots give the same text as one at a time, a shared prefix is reused, disconnects free slots, an oversized body is refused, and the API key is enforced.

### What the server does not do

No grammar or JSON schema constrained decoding, no tool call parsing (tools are rendered into the prompt, but the reply is returned as text), no speculative decoding, no vision inputs.

## From Python and C

The C library exposes the same engine: `hk_engine_load_from_file`, `hk_engine_forward_tokens`, `hk_engine_reset_cache`, `hk_engine_get_vocab_size` and friends in `include/hk.h`, and the Python package wraps them as `hk.native.NativeHKEngine`. Hugging Face style classes (`AutoModelForCausalLM`) in the Python package run PyTorch models; see the [Python API Reference](Python-API-Reference.md).
