# Downloading Models

`hk pull` fetches a model from the Hugging Face Hub straight into a `.hk` file.

```bash
hk pull Qwen/Qwen3-0.6B                          # a safetensors repository
hk pull bartowski/SmolLM2-135M-Instruct-GGUF     # a GGUF repository, best default quant
hk pull bartowski/SmolLM2-135M-Instruct-GGUF:Q8_0
hk pull owner/name --revision some-branch --force
```

## What it does

A GGUF or safetensors file is never saved as such. The bytes are converted while they arrive and written once, as `.hk`, so the disk use is the size of the result and the memory use is a few megabytes whatever the model size. The download is hashed as it streams, and the hash is compared with the SHA-256 the Hub lists before the file is moved into place. A wrong hash, a truncated body or any other error deletes the partial file; nothing half written is ever left in the cache.

- **Resume.** A dropped connection continues from the last byte with a range request.
- **Retries.** Rate limits (429) and server errors are retried with backoff, then reported.
- **Sharded repositories.** Safetensors repositories split over several files are converted shard by shard. If the Hub does not list a checksum for a shard, the summary says "no checksum available" instead of "checksum verified".
- **Authentication.** Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`, or log in with the Hugging Face tools, which write `~/.cache/huggingface/token`). The token is sent only to the Hub's own host and is dropped when a download is redirected to another host.
- **Choosing a file.** With a GGUF repository, a `:selector` picks the file whose name contains it (`Q8_0`, `IQ4_XS`, or an exact name). Without one, the first of Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, Q4_0, Q3_K_M, IQ4_XS, Q2_K that exists is used.

Safetensors repositories need `config.json` and a byte level BPE `tokenizer.json`. A repository with a SentencePiece or Unigram tokenizer is refused with a message; its GGUF version works.

## Where models go

`$HK_HOME`, else `$XDG_CACHE_HOME/hk`, else `~/.cache/hk`, in `models/<owner>--<name>/`.

```bash
hk list                       # what is cached
hk rm Qwen/Qwen3-0.6B         # delete it
hk search qwen3 --limit 10    # search the Hub (GGUF repositories by default, --all for everything)
```

## Environment variables

| Variable | Effect |
|:---|:---|
| `HK_HOME` | cache directory |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | Hub access token |
| `HF_ENDPOINT` | use another Hub, for example a mirror or a test server |
| `HK_HTTP_BACKOFF_MS` | base delay between retries |

## Not done yet

- Split GGUF files (`-00001-of-00003.gguf`) are refused with a clear message.
- An interrupted conversion starts over. The download resumes; the conversion does not.

The downloader is tested against a mock Hub that injects wrong checksums, dropped and truncated connections, rate limits, redirects to other hosts and missing authorization (`tests/hub_tests.zig`).
