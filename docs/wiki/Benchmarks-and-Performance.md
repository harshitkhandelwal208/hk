# Benchmarks and Performance

Every number on this page was measured with the tools in this repository, on the machine and conditions stated. Nothing here is projected to other hardware. Earlier versions of this page listed micro-benchmark speedups (for example "346x faster layer retrieval", file-load comparisons against SafeTensors) from Python scripts; those were not reproducible comparisons of like with like and have been removed.

## Method

- **Tool:** `hk-compare` (built by `zig build`) runs `hk-probe --bench` and llama.cpp's `llama-bench` on the same model, alternating hk, llama.cpp, hk, llama.cpp..., so slow moments on the machine (another process, thermal throttling) hit both. Both tools are summarized the same way: the **median** of the repeats, with the min-max range in brackets.
- **Workload:** prompt processing ("prefill") of 512 tokens, then generation ("decode") of 64 tokens, 6 threads, CPU only.
- **Models:** SmolLM2-135M-Instruct in nine formats and Qwen3-0.6B Q8_0, converted from the same GGUF files llama.cpp reads, so weights are identical.
- **Memory:** the tool polls `/proc/<pid>/status` during a real greedy generation and reports peak resident memory and peak *private* (anonymous) memory. Private memory is what the process costs beyond the model file, which the OS can drop from its page cache. llama.cpp is given a 1024-token context because by default it reserves the model's full training context up front; hk grows its KV cache as the context fills.
- **Machine:** AMD Ryzen 7 7445HS (6 cores / 12 threads, 6 MiB L2, 16 MiB L3, AVX-512 with VNNI), 15 GiB RAM, Linux. llama.cpp 0.5.0-dev, build 8216c84, GCC 16.2.1, CPU backend. hk selected the `avx512-vnni` kernels.
- **Caveats:** the run on 2026-10-05 was *not* on an idle machine (a browser was open) and the CPU is a laptop part that throttles. Run-to-run variation of 3 to 8 percent is visible in the ranges; differences smaller than that are not meaningful. These models are small: a 135M model is largely cache-resident and tests per-layer overhead more than DRAM bandwidth, while Qwen3-0.6B is closer to bandwidth bound.

## CPU results (tokens per second)

| Model | Prefill hk | Prefill llama.cpp | hk / llama.cpp | Decode hk | Decode llama.cpp | hk / llama.cpp |
|:---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M-Instruct-Q8_0 | 2633 [2381-2781] | 1958 [1827-1972] | 135% | 145.5 [144.8-149.1] | 147.9 [147.0-152.1] | 98% |
| SmolLM2-135M-Instruct-Q4_0 | 2985 [2863-3087] | 2178 [2125-2211] | 137% | 217.0 [208.5-218.1] | 226.7 [223.0-233.4] | 96% |
| SmolLM2-135M-Instruct-Q4_K_M | 2671 [2548-2721] | 1747 [1661-1763] | 153% | 191.2 [188.1-194.7] | 198.5 [197.1-203.8] | 96% |
| SmolLM2-135M-Instruct-Q5_K_M | 2555 [2544-2619] | 1063 [1059-1083] | 240% | 183.1 [181.4-193.5] | 190.7 [186.6-193.5] | 96% |
| SmolLM2-135M-Instruct-Q6_K | 2400 [2328-2581] | 1971 [1736-2058] | 122% | 151.5 [149.5-158.9] | 152.8 [149.6-161.3] | 99% |
| SmolLM2-135M-Instruct-Q3_K_M | 2458 [2224-2497] | 2176 [2029-2229] | 113% | 209.5 [202.0-217.6] | 222.4 [215.6-227.2] | 94% |
| SmolLM2-135M-Instruct-Q2_K | 2416 [2314-2532] | 2289 [2235-2386] | 106% | 220.0 [218.8-225.6] | 236.7 [226.8-245.2] | 93% |
| SmolLM2-135M-Instruct-IQ4_XS | 2172 [2066-2188] | 2394 [2307-2406] | 91% | 210.6 [202.0-224.3] | 228.5 [215.8-231.9] | 92% |
| SmolLM2-135M-Instruct-f16 | 1821 [1819-1846] | 1828 [1780-1842] | 100% | 82.6 [80.7-84.3] | 82.8 [81.8-84.5] | 100% |
| Qwen3-0.6B-Q8_0 | 774 [725-785] | 490 [484-498] | 158% | 33.5 [33.3-33.8] | 36.0 [35.5-36.5] | 93% |

Memory during a greedy generation (MiB). Private is memory the process owns; the rest of resident memory is the model file mapped from disk. llama.cpp is run with a 1024 token context, because by default it reserves the model's whole training context up front. hk allocates its cache as it fills.

| Model | hk peak resident | hk private | llama.cpp peak resident | llama.cpp private |
|:---|---:|---:|---:|---:|
| SmolLM2-135M-Instruct-Q8_0 | 145 | 8 | 197 | 45 |
| SmolLM2-135M-Instruct-Q4_0 | 100 | 8 | 210 | 109 |
| SmolLM2-135M-Instruct-Q4_K_M | 89 | 8 | 179 | 65 |
| SmolLM2-135M-Instruct-Q5_K_M | 119 | 8 | 177 | 56 |
| SmolLM2-135M-Instruct-Q6_K | 134 | 8 | 203 | 57 |
| SmolLM2-135M-Instruct-Q3_K_M | 104 | 10 | 212 | 109 |
| SmolLM2-135M-Instruct-Q2_K | 82 | 8 | 199 | 102 |
| SmolLM2-135M-Instruct-IQ4_XS | 101 | 10 | 201 | 101 |
| SmolLM2-135M-Instruct-f16 | 263 | 9 | 326 | 54 |
| Qwen3-0.6B-Q8_0 | 644 | 30 | 801 | 182 |

### Reading the CPU results

- **Prefill:** hk is faster than llama.cpp on eight of the ten models (106% to 240%), tied on f16 (100%), and behind on `IQ4_XS` (91%). The advantage comes from the register-tiled GEMM described in [Hardware and Kernels](Hardware-and-Kernels).
- **Decode:** hk is **slightly behind** on nearly every model: 92% to 99% of llama.cpp, tied on f16. Decode streams every weight once per token and is limited by memory bandwidth (about 25 GB/s measured on this machine) plus per-layer synchronization between threads; hk is close to that ceiling but has not matched llama.cpp here. This is the main open performance gap on CPU.

  Why it is hard to close: on this machine a plain multi-threaded read loop reaches about 25 GB/s regardless of thread count (2, 4, 6 or 8 threads all give 24.5 to 25.3 GB/s). For SmolLM2-135M Q4_K_M one decoded token reads about 104 MB of weights plus about 13 MB of KV cache, so the floor is about 4.7 ms per token (about 210 tokens/s). hk takes about 5.2 ms, roughly 90% of that floor, and llama.cpp about 5.0 ms. Decode speed is also flat from 4 to 8 threads, which confirms it is bandwidth bound. What is left is a few percent of overhead (synchronization between layers, tail effects), not arithmetic.
- **Memory:** hk's private memory stays under about 30 MiB on every model here (8 to 10 MiB for SmolLM2, 30 MiB for Qwen3-0.6B) against 45 to 182 MiB for llama.cpp, because weights stay memory mapped and the KV cache grows in segments. Peak resident memory (which includes the mapped file) is lower for hk on every model too. The private-memory gap is a stable result; absolute values depend on context length and thread count.

## GPU results (Vulkan)

Single NVIDIA GeForce RTX 3050 6GB Laptop GPU, Vulkan, same GGUF-derived models, 512-token prompt, 64-token generation. hk: one `hk-probe --bench` run with `HK_GPU=1`; llama.cpp: `llama-bench -ngl 99 -r 3` built with the Vulkan backend (same commit), mean of 3.

| Model | Prefill hk | Prefill llama.cpp | hk / llama.cpp | Decode hk | Decode llama.cpp | hk / llama.cpp |
|:---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M Q8_0 | 4930 | 15035 | 33% | 247 | 375 | 66% |
| SmolLM2-135M Q4_K_M | 5322 | 14714 | 36% | 279 | 428 | 65% |
| SmolLM2-135M f16 | 5927 | 18691 | 32% | 222 | 275 | 81% |
| Qwen3-0.6B Q8_0 | 1528 | 6394 | 24% | 112 | 151 | 74% |

**The GPU backend is substantially slower than llama.cpp's Vulkan backend.** It is correct (logits match the CPU path in the GPU tests) and it runs a whole model on the device, but it is untuned: no cooperative-matrix prefill, no kernel fusion beyond SwiGLU, one device only, no partial offload. Treat it as a working baseline, not a competitive implementation. Only this one GPU has been tested; AMD, Intel and Apple GPUs, and any other Vulkan driver, have not.

## What has not been measured

- ARM (NEON/dotprod) and Apple Silicon performance: the kernels compile and pass the portable tests but have never been run on that hardware.
- AVX2-only, AVX-VNNI-only and generic kernel levels on real older CPUs. They are tested for correctness against the portable definitions (`HK_KERNELS=avx2 zig build test`), and `hk-kernels` can benchmark each level on any machine, but there is no comparison against llama.cpp on such hardware.
- Larger models (7B and up), long contexts, multiple concurrent server requests, and the Python/PyTorch side.
- Perplexity/quality versus llama.cpp is checked on the tokenizer and logits level (see [Compatibility](Compatibility)), not across a benchmark suite.

## Reproduce

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/hk convert-gguf model.gguf model.hk
./zig-out/bin/hk-compare --llama-bin /path/to/llama.cpp/build/bin \
    --pair model.hk:model.gguf --threads 6 --repeats 5
./zig-out/bin/hk-kernels            # per-kernel microbenchmarks, one block per ISA level
HK_GPU=1 ./zig-out/bin/hk-probe model.hk --bench 512 64     # GPU speed
```

Close other programs and let the machine cool first, and compare the ranges, not just the medians.
