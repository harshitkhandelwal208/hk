# Benchmarks

These numbers come from `hk-compare` (`tools/hk_compare.zig`), which runs hk and llama.cpp on the same models, on the same machine, alternating between the two so a slow moment hits both. Reproduce with:

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/hk-compare --llama-bin /path/to/llama.cpp/build/bin \
    --pair model.hk:model.gguf --threads 6 --repeats 5 --out benchmarks/results.md
```

Each `.hk` was made from the same GGUF with `hk convert-gguf`, so the weights are identical. Speeds are tokens per second.

## Latest run

Conditions: llama.cpp 0.5.0-dev (build 1, commit 8216c84, GCC 16.2.1, CPU backend), hk built with `-Doptimize=ReleaseFast`, kernel level `avx512-vnni`. The machine was not idle: a web browser was open (about 20% of one core in `ps`), and the CPU is a laptop part that throttles, so single runs vary by several percent; use the bracketed ranges. 2026-10-05.

Machine: AMD Ryzen 7 7445HS w/ Radeon 740M Graphics, 6 threads, CPU only. Kernel level: see hk-probe.
Prompt 512 tokens, generate 64; median of 5 alternating runs of each tool, range in brackets.

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
