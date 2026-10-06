# HK — Context (start here)

> Read this file first, in any session, with any agent. It is the stable map.
> **Volatile state** (what is half-done right now, what to do next) lives in
> [docs/plan/HANDOFF.md](docs/plan/HANDOFF.md). **Plan vs. reality** lives in
> [docs/plan/STATUS.md](docs/plan/STATUS.md). The **target** is
> [docs/plan/MASTER_PLAN.md](docs/plan/MASTER_PLAN.md). The **queryable graph** is
> [docs/graph/](docs/graph/README.md).

## 1. What HK is

A local LLM runtime and universal model container written in **Zig 0.16**. One binary pulls a model from
Hugging Face, converts it to `.hk` while downloading (bounded memory), and runs it (CLI, chat, OpenAI-compatible
server). Weights are memory-mapped and never copied. The owner's stated reason for existing: *memory efficiency
done seriously* — "the most efficient option out there".

The long-term plan (160 sections) aims at a **universal neural platform**: four systems on one foundation —
**Container** (`.hk` format), **Runtime** (inference everywhere), **Research** (quantization, sparsity, growth,
self-training, lineage), **Tooling** (convert, serve, SDKs, profile, package). Everything in the plan is in
scope; it is staged by the dependency order in plan §151.

Repo: `github.com/harshitkhandelwal208/hk` · Apache-2.0 · packages: PyPI `hknt`, npm `hkntf`.

## 2. Where things stand (verified 2026-10-06)

| | |
|:--|:--|
| Version | 1.1.1 (tag `v1.1.1`); `HEAD` = `43cf891` on `main`. **Uncommitted work in tree** — see HANDOFF |
| Engine runs | Llama / Mistral (no sliding window) / Qwen2 / Qwen3 dense decoders — **3 architecture families** |
| Formats | 26 weight storage formats decoded; CPU kernel for every one; Vulkan decodes a subset |
| CPU perf | prefill 91–240 % of llama.cpp (faster on 8/10 models), decode 92–100 %, private RSS 5–9× lower |
| GPU | Vulkan whole-model-on-one-device; one NVIDIA card tested; slower than llama.cpp's Vulkan |
| Not present | CUDA in the engine, Metal, ROCm, NPU, MoE, SWA, multimodal, grammar, speculative decoding, distributed, sparse execution, signatures, `hk quantize` |
| Research half | Python prototypes (Net2Net growth, lineage, 2:4 storage, LoRA/QLoRA, SPIN, self-training) — unit-tested on small models only |
| Tests | `zig build test`: 138/138 pass (Debug, this machine). Python suite not run in the last session |

Do **not** quote "137+ architectures" as support. It is a Python name-mapping registry (169 aliases → 79
canonical names). The engine executes 3 families. Details and every other gap: STATUS.md.

## 3. Repository map

```
src/                    Zig core (library + C ABI)
  format.zig reader.zig writer.zig metadata.zig tensor_toc.zig   .hk container
  appendix.zig                                                   in-file version lineage (hash chain)
  integrity.zig transaction.zig                                  SHA-256 seal + metadata journal (NEW, uncommitted)
  platform.zig                                                   mmap / file mapping / in-place IO
  engine/        model, config(Arch enum), weights, kv, attention, ops, matmul, session   ← live inference path
  quant/         per-format dequant, vecdot, gemm, unpack, isa                           ← live kernels
  kernels/ + kernels.zig + kernels_impl.zig                      per-ISA-level build + runtime dispatch
  vk/ + shaders/ + vk/spv/                                       Vulkan backend (GLSL → checked-in SPIR-V)
  server/        api.zig (HTTP/OpenAI JSON/SSE), scheduler.zig (slots)
  hub/           HTTP, HF API, pull (resume, SHA-256)
  convert/       streaming GGUF / safetensors / HF → .hk; .hk → safetensors
  tokenizer/ tokenizer.zig chat/    BPE + SentencePiece; Jinja subset for chat templates
  sampler.zig                                                    temp/top-k/top-p/min-p/penalties
  tensor_ops.zig quantization.zig c_api.zig                      LEGACY stack behind C ABI / Python / bindings
  cuda.zig cuda/                                                 standalone CUDA (NOT in engine)
  growth.zig adaptive.zig expansion.zig sparsity.zig nf4.zig sandbox.zig   research-side Zig
tools/          hk_cli.zig (the `hk` binary), hk_compare.zig, hk_probe.zig, hk_gpu_bench.zig, GUI, generators
tests/          Zig suites (roundtrip, quant, engine, gpu, cuda, cli/server/hub e2e) + pytest + bindings/run.sh
python/hk/      Python SDK: modeling, torch/native bridges, quantization, pruning, trainer, adaptive/ (research)
bindings/       csharp go java js rust (over include/hk.h)         include/  hk.h hk.hpp
benchmarks/     results.md (reproduce with hk-compare)             docs/wiki/  user docs (auto-synced to GitHub wiki)
docs/plan/      MASTER_PLAN, STATUS, HANDOFF                       docs/graph/ knowledge graph (+ generator)
```

**Two kernel stacks** (most common source of wasted effort): the live engine uses `engine → quant + kernels → vk`.
`tensor_ops.zig` + `quantization.zig` (where the v1.1.0 "gemmF32 zero-skip / RoPE table / FP8 LUT"
optimizations live) are reached only through `c_api.zig`, Python, and the converters. `cuda.zig` is wired into
neither. Check which stack a change belongs to before optimizing anything.

## 4. Commands

```bash
zig build -Doptimize=ReleaseFast            # hk, libhk, hk-probe, hk-compare, hk-kernels, hk-tiny-model, ...
zig build test                              # unit tests (Debug => portable kernel level only; ~5 s)
zig build test -Doptimize=ReleaseFast       # every ISA level this CPU supports
zig build test -Doptimize=ReleaseFast -Dcpu=x86_64_v3   # AVX2 code in the library; also -Dcpu=x86_64
zig build test-e2e -Doptimize=ReleaseFast   # CLI + server + hub tests against the built binary
zig build test-cuda                         # standalone CUDA tests (needs a driver)
zig build bench                             # kernel microbenchmarks
bash tests/bindings/run.sh                  # C ABI + C++/Rust/Go/C#/Java/TS bindings against a fixture
pytest tests                                # Python (needs torch, numpy, safetensors, gguf installed)
HK_KERNELS=avx2 hk run model.hk "hi"        # force a kernel level;  HK_GPU=1 or -ngl N for Vulkan
./zig-out/bin/hk-compare --llama-bin <dir> --pair m.hk:m.gguf --threads 6 --repeats 5   # vs llama.cpp
python3 docs/graph/gen_graph.py --check     # refresh/verify the knowledge graph
```

Zig 0.16 specifics that bite: I/O goes through `std.Io` (`std.Options.debug_io`, `std.process.Init` for `main`),
`std.Io.Dir.cwd().createFile(io, path, …)`; `build.zig` fans the kernel library out once per ISA level
(`src/kernels/variants.zig`); never rely on the build machine's CPU features.

## 5. Standing rules (owner-set; apply to every agent)

1. **Memory efficiency is the product.** Stream, mmap, bounded buffers; never hold a whole model/tensor set in
   RAM; no intermediate files or extra copies. Measure peak RSS (`/usr/bin/time -v`) and report it. Flag any
   O(model) path instead of hiding it. (`HKWriter` still holds tensors in memory; converters use the streaming
   writer.) *Watch:* the new container seal re-hashes the **whole file** on every mutation — bounded RAM, O(file) time.
2. **Robustness means no silent failure.** A value is right or flagged/blank — never silently wrong. Test failure
   paths, not just success. A guard's test must fail when the guard is removed (check by temporarily disabling it).
   State limits plainly ("cannot guarantee X") instead of claiming zero failures.
3. **Docs say only what is true.** Every claim in README/wiki has a number or a test, or says it doesn't. The
   1.1.1 rewrite removed overstated claims (2:4 speedups, "secure sandbox", unmeasured tables); do not reintroduce them.
4. **Plan hierarchy (§158):** model quality > correctness > memory/IO > compute > acceleration > distributed >
   research. Research code must never compromise runtime correctness.
5. **No "faster" without a benchmark (§1.5):** baseline, same hardware/model/workload, repeats + variance, memory,
   correctness check. Never compare incompatible configurations.
6. **Optimization order (§152):** correctness → memory traffic → layout → vectorization → tiling → fusion → sync →
   launch overhead → accelerator-specific → autotune.
7. **Untrusted input:** model files, tokenizers, templates, HTTP, downloads are hostile. Bounds-check with
   overflow-safe arithmetic; validate before turning bytes into enums or slices; name the offending tensor in errors.
8. **Say what was and wasn't run.** Cross-compiled ≠ tested. One GPU ≠ GPUs. Python suite not run ≠ passing.
9. **Don't commit or push unless asked.** When asked, follow the repo's commit style and the attribution lines
   the harness provides.

## 6. Plan, compressed

Dependency order (§151): `FORMAT → MMAP → QUANT CORE → CPU KERNELS → GRAPH/MODEL ABSTRACTION → BACKEND ABSTRACTION
→ CUDA (prefill, decode, attention, fusion/graphs) → MEMORY PLANNER → PARTIAL OFFLOAD → VULKAN OPT → METAL → ROCm
→ NPU → MOBILE → DISTRIBUTED → ARCH EXPANSION → SERVER FEATURES → ADV. QUANT → SPARSITY EXEC → TRAINING →
ADAPTIVE GROWTH → LINEAGE → SELF-TRAINING → SELF-PLAY → ARCH SEARCH`.

We are at: FORMAT hardening (integrity + crash safety, in flight) with CPU kernels and a Vulkan backend already
standing; the graph/model abstraction and backend abstraction (the gate to CUDA/Metal/ROCm) do not exist.
Kernel priority (§153): GEMV, GEMM, attention, RMSNorm, RoPE, SwiGLU, KV ops, embedding, LM head, sampling.
Memory priority (§154): mapped weights, quantized execution, KV segmentation, buffer reuse, residency, partial
offload, KV compression, prefix sharing, sharding, distributed.

Milestones: 2.0 Unified Runtime · 2.1 HW Acceleration · 2.2 Model Compatibility · 2.3 Server Runtime ·
2.4 Universal Format · 2.5 Training · 2.6 Adaptive Research · 2.7 Self-Improvement · 3.0 Universal Runtime
(per-item truth in STATUS.md).

## 7. Glossary

| Term | Meaning |
|:--|:--|
| `.hk` | Container: 128 B header, metadata KV, tensor TOC, aligned payloads, optional append-only appendix |
| Appendix | Records after tensor data (LoRA, deltas, topology, code-eval…) chained by parent SHA-256 → in-file version history |
| Seal | `integrity.sealFile`: SHA-256 over the whole file (digest fields zeroed), header flag bit 9 |
| Journal | `<file>.hkmeta-journal`: snapshot of the metadata prefix taken before an in-place edit |
| ISA level | `avx512` / `avx2vnni` / `avx2` / `generic` (x86); `dotprod` / `neon` (ARM) — chosen at startup |
| Prefill / decode | batch prompt processing (compute-bound) / token-by-token generation (bandwidth-bound) |
| `-ngl N` | offload to GPU (Vulkan): whole model or CPU; no partial offload yet |
| Super-coalescing | page-aligned payload layout idea (`UNIVERSAL_PAGE_ALIGNED`); a layout convenience, not an accelerator claim |
| Dual-mode quant | base quantized tensor + optional residual (`dq*`, NF4); storage + Python only today |
| Net2Net | function-preserving widen/deepen (`growth.zig`, `adaptive/growth.py`) |
| SPIN | self-play fine-tuning loss (`adaptive/self_play.py`) |
| Slot | one concurrent conversation in `hk serve --slots N` |

## 8. Session protocol

**Start:** read this file → `docs/plan/HANDOFF.md` (top "Now" section + last log entry) → `git status` and
`git log -5` → run `zig build test` for a baseline → confirm the HANDOFF "in flight" list still matches the tree.
For a specific area, open the matching rows in STATUS.md and the matching node in `docs/graph/graph.json`.

**End:** update HANDOFF.md (what changed, what's verified, what's next, open questions); update the STATUS.md rows
you touched; run `python3 docs/graph/gen_graph.py` if you added/moved/removed files or changed status; if you
found a doc/code discrepancy, fix it or list it in STATUS "Known discrepancies".

## 9. Context system index

| File | Purpose | Changes |
|:--|:--|:--|
| `CONTEXT.md` | stable orientation (this file) | rarely |
| `AGENTS.md` / `CLAUDE.md` | agent working agreement (CLAUDE.md imports AGENTS.md) | rarely |
| `docs/plan/MASTER_PLAN.md` | the owner's 160-section target, verbatim | owner only |
| `docs/plan/STATUS.md` | plan § → code reality matrix with evidence | when status changes |
| `docs/plan/HANDOFF.md` | in-flight work, next steps, open decisions, session log | every session |
| `docs/graph/curated.json` | hand-maintained concept/plan/status nodes + edges | when architecture/status changes |
| `docs/graph/gen_graph.py` | builds `graph.json` + `GRAPH.md` from code imports + curated | tool |
| `docs/graph/graph.json`, `GRAPH.md` | generated; do not hand-edit | regenerate |
