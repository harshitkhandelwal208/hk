# HK — Handoff

Volatile, session-to-session state. **Newest information goes at the top of "Now"; the log at the bottom is
append-only.** Stable orientation: [CONTEXT.md](CONTEXT.md). Plan vs. reality: [STATUS.md](STATUS.md).

Rules for editing this file: state what was *run* vs. *only read*; never write "done" for something not
verified; when a "Now" item is finished, move one line of it to the log and delete it from "Now".

---

## Now

### 00. HK 2.1 (hardware acceleration) in progress — branch `feat/hk-2.1-rocm`, PR open

**Done this session, with what was run:**
- `src/gpu_prims.zig`: the 17 inline-PTX sites of `src/cuda_device.zig` moved behind per-architecture primitives (NVIDIA PTX unchanged; AMD via LLVM `amdgcn` intrinsics). The NVIDIA module still exports the same 73 (sm_70) / 109 (sm_80) kernels; `zig build test` and `test-cuda` pass on the RTX 3050.
- `build.zig` (`-Drocm`, default on for Linux/Windows) compiles the device code to a `gfx11-generic` HSA code object (RDNA3/3.5, wave32) and embeds it. Checked: 73 kernels, valid `EM_AMDGPU` ELF, disassembly has `ds_bpermute_b32`, `v_dot4_i32_iu8`, `s_barrier`. `gfx1030` compiles too (no hardware dot4, slower); GCN/CDNA (gfx9) are wave64 and refused.
- `src/hip.zig` (HIP runtime via dlopen, same surface as `cuda.zig`), `src/gpu/engine.zig` (the old CUDA engine, now `Make(runtime)`), `src/cuda/engine.zig` and `src/rocm/engine.zig` instantiate it. `backend.Device.rocm`, `probeRocm`, `--backend rocm`, caps and planner cases. `auto` order: CUDA (if tensor cores), Vulkan, CUDA, ROCm.
- Run: unit tests 197 passed + 7 skipped (the ROCm parity tests, no AMD device) at Debug, ReleaseFast, `-Dcpu=x86_64_v3`, `-Dcpu=x86_64`; e2e 26/26; `test-cuda` passes. Mutation-checked: letting `--backend rocm` fall back quietly turns a unit test and the e2e test red.
- **Never run on AMD hardware.** The machine has a Radeon 740M iGPU (HawkPoint2, gfx1103) but no `vulkan-radeon` (RADV) and no ROCm userspace, `/dev/kfd` exists, the user is not in the `render` group.

**Next concrete commands (need the owner, they use sudo):**
```
sudo pacman -S --needed vulkan-radeon hip-runtime-amd hsa-rocr    # RADV for the Vulkan backend on AMD; HIP runtime for ROCm
sudo usermod -aG render harshit                                   # then log out and in
HSA_OVERRIDE_GFX_VERSION=11.0.0 zig17 build test -Doptimize=ReleaseFast   # runs the ROCm parity tests; gfx1103 is not an official ROCm target
```
Then: fix whatever the first real run shows, benchmark against llama.cpp (HIP and Vulkan on the iGPU), write WMMA (`v_wmma_f16_16x16x16_f16`) kernels for prompts, build `gfx10_3_generic` and `gfx12_generic` objects, add HIP graph capture. Update STATUS row 58 and the wiki from "implemented, not run" only with what actually ran.

**Not started in 2.1:** 2:4 sparse tensor cores (owner: later), Metal (needs Apple hardware), Intel XPU / DirectML (Vulkan covers Intel and Windows GPUs; a D3D12 backend is a separate project), Android (Zig cannot provide bionic; needs the NDK, not installed; `build.zig` has an untested no-`libdl` tweak). Open decision: whether to treat Intel/Windows GPUs as "covered by Vulkan" in the plan.

**Zero dependencies (separate PR #10, `feat/zero-dep-python`):** wheels per platform with `hk` bundled, `torch`/`numpy` optional, plan section 1.6 and 161 (`hk.tensor`). The release workflow's new wheel matrix was not run (needs a tag).

### 0. HK 2.0.0 is released (2026-10-21)

PR #5 (`feat/hk-2.0-backends`) was merged into `main` by the owner (ruleset bypass; merge commit `030ba32`, tree identical to the tested `16f765e`). Tag `v2.0.0`
points at `030ba32`. The release workflow (run `39485175463`) finished green: native binaries for x86_64/aarch64 Linux, aarch64 macOS and x86_64 Windows, the GitHub
Release, PyPI `hknt` 2.0.0 (confirmed with `pip index versions hknt`), npm `hkntf`, crates.io `hknt`, NuGet and GitHub Packages. Not checked by installing each package.
CI on the PR head `16f765e` was green on all jobs; CI on `main` after the merge: see the Actions tab.

`main` requires signed commits and PRs. The owner authorised signed commits on their behalf (2026-10-21): an SSH signing key (`~/.ssh/github_signing_ed25519`, no passphrase,
registered on their GitHub account) signs every commit and tag in this clone (`gpg.format=ssh`, `commit.gpgsign`); GitHub shows them as verified. The owner also allowed `--admin` for PR #6 only;
ask before bypassing the ruleset again.

PR #6 (merged 2026-10-21, `cf4ca26`, docs only, no new tag, its CI had not finished when it was merged) cleaned the repository root (removed `PERFORMANCE_REPORT_KERNELS.md`, the four `run_*_demo.py` scripts,
`install.sh`/`install.ps1` (pinned to v1.0.0 and to asset names the release does not publish), `models/.gitkeep`; `CONTEXT.md` moved to `docs/plan/`), committed `graphify-out/`
(graph, report, html, manifest; cache ignored), and rewrote README/CONTRIBUTING. The GitHub About blurb and topics were updated with `gh repo edit` (no more "137+ architecture support").
Rebuild the graphs after code or doc moves: `graphify update .` and `python3 docs/graph/gen_graph.py`.

**Next: 2.1 hardware acceleration** (branch `feat/hk-2.1-hardware`, off `16f765e`, no commits yet). Candidates: ROCm/HIP (Zig can emit amdgcn code objects: `zig build-lib -dynamic
-target amdgcn-amdhsa -mcpu gfx1100` gives a valid object; `cuda_device.zig` primitives need an arch abstraction and wave-size handling), Metal, Intel XPU/DirectML (probably documented as
covered by Vulkan), Android Vulkan; none can be run on this machine (RTX 3050 only), so they ship as "implemented, not run" per the owner. Known performance gaps: Vulkan K-quant prefill and decode,
CUDA K-quant prefill (~75 %), CPU K-quant prefill. Add a Zig cache to CI (the cross-compile step takes 25 to 40 minutes cold). 2:4 sparse tensor cores are later work.

### Owner direction (2026-10-20)

Do **not** ship a "3.0.0". Implement the whole of `MASTER_PLAN.md` and release **in the plan's order**: 2.0 → 2.1 → … → 2.7 → 3.0
(§150), built along the dependency chain in §151. 3.0 is the *last* milestone. Also: **keep as much as possible in Zig**
(Python stays a thin layer over the C ABI; no second implementation of anything Zig already owns).

### A. DONE, COMMITTED (`a1ee4eb`, branch `feat/container-integrity`): container integrity + crash-safe edits (plan §4, §5; milestone 2.4 items, landed early)

Details and numbers: CHANGELOG `[Unreleased]`, STATUS rows 4–5, wiki `Format-Specification` "Integrity and crash safety".
Committed on top of `054964b`; not pushed, not tagged.

**Run (2026-10-20):** `zig build test` 148/148 at Debug, ReleaseFast, `-Dcpu=x86_64_v3`, `-Dcpu=x86_64`; `zig build test-e2e -Doptimize=ReleaseFast`
24/24; `bash tests/bindings/run.sh` 8 suites 0 failed; `pytest tests` 97 passed / 3 skipped (missing `openai`, `jax`, `tkinter`) in a scratch
venv at `/tmp/claude-1000/.../scratchpad/venv` (CPU torch) — recreate it, it is not in the repo. Mutation-checked: 9 Zig guards (G1–G9) and 4 CLI guards (C1–C4) each turn
red when disabled. Measured on a 2.1 GiB container: `hk verify` 1.27 s / 13 MiB RSS, `hk metadata set` 2.50 s / 13 MiB RSS (warm cache, one machine).

**Fixed while finishing (were silent failures):** converters never sealed; sealing could read the whole file into RAM when `mmap` failed;
an edit of an already-corrupt sealed file re-sealed it; `sealFile` wrote the header twice (crash window); appendix append/rollback had no
journal; a torn journal blocked all edits; `hk verify` / `hk metadata set` / `hk rollback` exited 0 on failure; `hk verify` hashed through
the mapping (2 GiB RSS); `hk_appendix_append` cast an untrusted byte to an enum; Python `metadata_set` fell back to a patcher that left a stale seal
and could corrupt an appendix, and Python `append_record`/`rollback_appendix` read the whole file and rewrote it non-atomically.
Python now goes through the C ABI only and raises without the native library.

**Decision I made (revisit if you disagree):** Open decision "seal cost" resolved as *keep whole-file SHA-256, streamed, verify-before-edit*.
Rationale: correctness first, bounded RAM, measured ~1.65 GiB/s. Sectioned/Merkle hashing would make edits O(changed) but needs a format change — fits the
2.4 "v2 format" item. Edits on a 70 GB model will take minutes; that is the known cost.

**Not done / limits:** no Windows/macOS/ARM run (`std.Io` replaced the Windows-specific writer in sealing, unrun there); crash tests simulate the
post-crash state by hand, no kill-9/power-loss harness; `hk_verify_integrity` is wrapped in C++ and Python only (Go/Rust/C#/Java/TS lack it);
no `hk compact`/`hk fsck`; no signatures (an attacker can recompute the seal); `hk run` does not verify the seal at load (O(file) cost) —
decide whether `hk pull` should.

### A2. Toolchain moved to Zig 0.17 (branch `feat/hk-2.0-backends`)

Owner asked for 0.17. `zig` on PATH is still 0.16 (pacman); 0.17 is `~/.local/bin/zig17`. Repo builds and passes on 0.17: unit 148/148 (Debug, ReleaseFast, v3, baseline),
e2e 24/24 (now never cached), bindings 8/8 (needs a `zig`→0.17 shim on PATH), pytest 97/3. CI action pin changed to 0.17.0 but **CI was not run**.
**Stage 0 spike result: GO** — a Zig-authored Q8_0 GEMV (shared memory, `shfl.sync`, `bar.sync`) compiled with
`zig17 build-obj -target nvptx64-cuda -mcpu sm_86 -O ReleaseFast k.zig -fno-emit-bin -femit-asm=k.ptx`, loaded with `cuModuleLoadData`, ran on the RTX 3050
and matched a CPU reference (rel err 1.25e-7). Shared memory must be file-scope `var x: [N]T addrspace(.shared)`. Spike files were in the scratchpad (not kept).

### B. HK 2.0 — Unified Runtime: **2.0.0 (cut 2026-10-21 from PR #5 `feat/hk-2.0-backends`)**

Plan: `~/.claude/plans/snug-foraging-quasar.md` (approved 2026-10-20). Checklist (§150): backend abstraction · CUDA engine · CUDA decode · CUDA prefill · CUDA attention · Vulkan under common API · CPU backend finalized · unified memory planner · automatic device selection.

| Stage | State | Evidence |
|:--|:--|:--|
| 0 Zig-authored CUDA kernels feasible? | **GO** | Zig 0.17 nvptx64: Q8_0 GEMV with shared memory + `shfl.sync` + `bar.sync` ran on the RTX 3050, rel err 1.25e-7 vs CPU |
| Toolchain → Zig 0.17 | done | `1f040ee`; unit/e2e/bindings/pytest green; CI action pin bumped, CI not run |
| 1a `backend.Device` seam | done | `2a07119`; Vulkan under it |
| 1b CPU executor extracted (`engine/cpu_backend.zig`) | done | `01d80b0`; logits byte-identical old vs new in 70/70 comparisons; **speed not re-measured** |
| 1 capability table (`backend/caps.zig`) | done (formats only) | per-operator table (§136) not done |
| 2 planner + `--backend` | done | `fb6c2c1`; 159/159 unit ×4 configs, e2e 26/26, guards mutation-checked |
| 3 CUDA engine | **done (first version)** | `src/cuda/engine.zig` + `src/cuda_device.zig`; parity vs CPU for llama/qwen2/qwen3 × f32/Q8_0, 26 GEMV/GEMM formats, multi-slot, long context; 20+ mutations red |
| 4 CUDA prefill | **done** | tensor-core `mma.sync` GEMM (`gemm_tc_<fmt>`, all 26 formats) + flash attention (`attention_tc_64/128`); Qwen3-0.6B Q8_0 prefill 350 -> 4,005 tok/s |
| 4c Vulkan prefill | **done** | cooperative-matrix GEMM (`mm_cm.comp`) + attention (`attention_cm.comp`); 1,487 -> ~4,700 tok/s |
| 4d Partial offload (planner v1) | **done** | `-ngl N` = first N layers; planner picks the largest fitting prefix; hybrid parity tests on both devices |
| 4b CUDA decode | **done** | warp-per-row GEMV + split-KV attention; decode 360 tok/s vs Vulkan 406 |
| 5 CUDA attention | **done** | online-softmax tiles, warp-cooperative key reads, split-KV for decode |
| 6 CUDA fusion + graphs | **evaluated, not done** | host launch cost measured ~0.4 ms/pass of ~7 ms; GEMV was 72 % of decode, so fixing memory traffic came first (§152). Revisit after Tensor Cores/integer dot products |
| 7 Vulkan under common API | partly (1a); parity harness not shared yet | |
| 8 CPU finalized + kernel-stack unification (owner decision 2) | **done** | `quantization.zig` and `tensor_ops.zig` deleted; encoders → `quant/encode.zig`, research formats → `quant/research.zig`, C ABI products → `engine/abi_ops.zig`. Found and fixed: private Q4_K layout, broken Q3_K encoder, silent block skipping in `dequantizeToF32`. The "legacy optimizations worth porting" turned out to be nothing: the legacy code was single-threaded (its thread pool ran inline) and slower |
| 9 Release gate: §148 compatibility contract | **done; 2.0.0 released 2026-10-21** | `docs/wiki/Compatibility.md` "Release contract"; `hk_abi_version()` = 2; tag `v2.0.0` |

**2.0 readiness (honest assessment, 2026-10-20 evening; branch `feat/hk-2.0-backends`, pushed, draft PR #5):** every §150 item exists and is verified on this machine only (Ryzen 7 7445HS, RTX 3050 6 GB laptop, Arch, driver 615.78). Since the first assessment: Vulkan prompt matmul and attention run on cooperative matrices, CUDA has tensor-core GEMM and flash attention (Zig inline PTX, sm_80 build of the device code), Vulkan has split-KV decode attention, partial offload of leading layers exists on both devices with a planner that picks the largest prefix that fits, and perplexity agrees with llama.cpp and Transformers (`benchmarks/results/2026-10-20-ppl-*.json`: Qwen3-0.6B Q8_0 wikitext-2, 20 windows of 512: llama.cpp 23.67-23.68, hk 23.64-23.68, Transformers f32 23.73). Real-model prefill/decode (tok/s, Qwen3-0.6B Q8_0, 512/128): hk-vulkan ~4,700/~117, hk-cuda ~4,000/~108 (llama.cpp Vulkan measured earlier: 7,032/145.6; llama.cpp CUDA baseline needs a CUDA-capable host compiler, see below). Remaining gaps before calling it 2.0: decode is 20-25 % behind llama.cpp on both devices (CUDA GEMV reaches ~58 % of memory bandwidth; no `dp4a`); prefill still ~1.5x behind llama.cpp Vulkan; automatic selection is a fixed order (Vulkan, then CUDA), not measured; no fusion/graphs; nothing run on a second machine, ARM, macOS or a non-NVIDIA GPU; CI on PR #5 is the first run against Zig 0.17 (Windows/cross-compile failed on an unguarded `std.c.getenv`, fixed in `2fa7da8`; macOS fails `server_tests: an oversized body is refused`, which also fails on `main`, and the test now prints the client error so the next run says why). The CPU executor is a peer struct rather than a union variant of `Device`.

**Benchmark environment (user-level, nothing installed system-wide):** `~/hk-bench/` holds llama.cpp builds (`build` Vulkan, `build-cpu`, `build-cuda` pending), models (`models/`), wikitext-2 (`data/`), CUDA 13.0.2 toolkit assembled from the runfile (`cuda/`; the installer needs libxml2.so.2 which Arch lacks, so it was unpacked with `--noexec` and the nvcc/cudart/cublas components copied), and a GCC 15.2 build in progress (`build_gcc.sh`; nvcc 13.0 rejects the system GCC 16). Harness: `benchmarks/suite.py` (speed, RSS, VRAM; interleaved repeats) and `benchmarks/perplexity.py` (accuracy).

**Release-cut steps (done for 2.0.0, repeat for each milestone):** (1) version in `build.zig.zon`, `pyproject.toml`, `python/hk/__init__.py`, `bindings/{js/package.json + package-lock.json, rust/Cargo.toml + Cargo.lock, java/pom.xml, csharp/Hk.csproj}`, `CITATION.cff`, the banner in `tools/hk_cli.zig`; (2) `## [Unreleased]` → `## [x.y.z] - date` in `CHANGELOG.md` with an upgrade note, a fresh empty `[Unreleased]` above; (3) `SECURITY.md`, `README.md`, `docs/plan/CONTEXT.md`, `STATUS.md`; (4) the whole matrix (Debug, ReleaseFast, `-Dcpu=x86_64_v3`, `-Dcpu=x86_64`, e2e, `tests/bindings/run.sh`, pytest, `gen_graph.py --check`) plus CI on all three OSes; (5) merge the PR into `main`; (6) tag `vX.Y.Z` — the release workflow then builds the native libraries, wheels and packages and publishes PyPI `hknt`, npm `hkntf`, crates.io `hknt`, NuGet and GitHub Packages (not undoable: a published version cannot be replaced); (7) wiki sync (automatic on `docs/wiki` changes).

**Bugs of mine found by the process (for the record):** an edit script that died halfway and left `engine.zig` half-edited while I reported it done (assert every anchor); a benchmark that stopped the prefill clock before the async GPU finished, which inflated prefill and deflated decode (end prefill with the first logits); `Engine.init` leaving defaulted fields `undefined`, visible only in Debug (keep Debug in the matrix); equivalent mutants ("every split covers the whole context", "never split") that look like test gaps but are not.

**Gotchas learned this session:** shell `ls` alias is broken (use `find`); `/usr/bin/time` is not installed (use Python `resource`); zsh does not word-split `$var` (use `${=var}`); `pgrep -f X` inside a waiting loop matches the loop itself — poll a log file instead; Zig caches test *runs* (e2e is now `has_side_effects`); no real model on disk, so no speed claims; tiny models via `hk-tiny-model [--q8]`; parity harness lives in the scratchpad (`parity.py`, old-vs-new `hk-probe` over ISA levels × threads × prompts) — recreate from this description if needed: build the previous commit in a `git worktree`, run both `hk-probe`s with `HK_KERNELS`/`HK_THREADS`, compare raw f32 output bytes.

### Open decisions (owner)

1. ~~Seal cost model~~ — resolved above (revisitable).
2. ~~**Which kernel stack is canonical?**~~ **DECIDED 2026-10-20: the engine stack.** Port the legacy optimizations worth keeping into `src/engine` + `src/quant` + `src/kernels`, then route the C API/Python through the engine kernels and retire `tensor_ops.zig`/`quantization.zig`. (Original question follows.) Plan §9/§15/§17/§20 say "retain the earlier optimizations", but those live in the legacy stack
   (`tensor_ops.zig`), not the engine. Options: port them into the engine stack; or retire the legacy stack and route C API/Python through the engine kernels.
   *Blocks the backend interface design.*
3. ~~**CUDA path**~~ **DECIDED 2026-10-20: adopt `src/cuda.zig` as the base** (validate it first — it is unmeasured and wired into nothing). (Original question follows.) Adopt `src/cuda.zig` as the base for an engine backend, or restart from the Vulkan engine's structure? (Old commit messages
   disagree on how well the old CUDA code worked; nothing re-measured.) *Blocks CUDA engine.*
4. **Sandbox policy (§33):** `CodeSandbox` silently falls back from Docker to an unsandboxed subprocess. Make it fail closed? Likely yes; changes behavior.
5. ~~Version number~~ — superseded: no version is assigned until a milestone is complete; the CLI banner still says 1.2.0 while packages say 1.1.1 (fix when 2.0 is cut).
6. **`build.zig` `-Dcuda` default-on** for non-macOS builds although CUDA is off the engine path.
7. ~~Commit the integrity work?~~ **DONE 2026-10-20:** commit `a1ee4eb` on branch `feat/container-integrity` (off `docs/context-system`). No tag, no push, no release.

### Verified facts worth not re-deriving

- Engine architectures: `llama` (+`mistral` alias), `qwen2`, `qwen3` (`src/engine/config.zig`).
- ISA levels per arch: `src/kernels/variants.zig`; override with `HK_KERNELS`.
- Server routes: `/health`, `/v1/health`, `/v1/models`, `/models`, `/metrics`, `/v1/chat/completions`, `/v1/completions` (+ unprefixed aliases).
- `hk-cuda-tests` passes here in ~2 s; unknown whether it hit the GPU.
- Local hardware: AMD Ryzen 7 7445HS (6C), NVIDIA RTX 3050 6GB Laptop (driver present), Arch Linux, Zig 0.16.0 at `/usr/bin/zig`.
- Shell quirk: `ls` is aliased to a tool that rejects directory args as `--icons` values — use `find`/`git ls-files`.

---

## Log (append-only, newest last)

### 2026-10-06 — context system created
- Created `CONTEXT.md`, `AGENTS.md`, `CLAUDE.md`, `docs/plan/{MASTER_PLAN,STATUS,HANDOFF}.md`, `docs/graph/*`.
- Stored the owner's 160-section plan verbatim as `MASTER_PLAN.md`.
- Surveyed repo at `43cf891` + dirty tree; ran `zig build test` (138/138). Did **not** run Python tests, e2e tests, or GPU benchmarks.
- Findings recorded in STATUS: two kernel stacks; "137+ architectures" is a Python alias registry (79 canonical, 169 aliases) while the engine runs 3 families;
  `NativeSandbox` does not enforce its timeout and is unused; `CodeSandbox` silently degrades Docker→subprocess; CLI/packaging version mismatch; stale `SECURITY.md`.
- Graph: `docs/graph/gen_graph.py --check` passes (79 curated nodes, 167 files, 368 import edges, 0 unowned files) and was mutation-tested (5 breakages caught).
  Mermaid diagrams: rendered with mermaid-cli 12.0.0 and viewed. First render showed an unreadable hairball and a 4,389px strip, so the generator now
  splits the dependency diagram into 2a (built) / 2b (planned), lays systems out side by side, and shows only coupling >= 2 imports. All 5 render.
  Still not run: the `jq` recipes (no jq installed) — equivalent Python queries were.
- Added a pointer memory (`hk-context-system`) to the owner's Claude memory index.
- No source code was modified. Nothing committed. The pre-existing uncommitted integrity/transaction work (item A) is untouched.

### 2026-10-20 — container integrity finished; graphify added; direction reset
- Finished item A (see Now). Found and fixed the silent failures listed there; added 9+5 tests, mutation-checked 13 guards, measured cost/RSS.
- Owner redirected: no 3.0.0 now; follow the §150 milestone order and the §151 chain; keep as much as possible in Zig. Removed the Python seal/journal mirror I had started writing and routed Python through the C ABI instead.
- Installed **graphify** (`pipx install graphifyy` 0.9.92; command `graphify`). Built the code graph with `graphify extract . --code-only` (local AST, no API key): 4,848 nodes, 9,617 edges, Zig/Python/C#/Rust/TS/Java included. `graphify claude install` appended a "graphify" section to `CLAUDE.md` and wrote PreToolUse hooks to `.claude/settings.json` (machine-specific absolute path — excluded via `.git/info/exclude`; re-run `graphify claude install` on another machine). `graphify-out/` is git-ignored. Not installed: git hooks (`graphify hook install`) — they would run on every commit. Community names are placeholders (naming needs an LLM backend). Refresh with `graphify update .`. `docs/graph/` (curated plan/status graph) stays the source for *meaning*; graphify is the source for *code structure*; both pass their checks.
- Not run: Windows/macOS/ARM, GPU benchmarks, anything in section B.

### 2026-10-21 — 2.0.0 released; root cleanup
- Owner merged PR #5 (override), I tagged `v2.0.0` on `030ba32`; release workflow green (see Now §0). Verified before tagging: `origin/main` tree equals the tested tree; Debug/ReleaseFast/v3/baseline 196/196, e2e 26/26, bindings 8 suites, pytest 97 passed / 3 skipped.
- Branch `docs/readme-landing`: repo root cleaned, `graphify-out/` committed, README/CONTRIBUTING/HANDOFF rewritten; `gen_graph.py --check` is run before the commit. Not run: a fresh `pip install hknt==2.0.0` / `npm install hkntf` smoke test.
- Merged PR #6 (admin bypass, owner-authorised for that PR) and updated the About blurb. Refreshed both graphs (`graphify update .`: 5,742 nodes, 11,135 edges; `gen_graph.py --check` 0 errors). CI on `main` after the merge was still running when this was written.
