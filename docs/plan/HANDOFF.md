# HK — Handoff

Volatile, session-to-session state. **Newest information goes at the top of "Now"; the log at the bottom is
append-only.** Stable orientation: [CONTEXT.md](../../CONTEXT.md). Plan vs. reality: [STATUS.md](STATUS.md).

Rules for editing this file: state what was *run* vs. *only read*; never write "done" for something not
verified; when a "Now" item is finished, move one line of it to the log and delete it from "Now".

---

## Now

### A. In flight: container integrity + crash-safe metadata edits (plan §4, §5) — UNCOMMITTED

Working tree on top of `43cf891` (16 changed paths; new files `src/integrity.zig`, `src/transaction.zig`).

| Piece | Where | State |
|:--|:--|:--|
| Whole-file SHA-256 seal, header flag bit 9 (`HAS_SHA256_CHECKSUM`) | `src/integrity.zig`, `format.zig`, `writer.zig` (seals after write), `reader.verifyIntegrity` | Written; unit test passes |
| Per-record CRC-32 in appendix; `0` = legacy record | `src/appendix.zig`, `python/hk/adaptive/appendix.py` | Written; unit test passes |
| Overflow-safe reader range validation | `src/reader.zig` (`checkedEnd`, `validateHeaderLayout`, `validateTensorRanges`) | Written; unit test passes |
| Metadata edit journal + recovery | `src/transaction.zig`, `metadata.patchFileMetadataInPlace` | Written; unit test passes |
| CLI: `hk verify` reports seal; new `hk repair` | `tools/hk_cli.zig` | Written; **no e2e test** for `repair` / seal reporting |
| Docs | `docs/wiki/Format-Specification.md`, `In-Container-Version-Lineage.md`, `Raw-Storage-and-Super-Coalescing.md` | Edited to match |
| Python mirror of the seal | `python/hk/adaptive/appendix.py` | Written; **Python suite not run** |

**Verified (2026-10-06):** `zig build test` → 17/17 steps, 138/138 tests pass (Debug). New tests:
`Appendix CRC-32 rejects a corrupted payload`, `container SHA-256 seal detects tensor corruption`,
`reader rejects an overflowing metadata range`, `metadata journal restores an interrupted mutation`.

**NOT verified:** `zig build test -Doptimize=ReleaseFast`, `-Dcpu=x86_64_v3`/`x86_64`, `zig build test-e2e`,
`bash tests/bindings/run.sh`, `pytest tests`, Windows (the `writer.zig` change closes the handle before the
second mapper opens it "to avoid a conflicting handle on Windows" — unrun). Mutation-style check ("does each new test
fail when its guard is removed?") not done — owner's robustness standard requires it before calling this finished.

**Risks / gaps found by reading the diff (not yet decided):**

1. **O(file) cost per mutation.** `sealFile` hashes the entire container (mmap, bounded RAM, but a multi-GB read) on
   *every* metadata patch and *every* appended appendix record, and rewrites the header twice. Fine for tests,
   costly for large models, and it cuts against the memory/IO-first principle. Options: seal only on explicit
   `hk seal`/write; hash sections separately (header+meta+TOC / each tensor / appendix) so appends only re-hash
   the appendix; Merkle tree. **Owner decision needed** (see Open decisions).
2. **Crash window in `sealFile`.** It first writes the header with the flag set and a *zeroed* digest, then hashes,
   then writes the final header. A crash between leaves a file that claims a seal and fails it. `hk repair` can only
   restore a *metadata journal*; with no journal (appendix append/rollback path) it reports `REPAIR INCOMPLETE`
   and cannot re-seal. Re-sealing blindly would bless a possibly truncated appendix, so this is deliberately left.
3. **Appendix append/rollback are still non-atomic** (in-place append + truncate; Python rewrites the whole file via
   `open(..., "wb")` — a crash mid-write can destroy the file; also O(file) RAM in Python). Plan §5 wants atomic appendix writes.
4. **`HKWriter.write` writes to the destination path directly** (no temp file + rename).
5. **No `hk compact` / `hk fsck`.** `hk verify` text still says "128-byte alignment" in its usage line.
6. **Legacy-file policy:** v1 files without seal verify as `legacy_missing` (warning, not failure). Fine for
   compatibility; means an attacker can strip the seal. Signatures (plan §4) are the real fix, not started.

**Suggested finish for A (in this order):**
1. Run the full not-verified list above; fix whatever breaks.
2. Mutation-check the four new tests (disable each guard, confirm red).
3. Add an e2e test in `tests/cli_tests.zig` for `hk verify` (sealed / tampered / legacy) and `hk repair`.
4. Resolve Open decision 1 (seal cost) before more code builds on `sealFile`.
5. Update `CHANGELOG.md` (it is the project's release notes), then commit when the owner asks.

### B. Recommended next arcs (after A), per plan §151

The plan's chain says the next structural gate is **graph/model abstraction → backend abstraction**, because CUDA,
Metal, ROCm, and the memory planner all hang off it. Today `engine/model.zig` imports the Vulkan engine directly
(`gpu_mod`), and CUDA sits unused in `src/cuda.zig`. Candidate order, smallest useful step first:

1. **Backend interface (plan §1.1, §137):** define `Backend` (capabilities query, buffer alloc, matmul-by-format,
   attention, norm, rope, kv ops, sync) behind which CPU and Vulkan sit; no behavior change; tests prove parity.
2. **Operator capability table (§136):** per op × dtype/quant × backend; replaces ad-hoc "GPU decodes these formats" lists
   and makes fallback diagnostics (§137) systematic.
3. **Wire CUDA into the engine through that interface** (reusing `src/cuda.zig`/PTX), measured against Vulkan and
   llama.cpp on the RTX 3050 with `hk-compare`. Prefill (GEMM) before decode before attention (§151, §153).
4. **Memory planner v0 (§1.2, §134):** answer "can it fit / where does each tensor live" for CPU+one GPU; enables
   partial offload (currently unsupported in the Vulkan engine).
5. **Architecture descriptor (§46):** replace the 3-value `Arch` enum with data-driven descriptors; then Gemma/Phi.

None of B is started. These are recommendations, not decisions.

### Open decisions (owner)

1. **Seal cost model** (Risk 1): whole-file hash per mutation vs sectioned hashes vs explicit seal command.
2. **Which kernel stack is canonical?** Plan §9/§15/§17/§20 say "retain the earlier optimizations", but those live in
   the legacy stack (`tensor_ops.zig`), not the engine. Options: port them into the engine stack; or retire the
   legacy stack and route C API/Python through the engine kernels (smaller surface, one place to optimize).
3. **CUDA path:** adopt `src/cuda.zig` as the base for an engine backend, or restart from the Vulkan engine's structure?
   (Commit history gives conflicting accounts of how well the old CUDA code worked; nothing re-measured.)
4. **Sandbox policy (§33):** `CodeSandbox` silently falls back from Docker to an unsandboxed subprocess — a plan
   violation. Make it fail closed (raise unless `allow_unsafe=True`)? Likely yes; needs a go-ahead since it changes behavior.
5. **Version number:** CLI banner says 1.2.0, packages say 1.1.1.
6. **`build.zig` `-Dcuda` default-on** for non-macOS builds although CUDA is off the engine path.

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
