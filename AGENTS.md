# AGENTS.md — working agreement for any coding agent on HK

This file is tool-agnostic (Claude Code, Codex, Cursor, Aider, a human). `CLAUDE.md` simply imports it.

## Orient in 3 minutes

1. Read [CONTEXT.md](CONTEXT.md) (stable map, commands, standing rules).
2. Read the **Now** section and the last log entry of [docs/plan/HANDOFF.md](docs/plan/HANDOFF.md).
3. `git status && git log --oneline -5`, then `zig build test` for a baseline. If the tree disagrees with HANDOFF,
   trust the tree and fix HANDOFF.
4. For the area you will touch: its rows in [docs/plan/STATUS.md](docs/plan/STATUS.md), and its node in
   [docs/graph/graph.json](docs/graph/README.md) (`jq` recipes inside).

The target is [docs/plan/MASTER_PLAN.md](docs/plan/MASTER_PLAN.md). It is the owner's document: do not rewrite it;
put proposed changes in HANDOFF "Open decisions".

## Non-negotiables

- **Memory first.** Streaming, mmap, bounded buffers. Report peak RSS for anything that touches model data.
- **No silent failure.** Right or flagged — never silently wrong. Every guard gets a test that goes red when the
  guard is removed (verify it by disabling the guard once). Never "fall back" to a less safe mode without saying so.
- **Truthful claims.** README/wiki/CHANGELOG/commit messages state only what was run. "Cross-compiled" ≠ "tested";
  "one GPU" ≠ "GPUs"; "not run" must be written as "not run".
- **Benchmarks before "faster"** (plan §1.5): same hardware/model/workload, repeats + variance, memory, correctness.
- **Correctness > speed > features > research** (plan §158).
- **Hostile inputs:** model files, tokenizers, chat templates, HTTP, downloads. Overflow-safe bounds checks;
  validate before casting bytes to enums/slices; errors name the offending tensor/field.
- **Don't commit/push/tag/release unless the owner asks.** Do not edit `build.zig.zon`'s fingerprint.

## Workflow

- **Which stack?** Engine path = `src/engine`, `src/quant`, `src/kernels`, `src/vk`. Legacy path = `src/tensor_ops.zig`,
  `src/quantization.zig`, `src/c_api.zig`. Standalone = `src/cuda*`. Confirm before optimizing (CONTEXT §3).
- **Tests live next to the code** (`test "…"` blocks, pulled in via `src/root.zig`'s `refAllDecls`) and in `tests/`
  (Zig suites registered in `build.zig`; pytest; `tests/bindings/run.sh`). New Zig test files must be added to `build.zig`.
- **Kernel changes:** run `zig build test -Doptimize=ReleaseFast` (all levels), then the forced low levels
  (`-Dcpu=x86_64_v3`, `-Dcpu=x86_64`), then a `hk-compare` run if you claim speed. All ISA levels must produce the same integers.
- **Format changes:** update `docs/wiki/Format-Specification.md` in the same change; keep old files readable (v1 compat);
  add a legacy-file test.
- **C ABI changes:** update `include/hk.h` and `hk.hpp`, every affected binding, and `tests/bindings/`.
- **User-visible changes:** `CHANGELOG.md` entry; wiki page if behavior is documented there (`docs/wiki` auto-syncs to the GitHub wiki).
- **Style:** match the surrounding Zig/Python — `//!` module docs that explain *why*, `///` on public items, error
  sets named and specific, no unexplained magic numbers, comments rare and about intent.
- **Python:** `python/hk/` is a high-level layer; the plan wants hot paths native. Don't add new NumPy-heavy hot loops.

## Keeping the context system honest

| You did… | Update… |
|:--|:--|
| Finished, started, or discovered anything about in-flight work | `docs/plan/HANDOFF.md` (Now + log) |
| Changed what works / what's verified | the touched rows in `docs/plan/STATUS.md` |
| Added/moved/deleted source files, or changed a subsystem's status | `docs/graph/curated.json` if a concept/status changed; then `python3 docs/graph/gen_graph.py` |
| Found a doc ≠ code mismatch | fix it, or add to STATUS "Known doc/code discrepancies" |
| Learned a durable owner preference | the agent's memory system (if it has one) and, if it's a project rule, `CONTEXT.md` §5 |

`python3 docs/graph/gen_graph.py --check` fails if the generated graph is stale or a curated node points at a path
that no longer exists. Run it before ending a session that touched structure.

## Handing off to another agent

Finish by making HANDOFF.md sufficient on its own: what is half-done (file + function), what was run vs. only read,
the next concrete command to run, and every decision you needed from the owner but could not get. Assume the next
agent has none of your conversation.

## Pitfalls already paid for

- Optimizing `tensor_ops.zig` and expecting `hk run` to speed up (it won't; see "two kernel stacks").
- Quoting "137+ architectures" or "CUDA support" as supported features (neither runs in the engine).
- Trusting `NativeSandbox`'s `timed_out` (it doesn't enforce a timeout) or `CodeSandbox`'s Docker mode (it can silently fall back).
- Editing a `.hk` file's metadata through an mmap of that same file — the TOC is read through the mapping you are overwriting (1.1.1 bug; see the in-place patch tests).
- Assuming Debug `zig build test` exercises SIMD kernels: Debug builds only the portable `generic` level.
- Appendix records are ignored by the engine; a LoRA/delta in the appendix does not change inference.
