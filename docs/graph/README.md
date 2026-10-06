# docs/graph — HK knowledge graph

A queryable map of the project that lets a fresh session or a different agent find "what is X, what depends on
it, what does the plan say about it, and is it real yet" without reading the whole tree.

| File | What | Edit by hand? |
|:--|:--|:--|
| `curated.json` | Concepts, subsystems, planned-but-missing items, milestones, standing rules, **status**, plan-section links, and the edges between them | **yes** — this is the source of truth for meaning |
| `gen_graph.py` | Merges `curated.json` with the real source tree (Zig `@import`, Python relative imports) | no |
| `graph.json` | Full graph: curated nodes + every source file + curated edges + `implemented_by` + `imports` + aggregated `coupling` | no (generated) |
| `GRAPH.md` | Mermaid diagrams (status map, dependencies, milestones, measured coupling) and a node index table | no (generated) |

## Two layers, on purpose

- **Curated** (judgment): what a subsystem *is*, whether it's done/prototype/not-started, which plan sections it
  serves, what blocks what. Wrong curation is a documentation bug — fix `curated.json` and `docs/plan/STATUS.md` together.
- **Measured** (facts): which files exist, who imports whom, how many lines. Cannot drift, because it is regenerated.
  Use it to *check* the curated layer: e.g. the claim "CUDA is not wired into the engine" can be tested with
  `imports` edges — the only file importing `src/cuda.zig` is `src/root.zig` (a re-export); `tests/test_cuda.zig` and
  `tools/hk_gpu_bench.zig` reach it as `hk.cuda`; neither `src/c_api.zig` nor anything under `src/engine/` touches it.

## Commands

```bash
python3 docs/graph/gen_graph.py            # regenerate graph.json + GRAPH.md
python3 docs/graph/gen_graph.py --check    # no writes; exit 1 if stale, a curated path is missing,
                                           # an edge points at an unknown node, or a status is invalid
```

`--check` is cheap (stdlib only, <1 s) and suitable for CI. It has been mutation-tested against: a dangling path, an
unknown edge endpoint, an invalid status, a stale `graph.json`, and a stale `GRAPH.md`.

## When to touch `curated.json`

- A subsystem changes status (e.g. `prototype` → `done`) — also update its row in `docs/plan/STATUS.md`.
- You add a new top-level module, backend, or planned item the plan cares about.
- A file moves: update the `paths` of the owning node (the check fails if you forget).
- A node's `paths` may be a file or a directory. A file is owned by the **longest matching** path, so a broad node
  (`tests`, `python/hk`, `tools`) can act as a catch-all without stealing files from specific nodes.

## Node and edge vocabulary

Node `type`: `root`, `system`, `subsystem`, `backend`, `planned`, `milestone`, `constraint`, `doc`, `file`.
Node `status`: `done`, `partial`, `in-progress` (uncommitted work), `prototype`, `storage-only`, `legacy`,
`untested-hw`, `not-started`.
Edge `type`: `contains` (hierarchy), `depends_on` (runtime/build dependency), `blocked_by` (planned work gated on
something), `requires` (milestone needs), `constrains` (standing rule applies), `supersedes`, plus generated
`implemented_by` and `imports`.

Plan sections are in each node's `plan` array and refer to `docs/plan/MASTER_PLAN.md` §N.

## Example queries

(The `jq` snippets below were not executed in the session that wrote them — `jq` was not installed; the equivalent
queries were checked with Python against `graph.json`. If one errors, the data is in the file; adjust the filter.)

```bash
jq '.nodes[] | select(.plan? // [] | index(57))' docs/graph/graph.json                       # everything touching plan §57 (CUDA)
jq -r '.nodes[] | select(.status=="prototype") | "\(.id)\t\(.label)"' docs/graph/graph.json  # research prototypes
jq -r '.edges[] | select(.t=="backend-abstraction" and .type=="blocked_by") | .s' docs/graph/graph.json
jq -r '.nodes[] | select(.type=="file" and .owner=="engine") | "\(.loc)\t\(.label)"' docs/graph/graph.json | sort -rn
jq -r '.coupling[] | select(.s=="format-core") | "\(.s) -> \(.t) (\(.imports))"' docs/graph/graph.json
```

## Known limits

- Import edges come from regexes over `@import("x.zig")` and Python `from . import …`. Tests and tools that import the
  `hk` module are linked to `src/root.zig`, not to individual files; edges touching `src/root.zig` are left out of `coupling`. Dynamic imports, `build.zig` wiring, and C-ABI
  calls from bindings are not captured.
- The five Mermaid diagrams in `GRAPH.md` were rendered with `@mermaid-js/mermaid-cli` 12.0.0 (headless Chromium) and inspected; all parse and are
  legible. To re-check after changing the generator: extract the ```` ```mermaid ```` blocks and run `mmdc -i block.mmd -o block.png`
  (add `-p puppeteer.json` with `{"args":["--no-sandbox"]}` where Chromium's sandbox is unavailable). CI does not run this.
