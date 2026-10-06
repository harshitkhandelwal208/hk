#!/usr/bin/env python3
"""Builds the HK knowledge graph.

Inputs : docs/graph/curated.json (hand-maintained concepts, plan items, status) and the source tree
         (Zig `@import`s, Python relative imports).
Outputs: docs/graph/graph.json (machine-readable) and docs/graph/GRAPH.md (Mermaid + tables).
Usage  : python3 docs/graph/gen_graph.py            write outputs (exit 1 on a consistency error)
         python3 docs/graph/gen_graph.py --check    write nothing; exit 1 if outputs are stale or inconsistent

Standard library only. Output is deterministic (no timestamps) so --check is meaningful in CI.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = ROOT / "docs" / "graph"
CURATED = GRAPH_DIR / "curated.json"
OUT_JSON = GRAPH_DIR / "graph.json"
OUT_MD = GRAPH_DIR / "GRAPH.md"

SCAN = [
    ("src", "*.zig"),
    ("tools", "*.zig"),
    ("tests", "*.zig"),
    ("python/hk", "*.py"),
    ("tests", "*.py"),
    ("tools", "*.py"),
    ("src", "*.cu"),
    ("shaders", "*.comp"),
    ("shaders", "*.glsl"),
]
MIN_COUPLING_SHOWN = 2  # single-import edges are in graph.json but make the diagram unreadable
SKIP_PARTS = {"__pycache__", ".zig-cache", "zig-out", "node_modules", "fixtures"}
ZIG_IMPORT = re.compile(r'@import\("([^"]+)"\)')
PY_FROM = re.compile(r"^\s*from\s+(\.+)([\w.]*)\s+import\s+(.+)$", re.M)
STATUS_ORDER = ["done", "partial", "in-progress", "prototype", "storage-only", "legacy", "untested-hw", "not-started"]
STATUS_COLOR = {
    "done": "#2e7d32", "partial": "#f9a825", "in-progress": "#1565c0", "prototype": "#8e24aa",
    "storage-only": "#6d4c41", "legacy": "#757575", "untested-hw": "#ef6c00", "not-started": "#c62828",
}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def collect_files() -> list[Path]:
    seen: dict[str, Path] = {}
    for base, pattern in SCAN:
        root = ROOT / base
        if not root.exists():
            continue
        for p in root.rglob(pattern):
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_PARTS):
                seen[rel(p)] = p
    return [seen[k] for k in sorted(seen)]


def count_loc(p: Path) -> int:
    try:
        with p.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def zig_imports(p: Path, known: set[str]) -> list[str]:
    out = []
    text = p.read_text(encoding="utf-8", errors="replace")
    for m in ZIG_IMPORT.finditer(text):
        name = m.group(1)
        if name.endswith(".zig"):
            target = rel((p.parent / name).resolve()) if (p.parent / name).resolve().is_relative_to(ROOT) else None
            if target in known:
                out.append(target)
        elif name == "hk" and "src/root.zig" in known and not rel(p).startswith("src/"):
            out.append("src/root.zig")
    return out


def py_imports(p: Path, known: set[str]) -> list[str]:
    out = []
    text = p.read_text(encoding="utf-8", errors="replace")
    pkg_dir = p.parent
    for m in PY_FROM.finditer(text):
        dots, mod, names = len(m.group(1)), m.group(2), m.group(3)
        base = pkg_dir
        for _ in range(dots - 1):
            base = base.parent
        candidates = []
        if mod:
            parts = mod.split(".")
            candidates.append(base.joinpath(*parts).with_suffix(".py"))
            candidates.append(base.joinpath(*parts, "__init__.py"))
        else:
            for n in re.split(r"[,\s()]+", names):
                n = n.strip()
                if n and n.isidentifier():
                    candidates.append((base / n).with_suffix(".py"))
                    candidates.append(base / n / "__init__.py")
        for c in candidates:
            try:
                r = rel(c.resolve())
            except ValueError:
                continue
            if r in known and r != rel(p):
                out.append(r)
                break
    return out


def expand_paths(paths: list[str], files: list[str]) -> tuple[set[str], list[str]]:
    covered: set[str] = set()
    dangling: list[str] = []
    for raw in paths:
        full = ROOT / raw
        if not full.exists():
            dangling.append(raw)
            continue
        prefix = raw.rstrip("/")
        for f in files:
            if f == prefix or f.startswith(prefix + "/"):
                covered.add(f)
    return covered, dangling


def mm_id(s: str) -> str:
    return "n_" + re.sub(r"[^A-Za-z0-9]", "_", s)


def mm_label(s: str) -> str:
    return s.replace('"', "'")


def build() -> tuple[dict, str, list[str]]:
    errors: list[str] = []
    cur = json.loads(CURATED.read_text(encoding="utf-8"))
    nodes = cur["nodes"]
    ids = [n["id"] for n in nodes]
    for dup, c in Counter(ids).items():
        if c > 1:
            errors.append(f"duplicate curated id: {dup}")
    idset = set(ids)
    for e in cur["edges"]:
        for end in ("s", "t"):
            if e[end] not in idset:
                errors.append(f"edge references unknown node: {e}")
    for n in nodes:
        if n["status"] not in STATUS_ORDER:
            errors.append(f"{n['id']}: unknown status {n['status']!r}")

    file_paths = collect_files()
    files = [rel(p) for p in file_paths]
    known = set(files)

    # curated node -> files it covers
    cover: dict[str, set[str]] = {}
    for n in nodes:
        covered, dangling = expand_paths(n.get("paths", []), files)
        cover[n["id"]] = covered
        for d in dangling:
            errors.append(f"{n['id']}: path does not exist: {d}")

    # file -> most specific owner (longest matching curated path prefix)
    owner: dict[str, str] = {}
    best_len: dict[str, int] = {}
    for n in nodes:
        for raw in n.get("paths", []):
            prefix = raw.rstrip("/")
            for f in cover[n["id"]]:
                if f == prefix or f.startswith(prefix + "/"):
                    if len(prefix) > best_len.get(f, -1):
                        best_len[f] = len(prefix)
                        owner[f] = n["id"]

    file_nodes, import_edges = [], []
    for p in file_paths:
        r = rel(p)
        lang = {"zig": "zig", "py": "python", "cu": "cuda", "comp": "glsl", "glsl": "glsl"}.get(p.suffix.lstrip("."), "other")
        file_nodes.append({"id": "file:" + r, "type": "file", "label": r, "lang": lang, "loc": count_loc(p), "owner": owner.get(r)})
        targets = zig_imports(p, known) if lang == "zig" else py_imports(p, known) if lang == "python" else []
        for t in sorted(set(targets)):
            import_edges.append({"s": "file:" + r, "t": "file:" + t, "type": "imports"})

    impl_edges = [
        {"s": nid, "t": "file:" + f, "type": "implemented_by"}
        for nid in ids for f in sorted(cover[nid]) if owner.get(f) == nid
    ]

    # aggregated coupling between curated subsystems
    # src/root.zig re-exports every module (and tests/tools import it as `hk`), so edges touching it say
    # nothing about real coupling; they stay in `imports` but are excluded from the aggregate.
    coupling: Counter = Counter()
    for e in import_edges:
        if "file:src/root.zig" in (e["s"], e["t"]):
            continue
        a, b = owner.get(e["s"][5:]), owner.get(e["t"][5:])
        if a and b and a != b:
            coupling[(a, b)] += 1

    orphans = sorted(f for f in files if f not in owner)
    status_count = Counter(n["status"] for n in nodes if n["type"] not in ("constraint", "root", "system", "milestone"))

    graph = {
        "schema_version": cur.get("schema_version", 1),
        "as_of": cur.get("as_of", ""),
        "generated_by": "docs/graph/gen_graph.py",
        "stats": {
            "curated_nodes": len(nodes), "curated_edges": len(cur["edges"]),
            "file_nodes": len(file_nodes), "import_edges": len(import_edges),
            "unowned_files": len(orphans), "status_counts": {s: status_count.get(s, 0) for s in STATUS_ORDER},
        },
        "nodes": nodes + file_nodes,
        "edges": cur["edges"] + impl_edges + import_edges,
        "coupling": [{"s": a, "t": b, "imports": c} for (a, b), c in sorted(coupling.items())],
        "unowned_files": orphans,
    }
    md = render_md(cur, graph, cover, owner, coupling, orphans, file_nodes)
    return graph, md, errors


def class_defs() -> list[str]:
    return [f"  classDef {s.replace('-', '_')} fill:{c},stroke:#222,color:#fff;" for s, c in STATUS_COLOR.items()]


def render_md(cur, graph, cover, owner, coupling, orphans, file_nodes) -> str:
    nodes = cur["nodes"]
    by_id = {n["id"]: n for n in nodes}
    loc = {f["label"]: f["loc"] for f in file_nodes}
    L: list[str] = []
    a = L.append
    a("# HK knowledge graph")
    a("")
    a("> **Generated by `docs/graph/gen_graph.py` — do not edit by hand.** Edit `curated.json` (concepts, status, plan links) and rerun.")
    a(f"> As of: {graph['as_of']}.")
    a("")
    s = graph["stats"]
    a(f"{s['curated_nodes']} curated nodes, {s['curated_edges']} curated edges, {s['file_nodes']} source files, "
      f"{s['import_edges']} import edges, {s['unowned_files']} files not owned by any curated node.")
    a("")
    a("## Using the graph")
    a("")
    a("`graph.json` has `nodes` (types: root, system, subsystem, backend, planned, milestone, constraint, doc, file) and `edges`")
    a("(types: contains, depends_on, blocked_by, requires, constrains, supersedes, implemented_by, imports). Plan sections are in each node's `plan` field.")
    a("")
    a("```bash")
    a("# everything the plan says about a section, and what implements it")
    a("jq '.nodes[] | select(.plan? // [] | index(57))' docs/graph/graph.json")
    a("# files behind a subsystem")
    a("jq -r '.edges[] | select(.s==\"engine\" and .type==\"implemented_by\") | .t' docs/graph/graph.json")
    a("# what is blocked by the backend abstraction")
    a("jq -r '.edges[] | select(.t==\"backend-abstraction\" and .type==\"blocked_by\") | .s' docs/graph/graph.json")
    a("# all not-started items")
    a("jq -r '.nodes[] | select(.status==\"not-started\") | .id' docs/graph/graph.json")
    a("# which curated subsystems import which (measured, not curated)")
    a("jq -r '.coupling[] | \"\\(.s) -> \\(.t) (\\(.imports))\"' docs/graph/graph.json")
    a("```")
    a("")
    a("Status legend: " + ", ".join(f"`{k}`" for k in STATUS_ORDER) + ". Meanings are in `docs/plan/STATUS.md`.")
    a("")
    a("Status counts (subsystems, backends, planned, docs): " + ", ".join(f"{k} {v}" for k, v in s["status_counts"].items() if v) + ".")
    a("")

    # 1. status map
    a("## 1. System map (colored by status)")
    a("")
    a("```mermaid")
    a("flowchart TB")
    parent: dict[str, str] = {}
    for e in cur["edges"]:
        if e["type"] == "contains" and e["t"] not in parent and by_id[e["s"]]["type"] == "system":
            parent[e["t"]] = e["s"]
    for sysn in [n for n in nodes if n["type"] == "system"]:
        a(f'  subgraph {mm_id(sysn["id"])}["{mm_label(sysn["label"])}"]')
        a("    direction TB")
        for n in nodes:
            if parent.get(n["id"]) == sysn["id"]:
                a(f'    {mm_id(n["id"])}["{mm_label(n["label"])}"]:::{n["status"].replace("-", "_")}')
        a("  end")
    a("\n".join(class_defs()))
    a("```")
    a("")

    # 2. architectural dependencies, split so each diagram stays readable
    def edge_diagram(title: str, intro: str, keep) -> None:
        a(title)
        a("")
        a(intro)
        a("")
        a("```mermaid")
        a("flowchart BT")
        used: set[str] = set()
        lines = []
        for e in cur["edges"]:
            if keep(e):
                used.update((e["s"], e["t"]))
                arrow = "-.->" if e["type"] == "blocked_by" else ("==>" if e["type"] == "supersedes" else "-->")
                lines.append(f'  {mm_id(e["s"])} {arrow} {mm_id(e["t"])}')
        for nid in sorted(used):
            n = by_id[nid]
            a(f'  {mm_id(nid)}["{mm_label(n["label"])}"]:::{n["status"].replace("-", "_")}')
        L.extend(lines)
        a("\n".join(class_defs()))
        a("```")
        a("")

    def is_planned(nid: str) -> bool:
        return by_id[nid]["type"] == "planned"

    edge_diagram("## 2a. What exists: dependencies between built subsystems",
                 "Solid arrow = depends on; double arrow = independent path that does not share runtime code with its target.",
                 lambda e: e["type"] in ("depends_on", "supersedes") and not is_planned(e["s"]) and not is_planned(e["t"]))
    edge_diagram("## 2b. What is planned, and what gates it",
                 "Dashed arrow = blocked by (the planned item cannot start until its target exists); solid = depends on.",
                 lambda e: e["type"] in ("blocked_by", "depends_on") and (is_planned(e["s"]) or is_planned(e["t"])))

    # 3. milestones
    a("## 3. Milestones and what they require")
    a("")
    a("```mermaid")
    a("flowchart TB")
    used = set()
    lines = []
    for e in cur["edges"]:
        if e["type"] == "requires":
            used.update((e["s"], e["t"]))
            lines.append(f'  {mm_id(e["s"])} --> {mm_id(e["t"])}')
    for nid in sorted(used):
        n = by_id[nid]
        a(f'  {mm_id(nid)}["{mm_label(n["label"])}"]:::{n["status"].replace("-", "_")}')
    L.extend(lines)
    a("\n".join(class_defs()))
    a("```")
    a("")

    # 4. measured coupling
    a("## 4. Measured coupling (source imports aggregated per curated subsystem)")
    a("")
    a("Counts are file-level import edges from `@import(\"x.zig\")` and Python relative imports. Edges to or from `src/root.zig`")
    a("(the library's re-export hub, which tests and tools import as `hk`) are excluded: it touches every module and would hide the real structure.")
    a(f"The diagram shows only pairs with at least {MIN_COUPLING_SHOWN} imports; every pair (including single imports) is in `graph.json` under `coupling`.")
    a("")
    a("```mermaid")
    a("flowchart LR")
    used = set()
    lines = []
    for (x, y), c in sorted(coupling.items()):
        if c < MIN_COUPLING_SHOWN:
            continue
        used.update((x, y))
        lines.append(f'  {mm_id(x)} -->|{c}| {mm_id(y)}')
    for nid in sorted(used):
        n = by_id[nid]
        a(f'  {mm_id(nid)}["{mm_label(n["label"])}"]:::{n["status"].replace("-", "_")}')
    L.extend(lines)
    a("\n".join(class_defs()))
    a("```")
    a("")

    # 5. node table
    a("## 5. Node index")
    a("")
    for typ, title in [("system", "Systems"), ("subsystem", "Subsystems"), ("backend", "Backends"), ("planned", "Planned (no code yet)"),
                       ("milestone", "Milestones"), ("constraint", "Standing constraints"), ("doc", "Docs")]:
        group = [n for n in nodes if n["type"] == typ]
        if not group:
            continue
        a(f"### {title}")
        a("")
        a("| id | label | status | plan § | files | LOC | note |")
        a("|:--|:--|:--|:--|--:|--:|:--|")
        for n in group:
            fs = sorted(f for f in cover[n["id"]] if owner.get(f) == n["id"])
            total = sum(loc.get(f, 0) for f in fs)
            plan = ", ".join(str(x) for x in n.get("plan", [])) or "—"
            note = n.get("note", "").replace("|", "\\|")
            a(f"| `{n['id']}` | {n['label']} | {n['status']} | {plan} | {len(fs) or '—'} | {total or '—'} | {note} |")
        a("")

    a("## 6. Files not owned by any curated node")
    a("")
    if orphans:
        a("Add them to a node's `paths` in `curated.json` (or accept them as incidental):")
        a("")
        for f in orphans:
            a(f"- `{f}`")
    else:
        a("None.")
    a("")
    return "\n".join(L) + "\n"


def main() -> int:
    check = "--check" in sys.argv[1:]
    graph, md, errors = build()
    js = json.dumps(graph, indent=1, sort_keys=False) + "\n"
    if check:
        for path, new in ((OUT_JSON, js), (OUT_MD, md)):
            old = path.read_text(encoding="utf-8") if path.exists() else None
            if old != new:
                errors.append(f"stale: {rel(path)} (run python3 docs/graph/gen_graph.py)")
    else:
        if not errors:
            OUT_JSON.write_text(js, encoding="utf-8")
            OUT_MD.write_text(md, encoding="utf-8")
    for e in errors:
        print("ERROR:", e, file=sys.stderr)
    s = graph["stats"]
    print(f"{'checked' if check else 'wrote'}: {s['curated_nodes']} curated nodes, {s['file_nodes']} files, "
          f"{s['import_edges']} import edges, {s['unowned_files']} unowned files; {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
