@AGENTS.md

# Claude Code specifics

- Read `CONTEXT.md` and the top of `docs/plan/HANDOFF.md` at the start of a session; update HANDOFF before ending one.
- Plan mode is appropriate for anything touching the `.hk` format, the C ABI, or the backend interface.
- Prefer `git ls-files`/`find` over `ls` in this environment (the `ls` alias rejects directory arguments).
- Use the Agent tool only when the owner asks for it; the docs under `docs/plan/` and `docs/graph/` are designed so a
  fresh agent does not need a conversation to get oriented.
