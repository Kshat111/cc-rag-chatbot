# CLAUDE.md

Rules Claude Code follows every session in this repo.

- Read docs/HANDOVER.md, docs/CONSTRAINTS.md and docs/ARCHITECTURE.md first.
- Update HANDOVER.md last.
- Before editing, explain your plan and at least one alternative, and wait for my approval.
- One logical change per request. If a task needs more, propose a split.
- Before any non-trivial change, draft a DECISIONS.md entry with options and trade-offs.
- Comment intent and assumptions in non-obvious code, never restate syntax.
- Use uv only. No new dependency without asking, and say why it is needed and who maintains it.
- All tunables live in config.py. No hardcoded keys, paths or model names elsewhere.
- Never read, print or edit .env. Never run destructive commands.
- Show real command output. Never claim tests pass without pasting them.
- At the end of each task, list what you assumed that I did not tell you.
