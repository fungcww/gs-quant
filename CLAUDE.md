# CLAUDE.md — gs-quant Quant Research Project

## Project Overview
Systematic trading strategy research built on top of the gs-quant library.
- Self-developed research scripts: `research/` (backtesters, optimizers, alpha engines)
- gs-quant extensions (custom math/indicators): `gs_quant/` (extend in-place, not in research scripts)
- Market data: daily OHLC persisted in `shared_data/market.db` (yfinance)
- Output charts: `charts/`

---

## Reference Map

| File | Purpose | When to read |
|------|---------|--------------|
| [.claude/roadmap.md](.claude/roadmap.md) | 4-phase roadmap + current milestone bookmark | Always — check current phase before suggesting next work |
| [.claude/strategy.md](.claude/strategy.md) | Active strategy rules, indicators, costs, data split | When modifying any signal or execution logic |
| [.claude/collaboration.md](.claude/collaboration.md) | Gemini workflow + Claude behaviour rules | Always — governs how to handle all incoming prompts |

---

@.claude/roadmap.md

---

@.claude/strategy.md

---

@.claude/collaboration.md
