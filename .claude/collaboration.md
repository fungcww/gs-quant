# Collaboration Notes

The user co-pilots with a Gemini AI agent — Gemini handles high-level planning and produces prompts/tasks, Claude is the implementation and critique layer.

## How to apply
- Treat pasted Gemini prompts as implementation requests, not gospel — evaluate them critically before acting.
- Freely raise concerns if a Gemini suggestion seems architecturally unsound, risky, redundant, or inconsistent with the existing codebase.
- Debate the decision if there is a clearly better alternative; briefly explain the tradeoff.
- Ask for clarification when the prompt is ambiguous or missing context needed to implement it correctly.
- Do not blindly implement something that could introduce bugs, security issues, or unnecessary complexity just because Gemini suggested it.

## gs-quant Library-First Rule
Before building any new strategy component, indicator, or mathematical calculation:
1. **Search the gs-quant library first** — check `gs_quant/timeseries/`, `gs_quant/analytics/`, `gs_quant/risk/`, and `gs_quant/markets/` for existing implementations.
2. **Use it if it exists** — do not rewrite math (e.g. volatility, correlation, regression, convexity) that gs-quant already provides; import and call the library function instead.
3. **Only build custom code** when the required logic is genuinely absent from gs-quant or requires project-specific adaptation that cannot be cleanly layered on top.
4. When uncertain, do a quick grep/search of the gs-quant source before concluding something is missing.
