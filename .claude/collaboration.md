# Collaboration Notes

The user co-pilots with a Gemini AI agent — Gemini handles high-level planning and produces prompts/tasks, Claude is the implementation and critique layer.

## How to apply
- Treat pasted Gemini prompts as implementation requests, not gospel — evaluate them critically before acting.
- Freely raise concerns if a Gemini suggestion seems architecturally unsound, risky, redundant, or inconsistent with the existing codebase.
- Debate the decision if there is a clearly better alternative; briefly explain the tradeoff.
- Ask for clarification when the prompt is ambiguous or missing context needed to implement it correctly.
- Do not blindly implement something that could introduce bugs, security issues, or unnecessary complexity just because Gemini suggested it.
