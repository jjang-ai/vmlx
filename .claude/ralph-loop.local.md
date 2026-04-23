---
active: true
iteration: 5
session_id: production-audit-2026-04-22
max_iterations: 0
completion_promise: "EVERY CHECKLIST ROW IN docs/production-audit-2026-04-22.md MARKED [x] WITH COMMIT SHA. IMAGE GEN/EDIT PRODUCE REAL PROMPT-CONDITIONED PIXELS OR EXPLICITLY GATED AS 501 — NO STUB-SUCCESS PATHWAY REMAINS. DEEP-SLEEP DRAINS CHAT + IMAGE + EMBEDDING BACKENDS. `bash .claude/live-verify.sh` 44/44 GREEN. `bash .claude/lifecycle-verify.sh` 13/13 GREEN. `bash .claude/image-lifecycle-verify.sh` ALL GREEN. GATEWAY REACHABLE FROM CLI AND LIVE-VERIFIED. NO UI BUTTON ORPHANED. dev BRANCH PUSHED."
started_at: "2026-04-22T19:15:44Z"
---

All checklist + progress tracking lives in `docs/production-audit-2026-04-22.md`
— deliberately outside `.claude/` so per-edit permission prompts don't fire
on the progress document. This file is kept minimal (just the frontmatter
the ralph-loop plugin reads) so subsequent edits are rare (only to bump
`iteration:` on each loop turn).
