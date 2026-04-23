---
name: ralph-production-audit
enabled: true
event: stop
pattern: .*
action: block
---

🔄 **vMLX Swift production audit loop — do not stop yet.**

Check `docs/production-audit-2026-04-22.md` for the next open `[ ]` row
(in block order: I → L → H → G → C → A → S → U → O → P → D). Pick the
TOP open row and execute this protocol:

1. Read the implicated source paths + the prior commit(s) that touched
   them (use `git log --oneline -- <path>`). No guessing.

2. **Ship real code**, not a workaround. If the fix needs an
   abstraction, ship it. If a placeholder returns success today, cut
   it or gate it to 501. No stub-returning-success pathways remain.

3. **Build**: `swift build -c release` must pass green.

4. **Live-verify** — boot a real server, hit the real route, measure
   real behavior. NEVER fake a pass. Use:
   - `.build/release/vmlxctl serve -m <path> --port <uniq> ...`
   - `curl -sS ... | python3 -c 'import json; ...'` for assertions
   - `/metrics` scrape for RAM/GPU delta measurement

5. **Cross-check** sibling harnesses — ALL three must stay green:
   - `bash .claude/live-verify.sh` (44/44)
   - `bash .claude/lifecycle-verify.sh` (13/13)
   - `bash .claude/image-lifecycle-verify.sh` (12/12)
   Any regression rolls the current row back to `[ ]`.

6. **Commit + push**:
   - `git -c user.name="Jinho Jang" -c user.email="eric@jangq.ai" commit -m "..."`
   - `git push origin dev`
   - No Co-Authored-By. No AI attribution.

7. **Mark `[x]` in `docs/production-audit-2026-04-22.md`** with commit
   SHA + one-line proof (curl excerpt, harness line, or file path).
   Update the "Commits this audit chain" list and "Harness state" line.

8. **Bump `iteration:`** in `.claude/ralph-loop.local.md` by one.

9. **If a row is TOO BIG** for one iteration, split into sub-rows FIRST
   (I7a pattern — "I2a", "L5a", etc.), then work the first sub-row.
   Never claim `[x]` on a partial.

## Scope — what "production level" means here

Every surface listed in `docs/production-audit-2026-04-22.md` blocks
1 through 11 MUST close before the loop exits:

- **Block 1 (Image gen/edit)**: Z-Image placeholder honest OR real DiT
  port shipped; Flux.1/Flux2/FIBO/QwenImage/SeedVR2 either produce
  pixels or return 501 with download hint; edit route round-trip
  verified when an edit model is locally available.
- **Block 2 (Lifecycle)**: deepSleep drains chat + image + embedding;
  softSleep drains transient state only; JIT re-hydrate works on all
  three.
- **Block 3 (Harness coverage)**: all three harness scripts shipped +
  green.
- **Block 4 (Gateway)**: `vmlxctl gateway` subcommand reachable + live-
  verified multi-engine routing.
- **Block 5 (Cache)**: hybrid SSM + paged + prefix + TQ + disk L2
  matrix green across Gemma4, Qwen3.6, MiniMax, Nemotron.
- **Block 6 (API)**: all 52 RouteCatalog routes live-verified with
  expected shapes or documented as blocked-on-model-asset.
- **Block 7 (Settings)**: 4-tier resolution per-field test coverage.
- **Block 8 (UI buttons)**: every button action traced to a real
  engine/setting method. No orphaned `TODO`/empty-action bugs.
- **Block 9 (Logs)**: category taxonomy + log-level propagation +
  no-PII guard verified.
- **Block 10 (Perf)**: decode tok/s targets for Qwen3.5-35B /
  Nemotron-Cascade / Gemma-4-26B met or gap documented.
- **Block 11 (Packaging)**: notarized DMG, fresh-user first-run,
  homebrew formula, auto-updater.

## Completion gate

This hook only releases when the user explicitly says "done" OR the
audit doc has zero `[ ]` rows AND all three harnesses are green AND
the dev branch matches the promise in `.claude/ralph-loop.local.md`.

No stopping early.
