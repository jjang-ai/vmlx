---
name: ralph-loop
enabled: true
iteration: 6
completion_promise: none
event: stop
---

# Ralph Loop — vMLX Swift production audit

Iteration 6 continues the N6 / N3 / N8 close-out + engine-level
generation_config.json wiring across all APIs, plus beta.10 build
with proper AppIcon compiled via actool.

Recently closed (iter-6):
- §371 Terminal reasoning surface — .reasoning role + auto-collapse
- §372 Tooltip i18n — 26-key L10n.Tooltip (en/ja/ko/zh) covering
  Chat + Server + Image + API .help() hovers
- §373 Engine sampling fallback — three-tier chain
  (request → loadedModelDefaults → resolved) so Qwen/Gemma/Nemotron
  recommended temp/top_p/top_k values reach the sampler by default
- §374 Build script wires AppIcon via actool so Dock/Finder/About
  no longer show a generic icon

Tracker: docs/audit/OPEN-FIX-LIST.md

Next up:
- Run full swift test after beta.10 builds
- Live-test §373 against Qwen / Gemma / Nemotron for default sampling
- F4 text selection sweep remaining spots
- Eric-visible final audit pass
