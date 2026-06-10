# Build Makefile Enhancements (2026-06-09)

Notes for PR description and changelog. Covers local build ergonomics work
alongside the Qwen3.5 VL MTP patch.

## Problem

Panel/desktop builds were hard to discover and easy to get wrong:

- `npm run build` always re-ran `bundle-python.sh` (~30 min) even for UI-only edits
- `build-and-install.sh` called `npm run build`, triggering the same rebundle trap
- Docs pointed at `npm install`, `npm run build`, `npm run dist` with conflicting meanings
- jang-tools `step37/` requirement and macOS BSD `tar` path bug blocked first-time builders
- No single entry point, resume guidance, or artifact checksums

## Solution: repo-root Makefile + panel delegation

Run **`make`** or **`make help`** from the repo root.

### One-shot targets

| Target | When |
|--------|------|
| `make first-dmg` | First time from scratch (bundle + DMG + SHA256SUMS) |
| `make release` / `make installable` | Staged `.app` → DMG (`--prepackaged`); removes `release/mac-arm64/` after pack |
| `make engine-dmg` | `check-sync` + `release` after engine/jang edits |
| `make engine-app` | Local staged `.app` after engine/jang edits (sync + app, no DMG) |
| `make engine-and-install` | `engine-app` + install to `/Applications` |
| `make install` | Auto-sync/rebuild if sources changed, then copy/sign — **no forced DMG** |

Stale ship artifacts (`vMLX-*.dmg`, `SHA256SUMS`, zip/blockmap) are removed when bundled-python syncs or the staged app rebuilds.

### Resume after failure

| Failed step | Re-run |
|-------------|--------|
| bundle | `make bundle` |
| staged app / pack | `make app` |
| DMG/pack | `make release` |
| engine/jang (local) | `make engine-app` or `make engine-and-install` |
| engine/jang (ship) | `make engine-dmg` |
| everything | `make first-dmg` |

## Files added

| File | Purpose |
|------|---------|
| `Makefile` | Repo-root orchestration (engine, git, stash, delegates panel) |
| `panel/Makefile` | Panel/bundle/DMG targets |
| `scripts/freshen-repo.sh` | `make freshen*` git fetch/checkout |
| `panel/scripts/sync-bundled-local-packages.sh` | Fast vmlx+jang reinstall (`make sync`) |
| `panel/scripts/hash-release-artifacts.sh` | SHA256SUMS after DMG (automatic) |
| `panel/scripts/prune-ship-artifacts.sh` | Drop stale DMG/SHA256SUMS when inputs change |
| `panel/scripts/release-ship-stale.sh` | Skip DMG repack when ship artifacts are current |
| `panel/scripts/install-from-release.sh` | Install from staged `.app` or DMG without rebundle |

## Files changed

| File | Change |
|------|--------|
| `panel/package.json` | `build` = panel only; `build:bundle`, `build:full` split; `dist:local` hint |
| `panel/scripts/build-and-install.sh` | Delegates to `make release` / `first-dmg`, no rebundle |
| `panel/scripts/bundle-python.sh` | macOS tar path fix; success next-step hints |
| `README.md` | Contributing → `make help` |
| `docs/development/build-test-deploy.md` | Canonical Makefile workflow |

## Other ergonomics

- `make doctor` — conda warning, jang branch/step37, SHA256SUMS status
- `make freshen-jang` — default branch with `step37/`
- `make stash` / `stash-apply` — PR prep with `STASH_ALL=1` for untracked files
- `make clean` / `clean-all` / `uninstall`
- `npm run dist` — still official release path (manifest gates); local DMG = `make release`

## PR split suggestion

1. **Bug fix PR:** `vmlx_engine/patches/mlx_vlm_mtp/qwen35_vl.py` only
2. **Build ergonomics PR:** Makefile, scripts, docs (this note)

## Fork / branch workflow (contributor)

```bash
# GitHub: fork jjang-ai/vmlx → your account

git remote add fork git@github.com:YOURUSER/vmlx.git
git checkout -b fix/qwen35-vl-mtp-and-build-ergonomics

# work …
make doctor
make first-dmg          # or: make bundle && make release
make engine-and-install # local: sync + app + /Applications

STASH_ALL=1 make stash  # optional before upstream sync
make freshen-all && make stash-apply

git add … && git commit
git push fork HEAD
# open PR to jjang-ai/vmlx
```

## Changelog snippet (panel)

```markdown
### Build tooling (local dev)
- Add repo-root and panel Makefiles with `first-dmg`, `release`, `engine-app`, `engine-and-install`, `engine-dmg`, `install`.
- `make install` uses staged `.app` (`make app`); DMG only for `make release` / `make engine-dmg`.
- Stop `npm run build` from re-running full `bundle-python.sh` on every panel compile.
- Auto-write `panel/release/SHA256SUMS` when packaging DMG.
- Fix macOS BSD tar member paths in `bundle-python.sh`.
- Add `sync-bundled-local-packages.sh` for fast vmlx_engine/jang_tools refresh.
```

## Do not commit accidentally

- `panel/package-lock.json` unless npm deps intentionally changed
- `panel/bundled-python/` (gitignored)
- `vmlx.egg-info/` from local pip installs
