# vMLX Private UI E2E Test Suite

Drives the **shipped** apps (Electron panel + SwiftUI Swift app) end-to-end
against real binaries, not mocks. Intended for in-repo dogfood: any time
you ship a panel or Swift change, run this once, attach the screenshot
+ smoke JSON to the PR.

## Two drivers

### Electron panel — `panel-driver/`
Playwright over Chrome DevTools Protocol. Launches `/Applications/vMLX.app`
with `--remote-debugging-port=9222`, connects via `chromium.connectOverCDP`,
walks the renderer DOM exactly like Chrome.

Setup once:
```
cd tests/e2e/panel-driver && npm install
```

Self-contained CLI:
```
node drive.mjs launch                  # launch panel with CDP
node drive.mjs smoke                   # full smoke pass + screenshot + report JSON
node drive.mjs shot out.png            # screenshot the visible renderer window
node drive.mjs eval "document.title"   # eval JS in renderer, print result
node drive.mjs click "button:has-text('Server')"
node drive.mjs type  "textarea[placeholder*='message']" "Hello"
node drive.mjs wait  "[role='alert']" 5000
node drive.mjs close
```

The `smoke` pass covers:
1. renderer URL + title resolved
2. **NO setup-screen / "Install Engine" surface** (regression guard for the
   user-reported 1.4.2 flow)
3. tab set present (Code / Chat / Server / Tools / Image / API)
4. screenshot baseline → `reports/smoke-<ts>.png`
5. zero console errors during boot
6. engine version surfaced (currently expected-fail until the panel
   exposes the version on the boot screen — informational)

### SwiftUI app — `swift-axdriver/`
Custom Swift CLI using the macOS Accessibility API
(`ApplicationServices.AXUIElement*`). System Events / AppleScript reports
SwiftUI windows as opaque AXGroup blobs; this driver pulls real role,
title, accessibilityIdentifier, description, and value via the C API.

Setup once:
```
cd tests/e2e/swift-axdriver && swift build -c release
# Grant Terminal "Accessibility" in System Settings → Privacy & Security
# (one-time prompt on first run)
```

Self-contained CLI (`<pid>` = the running vMLX SwiftUI process pid):
```
.build/release/vmlx-axdriver dump  <pid>                 # full AX tree
.build/release/vmlx-axdriver grep  <pid> "Load Model"    # find by needle
.build/release/vmlx-axdriver click <pid> "Server"        # click button by title or accessibilityIdentifier
.build/release/vmlx-axdriver type  <pid> "ChatInput" "Hi" # set value of text field
.build/release/vmlx-axdriver wait  <pid> "Send" 10       # wait up to 10s for element
.build/release/vmlx-axdriver shot  <pid> out.png         # screenshot the window (needs Screen Recording perm)
```

`shot` uses `CGWindowListCreateImage` which is deprecated on macOS 14+
but still functional; in 15+ Apple wants you on `ScreenCaptureKit` and
will eventually require Screen Recording TCC permission for any app
trying to read pixels of windows it doesn't own. We accept the
deprecation warning for now; ScreenCaptureKit migration tracked
separately.

## Orchestrator

```
tests/e2e/run-ui-suite.sh         # both apps, full pass
tests/e2e/run-ui-suite.sh panel   # panel only
tests/e2e/run-ui-suite.sh swift   # swift only
```

Writes a timestamped log + report to `tests/e2e/reports/`.

## What this gives us

- **Regression guard** for the "First-time setup / Install Engine"
  bug — `panel-driver smoke` fails if that string ever surfaces again.
- **Tab-set parity** — both Electron and SwiftUI must expose the same
  navigation surface (Chat, Server, Image, Terminal, API at minimum).
- **Live UI on every PR** — no need to manually click through; the
  smoke pass takes ~10s combined.
- **Real driving for me/Claude** — I can grep the AX tree, click buttons,
  type into fields, take screenshots, and assert on state changes
  without depending on `osascript` (which sees Electron + SwiftUI as
  AXGroup blobs).

## Permissions checklist

One-time on a fresh machine:
1. **Accessibility** for Terminal (or whichever shell you launch
   `vmlx-axdriver` from): System Settings → Privacy & Security →
   Accessibility → toggle Terminal on. Required for AX read + click.
2. **Screen Recording** for Terminal (optional): same panel → Screen
   Recording. Required only for `vmlx-axdriver shot`. The Electron
   path uses Playwright's CDP screenshot which does NOT need this.

## Scope it does NOT cover (yet)

- **Multi-window flow** (settings popovers, modal dialogs) — driver
  only inspects window 1.
- **Notarized DMG flow** — these drivers run against unsigned local
  builds; if the DMG ever fails to launch through Gatekeeper, this
  suite won't catch it. Add `spctl --assess` + `codesign -dv`
  pre-flight to the runner if/when needed.
- **Interactive engine load smoke** — picking a real model from the
  library and clicking "Load Model" works at the AX-click level
  (verified manually 2026-04-30) but is not asserted in the smoke
  pass because it ties up GPU memory; covered by `harness.sh` instead
  at the engine HTTP surface.
