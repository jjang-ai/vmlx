#!/usr/bin/env bash
# vMLX UI E2E suite — drives the LIVE shipped apps (Electron panel + SwiftUI
# Swift app) end-to-end. The two drivers (tests/e2e/panel-driver — Playwright
# over Electron CDP — and tests/e2e/swift-axdriver — macOS Accessibility API)
# both have to be set up once, then this script runs both against the
# installed apps and writes a report under tests/e2e/reports/.
#
# Setup (once):
#   cd tests/e2e/panel-driver && npm install
#   cd tests/e2e/swift-axdriver && swift build -c release
#   # Grant Terminal "Accessibility" in System Settings → Privacy & Security
#
# Usage:
#   tests/e2e/run-ui-suite.sh            # full pass on both apps
#   tests/e2e/run-ui-suite.sh panel      # panel only
#   tests/e2e/run-ui-suite.sh swift      # swift only
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
REPORTS="$HERE/reports"
mkdir -p "$REPORTS"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$REPORTS/run-ui-$TS.log"

mode=${1:-all}
panel_app=/Applications/vMLX.app
swift_app=/Applications/vMLX-Swift.app
swift_bin="$REPO_ROOT/.build/arm64-apple-macosx/release/vMLX"
ax="$HERE/swift-axdriver/.build/release/vmlx-axdriver"

note() { echo "[run-ui $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ── Panel ────────────────────────────────────────────────────────────────────
test_panel() {
    note "panel: ensure clean state"
    pkill -9 -f "vMLX.app" 2>/dev/null || true
    sleep 1

    note "panel: launch with CDP"
    nohup "$panel_app/Contents/MacOS/vMLX" --remote-debugging-port=9222 --no-sandbox \
        > "$REPORTS/panel-$TS.log" 2>&1 &
    sleep 4

    if ! curl -sf http://127.0.0.1:9222/json/version >/dev/null; then
        note "panel: CDP did not come up — see $REPORTS/panel-$TS.log"
        return 1
    fi
    note "panel: CDP up"

    pushd "$HERE/panel-driver" >/dev/null
    if [ ! -d node_modules ]; then npm install --silent 2>&1 | tail -3; fi
    note "panel: smoke pass"
    if node drive.mjs smoke 2>&1 | tee -a "$LOG"; then
        note "panel: SMOKE PASS"
    else
        note "panel: SMOKE FAIL"
        popd >/dev/null
        return 1
    fi
    popd >/dev/null

    note "panel: stop"
    pkill -9 -f "vMLX.app" 2>/dev/null || true
}

# ── Swift app ────────────────────────────────────────────────────────────────
test_swift() {
    note "swift: ensure clean state"
    pkill -9 -f "vMLX-Swift" 2>/dev/null || true
    pkill -9 -x "vMLX" 2>/dev/null || true
    sleep 1

    if [ ! -x "$swift_bin" ]; then
        note "swift: building (.build/release/vMLX missing)"
        (cd "$REPO_ROOT" && swift build -c release 2>&1 | tail -3) | tee -a "$LOG"
    fi

    note "swift: launch"
    nohup "$swift_bin" > "$REPORTS/swift-$TS.log" 2>&1 &
    SWIFT_PID=$!
    sleep 3

    if ! ps -p "$SWIFT_PID" > /dev/null; then
        note "swift: process died — see $REPORTS/swift-$TS.log"
        return 1
    fi
    note "swift: pid=$SWIFT_PID up"

    if [ ! -x "$ax" ]; then
        note "swift: building axdriver"
        (cd "$HERE/swift-axdriver" && swift build -c release 2>&1 | tail -3) | tee -a "$LOG"
    fi

    note "swift: AX dump"
    "$ax" dump "$SWIFT_PID" > "$REPORTS/swift-axtree-$TS.txt" 2>&1 || true
    AX_LINES=$(wc -l < "$REPORTS/swift-axtree-$TS.txt")
    note "swift: AX tree has $AX_LINES lines"

    note "swift: AX grep for visible tabs"
    "$ax" grep "$SWIFT_PID" Chat | head -5 | tee -a "$LOG" || true

    note "swift: window screenshot"
    "$ax" shot "$SWIFT_PID" "$REPORTS/swift-shot-$TS.png" 2>&1 | tee -a "$LOG" || true

    note "swift: stop"
    kill -TERM "$SWIFT_PID" 2>/dev/null || true
    sleep 1
    pkill -9 -x "vMLX" 2>/dev/null || true
}

# ── Run ──────────────────────────────────────────────────────────────────────
case "$mode" in
    panel) test_panel ;;
    swift) test_swift ;;
    all)
        test_panel || true
        test_swift || true
        ;;
    *) echo "usage: $0 [panel|swift|all]"; exit 1 ;;
esac

note "report dir: $REPORTS"
ls -la "$REPORTS" | tail -10 | tee -a "$LOG"
