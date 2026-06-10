#!/usr/bin/env bash
# Fast path: refresh only vmlx_engine + jang_tools inside an existing bundled-python.
# Invoked by: make sync  (see panel/Makefile)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$PANEL_DIR")"
BUNDLE_DIR="$PANEL_DIR/bundled-python"
PYTHON="$BUNDLE_DIR/python/bin/python3"
JANG_LOCAL="$(bash "$REPO_DIR/scripts/resolve-jang-tools.sh" "$REPO_DIR")"
VMLX_LOCAL="$REPO_DIR"

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: bundled Python missing: $PYTHON" >&2
  echo "       Run ./scripts/bundle-python.sh once first (full ~15–30 min)." >&2
  exit 1
fi

echo "==> Syncing local vmlx_engine + jang_tools into bundled-python"
if [ -f "$VMLX_LOCAL/pyproject.toml" ] && [ -d "$VMLX_LOCAL/vmlx_engine" ]; then
  echo "    vmlx: $VMLX_LOCAL"
  "$PYTHON" -m pip install --force-reinstall --no-deps --no-cache-dir "$VMLX_LOCAL"
else
  echo "ERROR: local vmlx checkout missing at $VMLX_LOCAL" >&2
  exit 1
fi

if [ -f "$JANG_LOCAL/pyproject.toml" ]; then
  echo "    jang: $JANG_LOCAL"
  "$PYTHON" -m pip install --force-reinstall --no-deps --no-cache-dir "$JANG_LOCAL"
else
  echo "ERROR: local jang-tools missing at $JANG_LOCAL" >&2
  exit 1
fi

for SCRIPT in "$BUNDLE_DIR/python/bin/"vmlx* "$BUNDLE_DIR/python/bin/"jang*; do
  if [ ! -f "$SCRIPT" ]; then
    continue
  fi
  FIRST_LINE="$(LC_ALL=C head -n 1 "$SCRIPT" 2>/dev/null || true)"
  if [[ "$FIRST_LINE" == '#!'*python* ]] && [[ "$FIRST_LINE" != '#!/bin/sh'* ]]; then
    TMP_SCRIPT="$(mktemp "${SCRIPT}.XXXXXX")"
    {
      printf '%s\n' '#!/bin/sh'
      printf '%s\n' "'''exec' \"\$(dirname \"\$0\")/python3\" -B -s \"\$0\" \"\$@\""
      printf '%s\n' "' '''"
      tail -n +2 "$SCRIPT"
    } > "$TMP_SCRIPT"
    chmod --reference="$SCRIPT" "$TMP_SCRIPT" 2>/dev/null || chmod +x "$TMP_SCRIPT"
    mv "$TMP_SCRIPT" "$SCRIPT"
  fi
done

echo "==> Local package sync OK"
date -u +%Y-%m-%dT%H:%M:%SZ > "$BUNDLE_DIR/.sync-stamp"
PANEL_DIR="$PANEL_DIR" "$SCRIPT_DIR/prune-ship-artifacts.sh"
echo "    Release gate: make verify  or  make release  (verify runs before DMG pack)"
