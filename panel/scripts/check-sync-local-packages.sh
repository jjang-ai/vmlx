#!/usr/bin/env bash
# Check whether bundled-python matches local vmlx_engine + jang_tools; sync if not.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$PANEL_DIR")"
BUNDLE_DIR="$PANEL_DIR/bundled-python"
SYNC_STAMP="$BUNDLE_DIR/.sync-stamp"
SOURCES_NEWER="$SCRIPT_DIR/sources-newer-than.sh"
SYNC_SCRIPT="$SCRIPT_DIR/sync-bundled-local-packages.sh"
JANG_TOOLS="${VMLX_JANG_TOOLS_SOURCE:-${VMLINUX_JANG_TOOLS_SOURCE:-$(bash "$REPO_DIR/scripts/resolve-jang-tools.sh" "$REPO_DIR")}}"

if [[ ! -x "$BUNDLE_DIR/python/bin/python3" ]]; then
  echo "==> bundled-python missing — run make bundle first" >&2
  exit 1
fi

sync_paths=("$REPO_DIR/vmlx_engine")
if [[ -d "$JANG_TOOLS/jang_tools" ]]; then
  sync_paths+=("$JANG_TOOLS/jang_tools")
elif [[ -d "$JANG_TOOLS" ]]; then
  sync_paths+=("$JANG_TOOLS")
fi

if [[ ! -f "$SYNC_STAMP" ]] || "$SOURCES_NEWER" "$SYNC_STAMP" "${sync_paths[@]}"; then
  echo "==> vmlx_engine / jang_tools changed — running make sync"
  exec "$SYNC_SCRIPT"
fi

echo "==> bundled-python matches local packages (no sync needed)"
