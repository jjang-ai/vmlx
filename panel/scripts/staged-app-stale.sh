#!/usr/bin/env bash
# Exit 0 if staged app is missing or older than build inputs (rebuild needed).
# Exit 1 if up to date. Makefile tests use: ! staged-app-stale.sh
set -euo pipefail

STAGED_APP="${1:?usage: staged-app-stale.sh STAGED_APP}"
PANEL_DIR="${PANEL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
BUNDLE_DIR="$PANEL_DIR/bundled-python"
SOURCES_NEWER="$PANEL_DIR/scripts/sources-newer-than.sh"

# shellcheck source=_build-input-paths.sh
source "$PANEL_DIR/scripts/_build-input-paths.sh"

if [[ ! -d "$STAGED_APP" ]]; then
  exit 0
fi

app_ref="$STAGED_APP/Contents/Info.plist"
if [[ ! -f "$app_ref" ]]; then
  app_ref="$STAGED_APP"
fi

if "$SOURCES_NEWER" "$app_ref" "${BUILD_INPUT_PATHS[@]}"; then
  exit 0
fi

exit 1
