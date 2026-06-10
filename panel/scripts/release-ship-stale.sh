#!/usr/bin/env bash
# Exit 0 if ship artifacts are missing or older than build inputs. Exit 1 if DMG is current.
set -euo pipefail

PANEL_DIR="${PANEL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
RELEASE_DIR="${VMLX_RELEASE_DIR:-$PANEL_DIR/release}"
STAGED_APP="${STAGED_APP:-$RELEASE_DIR/mac-arm64/vMLX.app}"
BUNDLE_DIR="$PANEL_DIR/bundled-python"
SOURCES_NEWER="$PANEL_DIR/scripts/sources-newer-than.sh"

# shellcheck source=_build-input-paths.sh
source "$PANEL_DIR/scripts/_build-input-paths.sh"

DMG="$(find "$RELEASE_DIR" -maxdepth 1 -type f -name 'vMLX-*.dmg' 2>/dev/null | sort | tail -1)"
if [[ -z "$DMG" ]]; then
  exit 0
fi

if "$SOURCES_NEWER" "$DMG" "${BUILD_INPUT_PATHS[@]}"; then
  exit 0
fi

if [[ -d "$STAGED_APP" ]]; then
  app_ref="$STAGED_APP/Contents/Info.plist"
  if [[ ! -f "$app_ref" ]]; then
    app_ref="$STAGED_APP"
  fi
  if "$SOURCES_NEWER" "$DMG" "$app_ref"; then
    exit 0
  fi
fi

exit 1
