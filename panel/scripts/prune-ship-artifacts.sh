#!/usr/bin/env bash
# Remove local ship artifacts (DMG, zip, checksums). Staged .app is handled separately.
set -euo pipefail

PANEL_DIR="${PANEL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
RELEASE_DIR="${VMLX_RELEASE_DIR:-$PANEL_DIR/release}"

if [[ ! -d "$RELEASE_DIR" ]]; then
  exit 0
fi

removed=0
while IFS= read -r -d '' artifact; do
  rm -f "$artifact"
  removed=1
done < <(
  find "$RELEASE_DIR" -maxdepth 1 -type f \( \
    -name 'vMLX-*.dmg' -o \
    -name 'vMLX-*.zip' -o \
    -name 'SHA256SUMS' -o \
    -name '*.blockmap' \
  \) -print0 2>/dev/null
)

if [[ "$removed" == "1" ]]; then
  echo "==> Removed stale ship artifacts from $RELEASE_DIR"
fi
