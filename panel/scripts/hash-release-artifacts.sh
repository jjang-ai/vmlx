#!/usr/bin/env bash
# Write SHA256 checksums for packaged release artifacts under panel/release/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL_DIR="$(dirname "$SCRIPT_DIR")"
RELEASE_DIR="${VMLX_RELEASE_DIR:-$PANEL_DIR/release}"
SUMS_FILE="${VMLX_SHA256SUMS_FILE:-$RELEASE_DIR/SHA256SUMS}"

if [[ ! -d "$RELEASE_DIR" ]]; then
  echo "ERROR: release directory missing: $RELEASE_DIR" >&2
  echo "       Run make dmg first." >&2
  exit 1
fi

mapfile -t ARTIFACTS < <(
  find "$RELEASE_DIR" -maxdepth 2 -type f \( \
    -name '*.dmg' -o -name '*.zip' \
  \) ! -name 'SHA256SUMS' | sort
)

if [[ ${#ARTIFACTS[@]} -eq 0 ]]; then
  echo "ERROR: no release artifacts found under $RELEASE_DIR" >&2
  echo "       Run make dmg first." >&2
  exit 1
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

{
  echo "# vMLX release artifact SHA256 checksums"
  echo "# generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# verify:    make verify-hashes"
  echo "#"
  for artifact in "${ARTIFACTS[@]}"; do
    rel="${artifact#"$RELEASE_DIR"/}"
    hash="$(shasum -a 256 "$artifact" | awk '{print $1}')"
    printf '%s  %s\n' "$hash" "$rel"
  done
} > "$TMP"

mv "$TMP" "$SUMS_FILE"

echo "==> Wrote $SUMS_FILE"
cat "$SUMS_FILE"
echo
DMG="$(find "$RELEASE_DIR" -maxdepth 1 -type f -name 'vMLX-*.dmg' | sort | tail -1)"
echo "✓ release ready"
if [[ -n "$DMG" ]]; then
  echo "  DMG:      $DMG"
fi
echo "  Hashes:   $SUMS_FILE"
echo "  Install:  make install"
echo "  Mount:    open \"$RELEASE_DIR\"/vMLX-*.dmg"
echo "  Verify:   make verify-hashes"
