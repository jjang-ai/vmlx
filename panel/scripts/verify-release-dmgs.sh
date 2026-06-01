#!/usr/bin/env bash
set -euo pipefail

# Verify final public DMG containers after signing, notarization, and stapling.
# This script does not build, sign, notarize, upload, or mutate release feeds.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PANEL_DIR"

VERSION="$(node -p "require('./package.json').version")"
DIST_DIR="${VMLINUX_RELEASE_OUTPUT_DIR:-release}"

verify_one() {
  local flavor="$1"
  local dmg="$DIST_DIR/vMLX-${VERSION}-${flavor}-arm64.dmg"

  if [[ ! -f "$dmg" ]]; then
    echo "ERROR: missing release DMG: $dmg" >&2
    exit 1
  fi

  echo "==> Verifying final ${flavor} DMG: $dmg"
  hdiutil verify "$dmg"
  codesign --verify --verbose=2 "$dmg"
  codesign -dv --verbose=4 "$dmg"
  xcrun stapler validate "$dmg"
  spctl --assess --type open --context context:primary-signature --verbose=4 "$dmg"
  shasum -a 256 "$dmg"
}

for flavor in sequoia tahoe; do
  verify_one "$flavor"
done
