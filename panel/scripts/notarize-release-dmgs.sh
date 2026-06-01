#!/usr/bin/env bash
set -euo pipefail

# Submit final signed public DMG containers to Apple, staple the accepted
# tickets, and print the post-staple hashes that must be used for release feeds.
# This script does not build, sign, upload, tag, publish, or mutate latest.json.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PANEL_DIR"

VERSION="$(node -p "require('./package.json').version")"
DIST_DIR="${VMLINUX_RELEASE_OUTPUT_DIR:-release}"
NOTARY_PROFILE="${VMLINUX_NOTARY_KEYCHAIN_PROFILE:-${VMLX_NOTARY_KEYCHAIN_PROFILE:-vmlx-notary}}"

notarize_one() {
  local flavor="$1"
  local dmg="$DIST_DIR/vMLX-${VERSION}-${flavor}-arm64.dmg"

  if [[ ! -f "$dmg" ]]; then
    echo "ERROR: missing release DMG: $dmg" >&2
    exit 1
  fi

  echo "==> Notarizing final signed ${flavor} DMG: $dmg"
  hdiutil verify "$dmg"
  codesign --verify --verbose=2 "$dmg"
  codesign -dv --verbose=4 "$dmg"
  xcrun notarytool submit "$dmg" \
    --keychain-profile "$NOTARY_PROFILE" \
    --wait \
    --output-format json
  xcrun stapler staple "$dmg"
  xcrun stapler validate "$dmg"
  spctl --assess --type open --context context:primary-signature --verbose=4 "$dmg"
  shasum -a 256 "$dmg"
}

for flavor in sequoia tahoe; do
  notarize_one "$flavor"
done
