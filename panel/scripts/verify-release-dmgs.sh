#!/usr/bin/env bash
set -euo pipefail

# Verify final public DMG containers after signing, notarization, and stapling.
# This script does not build, sign, notarize, upload, or mutate release feeds.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PANEL_DIR"

VERSION="$(node -p "require('./package.json').version")"
DIST_DIR="${VMLINUX_RELEASE_OUTPUT_DIR:-release}"

require_developer_id_signature() {
  local path="$1"
  local signature

  signature="$(codesign -dv --verbose=4 "$path" 2>&1)"
  printf '%s\n' "$signature"

  if ! grep -Fq "Authority=Developer ID Application: ShieldStack LLC (55KGF2S5AY)" <<<"$signature"; then
    echo "ERROR: $path is not signed by the ShieldStack Developer ID Application certificate" >&2
    exit 1
  fi

  if ! grep -Fq "TeamIdentifier=55KGF2S5AY" <<<"$signature"; then
    echo "ERROR: $path is not signed with the ShieldStack Developer ID team identifier" >&2
    exit 1
  fi

  if grep -Fq "Signature=adhoc" <<<"$signature"; then
    echo "ERROR: $path is ad-hoc signed and is not a public release DMG" >&2
    exit 1
  fi
}

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
  require_developer_id_signature "$dmg"
  xcrun stapler validate "$dmg"
  spctl --assess --type open --context context:primary-signature --verbose=4 "$dmg"
  shasum -a 256 "$dmg"
}

for flavor in sequoia tahoe; do
  verify_one "$flavor"
done
