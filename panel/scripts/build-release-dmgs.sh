#!/usr/bin/env bash
set -euo pipefail

# Build the two public macOS DMG flavors for the same source checkout.
#
# vmlx#169: macosx_26 MLX wheels ship Metal language 4.0 kernels that are
# valid on Tahoe but fail on Sequoia. Release packaging must therefore produce
# two clearly named DMGs from the same source:
#   - sequoia: macosx_14 wheels, works on Sonoma 14.5+, Sequoia 15, and Tahoe
#   - tahoe: native macosx_26 wheels, Tahoe-only
#
# This script only builds local artifacts. It does not tag, upload, publish,
# notarize, update the updater manifest, or create a GitHub release.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PANEL_DIR"

VERSION="$(node -p "require('./package.json').version")"
DIST_DIR="${VMLINUX_RELEASE_OUTPUT_DIR:-release}"

build_one() {
  local flavor="$1"
  local platform="$2"
  local wheel_tag

  case "$flavor" in
    sequoia) wheel_tag="macosx_14_0_arm64" ;;
    tahoe) wheel_tag="macosx_26_0_arm64" ;;
    *)
      echo "ERROR: unsupported release flavor: $flavor" >&2
      exit 1
      ;;
  esac

  echo "==> Building vMLX ${VERSION} ${flavor} DMG (${wheel_tag})"
  VMLX_BUNDLE_MLX_PLATFORM="$platform" ./scripts/bundle-python.sh
  ./scripts/verify-bundled-python.sh
  npx electron-vite build
  npx electron-builder --mac dmg \
    --config.directories.output="$DIST_DIR" \
    --config.mac.artifactName="vMLX-\${version}-${flavor}-\${arch}.\${ext}"
}

case "${1:-all}" in
  all)
    rm -rf "$DIST_DIR"
    build_one "sequoia" "compat"
    build_one "tahoe" "native"
    ;;
  sequoia)
    build_one "sequoia" "compat"
    ;;
  tahoe)
    build_one "tahoe" "native"
    ;;
  *)
    echo "Usage: $0 [all|sequoia|tahoe]" >&2
    exit 2
    ;;
esac

echo "==> Built DMG artifacts:"
find "$DIST_DIR" -maxdepth 1 -type f -name "vMLX-${VERSION}-*.dmg" -print | sort
