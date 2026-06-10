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
ROOT_DIR="$(dirname "$PANEL_DIR")"

cd "$PANEL_DIR"

VERSION="$(node -p "require('./package.json').version")"
DIST_DIR="${VMLINUX_RELEASE_OUTPUT_DIR:-release}"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi
PREPACKAGE_READY_MANIFEST_OUT="${VMLX_PREPACKAGE_READY_MANIFEST_OUT:-${VMLINUX_PREPACKAGE_READY_MANIFEST_OUT:-$ROOT_DIR/build/current-release-regression-manifest-pre-dmg-release-build.json}}"
case "$PREPACKAGE_READY_MANIFEST_OUT" in
  /*) ;;
  *) PREPACKAGE_READY_MANIFEST_OUT="$ROOT_DIR/$PREPACKAGE_READY_MANIFEST_OUT" ;;
esac
RELEASE_CODESIGN_IDENTITY="${VMLX_RELEASE_CODESIGN_IDENTITY:-${VMLINUX_RELEASE_CODESIGN_IDENTITY:-${CSC_NAME:-Developer ID Application: ShieldStack LLC (55KGF2S5AY)}}}"
CHECKPOINT_RELEASE_OVERRIDE="${VMLX_CHECKPOINT_RELEASE_OVERRIDE:-${VMLINUX_CHECKPOINT_RELEASE_OVERRIDE:-0}}"

echo "==> Checking pre-package release ledger before public DMG build"
if [[ "$CHECKPOINT_RELEASE_OVERRIDE" = "1" ]]; then
  echo "WARNING: VMLX_CHECKPOINT_RELEASE_OVERRIDE=1 — building a checkpoint DMG with open rows." >&2
  echo "WARNING: Release notes must list open rows from: $PREPACKAGE_READY_MANIFEST_OUT" >&2
  set +e
  (
    cd "$ROOT_DIR"
    "$PYTHON_BIN" "tests/cross_matrix/run_release_regression_manifest.py" \
      --out "$PREPACKAGE_READY_MANIFEST_OUT"
  )
  manifest_rc=$?
  set -e
  if [[ ! -f "$PREPACKAGE_READY_MANIFEST_OUT" ]]; then
    echo "ERROR: checkpoint override did not produce pre-package manifest: $PREPACKAGE_READY_MANIFEST_OUT" >&2
    exit "${manifest_rc:-1}"
  fi
  if [[ "$manifest_rc" -ne 0 ]]; then
    echo "WARNING: checkpoint pre-package manifest exited $manifest_rc; continuing only because override is explicit." >&2
  fi
else
  (
    cd "$ROOT_DIR"
    "$PYTHON_BIN" "tests/cross_matrix/run_release_regression_manifest.py" \
      --require-prepackage-ready \
      --out "$PREPACKAGE_READY_MANIFEST_OUT"
  )
fi

sign_bundled_python_native_files() {
  local bundled_python="$1"
  local identity="$2"

  if [[ ! -d "$bundled_python" ]]; then
    echo "ERROR: missing bundled Python at $bundled_python" >&2
    exit 1
  fi

  echo "==> Signing bundled Python native files with release identity"
  local signed_count=0
  while IFS= read -r native_file; do
    if file "$native_file" | grep -q "Mach-O"; then
      codesign --force --timestamp --options runtime --sign "$identity" "$native_file" >/dev/null
      signed_count=$((signed_count + 1))
    fi
  done < <(find "$bundled_python" -type f \( -name "*.dylib" -o -name "*.so" -o -perm +111 \))
  echo "  signed $signed_count bundled Python native files"
}

finalize_release_app_signature() {
  local app_path="$1"
  local identity="${2:-$RELEASE_CODESIGN_IDENTITY}"
  local entitlements="$PANEL_DIR/build/entitlements.mac.plist"

  if [[ ! -d "$app_path" ]]; then
    echo "ERROR: missing staged app at $app_path" >&2
    exit 1
  fi
  if [[ ! -f "$entitlements" ]]; then
    echo "ERROR: missing release entitlements at $entitlements" >&2
    exit 1
  fi

  local bundled_python="$app_path/Contents/Resources/bundled-python"
  if [[ -d "$bundled_python" ]]; then
    echo "==> Removing Python bytecode before release app seal"
    find "$bundled_python" -name "*.pyc" -type f -delete
    find "$bundled_python" -name "__pycache__" -type d -prune -exec rm -rf {} +
  fi

  sign_bundled_python_native_files "$bundled_python" "$identity"
  echo "==> Final release app seal/signature: $app_path"
  codesign --force --deep --timestamp --options runtime --entitlements "$entitlements" --sign "$identity" "$app_path"
  codesign --verify --deep --strict --verbose=2 "$app_path"
}

find_staged_app() {
  local staged_output="$1"
  local app_path

  app_path="$(find "$staged_output/mac-arm64" -maxdepth 2 -name "vMLX.app" -type d 2>/dev/null | head -1)"
  if [[ -z "$app_path" ]]; then
    app_path="$(find "$staged_output" -maxdepth 3 -name "vMLX.app" -type d | head -1)"
  fi
  if [[ -z "$app_path" ]]; then
    echo "ERROR: electron-builder did not produce a staged vMLX.app in $staged_output" >&2
    exit 1
  fi
  printf '%s\n' "$app_path"
}

build_one() {
  local flavor="$1"
  local platform="$2"
  local wheel_tag
  local staged_output="$DIST_DIR/${flavor}-app"
  local app_path

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
  rm -rf "$staged_output"
  npx electron-builder --mac --dir \
    --config.directories.output="$staged_output"
  app_path="$(find_staged_app "$staged_output")"
  finalize_release_app_signature "$app_path" "$RELEASE_CODESIGN_IDENTITY"
  npx electron-builder --mac dmg \
    --prepackaged "$app_path" \
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
