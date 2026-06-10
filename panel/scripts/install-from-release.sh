#!/usr/bin/env bash
# Install vMLX.app to /Applications from an existing release build.
# Does NOT re-run bundle-python. Expects make release (or make app) first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL_DIR="$(dirname "$SCRIPT_DIR")"
APP_NAME="vMLX.app"
APP_DEST="${VMLX_APP_DEST:-${VMLINUX_APP_DEST:-/Applications/$APP_NAME}}"
RELEASE_DIR="$PANEL_DIR/release"

sign_bundled_python_native_files() {
  local bundled_python="$1"
  local identity="${2:--}"

  if [[ ! -d "$bundled_python" ]]; then
    return 0
  fi

  echo "==> Signing bundled Python native files"
  local signed_count=0
  while IFS= read -r native_file; do
    if file "$native_file" | grep -q "Mach-O"; then
      codesign --force --sign "$identity" "$native_file" >/dev/null
      signed_count=$((signed_count + 1))
    fi
  done < <(
    {
      find "$bundled_python" -type f \( -name "*.dylib" -o -name "*.so" \) 2>/dev/null
      find "$bundled_python/python/bin" -type f 2>/dev/null
    } | sort -u
  )
  echo "  signed $signed_count bundled Python native files"
}

finalize_local_app_signature() {
  local app_path="$1"
  local identity="${VMLINUX_INSTALL_CODESIGN_IDENTITY:-${VMLX_INSTALL_CODESIGN_IDENTITY:--}}"

  if [[ ! -d "$app_path" ]]; then
    echo "ERROR: missing app: $app_path" >&2
    exit 1
  fi

  local bundled_python="$app_path/Contents/Resources/bundled-python"
  if [[ -d "$bundled_python" ]]; then
    find "$bundled_python" -name "*.pyc" -type f -delete 2>/dev/null || true
    find "$bundled_python" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
  fi

  if [[ "${VMLINUX_INSTALL_SKIP_FINAL_SIGN:-${VMLX_INSTALL_SKIP_FINAL_SIGN:-0}}" != "1" ]]; then
    sign_bundled_python_native_files "$bundled_python" "$identity"
    codesign --force --deep --sign "$identity" "$app_path"
  fi
  codesign --verify --deep --strict --verbose=2 "$app_path"
  xattr -dr com.apple.quarantine "$app_path" 2>/dev/null || true
}

find_staged_app() {
  local candidate
  candidate="$(find "$RELEASE_DIR/mac-arm64" -maxdepth 2 -name "$APP_NAME" -type d 2>/dev/null | head -1)"
  if [[ -n "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi
  find "$RELEASE_DIR" -maxdepth 3 -name "$APP_NAME" -type d 2>/dev/null | head -1
}

install_app_tree() {
  local app_path="$1"

  echo "==> Stopping running instances"
  pkill -f "$APP_NAME" 2>/dev/null || true
  pkill -f "vmlx-engine" 2>/dev/null || true
  sleep 1

  if [[ -d "$APP_DEST" ]]; then
    echo "==> Removing existing $APP_DEST"
    rm -rf "$APP_DEST"
  fi

  echo "==> Installing to $APP_DEST"
  cp -R "$app_path" "$APP_DEST"
  finalize_local_app_signature "$APP_DEST"

  echo
  echo "✓ vMLX installed: $APP_DEST"
  echo "  Launch: open \"$APP_DEST\""
  if [[ ! -f "$RELEASE_DIR/SHA256SUMS" ]]; then
    echo "  Ship:     make release    (build DMG for distribution)"
  fi
}

if [[ ! -d "$RELEASE_DIR" ]]; then
  echo "ERROR: no panel/release/ directory. Run: make app" >&2
  exit 1
fi

APP_PATH="$(find_staged_app || true)"
if [[ -n "$APP_PATH" && -d "$APP_PATH" ]]; then
  echo "==> Using staged app: $APP_PATH"
  install_app_tree "$APP_PATH"
  exit 0
fi

DMG="$(find "$RELEASE_DIR" -maxdepth 1 -type f -name 'vMLX-*.dmg' | sort | tail -1)"
if [[ -z "$DMG" ]]; then
  echo "ERROR: no vMLX.app or vMLX-*.dmg under $RELEASE_DIR" >&2
  echo "       Run: make app   (local) or make release   (DMG)" >&2
  exit 1
fi

MOUNT="$(mktemp -d /tmp/vmlx-mount.XXXXXX)"
cleanup() {
  hdiutil detach "$MOUNT" -quiet 2>/dev/null || true
  rmdir "$MOUNT" 2>/dev/null || true
}
trap cleanup EXIT

echo "==> Mounting $DMG"
hdiutil attach "$DMG" -mountpoint "$MOUNT" -nobrowse -quiet
APP_PATH="$(find "$MOUNT" -maxdepth 2 -name "$APP_NAME" -type d | head -1)"
if [[ -z "$APP_PATH" ]]; then
  echo "ERROR: $APP_NAME not found inside $DMG" >&2
  exit 1
fi
install_app_tree "$APP_PATH"
