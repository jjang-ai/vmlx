#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL_DIR="$(dirname "$SCRIPT_DIR")"
APP_NAME="vMLX.app"
DEST="/Applications/$APP_NAME"

cd "$PANEL_DIR"

sign_bundled_python_native_files() {
  local bundled_python="$1"
  local identity="$2"

  if [ ! -d "$bundled_python" ]; then
    return
  fi

  echo "==> Signing bundled Python native files: $bundled_python"
  local signed_count=0
  while IFS= read -r native_file; do
    if file "$native_file" | grep -q "Mach-O"; then
      codesign --force --sign "$identity" "$native_file" >/dev/null
      signed_count=$((signed_count + 1))
    fi
  done < <(find "$bundled_python" -type f \( -name "*.dylib" -o -name "*.so" -o -perm +111 \))
  echo "  signed $signed_count bundled Python native files"
}

finalize_local_app_signature() {
  local app_path="$1"
  local identity="${VMLINUX_INSTALL_CODESIGN_IDENTITY:--}"

  if [ ! -d "$app_path" ]; then
    echo "ERROR: Cannot finalize missing app: $app_path"
    exit 1
  fi

  local bundled_python="$app_path/Contents/Resources/bundled-python"
  if [ -d "$bundled_python" ]; then
    echo "==> Removing Python bytecode before app seal: $app_path"
    find "$bundled_python" -name "*.pyc" -type f -delete
    find "$bundled_python" -name "__pycache__" -type d -prune -exec rm -rf {} +
  fi

  # This script installs a local app-dir build, not the notarized release DMG.
  # Electron-builder may sign the .app before later copy/xattr/resource changes.
  # Re-seal at the end so spawned bundled-Python native libraries do not trip
  # macOS code-signing validation as "unsigned" or modified resources.
  if [ "${VMLINUX_INSTALL_SKIP_FINAL_SIGN:-0}" != "1" ]; then
    sign_bundled_python_native_files "$bundled_python" "$identity"
    echo "==> Final app seal/signature: $app_path"
    codesign --force --deep --sign "$identity" "$app_path"
  else
    echo "==> Skipping final app sign because VMLX_INSTALL_SKIP_FINAL_SIGN=1"
  fi

  echo "==> Verifying app signature: $app_path"
  codesign --verify --deep --strict --verbose=2 "$app_path"

  # Local unsigned/ad-hoc installs are intended to be runnable for smoke tests.
  # Notarized release artifacts are produced by the release pipeline, not here.
  xattr -dr com.apple.quarantine "$app_path" 2>/dev/null || true
}

# ─── Pre-build checklist ──────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              vMLX Pre-Build Compatibility Check             ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Check for bundled Python (required for standalone distribution)
echo ""
echo "  Bundled Python..."
if [ -d "bundled-python/python/bin" ]; then
  BUNDLE_SIZE=$(du -sh bundled-python 2>/dev/null | cut -f1)
  echo "  [PASS] Bundled Python found ($BUNDLE_SIZE)"
else
  echo "  [WARN] No bundled Python found. App will require system Python."
  echo "         Run: bash scripts/bundle-python.sh"
fi
echo ""
echo "  TypeScript compilation..."
if npm run typecheck 2>/dev/null; then
  echo "  [PASS] TypeScript: no type errors"
else
  echo "  [FAIL] TypeScript: type errors found"
  echo "  Fix type errors before building."
  exit 1
fi

# Check Python server syntax
echo ""
echo "  Python server syntax..."
PYTHON_ERRORS=0
REPO_DIR="$(dirname "$PANEL_DIR")"
for f in $(find "$REPO_DIR/vmlx_engine" -name "*.py" 2>/dev/null); do
  if ! python3 -c "import py_compile; py_compile.compile('$f', doraise=True)" 2>/dev/null; then
    echo "  [FAIL] Syntax error: $f"
    PYTHON_ERRORS=1
  fi
done
if [ "$PYTHON_ERRORS" -eq 0 ]; then
  echo "  [PASS] Python: no syntax errors"
else
  echo "  Fix Python syntax errors before building."
  exit 1
fi

# Check model registry consistency
echo ""
echo "  Model registry sync..."
TS_FAMILIES=$(grep -c "family:" "$PANEL_DIR/src/main/model-config-registry.ts" 2>/dev/null || echo "0")
PY_FAMILIES=$(grep -c "family_name=" "$REPO_DIR/vmlx_engine/model_configs.py" 2>/dev/null || echo "0")
echo "  TypeScript families: $TS_FAMILIES | Python families: $PY_FAMILIES"
if [ "$TS_FAMILIES" -gt 0 ] && [ "$PY_FAMILIES" -gt 0 ]; then
  echo "  [PASS] Both registries have model families"
else
  echo "  [WARN] Registry mismatch — check model-config-registry.ts vs model_configs.py"
fi

# Check for editable installs in bundled Python (CRITICAL: prevents shipping dev paths)
echo ""
echo "  Editable install check..."
if [ -d "bundled-python/python/lib/python3.12/site-packages" ]; then
  EDITABLE=$(find bundled-python/python/lib/python3.12/site-packages -maxdepth 1 -name "__editable__.*" -o -name "__editable___*" 2>/dev/null)
  if [ -n "$EDITABLE" ]; then
    echo "  [FAIL] Editable install found in bundled Python!"
    echo "         $EDITABLE"
    echo "         This ships hardcoded dev paths. Re-run: bash scripts/bundle-python.sh"
    exit 1
  else
    echo "  [PASS] No editable installs in bundled Python"
  fi
fi

# Check critical API field parity
echo ""
echo "  API field parity..."
RESP_FIELDS=$(grep -c "stream_options\|enable_thinking" "$REPO_DIR/vmlx_engine/api/models.py" 2>/dev/null || echo "0")
if [ "$RESP_FIELDS" -ge 3 ]; then
  echo "  [PASS] ResponsesRequest has stream_options + enable_thinking"
else
  echo "  [WARN] ResponsesRequest may be missing fields (found $RESP_FIELDS matches)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  All checks passed!                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─── Build ─────────────────────────────────────────────────────────────
echo "==> Installing dependencies..."
npm install

echo "==> Building app..."
npm run build

echo "==> Packaging for macOS..."
npx electron-builder --mac --dir

# Find the built .app in release/ — prefer mac-arm64 over mas (sandboxed)
APP_PATH=$(find release/mac-arm64 -name "$APP_NAME" -type d -maxdepth 2 2>/dev/null | head -1)
if [ -z "$APP_PATH" ]; then
  APP_PATH=$(find release -name "$APP_NAME" -type d -maxdepth 3 | head -1)
fi

if [ -z "$APP_PATH" ]; then
  echo "ERROR: Could not find $APP_NAME in release/"
  exit 1
fi

echo "==> Found: $APP_PATH"

finalize_local_app_signature "$APP_PATH"

# Stop running instances
echo "==> Stopping running instances..."
pkill -f "$APP_NAME" 2>/dev/null || true
pkill -f "vmlx-engine" 2>/dev/null || true
sleep 2

# Remove old installation if it exists
if [ -d "$DEST" ]; then
  echo "==> Removing existing $DEST"
  rm -rf "$DEST"
fi

echo "==> Copying to /Applications/"
cp -R "$APP_PATH" "$DEST"

finalize_local_app_signature "$DEST"

echo "==> Done! $APP_NAME installed to /Applications/"
echo "    Launch it from Spotlight or: open /Applications/$APP_NAME"
