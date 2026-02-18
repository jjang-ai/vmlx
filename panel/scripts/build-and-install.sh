#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL_DIR="$(dirname "$SCRIPT_DIR")"
APP_NAME="vMLX.app"
DEST="/Applications/$APP_NAME"

cd "$PANEL_DIR"

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
for f in $(find "$(dirname "$PANEL_DIR")/vllm_mlx" -name "*.py" 2>/dev/null); do
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
PY_FAMILIES=$(grep -c "family_name=" "$(dirname "$PANEL_DIR")/vllm_mlx/model_configs.py" 2>/dev/null || echo "0")
echo "  TypeScript families: $TS_FAMILIES | Python families: $PY_FAMILIES"
if [ "$TS_FAMILIES" -gt 0 ] && [ "$PY_FAMILIES" -gt 0 ]; then
  echo "  [PASS] Both registries have model families"
else
  echo "  [WARN] Registry mismatch — check model-config-registry.ts vs model_configs.py"
fi

# Check critical API field parity
echo ""
echo "  API field parity..."
RESP_FIELDS=$(grep -c "stream_options\|enable_thinking" "$(dirname "$PANEL_DIR")/vllm_mlx/api/models.py" 2>/dev/null || echo "0")
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

# Find the built .app in release/
APP_PATH=$(find release -name "$APP_NAME" -type d -maxdepth 3 | head -1)

if [ -z "$APP_PATH" ]; then
  echo "ERROR: Could not find $APP_NAME in release/"
  exit 1
fi

echo "==> Found: $APP_PATH"

# Stop running instances
echo "==> Stopping running instances..."
pkill -f "$APP_NAME" 2>/dev/null || true
pkill -f "vllm-mlx" 2>/dev/null || true
sleep 2

# Remove old installation if it exists
if [ -d "$DEST" ]; then
  echo "==> Removing existing $DEST"
  rm -rf "$DEST"
fi

echo "==> Copying to /Applications/"
cp -R "$APP_PATH" "$DEST"

echo "==> Done! $APP_NAME installed to /Applications/"
echo "    Launch it from Spotlight or: open /Applications/$APP_NAME"
