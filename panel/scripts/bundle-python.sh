#!/bin/bash
set -euo pipefail

# Build a relocatable Python environment with all vllm-mlx dependencies.
# Run once on dev machine before `npm run dist`.
# Output: panel/bundled-python/python/ (~1-2 GB)

PYTHON_VERSION="3.12.12"
BUILD_DATE="20260211"
ARCH="aarch64-apple-darwin"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="/Users/eric/mlx/vllm-mlx"
BUNDLE_DIR="$PANEL_DIR/bundled-python"

echo "==> Bundling Python $PYTHON_VERSION for standalone vMLX distribution"

# Clean previous build
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

# Download python-build-standalone (Astral's relocatable Python builds)
TARBALL="cpython-${PYTHON_VERSION}+${BUILD_DATE}-${ARCH}-install_only.tar.gz"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/${BUILD_DATE}/${TARBALL}"
echo "==> Downloading Python ${PYTHON_VERSION}..."
curl -L "$URL" | tar xz -C "$BUNDLE_DIR"

PYTHON="$BUNDLE_DIR/python/bin/python3"

# Verify Python works
"$PYTHON" --version

# Upgrade pip
echo "==> Upgrading pip..."
"$PYTHON" -m pip install --upgrade pip

# Install ALL dependencies (lean: no gradio, no dev tools, no pytz)
# Uses opencv-python-headless instead of opencv-python (no GUI deps, smaller)
echo "==> Installing dependencies..."
"$PYTHON" -m pip install \
  "mlx>=0.29.0" "mlx-lm>=0.30.2" "mlx-vlm>=0.1.0" \
  "transformers>=4.40.0" "tokenizers>=0.19.0" "huggingface-hub>=0.23.0" \
  "numpy>=1.24.0" "pillow>=10.0.0" \
  "opencv-python-headless>=4.8.0" \
  "fastapi>=0.100.0" "uvicorn>=0.23.0" \
  "mcp>=1.0.0" "jsonschema>=4.0.0" \
  "psutil>=5.9.0" "tqdm>=4.66.0" "pyyaml>=6.0" \
  "requests>=2.28.0" "tabulate>=0.9.0" "mlx-embeddings>=0.0.5"

# Install our customized vllm-mlx from source (--no-deps since all deps already installed)
echo "==> Installing vllm-mlx from source..."
"$PYTHON" -m pip install --no-deps "$REPO_DIR"

# Verify it works
echo "==> Verifying installation..."
"$PYTHON" -c "import vllm_mlx; print(f'vllm_mlx {vllm_mlx.__version__} imported OK')"
"$PYTHON" -m vllm_mlx.cli --help > /dev/null 2>&1 && echo "CLI OK"

# Clean up to reduce size
echo "==> Cleaning up..."
SITE="$BUNDLE_DIR/python/lib/python3.12/site-packages"

# Python bytecode (regenerated on import)
find "$BUNDLE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$BUNDLE_DIR" -name "*.pyc" -delete 2>/dev/null || true

# Unused stdlib modules
rm -rf "$BUNDLE_DIR/python/lib/python3.12/test"
rm -rf "$BUNDLE_DIR/python/lib/python3.12/ensurepip"
rm -rf "$BUNDLE_DIR/python/lib/python3.12/idlelib"
rm -rf "$BUNDLE_DIR/python/lib/python3.12/tkinter"
rm -rf "$BUNDLE_DIR/python/lib/python3.12/turtle"*
rm -rf "$BUNDLE_DIR/python/share" 2>/dev/null || true
# Unused .so for removed stdlib (tkinter)
rm -f "$BUNDLE_DIR/python/lib/python3.12/lib-dynload/_tkinter"*.so 2>/dev/null || true

# Test suites in site-packages (~80+ MB of test data never used at runtime)
find "$SITE" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find "$SITE" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# Packages not needed at runtime (transitive deps / dev-only tools)
# torch/torchvision: vllm-mlx uses MLX, not PyTorch. torch's bundled copy is
# broken (missing torchgen) and causes ModuleNotFoundError on startup.
# transformers gracefully degrades without torch via is_torch_available().
rm -rf "$SITE/torch" "$SITE/torch-"*.dist-info 2>/dev/null || true
rm -rf "$SITE/torchvision" "$SITE/torchvision-"*.dist-info 2>/dev/null || true
rm -rf "$SITE/torchgen" "$SITE/torchgen-"*.dist-info 2>/dev/null || true
rm -rf "$SITE/_soundfile_data" 2>/dev/null || true     # audio sample files (~2.9 MB)
rm -rf "$SITE/setuptools" 2>/dev/null || true          # build tool, not needed at runtime (~4.2 MB)
rm -rf "$SITE/setuptools"*.dist-info 2>/dev/null || true

# Keep pip (needed for engine auto-update at runtime via python3 -m pip)
# Only trim pip's vendored cache module
rm -rf "$SITE/pip/_vendor/cachecontrol" 2>/dev/null || true

echo ""
echo "==> Bundle size:"
du -sh "$BUNDLE_DIR"
echo ""
echo "==> Done! Bundled Python ready at: $BUNDLE_DIR"
echo "    Next: npm run build && npx electron-builder --mac"
