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
# soundfile: requires libsndfile.dylib which isn't bundled. Removing the package
# makes transformers.is_soundfile_available() return False and skip audio imports.
# mlx_vlm/utils.py is also patched to lazy-import soundfile for defense-in-depth.
rm -f "$SITE/soundfile.py" 2>/dev/null || true
rm -rf "$SITE/soundfile-"*.dist-info 2>/dev/null || true
rm -rf "$SITE/_soundfile"* 2>/dev/null || true
rm -rf "$SITE/setuptools" 2>/dev/null || true          # build tool, not needed at runtime (~4.2 MB)
rm -rf "$SITE/setuptools"*.dist-info 2>/dev/null || true

# Keep pip intact (needed for engine auto-update at runtime via python3 -m pip)
# NOTE: Do NOT remove pip/_vendor/* — pip 26+ requires cachecontrol and other
# vendored modules. Removing them breaks `python3 -m pip install`.
# Trim only safe targets: pip's test suite and docs.
rm -rf "$SITE/pip/_vendor/pygments" 2>/dev/null || true      # syntax highlighting, not needed for install
find "$SITE/pip" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true

# ====================================================================
# Patches for bundled dependencies (apply AFTER pip install, AFTER cleanup)
# These fix issues in transformers/mlx-vlm for torch-free environments.
# ====================================================================
echo "==> Applying bundled dependency patches..."

# 1. transformers/processing_utils.py: Allow None sub-processors (video_processor)
#    Without torchvision, Qwen2VL's video_processor loads as None. The type check
#    must allow None so image-only VLM usage works.
sed -i '' 's/if not isinstance(argument, proper_class):/if argument is not None and not isinstance(argument, proper_class):/' \
  "$SITE/transformers/processing_utils.py"

# 2. transformers/processing_utils.py: Skip ImportError when loading sub-processors
#    Video processor requires torchvision; gracefully skip when unavailable.
"$PYTHON" -c "
import re
path = '$SITE/transformers/processing_utils.py'
with open(path, 'r') as f:
    content = f.read()
# Wrap the auto_processor_class.from_pretrained call in try/except ImportError
old = '''            elif is_primary:
                # Primary non-tokenizer sub-processor: load via Auto class
                auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[sub_processor_type]
                sub_processor = auto_processor_class.from_pretrained(
                    pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                )
                args.append(sub_processor)'''
new = '''            elif is_primary:
                # Primary non-tokenizer sub-processor: load via Auto class
                auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[sub_processor_type]
                try:
                    sub_processor = auto_processor_class.from_pretrained(
                        pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                    )
                    args.append(sub_processor)
                except ImportError:
                    # Skip sub-processors that need unavailable backends (e.g. video needs torchvision)
                    pass'''
if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('  Patched: processing_utils.py sub-processor ImportError handling')
else:
    print('  Already patched or structure changed: processing_utils.py sub-processor')
"

# 3. transformers/models/auto/video_processing_auto.py: Null check for extractors
#    transformers 5.2.0 bug where extractors can be None
sed -i '' 's/if class_name in extractors:/if extractors is not None and class_name in extractors:/' \
  "$SITE/transformers/models/auto/video_processing_auto.py" 2>/dev/null || true

# 4. mlx_vlm/utils.py: Lazy-import soundfile (defense-in-depth)
#    Even after removing the soundfile package, patch the import to be lazy
#    in case soundfile gets pulled back in as a transitive dep.
sed -i '' 's/^import soundfile as sf$/# import soundfile as sf  # lazy-loaded: see _get_sf()/' \
  "$SITE/mlx_vlm/utils.py" 2>/dev/null || true

# 5. mlx_vlm/models/qwen3_5/language.py: Fix mRoPE dimension mismatch for MoE
#    mlx-vlm 0.3.12 bug: broadcasting with cos/sin can produce 5D tensors
"$PYTHON" -c "
path = '$SITE/mlx_vlm/models/qwen3_5/language.py'
try:
    with open(path, 'r') as f:
        content = f.read()
    old = '''    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = mx.concatenate([q_embed, q_pass], axis=-1)'''
    new = '''    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Fix mRoPE dimension mismatch for MoE models: broadcasting with cos/sin
    # can produce 5D tensors when q_pass is 4D (mlx-vlm 0.3.12 bug)
    if q_embed.ndim > q_pass.ndim and q_embed.ndim == 5:
        q_embed = q_embed[0]
        k_embed = k_embed[0]

    q_embed = mx.concatenate([q_embed, q_pass], axis=-1)'''
    if old in content:
        content = content.replace(old, new)
        with open(path, 'w') as f:
            f.write(content)
        print('  Patched: qwen3_5/language.py mRoPE dimension fix')
    else:
        print('  Already patched or structure changed: qwen3_5/language.py')
except FileNotFoundError:
    print('  Skipped: qwen3_5/language.py not found (model not in this mlx-vlm version)')
"

echo "==> Patches applied."

# ====================================================================
# Critical: Verify the Python shared library exists (prevents broken bundles)
# The bundled Python MUST include libpython3.12.dylib for the app to work.
# Without it, the app falls back to system Python which may have outdated or
# missing packages (e.g., mlx_vlm without qwen3_5_moe support).
# ====================================================================
echo "==> Verifying Python shared library..."
LIBPYTHON="$BUNDLE_DIR/python/lib/libpython3.12.dylib"
if [ -f "$LIBPYTHON" ]; then
  echo "  libpython3.12.dylib OK ($(du -h "$LIBPYTHON" | cut -f1))"
else
  # Check if it exists elsewhere in the bundle (some builds put it in different locations)
  FOUND=$(find "$BUNDLE_DIR" -name "libpython3.12*.dylib" 2>/dev/null | head -1)
  if [ -n "$FOUND" ]; then
    echo "  Found at: $FOUND — creating symlink"
    ln -sf "$FOUND" "$LIBPYTHON"
  else
    echo "ERROR: libpython3.12.dylib NOT FOUND in bundle!"
    echo "  The app will fall back to system Python, which may have outdated packages."
    echo "  This is a critical build issue — the bundle is incomplete."
    exit 1
  fi
fi

# Post-cleanup verification: ensure pip still works (catches vendor stripping bugs)
echo "==> Verifying pip is functional (needed for engine auto-update)..."
"$PYTHON" -s -m pip --version > /dev/null 2>&1 || { echo "ERROR: pip is broken after cleanup! Check vendor removals."; exit 1; }
echo "  pip OK"

# Verify path isolation
echo "==> Verifying path isolation..."
ENABLE_USER_SITE=$("$PYTHON" -s -c "import site; print(site.ENABLE_USER_SITE)" 2>&1)
if [ "$ENABLE_USER_SITE" = "False" ]; then
  echo "  ENABLE_USER_SITE=False with -s flag OK"
else
  echo "WARNING: -s flag did not suppress user site-packages"
fi

echo ""
echo "==> Bundle size:"
du -sh "$BUNDLE_DIR"
echo ""
echo "==> Done! Bundled Python ready at: $BUNDLE_DIR"
echo "    Next: npm run build && npx electron-builder --mac"
