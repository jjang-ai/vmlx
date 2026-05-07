#!/bin/bash
set -euo pipefail

# Build a relocatable Python environment with all vmlx-engine dependencies.
# Run once on dev machine before `npm run dist`.
# Output: panel/bundled-python/python/ (~1-2 GB)

PYTHON_VERSION="3.12.12"
BUILD_DATE="20260211"
ARCH="aarch64-apple-darwin"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PANEL_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$PANEL_DIR")"
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
#
# Build reproducibility note:
# On macOS Tahoe, pip prefers MLX wheels tagged macosx_26_0_arm64. Shipping
# that wheel in a DMG whose Info.plist allows Sonoma/Sequoia reopens the
# mlxstudio#104 "Metal language version 4.0 unsupported" crash. The same MLX
# release publishes macosx_14_0_arm64 wheels, so force those for the bundle.
MLX_VERSION="0.31.2"
MLX_LM_VERSION="0.31.3"
MLX_VLM_VERSION="0.4.4"
MFLUX_VERSION="0.17.5"
MLX_WHEEL_PLATFORM="${VMLX_BUNDLE_MLX_PLATFORM:-macosx_14_0_arm64}"
WHEELHOUSE="$BUNDLE_DIR/wheelhouse"
mkdir -p "$WHEELHOUSE"
echo "==> Installing MLX $MLX_VERSION wheels for $MLX_WHEEL_PLATFORM..."
"$PYTHON" -m pip download --only-binary=:all: --no-deps \
  --dest "$WHEELHOUSE" \
  --platform "$MLX_WHEEL_PLATFORM" \
  --implementation cp --python-version 312 --abi cp312 \
  "mlx==$MLX_VERSION"
"$PYTHON" -m pip download --only-binary=:all: --no-deps \
  --dest "$WHEELHOUSE" \
  --platform "$MLX_WHEEL_PLATFORM" \
  --implementation py --python-version 312 --abi none \
  "mlx-metal==$MLX_VERSION"
"$PYTHON" -m pip install "$WHEELHOUSE"/mlx-"$MLX_VERSION"-*.whl "$WHEELHOUSE"/mlx_metal-"$MLX_VERSION"-*.whl

echo "==> Installing dependencies..."
"$PYTHON" -m pip install \
  "mlx==$MLX_VERSION" "mlx-lm==$MLX_LM_VERSION" "mlx-vlm==$MLX_VLM_VERSION" \
  "transformers>=4.40.0" "tokenizers>=0.19.0" "huggingface-hub>=0.23.0" \
  "numpy>=1.24.0" "pillow>=10.0.0" \
  "opencv-python-headless>=4.8.0" \
  "fastapi>=0.100.0" "uvicorn>=0.23.0" \
  "mcp>=1.0.0" "jsonschema>=4.0.0" \
  "psutil>=5.9.0" "tqdm>=4.66.0" "pyyaml>=6.0" \
  "requests>=2.28.0" "tabulate>=0.9.0" "mlx-embeddings>=0.0.5" \
  "tiktoken>=0.7.0" \
  "soundfile>=0.12" \
  "mflux==$MFLUX_VERSION" \
  "timm>=1.0.20"  # Kimi K2.6 tokenizer + Nemotron-Omni audio (soundfile) + Omni RADIO ViT (timm)

# Install mlx-audio for STT/TTS (--no-deps: it pins exact mlx-lm/transformers versions
# that conflict with ours — we already have all the real deps above)
echo "==> Installing mlx-audio (STT/TTS)..."
"$PYTHON" -m pip install --no-deps "mlx-audio>=0.2.0"
# Install mlx-audio's transitive deps that we don't already have
"$PYTHON" -m pip install \
  librosa sounddevice miniaudio pyloudnorm numba

# Install vmlx-engine from PyPI (the local repo's pyproject.toml has a
# `setuptools.package-dir = engine` declaration that breaks `pip install
# .` here because the panel dir lives at panel/, not engine/. PyPI 1.4.3
# is a clean wheel with the same source so this is equivalent.)
#
# jang_tools — install from LOCAL source path, not PyPI. The published
# `jang` wheel lags the local development of jang_tools.dsv4 modules
# (pool_quant_cache.py, fused_pool_attn.py, fused_pool_attn_kernel.py,
# build_role_codebooks.py landed in jang 2.5.10 but only jang 2.5.9 was
# on PyPI as of v1.5.2 build, causing
# `ModuleNotFoundError: No module named 'jang_tools.dsv4.pool_quant_cache'`
# the moment a DSV4-Flash bundle was loaded). Pinning to local source
# ensures every DMG ships with whatever DSV4 runtime the engine actually
# needs. Falls back to PyPI if the local path doesn't exist (CI builds).
echo "==> Installing vmlx-engine + jang_tools (local source)..."
# Install vmlx-engine from local source so the bundle ships current fixes,
# not the lagging PyPI release. Falls back to PyPI if the local path is
# absent (CI builds). 2026-05-03: previous pin to PyPI vmlx==1.4.3
# silently overwrote shipped F1-F13 fixes on every rebuild.
VMLX_LOCAL="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -f "$VMLX_LOCAL/pyproject.toml" ] && [ -d "$VMLX_LOCAL/vmlx_engine" ]; then
  echo "    using local vmlx at $VMLX_LOCAL"
  "$PYTHON" -m pip install --no-deps "$VMLX_LOCAL"
else
  echo "    local vmlx missing, falling back to PyPI"
  "$PYTHON" -m pip install --no-deps "vmlx>=1.5.24"
fi
JANG_LOCAL="$HOME/jang/jang-tools"
if [ -f "$JANG_LOCAL/pyproject.toml" ]; then
  echo "    using local jang-tools at $JANG_LOCAL"
  "$PYTHON" -m pip install --no-deps "$JANG_LOCAL"
else
  echo "    local jang-tools missing, falling back to PyPI"
  "$PYTHON" -m pip install --no-deps "jang>=2.5.26"
fi

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
# torch/torchvision/torchgen: KEEP (2026-05-03 reversal of earlier strip).
# Earlier comment claimed "vmlx-engine uses MLX, not PyTorch" — true for the
# core LLM path, but VL bundles tell another story:
#   - Nemotron-Omni Stage-1 (default) bridges PT encoders to MLX LLM via
#     `jang_tools.nemotron_omni_chat.OmniChat` which calls
#     `AutoModel.from_pretrained(..., torch_dtype=bfloat16)`.
#   - Qwen3-VL / Qwen3.5-VL: transformers loads `Qwen3VLVideoProcessor`
#     eagerly via `AutoProcessor.from_pretrained`; that class hard-requires
#     torch+torchvision and raises a confusing `ImportError` upstream
#     that gets re-wrapped as "mlx-vlm is required" in mllm.py.
#   - mlx_vlm itself imports torch helpers in some processor paths.
# Stripping torch broke every VL bundle the moment a video processor or
# PT encoder loaded. Keep them. Cost: ~700 MB extra bundle weight.
# pyproject.toml already declares them as hard deps (`torch>=2.3.0`,
# `torchvision>=0.18.0`) — pip install -e . pulls them in cleanly.
# soundfile: KEEP. Required for Nemotron-3-Nano-Omni audio path
# (jang_tools.nemotron_omni_chat → ParakeetEncoder mel features).
# Modern pip install ships libsndfile_arm64.dylib alongside _soundfile_data,
# so the older "missing libsndfile.dylib" runtime crash no longer applies.
# Verified working at bundle time: see verify-bundled-python.sh soundfile gate.
rm -rf "$SITE/setuptools" 2>/dev/null || true          # build tool, not needed at runtime (~4.2 MB)
rm -rf "$SITE/setuptools"*.dist-info 2>/dev/null || true
rm -rf "$WHEELHOUSE" 2>/dev/null || true

# Keep pip intact (needed for engine auto-update at runtime via python3 -m pip)
# NOTE: Do NOT remove pip/_vendor/* — pip 26+ requires cachecontrol, pygments,
# rich, and other vendored modules. Removing them breaks `python3 -m pip install`.
# Only safe to remove: pip's test directories.
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
	
# --- Patch: mlx_vlm/models/gemma4/vision.py (Gemma 4 multi-image input) ---
# mlxstudio#88: some Gemma 4 processors return `pixel_values` as a Python
# list containing numpy arrays. mlx.concatenate requires mx.array inputs.
echo "  Patching mlx_vlm/models/gemma4/vision.py (pixel_values list coercion)..."
"$PYTHON" -c "
path = '$SITE/mlx_vlm/models/gemma4/vision.py'
try:
    with open(path, 'r') as f:
        content = f.read()
    old = '''    def __call__(self, pixel_values: mx.array) -> mx.array:
        if isinstance(pixel_values, list):
            pixel_values = mx.concatenate(pixel_values, axis=0)

        B, C, H, W = pixel_values.shape'''
    new = '''    def __call__(self, pixel_values: mx.array) -> mx.array:
        if isinstance(pixel_values, list):
            # mlxstudio#88: multi-image processors can hand us a Python list
            # containing numpy arrays instead of MLX arrays. Upstream
            # mx.concatenate only accepts mx.array inputs, so coerce each
            # element before concatenating.
            pixel_values = [
                v if isinstance(v, mx.array) else mx.array(v)
                for v in pixel_values
            ]
            pixel_values = mx.concatenate(pixel_values, axis=0)
        elif not isinstance(pixel_values, mx.array):
            pixel_values = mx.array(pixel_values)

        B, C, H, W = pixel_values.shape'''
    if old in content:
        content = content.replace(old, new)
        with open(path, 'w') as f:
            f.write(content)
        print('  Patched: gemma4 vision pixel_values list coercion')
    elif 'mlxstudio#88' in content and 'isinstance(v, mx.array)' in content:
        print('  Already patched: gemma4 vision pixel_values list coercion')
    else:
        print('  WARNING: gemma4 vision patch target not found')
except FileNotFoundError:
    print('  Skipped: gemma4 vision.py not found')
"

# --- Patch: mlx_lm MLA fp32 SDPA absorb (DeepSeek V3 / V3.2 / Mistral 4) ---
# The L==1 decode path for MLA-family models needs the q/k/v/mask cast to
# fp32 before scaled_dot_product_attention or the absorb branch produces
# logit drift. Upstream mlx_lm has carried this fix in some 0.31.x
# releases but regressed it in 0.31.3, so vMLX vendors the patched files
# at panel/scripts/patches/*.patched.py and copies them in at bundle time.
# Rationale: this is a CONTROLLED FORK of three upstream files (Apache-2.0,
# attribution preserved), not a runtime monkeypatch. Each rebuild lands
# the deterministic patched source.
PATCH_DIR="$REPO_DIR/panel/scripts/patches"
for entry in \
  "deepseek_v3.patched.py:mlx_lm/models/deepseek_v3.py:Kimi K2.6 fp32 MLA decode" \
  "deepseek_v32.patched.py:mlx_lm/models/deepseek_v32.py:DSV3.2 fp32 MLA decode" \
  "mistral4.patched.py:mlx_lm/models/mistral4.py:Mistral 4 fp32 MLA decode"; do
  IFS=":" read -r src rel desc <<< "$entry"
  src_path="$PATCH_DIR/$src"
  dst_path="$SITE/$rel"
  if [ ! -f "$src_path" ]; then
    echo "ERROR: required patch source missing: $src_path" >&2
    exit 1
  fi
  echo "  Installing $rel ($desc)..."
  mkdir -p "$(dirname "$dst_path")"
  cp "$src_path" "$dst_path"
  if ! grep -q 'q_sdpa.*astype(mx\.float32)' "$dst_path"; then
    echo "ERROR: $rel post-install does not contain q_sdpa fp32 cast" >&2
    exit 1
  fi
done
# Stage a copy under bundle research/ for build-traceability (matches the
# v3 historical layout so existing diagnostics keep working).
mkdir -p "$BUNDLE_DIR/python/lib/python3.12/research"
cp "$PATCH_DIR/deepseek_v3.patched.py" \
   "$BUNDLE_DIR/python/lib/python3.12/research/deepseek_v3_patched.py"

# --- New model class: bailing_hybrid (Ling-2.6-flash / Bailing-V2.5) ---
# mlx-lm 0.31.3 doesn't ship a model class for `bailing_hybrid` model_type
# yet; vMLX vendors the validated implementation under panel/scripts/patches/.
# This is a NEW file, not a patch over an upstream one — bundle-python.sh
# just copies it in. Hard-fail if the file is missing.
BAILING_SRC="$PATCH_DIR/bailing_hybrid.patched.py"
BAILING_DST="$SITE/mlx_lm/models/bailing_hybrid.py"
if [ ! -f "$BAILING_SRC" ]; then
  echo "ERROR: required source missing: $BAILING_SRC" >&2
  exit 1
fi
echo "  Installing mlx_lm/models/bailing_hybrid.py (Ling-2.6 / Bailing-V2.5 hybrid)..."
cp "$BAILING_SRC" "$BAILING_DST"
if ! grep -q 'class LanguageModel' "$BAILING_DST"; then
  echo "ERROR: bailing_hybrid post-install missing LanguageModel class" >&2
  exit 1
fi

# --- Patch: mlx_lm/models/ssm.py (Mamba/Nemotron-H hybrid state space model) ---
# Fix 1: mx.clip(dt, ...) upper-clips dt values, corrupting Mamba state transitions.
#         Replace with mx.maximum(dt, time_step_limit[0]) — only lower-clip.
# Fix 2: output_dtypes=[input_type, input_type] stores SSM state in bfloat16,
#         causing precision loss. State must be float32.
echo "  Patching mlx_lm/models/ssm.py (Mamba state fixes)..."
python3 -c "
import os, glob
base = '$BUNDLE_DIR/python/lib/python3.*/site-packages/mlx_lm/models/ssm.py'
paths = glob.glob(base)
if not paths:
    print('  Skipped: ssm.py not found')
else:
    path = paths[0]
    with open(path, 'r') as f:
        content = f.read()
    changed = False
    # Fix 1: clip -> maximum (line 10)
    old1 = 'return mx.clip(dt, time_step_limit[0], time_step_limit[1])'
    new1 = 'return mx.maximum(dt, time_step_limit[0])'
    if old1 in content:
        content = content.replace(old1, new1)
        changed = True
        print('  Patched: ssm.py dt clip -> maximum')
    else:
        print('  Already patched or structure changed: ssm.py dt fix')
    # Fix 2: state output dtype must be float32
    old2 = 'output_dtypes=[input_type, input_type]'
    new2 = 'output_dtypes=[input_type, mx.float32]'
    if old2 in content:
        content = content.replace(old2, new2)
        changed = True
        print('  Patched: ssm.py state dtype -> float32')
    else:
        print('  Already patched or structure changed: ssm.py dtype fix')
    if changed:
        with open(path, 'w') as f:
            f.write(content)
"

echo "==> Patches applied."

# Verify imports only after dependency patches are baked in. Importing
# vmlx_engine/CLI earlier can trigger runtime patch checks against a half-built
# bundle and produce false "patched source not found" diagnostics.
echo "==> Verifying installation..."
"$PYTHON" -c "import vmlx_engine; print(f'vmlx_engine {vmlx_engine.__version__} imported OK')"
"$PYTHON" -m vmlx_engine.cli --help > /dev/null 2>&1 && echo "CLI OK"

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

# Critical: reject editable installs (prevents shipping dev-machine paths to users)
echo "==> Checking for editable installs..."
EDITABLE_PTH=$(find "$SITE" -maxdepth 1 -name "__editable__.*" -o -name "__editable___*" 2>/dev/null)
if [ -n "$EDITABLE_PTH" ]; then
  echo "ERROR: Editable install detected in bundled Python!"
  echo "  Found: $EDITABLE_PTH"
  echo "  This would ship with hardcoded paths to your dev machine."
  echo "  Fix: re-run bundle-python.sh from scratch (it cleans the bundle dir)."
  exit 1
fi
echo "  No editable installs (good)"

# Verify path isolation
echo "==> Verifying path isolation..."
ENABLE_USER_SITE=$("$PYTHON" -s -c "import site; print(site.ENABLE_USER_SITE)" 2>&1)
if [ "$ENABLE_USER_SITE" = "False" ]; then
  echo "  ENABLE_USER_SITE=False with -s flag OK"
else
  echo "WARNING: -s flag did not suppress user site-packages"
fi

echo ""
# Post-bundle: rewrite shebangs in console scripts to the install location.
# pip bakes the source bundled-python path into shebangs (e.g.
# `#!/Users/eric/mlx/vllm-mlx/panel/bundled-python/python/bin/python3`),
# which would ship to users and never resolve. Rewrite to the .app install
# path so terminal users running `vmlx-serve` directly work.
#
# Use env -S so the script entry points also run with -B -s. Without -B,
# direct console-script use writes __pycache__ into the signed .app bundle
# and invalidates the sealed Resources signature after first launch.
echo "==> Rewriting console-script shebangs to install path..."
INSTALL_PYTHON="/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
INSTALL_SHEBANG="/usr/bin/env -S $INSTALL_PYTHON -B -s"
SOURCE_PYTHON="$BUNDLE_DIR/python/bin/python3"
SHEBANG_FIXED=0
for SCRIPT in "$BUNDLE_DIR/python/bin/"*; do
  if [ -f "$SCRIPT" ] && head -c 2 "$SCRIPT" 2>/dev/null | grep -q '^#!'; then
    if head -1 "$SCRIPT" 2>/dev/null | grep -qF "$SOURCE_PYTHON"; then
      sed -i '' "1s|^#!$SOURCE_PYTHON\$|#!$INSTALL_SHEBANG|" "$SCRIPT"
      SHEBANG_FIXED=$((SHEBANG_FIXED + 1))
    fi
  fi
done
echo "  rewrote $SHEBANG_FIXED shebangs to $INSTALL_SHEBANG"
# Sanity check — no script should still reference the source path
LEAKED=$(grep -lF "$SOURCE_PYTHON" "$BUNDLE_DIR/python/bin/"* 2>/dev/null | head -3 || true)
if [ -n "$LEAKED" ]; then
  echo "ERROR: source path still in shebangs after rewrite:"
  echo "$LEAKED"
  exit 1
fi
echo "  no source-path leaks (good)"

echo ""
echo "==> Bundle size:"
du -sh "$BUNDLE_DIR"
echo ""
echo "==> Done! Bundled Python ready at: $BUNDLE_DIR"
echo "    Next: npm run build && npx electron-builder --mac"
