#!/usr/bin/env bash
# Release-time sanity check: bundled-python must have all critical model
# modules that vMLX depends on. Runs before electron-builder packages the
# .app so we never ship a DMG that instantly ModuleNotFoundErrors on a
# model the user tries to load.
#
# Added after a user reported `ModuleNotFoundError: No module named
# 'mlx_vlm.models.gemma4'` on a fresh install — the bundled mlx_vlm 0.4.0
# had the gemma4 dir cherry-picked in at some point and we want to make
# sure we never regress the cherry-pick on a future rebuild.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PANEL="$(dirname "$HERE")"
PY="$PANEL/bundled-python/python/bin/python3"

if [ ! -x "$PY" ]; then
  echo "❌ bundled python missing: $PY"
  exit 1
fi

# Isolated imports — no user site, no PYTHONPATH leakage (same env as the
# running engine). -s suppresses user site-packages the way sessions.ts does.
PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" -s - <<'PYEOF'
import sys

REQUIRED = [
    # (import name, human label, remediation hint)
    ("mlx", "mlx core", "bundled mlx package broken"),
    ("mlx.nn", "mlx.nn", "bundled mlx package broken"),
    ("mlx_lm", "mlx-lm", "bundled mlx-lm package broken"),
    ("mlx_vlm", "mlx-vlm", "bundled mlx-vlm package broken"),
    ("mlx_vlm.models.gemma4", "mlx-vlm gemma4", "cherry-picked gemma4 dir missing or incomplete — re-sync from an mlx-vlm wheel that has it"),
    ("mlx_vlm.models.gemma3", "mlx-vlm gemma3", "bundled mlx-vlm gemma3 missing"),
    ("mlx_vlm.models.qwen3_vl", "mlx-vlm qwen3_vl", "bundled mlx-vlm qwen3_vl missing"),
    ("jang_tools", "jang-tools", "bundled jang-tools package missing"),
    ("jang_tools.load_jangtq", "jang_tools.load_jangtq", "JANGTQ fast-path loader missing from bundled jang-tools"),
    ("jang_tools.turboquant.tq_kernel", "jang_tools.turboquant.tq_kernel", "TQ Metal kernel runtime missing from bundled jang-tools"),
    ("jang_tools.turboquant.hadamard_kernel", "hadamard_kernel", "P3 Hadamard kernel missing"),
    ("jang_tools.turboquant.fused_gate_up_kernel", "fused_gate_up_kernel", "P17 fused kernel missing"),
    ("jang_tools.turboquant.gather_tq_kernel", "gather_tq_kernel", "P17 gather kernel missing"),
    # Kimi K2.6 runtime — research/KIMI-K2.6-VMLX-INTEGRATION.md §1.1
    ("jang_tools.load_jangtq_kimi_vlm", "jang_tools.load_jangtq_kimi_vlm", "Kimi VL loader missing (kimi_k25 remap + wired_limit + command-buffer split)"),
    ("jang_tools.kimi_prune.generate_vl", "jang_tools.kimi_prune.generate_vl", "Kimi chunked VL generate path missing — required by vmlx_engine.vlm.generate_vl"),
    ("vmlx_engine", "vmlx_engine", "bundled vmlx_engine missing"),
    ("vmlx_engine.utils.jang_loader", "vmlx_engine jang_loader", "bundled jang_loader missing"),
    ("vmlx_engine.api.ollama_adapter", "vmlx_engine ollama_adapter", "bundled ollama_adapter missing"),
    # Doc §1.3 + §1.4 import paths — shipping these means the Kimi integration
    # doc's code examples work verbatim on the shipped DMG, not just in dev.
    ("vmlx_engine.loaders.load_jangtq", "vmlx_engine.loaders.load_jangtq", "§1.3 text loader re-export missing"),
    ("vmlx_engine.loaders.load_jangtq_vlm", "vmlx_engine.loaders.load_jangtq_vlm", "§1.1 shared VLM loader re-export missing"),
    ("vmlx_engine.loaders.load_jangtq_kimi_vlm", "vmlx_engine.loaders.load_jangtq_kimi_vlm", "§1.4 Kimi VL loader re-export missing"),
    ("vmlx_engine.vlm.generate_vl", "vmlx_engine.vlm.generate_vl", "§1.4 chunked-prefill generate_vl re-export missing"),
    ("vmlx_engine.runtime_patches.kimi_k25_mla", "vmlx_engine.runtime_patches.kimi_k25_mla", "§1.2 Kimi MLA fp32-SDPA runtime-patch installer missing"),
]

failures = []
for mod, label, hint in REQUIRED:
    try:
        __import__(mod)
        print(f"  ok   {label:<40}  ({mod})")
    except Exception as e:
        failures.append((mod, label, hint, e))
        print(f"  FAIL {label:<40}  ({mod})  {type(e).__name__}: {e}")

if failures:
    print()
    print("RELEASE BLOCKED — bundled-python is missing critical modules:")
    for mod, label, hint, e in failures:
        print(f"  - {label}: {hint}")
    sys.exit(1)

# Extra spot-check: load the gemma4 Model class (catches broken relative
# imports that package-level __import__ won't catch).
try:
    from mlx_vlm.models.gemma4 import Model, LanguageModel, VisionModel  # noqa: F401
    print("  ok   gemma4 Model/LanguageModel/VisionModel classes")
except Exception as e:
    print(f"  FAIL gemma4 class import: {type(e).__name__}: {e}")
    sys.exit(1)

# Kimi K2.6 §1.2 fp32 MLA L==1 SDPA patch must be baked into bundled mlx_lm.
# If this fails, the bundled deepseek_v3.py wasn't patched at build time and
# Kimi K2.6 decode will produce repetition loops after ~14 tokens.
try:
    import inspect
    import mlx_lm.models.deepseek_v3 as _dv3
    _src = inspect.getsource(_dv3.DeepseekV3Attention.__call__)
    if "JANG fast fix" in _src and "q_sdpa" in _src:
        print("  ok   Kimi K2.6 fp32 MLA L==1 SDPA patch in bundled mlx_lm")
    else:
        print("  FAIL Kimi K2.6 fp32 MLA L==1 SDPA patch missing from bundled mlx_lm/models/deepseek_v3.py")
        print("       re-apply research/deepseek_v3_patched.py over it")
        sys.exit(1)
except Exception as e:
    print(f"  FAIL Kimi K2.6 MLA patch check: {type(e).__name__}: {e}")
    sys.exit(1)

# mlx_vlm kimi_k25 dispatch remap must be live (installed by
# vmlx_engine.__init__ at import time). Catches a silent regression if the
# remap block ever gets removed.
try:
    import vmlx_engine  # triggers remap install
    from mlx_vlm.utils import MODEL_REMAPPING
    from mlx_vlm.prompt_utils import MODEL_CONFIG
    if MODEL_REMAPPING.get("kimi_k25") != "kimi_vl":
        print("  FAIL kimi_k25 → kimi_vl remap missing in mlx_vlm.utils.MODEL_REMAPPING")
        sys.exit(1)
    if "kimi_k25" not in MODEL_CONFIG:
        print("  FAIL kimi_k25 missing in mlx_vlm.prompt_utils.MODEL_CONFIG")
        sys.exit(1)
    print("  ok   Kimi K2.6 mlx_vlm remap + prompt_utils config")
except Exception as e:
    print(f"  FAIL Kimi K2.6 mlx_vlm remap check: {type(e).__name__}: {e}")
    sys.exit(1)

print()
print("bundled-python: all critical imports ok")
PYEOF
