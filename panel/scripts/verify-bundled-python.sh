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
    ("vmlx_engine", "vmlx_engine", "bundled vmlx_engine missing"),
    ("vmlx_engine.utils.jang_loader", "vmlx_engine jang_loader", "bundled jang_loader missing"),
    ("vmlx_engine.api.ollama_adapter", "vmlx_engine ollama_adapter", "bundled ollama_adapter missing"),
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

print()
print("bundled-python: all critical imports ok")
PYEOF
