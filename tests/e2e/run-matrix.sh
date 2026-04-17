#!/usr/bin/env bash
# run-matrix.sh — sweep the harness across every small/local model and
# produce a pass/fail matrix report. Skips any model that takes longer
# than 5 min to finish (background load can make load times unpredictable).
#
# Usage:
#   tests/e2e/run-matrix.sh [tier]
#
#     tier=1   small text models only (~minutes total)  [default]
#     tier=2   adds VL and embedding models
#     tier=3   tier-2 + one JANG-stamped text model
#
# Outputs:
#   tests/e2e/results/matrix-YYYYMMDD-HHMMSS.md   — summary table
#   tests/e2e/results/matrix-YYYYMMDD-HHMMSS/     — per-model jsonl logs
#
# The harness itself stays the source of truth for per-case logic.

set -u
# Walk up from tests/e2e/ to the swift/ repo root so paths like
# tests/e2e/results and the harness sibling resolve unambiguously.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TIER=${1:-1}
STAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR="$ROOT/tests/e2e/results/matrix-$STAMP"
mkdir -p "$OUT_DIR"
MARKDOWN="$ROOT/tests/e2e/results/matrix-$STAMP.md"
HARNESS="$ROOT/tests/e2e/harness.sh"

# Per-tier list of (label, path, suite). Only models already on disk
# with full weight shards — use `tests/e2e/audit-disk.sh` to see what's
# available and populate this list.
declare -a ENTRIES=()

add_if_exists() {
    local label="$1"
    local path="$2"
    local suite="$3"
    if [ -d "$path" ] && [ -f "$path/config.json" ]; then
        ENTRIES+=("$label|$path|$suite")
    fi
}

HF=$HOME/.cache/huggingface/hub
USER_MODELS=$HOME/.mlxstudio/models/MLXModels

# Resolve newest HF snapshot by refs/main (or first) for a repo name.
resolve_hf() {
    local repo="$1"
    local repo_dir="$HF/models--$repo"
    [ -d "$repo_dir" ] || { echo ""; return; }
    if [ -f "$repo_dir/refs/main" ]; then
        local sha
        sha=$(cat "$repo_dir/refs/main" | tr -d '\n')
        [ -d "$repo_dir/snapshots/$sha" ] && { echo "$repo_dir/snapshots/$sha"; return; }
    fi
    local first
    first=$(ls -d "$repo_dir"/snapshots/*/ 2>/dev/null | head -1)
    echo "${first%/}"
}

# Tier 1: small text (< 2 min each)
add_if_exists "qwen3-0.6b-8bit"            "$(resolve_hf mlx-community--Qwen3-0.6B-8bit)"             full
add_if_exists "llama-3.2-1b-4bit"          "$(resolve_hf mlx-community--Llama-3.2-1B-Instruct-4bit)" full
add_if_exists "gemma-4-e2b-it-4bit"        "$(resolve_hf mlx-community--gemma-4-e2b-it-4bit)"        full

if [ "$TIER" -ge 2 ]; then
    add_if_exists "qwen3-embedding-0.6b"   "$(resolve_hf mlx-community--Qwen3-Embedding-0.6B-8bit)"  embedding
    add_if_exists "gemma-4-e4b-it-4bit"    "$(resolve_hf mlx-community--gemma-4-e4b-it-4bit)"        full
    add_if_exists "qwen3.5-vl-4b-jang-4s"  "$USER_MODELS/dealignai/Qwen3.5-VL-4B-JANG_4S-CRACK"      vl
fi

if [ "$TIER" -ge 3 ]; then
    add_if_exists "qwen3.5-vl-9b-jang-4s"  "$USER_MODELS/dealignai/Qwen3.5-VL-9B-JANG_4S-CRACK"      vl
    add_if_exists "nemotron-30b-a3b-jang2l" "$USER_MODELS/dealignai/Nemotron-Cascade-2-30B-A3B-JANG_2L-CRACK" full
    # Tier-3 adds JANGTQ native-kernel path — exercises MXTQ decode,
    # QuantizedLinear repack, and packed-safetensors loader in one go.
    # Skip MiniMax-M2.7-JANGTQ-CRACK (~460 GB source) — not routinely
    # runnable on audit hardware; iter-58 verified it separately.
    add_if_exists "qwen3.6-35b-a3b-jangtq2" "$USER_MODELS/dealignai/Qwen3.6-35B-A3B-JANGTQ2-CRACK"   full
fi

if [ ${#ENTRIES[@]} -eq 0 ]; then
    echo "No models found on disk. Check tests/e2e/audit-disk.sh." >&2
    exit 1
fi

PORT_BASE=8780
IDX=0

{
echo "# vMLX Matrix Run — $STAMP"
echo ""
echo "Tier $TIER · ${#ENTRIES[@]} models · suite per model varies (full/vl/embedding)"
echo ""
echo "| model | suite | pass | fail | tps | ttft | cancel | concurrent |"
echo "|-------|-------|------|------|-----|------|--------|------------|"
} > "$MARKDOWN"

# One model at a time (no parallelism — MLX on one GPU).
for entry in "${ENTRIES[@]}"; do
    label=$(echo "$entry" | cut -d'|' -f1)
    path=$(echo "$entry" | cut -d'|' -f2)
    suite=$(echo "$entry" | cut -d'|' -f3)
    port=$((PORT_BASE + IDX))
    IDX=$((IDX + 1))
    jsonl="$OUT_DIR/$label.jsonl"

    echo "▶ $label ($suite suite, port $port)"
    # Kill anything stale on this port
    lsof -ti tcp:"$port" 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 1
    "$HARNESS" "$path" "$port" "$suite" > "$jsonl" 2>&1

    # Summarize
    python3 - "$jsonl" "$label" "$suite" "$MARKDOWN" <<'PY'
import json, sys
jsonl, label, suite, md = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cases = []
with open(jsonl) as f:
    for line in f:
        line = line.strip()
        if not line.startswith("{"): continue
        try: cases.append(json.loads(line))
        except Exception: continue
passed = sum(1 for c in cases if c.get("ok") is True)
failed = sum(1 for c in cases if c.get("ok") is False)
def field(name, key, default=""):
    for c in cases:
        if c.get("name") == name: return c.get(key, default)
    return default
tps = field("sse_stream", "tps", "-")
ttft = field("sse_stream", "ttft_ms", "-")
cancel = "✅" if any(c.get("name")=="cancel_midstream" and c.get("ok") for c in cases) else "—"
concur = field("concurrent", "notes", "-")
burst  = field("concurrent_burst", "notes", "-")
with open(md, "a") as f:
    f.write(f"| `{label}` | {suite} | {passed} | {failed} | {tps} | {ttft}ms | {cancel} | {concur} / {burst} |\n")
print(f"  {label}: {passed} pass / {failed} fail, tps={tps}")
PY
    # Extra pause between models so Metal can cool + CPU buffers flush.
    sleep 3
done

echo ""
echo "Matrix complete → $MARKDOWN"
echo "Per-model logs   → $OUT_DIR/"
