#!/bin/bash
# iter-151 §219: perf-parity bench harness
#
# Runs a quick decode256 + prefill1024 bench on a target model and
# reports tokens_per_sec / ttft_ms / peak_memory_gb. Matches the
# methodology used in iter-144 / iter-149 diagnostic work. Usage:
#
#   ./scripts/bench-perf-parity.sh <model_path>
#
# Or with defaults (tries the fleet in order, uses first that exists):
#
#   ./scripts/bench-perf-parity.sh
#
# Requires a fresh release build: `swift build -c release`.
# Appends results to /tmp/vmlx-bench.log for post-run review.

set -euo pipefail

MODEL_PATH="${1:-}"
PORT="${VMLX_BENCH_PORT:-8765}"
VMLXCTL="${VMLXCTL_BIN:-.build/release/vmlxctl}"

if [ ! -x "$VMLXCTL" ]; then
    echo "ERROR: $VMLXCTL not found or not executable"
    echo "       Run: swift build -c release"
    exit 1
fi

# Default fleet — try in order, use first that exists
if [ -z "$MODEL_PATH" ]; then
    FLEET=(
        "/Users/eric/.mlxstudio/models/MLXModels/JANGQ-AI/Nemotron-Cascade-2-30B-A3B-JANG_4M"
        "/Users/eric/.mlxstudio/models/MLXModels/JANGQ-AI/Gemma-4-31B-it-JANG_4M"
        "/Users/eric/.mlxstudio/models/MLXModels/OsaurusAI/gemma-4-e2b-it-4bit"
    )
    for candidate in "${FLEET[@]}"; do
        if [ -d "$candidate" ] && [ -f "$candidate/config.json" ]; then
            MODEL_PATH="$candidate"
            echo "Using default: $MODEL_PATH"
            break
        fi
    done
    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: no default model found; pass a path as arg 1"
        exit 1
    fi
fi

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: $MODEL_PATH is not a valid model directory (missing config.json)"
    exit 1
fi

# Cleanup any stray server
pkill -f "vmlxctl serve" 2>/dev/null || true
sleep 2

# Boot
echo "=== Booting $MODEL_PATH on :$PORT ==="
$VMLXCTL serve --model "$MODEL_PATH" --port $PORT > /tmp/vmlx-bench-serve.log 2>&1 &
SERVE_PID=$!
trap "kill $SERVE_PID 2>/dev/null || true" EXIT

# Wait up to 90s for load
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
    echo "ERROR: server didn't start in 90s"
    tail -20 /tmp/vmlx-bench-serve.log
    exit 1
fi

MODEL_ID=$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
ms = [m['id'] for m in d['data'] if m.get('vmlx', {}).get('loaded')]
print(ms[0] if ms else d['data'][0]['id'])")

echo "Model ID: $MODEL_ID"
echo ""

# Warmup
echo "=== Warmup (discard) ==="
curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"warm\"}],\"max_tokens\":16}" \
    --max-time 120 || true

# decode256
echo ""
echo "=== decode256 (per-token decode throughput) ==="
curl -s -X POST "http://127.0.0.1:$PORT/admin/benchmark" \
    -H "Content-Type: application/json" \
    -d '{"suite":"decode256"}' \
    --max-time 300 | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
for k in ['tokens_per_sec', 'generation_tps', 'processing_tps', 'ttft_ms', 'total_ms', 'peak_memory_gb']:
    v = d.get(k)
    if v is not None:
        print(f'  {k}: {v:.2f}' if isinstance(v, (int, float)) else f'  {k}: {v}')"

# prefill1024
echo ""
echo "=== prefill1024 (prefill GEMM throughput) ==="
curl -s -X POST "http://127.0.0.1:$PORT/admin/benchmark" \
    -H "Content-Type: application/json" \
    -d '{"suite":"prefill1024"}' \
    --max-time 300 | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
for k in ['tokens_per_sec', 'generation_tps', 'processing_tps', 'ttft_ms', 'total_ms', 'peak_memory_gb']:
    v = d.get(k)
    if v is not None:
        print(f'  {k}: {v:.2f}' if isinstance(v, (int, float)) else f'  {k}: {v}')"

echo ""
echo "=== Done ==="
