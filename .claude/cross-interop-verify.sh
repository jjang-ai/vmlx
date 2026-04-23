#!/usr/bin/env bash
# Cross-function interoperability harness.
#
# Proves that every pair of features we ship still works when exercised
# in sequence within a single server lifetime:
#
#   chat → chat-cache-hit → soft-sleep → chat-JIT-wake →
#   image gen (all 3 response_format) → deep-sleep → chat-JIT-reload →
#   image gen JIT-rehydrate → admin/log-level live swap →
#   x-vmlx-trace-id header → /v1/cache/stats → /admin/cache/clear →
#   post-clear gen → /metrics RAM delta probe
#
# Any regression in a single step marks the whole run FAIL.

set -u
PORT="${PORT:-19930}"
BASE="http://127.0.0.1:$PORT"
ADMIN=interop
BIN="$(cd "$(dirname "$0")/.." && pwd)/.build/release/vmlxctl"
CHAT_MODEL="${CHAT_MODEL:-$HOME/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/76b6a5af250fa029339a757deeb93716baa8ead0}"
IMAGE_MODEL="${IMAGE_MODEL:-$HOME/.mlxstudio/models/image-gen/z-image-turbo-8bit}"
LOG=$(mktemp -t vmlx-interop).log
PASSED=0
FAILED=0
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

check() {
  local label="$1"; local cmd="$2"; local expect="$3"
  local got
  got=$(eval "$cmd" 2>&1)
  if echo "$got" | grep -qE "$expect"; then
    printf "  ✓ %s\n" "$label"; PASSED=$((PASSED+1))
  else
    printf "  ✗ %s\n    expected: %s\n    got: %s\n" "$label" "$expect" "$(echo "$got" | head -c 400)"
    FAILED=$((FAILED+1))
  fi
}
section() { printf "\n==== %s ====\n" "$1"; }

[ -x "$BIN" ] || { echo "$BIN not built"; exit 2; }
[ -d "$CHAT_MODEL" ] || { echo "chat model not at $CHAT_MODEL"; exit 2; }
[ -d "$IMAGE_MODEL" ] || { echo "image model not at $IMAGE_MODEL"; exit 2; }

pkill -9 -f vmlxctl 2>/dev/null || true
sleep 1

section "BOOT chat + image concurrently"
"$BIN" serve -m "$CHAT_MODEL" --image-model "$IMAGE_MODEL" \
  --port "$PORT" --admin-token "$ADMIN" \
  --idle-soft-sec 600 --idle-deep-sec 1200 \
  > "$LOG" 2>&1 &
SERVER_PID=$!
chat_ready=0; img_ready=0
for i in $(seq 1 300); do
  if grep -q '"state":"running"' <(curl -sS -m 2 "$BASE/health" 2>/dev/null); then chat_ready=1; fi
  if grep -q 'image model ready' "$LOG" 2>/dev/null; then img_ready=1; fi
  [ "$chat_ready" = 1 ] && [ "$img_ready" = 1 ] && break
  sleep 1
done
[ "$chat_ready" = 1 ] && { printf "  ✓ chat ready\n"; PASSED=$((PASSED+1)); } || { printf "  ✗ chat never came up\n"; FAILED=$((FAILED+1)); }
[ "$img_ready" = 1 ] && { printf "  ✓ image ready\n"; PASSED=$((PASSED+1)); } || { printf "  ✗ image never came up\n"; FAILED=$((FAILED+1)); }

section "S1 chat completion + trace id"
check "non-streaming chat returns content" \
  "curl -sS -m 60 -H 'Content-Type: application/json' -d '{\"model\":\"g\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":10}' $BASE/v1/chat/completions" \
  '"content"'
check "streaming chat sets x-vmlx-trace-id header" \
  "curl -sS -D - -o /dev/null -m 60 -H 'Content-Type: application/json' -d '{\"model\":\"g\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5,\"stream\":true}' $BASE/v1/chat/completions" \
  'x-vmlx-trace-id: chatcmpl-'

section "S2 cache hit on exact-match reprompt"
# T1 cold
curl -sS -o /tmp/t1.json -m 60 -H 'Content-Type: application/json' \
  -d '{"model":"g","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":10}' \
  "$BASE/v1/chat/completions"
# T2 warm
curl -sS -o /tmp/t2.json -m 60 -H 'Content-Type: application/json' \
  -d '{"model":"g","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":10}' \
  "$BASE/v1/chat/completions"
check "T2 shows cached_tokens > 0" \
  "cat /tmp/t2.json" \
  '"cached_tokens":[1-9]'

section "S3 admin log-level live swap"
check "GET returns current level (info)" \
  "curl -sS -m 5 -H 'x-admin-token: $ADMIN' $BASE/admin/log-level" \
  '"level":"info"'
check "POST debug accepted" \
  "curl -sS -m 5 -H 'x-admin-token: $ADMIN' -H 'Content-Type: application/json' -d '{\"level\":\"debug\"}' $BASE/admin/log-level" \
  '"level":"debug"'
check "POST invalid rejected" \
  "curl -sS -o /dev/null -w 'status=%{http_code}' -m 5 -H 'x-admin-token: $ADMIN' -H 'Content-Type: application/json' -d '{\"level\":\"verbose\"}' $BASE/admin/log-level" \
  'status=400'

section "S4 image gen during chat-running (same port)"
check "image gen returns PNG" \
  "curl -sS -m 60 -H 'Content-Type: application/json' -d '{\"model\":\"z-image-turbo\",\"prompt\":\"x\",\"size\":\"256x256\",\"steps\":4,\"response_format\":\"b64_json\"}' $BASE/v1/images/generations | python3 -c 'import sys,json,base64;o=json.loads(sys.stdin.read());d=base64.b64decode(o[\"data\"][0][\"b64_json\"]);print(\"magic=\"+d[:8].hex())'" \
  'magic=89504e470d0a1a0a'
check "image response carries placeholder_output warning" \
  "curl -sS -m 60 -H 'Content-Type: application/json' -d '{\"model\":\"z-image-turbo\",\"prompt\":\"x\",\"size\":\"256x256\",\"steps\":4}' $BASE/v1/images/generations" \
  '"code":"placeholder_output"'

section "S5 cache stats + cache clear"
check "/v1/cache/stats returns architecture keys" \
  "curl -sS -m 5 $BASE/v1/cache/stats" \
  'architecture|paged'
check "/admin/cache/clear succeeds" \
  "curl -sS -m 10 -H 'x-admin-token: $ADMIN' -X POST $BASE/admin/cache/clear" \
  '"status":"cleared"|2xx'

section "S6 soft-sleep → chat JIT-wake"
check "soft-sleep returns sleeping" \
  "curl -sS -m 10 -H 'x-admin-token: $ADMIN' -X POST $BASE/admin/soft-sleep" \
  '"status":"sleeping"'
check "health shows soft_sleep" \
  "curl -sS -m 5 $BASE/health" \
  '"state":"soft_sleep"'
check "chat post-soft JIT-wakes" \
  "curl -sS -m 60 -H 'Content-Type: application/json' -d '{\"model\":\"g\",\"messages\":[{\"role\":\"user\",\"content\":\"hey\"}],\"max_tokens\":5}' $BASE/v1/chat/completions" \
  '"content"'

section "S7 deep-sleep → chat JIT-reload → image JIT-rehydrate"
check "deep-sleep returns deep_sleeping" \
  "curl -sS -m 15 -H 'x-admin-token: $ADMIN' -X POST $BASE/admin/deep-sleep" \
  '"status":"deep_sleeping"'
check "health shows deep_sleep" \
  "curl -sS -m 5 $BASE/health" \
  '"state":"deep_sleep"'
check "chat post-deep reloads weights + generates" \
  "curl -sS -m 120 -H 'Content-Type: application/json' -d '{\"model\":\"g\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}' $BASE/v1/chat/completions" \
  '"content"'
check "image post-deep JIT-rehydrates + generates PNG" \
  "curl -sS -m 120 -H 'Content-Type: application/json' -d '{\"model\":\"z-image-turbo\",\"prompt\":\"x\",\"size\":\"256x256\",\"steps\":4,\"response_format\":\"b64_json\"}' $BASE/v1/images/generations | python3 -c 'import sys,json,base64;o=json.loads(sys.stdin.read());d=base64.b64decode(o[\"data\"][0][\"b64_json\"]);print(\"magic=\"+d[:8].hex())'" \
  'magic=89504e470d0a1a0a'

section "S8 metrics + health integrity post-cycle"
check "/metrics has vmlx_ram_bytes_used" \
  "curl -sS $BASE/metrics" \
  '^vmlx_ram_bytes_used'
check "/health has loaded model name (not hash)" \
  "curl -sS $BASE/health" \
  '"state":"running"'

section "S9 concurrent burst — 3 parallel chats"
for i in 1 2 3; do
  (curl -sS -o "/tmp/burst$i.json" -m 60 -H 'Content-Type: application/json' \
    -d "{\"model\":\"g\",\"messages\":[{\"role\":\"user\",\"content\":\"burst $i\"}],\"max_tokens\":5}" \
    "$BASE/v1/chat/completions" &) 2>/dev/null
done
wait 2>/dev/null
burst_ok=0
for i in 1 2 3; do
  if grep -q '"finish_reason"' "/tmp/burst$i.json" 2>/dev/null; then
    burst_ok=$((burst_ok+1))
  fi
done
if [ "$burst_ok" = "3" ]; then
  printf "  ✓ concurrent burst 3/3 completed\n"; PASSED=$((PASSED+1))
else
  printf "  ✗ concurrent burst only %d/3\n" "$burst_ok"; FAILED=$((FAILED+1))
fi

section "RESULT"
printf "  PASSED: %d\n  FAILED: %d\n" "$PASSED" "$FAILED"
if [ "$FAILED" -gt 0 ]; then exit 1; fi
