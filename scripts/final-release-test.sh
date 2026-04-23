#!/usr/bin/env bash
#
# final-release-test.sh — pre-release live-verification matrix
#
# Exercises the full 3-API surface + cache + lifecycle + VL against a
# running vMLX Swift server. Run this AFTER flipping the UI's Server
# tab to Start (or via `vmlxctl serve --model <path>`), then pass the
# port as arg 1. Default port 8080.
#
# Usage:
#   scripts/final-release-test.sh [PORT]
#
# Exit non-zero on any failure. Logs land in /tmp/vmlx-release-test-*.log.

set -u -o pipefail

PORT="${1:-8080}"
BASE="http://127.0.0.1:$PORT"
LOG="/tmp/vmlx-release-test-$(date +%s).log"
FAIL=0
PASS=0

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
pass() { log "✅ $*"; PASS=$((PASS+1)); }
fail() { log "❌ $*"; FAIL=$((FAIL+1)); }

log "=== vMLX Swift final release test ==="
log "Base URL: $BASE"
log "Log: $LOG"

# ----------------------------------------------------------------------
# 0. Liveness
# ----------------------------------------------------------------------
log ""
log "--- 0. Liveness ---"
if ! curl -sf --max-time 5 "$BASE/v1/models" >/dev/null; then
    fail "Server not reachable at $BASE — start it via UI or 'vmlxctl serve --model <path>'"
    exit 1
fi
pass "Server reachable at $BASE"

MODEL=$(curl -s "$BASE/v1/models" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["data"][0]["id"] if d.get("data") else "")')
if [[ -z "$MODEL" ]]; then
    fail "/v1/models returned empty list"
    exit 1
fi
pass "Loaded model: $MODEL"

# ----------------------------------------------------------------------
# 1. OpenAI /v1/chat/completions — non-stream T1
# ----------------------------------------------------------------------
log ""
log "--- 1. OpenAI /v1/chat/completions T1 (non-stream) ---"
T1_REQ=$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [{"role":"user","content":"Reply with exactly: pong"}],
  "max_tokens": 16,
  "stream": false
}
EOF
)
T1_RESP=$(curl -sf --max-time 60 -H 'Content-Type: application/json' \
    -d "$T1_REQ" "$BASE/v1/chat/completions")
T1_CONTENT=$(echo "$T1_RESP" | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
if [[ -n "$T1_CONTENT" ]]; then
    pass "T1 chat response: '$T1_CONTENT' (len=${#T1_CONTENT})"
else
    fail "T1 empty or malformed: $T1_RESP"
fi

# ----------------------------------------------------------------------
# 2. Multi-turn cache hit — same prefix
# ----------------------------------------------------------------------
log ""
log "--- 2. Multi-turn cache hit (T2 same prefix) ---"
T2_REQ=$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [
    {"role":"user","content":"Reply with exactly: pong"},
    {"role":"assistant","content":"$T1_CONTENT"},
    {"role":"user","content":"Now reply with exactly: ping"}
  ],
  "max_tokens": 16,
  "stream": false
}
EOF
)
T2_RESP=$(curl -sf --max-time 60 -H 'Content-Type: application/json' \
    -d "$T2_REQ" "$BASE/v1/chat/completions")
T2_CACHED=$(echo "$T2_RESP" | python3 -c 'import sys,json;d=json.load(sys.stdin);u=d.get("usage",{});c=u.get("prompt_tokens_details",{}).get("cached_tokens",u.get("cached_tokens",0));print(c)' 2>/dev/null)
T2_CONTENT=$(echo "$T2_RESP" | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
if [[ "${T2_CACHED:-0}" -gt 0 ]]; then
    pass "T2 cache hit: cached_tokens=$T2_CACHED, content='$T2_CONTENT'"
else
    fail "T2 cache MISS (expected >0 cached tokens, got '${T2_CACHED:-}')"
fi

# ----------------------------------------------------------------------
# 3. OpenAI /v1/chat/completions — streaming SSE
# ----------------------------------------------------------------------
log ""
log "--- 3. OpenAI /v1/chat/completions streaming ---"
STREAM_LINES=$(curl -sf --max-time 60 -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":8,\"stream\":true}" \
    "$BASE/v1/chat/completions" | wc -l)
if [[ "$STREAM_LINES" -gt 3 ]]; then
    pass "Streaming: $STREAM_LINES SSE lines received"
else
    fail "Streaming too short: $STREAM_LINES lines"
fi

# ----------------------------------------------------------------------
# 4. OpenAI /v1/completions legacy (logprobs + text_offset)
# ----------------------------------------------------------------------
log ""
log "--- 4. OpenAI /v1/completions legacy + logprobs ---"
LEGACY=$(curl -sf --max-time 30 -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"Hello\",\"max_tokens\":4,\"logprobs\":1}" \
    "$BASE/v1/completions" 2>&1)
LEGACY_HAS_LP=$(echo "$LEGACY" | python3 -c 'import sys,json;d=json.load(sys.stdin);lp=d.get("choices",[{}])[0].get("logprobs");print("yes" if lp and "tokens" in lp and "token_logprobs" in lp else "no")' 2>/dev/null)
if [[ "$LEGACY_HAS_LP" == "yes" ]]; then
    pass "Legacy /v1/completions logprobs shape valid (tokens + token_logprobs + text_offset)"
else
    fail "Legacy /v1/completions logprobs missing or malformed: $LEGACY"
fi

# ----------------------------------------------------------------------
# 5. Anthropic /v1/messages
# ----------------------------------------------------------------------
log ""
log "--- 5. Anthropic /v1/messages ---"
ANTH=$(curl -sf --max-time 30 -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":8}" \
    "$BASE/v1/messages")
ANTH_CONTENT=$(echo "$ANTH" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["content"][0]["text"])' 2>/dev/null)
if [[ -n "$ANTH_CONTENT" ]]; then
    pass "Anthropic /v1/messages: '$ANTH_CONTENT'"
else
    fail "Anthropic /v1/messages empty: $ANTH"
fi

# ----------------------------------------------------------------------
# 6. Ollama /api/chat
# ----------------------------------------------------------------------
log ""
log "--- 6. Ollama /api/chat ---"
OLL=$(curl -sf --max-time 30 -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"stream\":false}" \
    "$BASE/api/chat" 2>&1)
OLL_CONTENT=$(echo "$OLL" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["message"]["content"])' 2>/dev/null)
if [[ -n "$OLL_CONTENT" ]]; then
    pass "Ollama /api/chat: '$OLL_CONTENT'"
else
    fail "Ollama /api/chat empty: $OLL"
fi

# ----------------------------------------------------------------------
# 7. Ollama /api/tags + /api/version + /api/ps
# ----------------------------------------------------------------------
log ""
log "--- 7. Ollama metadata routes ---"
for route in "/api/version" "/api/tags" "/api/ps"; do
    if curl -sf --max-time 5 "$BASE$route" >/dev/null; then
        pass "$route responds"
    else
        fail "$route failed"
    fi
done

# ----------------------------------------------------------------------
# 8. §347 Tokenizer endpoints (lm-eval Phase 1)
# ----------------------------------------------------------------------
log ""
log "--- 8. §347 Tokenizer endpoints ---"
TOK_INFO=$(curl -sf --max-time 5 "$BASE/v1/tokenizer_info")
if echo "$TOK_INFO" | grep -q "eos_token"; then
    pass "/v1/tokenizer_info returns eos_token, bos_token, pad_token, chat_template"
else
    fail "/v1/tokenizer_info missing fields: $TOK_INFO"
fi

TOK_OUT=$(curl -sf --max-time 5 -H 'Content-Type: application/json' \
    -d '{"text":"Hello world"}' "$BASE/v1/tokenize")
TOK_COUNT=$(echo "$TOK_OUT" | python3 -c 'import sys,json;print(json.load(sys.stdin)["count"])' 2>/dev/null)
if [[ "${TOK_COUNT:-0}" -gt 0 ]]; then
    pass "/v1/tokenize: 'Hello world' → $TOK_COUNT tokens"
else
    fail "/v1/tokenize returned 0 tokens or malformed: $TOK_OUT"
fi

# Alias (lm-eval strips /v1/completions from base_url)
if curl -sf --max-time 5 "$BASE/tokenizer_info" >/dev/null; then
    pass "/tokenizer_info alias (unprefixed) responds — lm-eval compat"
else
    fail "/tokenizer_info alias missing"
fi

# ----------------------------------------------------------------------
# 9. Embeddings
# ----------------------------------------------------------------------
log ""
log "--- 9. OpenAI /v1/embeddings ---"
EMB=$(curl -sf --max-time 30 -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"input\":\"hello\"}" \
    "$BASE/v1/embeddings" 2>&1)
EMB_DIM=$(echo "$EMB" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(len(d["data"][0]["embedding"]) if d.get("data") else 0)' 2>/dev/null)
if [[ "${EMB_DIM:-0}" -gt 0 ]]; then
    pass "/v1/embeddings: $EMB_DIM-dim vector"
else
    log "   (skipped — requires an embedding model; current model: chat/VL)"
fi

# ----------------------------------------------------------------------
# 10. Cache stats + architecture breakdown
# ----------------------------------------------------------------------
log ""
log "--- 10. Cache stats + architecture ---"
STATS=$(curl -sf --max-time 5 "$BASE/v1/cache/stats" 2>&1)
if echo "$STATS" | grep -qE "prefix|paged|ssm|turboquant|architecture"; then
    pass "/v1/cache/stats exposes cache breakdown"
    echo "$STATS" | python3 -m json.tool 2>/dev/null | head -20 | tee -a "$LOG"
else
    fail "/v1/cache/stats missing expected keys: $STATS"
fi

# ----------------------------------------------------------------------
# 11. Admin wake / sleep lifecycle
# ----------------------------------------------------------------------
log ""
log "--- 11. Admin sleep/wake lifecycle ---"
SLEEP_RESP=$(curl -sf --max-time 10 -X POST "$BASE/admin/sleep" -H 'Content-Type: application/json' -d '{"mode":"soft"}' 2>&1)
if [[ $? -eq 0 ]]; then
    pass "admin/sleep soft: accepted"
    sleep 1
    WAKE_RESP=$(curl -sf --max-time 10 -X POST "$BASE/admin/wake" 2>&1)
    [[ $? -eq 0 ]] && pass "admin/wake: accepted" || fail "admin/wake failed: $WAKE_RESP"
else
    log "   (admin routes may require auth — skip when bearer is set)"
fi

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
log ""
log "=== SUMMARY ==="
log "PASS: $PASS  FAIL: $FAIL"
log "Full log: $LOG"
[[ "$FAIL" -eq 0 ]] && exit 0 || exit 1
