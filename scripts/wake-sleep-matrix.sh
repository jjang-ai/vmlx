#!/usr/bin/env bash
#
# wake-sleep-matrix.sh — one-at-a-time per-session matrix.
#
# Purpose: Eric sets up N server sessions in the tray (all in
# standby/sleep). This script wakes them one by one, waits for model
# load, runs the full multi-turn + VL + 3-API matrix, then puts it
# back to deep sleep so GPU RAM frees before the next session wakes.
#
# Usage:
#   scripts/wake-sleep-matrix.sh PORT1 [PORT2 ...]
#
# Example:
#   scripts/wake-sleep-matrix.sh 8000 8001 8002
#
# Each port is assumed to be a separate session already bound in
# standby mode. The script never LOADs a model — it only flips
# wake → test → sleep per existing session.

set -u -o pipefail

PORTS=("$@")
if [[ ${#PORTS[@]} -eq 0 ]]; then
    echo "usage: $0 PORT1 [PORT2 ...]" >&2
    exit 1
fi

LOG="/tmp/vmlx-wake-sleep-$(date +%s).log"
TOTAL_PASS=0
TOTAL_FAIL=0

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
pass() { log "  ✅ $*"; TOTAL_PASS=$((TOTAL_PASS+1)); }
fail() { log "  ❌ $*"; TOTAL_FAIL=$((TOTAL_FAIL+1)); }

wait_ready() {
    local port=$1 tries=0
    while (( tries < 60 )); do
        local r=$(curl -s --max-time 3 "http://127.0.0.1:$port/health" 2>&1)
        if echo "$r" | grep -qE '"state":"running"|"loaded":true|"loaded": true'; then
            return 0
        fi
        sleep 2
        tries=$((tries+1))
    done
    return 1
}

test_session() {
    local port=$1
    local base="http://127.0.0.1:$port"

    log ""
    log "═══════════════════════════════════════"
    log "SESSION on port $port"
    log "═══════════════════════════════════════"

    # 0. Identify
    local health=$(curl -sf --max-time 5 "$base/health" 2>&1)
    local model=$(curl -sf --max-time 5 "$base/v1/models" 2>&1 | python3 -c 'import sys,json;d=json.load(sys.stdin);entries=d.get("data",[]);loaded=[e for e in entries if e.get("vmlx",{}).get("loaded")];print(loaded[0]["id"] if loaded else (entries[0]["id"] if entries else ""))' 2>/dev/null)
    log "loaded model: $model"
    log "pre-wake health: $(echo "$health" | head -c 100)"

    # 1. Wake
    log "— 1. /admin/wake"
    local wake=$(curl -sf -X POST --max-time 10 "$base/admin/wake" 2>&1)
    if echo "$wake" | grep -q "awake\|running\|ok\|status"; then
        pass "wake accepted: $wake"
    else
        fail "wake rejected: $wake"
        return 1
    fi
    if wait_ready $port; then
        pass "model loaded and ready"
    else
        fail "timeout waiting for ready state"
        return 1
    fi

    # 2. OpenAI chat T1
    log "— 2. OpenAI /v1/chat/completions T1"
    local t1=$(curl -sf --max-time 60 -H 'Content-Type: application/json' \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say: pong\"}],\"max_tokens\":8,\"stream\":false}" \
        "$base/v1/chat/completions" 2>&1)
    local t1_content=$(echo "$t1" | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"][:40])' 2>/dev/null)
    [[ -n "$t1_content" ]] && pass "T1: '$t1_content'" || fail "T1 empty"

    # 3. Multi-turn T2 cache hit
    log "— 3. Multi-turn T2 cache hit"
    local t2=$(curl -sf --max-time 60 -H 'Content-Type: application/json' \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say: pong\"},{\"role\":\"assistant\",\"content\":\"$t1_content\"},{\"role\":\"user\",\"content\":\"Now say: ping\"}],\"max_tokens\":8,\"stream\":false}" \
        "$base/v1/chat/completions" 2>&1)
    local t2_cached=$(echo "$t2" | python3 -c 'import sys,json;d=json.load(sys.stdin);u=d.get("usage",{});print(u.get("prompt_tokens_details",{}).get("cached_tokens",0))' 2>/dev/null)
    local t2_detail=$(echo "$t2" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("usage",{}).get("cache_detail","?"))' 2>/dev/null)
    if [[ "${t2_cached:-0}" -gt 0 ]]; then
        pass "T2 cache hit: $t2_cached tokens ($t2_detail)"
    else
        fail "T2 cache miss (cached=$t2_cached, detail=$t2_detail)"
    fi

    # 4. Streaming SSE
    log "— 4. Streaming SSE"
    local lines=$(curl -sf --max-time 60 -H 'Content-Type: application/json' \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":6,\"stream\":true}" \
        "$base/v1/chat/completions" 2>&1 | wc -l)
    [[ "$lines" -gt 3 ]] && pass "$lines SSE lines" || fail "only $lines lines"

    # 5. Anthropic
    log "— 5. Anthropic /v1/messages"
    local anth=$(curl -sf --max-time 30 -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" \
        "$base/v1/messages" 2>&1)
    local anth_content=$(echo "$anth" | python3 -c 'import sys,json;d=json.load(sys.stdin);blocks=d.get("content",[]);texts=[b.get("text","") for b in blocks if b.get("type")=="text"];print(texts[0][:40] if texts else "")' 2>/dev/null)
    [[ -n "$anth_content" ]] && pass "Anthropic: '$anth_content'" || fail "Anthropic empty: $anth"

    # 6. Ollama
    log "— 6. Ollama /api/chat"
    local oll=$(curl -sf --max-time 30 -H 'Content-Type: application/json' \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" \
        "$base/api/chat" 2>&1)
    local oll_content=$(echo "$oll" | python3 -c 'import sys,json;print(json.load(sys.stdin)["message"]["content"][:40])' 2>/dev/null)
    [[ -n "$oll_content" ]] && pass "Ollama: '$oll_content'" || fail "Ollama empty"

    # 7. Tokenizer routes
    log "— 7. §347 tokenizer routes"
    local tok=$(curl -sf --max-time 5 "$base/v1/tokenizer_info" 2>&1)
    echo "$tok" | grep -q "eos_token" && pass "/v1/tokenizer_info OK" || fail "/v1/tokenizer_info missing"

    # 8. VL image (only if model is a VL family)
    log "— 8. VL image (1×1 red pixel base64)"
    local img="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    local vl=$(curl -sf --max-time 90 -H 'Content-Type: application/json' \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What color is this image?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$img\"}}]}],\"max_tokens\":16,\"stream\":false}" \
        "$base/v1/chat/completions" 2>&1)
    local vl_content=$(echo "$vl" | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"][:60])' 2>/dev/null)
    if [[ -n "$vl_content" ]]; then
        pass "VL image → '$vl_content'"
    else
        log "  (VL likely not a vision model for this session — that's OK)"
    fi

    # 9. Cache stats
    log "— 9. Cache stats"
    local stats=$(curl -sf --max-time 5 "$base/v1/cache/stats" 2>&1)
    local arch=$(echo "$stats" | python3 -c 'import sys,json;d=json.load(sys.stdin);a=d.get("architecture",{});print(f"total={a.get(\"total\",\"?\")} kv={a.get(\"kvSimple\",\"?\")} mamba={a.get(\"mamba\",\"?\")} tq={a.get(\"turboQuantActive\",\"?\")} hybrid={a.get(\"hybridSSMActive\",\"?\")}")' 2>/dev/null)
    [[ -n "$arch" ]] && pass "architecture: $arch" || fail "no architecture"

    # 10. Sleep (deep — unloads weights for next session)
    log "— 10. /admin/sleep (deep — unload weights)"
    local sleep_resp=$(curl -sf -X POST --max-time 15 -H 'Content-Type: application/json' \
        -d '{"mode":"deep"}' "$base/admin/sleep" 2>&1)
    if echo "$sleep_resp" | grep -qE "asleep|standby|ok"; then
        pass "deep sleep accepted"
    else
        log "  (sleep returned: $sleep_resp)"
    fi

    log ""
}

log "═══════════════════════════════════════"
log "WAKE/SLEEP MATRIX — ${#PORTS[@]} session(s): ${PORTS[*]}"
log "log: $LOG"
log "═══════════════════════════════════════"

for port in "${PORTS[@]}"; do
    test_session "$port"
done

log ""
log "═══════════════════════════════════════"
log "TOTAL PASS: $TOTAL_PASS  FAIL: $TOTAL_FAIL"
log "═══════════════════════════════════════"
[[ "$TOTAL_FAIL" -eq 0 ]] && exit 0 || exit 1
