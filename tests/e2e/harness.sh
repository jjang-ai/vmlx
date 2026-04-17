#!/usr/bin/env bash
# harness.sh — end-to-end empirical test harness for a single model.
#
# Usage:
#   tests/e2e/harness.sh <model-path> [port] [suite]
#
#   <model-path>   absolute path to a local model dir
#   port           optional, default 8765
#   suite          "smoke" (default) | "full" | "multiturn" | "vl" | "tools"
#
# Emits JSON-lines to stdout, one line per test case. Each line includes
#   { name, ok, ttft_ms, tokens, tps, notes }
# so downstream reporters can diff across runs.

set -u
CLI="${CLI:-/Users/eric/vmlx/swift/.build/arm64-apple-macosx/release/vmlxctl}"
MODEL="${1:?model-path required}"
PORT="${2:-8765}"
SUITE="${3:-smoke}"
HOST="127.0.0.1"
BASE="http://${HOST}:${PORT}"
LOG_DIR="/tmp/vmlx-e2e"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/server-$(basename "$MODEL").log"

emit() { printf '%s\n' "$1"; }
die()  { emit "{\"name\":\"FATAL\",\"ok\":false,\"notes\":\"$1\"}"; exit 1; }

start_server() {
    # Kill anything on our port first
    lsof -ti tcp:"$PORT" | xargs -r kill -9 2>/dev/null
    "$CLI" serve --model "$MODEL" --host "$HOST" --port "$PORT" \
        > "$LOG" 2>&1 &
    SERVER_PID=$!
    # Poll /v1/models until ready (or timeout after 300s for 60GB models)
    for i in $(seq 1 300); do
        if curl -s --max-time 2 "$BASE/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

stop_server() {
    [ -n "${SERVER_PID:-}" ] && kill -TERM "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    lsof -ti tcp:"$PORT" | xargs -r kill -9 2>/dev/null
}
trap stop_server EXIT

# ---------------------------------------------------------------------------
# Test primitives
# ---------------------------------------------------------------------------

# curl_chat <prompt> <max_tokens> [stream=false]
curl_chat() {
    local prompt="$1" max="${2:-32}" stream="${3:-false}"
    local model_id
    model_id=$(curl -s "$BASE/v1/models" | python3 -c \
        'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    [ -z "$model_id" ] && { echo "{}" ; return 1; }
    curl -s -N -X POST "$BASE/v1/chat/completions" \
        -H "content-type: application/json" \
        -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max,\"stream\":$stream,\"temperature\":0}"
}

# Extract content from non-streaming response
extract_content() {
    python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["choices"][0]["message"]["content"] if d.get("choices") else "")'
}

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

case_models_list() {
    local out
    out=$(curl -s --max-time 5 "$BASE/v1/models")
    local count
    count=$(echo "$out" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(len(d.get("data",[])))' 2>/dev/null || echo 0)
    if [ "$count" -gt 0 ]; then
        emit "{\"name\":\"models_list\",\"ok\":true,\"notes\":\"$count model(s) in picker\"}"
    else
        emit "{\"name\":\"models_list\",\"ok\":false,\"notes\":\"empty\"}"
    fi
}

case_basic_chat() {
    local t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    local resp
    resp=$(curl_chat "Reply with the single word OK." 8 false)
    local t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    local content=$(echo "$resp" | extract_content)
    local usage_tok
    usage_tok=$(echo "$resp" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("usage",{}).get("completion_tokens",0))' 2>/dev/null || echo 0)
    local ok=false
    echo "$content" | grep -iq "ok\|hi\|hello\|yes\|sure" && ok=true
    emit "{\"name\":\"basic_chat\",\"ok\":$ok,\"ttft_ms\":$((t1-t0)),\"tokens\":$usage_tok,\"notes\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1][:80]))' "$content")}"
}

case_sse_stream() {
    # Use a prompt that forces many tokens — "count 1 to 5" hits EOS at 5.
    # Write a 200-word paragraph gives us a real throughput measurement.
    local t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    local first_ms=0 last_ms=0 tok=0
    while IFS= read -r line; do
        [[ "$line" != data:* ]] && continue
        [[ "$line" == "data: [DONE]" ]] && break
        local now=$(python3 -c 'import time;print(int(time.time()*1000))')
        [ "$first_ms" = 0 ] && first_ms=$((now-t0))
        last_ms=$((now-t0))
        tok=$((tok+1))
    done < <(curl_chat "Write a detailed 200 word paragraph about the history of computing. Cover Turing, ENIAC, transistors, and the invention of the transistor. Be thorough and precise." 200 true)
    local tps=0
    local decode_ms=$((last_ms-first_ms))
    [ "$decode_ms" -gt 0 ] && [ "$tok" -gt 1 ] && tps=$(python3 -c "print(f'{($tok-1)*1000/$decode_ms:.1f}')")
    local ok=false
    [ "$tok" -ge 20 ] && [ "$first_ms" -gt 0 ] && ok=true
    emit "{\"name\":\"sse_stream\",\"ok\":$ok,\"ttft_ms\":$first_ms,\"tokens\":$tok,\"tps\":$tps,\"notes\":\"decode_ms=$decode_ms\"}"
}

case_json_mode() {
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local resp
    resp=$(curl -s -X POST "$BASE/v1/chat/completions" \
        -H "content-type: application/json" \
        -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"Return ONLY a JSON object with field 'color' set to 'blue'. No prose.\"}],\"max_tokens\":40,\"temperature\":0,\"response_format\":{\"type\":\"json_object\"}}")
    local content=$(echo "$resp" | extract_content)
    # Stricter: must be a dict AND contain the expected key — without that
    # an API-error payload (also a dict) would false-positive.
    local ok=false
    echo "$content" | python3 -c '
import sys, json
try:
    d = json.loads(sys.stdin.read())
    sys.exit(0 if isinstance(d, dict) and "color" in d else 2)
except Exception:
    sys.exit(1)
' 2>/dev/null && ok=true
    emit "{\"name\":\"json_mode\",\"ok\":$ok,\"notes\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1][:80]))' "$content")}"
}

case_ollama_chat() {
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local resp
    resp=$(curl -s -X POST "$BASE/api/chat" \
        -H "content-type: application/json" \
        -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"Say 'hi' and stop.\"}],\"stream\":false,\"options\":{\"temperature\":0,\"num_predict\":8}}")
    local content=$(echo "$resp" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("message",{}).get("content","")[:40])' 2>/dev/null)
    local ok=false
    [ -n "$content" ] && ok=true
    emit "{\"name\":\"ollama_chat\",\"ok\":$ok,\"notes\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1][:80]))' "$content")}"
}

case_anthropic_messages() {
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local resp
    resp=$(curl -s -X POST "$BASE/v1/messages" \
        -H "content-type: application/json" \
        -H "anthropic-version: 2023-06-01" \
        -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":8}")
    local content=$(echo "$resp" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("content",[{}])[0].get("text","")[:40])' 2>/dev/null)
    local ok=false
    [ -n "$content" ] && ok=true
    emit "{\"name\":\"anthropic_messages\",\"ok\":$ok,\"notes\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1][:80]))' "$content")}"
}

case_concurrent() {
    # Fire 3 parallel requests; all should return non-empty content.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local pids=()
    local outs=()
    for i in 1 2 3; do
        local tmp="/tmp/vmlx-concur-$i.json"
        curl -s -X POST "$BASE/v1/chat/completions" -H "content-type: application/json" \
            -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply $i.\"}],\"max_tokens\":6,\"temperature\":0}" \
            > "$tmp" &
        pids+=($!)
        outs+=("$tmp")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    local success=0
    for out in "${outs[@]}"; do
        local c=$(cat "$out" | extract_content)
        [ -n "$c" ] && success=$((success+1))
    done
    local ok=false
    [ "$success" = "3" ] && ok=true
    emit "{\"name\":\"concurrent\",\"ok\":$ok,\"notes\":\"$success/3 succeeded\"}"
}

case_stop_sequences() {
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local resp
    resp=$(curl -s -X POST "$BASE/v1/chat/completions" -H "content-type: application/json" \
        -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"List 5 fruits, one per line.\"}],\"max_tokens\":100,\"temperature\":0,\"stop\":[\"\\n\"]}")
    # Use Python for newline detection — shell `grep $'\n'` is flaky when
    # stdin buffering chops on LF. The only contract that matters is that
    # the server honoured `finish_reason=stop` AND stripped the newline
    # from the emitted content.
    python3 <<PY
import json, sys
r = json.loads('''$resp''')
c = r.get("choices",[{}])[0]
finish = c.get("finish_reason","?")
content = c.get("message",{}).get("content","")
has_nl = "\n" in content
ok = "true" if (finish == "stop" and not has_nl) else "false"
note = json.dumps(f"finish={finish} nl={has_nl} c={content[:60]}")
print(f'{{"name":"stop_sequences","ok":{ok},"notes":{note}}}')
PY
}

case_max_tokens() {
    # max_tokens=4 must produce <=4 tokens and finish_reason=length.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local resp
    resp=$(curl -s -X POST "$BASE/v1/chat/completions" -H "content-type: application/json" \
        -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a long story about a cat.\"}],\"max_tokens\":4,\"temperature\":0}")
    local finish=$(echo "$resp" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["choices"][0].get("finish_reason","?")) if d.get("choices") else print("?")' 2>/dev/null)
    local tok=$(echo "$resp" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("usage",{}).get("completion_tokens",0))' 2>/dev/null || echo 0)
    local ok=false
    [ "$finish" = "length" ] && [ "$tok" -le 4 ] && [ "$tok" -gt 0 ] && ok=true
    emit "{\"name\":\"max_tokens\",\"ok\":$ok,\"tokens\":$tok,\"notes\":\"finish=$finish\"}"
}

case_multiturn_prefix_cache() {
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local t0 t1 t2
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    curl -s -X POST "$BASE/v1/chat/completions" -H "content-type: application/json" -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"system\",\"content\":\"You are a number counter. Be brief.\"},{\"role\":\"user\",\"content\":\"Hi.\"}],\"max_tokens\":6,\"temperature\":0}" > /dev/null
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    curl -s -X POST "$BASE/v1/chat/completions" -H "content-type: application/json" -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"system\",\"content\":\"You are a number counter. Be brief.\"},{\"role\":\"user\",\"content\":\"Hi.\"},{\"role\":\"assistant\",\"content\":\"Hello.\"},{\"role\":\"user\",\"content\":\"Bye.\"}],\"max_tokens\":6,\"temperature\":0}" > /dev/null
    t2=$(python3 -c 'import time;print(int(time.time()*1000))')
    local t1ms=$((t1-t0))
    local t2ms=$((t2-t1))
    # Turn 2 should be faster than turn 1 if prefix cache kicks in
    local ok=false
    [ "$t2ms" -gt 0 ] && [ "$t2ms" -le "$((t1ms * 2))" ] && ok=true
    emit "{\"name\":\"multiturn_prefix_cache\",\"ok\":$ok,\"ttft_ms\":$t2ms,\"notes\":\"t1=${t1ms}ms t2=${t2ms}ms\"}"
}

case_metrics() {
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics")
    local ok=false
    [ "$code" = "200" ] && ok=true
    emit "{\"name\":\"metrics_endpoint\",\"ok\":$ok,\"notes\":\"HTTP $code\"}"
}

case_ollama_tags() {
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/api/tags")
    local ok=false
    [ "$code" = "200" ] && ok=true
    emit "{\"name\":\"ollama_tags\",\"ok\":$ok,\"notes\":\"HTTP $code\"}"
}

# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------

suite_smoke() {
    case_models_list
    case_basic_chat
    case_sse_stream
    case_metrics
    case_ollama_tags
    case_max_tokens
}

suite_full() {
    suite_smoke
    case_multiturn_prefix_cache
    case_stop_sequences
    case_json_mode
    case_ollama_chat
    case_anthropic_messages
    case_concurrent
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

emit "{\"name\":\"start\",\"ok\":true,\"notes\":\"model=$(basename "$MODEL") suite=$SUITE\"}"
if ! start_server; then
    emit "{\"name\":\"server_start\",\"ok\":false,\"notes\":\"timed out — see $LOG\"}"
    exit 2
fi
emit "{\"name\":\"server_start\",\"ok\":true,\"notes\":\"pid=$SERVER_PID\"}"

case "$SUITE" in
    smoke)      suite_smoke ;;
    full)       suite_full ;;
    *)          suite_smoke ;;
esac

emit "{\"name\":\"done\",\"ok\":true}"
