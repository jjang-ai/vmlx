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
# Environment variables:
#   CLI            path to vmlxctl (default: .build/arm64-apple-macosx/release/vmlxctl)
#   EMBEDDING_MODEL  when set, passed to --embedding-model; /v1/embeddings works
#   EXTRA_FLAGS      additional flags appended to `vmlxctl serve`
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
    local extra=()
    [ -n "${EMBEDDING_MODEL:-}" ] && extra+=(--embedding-model "$EMBEDDING_MODEL")
    [ -n "${EXTRA_FLAGS:-}" ] && extra+=($EXTRA_FLAGS)
    # `set -u` + empty array expansion trips `unbound variable` on bash 3 —
    # guard with the `+` form so the expansion disappears when `extra` is
    # empty instead of erroring.
    "$CLI" serve --model "$MODEL" --host "$HOST" --port "$PORT" \
        ${extra[@]+"${extra[@]}"} > "$LOG" 2>&1 &
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
    # IMPORTANT: don't spawn python in a per-chunk hot loop — that alone
    # burned 50ms × N chunks and capped "measured" tps at 6. Do all the
    # timing inline in one python process that reads curl's stdout.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    python3 <<PY
import json, subprocess, time, sys
body = {
    "model": "$model_id",
    "messages": [{"role":"user","content":"Write a detailed 200 word paragraph about the history of computing. Cover Turing, ENIAC, transistors, and the invention of the transistor. Be thorough and precise."}],
    "max_tokens": 200,
    "stream": True,
    "temperature": 0,
}
t0 = time.time()
proc = subprocess.Popen(
    ["curl","-sN","-X","POST","$BASE/v1/chat/completions",
     "-H","content-type: application/json","-d", json.dumps(body)],
    stdout=subprocess.PIPE, text=True)
first_ms = None
last_ms = None
tok = 0
for raw in proc.stdout:
    if not raw.startswith("data:"): continue
    if raw.strip() == "data: [DONE]": break
    now_ms = int((time.time() - t0) * 1000)
    if first_ms is None: first_ms = now_ms
    last_ms = now_ms
    tok += 1
proc.wait(timeout=2)
decode_ms = (last_ms or 0) - (first_ms or 0)
tps = 0.0
if decode_ms > 0 and tok > 1:
    tps = round((tok - 1) * 1000.0 / decode_ms, 1)
ok = "true" if (tok >= 20 and first_ms is not None) else "false"
print(json.dumps({
    "name":"sse_stream","ok":ok == "true","ttft_ms":first_ms or 0,
    "tokens":tok,"tps":tps,"notes":f"decode_ms={decode_ms}"
}))
PY
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

case_embeddings() {
    # Requires an embedding-capable model (bert / qwen3-embedding / etc).
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local resp
    resp=$(curl -s -X POST "$BASE/v1/embeddings" \
        -H "content-type: application/json" \
        -d "{\"model\":\"$model_id\",\"input\":[\"hello world\",\"goodbye world\"]}")
    local dim=$(echo "$resp" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(len(d["data"][0]["embedding"])) if d.get("data") else print(0)' 2>/dev/null || echo 0)
    local ok=false
    [ "$dim" -gt 64 ] && ok=true
    emit "{\"name\":\"embeddings\",\"ok\":$ok,\"notes\":\"dim=$dim\"}"
}

case_vision_chat() {
    # Generate a 64×64 solid-red PNG on the fly — Qwen-VL / Gemma-VL
    # preprocessors require images ≥ 32×32 (patch factor). A 1-pixel
    # image trips `Height: 1 must be larger than factor: 32`.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local img=$(python3 <<'PY'
import base64, struct, zlib
w = h = 64
def chunk(tag, data):
    crc = zlib.crc32(tag + data) & 0xffffffff
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
# scanlines: filter byte 0 + 3-byte RGB per pixel, solid red
raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
idat = zlib.compress(raw)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
print("data:image/png;base64," + base64.b64encode(png).decode())
PY
)
    local resp
    resp=$(curl -s -X POST "$BASE/v1/chat/completions" \
        -H "content-type: application/json" \
        -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What color is this image? One word.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"$img\"}}]}],\"max_tokens\":16,\"temperature\":0}")
    local content=$(echo "$resp" | extract_content)
    local err=$(echo "$resp" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("error",{}).get("message","")[:80])' 2>/dev/null)
    local ok=false
    [ -n "$content" ] && [ -z "$err" ] && ok=true
    local note="$content"
    [ -n "$err" ] && note="ERR: $err"
    emit "{\"name\":\"vision_chat\",\"ok\":$ok,\"notes\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1][:80]))' "$note")}"
}

case_audio_transcription() {
    # Use the whisper model's own test wav if present; else skip.
    local wav=""
    for cand in \
        /Users/eric/.cache/huggingface/hub/models--mlx-community--whisper-tiny-mlx/snapshots/*/test.wav \
        /Users/eric/vmlx/swift/tests/e2e/fixtures/hello.wav; do
        eval "expanded=$cand" 2>/dev/null
        [ -f "$expanded" ] && wav="$expanded" && break
    done
    if [ -z "$wav" ]; then
        # Generate a 1-second silent wav on the fly with python so the
        # harness doesn't require a checked-in fixture.
        wav=/tmp/vmlx-test-silent.wav
        python3 - <<PY
import struct, wave
with wave.open("$wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b"\\x00\\x00" * 16000)
PY
    fi
    local resp
    resp=$(curl -s -X POST "$BASE/v1/audio/transcriptions" \
        -F "file=@$wav" -F "model=whisper-tiny-mlx")
    local text=$(echo "$resp" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("text",""))' 2>/dev/null)
    # Even a silent wav should return a valid (possibly empty) text field.
    local has_text=$(echo "$resp" | python3 -c 'import json,sys;d=json.load(sys.stdin);print("yes" if "text" in d else "no")' 2>/dev/null)
    local ok=false
    [ "$has_text" = "yes" ] && ok=true
    emit "{\"name\":\"audio_transcription\",\"ok\":$ok,\"notes\":$(python3 -c 'import json,sys;print(json.dumps((sys.argv[1] or "[silent]")[:60]))' "$text")}"
}

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

suite_vl() {
    # VL models: skip max_tokens/stop-sequence chat-template tests that
    # degenerate on a vision-stamp-heavy template; keep the basic chat +
    # vision_chat as the marker case.
    case_models_list
    case_basic_chat
    case_sse_stream
    case_vision_chat
    case_multiturn_prefix_cache
    case_concurrent
}

suite_embedding() {
    case_models_list
    case_metrics
    case_embeddings
}

suite_audio() {
    case_models_list
    case_audio_transcription
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
    vl)         suite_vl ;;
    embedding)  suite_embedding ;;
    audio)      suite_audio ;;
    *)          suite_smoke ;;
esac

emit "{\"name\":\"done\",\"ok\":true}"
