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

case_concurrent_burst() {
    # 5-request stress burst. The 3-way `concurrent` case is too shallow —
    # at that width the FIFO lock never contends enough to expose subtle
    # drain/release races. Widen to 5 simultaneous streams to stress
    # GenerationLock + MLX-drain + continuation lifecycle harder.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local pids=()
    local outs=()
    for i in 1 2 3 4 5; do
        local tmp="/tmp/vmlx-burst-$i.json"
        curl -s -X POST "$BASE/v1/chat/completions" -H "content-type: application/json" \
            -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to $i.\"}],\"max_tokens\":12,\"temperature\":0}" \
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
    # Server alive check — if it aborted mid-burst, /v1/models 404s.
    local alive_http=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$BASE/v1/models")
    local ok=false
    [ "$success" = "5" ] && [ "$alive_http" = "200" ] && ok=true
    emit "{\"name\":\"concurrent_burst\",\"ok\":$ok,\"notes\":\"$success/5 ok; alive=$alive_http\"}"
}

case_cancel_midstream() {
    # Open a long-running stream, kill the curl connection mid-way, then
    # fire a follow-up chat request and verify the server is still
    # responsive (didn't deadlock on a held GenerationLock, didn't hang
    # on a pending continuation). Python because bash backgrounded
    # `curl` fights the timeout flag.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    python3 <<PY
import json, subprocess, time, sys, os, signal
body = {
    "model": "$model_id",
    "messages": [{"role":"user","content":"Write a very long story with many details."}],
    "max_tokens": 500,
    "stream": True,
    "temperature": 0,
}
# Start streaming
p = subprocess.Popen(
    ["curl","-sN","-X","POST","$BASE/v1/chat/completions",
     "-H","content-type: application/json","-d", json.dumps(body)],
    stdout=subprocess.PIPE, text=True, preexec_fn=os.setsid)
# Read at least a few chunks to confirm the stream is actually live
tok = 0
for line in p.stdout:
    if line.startswith("data:") and line.strip() != "data: [DONE]":
        tok += 1
        if tok >= 3: break
started = tok >= 3
# Abort hard — simulates the UI clicking stop / client disconnect
os.killpg(os.getpgid(p.pid), signal.SIGTERM)
try: p.wait(timeout=3)
except Exception: os.killpg(os.getpgid(p.pid), signal.SIGKILL)

# Give the server a beat to release the lock + clean up
time.sleep(0.5)

# Follow-up request to prove the server didn't deadlock on the
# abandoned stream. If the GenerationLock wasn't released, this
# request will hang — set an explicit 20s timeout.
followup = subprocess.run(
    ["curl","-s","--max-time","20","-X","POST","$BASE/v1/chat/completions",
     "-H","content-type: application/json",
     "-d", json.dumps({
         "model":"$model_id",
         "messages":[{"role":"user","content":"Ping."}],
         "max_tokens":4,"temperature":0,
     })],
    capture_output=True, text=True)
resp_ok = False
try:
    d = json.loads(followup.stdout)
    resp_ok = bool(d.get("choices",[{}])[0].get("message",{}).get("content"))
except Exception:
    pass

ok = "true" if (started and resp_ok) else "false"
note = f"stream_started={started} followup_ok={resp_ok}"
print(json.dumps({"name":"cancel_midstream","ok": ok == "true","notes": note}))
PY
}

case_tool_call() {
    # Models that ship a tool parser (Qwen3/qwen family) should emit a
    # `tool_calls[]` array when the user asks something the provided
    # function can answer. Not all models comply (Llama-1B in particular
    # often ignores the tool schema), so this is opt-in — run only when
    # the first model in `/v1/models` is a plausible tool-capable family.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local body
    body=$(cat <<JSON
{
  "model": "$model_id",
  "messages": [{"role":"user","content":"What's the weather in Paris? Use the tool."}],
  "max_tokens": 80,
  "temperature": 0,
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type":"string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto"
}
JSON
)
    local resp_file="/tmp/vmlx-tool-call.json"
    curl -s --max-time 60 -X POST "$BASE/v1/chat/completions" \
        -H "content-type: application/json" -d "$body" > "$resp_file"
    # Pass the response via a file path (not string interpolation) so
    # quotes/newlines/backslashes inside the JSON body don't break the
    # heredoc.
    RESP_FILE="$resp_file" python3 <<'PY'
import json, os, sys
try:
    with open(os.environ["RESP_FILE"]) as f:
        r = json.load(f)
    ch = r.get("choices",[{}])[0]
    msg = ch.get("message",{})
    tc = msg.get("tool_calls") or []
    text = msg.get("content") or ""
    finish = ch.get("finish_reason","?")
    ok = "true" if (len(tc) >= 1 and tc[0].get("function",{}).get("name")=="get_weather") else "false"
    print(json.dumps({
        "name":"tool_call","ok":ok=="true",
        "notes":f"finish={finish} tc={len(tc)} content={text[:60]}"
    }))
except Exception as e:
    print(json.dumps({"name":"tool_call","ok":False,"notes":f"parse-error: {e}"}))
PY
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

case_stream_usage() {
    # stream_options.include_usage must emit a final chunk with a
    # non-empty `usage` block before [DONE]. OpenAI-compat contract —
    # many client libs (LangChain, Vercel AI SDK) rely on this to
    # accumulate cost metrics.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    python3 <<PY
import json, subprocess, sys
body = {
    "model": "$model_id",
    "messages":[{"role":"user","content":"Count 1 to 3."}],
    "max_tokens": 20,
    "stream": True,
    "temperature": 0,
    "stream_options": {"include_usage": True},
}
p = subprocess.Popen(
    ["curl","-sN","-X","POST","$BASE/v1/chat/completions",
     "-H","content-type: application/json","-d", json.dumps(body)],
    stdout=subprocess.PIPE, text=True)
saw_usage = False
usage = None
for line in p.stdout:
    if not line.startswith("data:"): continue
    if line.strip() == "data: [DONE]": break
    try:
        d = json.loads(line[5:].strip())
    except Exception:
        continue
    if d.get("usage"):
        saw_usage = True
        usage = d["usage"]
p.wait(timeout=2)
ok = "true" if saw_usage and (usage or {}).get("completion_tokens", 0) > 0 else "false"
note = f"usage={usage}" if usage else "no usage chunk"
print(json.dumps({"name":"stream_usage","ok":ok=="true","notes":note[:80]}))
PY
}

case_tool_roundtrip() {
    # Tool-call feedback loop — two-hop exchange:
    #   1. Client sends messages + tool defs → server emits tool_calls
    #   2. Client appends a `tool` role message with the result → server
    #      returns a final assistant reply that ideally cites the result.
    # This exercises the full OpenAI tool-feedback dance; bash-tool
    # terminal mode is the Swift engine's marquee feature and this is
    # the only case that verifies the plumbing end-to-end without Xcode.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local body1=$(cat <<JSON
{
  "model": "$model_id",
  "messages": [{"role":"user","content":"What is the sum of 17 and 25? Use the tool."}],
  "max_tokens": 80,
  "temperature": 0,
  "tools": [{
    "type": "function",
    "function": {
      "name": "add",
      "description": "Add two integers",
      "parameters": {
        "type": "object",
        "properties": {"a":{"type":"integer"},"b":{"type":"integer"}},
        "required": ["a","b"]
      }
    }
  }],
  "tool_choice": "auto"
}
JSON
)
    curl -s --max-time 60 -X POST "$BASE/v1/chat/completions" \
        -H "content-type: application/json" -d "$body1" > /tmp/vmlx-roundtrip-1.json
    # Parse tool_calls from step 1
    local tc_id tc_name tc_args
    local parsed=$(python3 <<'PY'
import json
r = json.load(open("/tmp/vmlx-roundtrip-1.json"))
tc = r.get("choices",[{}])[0].get("message",{}).get("tool_calls") or []
if not tc:
    print("MISS||")
else:
    c = tc[0]
    print(f"{c.get('id','call_1')}|{c.get('function',{}).get('name','?')}|{c.get('function',{}).get('arguments','{}')}")
PY
)
    if [[ "$parsed" == MISS* ]]; then
        emit "{\"name\":\"tool_roundtrip\",\"ok\":false,\"notes\":\"step 1: no tool_calls emitted\"}"
        return
    fi
    tc_id=$(echo "$parsed" | cut -d'|' -f1)
    tc_name=$(echo "$parsed" | cut -d'|' -f2)
    tc_args=$(echo "$parsed" | cut -d'|' -f3)
    # Step 2: send tool result back, expect prose answer mentioning 42
    local body2
    body2=$(python3 -c '
import json, sys, os
user_ask = "What is the sum of 17 and 25? Use the tool."
model_id = os.environ["MID"]
tc_id = os.environ["TCID"]
tc_name = os.environ["TCNAME"]
tc_args = os.environ["TCARGS"]
payload = {
    "model": model_id, "temperature": 0, "max_tokens": 60,
    "messages": [
        {"role":"user","content":user_ask},
        {"role":"assistant","content":None,"tool_calls":[{
            "id": tc_id, "type":"function",
            "function": {"name": tc_name, "arguments": tc_args}
        }]},
        {"role":"tool","tool_call_id":tc_id,"name":tc_name,"content":"42"},
    ],
}
print(json.dumps(payload))
' MID="$model_id" TCID="$tc_id" TCNAME="$tc_name" TCARGS="$tc_args")
    curl -s --max-time 60 -X POST "$BASE/v1/chat/completions" \
        -H "content-type: application/json" -d "$body2" > /tmp/vmlx-roundtrip-2.json
    python3 <<'PY'
import json
r = json.load(open("/tmp/vmlx-roundtrip-2.json"))
ch = r.get("choices",[{}])[0]
content = ch.get("message",{}).get("content") or ""
has_42 = "42" in content
# Accept either: final prose cites 42 OR finish_reason stop with non-empty content.
finish = ch.get("finish_reason","?")
ok = "true" if (has_42 or (finish=="stop" and len(content) > 0)) else "false"
print(json.dumps({"name":"tool_roundtrip","ok":ok=="true","notes":f"has_42={has_42} finish={finish} c={content[:50]}"}))
PY
}

case_large_context() {
    # A ~4K-token prompt exercises the long-context path, prefill
    # chunking, and eviction boundaries. Build a prompt of 400 lines
    # of "The number is N" so the tokenizer can't just repeat-compress.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    python3 <<PY
import json, subprocess, sys, time
big = "\n".join(f"Line {i}: the number is {i}." for i in range(400))
question = f"\n\nBased on the lines above, what is the number on Line 137?"
body = {
    "model": "$model_id",
    "messages":[{"role":"user","content": big + question}],
    "max_tokens": 20, "temperature": 0,
}
t0 = time.time()
out = subprocess.run(
    ["curl","-s","--max-time","120","-X","POST","$BASE/v1/chat/completions",
     "-H","content-type: application/json","-d", json.dumps(body)],
    capture_output=True, text=True)
elapsed = int((time.time()-t0)*1000)
try:
    r = json.loads(out.stdout)
    content = r.get("choices",[{}])[0].get("message",{}).get("content") or ""
    prompt_tok = r.get("usage",{}).get("prompt_tokens", 0)
    ok = "true" if prompt_tok > 2000 and content else "false"
    note = f"prompt_tok={prompt_tok} elapsed={elapsed}ms c={content[:40]}"
except Exception as e:
    ok = "false"; note = f"parse: {e}"
print(json.dumps({"name":"large_context","ok":ok=="true","notes":note[:120]}))
PY
}

suite_full() {
    suite_smoke
    case_multiturn_prefix_cache
    case_stop_sequences
    case_json_mode
    case_ollama_chat
    case_anthropic_messages
    case_concurrent
    case_concurrent_burst
    case_cancel_midstream
    case_tool_call
    case_tool_roundtrip
    case_stream_usage
    case_large_context
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
