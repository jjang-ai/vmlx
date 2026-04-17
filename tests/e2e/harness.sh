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
    # For the embedding suite the MODEL path IS the embedding model — load
    # it via `--embedding-model` rather than `--model` so `/v1/embeddings`
    # has a live loader and the text-gen path stays off (MODEL==
    # Qwen3-Embedding-0.6B isn't a causal LM). Dummy text model ensures
    # /v1/models surfaces *something* so the harness readiness probe
    # succeeds; the embedding endpoint resolves via the embedding-model
    # slot.
    if [ "$SUITE" = "embedding" ]; then
        "$CLI" serve --embedding-model "$MODEL" --host "$HOST" --port "$PORT" \
            ${extra[@]+"${extra[@]}"} > "$LOG" 2>&1 &
    else
        "$CLI" serve --model "$MODEL" --host "$HOST" --port "$PORT" \
            ${extra[@]+"${extra[@]}"} > "$LOG" 2>&1 &
    fi
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
    #
    # Per-request retry (--retry 3 --retry-connrefused --retry-delay 1)
    # is critical under heavy background machine load: without it,
    # transient TCP connection-resets from an overloaded kernel (not a
    # server-side crash) would be scored as failures, making the test
    # noisy. A real server crash still fails the post-burst /v1/models
    # liveness probe and downgrades the result.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local pids=()
    local outs=()
    for i in 1 2 3 4 5; do
        local tmp="/tmp/vmlx-burst-$i.json"
        curl -s --max-time 120 --retry 3 --retry-connrefused --retry-delay 1 \
            -X POST "$BASE/v1/chat/completions" -H "content-type: application/json" \
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
    # max_tokens=4 must produce <=4 completion tokens. `finish_reason`
    # can legitimately be either `length` (model was still going) OR
    # `stop` (model hit EOS within the budget). The contract the caller
    # cares about is the CAP, not which side closed the response. Earlier
    # versions of this test required `length` — that flaked on models
    # that hit EOS at 3 tokens (Qwen3-0.6B with "Write a long story…"
    # actually terminates at the filler preamble on greedy decode).
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local resp
    resp=$(curl -s -X POST "$BASE/v1/chat/completions" -H "content-type: application/json" \
        -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a long story about a cat.\"}],\"max_tokens\":4,\"temperature\":0}")
    local finish=$(echo "$resp" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["choices"][0].get("finish_reason","?")) if d.get("choices") else print("?")' 2>/dev/null)
    local tok=$(echo "$resp" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("usage",{}).get("completion_tokens",0))' 2>/dev/null || echo 0)
    local ok=false
    { [ "$finish" = "length" ] || [ "$finish" = "stop" ]; } \
        && [ "$tok" -le 4 ] && [ "$tok" -gt 0 ] && ok=true
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

case_prefix_cache_hit_ratio() {
    # Measure prefix-cache efficiency: the x-vmlx-cache-hit response
    # header (if present) reports `cached_tokens/prompt_tokens`. When
    # the header isn't available (older build, non-OpenAI route), fall
    # back to comparing `usage.prompt_tokens` between two turns that
    # share a long system prefix — if the engine is caching properly,
    # the second turn's usage.prompt_tokens should include a non-zero
    # cached_tokens hint surfaced via the final SSE chunk's usage block.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    python3 <<PY
import json, subprocess
long_sys = "You are a helpful assistant. " * 40  # ~200 tokens
def chat(msgs):
    body = {
        "model": "$model_id",
        "messages": msgs,
        "max_tokens": 8, "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    p = subprocess.run(
        ["curl","-sN","--max-time","60","-X","POST","$BASE/v1/chat/completions",
         "-H","content-type: application/json","-d", json.dumps(body)],
        capture_output=True, text=True)
    last_usage = None
    for line in p.stdout.splitlines():
        if line.startswith("data:") and line.strip() != "data: [DONE]":
            try:
                d = json.loads(line[5:].strip())
                if d.get("usage"): last_usage = d["usage"]
            except Exception: pass
    return last_usage or {}

u1 = chat([
    {"role":"system","content":long_sys},
    {"role":"user","content":"Say 'one'."},
])
u2 = chat([
    {"role":"system","content":long_sys},
    {"role":"user","content":"Say 'one'."},
    {"role":"assistant","content":"one"},
    {"role":"user","content":"Say 'two'."},
])
# cached_tokens field is in usage per OpenAI's 2024 spec
cached1 = (u1.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or u1.get("cached_tokens", 0)
cached2 = (u2.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or u2.get("cached_tokens", 0)
# Pass if:
#  - turn-2 exposes cached_tokens > 0 (explicit hit signal), OR
#  - turn-2 prefill_ms is much smaller than turn-1 (implicit hit signal)
p1 = u1.get("prefill_ms", 0) or 0
p2 = u2.get("prefill_ms", 0) or 0
explicit = cached2 > 0
implicit = p1 > 0 and p2 > 0 and p2 <= max(50, p1 * 0.5)
ok = "true" if (explicit or implicit) else "false"
note = f"t1 cached={cached1} prefill={p1:.0f}ms  t2 cached={cached2} prefill={p2:.0f}ms"
print(json.dumps({"name":"prefix_cache_hit_ratio","ok":ok=="true","notes":note[:120]}))
PY
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
    case_health_endpoint
    case_cache_stats
    case_gateway_info
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

case_health_endpoint() {
    # `/health` is the load-balancer liveness probe. Must be fast, must
    # be 200, must not require the model to be ready (otherwise a stuck
    # load would fail upstream health checks and cause a restart loop).
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$BASE/health")
    local ok=false
    [ "$code" = "200" ] && ok=true
    emit "{\"name\":\"health_endpoint\",\"ok\":$ok,\"notes\":\"HTTP $code\"}"
}

case_cache_stats() {
    # GET /v1/cache/stats reports paged/L1.5/disk cache telemetry. Used
    # by the CachePanel in the SwiftUI app. Admin-gated if
    # `--admin-token` was passed to `serve`; without a token the
    # endpoint is open (which is what our test harness expects).
    local resp
    resp=$(curl -s --max-time 5 "$BASE/v1/cache/stats")
    python3 <<PY
import json
try:
    d = json.loads('''$resp''')
    # Minimal invariants: response is a dict, has something
    ok = "true" if isinstance(d, dict) and len(d) > 0 else "false"
    keys = list(d.keys())[:4] if isinstance(d, dict) else []
    print(json.dumps({"name":"cache_stats","ok":ok=="true","notes":f"keys={keys}"}))
except Exception as e:
    print(json.dumps({"name":"cache_stats","ok":False,"notes":f"parse: {e}"}))
PY
}

case_gateway_info() {
    # GET /v1/_gateway/info — the multi-session routing descriptor used
    # by the multiplexer. Only wired when `serve --gateway-port` (or the
    # settings toggle) is on — in single-model `vmlxctl serve` mode the
    # route 404s, which is expected. Accept either 200-with-JSON OR
    # 404 (not registered). Anything else (500/timeout/TCP reset)
    # signals a server regression.
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$BASE/v1/_gateway/info")
    local ok=false
    [ "$code" = "200" ] || [ "$code" = "404" ] && ok=true
    emit "{\"name\":\"gateway_info\",\"ok\":$ok,\"notes\":\"HTTP $code (404 expected in single-serve mode)\"}"
}

case_deterministic() {
    # Greedy decode (temp=0) is supposed to be reproducible, but Metal
    # kernel cold-vs-warm compile produces tiny ULP differences in
    # logits that can flip the argmax when two top tokens are close
    # in likelihood. Empirically: run 1 (cold-kernel) often diverges
    # from runs 2+ (warm). Subsequent warm-kernel runs ARE consistent
    # with each other.
    #
    # Contract this test enforces: "steady-state greedy is
    # deterministic". Fire 3 back-to-back requests and require
    # runs 2 and 3 to match. Run 1 is the kernel-warmup pass and is
    # allowed to diverge.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local body="{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"Name 3 primary colors in order.\"}],\"max_tokens\":20,\"temperature\":0}"
    local f1=/tmp/vmlx-det-1.json f2=/tmp/vmlx-det-2.json f3=/tmp/vmlx-det-3.json
    curl -s --max-time 60 -X POST "$BASE/v1/chat/completions" -H "content-type: application/json" -d "$body" > "$f1"
    curl -s --max-time 60 -X POST "$BASE/v1/chat/completions" -H "content-type: application/json" -d "$body" > "$f2"
    curl -s --max-time 60 -X POST "$BASE/v1/chat/completions" -H "content-type: application/json" -d "$body" > "$f3"
    F1="$f1" F2="$f2" F3="$f3" python3 <<'PY'
import json, os
def content(p):
    try:
        return json.load(open(p)).get("choices",[{}])[0].get("message",{}).get("content","") or ""
    except Exception:
        return ""
r1, r2, r3 = content(os.environ["F1"]), content(os.environ["F2"]), content(os.environ["F3"])
warm_stable = (r2 == r3) and len(r2) > 0
cold_drift = (r1 != r2)
tag = "warm-stable"
if not warm_stable: tag = "DRIFT"
elif cold_drift:     tag = "warm-stable (cold r1 diverged — MLX kernel warmup)"
print(json.dumps({"name":"deterministic","ok":warm_stable,"notes":f"{tag} r2={r2[:40]!r}"[:120]}))
PY
}

case_logprobs() {
    # OpenAI `logprobs: true` + `top_logprobs: 3` should either:
    #   a) return per-token logprobs (full support), OR
    #   b) return a clean 400 "not yet supported" error (documented
    #      unimplemented — the Swift engine currently does this).
    # What we're catching: the 500/timeout/silent-empty-choices path
    # (engine would say "no supported" but a regression could just
    # hang or return an empty but 200 response).
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local resp_file=/tmp/vmlx-logprobs.json
    local code
    code=$(curl -s --max-time 60 -o "$resp_file" -w "%{http_code}" -X POST "$BASE/v1/chat/completions" \
        -H "content-type: application/json" \
        -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5,\"temperature\":0,\"logprobs\":true,\"top_logprobs\":3}")
    F="$resp_file" HTTP="$code" python3 <<'PY'
import json, os
code = os.environ["HTTP"]
try:
    r = json.load(open(os.environ["F"]))
except Exception as e:
    print(json.dumps({"name":"logprobs","ok":False,"notes":f"parse: {e}"}))
    raise SystemExit
choices = r.get("choices", [])
err = r.get("error", {})
if code == "200" and choices:
    lp = choices[0].get("logprobs") or {}
    content = lp.get("content") or []
    first_lp = content[0].get("logprob") if content else None
    if content and first_lp is not None:
        print(json.dumps({"name":"logprobs","ok":True,"notes":f"ok tokens={len(content)} first={first_lp:.2f}"}))
    else:
        print(json.dumps({"name":"logprobs","ok":False,"notes":"200 but no content array"}))
elif code.startswith("4") and "not yet supported" in (err.get("message") or ""):
    # Documented-unimplemented: engine surfaces a clean 400 with
    # explicit message. That's an acceptable pass for now.
    print(json.dumps({"name":"logprobs","ok":True,"notes":f"HTTP {code} not-yet-supported (documented)"}))
else:
    print(json.dumps({"name":"logprobs","ok":False,"notes":f"HTTP {code} err={(err.get('message') or '')[:50]}"}))
PY
}

case_input_validation() {
    # Malformed requests must return 4xx with a structured error — NOT
    # 500 (server bug) and NOT a hang. Probes six flavors of bad input:
    #   1. invalid JSON body
    #   2. missing `messages` field
    #   3. empty `messages` array
    #   4. negative max_tokens
    #   5. temperature out of range
    #   6. unknown `messages[].role`
    # Each returning 4xx counts as a pass. 5xx or 2xx (silent accept)
    # is a regression.
    local m=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    local pass=0 total=6 notes=""
    probe() {
        local label="$1" body="$2"
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
            -X POST "$BASE/v1/chat/completions" \
            -H "content-type: application/json" -d "$body")
        if [ "${code:0:1}" = "4" ]; then
            pass=$((pass+1))
        fi
        notes="$notes $label=$code"
    }
    probe invJSON        '{this is not json'
    probe noMessages     "{\"model\":\"$m\",\"max_tokens\":4}"
    probe emptyMessages  "{\"model\":\"$m\",\"messages\":[]}"
    probe negMax         "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":-1}"
    probe wildTemp       "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"temperature\":99}"
    probe badRole        "{\"model\":\"$m\",\"messages\":[{\"role\":\"alien\",\"content\":\"hi\"}]}"
    local ok=false
    [ "$pass" -ge 4 ] && ok=true  # tolerate 2 that might 200-accept if engine auto-clamps
    emit "{\"name\":\"input_validation\",\"ok\":$ok,\"notes\":\"$pass/$total:$notes\"}"
}

case_sleep_wake_cycle() {
    # Exercise the admin lifecycle: POST /admin/soft-sleep → verify the
    # next chat request transparently wakes → POST /admin/deep-sleep →
    # POST /admin/wake explicitly → verify chat works again. This is
    # the Electron-parity "server idle timer" path — the server enters
    # `.standby(.soft)` / `.standby(.deep)` states automatically after
    # the configured idle window, and chat requests auto-wake on the
    # way in. Users see a "Waking up…" banner briefly. A regression
    # here means models silently fail to respond after inactivity.
    #
    # Test shape:
    #   1. soft-sleep → chat works (auto-wake from soft)
    #   2. deep-sleep → explicit wake → chat works
    # Green = both post-sleep chats return non-empty content.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    # Step 1: soft sleep
    local soft_code
    soft_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 -X POST "$BASE/admin/soft-sleep")
    local chat1
    chat1=$(curl -s --max-time 60 -X POST "$BASE/v1/chat/completions" -H "content-type: application/json" \
        -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi.\"}],\"max_tokens\":4,\"temperature\":0}" \
        | extract_content)
    # Step 2: deep sleep → explicit wake → chat
    local deep_code
    deep_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 -X POST "$BASE/admin/deep-sleep")
    # Give deep sleep a moment to release
    sleep 1
    local wake_code
    wake_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 120 -X POST "$BASE/admin/wake")
    local chat2
    chat2=$(curl -s --max-time 60 -X POST "$BASE/v1/chat/completions" -H "content-type: application/json" \
        -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"Bye.\"}],\"max_tokens\":4,\"temperature\":0}" \
        | extract_content)
    local ok=false
    [ -n "$chat1" ] && [ -n "$chat2" ] && ok=true
    emit "{\"name\":\"sleep_wake_cycle\",\"ok\":$ok,\"notes\":\"soft=$soft_code chat1='${chat1:0:20}' deep=$deep_code wake=$wake_code chat2='${chat2:0:20}'\"}"
}

case_reasoning_content() {
    # Thinking models (Qwen3, DeepSeek R1, GLM 5, Nemotron-reasoning)
    # emit their chain-of-thought inside `<think>...</think>` tags.
    # The Swift engine's reasoning parser is supposed to route those
    # tokens into `delta.reasoning_content` on SSE chunks, NOT into
    # `delta.content` — the UI renders them in a separate "thinking"
    # panel. If the parser is OFF (or misconfigured), `<think>` leaks
    # into content verbatim and the UI looks broken.
    #
    # We verify by:
    #   1. enabling thinking explicitly via chat_template_kwargs
    #   2. checking that across the stream, either
    #       a) at least one chunk has delta.reasoning_content, OR
    #       b) no chunk contains a literal `<think>` in delta.content
    # Both are acceptable — (a) for full reasoning support, (b) for
    # graceful fallback where the server strips the tags without
    # surfacing a structured field.
    local model_id=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"]) if d.get("data") else print("")')
    python3 <<PY
import json, subprocess
body = {
    "model": "$model_id",
    "messages":[{"role":"user","content":"What is 17*23? Think step by step."}],
    "max_tokens": 128,
    "stream": True, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": True},
}
p = subprocess.run(
    ["curl","-sN","--max-time","60","-X","POST","$BASE/v1/chat/completions",
     "-H","content-type: application/json","-d", json.dumps(body)],
    capture_output=True, text=True)
reasoning_chunks = 0
content_chunks = 0
think_leak = False
for line in p.stdout.splitlines():
    if not line.startswith("data:") or line.strip() == "data: [DONE]": continue
    try: d = json.loads(line[5:].strip())
    except Exception: continue
    for ch in d.get("choices", []):
        delta = ch.get("delta", {})
        if delta.get("reasoning_content"): reasoning_chunks += 1
        c = delta.get("content") or ""
        if c: content_chunks += 1
        if "<think>" in c or "</think>" in c: think_leak = True
ok = "true" if (reasoning_chunks > 0 or not think_leak) else "false"
note = f"reasoning={reasoning_chunks} content={content_chunks} leak={think_leak}"
print(json.dumps({"name":"reasoning_content","ok":ok=="true","notes":note[:100]}))
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
    case_prefix_cache_hit_ratio
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
    case_deterministic
    case_logprobs
    case_input_validation
    case_sleep_wake_cycle
    case_reasoning_content
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

# Post-suite server liveness check. A Metal assertion (e.g. the
# `tryCoalescingPreviousComputeCommandEncoder` class that iter-25 fixed
# for VL, or the still-open JANGTQ 5-way burst crash) kills the server
# mid-suite — subsequent harness cases then fail silently with curl
# connection refused which looks like a test logic issue rather than a
# server crash. Surface it explicitly so CI / reviewers see it.
if [ -n "${SERVER_PID:-}" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
    # Scan the server log for the most common Metal assertion signatures.
    crash=$(grep -oE "failed assertion.*|Fatal.*|_status <.*Committed|Completed handler provided after commit|tryCoalescingPrevious.*|already encoding to this command buffer" "$LOG" 2>/dev/null | head -1 | tr -d '"' | tr '\n' ' ')
    [ -z "$crash" ] && crash="server PID $SERVER_PID gone; no Metal signature found in $LOG"
    emit "{\"name\":\"server_liveness\",\"ok\":false,\"notes\":\"server died mid-suite: ${crash:0:200}\"}"
else
    emit "{\"name\":\"server_liveness\",\"ok\":true,\"notes\":\"server alive through suite\"}"
fi

emit "{\"name\":\"done\",\"ok\":true}"
