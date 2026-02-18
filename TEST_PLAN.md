# vMLX Feature Verification Test Plan

## Setup

Pick a small model for fast testing. Recommended: `mlx-community/Qwen3-0.6B-4bit` or any small MLX model you have locally.

All `curl` commands assume the server is at `localhost:PORT` — replace PORT with actual port shown in the session header.

---

## 1. KV Cache Quantization

**What it does:** Compresses KV cache entries from fp16 to 4-bit or 8-bit, reducing memory per cached token.

### Configure
- Create/edit session → Server Settings → KV Cache Quantization section
- Set quantization to `4` (4-bit) or `8` (8-bit)
- Keep continuous batching ON, prefix cache ON, paged cache ON
- Start session

### Verify
```bash
# Health endpoint reports KV quant status
curl -s http://localhost:PORT/health | python3 -m json.tool

# Look for:
# "kv_cache_quantization": {
#     "enabled": true,
#     "bits": 4,
#     "group_size": 64
# }
```

```bash
# Cache stats also report it
curl -s http://localhost:PORT/v1/cache/stats | python3 -m json.tool

# Look for kv_cache_quantization section
```

### Compare memory
1. Start session WITHOUT KV quant → send a long prompt → check `/health` memory `active_mb`
2. Stop, restart WITH `--kv-cache-quantization 4` → send same prompt → check `active_mb`
3. 4-bit should use noticeably less memory for cached states

---

## 2. Paged KV Cache

**What it does:** Allocates KV cache in fixed-size blocks instead of contiguous memory. Reduces fragmentation.

### Configure
- Server Settings → Paged KV Cache → enabled, block size 64, max blocks 1000
- Start session

### Verify
```bash
curl -s http://localhost:PORT/v1/cache/stats | python3 -m json.tool

# Look for scheduler_cache with paged-cache-specific fields:
# "block_size", "max_blocks", "allocated_blocks", "free_blocks",
# "shared_blocks", "total_tokens_cached", "utilization", "cache_hit_rate"
```

Send same prompt twice:
```bash
# First request (cache miss)
curl -s http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Explain quantum computing in detail."}],"max_tokens":50}'

# Second request (cache hit — should be faster TTFT)
curl -s http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Explain quantum computing in detail."}],"max_tokens":50}'

# Check stats again — cache_hits should increment
curl -s http://localhost:PORT/v1/cache/stats | python3 -m json.tool
```

---

## 3. Persistent Disk Cache

**What it does:** Saves prefix cache entries to disk as .safetensors files. Survives server restarts.

### Configure
- Server Settings → Disk Cache → Enable
- Set max size (e.g., 10 GB)
- Set a custom directory (e.g., `/tmp/vmlx-disk-cache-test`) for easy inspection
- **IMPORTANT:** Paged cache must be OFF (they're incompatible — UI should show warning)
- Start session

### Verify
```bash
# Send a prompt to populate the cache
curl -s http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"What is the capital of France?"}],"max_tokens":20}'

# Check disk cache stats
curl -s http://localhost:PORT/v1/cache/stats | python3 -m json.tool

# Look for:
# "disk_cache": {
#     "entries": 1,        ← should be > 0
#     "total_size_mb": ...,
#     "hits": 0,
#     "misses": 1,
#     "stores": 1          ← should be > 0
# }
```

```bash
# Verify files actually exist on disk
ls -la /tmp/vmlx-disk-cache-test/

# Should see .safetensors files (or model-specific subdirectory)
```

**Persistence test:**
1. Note the disk cache stats (entries, stores)
2. Stop the session
3. Start the session again (same config)
4. Send the SAME prompt again
5. Check `/v1/cache/stats` — `disk_cache.hits` should be > 0 (loaded from disk instead of recomputing)

---

## 4. Prefix Cache (Memory)

**What it does:** Caches computed attention states for common prompt prefixes. Reduces TTFT on repeated system prompts.

### Verify
```bash
# Send first request
curl -s http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"system","content":"You are a very detailed expert on biology."},{"role":"user","content":"What is DNA?"}],"max_tokens":30}'

# Check cache entries
curl -s http://localhost:PORT/v1/cache/entries | python3 -m json.tool

# Send second request sharing same system prompt
curl -s http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"system","content":"You are a very detailed expert on biology."},{"role":"user","content":"What is RNA?"}],"max_tokens":30}'

# Check cache stats — hit count should increase
curl -s http://localhost:PORT/v1/cache/stats | python3 -m json.tool
```

---

## 5. Continuous Batching

**What it does:** Allows multiple concurrent requests to be batched together instead of processed serially.

### Verify
```bash
# Send two requests simultaneously (use & for background)
time curl -s http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Count to 50."}],"max_tokens":200}' > /dev/null &

time curl -s http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Count to 50."}],"max_tokens":200}' > /dev/null &

wait
# Both should complete in roughly similar time (batched), not one-after-the-other
```

```bash
# Health endpoint shows engine type
curl -s http://localhost:PORT/health | python3 -m json.tool

# "engine_type": "batched"  ← confirms BatchedEngine (continuous batching)
# vs "engine_type": "simple" when CB is off
```

---

## 6. Stop / Cancel Inference

**What it does:** Client abort or server cancel should immediately stop token generation.

### Test from vMLX UI
1. Send a long prompt (e.g., "Write a 2000 word essay on the history of computing")
2. While tokens are streaming, click the Stop button
3. Generation should halt immediately — no more tokens appear
4. The partial response should remain visible

### Test via API cancel endpoint
```bash
# Start a long streaming request, capture the request ID
curl -s -N http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Write a very long essay about space exploration, at least 2000 words."}],"max_tokens":2000,"stream":true}' &
CURL_PID=$!

# Wait a few seconds for tokens to start streaming
sleep 3

# Find the request ID from the stream output (chatcmpl-xxx in the id field)
# Then cancel it:
curl -s -X POST http://localhost:PORT/v1/chat/completions/CHATCMPL_ID/cancel

# The streaming curl should stop receiving data
kill $CURL_PID 2>/dev/null
```

---

## 7. Rate Limiting

**What it does:** Limits requests per minute to prevent overload.

### Configure
- Server Settings → Rate Limit → set to 5 req/min
- Start session

### Verify
```bash
# Rapid-fire 10 requests
for i in $(seq 1 10); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}')
  echo "Request $i: HTTP $CODE"
done

# First 5 should return 200, rest should return 429 (Too Many Requests)
```

---

## 8. Max Tokens (Server-side cap)

**What it does:** Hard cap on output tokens regardless of what client requests.

### Configure
- Server Settings → Performance → Max Tokens → set to 50
- Start session

### Verify
```bash
# Request 500 tokens, but server cap is 50
curl -s http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Count from 1 to 1000, one number per line."}],"max_tokens":500}'

# Response should cap at ~50 tokens even though 500 were requested
# Check usage.completion_tokens in the response
```

---

## 9. Timeout

**What it does:** Kills requests that exceed the timeout duration.

### Configure
- Server Settings → Timeout → set to 10 seconds
- Start session

### Verify
```bash
# Request a very long generation
time curl -s http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Write an extremely detailed 10000 word essay."}],"max_tokens":10000}'

# Should either complete within timeout or return a timeout error after ~10s
```

---

## 10. Reasoning (enable_thinking)

**What it does:** Enables the model's internal reasoning/thinking mode via chat template.

### Test in vMLX UI
1. Open Chat Settings → set Thinking to "On"
2. Send: "What is 15 * 37?"
3. Should see a collapsible "Thinking" section above the answer (if model supports it)
4. Check server logs for `enable_thinking: true` in the request

### Verify via API
```bash
curl -s http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"What is 15 * 37?"}],"max_tokens":200,"enable_thinking":true,"chat_template_kwargs":{"enable_thinking":true},"stream":false}'

# Response should include reasoning content (model-dependent)
```

---

## 11. Tool Calling (Built-in Coding Tools)

**What it does:** Model can call file I/O, search, shell, web tools.

### Test in vMLX UI
1. Chat Settings → Enable Built-in Coding Tools → ON
2. Set Working Directory to a project folder
3. Send: "List the files in the current directory"
4. Model should make a `list_directory` or `run_command` tool call
5. Tool execution status should show in the UI (executing → result)
6. Model should respond with the file listing

### Verify working directory warning
1. Chat Settings → Enable Built-in Coding Tools → ON
2. Do NOT set a working directory
3. Should see orange warning: "Working directory required"

---

## 12. Stop Sequences

**What it does:** Custom strings that stop generation when encountered.

### Verify
```bash
# Tell the model to count, stop at "5"
curl -s http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Count from 1 to 20, one per line."}],"max_tokens":200,"stop":["5"]}'

# Output should stop at or before "5"
```

---

## 13. Temperature / Top-P / Top-K / Min-P / Repeat Penalty

**What it does:** Sampling parameters that control output randomness and diversity.

### Verify temperature
```bash
# Temperature 0 → deterministic (same output each time)
for i in 1 2 3; do
  curl -s http://localhost:PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":10,"temperature":0}' \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
done
# All 3 outputs should be identical

# Temperature 2.0 → very random
for i in 1 2 3; do
  curl -s http://localhost:PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"Tell me a random word."}],"max_tokens":5,"temperature":2.0}' \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
done
# Outputs should vary
```

---

## 14. Wire API: Responses API vs Chat Completions

**What it does:** Two different API formats for the same underlying engine.

### Verify Chat Completions
```bash
curl -s http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello."}],"max_tokens":20}'
# Should return: choices[0].message.content
```

### Verify Responses API
```bash
curl -s http://localhost:PORT/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"default","input":"Say hello.","max_output_tokens":20}'
# Should return: output[0].content[0].text
```

### Verify both in vMLX UI
1. Chat Settings → Wire API → "Chat Completions" → send a message → should work
2. Chat Settings → Wire API → "Responses" → send a message → should work
3. Both should produce equivalent output

---

## 15. Memory Info

**What it does:** Reports Metal GPU memory usage.

### Verify
```bash
curl -s http://localhost:PORT/health | python3 -m json.tool

# Look for:
# "memory": {
#     "active_mb": 1234.5,
#     "peak_mb": 1500.0,
#     "cache_mb": 200.0
# }
```

Send a few prompts and re-check — `active_mb` should increase, `peak_mb` should be >= active.

---

## 16. Model Download Tab

### Test in vMLX UI
1. New Session → Download tab
2. Should show ShieldStack LLC recommended models immediately
3. Search "qwen3 mlx" → should show results from HuggingFace
4. Change download directory via "Change" button
5. Start a download → progress should stream
6. Cancel mid-download → should stop cleanly
7. After successful download → switches to Local tab and model appears in scan

### Verify partial download protection
1. Start a download, cancel midway
2. Go to Local tab → the partial model should NOT appear in the model list
3. Check the download directory — should have `.vmlx-downloading` marker file

---

## 17. Remote Session (OpenAI-compatible endpoint)

### Test in vMLX UI
1. New Session → Remote tab
2. Enter a remote endpoint URL (e.g., `https://api.openai.com/v1` or any compatible endpoint)
3. Enter API key and model name
4. Connect → should show "running" status
5. Send messages → should work
6. Chat Settings should show the remote URL (not localhost)
7. `enable_thinking` should NOT be auto-sent (check server doesn't reject unknown fields)

---

## 18. Feature Compatibility Warnings

### Test in vMLX UI
1. New Session → Server Settings:
   - Enable Disk Cache → Enable Paged Cache → should see warning on both sections
   - Disk Cache checkbox should be disabled when Paged Cache is on
   - Turn OFF continuous batching → turn ON prefix cache → should see info note: "Continuous batching will be auto-enabled at launch"
   - Turn OFF continuous batching → Prefix Cache, Paged Cache, KV Quant, Disk Cache sections should all show orange warning

---

## Quick Smoke Test Script

Run this after starting a session to quickly verify the basics:

```bash
PORT=8000  # ← change to your actual port

echo "=== Health ==="
curl -s http://localhost:$PORT/health | python3 -m json.tool

echo ""
echo "=== Engine Type ==="
curl -s http://localhost:$PORT/health | python3 -c "import sys,json; h=json.load(sys.stdin); print(f'Engine: {h.get(\"engine_type\")}, Model: {h.get(\"model_name\")}')"

echo ""
echo "=== KV Cache Quant ==="
curl -s http://localhost:$PORT/health | python3 -c "import sys,json; h=json.load(sys.stdin); print(h.get('kv_cache_quantization', 'not reported'))"

echo ""
echo "=== Cache Stats ==="
curl -s http://localhost:$PORT/v1/cache/stats | python3 -m json.tool

echo ""
echo "=== Memory ==="
curl -s http://localhost:$PORT/health | python3 -c "import sys,json; h=json.load(sys.stdin); m=h.get('memory',{}); print(f'Active: {m.get(\"active_mb\",\"?\")} MB, Peak: {m.get(\"peak_mb\",\"?\")} MB, Cache: {m.get(\"cache_mb\",\"?\")} MB')"

echo ""
echo "=== Quick inference test ==="
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":5}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'Response: {r[\"choices\"][0][\"message\"][\"content\"]}, Tokens: {r[\"usage\"]}')"

echo ""
echo "=== Responses API test ==="
curl -s http://localhost:$PORT/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"default","input":"Say hi in one word.","max_output_tokens":5}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'Response: {r[\"output\"][0][\"content\"][0][\"text\"]}')"
```

---

## Test Configurations Matrix

| Test | CB | Prefix | Paged | Disk | KV Quant | Expected Engine |
|------|-----|--------|-------|------|----------|----------------|
| All defaults | ON | ON | ON | OFF | none | batched |
| Disk cache | ON | ON | OFF | ON | none | batched |
| KV quant | ON | ON | ON | OFF | 4-bit | batched |
| Simple mode | OFF | OFF | - | - | - | simple |
| Prefix auto-enables CB | OFF | ON | ON | OFF | none | batched (auto) |
| Everything maxed | ON | ON | ON | OFF | 4-bit | batched |
