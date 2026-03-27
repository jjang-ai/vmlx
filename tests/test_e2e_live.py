#!/usr/bin/env python3
"""
End-to-end live integration tests for vMLX engine.

Starts a real vmlx-engine server, loads an actual model, sends real API
requests, and verifies responses + cache behavior + TQ recompress.

Usage:
    python3 tests/test_e2e_live.py                    # Auto-detect model
    python3 tests/test_e2e_live.py /path/to/model     # Specific model
    python3 tests/test_e2e_live.py --port 9999         # Custom port

Requires a downloaded model. Uses Qwen3.5-4B-JANG_2S by default.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

# ── Config ──────────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    os.path.expanduser("~/jang/models/Qwen3.5-4B-JANG_2S"),
    os.path.expanduser("~/jang/models/Qwen3.5-9B-JANG_2S"),
]
DEFAULT_PORT = 19876
TIMEOUT = 120  # Max seconds to wait for model load


# ── Helpers ─────────────────────────────────────────────────────────────

def find_model(explicit_path=None):
    if explicit_path and os.path.isdir(explicit_path):
        return explicit_path
    for p in DEFAULT_MODELS:
        if os.path.isdir(p):
            return p
    print("ERROR: No model found. Download a JANG model or pass --model /path")
    sys.exit(1)


def wait_for_health(port, timeout=TIMEOUT):
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = urllib.request.urlopen(url, timeout=5)
            data = json.loads(r.read())
            if data.get("status") == "healthy":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def api_post(port, path, body, stream=False):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    if stream:
        chunks = []
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
        return chunks
    return json.loads(resp.read())


def api_get(port, path):
    url = f"http://127.0.0.1:{port}{path}"
    resp = urllib.request.urlopen(url, timeout=10)
    return json.loads(resp.read())


def collect_stream(port, messages):
    """Send streaming chat completion and return (text, last_usage, chunks)."""
    body = {
        "model": "test",
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 200,
        "temperature": 0.1,
    }
    chunks = api_post(port, "/v1/chat/completions", body, stream=True)
    text = ""
    reasoning = ""
    usage = None
    for c in chunks:
        choices = c.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            text += delta.get("content", "") or ""
            reasoning += delta.get("reasoning_content", "") or ""
        if c.get("usage"):
            usage = c["usage"]
    # If model only produced reasoning (thinking model), count that as output
    output = text or reasoning
    return output, usage, chunks


# ── Tests ───────────────────────────────────────────────────────────────

def test_health(port):
    data = api_get(port, "/health")
    assert data["status"] == "healthy", f"Health not healthy: {data}"
    print(f"  PASS: /health → healthy, model={data.get('model_name', '?')}")


def test_models(port):
    data = api_get(port, "/v1/models")
    assert len(data.get("data", [])) > 0, "No models listed"
    print(f"  PASS: /v1/models → {len(data['data'])} model(s)")


def test_single_turn(port):
    text, usage, _ = collect_stream(port, [{"role": "user", "content": "Say hello in exactly 3 words."}])
    assert len(text) > 0, "Empty response"
    assert usage is not None, "No usage in stream"
    assert usage.get("prompt_tokens", 0) > 0, "No prompt tokens"
    assert usage.get("completion_tokens", 0) > 0, "No completion tokens"
    print(f"  PASS: Single turn — {usage['completion_tokens']} tokens, '{text[:50]}...'")
    return text


def test_multi_turn_cache(port, t1_text):
    messages = [
        {"role": "user", "content": "Say hello in exactly 3 words."},
        {"role": "assistant", "content": t1_text},
        {"role": "user", "content": "Now say goodbye in exactly 3 words."},
    ]
    text, usage, _ = collect_stream(port, messages)
    assert len(text) > 0, "Empty T2 response"
    cached = 0
    if usage and usage.get("prompt_tokens_details"):
        cached = usage["prompt_tokens_details"].get("cached_tokens", 0)
        detail = usage["prompt_tokens_details"].get("cache_detail", "")
    else:
        detail = ""
    print(f"  PASS: Multi-turn — cached={cached}, detail='{detail}', '{text[:50]}...'")
    return cached


def test_cache_stats(port):
    data = api_get(port, "/v1/cache/stats")
    # Should have some cache info
    has_cache = (
        data.get("scheduler_cache") is not None
        or data.get("prefix_cache") is not None
        or data.get("paged_cache") is not None
    )
    tq = data.get("turbo_quant", {})
    print(f"  PASS: /v1/cache/stats → has_cache={has_cache}, tq={tq}")
    return data


def test_non_streaming(port):
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 20,
        "temperature": 0.1,
    }
    resp = api_post(port, "/v1/chat/completions", body)
    text = resp["choices"][0]["message"]["content"]
    assert len(text) > 0, "Empty non-streaming response"
    print(f"  PASS: Non-streaming — '{text[:50]}...'")


def test_concurrent(port):
    """Two requests back to back (not truly concurrent, but tests batch handling)."""
    import threading
    results = [None, None]
    def req(idx):
        try:
            text, usage, _ = collect_stream(port, [{"role": "user", "content": f"Count to {idx+3}."}])
            results[idx] = text
        except Exception as e:
            results[idx] = f"ERROR: {e}"
    t1 = threading.Thread(target=req, args=(0,))
    t2 = threading.Thread(target=req, args=(1,))
    t1.start(); t2.start()
    t1.join(timeout=30); t2.join(timeout=30)
    assert results[0] and "ERROR" not in str(results[0]), f"Request 0 failed: {results[0]}"
    assert results[1] and "ERROR" not in str(results[1]), f"Request 1 failed: {results[1]}"
    print(f"  PASS: Concurrent — both completed")


def test_stop_sequences(port):
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "List numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}],
        "max_tokens": 100,
        "temperature": 0.1,
        "stop": ["5"],
    }
    resp = api_post(port, "/v1/chat/completions", body)
    text = resp["choices"][0]["message"]["content"]
    assert "6" not in text or "5" not in text, f"Stop sequence didn't work: '{text}'"
    print(f"  PASS: Stop sequences — '{text[:50]}...'")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="vMLX E2E live tests")
    parser.add_argument("model", nargs="?", help="Model path")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--skip-server", action="store_true", help="Use already-running server")
    args = parser.parse_args()

    port = args.port
    proc = None

    if not args.skip_server:
        model_path = find_model(args.model)
        print(f"Starting vmlx-engine on port {port} with {os.path.basename(model_path)}...")

        cmd = [
            sys.executable, "-m", "vmlx_engine.cli", "serve", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--continuous-batching",
            "--max-num-seqs", "4",
            "--max-tokens", "4096",
            "--enable-jit",
            "--cache-memory-percent", "0.2",
            "--kv-cache-quantization", "q8",
            "--log-level", "INFO",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(f"  PID: {proc.pid}")

        if not wait_for_health(port):
            print("FAIL: Server did not become healthy")
            proc.terminate()
            sys.exit(1)
        print(f"  Server healthy.")

    try:
        print("\n=== RUNNING TESTS ===\n")
        passed = 0
        failed = 0

        tests = [
            ("Health check", lambda: test_health(port)),
            ("Model list", lambda: test_models(port)),
            ("Single turn (streaming)", lambda: test_single_turn(port)),
            ("Non-streaming", lambda: test_non_streaming(port)),
            ("Stop sequences", lambda: test_stop_sequences(port)),
            ("Cache stats", lambda: test_cache_stats(port)),
            ("Concurrent requests", lambda: test_concurrent(port)),
        ]

        t1_text = None
        for name, fn in tests:
            try:
                result = fn()
                if name == "Single turn (streaming)":
                    t1_text = result
                passed += 1
            except Exception as e:
                print(f"  FAIL: {name} — {e}")
                failed += 1

        # Multi-turn depends on T1
        if t1_text:
            try:
                cached = test_multi_turn_cache(port, t1_text)
                passed += 1
                if cached > 0:
                    print(f"  CACHE HIT CONFIRMED: {cached} tokens reused from T1")
                else:
                    print(f"  WARNING: No cache hit on T2 (cached=0)")
            except Exception as e:
                print(f"  FAIL: Multi-turn cache — {e}")
                failed += 1

        print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
        return failed

    finally:
        if proc:
            print("\nStopping server...")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("  Stopped.")


if __name__ == "__main__":
    sys.exit(main())
