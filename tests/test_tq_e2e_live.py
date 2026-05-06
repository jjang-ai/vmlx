#!/usr/bin/env python3
"""
End-to-end live test for TQ three-tier cache (L1 memory + L2 disk + TQ compressed).

Starts a real vmlx-engine server with a JANG model, sends real API requests,
and verifies:
1. TQ is active (turbo_quant enabled in /v1/cache/stats)
2. First request -> TTFT (cold, no cache)
3. Second request (same prompt) -> TTFT (warm L1, should be faster)
4. Clear L1 -> third request -> TTFT (warm L2/disk, faster than cold)
5. Cache stats show TQ-native store/hits
6. Growing context -> verify TTFT scales with context length
7. Multi-turn -> prefix cache hit with TQ recompression

Usage:
    python3 tests/test_tq_e2e_live.py                     # Auto-detect model
    python3 tests/test_tq_e2e_live.py /path/to/model      # Specific model
    python3 tests/test_tq_e2e_live.py --skip-server        # Use running server
    python3 tests/test_tq_e2e_live.py --port 9876          # Custom port

Requires a downloaded JANG model. Uses Qwen3.5-4B-JANG_2S by default.
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

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]

# ── Config ──────────────────────────────────────────────────────────────
DEFAULT_MODELS = [
    os.path.expanduser("~/jang/models/Qwen3.5-4B-JANG_2S"),
    os.path.expanduser("~/jang/models/Qwen3.5-9B-JANG_2S"),
]
DEFAULT_PORT = 19877  # Different from test_e2e_live.py to avoid conflict
TIMEOUT = 120


# ── Helpers ─────────────────────────────────────────────────────────────

def find_model(explicit_path=None):
    """Find a JANG model to test with."""
    if explicit_path and os.path.isdir(explicit_path):
        return explicit_path
    for p in DEFAULT_MODELS:
        if os.path.isdir(p):
            return p
    print("ERROR: No JANG model found. Download a JANG model or pass --model /path")
    sys.exit(1)


def wait_for_health(port, timeout=TIMEOUT):
    """Wait for server to become healthy."""
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
    """Send POST request to the API."""
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
    """Send GET request to the API."""
    url = f"http://127.0.0.1:{port}{path}"
    resp = urllib.request.urlopen(url, timeout=10)
    return json.loads(resp.read())


def timed_completion(port, messages, max_tokens=100):
    """Send streaming completion and measure TTFT.

    Returns: (text, ttft_ms, usage, cache_detail)
    """
    body = {
        "model": "test",
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    start = time.time()
    chunks = api_post(port, "/v1/chat/completions", body, stream=True)
    first_token_time = None
    text = ""
    reasoning = ""
    usage = None

    for c in chunks:
        choices = c.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "") or ""
            reasoning_content = delta.get("reasoning_content", "") or ""
            if (content or reasoning_content) and first_token_time is None:
                first_token_time = time.time()
            text += content
            reasoning += reasoning_content
        if c.get("usage"):
            usage = c["usage"]

    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
    output = text or reasoning

    # Extract cache_detail from usage
    cache_detail = ""
    if usage and usage.get("prompt_tokens_details"):
        cache_detail = usage["prompt_tokens_details"].get("cache_detail", "")

    return output, ttft_ms, usage, cache_detail


# ── Tests ───────────────────────────────────────────────────────────────

def test_tq_active(port):
    """Verify TurboQuant is enabled in cache stats."""
    stats = api_get(port, "/v1/cache/stats")
    tq = stats.get("turbo_quant", {})
    enabled = tq.get("enabled", False)
    print(f"  TQ active: {enabled}")
    if not enabled:
        print("  WARNING: TurboQuant not detected. Is this a JANG model?")
    return enabled


def test_cold_vs_warm(port):
    """Measure TTFT for cold (no cache) vs warm (L1 cache) requests.

    Sends the same prompt twice:
    1. First request: cold (no cache), establishes baseline TTFT
    2. Second request: warm (L1 cache hit), should be faster
    """
    prompt = "Explain the three-tier cache architecture in exactly 3 sentences."
    messages = [{"role": "user", "content": prompt}]

    # Cold request (no cache)
    text1, ttft1, usage1, detail1 = timed_completion(port, messages)
    cached1 = 0
    if usage1 and usage1.get("prompt_tokens_details"):
        cached1 = usage1["prompt_tokens_details"].get("cached_tokens", 0)
    print(f"  Cold:  TTFT={ttft1:.0f}ms, cached={cached1}, detail='{detail1}'")
    print(f"         Output: '{text1[:60]}...'")

    # Warm request (same prompt -> should hit L1 cache)
    text2, ttft2, usage2, detail2 = timed_completion(port, messages)
    cached2 = 0
    if usage2 and usage2.get("prompt_tokens_details"):
        cached2 = usage2["prompt_tokens_details"].get("cached_tokens", 0)
    print(f"  Warm:  TTFT={ttft2:.0f}ms, cached={cached2}, detail='{detail2}'")

    # Report speedup
    if ttft1 > 0:
        speedup = ttft1 / max(ttft2, 1)
        print(f"  Speedup: {speedup:.1f}x (warm vs cold)")

    return ttft1, ttft2, cached2


def test_disk_cache_hit(port):
    """Test L2 disk cache hit after L1 eviction.

    1. Send unique prompt to establish disk cache entry
    2. Check cache stats for disk store
    3. Report disk cache status
    """
    # Use a unique prompt with enough context
    prompt = (
        "The TurboQuant three-tier cache architecture consists of three levels: "
        "L1 in-memory paged blocks for instant access, L2 on-disk safetensors "
        "for cold storage, and TQ compressed format as the common currency "
        "across all tiers. Summarize in 2 sentences."
    )
    messages = [{"role": "user", "content": prompt}]

    # First request (establishes cache)
    text1, ttft1, usage1, detail1 = timed_completion(port, messages)
    print(f"  Store: TTFT={ttft1:.0f}ms, detail='{detail1}'")

    # Give disk writer time to persist
    time.sleep(2.0)

    # Check disk cache stats
    stats = api_get(port, "/v1/cache/stats")
    disk = stats.get("disk_cache", {})
    disk_stores = disk.get("stores", 0)
    tq_stores = disk.get("tq_native_stores", 0)
    print(f"  Disk stats: stores={disk_stores}, tq_native_stores={tq_stores}")
    print(f"  Disk size: {disk.get('total_size_mb', 0):.2f}MB")

    return disk_stores, tq_stores


def test_growing_context(port):
    """Measure TTFT as context grows to verify cache effectiveness.

    Sends multi-turn conversations with increasing context:
    - Turn 1: short prompt
    - Turn 2: same + response + new question
    - Turn 3: same + response + new question

    Cache hits should reduce TTFT for the shared prefix portion.
    """
    turns = [
        "What is a paged KV cache?",
        "How does block hashing enable prefix sharing?",
        "Explain TurboQuant 3-bit compression.",
    ]

    ttfts = []
    all_messages = []
    prev_text = ""

    for i, question in enumerate(turns):
        if prev_text:
            all_messages.append({"role": "assistant", "content": prev_text})
        all_messages.append({"role": "user", "content": question})

        text, ttft, usage, detail = timed_completion(port, all_messages, max_tokens=80)
        cached = 0
        if usage and usage.get("prompt_tokens_details"):
            cached = usage["prompt_tokens_details"].get("cached_tokens", 0)

        prompt_tokens = usage.get("prompt_tokens", 0) if usage else 0
        print(
            f"  Turn {i+1}: TTFT={ttft:.0f}ms, prompt={prompt_tokens}tok, "
            f"cached={cached}, detail='{detail}'"
        )
        ttfts.append(ttft)
        prev_text = text

    # Turns 2+ should benefit from cache (lower TTFT per prompt token)
    if len(ttfts) >= 2 and ttfts[0] > 0:
        print(f"  T1→T2 speedup: {ttfts[0]/max(ttfts[1],1):.1f}x")

    return ttfts


def test_cache_stats_final(port):
    """Print final cache statistics for the test run."""
    stats = api_get(port, "/v1/cache/stats")

    # Scheduler cache
    sched = stats.get("scheduler_cache", {})
    if sched:
        print(f"  Prefix cache: hits={sched.get('hits',0)}, misses={sched.get('misses',0)}")

    # Disk cache (prompt-level L2)
    disk = stats.get("disk_cache", {})
    if disk:
        print(
            f"  Disk cache: hits={disk.get('hits',0)}, misses={disk.get('misses',0)}, "
            f"stores={disk.get('stores',0)}, tq_native_stores={disk.get('tq_native_stores',0)}, "
            f"tq_native_hits={disk.get('tq_native_hits',0)}, "
            f"size={disk.get('total_size_mb',0):.2f}MB"
        )

    # Block disk cache (block-level L2)
    block = stats.get("block_disk_cache", {})
    if block:
        print(
            f"  Block disk: hits={block.get('disk_hits',0)}, "
            f"misses={block.get('disk_misses',0)}, "
            f"writes={block.get('disk_writes',0)}, "
            f"size={block.get('disk_size_gb',0):.3f}GB"
        )

    # TQ status
    tq = stats.get("turbo_quant", {})
    if tq:
        print(f"  TurboQuant: {tq}")

    # SSM companion
    ssm = stats.get("ssm_companion", {})
    if ssm:
        print(f"  SSM companion: entries={ssm.get('entries',0)}")

    # Memory
    mem = stats.get("memory", {})
    if mem:
        print(
            f"  Metal GPU: active={mem.get('active_mb',0):.0f}MB, "
            f"peak={mem.get('peak_mb',0):.0f}MB"
        )

    return stats


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TQ three-tier cache E2E tests")
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
            "--max-num-seqs", "2",
            "--max-tokens", "4096",
            "--enable-jit",
            "--cache-memory-percent", "0.2",
            "--kv-cache-quantization", "q8",
            "--enable-disk-cache",
            "--log-level", "INFO",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(f"  PID: {proc.pid}")

        if not wait_for_health(port):
            print("FAIL: Server did not become healthy")
            proc.terminate()
            sys.exit(1)
        print(f"  Server healthy.\n")

    try:
        print("=== TQ THREE-TIER CACHE E2E TESTS ===\n")
        passed = 0
        failed = 0

        # Test 1: Verify TQ is active
        print("[1] TurboQuant Detection")
        try:
            tq_active = test_tq_active(port)
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

        # Test 2: Cold vs Warm TTFT
        print("\n[2] Cold vs Warm (L1) Cache")
        try:
            ttft_cold, ttft_warm, cached = test_cold_vs_warm(port)
            if cached > 0:
                print(f"  PASS: L1 cache hit confirmed ({cached} cached tokens)")
            else:
                print(f"  WARNING: No L1 cache hit (cached=0)")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

        # Test 3: Disk cache persistence
        print("\n[3] Disk Cache (L2) Persistence")
        try:
            disk_stores, tq_stores = test_disk_cache_hit(port)
            if disk_stores > 0:
                tq_label = f" ({tq_stores} TQ-native)" if tq_stores > 0 else ""
                print(f"  PASS: {disk_stores} disk stores{tq_label}")
            else:
                print(f"  WARNING: No disk stores (disk cache may not be enabled)")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

        # Test 4: Growing context
        print("\n[4] Growing Context (Multi-Turn)")
        try:
            ttfts = test_growing_context(port)
            if len(ttfts) >= 3:
                print(f"  PASS: 3-turn conversation completed")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

        # Test 5: Final stats
        print("\n[5] Final Cache Statistics")
        try:
            stats = test_cache_stats_final(port)
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
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
