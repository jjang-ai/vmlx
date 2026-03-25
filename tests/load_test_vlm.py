#!/usr/bin/env python3
"""
Comprehensive VLM Load Test for vMLX
=====================================
Tests all cache/batching features with a VLM model:
- Continuous batching
- Paged KV cache
- KV cache quantization (q8)
- Prefix cache (block-level deduplication)
- Multi-turn conversations
- Concurrent requests
- Image handling

Monitors and validates:
- Cache hit/miss rates
- Token throughput (TPS)
- Memory usage via cache stats
- Prefix cache deduplication
- KV quant memory savings
"""

import asyncio
import base64
import json
import os
import sys
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

import aiohttp

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("VLLM_API_KEY", "")

# Test image: a simple 64x64 red square as base64
def make_test_image(color=(255, 0, 0), size=(64, 64)):
    """Create a minimal BMP test image (no PIL needed)."""
    w, h = size
    r, g, b = color
    # BMP header
    row_size = (w * 3 + 3) & ~3  # rows padded to 4 bytes
    pixel_data_size = row_size * h
    file_size = 54 + pixel_data_size
    header = bytearray(54)
    # BM magic
    header[0:2] = b'BM'
    header[2:6] = file_size.to_bytes(4, 'little')
    header[10:14] = (54).to_bytes(4, 'little')  # pixel data offset
    header[14:18] = (40).to_bytes(4, 'little')  # DIB header size
    header[18:22] = w.to_bytes(4, 'little')
    header[22:26] = h.to_bytes(4, 'little')
    header[26:28] = (1).to_bytes(2, 'little')  # planes
    header[28:30] = (24).to_bytes(2, 'little')  # bits per pixel
    header[34:38] = pixel_data_size.to_bytes(4, 'little')
    # Pixel data (BGR in BMP)
    pixels = bytearray()
    for _ in range(h):
        row = bytearray()
        for _ in range(w):
            row += bytes([b, g, r])
        row += b'\x00' * (row_size - w * 3)
        pixels += row
    return bytes(header) + bytes(pixels)


def image_to_data_url(img_bytes: bytes, mime="image/bmp") -> str:
    return f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"


RED_IMAGE = image_to_data_url(make_test_image((255, 0, 0)))
BLUE_IMAGE = image_to_data_url(make_test_image((0, 0, 255)))
GREEN_IMAGE = image_to_data_url(make_test_image((0, 255, 0)))

SYSTEM_PROMPT = (
    "You are a helpful vision assistant. Describe images accurately and concisely. "
    "Keep responses under 50 words."
)


def headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


async def get_cache_stats(session: aiohttp.ClientSession) -> Dict[str, Any]:
    """Fetch cache stats from the server."""
    async with session.get(f"{BASE_URL}/v1/cache/stats", headers=headers()) as resp:
        if resp.status == 200:
            return await resp.json()
        return {"error": f"HTTP {resp.status}"}


async def get_cache_entries(session: aiohttp.ClientSession) -> Dict[str, Any]:
    """Fetch cache entries from the server."""
    async with session.get(f"{BASE_URL}/v1/cache/entries", headers=headers()) as resp:
        if resp.status == 200:
            return await resp.json()
        return {"error": f"HTTP {resp.status}"}


async def get_health(session: aiohttp.ClientSession) -> Dict[str, Any]:
    """Check server health."""
    async with session.get(f"{BASE_URL}/health", headers=headers()) as resp:
        if resp.status == 200:
            return await resp.json()
        return {"error": f"HTTP {resp.status}"}


async def chat_completion(
    session: aiohttp.ClientSession,
    messages: List[Dict],
    max_tokens: int = 100,
    temperature: float = 0.1,
    stream: bool = False,
) -> Dict[str, Any]:
    """Send a chat completion request."""
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "stream_options": {"include_usage": True} if stream else None,
    }
    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}

    start = time.time()
    if stream:
        content = ""
        usage = {}
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers=headers(),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"error": f"HTTP {resp.status}: {text}"}
            async for line in resp.content:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                content += delta["content"]
                        if "usage" in chunk and chunk["usage"]:
                            usage = chunk["usage"]
                    except json.JSONDecodeError:
                        pass
        elapsed = time.time() - start
        return {
            "content": content,
            "usage": usage,
            "elapsed": elapsed,
            "tps": usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0,
        }
    else:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers=headers(),
        ) as resp:
            elapsed = time.time() - start
            if resp.status != 200:
                text = await resp.text()
                return {"error": f"HTTP {resp.status}: {text}"}
            result = await resp.json()
            usage = result.get("usage", {})
            content = result["choices"][0]["message"]["content"]
            return {
                "content": content,
                "usage": usage,
                "elapsed": elapsed,
                "tps": usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0,
            }


def make_vision_message(text: str, image_url: str) -> List[Dict]:
    """Create a message with an image."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text},
            ],
        },
    ]


def make_text_followup(history: List[Dict], text: str) -> List[Dict]:
    """Add a text-only follow-up to conversation history."""
    return history + [{"role": "user", "content": text}]


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_stats(stats: Dict[str, Any], label: str = ""):
    if label:
        print(f"\n--- {label} ---")
    print(json.dumps(stats, indent=2, default=str))


async def run_tests():
    timeout = aiohttp.ClientTimeout(total=600, sock_read=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # ==========================================
        # TEST 0: Health check
        # ==========================================
        print_section("TEST 0: Health Check")
        health = await get_health(session)
        print(f"Health: {json.dumps(health, indent=2)}")
        if "error" in health:
            print("ERROR: Server not responding. Is it running?")
            sys.exit(1)

        # Get baseline cache stats
        stats_baseline = await get_cache_stats(session)
        print_stats(stats_baseline, "Baseline Cache Stats")

        # ==========================================
        # TEST 1: Single VLM request (warm-up)
        # ==========================================
        print_section("TEST 1: Single VLM Request (Warm-up + Baseline)")
        msgs = make_vision_message("What color is this image?", RED_IMAGE)
        result = await chat_completion(session, msgs, max_tokens=50, stream=True)
        print(f"Response: {result.get('content', 'ERROR')[:200]}")
        print(f"Usage: {result.get('usage', {})}")
        print(f"Elapsed: {result.get('elapsed', 0):.2f}s, TPS: {result.get('tps', 0):.1f}")
        if "error" in result:
            print(f"ERROR: {result['error']}")
            sys.exit(1)

        stats_after_1 = await get_cache_stats(session)
        print_stats(stats_after_1, "Cache Stats After Test 1")

        # ==========================================
        # TEST 2: Same prompt = prefix cache HIT
        # ==========================================
        print_section("TEST 2: Identical Prompt (Prefix Cache Hit Test)")
        result2 = await chat_completion(session, msgs, max_tokens=50, stream=True)
        print(f"Response: {result2.get('content', 'ERROR')[:200]}")
        print(f"Usage: {result2.get('usage', {})}")
        print(f"Elapsed: {result2.get('elapsed', 0):.2f}s, TPS: {result2.get('tps', 0):.1f}")

        stats_after_2 = await get_cache_stats(session)
        print_stats(stats_after_2, "Cache Stats After Test 2 (should show hits)")

        # Check for cache hit
        sched_cache = stats_after_2.get("scheduler_cache", {})
        hits = sched_cache.get("hits", 0)
        print(f"\n>>> CACHE HITS: {hits} (expected >= 1 for prefix cache)")

        # ==========================================
        # TEST 3: Multi-turn conversation
        # ==========================================
        print_section("TEST 3: Multi-Turn VLM Conversation")
        # Turn 1: image + question
        turn1_msgs = make_vision_message("What color is this square?", RED_IMAGE)
        r1 = await chat_completion(session, turn1_msgs, max_tokens=50, stream=True)
        print(f"Turn 1: {r1.get('content', 'ERROR')[:200]}")
        print(f"  Usage: {r1.get('usage', {})}, TPS: {r1.get('tps', 0):.1f}")

        # Turn 2: text follow-up (should reuse prefix cache for system+image)
        turn2_msgs = turn1_msgs + [
            {"role": "assistant", "content": r1.get("content", "")},
            {"role": "user", "content": "What would it look like if it were blue instead?"},
        ]
        try:
            r2 = await chat_completion(session, turn2_msgs, max_tokens=50, stream=True)
            if "error" in r2:
                print(f"Turn 2: ERROR - {r2['error'][:200]}")
                print("  (Multi-turn VLM may hit mlx-vlm processor issue, continuing)")
            else:
                print(f"Turn 2: {r2.get('content', 'ERROR')[:200]}")
                print(f"  Usage: {r2.get('usage', {})}, TPS: {r2.get('tps', 0):.1f}")
        except Exception as e:
            print(f"Turn 2: EXCEPTION - {e}")
            r2 = {"content": ""}

        # Turn 3: text follow-up
        turn3_msgs = turn2_msgs + [
            {"role": "assistant", "content": r2.get("content", "")},
            {"role": "user", "content": "Summarize what we discussed about the colors."},
        ]
        try:
            r3 = await chat_completion(session, turn3_msgs, max_tokens=80, stream=True)
            if "error" in r3:
                print(f"Turn 3: ERROR - {r3['error'][:200]}")
            else:
                print(f"Turn 3: {r3.get('content', 'ERROR')[:200]}")
                print(f"  Usage: {r3.get('usage', {})}, TPS: {r3.get('tps', 0):.1f}")
        except Exception as e:
            print(f"Turn 3: EXCEPTION - {e}")

        stats_after_3 = await get_cache_stats(session)
        print_stats(stats_after_3, "Cache Stats After Multi-Turn")

        # ==========================================
        # TEST 4: Concurrent requests (continuous batching)
        # ==========================================
        print_section("TEST 4: Concurrent VLM Requests (Continuous Batching)")
        concurrent_msgs = [
            make_vision_message("Describe this red image briefly.", RED_IMAGE),
            make_vision_message("Describe this blue image briefly.", BLUE_IMAGE),
            make_vision_message("Describe this green image briefly.", GREEN_IMAGE),
        ]
        start = time.time()
        tasks = [
            chat_completion(session, m, max_tokens=50, stream=True)
            for m in concurrent_msgs
        ]
        results = await asyncio.gather(*tasks)
        total_elapsed = time.time() - start
        total_tokens = 0
        for i, r in enumerate(results):
            tokens = r.get("usage", {}).get("completion_tokens", 0)
            total_tokens += tokens
            print(f"  Request {i+1}: {r.get('content', 'ERROR')[:100]}")
            print(f"    Tokens: {tokens}, Individual TPS: {r.get('tps', 0):.1f}")
            if "error" in r:
                print(f"    ERROR: {r['error']}")
        print(f"\n>>> Total wall-clock: {total_elapsed:.2f}s for {len(results)} requests")
        print(f">>> Total tokens: {total_tokens}")
        print(f">>> Aggregate TPS: {total_tokens/total_elapsed:.1f}")

        stats_after_4 = await get_cache_stats(session)
        print_stats(stats_after_4, "Cache Stats After Concurrent Test")

        # ==========================================
        # TEST 5: Shared system prompt (prefix dedup)
        # ==========================================
        print_section("TEST 5: Shared System Prompt Deduplication")
        # All 3 requests share the same system prompt — cache blocks should be shared
        shared_msgs = [
            make_vision_message("Is this red?", RED_IMAGE),
            make_vision_message("Is this blue?", BLUE_IMAGE),
            make_vision_message("Is this green?", GREEN_IMAGE),
        ]
        # Send sequentially to ensure cache fills from first, hits on subsequent
        for i, m in enumerate(shared_msgs):
            r = await chat_completion(session, m, max_tokens=30, stream=True)
            print(f"  Request {i+1}: {r.get('content', 'ERROR')[:100]}")
            print(f"    Usage: {r.get('usage', {})}, TPS: {r.get('tps', 0):.1f}")

        stats_after_5 = await get_cache_stats(session)
        print_stats(stats_after_5, "Cache Stats After Dedup Test")

        # Check tokens_saved metric
        sched_cache = stats_after_5.get("scheduler_cache", {})
        tokens_saved = sched_cache.get("tokens_saved", 0)
        total_hits = sched_cache.get("hits", 0)
        print(f"\n>>> TOTAL CACHE HITS: {total_hits}")
        print(f">>> TOKENS SAVED: {tokens_saved}")

        # ==========================================
        # TEST 6: Non-streaming (completeness)
        # ==========================================
        print_section("TEST 6: Non-Streaming VLM Request")
        r_ns = await chat_completion(
            session,
            make_vision_message("What is shown in this image?", RED_IMAGE),
            max_tokens=50,
            stream=False,
        )
        print(f"Response: {r_ns.get('content', 'ERROR')[:200]}")
        print(f"Usage: {r_ns.get('usage', {})}")
        print(f"Elapsed: {r_ns.get('elapsed', 0):.2f}s")

        # ==========================================
        # FINAL: Summary and Validation
        # ==========================================
        print_section("FINAL SUMMARY")
        final_stats = await get_cache_stats(session)
        print_stats(final_stats, "Final Cache Stats")

        # Validate KV quantization is active
        kv_quant = final_stats.get("kv_cache_quantization", {})
        if kv_quant:
            print(f"\n[OK] KV Cache Quantization: {kv_quant.get('bits')}-bit, group_size={kv_quant.get('group_size')}")
        else:
            print("\n[??] KV Cache Quantization: Not reported in stats")

        # Validate scheduler stats
        sched = final_stats.get("scheduler_stats", {})
        print(f"\n[STATS] Requests processed: {sched.get('num_requests_processed', '?')}")
        print(f"[STATS] Total prompt tokens: {sched.get('total_prompt_tokens', '?')}")
        print(f"[STATS] Total completion tokens: {sched.get('total_completion_tokens', '?')}")

        # Validate cache metrics
        cache = final_stats.get("scheduler_cache", {})
        if cache:
            h = cache.get("hits", 0)
            m = cache.get("misses", 0)
            rate = cache.get("hit_rate", 0)
            saved = cache.get("tokens_saved", 0)
            print(f"\n[CACHE] Hits: {h}, Misses: {m}, Hit Rate: {rate:.2%}")
            print(f"[CACHE] Tokens Saved by Prefix Cache: {saved}")
            # Memory savings calculation
            # 80KB per token at fp16, 40KB at q8
            if kv_quant and kv_quant.get("bits") == 8:
                bytes_per_token = 40960  # q8
                quant_label = "q8"
            elif kv_quant and kv_quant.get("bits") == 4:
                bytes_per_token = 20480  # q4
                quant_label = "q4"
            else:
                bytes_per_token = 81920  # fp16
                quant_label = "fp16"

            if saved > 0:
                mem_saved_mb = saved * bytes_per_token / 1024 / 1024
                print(f"[CACHE] Estimated Memory Saved: {mem_saved_mb:.1f} MB ({quant_label})")

            # Paged cache specific stats
            for key in ["total_blocks", "used_blocks", "free_blocks", "evictions",
                        "memory_used_mb", "memory_capacity_mb"]:
                if key in cache:
                    print(f"[PAGED] {key}: {cache[key]}")

        # Vision cache stats
        vcache = final_stats.get("vision_embedding_cache", {}) or final_stats.get(
            "scheduler_stats", {}
        ).get("vision_embedding_cache", {})
        if vcache:
            print(f"\n[VISION] Vision Cache Stats: {json.dumps(vcache, indent=2)}")

        print(f"\n{'='*70}")
        print("  LOAD TEST COMPLETE")
        print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(run_tests())
