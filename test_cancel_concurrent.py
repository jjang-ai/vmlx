#!/usr/bin/env python3
"""Test request cancellation with concurrent cancel during generation."""

import asyncio
import json
import aiohttp
import sys

async def test_concurrent_cancellation():
    """Test cancellation while generation is actively happening."""
    base_url = "http://localhost:8092"

    print("=" * 60)
    print("Testing Concurrent Request Cancellation")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # Test 1: Cancel after a short delay
        print("\n1. Starting long-running streaming request...")
        request_data = {
            "model": "default",
            "messages": [{"role": "user", "content": "Count from 1 to 10000, writing each number on its own line. Start now: 1"}],
            "stream": True,
            "max_tokens": 5000
        }

        request_id = None
        chunks_received = 0
        cancel_sent = False

        async def send_cancel_after_delay():
            """Send cancel request after 2 seconds."""
            nonlocal cancel_sent
            await asyncio.sleep(2)  # Wait 2 seconds for generation to start
            if request_id and not cancel_sent:
                print(f"\n2. Sending cancel request for {request_id} (after 2 seconds)...")
                try:
                    async with session.post(
                        f"{base_url}/v1/chat/completions/{request_id}/cancel",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as cancel_response:
                        cancel_result = await cancel_response.json()
                        print(f"   Cancel response status: {cancel_response.status}")
                        print(f"   Cancel response: {cancel_result}")
                        cancel_sent = True
                        if cancel_result.get('success'):
                            print("   ✅ Cancel succeeded!")
                        else:
                            print("   ⚠️  Cancel returned non-success (request may have finished)")
                except Exception as e:
                    print(f"   ❌ Cancel failed with error: {e}")

        # Start the cancel task
        cancel_task = asyncio.create_task(send_cancel_after_delay())

        # Start streaming request
        try:
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                print(f"   Response status: {response.status}")

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line or not line.startswith('data: '):
                        continue

                    if line == 'data: [DONE]':
                        print("   Received [DONE]")
                        break

                    try:
                        data = json.loads(line[6:])

                        # Extract request ID from first chunk
                        if request_id is None:
                            request_id = data.get('id')
                            print(f"   Request ID: {request_id}")

                        # Count chunks and show first few
                        chunks_received += 1
                        if chunks_received <= 5:
                            content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            if content:
                                print(f"   Chunk {chunks_received}: {content[:50]}...")

                        # Check finish reason
                        finish_reason = data.get('choices', [{}])[0].get('finish_reason')
                        if finish_reason == 'abort':
                            print(f"   ✅ Received abort finish_reason in chunk {chunks_received}")
                            break
                        elif finish_reason:
                            print(f"   Finished with reason: {finish_reason}")

                    except json.JSONDecodeError:
                        continue

        except asyncio.TimeoutError:
            print("   ⚠️  Request timed out (this might be expected if cancelled)")

        # Wait for cancel task to complete
        await cancel_task

        print(f"\n   Total chunks received: {chunks_received}")
        if cancel_sent and chunks_received < 100:  # Expecting thousands of chunks for 1-10000
            print(f"   ✅ Generation stopped early (received {chunks_received} chunks, expected ~1000+)")
        elif not cancel_sent:
            print("   ⚠️  Cancel wasn't sent (request may have finished too quickly)")
        else:
            print(f"   ⚠️  Received many chunks ({chunks_received}), cancel may not have stopped generation")

        # Give server a moment
        await asyncio.sleep(1)

        # Test 2: Verify server continues working
        print("\n3. Testing that server continues working...")
        request_data2 = {
            "model": "default",
            "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}],
            "stream": True,
            "max_tokens": 20
        }

        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=request_data2,
            headers={"Content-Type": "application/json"}
        ) as response:
            print(f"   Response status: {response.status}")

            full_response = ""
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line or not line.startswith('data: '):
                    continue

                if line == 'data: [DONE]':
                    break

                try:
                    data = json.loads(line[6:])
                    content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if content:
                        full_response += content
                except json.JSONDecodeError:
                    continue

            print(f"   Response: {full_response}")
            if full_response:
                print("   ✅ Server continues working correctly!")
            else:
                print("   ❌ Server failed to respond!")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(test_concurrent_cancellation())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
