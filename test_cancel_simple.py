#!/usr/bin/env python3
"""Simple test of request cancellation."""

import asyncio
import json
import aiohttp
import sys

async def test_simple_cancellation():
    """Test basic cancellation functionality."""
    base_url = "http://localhost:8092"

    print("=" * 60)
    print("Simple Cancellation Test")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # Test 1: Normal request completes successfully
        print("\n1. Testing normal completion...")
        request_data = {
            "model": "default",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
            "max_tokens": 10
        }

        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=request_data
        ) as response:
            chunks = 0
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: ') and line != 'data: [DONE]':
                    chunks += 1
            print(f"   ✅ Normal request completed with {chunks} chunks")

        # Test 2: Start a request and immediately try to cancel it
        print("\n2. Testing immediate cancellation...")
        request_data = {
            "model": "default",
            "messages": [{"role": "user", "content": "Write 10 sentences."}],
            "stream": True,
            "max_tokens": 200
        }

        request_id = None

        async def stream_and_cancel():
            """Stream response and try to cancel after getting request ID."""
            nonlocal request_id
            chunks = 0
            cancelled = False

            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line or not line.startswith('data: '):
                        continue

                    if line == 'data: [DONE]':
                        break

                    try:
                        data = json.loads(line[6:])
                        if request_id is None:
                            request_id = data.get('id')
                            print(f"   Got request ID: {request_id}")

                        chunks += 1

                        # After first chunk, try to cancel
                        if chunks == 1 and not cancelled:
                            await asyncio.sleep(0.5)  # Small delay to let generation start
                            print(f"   Attempting to cancel {request_id}...")
                            try:
                                async with session.post(
                                    f"{base_url}/v1/chat/completions/{request_id}/cancel",
                                    timeout=aiohttp.ClientTimeout(total=2)
                                ) as cancel_resp:
                                    cancel_data = await cancel_resp.json()
                                    print(f"   Cancel status: {cancel_resp.status}")
                                    print(f"   Cancel response: {cancel_data}")
                                    cancelled = True
                                    if cancel_data.get('success'):
                                        print("   ✅ Cancel endpoint succeeded")
                                    else:
                                        print("   ⚠️  Cancel returned non-success")
                            except Exception as e:
                                print(f"   Cancel request error: {e}")

                        # Check for abort finish reason
                        finish_reason = data.get('choices', [{}])[0].get('finish_reason')
                        if finish_reason == 'abort':
                            print(f"   ✅ Received 'abort' finish_reason")
                            break
                        elif finish_reason:
                            print(f"   Request finished with: {finish_reason}")

                    except json.JSONDecodeError:
                        pass

            return chunks, cancelled

        try:
            chunks, cancelled = await stream_and_cancel()
            print(f"   Total chunks: {chunks}")
            if cancelled:
                print(f"   ✅ Cancellation test completed")
            else:
                print(f"   ⚠️  Cancellation was not attempted")
        except Exception as e:
            print(f"   Error: {e}")

        # Test 3: Verify server still works
        print("\n3. Testing server continues to work...")
        request_data = {
            "model": "default",
            "messages": [{"role": "user", "content": "Say bye"}],
            "stream": True,
            "max_tokens": 10
        }

        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=request_data
        ) as response:
            chunks = 0
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: ') and line != 'data: [DONE]':
                    chunks += 1
            print(f"   ✅ Server working fine with {chunks} chunks")

    print("\n" + "=" * 60)
    print("All Tests Complete!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(test_simple_cancellation())
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
