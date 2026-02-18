#!/usr/bin/env python3
"""Live test of request cancellation functionality."""

import asyncio
import json
import aiohttp
import sys

async def test_cancellation():
    """Test that cancellation works correctly."""
    base_url = "http://localhost:8092"

    print("=" * 60)
    print("Testing Request Cancellation")
    print("=" * 60)

    # Test 1: Start a streaming request and cancel it
    print("\n1. Starting streaming request...")
    async with aiohttp.ClientSession() as session:
        request_data = {
            "model": "default",
            "messages": [{"role": "user", "content": "Count from 1 to 1000, writing each number on its own line. Start now: 1"}],
            "stream": True,
            "max_tokens": 2000
        }

        request_id = None
        chunks_received = 0

        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            print(f"   Response status: {response.status}")

            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line or not line.startswith('data: '):
                    continue

                if line == 'data: [DONE]':
                    print("   Received [DONE] - request completed before cancel")
                    break

                try:
                    data = json.loads(line[6:])  # Remove 'data: ' prefix

                    # Extract request ID from first chunk
                    if request_id is None:
                        request_id = data.get('id')
                        print(f"   Request ID: {request_id}")

                    # Count chunks
                    chunks_received += 1
                    content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if content:
                        print(f"   Chunk {chunks_received}: {content[:50]}...")

                    # After 3 chunks, cancel the request
                    if chunks_received == 3:
                        print(f"\n2. Cancelling request {request_id}...")
                        async with session.post(
                            f"{base_url}/v1/chat/completions/{request_id}/cancel"
                        ) as cancel_response:
                            cancel_result = await cancel_response.json()
                            print(f"   Cancel response: {cancel_result}")
                            if cancel_result.get('success'):
                                print("   ✅ Cancel succeeded!")
                            else:
                                print("   ❌ Cancel failed!")
                        break

                except json.JSONDecodeError:
                    continue

        print(f"\n   Total chunks received before cancel: {chunks_received}")

        # Give server a moment to process the cancellation
        await asyncio.sleep(1)

        # Test 2: Verify server continues working with a new request
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
        asyncio.run(test_cancellation())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
