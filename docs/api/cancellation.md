# Request Cancellation API

Stop ongoing inference requests immediately to save GPU compute and improve user experience.

## Overview

vMLX provides two ways to cancel ongoing requests:

1. **Explicit Cancellation**: Call the cancel endpoint with request ID
2. **Auto-Detection**: Close the stream connection (automatically aborts)

## Quick Start

### Get Request ID

The request ID is returned in the first SSE chunk:

```bash
curl -N http://localhost:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Tell me a long story"}],
    "stream": true
  }'

# First chunk:
# data: {"id":"chatcmpl-abc123",...}
```

### Cancel via API

```bash
curl -X POST http://localhost:8092/v1/chat/completions/chatcmpl-abc123/cancel \
  -H "Authorization: Bearer your-api-key"

# Response:
# {"success": true, "message": "Request chatcmpl-abc123 cancelled"}
```

### Cancel via Stream Close

```javascript
const reader = response.body.getReader();
// ... read chunks ...

// Cancel by closing stream
await reader.cancel();  // vMLX automatically aborts request
```

## Endpoints

### POST /v1/chat/completions/{request_id}/cancel

Cancel an ongoing chat completion request.

**Parameters:**
- `request_id` (path, required): Request ID from response (e.g., chatcmpl-abc123)

**Headers:**
- `Authorization: Bearer <api-key>` (if API key enabled)

**Response (Success):**
```json
{
  "success": true,
  "message": "Request chatcmpl-abc123 cancelled"
}
```

**Response (Not Found):**
```json
{
  "detail": "Request chatcmpl-abc123 not found or already finished"
}
```
Status: `404 Not Found`

### POST /v1/completions/{request_id}/cancel

Same as chat completions, but for `/v1/completions` endpoint.

### POST /v1/responses/{request_id}/cancel

Cancel an ongoing Responses API request. Used by agentic tools (Codex CLI, OpenCode, etc.).

**Parameters:**
- `request_id` (path, required): Response ID from SSE events (e.g., resp_abc123)

**Response (Success):**
```json
{
  "success": true,
  "message": "Request resp_abc123 cancelled"
}
```

## Frontend Integration

### JavaScript / Fetch API

```javascript
let currentRequestId = null;
let activeReader = null;

async function sendMessage(message) {
    const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: 'default',
            messages: [{ role: 'user', content: message }],
            stream: true
        })
    });

    const reader = response.body.getReader();
    activeReader = reader;
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\\n').filter(line => line.startsWith('data: '));

        for (const line of lines) {
            if (line.includes('[DONE]')) break;
            const data = JSON.parse(line.slice(6));

            // Store request ID from first chunk
            if (!currentRequestId && data.id) {
                currentRequestId = data.id;
            }

            // Display content
            const content = data.choices[0]?.delta?.content || '';
            appendToChat(content);
        }
    }
}

async function cancelGeneration() {
    // Option 1: Close stream (auto-aborts on server)
    if (activeReader) {
        await activeReader.cancel();
        activeReader = null;
    }

    // Option 2: Explicit cancel via API (optional)
    if (currentRequestId) {
        await fetch(\`/v1/chat/completions/\${currentRequestId}/cancel\`, {
            method: 'POST'
        });
    }

    currentRequestId = null;
}
```

### Python / OpenAI SDK

```python
from openai import OpenAI
import requests

client = OpenAI(base_url="http://localhost:8092/v1", api_key="your-api-key")

# Start streaming
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a long story"}],
    stream=True
)

# Get request ID
request_id = None
for chunk in stream:
    if not request_id:
        request_id = chunk.id
        print(f"Request ID: {request_id}")

    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)

    # Cancel mid-stream
    if should_cancel():
        requests.post(
            f"http://localhost:8092/v1/chat/completions/{request_id}/cancel",
            headers={"Authorization": "Bearer your-api-key"}
        )
        break
```

### exploit.bot (No Changes Needed)

exploit.bot's existing cancel button works automatically:

```javascript
// Already implemented in exploit.bot
async function cancelMessage() {
    if (activeStreamReader) {
        await activeStreamReader.cancel();  // Auto-aborts on vMLX
        activeStreamReader = null;
    }
}
```

## Behavior

### What Happens When Cancelled

1. **Token generation stops immediately** - GPU inference halts
2. **Partial response preserved** - Tokens generated so far are kept
3. **Finish reason set to "abort"** - In final response chunk
4. **Resources cleaned up** - Memory and cache entries released

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| Request not found | 404 error |
| Already finished | 404 error |
| Double cancellation | First succeeds, second returns 404 |
| Cancel during prefill | Aborts immediately, may return empty |
| Cancel during generation | Stops at current token, returns partial |

## Performance Impact

- **GPU compute**: Stops immediately, no wasted cycles
- **Memory**: Request resources freed
- **Cache**: Partial KV cache may be preserved for prefix reuse
- **Latency**: Cancel response typically < 10ms

## Troubleshooting

### "Request not found" error

**Cause**: Request ID doesn't exist or already finished

**Solution**: Ensure you're using the correct ID from the response

### Cancel doesn't stop generation

**Cause**: May be using SimpleEngine (synchronous mode)

**Solution**: Use `--continuous-batching` flag:
```bash
vmlx-engine serve model --continuous-batching
```

### No request ID in response

**Cause**: Not streaming or old vMLX version

**Solution**:
- Use `stream: true` in request
- Update vMLX to latest version

## Examples

See `tests/test_request_cancellation.py` for comprehensive examples.
