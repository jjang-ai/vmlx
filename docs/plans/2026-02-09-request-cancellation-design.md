# Request Cancellation Design

**Date**: 2026-02-09
**Status**: Approved for Implementation
**Compatibility**: exploit.bot, vMLX panel

---

## Overview

Add request cancellation support to vMLX that allows frontend applications to stop ongoing inference immediately, saving compute resources and improving user experience.

**Key Features:**
- OpenAI-compatible cancellation endpoint
- Unified request ID (response_id = request_id)
- Auto-detection of closed client connections
- Full compatibility with exploit.bot's cancel button
- Works seamlessly with vMLX panel

---

## Use Case

**Problem**: Users click "stop" button in frontend, but inference continues running on GPU, wasting compute.

**Solution**:
1. User clicks stop button in frontend (exploit.bot, vMLX panel)
2. Frontend calls `reader.cancel()` to close stream OR calls cancel API endpoint
3. Backend detects cancellation and calls `engine.abort_request(request_id)`
4. GPU inference stops immediately, partial response is preserved

---

## Architecture

### Current State

```
Frontend                    Backend (vMLX)
   │                             │
   ├─ POST /v1/chat/completions ─┤
   │                             │
   │  ◄── SSE stream (tokens) ── │
   │                             │
   └─ [No way to stop] ──────────┘
```

### New State

```
Frontend                         Backend (vMLX)
   │                                  │
   ├─ POST /v1/chat/completions ─────┤ (request_id: chatcmpl-abc123)
   │                                  │
   │  ◄── SSE stream (tokens) ────── │
   │                                  │
   ├─ reader.cancel() ───────────────┤ (closes connection)
   │    OR                            │ (auto-detects, aborts request)
   │                                  │
   ├─ POST .../abc123/cancel ────────┤ (explicit cancel)
   │                                  │
   └─ ✓ Inference stopped ◄─────────┘
```

---

## Design Details

### 1. Unified Request ID

**Problem**: Currently `response_id` (chatcmpl-abc123) is separate from internal `request_id` (UUID).

**Solution**: Use `response_id` as the `request_id` throughout the system.

**Changes**:
- `server.py` line ~1211: Pass `request_id=response_id` to engine
- `engine/base.py`: Add `request_id` parameter to `stream_chat()` signature
- `engine/simple.py` & `engine/batched.py`: Pass `request_id` to `engine.add_request()`

**Benefits**:
- Client knows the request ID immediately (from first SSE chunk)
- Can cancel using the same ID they received
- OpenAI-compatible pattern

---

### 2. Cancel Endpoint

**Endpoint**: `POST /v1/chat/completions/{request_id}/cancel`

**Request**:
```bash
POST /v1/chat/completions/chatcmpl-abc123/cancel
Authorization: Bearer <api-key>
```

**Response (Success)**:
```json
{
  "success": true,
  "message": "Request chatcmpl-abc123 cancelled"
}
```

**Response (Not Found)**:
```json
{
  "detail": "Request chatcmpl-abc123 not found or already finished"
}
```
Status: `404 Not Found`

**Implementation**:
```python
@app.post(
    "/v1/chat/completions/{request_id}/cancel",
    dependencies=[Depends(verify_api_key)]
)
async def cancel_chat_completion(request_id: str):
    """
    Cancel an ongoing chat completion request.

    Stops token generation immediately and marks the request as aborted.
    Any tokens generated so far are preserved in the partial response.

    Args:
        request_id: The request ID (e.g., chatcmpl-abc123)

    Returns:
        Success message or 404 if request not found
    """
    engine = get_engine()
    success = await engine.abort_request(request_id)

    if success:
        return {"success": True, "message": f"Request {request_id} cancelled"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Request {request_id} not found or already finished"
        )
```

**Also add for `/v1/completions`**:
```python
@app.post(
    "/v1/completions/{request_id}/cancel",
    dependencies=[Depends(verify_api_key)]
)
async def cancel_completion(request_id: str):
    # Same implementation as chat completions
```

---

### 3. Auto-Detection of Closed Connections

**Feature**: Automatically abort request when client closes stream connection.

**How it Works**:
1. Client calls `reader.cancel()` to close stream
2. FastAPI detects closed connection via `request.is_disconnected()`
3. Automatically calls `engine.abort_request(request_id)`
4. No explicit cancel API call needed

**Implementation**:

```python
async def stream_chat_completion(
    engine: BaseEngine,
    messages: list,
    request: ChatCompletionRequest,
    fastapi_request: Request = None,  # Add Request dependency
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion with auto-detection of closed connections."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    kwargs['request_id'] = response_id

    try:
        async for output in engine.stream_chat(messages=messages, **kwargs):
            # Check if client disconnected (reader.cancel() was called)
            if fastapi_request and await fastapi_request.is_disconnected():
                logger.info(f"Client disconnected, aborting {response_id}")
                await engine.abort_request(response_id)
                break

            # Build and yield chunk
            # ... normal streaming code ...
            yield f"data: {chunk.model_dump_json()}\n\n"

    except Exception as e:
        logger.error(f"Stream error for {response_id}: {e}")
        await engine.abort_request(response_id)
        raise
```

**Update endpoint to pass Request**:
```python
@app.post("/v1/chat/completions", ...)
async def create_chat_completion(
    request: ChatCompletionRequest,
    fastapi_request: Request,  # Add this
):
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(
                engine,
                messages,
                request,
                fastapi_request=fastapi_request,  # Pass it through
                **chat_kwargs
            ),
            media_type="text/event-stream",
        )
```

---

### 4. Engine Changes

**File**: `vllm_mlx/engine/base.py`

```python
async def stream_chat(
    self,
    messages: list,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop: Optional[List[str]] = None,
    request_id: Optional[str] = None,  # ADD THIS PARAMETER
) -> AsyncIterator[GenerationOutput]:
    """
    Stream chat completion responses.

    Args:
        messages: Chat messages in OpenAI format
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        stop: Stop sequences
        request_id: Optional custom request ID (for cancellation support)
    """
```

**File**: `vllm_mlx/engine/batched.py` (line ~510, ~587)

```python
# In generate() method
request_id = await self._engine.add_request(
    prompt=prompt_token_ids,
    sampling_params=sampling_params,
    request_id=request_id,  # Pass custom ID if provided
)

# In stream_chat() method
request_id = await self._engine.add_request(
    prompt=final_prompt_ids,
    sampling_params=sampling_params,
    request_id=request_id,  # Pass custom ID if provided
    images=images,
    videos=videos,
)
```

**File**: `vllm_mlx/engine/simple.py` (similar changes)

**Note**: `engine_core.py` already supports `request_id` parameter (line 191), no changes needed!

---

## Error Handling

### Edge Cases

1. **Request Not Found**: Return 404 if request_id doesn't exist or already finished
2. **Double Cancellation**: Safe to call multiple times (idempotent), second call returns 404
3. **Cancel During Prefill**: Aborts immediately, returns empty or partial output
4. **Cancel During Generation**: Stops at current token, returns partial response with `finish_reason: "abort"`
5. **Connection Already Closed**: No error, just log and cleanup

### Error Responses

```python
# Request not found
{
  "detail": "Request chatcmpl-abc123 not found or already finished"
}

# Engine not initialized
{
  "detail": "Service unavailable"
}
```

---

## Frontend Integration

### exploit.bot Integration

**Current Code** (already works, no changes needed):

```javascript
// Store reader when streaming starts
const reader = response.body.getReader();
activeStreamReader = reader;

// Cancel button handler
async function cancelMessage() {
    if (activeStreamReader) {
        await activeStreamReader.cancel();  // Closes connection
        activeStreamReader = null;
    }
}
```

**How it Works with vMLX**:
1. exploit.bot calls `POST /v1/chat/completions` → gets `chatcmpl-abc123` in first chunk
2. User clicks cancel → `reader.cancel()` closes stream
3. vMLX detects closed connection → automatically aborts request
4. GPU inference stops immediately

**Optional**: exploit.bot can also call cancel endpoint explicitly:
```javascript
async function cancelMessage() {
    if (currentRequestId) {
        // Explicit cancel via API
        await fetch(`/api/chat/${chatId}/cancel`, { method: 'POST' });
    }
    if (activeStreamReader) {
        await activeStreamReader.cancel();
    }
}
```

---

### vMLX Panel Integration

**Add Cancel Button to Panel**:

```javascript
// In vMLX panel streaming code
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
        const lines = chunk.split('\n').filter(line => line.startsWith('data: '));

        for (const line of lines) {
            if (line.includes('[DONE]')) break;
            const data = JSON.parse(line.slice(6));

            // Store request ID from first chunk
            if (!currentRequestId && data.id) {
                currentRequestId = data.id;
                console.log('Request ID:', currentRequestId);
            }

            // Display content
            const content = data.choices[0]?.delta?.content || '';
            appendToChat(content);
        }
    }
}

async function cancelGeneration() {
    if (activeReader) {
        // Close stream (triggers auto-abort on server)
        await activeReader.cancel();
        activeReader = null;
    }

    // Optional: explicit cancel via API
    if (currentRequestId) {
        await fetch(`/v1/chat/completions/${currentRequestId}/cancel`, {
            method: 'POST',
            headers: { 'Authorization': 'Bearer <api-key>' }
        });
    }

    currentRequestId = null;
}

// Add cancel button to UI
<button onclick="cancelGeneration()" id="cancel-btn">
    ⏹️ Stop Generation
</button>
```

---

## API Usage Documentation

### For Frontend Developers

**Basic Usage**:

```javascript
// 1. Start streaming request
const response = await fetch('http://localhost:8092/v1/chat/completions', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer your-api-key'
    },
    body: JSON.stringify({
        model: 'default',
        messages: [{ role: 'user', content: 'Tell me a long story' }],
        stream: true
    })
});

// 2. Store request ID from first chunk
const reader = response.body.getReader();
const decoder = new TextDecoder();

let requestId = null;
while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n').filter(l => l.startsWith('data: '));

    for (const line of lines) {
        if (line.includes('[DONE]')) break;
        const data = JSON.parse(line.slice(6));

        // First chunk contains request ID
        if (!requestId) {
            requestId = data.id;  // e.g., "chatcmpl-abc123"
            console.log('Request ID:', requestId);
        }

        // Display content
        console.log(data.choices[0]?.delta?.content || '');
    }
}

// 3. To cancel, call the cancel endpoint
async function cancelRequest() {
    if (requestId) {
        const response = await fetch(
            `http://localhost:8092/v1/chat/completions/${requestId}/cancel`,
            {
                method: 'POST',
                headers: { 'Authorization': 'Bearer your-api-key' }
            }
        );

        const result = await response.json();
        console.log(result);  // { "success": true, "message": "..." }
    }
}
```

**Or simpler - just close the stream**:

```javascript
// Client-side cancellation (no API call needed)
await reader.cancel();  // vMLX auto-detects and aborts
```

---

### cURL Examples

**Start streaming request**:
```bash
curl -N http://localhost:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Tell me a long story"}],
    "stream": true
  }'

# First chunk returns:
# data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk",...}
```

**Cancel the request**:
```bash
curl -X POST http://localhost:8092/v1/chat/completions/chatcmpl-abc123/cancel \
  -H "Authorization: Bearer your-api-key"

# Response:
# {"success": true, "message": "Request chatcmpl-abc123 cancelled"}
```

---

### Python SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8092/v1",
    api_key="your-api-key"
)

# Start streaming
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a long story"}],
    stream=True
)

# Get request ID from first chunk
request_id = None
for chunk in stream:
    if not request_id:
        request_id = chunk.id
        print(f"Request ID: {request_id}")

    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)

    # To cancel mid-stream:
    if should_cancel():
        # Call cancel endpoint via requests
        import requests
        requests.post(
            f"http://localhost:8092/v1/chat/completions/{request_id}/cancel",
            headers={"Authorization": "Bearer your-api-key"}
        )
        break
```

---

## Testing Plan

### Unit Tests

**Test 1: abort_request() method**
```python
def test_abort_request():
    scheduler = Scheduler()
    request = Request(request_id="test-123", ...)
    scheduler.add_request(request)

    # Abort the request
    result = scheduler.abort_request("test-123")
    assert result == True
    assert request.status == RequestStatus.FINISHED_ABORTED

    # Abort again (idempotent)
    result = scheduler.abort_request("test-123")
    assert result == False  # Already finished
```

### Integration Tests

**Test 2: Cancel endpoint**
```python
async def test_cancel_endpoint(client):
    # Start streaming request
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Count to 100"}],
            "stream": True
        }
    )

    # Get request ID from first chunk
    request_id = None
    async for line in response.iter_lines():
        if line.startswith(b"data: "):
            data = json.loads(line[6:])
            request_id = data["id"]
            break

    # Cancel the request
    cancel_response = await client.post(
        f"/v1/chat/completions/{request_id}/cancel"
    )
    assert cancel_response.status_code == 200
    assert cancel_response.json()["success"] == True
```

**Test 3: Auto-detection of closed connection**
```python
async def test_connection_closed():
    # Start streaming
    response = await client.post("/v1/chat/completions", ...)

    # Close connection mid-stream
    await response.aclose()

    # Verify request was aborted (check logs or internal state)
    # Should see: "Client disconnected, aborting chatcmpl-..."
```

### Manual Testing

**Test with vMLX Panel**:
1. Start vMLX server: `vllm-mlx serve model --continuous-batching`
2. Open vMLX panel in browser
3. Send a long request (e.g., "Write a 10 paragraph essay")
4. Click stop button mid-generation
5. Verify: Generation stops immediately, GPU usage drops

**Test with exploit.bot**:
1. Configure exploit.bot to use vMLX backend
2. Send message, click cancel button
3. Verify: Stream stops, no errors in console
4. Send another message immediately (should work)

---

## Files to Modify

### Core Implementation

1. **vllm_mlx/server.py** (~50 lines changed)
   - Add cancel endpoints (chat & completions)
   - Modify `stream_chat_completion()` to accept and pass `request_id`
   - Add connection detection logic
   - Add `fastapi_request: Request` parameter

2. **vllm_mlx/engine/base.py** (~5 lines)
   - Add `request_id: Optional[str] = None` parameter to `stream_chat()` signature

3. **vllm_mlx/engine/batched.py** (~10 lines)
   - Pass `request_id` to `engine.add_request()` in `generate()` method
   - Pass `request_id` to `engine.add_request()` in `stream_chat()` method

4. **vllm_mlx/engine/simple.py** (~10 lines)
   - Same changes as batched.py

### Documentation

5. **docs/api/cancellation.md** (NEW)
   - Complete API documentation
   - Frontend integration examples
   - cURL examples

6. **README.md** (~10 lines)
   - Add "Request Cancellation" feature to features list
   - Link to cancellation docs

7. **CHANGELOG.md** (~20 lines)
   - Document new feature
   - API compatibility notes

---

## Rollout Plan

### Phase 1: Core Implementation
- ✅ Design approved
- [ ] Implement unified request ID
- [ ] Add cancel endpoints
- [ ] Add connection detection
- [ ] Unit tests

### Phase 2: Documentation
- [ ] Write API documentation
- [ ] Add frontend integration examples
- [ ] Update README and CHANGELOG

### Phase 3: Testing
- [ ] Integration tests
- [ ] Manual testing with vMLX panel
- [ ] Manual testing with exploit.bot

### Phase 4: Deployment
- [ ] Merge to main
- [ ] Update deployed servers
- [ ] Notify frontend teams

---

## Success Criteria

✅ Cancel button in exploit.bot stops inference immediately
✅ vMLX panel can cancel requests via UI
✅ No errors when cancelling mid-stream
✅ GPU compute stops immediately after cancel
✅ Partial responses are preserved
✅ OpenAI SDK compatibility maintained
✅ All existing tests pass
✅ New cancellation tests pass

---

## Notes

- Engine core already supports `request_id` parameter (no changes needed)
- `abort_request()` method already exists in scheduler (reuse existing code)
- Connection detection is a bonus feature (nice-to-have, not required)
- exploit.bot will work with just connection detection (no code changes needed)
- vMLX panel will need UI changes to add cancel button (separate task)

---

## References

- exploit.bot cancel implementation: `/var/www/chat/routes/chat.js` line 743-754
- vMLX abort_request: `vllm_mlx/scheduler.py` line 726-759
- FastAPI disconnection detection: `request.is_disconnected()`
- OpenAI API streaming: https://platform.openai.com/docs/api-reference/streaming
