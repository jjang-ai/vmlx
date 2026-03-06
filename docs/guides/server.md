# OpenAI-Compatible Server

vmlx-engine provides a FastAPI server with full OpenAI API compatibility.

## Starting the Server

### Simple Mode (Default)

Maximum throughput for single user:

```bash
vmlx-engine serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
```

### Continuous Batching Mode

For multiple concurrent users:

```bash
vmlx-engine serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

### With Paged Cache

Memory-efficient caching for production:

```bash
vmlx-engine serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching --use-paged-cache
```

## Server Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port` | Server port | 8000 |
| `--host` | Server host | 0.0.0.0 |
| `--api-key` | API key for authentication | None |
| `--rate-limit` | Requests per minute per client (0 = disabled) | 0 |
| `--timeout` | Request timeout in seconds | 300 |
| `--continuous-batching` | Enable batching for multi-user | False |
| `--use-paged-cache` | Enable paged KV cache | False |
| `--cache-memory-mb` | Cache memory limit in MB | Auto |
| `--cache-memory-percent` | Fraction of RAM for cache | 0.30 |
| `--max-tokens` | Default max tokens | 32768 |
| `--default-temperature` | Default temperature when not specified | None |
| `--default-top-p` | Default top_p when not specified | None |
| `--stream-interval` | Tokens per stream chunk | 1 |
| `--mcp-config` | Path to MCP config file | None |
| `--reasoning-parser` | Parser for reasoning models (`qwen3`, `deepseek_r1`, `openai_gptoss`) | None |
| `--embedding-model` | Pre-load an embedding model at startup | None |
| `--enable-auto-tool-choice` | Enable automatic tool calling | False |
| `--tool-call-parser` | Tool call parser (see [Tool Calling](tool-calling.md)) | None |
| `--kv-cache-quantization` | Quantize KV cache in prefix storage (`none`, `q4`, `q8`) | none |
| `--kv-cache-group-size` | Group size for KV cache quantization | 64 |

## API Endpoints

### Chat Completions

```bash
POST /v1/chat/completions
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Non-streaming
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

# Streaming
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Responses API

```bash
POST /v1/responses
```

The Responses API is the wire format used by agentic coding tools like Codex CLI, OpenCode, and Cline. It supports streaming with SSE events and multi-turn tool calling.

```python
import httpx

response = httpx.post("http://localhost:8000/v1/responses", json={
    "model": "default",
    "stream": True,
    "input": "What is 2 + 2?",
    "max_output_tokens": 100
})
```

**With tool calling** (for agentic tools):

```python
response = httpx.post("http://localhost:8000/v1/responses", json={
    "model": "default",
    "stream": True,
    "tools": [{"type": "function", "name": "shell", "description": "Run command",
               "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}],
    "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List files"}]},
        {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{\"command\": \"ls\"}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "file1.txt\nfile2.txt"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Read file1.txt"}]}
    ]
})
```

**Codex CLI configuration** (`~/.codex/config.toml`):
```toml
model_provider = "vmlx"
model = "your-model-name"
[model_providers.vmlx]
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
```

### Completions

```bash
POST /v1/completions
```

```python
response = client.completions.create(
    model="default",
    prompt="The capital of France is",
    max_tokens=50
)
```

### Models

```bash
GET /v1/models
```

Returns available models.

### Embeddings

```bash
POST /v1/embeddings
```

```python
response = client.embeddings.create(
    model="mlx-community/multilingual-e5-small-mlx",
    input="Hello world"
)
print(response.data[0].embedding[:5])  # First 5 dimensions
```

See [Embeddings Guide](embeddings.md) for details.

### Request Cancellation

```bash
POST /v1/chat/completions/{request_id}/cancel
POST /v1/completions/{request_id}/cancel
POST /v1/responses/{request_id}/cancel
```

Cancel an in-progress request. The request ID is returned in the first streaming chunk. See [Cancellation API](../api/cancellation.md) for details.

### Health Check

```bash
GET /health
```

Returns server status.

### Cache Management

Inspect, warm, and clear the prefix cache.

```bash
# Cache statistics (entries, memory, hit/miss, quantization info)
GET /v1/cache/stats

# List cached prefixes with token counts and memory
GET /v1/cache/entries

# Pre-warm cache with system prompts
POST /v1/cache/warm
{"prompts": ["You are a helpful assistant.", "You are a coding expert."]}

# Clear cache (type: prefix, multimodal, or all)
DELETE /v1/cache?type=prefix
DELETE /v1/cache?type=all
```

```bash
# Example: warm cache then verify
curl -X POST http://localhost:8000/v1/cache/warm \
  -H "Content-Type: application/json" \
  -d '{"prompts":["You are a helpful assistant."]}'

curl http://localhost:8000/v1/cache/stats
curl http://localhost:8000/v1/cache/entries
```

## KV Cache Quantization

Reduce prefix cache memory by 2-4x using quantized storage. During inference, full-precision KVCache is used (no quality loss). When storing to prefix cache, KV data is quantized; when retrieving, it's dequantized back.

```bash
# q8 quantization (2x memory reduction, minimal quality loss)
vmlx-engine serve mlx-community/Qwen3-8B-4bit \
  --continuous-batching --kv-cache-quantization q8

# q4 quantization (4x memory reduction)
vmlx-engine serve mlx-community/Qwen3-8B-4bit \
  --continuous-batching --kv-cache-quantization q4

# Check quantization is active
curl http://localhost:8000/v1/cache/stats
```

Works with all three cache backends (memory-aware, paged, legacy), continuous batching, tool calls, and reasoning models.

## Tool Calling

Enable OpenAI-compatible tool calling with `--enable-auto-tool-choice`:

```bash
vmlx-engine serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

Use the `--tool-call-parser` option to select the parser for your model:

| Parser | Models |
|--------|--------|
| `auto` | Auto-detect (tries all parsers) |
| `mistral` | Mistral, Devstral |
| `qwen` | Qwen, Qwen3 |
| `llama` | Llama 3.x, 4.x |
| `hermes` | Hermes, NousResearch |
| `deepseek` | DeepSeek V3, R1 |
| `kimi` | Kimi K2, Moonshot |
| `granite` | IBM Granite 3.x, 4.x |
| `nemotron` | NVIDIA Nemotron |
| `xlam` | Salesforce xLAM |
| `functionary` | MeetKai Functionary |
| `glm47` | GLM-4.7, GLM-4.7-Flash |

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]
)

if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"{tc.function.name}: {tc.function.arguments}")
```

See [Tool Calling Guide](tool-calling.md) for full documentation.

## Reasoning Models

For models that show their thinking process (Qwen3, DeepSeek-R1), use `--reasoning-parser` to separate reasoning from the final answer:

```bash
# Qwen3 models
vmlx-engine serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# DeepSeek-R1 models
vmlx-engine serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

The API response includes a `reasoning` field with the model's thought process:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)

print(response.choices[0].message.reasoning)  # Step-by-step thinking
print(response.choices[0].message.content)    # Final answer
```

For streaming, reasoning chunks arrive first, followed by content chunks:

```python
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.reasoning:
        print(f"[Thinking] {delta.reasoning}")
    if delta.content:
        print(delta.content, end="")
```

See [Reasoning Models Guide](reasoning.md) for full details.

## Structured Output (JSON Mode)

Force the model to return valid JSON using `response_format`:

### JSON Object Mode

Returns any valid JSON:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={"type": "json_object"}
)
# Output: {"colors": ["red", "blue", "green"]}
```

### JSON Schema Mode

Returns JSON matching a specific schema:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "colors",
            "schema": {
                "type": "object",
                "properties": {
                    "colors": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["colors"]
            }
        }
    }
)
# Output validated against schema
data = json.loads(response.choices[0].message.content)
assert "colors" in data
```

### Curl Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "List 3 colors"}],
    "response_format": {"type": "json_object"}
  }'
```

## Curl Examples

### Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Streaming Configuration

Control streaming behavior with `--stream-interval`:

| Value | Behavior |
|-------|----------|
| `1` (default) | Send every token immediately |
| `2-5` | Batch tokens before sending |
| `10+` | Maximum throughput, chunkier output |

```bash
# Smooth streaming
vmlx-engine serve model --continuous-batching --stream-interval 1

# Batched streaming (better for high-latency networks)
vmlx-engine serve model --continuous-batching --stream-interval 5
```

## Open WebUI Integration

```bash
# 1. Start vmlx-engine server
vmlx-engine serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# 2. Start Open WebUI
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main

# 3. Open http://localhost:3000
```

## Production Deployment

### With systemd

Create `/etc/systemd/system/vmlx-engine.service`:

```ini
[Unit]
Description=vMLX Engine Server
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/vmlx-engine serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching --use-paged-cache --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable vmlx-engine
sudo systemctl start vmlx-engine
```

### Recommended Settings

For production with 50+ concurrent users:

```bash
vmlx-engine serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120 \
  --port 8000
```
