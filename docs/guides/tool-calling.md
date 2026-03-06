# Tool Calling

vmlx-engine supports OpenAI-compatible tool calling (function calling) with automatic parsing for many popular model families.

## Quick Start

Enable tool calling by adding the `--enable-auto-tool-choice` flag when starting the server. The correct parser is auto-detected from the model name:

```bash
# Auto-detect parser from model name (recommended)
vmlx-engine serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice

# Or specify parser explicitly
vmlx-engine serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

Then use tools with the standard OpenAI API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

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
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }]
)

# Check for tool calls
if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"Function: {tc.function.name}")
        print(f"Arguments: {tc.function.arguments}")
```

## Supported Parsers

Use `--tool-call-parser` to select a parser for your model family:

| Parser | Aliases | Models | Format |
|--------|---------|--------|--------|
| `auto` | | Any model | Auto-detects format (tries all parsers) |
| `mistral` | | Mistral, Devstral | `[TOOL_CALLS]` JSON array |
| `qwen` | `qwen3` | Qwen, Qwen3 | `<tool_call>` XML or `[Calling tool:]` |
| `llama` | `llama3`, `llama4` | Llama 3.x, 4.x | `<function=name>` tags |
| `hermes` | `nous` | Hermes, NousResearch | `<tool_call>` JSON in XML |
| `deepseek` | `deepseek_v3`, `deepseek_r1` | DeepSeek V3, R1 | Unicode delimiters |
| `kimi` | `kimi_k2`, `moonshot` | Kimi K2, Moonshot | `<\|tool_call_begin\|>` tokens |
| `granite` | `granite3` | IBM Granite 3.x, 4.x | `<\|tool_call\|>` or `<tool_call>` |
| `nemotron` | `nemotron3` | NVIDIA Nemotron | `<tool_call><function=...><parameter=...>` |
| `xlam` | | Salesforce xLAM | JSON with `tool_calls` array |
| `functionary` | `meetkai` | MeetKai Functionary | Multiple function blocks |
| `glm47` | `glm4` | GLM-4.7, GLM-4.7-Flash | `<tool_call>` with `<arg_key>`/`<arg_value>` XML |

## Model Examples

### Mistral / Devstral

```bash
# Devstral Small (optimized for coding and tool use)
vmlx-engine serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral

# Mistral Instruct
vmlx-engine serve mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral
```

### Qwen

```bash
# Qwen3
vmlx-engine serve mlx-community/Qwen3-4B-4bit \
  --enable-auto-tool-choice --tool-call-parser qwen
```

### Llama

```bash
# Llama 3.2
vmlx-engine serve mlx-community/Llama-3.2-3B-Instruct-4bit \
  --enable-auto-tool-choice --tool-call-parser llama
```

### DeepSeek

```bash
# DeepSeek V3
vmlx-engine serve mlx-community/DeepSeek-V3-0324-4bit \
  --enable-auto-tool-choice --tool-call-parser deepseek
```

### IBM Granite

```bash
# Granite 4.0
vmlx-engine serve mlx-community/granite-4.0-tiny-preview-4bit \
  --enable-auto-tool-choice --tool-call-parser granite
```

### NVIDIA Nemotron

```bash
# Nemotron 3 Nano
vmlx-engine serve mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-6Bit \
  --enable-auto-tool-choice --tool-call-parser nemotron
```

### GLM-4.7

```bash
# GLM-4.7 Flash
vmlx-engine serve lmstudio-community/GLM-4.7-Flash-MLX-8bit \
  --enable-auto-tool-choice --tool-call-parser glm47
```

### Kimi K2

```bash
# Kimi K2
vmlx-engine serve mlx-community/Kimi-K2-Instruct-4bit \
  --enable-auto-tool-choice --tool-call-parser kimi
```

### Salesforce xLAM

```bash
# xLAM
vmlx-engine serve mlx-community/xLAM-2-fc-r-4bit \
  --enable-auto-tool-choice --tool-call-parser xlam
```

## Auto Parser

When `--enable-auto-tool-choice` is used without specifying `--tool-call-parser`, the parser defaults to `auto`. Auto-detection works in two stages:

**Stage 1: Model name detection** — vmlx-engine checks the model name against its registry and selects a model-specific parser (e.g., Qwen -> `qwen`, Mistral -> `mistral`). This is the most reliable method.

**Stage 2: Format detection (fallback)** — If no model-specific parser is matched, the generic parser tries to detect the format from the output:
1. Mistral (`[TOOL_CALLS]`)
2. Qwen bracket (`[Calling tool:]`)
3. Nemotron (`<tool_call><function=...><parameter=...>`)
4. Qwen/Hermes XML (`<tool_call>{...}</tool_call>`)
5. Llama (`<function=name>{...}</function>`)
6. Raw JSON

```bash
# Auto is the default when no parser is specified
vmlx-engine serve mlx-community/Qwen3-4B-4bit \
  --enable-auto-tool-choice

# Equivalent to:
vmlx-engine serve mlx-community/Qwen3-4B-4bit \
  --enable-auto-tool-choice --tool-call-parser auto
```

## Streaming Tool Calls

Tool calls work with streaming. The tool call information is sent when the model finishes generating:

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's 25 * 17?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate math expressions",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.tool_calls:
        for tc in chunk.choices[0].delta.tool_calls:
            print(f"Tool call: {tc.function.name}({tc.function.arguments})")
```

## Handling Tool Results

After receiving a tool call, execute the function and send the result back:

```python
import json

# First request - model decides to call a tool
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[weather_tool]
)

# Get the tool call
tool_call = response.choices[0].message.tool_calls[0]
tool_call_id = tool_call.id
function_name = tool_call.function.name
arguments = json.loads(tool_call.function.arguments)

# Execute the function (your implementation)
result = get_weather(**arguments)  # {"temperature": 22, "condition": "sunny"}

# Send result back to model
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {"role": "assistant", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)}
    ],
    tools=[weather_tool]
)

print(response.choices[0].message.content)
# "The weather in Tokyo is sunny with a temperature of 22C."
```

## Think Tag Handling

Models that produce `<think>...</think>` reasoning tags (like DeepSeek-R1, Qwen3, GLM-4.7) are handled automatically. The parser strips thinking content before extracting tool calls, so reasoning tags never interfere with tool call parsing.

This works even when `<think>` was injected in the prompt (implicit think tags with only a closing `</think>`).

## Responses API (Agentic Tools)

The Responses API (`/v1/responses`) provides tool calling support for agentic coding tools like Codex CLI, OpenCode, and Cline. Multi-turn tool use is fully supported:

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "stream": true,
    "tools": [{"type": "function", "name": "shell", "description": "Run command",
               "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}],
    "input": [
      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List files"}]},
      {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{\"command\": \"ls\"}"},
      {"type": "function_call_output", "call_id": "call_1", "output": "file1.txt\nfile2.txt"},
      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Read file1.txt"}]}
    ]
  }'
```

SSE events follow the OpenAI Responses API spec:
- `response.created` — Response started
- `response.output_item.added` — New output item (text message or function_call)
- `response.function_call_arguments.delta` — Streaming function arguments
- `response.function_call_arguments.done` — Complete arguments
- `response.output_text.delta` — Streaming text content
- `response.completed` — Response finished with usage stats

## Parser Fallback

When a specific parser (e.g., `--tool-call-parser qwen`) doesn't find tool calls in the model's output, vmlx-engine automatically falls back to the generic parser. The generic parser tries all known formats:

1. Mistral (`[TOOL_CALLS]`)
2. Qwen/Hermes XML (`<tool_call>{...}</tool_call>`)
3. Nemotron (`<tool_call><function=...><parameter=...>`)
4. Llama (`<function=name>{...}</function>`)
5. Raw JSON

This means you can safely select a specific parser even if the model occasionally uses a different format.

## Tool Fallback Injection

Some models' chat templates silently drop tool definitions under certain conditions (e.g., Qwen 3.5 family when `enable_thinking=False`). vmlx-engine automatically detects this and injects tool schemas as a system prompt fallback:

1. After applying the chat template, vmlx-engine checks if the first tool name appears in the rendered prompt
2. If tools are missing, it injects an XML `<tool_call>` instruction set into the system message
3. The template is re-applied with the modified messages

This is **model-agnostic** — it works for any model family, including:
- Qwen 2.5, 3, 3.5, and VL variants (when thinking is disabled)
- Models without native tool-aware templates
- Custom fine-tuned models

No user configuration is needed — the fallback activates automatically when tools are requested but the template drops them.

## CLI Reference

| Option | Description |
|--------|-------------|
| `--enable-auto-tool-choice` | Enable automatic tool calling (auto-detects parser from model name) |
| `--tool-call-parser` | Override auto-detected parser (see table above) |

See [CLI Reference](../reference/cli.md) for all options.
