# Reasoning Models

vmlx-engine supports reasoning models that show their thinking process before giving an answer. Models like Qwen3 and DeepSeek-R1 wrap their reasoning in `<think>...</think>` tags, and vmlx-engine can parse these tags to separate the reasoning from the final response.

## Why Use Reasoning Parsing?

When a reasoning model generates output, it typically looks like this:

```
<think>
Let me analyze this step by step.
First, I need to consider the constraints.
The answer should be a prime number less than 10.
Checking: 2, 3, 5, 7 are all prime and less than 10.
</think>
The prime numbers less than 10 are: 2, 3, 5, 7.
```

Without reasoning parsing, you get the raw output with the tags included. With reasoning parsing enabled, the thinking process and final answer are separated into distinct fields in the API response.

## Getting Started

### Start the Server with Reasoning Parser

The easiest way is to use `auto`, which detects the correct parser from the model name:

```bash
# Auto-detect parser from model name (recommended)
vmlx-engine serve mlx-community/Qwen3-8B-4bit --reasoning-parser auto

# Explicit parser selection
vmlx-engine serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# For DeepSeek-R1 models
vmlx-engine serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

### Auto-Detection

When you pass `--reasoning-parser auto`, vmlx-engine uses its model config registry to match the model name against known patterns and select the appropriate parser:

| Model Pattern | Parser Selected |
|---------------|----------------|
| Qwen3 (all variants) | `qwen3` |
| DeepSeek-R1, DeepSeek-V3 | `deepseek_r1` |
| GLM-4.7 Flash, GPT-OSS | `openai_gptoss` |
| GLM-4.7, GLM-Z1 | `deepseek_r1` |
| Unknown model | No parser (content passed through) |

Auto-detection also works for tool call parsers with `--enable-auto-tool-choice` (defaults to `--tool-call-parser auto`).

### API Response Format

When reasoning parsing is enabled, the API response includes a `reasoning` field:

**Non-streaming response:**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The prime numbers less than 10 are: 2, 3, 5, 7.",
      "reasoning": "Let me analyze this step by step.\nFirst, I need to consider the constraints.\nThe answer should be a prime number less than 10.\nChecking: 2, 3, 5, 7 are all prime and less than 10."
    }
  }]
}
```

**Streaming response:**

Chunks are sent separately for reasoning and content. During the reasoning phase, chunks have `reasoning` populated. When the model transitions to the final answer, chunks have `content` populated:

```json
{"delta": {"reasoning": "Let me analyze"}}
{"delta": {"reasoning": " this step by step."}}
{"delta": {"reasoning": "\nFirst, I need to"}}
...
{"delta": {"content": "The prime"}}
{"delta": {"content": " numbers less than 10"}}
{"delta": {"content": " are: 2, 3, 5, 7."}}
```

## Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Non-streaming
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What are the prime numbers less than 10?"}]
)

message = response.choices[0].message
print("Reasoning:", message.reasoning)  # The thinking process
print("Answer:", message.content)        # The final answer
```

### Streaming with Reasoning

```python
reasoning_text = ""
content_text = ""

stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Solve: 2 + 2 = ?"}],
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if hasattr(delta, 'reasoning') and delta.reasoning:
        reasoning_text += delta.reasoning
        print(f"[Thinking] {delta.reasoning}", end="")
    if delta.content:
        content_text += delta.content
        print(delta.content, end="")

print(f"\n\nFinal reasoning: {reasoning_text}")
print(f"Final answer: {content_text}")
```

## Supported Parsers

### Auto (`auto`)

Automatically detects the correct parser based on the model name. This is the recommended option — it works for all known reasoning models and gracefully falls back to no parsing for models that don't use thinking tags.

```bash
vmlx-engine serve mlx-community/Qwen3-8B-4bit --reasoning-parser auto
```

### Qwen3 Parser (`qwen3`)

For Qwen3 models that use explicit `<think>` and `</think>` tags.

- If tags are missing, output is treated as regular content (no reasoning)
- Best for: Qwen3-0.6B, Qwen3-4B, Qwen3-8B, Qwen3-Coder-Next, GLM-4.7 and similar models

```bash
vmlx-engine serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

### DeepSeek-R1 Parser (`deepseek_r1`)

For DeepSeek-R1 models that may omit the opening `<think>` tag.

- More lenient than Qwen3 parser
- Handles implicit reasoning mode where `<think>` is injected in the prompt
- Content before `</think>` is treated as reasoning even without `<think>`
- Also used for GLM-4.7 and GLM-Z1 (non-Flash variants that use `<think>` tags with `think_in_template=True`)

```bash
vmlx-engine serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

### GPT-OSS / Harmony Parser (`openai_gptoss`)

For GLM-4.7 Flash and GPT-OSS models that use the Harmony protocol with channel markers instead of `<think>` tags.

- Uses `<|channel|>analysis` for reasoning and `<|channel|>final` for content
- Stateful streaming with `_harmony_active`, `_emitted_reasoning`, `_emitted_content` tracking
- `think_in_template=False` — reasoning is NOT injected in the chat template

```bash
vmlx-engine serve lmstudio-community/GLM-4.7-Flash-MLX-8bit --reasoning-parser openai_gptoss
```

## How It Works

The reasoning parser uses text-based detection to identify thinking tags in the model output. During streaming, it tracks the current position in the output to correctly route each token to either `reasoning` or `content`.

```
Model Output:        <think>Step 1: analyze...</think>The answer is 42.
                     ├─────────────────────┤├─────────────────────┤
Parsed:              │     reasoning       ││       content       │
                     └─────────────────────┘└─────────────────────┘
```

When no thinking tags are present in the output, all text is routed to `content`. This means `--reasoning-parser auto` is safe to use with any model — models that don't produce `<think>` tags will work normally.

## Tips for Best Results

### Prompting

Reasoning models work best when you encourage step-by-step thinking:

```python
messages = [
    {"role": "system", "content": "Think through problems step by step before answering."},
    {"role": "user", "content": "What is 17 × 23?"}
]
```

### Handling Missing Reasoning

Some prompts may not trigger reasoning. In these cases, `reasoning` will be `None` and all output goes to `content`:

```python
message = response.choices[0].message
if message.reasoning:
    print(f"Model's thought process: {message.reasoning}")
print(f"Answer: {message.content}")
```

### Temperature and Reasoning

Lower temperatures tend to produce more consistent reasoning patterns:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    temperature=0.3  # More focused reasoning
)
```

## Backward Compatibility

When `--reasoning-parser` is not specified, the server behaves as before:
- Thinking tags are included in the `content` field
- No `reasoning` field is added to responses

This ensures existing applications continue to work without changes.

## Example: Math Problem Solver

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

def solve_math(problem: str) -> dict:
    """Solve a math problem and return reasoning + answer."""
    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a math tutor. Show your work."},
            {"role": "user", "content": problem}
        ],
        temperature=0.2
    )

    message = response.choices[0].message
    return {
        "problem": problem,
        "work": message.reasoning,
        "answer": message.content
    }

result = solve_math("If a train travels 120 km in 2 hours, what is its average speed?")
print(f"Problem: {result['problem']}")
print(f"\nWork shown:\n{result['work']}")
print(f"\nFinal answer: {result['answer']}")
```

## Curl Examples

### Non-streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 15% of 80?"}]
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 15% of 80?"}],
    "stream": true
  }'
```

## Troubleshooting

### No reasoning field in response

- Make sure you started the server with `--reasoning-parser` (try `auto`)
- Check that the model actually uses thinking tags (not all prompts trigger reasoning)
- Models that don't output `<think>` tags will have `reasoning: null` — this is expected

### All text appears as reasoning, content is empty

- This was a bug in earlier versions. Update to the latest version.
- The fix: when no `<think>` tags are present, text defaults to `content` (not `reasoning`)

### Reasoning appears in content

- The model may not be using the expected tag format
- Try `--reasoning-parser auto` to let the system detect the right parser
- Try a different parser (`qwen3` vs `deepseek_r1`)

### Truncated reasoning

- Increase `--max-tokens` if the model is hitting the token limit mid-thought

## Related

- [Supported Models](../reference/models.md) - Models that support reasoning
- [Configuration Reference](../reference/configuration.md) - All server options
- [CLI Reference](../reference/cli.md) - Command line options
