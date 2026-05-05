# Autoresearch for vmlx Agents

> Autonomous agent optimization using vmlx_engine and local MLX models on Apple Silicon.

## The Idea

Give an AI coding agent a working agent implementation and an evaluation dataset. Let it experiment autonomously: modify the agent code, run evals locally on Apple Silicon, check if scores improved, keep or discard, and repeat. You wake up in the morning to a log of experiments and (hopefully) a better agent.

```
┌─────────────────────────────────────────────────────┐
│                  EXPERIMENT LOOP                     │
│                                                      │
│  1. Read agent.py + results so far                   │
│  2. Propose a change (prompt, tools, model)            │
│  3. Edit agent.py                                    │
│  4. git commit                                       │
│  5. Run evaluation: python run_eval.py               │
│  6. Parse scores from eval output                    │
│  7. If improved → keep commit                        │
│     If worse   → git reset  (discard)                │
│  8. Log result to results.tsv                        │
│  9. Repeat forever                                   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Comparison with original autoresearch-agents

| | autoresearch-agents | vmlx autoresearch |
|---|---|---|
| **LLM Backend** | OpenAI API (cloud) | vmlx_engine + MLX (local) |
| **API Keys needed** | OpenAI, LangSmith | None |
| **Cost** | API usage fees | Free (runs on local GPU) |
| **Privacy** | Data sent to cloud | Data stays local |
| **Latency** | Network-dependent | Fast local inference |
| **Evaluation** | LangSmith | Local code-based |

## Project Structure

```
agent.py        — YOUR agent implementation (the coding agent optimizes this)
run_eval.py     — YOUR evaluation harness + metrics (customize before starting)
dataset.json    — YOUR evaluation dataset (customize before starting)
program.md      — instructions for the AI coding agent (customize before starting)
results.tsv     — experiment log (auto-generated)
```

**Before you start the autonomous loop**, you customize everything to fit your use case. Once the loop begins, `run_eval.py` and `dataset.json` are fixed — only `agent.py` changes.

## Quick Start

### Prerequisites

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4)
- vmlx_engine installed in your environment
- mlx-lm installed (`poetry install`)
- MLX models downloaded (will be auto-downloaded on first use)

### Setup

```bash
# 1. Verify vmlx_engine is available (run from parent vmlx directory)
cd /Users/ludovicclaude/Exploration/vmlx
poetry run python -c "from vmlx_engine.engine.simple import SimpleEngine; print('vmlx_engine OK')"

# 2. Verify the baseline agent works (run as module from vmlx directory)
poetry run python -m autoresearch.agent "What is the capital of France?"

# 3. Run a single evaluation (run as module from vmlx directory)
poetry run python -m autoresearch.run_eval
```

### Running the Autonomous Agent

Point your coding agent (Claude Code, Cursor, Codex, etc.) at this directory and send this prompt:

<details>
<summary><b>📋 Copy-paste prompt for Claude Code / Cursor / Codex</b></summary>

```
I want you to autonomously optimize an AI agent using an eval-driven experiment loop.

Here's how it works:
- `agent.py` is the agent implementation. This is the ONLY file you modify.
- `run_eval.py` is the evaluation harness. It runs the agent against a fixed dataset
  and scores it locally using vmlx_engine. Do NOT modify this file.
- `dataset.json` is the evaluation dataset. Do NOT modify this file.
- `program.md` has detailed instructions for the experiment loop.

Read program.md now and follow the setup instructions. Once setup is confirmed,
start the experiment loop and run autonomously — do not stop to ask me questions.
Keep experimenting until I interrupt you.
```

</details>

The coding agent will then autonomously iterate on `agent.py`, running evals and tracking results. You can walk away and come back to a log of experiments in `results.tsv`.

#### First time here? Use this prompt instead to set everything up from scratch:

<details>
<summary><b>📋 Copy-paste setup + run prompt</b></summary>

```
I want to set up and run autoresearch for vmlx agents — an autonomous experiment loop
that optimizes an AI agent using local MLX inference on Apple Silicon.

First, help me get set up:
1. Verify vmlx_engine is available
2. Verify the agent works: cd /Users/ludovicclaude/Exploration/vmlx && python -m autoresearch.agent "What is 2+2?"
3. Run a baseline eval: cd /Users/ludovicclaude/Exploration/vmlx && python -m autoresearch.run_eval

If I want to customize this for my own agent, walk me through:
- Replacing agent.py with my own agent implementation
- Replacing dataset.json with my own test cases
- Updating the evaluators in run_eval.py for my use case

Once everything is set up and the baseline looks good, read program.md and start
the autonomous experiment loop. Do not stop to ask me questions — keep experimenting
until I interrupt you.
```

</details>

## Bring Your Own Everything

This ships with a working example (a Q&A agent with calculator tools), but it's designed as a **template**. Customize all three components before starting the autonomous loop.

### Bring Your Own Agent

Replace `agent.py` with any agent implementation using vmlx_engine. The key contract: your agent must expose `run_agent_with_tools(question: str) -> dict` that returns a dict with at least:
- `response`: the text response
- `tools_used`: list of tool names used (can be empty)
- `error`: optional boolean indicating an error occurred

**Example:**

```python
from vmlx_engine.simple import SimpleEngine

async def my_agent(question: str) -> dict:
    engine = SimpleEngine("mlx-community/Llama-3.2-3B-Instruct-4bit")
    await engine.start()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]
    
    output = await engine.chat(messages=messages, max_tokens=256)
    
    return {
        "response": output.text,
        "tools_used": [],
    }

def run_agent_with_tools(question: str) -> dict:
    import asyncio
    return asyncio.run(my_agent(question))
```

### Bring Your Own Dataset

Replace `dataset.json` with your evaluation cases. The format is a JSON array of objects with `inputs` and `outputs`:

```json
[
  {
    "inputs": {"question": "Your input here"},
    "outputs": {"answer": "Expected output", "expected_tool_use": true}
  }
]
```

The field names are up to you — just make sure your evaluators in `run_eval.py` reference the same fields. Some ideas:

- **Q&A agent**: `inputs: {question}`, `outputs: {answer}`
- **RAG agent**: `inputs: {question}`, `outputs: {answer, source_docs}`
- **Code agent**: `inputs: {task}`, `outputs: {code, test_result}`
- **Customer support**: `inputs: {ticket}`, `outputs: {response, category, priority}`

### Bring Your Own Evaluators

Modify the evaluators in `run_eval.py` to match your quality criteria. Each evaluator is a function that takes `(outputs, example)` and returns `{"score": number, "comment": "..."}`.

```python
# Code-based evaluator
def exact_match(outputs: dict, example: dict) -> dict:
    actual = outputs.get("response", "")
    expected = example.get("outputs", {}).get("answer", "")
    return {"score": 1 if expected.lower() in actual.lower() else 0, "comment": ""}
```

After modifying evaluators, update `program.md` to reflect the new metric names in the output format and TSV columns.

### Update program.md

After customizing the above, update `program.md` to match:
- Update the file list in the Setup section
- Update the "Ideas to try" section with domain-specific suggestions
- Update the output format section if your metrics changed
- Update the TSV columns to match your evaluator names

## How It Works

### The Evaluation (`run_eval.py`)

Uses local code-based evaluators to score the agent. No network calls required. The evaluators check:
- **Correctness**: Does the response contain the expected answer?
- **Helpfulness**: Is the response clear and complete?
- **Tool Usage**: Did the agent use tools when expected?

All agent runs are local and fast — perfect for rapid iteration.

### The Experiment Loop (`program.md`)

The `program.md` file is the "skill" that drives the autonomous coding agent. It tells the agent to:
1. Edit `agent.py` with an experimental idea
2. Commit, run eval, check scores
3. Keep improvements, discard regressions
4. Log everything to `results.tsv`
5. Never stop until interrupted

## Available MLX Models

Popular MLX models for agent tasks:

- `mlx-community/Llama-3.2-3B-Instruct-4bit` - Fast, good baseline
- `mlx-community/Llama-3.2-1B-Instruct-4bit` - Very fast, lightweight
- `mlx-community/Llama-3.2-3B-Instruct-8bit` - Higher quality
- `mlx-community/Qwen2.5-3B-Instruct-4bit` - Good for tool calling
- `mlx-community/Phi-4-mini-instruct-4bit` - Small but capable

Change the `MODEL` constant in `agent.py` to experiment with different models.

## License

MIT
