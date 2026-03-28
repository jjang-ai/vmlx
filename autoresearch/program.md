# vmlx autoresearch for agents

This is an experiment to have an LLM autonomously improve an agent through iterative experimentation, evaluated locally with vmlx_engine.

## Differences from original autoresearch-agents

- Uses **vmlx_engine** with MLX models for local inference on Apple Silicon
- No LangSmith required - evaluation happens entirely locally
- No OpenAI API key needed - runs entirely on local GPU
- Faster iteration cycles due to local execution

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The project is small. Read these files for full context:
   - `README.md` — project context and customization guide.
   - `agent.py` — the file you modify. The agent implementation.
   - `run_eval.py` — evaluation harness and metrics. Fixed during the experiment loop.
   - `dataset.json` — evaluation dataset. Fixed during the experiment loop.
4. **Verify setup**: Check that the vmlx_engine is available and the model can be loaded.
5. **Run the baseline**: Your first run establishes the baseline. Run `cd /Users/ludovicclaude/Exploration/vmlx && python -m autoresearch.run_eval` and record the results.
6. **Initialize results.tsv**: Create `results.tsv` with header row and the baseline entry.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs the agent against a fixed evaluation dataset using vmlx_engine locally. You launch it simply as: `cd .. && poetry run python -m autoresearch.run_eval > eval.log 2>&1`.

**What you CAN do:**
- Modify `agent.py` — this is the only file you edit. Everything is fair game: system prompt, tools, model choice, temperature, agent architecture, etc.
- Change the MODEL constant to try different MLX models (e.g., `mlx-community/Llama-3.2-1B-Instruct-4bit` for faster inference, or larger models for better quality)
- Add new tools or improve existing ones
- Modify the agent loop logic (max iterations, tool handling, etc.)

**What you CANNOT do:**
- Modify `run_eval.py`. It is read-only during the experiment loop. It contains the evaluation harness and metrics.
- Modify `dataset.json`. It is read-only during the experiment loop. It contains the test cases.
- Install new packages or add dependencies beyond what's already available.
- Modify the evaluators. The scoring functions in `run_eval.py` are the ground truth metrics.

**The goal is simple: get the highest `overall_score`.** Since the evaluation is fixed, you don't need to worry about the eval pipeline — it's always the same. Everything in `agent.py` is fair game. The only constraint is that the code runs without crashing and the agent returns valid responses matching the contract expected by `run_eval.py`.

**Cost**: No API costs - runs entirely on local hardware! Use any MLX model you want.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.01 overall_score improvement that adds 30 lines of hacky prompt engineering? Probably not worth it. A 0.01 improvement from simplifying the prompt? Definitely keep. An improvement of ~0 but much simpler code? Keep.

## Output format

Once the evaluation finishes it prints a summary like this:

```
---
avg_correctness: 0.850000
avg_helpfulness: 0.900000
avg_tool_usage: 0.750000
overall_score: 0.833333
num_examples: 20
num_errors: 0
```

You can extract the key metric from the log file:

```
grep "^overall_score:" eval.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns:

```
commit	overall_score	correctness	helpfulness	tool_usage	status	description
```

1. git commit hash (short, 7 chars)
2. overall_score achieved (e.g. 0.833333) — use 0.000000 for crashes
3. correctness score
4. helpfulness score
5. tool_usage score
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	overall_score	correctness	helpfulness	tool_usage	status	description
a1b2c3d	0.833333	0.850000	0.900000	0.750000	keep	baseline
b2c3d4e	0.866667	0.900000	0.900000	0.800000	keep	improved system prompt with explicit tool instructions
c3d4e5f	0.800000	0.800000	0.850000	0.750000	discard	switched to smaller model
d4e5f6g	0.000000	0.000000	0.000000	0.000000	crash	added broken custom tool
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Think about what to try next. Consider:
   - Improving the system prompt (more specific instructions, better formatting guidance)
   - Adding or improving tools (better descriptions, more capable implementations)
   - Changing the model (different MLX models, 4bit vs 8bit quantization)
   - Temperature tuning (try temperature=0.1 or 0.2 for slightly more creative responses)
   - Output formatting (add instructions about response format - concise, direct answers)
   - Tool improvements (make tools more robust, handle edge cases better)
   - Agent loop logic (max iterations, handling edge cases)
3. Edit `agent.py` with your experimental idea
4. git commit
5. Run the experiment: `cd /Users/ludovicclaude/Exploration/vmlx && python -m autoresearch.run_eval > eval.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. Read out the results: `grep "^overall_score:\|^avg_" eval.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 eval.log` to read the error and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the TSV
9. If overall_score improved (higher), you "advance" the branch, keeping the git commit
10. If overall_score is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind further back but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~30-60 seconds depending on model size. If a run exceeds 5 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes, use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the TSV, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the agent code, look at which eval cases are failing, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

**Model loading**: Each experiment run will load the model from the vmlx directory. This can take a few seconds. The model stays loaded in memory between runs in the same Python process, but each `python -m autoresearch.run_eval` invocation starts fresh.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~45 seconds then you can run approx 80/hour, for a total of about 640 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Ideas to try

Here are some starting ideas (but don't limit yourself to these):

- **Prompt engineering**: Add explicit instructions for when to use tools vs. answer directly
- **Tool descriptions**: Make tool docstrings more descriptive so the LLM knows when to use them
- **Few-shot examples**: Add examples to the system prompt showing correct behavior
- **Model upgrade**: Try a more capable MLX model (e.g., 8B instead of 3B, or 8bit instead of 4bit)
- **Temperature tuning**: Try temperature=0.1 or 0.2 for slightly more creative responses
- **Output formatting**: Add instructions about response format (concise, direct answers)
- **Tool improvements**: Make tools more robust, handle edge cases better
- **Max iterations**: Adjust the max_iterations in the agent loop
- **System prompt**: Try different system prompts emphasizing different aspects
- **Model comparison**: Compare different model families (Llama, Qwen, etc.)
