# vMLX — Installation Guide

## Requirements

- **macOS 13+** (Ventura or later)
- **Apple Silicon** (M1, M2, M3, M4, or later) — Intel Macs are NOT supported
- At least **16GB unified memory** recommended (8GB minimum for small models)
- MLX-format model weights downloaded locally (e.g., from Hugging Face `mlx-community`)

## Installation

### Quick Install

1. Copy `vMLX.app` to `/Applications/`
2. Double-click to launch

### First Launch — Gatekeeper Warning

Since vMLX is not signed with an Apple Developer ID, macOS will block it on first launch.

**Option A — Right-click method (recommended):**
1. Right-click (or Control-click) `vMLX.app` in Applications
2. Select "Open" from the context menu
3. Click "Open" in the dialog that appears
4. You only need to do this once — subsequent launches work normally

**Option B — Terminal method:**
```bash
xattr -cr /Applications/vMLX.app
```

**Option C — System Settings:**
1. Try to open vMLX normally (it will be blocked)
2. Go to System Settings → Privacy & Security
3. Scroll down — you'll see "vMLX was blocked"
4. Click "Open Anyway"

## What's Included

vMLX is fully self-contained. No additional software installation needed.

- Bundled Python 3.12 interpreter
- vLLM-MLX inference engine
- All Python dependencies (MLX, transformers, etc.)
- Electron app shell

## Getting Started

### 1. Launch vMLX

The app starts directly — no setup wizard needed (Python is bundled).

### 2. Add Model Directories

Click **New Session** → the model picker scans default directories:
- `~/.lmstudio/models/`
- `~/.cache/huggingface/hub/`

To add custom directories, click the folder icon in the model picker.

### 3. Create a Session

1. Select a model from the list
2. Configure server settings (defaults work well for most cases)
3. Click **Launch** — the server starts and you enter the chat interface

### 4. Chat

- Type messages and get streaming responses
- Use **Chat Settings** (gear icon) to adjust temperature, system prompt, etc.
- Use **Server Settings** (gear icon) to adjust server parameters (requires restart)
- Chat history is saved per model and persists across sessions

## Model Recommendations

For best results on Apple Silicon:

| RAM | Recommended Models |
|-----|-------------------|
| 8GB | 1B-3B models (4-bit quantized) |
| 16GB | 7B-8B models (4-bit) or 3B models (8-bit) |
| 32GB | 14B-32B models (4-bit) or 8B models (8-bit) |
| 64GB | 70B models (4-bit) or 32B models (8-bit) |
| 96GB+ | 70B models (8-bit) or larger |

Look for models on Hugging Face under `mlx-community` — they're pre-converted for MLX.

## Troubleshooting

### App won't open
- See "First Launch — Gatekeeper Warning" above
- Make sure you're on Apple Silicon (check Apple menu → About This Mac → Chip)

### Model won't load / out of memory
- Close other apps to free memory
- Try a smaller or more quantized model (4-bit instead of 8-bit)
- Check Activity Monitor → Memory tab for available memory

### Server crashes during generation
- Reduce **Prefill Batch Size** in Server Settings (try 256 or 128)
- Reduce **Max Tokens** to limit generation length
- Some large models need **Paged KV Cache** enabled (auto-detected for most)

### Chat shows garbled output or leaked tokens
- Go to Server Settings → Tool Integration
- Verify the **Reasoning Parser** matches your model:
  - Qwen3 / StepFun / MiniMax: `qwen3`
  - DeepSeek R1: `deepseek_r1`
  - GLM-4.7 / GLM-4.7 Flash: `openai_gptoss` (GPT-OSS / Harmony)
  - Most other models: `None` or `Auto-detect`
- Set to **Auto-detect** if unsure — it reads config.json

### Slow first response
- First message is always slower (prompt processing + model warmup)
- Enable **Prefix Cache** in Server Settings for faster follow-up messages
- Prefix cache is ON by default

### Model repeats itself in a loop
- Lower the **Temperature** in Chat Settings (try 0.6-0.7)
- This is model behavior, not a vMLX bug — some models are prone to repetition

## Updating

When a new version of vMLX is available:
1. Quit vMLX (Cmd+Q)
2. Replace `/Applications/vMLX.app` with the new version
3. Relaunch — your chat history and settings are preserved

Chat history and settings are stored in:
```
~/Library/Application Support/vllm-mlx-panel/
```

## Support

For issues, visit: https://github.com/ASquare04/vllm-mlx/issues
