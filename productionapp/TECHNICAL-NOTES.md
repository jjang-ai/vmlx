# vMLX — Technical Notes for Distribution

## Build Info

- **Architecture**: arm64 (Apple Silicon only)
- **Electron**: 28.3.3
- **Bundled Python**: 3.12.8 (python-build-standalone, relocatable)
- **Code Signing**: Ad-hoc (no Apple Developer ID) — users must bypass Gatekeeper
- **App Size**: ~1.4GB uncompressed, ~500-600MB compressed (DMG/zip)

## What's Inside the .app

```
vMLX.app/Contents/
├── MacOS/vMLX                          ← Electron binary (arm64)
├── Resources/
│   ├── app.asar                        ← Electron app bundle (~36MB)
│   ├── bundled-python/                 ← Self-contained Python 3.12 (~1.1GB)
│   │   └── python/
│   │       ├── bin/python3
│   │       └── lib/python3.12/site-packages/
│   │           ├── vllm_mlx/           ← Engine (installed from source)
│   │           ├── mlx/                ← Apple MLX framework
│   │           ├── transformers/       ← HuggingFace transformers
│   │           └── ...                 ← All other deps
│   └── vllm-mlx-source/               ← Engine source for auto-updates
│       ├── pyproject.toml
│       └── vllm_mlx/
└── Frameworks/                         ← Electron frameworks
```

## Engine Auto-Update Mechanism

On each launch, the app:
1. Reads installed version: `python3 -c "import vllm_mlx; print(vllm_mlx.__version__)"`
2. Reads source version: `vllm-mlx-source/pyproject.toml` → `version = "x.y.z"`
3. If different: runs `pip install --force-reinstall --no-deps <source-path>` (~5s)

This means engine updates are delivered by rebuilding the app with updated source.
Python dependency updates require rebuilding bundled-python (rare).

## Known Limitations

### Not Code-Signed
- Users will see Gatekeeper warning on first launch
- To properly sign: need Apple Developer ID ($99/year) + `electron-builder` signing config
- Notarization also required for smooth distribution outside App Store

### No Auto-Update
- No Sparkle/electron-updater configured
- Users must manually replace the .app
- Chat history/settings persist across updates (stored in ~/Library/Application Support/)

### arm64 Only
- No Intel (x86_64) support — MLX requires Apple Silicon
- Universal binary not possible since MLX has no Intel backend

### Unsigned Python Binaries
- The bundled Python and its .so/.dylib files are ad-hoc signed
- Some enterprise Macs with strict policies may block execution
- Workaround: `xattr -cr /Applications/vMLX.app` removes quarantine

## Model Compatibility

### Reasoning Parsers (server-side extraction)

| Parser | Models | Protocol |
|--------|--------|----------|
| `qwen3` | Qwen3, Qwen3-Coder, QwQ-32B, StepFun, MiniMax-M2.5 | `<think>...</think>` (strict) |
| `deepseek_r1` | DeepSeek-R1, R1-Distill, R1-0528 | `<think>...</think>` (lenient) |
| `openai_gptoss` | GLM-4.7, GLM-4.7 Flash, GLM-Z1, GPT-OSS | Harmony protocol channels |

### Tool Call Parsers

| Parser | Models |
|--------|--------|
| `qwen` | Qwen2, Qwen2.5, Qwen3, QwQ |
| `llama` | Llama 3.x, Llama 4 |
| `mistral` | Mistral, Mixtral, Pixtral, Codestral, Devstral |
| `hermes` | Hermes 2/3/4 |
| `deepseek` | DeepSeek-V2, V3 (native only) |
| `glm47` | GLM-4, GLM-4.7, GLM-Z1 |
| `minimax` | MiniMax M1, M2, M2.5 |
| `granite` | IBM Granite 3.x |
| `functionary` | MeetKai Functionary |
| `xlam` | Salesforce xLAM |
| `kimi` | Kimi-K2, Moonshot |
| `step3p5` | StepFun Step-3.5 |
| `nemotron` | NVIDIA Nemotron |

### Cache Types (auto-detected)

| Type | Models | Notes |
|------|--------|-------|
| KV | Most transformers | Standard attention cache |
| Mamba | Falcon-Mamba, RWKV, Qwen3-Next | SSM state, paged cache auto-enabled |
| Hybrid | Nemotron, Jamba | Mixed KV+Mamba, paged cache auto-enabled |

### KV Cache Quantization

Not yet exposed. mlx-lm supports `kv_bits` (4/8) in `generate_step()` but not in
`BatchGenerator` (continuous batching). Will be added when upstream supports it.

## Data Locations

| Data | Path |
|------|------|
| Database (chats, sessions, settings) | `~/Library/Application Support/vllm-mlx-panel/chats.db` |
| Model weights | User-configured dirs (default: `~/.lmstudio/models/`, `~/.cache/huggingface/hub/`) |
| Server logs | In-memory (visible in session loading screen) |

## Building a New Release

```bash
cd /Users/eric/mlx/vllm-mlx/panel

# 1. Bundle Python (only when deps change, ~15 min)
bash scripts/bundle-python.sh

# 2. Build + package
npm run build && npx electron-builder --mac --dir

# 3. Copy to productionapp
rm -rf ../productionapp/vMLX.app
cp -R release/mac-arm64/vMLX.app ../productionapp/vMLX.app

# 4. Optional: create DMG
npx electron-builder --mac dmg
cp release/*.dmg ../productionapp/
```

## CRITICAL: Proprietary Model Restrictions

**MiniMax Prism Pro** model weights are proprietary. NEVER distribute, bundle, or
include MiniMax Prism Pro weights in any release. All other open-weight models are fine.
