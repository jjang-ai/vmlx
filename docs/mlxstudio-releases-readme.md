<p align="center">
  <img src="https://vmlx.net/logos/png/wordmark-transparent-600x150.png" alt="MLX Studio" width="400">
</p>

<p align="center">
  <strong>The native macOS desktop app for local AI inference on Apple Silicon</strong>
</p>

<p align="center">
  <a href="https://github.com/jjang-ai/mlxstudio/releases/latest"><img src="https://img.shields.io/github/v/release/jjang-ai/mlxstudio?style=flat-square&label=Latest%20Release&color=blue" alt="Latest Release"></a>
  <a href="https://github.com/jjang-ai/mlxstudio/releases"><img src="https://img.shields.io/github/downloads/jjang-ai/mlxstudio/total?style=flat-square&label=Downloads&color=green" alt="Downloads"></a>
  <img src="https://img.shields.io/badge/Platform-macOS%20ARM64-lightgrey?style=flat-square&logo=apple" alt="Platform">
  <a href="https://github.com/jjang-ai/mlxstudio/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-orange?style=flat-square" alt="License"></a>
</p>

---

MLX Studio is a full-featured desktop application for running large language models, vision models, and image generation locally on your Mac. Built on [vMLX Engine](https://github.com/jjang-ai/vmlx) and Apple's [MLX](https://github.com/ml-explore/mlx) framework, it delivers GPU-accelerated inference with zero cloud dependencies.

## Screenshots

<table>
  <tr>
    <td align="center"><img src="assets/chat-tab.png" width="400"><br><b>Chat Interface</b></td>
    <td align="center"><img src="assets/image-tab.png" width="400"><br><b>Image Generation</b></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/tools-tab.png" width="400"><br><b>Developer Tools</b></td>
    <td align="center"><img src="assets/menu-bar.png" width="400"><br><b>Menu Bar Tray</b></td>
  </tr>
</table>

## Download

> **[Download the latest DMG from Releases](https://github.com/jjang-ai/mlxstudio/releases/latest)**

1. Download `MLXStudio-X.Y.Z-arm64.dmg` from the latest release
2. Open the DMG and drag **MLX Studio** to your Applications folder
3. Launch from Applications or Spotlight

All releases are signed and notarized by Apple for Gatekeeper compatibility.

## Quick Start

1. **Launch** MLX Studio from Applications
2. **Pick a model** -- browse and download from Hugging Face directly in the app
3. **Chat** -- start a conversation, attach images, or use tool calling
4. That's it. No Python setup, no terminal, no configuration files.

## Features

- **Chat** -- Multi-turn conversations with streaming, markdown rendering, and code highlighting
- **Vision** -- Send images to multimodal models (Qwen-VL, LLaVA, Pixtral, and more)
- **Reasoning** -- Native support for thinking/reasoning models (DeepSeek-R1, QwQ, GLM-Z1)
- **Tool Calling** -- Function calling with structured output for agent workflows
- **Image Generation** -- Text-to-image with MLX-powered diffusion models
- **Text-to-Speech** -- Local TTS via Kokoro with multiple voices
- **Speech-to-Text** -- Whisper-based transcription, fully on-device
- **OpenAI-Compatible API** -- Drop-in replacement server on `localhost` for any client
- **Menu Bar Mode** -- Runs quietly in the tray; always one click away
- **Developer Tools** -- Model inspection, conversion, quantization, and diagnostics
- **Session Management** -- Save, restore, and organize conversations with SQLite persistence
- **JANG Quantization** -- First-class support for [JANG mixed-precision quantization](https://github.com/jjang-ai/vmlx) formats

### Engine Highlights

- **Continuous Batching** with PagedAttention for efficient memory use
- **Speculative Decoding** for faster generation on supported models
- **KV Cache Quantization** to fit larger contexts in limited memory
- **Prefix Caching** for faster repeated prompts
- **Hybrid Model Support** (Mamba/SSM + Transformer architectures)

## System Requirements

| Requirement | Minimum |
|---|---|
| **macOS** | 14.0 Sonoma or later |
| **Chip** | Apple Silicon (M1, M2, M3, M4 -- any variant) |
| **RAM** | 8 GB (16 GB+ recommended for larger models) |
| **Disk** | ~500 MB for the app; models vary (1--50 GB each) |

## What's Included

MLX Studio bundles everything needed to run local AI:

| Mode | Description |
|---|---|
| **Chat** | Conversational UI with vision, reasoning, and tool support |
| **Server** | OpenAI-compatible API server (`/v1/chat/completions`, `/v1/responses`) |
| **Image** | Text-to-image generation with diffusion models |
| **Tools** | Model conversion, quantization, diagnostics, and inspection |
| **API** | Interactive API documentation and testing playground |

A bundled Python 3.12 environment with all dependencies is included -- no system Python or virtual environments required.

## Build from Source

If you prefer to build MLX Studio yourself:

```bash
# Clone the source repository
git clone https://github.com/jjang-ai/vmlx.git
cd vmlx

# Set up the Python engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Build the Electron app
cd panel
npm install
npm run build
npx electron-builder --mac --dir

# The app is in panel/release/mac-arm64/MLX Studio.app
```

For a distributable DMG:

```bash
npx electron-builder --mac dmg
```

## Also Available

| Resource | Link |
|---|---|
| **PyPI** (CLI / Python library) | `pip install vmlx-engine` -- [pypi.org/project/vmlx-engine](https://pypi.org/project/vmlx-engine/) |
| **Source Code** | [github.com/jjang-ai/vmlx](https://github.com/jjang-ai/vmlx) |
| **JANG Quantization** | Mixed-precision quantization for MLX -- [docs](https://github.com/jjang-ai/vmlx) |
| **Models on Hugging Face** | [huggingface.co/jjang-ai](https://huggingface.co/jjang-ai) |
| **Website** | [vmlx.net](https://vmlx.net) |

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Contact

**Jinho Jang** -- [eric@jangq.ai](mailto:eric@jangq.ai)

Built by [JANGQ AI](https://github.com/jjang-ai)
