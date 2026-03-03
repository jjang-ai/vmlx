# Contributing

We welcome contributions to vllm-mlx!

## Getting Started

```bash
# Clone the repository
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx

# Install with dev dependencies
pip install -e ".[dev]"
```

## Development Workflow

### Running Tests

**Important**: Always use the project's `.venv` Python — system Python won't have the dependencies.

```bash
# Run ALL engine tests (~1000+ tests)
.venv/bin/python -m pytest tests/ -v

# Critical test suites to run before any build:
.venv/bin/python -m pytest tests/test_reasoning_tool_interaction.py -v  # 61 tests: reasoning + tools
.venv/bin/python -m pytest tests/test_tool_fallback_injection.py -v     # 4 tests: template fallback
.venv/bin/python -m pytest tests/test_tool_format.py -v                 # 54+ tests: tool formats

# Panel tests (TypeScript)
cd panel && npx vitest run   # 80+ tests
```

### Pre-Build Checklist

Before every production build, run this:

```bash
# 1. Engine tests
.venv/bin/python -m pytest tests/ -v 2>&1 | tail -5

# 2. Panel tests
cd panel && npx vitest run 2>&1 | tail -5

# 3. Build and install
cd panel && npm run build && npm run dist

# 4. Deploy
killall vMLX 2>/dev/null || true
rm -rf /Applications/vMLX.app
cp -R release/mac-arm64/vMLX.app /Applications/
xattr -cr /Applications/vMLX.app
open /Applications/vMLX.app
```

See [Build, Test & Deploy](build-test-deploy.md) for complete details including feature cohesion matrix and dependency chain.

```bash
# Run with coverage
.venv/bin/python -m pytest --cov=vllm_mlx tests/
```

### Code Style

```bash
# Format code
black vllm_mlx/
isort vllm_mlx/

# Type checking
mypy vllm_mlx/
```

### Running Benchmarks

```bash
# LLM benchmark
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit

# Image benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Video benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --video
```

## Areas for Contribution

- **Bug fixes** - Fix issues and improve stability
- **Performance optimizations** - Improve inference speed
- **New features** - Add functionality
- **Documentation** - Improve docs and examples
- **Benchmarks** - Test on different Apple Silicon chips
- **Model support** - Test and add new models

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure they pass
5. Submit a pull request

## Code Structure

See [Architecture](architecture.md) for details on the codebase structure.

## Testing on Different Hardware

If you have access to different Apple Silicon chips (M1, M2, M3, M4), benchmark results are valuable:

```bash
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit --output results_m4.json
```

## Questions?

Open an issue at [GitHub Issues](https://github.com/waybarrios/vllm-mlx/issues).
