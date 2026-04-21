#!/bin/bash
# vMLX Mission Init Script
# Idempotent environment setup

set -e

echo "=== vMLX Mission Init ==="

# Verify Swift is available
if ! command -v swift &> /dev/null; then
    echo "ERROR: Swift not found in PATH"
    exit 1
fi

SWIFT_VERSION=$(swift --version | head -1)
echo "Swift: $SWIFT_VERSION"

# Verify lm-eval is available
if [ -f "$HOME/.local/pipx/venvs/lm-eval/bin/lm-eval" ]; then
    LM_EVAL_VERSION=$($HOME/.local/pipx/venvs/lm-eval/bin/lm-eval --version 2>/dev/null || echo "unknown")
    echo "lm-eval: $LM_EVAL_VERSION"
else
    echo "WARNING: lm-eval not found at ~/.local/pipx/venvs/lm-eval/bin/lm-eval"
fi

# Verify openai package in lm-eval venv
if $HOME/.local/pipx/venvs/lm-eval/bin/python -c "import openai" 2>/dev/null; then
    OPENAI_VERSION=$($HOME/.local/pipx/venvs/lm-eval/bin/python -c "import openai; print(openai.__version__)")
    echo "openai (lm-eval venv): $OPENAI_VERSION"
else
    echo "WARNING: openai package not found in lm-eval venv"
fi

# Check for port conflicts
if lsof -ti :8080 &>/dev/null; then
    echo "WARNING: Port 8080 is in use"
fi

echo "=== Init Complete ==="
