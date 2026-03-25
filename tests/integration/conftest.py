"""
Integration test fixtures — manages a live vmlx-engine server process.

Usage:
    pytest tests/integration/ --model <model-path> [-v]

The server starts once per session and is shared across all tests.
"""

import os
import subprocess
import sys
import time

import pytest
import requests


def pytest_addoption(parser):
    parser.addoption("--model", action="store", default=None, help="Model path for integration tests")
    parser.addoption("--port", action="store", default="9876", help="Server port")
    parser.addoption("--ci", action="store_true", default=False, help="CI mode (fail fast)")


@pytest.fixture(scope="session")
def server(request):
    """Start a vmlx-engine server for the test session."""
    model = request.config.getoption("--model")
    port = int(request.config.getoption("--port"))

    if not model:
        pytest.skip("No --model specified, skipping integration tests")

    base_url = f"http://127.0.0.1:{port}"

    # Check if server is already running on this port
    try:
        r = requests.get(f"{base_url}/health", timeout=2)
        if r.ok:
            yield {"base_url": base_url, "model": model, "port": port, "managed": False}
            return
    except Exception:
        pass

    # Start server
    cmd = [sys.executable, "-m", "vmlx_engine.cli", "serve",
           "--model", model, "--port", str(port), "--host", "127.0.0.1"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for server to be ready
    for _ in range(120):
        try:
            r = requests.get(f"{base_url}/health", timeout=1)
            if r.ok:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        proc.kill()
        pytest.fail(f"Server failed to start within 120s. Model: {model}")

    yield {"base_url": base_url, "model": model, "port": port, "managed": True, "proc": proc}

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def base_url(server):
    return server["base_url"]


@pytest.fixture
def model_name(server):
    return server["model"]
