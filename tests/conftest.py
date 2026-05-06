# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


EXTERNAL_MODEL_TEST_FILES = {
    "test_batching_deterministic.py",
    "test_continuous_batching.py",
    "test_emoji_comprehensive.py",
    "test_model_registry.py",
    "test_streaming_detokenizer.py",
}


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--server-url",
        action="store",
        default="http://localhost:8000",
        help="URL of the vmlx-engine server for integration tests",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that require model loading",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (requires model loading)"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires running server)",
    )


def pytest_sessionstart(session):
    """Expose the current checkout at legacy source-pin paths.

    Several regression guards were written against the packaged build root
    used during v1.3.66 triage. In a normal checkout those absolute paths do
    not exist, but the guards are intended to inspect this source tree.
    """
    repo_root = Path(__file__).resolve().parents[1]
    for legacy_root in (
        Path("/private/tmp/vmlx-1.3.66-build"),
        Path("/tmp/vmlx-1.3.66-build"),
    ):
        try:
            if legacy_root.exists() or legacy_root.is_symlink():
                continue
            legacy_root.parent.mkdir(parents=True, exist_ok=True)
            legacy_root.symlink_to(repo_root, target_is_directory=True)
        except OSError:
            # The individual source-pin tests will report the missing path.
            pass


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is passed."""
    for item in items:
        if Path(str(item.fspath)).name in EXTERNAL_MODEL_TEST_FILES:
            item.add_marker(pytest.mark.slow)

    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip integration tests unless server URL is explicitly provided
    skip_integration = pytest.mark.skip(reason="Integration tests require --server-url")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def server_url(request):
    """Get server URL from command line."""
    return request.config.getoption("--server-url")
