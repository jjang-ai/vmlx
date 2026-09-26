"""CPU-only persisted-state identity checks; no model or MLX initialization."""
import importlib.util
from pathlib import Path
import shutil

SOURCE = Path(__file__).parents[1] / "vmlx_engine/jangtq2/runtime_identity.py"


def load(path):
    spec = importlib.util.spec_from_file_location("isolated_jangtq2_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installed_sources_change_identity_without_git(tmp_path, monkeypatch):
    monkeypatch.delenv("JANGTQ2_DECODE_ROT", raising=False)
    monkeypatch.delenv("JANGTQ2_PREFILL", raising=False)
    shutil.copyfile(SOURCE, tmp_path / SOURCE.name)
    kernel = tmp_path / "kernels.py"
    kernel.write_text("kernel_v1")
    module = load(tmp_path / SOURCE.name)
    first = module.runtime_identity()
    kernel.write_text("kernel_v2")
    assert module.runtime_identity() != first


def test_routes_are_separate_and_frozen(monkeypatch):
    monkeypatch.setenv("JANGTQ2_DECODE_ROT", "host")
    monkeypatch.setenv("JANGTQ2_PREFILL", "steel")
    steel = load(SOURCE)
    first = steel.runtime_identity()
    monkeypatch.setenv("JANGTQ2_PREFILL", "nax")
    assert steel.runtime_identity() == first
    assert load(SOURCE).runtime_identity() != first
    monkeypatch.setenv("JANGTQ2_DECODE_ROT", "kernel")
    assert load(SOURCE).runtime_identity() != first


def test_invalid_route_rejected(monkeypatch):
    import pytest
    monkeypatch.setenv("JANGTQ2_DECODE_ROT", "typo")
    with pytest.raises(ValueError, match="DECODE_ROT"):
        load(SOURCE)
