import importlib.util
from pathlib import Path


def _load_gate_module():
    path = Path("panel/scripts/release-gate-python-app.py").resolve()
    spec = importlib.util.spec_from_file_location("release_gate_python_app", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_release_gate_loop_detector_catches_word_loop():
    gate = _load_gate_module()
    assert gate.obvious_loop("state " * 80)


def test_release_gate_loop_detector_catches_no_space_cjk_phrase_loop():
    gate = _load_gate_module()
    text = "音苷苷和音诺族的对策" * 80
    assert gate.obvious_loop(text)


def test_release_gate_loop_detector_catches_emoji_loop():
    gate = _load_gate_module()
    assert gate.obvious_loop("👀" * 200)


def test_release_gate_loop_detector_allows_short_clean_answer():
    gate = _load_gate_module()
    assert not gate.obvious_loop("Paris is the capital of France.")
