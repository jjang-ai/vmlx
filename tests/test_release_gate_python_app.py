import importlib.util
import subprocess
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


class _FakeGate:
    def __init__(self, stdout: str):
        self.stdout = stdout
        self.records = []
        self.run_cmd = None

    def run(self, name, cmd, **kwargs):
        self.run_cmd = cmd
        self.records.append((name, "RUN", kwargs))
        return subprocess.CompletedProcess(cmd, 0, self.stdout, "")

    def record(self, name, status, detail=""):
        self.records.append((name, status, detail))


def test_packaged_bundled_version_parity_passes_when_import_version_matches():
    gate_module = _load_gate_module()
    gate = _FakeGate("import ok\n1.5.25\n")

    gate_module.check_packaged_bundled_import_version(
        gate, Path("/app/python3"), "1.5.25", "1.5.25"
    )

    assert gate.records[-1] == (
        "packaged bundled version",
        "PASS",
        "app=1.5.25, bundled=1.5.25, expected=1.5.25",
    )
    assert "mflux" in " ".join(gate.run_cmd)


def test_packaged_bundled_version_parity_fails_on_stale_bundled_engine():
    gate_module = _load_gate_module()
    gate = _FakeGate("1.5.23\n")

    gate_module.check_packaged_bundled_import_version(
        gate, Path("/app/python3"), "1.5.25", "1.5.25"
    )

    assert gate.records[-1] == (
        "packaged bundled version",
        "FAIL",
        "app=1.5.25, bundled=1.5.23, expected=1.5.25",
    )
