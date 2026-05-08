import importlib.util
import json
import os
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


def test_electron_builder_runs_bundled_python_gate_before_packaging():
    pkg = json.loads(Path("panel/package.json").read_text())
    hook = pkg["build"].get("beforePack")
    assert hook == "scripts/electron-builder-before-pack.cjs"

    hook_src = Path("panel/scripts/electron-builder-before-pack.cjs").read_text()
    assert "verify-bundled-python.sh" in hook_src
    assert "electron-vite" in hook_src
    assert "VMLX_BEFORE_PACK_SKIP_VITE" in hook_src
    assert "require.main === module" in hook_src


def test_electron_builder_before_pack_hook_runs_verifier_in_direct_smoke(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    verifier = scripts / "verify-bundled-python.sh"
    verifier.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho ok > \"$PWD/verify-ran\"\n")
    verifier.chmod(0o755)

    env = dict(os.environ)
    env["VMLX_BEFORE_PACK_SKIP_VITE"] = "1"
    proc = subprocess.run(
        ["node", str(Path("panel/scripts/electron-builder-before-pack.cjs").resolve())],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "verify-ran").read_text() == "ok\n"
    assert "skipped electron-vite build" in proc.stdout


def test_electron_builder_before_pack_hook_rejects_skip_vite_in_pack_context(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    verifier = scripts / "verify-bundled-python.sh"
    verifier.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho ok > \"$PWD/verify-ran\"\n")
    verifier.chmod(0o755)

    hook_path = Path("panel/scripts/electron-builder-before-pack.cjs").resolve()
    js = (
        "process.env.VMLX_BEFORE_PACK_SKIP_VITE = '1';"
        f"const hook = require({json.dumps(str(hook_path))});"
        f"hook({{packager: {{projectDir: {json.dumps(str(tmp_path))}}}}})"
        ".then(() => process.exit(0))"
        ".catch((err) => { console.error(err.message); process.exit(3); });"
    )
    proc = subprocess.run(
        ["node", "-e", js],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 3
    assert (tmp_path / "verify-ran").read_text() == "ok\n"
    assert "only allowed for direct hook smoke tests" in proc.stderr
