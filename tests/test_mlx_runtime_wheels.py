"""Wheel provenance must fail closed across upstream MLX upgrades."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "panel/scripts/mlx-runtime.py"
    loader = importlib.util.spec_from_file_location("mlx_runtime_build", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    spec = {
        "version": "0.99.0", "source": {"kind": "pypi"},
        "platforms": {"macosx_14_0_arm64": {"deployment_target": "14.0"}},
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    monkeypatch.setattr(module, "SPEC_PATH", spec_path)
    return module, spec


def write_wheels(directory, spec, platform="macosx_14_0_arm64"):
    directory.mkdir(exist_ok=True)
    for name, tag in [("mlx", "cp312-cp312"), ("mlx_metal", "py3-none")]:
        prefix = f"{name}-{spec['version']}"
        with zipfile.ZipFile(directory / f"{prefix}-{tag}-{platform}.whl", "w") as archive:
            archive.writestr(f"{prefix}.dist-info/METADATA", f"Name: {name.replace('_', '-')}\nVersion: {spec['version']}\n")
            archive.writestr(f"{prefix}.dist-info/WHEEL", f"Tag: {tag}-{platform}\n")


def test_official_release_uses_exact_pair_and_attests_hashes(runtime, tmp_path, monkeypatch):
    module, spec = runtime
    output = tmp_path / "wheels"
    calls = []

    def download(args, **kwargs):
        calls.append(args)
        write_wheels(output, spec)

    monkeypatch.setattr(module, "run", download)
    module.prepare(output, spec, "macosx_14_0_arm64", None)
    assert [args[-1] for args in calls] == ["mlx==0.99.0", "mlx-metal==0.99.0"]
    assert all("--no-deps" in args for args in calls)
    assert len(module.verify(output, spec, "macosx_14_0_arm64")["wheels"]) == 2
    with next(output.glob("mlx-*.whl")).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="provenance mismatch: wheels"):
        module.verify(output, spec, "macosx_14_0_arm64")


def test_wheel_pair_rejects_wrong_platform(runtime, tmp_path):
    module, spec = runtime
    write_wheels(tmp_path, spec, "macosx_26_0_arm64")
    with pytest.raises(ValueError, match="Expected one mlx wheel"):
        module.wheel_records(tmp_path, spec, "macosx_14_0_arm64")


def test_new_source_spec_cannot_reuse_old_attestation(runtime, tmp_path, monkeypatch):
    module, spec = runtime
    output = tmp_path / "wheels"
    monkeypatch.setattr(module, "run", lambda *a, **kw: write_wheels(output, spec))
    module.prepare(output, spec, "macosx_14_0_arm64", None)
    module.SPEC_PATH.write_text(json.dumps({**spec, "upstream_prs": [4734]}))
    with pytest.raises(ValueError, match="runtime_spec_sha256"):
        module.prepare(output, spec, "macosx_14_0_arm64", None)
