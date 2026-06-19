import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_preflight_module():
    script = ROOT / "panel/scripts/scoped-release-preflight-66.py"
    spec = importlib.util.spec_from_file_location("scoped_release_preflight_66", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scoped_release_preflight_66_exists_and_targets_current_version():
    script = ROOT / "panel/scripts/scoped-release-preflight-66.py"
    assert script.exists()
    source = script.read_text(encoding="utf-8")
    assert "1.5.66 MM3 + Gemma 4 compatibility gate" in source
    assert "--expected-version" in source
    assert 'panel_version == expected_version' in source
    assert 'lock_version == expected_version' in source
    assert 'lock_root_version == expected_version' in source
    assert 'f\'version = "{expected_version}"\'' in source
    assert 'f\'__version__ = "{expected_version}"\'' in source
    assert '"scope": f"{args.expected_version}-mm3-gemma4"' in source
    assert "current-scoped-release-preflight-66.json" in source


def test_release_dmg_build_routes_1566_and_1567_mm3_gemma_scope_to_strict_preflight():
    source = (ROOT / "panel/scripts/build-release-dmgs.sh").read_text(
        encoding="utf-8"
    )
    assert 'if [[ "$VERSION" == "1.5.66" || "$VERSION" == "1.5.67" ]]' in source
    assert '"panel/scripts/scoped-release-preflight-66.py"' in source
    assert '--expected-version "$VERSION"' in source


def test_gemma_release_preflight_requires_serving_generation_config(tmp_path):
    module = _load_preflight_module()
    model_dir = tmp_path / "gemma4"
    model_dir.mkdir()

    (model_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "do_sample": True,
                "temperature": 1.0,
                "top_k": 64,
                "top_p": 0.95,
                "eos_token_id": [1, 106, 50],
            }
        ),
        encoding="utf-8",
    )
    failures = []
    module.validate_gemma_generation_config(
        "gemma4-e2b-jang4m-vl-current-64", model_dir, failures
    )
    assert failures == []

    (model_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "eos_token_id": [1, 106, 50],
                "max_new_tokens": 256,
            }
        ),
        encoding="utf-8",
    )
    failures = []
    module.validate_gemma_generation_config(
        "gemma4-e2b-jang4m-vl-current-64", model_dir, failures
    )
    assert any("do_sample" in failure for failure in failures)
    assert any("temperature" in failure for failure in failures)
    assert any("top_k" in failure for failure in failures)
    assert any("top_p" in failure for failure in failures)


def test_gemma_release_preflight_requires_multitoken_eos(tmp_path):
    module = _load_preflight_module()
    model_dir = tmp_path / "gemma4"
    model_dir.mkdir()
    (model_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "do_sample": True,
                "temperature": 1.0,
                "top_k": 64,
                "top_p": 0.95,
                "eos_token_id": 1,
            }
        ),
        encoding="utf-8",
    )
    failures = []
    module.validate_gemma_generation_config(
        "gemma4-e2b-jang4m-vl-current-64", model_dir, failures
    )
    assert any("eos_token_id" in failure and "[1, 106, 50]" in failure for failure in failures)
