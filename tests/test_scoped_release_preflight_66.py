from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scoped_release_preflight_66_exists_and_targets_current_version():
    script = ROOT / "panel/scripts/scoped-release-preflight-66.py"
    assert script.exists()
    source = script.read_text(encoding="utf-8")
    assert "1.5.66 MM3 + Gemma 4 compatibility gate" in source
    assert 'panel_version == "1.5.66"' in source
    assert 'lock_version == "1.5.66"' in source
    assert 'lock_root_version == "1.5.66"' in source
    assert 'version = "1.5.66"' in source
    assert '__version__ = "1.5.66"' in source
    assert '"scope": "1.5.66-mm3-gemma4"' in source
    assert "current-scoped-release-preflight-66.json" in source


def test_release_dmg_build_routes_1566_mm3_gemma_scope_to_66_preflight():
    source = (ROOT / "panel/scripts/build-release-dmgs.sh").read_text(
        encoding="utf-8"
    )
    assert 'if [[ "$VERSION" == "1.5.66" ]]' in source
    assert '"panel/scripts/scoped-release-preflight-66.py"' in source
