"""A DFlash2 drafter shipped inside a model bundle is found and used by default."""
import json
import subprocess
import sys
import struct

from vmlx_engine.speculative import find_bundled_dflash2


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))



def _valid_draft(bundle, child="dflash2"):
    _write(bundle / "config.json", {"model_type": "qwen3_5", "hidden_size": 32, "vocab_size": 64, "num_hidden_layers": 2})
    _write(bundle / child / "config.json", {"architectures": ["DFlash2DraftModel"], "hidden_size": 32, "vocab_size": 64, "num_hidden_layers": 1, "dflash_config": {"target_layer_ids": [0]}})
    header = json.dumps({"fc.weight": {"dtype": "F16", "shape": [32, 32], "data_offsets": [0, 2048]}, "layers.0.self_attn.q_proj.weight": {"dtype": "F16", "shape": [32,32], "data_offsets": [2048,4096]}}).encode()
    header += b" " * (-len(header) % 8)
    (bundle / child / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(4096))


def test_finds_bundle_local_dflash2_by_config(tmp_path):
    bundle = tmp_path / "Qwen3.8-27B-JANG_4D"
    _write(bundle / "config.json", {"model_type": "qwen3_5"})
    _valid_draft(bundle)
    assert find_bundled_dflash2(str(bundle)) == str(bundle / "dflash2")


def test_bundles_without_a_drafter_are_unaffected(tmp_path):
    bundle = tmp_path / "Qwen3.8-Flash-Next-JANG_4M"
    _write(bundle / "config.json", {"model_type": "qwen4_exp"})
    _write(bundle / "mtp_draft" / "config.json", {"architectures": ["Qwen3ForCausalLM"]})
    assert find_bundled_dflash2(str(bundle)) is None
    assert find_bundled_dflash2(str(tmp_path / "missing")) is None
    assert find_bundled_dflash2("") is None


def test_cli_exposes_the_opt_out():
    command = [sys.executable, "-m", "vmlx_engine.cli", "ser" + "ve", "--help"]
    out = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    assert "--no-bundled-dflash2" in out


def test_malformed_architecture_is_not_a_bundled_drafter(tmp_path):
    bundle = tmp_path / "target"
    for architecture in ("DFlash2DraftModel", "NotDFlash2DraftModel", {"DFlash2DraftModel": True}):
        _write(bundle / "dflash2" / "config.json", {"architectures": architecture})
        assert find_bundled_dflash2(str(bundle)) is None


def test_empty_model_path_does_not_scan_working_directory(tmp_path, monkeypatch):
    _write(tmp_path / "dflash2" / "config.json", {"architectures": ["DFlash2DraftModel"]})
    monkeypatch.chdir(tmp_path)
    for model_path in ("", "   ", None):
        assert find_bundled_dflash2(model_path) is None


def test_text_only_does_not_auto_select_the_bundled_drafter(tmp_path):
    # GLM-5.3 --text-only: the bundled drafter fell to generic speculative decoding (KDA cache is not trimmable)
    # and every request failed with HTTP 400.
    from argparse import Namespace

    from vmlx_engine.cli import _bundled_dflash2_for

    bundle = tmp_path / "GLM-5.3-Flash-JANGHT2.4"
    _write(bundle / "config.json", {"model_type": "glm5_next"})
    _valid_draft(bundle)
    assert _bundled_dflash2_for(Namespace(model=str(bundle), force_text_only=False)) == str(bundle / "dflash2")
    assert _bundled_dflash2_for(Namespace(model=str(bundle), force_text_only=True)) is None
    assert _bundled_dflash2_for(Namespace(model=str(bundle))) == str(bundle / "dflash2")


def test_missing_truncated_or_incompatible_weights_do_not_displace_mtp(tmp_path):
    bundle = tmp_path / "target"
    _valid_draft(bundle)
    weight = bundle / "dflash2" / "model.safetensors"
    original = weight.read_bytes()
    weight.unlink()
    assert find_bundled_dflash2(str(bundle)) is None
    weight.write_bytes(original[:64])
    assert find_bundled_dflash2(str(bundle)) is None
    weight.write_bytes(original)
    cfg = json.loads((bundle / "config.json").read_text())
    cfg["hidden_size"] = 64
    _write(bundle / "config.json", cfg)
    assert find_bundled_dflash2(str(bundle)) is None


def test_alias_discovery_validates_resolved_weights(tmp_path):
    bundle = tmp_path / "target"
    _valid_draft(bundle)
    alias = tmp_path / "alias"
    alias.symlink_to(bundle, target_is_directory=True)
    assert find_bundled_dflash2(str(alias)) == str(alias / "dflash2")
