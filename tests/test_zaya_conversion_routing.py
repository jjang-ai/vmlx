import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def _write_zaya_source(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "zaya",
                "architectures": ["ZayaForCausalLM"],
                "hidden_size": 2048,
                "num_hidden_layers": 80,
            }
        )
    )
    return path


def test_zaya_model_dir_detection_accepts_model_type_and_architecture(tmp_path):
    from vmlx_engine.commands.convert import _is_zaya_model_dir

    model_dir = _write_zaya_source(tmp_path / "ZAYA1-8B")
    assert _is_zaya_model_dir(model_dir)

    cfg = json.loads((model_dir / "config.json").read_text())
    cfg.pop("model_type")
    (model_dir / "config.json").write_text(json.dumps(cfg))
    assert _is_zaya_model_dir(model_dir)


def test_zaya_jang_profile_maps_generic_jang_profile_to_tq_bits():
    from vmlx_engine.commands.convert import _zaya_jangtq_profile

    assert _zaya_jangtq_profile("JANG_4L", 4) == "JANGTQ4"
    assert _zaya_jangtq_profile("JANG_2S", 2) == "JANGTQ2"
    assert _zaya_jangtq_profile("JANGTQ4", 4) == "JANGTQ4"


def test_uniform_convert_routes_zaya_to_zaya_mxfp_converter(
    tmp_path, monkeypatch
):
    from vmlx_engine.commands import convert as convert_mod
    import vmlx_engine.utils.model_inspector as inspector

    model_dir = _write_zaya_source(tmp_path / "ZAYA1-8B")
    out_dir = tmp_path / "out"
    calls = {}

    monkeypatch.setattr(inspector, "resolve_model_path", lambda _m: str(model_dir))
    monkeypatch.setattr(
        inspector,
        "inspect_model",
        lambda _p: SimpleNamespace(
            architecture="ZayaForCausalLM",
            needs_latent_moe=False,
        ),
    )
    monkeypatch.setattr(inspector, "format_model_info", lambda _info: "ZAYA info")
    monkeypatch.setattr(convert_mod, "_preflight_check", lambda _info, _bits: None)
    monkeypatch.setattr(
        convert_mod,
        "_run_conversion",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("generic convert used")),
    )
    monkeypatch.setattr(
        convert_mod,
        "_run_zaya_converter_subprocess",
        lambda **kwargs: calls.setdefault("converter", kwargs),
    )
    monkeypatch.setattr(
        convert_mod,
        "_finish_zaya_conversion",
        lambda output_dir, skip_verify: calls.setdefault(
            "finish", (output_dir, skip_verify)
        ),
    )

    args = argparse.Namespace(
        model="Zyphra/ZAYA1-8B",
        output=str(out_dir),
        bits=4,
        group_size=32,
        mode="default",
        dtype=None,
        force=False,
        skip_verify=True,
        trust_remote_code=True,
        jang_profile=None,
    )
    convert_mod.convert_command(args)

    assert calls["converter"]["module"] == "jang_tools.convert_zaya_mxfp4"
    assert calls["converter"]["source"] == str(model_dir)
    assert calls["converter"]["output"] == out_dir
    assert calls["converter"]["extra_args"] == [
        "--bits",
        "4",
        "--group-size",
        "32",
    ]
    assert calls["finish"] == (out_dir, True)


def test_jang_convert_routes_zaya_to_zaya_jangtq_converter(tmp_path, monkeypatch):
    from vmlx_engine.commands import convert as convert_mod
    import vmlx_engine.utils.model_inspector as inspector

    model_dir = _write_zaya_source(tmp_path / "ZAYA1-8B")
    out_dir = tmp_path / "out"
    calls = {}

    monkeypatch.setattr(inspector, "resolve_model_path", lambda _m: str(model_dir))
    monkeypatch.setattr(
        inspector,
        "inspect_model",
        lambda _p: SimpleNamespace(param_count_billions=1.8),
    )
    monkeypatch.setattr(inspector, "available_memory_gb", lambda: 128.0)
    monkeypatch.setattr(inspector, "total_memory_gb", lambda: 128.0)
    monkeypatch.setattr(
        convert_mod,
        "_run_zaya_converter_subprocess",
        lambda **kwargs: calls.setdefault("converter", kwargs),
    )
    monkeypatch.setattr(
        convert_mod,
        "_finish_zaya_conversion",
        lambda output_dir, skip_verify: calls.setdefault(
            "finish", (output_dir, skip_verify)
        ),
    )

    args = argparse.Namespace(
        model="Zyphra/ZAYA1-8B",
        output=str(out_dir),
        force=False,
        jang_profile="JANG_4L",
        jang_method="rtn",
        group_size=32,
        calibration_method="weights",
        imatrix_path=None,
        use_awq=False,
        awq_alpha=0.25,
        skip_verify=True,
    )
    convert_mod._jang_convert_command(args)

    assert calls["converter"]["module"] == "jang_tools.convert_zaya_jangtq"
    assert calls["converter"]["source"] == str(model_dir)
    assert calls["converter"]["output"] == out_dir
    assert calls["converter"]["extra_args"] == [
        "JANGTQ4",
        "--group-size",
        "32",
    ]
    assert calls["finish"] == (out_dir, True)
