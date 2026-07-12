"""Explicit remote-code trust contracts for JANG conversion smoke tests."""

import argparse
import importlib
from types import SimpleNamespace

import pytest

from vmlx_engine.commands import convert as convert_mod


class _Tokenizer:
    def encode(self, _text):
        return [1]

    def decode(self, _tokens):
        return " OK"


@pytest.mark.parametrize(
    ("trust_remote_code", "expected_kwargs"),
    [
        (False, {}),
        (
            True,
            {"tokenizer_config_extra": {"trust_remote_code": True}},
        ),
    ],
)
def test_jang_smoke_forwards_only_explicit_trust(
    monkeypatch,
    trust_remote_code,
    expected_kwargs,
):
    from vmlx_engine.utils import jang_loader

    calls = []
    mlx_generate = importlib.import_module("mlx_lm.generate")
    mlx_sample = importlib.import_module("mlx_lm.sample_utils")

    def fake_load(model_path, **kwargs):
        calls.append((model_path, kwargs))
        return object(), _Tokenizer()

    monkeypatch.setattr(jang_loader, "load_jang_model", fake_load)
    monkeypatch.setattr(mlx_sample, "make_sampler", lambda **_kwargs: object())
    monkeypatch.setattr(
        mlx_generate,
        "generate_step",
        lambda **_kwargs: iter([(2, None)]),
    )

    success, message = convert_mod._jang_smoke_test(
        "/models/converted-jang",
        trust_remote_code=trust_remote_code,
    )

    assert success is True
    assert "capital of France" in message
    assert calls == [("/models/converted-jang", expected_kwargs)]


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_jang_convert_command_preserves_trust_choice(
    tmp_path,
    monkeypatch,
    trust_remote_code,
):
    import vmlx_engine.utils.model_inspector as inspector

    jang_convert = importlib.import_module("jang_tools.convert")

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"llama"}')
    output = tmp_path / "output"
    observed = {}

    monkeypatch.setattr(inspector, "resolve_model_path", lambda _model: str(source))
    monkeypatch.setattr(
        inspector,
        "inspect_model",
        lambda _path: SimpleNamespace(param_count_billions=0.5),
    )
    monkeypatch.setattr(inspector, "available_memory_gb", lambda: 128.0)
    monkeypatch.setattr(inspector, "total_memory_gb", lambda: 128.0)

    def fake_conversion(**_kwargs):
        output.mkdir(exist_ok=True)
        (output / "model.safetensors").write_bytes(b"converted")
        return {"actual_bits": 6.0, "total_weight_gb": 0.1}

    monkeypatch.setattr(jang_convert, "convert_model", fake_conversion)

    def fake_smoke(_path, *, trust_remote_code=False):
        observed["trust_remote_code"] = trust_remote_code
        return True, "ok"

    monkeypatch.setattr(convert_mod, "_jang_smoke_test", fake_smoke)

    convert_mod._jang_convert_command(
        argparse.Namespace(
            model=str(source),
            output=str(output),
            force=False,
            jang_profile="JANG_6M",
            jang_method="rtn",
            group_size=64,
            calibration_method="weights",
            imatrix_path=None,
            use_awq=False,
            awq_alpha=0.25,
            skip_verify=False,
            trust_remote_code=trust_remote_code,
        )
    )

    assert observed == {"trust_remote_code": trust_remote_code}


def test_jang_loader_routes_tokenizer_config_to_v2(tmp_path, monkeypatch):
    from vmlx_engine.utils import jang_loader

    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v4"}')
    (tmp_path / "jang_config.json").write_text(
        '{"version":2,"weight_format":"affine","quantization":{}}'
    )
    expected = (object(), object())
    observed = {}

    monkeypatch.setattr(jang_loader, "_is_v2_model", lambda _path: True)

    def fake_v2(_path, _config, **kwargs):
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(jang_loader, "_load_jang_v2", fake_v2)
    monkeypatch.setattr(
        jang_loader,
        "_ensure_jang_family_runtime_supported",
        lambda _path, _config: None,
    )

    result = jang_loader.load_jang_model(
        tmp_path,
        skip_eval=True,
        tokenizer_config_extra={"trust_remote_code": True},
    )

    assert result is expected
    assert observed["tokenizer_config_extra"] == {"trust_remote_code": True}
