"""Post-conversion smoke-test trust-remote-code contract."""

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
        (True, {"tokenizer_config": {"trust_remote_code": True}}),
    ],
)
def test_smoke_load_forwards_only_explicit_trust(
    monkeypatch, trust_remote_code, expected_kwargs
):
    calls = []
    mlx_lm = importlib.import_module("mlx_lm")
    mlx_generate = importlib.import_module("mlx_lm.generate")
    mlx_sample = importlib.import_module("mlx_lm.sample_utils")

    def fake_load(model_path, **kwargs):
        calls.append((model_path, kwargs))
        return object(), _Tokenizer()

    monkeypatch.setattr(mlx_lm, "load", fake_load)
    monkeypatch.setattr(mlx_sample, "make_sampler", lambda **_kwargs: object())
    monkeypatch.setattr(
        mlx_generate,
        "generate_step",
        lambda **_kwargs: iter([(2, None)]),
    )
    monkeypatch.setattr(
        "vmlx_engine.utils.nemotron_latent_moe.ensure_latent_moe_support",
        lambda _path: None,
    )

    success, message = convert_mod._smoke_test(
        "/models/converted",
        trust_remote_code=trust_remote_code,
    )

    assert success is True
    assert "capital of France" in message
    assert calls == [("/models/converted", expected_kwargs)]


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_convert_command_preserves_trust_choice(
    tmp_path, monkeypatch, trust_remote_code
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        '{"model_type":"llama","architectures":["LlamaForCausalLM"]}'
    )
    (source / "model.safetensors").write_bytes(b"source")
    output = tmp_path / "output"
    observed = {}

    import vmlx_engine.utils.model_inspector as inspector

    monkeypatch.setattr(inspector, "resolve_model_path", lambda _model: str(source))
    monkeypatch.setattr(
        inspector,
        "inspect_model",
        lambda _path: SimpleNamespace(
            architecture="LlamaForCausalLM",
            needs_latent_moe=False,
        ),
    )
    monkeypatch.setattr(inspector, "format_model_info", lambda _info: "Llama")
    monkeypatch.setattr(convert_mod, "_preflight_check", lambda _info, _bits: None)

    def fake_conversion(**_kwargs):
        output.mkdir()
        (output / "model.safetensors").write_bytes(b"converted")

    monkeypatch.setattr(convert_mod, "_run_conversion", fake_conversion)

    def fake_smoke(_path, *, trust_remote_code=False):
        observed["trust_remote_code"] = trust_remote_code
        return True, "ok"

    monkeypatch.setattr(convert_mod, "_smoke_test", fake_smoke)

    convert_mod.convert_command(
        argparse.Namespace(
            model=str(source),
            output=str(output),
            bits=4,
            group_size=64,
            mode="default",
            dtype=None,
            force=False,
            skip_verify=False,
            trust_remote_code=trust_remote_code,
            jang_profile=None,
        )
    )

    assert observed == {"trust_remote_code": trust_remote_code}
