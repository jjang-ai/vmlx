import json


def _write_model(path, *, model_type="qwen3_5_moe", weight_format="jang", profile="JANG_1L"):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (path / "jang_config.json").write_text(
        json.dumps(
            {
                "version": 2,
                "weight_format": weight_format,
                "quantization": {"profile": profile},
            }
        )
    )


def test_n2_jang1l_affine_defers_startup_eval(tmp_path):
    from vmlx_engine.utils.tokenizer import _should_defer_jang_startup_eval

    model = tmp_path / "Nex-N2-Pro-JANG_1L"
    _write_model(model)

    assert _should_defer_jang_startup_eval(str(model)) is True


def test_n2_jangtq_does_not_defer_startup_eval(tmp_path):
    from vmlx_engine.utils.tokenizer import _should_defer_jang_startup_eval

    model = tmp_path / "Nex-N2-Pro-JANGTQ2"
    _write_model(model, weight_format="mxtq", profile="JANGTQ2")

    assert _should_defer_jang_startup_eval(str(model)) is False


def test_non_n2_jang1l_does_not_defer_startup_eval(tmp_path):
    from vmlx_engine.utils.tokenizer import _should_defer_jang_startup_eval

    model = tmp_path / "Other-JANG_1L"
    _write_model(model, model_type="qwen2_moe")

    assert _should_defer_jang_startup_eval(str(model)) is False


def test_load_model_with_fallback_passes_skip_eval_for_n2_jang1l(tmp_path, monkeypatch):
    from vmlx_engine.utils import tokenizer
    from vmlx_engine.api import utils as api_utils

    model = tmp_path / "Nex-N2-Pro-JANG_1L"
    _write_model(model)
    calls = []

    monkeypatch.setattr(tokenizer, "_register_mimo_v2_runtime_for_mlx_lm", lambda: False)
    monkeypatch.setattr(tokenizer, "_inject_chat_template_if_missing", lambda *_a, **_k: None)
    monkeypatch.setattr(tokenizer, "_needs_tokenizer_fallback", lambda _path: False)
    monkeypatch.setattr(api_utils, "resolve_to_local_path", lambda name: str(model))

    def fake_load_jang_model(path, **kwargs):
        calls.append((path, kwargs))
        return object(), object()

    monkeypatch.setattr("vmlx_engine.utils.jang_loader.is_jang_model", lambda _path: True)
    monkeypatch.setattr("vmlx_engine.utils.jang_loader.load_jang_model", fake_load_jang_model)

    tokenizer.load_model_with_fallback(str(model))

    assert calls == [(str(model), {"skip_eval": True})]


def test_n2_wired_limit_plan_reports_tight_metal_cap():
    from vmlx_engine.utils.jang_loader import _wired_limit_plan_for_model

    gib = 1024**3
    plan = _wired_limit_plan_for_model(
        total_bytes=int(106.725 * gib),
        max_working_set_bytes=int(107.10 * gib),
    )

    assert plan["target_bytes"] == int(107.10 * gib)
    assert plan["capped_by_working_set"] is True
    assert plan["headroom_bytes"] < int(0.5 * gib)
    assert plan["tight_headroom"] is True
    assert "106.72 GiB" in plan["message"]
    assert "107.10 GiB" in plan["message"]
    assert "headroom=0.38 GiB" in plan["message"]
    assert "capped_by_metal_ws=true" in plan["message"]
