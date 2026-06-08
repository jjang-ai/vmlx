import json


def _write_step3p7_model(tmp_path, *, jang_has_vision):
    model_dir = tmp_path / "Step-3.7-Flash-JANG_2L-CRACK"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Step3p7ForConditionalGeneration"],
                "model_type": "step3p7",
                "text_config": {"model_type": "step3p5"},
                "vision_config": {"hidden_size": 4096},
            }
        )
    )
    (model_dir / "jang_config.json").write_text(
        json.dumps(
            {
                "format": "jang",
                "architecture": {
                    "type": "step3p7",
                    "text_model_type": "step3p5",
                    "has_audio": False,
                    "has_mtp_tensors": False,
                    "has_vision": jang_has_vision,
                },
            }
        )
    )
    return model_dir


def test_step3p7_jang_advertised_vision_routes_vlm_when_source_runtime_exists(tmp_path):
    from vmlx_engine.api import utils

    model_dir = _write_step3p7_model(tmp_path, jang_has_vision=True)
    utils._IS_MLLM_CACHE.clear()

    assert utils.is_mllm_model(str(model_dir)) is True


def test_step3p7_jang_advertised_vision_blocks_vlm_when_runtime_missing(
    tmp_path,
    monkeypatch,
):
    from vmlx_engine.api import utils

    model_dir = _write_step3p7_model(tmp_path, jang_has_vision=True)
    utils._IS_MLLM_CACHE.clear()
    monkeypatch.setattr(utils, "_source_step3p7_vlm_runtime_available", lambda: False)

    assert utils.is_mllm_model(str(model_dir)) is False
    assert utils.is_mllm_model(str(model_dir), force_mllm=True) is False


def test_step3p7_jang_advertised_vision_force_mllm_uses_source_runtime(tmp_path):
    from vmlx_engine.api import utils

    model_dir = _write_step3p7_model(tmp_path, jang_has_vision=True)
    utils._IS_MLLM_CACHE.clear()

    assert utils.is_mllm_model(str(model_dir), force_mllm=True) is True


def test_step3p7_jang_text_only_view_stays_text_only(tmp_path):
    from vmlx_engine.api import utils

    model_dir = _write_step3p7_model(tmp_path, jang_has_vision=False)
    utils._IS_MLLM_CACHE.clear()

    assert utils.is_mllm_model(str(model_dir)) is False
