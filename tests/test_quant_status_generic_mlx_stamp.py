"""A generic "mlx" weight_format stamp names the loader, not the codec.

The 2026-08 Nemotron bundles declare weight_format "mlx" with the actual
codec in quantization.mode (mxfp8/mxfp4). The status display showed "mlx" —
uninformative — and inferred the codec via the generic q_cfg fallback.
"""
import json

from vmlx_engine.server import _model_quantization_status


def _write_bundle(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_generic_mlx_stamp_surfaces_the_quantization_mode(tmp_path):
    path = _write_bundle(
        tmp_path,
        {
            "model_type": "nemotron_h",
            "weight_format": "mlx",
            "quantization": {"mode": "mxfp8", "group_size": 32, "bits": 8},
        },
    )
    status = _model_quantization_status(path)
    assert status["weight_format"] == "mxfp8"


def test_concrete_weight_formats_are_untouched(tmp_path):
    path = _write_bundle(
        tmp_path,
        {
            "model_type": "nemotron_h",
            "weight_format": "affine",
            "quantization": {"mode": "affine"},
        },
    )
    status = _model_quantization_status(path)
    assert status["weight_format"] == "affine"


def test_mlx_stamp_with_no_codec_stays_mlx(tmp_path):
    path = _write_bundle(
        tmp_path, {"model_type": "llama", "weight_format": "mlx"}
    )
    status = _model_quantization_status(path)
    assert status["weight_format"] == "mlx"


def test_mixed_jang_does_not_report_fallback_config_as_global_target(tmp_path):
    path = _write_bundle(tmp_path, {"quantization": {"bits": 8, "group_size": 64}})
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "weight_format": "mixed_affine_mxfp4", "profile": "JANG_2L",
        "quantization": {"method": "mixed", "nonexpert": {"bits": 8},
                         "routed_expert_units": {"affine2/g128": 109, "affine3/g128": 22, "mxfp4": 10}},
    }))
    status = _model_quantization_status(path)
    assert status["mixed_precision"] is True
    assert status["config_bits"] == 8
    assert "target_bits" not in status
    assert "actual_bits" not in status  # Never invent an average from a tier name.


def test_loaded_inkling_reports_actual_mixed_expert_modes():
    from types import SimpleNamespace

    from vmlx_engine.server import _loaded_inkling_weight_quantization

    model = SimpleNamespace(_inkling_loaded_expert_modes=("nvfp4", "mxfp8"))
    engine = SimpleNamespace(_model=model, _loaded=False)
    assert _loaded_inkling_weight_quantization(engine) == {}

    engine._loaded = True
    quantization = _loaded_inkling_weight_quantization(engine)
    assert quantization["weight_format"] == "nvfp4/mxfp8"
    assert quantization["codec"] == "floating_point_quantized_matmul"
    assert quantization["mixed_precision"] is True
    assert quantization["quantization_metadata_source"] == "loaded_expert_projections"
    assert "actual_bits" not in quantization and "target_bits" not in quantization

    model._inkling_loaded_expert_modes = ()
    assert _loaded_inkling_weight_quantization(engine) == {}
