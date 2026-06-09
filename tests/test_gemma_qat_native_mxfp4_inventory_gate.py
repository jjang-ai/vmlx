import json
from pathlib import Path

from tests.cross_matrix import run_gemma_qat_native_mxfp4_inventory_gate as gate


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _bundle(
    root: Path,
    name: str,
    *,
    model_type: str,
    text_model_type: str | None = None,
    weight_format: str | None = None,
    vision: bool = True,
    audio: bool = True,
    video: bool = True,
) -> Path:
    path = root / name
    config = {"model_type": model_type}
    if text_model_type:
        config["text_config"] = {"model_type": text_model_type}
    if vision:
        config["vision_config"] = {"model_type": f"{model_type}_vision"}
        config["image_token_id"] = 1
    if audio:
        config["audio_config"] = {"model_type": f"{model_type}_audio"}
        config["audio_token_id"] = 2
    if video:
        config["video_token_id"] = 3
    if weight_format:
        config["weight_format"] = weight_format
        config["quantization"] = {
            "method": weight_format,
            "mode": weight_format,
            "bits": 4 if weight_format == "mxfp4" else 8,
            "group_size": 32,
        }
    _write_json(path / "config.json", config)
    _write_json(path / "generation_config.json", {"do_sample": False})
    _write_json(path / "tokenizer_config.json", {"chat_template": "stub"})
    _write_json(path / "processor_config.json", {"processor_class": "stub"})
    if weight_format:
        _write_json(path / "jang_config.json", {"weight_format": weight_format})
    return path


def test_gate_reports_missing_gemma3n_qat_and_open_present_gemma4_rows(tmp_path):
    _bundle(
        tmp_path,
        "gemma-4-12B-it-MXFP4",
        model_type="gemma4_unified",
        text_model_type="gemma4_unified_text",
        weight_format="mxfp4",
    )
    _bundle(
        tmp_path,
        "Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        model_type="gemma4",
        text_model_type="gemma4_text",
        weight_format="jang",
    )
    _bundle(
        tmp_path,
        "Gemma-4-31B-it-JANG_4M-MTP",
        model_type="gemma4",
        text_model_type="gemma4_text",
        weight_format="jang",
    )

    artifact = gate.build_artifact((tmp_path,))

    assert artifact["status"] == "open"
    assert artifact["checks"]["gemma3n_e2b_qat_present"] is False
    assert artifact["checks"]["gemma3n_e4b_qat_present"] is False
    assert artifact["checks"]["gemma4_12b_native_mxfp4_present"] is True
    assert artifact["checks"]["gemma4_26b_present"] is True
    assert artifact["checks"]["gemma4_31v_or_31b_present"] is True
    assert artifact["required_rows"]["gemma4_12b_native_mxfp4"]["status"] == "open"
    assert (
        "responses_function_call_arguments_delta_done_output_item"
        in artifact["required_rows"]["gemma4_12b_native_mxfp4"]["live_proof_required"]
    )
    assert artifact["required_rows"]["gemma3n_e2b_qat_native4"]["status"] == "missing"


def test_gate_keeps_gemma3n_qat_open_until_live_proof_exists(tmp_path):
    _bundle(
        tmp_path,
        "gemma-3n-E2B-it-qat-q4_0",
        model_type="gemma3n",
        text_model_type=None,
        weight_format="mxfp4",
    )
    _bundle(
        tmp_path,
        "gemma-3n-E4B-it-qat-q4_0",
        model_type="gemma3n",
        text_model_type=None,
        weight_format="mxfp4",
    )

    artifact = gate.build_artifact((tmp_path,))

    assert artifact["status"] == "open"
    assert artifact["checks"]["gemma3n_e2b_qat_present"] is True
    assert artifact["checks"]["gemma3n_e4b_qat_present"] is True
    assert artifact["required_rows"]["gemma3n_e2b_qat_native4"]["status"] == "open"
    assert artifact["required_rows"]["gemma3n_e4b_qat_native4"]["tool_parser"] == "gemma3"
    assert artifact["required_rows"]["gemma3n_e4b_qat_native4"]["reasoning_parser"] is None
    assert artifact["required_rows"]["gemma3n_e4b_qat_native4"]["live_proof_status"] == "missing"
    assert artifact["checks"]["all_required_live_proofs_present"] is False
