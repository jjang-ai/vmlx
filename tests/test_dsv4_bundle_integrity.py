import json

import numpy as np
import pytest
from safetensors.numpy import save_file


def _write_bundle(tmp_path, *, model_type="deepseek_v4", dtype=np.float32):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    save_file(
        {
            "hc_head_fn": np.zeros((1,), dtype=dtype),
            "layers.0.attn.attn_sink": np.zeros((1,), dtype=np.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    return tmp_path


def test_dsv4_control_tensor_validator_rejects_f16(tmp_path):
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _validate_dsv4_control_tensors,
    )

    bundle = _write_bundle(tmp_path, dtype=np.float16)

    with pytest.raises(RuntimeError) as exc:
        _validate_dsv4_control_tensors(bundle)

    msg = str(exc.value)
    assert "DSV4 bundle is known-bad" in msg
    assert "critical control tensors are not F32" in msg
    assert "hc_head_fn=F16" in msg


def test_dsv4_control_tensor_validator_accepts_f32(tmp_path):
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _audit_dsv4_control_tensor_dtypes,
        _validate_dsv4_control_tensors,
    )

    bundle = _write_bundle(tmp_path, dtype=np.float32)

    _validate_dsv4_control_tensors(bundle)
    report = _audit_dsv4_control_tensor_dtypes(bundle)
    assert report["checked"] is True
    assert report["critical_count"] == 2
    assert report["non_f32_count"] == 0


def test_dsv4_control_tensor_validator_skips_non_dsv4(tmp_path):
    from vmlx_engine.loaders.load_jangtq_dsv4 import (
        _audit_dsv4_control_tensor_dtypes,
        _validate_dsv4_control_tensors,
    )

    bundle = _write_bundle(tmp_path, model_type="qwen3", dtype=np.float16)

    _validate_dsv4_control_tensors(bundle)
    report = _audit_dsv4_control_tensor_dtypes(bundle)
    assert report["checked"] is False
