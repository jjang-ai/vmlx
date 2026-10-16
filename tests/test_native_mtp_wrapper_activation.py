"""Wrapper MTP activation and failed recurrent rollback state contracts."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("allow_runtime", [True, False])
def test_wrapper_activation_replaces_previous_bundle_state(monkeypatch, tmp_path, allow_runtime):
    from vmlx_engine import native_mtp as m

    flags, heads = [], []
    monkeypatch.setattr(m, "_ACTIVE_NATIVE_MTP_MODEL_PATH", Path("/previous-bundle"))
    monkeypatch.setattr(m, "_set_mtp_active", flags.append)
    monkeypatch.setitem(sys.modules, "vmlx_engine.patches.mlx_vlm_mtp.qwen35_vl",
                        SimpleNamespace(set_active_bundle_mtp_layers=heads.append))
    monkeypatch.setattr(m, "inspect_native_mtp_bundle", lambda path: {
        "runtime_adapter": "nemotron_h_head", "runtime_available": True,
        "family": "nemotron_h", "status": "native_runtime_ready",
        "effective_depth": 3, "effective_depth_source": "default",
    })
    for key in ("VMLINUX_NATIVE_MTP_DEPTH", "VMLX_NATIVE_MTP_DEPTH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(m, "_env_disabled", lambda *a: False)
    monkeypatch.setattr(m, "_model_tuning_depth", lambda path:
                        (1, "vmlx_mtp_tuning:test") if path == tmp_path else (None, None))
    result = m.maybe_apply_native_mtp(tmp_path, allow_runtime=allow_runtime)
    assert result["runtime_active"] is allow_runtime
    assert flags == [False] and heads == [0]
    assert m._ACTIVE_NATIVE_MTP_MODEL_PATH == (tmp_path if allow_runtime else None)
    if allow_runtime:
        assert m.native_mtp_effective_depth() == (1, "vmlx_mtp_tuning:test")
    else:
        assert result["status"] == "runtime_disabled"
        assert result["runtime_reason"] == "native MTP wrapper runtime disabled"

