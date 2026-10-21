"""Which DFlash2 targets route their affine projections through the lane matmul (models/mllm._dflash_target_lane_qmm)."""
import pytest

from vmlx_engine.models.mllm import _dflash_target_lane_qmm


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("VMLX_LANE_QMM", raising=False)
    monkeypatch.delenv("VMLX_MIMO_LANE_QMM", raising=False)


def test_default_targets_take_the_lane():
    assert _dflash_target_lane_qmm(True, "qwen3_5", False) == (True, "")


def test_no_dflash2_no_lane():
    assert _dflash_target_lane_qmm(False, "qwen3_5", False) == (False, "")


def test_glm5_next_keeps_native_kernels():
    lane, why = _dflash_target_lane_qmm(True, "glm5_next", False)
    assert not lane and "glm5_next" in why


def test_mimo_v2_keeps_mlx_kernels_by_default():
    lane, why = _dflash_target_lane_qmm(True, "mimo_v2", True)
    assert not lane and "MiMo-V2" in why


def test_mimo_v2_lane_restored_by_env(monkeypatch):
    monkeypatch.setenv("VMLX_MIMO_LANE_QMM", "1")
    assert _dflash_target_lane_qmm(True, "mimo_v2", True) == (True, "")


def test_global_off_switch(monkeypatch):
    monkeypatch.setenv("VMLX_LANE_QMM", "0")
    lane, why = _dflash_target_lane_qmm(True, "qwen3_5", False)
    assert not lane and why == "VMLX_LANE_QMM=0"
    monkeypatch.setenv("VMLX_MIMO_LANE_QMM", "1")
    assert _dflash_target_lane_qmm(True, "mimo_v2", True)[0] is False


def test_native_mtp_lane_qmm_is_opt_in(monkeypatch):
    from vmlx_engine.models.mllm import _native_mtp_lane_qmm_requested

    monkeypatch.delenv("VMLX_NATIVE_MTP_LANE_QMM", raising=False)
    assert not _native_mtp_lane_qmm_requested()
    monkeypatch.setenv("VMLX_NATIVE_MTP_LANE_QMM", "1")
    assert _native_mtp_lane_qmm_requested()
    monkeypatch.setenv("VMLX_NATIVE_MTP_LANE_QMM", "0")
    assert not _native_mtp_lane_qmm_requested()
