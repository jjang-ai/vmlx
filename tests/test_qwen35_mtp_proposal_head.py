"""Qwen3.5/3.8 VLM-lane proposal head: stamp-driven, env kill switch."""

import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine import native_mtp
from vmlx_engine.native_mtp_proposal_stamp import STAMP_FILENAME
from vmlx_engine.patches.mlx_vlm_mtp.qwen35_vl import _qwen35_mtp_proposal_head


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("VMLINUX_QWEN35_MTP_DRAFT_HEAD_BITS", raising=False)
    monkeypatch.delenv("VMLX_QWEN35_MTP_DRAFT_HEAD_BITS", raising=False)


def _model(head_bits: int, group_size: int = 64):
    lin = nn.Linear(256, 512, bias=False).to_quantized(
        group_size=group_size, bits=head_bits
    )
    mx.eval(lin.weight, lin.scales, lin.biases)
    return SimpleNamespace(
        lm_head=lin, args=SimpleNamespace(tie_word_embeddings=False)
    )


def _confirming_config(bundle, head_bits: int, group_size: int = 64):
    (bundle / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_vl",
                "tie_word_embeddings": False,
                "jang_config": {"calibrated": True},
                "quantization": {
                    "language_model.lm_head": {
                        "bits": head_bits,
                        "group_size": group_size,
                        "mode": "affine",
                    }
                },
            }
        )
    )


def test_q8_g64_head_builds_and_stamps(monkeypatch, tmp_path):
    monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", tmp_path)
    _confirming_config(tmp_path, 8)
    model = _model(8)
    head = _qwen35_mtp_proposal_head(model)
    assert head is not None
    assert head.bits == 4
    stamp = json.loads((tmp_path / STAMP_FILENAME).read_text())
    assert stamp["eligible"] is True
    assert stamp["family"] == "qwen3_5"
    # Second call returns the cached head without re-resolving.
    assert _qwen35_mtp_proposal_head(model) is head


@pytest.mark.parametrize("group_size", [64, 128])
def test_stamped_proposal_keeps_loaded_group_size_and_target(
    monkeypatch, tmp_path, caplog, group_size,
):
    """A pre-existing eligible layout stamp must not hit a fixed-width shell."""
    import logging

    monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", tmp_path)
    _confirming_config(tmp_path, 8, group_size)
    stamp_path = tmp_path / STAMP_FILENAME
    stamp_path.write_text(json.dumps({
        "version": 1, "family": "qwen3_5", "eligible": True,
        "proposal_bits": 4,
        "source": {"bits": 8, "group_size": group_size,
                   "mode": "affine", "tied": False},
    }))
    stamp_before = stamp_path.read_bytes()
    model = _model(8, group_size)
    target = model.lm_head
    saved_arrays = [mx.array(getattr(target, key))
                    for key in ("weight", "scales", "biases")]
    x = mx.random.normal((1, 1, 256))
    target_before = target(x)
    mx.eval(target_before, *saved_arrays)

    with caplog.at_level(logging.INFO):
        proposal = _qwen35_mtp_proposal_head(model)

    assert proposal is not None, model._vmlx_mtp_draft_head_state
    assert proposal.group_size == group_size
    assert proposal.bits == 4
    assert model.lm_head is target
    dense = mx.dequantize(*saved_arrays, group_size=group_size, bits=8)
    expected = mx.quantize(dense, group_size=group_size, bits=4)
    for key, before, want in zip(("weight", "scales", "biases"), saved_arrays, expected):
        assert bool(mx.array_equal(getattr(target, key), before))
        assert bool(mx.array_equal(getattr(proposal, key), want))
    actual = proposal(x)
    after = target(x)
    mx.eval(actual, after)
    assert actual.shape == (1, 1, 512)
    assert bool(mx.all(mx.isfinite(actual)))
    assert bool(mx.array_equal(target_before, after))
    assert _qwen35_mtp_proposal_head(model) is proposal
    assert stamp_path.read_bytes() == stamp_before
    assert f"q8/g{group_size} -> q4/g{group_size}" in caplog.text


def test_27b_low_bit_head_stamps_ineligible_and_uses_full_head(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", tmp_path)
    _confirming_config(tmp_path, 4, group_size=128)
    model = _model(4, group_size=128)
    assert _qwen35_mtp_proposal_head(model) is None
    assert model._vmlx_mtp_draft_head_state["reason"] == (
        "native_head_already_low_bit"
    )
    stamp = json.loads((tmp_path / STAMP_FILENAME).read_text())
    assert stamp["eligible"] is False


def test_unstamped_g128_does_not_expand_proposal_eligibility(monkeypatch, tmp_path):
    monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", tmp_path)
    _confirming_config(tmp_path, 8, group_size=128)
    model = _model(8, group_size=128)
    target = model.lm_head
    assert _qwen35_mtp_proposal_head(model) is None
    assert model._vmlx_mtp_draft_head_state["reason"] == "unmeasured_layout_q8_g128"
    assert model.lm_head is target


def test_env_zero_kills_even_when_stamped_eligible(monkeypatch, tmp_path):
    monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", tmp_path)
    monkeypatch.setenv("VMLX_QWEN35_MTP_DRAFT_HEAD_BITS", "0")
    _confirming_config(tmp_path, 8)
    model = _model(8)
    assert _qwen35_mtp_proposal_head(model) is None
    assert model._vmlx_mtp_draft_head_state["reason"] == "disabled_by_env"
    # The one-time check still stamped the bundle for later launches.
    assert json.loads((tmp_path / STAMP_FILENAME).read_text())["eligible"] is True


def test_no_active_path_still_builds_without_stamp(monkeypatch):
    monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", None)
    model = _model(8)
    head = _qwen35_mtp_proposal_head(model)
    assert head is not None
