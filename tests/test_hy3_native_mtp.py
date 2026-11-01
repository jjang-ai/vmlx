# SPDX-License-Identifier: Apache-2.0
"""Hy3 (hy_v3) native MTP — autodetect + model-side contract pins.

The Hy3 JANG converter (jang_tools.convert_hy3_jang, profile JANG_2L with
mtp_policy=preserve-affine8) re-namespaces the DSV3-style MTP layer
(model.layers.80.*) to mtp.0.* with final param names. jang_tools.hy3.model
supplies Hy3MTPLayer + the hooks the vMLX text MTP lane calls (mtp_forward /
make_mtp_cache / return_hidden __call__); vmlx_engine/text_mtp/hy3_family.py
gates head attachment on draft_head_loading_enabled().
"""

from __future__ import annotations

import json

import pytest


def _write_hy3_jang_2l_native_bundle(path):
    """Mirror the stamps jang_tools.convert_hy3_jang writes for JANG_2L +
    mtp_policy=preserve-affine8."""
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "hy_v3",
                "architectures": ["HYV3ForCausalLM"],
                "num_hidden_layers": 80,
                "num_nextn_predict_layers": 1,
            }
        )
    )
    (path / "jang_config.json").write_text(
        json.dumps(
            {
                "format": "jang",
                "format_version": "2.0",
                "profile": "JANG_2L",
                "runtime": {
                    "bundle_has_mtp": True,
                    "mtp_layers": 1,
                    "mtp_mode": "preserved_native_candidate",
                    "mtp_num_speculative_tokens": 2,
                },
                "capabilities": {
                    "family": "hy_v3",
                    "modality": "text",
                    "cache_type": "kv",
                },
            }
        )
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001.safetensors",
                    "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
                    "mtp.0.eh_proj.weight": "model-00001.safetensors",
                    "mtp.0.eh_proj.scales": "model-00001.safetensors",
                    "mtp.0.enorm.weight": "model-00001.safetensors",
                    "mtp.0.hnorm.weight": "model-00001.safetensors",
                    "mtp.0.final_layernorm.weight": "model-00001.safetensors",
                    "mtp.0.block.self_attn.q_proj.weight": "model-00001.safetensors",
                    "mtp.0.block.mlp.switch_mlp.gate_proj.weight": "model-00001.safetensors",
                }
            }
        )
    )
    # Hy3's affine verifier batches two tokens while ordinary greedy decode
    # uses one. Runtime readiness therefore requires an explicit real-artifact
    # token-identity attestation; speed/coherence alone is not sufficient.
    (path / "vmlx_mtp_tuning.json").write_text(
        json.dumps(
            {
                "native_mtp": {
                    "best_depth": 1,
                    "validated": True,
                    "output_equivalent": True,
                }
            }
        )
    )


class TestHy3NativeMtpAutodetect:
    def test_jang_2l_native_bundle_is_native_runtime_ready(self, tmp_path, monkeypatch):
        from vmlx_engine.native_mtp import inspect_native_mtp_bundle

        monkeypatch.delenv("VMLINUX_NATIVE_MTP", raising=False)
        _write_hy3_jang_2l_native_bundle(tmp_path)

        status = inspect_native_mtp_bundle(str(tmp_path))

        assert status["family"] == "hy_v3"
        assert status["config_num_nextn_predict_layers"] == 1
        assert status["index_has_mtp_tensors"] is True
        assert status["index_mtp_layer_count"] == 1
        assert status["jang_drop_mtp"] is not True
        assert status["issues"] == []
        assert status["artifact_available"] is True
        assert status["runtime_supported"] is True
        assert status["runtime_available"] is True
        assert status["status"] == "native_runtime_ready"
        assert status["runtime_scope"] == "text"

    def test_bundle_declared_block_disables_runtime_but_keeps_artifact(
        self, tmp_path, monkeypatch
    ):
        """A bundle that MEASURED MTP as a slowdown can declare it blocked.

        Hy3-JANG_2L does: verify forwards on an MoE cost ~35% more per extra
        token (expert gather scales with distinct experts), so at 65.8%
        depth-1 acceptance MTP is a net loss (29.9 vs 33.4 tok/s).
        """
        from vmlx_engine.native_mtp import inspect_native_mtp_bundle

        monkeypatch.delenv("VMLINUX_NATIVE_MTP", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_FORCE", raising=False)
        _write_hy3_jang_2l_native_bundle(tmp_path)
        jc = json.loads((tmp_path / "jang_config.json").read_text())
        jc["runtime"]["native_mtp_blocked"] = "measured net slowdown on JANG_2L"
        (tmp_path / "jang_config.json").write_text(json.dumps(jc))

        status = inspect_native_mtp_bundle(str(tmp_path))

        # artifact is still valid & detected — this is an acceleration block
        assert status["artifact_available"] is True
        assert status["runtime_supported"] is True
        assert status["index_has_mtp_tensors"] is True
        assert status["issues"] == []
        # ...but the runtime will not engage
        assert status["runtime_available"] is False
        assert status["runtime_validation_blocked"] is True
        assert status["status"] == "runtime_validation_blocked"
        assert "measured net slowdown" in status["runtime_reason"]

        # and it is overridable for experiments
        monkeypatch.setenv("VMLX_NATIVE_MTP_FORCE", "1")
        forced = inspect_native_mtp_bundle(str(tmp_path))
        assert forced["runtime_available"] is True
        assert forced["status"] == "native_runtime_ready"

    def test_missing_output_equivalence_blocks_affine_runtime_but_keeps_artifact(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.native_mtp import inspect_native_mtp_bundle

        monkeypatch.delenv("VMLINUX_NATIVE_MTP_FORCE", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP_FORCE", raising=False)
        _write_hy3_jang_2l_native_bundle(tmp_path)
        tuning = json.loads((tmp_path / "vmlx_mtp_tuning.json").read_text())
        tuning["native_mtp"].pop("output_equivalent")
        tuning["native_mtp"]["measured"] = (
            "coherent but byte-divergent multi-row verifier"
        )
        (tmp_path / "vmlx_mtp_tuning.json").write_text(json.dumps(tuning))

        status = inspect_native_mtp_bundle(tmp_path)

        assert status["artifact_available"] is True
        assert status["runtime_supported"] is True
        assert status["runtime_available"] is False
        assert status["runtime_validation_blocked"] is True
        assert status["status"] == "runtime_validation_blocked"
        assert "output_equivalent is missing" in status["runtime_reason"]
        assert "token-identical greedy output" in status["runtime_reason"]

    def test_env_kill_switch_still_wins(self, tmp_path, monkeypatch):
        from vmlx_engine.native_mtp import inspect_native_mtp_bundle

        monkeypatch.setenv("VMLINUX_NATIVE_MTP", "0")
        _write_hy3_jang_2l_native_bundle(tmp_path)

        status = inspect_native_mtp_bundle(str(tmp_path))

        assert status["runtime_supported"] is True
        assert status["runtime_available"] is False
        assert status["status"] == "runtime_disabled"


def _tiny_hy3_args():
    from jang_tools.hy3.model import ModelArgs

    return ModelArgs.from_dict(
        {
            "model_type": "hy_v3",
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 96,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_shared_experts": 1,
            "first_k_dense_replace": 1,
            "route_norm": True,
            "router_scaling_factor": 2.826,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 11158840.0, "rope_type": "default"},
            "max_position_embeddings": 4096,
            "tie_word_embeddings": False,
            "num_nextn_predict_layers": 1,
        }
    )


class TestHy3ModelMtpContract:
    def test_call_returns_prenorm_hidden_and_mtp_forward_shapes(self):
        import mlx.core as mx

        from jang_tools.hy3.model import Model

        model = Model(_tiny_hy3_args())
        model.attach_mtp()

        x = mx.array([[1, 2, 3]])
        logits, hidden = model(x, return_hidden=True)
        assert logits.shape == (1, 3, 128)
        assert hidden.shape == (1, 3, 64)

        mtp_cache = model.make_mtp_cache()
        assert len(mtp_cache) == 1

        draft_logits = model.mtp_forward(
            hidden[:, -1:, :], mx.array([[7]]), mtp_cache
        )
        assert draft_logits.shape == (1, 1, 128)
        # cache advanced by the 1-token draft step
        assert mtp_cache[0].offset == 1

        # second (recursive/verify-cycle) step continues the same cache
        d2, h2 = model.mtp_forward(
            hidden[:, -1:, :], mx.array([[9]]), mtp_cache, return_hidden=True
        )
        assert d2.shape == (1, 1, 128)
        assert h2.shape == (1, 1, 64)
        assert mtp_cache[0].offset == 2

    def test_sanitize_strips_mtp_without_head_and_keeps_with_head(self):
        import mlx.core as mx

        from jang_tools.hy3.model import Model
        from vmlx_engine.text_mtp import draft_head_loading_enabled, set_draft_head_loading

        weights = {
            "model.norm.weight": mx.ones((64,)),
            "mtp.0.enorm.weight": mx.ones((64,)),
            "model.layers.2.eh_proj.weight": mx.ones((64, 128)),  # legacy MTP
        }

        # Another test module may have left the process-wide MTP gate on; once
        # the patch is applied, Model.__init__ auto-attaches the head when it
        # is. Pin the flag so `bare` really is head-free.
        prev = draft_head_loading_enabled()
        set_draft_head_loading(False)
        try:
            bare = Model(_tiny_hy3_args())
        finally:
            set_draft_head_loading(prev)
        out = bare.sanitize(dict(weights))
        assert "mtp.0.enorm.weight" not in out
        assert "model.layers.2.eh_proj.weight" not in out
        assert "model.norm.weight" in out

        headed = Model(_tiny_hy3_args())
        headed.attach_mtp()
        out = headed.sanitize(dict(weights))
        assert "mtp.0.enorm.weight" in out
        # legacy source-style MTP names are still dropped (never loaded)
        assert "model.layers.2.eh_proj.weight" not in out

    def test_vmlx_patch_gates_head_attachment(self):
        from vmlx_engine.text_mtp import hy3_family
        from vmlx_engine.text_mtp import (
            draft_head_loading_enabled,
            set_draft_head_loading,
        )

        assert hy3_family.install() is True

        import sys

        hy = sys.modules["mlx_lm.models.hy_v3"]
        assert "__vmlx_text_mtp__" in hy.Model.__dict__

        prev = draft_head_loading_enabled()
        try:
            set_draft_head_loading(False)
            assert not hasattr(hy.Model(_tiny_hy3_args()), "mtp")
            set_draft_head_loading(True)
            model = hy.Model(_tiny_hy3_args())
            assert hasattr(model, "mtp") and len(model.mtp) == 1
        finally:
            set_draft_head_loading(prev)

    def test_model_has_native_mtp_runtime_walk_finds_hy3_head(self):
        from jang_tools.hy3.model import Model
        from vmlx_engine.native_mtp import model_has_native_mtp_runtime
        from vmlx_engine.text_mtp import draft_head_loading_enabled, set_draft_head_loading

        prev = draft_head_loading_enabled()
        set_draft_head_loading(False)  # see note above: patched __init__ reads this
        try:
            model = Model(_tiny_hy3_args())
        finally:
            set_draft_head_loading(prev)
        assert model_has_native_mtp_runtime(model) is False
        model.attach_mtp()
        assert model_has_native_mtp_runtime(model) is True


def test_hy3_family_depth_fallback_is_one(tmp_path):
    """An hy_v3 MTP bundle WITHOUT a tuning sidecar must default to depth 1
    (2026-07-10 sweep: d1 +10%, d2 -6% @24.5% accept, d3 -31% @1.9%).
    Qwen artifacts keep the D3 fallback."""
    import json
    from vmlx_engine.native_mtp import native_mtp_effective_depth

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "hy_v3"}))
    (tmp_path / "jang_config.json").write_text(json.dumps({}))
    depth, source = native_mtp_effective_depth(str(tmp_path))
    assert depth == 1
    assert source == "family_default:hy_v3"


def test_hy3_tuning_sidecar_outranks_family_fallback(tmp_path):
    import json
    from vmlx_engine.native_mtp import native_mtp_effective_depth

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "hy_v3"}))
    (tmp_path / "jang_config.json").write_text(json.dumps({}))
    (tmp_path / "vmlx_mtp_tuning.json").write_text(
        json.dumps({"native_mtp": {"best_depth": 2, "validated": True}})
    )
    depth, source = native_mtp_effective_depth(str(tmp_path))
    assert depth == 2
    assert source == "vmlx_mtp_tuning.json:native_mtp.best_depth"


def test_invalid_depth_env_does_not_bypass_sidecar(tmp_path, monkeypatch):
    """Audit H4: an invalid VMLX_NATIVE_MTP_DEPTH must be ignored (resolve
    from tuning/family), not silently become the generic D3 default."""
    import json
    from vmlx_engine.native_mtp import native_mtp_effective_depth

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "hy_v3"}))
    (tmp_path / "jang_config.json").write_text(json.dumps({}))
    (tmp_path / "vmlx_mtp_tuning.json").write_text(
        json.dumps({"native_mtp": {"best_depth": 1, "validated": True}})
    )
    monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "banana")
    depth, source = native_mtp_effective_depth(str(tmp_path))
    assert depth == 1
    assert source == "vmlx_mtp_tuning.json:native_mtp.best_depth"


def test_blocked_tuning_file_leaks_no_depth(tmp_path):
    """Audit H4: a tuning file whose native_mtp block is invalid must not
    leak a depth through the legacy top-level fallback keys."""
    import json
    from vmlx_engine.native_mtp import _model_tuning_depth

    (tmp_path / "vmlx_mtp_tuning.json").write_text(json.dumps({
        "native_mtp": {"blocked": True, "best_depth": 2},
        "best_native_mtp_depth": {"best_depth": 3},
        "best_depth": 3,
    }))
    depth, source = _model_tuning_depth(str(tmp_path))
    assert depth is None and source is None


def test_nested_tuning_block_needs_validated_true_above_depth_one(tmp_path):
    """One evidence rule for every sidecar format: a nested block that merely
    omits ``validated`` (or a sweep block without it) may recommend depth 1
    but not deeper; ``validated: true`` unlocks its stated depth."""
    import json
    from vmlx_engine.native_mtp import _model_tuning_depth

    (tmp_path / "vmlx_mtp_tuning.json").write_text(json.dumps({"native_mtp": {"best_depth": 3}}))
    assert _model_tuning_depth(str(tmp_path)) == (None, None)
    (tmp_path / "vmlx_mtp_tuning.json").write_text(json.dumps({"native_mtp": {"best_depth": 1}}))
    assert _model_tuning_depth(str(tmp_path)) == (1, "vmlx_mtp_tuning.json:native_mtp.best_depth")
    (tmp_path / "vmlx_mtp_tuning.json").write_text(json.dumps({"native_mtp": {"best_depth": 3, "validated": True}}))
    assert _model_tuning_depth(str(tmp_path)) == (3, "vmlx_mtp_tuning.json:native_mtp.best_depth")
    (tmp_path / "vmlx_mtp_tuning.json").write_text(json.dumps({"best_native_mtp_depth": {"best_depth": 2}}))
    assert _model_tuning_depth(str(tmp_path)) == (None, None)
    (tmp_path / "vmlx_mtp_tuning.json").write_text(json.dumps({"best_native_mtp_depth": {"best_depth": 2, "validated": True}}))
    assert _model_tuning_depth(str(tmp_path)) == (2, "vmlx_mtp_tuning.json:best_native_mtp_depth.best_depth")
