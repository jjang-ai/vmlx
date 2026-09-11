from types import SimpleNamespace

import pytest


def _rows(contract):
    return {row["id"]: row for row in contract["features"]}


def test_acceleration_family_resolves_nested_model_types():
    from vmlx_engine.acceleration_contract import acceleration_family_from_config

    assert acceleration_family_from_config({"model_type": "qwen4_exp"}) == "qwen4_exp"
    assert (
        acceleration_family_from_config(
            {"model_type": "wrapper", "text_config": {"model_type": "qwen3_5"}}
        )
        == "qwen3_5"
    )
    assert (
        acceleration_family_from_config(
            {"llm_config": {"model_type": "qwen3_5_moe_text"}}
        )
        == "qwen3_5_moe"
    )
    assert acceleration_family_from_config({"model_type": "unknown"}) is None


def test_qwen4_contract_never_calls_requested_kernel_installed(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract

    monkeypatch.setenv("VMLX_QWEN4_FUSED_GDN_CONV", "1")
    contract = build_acceleration_contract("qwen4_exp")
    rows = _rows(contract)

    assert contract["native_state"] == ["QSA", "GDN", "PLE", "n-gram", "MoE"]
    assert rows["projection_groups"]["state"] == "configured_unattested"
    assert rows["gdn_conv_state"]["requested"] is True
    assert rows["gdn_conv_state"]["state"] == "configured_unattested"
    assert rows["affine_moe_pair"]["requested"] is True
    assert rows["affine_moe_pair"]["selection_source"] == "default"
    assert rows["affine_moe"]["requested"] is False
    assert rows["affine_moe"]["state"] == "disabled"
    assert contract["summary"]["installed"] == 0


def test_qwen4_affine_moe_pair_default_has_explicit_opt_out(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract

    monkeypatch.delenv("VMLX_QWEN4_FUSED_MOE_PAIR", raising=False)
    rows = _rows(build_acceleration_contract("qwen4_exp"))
    assert rows["affine_moe_pair"]["requested"] is True
    assert rows["affine_moe_pair"]["state"] == "configured_unattested"

    monkeypatch.setenv("VMLX_QWEN4_FUSED_MOE_PAIR", "0")
    rows = _rows(build_acceleration_contract("qwen4_exp"))
    assert rows["affine_moe_pair"]["requested"] is False
    assert rows["affine_moe_pair"]["state"] == "disabled"


def test_glm_mhc_default_has_explicit_opt_out(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract
    from vmlx_engine.metal.glm5_mhc_decode import fused_glm5_mhc_requested

    monkeypatch.delenv("VMLX_GLM5_FUSED_MHC", raising=False)
    rows = _rows(build_acceleration_contract("glm5_next"))
    assert rows["mhc_transform"]["requested"] is True
    assert rows["mhc_transform"]["selection_source"] == "default"
    assert fused_glm5_mhc_requested() is True

    monkeypatch.setenv("VMLX_GLM5_FUSED_MHC", "0")
    rows = _rows(build_acceleration_contract("glm5_next"))
    assert rows["mhc_transform"]["requested"] is False
    assert rows["mhc_transform"]["state"] == "disabled"
    assert fused_glm5_mhc_requested() is False


def test_glm_kda_decode_defaults_have_explicit_opt_out(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract
    from vmlx_engine.metal.kda_conv_decode import fused_kda_conv_requested
    from vmlx_engine.metal.kda_step_decode import fused_kda_step_requested

    for name in ("VMLX_GLM5_FUSED_KDA_CONV", "VMLX_GLM5_FUSED_KDA_STEP"):
        monkeypatch.delenv(name, raising=False)
    rows = _rows(build_acceleration_contract("glm5_next"))
    assert rows["kda_conv_state"]["requested"] is True
    assert rows["kda_conv_state"]["selection_source"] == "default"
    assert rows["kda_recurrent_step"]["requested"] is True
    assert rows["kda_recurrent_step"]["selection_source"] == "default"
    assert fused_kda_conv_requested() is True
    assert fused_kda_step_requested() is True

    monkeypatch.setenv("VMLX_GLM5_FUSED_KDA_CONV", "0")
    monkeypatch.setenv("VMLX_GLM5_FUSED_KDA_STEP", "0")
    rows = _rows(build_acceleration_contract("glm5_next"))
    assert rows["kda_conv_state"]["requested"] is False
    assert rows["kda_conv_state"]["state"] == "disabled"
    assert rows["kda_recurrent_step"]["requested"] is False
    assert rows["kda_recurrent_step"]["state"] == "disabled"
    assert fused_kda_conv_requested() is False
    assert fused_kda_step_requested() is False


def test_glm_mhc_verify_fusions_default_on_with_explicit_opt_out(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract
    from vmlx_engine.metal.glm5_hc_place_decode import (
        fused_glm5_hc_place_verify_requested,
    )
    from vmlx_engine.metal.glm5_mhc_decode import (
        fused_glm5_mhc_verify_requested,
    )

    for name in (
        "VMLX_GLM5_FUSED_MHC_VERIFY",
        "VMLINUX_GLM5_FUSED_MHC_VERIFY",
        "VMLX_GLM5_FUSED_HC_PLACE_VERIFY",
        "VMLINUX_GLM5_FUSED_HC_PLACE_VERIFY",
    ):
        monkeypatch.delenv(name, raising=False)
    rows = _rows(build_acceleration_contract("glm5_next"))
    assert rows["mhc_verify_transform"]["requested"] is True
    assert rows["hc_place_verify"]["requested"] is True
    assert fused_glm5_mhc_verify_requested() is True
    assert fused_glm5_hc_place_verify_requested() is True

    monkeypatch.setenv("VMLX_GLM5_FUSED_MHC_VERIFY", "0")
    monkeypatch.setenv("VMLX_GLM5_FUSED_HC_PLACE_VERIFY", "0")
    rows = _rows(build_acceleration_contract("glm5_next"))
    assert rows["mhc_verify_transform"]["requested"] is False
    assert rows["hc_place_verify"]["requested"] is False
    assert fused_glm5_mhc_verify_requested() is False
    assert fused_glm5_hc_place_verify_requested() is False


def test_glm_aligned_mtp_head_cache_is_exact_opt_in(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract

    monkeypatch.delenv("VMLX_GLM5_ALIGNED_MTP_HEAD_CACHE", raising=False)
    monkeypatch.delenv("VMLINUX_GLM5_ALIGNED_MTP_HEAD_CACHE", raising=False)
    rows = _rows(build_acceleration_contract("glm5_next"))
    assert rows["aligned_mtp_head_cache"]["requested"] is False
    assert rows["aligned_mtp_head_cache"]["state"] == "disabled"

    monkeypatch.setenv("VMLX_GLM5_ALIGNED_MTP_HEAD_CACHE", "1")
    rows = _rows(build_acceleration_contract("glm5_next"))
    assert rows["aligned_mtp_head_cache"]["requested"] is True
    assert rows["aligned_mtp_head_cache"]["state"] == "configured_unattested"


def test_glm_mtp_prompt_priming_is_exact_opt_in(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract

    monkeypatch.delenv("VMLX_GLM5_MTP_PROMPT_PRIMING", raising=False)
    monkeypatch.delenv("VMLINUX_GLM5_MTP_PROMPT_PRIMING", raising=False)
    rows = _rows(build_acceleration_contract("glm5_next"))
    assert rows["mtp_prompt_priming"]["requested"] is False
    assert rows["mtp_prompt_priming"]["state"] == "disabled"

    monkeypatch.setenv("VMLX_GLM5_MTP_PROMPT_PRIMING", "1")
    rows = _rows(build_acceleration_contract("glm5_next"))
    assert rows["mtp_prompt_priming"]["requested"] is True
    assert rows["mtp_prompt_priming"]["state"] == "configured_unattested"


def test_ernie_mtp_prompt_priming_defaults_on_with_explicit_opt_out(monkeypatch):
    from vmlx_engine.acceleration_contract import (
        acceleration_family_from_config,
        build_acceleration_contract,
    )

    assert acceleration_family_from_config({"model_type": "ernie4_5_moe"}) == "ernie4_5"
    monkeypatch.delenv("VMLX_ERNIE45_MTP_PROMPT_PRIMING", raising=False)
    monkeypatch.delenv("VMLINUX_ERNIE45_MTP_PROMPT_PRIMING", raising=False)
    contract = build_acceleration_contract("ernie4_5_moe")
    assert contract["known_family"] is True
    rows = _rows(contract)
    assert rows["mtp_prompt_priming"]["requested"] is True
    assert rows["mtp_prompt_priming"]["selection_source"] == "default"
    assert rows["mtp_prompt_priming"]["state"] == "configured_unattested"

    monkeypatch.setenv("VMLX_ERNIE45_MTP_PROMPT_PRIMING", "0")
    rows = _rows(build_acceleration_contract("ernie4_5_moe"))
    assert rows["mtp_prompt_priming"]["requested"] is False
    assert rows["mtp_prompt_priming"]["state"] == "disabled"


@pytest.mark.parametrize(
    "vmlinux,vmlx",
    [
        (None, None), ("", None), (None, ""), ("0", None), (None, "0"), ("1", None),
        (None, "1"), ("false", "1"), ("1", "0"), ("0", "1"), ("off", "off"), ("yes", "no"),
    ],
)
def test_ernie_priming_report_matches_runtime_gate(monkeypatch, vmlinux, vmlx):
    """Both aliases, any combination: what /health reports must be what the gate does."""
    from vmlx_engine.acceleration_contract import build_acceleration_contract
    from vmlx_engine.patches.mlx_lm_mtp.batch_generator import _glm_prompt_priming_enabled

    for name, value in (
        ("VMLINUX_ERNIE45_MTP_PROMPT_PRIMING", vmlinux),
        ("VMLX_ERNIE45_MTP_PROMPT_PRIMING", vmlx),
    ):
        monkeypatch.delenv(name, raising=False)
        if value is not None:
            monkeypatch.setenv(name, value)
    reported = _rows(build_acceleration_contract("ernie4_5_moe"))["mtp_prompt_priming"]["requested"]
    actual = _glm_prompt_priming_enabled(type("Model", (), {"model_type": "ernie4_5_moe"})())
    assert reported is actual


def test_glm_vectorized_kda_verify_defaults_on_with_explicit_opt_out(
    monkeypatch,
):
    from vmlx_engine.acceleration_contract import build_acceleration_contract

    monkeypatch.delenv("VMLX_GLM5_VECTOR_KDA_VERIFY", raising=False)
    monkeypatch.delenv("VMLINUX_GLM5_VECTOR_KDA_VERIFY", raising=False)
    rows = _rows(build_acceleration_contract("glm5_next"))
    assert rows["vectorized_kda_verify"]["requested"] is True
    assert rows["vectorized_kda_verify"]["selection_source"] == "default"

    monkeypatch.setenv("VMLX_GLM5_VECTOR_KDA_VERIFY", "0")
    rows = _rows(build_acceleration_contract("glm5_next"))
    assert rows["vectorized_kda_verify"]["requested"] is False
    assert rows["vectorized_kda_verify"]["state"] == "disabled"


def test_qwen35_gdn_conv_candidate_is_family_scoped(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract

    monkeypatch.setenv("VMLX_QWEN35_FUSED_GDN_CONV", "1")
    qwen35 = _rows(build_acceleration_contract("qwen3_5"))
    qwen35_moe = _rows(build_acceleration_contract("qwen3_5_moe"))
    qwen4 = _rows(build_acceleration_contract("qwen4_exp"))
    assert qwen35["gdn_conv_state"]["requested"] is True
    assert qwen35_moe["gdn_conv_state"]["requested"] is True
    assert qwen4["gdn_conv_state"]["requested"] is False


def test_qwen35_gdn_gate_candidate_is_family_scoped(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract

    monkeypatch.setenv("VMLX_QWEN35_FUSED_GDN_GATES", "1")
    qwen35 = _rows(build_acceleration_contract("qwen3_5"))
    qwen35_moe = _rows(build_acceleration_contract("qwen3_5_moe"))
    qwen4 = _rows(build_acceleration_contract("qwen4_exp"))
    assert qwen35["gdn_gate_terms"]["requested"] is True
    assert qwen35_moe["gdn_gate_terms"]["requested"] is True
    assert "gdn_gate_terms" not in qwen4


def test_runtime_attestation_distinguishes_installed_from_observed(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract

    monkeypatch.setenv("VMLX_GLM5_FUSED_KDA_CONV", "1")
    contract = build_acceleration_contract(
        "glm5_next",
        {
            "features": {
                "projection_groups": {"installed": 80},
                "startup_warmup": {"installed": True, "observed_calls": 1},
                "kda_conv_state": {"installed": 34, "observed_calls": 0},
            }
        },
    )
    rows = _rows(contract)

    assert rows["projection_groups"]["state"] == "installed_unobserved"
    assert rows["startup_warmup"]["state"] == "active_observed"
    assert rows["kda_conv_state"]["state"] == "installed_unobserved"
    assert contract["summary"] == {
        "requested": 9,
        "installed": 3,
        "observed": 1,
        "source_only_or_unattested": 6,
    }


def test_dsv4_contract_preserves_production_defaults(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract

    monkeypatch.delenv("VMLX_DSV4_FUSED_MOE_PAIR", raising=False)
    monkeypatch.delenv("VMLX_DSV4_INDEXER_KERNEL", raising=False)
    monkeypatch.delenv("VMLX_DSV4_LM_HEAD_MODE", raising=False)
    monkeypatch.delenv("VMLX_DSV4_ROPE_CACHE", raising=False)
    contract = build_acceleration_contract("deepseek_v4")
    rows = _rows(contract)

    assert rows["runtime_patch"]["requested"] is True
    assert rows["indexer_scores"]["requested"] is False
    assert rows["fused_moe_pair"]["requested"] is True
    assert rows["lm_head"]["selection"] == "qmm"
    assert rows["lm_head"]["requested"] is True
    assert rows["rope_cache"]["requested"] is True
    assert rows["affine_moe"]["requested"] is False


def test_dsv4_indexer_candidate_is_prefill_only(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract

    monkeypatch.setenv("VMLX_DSV4_INDEXER_KERNEL", "1")
    rows = _rows(build_acceleration_contract("deepseek_v4"))

    assert rows["indexer_scores"]["requested"] is True
    assert rows["indexer_scores"]["scopes"] == ["prefill"]
    assert "indexer_scores" not in _rows(
        build_acceleration_contract("qwen4_exp")
    )


def test_model_attestation_merges_and_returns_a_copy():
    from vmlx_engine.acceleration_contract import (
        find_acceleration_attestation,
        record_acceleration_attestation,
    )

    model = SimpleNamespace()
    record_acceleration_attestation(
        model,
        "glm5_next",
        {"projection_groups": {"installed": 79}},
    )
    record_acceleration_attestation(
        model,
        "glm5_next",
        {
            "projection_groups": {"observed_calls": 2},
            "startup_warmup": {"installed": True, "observed_calls": 1},
        },
    )

    found = find_acceleration_attestation([object(), model])
    assert found["family"] == "glm5_next"
    assert found["features"]["projection_groups"] == {
        "installed": 79,
        "observed_calls": 2,
    }
    found["features"]["projection_groups"]["installed"] = 0
    assert model._vmlx_acceleration_attestation["features"]["projection_groups"][
        "installed"
    ] == 79


def test_server_acceleration_surface_embeds_family_contract(monkeypatch, tmp_path):
    import vmlx_engine.server as server

    (tmp_path / "config.json").write_text(
        '{"model_type":"glm5_next","quantization":{"bits":4,"group_size":128}}'
    )
    monkeypatch.setattr(
        server,
        "_mlx_metal_na_status",
        lambda: {"available": True, "nax_symbols": 1, "naxtile_symbols": 1},
    )
    monkeypatch.setattr(
        server,
        "_host_supports_metal_na",
        lambda: {"supported": True, "brand": "Apple M5 Max"},
    )
    monkeypatch.setattr(
        server,
        "_loaded_acceleration_attestation",
        lambda: {
            "family": "glm5_next",
            "features": {"startup_warmup": {"installed": True, "observed_calls": 1}},
        },
    )

    status = server._model_acceleration_status(str(tmp_path))

    assert status["metal_na_active_on_host"] is True
    family = status["family_runtime"]
    assert family["schema"] == "vmlx-runtime-acceleration-v1"
    assert family["family"] == "glm5_next"
    assert _rows(family)["startup_warmup"]["state"] == "active_observed"


def test_server_reports_observed_qwen_kernel_only_after_qualifying_call(
    monkeypatch, tmp_path
):
    import vmlx_engine.server as server
    from vmlx_engine.metal import gdn_conv_decode

    (tmp_path / "config.json").write_text('{"model_type":"qwen4_exp"}')
    monkeypatch.setenv("VMLX_QWEN4_FUSED_GDN_CONV", "1")
    monkeypatch.setattr(server, "_loaded_acceleration_attestation", lambda: None)
    monkeypatch.setattr(gdn_conv_decode, "_OBSERVED", False)

    before = server._family_acceleration_contract(str(tmp_path))
    assert _rows(before)["gdn_conv_state"]["state"] == "configured_unattested"

    monkeypatch.setattr(gdn_conv_decode, "_OBSERVED", True)
    after = server._family_acceleration_contract(str(tmp_path))
    assert _rows(after)["gdn_conv_state"]["state"] == "active_observed"
    assert _rows(after)["gdn_conv_state"]["runtime"]["observed_calls"] == 1


def test_dsv4_indexer_runtime_keeps_decode_fallback_separate_from_hits():
    import vmlx_engine.server as server

    status = server._dsv4_indexer_runtime_feature(
        {
            "self_test": "passed",
            "calls": 588,
            "reason": None,
            "declined": "tile work below kernel threshold",
        }
    )

    assert status == {
        "installed": True,
        "observed_calls": 588,
        "reason": None,
        "last_fallback_reason": "tile work below kernel threshold",
    }
