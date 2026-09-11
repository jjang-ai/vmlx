# SPDX-License-Identifier: Apache-2.0
"""ERNIE-4.5 MoE vendored text runtime — synthetic contract tests.

Real-bundle reference comparisons and numerical output-equivalence limits are
recorded separately in the introducing PR. These tests pin the source contract
on a tiny random model: registration, config parsing, the documented
departures from upstream mlx-lm (aux-loss-free routing bias kept for SELECTION only,
config-driven norm_min, pre-norm hidden for the native-MTP seam, MTP head attach + sanitize
renames), cache layout, and the engine-side detection helpers.
"""
import json
from unittest.mock import patch

import mlx.core as mx
import pytest

from vmlx_engine.models.ernie4_5.register import register_ernie4_5_runtime

TINY_CFG = {
    "model_type": "ernie4_5_moe",
    "hidden_size": 64,
    "intermediate_size": 96,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 256,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "use_bias": False,
    "tie_word_embeddings": True,
    "moe_num_experts": 8,
    "moe_k": 2,
    "moe_layer_start_index": 1,  # layer 0 dense, layer 1 MoE (as the real 28-layer model)
    "moe_layer_interval": 1,
    "moe_num_shared_experts": 1,
    "moe_norm_min": 1e-12,
    "num_nextn_predict_layers": 1,
}


@pytest.fixture(scope="module")
def ernie():
    register_ernie4_5_runtime()
    import mlx_lm.models.ernie4_5_moe as mod

    return mod


@pytest.fixture(scope="module")
def tiny_model(ernie):
    mx.random.seed(0)
    model = ernie.Model(ernie.ModelArgs.from_dict(TINY_CFG))
    model.eval()
    mx.eval(model.parameters())
    return model


class TestRegistration:
    def test_registers_vendored_module_over_upstream(self, ernie):
        assert ernie.__file__.replace("\\", "/").endswith(
            "vmlx_engine/models/ernie4_5/ernie4_5_moe.py"
        )
        # Idempotent: a second call keeps the same module object.
        register_ernie4_5_runtime()
        import mlx_lm.models.ernie4_5_moe as again

        assert again is ernie

    def test_args_parse_departures_from_real_config_keys(self, ernie):
        args = ernie.ModelArgs.from_dict(TINY_CFG)
        assert args.moe_norm_min == 1e-12
        assert args.num_nextn_predict_layers == 1
        assert args.moe_k == 2 and args.moe_num_experts == 8

    def test_model_configs_registry_row(self):
        import vmlx_engine.model_configs  # noqa: F401  (registers families)
        from vmlx_engine.model_config_registry import get_model_config_registry

        registry = get_model_config_registry()
        with patch(
            "vmlx_engine.model_config_registry.load_config",
            lambda _path: {"model_type": "ernie4_5_moe"},
        ):
            registry.clear_cache()
            cfg = registry.lookup("ERNIE-4.5-21B-A3B-PT")
        assert cfg.family_name == "ernie4_5"
        assert cfg.cache_type == "kv"
        assert cfg.eos_tokens == ["</s>", "<|end_of_sentence|>"]
        assert cfg.tool_parser is None and cfg.supports_native_tools is False
        assert cfg.reasoning_parser is None and cfg.supports_thinking is False
        assert cfg.is_mllm is False


class TestCacheAndForward:
    def test_cache_layout_is_plain_kv(self, tiny_model):
        from mlx_lm.models.cache import KVCache

        cache = tiny_model.make_cache()
        assert len(cache) == 2 and all(isinstance(c, KVCache) for c in cache)
        mtp_cache = tiny_model.make_mtp_cache()
        assert len(mtp_cache) == 1 and isinstance(mtp_cache[0], KVCache)
        assert tiny_model.mtp is not None

    def test_call_returns_pre_norm_hidden_and_logits_agree(self, tiny_model):
        x = mx.array([[3, 17, 42, 99, 7]])
        logits, hidden = tiny_model(x, return_hidden=True)
        assert logits.shape == (1, 5, TINY_CFG["vocab_size"])
        assert hidden.shape == (1, 5, TINY_CFG["hidden_size"])
        # hidden is PRE-norm: normalising it and projecting reproduces the logits.
        re_logits = tiny_model._logits(tiny_model.model.norm(hidden))
        assert mx.allclose(re_logits, logits, atol=1e-5).item()
        # ... and it is NOT already normalised (rms differs from 1 in general).
        rms = mx.sqrt(mx.mean(hidden.astype(mx.float32) ** 2, axis=-1))
        assert not mx.allclose(rms, mx.ones_like(rms), atol=1e-2).item()
        only_hidden = tiny_model(x, return_logits=False)
        assert mx.array_equal(only_hidden, hidden).item()

    def test_prefill_then_decode_shapes_and_offsets(self, tiny_model):
        cache = tiny_model.make_cache()
        x = mx.array([[3, 17, 42, 99]])
        logits = tiny_model(x, cache=cache)
        mx.eval(logits)
        assert logits.shape == (1, 4, TINY_CFG["vocab_size"])
        assert all(c.offset == 4 for c in cache)
        step = tiny_model(mx.array([[7]]), cache=cache)
        mx.eval(step)
        assert step.shape == (1, 1, TINY_CFG["vocab_size"])
        assert all(c.offset == 5 for c in cache)

    def test_mtp_forward_contract(self, tiny_model):
        x = mx.array([[3, 17, 42, 99, 7]])
        logits, hidden = tiny_model(x, return_hidden=True)
        mtp_cache = tiny_model.make_mtp_cache()
        # hidden at t + token t+1 -> distribution over token t+2
        head_logits = tiny_model.mtp_forward(hidden[:, :-1], x[:, 1:], mtp_cache)
        mx.eval(head_logits)
        assert head_logits.shape == (1, 4, TINY_CFG["vocab_size"])
        assert mtp_cache[0].offset == 4
        with_hidden = tiny_model.mtp_forward(
            hidden[:, -1:], x[:, -1:], mtp_cache, return_hidden=True
        )
        assert isinstance(with_hidden, tuple) and with_hidden[1].shape == (1, 1, TINY_CFG["hidden_size"])
        assert mtp_cache[0].offset == 5
        # Chained drafting feeds the returned hidden back in as the next "previous
        # hidden", which mtp_forward norms. So the returned hidden must be the head's
        # PRE-norm output: norm + project reproduces the logits, projecting it raw does not.
        chain_logits, chain_hidden = with_hidden
        assert mx.allclose(tiny_model._logits(tiny_model.model.norm(chain_hidden)), chain_logits, atol=1e-5).item()
        assert not mx.allclose(tiny_model._logits(chain_hidden), chain_logits, atol=1e-3).item()

    def test_router_bias_selects_experts_but_does_not_weight_them(self, ernie):
        """Departure 1 (aux-loss-free routing): e_score_correction_bias is added to the
        softmax probs ONLY to pick the top-k experts; the combine weights use the
        unbiased probs. A uniform bias therefore changes nothing, while a targeted
        bias changes which experts run and hence the output."""
        mx.random.seed(1)
        args = ernie.ModelArgs.from_dict(TINY_CFG)
        mlp = ernie.Ernie4_5_MoeMLP(args)
        mx.eval(mlp.parameters())
        x = mx.random.normal((1, 3, args.hidden_size))
        base = mlp(x)
        mx.eval(base)
        # Uniform shift: same selection, same weights -> identical output.
        mlp.e_score_correction_bias = mx.full((args.moe_num_experts,), 5.0, dtype=mx.float32)
        assert mx.allclose(mlp(x), base, atol=1e-6).item()
        # Force the two least-likely experts of token 0 into the top-k.
        mlp.e_score_correction_bias = mx.zeros((args.moe_num_experts,), dtype=mx.float32)
        probs = mlp.gate_act(mlp._router_logits(x))[0, 0]
        worst = mx.argsort(probs)[:2].tolist()
        bias = mx.zeros((args.moe_num_experts,), dtype=mx.float32)
        for e in worst:
            bias[e] = 10.0
        mlp.e_score_correction_bias = bias
        flipped = mlp(x)
        mx.eval(flipped)
        assert not mx.allclose(flipped[0, 0], base[0, 0], atol=1e-4).item()
        # Router logits are computed in float32 regardless of activation dtype.
        assert mlp._router_logits(x.astype(mx.float16)).dtype == mx.float32


class TestSanitize:
    def test_renames_ernie_mtp_keys_onto_the_head(self, tiny_model):
        w = {
            "model.mtp_block.0.self_attn.q_proj.weight": mx.zeros((1,)),
            "model.mtp_emb_norm.0.weight": mx.zeros((1,)),
            "model.mtp_hidden_norm.0.weight": mx.zeros((1,)),
            "model.mtp_linear_proj.0.weight": mx.zeros((1,)),
            "model.layers.0.self_attn.q_proj.weight": mx.zeros((1,)),
        }
        out = tiny_model.sanitize(w)
        assert set(out) == {
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.emb_norm.weight",
            "mtp.hidden_norm.weight",
            "mtp.linear_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        }

    def test_drops_mtp_keys_when_no_head_is_configured(self, ernie):
        cfg = dict(TINY_CFG, num_nextn_predict_layers=0)
        model = ernie.Model(ernie.ModelArgs.from_dict(cfg))
        assert not hasattr(model, "mtp") and model.make_mtp_cache() == []
        out = model.sanitize({"model.mtp_block.0.x": mx.zeros((1,)), "model.embed_tokens.weight": mx.zeros((1,))})
        assert set(out) == {"model.embed_tokens.weight"}

    def test_routing_bias_is_flattened_to_float32_on_the_moe_block(self, tiny_model):
        raw = mx.arange(8, dtype=mx.bfloat16).reshape(1, 8)  # HF stores [1, E]
        out = tiny_model.sanitize({"model.layers.1.mlp.moe_statics.e_score_correction_bias": raw})
        key = "model.layers.1.mlp.e_score_correction_bias"
        assert key in out and out[key].shape == (8,) and out[key].dtype == mx.float32

    def test_quantised_routing_bias_is_a_loud_converter_defect(self, tiny_model):
        with pytest.raises(ValueError, match="float32 passthrough"):
            tiny_model.sanitize({"model.layers.1.mlp.moe_statics.e_score_correction_bias.scales": mx.zeros((1,))})

    def test_stacks_experts_and_drops_tied_lm_head(self, tiny_model):
        w = {"lm_head.weight": mx.zeros((1,)), "lm_head.scales": mx.zeros((1,)), "lm_head.biases": mx.zeros((1,))}
        for e in range(TINY_CFG["moe_num_experts"]):
            for m in ("gate_proj", "up_proj"):
                w[f"model.layers.1.mlp.experts.{e}.{m}.weight"] = mx.zeros((32, 64))
            w[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = mx.zeros((64, 32))
        out = tiny_model.sanitize(w)
        assert not any(k.startswith("lm_head.") for k in out)  # weight AND quant sidecars
        assert out["model.layers.1.mlp.switch_mlp.gate_proj.weight"].shape == (8, 32, 64)
        assert out["model.layers.1.mlp.switch_mlp.down_proj.weight"].shape == (8, 64, 32)
        assert not any(".experts." in k for k in out)

    def test_stacks_quantised_expert_sidecars_too(self, tiny_model):
        w = {}
        for e in range(TINY_CFG["moe_num_experts"]):
            w[f"model.layers.1.mlp.experts.{e}.up_proj.weight"] = mx.zeros((32, 8), dtype=mx.uint32)
            w[f"model.layers.1.mlp.experts.{e}.up_proj.scales"] = mx.zeros((32, 1))
            w[f"model.layers.1.mlp.experts.{e}.up_proj.biases"] = mx.zeros((32, 1))
        out = tiny_model.sanitize(w)
        assert set(out) == {
            "model.layers.1.mlp.switch_mlp.up_proj.weight",
            "model.layers.1.mlp.switch_mlp.up_proj.scales",
            "model.layers.1.mlp.switch_mlp.up_proj.biases",
        }
        assert out["model.layers.1.mlp.switch_mlp.up_proj.scales"].shape == (8, 32, 1)

class TestWeightLoading:
    def _model(self, ernie, **overrides):
        return ernie.Model(ernie.ModelArgs.from_dict(dict(TINY_CFG, **overrides)))

    def _weights(self, model):
        from mlx.utils import tree_flatten
        return dict(tree_flatten(model.parameters()))

    @pytest.mark.parametrize("disabled_alias", ["VMLX_NATIVE_MTP", "VMLINUX_NATIVE_MTP"])
    @pytest.mark.parametrize("sharded", [False, True])
    @pytest.mark.parametrize("quantized", [False, True])
    def test_disabled_head_checkpoint_loads_backbone_only(
        self, ernie, monkeypatch, disabled_alias, sharded, quantized
    ):
        import mlx.nn as nn
        from types import SimpleNamespace
        from vmlx_engine.patches.mlx_lm_mtp.batch_generator import _is_mtp_eligible

        # The disable switch wins even when the other alias enables MTP.
        for alias in ("VMLX_NATIVE_MTP", "VMLINUX_NATIVE_MTP"):
            monkeypatch.setenv(alias, "1")
        source = self._model(ernie)
        if quantized:
            nn.quantize(source, group_size=32, bits=4)
        weights = self._weights(source)
        assert any(k.startswith("mtp.") for k in weights)

        monkeypatch.setenv(disabled_alias, "0")
        model = self._model(ernie)
        if quantized:
            nn.quantize(model, group_size=32, bits=4)
        assert not hasattr(model, "mtp")
        assert not _is_mtp_eligible(SimpleNamespace(model=model, uids=[0]))
        if sharded:
            for group in (
                {k: v for k, v in weights.items() if k.startswith("mtp.")},
                {k: v for k, v in weights.items() if not k.startswith("mtp.")},
            ):
                model.load_weights(list(model.sanitize(group).items()), strict=False)
            model.finalize_ernie_weight_loading()
        else:
            model.load_weights(list(model.sanitize(weights).items()))
        assert model.make_mtp_cache() == []
        loaded = self._weights(model)
        assert set(loaded) == {k for k in weights if not k.startswith("mtp.")}
        assert all(mx.array_equal(v, weights[k]).item() for k, v in loaded.items())
        inputs = mx.array([[3, 17, 42]])
        assert mx.array_equal(model(inputs), source(inputs)).item()

    def test_legacy_bundle_requires_explicit_bias_opt_in(self, ernie, monkeypatch, caplog):
        monkeypatch.delenv("VMLX_ERNIE45_ALLOW_MISSING_ROUTING_BIAS", raising=False)
        model = self._model(ernie)
        weights = {k: v for k, v in self._weights(model).items()
                   if not k.startswith("mtp.") and "e_score_correction_bias" not in k}
        sanitized = model.sanitize(weights)
        assert hasattr(model, "mtp")  # sanitization cannot infer absence
        with pytest.raises(ValueError, match="missing routing bias"):
            model.load_weights(list(sanitized.items()))
        monkeypatch.setenv("VMLX_ERNIE45_ALLOW_MISSING_ROUTING_BIAS", "1")
        model.load_weights(list(sanitized.items()))
        assert not hasattr(model, "mtp")
        assert mx.all(model.layers[1].mlp.e_score_correction_bias == 0).item()
        assert "legacy compatibility" in caplog.text and "AR-only" in caplog.text
        assert model(mx.array([[3, 17, 42]])).shape == (1, 3, TINY_CFG["vocab_size"])

    def test_no_head_with_valid_bias_loads_ar_without_compatibility_flag(self, ernie, monkeypatch):
        monkeypatch.delenv("VMLX_ERNIE45_ALLOW_MISSING_ROUTING_BIAS", raising=False)
        model = self._model(ernie)
        weights = {k: v for k, v in self._weights(model).items() if not k.startswith("mtp.")}
        model.load_weights(list(model.sanitize(weights).items()))
        assert not hasattr(model, "mtp")

    def test_shard_with_both_endpoints_never_detaches_head_or_overwrites_bias(self, ernie):
        source = self._model(ernie)
        full = self._weights(source)
        full["model.layers.1.mlp.e_score_correction_bias"] = mx.arange(8, dtype=mx.float32)
        endpoints = {k: full.pop(k) for k in (
            "model.embed_tokens.weight", "model.layers.1.input_layernorm.weight")}
        model = self._model(ernie)
        # Head and real bias are loaded BEFORE the misleading two-tensor shard.
        model.load_weights(list(model.sanitize(full).items()), strict=False)
        sanitized = model.sanitize(endpoints)
        assert set(sanitized) == set(endpoints) and hasattr(model, "mtp")
        model.load_weights(list(sanitized.items()), strict=False)
        model.finalize_ernie_weight_loading()
        assert hasattr(model, "mtp")
        assert mx.array_equal(model.layers[1].mlp.e_score_correction_bias, mx.arange(8)).item()

    @pytest.mark.parametrize("quantized", [False, True])
    def test_split_experts_and_sidecars_reach_the_actual_module(self, ernie, quantized):
        import mlx.nn as nn
        model = self._model(ernie)
        if quantized:
            nn.quantize(model, group_size=32, bits=4)
        full = self._weights(model)
        prefix = "model.layers.1.mlp.switch_mlp.up_proj."
        suffixes = ("weight", "scales", "biases") if quantized else ("weight",)
        stacks = {s: full.pop(prefix + s) for s in suffixes}
        model.load_weights(list(model.sanitize(full).items()), strict=False)
        wanted = {}
        for suffix in suffixes:
            shape = stacks[suffix].shape
            wanted[suffix] = mx.stack([mx.full(shape[1:], i + 10, dtype=stacks[suffix].dtype) for i in range(8)])
            # Reverse shard order also exercises groups that start without expert 0.
            for indices in (range(4, 8), range(4)):
                shard = {f"model.layers.1.mlp.experts.{i}.up_proj.{suffix}": wanted[suffix][i] for i in indices}
                out = model.sanitize(shard)
                assert not any(".experts." in k for k in out)
                model.load_weights(list(out.items()), strict=False)
        model.finalize_ernie_weight_loading()
        loaded = self._weights(model)
        for suffix in suffixes:
            assert mx.array_equal(loaded[prefix + suffix], wanted[suffix]).item()
        assert not model._ernie_pending_experts

    def test_incomplete_group_fails_at_end_of_load(self, ernie):
        model = self._model(ernie)
        full = self._weights(model)
        stack = full.pop("model.layers.1.mlp.switch_mlp.up_proj.weight")
        full["model.layers.1.mlp.experts.0.up_proj.weight"] = stack[0]
        model.load_weights(list(model.sanitize(full).items()), strict=False)
        with pytest.raises(ValueError, match="incomplete expert groups"):
            model.finalize_ernie_weight_loading()

    @pytest.mark.parametrize("missing", ["model.layers.0.self_attn.q_proj.weight", "mtp.linear_proj.weight"])
    def test_missing_backbone_or_partial_head_is_never_accepted(self, ernie, missing):
        model = self._model(ernie)
        full = self._weights(model)
        full.pop(missing)
        model.load_weights(list(model.sanitize(full).items()), strict=False)
        with pytest.raises(ValueError, match="missing parameters|incomplete MTP"):
            model.finalize_ernie_weight_loading()

    def test_partial_bias_loss_is_not_legacy_compatibility(self, ernie, monkeypatch):
        monkeypatch.setenv("VMLX_ERNIE45_ALLOW_MISSING_ROUTING_BIAS", "1")
        model = self._model(ernie, num_hidden_layers=3)
        full = self._weights(model)
        full.pop("model.layers.2.mlp.e_score_correction_bias")
        with pytest.raises(ValueError, match="missing routing bias"):
            model.load_weights(list(model.sanitize(full).items()))

    def test_strict_shape_validation_is_preserved(self, ernie):
        model = self._model(ernie)
        full = self._weights(model)
        full["model.layers.0.self_attn.q_proj.weight"] = mx.zeros((1,))
        with pytest.raises(ValueError, match="Expected shape"):
            model.load_weights(list(model.sanitize(full).items()))

    @pytest.mark.parametrize("quantized", [False, True])
    def test_real_mlx_loader_legacy_checkpoint_policy(self, ernie, tmp_path, monkeypatch, quantized):
        import mlx.nn as nn
        from mlx_lm.utils import load_model
        source = self._model(ernie)
        cfg = dict(TINY_CFG)
        if quantized:
            nn.quantize(source, group_size=32, bits=4)
            cfg["quantization"] = {"group_size": 32, "bits": 4}
        weights = {k: v for k, v in self._weights(source).items()
                   if not k.startswith("mtp.") and "e_score_correction_bias" not in k}
        mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        monkeypatch.delenv("VMLX_ERNIE45_ALLOW_MISSING_ROUTING_BIAS", raising=False)
        with pytest.raises(ValueError, match="missing routing bias"):
            load_model(tmp_path)
        monkeypatch.setenv("VMLX_ERNIE45_ALLOW_MISSING_ROUTING_BIAS", "1")
        loaded, _ = load_model(tmp_path)
        assert not hasattr(loaded, "mtp")
        assert mx.array_equal(source(mx.array([[3, 17]])), loaded(mx.array([[3, 17]]))).item()

    def test_public_jang_loader_finalizes_shard_loading(self, ernie, tmp_path, monkeypatch):
        from vmlx_engine import model_bundle_integrity
        from vmlx_engine.utils import jang_loader
        (tmp_path / "config.json").write_text(json.dumps(TINY_CFG))
        (tmp_path / "jang_config.json").write_text(json.dumps({"format": "jang", "version": 2}))
        monkeypatch.setattr(model_bundle_integrity, "prepare_model_bundle_for_load", lambda *a, **kw: (str(tmp_path), {}))
        model = self._model(ernie)
        weights = self._weights(model)
        weights.pop("model.layers.0.self_attn.q_proj.weight")
        model.load_weights(list(model.sanitize(weights).items()), strict=False)
        monkeypatch.setattr(jang_loader, "_load_jang_v2", lambda *a, **kw: (model, object()))
        with pytest.raises(ValueError, match="missing parameters.*q_proj"):
            jang_loader.load_jang_model(tmp_path)

    def test_real_mlx_loader_merges_shards_before_completing_load(self, ernie, tmp_path):
        from mlx_lm.utils import load_model
        source = self._model(ernie)
        full = self._weights(source)
        # Entire backbone in one shard, head in another: absence cannot be
        # inferred even when every backbone parameter is present in a shard.
        backbone = {k: v for k, v in full.items() if not k.startswith("mtp.")}
        head = {k: v for k, v in full.items() if k.startswith("mtp.")}
        mx.save_safetensors(str(tmp_path / "model-00001.safetensors"), backbone)
        mx.save_safetensors(str(tmp_path / "model-00002.safetensors"), head)
        (tmp_path / "config.json").write_text(json.dumps(TINY_CFG))
        loaded, _ = load_model(tmp_path)
        assert hasattr(loaded, "mtp")
        loaded.finalize_ernie_weight_loading()
        x = mx.array([[3, 17, 42]])
        assert mx.array_equal(source(x), loaded(x)).item()


class TestEngineDetection:
    def test_native_mtp_key_detection_sees_ernie_layout(self):
        from vmlx_engine.native_mtp import _mtp_keys_from_weight_keys, _mtp_layer_count_from_keys

        keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.mtp_block.0.self_attn.q_proj.weight",
            "model.mtp_emb_norm.0.weight",
            "model.mtp_linear_proj.0.weight",
            "model.layers.3.mlp.experts.0.mtp_unrelated.weight",
        ]
        found = _mtp_keys_from_weight_keys(keys)
        assert found == keys[1:4]
        assert _mtp_layer_count_from_keys(found) == 1

    def test_family_alias_and_default_depth(self, tmp_path, monkeypatch):
        from vmlx_engine.native_mtp import (
            _FAMILY_ALIAS,
            _RUNTIME_SUPPORTED_FAMILIES,
            native_mtp_effective_depth,
        )

        assert _FAMILY_ALIAS["ernie4_5_moe"] == "ernie4_5"
        assert "ernie4_5" in _RUNTIME_SUPPORTED_FAMILIES
        for name in ("VMLINUX_NATIVE_MTP_DEPTH", "VMLX_NATIVE_MTP_DEPTH"):
            monkeypatch.delenv(name, raising=False)
        (tmp_path / "config.json").write_text(json.dumps(TINY_CFG))
        # No sidecar, no stamp: the family default is the conservative depth 1.
        assert native_mtp_effective_depth(tmp_path) == (1, "family_default:ernie4_5")
        (tmp_path / "vmlx_mtp_tuning.json").write_text(
            json.dumps({"native_mtp": {"best_depth": 1, "validated": True}})
        )
        assert native_mtp_effective_depth(tmp_path) == (
            1,
            "vmlx_mtp_tuning.json:native_mtp.best_depth",
        )
