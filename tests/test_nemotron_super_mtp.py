"""Nemotron-3.5-Super-VL native MTP adapter (models/nemotron_bolt_vl.py) + the atomic MTP rollback in the MLLM lane."""
import json

import mlx.core as mx
import mlx.nn as nn
import pytest

pytest.importorskip("jang_tools.nemotron_mtp")


def _args():
    from mlx_lm.models.nemotron_h import ModelArgs

    return ModelArgs(
        model_type="nemotron_h", vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        max_position_embeddings=256, num_attention_heads=4, num_key_value_heads=2, attention_bias=False,
        mamba_num_heads=4, mamba_head_dim=8, mamba_proj_bias=False, ssm_state_size=64, conv_kernel=4, n_groups=1,
        mlp_bias=False, layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True, hybrid_override_pattern=["M", "*", "E"],
        head_dim=8, moe_intermediate_size=32, moe_shared_expert_intermediate_size=32, moe_latent_size=32, n_group=1,
        n_routed_experts=4, n_shared_experts=1, topk_group=1, num_experts_per_tok=2, norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )


def _lm():
    from jang_tools.nemotron_mtp import NemotronHMTP
    from mlx_lm.models.nemotron_h import Model as NemotronH

    from vmlx_engine.models.nemotron_bolt_vl import LanguageModel

    mx.random.seed(0)
    inner = NemotronH(_args())
    head = NemotronHMTP(inner.args)
    mx.eval(inner.parameters(), head.parameters())
    return LanguageModel(inner, head)


def test_quantization_inference_uses_the_true_input_width():
    from vmlx_engine.models.nemotron_bolt_vl import _quantization_from_tensors

    w = {"p.weight": mx.zeros((8, 256), dtype=mx.uint32), "p.scales": mx.zeros((8, 32))}
    assert _quantization_from_tensors(w, "p", 1024) == (8, 32)
    assert _quantization_from_tensors(w, "p", 2048) == (4, 64)
    assert _quantization_from_tensors(w, "p", 4096) == (2, 128)
    assert _quantization_from_tensors({"q.weight": mx.zeros((8, 32))}, "q", 32) is None
    with pytest.raises(ValueError):
        _quantization_from_tensors(w, "p", 1000)


def test_return_hidden_is_post_norm_backbone_output():
    lm = _lm()
    ids = mx.array([[3, 7, 11, 5]])
    logits, h = lm(ids, cache=lm.make_cache(), return_hidden=True, n_confirmed=1)
    ref = lm(ids, cache=lm.make_cache()).logits
    assert mx.allclose(logits, ref, atol=1e-5).item()
    assert mx.allclose(lm.inner.lm_head(h), logits, atol=1e-5).item()


def test_mtp_forward_multi_pair_equals_one_pair_at_a_time():
    lm = _lm()
    ids = mx.array([[3, 7, 11, 5, 9]])
    _, hidden = lm(ids, cache=lm.make_cache(), return_hidden=True)
    nxt = mx.array([[7, 11, 5, 9]])
    batch_logits = lm.mtp_forward(hidden[:, :4], nxt, lm.make_mtp_cache())
    cache = lm.make_mtp_cache()
    single = mx.concatenate([lm.mtp_forward(hidden[:, i:i + 1], nxt[:, i:i + 1], cache) for i in range(4)], axis=1)
    causal_err = mx.max(mx.abs(batch_logits - single)).item()      # row-count matmul numerics only (~1e-3)
    # control: the same pairs WITHOUT a causal mask (what jang_tools' __call__ does) — earlier positions see later
    # tokens, so the error must be far larger; proves the comparison can detect a missing mask
    entry, moe = lm.mtp.layers
    x = entry.fuse(lm.inner.backbone.embeddings(nxt), hidden[:, :4])
    unmasked = lm.inner.lm_head(moe.final_layernorm(moe(entry(x, mask=None, cache=lm.make_mtp_cache()[0]))))
    leak_err = mx.max(mx.abs(unmasked[:, :3] - single[:, :3])).item()
    assert causal_err < 1e-2 and leak_err > 20 * causal_err, (causal_err, leak_err)
    logits, h2 = lm.mtp_forward(hidden[:, :1], nxt[:, :1], lm.make_mtp_cache(), return_hidden=True)
    assert h2.shape == (1, 1, 32) and logits.shape == (1, 1, 97)


def test_head_loader_quantizes_each_module_as_stored(tmp_path):
    from jang_tools.nemotron_mtp import NemotronHMTP
    from mlx.utils import tree_flatten

    from vmlx_engine.models.nemotron_bolt_vl import load_mtp_head

    lm = _lm()
    src = NemotronHMTP(lm.inner.args)
    nn.quantize(src, group_size=32, bits=8, class_predicate=lambda p, m: p.endswith("eh_proj"))
    nn.quantize(src, group_size=32, bits=4, class_predicate=lambda p, m: p.endswith(("switch_mlp.fc1", "switch_mlp.fc2")))
    mx.eval(src.parameters())
    flat = {"mtp." + k: v for k, v in tree_flatten(src.parameters())}
    mx.save_safetensors(str(tmp_path / "model-00001-of-00001.safetensors"), flat)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in flat}}))
    head = load_mtp_head(lm.inner, tmp_path)
    assert head.layers[0].eh_proj.bits == 8 and head.layers[0].eh_proj.group_size == 32
    assert head.layers[1].mixer.switch_mlp.fc1.bits == 4
    assert isinstance(head.layers[0].mixer.q_proj, nn.Linear) and not hasattr(head.layers[0].mixer.q_proj, "bits")


class _KV:
    def __init__(self):
        self.trimmed = 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.trimmed += n


class _SSM:
    rollback_state = None
    rollback_to = None

    def is_trimmable(self):
        return False


def test_rollback_is_atomic_when_a_recurrent_layer_has_no_point():
    from vmlx_engine.mllm_batch_generator import _native_mtp_rollback_to_confirmed_impl

    kv1, kv2 = _KV(), _KV()
    assert _native_mtp_rollback_to_confirmed_impl([kv1, _SSM(), kv2], 1) is False
    assert kv1.trimmed == 0 and kv2.trimmed == 0             # nothing mutated: replay path trims exactly once
    assert _native_mtp_rollback_to_confirmed_impl([kv1, kv2], 2) is True
    assert kv1.trimmed == 2 and kv2.trimmed == 2


def test_mamba_rollback_points_match_a_clean_recompute():
    """Verify forward with n_confirmed=1 over [t0, d1, d2]: rollback_state == state after t0 alone, rollback_to(k) ==
    state after t0 + k drafts computed from scratch; logits identical to the unmarked forward."""
    from mlx_lm.models.cache import ArraysCache

    from vmlx_engine.models import nemotron_mamba_rollback as rb

    rb.apply()
    lm = _lm()
    prefix, block = mx.array([[3, 7, 11, 5]]), mx.array([[9, 2, 14]])

    def primed():
        c = lm.make_cache(); lm(prefix, cache=c); mx.eval([x for e in c for x in (e.state if isinstance(e, ArraysCache) else [e.keys])])
        return c

    c = primed()
    logits_m, _ = lm(block, cache=c, return_hidden=True, n_confirmed=1)
    c_plain = primed()
    logits_p = lm(block, cache=c_plain).logits
    assert mx.allclose(logits_m, logits_p, atol=2e-3).item()
    ssm = [e for e in c if isinstance(e, ArraysCache)]
    assert ssm and all(e.rollback_state is not None and e.rollback_to is not None for e in ssm)
    for k in (0, 1, 2):
        ref = primed()
        if k + 1:
            lm(block[:, :1 + k], cache=ref)
        ref_ssm = [e for e in ref if isinstance(e, ArraysCache)]
        for live, want in zip(ssm, ref_ssm):
            conv, st = live.rollback_state if k == 0 else live.rollback_to(k)
            assert mx.allclose(conv, want[0], atol=2e-3).item() and mx.allclose(st, want[1], atol=2e-3).item(), k
    # an unmarked forward clears stale points
    lm(mx.array([[4]]), cache=c)
    assert all(e.rollback_state is None and e.rollback_to is None for e in ssm)
