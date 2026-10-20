"""DSpark speculative decoding on MiniCPM-V-4.7: drafter contract, exact
verify + rollback, greedy equality with AR decoding, and serving routing."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from tests.test_minicpmv47_runtime import _randomize, _text_config
from vmlx_engine.dspark import DSparkDrafter, DSparkStats, generate_ar, generate_dspark, load_drafter
from vmlx_engine.dspark.generate import choose_width
from vmlx_engine.dspark.minicpm_target import MiniCPMDSparkTarget
from vmlx_engine.models.mllm import MLXMultimodalLM, _load_dspark_runtime

PROMPT = [5, 17, 300, 4000, 12, 9, 77, 1001, 3, 8, 42]  # 11 rows: not a multiple of any chunk below


@pytest.fixture(autouse=True)
def _row_exact_on(monkeypatch):
    monkeypatch.setenv("VMLX_ROW_EXACT_VERIFY_QMV", "1")


def _drafter_config(hidden: int = 16, taps=(0, 2, 3)) -> dict:
    vocab = _text_config(False)["vocab_size"]
    return {
        "architectures": ["Qwen3DSparkModel"],
        "model_type": "qwen3",
        "block_size": 8,
        "dflash_config": {
            "attention_mode": "gqa",
            "enable_confidence_head": True,
            "markov_head_type": "vanilla",
            "markov_rank": 8,
            "mask_token_id": 248076,
            "projector_type": "dspark",
            "target_layer_ids": list(taps),
        },
        "head_dim": 8,
        "hidden_size": hidden,
        "intermediate_size": 64,
        "layer_types": ["full_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default", "rope_dim": 8},
        "vocab_size": vocab,
        "draft_vocab_size": vocab,
        "markov_rank": 8,
        "enable_confidence_head": True,
    }


def _save_drafter(path, cfg: dict, seed: int = 0):
    path.mkdir(parents=True, exist_ok=True)
    model = DSparkDrafter(cfg)
    mx.random.seed(seed)
    flat = {k: (mx.random.normal(v.shape) * 0.05).astype(mx.bfloat16) for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(str(path / "model.safetensors"), flat)
    (path / "config.json").write_text(json.dumps(cfg))
    return path


def _gdn_text_config() -> dict:
    # The multi-row gated-delta Metal kernel needs key heads of >= 32 lanes.
    cfg = _text_config(False)
    cfg.update(linear_key_head_dim=32, linear_value_head_dim=32)
    return cfg


def _tiny_model():
    from mlx_vlm.models.minicpmv4_7 import TextConfig, VisionConfig

    from vmlx_engine.models.minicpmv4_7.config import ModelConfig
    from vmlx_engine.models.minicpmv4_7.minicpmv4_7 import Model

    config = ModelConfig.from_dict(
        {"model_type": "minicpmv4_7", "text_config": _gdn_text_config(), "vision_config": {},
         "insert_layer_id": 0, "downsample_mode": "16x"}
    )
    config.text_config = TextConfig.from_dict(_gdn_text_config())
    config.vision_config = VisionConfig.from_dict(
        {"hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2, "num_attention_heads": 2,
         "image_size": 56, "patch_size": 14}
    )
    return Model(config)


def _tiny_target(seed: int = 3, eval_mode: bool = True) -> MiniCPMDSparkTarget:
    model = _tiny_model()
    _randomize(model, seed=seed)
    model.train(not eval_mode)  # eval: Metal gated-delta kernel; train: ops recurrence
    return MiniCPMDSparkTarget(model)


# ------------------------------------------------------------------ drafter
@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["dflash_config"].update(projector_type="dflash"),
        lambda c: c["dflash_config"].update(causal=True),
        lambda c: c["dflash_config"].update(markov_head_type="gated"),
        lambda c: c.update(layer_types=["sliding_attention", "full_attention"]),
        lambda c: c["rope_parameters"].update(rope_dim=4),
        lambda c: c.update(draft_vocab_size=32000),
        lambda c: (c.pop("enable_confidence_head"), c["dflash_config"].pop("enable_confidence_head")),
    ],
)
def test_drafter_rejects_unsupported_variants(mutate):
    cfg = _drafter_config()
    mutate(cfg)
    with pytest.raises(ValueError, match="DSpark drafter not supported"):
        DSparkDrafter(cfg)


def test_prequantized_drafter_equals_load_time_quantization(tmp_path):
    pytest.importorskip("jang_tools.format.aligned_safetensors")
    from vmlx_engine.dspark import write_quantized_drafter

    src = _save_drafter(tmp_path / "bf16", _drafter_config(hidden=64))
    dst = write_quantized_drafter(src, tmp_path / "q8")
    at_load = dict(tree_flatten(load_drafter(src, bits=8).parameters()))
    stored = dict(tree_flatten(load_drafter(dst).parameters()))
    assert at_load.keys() == stored.keys()
    assert all(bool(mx.array_equal(at_load[k], stored[k]).item()) for k in at_load)
    assert any(k.endswith(".scales") for k in stored)
    assert not any(k.endswith(".scales") for k in stored if k.startswith(("markov_head", "confidence_head")))
    assert json.loads((dst / "config.json").read_text())["quantization"]["bits"] == 8
    with pytest.raises(FileExistsError):
        write_quantized_drafter(src, dst)


def test_choose_width_maximizes_expected_tokens_per_cost():
    assert choose_width([0.99] * 7, 7) == 7
    assert choose_width([0.0] * 7, 7) == 1
    # A confident pair followed by a near-certain miss stops at the pair.
    assert choose_width([0.95, 0.95, 0.05, 0.95, 0.95, 0.95, 0.95], 7) == 2
    assert choose_width([0.99] * 7, 3) == 3


# ------------------------------------------------------- exact verify + rollback
@pytest.mark.parametrize("eval_mode", [True, False])
@pytest.mark.parametrize("width", [2, 5, 7])
def test_verify_rows_and_rollback_equal_decode_steps(width, eval_mode):
    target = _tiny_target(eval_mode=eval_mode)
    prompt = mx.array([PROMPT])
    rows = mx.array([[11, 22, 33, 44, 55, 66, 77][:width]])
    follow = [mx.array([500]), mx.array([501])]
    for keep in sorted({1, max(1, width // 2), width}):
        cache = target.make_cache()
        target.prefill(prompt, cache, (), None)
        logits, _, sink = target.verify(rows, cache, ())
        target.rollback(cache, sink, kept=keep, total=width)
        after = [target.decode(t, cache) for t in follow]

        ref_cache = target.make_cache()
        target.prefill(prompt, ref_cache, (), None)
        ref_rows = [target.decode(rows[0, i:i + 1], ref_cache) for i in range(width)]
        ref_cache = target.make_cache()
        target.prefill(prompt, ref_cache, (), None)
        for i in range(keep):
            target.decode(rows[0, i:i + 1], ref_cache)
        ref_after = [target.decode(t, ref_cache) for t in follow]

        assert all(bool(mx.array_equal(logits[i], ref_rows[i]).item()) for i in range(width))
        assert all(bool(mx.array_equal(a, b).item()) for a, b in zip(after, ref_after)), keep


class _OracleDrafter(DSparkDrafter):
    """Drafts the true AR continuation, wrong at every fifth output position.

    Gives rounds that accept everything, nothing, and partial prefixes, so the
    loop's bookkeeping and the GDN rollback run on every acceptance count.
    """

    def __init__(self, cfg, reference, prompt_len):
        super().__init__(cfg)
        self.reference, self.prompt_len = reference, prompt_len

    def draft(self, block_hidden, anchor, lm_head, count, sampler=None):
        start = self.ctx_len - self.prompt_len + 1
        tokens = []
        for pos in range(start, start + count):
            tok = self.reference[pos] if pos < len(self.reference) else 7
            tokens.append(tok + 1 if pos % 5 == 4 else tok)
        return mx.array(tokens), mx.full((count,), 0.9, dtype=mx.float32)


def _oracle(tmp_path, reference) -> _OracleDrafter:
    src = _save_drafter(tmp_path / "d", _drafter_config())
    drafter = _OracleDrafter(_drafter_config(), reference, len(PROMPT))
    drafter.load_weights(tree_flatten(load_drafter(src, bits=16).parameters()), strict=True)
    return drafter


@pytest.mark.parametrize("prefill_step", [None, 4])
@pytest.mark.parametrize("fixed_width", [None, 1, 3, 7])
def test_dspark_greedy_equals_ar(tmp_path, prefill_step, fixed_width):
    target = _tiny_target()
    reference = generate_ar(target, PROMPT, 30, set(), prefill_step=prefill_step)
    drafter = _oracle(tmp_path, reference)
    stats = DSparkStats()
    out = generate_dspark(target, drafter, PROMPT, 30, set(), prefill_step=prefill_step,
                          fixed_width=fixed_width, stats=stats)
    assert out == reference
    assert stats.tokens == len(reference)
    if fixed_width != 1:
        assert stats.accepted > 0 and stats.accepted < stats.drafted


def test_dspark_stops_at_eos_like_ar(tmp_path):
    target = _tiny_target()
    free = generate_ar(target, PROMPT, 30, set())
    eos = {free[9]}
    reference = generate_ar(target, PROMPT, 30, eos)
    drafter = _oracle(tmp_path, free)
    assert generate_dspark(target, drafter, PROMPT, 30, eos, fixed_width=7) == reference


# ------------------------------------------------------------------ serving
def _bundle(tmp_path, *, spec=True, drafter=True, model_type="minicpmv4_7", hidden=16, default=None):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": model_type, "text_config": _gdn_text_config()}))
    stamp = {"chat": {}}
    if spec:
        stamp["speculative"] = {"method": "dspark", "drafter_path": "dspark"}
        if default is not None:
            stamp["speculative"]["default"] = default
    (root / "jang_config.json").write_text(json.dumps(stamp))
    if drafter:
        _save_drafter(root / "dspark", _drafter_config(hidden=hidden))
    return root


def test_load_dspark_runtime_contract(tmp_path, monkeypatch):
    model = _tiny_model()
    root = _bundle(tmp_path)
    runtime = _load_dspark_runtime(model, root)
    assert runtime is not None and runtime["max_draft"] == 7
    assert isinstance(runtime["target"], MiniCPMDSparkTarget)
    monkeypatch.setenv("VMLX_DSPARK", "0")
    assert _load_dspark_runtime(model, root) is None


@pytest.mark.parametrize("stamped", [1, 3, 99])
def test_stamped_max_draft_never_lowers_the_adaptive_width_ceiling(tmp_path, stamped):
    # The width is chosen per round; a bundle's speculative.max_draft stamp is advisory, never a hard depth.
    root = _bundle(tmp_path)
    stamp = json.loads((root / "jang_config.json").read_text())
    stamp["speculative"]["max_draft"] = stamped
    (root / "jang_config.json").write_text(json.dumps(stamp))
    assert _load_dspark_runtime(_tiny_model(), root)["max_draft"] == 7


@pytest.mark.parametrize(
    "kwargs, outcome",
    [
        ({"spec": False}, "engaged"),              # the drafter FILES decide; no stamp needed
        ({"model_type": "qwen3_5"}, None),
        ({"drafter": False}, None),                # declared but shipped without the drafter: AR, never an error
        ({"spec": False, "drafter": False}, None),
        ({"hidden": 32}, ValueError),
    ],
)
def test_load_dspark_runtime_follows_the_drafter_files(tmp_path, kwargs, outcome):
    root = _bundle(tmp_path, **kwargs)
    if outcome == "engaged":
        assert _load_dspark_runtime(_tiny_model(), root) is not None
    elif outcome is None:
        assert _load_dspark_runtime(_tiny_model(), root) is None
    else:
        with pytest.raises(outcome):
            _load_dspark_runtime(_tiny_model(), root)


def test_load_dspark_runtime_ignores_a_sidecar_that_does_not_declare_dspark(tmp_path):
    # Same rule as panel dsparkBundle.configDeclaresDspark, which picks the serving lane from it.
    root = _bundle(tmp_path)
    cfg = json.loads((root / "dspark" / "config.json").read_text())
    cfg["architectures"] = ["Qwen3ForCausalLM"]
    cfg["dflash_config"]["projector_type"] = "mlp"
    (root / "dspark" / "config.json").write_text(json.dumps(cfg))
    assert _load_dspark_runtime(_tiny_model(), root) is None


@pytest.mark.parametrize(
    "media, kwargs, routed",
    [
        ({}, {}, True),
        ({}, {"repetition_penalty": 1.0, "top_p": 0.9}, True),
        ({"images": ["x"]}, {}, True),            # planned by the caller (_dspark_media_plan) or native
        ({"videos": ["x"]}, {}, False),
        ({"audio": ["x"]}, {}, False),
        ({}, {"repetition_penalty": 1.1}, False),
        ({}, {"presence_penalty": 0.5}, False),
        ({}, {"logit_bias": {1: 2.0}}, False),
        ({}, {"stop": ["\n"]}, False),
        ({}, {"thinking_budget": 64}, False),
    ],
)
def test_requests_route_to_dspark(media, kwargs, routed):
    engine = SimpleNamespace(_dspark={"loaded": True})
    args = (media.get("images", []), media.get("videos", []), media.get("audio", []), 0.0, kwargs)
    assert MLXMultimodalLM._dspark_eligible(engine, *args) is routed


def test_no_dspark_route_without_runtime_or_row_exact_verify(monkeypatch):
    assert MLXMultimodalLM._dspark_eligible(SimpleNamespace(_dspark=None), [], [], [], 0.0, {}) is False
    monkeypatch.setenv("VMLX_ROW_EXACT_VERIFY_QMV", "0")
    assert MLXMultimodalLM._dspark_eligible(SimpleNamespace(_dspark={"x": 1}), [], [], [], 0.0, {}) is False


# ------------------------------------------------------------- MoE target (35B-A3B shape, tiny)
@pytest.fixture(autouse=True)
def _stock_switchglu(monkeypatch):
    """The exactness tests compare verify rows against stock 1-row SwitchGLU decode. MiMo-V2 capability discovery
    (server._mimo_v2_runtime_module, only for model_type mimo_v2) installs a process-global SwitchGLU decode fast
    path that other tests leave in place; a MiniCPM serving process never has it, so run these on the stock call."""
    from mlx_lm.models.switch_layers import SwitchGLU

    original = getattr(SwitchGLU, "_vmlx_mimo_original_affine_decode_call", None)
    if original is not None:
        monkeypatch.setattr(SwitchGLU, "__call__", original)
        monkeypatch.setattr(SwitchGLU, "_vmlx_mimo_affine_decode_fast_path", False)
    yield


def _moe_text_config() -> dict:
    # Quantizable expert shapes (64) and 16 experts x top-8: an 8-row verify reaches SwitchGLU's 64-index sort
    # threshold, where the stock multi-row block switches kernels (the case the per-row MoE verify exists for).
    cfg = _text_config(True)
    cfg.update(hidden_size=64, intermediate_size=64, moe_intermediate_size=64, shared_expert_intermediate_size=64,
               num_experts=16, num_experts_per_tok=8, linear_key_head_dim=32, linear_value_head_dim=32)
    return cfg


def _tiny_moe_target(seed: int = 5, eval_mode: bool = True) -> MiniCPMDSparkTarget:
    import mlx.nn as nn
    from mlx_vlm.models.minicpmv4_7 import TextConfig, VisionConfig

    from vmlx_engine.models.minicpmv4_7.config import ModelConfig
    from vmlx_engine.models.minicpmv4_7.minicpmv4_7 import Model

    config = ModelConfig.from_dict(
        {"model_type": "minicpmv4_7", "text_config": _moe_text_config(), "vision_config": {},
         "insert_layer_id": 0, "downsample_mode": "16x"}
    )
    config.text_config = TextConfig.from_dict(_moe_text_config())
    config.vision_config = VisionConfig.from_dict(
        {"hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2, "num_attention_heads": 2,
         "image_size": 56, "patch_size": 14}
    )
    model = Model(config)
    _randomize(model, seed=seed)
    # Routed experts affine 8 (the bundles' JANG_8M layout); routers stay unquantized like the bundles' bf16 routers.
    nn.quantize(model.language_model, group_size=64, bits=8,
                class_predicate=lambda path, m: "switch_mlp" in path and hasattr(m, "to_quantized"))
    model.set_dtype(mx.bfloat16)
    model.train(not eval_mode)
    mx.eval(model.parameters())
    return MiniCPMDSparkTarget(model)


def _verify_vs_decode(target, width):
    prompt = mx.array([PROMPT])
    rows = mx.array([[11, 22, 33, 44, 55, 66, 77, 88][:width]])
    cache = target.make_cache()
    target.prefill(prompt, cache, (), None)
    logits, _, sink = target.verify(rows, cache, ())
    ref_cache = target.make_cache()
    target.prefill(prompt, ref_cache, (), None)
    ref_rows = [target.decode(rows[0, i:i + 1], ref_cache) for i in range(width)]
    return [bool(mx.array_equal(logits[i], ref_rows[i]).item()) for i in range(width)], cache, sink, rows


@pytest.mark.parametrize("eval_mode", [True, False])
@pytest.mark.parametrize("width", [2, 5, 7, 8])
def test_moe_verify_rows_and_rollback_equal_decode_steps(width, eval_mode):
    target = _tiny_moe_target(eval_mode=eval_mode)
    assert target.moe
    follow = [mx.array([500]), mx.array([501])]
    for keep in sorted({1, max(1, width // 2), width}):
        same, cache, sink, rows = _verify_vs_decode(target, width)
        assert all(same), same
        target.rollback(cache, sink, kept=keep, total=width)
        after = [target.decode(t, cache) for t in follow]
        ref_cache = target.make_cache()
        target.prefill(mx.array([PROMPT]), ref_cache, (), None)
        for i in range(keep):
            target.decode(rows[0, i:i + 1], ref_cache)
        ref_after = [target.decode(t, ref_cache) for t in follow]
        assert all(bool(mx.array_equal(a, b).item()) for a, b in zip(after, ref_after)), keep


def test_moe_control_stock_multi_row_block_is_not_row_exact_at_eight_rows():
    """Fail-closed control: the per-row MoE route is what makes 8-row verify exact on this shape."""
    target = _tiny_moe_target()
    target.moe = False                      # stock multi-row sparse MoE block inside verify
    same, *_ = _verify_vs_decode(target, 8)
    assert not all(same), "the control did not reproduce the stock block's 8-row mismatch; the MoE test would be vacuous"


@pytest.mark.parametrize("fixed_width", [None, 3, 7])
def test_moe_dspark_greedy_equals_ar(tmp_path, fixed_width):
    target = _tiny_moe_target()
    reference = generate_ar(target, PROMPT, 30, set())
    src = _save_drafter(tmp_path / "d", _drafter_config(hidden=64))
    drafter = _OracleDrafter(_drafter_config(hidden=64), reference, len(PROMPT))
    drafter.load_weights(tree_flatten(load_drafter(src, bits=16).parameters()), strict=True)
    stats = DSparkStats()
    out = generate_dspark(target, drafter, PROMPT, 30, set(), fixed_width=fixed_width, stats=stats)
    assert out == reference
    assert stats.accepted > 0 and stats.accepted < stats.drafted


@pytest.mark.parametrize("default, env, engaged", [
    (None, None, True), (True, None, True), (False, None, True), (False, "1", True), (True, "0", False),
    (None, "0", False),
])
def test_load_dspark_runtime_default_on_when_the_drafter_ships(tmp_path, monkeypatch, default, env, engaged):
    """A shipped drafter is used by default whatever the stamp says; VMLX_DSPARK=0 (serve --dspark off) opts out."""
    if env is None:
        monkeypatch.delenv("VMLX_DSPARK", raising=False)
    else:
        monkeypatch.setenv("VMLX_DSPARK", env)
    root = _bundle(tmp_path, default=default)
    runtime = _load_dspark_runtime(_tiny_model(), root)
    assert (runtime is not None) == engaged


def test_moe_verify_routes_below_and_at_the_expert_sort_threshold(monkeypatch):
    """Stock multi-row block below rows x top_k = 64 (one-row expert kernels, cheaper), one-row blocks at it."""
    from vmlx_engine.dspark import minicpm_target as T

    target = _tiny_moe_target()
    assert target.preferred_max_draft == 6                    # 7 rows x top-8 = 56 < 64
    block = target.nm.layers[0].mlp
    assert T._moe_rows_stay_on_decode_kernels(block, 7) and not T._moe_rows_stay_on_decode_kernels(block, 8)
    calls = []
    real = type(block).__call__

    def spy(self, x):
        calls.append(int(x.shape[1]))
        return real(self, x)
    monkeypatch.setattr(type(block), "__call__", spy)
    rows = mx.array([[11, 22, 33, 44, 55, 66, 77, 88]])
    for width, expected_calls in ((7, [7]), (8, [1] * 8)):
        cache = target.make_cache()
        target.prefill(mx.array([PROMPT]), cache, (), None)
        calls.clear()
        logits, _, _ = target.verify(rows[:, :width], cache, ())
        mx.eval(logits)
        assert calls == expected_calls * len(target.nm.layers), (width, calls[:10])
    same7, *_ = _verify_vs_decode(target, 7)
    same8, *_ = _verify_vs_decode(target, 8)
    assert all(same7) and all(same8)


# ------------------------------------------------------------- AR fallback (measured-cost policy)
def test_ar_step_equals_decode_and_returns_taps():
    for make in (_tiny_target, _tiny_moe_target):
        target = make()
        taps = (0, 2, 3)
        c1, c2 = target.make_cache(), target.make_cache()
        target.prefill(mx.array([PROMPT]), c1, taps, None)
        target.prefill(mx.array([PROMPT]), c2, taps, None)
        for tok in (500, 501, 502):
            a, feats = target.ar_step(mx.array([tok]), c1, taps)
            b = target.decode(mx.array([tok]), c2)
            assert bool(mx.array_equal(a, b).item()) and feats.shape[:2] == (1, 1)


class _ScriptedFallback:
    """Forces a fixed action sequence (True = AR step) to exercise every interleaving."""

    def __init__(self, plan):
        self.plan, self.i = list(plan), 0

    def next_is_ar(self):
        out = self.plan[self.i % len(self.plan)]
        self.i += 1
        return out

    def record_draft(self, tokens, seconds, rows=None):
        pass

    def record_ar(self, seconds):
        pass

    def burst_size(self):
        return 3

    def record_ar_burst(self, steps, seconds):
        pass


@pytest.mark.parametrize("plan", [[False, True], [True, True, False], [False, False, True, True, True]])
@pytest.mark.parametrize("moe", [False, True])
def test_dspark_with_ar_steps_interleaved_equals_ar(tmp_path, plan, moe):
    target = _tiny_moe_target() if moe else _tiny_target()
    hidden = 64 if moe else 16
    reference = generate_ar(target, PROMPT, 30, set())
    src = _save_drafter(tmp_path / "d", _drafter_config(hidden=hidden))
    drafter = _OracleDrafter(_drafter_config(hidden=hidden), reference, len(PROMPT))
    drafter.load_weights(tree_flatten(load_drafter(src, bits=16).parameters()), strict=True)
    stats = DSparkStats()
    out = generate_dspark(target, drafter, PROMPT, 30, set(), stats=stats, ar_fallback=_ScriptedFallback(plan))
    assert out == reference
    assert stats.ar_tokens > 0 and stats.rounds > 0


def test_ar_fallback_policy_switches_on_measured_rates():
    from vmlx_engine.dspark.generate import ARFallback

    p = ARFallback()
    assert not p.next_is_ar()                     # first action: a draft round
    p.record_draft(tokens=2, seconds=0.030)       # ~67 tok/s drafting
    assert p.next_is_ar()                         # second action: one AR step seeds its cost
    p.record_ar(0.010)                            # 100 tok/s AR > 67 * 1.05 -> burst
    assert p.next_is_ar() and p.ar_left == p.burst
    actions = []
    for _ in range(p.probe_every + 2):            # AR steps until the probe is due, then exactly one draft round
        is_ar = p.next_is_ar()
        actions.append(is_ar)
        if is_ar:
            p.record_ar(0.010)
        else:
            p.record_draft(tokens=2, seconds=0.030)
    assert False in actions, "the policy never re-probed drafting"
    first_draft = actions.index(False)
    assert all(actions[:first_draft]) and actions[first_draft + 1] is True   # back to AR after one probe round
    q = ARFallback()
    q.record_draft(tokens=6, seconds=0.030)       # 200 tok/s drafting
    q.ar_seconds = 0.010
    assert not q.next_is_ar()                     # drafting pays: stay on draft rounds


def test_drafter_cache_context_equals_concatenated_reference(tmp_path):
    """The KVCache-backed context (append + trim of the block rows) gives the same block hidden states as attending
    over an explicit concatenation, across two add_context calls and two consecutive blocks."""
    import mlx.nn as nn

    src = _save_drafter(tmp_path / "d", _drafter_config(hidden=16))
    drafter = load_drafter(src, bits=16)
    mx.random.seed(4)
    feats1 = mx.random.normal((1, 9, 3 * 16)).astype(mx.bfloat16)
    feats2 = mx.random.normal((1, 5, 3 * 16)).astype(mx.bfloat16)
    block = mx.random.normal((1, drafter.block, 16)).astype(mx.bfloat16)

    def reference(feature_chunks, blk):
        ks, vs, pos = [[] for _ in drafter.layers], [[] for _ in drafter.layers], 0
        for f in feature_chunks:
            ctx = drafter.hidden_norm(drafter.fc(f))
            n = int(f.shape[1])
            for li, layer in enumerate(drafter.layers):
                a = layer.self_attn
                k = a.k_norm(a.k_proj(ctx).reshape(1, n, a.kv_heads, a.head_dim)).transpose(0, 2, 1, 3)
                ks[li].append(drafter._rope(k, pos))
                vs[li].append(a.v_proj(ctx).reshape(1, n, a.kv_heads, a.head_dim).transpose(0, 2, 1, 3))
            pos += n
        h, L = blk, int(blk.shape[1])
        for li, layer in enumerate(drafter.layers):
            a = layer.self_attn
            x = layer.input_layernorm(h)
            q = drafter._rope(a.q_norm(a.q_proj(x).reshape(1, L, a.heads, a.head_dim)).transpose(0, 2, 1, 3), pos)
            k = drafter._rope(a.k_norm(a.k_proj(x).reshape(1, L, a.kv_heads, a.head_dim)).transpose(0, 2, 1, 3), pos)
            v = a.v_proj(x).reshape(1, L, a.kv_heads, a.head_dim).transpose(0, 2, 1, 3)
            k = mx.concatenate(ks[li] + [k], axis=2)
            v = mx.concatenate(vs[li] + [v], axis=2)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.head_dim**-0.5, mask=None)
            h = h + a.o_proj(o.transpose(0, 2, 1, 3).reshape(1, L, -1))
            y = layer.post_attention_layernorm(h)
            h = h + layer.mlp.down_proj(nn.silu(layer.mlp.gate_proj(y)) * layer.mlp.up_proj(y))
        return drafter.norm(h)

    drafter.reset()
    drafter.add_context(feats1)
    first = drafter(block)
    mx.eval(first)
    drafter.add_context(feats2)
    second = drafter(block)
    mx.eval(second)
    assert drafter.ctx_len == 14 and drafter.context_caches()[0].offset == 14
    for got, ref in ((first, reference([feats1], block)), (second, reference([feats1, feats2], block))):
        diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item()
        assert diff <= 2e-2 * (mx.abs(ref.astype(mx.float32)).max().item() + 1e-6), diff


# ------------------------------------------------------------- multi-turn reuse (dspark/session.py)
def _drafter_for(target, tmp_path, reference, hidden):
    src = _save_drafter(tmp_path / "d", _drafter_config(hidden=hidden))
    drafter = _OracleDrafter(_drafter_config(hidden=hidden), reference, len(PROMPT))
    drafter.load_weights(tree_flatten(load_drafter(src, bits=16).parameters()), strict=True)
    return drafter


def _offset(target, cache):
    from vmlx_engine.dspark.minicpm_target import _cache_offset
    return _cache_offset(cache, target.nm.fa_idx)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("plan", [None, [False, True], [True, True, False]])
def test_session_state_tracks_exactly_what_the_caches_hold(tmp_path, moe, plan):
    from vmlx_engine.dspark.generate import iter_dspark
    target = _tiny_moe_target() if moe else _tiny_target()
    reference = generate_ar(target, PROMPT, 30, set())
    drafter = _drafter_for(target, tmp_path, reference, 64 if moe else 16)
    eos = {reference[13]}                                     # a stop inside the run (exercises an EOS mid-burst)
    session = {}
    out = []
    for chunk in iter_dspark(target, drafter, PROMPT, 30, eos, session=session,
                             ar_fallback=_ScriptedFallback(plan) if plan else None):
        out.extend(chunk)
    fw = session["forwarded"]
    assert fw[: len(PROMPT)] == PROMPT and fw[len(PROMPT):] == out[: len(fw) - len(PROMPT)]
    assert _offset(target, session["target_cache"]) == len(fw) == drafter.ctx_len


@pytest.mark.parametrize("moe", [False, True])
def test_resume_from_a_snapshot_equals_ar_with_the_same_segments(tmp_path, moe):
    """A cold request snapshots at a cut; a second prompt sharing that prefix resumes from the snapshot and must equal
    AR run with the same prefill segmentation."""
    from mlx_lm.models.cache import KVCache
    from vmlx_engine.dspark import session as S
    from vmlx_engine.dspark.generate import iter_dspark
    target = _tiny_moe_target() if moe else _tiny_target()
    cut = 6
    reference = generate_ar(target, PROMPT, 20, set(), prefill_cuts=[cut])
    drafter = _drafter_for(target, tmp_path, reference, 64 if moe else 16)
    snaps = []
    make_draft = lambda: [KVCache() for _ in drafter.layers]
    for _ in iter_dspark(target, drafter, PROMPT, 20, set(), snapshot_cuts=[cut],
                         on_snapshot=lambda c, tc, dc: snaps.append((c, S.clone(tc, target.make_cache), S.clone(dc, make_draft)))):
        pass
    assert [c for c, *_ in snaps] == [cut] and snaps[0][1] is not None and snaps[0][2] is not None
    prompt2 = PROMPT[:cut] + [901, 902, 903, 904, 905]
    ref2 = generate_ar(target, prompt2, 20, set(), prefill_cuts=[cut])
    drafter2 = _drafter_for(target, tmp_path / "2", ref2, 64 if moe else 16)
    drafter2.prompt_len = len(prompt2)
    resume = {"target_cache": snaps[0][1], "draft_cache": snaps[0][2], "cache_len": cut}
    stats = DSparkStats()
    out = []
    for chunk in iter_dspark(target, drafter2, prompt2, 20, set(), resume=resume, stats=stats):
        out.extend(chunk)
    assert out == ref2 and stats.cached_tokens == cut and stats.prefill_tokens == len(prompt2) - cut


@pytest.mark.parametrize("moe", [False, True])
def test_resume_from_the_end_of_turn_state_equals_ar(tmp_path, moe):
    from vmlx_engine.dspark.generate import iter_dspark
    target = _tiny_moe_target() if moe else _tiny_target()
    reference = generate_ar(target, PROMPT, 12, set())
    drafter = _drafter_for(target, tmp_path, reference, 64 if moe else 16)
    session = {}
    for _ in iter_dspark(target, drafter, PROMPT, 12, set(), session=session, ar_fallback=_ScriptedFallback([False, True])):
        pass
    fw = list(session["forwarded"])
    prompt2 = fw + [777, 778, 779]
    # Reference built the way the stored cache was: the prompt prefilled, every generated token DECODED (verify rows
    # and AR steps are bit-equal to decode steps), then the new turn prefilled and greedy decode.
    ref_cache = target.make_cache()
    target.prefill(mx.array([PROMPT]), ref_cache, (), None)
    for t in fw[len(PROMPT):]:
        target.decode(mx.array([t]), ref_cache)
    logits, _ = target.prefill(mx.array([prompt2[len(fw):]]), ref_cache, (), None)
    tok = mx.argmax(logits).reshape(1)
    ref2 = [int(tok.item())]
    while len(ref2) < 15:
        tok = mx.argmax(target.decode(tok, ref_cache)).reshape(1)
        ref2.append(int(tok.item()))
    drafter2 = _drafter_for(target, tmp_path / "2", ref2, 64 if moe else 16)
    drafter2.prompt_len = len(prompt2)
    resume = {"target_cache": session["target_cache"], "draft_cache": drafter.context_caches(), "cache_len": len(fw)}
    out = []
    for chunk in iter_dspark(target, drafter2, prompt2, 15, set(), resume=resume):
        out.extend(chunk)
    assert out == ref2


def test_session_store_round_trip_ram_tier(monkeypatch):
    from mlx_lm.models.cache import KVCache
    from vmlx_engine.dspark import session as S
    monkeypatch.setattr(S, "SSD", None)
    S.clear()
    target = _tiny_target()
    cache = target.make_cache()
    target.prefill(mx.array([PROMPT]), cache, (), None)
    entry = {"kind": "turn", "tokens": list(PROMPT), "cache_len": len(PROMPT), "target_cache": cache,
             "draft_cache": [KVCache()], "model_key": "k"}
    receipt = S.store(entry)
    assert receipt.done() and receipt.vmlx_lookup_served_from_ram
    assert S.lookup("other", PROMPT + [1], None, target.make_cache, list) is None      # model key guard
    hit = S.lookup("k", PROMPT + [1, 2], None, target.make_cache, list)
    assert hit is entry and hit["source"] == "dspark-ram"
    assert S.lookup("k", PROMPT + [1, 2], None, target.make_cache, list) is None        # ownership moved
    monkeypatch.setenv("VMLX_DSPARK_PREFIX_REUSE", "0")
    S.store(dict(entry))
    assert S.lookup("k", PROMPT + [1, 2], None, target.make_cache, list) is None
    S.clear()


# ------------------------------------------------------------- image requests (canvas M-RoPE + merged embeddings)
def _media_inputs(model):
    """One 8x8-patch crop (4 tokens in 16x) inside a text prompt, the same layout as the runtime tests."""
    from tests.test_minicpmv47_runtime import _image_prompt

    ids = _image_prompt(4)
    pixels = mx.random.normal((1, 3, 14, 64 * 14), key=mx.random.key(11))
    sizes = mx.array([[8, 8]], dtype=mx.int32)
    return ids, pixels, sizes


def _native_media_ar(model, ids, pixels, sizes, max_new, follow=()):
    """The native VLM route: the model's own forward with pixels (vision tower, scatter, whole-prompt canvas plan),
    then served decode steps (language-model positions = offset + rope delta). ``follow`` = (generated, new_text):
    decode those generated tokens, prefill the new text, and greedy-decode ``max_new`` more (the next turn)."""
    from mlx_vlm.models.cache import make_prompt_cache

    lm = model.language_model
    cache = make_prompt_cache(lm)
    logits = model(ids, pixel_values=pixels, target_sizes=sizes, cache=cache).logits[0, -1]
    if follow:
        generated, new_text = follow
        for t in generated:
            lm(mx.array([[t]]), cache=cache)
        logits = lm(mx.array([new_text]), cache=cache).logits[0, -1]
    tok = mx.argmax(logits).reshape(1)
    out = [int(tok.item())]
    while len(out) < max_new:
        tok = mx.argmax(lm(tok.reshape(1, 1), cache=cache).logits[0, -1]).reshape(1)
        out.append(int(tok.item()))
    lm.reset_position_state()
    return out


def _plan(model, ids, pixels, sizes):
    feats = model.get_input_embeddings(ids, pixels, target_sizes=sizes)
    lm = model.language_model
    positions, delta = lm._position_ids, int(lm._rope_deltas.reshape(-1)[0].item())
    lm.reset_position_state()
    return feats.inputs_embeds, positions, delta


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("fixed_width", [None, 1, 3, 7])
def test_media_dspark_greedy_equals_native_ar(tmp_path, moe, fixed_width):
    target = _tiny_moe_target() if moe else _tiny_target()
    ids, pixels, sizes = _media_inputs(target.model)
    reference = _native_media_ar(target.model, ids, pixels, sizes, 24)
    embeds, positions, delta = _plan(target.model, ids, pixels, sizes)
    assert delta != 0                       # the canvas plan compresses the image: text after it is offset by delta
    drafter = _drafter_for(target, tmp_path, reference, 64 if moe else 16)
    drafter.prompt_len = int(ids.shape[1])
    target.set_media(embeds, positions, delta)
    try:
        out = generate_dspark(target, drafter, ids[0].tolist(), 24, set(), fixed_width=fixed_width)
    finally:
        target.set_media()
    assert out == reference
    if fixed_width is None:
        # Control: the same request without the plan (placeholder embeddings, sequence positions) is NOT the answer.
        assert generate_dspark(target, drafter, ids[0].tolist(), 24, set()) != reference


@pytest.mark.parametrize("moe", [False, True])
def test_media_follow_up_turn_resumes_with_the_cached_rope_delta(tmp_path, moe):
    """Turn 2 resumes the end-of-turn state of an image turn: only the rope delta is set (no embeddings, vision tower
    skipped), and the new text + decode equal the native route that kept the same cache."""
    from vmlx_engine.dspark.generate import iter_dspark

    target = _tiny_moe_target() if moe else _tiny_target()
    ids, pixels, sizes = _media_inputs(target.model)
    prompt = ids[0].tolist()
    reference = _native_media_ar(target.model, ids, pixels, sizes, 10)
    embeds, positions, delta = _plan(target.model, ids, pixels, sizes)
    drafter = _drafter_for(target, tmp_path, reference, 64 if moe else 16)
    drafter.prompt_len = len(prompt)
    session = {}
    target.set_media(embeds, positions, delta)
    try:
        for _ in iter_dspark(target, drafter, prompt, 10, set(), session=session):
            pass
    finally:
        target.set_media()
    forwarded = list(session["forwarded"])
    new_text = [777, 778, 779]
    ref2 = _native_media_ar(target.model, ids, pixels, sizes, 12, follow=(forwarded[len(prompt):], new_text))
    drafter2 = _drafter_for(target, tmp_path / "2", ref2, 64 if moe else 16)
    drafter2.prompt_len = len(forwarded) + len(new_text)
    resume = {"target_cache": session["target_cache"], "draft_cache": drafter.context_caches(),
              "cache_len": len(forwarded)}
    import copy

    control_resume = {"target_cache": copy.deepcopy(session["target_cache"]),
                      "draft_cache": copy.deepcopy(drafter.context_caches()), "cache_len": len(forwarded)}
    target.set_media(rope_delta=delta)
    try:
        out = []
        for chunk in iter_dspark(target, drafter2, forwarded + new_text, 12, set(), resume=resume):
            out.extend(chunk)
    finally:
        target.set_media()
    assert out == ref2
    # Control: the same resume at delta 0 (positions as if the image took its token count) is NOT the answer.
    control = []
    for chunk in iter_dspark(target, drafter2, forwarded + new_text, 12, set(), resume=control_resume):
        control.extend(chunk)
    assert control != ref2


def test_salted_media_keys_follow_the_media_identity():
    from vmlx_engine.models.mllm import _dspark_salted_media_keys

    media = {248056}
    tokens = [1, 2, 248056, 248056, 248056, 9, 10]
    a = _dspark_salted_media_keys(tokens, media, "salt-a")
    assert a == _dspark_salted_media_keys(tokens, media, "salt-a")             # same media: identical keys
    b = _dspark_salted_media_keys(tokens, media, "salt-b")
    assert a[:2] == b[:2] == tokens[:2] and a[5:] == b[5:] == tokens[5:]       # text positions keep their ids
    assert all(a[i] != b[i] for i in (2, 3, 4))                                # other media: diverge at the first
    assert len(set(a[2:5])) == 3 and all(k >= 0x80000000 for k in a[2:5])      # per position, above any vocab id


# ------------------------------------------------------------- candidate-only Markov drafting
def _real_drafter(tmp_path, hidden=16):
    return load_drafter(_save_drafter(tmp_path / "real", _drafter_config(hidden=hidden)), bits=16)


def _draft_inputs(drafter, target, seed=2):
    block = mx.random.normal((1, drafter.block, target.lm_head.weight.shape[-1] if hasattr(target.lm_head, "weight")
                              else 16), key=mx.random.key(seed)).astype(mx.bfloat16)
    return drafter(block), mx.array([17])


def test_candidate_markov_equals_full_chain_when_the_bias_cannot_reorder(tmp_path):
    target = _tiny_target()
    drafter = _real_drafter(tmp_path)
    hidden, anchor = _draft_inputs(drafter, target)
    w2 = drafter.markov_head.markov_w2
    w2.weight = mx.zeros_like(w2.weight)                    # no Markov bias: argmax of the base logits
    full, full_conf = drafter.draft(hidden, anchor, target.lm_head, 7, candidates=0)
    cand, cand_conf = drafter.draft(hidden, anchor, target.lm_head, 7, candidates=4)
    assert full.tolist() == cand.tolist()
    assert mx.allclose(full_conf, cand_conf, atol=1e-5).item()


def test_candidate_markov_samples_within_the_candidates(tmp_path):
    from vmlx_engine.sampling import make_sampler

    target = _tiny_target()
    drafter = _real_drafter(tmp_path)
    hidden, anchor = _draft_inputs(drafter, target, seed=3)
    logits = target.lm_head(hidden[:, :7])[0].astype(mx.float32)
    top = mx.argpartition(-logits, kth=5, axis=-1)[:, :6]
    mx.random.seed(0)
    toks, _ = drafter.draft(hidden, anchor, target.lm_head, 7, sampler=make_sampler(temp=1.0), candidates=6)
    for i, t in enumerate(toks.tolist()):
        assert t in top[i].tolist()


@pytest.mark.parametrize("candidates", ["0", "32"])
def test_real_drafter_keeps_greedy_equal_to_ar(tmp_path, monkeypatch, candidates):
    monkeypatch.setenv("VMLX_DSPARK_MARKOV_CANDIDATES", candidates)
    target = _tiny_target()
    reference = generate_ar(target, PROMPT, 24, set())
    drafter = _real_drafter(tmp_path)
    assert generate_dspark(target, drafter, PROMPT, 24, set()) == reference



# ------------------------------------------------------------- width choice from measured round costs
def test_measured_round_costs_pick_a_narrower_width_than_the_static_model(monkeypatch):
    from vmlx_engine.dspark.generate import ARFallback, choose_width

    monkeypatch.setenv("VMLX_DSPARK_MEASURED_WIDTH", "1")
    conf = [0.8] * 6
    policy = ARFallback()
    assert policy.round_seconds(2) is None                         # nothing measured: the static model decides
    assert policy.choose_width(conf, 6, 0.12) == choose_width(conf, 6, 0.12) == 5     # E/(1 + 0.12 k) peaks at 5
    policy.ar_seconds = 0.01                                       # prior: 1 + 0.5 + 0.3 * (T - 1) AR steps
    assert abs(policy.round_seconds(4) - 0.01 * (1.5 + 0.9)) < 1e-12
    # Measured like the 35B-A3B: rows are expensive (2 rows 20 ms, 7 rows 45 ms; 4 rows interpolated to 30 ms).
    policy.record_draft(2, 0.020, rows=2)
    policy.record_draft(4, 0.045, rows=7)
    assert abs(policy.round_seconds(4) - 0.030) < 1e-12
    assert policy.choose_width(conf, 6, 0.12) == 3                 # E 2.95 / 30 ms beats 1.8 / 20 ms and 3.95 / 45 ms
    monkeypatch.setenv("VMLX_DSPARK_MEASURED_WIDTH", "0")
    assert policy.choose_width(conf, 6, 0.12) == 5                 # off: the static model
    monkeypatch.delenv("VMLX_DSPARK_MEASURED_WIDTH")
    assert policy.choose_width(conf, 6, 0.12) == 5                 # unset = static (the default)


def test_measured_width_choice_keeps_greedy_equal_to_ar(tmp_path):
    from vmlx_engine.dspark.generate import ARFallback

    target = _tiny_target()
    reference = generate_ar(target, PROMPT, 30, set())
    out = generate_dspark(target, _oracle(tmp_path, reference), PROMPT, 30, set(), ar_fallback=ARFallback())
    assert out == reference


# ------------------------------------------------------------- pipelined rounds (one sync per draft round)
@pytest.mark.parametrize("moe", [False, True])
def test_pipelined_rounds_keep_greedy_equal_to_ar(tmp_path, monkeypatch, moe):
    from vmlx_engine.dspark.generate import ARFallback

    monkeypatch.setenv("VMLX_DSPARK_PIPELINE", "1")
    target = _tiny_moe_target() if moe else _tiny_target()
    reference = generate_ar(target, PROMPT, 30, set())
    drafter = _drafter_for(target, tmp_path, reference, 64 if moe else 16)
    stats = DSparkStats()
    out = generate_dspark(target, drafter, PROMPT, 30, set(), ar_fallback=ARFallback(), stats=stats)
    assert out == reference
    assert stats.rounds > 0 and 0 < stats.accepted < stats.drafted       # partial acceptances, rollbacks exercised
    real = generate_dspark(target, _real_drafter(tmp_path, 64 if moe else 16), PROMPT, 30, set(), ar_fallback=ARFallback())
    assert real == reference


def test_planned_width_follows_the_acceptance_history(monkeypatch):
    from vmlx_engine.dspark.generate import ARFallback

    monkeypatch.setenv("VMLX_DSPARK_MEASURED_WIDTH", "1")
    policy = ARFallback()
    policy.ar_seconds = 0.01
    policy.record_draft(3, 0.020, rows=2)
    policy.record_draft(3, 0.045, rows=7)
    for _ in range(20):
        policy.record_acceptance(6, 6)                 # every draft accepted
    wide = policy.planned_width(6, 0.12)
    for _ in range(40):
        policy.record_acceptance(6, 0)                 # first draft always rejected
    assert policy.planned_width(6, 0.12) == 1 < wide



def _video_inputs():
    """One 8x8-patch video frame (4 tokens in 16x) inside a text prompt; MiniCPM video takes the processor's native
    videos= path (pixel_values_videos / target_sizes_videos), not frames-as-images."""
    from tests.test_minicpmv47_runtime import MARKERS, VIDEO_PAD

    ids = mx.array([[5, 6, MARKERS[2]] + [VIDEO_PAD] * 4 + [MARKERS[3], 198, 7, 8]])
    kw = {"pixel_values_videos": mx.random.normal((1, 3, 14, 64 * 14), key=mx.random.key(5)),
          "target_sizes_videos": mx.array([[8, 8]], dtype=mx.int32)}
    return ids, kw


@pytest.mark.parametrize("moe", [False, True])
def test_video_dspark_greedy_equals_native_ar(tmp_path, moe):
    from mlx_vlm.models.cache import make_prompt_cache

    target = _tiny_moe_target() if moe else _tiny_target()
    model, lm = target.model, target.model.language_model
    ids, kw = _video_inputs()
    cache = make_prompt_cache(lm)
    tok = mx.argmax(model(ids, cache=cache, **kw).logits[0, -1]).reshape(1)
    reference = [int(tok.item())]
    while len(reference) < 20:
        tok = mx.argmax(lm(tok.reshape(1, 1), cache=cache).logits[0, -1]).reshape(1)
        reference.append(int(tok.item()))
    lm.reset_position_state()
    feats = model.get_input_embeddings(ids, None, **kw)
    positions, delta = lm._position_ids, int(lm._rope_deltas.reshape(-1)[0].item())
    lm.reset_position_state()
    assert delta != 0
    drafter = _drafter_for(target, tmp_path, reference, 64 if moe else 16)
    drafter.prompt_len = int(ids.shape[1])
    target.set_media(feats.inputs_embeds, positions, delta)
    try:
        out = generate_dspark(target, drafter, ids[0].tolist(), 20, set())
    finally:
        target.set_media()
    assert out == reference


# ------------------------------------------------------------- opt-in fast verify (stock multi-row kernels)
@pytest.mark.parametrize("moe", [False, True])
def test_fast_verify_rows_match_the_stock_forward_and_roll_back(tmp_path, monkeypatch, moe):
    """VMLX_DSPARK_VERIFY=fast: verify returns the stock multi-row forward's logits, and rollback leaves the caches
    exactly where a forward of only the kept rows would (the GDN replay, KV trim)."""
    monkeypatch.setenv("VMLX_DSPARK_VERIFY", "fast")
    target = _tiny_moe_target() if moe else _tiny_target()
    c1, c2 = target.make_cache(), target.make_cache()
    for c in (c1, c2):
        target.prefill(mx.array([PROMPT]), c, (), None)
    rows = mx.array([[11, 12, 13, 14, 15]])
    lg, _, sink = target.verify(rows, c1, ())
    ref, _ = target._stock(rows, c2, (), logits=True)
    assert mx.array_equal(lg, ref[0])
    target.rollback(c1, sink, kept=2, total=5)
    c3 = target.make_cache()
    target.prefill(mx.array([PROMPT]), c3, (), None)
    target._stock(rows[:, :2], c3, (), logits=False)
    nxt = mx.array([[21]])
    a = target._stock(nxt, c1, (), logits=True)[0]
    b = target._stock(nxt, c3, (), logits=True)[0]
    assert mx.allclose(a, b, atol=1e-2).item()


def test_fast_verify_keeps_dspark_eligible_without_row_exact(monkeypatch):
    monkeypatch.setenv("VMLX_ROW_EXACT_VERIFY_QMV", "0")
    assert MLXMultimodalLM._dspark_eligible(SimpleNamespace(_dspark={"x": 1}), [], [], [], 0.0, {}) is False
    monkeypatch.setenv("VMLX_DSPARK_VERIFY", "fast")
    assert MLXMultimodalLM._dspark_eligible(SimpleNamespace(_dspark={"x": 1}), [], [], [], 0.0, {}) is True


@pytest.mark.parametrize("window", ["0", "4"])
def test_draft_window_keeps_greedy_equal_to_ar(tmp_path, monkeypatch, window):
    monkeypatch.setenv("VMLX_DSPARK_DRAFT_WINDOW", window)
    target = _tiny_target()
    reference = generate_ar(target, PROMPT, 24, set())
    assert generate_dspark(target, _real_drafter(tmp_path), PROMPT, 24, set()) == reference


def test_draft_window_limits_the_attended_context(tmp_path, monkeypatch):
    target = _tiny_target()
    drafter = _real_drafter(tmp_path)
    drafter.reset()
    drafter.add_context(mx.random.normal((1, 40, drafter.fc.weight.shape[-1]), key=mx.random.key(3)).astype(mx.bfloat16))
    block = mx.random.normal((1, drafter.block, 16), key=mx.random.key(4)).astype(mx.bfloat16)
    full = drafter(block)
    monkeypatch.setenv("VMLX_DSPARK_DRAFT_WINDOW", "40")
    assert mx.array_equal(drafter(block), full)                  # window >= context: identical
    monkeypatch.setenv("VMLX_DSPARK_DRAFT_WINDOW", "4")
    assert not mx.array_equal(drafter(block), full)              # a real restriction
    assert drafter.ctx_len == 40                                 # context untouched


@pytest.mark.parametrize("model_type, expected", [("minicpmv4_7", True), ("qwen3_5_moe", False)])
def test_text_only_loads_the_multimodal_runtime_when_no_text_runtime_exists(tmp_path, model_type, expected):
    # Multimodal Force Off (--text-only) on MiniCPM-V-4.7 sent the bundle to mlx_lm, which raises "Model type
    # minicpmv4_7 not supported": the session never loaded. Media stays refused at the API (_enforce_text_only_override).
    from vmlx_engine.api.utils import is_mllm_model

    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type, "vision_config": {}}))
    assert is_mllm_model(str(tmp_path), force_text_only=True) is expected


def test_chunk_cuts_only_with_an_ssd_tier_and_chunk_writer_chains(monkeypatch):
    """Cross-lane port of the MiMo chunk checkpoints (2026-10-17): with an SSD tier (SSD-only RAM policy) the DSpark
    lane also cuts every chunk_every() absolute tokens; each chunk is stored at once as a delta on the previous chunk,
    and deferred entries take the closest stored chunk below them as parent."""
    from concurrent.futures import Future
    from vmlx_engine.dspark import session as S

    tok = type("T", (), {"convert_tokens_to_ids": lambda self, t: -1})()
    monkeypatch.setenv("VMLX_DSPARK_CHUNK_SNAPSHOT_EVERY", "16")
    monkeypatch.setattr(S, "SSD", None)
    assert [k for k, _ in S.snapshot_cuts(tok, list(range(50)), 0)] == []           # RAM-only: no chunk entries
    monkeypatch.setattr(S, "SSD", object())
    assert S.snapshot_cuts(tok, list(range(50)), 20) == [("chunk", 32), ("chunk", 48)]
    stored = []

    def fake_store(entry, parent=None):
        stored.append((entry["cache_len"], parent and parent["len"]))
        r = Future(); r.set_result({"durable": True})
        r.vmlx_ssd_ref = {"sig": str(entry["cache_len"]), "len": entry["cache_len"], "depth": len(stored)}
        return r

    w = S.ChunkWriter({"len": 20, "sig": "resume"}, store_fn=fake_store)
    for c in (32, 48):
        w.put({"cache_len": c})
    assert stored == [(32, 20), (48, 32)]                                            # chained, not all on the resume
    assert w.parent_for(50, {"len": 20})["len"] == 48 and w.parent_for(30, {"len": 20})["len"] == 20
