"""Source-executed Naive buffer bound; global admission remains independent."""

from types import SimpleNamespace

import pytest

from test_cache_cleanup_phase_timing import ROOT, load_functions
from test_naive_prefill_phase_trace import owner


def configured_owner(monkeypatch, prior=0, dtype="bfloat16", itemsize=2):
    ns, obj, req, events, logs, clocks = owner(monkeypatch, enabled=False)
    obj.model.model = SimpleNamespace(
        embed_tokens=SimpleNamespace(
            weight=SimpleNamespace(dtype=dtype, itemsize=itemsize)
        )
    )
    obj.model.layers = [
        SimpleNamespace(
            is_swa=False,
            self_attn=SimpleNamespace(n_heads=64, indexer=SimpleNamespace(n_heads=16)),
        ),
        SimpleNamespace(is_swa=True, self_attn=SimpleNamespace(n_heads=64)),
    ]
    ns["_chunk_attention_budget_bytes"] = lambda: 4 * 1024**3
    load_functions(
        ROOT / "utils/prefill_admission.py", {"max_prefill_chunk_tokens"}, ns
    )
    obj.prefill_step_size = 2048
    req.context_tokens = list(range(prior))
    return ns, obj, req, events, logs, clocks


@pytest.mark.parametrize(
    "prior,dtype,itemsize",
    [
        (0, "bfloat16", 2),
        (4096, "bfloat16", 2),
        (20480, "bfloat16", 2),
        (20480, "float32", 4),
    ],
)
def test_bound_precedes_slice_and_preserves_every_token(
    monkeypatch, prior, dtype, itemsize
):
    ns, obj, req, events, logs, clocks = configured_owner(
        monkeypatch, prior, dtype, itemsize
    )
    tokens = list(range(100000, 104500))
    ns["_prefill"](obj, tokens, req)
    chunks = [e[1] for e in events if isinstance(e, tuple)]
    assert [t for chunk in chunks for t in chunk] == tokens
    assert req.context_tokens == list(range(prior)) + tokens
    context = prior
    for chunk in chunks:
        assert 64 * len(chunk) * (context + len(chunk)) * 4 <= 4 * 1024**3
        context += len(chunk)
    if prior <= 4096:
        assert [len(c) for c in chunks] == [2048, 2048, 404]
        assert logs == []
    else:
        assert len(chunks[0]) < 2048
        assert len(chunks[0]) == (4 * 1024**3) // (64 * (prior + 2048) * 4)
        assert logs[0][1] == req.uid
    assert events == [
        e
        for chunk in chunks
        for e in (
            "valve",
            "reset_peak",
            ("chunk", chunk),
            "state_eval",
            "sync",
            "clear",
        )
    ]
    assert clocks == []


@pytest.mark.parametrize(
    "unknown",
    [
        "family",
        "embedding",
        "dtype",
        "heads",
        "indexer",
        "layers",
        "swa",
        "oversized_indexer",
    ],
)
def test_unknown_or_other_family_keeps_configured_width(monkeypatch, unknown):
    ns, obj, req, events, logs, clocks = configured_owner(monkeypatch, 20480)
    if unknown == "family":
        obj.model.model_type = "other"
    elif unknown == "embedding":
        obj.model.model.embed_tokens = None
    elif unknown == "dtype":
        obj.model.model.embed_tokens.weight.dtype = "uint32"
    elif unknown == "heads":
        obj.model.layers[0].self_attn.n_heads = None
    elif unknown == "indexer":
        obj.model.layers[0].self_attn.indexer = None
    elif unknown == "layers":
        obj.model.layers = []
    elif unknown == "swa":
        obj.model.layers[0].is_swa = None
    elif unknown == "oversized_indexer":
        obj.model.layers[0].self_attn.indexer.n_heads = 128
    ns["_prefill"](obj, [1] * 2049, req)
    assert [len(e[1]) for e in events if isinstance(e, tuple)] == [2048, 1]
    assert logs == []


def test_existing_global_valve_can_still_reject_smaller_chunk_before_mutation(
    monkeypatch,
):
    ns, obj, req, events, logs, clocks = configured_owner(monkeypatch, 20480)
    original_context = list(req.context_tokens)
    ns.update(
        PrefillAdmissionError=RuntimeError,
        _GIB=1024**3,
        wired_limit_advisory=lambda limit: "",
        get_effective_metal_working_set_bytes=lambda mx: (0, 9),
    )
    load_functions(
        ROOT / "utils/prefill_admission.py",
        {"prefill_valve_check", "project_chunk_peak_bytes"},
        ns,
    )
    ns["_prefill_valve_check"] = ns["prefill_valve_check"]
    with pytest.raises(RuntimeError, match="prefill admission rejected chunk"):
        ns["_prefill"](obj, [1] * 2048, req)
    assert 0 < logs[0][3] < 2048  # The global refusal follows the geometry clamp.
    assert req.context_tokens == original_context
    assert events == []


def test_prior_observed_peak_not_reset_or_scaled_when_width_shrinks(monkeypatch):
    ns, obj, req, events, logs, clocks = configured_owner(monkeypatch, 5000)
    observed = []
    ns["_prefill_valve_check"] = lambda a, l, t, m, **kw: observed.append(
        (t, kw["chunk_end"] - kw["chunk_start"])
    )
    ns["_prefill"](obj, list(range(6000)), req)
    assert observed[0] == (0, 2048)
    assert all(t == 10 for t, width in observed[1:])
    assert any(width < 2048 for t, width in observed[1:])


def packed_embedding(ns):
    embedding = ns["QuantizedEmbedding"]()
    embedding.mode = "affine"
    embedding.weight = SimpleNamespace(dtype="uint32", itemsize=4, shape=(152576, 1024))
    embedding.scales = SimpleNamespace(dtype="bfloat16", itemsize=2, shape=(152576, 64))
    embedding.biases = SimpleNamespace(dtype="bfloat16", itemsize=2, shape=(152576, 64))
    return embedding


def test_actual_packed_embedding_logs_hidden_dtype_but_bounds_fp32_scores(monkeypatch):
    ns, obj, req, events, logs, clocks = configured_owner(monkeypatch, 20480)
    obj.model.model.embed_tokens = packed_embedding(ns)
    assert ns["_naive_prefill_attention_geometry"](obj.model) == (64, 2)
    ns["_prefill"](obj, list(range(2048)), req)
    chunks = [e[1] for e in events if isinstance(e, tuple)]
    assert len(chunks[0]) == 744
    assert [t for chunk in chunks for t in chunk] == list(range(2048))
    assert all(row[-2:] == (2, 4) for row in logs)


@pytest.mark.parametrize(
    "unsupported",
    [
        "mixed_dtype",
        "missing_bias",
        "uint8_scale",
        "non_affine",
        "unknown_class",
        "shape",
        "itemsize",
    ],
)
def test_packed_embedding_unknown_contract_retains_old_path(monkeypatch, unsupported):
    ns, obj, req, events, logs, clocks = configured_owner(monkeypatch, 20480)
    emb = packed_embedding(ns)
    if unsupported == "mixed_dtype":
        emb.biases.dtype = "float32"
    elif unsupported == "missing_bias":
        emb.biases = None
    elif unsupported == "uint8_scale":
        emb.scales.dtype = emb.biases.dtype = "uint8"
    elif unsupported == "non_affine":
        emb.mode = "mxfp4"
    elif unsupported == "unknown_class":
        emb = SimpleNamespace(**vars(emb))
    elif unsupported == "shape":
        emb.biases.shape = (152576, 32)
    elif unsupported == "itemsize":
        emb.biases.itemsize = 4
    obj.model.model.embed_tokens = emb
    assert ns["_naive_prefill_attention_geometry"](obj.model) is None
    ns["_prefill"](obj, [1] * 2048, req)
    assert [len(e[1]) for e in events if isinstance(e, tuple)] == [2048]
    assert logs == []


@pytest.mark.parametrize("configured", [128, 512])
def test_refault_prefix_and_configured_smaller_width_are_preserved(
    monkeypatch, configured
):
    ns, obj, req, events, logs, clocks = configured_owner(monkeypatch, 20480)
    obj.model.model.embed_tokens = packed_embedding(ns)
    obj.prefill_step_size = configured
    native_cache = req.cache
    native_state = req.cache[0].state
    prefix = list(req.context_tokens)
    tokens = list(range(70000, 71001))
    ns["_prefill"](obj, tokens, req)
    chunks = [e[1] for e in events if isinstance(e, tuple)]
    assert len(chunks[0]) == configured
    assert all(len(chunk) <= configured for chunk in chunks)
    assert req.context_tokens == prefix + tokens
    assert req.cache is native_cache and req.cache[0].state is native_state
    assert events.count("state_eval") == events.count("sync") == len(chunks)
    assert logs == []
