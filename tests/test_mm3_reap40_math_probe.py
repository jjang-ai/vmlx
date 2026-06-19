from __future__ import annotations

import mlx.core as mx

from tests.cross_matrix import run_mm3_reap40_math_probe as probe


def test_mm3_reap40_probe_forces_thinking_off_in_api_payload(monkeypatch):
    captured = {}

    def fake_post_json(url, payload, *, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {
            "choices": [
                {
                    "message": {
                        "content": "The result is \\boxed{42}.",
                    }
                }
            ]
        }

    monkeypatch.setattr(probe, "post_json", fake_post_json)

    row = probe.run_api_task(
        probe.TASKS[0],
        server_url="http://127.0.0.1:8000",
        model="m3",
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        thinking_mode="off",
        timeout=3.0,
    )

    assert row["pass"] is True
    assert captured["payload"]["enable_thinking"] is False
    assert captured["payload"]["chat_template_kwargs"] == {
        "thinking_mode": "disabled",
    }


def test_mm3_reap40_probe_records_msa_cache_invariant_mismatch():
    Sparse = type("MiniMaxM3SparseCache", (), {})
    cache = Sparse()
    cache.offset = 3
    cache.keys = mx.zeros((1, 4, 3, 8))
    cache.values = mx.zeros((1, 4, 3, 8))
    cache.idx_keys = mx.zeros((1, 1, 2, 8))

    snapshot = probe._cache_snapshot([cache])

    assert snapshot["m3_sparse_invariants_ok"] is False
    assert snapshot["mismatch_layers"] == [0]
    assert snapshot["layers"][0] == {
        "layer": 0,
        "class": "MiniMaxM3SparseCache",
        "offset": 3,
        "keys_len": 3,
        "values_len": 3,
        "idx_keys_len": 2,
    }


def test_mm3_reap40_probe_runtime_template_uses_m3_thinking_vocab():
    class Tokenizer:
        chat_template = "m3-template"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["thinking_mode"] == "enabled"
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["tokenize"] is False
            return "rendered"

        def encode(self, text, add_special_tokens=False):
            assert text == "rendered"
            assert add_special_tokens is False
            return [1, 2, 3]

    rendered, ids = probe._chat_prompt_ids(
        Tokenizer(),
        probe.TASKS[0],
        thinking_mode="on",
    )

    assert rendered == "rendered"
    assert ids == [1, 2, 3]
