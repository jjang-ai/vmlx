from pathlib import Path
import os

import pytest

_OFFICIAL_VEC = Path("/tmp/ds4-read/tests/test-vectors/official.vec")

from tests.cross_matrix.dsv4_vector_probe import (
    _cache_disabled_requested,
    _disable_attention_sinks,
    _disable_attention_sinks_requested,
    _prepare_vector_probe_environment,
    bytes_to_hex,
    parse_official_vec,
    token_id_for_bytes,
    token_bytes,
)


def test_parse_ds4_official_vec_fixture():
    if not _OFFICIAL_VEC.exists():
        pytest.skip(
            f"DSV4 official vector fixture not present at {_OFFICIAL_VEC}; "
            "skipping (fixture lives outside the repo and is only mounted on "
            "the DSV4 contributor box)."
        )
    cases = parse_official_vec(
        _OFFICIAL_VEC,
        base_dir=Path("/tmp/ds4-read"),
    )

    ids = [case.case_id for case in cases]
    assert "short_reasoning_plain" in ids
    assert "long_code_audit" in ids

    short = next(case for case in cases if case.case_id == "short_reasoning_plain")
    assert short.ctx == 4096
    assert short.steps[0].selected_hex == "3136"
    assert short.prompt_path.name == "short_reasoning_plain.txt"
    assert short.prompt_path.is_absolute()
    assert short.prompt_path.exists()


def test_token_bytes_encodes_decoded_token_as_utf8():
    class Tok:
        def decode(self, ids):
            assert ids == [123]
            return " gamma"

    assert bytes_to_hex(token_bytes(Tok(), 123)) == "2067616d6d61"


def test_token_id_for_bytes_requires_single_round_trip_token():
    class Tok:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            if text == " gamma":
                return [123]
            return [1, 2]

        def decode(self, ids):
            if ids == [123]:
                return " gamma"
            return "bad"

    assert token_id_for_bytes(Tok(), b" gamma") == 123
    assert token_id_for_bytes(Tok(), b"split") is None


def test_disable_attention_sinks_sets_large_negative_values():
    import mlx.core as mx
    import numpy as np

    class WithSink:
        def __init__(self):
            self.attn_sink = mx.array([1.0, 2.0], dtype=mx.float32)

    class WithoutSink:
        pass

    sink = WithSink()

    class Model:
        def named_modules(self):
            return [("sink", sink), ("plain", WithoutSink())]

    assert _disable_attention_sinks(Model()) == 1
    assert np.asarray(sink.attn_sink).max() < -1e8


def test_disable_attention_sinks_env_uses_standard_vmlx_prefix(monkeypatch):
    monkeypatch.setenv("VMLX_DSV4_VECTOR_DISABLE_SINKS", "1")
    monkeypatch.delenv("VMLINUX_DSV4_VECTOR_DISABLE_SINKS", raising=False)

    assert _disable_attention_sinks_requested()


def test_no_cache_env_uses_standard_vmlx_prefix(monkeypatch):
    monkeypatch.setenv("VMLX_DSV4_VECTOR_NO_CACHE", "1")
    monkeypatch.delenv("VMLINUX_DSV4_VECTOR_NO_CACHE", raising=False)

    assert _cache_disabled_requested()


def test_prepare_vector_probe_environment_disables_hard_rep_block(monkeypatch):
    monkeypatch.delenv("VMLX_DSV4_HARD_REP_BLOCK", raising=False)
    monkeypatch.delenv("VMLINUX_DSV4_HARD_REP_BLOCK", raising=False)

    meta = _prepare_vector_probe_environment()

    assert os.environ["VMLX_DSV4_HARD_REP_BLOCK"] == "0"
    assert "VMLINUX_DSV4_HARD_REP_BLOCK" not in os.environ
    assert meta["runtime_path"] == "direct_model_call"
    assert meta["dsv4_batch_generator_used"] is False
    assert meta["logit_warps"]["hard_repetition_block"] == "bypassed_by_direct_model_call"
    assert meta["logit_warps"]["hard_repetition_block_env"] == "VMLX_DSV4_HARD_REP_BLOCK=0"


def test_parse_official_vec_rejects_malformed_top_count(tmp_path):
    vec = tmp_path / "bad.vec"
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("hi", encoding="utf-8")
    vec.write_text(
        "\n".join(
            [
                "# ds4-official-logprob-vectors-v1",
                f"case bad 128 1 {prompt.name}",
                "step 0 61 2",
                "top 61 0",
                "end",
            ]
        ),
        encoding="ascii",
    )

    try:
        parse_official_vec(vec, base_dir=tmp_path)
    except ValueError as exc:
        assert "expected 2 top rows" in str(exc)
    else:
        raise AssertionError("malformed vector should have raised")
