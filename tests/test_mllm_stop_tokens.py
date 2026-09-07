"""The MLLM scheduler's stop set must include the bundle's own declared stop ids.

Measured on gemma-4-26B-A4B-it-qat-JANG_4M (probe-gemma-empty-081617): after
``<tool_call|>`` the model emits ``<|tool_response>`` (id 50, declared in the
bundle's generation_config eos list) and, because this lane only read the
tokenizer eos plus the registry strings, kept emitting it until max_tokens —
400 tokens rendered as nothing after every tool call. The text lane honoured
the declaration through collect_multi_eos_ids all along.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


class _ProcessorTokenizer:
    def __init__(self, path: Path, eos_token_id: int):
        self.name_or_path = str(path)
        self.eos_token_id = eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False):
        table = {"<eos>": [1], "<turn|>": [106]}
        return table.get(text, [9, 9])


def _scheduler_stub(processor):
    from vmlx_engine.mllm_scheduler import MLLMScheduler

    stub = SimpleNamespace(processor=processor)
    return MLLMScheduler._get_stop_tokens(stub)


def test_mllm_stop_set_includes_bundle_declared_eos_ids(tmp_path: Path):
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 106, 50]}))
    tok = _ProcessorTokenizer(tmp_path, eos_token_id=1)

    stops = _scheduler_stub(tok)

    assert {1, 106, 50} <= stops


def test_mllm_stop_set_includes_jang_declared_ids(tmp_path: Path):
    (tmp_path / "jang_config.json").write_text(json.dumps({"chat": {"stop_token_ids": [1, 77]}}))
    tok = _ProcessorTokenizer(tmp_path, eos_token_id=1)

    assert 77 in _scheduler_stub(tok)


def test_mllm_stop_set_without_declarations_is_the_tokenizer_eos(tmp_path: Path):
    tok = _ProcessorTokenizer(tmp_path, eos_token_id=1)

    assert 1 in _scheduler_stub(tok)
    assert 50 not in _scheduler_stub(tok)
