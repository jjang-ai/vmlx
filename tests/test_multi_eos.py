from pathlib import Path

from vmlx_engine.utils.multi_eos import collect_multi_eos_ids


class _FakeTokenizer:
    def __init__(
        self,
        eos_token_id: int = 2,
        token_to_id: dict[str, int] | None = None,
        unk_token_id: int = 0,
        unk_token: str = "<unk>",
    ):
        self.eos_token_id = eos_token_id
        self.convert_map = token_to_id or {}
        self.unk_token_id = unk_token_id
        self.unk_token = unk_token

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.convert_map.get(token, self.unk_token_id)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(payload))


def test_collect_multi_eos_ids_merges_generation_and_registry_tokens(tmp_path: Path):
    _write_json(
        tmp_path / "generation_config.json",
        {"eos_token_id": [10, 20]},
    )
    _write_json(
        tmp_path / "tokenizer_config.json",
        {},
    )

    tok = _FakeTokenizer(
        eos_token_id=1,
        token_to_id={"<|extra|>": 200},
    )
    resolved, unresolved = collect_multi_eos_ids(
        tok,
        str(tmp_path),
        registry_eos_tokens=["<|extra|>"],
        reasoning_parser=None,
        use_rust_tokenizer=False,
    )

    assert resolved == [1, 10, 20, 200]
    assert unresolved == []


def test_collect_multi_eos_ids_uses_deepseek_role_tokens(tmp_path: Path):
    _write_json(
        tmp_path / "generation_config.json",
        {"eos_token_id": [10]},
    )
    _write_json(
        tmp_path / "tokenizer_config.json",
        {"chat": {"role_tokens": {"user": "<|User|>", "assistant": "<|Assistant|>","latest_reminder":"<|latest_reminder|>"}}},
    )
    tok = _FakeTokenizer(
        eos_token_id=1,
        token_to_id={
            "<|User|>": 101,
            "<|Assistant|>": 102,
            "<|latest_reminder|>": 103,
        },
    )
    resolved, unresolved = collect_multi_eos_ids(
        tok,
        str(tmp_path),
        registry_eos_tokens=[],
        reasoning_parser="deepseek_r1",
        use_rust_tokenizer=False,
    )

    assert resolved == [1, 10, 101, 102, 103]
    assert unresolved == []


def test_collect_multi_eos_ids_ignores_role_tokens_for_non_deepseek(tmp_path: Path):
    _write_json(
        tmp_path / "tokenizer_config.json",
        {"chat": {"role_tokens": {"user": "<|User|>"}}},
    )
    tok = _FakeTokenizer(eos_token_id=1, token_to_id={"<|extra|>": 200})
    resolved, unresolved = collect_multi_eos_ids(
        tok,
        str(tmp_path),
        registry_eos_tokens=["<|extra|>"],
        reasoning_parser="qwen3",
        use_rust_tokenizer=False,
    )

    assert resolved == [1, 200]
    assert unresolved == []


def test_collect_multi_eos_ids_rejects_unknown_tokens_as_unresolved(tmp_path: Path):
    _write_json(
        tmp_path / "generation_config.json",
        {"eos_token_id": 1},
    )
    tok = _FakeTokenizer(
        eos_token_id=1,
        token_to_id={},
        unk_token_id=0,
        unk_token="<unk>",
    )

    resolved, unresolved = collect_multi_eos_ids(
        tok,
        str(tmp_path),
        registry_eos_tokens=["<|missing|>"],
        reasoning_parser=None,
        use_rust_tokenizer=False,
    )

    assert resolved == [1]
    assert unresolved == ["<|missing|>"]
