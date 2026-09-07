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


def test_collect_multi_eos_ids_resolves_variant_suffixed_spellings(tmp_path: Path):
    """Hy3-style bundle: vocab has only `:opensource`-suffixed specials, the
    registry declares the canonical bare spelling. The resolver must fall back
    through the tag-dialect map instead of leaving the stop set uninstalled."""
    _write_json(tmp_path / "tokenizer_config.json", {})

    class _Entry:
        def __init__(self, content: str):
            self.content = content

    tok = _FakeTokenizer(
        eos_token_id=1,
        token_to_id={
            "<｜hy_eos:opensource｜>": 120,
            "<｜hy_User:opensource｜>": 121,
            "<｜hy_Assistant:opensource｜>": 122,
        },
    )
    tok.added_tokens_decoder = {
        10: _Entry("</think:opensource>"),
        120: _Entry("<｜hy_eos:opensource｜>"),
        121: _Entry("<｜hy_User:opensource｜>"),
        122: _Entry("<｜hy_Assistant:opensource｜>"),
    }

    resolved, unresolved = collect_multi_eos_ids(
        tok,
        str(tmp_path),
        registry_eos_tokens=["<｜hy_eos｜>", "<｜hy_User｜>", "<｜hy_Assistant｜>"],
        reasoning_parser="qwen3",
        use_rust_tokenizer=False,
    )

    assert resolved == [1, 120, 121, 122]
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


def test_collect_multi_eos_ids_honors_jang_config_stop_token_ids(tmp_path: Path):
    """jang_config.chat.stop_token_ids is the bundle's own declared stop set.

    Every 2026-08 Nemotron stamp declares it, and before this reader existed
    the key was parsed by NOTHING — a bundle carrying only this spelling (no
    generation_config.json) kept generating past its turn boundary.
    """
    _write_json(
        tmp_path / "jang_config.json",
        {"chat": {"stop_token_ids": [2, 11]}},
    )

    tok = _FakeTokenizer(eos_token_id=11)
    resolved, unresolved = collect_multi_eos_ids(
        tok,
        str(tmp_path),
        registry_eos_tokens=None,
        reasoning_parser=None,
        use_rust_tokenizer=False,
    )

    assert resolved == [11, 2]
    assert unresolved == []


def test_collect_multi_eos_ids_ignores_malformed_jang_stop_ids(tmp_path: Path):
    """Non-int, negative, and boolean entries must not poison the stop set."""
    _write_json(
        tmp_path / "jang_config.json",
        {"chat": {"stop_token_ids": ["11", -3, True, None, 7]}},
    )

    tok = _FakeTokenizer(eos_token_id=1)
    resolved, _ = collect_multi_eos_ids(
        tok,
        str(tmp_path),
        registry_eos_tokens=None,
        reasoning_parser=None,
        use_rust_tokenizer=False,
    )

    assert resolved == [1, 7]


def test_bundle_declared_stop_ids_reads_both_spellings_in_order(tmp_path: Path):
    """The bundle's declared stop set is one contract with two spellings.

    Gemma-4 declares ``eos_token_id: [1, 106, 50]`` — ``<eos>``, ``<turn|>``
    and ``<|tool_response>``, the tool-role opener the assistant must never
    produce. Both lanes read it through this helper; the jang spelling is
    merged after it, duplicates and malformed entries dropped.
    """
    from vmlx_engine.utils.multi_eos import bundle_declared_stop_ids

    _write_json(tmp_path / "generation_config.json", {"eos_token_id": [1, 106, 50, True, "x"]})
    _write_json(tmp_path / "jang_config.json", {"chat": {"stop_token_ids": [50, 7, -1, None]}})

    assert bundle_declared_stop_ids(str(tmp_path)) == [1, 106, 50, 7]
    assert bundle_declared_stop_ids(str(tmp_path / "missing")) == []
