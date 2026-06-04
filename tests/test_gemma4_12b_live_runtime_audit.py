from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve().parent / "cross_matrix/run_gemma4_12b_live_runtime_audit.py"
SPEC = importlib.util.spec_from_file_location("gemma4_12b_live_runtime_audit", SCRIPT)
assert SPEC is not None
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_cache_warm_uses_prompts_contract(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_request_json(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 60,
    ) -> tuple[int, dict[str, Any], str]:
        captured.update(
            {
                "method": method,
                "url": url,
                "payload": payload,
                "timeout": timeout,
            }
        )
        return 200, {"warmed": 1}, '{"warmed":1}'

    monkeypatch.setattr(audit, "_request_json", fake_request_json)

    result = audit._run_cache_warm("http://127.0.0.1:9999", "gemma4-test")

    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:9999/v1/cache/warm"
    assert captured["payload"] == {
        "prompts": ["Cache warm probe. Remember codeword ORCHID-4729."],
    }
    assert result["code"] == 200
