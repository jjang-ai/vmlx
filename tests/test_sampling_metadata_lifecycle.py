"""Defaults are load-scoped; status inspection is not request admission."""
import json
import logging

import pytest
from fastapi import HTTPException


def test_status_projection_does_not_warn_or_consume_request_dedup(monkeypatch, caplog):
    from vmlx_engine import server
    monkeypatch.setattr(server, '_default_max_tokens_explicit', False)
    monkeypatch.setattr(server, '_bundle_sampling_default', lambda *a: None)
    monkeypatch.setattr(server, '_metal_projected_output_token_cap', lambda *a: 11436)
    monkeypatch.setattr(server, '_loaded_model_is_reasoning_capable', lambda: True)
    monkeypatch.setattr(server, '_resolve_repetition_penalty', lambda *a, **k: None)
    monkeypatch.setattr(server, '_projected_guard_warned', set())
    with caplog.at_level(logging.WARNING):
        status = server._model_effective_defaults_status('diagnostic')
        assert status['effective_defaults']['max_output_tokens'] == 11436
        assert not server._projected_guard_warned
        assert not any('headroom guard clamped' in r.message for r in caplog.records)
        assert server._resolve_max_tokens(4096, 'diagnostic') == 4096
        assert server._resolve_max_tokens(None, 'diagnostic') == 11436
        assert sum('headroom guard clamped implicit' in r.message for r in caplog.records) == 1
    # Suppressing telemetry warnings must never bypass explicit safety.
    with pytest.raises(HTTPException) as exc:
        server._resolve_max_tokens(16384, 'diagnostic', emit_warning=False)
    assert exc.value.status_code == 413


@pytest.mark.parametrize('metadata', ['generation_config.json', 'jang_config.json'])
@pytest.mark.parametrize('initially_present', [True, False])
def test_load_refresh_invalidates_same_path_sampling_metadata(tmp_path, monkeypatch, metadata, initially_present):
    from vmlx_engine import server
    def write(value):
        doc = {'max_new_tokens':value}
        if metadata == 'jang_config.json':
            doc = {'chat': {'sampling_defaults':doc}}
        (tmp_path / metadata).write_text(json.dumps(doc))
    monkeypatch.setattr(server, '_model_path', str(tmp_path))
    monkeypatch.setattr(server, '_jang_sampling_defaults_cache', {})
    monkeypatch.setattr(server, '_generation_defaults_cache', {})
    monkeypatch.setattr(server, '_estimate_max_prompt_tokens', lambda: 8192)
    monkeypatch.setattr(server, '_cli_args', {})
    monkeypatch.setattr(server, '_max_prompt_tokens', 0)
    # Avoid mutating the test process's unrelated declared-context registry.
    from vmlx_engine import context_limits
    monkeypatch.setattr(context_limits, 'set_declared_context_tokens', lambda _: None)
    if initially_present:
        write(512)
    assert server._bundle_sampling_default(str(tmp_path), 'max_new_tokens') == (512 if initially_present else None)
    write(2048)
    # A running generation keeps its cached metadata until a load refresh.
    assert server._bundle_sampling_default(str(tmp_path), 'max_new_tokens') == (512 if initially_present else None)
    server._refresh_loaded_max_prompt_tokens('test_successful_model_reload')
    assert server._bundle_sampling_default(str(tmp_path), 'max_new_tokens') == 2048
