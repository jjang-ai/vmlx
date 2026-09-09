"""Translated Ollama input validation must remain a client error."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient

@pytest.mark.parametrize('path,extra', [('/api/chat', {'messages': [{'role':'user','content':'hi'}]}),
    ('/api/generate', {'prompt':'hi'}), ('/api/generate', {'prompt':'hi','raw':True})])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('value', ['not-a-number', 12.5, 'Infinity'])
def test_invalid_num_predict_is_400_before_generation(monkeypatch, path, extra, stream, value):
    from vmlx_engine import server
    monkeypatch.setattr(server, '_engine', SimpleNamespace(is_mllm=False))
    monkeypatch.setattr(server, '_api_key', None)
    monkeypatch.setattr(server, '_standby_state', None)
    generation = Mock(side_effect=AssertionError('invalid request reached inference'))
    monkeypatch.setattr(server, 'create_chat_completion', generation)
    monkeypatch.setattr(server, 'create_completion', generation)
    client = TestClient(server.app, raise_server_exceptions=False)
    result = client.post(path, json={'model':'test','stream':stream,'options':{'num_predict':value},**extra})
    client.close()
    assert result.status_code == 400, result.text
    assert result.json()['code'] == 'invalid_request_error'
    assert 'num_predict' in result.json()['error']
    generation.assert_not_called()
