"""The JSON/event fast path follows the same terminal publication as streams.

Default: the JSON result returns as soon as generation finishes and the next
admission is the persistence fence. VMLX_TERMINAL_PUBLISH=strict holds it
behind the fence, exactly like a strict stream.
"""
import asyncio
from types import MethodType
import pytest
from vmlx_engine.engine_core import EngineCore
from vmlx_engine.request import RequestOutput
from vmlx_engine.persistence_outcome import LEDGER

@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['stored', 'failed', 'skipped'])
async def test_generate_does_not_publish_or_cleanup_before_persistence_in_strict_mode(
    outcome, caplog, monkeypatch
):
    monkeypatch.setenv('VMLX_TERMINAL_PUBLISH', 'strict')
    core = EngineCore.__new__(EngineCore)
    core._terminal_cleanup_complete = asyncio.Event()
    finished = asyncio.Event()
    finished.set()
    output = RequestOutput(request_id='json-fence', output_text='tool result', finished=True)
    class Collector:
        def __init__(self): self.items = [output]
        def get_nowait(self): return self.items.pop(0) if self.items else None
    async def add(self, **kwargs): return 'json-fence'
    cleaned = []
    core.add_request = MethodType(add, core)
    core._finished_events = {'json-fence': finished}
    core._output_collectors = {'json-fence': Collector()}
    core._cleanup_request = lambda rid: cleaned.append(rid)
    with caplog.at_level('INFO'):
        task = asyncio.create_task(core.generate('prompt'))
        await asyncio.sleep(0)
        premature = task.done()
        before_cleanup = list(cleaned)
        LEDGER.record('json-fence', outcome, 'test outcome', retained_tokens=64)
        core._terminal_cleanup_complete.set()
        result = await task
    assert not premature, 'JSON response became visible while persistence was unfinished'
    assert before_cleanup == []
    assert result is output
    assert cleaned == ['json-fence']
    assert f'cache_outcome={outcome}' in caplog.text
    assert LEDGER.peek('json-fence') is None


@pytest.mark.asyncio
async def test_generate_publishes_json_result_before_persistence_settles():
    core = EngineCore.__new__(EngineCore)
    core._terminal_cleanup_complete = asyncio.Event()
    finished = asyncio.Event()
    finished.set()
    output = RequestOutput(request_id='json-early', output_text='tool result', finished=True)
    class Collector:
        def __init__(self): self.items = [output]
        def get_nowait(self): return self.items.pop(0) if self.items else None
    async def add(self, **kwargs): return 'json-early'
    cleaned = []
    core.add_request = MethodType(add, core)
    core._finished_events = {'json-early': finished}
    core._output_collectors = {'json-early': Collector()}
    core._cleanup_request = lambda rid: cleaned.append(rid)

    result = await asyncio.wait_for(core.generate('prompt'), timeout=1.0)

    assert result is output
    assert cleaned == ['json-early']
    # The fence is still closed: the next admission, not this result, waits.
    assert not core._terminal_cleanup_complete.is_set()
