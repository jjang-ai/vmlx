"""The speed harness must not count malformed/incomplete streams as wins."""
import importlib.util
import json
from pathlib import Path
import pytest

def gate():
    spec=importlib.util.spec_from_file_location("qwen4_speed_gate",Path(__file__).resolve().parents[1]/"bench/qwen4_speed_gate.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def event(payload):
    return ("data: "+json.dumps(payload)+"\n\n").encode()

def good():
    return [
        event({"id":"req-test","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":None}]}),
        event({"id":"req-test","choices":[{"index":0,"delta":{},"finish_reason":"length"}],
               "usage":{"prompt_tokens":8192,"completion_tokens":128}}),
        b"data: [DONE]\n\n",
    ]

def run(monkeypatch,lines):
    g=gate()
    class Response:
        def __enter__(self): return iter(lines)
        def __exit__(self,*args): pass
    monkeypatch.setattr(g.urllib.request,"urlopen",lambda *args,**kwargs: Response())
    return g.request_stream(url="http://unused/v1/chat/completions",model="fixture",text="fixture",
        max_tokens=128,temperature=0,top_p=1,top_k=0,generation_mode=None,depth=None,timeout=1)

@pytest.mark.parametrize("defect",["error","malformed","missing_done","duplicate_done","missing_finish","missing_usage","changed_id","stats_only_usage","error_finish"])
def test_invalid_stream_is_not_a_speed_success(monkeypatch,defect):
    lines=good()
    if defect=="error": lines.insert(2,event({"error":{"code":"runtime_failure","message":"failed"}}))
    elif defect=="malformed": lines.insert(2,b"data: {broken}\n")
    elif defect=="missing_done": lines.pop()
    elif defect=="duplicate_done": lines.append(lines[-1])
    elif defect=="missing_finish":
        lines[1]=event({"id":"req-test","choices":[],"usage":{"prompt_tokens":8192,"completion_tokens":128}})
    elif defect=="missing_usage":
        lines[1]=event({"id":"req-test","choices":[{"index":0,"delta":{},"finish_reason":"length"}]})
    elif defect=="changed_id": lines.insert(1,event({"id":"other","choices":[]}))
    elif defect=="stats_only_usage":
        lines[1]=event({"id":"req-test","choices":[{"delta":{},"finish_reason":"length"}],
            "usage":{"other":1},"stats":{"prompt_tokens":8192,"generated_tokens":128}})
    elif defect=="error_finish":
        lines[1]=event({"id":"req-test","choices":[{"delta":{},"finish_reason":"error"}],
            "usage":{"prompt_tokens":8192,"completion_tokens":128}})
    row=run(monkeypatch,lines)
    assert row["protocol_valid"] is False
    assert row["protocol_errors"]
    assert row["raw_sse_lines"]

def test_valid_stream_retains_separate_reasoning_and_content(monkeypatch):
    lines=good()
    lines[0]=event({"id":"req-test","choices":[{"index":0,"delta":{"content":"answer","reasoning_content":"thought"},"finish_reason":None}]})
    row=run(monkeypatch,lines)
    assert row["protocol_valid"] is True
    assert row["protocol_errors"]==[]
    assert row["output_text"]=="answer"
    assert row["reasoning_text"]=="thought"
    assert row["done_events"]==1 and row["finish_events"]==1

@pytest.mark.parametrize("bad_warmup",[True,False])
def test_main_preserves_invalid_receipt_and_exits_before_reporting_speed(monkeypatch,tmp_path,bad_warmup):
    import sys
    g=gate()
    output=tmp_path/"receipt.json"
    monkeypatch.setattr(sys,"argv",["speed","--url","http://unused/v1/chat/completions",
        "--model","fixture","--label","fixture","--prompt-seed","fixture","--output",str(output),
        "--contexts","1024","--trials","1"])
    monkeypatch.setattr(g,"host_snapshot",lambda *args: {})
    bad={"protocol_valid":False,"protocol_errors":["server error"],"raw_sse_lines":["data: error"]}
    rows=iter([bad] if bad_warmup else [{"protocol_valid":True},bad])
    monkeypatch.setattr(g,"request_stream",lambda **kwargs: next(rows))
    with pytest.raises(RuntimeError,match="Invalid"):
        g.main()
    receipt=json.loads(output.read_text())
    assert receipt["rows"]==[]
    assert (receipt["warmup"] if bad_warmup else receipt["invalid_attempts"][0])==bad
