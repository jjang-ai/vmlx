"""Health probes must not import/activate a kernel or imply completed GPU work."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def probe():
    path = Path(__file__).resolve().parents[1] / "vmlx_engine/server.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef)
                and n.name == "_loaded_gdn_dispatch_status")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["_loaded_gdn_dispatch_status"]()


def test_unloaded_kernel_stays_unloaded_and_counters_are_unknown():
    name = "vmlx_engine.metal.gdn_packed"
    with patch.dict(sys.modules):
        sys.modules.pop(name, None)
        result = probe()
        assert name not in sys.modules
    assert not result["module_loaded"]
    assert not result["counters_available"]
    assert result["packed_calls"] is None
    assert result["stock_calls"] is None


def test_live_counter_snapshots_do_not_reset_or_alias_runtime_state():
    stats = {"packed_calls": 48, "stock_calls": 3}
    with patch.dict(sys.modules, {
        "vmlx_engine.metal.gdn_packed": SimpleNamespace(STATS=stats)
    }):
        first = probe()
        stats["packed_calls"] += 48
        second = probe()
    assert first["packed_calls"] == 48
    assert second["packed_calls"] == 96
    assert second["stock_calls"] == 3
    assert stats == {"packed_calls": 96, "stock_calls": 3}
    assert second["scope"] == "process_cumulative_graph_dispatch"
    assert "not GPU timings" in second["note"]
