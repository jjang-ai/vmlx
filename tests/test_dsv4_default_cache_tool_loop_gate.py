from argparse import Namespace
from pathlib import Path


def test_dsv4_default_cache_tool_loop_gate_dry_run_pins_default_cache_flags():
    from tests.cross_matrix.run_dsv4_default_cache_tool_loop_gate import run

    result = run(
        Namespace(
            model="/models/dsv4",
            python=Path("/python"),
            port=8899,
            out=Path("unused.json"),
            timeout=1,
            request_timeout=1,
            min_free_gb=0,
            dry_run=True,
            pool_quant=False,
        )
    )

    assert result["status"] == "dry_run"
    assert "--dsv4-enable-prefix-cache" in result["cmd"]
    assert "--use-paged-cache" in result["cmd"]
    assert "--enable-block-disk-cache" in result["cmd"]
    assert "--disable-prefix-cache" not in result["cmd"]
    assert "--kv-cache-quantization" not in result["cmd"]
    assert "--tool-call-parser" in result["cmd"]
    assert "dsml" in result["cmd"]
    assert result["env"]["DSV4_LONG_CTX"] == "1"
    assert result["env"]["DSV4_POOL_QUANT"] == "0"


def test_dsv4_default_cache_tool_loop_gate_memory_preflight_skips_before_spawn(
    monkeypatch,
):
    import tests.cross_matrix.run_dsv4_default_cache_tool_loop_gate as gate

    monkeypatch.setattr(
        gate,
        "resource_snapshot",
        lambda name, proc=None: {
            "name": name,
            "system_memory": {"available_gb": 8.0},
        },
    )

    result = gate.run(
        Namespace(
            model="/models/dsv4",
            python=Path("/python"),
            port=8899,
            out=Path("unused.json"),
            timeout=1,
            request_timeout=1,
            min_free_gb=120,
            dry_run=False,
            pool_quant=False,
        )
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_free_memory"
    assert "cmd" in result
    assert "--dsv4-enable-prefix-cache" in result["cmd"]
