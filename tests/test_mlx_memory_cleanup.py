from types import SimpleNamespace
import ast
from pathlib import Path


def test_clear_mlx_memory_cache_prefers_top_level_api():
    from vmlx_engine.mlx_memory import clear_mlx_memory_cache

    calls = []
    fake_mx = SimpleNamespace(
        metal=SimpleNamespace(clear_cache=lambda: calls.append("metal")),
        clear_cache=lambda: calls.append("top"),
    )

    assert clear_mlx_memory_cache(mx=fake_mx) == "mx.clear_cache"
    assert calls == ["top"]


def test_clear_mlx_memory_cache_falls_back_to_metal_api():
    from vmlx_engine.mlx_memory import clear_mlx_memory_cache

    calls = []
    fake_mx = SimpleNamespace(metal=SimpleNamespace(clear_cache=lambda: calls.append("metal")))

    assert clear_mlx_memory_cache(mx=fake_mx) == "mx.metal.clear_cache"
    assert calls == ["metal"]


def test_clear_mlx_memory_cache_logs_missing_api(caplog):
    from vmlx_engine.mlx_memory import clear_mlx_memory_cache

    fake_mx = SimpleNamespace()

    assert clear_mlx_memory_cache(mx=fake_mx) is None
    assert "No known MLX memory cache clearing API available" in caplog.text


def test_clear_mlx_memory_cache_logs_api_failure(caplog):
    from vmlx_engine.mlx_memory import clear_mlx_memory_cache

    def boom():
        raise RuntimeError("top cache failed")

    fake_mx = SimpleNamespace(clear_cache=boom)

    assert clear_mlx_memory_cache(mx=fake_mx) is None
    assert "MLX memory cache cleanup via mx.clear_cache failed" in caplog.text


def test_runtime_paths_do_not_call_removed_clear_memory_cache_directly():
    repo_root = Path(__file__).resolve().parents[1]
    runtime_paths = [
        repo_root / "vmlx_engine" / "engine" / "simple.py",
        repo_root / "vmlx_engine" / "image_gen.py",
        repo_root / "vmlx_engine" / "mllm_scheduler.py",
        repo_root / "vmlx_engine" / "models" / "mllm.py",
        repo_root / "vmlx_engine" / "reranker.py",
        repo_root / "vmlx_engine" / "scheduler.py",
        repo_root / "vmlx_engine" / "server.py",
        repo_root / "vmlx_engine" / "speculative.py",
        repo_root / "vmlx_engine" / "worker.py",
    ]

    offenders = []
    for path in runtime_paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "clear_memory_cache":
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")

    assert offenders == []


def test_runtime_paths_do_not_call_deprecated_metal_clear_cache_directly():
    repo_root = Path(__file__).resolve().parents[1]
    runtime_paths = [
        repo_root / "vmlx_engine" / "speculative.py",
        repo_root / "vmlx_engine" / "worker.py",
    ]

    offenders = []
    for path in runtime_paths:
        source = path.read_text(encoding="utf-8")
        if "mx.metal.clear_cache()" in source or "metal.clear_cache()" in source:
            offenders.append(str(path.relative_to(repo_root)))

    assert offenders == []


def test_admin_sleep_paths_use_shared_mlx_memory_cache_helper():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "vmlx_engine" / "server.py").read_text(encoding="utf-8")

    soft_start = source.index('async def admin_soft_sleep')
    deep_start = source.index('async def admin_deep_sleep')
    wake_start = source.index('async def admin_wake')
    soft_source = source[soft_start:deep_start]
    deep_source = source[deep_start:wake_start]

    assert "clear_mlx_memory_cache(log=logger)" in soft_source
    assert "clear_mlx_memory_cache(log=logger)" in deep_source
    assert "mx.clear_cache()" not in soft_source
    assert "mx.clear_cache()" not in deep_source
    assert "mx.metal.clear_cache()" not in soft_source
    assert "mx.metal.clear_cache()" not in deep_source
