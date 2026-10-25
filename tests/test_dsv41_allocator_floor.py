"""DeepSeek-V4.1 raises the serving allocator-cache bound to its prefill floor; nothing else changes."""
import json

from vmlx_engine import mlx_memory as M


class _FakeMx:
    def __init__(self):
        self.limit = None

    def set_cache_limit(self, n):
        prev, self.limit = self.limit, n
        return prev


def _bundle(tmp_path, model_type):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return tmp_path


def test_floor_only_for_dsv41(tmp_path):
    assert M.model_cache_limit_floor_bytes(_bundle(tmp_path, "deepseek_v41")) == 16 * 1024**3
    assert M.model_cache_limit_floor_bytes(_bundle(tmp_path, "deepseek_v4")) is None
    assert M.model_cache_limit_floor_bytes(tmp_path / "missing") is None
    assert M.model_cache_limit_floor_bytes(None) is None


def test_floor_raises_default_not_override(monkeypatch):
    resident = 197 * 1024**3
    monkeypatch.delenv("VMLX_MLX_CACHE_LIMIT_MB", raising=False)
    mx = _FakeMx()
    assert M.apply_serving_cache_limit(resident, mx=mx) == 4 * 1024**3
    assert M.apply_serving_cache_limit(resident, mx=mx, floor_bytes=16 * 1024**3) == 16 * 1024**3
    monkeypatch.setenv("VMLX_MLX_CACHE_LIMIT_MB", "2048")
    assert M.apply_serving_cache_limit(resident, mx=mx, floor_bytes=16 * 1024**3) == 2048 * 1024**2
    monkeypatch.setenv("VMLX_MLX_CACHE_LIMIT_MB", "0")
    assert M.apply_serving_cache_limit(resident, mx=mx, floor_bytes=16 * 1024**3) is None
