# SPDX-License-Identifier: Apache-2.0
"""Speed contract: a text-only DFlash2 target (Ripple) routes its affine projections through the lane matmul.

Without it every extra verify row cost about one decode step (measured Ripple JANG_4D: verify w16 667 -> 157 ms,
decode 11.5 -> 28.5 tok/s on code). These tests fail if a refactor drops the install or ignores its opt-out."""
from types import SimpleNamespace

import vmlx_engine.speculative as spec
from vmlx_engine.metal import lane_qmm
from vmlx_engine.models.llm import MLXLanguageModel


def _model(monkeypatch, dflash2: bool, calls: list):
    monkeypatch.setattr(spec, "is_dflash2_enabled", lambda: dflash2)
    monkeypatch.setattr(lane_qmm, "install", lambda m, tile=True: calls.append((m, tile)) or {"lane": 1, "tiled": 1, "skipped": {}})
    m = MLXLanguageModel.__new__(MLXLanguageModel)
    inner = SimpleNamespace(args=SimpleNamespace(model_type="ricursive"))
    inner.language_model = inner
    m.model = inner
    return m, inner


def test_text_dflash2_target_installs_the_lane_matmul(monkeypatch):
    monkeypatch.delenv("VMLX_LANE_QMM", raising=False)
    calls: list = []
    m, inner = _model(monkeypatch, True, calls)
    m._install_dflash2_lane_matmul()
    assert calls and calls[0][0] is inner and calls[0][1] is True


def test_lane_matmul_is_not_installed_without_dflash2_or_when_opted_out(monkeypatch):
    calls: list = []
    m, _ = _model(monkeypatch, False, calls)
    m._install_dflash2_lane_matmul()
    assert not calls
    monkeypatch.setenv("VMLX_LANE_QMM", "0")
    m, _ = _model(monkeypatch, True, calls)
    m._install_dflash2_lane_matmul()
    assert not calls


def test_load_installs_it(monkeypatch):
    """load() itself calls the install (the measured default path), not only an optional helper."""
    import inspect

    assert "_install_dflash2_lane_matmul()" in inspect.getsource(MLXLanguageModel.load)
