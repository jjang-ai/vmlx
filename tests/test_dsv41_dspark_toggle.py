"""DeepSeek-V4.1 follows ``vmlx serve --dspark on|off|auto`` like every bundled-DSpark family.

The CLI exports an explicit choice as ``VMLX_DSPARK=1|0`` before the model loads; the V4.1 loader reads it first, then
the diagnostic ``VMLX_DSV41_DSPARK`` lever, then attaches the bundle's draft by default. Without ``model._dspark`` the
scheduler serves the plain SingleBatchGenerator (AR), so ``--dspark off`` leaves the DSpark lane entirely.
No model weights: the draft loader is a stub that records its calls.
"""

from __future__ import annotations

import inspect
import json
import sys
import types
from argparse import Namespace
from types import SimpleNamespace

import pytest

from vmlx_engine.cli import export_dspark_choice
from vmlx_engine.loaders import load_jangtq_dsv41 as L

LEVERS = ("VMLX_DSPARK", "VMLX_DSV41_DSPARK")


@pytest.fixture(autouse=True)
def _clean_levers(monkeypatch):
    # setenv first so monkeypatch restores the original state, also for the os.environ write export_dspark_choice does.
    for name in LEVERS:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture
def draft_stub(monkeypatch):
    """jang_tools.dsv41.dspark.load_dspark replaced by a recorder (the real one needs the 1 GB draft)."""
    calls = []

    def load_dspark(path, args, embed, head):
        calls.append(path)
        return SimpleNamespace(bs=4)

    pkg = types.ModuleType("jang_tools.dsv41")
    mod = types.ModuleType("jang_tools.dsv41.dspark")
    mod.load_dspark = load_dspark
    pkg.dspark = mod
    monkeypatch.setitem(sys.modules, "jang_tools.dsv41", pkg)
    monkeypatch.setitem(sys.modules, "jang_tools.dsv41.dspark", mod)
    return calls


def _bundle(tmp_path, has_mtp=True):
    runtime = {"bundle_has_mtp": True, "mtp_mode": "dspark_subdir", "dspark_dir": "dspark"} if has_mtp else {}
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41", "runtime": runtime}))
    return tmp_path


def _model():
    return SimpleNamespace(runtime=SimpleNamespace(args=object(), model=SimpleNamespace(embed=object(), head=object())))


@pytest.mark.parametrize("mode, exported", [("on", "1"), ("off", "0"), ("auto", None)])
def test_cli_exports_the_explicit_choice(monkeypatch, mode, exported):
    export_dspark_choice(Namespace(dspark=mode))
    import os

    assert os.environ.get("VMLX_DSPARK") == exported


def test_cli_auto_keeps_an_inherited_choice(monkeypatch):
    # auto exports nothing; a choice already in the environment stays (same as the other DSpark families).
    monkeypatch.setenv("VMLX_DSPARK", "0")
    export_dspark_choice(Namespace(dspark="auto"))
    assert L.dsv41_dspark_decision()[0] is False


@pytest.mark.parametrize(
    "mode, lever, attached",
    [
        ("auto", None, True),   # default: the bundle's draft is used
        ("on", None, True),
        ("off", None, False),   # the bug: --dspark off used to attach the draft anyway
        ("auto", "0", False),   # diagnostic lever under auto
        ("auto", "1", True),
        ("on", "0", True),      # an explicit --dspark on is the user's choice
        ("off", "1", False),    # an explicit --dspark off is the user's choice
    ],
)
def test_flag_governs_the_loader(monkeypatch, tmp_path, draft_stub, mode, lever, attached):
    if lever is not None:
        monkeypatch.setenv("VMLX_DSV41_DSPARK", lever)
    export_dspark_choice(Namespace(dspark=mode))
    model = _model()
    L._maybe_load_dsv41_dspark(model, _bundle(tmp_path))
    assert (getattr(model, "_dspark", None) is not None) is attached
    # Off never even reads the draft weights.
    assert draft_stub == ([str(tmp_path)] if attached else [])


@pytest.mark.parametrize("spelling", ["0", "false", "off", "no", " OFF "])
def test_off_spellings(monkeypatch, spelling):
    monkeypatch.setenv("VMLX_DSPARK", spelling)
    assert L.dsv41_dspark_decision()[0] is False
    monkeypatch.delenv("VMLX_DSPARK")
    monkeypatch.setenv("VMLX_DSV41_DSPARK", spelling)
    assert L.dsv41_dspark_decision()[0] is False


def test_decision_reason_names_the_source(monkeypatch):
    assert "auto" in L.dsv41_dspark_decision()[1]
    monkeypatch.setenv("VMLX_DSPARK", "0")
    assert "--dspark off" in L.dsv41_dspark_decision()[1]
    monkeypatch.setenv("VMLX_DSPARK", "1")
    assert "--dspark on" in L.dsv41_dspark_decision()[1]
    monkeypatch.delenv("VMLX_DSPARK")
    monkeypatch.setenv("VMLX_DSV41_DSPARK", "0")
    assert "VMLX_DSV41_DSPARK" in L.dsv41_dspark_decision()[1]


@pytest.mark.parametrize("mode", ["on", "auto"])
def test_bundle_without_a_draft_serves_ar(monkeypatch, tmp_path, draft_stub, mode):
    export_dspark_choice(Namespace(dspark=mode))
    model = _model()
    L._maybe_load_dsv41_dspark(model, _bundle(tmp_path, has_mtp=False))
    assert getattr(model, "_dspark", None) is None
    assert draft_stub == []


def test_off_logs_the_receipt(monkeypatch, tmp_path, draft_stub, caplog):
    export_dspark_choice(Namespace(dspark="off"))
    with caplog.at_level("INFO", logger=L.logger.name):
        L._maybe_load_dsv41_dspark(_model(), _bundle(tmp_path))
    assert any("dSpark draft not attached: --dspark off" in r.getMessage() for r in caplog.records)


def test_scheduler_selects_the_dsv41_lane_only_when_the_draft_is_attached():
    # The DSpark lane is keyed on model._dspark (set only by the loader above); without it the plain
    # SingleBatchGenerator branch serves the request.
    from vmlx_engine.scheduler import Scheduler

    src = inspect.getsource(Scheduler._create_batch_generator)
    lane = src.index("DSV41DSparkGenerator(")
    gate = src.rindex('getattr(self.model, "_dspark", None) is not None', 0, lane)
    assert "max_num_seqs" in src[gate:lane]
    assert src.index("return SingleBatchGenerator(", lane) > lane


def test_serve_parser_accepts_the_three_values():
    from vmlx_engine import cli

    src = inspect.getsource(cli)
    block = src[src.index('"--dspark",'):][:200]
    assert 'choices=("auto", "on", "off")' in block and 'default="auto"' in block
    serve = inspect.getsource(cli.serve_command)
    # Exported before the model loads, so the V4.1 loader sees the choice.
    assert serve.index("export_dspark_choice(args)") < serve.index("load_model(")
