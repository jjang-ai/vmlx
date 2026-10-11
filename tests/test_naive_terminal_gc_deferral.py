"""Source-executed text terminal ownership and two-store GC policy."""

import ast
import gc
import textwrap
import weakref
from types import SimpleNamespace

import pytest

from .test_cache_cleanup_phase_timing import ROOT, load_functions


def owner_namespace(order):
    namespace = {
        "logger": SimpleNamespace(debug=lambda *args: None),
        "clear_mlx_memory_cache": lambda **kwargs: order.append("clear"),
    }
    load_functions(
        ROOT / "scheduler.py", {"_cleanup_finished_after_terminal_dispatch"}, namespace
    )
    return namespace


@pytest.mark.parametrize(
    "model,running,finished,expected",
    [
        ("naive_n05_flash", {"r"}, {"r"}, True),
        # Every family's sole terminal owner defers its store collections to the
        # one collection after the cleanup frame (2026-10-11: the extra full
        # collections freed no MLX/NumPy buffers and cost 45-70 ms each).
        ("qwen4", {"r"}, {"r"}, True),
        ("qwen4", {"r", "s"}, {"r"}, False),
        ("naive_n05_flash", {"r", "s"}, {"r"}, False),
        ("naive_n05_flash", {"r", "s"}, {"r", "s"}, False),
        ("naive_n05_flash", set(), {"r"}, False),
        ("naive_n05_flash", set(), set(), False),
    ],
)
def test_sole_terminal_owner_defers(
    monkeypatch, model, running, finished, expected
):
    order = []
    namespace = owner_namespace(order)
    monkeypatch.setattr(gc, "collect", lambda: order.append("gc"))
    obj = SimpleNamespace(_model_type_for_runtime=model, running=dict.fromkeys(running))
    received = []

    def cleanup(ids, **kwargs):
        received.append(kwargs.get("_defer_final_store_gc", False))
        for key in ids:
            obj.running.pop(key, None)

    obj._cleanup_finished = cleanup
    namespace["_cleanup_finished_after_terminal_dispatch"](obj, finished)
    assert received == [expected]


def execute_actual_two_store_region(obj, request, defer):
    source = (ROOT / "scheduler.py").read_text()
    tree = ast.parse(source)
    cleanup = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_cleanup_finished"
    )
    assignments = [node for node in ast.walk(cleanup) if isinstance(node, ast.Assign)]
    start = next(
        node.lineno
        for node in assignments
        if any(
            isinstance(target, ast.Name) and target.id == "naive_prompt"
            for target in node.targets
        )
    )
    end = next(
        node.end_lineno
        for node in assignments
        if any(
            isinstance(target, ast.Name) and target.id == "_stored_block_table"
            for target in node.targets
        )
    )
    body = textwrap.dedent("\n".join(source.splitlines()[start - 1 : end]))
    namespace = dict(
        self=obj,
        request=request,
        request_id="r",
        store_tokens=[1, 2],
        cache_data=["terminal"],
        _paged_store_kwargs={},
        _defer_final_store_gc=defer,
    )
    function = ast.FunctionDef(
        name="actual_store_region",
        args=cleanup.args,
        body=ast.parse(body).body
        + [ast.Return(value=ast.Name(id="_stored_block_table", ctx=ast.Load()))],
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ROOT / "scheduler.py"), "exec"),
        namespace,
    )
    if defer is None:
        return namespace["actual_store_region"](obj, {"r"})
    return namespace["actual_store_region"](obj, {"r"}, _defer_final_store_gc=defer)


@pytest.mark.parametrize("defer", [None, False, True])
def test_actual_checkpoint_then_terminal_store_keeps_checkpoint_gc(defer):
    calls = []

    def store(request_id, tokens, state, **kwargs):
        calls.append((request_id, kwargs.get("defer_post_fence_gc", False)))
        return request_id

    cache = SimpleNamespace(
        store_cache=store, release_cache=lambda key: calls.append(("release", key))
    )
    request = SimpleNamespace(_naive_prompt_checkpoint=([1], ["prompt"]))
    assert (
        execute_actual_two_store_region(
            SimpleNamespace(block_aware_cache=cache), request, defer
        )
        == "r"
    )
    assert calls == [
        ("r:native-prompt", False),
        ("release", "r:native-prompt"),
        ("r", bool(defer)),
    ]
    assert request._naive_prompt_checkpoint is None


@pytest.mark.parametrize("fail", [False, True])
def test_repayment_after_frame_and_on_cleanup_failure(monkeypatch, fail):
    order, refs = [], []
    namespace = owner_namespace(order)
    collect = gc.collect
    monkeypatch.setattr(gc, "collect", lambda: (order.append("gc"), collect())[1])

    class Payload:
        pass

    error = ValueError("cleanup failed")
    obj = SimpleNamespace(
        _model_type_for_runtime="naive_n05_flash", running={"r": object()}
    )

    def cleanup(*args, **kwargs):
        payload = Payload()
        payload.cycle = payload
        refs.append(weakref.ref(payload))
        if fail:
            raise error
        obj.running.clear()

    obj._cleanup_finished = cleanup
    if fail:
        with pytest.raises(ValueError) as caught:
            namespace["_cleanup_finished_after_terminal_dispatch"](obj, {"r"})
        assert caught.value is error
    else:
        namespace["_cleanup_finished_after_terminal_dispatch"](obj, {"r"})
        assert refs[0]() is None
    assert order == ["gc", "clear"]


def test_original_failure_survives_repayment_failure(monkeypatch):
    namespace = owner_namespace([])
    error = ValueError("original")

    def fail_cleanup(*args, **kwargs):
        raise error

    def fail_clear(**kwargs):
        raise RuntimeError("repayment")

    namespace["clear_mlx_memory_cache"] = fail_clear
    obj = SimpleNamespace(
        _model_type_for_runtime="naive_n05_flash",
        running={"r": object()},
        _cleanup_finished=fail_cleanup,
    )
    with pytest.raises(ValueError) as caught:
        namespace["_cleanup_finished_after_terminal_dispatch"](obj, {"r"})
    assert caught.value is error


def test_outer_gc_failure_still_clears(monkeypatch):
    order = []
    namespace = owner_namespace(order)

    def collect():
        order.append("gc")
        raise RuntimeError("collection")

    monkeypatch.setattr(gc, "collect", collect)
    obj = SimpleNamespace(
        _model_type_for_runtime="naive_n05_flash", running={"r": object()}
    )

    def cleanup(*args, **kwargs):
        obj.running.clear()

    obj._cleanup_finished = cleanup
    namespace["_cleanup_finished_after_terminal_dispatch"](obj, {"r"})
    assert order == ["gc", "clear"]


def test_non_deferred_cleanup_error_does_not_add_repayment(monkeypatch):
    """A cleanup that did not defer (another request still running) and failed
    gets no extra repayment collection."""
    order = []
    namespace = owner_namespace(order)
    monkeypatch.setattr(gc, "collect", lambda: order.append("gc"))

    def cleanup(*args, **kwargs):
        assert kwargs == {}
        raise ValueError("original")

    obj = SimpleNamespace(
        _model_type_for_runtime="qwen4",
        running={"r": object(), "s": object()},
        _cleanup_finished=cleanup,
    )
    with pytest.raises(ValueError):
        namespace["_cleanup_finished_after_terminal_dispatch"](obj, {"r"})
    assert order == []
