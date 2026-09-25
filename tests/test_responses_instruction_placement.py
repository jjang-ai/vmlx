"""CPU-only route assembly regressions; execute the owning route statements."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


def assemble(instructions, previous, current, *, native_order=True):
    source = (Path(__file__).parents[1] / "vmlx_engine/server.py").read_text()
    tree = ast.parse(source)
    route = next(
        n
        for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "create_response"
    )

    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        )

    start = next(i for i, n in enumerate(route.body) if assigns(n, "history_messages"))
    stop = next(
        i
        for i, n in enumerate(route.body[start:], start)
        if assigns(n, "messages")
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "_canonicalize_mimo_v26_tool_history"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=route.body[start:stop], type_ignores=[])
    )

    def convert(items, request_instructions, **kwargs):
        return (
            [{"role": "system", "content": request_instructions}]
            if request_instructions
            else []
        ) + copy.deepcopy(items)

    env = dict(
        request=SimpleNamespace(
            input=current,
            instructions=instructions,
            previous_response_id="prior" if previous else None,
            model="unit",
        ),
        _preserve_mm=False,
        _native_omni_resp=False,
        _responses_has_media=False,
        _responses_input_to_messages=convert,
        _preserves_native_developer_role=lambda _: True,
        _preserves_native_system_order=lambda _: native_order,
        _responses_get_history=lambda *a, **k: copy.deepcopy(previous),
        _enforce_text_only_override=lambda *a: None,
        _messages_requested_modalities=lambda _: [],
        _responses_should_scrub_multimodal_history_for_followup=lambda *a, **k: False,
        logger=SimpleNamespace(debug=lambda *a: None),
    )
    exec(compile(module, "<actual Responses route assembly>", "exec"), env)
    return env["messages"], env["history_messages"]


class InstructionPlacement(unittest.TestCase):
    def test_chained_instructions_precede_history_and_remain_transient(self):
        history = [
            {"role": "user", "content": "inventory"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call1"}]},
        ]
        result = [{"role": "tool", "tool_call_id": "call1", "content": "42"}]
        messages, persisted = assemble("current instructions", history, result)
        self.assertEqual(
            messages,
            [{"role": "system", "content": "current instructions"}] + history + result,
        )
        self.assertEqual(persisted, history + result)
        self.assertEqual(history[0]["role"], "user")

    def test_preserves_authored_native_system_developer_order_and_duplicates(self):
        history = [
            {"role": "user", "content": "u"},
            {"role": "developer", "content": "native later"},
            {"role": "assistant", "content": "a"},
        ]
        current = [
            {"role": "system", "content": "native later"},
            {"role": "user", "content": "next"},
        ]
        messages, persisted = assemble("transient", history, current)
        self.assertEqual(messages[1:], history + current)
        self.assertEqual(persisted, history + current)
        self.assertEqual(messages[0], {"role": "system", "content": "transient"})

    def test_omitted_or_empty_instructions_do_not_create_a_message(self):
        for instructions in (None, ""):
            history = [{"role": "user", "content": "u"}]
            current = [{"role": "user", "content": "next"}]
            messages, persisted = assemble(instructions, history, current)
            self.assertEqual(messages, history + current)
            self.assertEqual(persisted, messages)

    def test_generic_history_keeps_base_then_current_instruction_order(self):
        previous = [
            {"role": "system", "content": "base instructions"},
            {"role": "user", "content": "first"},
        ]
        current = [{"role": "user", "content": "next"}]
        messages, persisted = assemble(
            "new instructions", previous, current, native_order=False
        )
        self.assertEqual(
            messages,
            previous + [{"role": "system", "content": "new instructions"}] + current,
        )
        self.assertEqual(persisted, previous + current)

    def test_first_request_and_replacement_do_not_persist_instructions(self):
        current = [{"role": "user", "content": "first"}]
        initial, persisted = assemble("old", [], current)
        messages, again = assemble(
            "new", persisted, [{"role": "user", "content": "second"}]
        )
        self.assertEqual(initial[0]["content"], "old")
        self.assertEqual(messages[0]["content"], "new")
        self.assertNotIn("old", str(messages))
        self.assertNotIn("new", str(again))


if __name__ == "__main__":
    unittest.main()
