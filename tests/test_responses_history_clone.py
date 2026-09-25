# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the actual history clone/store owners without model imports."""

import ast
from collections import OrderedDict
import json
from pathlib import Path
import threading
import unittest


def history_owners():
    source = Path(__file__).parents[1] / "vmlx_engine" / "server.py"
    names = {"_clone_response_messages", "_responses_store_history"}
    nodes = [
        node for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "json": json,
        "_responses_history": OrderedDict(),
        "_responses_history_lock": threading.Lock(),
        "_responses_was_reasoning_only": set(),
        "_RESPONSES_HISTORY_MAX": 2,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class TestResponsesHistoryClone(unittest.TestCase):
    def setUp(self):
        self.owner = history_owners()
        self.clone = self.owner["_clone_response_messages"]
        self.store = self.owner["_responses_store_history"]
        self.history = self.owner["_responses_history"]

    def test_media_strings_share_but_mutable_containers_do_not(self):
        media = "audio-payload-" * 10000
        value = [{"role": "user", "content": [{"type": "input_audio", "data": media}]}]
        cloned = self.clone(value)
        self.assertEqual(cloned, value)
        self.assertIs(cloned[0]["content"][0]["data"], media)
        cloned[0]["content"].append({"type": "text", "text": "changed"})
        self.assertEqual(len(value[0]["content"]), 1)

    def test_nested_input_and_sibling_branch_mutations_are_isolated(self):
        value = [{"role": "user", "content": [{"type": "text", "text": "雪 😀"}]}]
        self.store("a", value)
        value[0]["content"][0]["text"] = "caller mutation"
        branch = self.clone(self.history["a"])
        self.store("b", branch)
        branch[0]["content"][0]["text"] = "branch mutation"
        self.assertEqual(self.history["a"], self.history["b"])
        self.assertEqual(self.history["a"][0]["content"][0]["text"], "雪 😀")

    def test_overwrite_eviction_and_reasoning_markers(self):
        self.store("a", [{"role": "user", "content": "a"}], reasoning_only=True)
        self.store("b", [{"role": "user", "content": "b"}])
        self.store("a", [{"role": "user", "content": "replacement"}])
        self.store("c", [{"role": "user", "content": "c"}])
        self.assertEqual(list(self.history), ["a", "c"])
        self.assertEqual(self.history["a"][0]["content"], "replacement")
        self.assertEqual(self.owner["_responses_was_reasoning_only"], set())

    def test_literal_tool_arguments_ids_and_result_order(self):
        value = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "function": {"name": "read_file", "arguments": '{"path":"[\\"beta\\",2]"}'}},
                {"id": "call_2", "function": {"name": "read_file", "arguments": '{"path":"{\\"slot\\":\\"alpha\\"}"}'}},
            ]},
            {"role": "tool", "tool_call_id": "call_2", "content": "14/2/K7"},
            {"role": "tool", "tool_call_id": "call_1", "content": "9/4/P4"},
        ]
        self.assertEqual(self.clone(value), json.loads(json.dumps(value)))

    def test_repeated_mutable_reference_is_copied_independently(self):
        shared = {"items": [1]}
        cloned = self.clone([{"a": shared, "b": shared}])
        cloned[0]["a"]["items"].append(2)
        self.assertEqual(cloned[0]["b"]["items"], [1])
        self.assertEqual(shared["items"], [1])

    def test_legacy_tuple_numeric_key_and_subclass_normalization(self):
        class Text(str):
            pass
        for value in [[{"tuple": (1, 2)}], [{1: "numeric", None: "null"}], [{"text": Text("value")}]]:
            with self.subTest(value=value):
                result = self.clone(value)
                self.assertEqual(result, json.loads(json.dumps(value)))
                if "text" in result[0]:
                    self.assertIs(type(result[0]["text"]), str)

    def test_legacy_nonserializable_and_cycle_shallow_fallback(self):
        cycle = []
        cycle.append(cycle)
        for child in [cycle, {1, 2}]:
            value = [{"nested": child}]
            result = self.clone(value)
            self.assertIsNot(result[0], value[0])
            self.assertIs(result[0]["nested"], child)

    def test_independently_parsed_equal_strings_are_not_interned(self):
        first = "large-media-" * 10000
        second = first.encode().decode()
        self.assertIsNot(first, second)
        self.assertIsNot(self.clone([{"data": first}])[0]["data"], self.clone([{"data": second}])[0]["data"])


if __name__ == "__main__":
    unittest.main()
