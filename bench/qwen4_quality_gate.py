"""Deterministic, scored API regression cases for Qwen4 runtime changes.

Run against one idle localhost server at a time. Results include the exact
request, raw response, usage, and request-matched cache telemetry. This bounded
regression set is not a general model-capability benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def cases(tokenizer_path: Path, extended: bool = False):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    result = []
    for i, (question, expected) in enumerate(
        [
            ("What is 37 * 24? Return only the integer.", "888"),
            ("What is 1000 - 7 * 83? Return only the integer.", "419"),
            (
                "What is the greatest common divisor of 84 and 126? Return only the integer.",
                "42",
            ),
            (
                "A train travels 180 km in 2.5 hours. What is its average speed in km/h? Return only the number.",
                "72",
            ),
            (
                "All zibs are loms. No loms are tars. Can any zib be a tar? Return only YES or NO.",
                "NO",
            ),
            (
                "Sort these integers ascending: 9, -3, 0, 9, 2. Return only a JSON array.",
                [-3, 0, 2, 9, 9],
            ),
            (
                "Convert these records to a JSON object mapping name to score: Ada=7; Bo=0; Cy=12. Return only the JSON object.",
                {"Ada": 7, "Bo": 0, "Cy": 12},
            ),
            (
                'Given {"a":false,"b":null,"c":0,"d":""}, return the same JSON object exactly in meaning, preserving value types. Return only JSON.',
                {"a": False, "b": None, "c": 0, "d": ""},
            ),
            (
                "Return only JSON with keys total and count for the numbers 4, 8, -2, 10.",
                {"total": 20, "count": 4},
            ),
            ("Find the missing term: 3, 6, 12, 24, __. Return only the integer.", "48"),
        ]
    ):
        result.append(
            {
                "id": f"answer-{i}",
                "prompt": question,
                "expected": expected,
                "kind": "json" if isinstance(expected, (dict, list)) else "text",
            }
        )
    coding = [
        (
            "stable_unique",
            "return the first occurrence of each integer in input list xs, preserving order",
            [([[3, 1, 3, 2, 1]], [3, 1, 2]), ([[]], []), ([[0, 0, -1]], [0, -1])],
        ),
        (
            "gcd",
            "return the nonnegative greatest common divisor of integers a and b, including zero inputs",
            [([84, 126], 42), ([0, 5], 5), ([-12, 8], 4), ([0, 0], 0)],
        ),
        (
            "balanced",
            "return whether string s has balanced parentheses; ignore all characters except ( and )",
            [(["a(b(c)d)"], True), (["())("], False), ([""], True), (["(()"], False)],
        ),
        (
            "two_sum",
            "given list nums and integer target, return indices of two distinct elements summing to target, or None if absent; return the indices in ascending order",
            [([[2, 7, 11, 15], 9], [0, 1]), ([[3, 3], 6], [0, 1]), ([[1, 2], 8], None)],
        ),
        (
            "merge_intervals",
            "merge overlapping closed intervals in list intervals and return them sorted by start; touching endpoints overlap",
            [
                ([[[5, 7], [1, 3], [2, 4]]], [[1, 4], [5, 7]]),
                ([[]], []),
                ([[[1, 2], [2, 3]]], [[1, 3]]),
            ],
        ),
        (
            "rotate",
            "return list xs rotated right by integer k; handle empty lists and k larger than the list length",
            [([[1, 2, 3], 4], [3, 1, 2]), ([[], 7], []), ([[1, 2], 0], [1, 2])],
        ),
    ]
    for name, spec, checks in coding:
        result.append(
            {
                "id": "code-" + name,
                "kind": "code",
                "name": name,
                "checks": checks,
                "prompt": f"Write a Python function named {name} to {spec}. Return only the function code. No imports, type annotations or explanation. Use ordinary Python builtins. Keep it under 30 lines.",
            }
        )
    for i, (city, unit) in enumerate([("Oslo", "celsius"), ("Tokyo", "fahrenheit")]):
        result.append(
            {
                "id": f"tool-{i}",
                "kind": "tool",
                "prompt": f"Use get_weather to obtain the weather in {city} with unit {unit}.",
                "expected": {"city": city, "unit": unit},
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                    "unit": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                    },
                                },
                                "required": ["city", "unit"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
            }
        )
    for context in [1024, 4096, 8192, 32768] + ([65536, 131000] if extended else []):
        for position in ("middle", "tail"):
            answer = hashlib.sha256(
                f"qwen4-gate-{context}-{position}".encode()
            ).hexdigest()[:12]
            needle = (
                f"\nThe unique account recovery code for account ORCHID is {answer}.\n"
            )
            filler = "This reference record describes routine inventory with no recovery credentials.\n"
            prefix = f"Qualification case {context}-{position}. Read the records and answer the final question.\n"
            suffix = "\nWhat is the recovery code for account ORCHID? Return only the exact code."

            def count(s):
                return len(tokenizer.encode(s, add_special_tokens=False).ids)

            def render(
                n,
                position=position,
                prefix=prefix,
                filler=filler,
                needle=needle,
                suffix=suffix,
            ):
                split = n // 2 if position == "middle" else n
                return prefix + filler * split + needle + filler * (n - split) + suffix

            low, high = 0, context
            while low < high:
                mid = (low + high + 1) // 2
                if count(render(mid)) <= context - 5:
                    low = mid
                else:
                    high = mid - 1
            result.append(
                {
                    "id": f"retrieval-{context}-{position}",
                    "kind": "text",
                    "prompt": render(low),
                    "expected": answer,
                    "nominal_context": context,
                }
            )
    for case in result:
        case["prompt"] = (
            f"Independent case {case['id']} / 20260908-Q4.\n" + case["prompt"]
        )
    return result


CODE_RUNNER = r"""
import ast,json,sys,resource
resource.setrlimit(resource.RLIMIT_CPU,(1,1))
data=json.load(sys.stdin); tree=ast.parse(data['code'])
allowed_attributes={'append','get','items','values','keys','pop','sort','copy','add','remove','lower','upper','strip','split','join','setdefault','extend','index','count'}
for node in ast.walk(tree):
 if isinstance(node,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.With,ast.AsyncWith,ast.Global,ast.Nonlocal)):
  raise ValueError('unsupported statement')
 if isinstance(node,ast.Name) and node.id.startswith('__'): raise ValueError('private name')
 if isinstance(node,ast.Attribute) and node.attr not in allowed_attributes: raise ValueError('unsupported attribute')
allowed={k:__builtins__.__dict__[k] for k in ['len','range','enumerate','zip','sorted','reversed','sum','min','max','abs','all','any','int','str','bool','list','tuple','dict','set','isinstance','ValueError']}
scope={'__builtins__':allowed};exec(compile(tree,'<generated>','exec'),scope)
out=[scope[data['name']](*args) for args,expected in data['checks']]
print(json.dumps(out))
"""


def grade(case, message):
    text = (message.get("content") or "").strip()
    if case["kind"] == "tool":
        calls = message.get("tool_calls") or []
        if len(calls) != 1:
            return False, "expected one tool call"
        f = calls[0]["function"]
        return f["name"] == "get_weather" and json.loads(f["arguments"]) == case[
            "expected"
        ], "tool schema and arguments"
    if case["kind"] == "json":
        parsed = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text))
        return json.dumps(parsed, sort_keys=True) == json.dumps(
            case["expected"], sort_keys=True
        ), "typed JSON equality"
    if case["kind"] == "code":
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
        code = match.group(1) if match else text
        # Generated functions execute in an isolated, bounded child with
        # imports/private attributes forbidden and restricted builtins.
        payload = {**case, "code": code}
        p = subprocess.run(
            [sys.executable, "-I", "-c", CODE_RUNNER],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=3,
        )
        if p.returncode:
            return False, p.stderr[-800:]
        return json.loads(p.stdout) == [
            expected for args, expected in case["checks"]
        ], "behavioral code tests"
    return text.strip("` \n") == case["expected"], "exact answer"


def request(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def run(url, model, dataset, output, label):
    rows = []
    for case in dataset:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "temperature": 0,
            "max_tokens": 512 if case["kind"] == "code" else 96,
            "seed": 20260908,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "enable_thinking": False,
        }
        if case.get("tools"):
            body.update(tools=case["tools"], tool_choice="auto")
        start = time.monotonic()
        try:
            response = request(url, body)
            ok, reason = grade(case, response["choices"][0]["message"])
            with urllib.request.urlopen(
                url.split("/v1/")[0] + "/health", timeout=5
            ) as f:
                health = json.load(f)
            cache = (
                health.get("scheduler", {})
                .get("batch_generator", {})
                .get("last_cache_execution", {})
            )
            rows.append(
                {
                    "id": case["id"],
                    "kind": case["kind"],
                    "passed": ok,
                    "reason": reason,
                    "wall_s": time.monotonic() - start,
                    "request": body,
                    "response": response,
                    "cache": cache,
                    "cache_matches_request": cache.get("request_id")
                    == response.get("id"),
                    "mtp": health.get("mtp"),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "id": case["id"],
                    "kind": case["kind"],
                    "passed": False,
                    "reason": str(exc),
                    "request": body,
                }
            )
        output.write_text(
            json.dumps(
                {
                    "label": label,
                    "passed": sum(r["passed"] for r in rows),
                    "total": len(rows),
                    "rows": rows,
                },
                indent=2,
            )
        )
        print(label, case["id"], "PASS" if rows[-1]["passed"] else "FAIL", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="qwen-next")
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--extended", action="store_true")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    rows = run(a.url, a.model, cases(a.tokenizer, a.extended), a.output, a.label)
    return 0 if rows and all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
