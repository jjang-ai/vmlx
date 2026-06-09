#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply vMLX structured-output repair diagnostics to benchmark JSONL.

This is caller-side post-generation repair. It does not perform guided decoding
and must not be used as evidence that a model natively emits valid JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vmlx_engine.api.tool_calling import repair_json_output, repair_xml_output


TEXT_FIELDS = ("text", "raw_response", "response", "output", "content")


def _load_json_arg(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    value = value.strip()
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(value)


def _response_format_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.format == "xml":
        return {"type": "xml"}
    if args.response_format_json:
        loaded = _load_json_arg(args.response_format_json)
        if not isinstance(loaded, dict):
            raise SystemExit("--response-format-json must be a JSON object")
        return loaded
    if args.schema_json:
        schema = _load_json_arg(args.schema_json)
        if not isinstance(schema, dict):
            raise SystemExit("--schema-json must be a JSON object")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": args.schema_name,
                "schema": schema,
            },
        }
    return {"type": "json_object"}


def _record_text(record: dict[str, Any], explicit_field: str | None) -> str:
    fields = (explicit_field,) if explicit_field else TEXT_FIELDS
    for field in fields:
        if not field:
            continue
        value = record.get(field)
        if isinstance(value, str):
            return value
    raise KeyError(f"record has no text field in {', '.join(fields)}")


def repair_records(
    records: list[dict[str, Any]],
    *,
    response_format: dict[str, Any],
    text_field: str | None = None,
    xml_root_tag: str | None = None,
    required_xml_fields: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repaired_records: list[dict[str, Any]] = []
    summary = {
        "records": 0,
        "valid": 0,
        "invalid": 0,
        "raw_json_ok": 0,
        "raw_schema_ok": 0,
        "raw_xml_ok": 0,
        "raw_fields_ok": 0,
        "repair_needed": 0,
        "repair_actions": {},
    }

    for index, record in enumerate(records):
        text = _record_text(record, text_field)
        if response_format.get("type") == "xml":
            report = repair_xml_output(
                text,
                root_tag=xml_root_tag,
                required_fields=required_xml_fields,
            )
        else:
            report = repair_json_output(text, response_format)
        out = dict(record)
        is_valid = bool(report["is_valid"])
        if response_format.get("type") == "xml":
            out["structured_output"] = {
                "index": index,
                "is_valid": is_valid,
                "error": report["error"],
                "raw_xml_ok": report["raw_xml_ok"],
                "raw_fields_ok": report["raw_fields_ok"],
                "repair_needed": report["repair_needed"],
                "repair_actions": report["repair_actions"],
                "xml": report["xml"] if is_valid else None,
            }
        else:
            out["structured_output"] = {
                "index": index,
                "is_valid": is_valid,
                "error": report["error"],
                "raw_json_ok": report["raw_json_ok"],
                "raw_schema_ok": report["raw_schema_ok"],
                "repair_needed": report["repair_needed"],
                "repair_actions": report["repair_actions"],
                "parsed": report["parsed"] if is_valid else None,
            }
        repaired_records.append(out)

        summary["records"] += 1
        summary["valid" if report["is_valid"] else "invalid"] += 1
        if report.get("raw_json_ok"):
            summary["raw_json_ok"] += 1
        if report.get("raw_schema_ok"):
            summary["raw_schema_ok"] += 1
        if report.get("raw_xml_ok"):
            summary["raw_xml_ok"] += 1
        if report.get("raw_fields_ok"):
            summary["raw_fields_ok"] += 1
        if report["repair_needed"]:
            summary["repair_needed"] += 1
        for action in report["repair_actions"]:
            actions = summary["repair_actions"]
            actions[action] = actions.get(action, 0) + 1

    return repaired_records, summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        records.append(record)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--text-field")
    parser.add_argument("--format", choices=("json", "xml"), default="json")
    parser.add_argument("--response-format-json")
    parser.add_argument("--schema-json")
    parser.add_argument("--schema-name", default="benchmark_output")
    parser.add_argument("--xml-root-tag")
    parser.add_argument("--required-xml-field", action="append", default=[])
    args = parser.parse_args(argv)

    response_format = _response_format_from_args(args)
    records = _read_jsonl(Path(args.input_jsonl))
    repaired_records, summary = repair_records(
        records,
        response_format=response_format,
        text_field=args.text_field,
        xml_root_tag=args.xml_root_tag,
        required_xml_fields=tuple(args.required_xml_field),
    )
    summary["response_format"] = response_format
    _write_jsonl(Path(args.out_jsonl), repaired_records)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return 0 if summary["invalid"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
