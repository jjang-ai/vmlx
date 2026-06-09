# SPDX-License-Identifier: Apache-2.0

import json

from bench.structured_output_repair_report import main, repair_records


CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "visible_text": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["visible_text"],
}


def _response_format():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "clip",
            "schema": CLIP_SCHEMA,
        },
    }


def test_repair_records_preserves_raw_vs_repaired_counts():
    records = [
        {"id": "native", "text": '{"visible_text": ["OK"]}'},
        {
            "id": "qwen35",
            "text": '{"visible_text": "CLIPFARM STRESS STREAM", "0-15 M00 ALERT START"}',
        },
        {"id": "bad", "text": "not json"},
    ]

    repaired, summary = repair_records(records, response_format=_response_format())

    assert summary["records"] == 3
    assert summary["valid"] == 2
    assert summary["invalid"] == 1
    assert summary["raw_json_ok"] == 1
    assert summary["raw_schema_ok"] == 1
    assert summary["repair_needed"] == 1
    assert summary["repair_actions"]["syntax_repair"] == 1
    assert repaired[1]["structured_output"]["parsed"] == {
        "visible_text": ["CLIPFARM STRESS STREAM", "0-15 M00 ALERT START"]
    }
    assert repaired[1]["structured_output"]["raw_json_ok"] is False
    assert repaired[1]["structured_output"]["repair_needed"] is True
    assert repaired[2]["structured_output"]["is_valid"] is False


def test_repair_records_reports_raw_and_repaired_rates():
    records = [
        {"id": "native", "text": '{"visible_text": ["OK"]}'},
        {
            "id": "repaired",
            "text": '{"visible_text": "CLIPFARM STRESS STREAM", "0-15 M00 ALERT START"}',
        },
        {"id": "invalid", "text": "not json"},
    ]

    _, summary = repair_records(records, response_format=_response_format())

    assert summary["raw_json_ok_rate"] == 1 / 3
    assert summary["raw_schema_ok_rate"] == 1 / 3
    assert summary["validated_rate"] == 2 / 3
    assert summary["invalid_rate"] == 1 / 3
    assert summary["repair_needed_rate"] == 1 / 3


def test_repair_records_does_not_store_schema_invalid_json_as_parsed_payload():
    records = [
        {
            "id": "schema_invalid",
            "text": '{"visible_text": 123}',
        },
    ]

    repaired, summary = repair_records(records, response_format=_response_format())

    assert summary["records"] == 1
    assert summary["valid"] == 0
    assert summary["invalid"] == 1
    assert repaired[0]["structured_output"]["is_valid"] is False
    assert repaired[0]["structured_output"]["raw_json_ok"] is True
    assert repaired[0]["structured_output"]["raw_schema_ok"] is False
    assert repaired[0]["structured_output"]["parsed"] is None


def test_repair_records_supports_xml_root_and_required_fields():
    records = [
        {
            "id": "native_xml",
            "text": "<catalog><visible_text>OK</visible_text></catalog>",
        },
        {
            "id": "fenced_xml",
            "text": "Result:\n```xml\n<catalog><visible_text>FIXED</visible_text></catalog>\n```",
        },
        {"id": "bad_xml", "text": "<catalog><label>missing</label></catalog>"},
    ]

    repaired, summary = repair_records(
        records,
        response_format={"type": "xml"},
        xml_root_tag="catalog",
        required_xml_fields=("visible_text",),
    )

    assert summary["records"] == 3
    assert summary["valid"] == 2
    assert summary["invalid"] == 1
    assert summary["raw_xml_ok"] == 2
    assert summary["raw_fields_ok"] == 1
    assert summary["repair_needed"] == 1
    assert summary["repair_actions"]["extract_xml_block"] == 1
    assert repaired[1]["structured_output"]["xml"] == (
        "<catalog><visible_text>FIXED</visible_text></catalog>"
    )
    assert repaired[1]["structured_output"]["raw_xml_ok"] is False
    assert repaired[2]["structured_output"]["is_valid"] is False


def test_cli_writes_repaired_jsonl_and_summary(tmp_path):
    input_path = tmp_path / "clips.jsonl"
    output_path = tmp_path / "repaired.jsonl"
    summary_path = tmp_path / "summary.json"
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(CLIP_SCHEMA))
    input_path.write_text(
        '{"id":"native","text":"{\\"visible_text\\":[\\"OK\\"]}"}\n'
        '{"id":"fixed","text":"{\\"visible_text\\": \\"ONE\\", \\"TWO\\"}"}\n'
    )

    rc = main(
        [
            "--input-jsonl",
            str(input_path),
            "--out-jsonl",
            str(output_path),
            "--summary-json",
            str(summary_path),
            "--schema-json",
            str(schema_path),
            "--schema-name",
            "clip",
        ]
    )

    assert rc == 0
    summary = json.loads(summary_path.read_text())
    assert summary["records"] == 2
    assert summary["valid"] == 2
    assert summary["raw_json_ok"] == 1
    assert summary["repair_needed"] == 1
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[1]["structured_output"]["parsed"] == {"visible_text": ["ONE", "TWO"]}


def test_cli_writes_xml_repair_jsonl_and_summary(tmp_path):
    input_path = tmp_path / "clips.xml.jsonl"
    output_path = tmp_path / "repaired.xml.jsonl"
    summary_path = tmp_path / "summary.xml.json"
    input_path.write_text(
        '{"id":"native","text":"<catalog><visible_text>OK</visible_text></catalog>"}\n'
        '{"id":"fixed","text":"Result:\\n```xml\\n<catalog><visible_text>ONE</visible_text></catalog>\\n```"}\n'
    )

    rc = main(
        [
            "--input-jsonl",
            str(input_path),
            "--out-jsonl",
            str(output_path),
            "--summary-json",
            str(summary_path),
            "--format",
            "xml",
            "--xml-root-tag",
            "catalog",
            "--required-xml-field",
            "visible_text",
        ]
    )

    assert rc == 0
    summary = json.loads(summary_path.read_text())
    assert summary["records"] == 2
    assert summary["valid"] == 2
    assert summary["raw_xml_ok"] == 1
    assert summary["repair_needed"] == 1
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[1]["structured_output"]["xml"] == (
        "<catalog><visible_text>ONE</visible_text></catalog>"
    )
