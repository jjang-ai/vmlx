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
