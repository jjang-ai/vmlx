# SPDX-License-Identifier: Apache-2.0
"""
Tests for structured output (JSON Schema) functionality.

Tests the JSON parsing, validation, and response_format handling.
"""

import json
import pytest
from vmlx_engine.api.tool_calling import (
    validate_json_schema,
    extract_json_from_text,
    extract_xml_from_text,
    parse_json_output,
    build_json_system_prompt,
)
from vmlx_engine.api.models import ResponseFormat, ResponseFormatJsonSchema


class TestValidateJsonSchema:
    """Tests for validate_json_schema function."""

    def test_valid_object(self):
        """Test validation of a valid object against schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        data = {"name": "Alice", "age": 30}
        is_valid, error = validate_json_schema(data, schema)
        assert is_valid is True
        assert error is None

    def test_invalid_type(self):
        """Test validation fails for wrong type."""
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        data = {"age": "not an integer"}
        is_valid, error = validate_json_schema(data, schema)
        assert is_valid is False
        assert error is not None
        assert "integer" in error.lower() or "type" in error.lower()

    def test_missing_required(self):
        """Test validation fails for missing required field."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = {}
        is_valid, error = validate_json_schema(data, schema)
        assert is_valid is False
        assert error is not None

    def test_array_validation(self):
        """Test validation of array types."""
        schema = {
            "type": "object",
            "properties": {"colors": {"type": "array", "items": {"type": "string"}}},
        }
        # Valid
        is_valid, _ = validate_json_schema({"colors": ["red", "blue"]}, schema)
        assert is_valid is True

        # Invalid - number in array
        is_valid, _ = validate_json_schema({"colors": ["red", 123]}, schema)
        assert is_valid is False


class TestExtractJsonFromText:
    """Tests for extract_json_from_text function."""

    def test_pure_json(self):
        """Test extraction from pure JSON string."""
        text = '{"name": "test", "value": 42}'
        result = extract_json_from_text(text)
        assert result == {"name": "test", "value": 42}

    def test_json_in_markdown(self):
        """Test extraction from markdown code block."""
        text = """Here is the result:
```json
{"name": "test", "value": 42}
```
"""
        result = extract_json_from_text(text)
        assert result == {"name": "test", "value": 42}

    def test_json_in_plain_code_block(self):
        """Test extraction from plain code block without json marker."""
        text = """Result:
```
{"items": [1, 2, 3]}
```
"""
        result = extract_json_from_text(text)
        assert result == {"items": [1, 2, 3]}

    def test_json_embedded_in_text(self):
        """Test extraction from JSON embedded in text."""
        text = 'The answer is: {"result": true} and that concludes the analysis.'
        result = extract_json_from_text(text)
        assert result == {"result": True}

    def test_array_extraction(self):
        """Test extraction of JSON arrays."""
        text = 'Colors: ["red", "green", "blue"]'
        result = extract_json_from_text(text)
        assert result == ["red", "green", "blue"]

    def test_no_json(self):
        """Test returns None when no JSON found."""
        text = "This is just plain text with no JSON."
        result = extract_json_from_text(text)
        assert result is None

    def test_invalid_json(self):
        """Test returns None for invalid JSON."""
        text = '{"broken": json, not valid}'
        result = extract_json_from_text(text)
        assert result is None

    def test_nested_json(self):
        """Test extraction of nested JSON."""
        text = '{"outer": {"inner": {"deep": "value"}}}'
        result = extract_json_from_text(text)
        assert result == {"outer": {"inner": {"deep": "value"}}}

    def test_repairs_markdown_trailing_comma_and_python_literals(self):
        text = """```json
{"ok": True, "missing": None, "items": ["a", "b",],}
```"""
        result = extract_json_from_text(text)
        assert result == {"ok": True, "missing": None, "items": ["a", "b"]}

    def test_repairs_missing_closing_brace(self):
        result = extract_json_from_text('prefix {"name": "clip", "score": 7')
        assert result == {"name": "clip", "score": 7}

    def test_repairs_adjacent_string_fragments_for_schema_array(self):
        schema = {
            "type": "object",
            "properties": {
                "visible_text": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["visible_text"],
        }
        text = '{"visible_text": "CLIPFARM STRESS STREAM", "0-15 M00 ALERT START"}'
        result = extract_json_from_text(text, schema)
        assert result == {
            "visible_text": ["CLIPFARM STRESS STREAM", "0-15 M00 ALERT START"]
        }

    def test_coerces_schema_string_field_to_array(self):
        schema = {
            "type": "object",
            "properties": {
                "visible_text": {"type": "array", "items": {"type": "string"}}
            },
        }
        result = extract_json_from_text('{"visible_text": "single"}', schema)
        assert result == {"visible_text": ["single"]}


class TestExtractXmlFromText:
    """Tests for conservative XML extraction from model output."""

    def test_extracts_xml_from_markdown(self):
        text = """Result:
```xml
<catalog><visible_text>CLIPFARM</visible_text></catalog>
```"""
        assert (
            extract_xml_from_text(text, root_tag="catalog")
            == "<catalog><visible_text>CLIPFARM</visible_text></catalog>"
        )

    def test_extracts_xml_from_surrounding_text(self):
        text = "before <tool><name>x</name></tool> after"
        assert extract_xml_from_text(text, root_tag="tool") == "<tool><name>x</name></tool>"

    def test_xml_fails_closed_when_invalid(self):
        assert extract_xml_from_text("before <tool><name>x</tool> after") is None


class TestParseJsonOutput:
    """Tests for parse_json_output function."""

    def test_no_response_format(self):
        """Test with no response_format returns original text."""
        text = "Hello, world!"
        cleaned, parsed, is_valid, error = parse_json_output(text, None)
        assert cleaned == text
        assert parsed is None
        assert is_valid is True
        assert error is None

    def test_text_format(self):
        """Test with type='text' returns original text."""
        text = "Hello, world!"
        response_format = {"type": "text"}
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert cleaned == text
        assert parsed is None
        assert is_valid is True

    def test_json_object_valid(self):
        """Test json_object mode extracts valid JSON."""
        text = '{"name": "test"}'
        response_format = {"type": "json_object"}
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"name": "test"}
        assert is_valid is True
        assert error is None

    def test_json_object_invalid(self):
        """Test json_object mode fails for non-JSON."""
        text = "This is not JSON"
        response_format = {"type": "json_object"}
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed is None
        assert is_valid is False
        assert "Failed to extract" in error

    def test_json_schema_valid(self):
        """Test json_schema mode validates against schema."""
        text = '{"name": "Alice", "age": 30}'
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
        }
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"name": "Alice", "age": 30}
        assert is_valid is True
        assert error is None

    def test_json_schema_invalid(self):
        """Test json_schema mode fails validation for wrong data."""
        text = '{"name": 123}'  # name should be string
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"name": 123}
        assert is_valid is False
        assert "validation failed" in error.lower()

    def test_json_schema_repairs_qwen_adjacent_string_array(self):
        text = '{"visible_text": "CLIPFARM STRESS STREAM", "0-15 M00 ALERT START"}'
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "clip",
                "schema": {
                    "type": "object",
                    "properties": {
                        "visible_text": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["visible_text"],
                },
            },
        }
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert cleaned == text
        assert parsed == {
            "visible_text": ["CLIPFARM STRESS STREAM", "0-15 M00 ALERT START"]
        }
        assert is_valid is True
        assert error is None

    def test_json_schema_coerces_string_to_array(self):
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "clip",
                "schema": {
                    "type": "object",
                    "properties": {
                        "visible_text": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                },
            },
        }
        _, parsed, is_valid, error = parse_json_output(
            '{"visible_text": "CLIPFARM"}', response_format
        )
        assert parsed == {"visible_text": ["CLIPFARM"]}
        assert is_valid is True
        assert error is None

    def test_response_format_model(self):
        """Test with ResponseFormat Pydantic model."""
        text = '{"result": true}'
        response_format = ResponseFormat(type="json_object")
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"result": True}
        assert is_valid is True

    def test_response_format_with_json_schema_model(self):
        """Test with ResponseFormat and ResponseFormatJsonSchema models."""
        text = '{"colors": ["red", "blue"]}'
        json_schema = ResponseFormatJsonSchema(
            name="colors",
            schema={
                "type": "object",
                "properties": {
                    "colors": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["colors"],
            },
        )
        response_format = ResponseFormat(type="json_schema", json_schema=json_schema)
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"colors": ["red", "blue"]}
        assert is_valid is True


class TestBuildJsonSystemPrompt:
    """Tests for build_json_system_prompt function."""

    def test_no_format(self):
        """Test returns None for no format."""
        result = build_json_system_prompt(None)
        assert result is None

    def test_text_format(self):
        """Test returns None for text format."""
        result = build_json_system_prompt({"type": "text"})
        assert result is None

    def test_json_object(self):
        """Test prompt for json_object mode."""
        result = build_json_system_prompt({"type": "json_object"})
        assert result is not None
        assert "valid JSON" in result
        assert "only" in result.lower()

    def test_json_schema(self):
        """Test prompt for json_schema mode."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "description": "A person object",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }
        result = build_json_system_prompt(response_format)
        assert result is not None
        assert "person" in result
        assert "A person object" in result
        assert "JSON Schema" in result

    def test_json_schema_model(self):
        """Test prompt with ResponseFormat model."""
        json_schema = ResponseFormatJsonSchema(
            name="output", description="Output format", schema={"type": "object"}
        )
        response_format = ResponseFormat(type="json_schema", json_schema=json_schema)
        result = build_json_system_prompt(response_format)
        assert result is not None
        assert "output" in result


class TestInjectJsonInstruction:
    """Tests for _inject_json_instruction function in server."""

    def test_inject_new_system_message(self):
        """Test injecting instruction when no system message exists."""
        from vmlx_engine.server import _inject_json_instruction

        messages = [{"role": "user", "content": "Hello"}]
        result = _inject_json_instruction(messages, "Return JSON only")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "Return JSON only" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_append_to_existing_system(self):
        """Test appending to existing system message."""
        from vmlx_engine.server import _inject_json_instruction

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = _inject_json_instruction(messages, "Return JSON only")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "You are helpful." in result[0]["content"]
        assert "Return JSON only" in result[0]["content"]

    def test_does_not_modify_original(self):
        """Test that original messages are not modified."""
        from vmlx_engine.server import _inject_json_instruction

        original = [{"role": "user", "content": "Hello"}]
        original_content = original[0]["content"]
        result = _inject_json_instruction(original, "Return JSON only")

        # Original should be unchanged
        assert len(original) == 1
        assert original[0]["content"] == original_content


class TestStructuredOutputApiRepair:
    """Route-level tests proving repair is used by API response handling."""

    @pytest.mark.asyncio
    async def test_chat_completion_repairs_and_canonicalizes_schema_json(
        self, monkeypatch
    ):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ChatCompletionRequest, Message
        from vmlx_engine.engine.base import GenerationOutput

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False

            async def chat(self, *, messages, **kwargs):
                return GenerationOutput(
                    text='{"visible_text": "CLIPFARM STRESS STREAM", "0-15 M00 ALERT START"}',
                    prompt_tokens=5,
                    completion_tokens=8,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_engine", _Engine())
        monkeypatch.setattr(server, "_served_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_model_type", "llm")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_mcp_manager", None)
        monkeypatch.setattr(server, "_max_prompt_tokens", None)

        response = await server.create_chat_completion(
            ChatCompletionRequest(
                model="loaded-model",
                messages=[Message(role="user", content="return clip metadata")],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "clip",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "visible_text": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                }
                            },
                            "required": ["visible_text"],
                        },
                    },
                },
            ),
            fastapi_request=None,
        )

        assert json.loads(response.choices[0].message.content) == {
            "visible_text": ["CLIPFARM STRESS STREAM", "0-15 M00 ALERT START"]
        }


# Integration test - run only if model available
@pytest.mark.skip(reason="Requires model loaded")
class TestStructuredOutputIntegration:
    """Integration tests for structured output with real model."""

    @pytest.fixture
    def client(self):
        """Create OpenAI client pointing to local server."""
        from openai import OpenAI

        return OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

    def test_json_object_mode(self, client):
        """Test json_object mode returns valid JSON."""
        response = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": "List 3 colors"}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        # Should be valid JSON
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_json_schema_mode(self, client):
        """Test json_schema mode returns valid structured data."""
        response = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": "List 3 colors"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "colors",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "colors": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["colors"],
                    },
                },
            },
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        assert "colors" in data
        assert isinstance(data["colors"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
