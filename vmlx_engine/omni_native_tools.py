"""Native Omni tool contracts, before rendering or executing model output.

The vendor template renders mapping arguments and omits result IDs. Normalize
only validated objects and order a complete result batch by its call IDs.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import re

from fastapi import HTTPException
from jsonschema import validators
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012, specification_with


def _object_arguments(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"duplicate argument key {key!r}")
            result[key] = item
        return result

    def nonfinite(value):
        raise ValueError(f"non-finite JSON value {value}")

    if isinstance(value, str):
        value = json.loads(value, object_pairs_hook=pairs, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be a JSON object")
    # Mapping input must meet the same JSON contract as string input.
    json.dumps(value, allow_nan=False)
    return value


def _schema_validator(schema):
    cls = validators.validator_for(schema)
    cls.check_schema(schema)
    spec = specification_with(schema.get('$schema', ''), default=DRAFT202012)
    root = Resource.from_contents(schema, default_specification=spec)
    registry = Registry()  # No network retrieval callback.
    resolver = registry.resolver_with_root(root)
    # Walk schema resources, not arbitrary dictionaries in defaults/examples.
    # Check references even in currently unused optional properties.
    remaining = [(root, resolver)]
    visited = set()
    while remaining:
        resource, scope = remaining.pop()
        if id(resource.contents) in visited:
            continue
        visited.add(id(resource.contents))
        if len(visited) > 4096:
            raise ValueError('tool schema exceeds native reference traversal limit')
        node = resource.contents
        if isinstance(node, dict):
            for key in ('$ref', '$dynamicRef', '$recursiveRef'):
                if key in node:
                    try:
                        scope.lookup(node[key])
                    except Exception as error:
                        raise ValueError(f'unresolvable offline schema reference {node[key]!r}') from error
        remaining.extend((child, scope.in_subresource(child)) for child in resource.subresources())
    return cls(schema, registry=registry)


@dataclass
class NativeToolContract:
    messages: list
    template_tools: list
    validators: dict
    enabled: bool
    active: bool

    def validate_arguments(self, name, arguments):
        if name not in self.validators:
            raise ValueError(f'unknown native tool {name!r}')
        arguments = _object_arguments(arguments)
        try:
            error = next(self.validators[name].iter_errors(arguments), None)
        except Exception as error:
            raise ValueError(f'tool {name!r} schema could not be resolved offline') from error
        if error is not None:
            raise ValueError(f'tool {name!r} arguments violate schema: {error.message}')
        return arguments


class NativeToolOutput:
    """Hold control markup until the complete call batch passes validation.

    Feed only the visible rail. Reasoning is handled separately by the native
    rail splitter and must never be searched for executable calls.
    """
    _OPEN = ('<tool_call>', '<function=')

    def __init__(self, contract):
        self.contract = contract
        self.pending = ''
        self.in_calls = False

    def feed(self, text):
        self.pending += text
        if self.in_calls:
            return ''
        starts = [self.pending.find(marker) for marker in self._OPEN if marker in self.pending]
        if starts:
            offset = min(starts)
            visible, self.pending = self.pending[:offset], self.pending[offset:]
            self.in_calls = True
            return visible
        held = max((size for marker in self._OPEN for size in range(1, len(marker))
                    if self.pending.endswith(marker[:size])), default=0)
        visible = self.pending[:-held] if held else self.pending
        self.pending = self.pending[-held:] if held else ''
        return visible

    def finish(self, finish_reason):
        if not self.in_calls:
            if self.pending.startswith(('<tool', '<function')):
                raise ValueError('incomplete native tool markup')
            tail, self.pending = self.pending, ''
            return tail, []
        if not self.contract.enabled:
            raise ValueError('native tool call generated while tool_choice is none')
        if finish_reason != 'stop':
            raise ValueError('native tool generation ended before a complete stop')
        from .tool_parsers.nemotron_tool_parser import NemotronToolParser
        parser = NemotronToolParser()
        remaining = self.pending.strip()
        calls = []
        while remaining:
            wrapped = remaining.startswith('<tool_call>')
            expression = (r'<tool_call>\s*(<function=[^>]+>.*?</function>)\s*</tool_call>'
                          if wrapped else r'(<function=[^>]+>.*?</function>)')
            match = re.match(expression, remaining, re.DOTALL)
            if match is None:
                raise ValueError('malformed or incomplete native tool call batch')
            segment = match.group(1)
            function = re.fullmatch(r'<function=([^>]+)>(.*?)</function>', segment, re.DOTALL)
            body = function.group(2).strip()
            if body.startswith('{'):
                _object_arguments(body)
            elif body:
                parameters = list(parser.PARAM_PATTERN.finditer(body))
                cursor, names = 0, set()
                for parameter in parameters:
                    name = parameter.group(1).strip()
                    if body[cursor:parameter.start()].strip() or not name or name in names:
                        raise ValueError('ambiguous native XML tool parameters')
                    names.add(name)
                    cursor = parameter.end()
                if body[cursor:].strip() or not parameters:
                    raise ValueError('malformed native XML tool parameters')
            extracted = parser.extract_tool_calls(segment, {'tools': self.contract.template_tools})
            if len(extracted.tool_calls) != 1:
                raise ValueError('native tool parser did not produce exactly one call')
            call = extracted.tool_calls[0]
            arguments = self.contract.validate_arguments(call['name'], call['arguments'])
            calls.append({'id': call['id'], 'type': 'function', 'function': {
                'name': call['name'], 'arguments': json.dumps(arguments, ensure_ascii=False, allow_nan=False)}})
            remaining = remaining[match.end():].strip()
        self.pending = ''
        return '', calls


def prepare_native_tools(tools, choice, messages):
    """Return a request-local contract; never mutate a client's transcript."""
    try:
        if choice not in (None, 'auto', 'none'):
            raise ValueError('native Omni tool_choice supports auto or none; forced choices are unsupported')
        catalog = []
        checks = {}
        for tool in tools or []:
            item = tool.model_dump(exclude_none=True) if hasattr(tool, 'model_dump') else deepcopy(tool)
            if not isinstance(item, dict) or item.get('type') != 'function':
                raise ValueError('native Omni supports function tools only')
            function = item.get('function')
            name = function.get('name') if isinstance(function, dict) else None
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]+', name):
                raise ValueError('invalid native function tool name')
            if name in checks:
                raise ValueError(f'duplicate native tool name {name!r}')
            schema = function.get('parameters', {})
            if not isinstance(schema, dict):
                raise ValueError('native tool parameters must be a schema object')
            checks[name] = _schema_validator(schema)
            catalog.append(item)
        history = deepcopy(messages)
        active = bool(catalog) or any(m.get('tool_calls') or m.get('role') == 'tool' for m in history)
        contract = NativeToolContract(history, catalog if choice != 'none' else [], checks,
                                      bool(catalog) and choice != 'none', active)
        normalized = []
        seen_ids = set()
        index = 0
        while index < len(history):
            message = history[index]
            if message.get('role') == 'tool':
                raise ValueError('orphan native tool-result history')
            normalized.append(message)
            index += 1
            calls = message.get('tool_calls')
            if not calls:
                continue
            if message.get('role') != 'assistant' or not isinstance(calls, list):
                raise ValueError('tool calls require an assistant call list')
            ordered_ids = []
            for call in calls:
                identifier = call.get('id')
                if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
                    raise ValueError('tool call IDs must be nonempty and unique')
                seen_ids.add(identifier)
                ordered_ids.append(identifier)
                function = call.get('function')
                if call.get('type', 'function') != 'function' or not isinstance(function, dict):
                    raise ValueError('invalid function tool history')
                # Historical calls may use a catalog/schema since removed or
                # changed by the client. Validate their wire shape, not today's
                # execution schema; only newly generated calls use checks.
                name = function.get('name')
                if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]+', name):
                    raise ValueError('invalid historical native tool name')
                function['arguments'] = _object_arguments(function.get('arguments'))
            results = {}
            while index < len(history) and history[index].get('role') == 'tool':
                result = history[index]
                identifier = result.get('tool_call_id')
                if identifier not in ordered_ids or identifier in results:
                    raise ValueError('duplicate or unknown native tool-result ID')
                content = result.get('content')
                if isinstance(content, list) and all(p.get('type') == 'text' for p in content):
                    result['content'] = ''.join(p.get('text', '') for p in content)
                elif not isinstance(content, str):
                    raise ValueError('native tool results require text content')
                results[identifier] = result
                index += 1
            if set(results) != set(ordered_ids):
                raise ValueError('native tool-result history requires a complete contiguous result batch')
            normalized.extend(results[identifier] for identifier in ordered_ids)
        contract.messages = normalized
        return contract
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise HTTPException(400, f'Invalid native Omni tool contract: {error}') from error
    except Exception as error:
        # JSON Schema validators raise SchemaError outside ValueError.
        from jsonschema.exceptions import SchemaError
        if isinstance(error, SchemaError):
            raise HTTPException(400, f'Invalid native Omni tool schema: {error.message}') from error
        raise
