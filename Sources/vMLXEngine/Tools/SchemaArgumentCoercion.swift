//
//  SchemaArgumentCoercion.swift
//  vMLXEngine
//
//  §338 — JSON-Schema-driven coercion of tool arguments.
//
//  LLMs frequently emit JSON tool arguments where a numeric parameter
//  appears as a *quoted string* (`{"page": "3"}` instead of
//  `{"page": 3}`). When the downstream tool (MCP server, bash
//  validator, etc.) runs JSON-Schema validation against the declared
//  parameters and the schema says `{"type": "integer"}`, the string
//  form fails validation and the tool call errors out — the model
//  did its job but the strict validator rejects it.
//
//  vmlx#47 reported this for MCP tools: a request expecting
//  `{"offset": integer}` got `{"offset": "25"}` and the MCP server
//  returned `Invalid arguments`. The Python fix walked the schema
//  and coerced leaf values before dispatch. This is the Swift mirror.
//
//  Scope: coerces **only** when the raw value is a recognizable form
//  of the target type and the schema is unambiguous at that path. Any
//  ambiguity (no schema, union types, unexpected shape) returns the
//  value unchanged — the tool's own validator then produces its own
//  clear error. Never mangles values the LLM got right.
//
//  Coercions applied:
//    - String → Int    when schema.type == "integer" AND string parses as Int
//    - String → Double when schema.type == "number"  AND string parses as Double
//    - String → Bool   when schema.type == "boolean" AND string is "true"/"false"
//    - Double → Int    when schema.type == "integer" AND value is whole number
//    - Recurses into `object` via `properties`; recurses into `array` via `items`.

import Foundation

/// Coerce tool-argument values to the types their schema declares,
/// limited to the safe cases enumerated above. Non-destructive: any
/// value the schema doesn't cover, or that can't be safely coerced,
/// is passed through unchanged.
///
/// - Parameters:
///   - arguments: raw parsed JSON arguments (the LLM's output after
///     `JSONSerialization.jsonObject`).
///   - schema: the tool's JSON Schema `inputSchema` (typically
///     `{"type":"object","properties":{...}}`), as a parsed
///     `[String: Any]`.
/// - Returns: a shallow-copied args dict with leaves coerced where
///   unambiguous.
public func coerceToolArguments(
    _ arguments: [String: Any],
    schema: [String: Any]
) -> [String: Any] {
    guard let props = schema["properties"] as? [String: Any] else {
        return arguments
    }
    var out = arguments
    for (key, raw) in arguments {
        guard let propSchema = props[key] as? [String: Any] else { continue }
        out[key] = coerceValue(raw, schema: propSchema)
    }
    return out
}

/// Recursive leaf coercion against a single property schema. Extracted
/// so array+object recursion stays consistent with the top-level walk.
private func coerceValue(_ value: Any, schema: [String: Any]) -> Any {
    let t = (schema["type"] as? String) ?? ""

    // Object recursion: walk `properties`.
    if t == "object", let dict = value as? [String: Any] {
        return coerceToolArguments(dict, schema: schema)
    }

    // Array recursion: coerce each element against `items` if present.
    if t == "array", let arr = value as? [Any],
       let itemSchema = schema["items"] as? [String: Any] {
        return arr.map { coerceValue($0, schema: itemSchema) }
    }

    // Integer: accept Int / Double-that-is-whole / parseable numeric String.
    if t == "integer" {
        if value is Int { return value }
        if let d = value as? Double, d.truncatingRemainder(dividingBy: 1) == 0,
           d >= Double(Int.min), d <= Double(Int.max)
        {
            return Int(d)
        }
        if let s = value as? String, let i = Int(s) {
            return i
        }
        return value
    }

    // Number: accept Int / Double / parseable Double String.
    if t == "number" {
        if value is Double { return value }
        if let i = value as? Int { return Double(i) }
        if let s = value as? String, let d = Double(s) {
            return d
        }
        return value
    }

    // Boolean: accept Bool / "true"/"false" string (case-insensitive).
    if t == "boolean" {
        if value is Bool { return value }
        if let s = value as? String {
            switch s.lowercased() {
            case "true": return true
            case "false": return false
            default: return value
            }
        }
        return value
    }

    // String / no-coercion / unknown type: pass through.
    return value
}
