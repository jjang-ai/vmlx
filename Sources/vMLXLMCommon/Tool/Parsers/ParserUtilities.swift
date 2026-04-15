// Copyright © 2025 Apple Inc.

import Foundation

// MARK: - JSON to Sendable Bridge

/// Convert a JSON-deserialized value to `any Sendable`.
///
/// `JSONSerialization` returns `Any`, but all JSON types it produces
/// (String, NSNumber, NSNull, Array, Dictionary) are Sendable.
func asSendable(_ value: Any) -> any Sendable {
    switch value {
    case let s as String: return s
    case let n as NSNumber: return n
    case let a as [Any]: return a.map(asSendable)
    case let d as [String: Any]: return d.mapValues(asSendable)
    case let null as NSNull: return null
    default: return "\(value)"
    }
}

/// Deserialize JSON data, returning a Sendable value.
func deserializeJSON(_ data: Data) -> (any Sendable)? {
    guard let object = try? JSONSerialization.jsonObject(with: data) else { return nil }
    return asSendable(object)
}

// MARK: - Basic Deserialization

/// Deserialize a string value to JSON or return as string.
///
/// Attempts JSON parsing first, falling back to the original string value.
/// Reference: Python's `ast.literal_eval` / `json.loads` pattern
func tryParseJSON(_ value: String) -> (any Sendable)? {
    guard let data = value.data(using: .utf8) else { return nil }
    return deserializeJSON(data)
}

func deserialize(_ value: String) -> any Sendable {
    tryParseJSON(value) ?? value
}

// MARK: - Schema Lookup Functions

/// Check if a parameter is a string type in the tool schema.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/glm47.py
func isStringType(funcName: String, argName: String, tools: [[String: any Sendable]]?) -> Bool {
    guard let type = getParameterType(funcName: funcName, paramName: argName, tools: tools) else {
        return false
    }
    return type == "string"
}

/// Get the parameter type from tool schema for a specific function and parameter.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/qwen3_coder.py
func getParameterType(
    funcName: String, paramName: String, tools: [[String: any Sendable]]?
) -> String? {
    guard let tools else { return nil }
    for tool in tools {
        guard let function = tool["function"] as? [String: any Sendable],
            function["name"] as? String == funcName,
            let parameters = function["parameters"] as? [String: any Sendable],
            let properties = parameters["properties"] as? [String: any Sendable],
            let param = properties[paramName] as? [String: any Sendable],
            let type = param["type"] as? String
        else { continue }
        return type
    }
    return nil
}

/// Get parameter configuration for a function from tools schema.
func getParameterConfig(
    funcName: String, tools: [[String: any Sendable]]?
) -> [String: any Sendable] {
    guard let tools else { return [:] }
    for tool in tools {
        guard let function = tool["function"] as? [String: any Sendable],
            function["name"] as? String == funcName,
            let parameters = function["parameters"] as? [String: any Sendable],
            let properties = parameters["properties"] as? [String: any Sendable]
        else { continue }
        return properties
    }
    return [:]
}

// MARK: - Schema Type Extraction

/// Extract types from JSON schema (handles anyOf, oneOf, allOf, enums).
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/minimax_m2.py
func extractTypesFromSchema(_ schema: [String: any Sendable]?) -> [String] {
    guard let schema else { return ["string"] }

    var types: Set<String> = []

    // Handle direct "type" field
    if let typeValue = schema["type"] {
        if let typeString = typeValue as? String {
            types.insert(typeString)
        } else if let typeArray = typeValue as? [String] {
            types.formUnion(typeArray)
        }
    }

    // Handle enum - infer types from enum values
    if let enumValues = schema["enum"] as? [any Sendable], !enumValues.isEmpty {
        for value in enumValues {
            switch value {
            case is NSNull: types.insert("null")
            case is Bool: types.insert("boolean")
            case is Int: types.insert("integer")
            case is Double: types.insert("number")
            case is String: types.insert("string")
            case is [any Sendable]: types.insert("array")
            case is [String: any Sendable]: types.insert("object")
            default: break
            }
        }
    }

    // Handle anyOf, oneOf, allOf - recursively extract types
    for choiceField in ["anyOf", "oneOf", "allOf"] {
        if let choices = schema[choiceField] as? [[String: any Sendable]] {
            for choice in choices {
                types.formUnion(extractTypesFromSchema(choice))
            }
        }
    }

    return types.isEmpty ? ["string"] : Array(types)
}

// MARK: - Type Conversion

/// Convert parameter value based on multiple possible types.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/minimax_m2.py
func convertValueWithTypes(_ value: String, types: [String]) -> any Sendable {
    let lowerValue = value.lowercased()

    // Handle null values
    if ["null", "none", "nil"].contains(lowerValue) {
        return NSNull()
    }

    let normalizedTypes = Set(types.map { $0.lowercased() })

    // Priority: integer > number > boolean > object > array > string
    let typePriority = [
        "integer", "int", "number", "float", "boolean", "bool",
        "object", "array", "string", "str", "text",
    ]

    for paramType in typePriority {
        guard normalizedTypes.contains(paramType) else { continue }

        switch paramType {
        case "string", "str", "text":
            return value

        case "integer", "int":
            if let intVal = Int(value) {
                return intVal
            }

        case "number", "float":
            if let floatVal = Double(value) {
                let intVal = Int(floatVal)
                return floatVal != Double(intVal) ? floatVal : intVal
            }

        case "boolean", "bool":
            let trimmed = lowerValue.trimmingCharacters(in: .whitespaces)
            if ["true", "1", "yes", "on"].contains(trimmed) {
                return true
            } else if ["false", "0", "no", "off"].contains(trimmed) {
                return false
            }

        case "object", "array":
            if let json = tryParseJSON(value) {
                return json
            }

        default:
            continue
        }
    }

    // Fallback: try JSON parse, then return as string
    return tryParseJSON(value) ?? value
}

/// Convert parameter value based on schema type.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/qwen3_coder.py
func convertParameterValue(
    _ value: String, paramName: String, funcName: String, tools: [[String: any Sendable]]?
) -> any Sendable {
    guard let paramType = getParameterType(funcName: funcName, paramName: paramName, tools: tools)
    else {
        return value
    }

    let type = paramType.lowercased()

    // String types - return as-is
    if ["string", "str", "text", "varchar", "char", "enum"].contains(type) {
        return value
    }

    // Integer types
    if type.hasPrefix("int") || type.hasPrefix("uint")
        || type.hasPrefix("long") || type.hasPrefix("short")
        || type.hasPrefix("unsigned")
    {
        return Int(value) ?? value
    }

    // Float types
    if type.hasPrefix("num") || type.hasPrefix("float") {
        if let floatVal = Double(value) {
            let intVal = Int(floatVal)
            return floatVal != Double(intVal) ? floatVal : intVal
        }
        return value
    }

    // Boolean types
    if ["boolean", "bool", "binary"].contains(type) {
        return ["true", "1", "yes", "on"].contains(
            value.lowercased().trimmingCharacters(in: .whitespaces))
    }

    // Object/Array types - JSON decode
    if ["object", "array"].contains(type) || type.hasPrefix("dict") || type.hasPrefix("list") {
        if let json = tryParseJSON(value) {
            return json
        }
    }

    return value
}

// MARK: - String Utilities

/// Extract name from a potentially quoted string.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/minimax_m2.py
func extractName(_ nameStr: String) -> String {
    let trimmed = nameStr.trimmingCharacters(in: .whitespaces)
    if (trimmed.hasPrefix("\"") && trimmed.hasSuffix("\""))
        || (trimmed.hasPrefix("'") && trimmed.hasSuffix("'"))
    {
        return String(trimmed.dropFirst().dropLast())
    }
    return trimmed
}
