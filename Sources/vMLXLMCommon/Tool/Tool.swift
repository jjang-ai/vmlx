// Copyright © 2025 Apple Inc.

import Foundation

public typealias ToolSpec = [String: any Sendable]

/// Protocol defining the requirements for a tool.
public protocol ToolProtocol: Sendable {
    /// The JSON Schema describing the tool's interface.
    var schema: ToolSpec { get }
}

public struct Tool<Input: Codable, Output: Codable>: ToolProtocol {
    /// The JSON Schema describing the tool's interface.
    public let schema: ToolSpec

    /// The handler for the tool.
    public let handler: @Sendable (Input) async throws -> Output

    /// The name of the tool extracted from the schema
    public var name: String {
        let function = schema["function"] as? [String: any Sendable]
        let name = function?["name"] as? String
        return name ?? ""
    }

    public init(
        name: String,
        description: String,
        parameters: [ToolParameter],
        handler: @Sendable @escaping (Input) async throws -> Output
    ) {
        var properties = [String: any Sendable]()
        var requiredParams = [String]()

        for param in parameters {
            properties[param.name] = param.schema
            if param.isRequired {
                requiredParams.append(param.name)
            }
        }

        self.schema =
            [
                "type": "function",
                "function": [
                    "name": name,
                    "description": description,
                    "parameters": [
                        "type": "object",
                        "properties": properties,
                        "required": requiredParams,
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec

        self.handler = handler
    }

    public init(schema: ToolSpec, handler: @Sendable @escaping (Input) async throws -> Output) {
        self.schema = schema
        self.handler = handler
    }
}
