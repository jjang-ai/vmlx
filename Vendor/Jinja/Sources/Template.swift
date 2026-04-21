//
//  Template.swift
//
//
//  Created by John Mai on 2024/3/23.
//

import Foundation

public struct Template {
    var parsed: Program

    public init(_ template: String) throws {
        let tokens = try tokenize(template, options: PreprocessOptions(trimBlocks: true, lstripBlocks: true))
        self.parsed = try parse(tokens: tokens)
    }

    public func render(_ items: [String: Any?]) throws -> String {
        return try self.render(items, environment: nil)
    }

    func render(_ items: [String: Any?], environment parentEnvironment: Environment?) throws -> String {
        let base = parentEnvironment ?? Environment.sharedBase
        let env = Environment(parent: base)

        for (key, value) in items {
            if let value {
                try env.set(name: key, value: value)
            }
        }

        let interpreter = Interpreter(env: env)
        let result = try interpreter.run(program: self.parsed) as! StringValue

        return result.value
    }
}
