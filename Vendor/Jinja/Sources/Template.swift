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
            } else {
                // Previously nil values were silently skipped, which left
                // the variable UndefinedValue. But Nemotron-family chat
                // templates use `{% set X = X if X is defined else None %}`
                // which — under swift-jinja's flaky undefined handling —
                // ended up with X as StringValue("") in practice (root
                // cause of hybrid-SSM cache-key drift per §240). Explicitly
                // materialize `nil` as NullValue so `is none` tests return
                // true and `is defined` returns true. Callers that want
                // UndefinedValue can simply not include the key.
                try env.set(name: key, value: NullValue())
            }
        }

        let interpreter = Interpreter(env: env)
        let result = try interpreter.run(program: self.parsed) as! StringValue

        return result.value
    }
}
