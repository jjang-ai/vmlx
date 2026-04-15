// Copyright © 2024 Apple Inc.

import Foundation

public actor ProcessorTypeRegistry {

    /// Creates an empty registry.
    public init() {
        self.creators = [:]
    }

    /// Creates a registry with given creators.
    public init(creators: [String: (Data, any Tokenizer) throws -> any UserInputProcessor]) {
        self.creators = creators
    }

    private var creators: [String: (Data, any Tokenizer) throws -> any UserInputProcessor]

    /// Add a new model to the type registry.
    public func registerProcessorType(
        _ type: String,
        creator:
            @escaping (
                Data,
                any Tokenizer
            ) throws -> any UserInputProcessor
    ) {
        creators[type] = creator
    }

    /// Given a `processorType` and configuration data instantiate a new `UserInputProcessor`.
    public func createModel(configuration: Data, processorType: String, tokenizer: any Tokenizer)
        throws -> sending any UserInputProcessor
    {
        guard let creator = creators[processorType] else {
            throw ModelFactoryError.unsupportedProcessorType(processorType)
        }
        return try creator(configuration, tokenizer)
    }

}
