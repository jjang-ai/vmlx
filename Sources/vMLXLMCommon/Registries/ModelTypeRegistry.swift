// Copyright © 2024 Apple Inc.

import Foundation

public actor ModelTypeRegistry {

    /// Creates an empty registry.
    public init() {
        self.creators = [:]
    }

    /// Creates a registry with given creators.
    public init(creators: [String: (Data) throws -> any LanguageModel]) {
        self.creators = creators
    }

    private var creators: [String: (Data) throws -> any LanguageModel]

    /// Add a new model to the type registry.
    public func registerModelType(
        _ type: String, creator: @escaping (Data) throws -> any LanguageModel
    ) {
        creators[type] = creator
    }

    /// Given a `modelType` and configuration data instantiate a new `LanguageModel`.
    public func createModel(configuration: Data, modelType: String) throws -> sending LanguageModel
    {
        guard let creator = creators[modelType] else {
            throw ModelFactoryError.unsupportedModelType(modelType)
        }
        return try creator(configuration)
    }

    /// Whether a creator for the given `modelType` is registered.
    /// Used by regression tests to guard against accidental model
    /// drops (Swift analogue of the v1.3.51 `mlx-vlm` bundled-import
    /// post-mortem — see ``ModelFactoryRegistrationTests``).
    public func contains(modelType: String) -> Bool {
        creators[modelType] != nil
    }

    /// Return a snapshot of every registered `modelType` key.
    /// Intended for diagnostics / regression tests only — the main
    /// code paths should use `createModel` directly.
    public func registeredModelTypes() -> [String] {
        Array(creators.keys).sorted()
    }

}
