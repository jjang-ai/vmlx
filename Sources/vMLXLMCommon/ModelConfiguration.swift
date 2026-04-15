// Copyright © 2024 Apple Inc.

import Foundation

/// Configuration for a given model name with overrides for prompts and tokens.
///
/// See e.g. `MLXLM.ModelRegistry` for an example of use.
public struct ModelConfiguration: Sendable {

    public enum DirectoryError: LocalizedError, Equatable {
        case unresolvedModelDirectory(String)
        case unresolvedTokenizerDirectory(String)

        public var errorDescription: String? {
            switch self {
            case .unresolvedModelDirectory(let id):
                return "Model configuration '\(id)' has not been resolved to a local directory."
            case .unresolvedTokenizerDirectory(let id):
                return "Tokenizer source '\(id)' has not been resolved to a local directory."
            }
        }
    }

    public enum Identifier: Sendable {
        case id(String, revision: String = "main")
        case directory(URL)
    }

    public var id: Identifier

    public var name: String {
        switch id {
        case .id(let id, _):
            id
        case .directory(let url):
            url.deletingLastPathComponent().lastPathComponent + "/" + url.lastPathComponent
        }
    }

    /// The resolved local directory containing model files.
    ///
    /// - Throws: ``DirectoryError/unresolvedModelDirectory(_:)`` if this configuration still
    ///   identifies a remote model by ID rather than a local directory.
    package var modelDirectory: URL {
        get throws {
            switch id {
            case .directory(let directory):
                return directory
            case .id(let id, _):
                throw DirectoryError.unresolvedModelDirectory(id)
            }
        }
    }

    /// The resolved local directory containing tokenizer files.
    ///
    /// If ``tokenizerSource`` is `nil`, this falls back to ``modelDirectory``.
    ///
    /// - Throws: ``DirectoryError/unresolvedTokenizerDirectory(_:)`` if the tokenizer still
    ///   points to a remote source by ID, or ``DirectoryError/unresolvedModelDirectory(_:)``
    ///   if no separate tokenizer source is set and the model itself is unresolved.
    package var tokenizerDirectory: URL {
        get throws {
            switch tokenizerSource {
            case .directory(let directory):
                return directory
            case .id(let id, _):
                throw DirectoryError.unresolvedTokenizerDirectory(id)
            case nil:
                return try modelDirectory
            }
        }
    }

    /// Where to load the tokenizer from when it differs from the model directory.
    ///
    /// - `.id`: download from a remote provider (requires a ``Downloader``)
    /// - `.directory`: load from a local path
    /// - `nil`: use the same directory as the model
    public let tokenizerSource: TokenizerSource?

    /// A reasonable default prompt for the model
    public var defaultPrompt: String

    /// Additional tokens to use for end of string (specified as strings, converted to IDs at runtime)
    public var extraEOSTokens: Set<String>

    /// EOS token IDs loaded from config.json/generation_config.json
    public var eosTokenIds: Set<Int> = []

    /// Tool call format for this model (nil = default JSON format)
    public var toolCallFormat: ToolCallFormat?

    public init(
        id: String, revision: String = "main",
        tokenizerSource: TokenizerSource? = nil,
        defaultPrompt: String = "",
        extraEOSTokens: Set<String> = [],
        toolCallFormat: ToolCallFormat? = nil
    ) {
        self.id = .id(id, revision: revision)
        self.tokenizerSource = tokenizerSource
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.toolCallFormat = toolCallFormat
    }

    public init(
        directory: URL,
        tokenizerSource: TokenizerSource? = nil,
        defaultPrompt: String = "",
        extraEOSTokens: Set<String> = [],
        eosTokenIds: Set<Int> = [],
        toolCallFormat: ToolCallFormat? = nil
    ) {
        self.id = .directory(directory)
        self.tokenizerSource = tokenizerSource
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.eosTokenIds = eosTokenIds
        self.toolCallFormat = toolCallFormat
    }

    /// Maps this configuration's behavioral properties into a
    /// ``ResolvedModelConfiguration`` with the given directories.
    ///
    /// This is a pure data mapping with no I/O. The directories should
    /// already be resolved (downloaded or local) before calling this method.
    public func resolved(
        modelDirectory: URL, tokenizerDirectory: URL
    ) -> ResolvedModelConfiguration {
        ResolvedModelConfiguration(
            modelDirectory: modelDirectory,
            tokenizerDirectory: tokenizerDirectory,
            name: name,
            defaultPrompt: defaultPrompt,
            extraEOSTokens: extraEOSTokens,
            eosTokenIds: eosTokenIds,
            toolCallFormat: toolCallFormat)
    }

}

extension ModelConfiguration: Equatable {

}

extension ModelConfiguration.Identifier: Equatable {

    public static func == (lhs: ModelConfiguration.Identifier, rhs: ModelConfiguration.Identifier)
        -> Bool
    {
        switch (lhs, rhs) {
        case (.id(let lhsID, let lhsRevision), .id(let rhsID, let rhsRevision)):
            lhsID == rhsID && lhsRevision == rhsRevision
        case (.directory(let lhsURL), .directory(let rhsURL)):
            lhsURL == rhsURL
        default:
            false
        }
    }
}
