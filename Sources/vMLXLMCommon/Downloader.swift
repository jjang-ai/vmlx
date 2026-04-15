import Foundation

/// A protocol for downloading model repository snapshots to local directories.
///
/// Conforming types encapsulate the full download lifecycle — cache check, network
/// download, and fallback to cache on failure. Each conformance owns its own caching
/// strategy. The return value is always a local directory URL containing the requested files.
///
/// The protocol is provider-agnostic. `id` is a plain `String` that each conformance
/// interprets however it wants (e.g. `"org/model"` for Hugging Face, a four-part
/// Kaggle handle, an S3 path). `revision` is optional for providers without versioning.
///
/// ## See Also
/// - ``ResolvedModelConfiguration``
/// - ``TokenizerSource``
public protocol Downloader: Sendable {
    /// Download (or retrieve from cache) a snapshot of a repository.
    ///
    /// - Parameters:
    ///   - id: Provider-specific repository identifier
    ///   - revision: Optional revision (branch, tag, commit hash, version number).
    ///     Providers without versioning receive `nil`.
    ///   - patterns: Glob patterns to filter which files to download
    ///     (e.g. `["*.safetensors", "*.json", "*.jinja"]`)
    ///   - useLatest: When `true`, check the provider for updates even if a cached
    ///     version exists. When `false`, return the cached version if available.
    ///   - progressHandler: Callback for download progress
    /// - Returns: Local directory URL containing the downloaded files
    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL
}

/// Identifies where a tokenizer should be loaded from.
///
/// Used by ``ModelConfiguration`` to specify an alternate tokenizer source
/// when the tokenizer files are not co-located with the model weights.
///
/// - ``id(_:)`` downloads tokenizer files from a remote provider (requires a ``Downloader``)
/// - ``directory(_:)`` loads tokenizer files from a local path
///
/// When `nil` on a ``ModelConfiguration``, the tokenizer is loaded from the
/// same directory as the model.
public enum TokenizerSource: Sendable, Equatable {
    /// A provider-specific repository identifier for downloading tokenizer files.
    /// - Parameters:
    ///   - id: The repository identifier (e.g. `"org/tokenizer-name"`).
    ///   - revision: Optional revision (branch, tag, commit hash). When `nil`,
    ///     the ``Downloader`` decides the default (typically `"main"`).
    case id(String, revision: String? = nil)
    /// A local directory containing tokenizer files.
    case directory(URL)
}

/// A fully resolved model configuration where all sources have been resolved
/// to local directory paths.
///
/// Created by resolving a ``ModelConfiguration`` — downloading remote sources
/// via a ``Downloader`` and mapping behavioral properties. Factory implementations
/// receive this type in their `_load` method, so they work purely with local files.
///
/// ## See Also
/// - ``ModelConfiguration/resolved(modelDirectory:tokenizerDirectory:)``
/// - ``Downloader``
public struct ResolvedModelConfiguration: Sendable {
    public var modelDirectory: URL
    public var tokenizerDirectory: URL
    public var name: String
    public var defaultPrompt: String
    public var extraEOSTokens: Set<String>
    public var eosTokenIds: Set<Int>
    public var toolCallFormat: ToolCallFormat?

    public init(
        modelDirectory: URL,
        tokenizerDirectory: URL,
        name: String,
        defaultPrompt: String,
        extraEOSTokens: Set<String>,
        eosTokenIds: Set<Int>,
        toolCallFormat: ToolCallFormat?
    ) {
        self.modelDirectory = modelDirectory
        self.tokenizerDirectory = tokenizerDirectory
        self.name = name
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.eosTokenIds = eosTokenIds
        self.toolCallFormat = toolCallFormat
    }
}

extension ResolvedModelConfiguration {
    /// Convenience for loading everything from a single local directory.
    public init(directory: URL) {
        self.init(
            modelDirectory: directory,
            tokenizerDirectory: directory,
            name: directory.deletingLastPathComponent().lastPathComponent + "/"
                + directory.lastPathComponent,
            defaultPrompt: "",
            extraEOSTokens: [],
            eosTokenIds: [],
            toolCallFormat: nil)
    }
}
