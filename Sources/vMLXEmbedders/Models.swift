// Copyright © 2024 Apple Inc.

import Foundation
import vMLXLMCommon

/// A registry and configuration provider for embedding models.
///
/// `ModelConfiguration` manages how models are identified (either via Hugging Face Hub IDs or local file URLs)
/// and provides a mechanism to override tokenizer settings. It includes a global registry of
/// well-known models (like BGE, E5, and Snowflake Arctic) to simplify initialization.
///
/// ### Example
/// ```swift
/// // Using a pre-registered model
/// let config = ModelConfiguration.bge_small
///
/// // Using a custom local directory
/// let customConfig = ModelConfiguration(directory: myURL)
/// ```
public struct ModelConfiguration: Sendable {

    /// The backing storage for the model's location.
    public enum Identifier: Sendable {
        /// A Hugging Face Hub repository identifier (e.g., "BAAI/bge-small-en-v1.5").
        case id(String, revision: String = "main")
        /// A file system URL pointing to a local model directory.
        case directory(URL)
    }

    /// The model's identifier (ID or Directory).
    public var id: Identifier

    /// A display-friendly name for the model.
    ///
    /// For Hub models, this returns the repo ID. For local directories,
    /// it returns a path-based name (e.g., "ParentDir/ModelDir").
    public var name: String {
        switch id {
        case .id(let string, _):
            string
        case .directory(let url):
            url.deletingLastPathComponent().lastPathComponent + "/" + url.lastPathComponent
        }
    }

    /// Where to load the tokenizer from when it differs from the model directory.
    ///
    /// - `.id`: download from a remote provider (requires a ``Downloader``)
    /// - `.directory`: load from a local path
    /// - `nil`: use the same directory as the model
    public let tokenizerSource: TokenizerSource?

    /// Initializes a configuration using a Hub repository ID.
    /// - Parameters:
    ///   - id: The Hugging Face repo ID.
    ///   - revision: The Git revision to use (defaults to "main").
    ///   - tokenizerSource: Optional alternate source for the tokenizer.
    public init(
        id: String,
        revision: String = "main",
        tokenizerSource: TokenizerSource? = nil
    ) {
        self.id = .id(id, revision: revision)
        self.tokenizerSource = tokenizerSource
    }

    /// Initializes a configuration using a local directory.
    /// - Parameters:
    ///   - directory: The `URL` of the model on disk.
    ///   - tokenizerSource: Optional alternate source for the tokenizer.
    public init(
        directory: URL,
        tokenizerSource: TokenizerSource? = nil
    ) {
        self.id = .directory(directory)
        self.tokenizerSource = tokenizerSource
    }

    // MARK: - Registry Management

    /// Global registry of available model configurations.
    @MainActor
    public static var registry = [String: ModelConfiguration]()

    /// Registers an array of configurations into the global registry.
    /// - Parameter configurations: The models to register.
    @MainActor
    public static func register(configurations: [ModelConfiguration]) {
        bootstrap()

        for c in configurations {
            registry[c.name] = c
        }
    }

    /// Retrieves a configuration by its ID or name.
    ///
    /// If the ID is not found in the registry, a new `ModelConfiguration` is
    /// created on-the-fly using the provided string as a Hub ID.
    ///
    /// - Parameter id: The model name or Hub ID.
    /// - Returns: A `ModelConfiguration` instance.
    @MainActor
    public static func configuration(id: String) -> ModelConfiguration {
        bootstrap()

        if let c = registry[id] {
            return c
        } else {
            return ModelConfiguration(id: id)
        }
    }

    /// Returns all registered model configurations.
    @MainActor
    public static var models: some Collection<ModelConfiguration> & Sendable {
        bootstrap()
        return Self.registry.values
    }
}

// MARK: - Predefined Models

extension ModelConfiguration {
    /// BGE Micro v2 (TaylorAI) - optimized for extremely low latency.
    public static let bge_micro = ModelConfiguration(id: "TaylorAI/bge-micro-v2")
    /// GTE Tiny - a small, efficient embedding model.
    public static let gte_tiny = ModelConfiguration(id: "TaylorAI/gte-tiny")
    /// MiniLM-L6 - the industry-standard small embedding model.
    public static let minilm_l6 = ModelConfiguration(id: "sentence-transformers/all-MiniLM-L6-v2")
    /// Snowflake Arctic Embed XS.
    public static let snowflake_xs = ModelConfiguration(id: "Snowflake/snowflake-arctic-embed-xs")
    /// MiniLM-L12 - a more accurate version of MiniLM.
    public static let minilm_l12 = ModelConfiguration(id: "sentence-transformers/all-MiniLM-L12-v2")
    /// BGE Small en v1.5.
    public static let bge_small = ModelConfiguration(id: "BAAI/bge-small-en-v1.5")
    /// Multilingual E5 Small - supports over 100 languages.
    public static let multilingual_e5_small = ModelConfiguration(
        id: "intfloat/multilingual-e5-small")
    /// BGE Base en v1.5.
    public static let bge_base = ModelConfiguration(id: "BAAI/bge-base-en-v1.5")
    /// Nomic Embed Text v1.
    public static let nomic_text_v1 = ModelConfiguration(id: "nomic-ai/nomic-embed-text-v1")
    /// Nomic Embed Text v1.5 - supports Matryoshka embeddings.
    public static let nomic_text_v1_5 = ModelConfiguration(id: "nomic-ai/nomic-embed-text-v1.5")
    /// BGE Large en v1.5.
    public static let bge_large = ModelConfiguration(id: "BAAI/bge-large-en-v1.5")
    /// Snowflake Arctic Embed L.
    public static let snowflake_lg = ModelConfiguration(id: "Snowflake/snowflake-arctic-embed-l")
    /// BGE-M3 - Multi-lingual, Multi-functional, Multi-granularity.
    public static let bge_m3 = ModelConfiguration(id: "BAAI/bge-m3")
    /// Mixedbread AI Large v1.
    public static let mixedbread_large = ModelConfiguration(
        id: "mixedbread-ai/mxbai-embed-large-v1")
    /// Qwen3 Embedding 0.6B - 4-bit quantized version.
    public static let qwen3_embedding = ModelConfiguration(
        id: "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")

    private enum BootstrapState: Sendable {
        case idle
        case bootstrapping
        case bootstrapped
    }

    /// Internal state to ensure the registry is only populated once.
    @MainActor
    static private var bootstrapState = BootstrapState.idle

    /// Populates the registry with default models if it hasn't been done already.
    @MainActor
    static func bootstrap() {
        switch bootstrapState {
        case .idle:
            bootstrapState = .bootstrapping
            register(configurations: [
                bge_micro, gte_tiny, minilm_l6, snowflake_xs, minilm_l12,
                bge_small, multilingual_e5_small, bge_base, nomic_text_v1,
                nomic_text_v1_5, bge_large, snowflake_lg, bge_m3,
                mixedbread_large, qwen3_embedding,
            ])
            bootstrapState = .bootstrapped
        case .bootstrapping, .bootstrapped:
            break
        }
    }
}
