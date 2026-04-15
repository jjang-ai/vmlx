// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

/// Errors encountered during the model loading and initialization process.
///
/// This enum provides detailed feedback for failures in model type identification,
/// file access, JSON decoding, and missing configuration files.
public enum EmbedderError: LocalizedError {

    /// The specified `model_type` in `config.json` is not supported by the current implementation.
    case unsupportedModelType(String)

    /// A required file could not be read from the disk.
    /// - Parameters:
    ///     - fileName: The name of the file (e.g., "config.json").
    ///     - modelName: The name/ID of the model being loaded.
    ///     - error: The underlying system error.
    case configurationFileError(String, String, Error)

    /// The configuration file exists but contains invalid JSON or missing required fields.
    case configurationDecodingError(String, String, DecodingError)

    /// A human-readable description of the error.
    public var errorDescription: String? {
        switch self {
        case .unsupportedModelType(let type):
            return "Unsupported model type: \(type)"
        case .configurationFileError(let file, let modelName, let error):
            return "Error reading '\(file)' for model '\(modelName)': \(error.localizedDescription)"
        case .configurationDecodingError(let file, let modelName, let decodingError):
            let errorDetail = extractDecodingErrorDetail(decodingError)
            return "Failed to parse \(file) for model '\(modelName)': \(errorDetail)"
        }
    }

    /// Internal helper to provide specific details about JSON decoding failures,
    /// such as the exact key path that failed.
    private func extractDecodingErrorDetail(_ error: DecodingError) -> String {
        switch error {
        case .keyNotFound(let key, let context):
            let path = (context.codingPath + [key]).map { $0.stringValue }.joined(separator: ".")
            return "Missing field '\(path)'"
        case .typeMismatch(_, let context):
            let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
            return "Type mismatch at '\(path)'"
        case .valueNotFound(_, let context):
            let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
            return "Missing value at '\(path)'"
        case .dataCorrupted(let context):
            if context.codingPath.isEmpty {
                return "Invalid JSON"
            } else {
                let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
                return "Invalid data at '\(path)'"
            }
        @unknown default:
            return error.localizedDescription
        }
    }
}

/// Resolve model and tokenizer directories from a ``ModelConfiguration``
/// using a ``Downloader``.
///
/// - Parameters:
///   - downloader: The downloader to use for fetching remote resources.
///   - configuration: The configuration identifying the model.
///   - useLatest: When true, always checks the provider for updates.
///   - progressHandler: A closure to monitor download progress.
/// - Returns: A tuple of (modelDirectory, tokenizerDirectory).
func resolveDirectories(
    from downloader: any Downloader,
    configuration: ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> (modelDirectory: URL, tokenizerDirectory: URL) {
    let modelDirectory: URL
    switch configuration.id {
    case .id(let id, let revision):
        modelDirectory = try await downloader.download(
            id: id, revision: revision,
            matching: modelDownloadPatterns,
            useLatest: useLatest,
            progressHandler: progressHandler)
    case .directory(let directory):
        modelDirectory = directory
    }

    let tokenizerDirectory: URL
    switch configuration.tokenizerSource {
    case .id(let id, let revision):
        tokenizerDirectory = try await downloader.download(
            id: id, revision: revision,
            matching: tokenizerDownloadPatterns,
            useLatest: useLatest,
            progressHandler: { _ in })
    case .directory(let directory):
        tokenizerDirectory = directory
    case nil:
        tokenizerDirectory = modelDirectory
    }

    return (modelDirectory, tokenizerDirectory)
}

/// Asynchronously loads the `EmbeddingModel` and its associated `Tokenizer`.
///
/// This is the primary high-level function for initializing an embedding pipeline.
/// It leverages `async let` to load the tokenizer in parallel while the model
/// structure is being built synchronously.
///
/// - Parameters:
///   - downloader: The ``Downloader`` to use for fetching remote resources.
///   - configuration: The model configuration.
///   - useLatest: When true, always checks the provider for updates.
///   - progressHandler: A closure for tracking download progress.
/// - Returns: A tuple containing the initialized `EmbeddingModel` and `Tokenizer`.
public func load(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> (EmbeddingModel, Tokenizer) {
    let (modelDirectory, tokenizerDirectory) = try await resolveDirectories(
        from: downloader, configuration: configuration, useLatest: useLatest,
        progressHandler: progressHandler)

    async let tokenizerTask = tokenizerLoader.load(from: tokenizerDirectory)
    let model = try loadSynchronous(modelDirectory: modelDirectory, modelName: configuration.name)
    let tokenizer = try await tokenizerTask

    return (model, tokenizer)
}

/// Synchronously initializes the model architecture, loads weights, and applies quantization.
///
/// This function performs the following steps:
/// 1. Reads and decodes `config.json`.
/// 2. Instantiates the specific model class based on `model_type`.
/// 3. Recursively scans the directory for `.safetensors` weight files.
/// 4. Applies quantization if defined in the configuration.
/// 5. Updates the model parameters and performs an initial evaluation (`eval`).
///
/// - Parameters:
///   - modelDirectory: The local `URL` containing model files.
///   - modelName: The display name of the model for error reporting.
/// - Returns: A fully initialized and weighted `EmbeddingModel`.
func loadSynchronous(modelDirectory: URL, modelName: String) throws -> EmbeddingModel {
    let configurationURL = modelDirectory.appending(component: "config.json")
    let configData: Data
    do {
        configData = try Data(contentsOf: configurationURL)
    } catch {
        throw EmbedderError.configurationFileError(
            configurationURL.lastPathComponent, modelName, error)
    }

    let baseConfig: BaseConfiguration
    do {
        baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
    } catch let error as DecodingError {
        throw EmbedderError.configurationDecodingError(
            configurationURL.lastPathComponent, modelName, error)
    }

    let modelType = ModelType(rawValue: baseConfig.modelType)
    let model: EmbeddingModel
    do {
        model = try modelType.createModel(configuration: configData)
    } catch let error as DecodingError {
        throw EmbedderError.configurationDecodingError(
            configurationURL.lastPathComponent, modelName, error)
    }

    var weights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let w = try loadArrays(url: url)
            for (key, value) in w {
                weights[key] = value
            }
        }
    }

    weights = model.sanitize(weights: weights)

    if let perLayerQuantization = baseConfig.perLayerQuantization {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                return perLayerQuantization.quantization(layer: path)?.asTuple
            } else {
                return nil
            }
        }
    }

    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)

    return model
}

/// Asynchronously loads a `ModelContainer` for thread-safe model access.
///
/// The `ModelContainer` is recommended for applications where multiple threads
/// or tasks may need to access the embedding model simultaneously.
///
/// - Parameters:
///   - downloader: The ``Downloader`` to use for fetching remote resources.
///   - configuration: The model configuration.
///   - useLatest: When true, always checks the provider for updates.
///   - progressHandler: A closure for tracking download progress.
/// - Returns: A thread-safe `ModelContainer` instance.
public func loadModelContainer(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> ModelContainer {
    let (modelDirectory, tokenizerDirectory) = try await resolveDirectories(
        from: downloader, configuration: configuration, useLatest: useLatest,
        progressHandler: progressHandler)

    return try await ModelContainer(
        modelDirectory: modelDirectory,
        tokenizerDirectory: tokenizerDirectory,
        configuration: configuration,
        tokenizerLoader: tokenizerLoader)
}

/// Load an embedding model from a local directory.
///
/// No downloader is needed — the model and tokenizer are loaded from
/// the given directory.
///
/// - Parameter directory: The local directory containing model files.
/// - Returns: A tuple containing the initialized `EmbeddingModel` and `Tokenizer`.
public func load(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader
) async throws -> (EmbeddingModel, Tokenizer) {
    let name =
        directory.deletingLastPathComponent().lastPathComponent + "/"
        + directory.lastPathComponent
    async let tokenizerTask = tokenizerLoader.load(from: directory)
    let model = try loadSynchronous(modelDirectory: directory, modelName: name)
    let tokenizer = try await tokenizerTask
    return (model, tokenizer)
}

/// Load an embedding model container from a local directory.
///
/// No downloader is needed — the model and tokenizer are loaded from
/// the given directory.
///
/// - Parameters:
///   - directory: The local directory containing model files.
///   - tokenizerLoader: The ``TokenizerLoader`` to use for loading the tokenizer.
/// - Returns: A thread-safe `ModelContainer` instance.
public func loadModelContainer(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader
) async throws -> ModelContainer {
    try await ModelContainer(
        modelDirectory: directory,
        tokenizerDirectory: directory,
        configuration: ModelConfiguration(directory: directory),
        tokenizerLoader: tokenizerLoader)
}
