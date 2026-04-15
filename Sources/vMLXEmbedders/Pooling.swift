// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

/// Configuration for the `Pooling` layer, typically loaded from a JSON file.
///
/// This struct defines which pooling strategy to apply and the output dimensions,
/// mapping keys from the standard `sentence-transformers` configuration format.
public struct PoolingConfiguration: Codable {

    /// The target output dimension for the embeddings.
    public let dimension: Int

    /// Whether to use the CLS token (special classification token) for pooling.
    public let poolingModeClsToken: Bool

    /// Whether to use the mean of all sequence tokens for pooling.
    public let poolingModeMeanTokens: Bool

    /// Whether to use the maximum value across sequence tokens for pooling.
    public let poolingModeMaxTokens: Bool

    /// Whether to use the last token in the sequence for pooling.
    public let poolingModeLastToken: Bool

    enum CodingKeys: String, CodingKey {
        case dimension = "word_embedding_dimension"
        case poolingModeClsToken = "pooling_mode_cls_token"
        case poolingModeMeanTokens = "pooling_mode_mean_tokens"
        case poolingModeMaxTokens = "pooling_mode_max_tokens"
        case poolingModeLastToken = "pooling_mode_lasttoken"
    }
}

/// Loads a `Pooling` module from a specific model directory.
///
/// It looks for a configuration file located at `1_Pooling/config.json` within the provided directory.
/// If the file is missing or invalid, it falls back to the embedding model's built-in pooling strategy.
/// If neither source provides a strategy, it returns a `Pooling` module with strategy `.none`.
///
/// - Parameter modelDirectory: The base URL of the model weights and configuration.
/// - Parameter model: The loaded embedding model.
/// - Returns: An initialized `Pooling` module.
func loadPooling(modelDirectory: URL, model: EmbeddingModel) -> Pooling {
    let configurationURL = modelDirectory.appending(components: "1_Pooling", "config.json")
    if let poolingConfig = try? JSONDecoder.json5().decode(
        PoolingConfiguration.self, from: Data(contentsOf: configurationURL))
    {
        return Pooling(config: poolingConfig)
    }

    if let strategy = model.poolingStrategy {
        return Pooling(strategy: strategy)
    }

    return Pooling(strategy: .none)
}

/// A module that performs pooling operations on hidden states to produce fixed-sized sentence embeddings.
///
/// `Pooling` takes the sequence of hidden states from a transformer model and collapses them
/// into a single vector using strategies like mean, max, or token selection.
public class Pooling: Module {

    /// Supported pooling strategies.
    public enum Strategy {
        /// Average all token embeddings (weighted by mask).
        case mean
        /// Use the pooled output (CLS) provided by the model.
        case cls
        /// Use the first token in the sequence.
        case first
        /// Use the last token in the sequence.
        case last
        /// Use the maximum value across the sequence length for each dimension.
        case max
        /// Return the existing pooled output or the raw hidden states.
        case none
    }

    /// The active strategy used for pooling hidden states.
    public private(set) var strategy: Strategy

    /// Optional dimension to truncate the resulting embedding to.
    public private(set) var dimension: Int?

    /// Initializes a `Pooling` module with a specific strategy.
    /// - Parameters:
    ///   - strategy: The `Strategy` to use for pooling.
    ///   - dimension: An optional integer to truncate the output vector.
    public init(
        strategy: Strategy, dimension: Int? = nil
    ) {
        self.strategy = strategy
        self.dimension = dimension
    }

    /// Initializes a `Pooling` module from a `PoolingConfiguration`.
    ///
    /// The initialization follows a priority order: CLS > Mean > Max > Last.
    /// If no specific mode is enabled in the config, it defaults to `.first`.
    ///
    /// - Parameter config: The configuration object containing pooling flags.
    public init(
        config: PoolingConfiguration
    ) {
        dimension = config.dimension
        if config.poolingModeClsToken {
            strategy = .cls
        } else if config.poolingModeMeanTokens {
            strategy = .mean
        } else if config.poolingModeMaxTokens {
            strategy = .max
        } else if config.poolingModeLastToken {
            strategy = .last
        } else {
            strategy = .first
        }
    }

    /// Processes the input hidden states according to the configured strategy.
    ///
    /// - Parameters:
    ///   - inputs: The `EmbeddingModelOutput` containing hidden states and/or pooled output.
    ///   - mask: An optional `MLXArray` mask (usually 1 for tokens, 0 for padding).
    ///   - normalize: If `true`, L2 normalizes the resulting vector.
    ///   - applyLayerNorm: If `true`, applies Layer Normalization before truncation/normalization.
    /// - Returns: An `MLXArray` representing the pooled embedding.
    public func callAsFunction(
        _ inputs: EmbeddingModelOutput,
        mask: MLXArray? = nil,
        normalize: Bool = false,
        applyLayerNorm: Bool = false
    ) -> MLXArray {
        let _mask = mask ?? MLXArray.ones(Array(inputs.hiddenStates?.shape[0 ..< 2] ?? [0]))

        var pooled: MLXArray
        switch self.strategy {
        case .mean:
            pooled =
                sum(
                    inputs.hiddenStates! * _mask.expandedDimensions(axes: [-1]),
                    axis: 1)
                / sum(_mask, axis: -1, keepDims: true)
        case .max:
            pooled = MLX.max(
                inputs.hiddenStates! * _mask.expandedDimensions(axes: [-1]), axis: 1)
        case .first:
            pooled = inputs.hiddenStates![0..., 0, 0...]
        case .last:
            let hiddenStates = inputs.hiddenStates!
            let tokenCounts = sum(_mask.asType(.int32), axis: -1)
            let tokenIndices = MLX.maximum(
                tokenCounts - MLXArray(Int32(1)),
                MLXArray(Int32(0))
            )
            let indices = tokenIndices.expandedDimensions(axes: [1, 2])
            let gathered = MLX.takeAlong(hiddenStates, indices, axis: 1)
            pooled = MLX.squeezed(gathered, axis: 1)
        case .cls:
            pooled =
                inputs.pooledOutput
                ?? inputs.hiddenStates![0..., 0, 0...]
        case .none:
            pooled = inputs.pooledOutput ?? inputs.hiddenStates!
        }

        if applyLayerNorm {
            pooled = MLXFast.layerNorm(pooled, eps: 1e-5)
        }

        if let dimension {
            pooled = pooled[0..., 0 ..< dimension]
        }

        if normalize {
            pooled = pooled.l2Normalized()
        }

        return pooled
    }
}
