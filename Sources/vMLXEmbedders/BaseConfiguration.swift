// Copyright © 2025 Apple Inc.

import Foundation
import MLX

/// The fundamental configuration for any MLX-based model.
///
/// `BaseConfiguration` provides the metadata necessary to identify the model architecture
/// (`modelType`) and describes the quantization parameters used to compress the model's weights.
/// It is designed to be decoded directly from a model repository's `config.json`.
public struct BaseConfiguration: Codable, Sendable {

    /// The architecture identifier (e.g., "bert", "roberta", "xlm-roberta").
    public let modelType: String

    /// Configuration parameters for weight quantization.
    ///
    /// MLX uses group-wise quantization to reduce memory footprint. This struct
    /// defines how weights are grouped and the precision (bits) used for each group.
    public struct Quantization: Codable, Sendable, Equatable {

        /// Initializes a new quantization configuration.
        /// - Parameters:
        ///   - groupSize: The number of weights that share the same scale and bias.
        ///   - bits: The bit-depth of the quantized weights (e.g., 4 or 8).
        public init(groupSize: Int, bits: Int) {
            self.groupSize = groupSize
            self.bits = bits
        }

        /// The size of the quantization group.
        public let groupSize: Int

        /// The number of bits per weight.
        public let bits: Int

        /// Internal storage for the quantization mode.
        private var _mode: QuantizationMode? = nil

        /// The quantization method to use (defaults to `.affine`).
        ///
        /// Affine quantization (asymmetric) uses both a scale and a zero-point
        /// to map floating point values to integers.
        public var mode: QuantizationMode { _mode ?? .affine }

        /// Converts the configuration into a tuple format compatible with `MLX.quantize`.
        public var asTuple: (Int, Int, QuantizationMode) { (groupSize, bits, mode) }

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits = "bits"
            case _mode = "mode"
        }
    }

    /// Instructions for handling individual layers during the quantization process.
    public enum QuantizationOption: Sendable {
        /// Do not quantize this specific layer (keep it in high precision).
        case skip
        /// Quantize this layer using the provided parameters.
        case quantize(Quantization)
    }

    /// A container for per-layer quantization settings.
    ///
    /// This allows for "Mixed-Precision" or "Heterogeneous" quantization, where
    /// sensitive layers (like the embedding head) can be kept at higher precision
    /// while the rest of the model is compressed.
    public struct PerLayerQuantization: Sendable {
        /// The default quantization for any layer not explicitly named in `perLayerQuantization`.
        public var quantization: Quantization? = nil

        /// A dictionary mapping layer paths (e.g., "model.embed_tokens") to their quantization options.
        public var perLayerQuantization: [String: QuantizationOption]

        public init(
            quantization: BaseConfiguration.Quantization? = nil,
            perLayerQuantization: [String: BaseConfiguration.QuantizationOption]
        ) {
            self.quantization = quantization
            self.perLayerQuantization = perLayerQuantization
        }

        /// Resolves the quantization parameters for a specific layer.
        /// - Parameter layer: The path/name of the layer.
        /// - Returns: The `Quantization` settings to apply, or `nil` if the layer should be skipped.
        public func quantization(layer: String) -> Quantization? {
            if let perLayer = perLayerQuantization[layer] {
                switch perLayer {
                case .skip:
                    return nil
                case .quantize(let quantization):
                    return quantization
                }
            } else {
                return quantization
            }
        }
    }

    /// An internal container designed to handle the mixed JSON structure found in `config.json`.
    ///
    /// Quantization configs in MLX often interleave global keys (like `bits`) with
    /// specific layer keys (like `model.layers.0...`). This container uses manual
    /// decoding to separate these interleaved values.
    struct QuantizationContainer: Codable, Sendable {
        var quantization: Quantization
        var perLayerQuantization: PerLayerQuantization

        /// A custom CodingKey used to iterate over arbitrary layer names in JSON.
        internal struct _DictionaryCodingKey: CodingKey {
            internal let stringValue: String
            internal let intValue: Int?

            internal init(stringValue: String) {
                self.stringValue = stringValue
                self.intValue = Int(stringValue)
            }

            internal init(intValue: Int) {
                self.stringValue = "\(intValue)"
                self.intValue = intValue
            }
        }

        init(from decoder: any Decoder) throws {
            // 1. Decode global quantization (bits/group_size) from the current level
            self.quantization = try Quantization(from: decoder)

            // 2. Decode interleaved per-layer values
            var perLayerQuantization = [String: QuantizationOption]()
            let container = try decoder.container(keyedBy: _DictionaryCodingKey.self)

            for key in container.allKeys {
                switch key.stringValue {
                case Quantization.CodingKeys.groupSize.rawValue: continue
                case Quantization.CodingKeys.bits.rawValue: continue
                case Quantization.CodingKeys._mode.rawValue: continue

                default:
                    // If the value is a boolean 'false', we treat it as .skip
                    if let f = try? container.decode(Bool.self, forKey: key) {
                        if !f {
                            perLayerQuantization[key.stringValue] = .skip
                        }
                    } else {
                        // Otherwise, try to decode a specific Quantization object for this layer
                        perLayerQuantization[key.stringValue] = .quantize(
                            try container.decode(Quantization.self, forKey: key))
                    }
                }
            }
            self.perLayerQuantization = PerLayerQuantization(
                quantization: quantization, perLayerQuantization: perLayerQuantization)
        }

        func encode(to encoder: any Encoder) throws {
            try quantization.encode(to: encoder)

            var container = encoder.container(keyedBy: _DictionaryCodingKey.self)
            for (key, value) in perLayerQuantization.perLayerQuantization {
                switch value {
                case .skip:
                    try container.encode(false, forKey: .init(stringValue: key))
                case .quantize(let q):
                    try container.encode(q, forKey: .init(stringValue: key))
                }
            }
        }
    }

    /// Internal storage for quantization details extracted from `config.json`.
    var quantizationContainer: QuantizationContainer?

    /// The default quantization settings.
    @available(*, deprecated, message: "Please use perLayerQuantization instead")
    public var quantization: Quantization? {
        quantizationContainer?.quantization
    }

    /// The per-layer quantization settings, including the default fallback.
    public var perLayerQuantization: PerLayerQuantization? {
        quantizationContainer?.perLayerQuantization
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantizationContainer = "quantization"
    }
}
