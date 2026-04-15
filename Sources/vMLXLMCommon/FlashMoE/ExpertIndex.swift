// SPDX-License-Identifier: Apache-2.0
//
// ExpertIndex — safetensors header scanner that locates per-expert
// weight tensors in a model directory without loading any weight data.
//
// Mirrors `vmlx_engine/utils/smelt_loader.py:ExpertIndex.build`. Reads
// only the first 8 bytes (header size) + the JSON header of each
// safetensors file; computes absolute byte offsets for every
// `switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}`
// tensor it finds.
//
// Supported layout families (regex-matched):
//   - Nemotron:  backbone.layers.N.mixer.switch_mlp.(fc1|fc2).{weight|scales|biases}
//   - MiniMax:   *.layers.N.block_sparse_moe.switch_mlp.(gate|up|down)_proj.{...}
//   - Gemma 4:   *.language_model.layers.N.switch_mlp.(gate|up|down)_proj.{...}
//   - Qwen/Mistral text + Mistral VLM: *.layers.N.mlp.switch_mlp.(gate|up|down)_proj.{...}
//
// Nemotron's fc1/fc2 are normalized to up_proj/down_proj so downstream
// code treats them uniformly.

import Foundation

/// Location and metadata for a single tensor inside a safetensors file.
public struct TensorInfo: Sendable, Equatable {
    /// Absolute path to the safetensors file holding the tensor.
    public let filePath: URL
    /// Absolute byte offset to the start of tensor data:
    /// `8 + header_size + data_offsets[0]`.
    public let absOffset: Int
    /// `data_offsets[1] - data_offsets[0]`.
    public let numBytes: Int
    /// Tensor shape (first dim is always `num_experts` for expert tensors).
    public let shape: [Int]
    /// Safetensors dtype string: `U32`, `F16`, `BF16`, `F32`, `I32`, `U8`.
    public let dtype: String

    public init(filePath: URL, absOffset: Int, numBytes: Int, shape: [Int], dtype: String) {
        self.filePath = filePath
        self.absOffset = absOffset
        self.numBytes = numBytes
        self.shape = shape
        self.dtype = dtype
    }
}

/// All tensors (`.weight`, `.scales`, `.biases`) for one projection of an MoE layer.
public struct ProjectionTensors: Sendable, Equatable {
    public var weight: TensorInfo?
    public var scales: TensorInfo?
    public var biases: TensorInfo?

    public init(weight: TensorInfo? = nil, scales: TensorInfo? = nil, biases: TensorInfo? = nil) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
    }

    public var totalBytes: Int {
        (weight?.numBytes ?? 0) + (scales?.numBytes ?? 0) + (biases?.numBytes ?? 0)
    }

    public var allTensors: [TensorInfo] {
        [weight, scales, biases].compactMap { $0 }
    }
}

/// Expert weight locations for a single transformer layer.
///
/// Nemotron uses 2-projection SwitchMLP (`up_proj` + `down_proj` only,
/// no gate). Other families use 3-projection.
public struct LayerExpertInfo: Sendable, Equatable {
    public let layerIdx: Int
    public var gateProj: ProjectionTensors?
    public var upProj: ProjectionTensors?
    public var downProj: ProjectionTensors?

    public init(
        layerIdx: Int,
        gateProj: ProjectionTensors? = nil,
        upProj: ProjectionTensors? = nil,
        downProj: ProjectionTensors? = nil
    ) {
        self.layerIdx = layerIdx
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
    }

    public var totalBytes: Int {
        (gateProj?.totalBytes ?? 0)
            + (upProj?.totalBytes ?? 0)
            + (downProj?.totalBytes ?? 0)
    }

    /// Infer expert count from the first available weight tensor's `shape[0]`.
    public var numExperts: Int? {
        for proj in [gateProj, upProj, downProj] {
            if let w = proj?.weight, let first = w.shape.first {
                return first
            }
        }
        return nil
    }
}

/// Complete expert-weight map for a model directory.
public struct ExpertIndex: Sendable, Equatable {
    /// Per-layer expert tensor locations, keyed by layer index.
    public var layers: [Int: LayerExpertInfo]
    /// Absolute model directory the index was scanned from.
    public var modelPath: URL?
    /// Expert count inferred from the first expert tensor's `shape[0]`.
    public var numExperts: Int
    /// Number of layers that carry expert weights.
    public var numMoeLayers: Int
    /// Total bytes consumed by expert tensors.
    public var expertSizeBytes: Int
    /// Total bytes consumed by non-expert (backbone) tensors in the scan.
    public var backboneBytes: Int

    public init(
        layers: [Int: LayerExpertInfo] = [:],
        modelPath: URL? = nil,
        numExperts: Int = 0,
        numMoeLayers: Int = 0,
        expertSizeBytes: Int = 0,
        backboneBytes: Int = 0
    ) {
        self.layers = layers
        self.modelPath = modelPath
        self.numExperts = numExperts
        self.numMoeLayers = numMoeLayers
        self.expertSizeBytes = expertSizeBytes
        self.backboneBytes = backboneBytes
    }

    // MARK: - Build

    public enum BuildError: Error, CustomStringConvertible {
        case noSafetensorsFound(URL)
        case invalidHeader(URL, String)

        public var description: String {
            switch self {
            case .noSafetensorsFound(let u):
                return "flash-moe: no .safetensors files in \(u.path)"
            case .invalidHeader(let u, let reason):
                return "flash-moe: invalid safetensors header in \(u.path): \(reason)"
            }
        }
    }

    /// Scan a model directory and build an ExpertIndex.
    ///
    /// Only safetensors file headers are read — weight data stays on
    /// disk. This is fast (milliseconds) even for 400GB models.
    public static func build(modelPath: URL) throws -> ExpertIndex {
        let fm = FileManager.default
        let contents = (try? fm.contentsOfDirectory(
            at: modelPath,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
        let safetensorFiles = contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !safetensorFiles.isEmpty else {
            throw BuildError.noSafetensorsFound(modelPath)
        }

        var layers: [Int: LayerExpertInfo] = [:]
        var expertBytes = 0
        var backboneBytes = 0
        var numExperts = 0

        for file in safetensorFiles {
            let (absDataOrigin, header) = try readSafetensorsHeader(at: file)
            // header is [tensor_name: {"dtype","shape","data_offsets"}]
            for (name, entry) in header {
                guard let entryDict = entry as? [String: Any],
                      let dtype = entryDict["dtype"] as? String,
                      let shapeRaw = entryDict["shape"] as? [Any],
                      let offs = entryDict["data_offsets"] as? [Any],
                      offs.count == 2
                else { continue }
                let shape = shapeRaw.compactMap { ($0 as? Int) ?? (($0 as? NSNumber)?.intValue) }
                guard shape.count == shapeRaw.count else { continue }
                let start = (offs[0] as? Int) ?? ((offs[0] as? NSNumber)?.intValue ?? 0)
                let end = (offs[1] as? Int) ?? ((offs[1] as? NSNumber)?.intValue ?? 0)
                let numBytes = end - start
                let absOffset = absDataOrigin + start
                let ti = TensorInfo(
                    filePath: file,
                    absOffset: absOffset,
                    numBytes: numBytes,
                    shape: shape,
                    dtype: dtype
                )

                // Classify: is this an expert tensor?
                if let (layerIdx, proj, suffix) = matchExpertKey(name) {
                    var info = layers[layerIdx] ?? LayerExpertInfo(layerIdx: layerIdx)
                    info.apply(projection: proj, suffix: suffix, tensor: ti)
                    layers[layerIdx] = info
                    expertBytes += numBytes
                    if numExperts == 0, let n = shape.first {
                        numExperts = n
                    }
                } else {
                    backboneBytes += numBytes
                }
            }
        }

        return ExpertIndex(
            layers: layers,
            modelPath: modelPath,
            numExperts: numExperts,
            numMoeLayers: layers.count,
            expertSizeBytes: expertBytes,
            backboneBytes: backboneBytes
        )
    }

    // MARK: - Private: safetensors header reader

    /// Reads the first `8 + headerSize` bytes of a safetensors file and
    /// parses the JSON header. Returns `(absDataOrigin, headerDict)`
    /// where `absDataOrigin = 8 + headerSize` is the absolute byte offset
    /// of the first tensor's data section.
    private static func readSafetensorsHeader(
        at url: URL
    ) throws -> (absDataOrigin: Int, header: [String: Any]) {
        guard let fh = try? FileHandle(forReadingFrom: url) else {
            throw BuildError.invalidHeader(url, "cannot open for reading")
        }
        defer { try? fh.close() }
        let sizeData = fh.readData(ofLength: 8)
        guard sizeData.count == 8 else {
            throw BuildError.invalidHeader(url, "file shorter than 8 bytes")
        }
        let headerSize: UInt64 = sizeData.withUnsafeBytes { ptr in
            ptr.load(as: UInt64.self).littleEndian
        }
        guard headerSize > 0, headerSize < 100_000_000 else {
            throw BuildError.invalidHeader(url, "implausible header size \(headerSize)")
        }
        let headerData = fh.readData(ofLength: Int(headerSize))
        guard headerData.count == Int(headerSize) else {
            throw BuildError.invalidHeader(url, "short read on header")
        }
        guard let obj = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw BuildError.invalidHeader(url, "JSON parse failed")
        }
        return (8 + Int(headerSize), obj)
    }

    // MARK: - Private: expert-key pattern matching

    /// Matches the five supported expert-key patterns. Returns
    /// `(layerIdx, projection, suffix)` on success.
    ///
    /// Projection is one of `"gate_proj"`, `"up_proj"`, `"down_proj"`.
    /// Nemotron's `fc1`/`fc2` are normalized to `up_proj`/`down_proj`.
    /// Suffix is one of `"weight"`, `"scales"`, `"biases"`.
    static func matchExpertKey(_ key: String) -> (layerIdx: Int, projection: String, suffix: String)? {
        // Extract "switch_mlp.<proj>.<suffix>" tail quickly.
        // Accept fc1/fc2 as well (Nemotron).
        let validSuffixes: Set<String> = ["weight", "scales", "biases"]
        let validProjections: Set<String> = ["gate_proj", "up_proj", "down_proj", "fc1", "fc2"]

        let comps = key.split(separator: ".").map(String.init)
        guard comps.count >= 4 else { return nil }
        let suffix = comps[comps.count - 1]
        let rawProj = comps[comps.count - 2]
        guard validSuffixes.contains(suffix),
              validProjections.contains(rawProj)
        else { return nil }
        guard comps[comps.count - 3] == "switch_mlp" else { return nil }

        // Find the "layers.N" pair earlier in the path.
        var layerIdx: Int?
        var i = 0
        while i < comps.count - 1 {
            if comps[i] == "layers", let n = Int(comps[i + 1]) {
                layerIdx = n
                break
            }
            i += 1
        }
        guard let idx = layerIdx else { return nil }

        // Normalize Nemotron fc1/fc2 → up_proj/down_proj.
        let proj: String
        switch rawProj {
        case "fc1": proj = "up_proj"
        case "fc2": proj = "down_proj"
        default:    proj = rawProj
        }
        return (idx, proj, suffix)
    }
}

extension LayerExpertInfo {
    /// Assign a located tensor into the correct projection slot.
    mutating func apply(projection: String, suffix: String, tensor: TensorInfo) {
        switch projection {
        case "gate_proj": gateProj = gateProj ?? ProjectionTensors(); gateProj = assigned(gateProj!, suffix: suffix, tensor: tensor)
        case "up_proj":   upProj   = upProj   ?? ProjectionTensors(); upProj   = assigned(upProj!,   suffix: suffix, tensor: tensor)
        case "down_proj": downProj = downProj ?? ProjectionTensors(); downProj = assigned(downProj!, suffix: suffix, tensor: tensor)
        default: break
        }
    }

    private func assigned(
        _ p: ProjectionTensors, suffix: String, tensor: TensorInfo
    ) -> ProjectionTensors {
        var out = p
        switch suffix {
        case "weight": out.weight = tensor
        case "scales": out.scales = tensor
        case "biases": out.biases = tensor
        default: break
        }
        return out
    }
}
