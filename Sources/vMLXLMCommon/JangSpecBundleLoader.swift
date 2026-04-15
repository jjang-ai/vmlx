// Copyright © 2025 JANG. All rights reserved.
//
// JangSpecBundleLoader — read a `.jangspec` bundle into the
// `[String: MLXArray]` dict that the existing `loadWeights()` pipeline
// expects.
//
// A `.jangspec` bundle is a self-contained directory holding:
//
//   <name>.jangspec/
//     jangspec.json                 manifest (bundle_version, tensor lists, sizes)
//     config.json                   model config (copied from source for factory detection)
//     jang_config.json              JANG quant metadata (copied from source)
//     tokenizer.json                tokenizer (copied from source)
//     tokenizer_config.json
//     target/
//       config.json                 same as root, also kept for streaming runtime
//       jang_config.json
//       hot_core.safetensors        attention/router/norm/embed/lm_head, mmap-ready
//       experts.jsidx               flat binary index, one entry per (layer, expert)
//       experts-00000.bin           per-expert blobs, 4 KB-aligned
//       experts-00001.bin
//       ...
//
// This loader is the inverse of the Python `jang_tools.jangspec.builder`
// and produces the same `{tensor_name: MLXArray}` dict that the existing
// `loadWeights(modelDirectory:)` safetensors enumeration would produce on
// the source JANG model directory. Downstream sanitize, JANG MoE gate
// dequant, per-layer quant inference, and `model.update` all run unchanged.

import Foundation
import MLX

// MARK: - Format constants
//
// These mirror `jang_tools/jangspec/format.py` exactly. If the on-disk
// format ever changes, update both sites in lockstep.

public enum JangSpecBundleFormat {
    public static let bundleVersion: Int = 1

    public static let manifestFilename = "jangspec.json"
    public static let indexFilename = "target/experts.jsidx"
    public static let hotCoreFilename = "target/hot_core.safetensors"

    public static func expertFilename(idx: Int) -> String {
        return String(format: "target/experts-%05d.bin", idx)
    }

    public static let blobAlignment: Int = 4096
    public static let blobMagic: UInt32 = 0x4550_534A    // "JSPE"
    public static let indexMagic: UInt32 = 0x58_494A_53  // "SJIX"

    public static let blobHeaderSize: Int = 32
    public static let tensorHeaderSize: Int = 36
    public static let indexEntrySize: Int = 28
    public static let indexHeaderSize: Int = 24

    public enum TensorKind: UInt8 {
        case gate = 0
        case up = 1
        case down = 2
    }

    public enum TensorDType: UInt32 {
        case qweight = 0   // uint32 packed
        case scales = 1    // float16
        case biases = 2    // float16
    }
}

// MARK: - Errors

public enum JangSpecBundleError: Error, CustomStringConvertible {
    case fileMissing(URL)
    case unsupportedVersion(field: String, value: Int, supported: Int)
    case truncated(URL, expected: Int, actual: Int)
    case missingEntry(layer: Int, expert: Int)
    case invalidManifest(String)
    case invalidBlob(String)
    case invalidIndex(String)
    case missingBaseName(String)

    public var description: String {
        switch self {
        case .fileMissing(let url):
            return "jangspec: file missing: \(url.path)"
        case .unsupportedVersion(let field, let value, let supported):
            return "jangspec: unsupported \(field) version \(value), supported \(supported)"
        case .truncated(let url, let e, let a):
            return "jangspec: truncated \(url.lastPathComponent): expected \(e), got \(a)"
        case .missingEntry(let layer, let expert):
            return "jangspec: no entry for (layer=\(layer), expert=\(expert))"
        case .invalidManifest(let m):
            return "jangspec: invalid manifest: \(m)"
        case .invalidBlob(let m):
            return "jangspec: invalid blob: \(m)"
        case .invalidIndex(let m):
            return "jangspec: invalid index: \(m)"
        case .missingBaseName(let n):
            return "jangspec: cannot parse layer index from base name: \(n)"
        }
    }
}

// MARK: - Manifest

/// Mirror of `jang_tools.jangspec.manifest.Manifest`. Decoded from
/// `jangspec.json` via `JSONDecoder`.
public struct JangSpecBundleManifest: Codable, Sendable {
    public var bundleVersion: Int
    public var sourceJang: String
    public var targetArch: String
    public var nLayers: Int
    public var nExpertsPerLayer: Int
    public var targetTopK: Int
    public var hotCoreTensors: [String]
    public var expertTensorNames: [String]
    public var nExpertsTotal: Int

    enum CodingKeys: String, CodingKey {
        case bundleVersion = "bundle_version"
        case sourceJang = "source_jang"
        case targetArch = "target_arch"
        case nLayers = "n_layers"
        case nExpertsPerLayer = "n_experts_per_layer"
        case targetTopK = "target_top_k"
        case hotCoreTensors = "hot_core_tensors"
        case expertTensorNames = "expert_tensor_names"
        case nExpertsTotal = "n_experts_total"
    }
}

// MARK: - Index entry

private struct ExpertIndexEntry {
    let layerIdx: Int
    let expertID: Int
    let fileID: Int
    let offset: Int
    let nbytes: Int
}

private struct ExpertKey: Hashable {
    let layer: Int
    let expert: Int
}

// MARK: - Bundle loader

public enum JangSpecBundleLoader {

    /// Returns true iff `directory` looks like a `.jangspec` bundle.
    public static func isBundle(at directory: URL) -> Bool {
        let manifest = directory.appendingPathComponent(JangSpecBundleFormat.manifestFilename)
        return FileManager.default.fileExists(atPath: manifest.path)
    }

    /// Load every tensor a model needs from a `.jangspec` bundle.
    ///
    /// The returned dict has the same keys and dtypes as a vanilla
    /// safetensors enumeration on the source JANG directory.
    ///
    /// Two expert-loading paths:
    ///
    /// **Fat layout (preferred, default for bundles built after 2026-04-14):**
    /// If `<bundle>/experts.safetensors` exists, mmap it directly via
    /// `loadArraysAndMetadata`. The file holds every expert tensor in its
    /// original 3D `[E, ...]` stacked form under the full source tensor
    /// names. Load perf is identical to the source-directory path — zero
    /// restack cost, zero transient RAM peak.
    ///
    /// **Per-blob streaming layout (legacy, still supported):**
    /// If `experts.safetensors` is absent, walk `target/experts.jsidx` and
    /// restack the per-expert blobs via `MLX.stacked(_:axis: 0)`. Used by
    /// future SSD streaming work; currently ~3× the RAM peak of the fat
    /// layout because of the transient per-expert MLXArrays.
    public static func loadWeights(from directory: URL) throws -> [String: MLXArray] {
        let manifestURL = directory.appendingPathComponent(
            JangSpecBundleFormat.manifestFilename)
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw JangSpecBundleError.fileMissing(manifestURL)
        }

        // 1. Manifest.
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(
            JangSpecBundleManifest.self, from: manifestData)
        guard manifest.bundleVersion == JangSpecBundleFormat.bundleVersion else {
            throw JangSpecBundleError.unsupportedVersion(
                field: "bundle",
                value: manifest.bundleVersion,
                supported: JangSpecBundleFormat.bundleVersion
            )
        }

        var out: [String: MLXArray] = [:]

        // 2. Hot core — read every tensor via the existing safetensors loader.
        let hotCoreURL = directory.appendingPathComponent(
            JangSpecBundleFormat.hotCoreFilename)
        guard FileManager.default.fileExists(atPath: hotCoreURL.path) else {
            throw JangSpecBundleError.fileMissing(hotCoreURL)
        }
        let (hotArrays, _) = try loadArraysAndMetadata(url: hotCoreURL)
        for (key, value) in hotArrays {
            out[key] = value
        }

        // 3. Fat expert layout — preferred path.
        let fatURL = directory.appendingPathComponent("experts.safetensors")
        if FileManager.default.fileExists(atPath: fatURL.path) {
            let (fatArrays, _) = try loadArraysAndMetadata(url: fatURL)
            for (key, value) in fatArrays {
                out[key] = value
            }
            return out
        }

        // 4. Legacy per-blob path (only reached if experts.safetensors is absent).
        let indexURL = directory.appendingPathComponent(
            JangSpecBundleFormat.indexFilename)
        let entriesByKey = try parseIndex(at: indexURL)

        var shardCache: [Int: Data] = [:]
        func shard(forID id: Int) throws -> Data {
            if let hit = shardCache[id] { return hit }
            let url = directory.appendingPathComponent(
                JangSpecBundleFormat.expertFilename(idx: id))
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw JangSpecBundleError.fileMissing(url)
            }
            let data = try Data(contentsOf: url, options: .mappedIfSafe)
            shardCache[id] = data
            return data
        }

        let layerGroups = try groupExpertBasesByLayer(manifest.expertTensorNames)

        // For each (layer, base), walk experts in order and stack.
        //
        // Wrap each layer's body in `autoreleasepool` so the ~256 per-expert
        // intermediate MLXArrays get released as soon as the layer's
        // `stacked()` calls finish. Without this boundary, all 62 layers'
        // worth of per-expert temporaries accumulate in the outer pool and
        // peak RSS hits ~3× the model size (89 GB for a 30 GB model).
        for (layerIdx, baseNames) in layerGroups {
            try autoreleasepool {
                try restackLayer(
                    layerIdx: layerIdx,
                    baseNames: baseNames,
                    manifest: manifest,
                    entriesByKey: entriesByKey,
                    shardLookup: shard,
                    into: &out
                )
            }
        }

        return out
    }

    /// Restack one layer's per-expert blobs into 3D `[E, ...]` tensors and
    /// emit them under the `{base}.{weight,scales,biases}` keys.
    ///
    /// Lives in its own function so callers can wrap each invocation in
    /// `autoreleasepool { ... }` — the per-expert MLXArrays the function
    /// allocates internally are then released as soon as it returns,
    /// keeping peak RSS bounded to one layer's worth of temporaries
    /// instead of the whole model's.
    private static func restackLayer(
        layerIdx: Int,
        baseNames: [String],
        manifest: JangSpecBundleManifest,
        entriesByKey: [ExpertKey: ExpertIndexEntry],
        shardLookup: (Int) throws -> Data,
        into out: inout [String: MLXArray]
    ) throws {
        // Each layer has gate_proj / up_proj / down_proj. Map kind -> base name.
        var kindToBase: [JangSpecBundleFormat.TensorKind: String] = [:]
        for base in baseNames {
            if base.hasSuffix(".switch_mlp.gate_proj") {
                kindToBase[.gate] = base
            } else if base.hasSuffix(".switch_mlp.up_proj") {
                kindToBase[.up] = base
            } else if base.hasSuffix(".switch_mlp.down_proj") {
                kindToBase[.down] = base
            }
        }

        // Per-base accumulators. We collect one MLXArray per expert
        // and `stacked()` them at the end into a 3D `[E, ...]` tensor.
        struct Slots {
            var qweight: [MLXArray] = []
            var scales: [MLXArray] = []
            var biases: [MLXArray] = []
        }
        var slots: [JangSpecBundleFormat.TensorKind: Slots] = [
            .gate: Slots(),
            .up: Slots(),
            .down: Slots(),
        ]

        for expertID in 0..<manifest.nExpertsPerLayer {
            try autoreleasepool {
                let key = ExpertKey(layer: layerIdx, expert: expertID)
                guard let entry = entriesByKey[key] else {
                    throw JangSpecBundleError.missingEntry(
                        layer: layerIdx, expert: expertID)
                }
                let shardData = try shardLookup(entry.fileID)
                let blobBytes = shardData.subdata(
                    in: entry.offset..<(entry.offset + entry.nbytes))
                let parsed = try parseExpertBlob(blobBytes)

                for kind in [
                    JangSpecBundleFormat.TensorKind.gate,
                    .up,
                    .down,
                ] {
                    guard kindToBase[kind] != nil else { return }
                    guard let triple = parsed.tensors[kind] else { return }
                    slots[kind]!.qweight.append(triple.qweight)
                    slots[kind]!.scales.append(triple.scales)
                    slots[kind]!.biases.append(triple.biases)
                }
            }
        }

        // Stack and emit. Each `stacked()` call materializes a new MLXArray;
        // dropping the input slot arrays after each base lets the per-expert
        // temporaries get released as soon as the stack consumes them.
        for kind in [
            JangSpecBundleFormat.TensorKind.gate,
            .up,
            .down,
        ] {
            guard let base = kindToBase[kind] else { continue }
            let s = slots[kind]!
            guard !s.qweight.isEmpty else { continue }
            out["\(base).weight"] = stacked(s.qweight, axis: 0)
            out["\(base).scales"] = stacked(s.scales, axis: 0)
            out["\(base).biases"] = stacked(s.biases, axis: 0)
            slots[kind] = Slots()  // drop refs immediately
        }
    }

    // MARK: - Index parsing

    private static func parseIndex(at url: URL) throws -> [ExpertKey: ExpertIndexEntry] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw JangSpecBundleError.fileMissing(url)
        }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= JangSpecBundleFormat.indexHeaderSize else {
            throw JangSpecBundleError.truncated(
                url, expected: JangSpecBundleFormat.indexHeaderSize, actual: data.count)
        }

        let (magic, version, _, _, nEntries): (UInt32, UInt16, UInt32, UInt32, UInt64) =
            data.withUnsafeBytes { raw in
                let m = raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self)
                let v = raw.loadUnaligned(fromByteOffset: 4, as: UInt16.self)
                let nL = raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self)
                let nE = raw.loadUnaligned(fromByteOffset: 12, as: UInt32.self)
                let n = raw.loadUnaligned(fromByteOffset: 16, as: UInt64.self)
                return (m, v, nL, nE, n)
            }
        guard magic == JangSpecBundleFormat.indexMagic else {
            throw JangSpecBundleError.invalidIndex(
                String(format: "bad magic 0x%08x", magic))
        }
        guard version == 1 else {
            throw JangSpecBundleError.unsupportedVersion(
                field: "index", value: Int(version), supported: 1)
        }

        let count = Int(nEntries)
        let expectedSize =
            JangSpecBundleFormat.indexHeaderSize + count * JangSpecBundleFormat.indexEntrySize
        guard data.count >= expectedSize else {
            throw JangSpecBundleError.truncated(url, expected: expectedSize, actual: data.count)
        }

        var entries: [ExpertKey: ExpertIndexEntry] = [:]
        entries.reserveCapacity(count)
        data.withUnsafeBytes { raw in
            var cursor = JangSpecBundleFormat.indexHeaderSize
            for _ in 0..<count {
                let layer = raw.loadUnaligned(fromByteOffset: cursor + 0, as: UInt32.self)
                let expert = raw.loadUnaligned(fromByteOffset: cursor + 4, as: UInt32.self)
                let fileID = raw.loadUnaligned(fromByteOffset: cursor + 8, as: UInt16.self)
                // 2 bytes pad at cursor + 10
                let offset = raw.loadUnaligned(fromByteOffset: cursor + 12, as: UInt64.self)
                let nbytes = raw.loadUnaligned(fromByteOffset: cursor + 20, as: UInt64.self)
                entries[ExpertKey(layer: Int(layer), expert: Int(expert))] =
                    ExpertIndexEntry(
                        layerIdx: Int(layer),
                        expertID: Int(expert),
                        fileID: Int(fileID),
                        offset: Int(offset),
                        nbytes: Int(nbytes)
                    )
                cursor += JangSpecBundleFormat.indexEntrySize
            }
        }
        return entries
    }

    // MARK: - Blob parsing

    private struct ParsedBlob {
        struct Triple {
            let qweight: MLXArray
            let scales: MLXArray
            let biases: MLXArray
        }
        let layerIdx: Int
        let expertID: Int
        let bits: Int
        let tensors: [JangSpecBundleFormat.TensorKind: Triple]
    }

    private static func parseExpertBlob(_ data: Data) throws -> ParsedBlob {
        guard data.count >= JangSpecBundleFormat.blobHeaderSize else {
            throw JangSpecBundleError.invalidBlob("blob too short")
        }
        let (magic, version, nTensors, layer, expert, payloadOffset, payloadBytes):
            (UInt32, UInt16, UInt16, UInt32, UInt32, UInt64, UInt64) =
            data.withUnsafeBytes { raw in
                let m = raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self)
                let v = raw.loadUnaligned(fromByteOffset: 4, as: UInt16.self)
                let n = raw.loadUnaligned(fromByteOffset: 6, as: UInt16.self)
                let l = raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self)
                let e = raw.loadUnaligned(fromByteOffset: 12, as: UInt32.self)
                let po = raw.loadUnaligned(fromByteOffset: 16, as: UInt64.self)
                let pb = raw.loadUnaligned(fromByteOffset: 24, as: UInt64.self)
                return (m, v, n, l, e, po, pb)
            }
        guard magic == JangSpecBundleFormat.blobMagic else {
            throw JangSpecBundleError.invalidBlob(
                String(format: "bad magic 0x%08x", magic))
        }
        guard version == 1 else {
            throw JangSpecBundleError.unsupportedVersion(
                field: "blob", value: Int(version), supported: 1)
        }
        guard nTensors == 9 else {
            throw JangSpecBundleError.invalidBlob("expected 9 tensor entries, got \(nTensors)")
        }

        let payOff = Int(payloadOffset)
        let payBytes = Int(payloadBytes)
        guard data.count >= payOff + payBytes else {
            throw JangSpecBundleError.invalidBlob("declared payload exceeds data length")
        }

        var bitsSeen: Int? = nil
        var collected: [JangSpecBundleFormat.TensorKind: [JangSpecBundleFormat.TensorDType: MLXArray]] = [:]

        for i in 0..<Int(nTensors) {
            let cursor = JangSpecBundleFormat.blobHeaderSize + i * JangSpecBundleFormat.tensorHeaderSize
            let (kindRaw, bitsVal, dtypeRaw, d0, d1, d2, off, nb):
                (UInt8, UInt8, UInt32, UInt32, UInt32, UInt32, UInt64, UInt64) =
                data.withUnsafeBytes { raw in
                    let k = raw.loadUnaligned(fromByteOffset: cursor + 0, as: UInt8.self)
                    let b = raw.loadUnaligned(fromByteOffset: cursor + 1, as: UInt8.self)
                    let dt = raw.loadUnaligned(fromByteOffset: cursor + 4, as: UInt32.self)
                    let x = raw.loadUnaligned(fromByteOffset: cursor + 8, as: UInt32.self)
                    let y = raw.loadUnaligned(fromByteOffset: cursor + 12, as: UInt32.self)
                    let z = raw.loadUnaligned(fromByteOffset: cursor + 16, as: UInt32.self)
                    let o = raw.loadUnaligned(fromByteOffset: cursor + 20, as: UInt64.self)
                    let n = raw.loadUnaligned(fromByteOffset: cursor + 28, as: UInt64.self)
                    return (k, b, dt, x, y, z, o, n)
                }
            guard let kind = JangSpecBundleFormat.TensorKind(rawValue: kindRaw) else {
                throw JangSpecBundleError.invalidBlob("unknown tensor kind \(kindRaw)")
            }
            guard let dtype = JangSpecBundleFormat.TensorDType(rawValue: dtypeRaw) else {
                throw JangSpecBundleError.invalidBlob("unknown tensor dtype \(dtypeRaw)")
            }

            let bi = Int(bitsVal)
            if let prev = bitsSeen {
                if prev != bi {
                    throw JangSpecBundleError.invalidBlob(
                        "mixed bits in one blob: \(prev) vs \(bi)")
                }
            } else {
                bitsSeen = bi
            }

            let start = payOff + Int(off)
            let end = start + Int(nb)
            guard end <= data.count else {
                throw JangSpecBundleError.invalidBlob("tensor slice out of range")
            }

            // Materialize as MLXArray from the raw byte slice using the
            // explicit-dtype `MLXArray(_:_:dtype:)` initializer. Zero-copy
            // over the mmap-backed Data slice.
            let dims = [Int(d0), Int(d1), Int(d2)].filter { $0 != 0 }
            let payloadSlice = data.subdata(in: start..<end)
            let arr: MLXArray
            switch dtype {
            case .qweight:
                arr = MLXArray(payloadSlice, dims, dtype: .uint32)
            case .scales, .biases:
                arr = MLXArray(payloadSlice, dims, dtype: .float16)
            }
            collected[kind, default: [:]][dtype] = arr
        }

        var tensors: [JangSpecBundleFormat.TensorKind: ParsedBlob.Triple] = [:]
        for kind in [
            JangSpecBundleFormat.TensorKind.gate,
            .up,
            .down,
        ] {
            guard let kindMap = collected[kind] else { continue }
            guard let q = kindMap[.qweight],
                  let s = kindMap[.scales],
                  let b = kindMap[.biases]
            else { continue }
            tensors[kind] = ParsedBlob.Triple(qweight: q, scales: s, biases: b)
        }

        return ParsedBlob(
            layerIdx: Int(layer),
            expertID: Int(expert),
            bits: bitsSeen ?? 0,
            tensors: tensors
        )
    }

    // MARK: - Expert grouping

    /// Match the layer index in tensor names like
    /// "model.language_model.layers.7.switch_mlp.gate_proj".
    private static let layerRegex = try! NSRegularExpression(
        pattern: #"\.?layers\.(\d+)\."#)

    private static func layerIndex(of base: String) throws -> Int {
        let range = NSRange(base.startIndex..., in: base)
        guard let match = layerRegex.firstMatch(in: base, range: range),
              match.numberOfRanges >= 2,
              let r = Range(match.range(at: 1), in: base),
              let idx = Int(base[r])
        else {
            throw JangSpecBundleError.missingBaseName(base)
        }
        return idx
    }

    private static func groupExpertBasesByLayer(_ baseNames: [String]) throws
        -> [(Int, [String])]
    {
        var byLayer: [Int: [String]] = [:]
        for base in baseNames {
            let idx = try layerIndex(of: base)
            byLayer[idx, default: []].append(base)
        }
        return byLayer.keys.sorted().map { ($0, byLayer[$0]!) }
    }
}
