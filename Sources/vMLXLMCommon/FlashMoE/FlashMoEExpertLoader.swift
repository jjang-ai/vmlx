// SPDX-License-Identifier: Apache-2.0
//
// FlashMoEExpertLoader — on-demand expert loader with slot-bank caching.
//
// Port of `vmlx_engine/utils/flash_moe_loader.py:FlashMoEExpertLoader`
// — reads individual experts out of safetensors files via direct
// file I/O at precomputed byte offsets, no full-file mmap or load.
//
// Threading: reads are issued on a `DispatchQueue` pool sized by
// `FlashMoEConfig.cacheIOSplit` so top-K simultaneous expert loads
// (common when multiple tokens route to disjoint experts) can
// parallelise across SSD channels.
//
// MLX construction: raw bytes are read into `Data`, then wrapped
// into an `MLXArray` via `Data(bytes:)` reshaping. BF16 is the
// gotcha — Swift/Foundation has no native BF16 type, so we read
// 2 bytes per element into a `Float16` stride and let MLX reinterpret
// via `asType(.bfloat16)`. Same approach as the Python loader.

import Foundation
import MLX

/// Loads MoE expert weights on-demand from SSD with slot-bank caching.
public final class FlashMoEExpertLoader: @unchecked Sendable {

    // MARK: - Properties

    /// The expert index. Exposed publicly so `FlashMoE.apply` can
    /// check which layers have MoE weights before asking a layer
    /// to install a shim.
    public let index: ExpertIndex
    private let cache: SlotBankCache
    private let ioSplit: Int
    private let ioQueue: DispatchQueue

    /// Total number of cache-miss loads served from disk.
    public private(set) var diskLoadCount: Int = 0
    /// Cumulative wall-clock time spent on disk reads (ms).
    public private(set) var diskLoadTimeMs: Double = 0
    private let statsLock = NSLock()

    // MARK: - Init

    public init(index: ExpertIndex, cache: SlotBankCache, ioSplit: Int = 4) {
        self.index = index
        self.cache = cache
        self.ioSplit = max(1, ioSplit)
        // Concurrent queue — DispatchQueue.global target allows the
        // thread pool to scale down when I/O quiets down.
        self.ioQueue = DispatchQueue(
            label: "vmlx.flashmoe.io",
            qos: .userInitiated,
            attributes: .concurrent,
            target: .global(qos: .userInitiated)
        )
    }

    // MARK: - Single-expert load

    /// Load a single expert, checking the cache first. Returns `nil`
    /// if the layer has no expert tensors in the index.
    public func loadExpert(layerIdx: Int, expertIdx: Int) -> ExpertWeightSet? {
        if let hit = cache.get(layerIdx: layerIdx, expertIdx: expertIdx) {
            return hit
        }
        guard let layerInfo = index.layers[layerIdx] else { return nil }

        let t0 = DispatchTime.now()
        let expert = loadExpertFromDisk(layerInfo: layerInfo, expertIdx: expertIdx)
        let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000

        statsLock.withLock {
            diskLoadCount += 1
            diskLoadTimeMs += elapsedMs
        }
        cache.put(expert)
        return expert
    }

    // MARK: - Parallel load (top-K experts for one layer)

    /// Load multiple experts for one layer in parallel. Cached hits are
    /// returned directly; misses are dispatched across the I/O queue.
    public func loadExpertsParallel(
        layerIdx: Int,
        expertIndices: [Int]
    ) -> [Int: ExpertWeightSet] {
        var result: [Int: ExpertWeightSet] = [:]
        var toLoad: [Int] = []

        for eidx in expertIndices {
            if let hit = cache.get(layerIdx: layerIdx, expertIdx: eidx) {
                result[eidx] = hit
            } else {
                toLoad.append(eidx)
            }
        }
        guard !toLoad.isEmpty, let layerInfo = index.layers[layerIdx] else {
            return result
        }

        let t0 = DispatchTime.now()
        let group = DispatchGroup()
        let resultLock = NSLock()

        for eidx in toLoad {
            group.enter()
            ioQueue.async { [self] in
                let expert = loadExpertFromDisk(layerInfo: layerInfo, expertIdx: eidx)
                cache.put(expert)
                resultLock.withLock { result[eidx] = expert }
                group.leave()
            }
        }
        group.wait()

        let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000
        statsLock.withLock {
            diskLoadCount += toLoad.count
            diskLoadTimeMs += elapsedMs
        }
        return result
    }

    // MARK: - Stats

    public struct Stats: Sendable, Equatable {
        public var slotBank: SlotBankStats
        public var diskLoads: Int
        public var diskLoadTimeMs: Double
        public var avgLoadMs: Double
        public var ioWorkers: Int
    }

    public func stats() -> Stats {
        let cacheStats = cache.stats()
        let (loads, time) = statsLock.withLock { (diskLoadCount, diskLoadTimeMs) }
        return Stats(
            slotBank: cacheStats,
            diskLoads: loads,
            diskLoadTimeMs: time,
            avgLoadMs: loads > 0 ? time / Double(loads) : 0,
            ioWorkers: ioSplit
        )
    }

    // MARK: - Private: disk read

    private func loadExpertFromDisk(
        layerInfo: LayerExpertInfo, expertIdx: Int
    ) -> ExpertWeightSet {
        var tensors: [ExpertProjection: [ExpertTensorSuffix: MLXArray]] = [:]
        let projections: [(ExpertProjection, ProjectionTensors?)] = [
            (.gateProj, layerInfo.gateProj),
            (.upProj,   layerInfo.upProj),
            (.downProj, layerInfo.downProj),
        ]
        for (name, maybeProj) in projections {
            guard let proj = maybeProj else { continue }
            let inner = loadProjectionExpert(proj: proj, expertIdx: expertIdx)
            if !inner.isEmpty {
                tensors[name] = inner
            }
        }
        return ExpertWeightSet(
            layerIdx: layerInfo.layerIdx,
            expertIdx: expertIdx,
            tensors: tensors
        )
    }

    private func loadProjectionExpert(
        proj: ProjectionTensors, expertIdx: Int
    ) -> [ExpertTensorSuffix: MLXArray] {
        var result: [ExpertTensorSuffix: MLXArray] = [:]
        for (suffix, maybeTI) in [
            (ExpertTensorSuffix.weight, proj.weight),
            (ExpertTensorSuffix.scales, proj.scales),
            (ExpertTensorSuffix.biases, proj.biases),
        ] {
            guard let ti = maybeTI,
                  let arr = preadExpertTensor(ti: ti, expertIdx: expertIdx)
            else { continue }
            result[suffix] = arr
        }
        return result
    }

    /// Reads a single expert's slice from a safetensors file and
    /// returns it as an `MLXArray` with shape `ti.shape[1...]`.
    ///
    /// For tensors without a leading expert dim (rare — shouldn't
    /// happen for properly-indexed expert tensors) the full tensor
    /// is returned.
    private func preadExpertTensor(ti: TensorInfo, expertIdx: Int) -> MLXArray? {
        // Compute per-expert byte count.
        guard ti.shape.count >= 2 else {
            return readFullTensor(ti: ti)
        }
        let expertShape = Array(ti.shape.dropFirst())
        let expertElements = expertShape.reduce(1, *)
        let bytesPerElement = bytesPerElement(dtype: ti.dtype)
        let expertBytes = expertElements * bytesPerElement
        let offset = ti.absOffset + expertIdx * expertBytes

        guard let data = readBytes(from: ti.filePath, offset: offset, count: expertBytes),
              data.count == expertBytes
        else { return nil }

        return makeMLXArray(data: data, shape: expertShape, dtype: ti.dtype)
    }

    private func readFullTensor(ti: TensorInfo) -> MLXArray? {
        guard let data = readBytes(from: ti.filePath, offset: ti.absOffset, count: ti.numBytes) else {
            return nil
        }
        return makeMLXArray(data: data, shape: ti.shape, dtype: ti.dtype)
    }

    private func readBytes(from url: URL, offset: Int, count: Int) -> Data? {
        guard let fh = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? fh.close() }
        do {
            try fh.seek(toOffset: UInt64(offset))
        } catch {
            return nil
        }
        return fh.readData(ofLength: count)
    }

    // MARK: - dtype helpers

    private func bytesPerElement(dtype: String) -> Int {
        switch dtype {
        case "F16", "BF16":   return 2
        case "F32", "I32", "U32": return 4
        case "U8":            return 1
        default:              return 1
        }
    }

    private func makeMLXArray(data: Data, shape: [Int], dtype: String) -> MLXArray? {
        // Fast path: construct from raw bytes per dtype.
        // MLX.Swift exposes `MLXArray(bytes:shape:dtype:)`-style initializers
        // through Data conversion helpers — we follow the same pattern as
        // JangMXTQDequant's raw-bytes loader.
        // Explicit intermediate typed arrays: under the xcodebuild
        // Swift compiler, the overload resolution for
        // `MLXArray(Array(ptr), shape)` picks the wrong init and fails
        // with "cannot convert Array<Float16> to [Int]". Annotating each
        // intermediate as the element array forces the typed generic
        // init. SwiftPM was tolerating this under looser inference.
        switch dtype {
        case "F16":
            let arr = data.withUnsafeBytes { raw -> MLXArray in
                let ptr = raw.bindMemory(to: Float16.self)
                let values: [Float16] = Array(ptr)
                return MLXArray(values, shape)
            }
            return arr
        case "BF16":
            let arr = data.withUnsafeBytes { raw -> MLXArray in
                let ptr = raw.bindMemory(to: Float16.self)
                let values: [Float16] = Array(ptr)
                return MLXArray(values, shape)
            }
            return arr.asType(DType.bfloat16)
        case "F32":
            let arr = data.withUnsafeBytes { raw -> MLXArray in
                let ptr = raw.bindMemory(to: Float.self)
                let values: [Float] = Array(ptr)
                return MLXArray(values, shape)
            }
            return arr
        case "U32":
            let arr = data.withUnsafeBytes { raw -> MLXArray in
                let ptr = raw.bindMemory(to: UInt32.self)
                let values: [UInt32] = Array(ptr)
                return MLXArray(values, shape)
            }
            return arr
        case "I32":
            let arr = data.withUnsafeBytes { raw -> MLXArray in
                let ptr = raw.bindMemory(to: Int32.self)
                let values: [Int32] = Array(ptr)
                return MLXArray(values, shape)
            }
            return arr
        case "U8":
            let arr = data.withUnsafeBytes { raw -> MLXArray in
                let ptr = raw.bindMemory(to: UInt8.self)
                let values: [UInt8] = Array(ptr)
                return MLXArray(values, shape)
            }
            return arr
        default:
            return nil
        }
    }
}
