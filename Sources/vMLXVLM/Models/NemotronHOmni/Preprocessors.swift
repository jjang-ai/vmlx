// Preprocessors.swift
// Image / audio / video preprocessors for Nemotron-3-Nano-Omni.
//
// Apple-native: CIImage + AVFoundation + Accelerate (vDSP). No PyTorch /
// torchvision dependency.

import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
import Accelerate
import AVFoundation
import MLX

// MARK: - Image preprocessing (NVLM 1-D dynamic tile)

/// OpenAI CLIP normalization mean (matches source `norm_mean`)
public let NEMOTRON_OMNI_CLIP_MEAN: [Float] = [0.48145466, 0.4578275, 0.40821073]
/// OpenAI CLIP normalization std (matches source `norm_std`)
public let NEMOTRON_OMNI_CLIP_STD: [Float] = [0.26862954, 0.26130258, 0.27577711]

private let imagePreprocessContext = CIContext(options: [
    .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB) as Any
])

/// Pick the (cols, rows) tile grid whose aspect ratio best matches the input.
/// Mirrors `_find_closest_aspect_ratio` in image_processor.py.
private func findClosestAspectRatio(
    aspectRatio: Float,
    targetRatios: [(Int, Int)],
    width: Int, height: Int,
    imageSize: Int
) -> (Int, Int) {
    var bestDiff = Float.infinity
    var best = (1, 1)
    let area = Float(width * height)
    for ratio in targetRatios {
        let target = Float(ratio.0) / Float(ratio.1)
        let diff = abs(aspectRatio - target)
        if diff < bestDiff {
            bestDiff = diff
            best = ratio
        } else if diff == bestDiff {
            // Tie-break: prefer ratio that fits more area
            if area > 0.5 * Float(imageSize * imageSize * ratio.0 * ratio.1) {
                best = ratio
            }
        }
    }
    return best
}

/// Build candidate (cols, rows) tile grids with cols*rows in [minNum, maxNum].
private func buildTargetRatios(minNum: Int, maxNum: Int) -> [(Int, Int)] {
    var set = Set<String>()
    var ratios: [(Int, Int)] = []
    for n in minNum ... maxNum {
        for c in 1 ... n {
            for r in 1 ... n {
                let prod = c * r
                if prod >= minNum && prod <= maxNum {
                    let key = "\(c)x\(r)"
                    if !set.contains(key) {
                        set.insert(key)
                        ratios.append((c, r))
                    }
                }
            }
        }
    }
    // Sort by total tile count (matches Python `sorted(..., key=lambda x: x[0] * x[1])`)
    ratios.sort { $0.0 * $0.1 < $1.0 * $1.1 }
    return ratios
}

/// NVLM-style dynamic tiling. Returns a list of CIImage tiles (each at
/// imageSize×imageSize), with optional thumbnail appended.
public func nemotronOmniDynamicPreprocess(
    _ image: CIImage,
    imageSize: Int = 512,
    minNum: Int = 1,
    maxNum: Int = 12,
    useThumbnail: Bool = true
) -> [CIImage] {
    let extent = image.extent
    let origW = max(1, Int(extent.width.rounded()))
    let origH = max(1, Int(extent.height.rounded()))
    let aspect = Float(origW) / Float(origH)

    let targetRatios = buildTargetRatios(minNum: minNum, maxNum: maxNum)
    let (cols, rows) = findClosestAspectRatio(
        aspectRatio: aspect, targetRatios: targetRatios,
        width: origW, height: origH, imageSize: imageSize)

    let targetW = imageSize * cols
    let targetH = imageSize * rows

    // Bicubic resize to (targetW × targetH)
    let resized = MediaProcessing.resampleBicubic(
        MediaProcessing.inSRGBToneCurveSpace(image),
        to: CGSize(width: targetW, height: targetH))

    // Slice into cols × rows tiles. CIImage origin is bottom-left so we
    // crop using a translated rect for top-down ordering matching Python:
    // for i in range(blocks): col = i%cols, row = i//cols.
    var tiles: [CIImage] = []
    let imgRect = resized.extent
    for i in 0 ..< (cols * rows) {
        let col = i % cols
        let row = i / cols
        // Top-down crop: row=0 is top in image space, but CI origin is bottom.
        let yTop = imgRect.height - CGFloat((row + 1) * imageSize)
        let cropRect = CGRect(
            x: CGFloat(col * imageSize),
            y: yTop,
            width: CGFloat(imageSize),
            height: CGFloat(imageSize))
        let tile = resized
            .cropped(to: cropRect)
            .transformed(by: CGAffineTransform(translationX: -cropRect.minX, y: -cropRect.minY))
        tiles.append(tile)
    }

    if useThumbnail && (cols * rows) != 1 {
        let thumb = MediaProcessing.resampleBicubic(
            MediaProcessing.inSRGBToneCurveSpace(image),
            to: CGSize(width: imageSize, height: imageSize))
        tiles.append(thumb)
    }
    return tiles
}

/// Render a CIImage tile (sized imageSize×imageSize) into a planar
/// (3, imageSize, imageSize) Float32 array, normalized by CLIP mean/std.
private func rasterizeTile(
    _ tile: CIImage,
    imageSize: Int,
    mean: [Float], std: [Float]
) -> MLXArray {
    // Render to RGBA Float32 bitmap.
    let bytesPerRow = imageSize * 4 * MemoryLayout<Float>.size
    var data = Data(count: imageSize * imageSize * 4 * MemoryLayout<Float>.size)
    data.withUnsafeMutableBytes { ptr in
        imagePreprocessContext.render(
            tile,
            toBitmap: ptr.baseAddress!,
            rowBytes: bytesPerRow,
            bounds: CGRect(x: 0, y: 0, width: imageSize, height: imageSize),
            format: .RGBAf,
            colorSpace: nil)
    }

    // Build (H, W, 4) array, drop alpha → (H, W, 3), normalize, transpose to (3, H, W)
    var arr = MLXArray(data, [imageSize, imageSize, 4], type: Float32.self)
    arr = arr[0..., 0..., ..<3] // (H, W, 3)
    let meanT = MLXArray(mean).reshaped([1, 1, 3])
    let stdT = MLXArray(std).reshaped([1, 1, 3])
    arr = (arr - meanT) / stdT
    // (H, W, 3) → (3, H, W)
    return arr.transposed(2, 0, 1)
}

/// Process one or more CIImages into model-ready tile pixel values.
///
/// Returns: pixelValues of shape (totalTiles, 3, imageSize, imageSize) and
/// per-input tile counts.
public func nemotronOmniPreprocessImages(
    _ images: [CIImage],
    imageSize: Int = 512,
    minNum: Int = 1,
    maxNum: Int = 12,
    useThumbnail: Bool = true
) -> (pixelValues: MLXArray, tileCounts: [Int]) {
    var allTiles: [MLXArray] = []
    var counts: [Int] = []
    for img in images {
        let tiles = nemotronOmniDynamicPreprocess(
            img, imageSize: imageSize,
            minNum: minNum, maxNum: maxNum,
            useThumbnail: useThumbnail)
        for t in tiles {
            allTiles.append(rasterizeTile(
                t, imageSize: imageSize,
                mean: NEMOTRON_OMNI_CLIP_MEAN, std: NEMOTRON_OMNI_CLIP_STD))
        }
        counts.append(tiles.count)
    }
    if allTiles.isEmpty {
        return (MLXArray.zeros([0, 3, imageSize, imageSize]), [])
    }
    let stacked = MLX.stacked(allTiles, axis: 0) // (N, 3, H, W)
    return (stacked, counts)
}

// MARK: - Audio preprocessing (parakeet mel STFT)

/// Hann window of length n (periodic=False matches np.hanning / scipy default).
private func hannWindow(length: Int, periodic: Bool = false) -> [Float] {
    let n = periodic ? length + 1 : length
    var w = [Float](repeating: 0, count: n)
    for i in 0 ..< n {
        w[i] = Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double(i) / Double(n - 1)))
    }
    return Array(w.prefix(length))
}

/// Slaney-norm mel filterbank, shape (nMels, nFFT/2+1).
/// Pure-Swift port of `_slaney_mel_filterbank` in audio_features.py.
private func slaneyMelFilterbank(
    sampleRate: Int, nFFT: Int, nMels: Int,
    fmin: Float, fmax: Float
) -> [[Float]] {
    let nBins = nFFT / 2 + 1
    var fftFreqs = [Float](repeating: 0, count: nBins)
    let nyquist = Float(sampleRate) / 2
    for i in 0 ..< nBins {
        fftFreqs[i] = Float(i) * nyquist / Float(nBins - 1)
    }

    // Slaney hz <-> mel
    let fSp: Float = 200.0 / 3
    let minLogHz: Float = 1000.0
    let minLogMel = (minLogHz - 0) / fSp
    let logstep = log(Float(6.4)) / 27.0

    func hzToMel(_ f: Float) -> Float {
        if f >= minLogHz {
            return minLogMel + log(f / minLogHz) / logstep
        }
        return f / fSp
    }
    func melToHz(_ m: Float) -> Float {
        if m >= minLogMel {
            return minLogHz * exp(logstep * (m - minLogMel))
        }
        return fSp * m
    }

    let melMin = hzToMel(fmin)
    let melMax = hzToMel(fmax)
    let nPts = nMels + 2
    var melPts = [Float](repeating: 0, count: nPts)
    for i in 0 ..< nPts {
        melPts[i] = melMin + (melMax - melMin) * Float(i) / Float(nPts - 1)
    }
    let hzPts = melPts.map { melToHz($0) }

    var fb = [[Float]](repeating: [Float](repeating: 0, count: nBins), count: nMels)
    for i in 0 ..< nMels {
        let l = hzPts[i]
        let c = hzPts[i + 1]
        let r = hzPts[i + 2]
        let widthHz = max(r - l, 1e-12)
        let scale = 2.0 / widthHz // Slaney norm
        for k in 0 ..< nBins {
            let f = fftFreqs[k]
            let rising = (f - l) / max(c - l, 1e-12)
            let falling = (r - f) / max(r - c, 1e-12)
            let tri = max(0, min(rising, falling))
            fb[i][k] = tri * scale
        }
    }
    return fb
}

/// Cache for mel filterbank (computed once per (sr, nFFT, nMels, fmin, fmax)).
private final class MelFilterbankCache: @unchecked Sendable {
    static let shared = MelFilterbankCache()
    private var entries: [String: [[Float]]] = [:]
    private let queue = DispatchQueue(label: "nemotron.omni.mel.cache")

    func get(sr: Int, nFFT: Int, nMels: Int, fmin: Float, fmax: Float) -> [[Float]] {
        let key = "\(sr)|\(nFFT)|\(nMels)|\(fmin)|\(fmax)"
        return queue.sync {
            if let cached = entries[key] { return cached }
            let fb = slaneyMelFilterbank(
                sampleRate: sr, nFFT: nFFT, nMels: nMels, fmin: fmin, fmax: fmax)
            entries[key] = fb
            return fb
        }
    }
}

/// Preemphasis: y[t] = x[t] - coef * x[t-1], y[0] = x[0].
private func preemphasis(_ waveform: [Float], coef: Float) -> [Float] {
    if coef == 0 { return waveform }
    var out = [Float](repeating: 0, count: waveform.count)
    if !waveform.isEmpty { out[0] = waveform[0] }
    for i in 1 ..< waveform.count {
        out[i] = waveform[i] - coef * waveform[i - 1]
    }
    return out
}

/// Real FFT via vDSP. Returns interleaved complex frames as
/// (nBins=nFFT/2+1, nFrames). Each frame uses a centered, Hann-windowed
/// segment of the waveform with constant zero padding.
private func stftRFFT(
    waveform: [Float], nFFT: Int, hopLength: Int,
    winLength: Int, window: [Float]
) -> (real: [Float], imag: [Float], nFrames: Int) {
    let nBins = nFFT / 2 + 1

    // Center pad waveform with nFFT/2 zeros on each side
    let pad = nFFT / 2
    var padded = [Float](repeating: 0, count: pad)
    padded.append(contentsOf: waveform)
    padded.append(contentsOf: [Float](repeating: 0, count: pad))

    // Window padded to nFFT (centered when winLength < nFFT)
    var fullWindow = [Float](repeating: 0, count: nFFT)
    if winLength < nFFT {
        let off = (nFFT - winLength) / 2
        for i in 0 ..< winLength {
            fullWindow[off + i] = window[i]
        }
    } else {
        for i in 0 ..< nFFT {
            fullWindow[i] = i < window.count ? window[i] : 0
        }
    }

    let nFrames = max(0, 1 + (padded.count - nFFT) / hopLength)
    var realOut = [Float](repeating: 0, count: nBins * nFrames)
    var imagOut = [Float](repeating: 0, count: nBins * nFrames)

    let log2n = vDSP_Length(log2(Double(nFFT)))
    guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
        return (realOut, imagOut, nFrames)
    }
    defer { vDSP_destroy_fftsetup(setup) }

    var realIn = [Float](repeating: 0, count: nFFT / 2)
    var imagIn = [Float](repeating: 0, count: nFFT / 2)
    var split = DSPSplitComplex(realp: &realIn, imagp: &imagIn)

    var frame = [Float](repeating: 0, count: nFFT)
    for f in 0 ..< nFrames {
        let start = f * hopLength
        for i in 0 ..< nFFT {
            frame[i] = padded[start + i] * fullWindow[i]
        }
        // Pack real array into split-complex (length nFFT/2)
        frame.withUnsafeBytes { rawBuf in
            let ptr = rawBuf.baseAddress!.assumingMemoryBound(to: DSPComplex.self)
            // ctoz needs a non-mutating split for input — but vDSP wants mutable;
            // declare anew.
            realIn.withUnsafeMutableBufferPointer { rPtr in
                imagIn.withUnsafeMutableBufferPointer { iPtr in
                    var s = DSPSplitComplex(realp: rPtr.baseAddress!, imagp: iPtr.baseAddress!)
                    vDSP_ctoz(ptr, 2, &s, 1, vDSP_Length(nFFT / 2))
                    vDSP_fft_zrip(setup, &s, 1, log2n, FFTDirection(FFT_FORWARD))
                    // Unpack: bin 0 real = realp[0]; bin N/2 real = imagp[0];
                    // bins 1..N/2-1 packed in (realp[k], imagp[k]). vDSP zrip
                    // returns them already scaled by 2× for r2c convention, so
                    // we scale by 0.5 to match numpy.fft.rfft.
                    realOut[f * nBins + 0] = rPtr[0] * 0.5
                    imagOut[f * nBins + 0] = 0
                    realOut[f * nBins + nFFT / 2] = iPtr[0] * 0.5
                    imagOut[f * nBins + nFFT / 2] = 0
                    for k in 1 ..< (nFFT / 2) {
                        realOut[f * nBins + k] = rPtr[k] * 0.5
                        imagOut[f * nBins + k] = iPtr[k] * 0.5
                    }
                }
            }
        }
    }
    return (realOut, imagOut, nFrames)
}

/// Compute parakeet mel features for a single waveform.
/// Mirrors `extract_mel_features` in audio_features.py.
/// Returns shape (1, nFrames, nMels) MLXArray.
public func nemotronOmniExtractMelFeatures(
    _ waveform: [Float],
    sampleRate: Int = 16000,
    nFFT: Int = 512,
    hopLength: Int = 160,
    winLength: Int = 400,
    nMels: Int = 128,
    preemphasisCoef: Float = 0.97,
    normalize: Bool = true,
    fmin: Float = 0,
    fmax: Float? = nil
) -> MLXArray {
    let waveform = preemphasis(waveform, coef: preemphasisCoef)
    let window = hannWindow(length: winLength, periodic: false)
    let actualFmax = fmax ?? Float(sampleRate) / 2

    let (real, imag, nFrames) = stftRFFT(
        waveform: waveform, nFFT: nFFT, hopLength: hopLength,
        winLength: winLength, window: window)

    let nBins = nFFT / 2 + 1
    let fb = MelFilterbankCache.shared.get(
        sr: sampleRate, nFFT: nFFT, nMels: nMels, fmin: fmin, fmax: actualFmax)

    // Power spectrum + mel projection (per frame). Layout: (nMels, nFrames).
    var mel = [Float](repeating: 0, count: nMels * nFrames)
    let logZeroGuard = Float(pow(2.0, -24.0))
    for f in 0 ..< nFrames {
        // Power: |X|^2 per bin
        var power = [Float](repeating: 0, count: nBins)
        for k in 0 ..< nBins {
            let r = real[f * nBins + k]
            let i = imag[f * nBins + k]
            power[k] = r * r + i * i
        }
        // mel[m, f] = sum_k fb[m][k] * power[k]
        for m in 0 ..< nMels {
            var s: Float = 0
            let row = fb[m]
            for k in 0 ..< nBins {
                s += row[k] * power[k]
            }
            mel[m * nFrames + f] = log(s + logZeroGuard)
        }
    }

    // Reshape to (nFrames, nMels) row-major (transpose)
    var melT = [Float](repeating: 0, count: nFrames * nMels)
    for f in 0 ..< nFrames {
        for m in 0 ..< nMels {
            melT[f * nMels + m] = mel[m * nFrames + f]
        }
    }

    if normalize && nFrames > 1 {
        // Per-sample (whole utterance) zero-mean unit-variance.
        // Mean over nFrames axis (PyTorch source computes along time per channel).
        var mean = [Float](repeating: 0, count: nMels)
        for f in 0 ..< nFrames {
            for m in 0 ..< nMels {
                mean[m] += melT[f * nMels + m]
            }
        }
        for m in 0 ..< nMels { mean[m] /= Float(nFrames) }

        var variance = [Float](repeating: 0, count: nMels)
        for f in 0 ..< nFrames {
            for m in 0 ..< nMels {
                let d = melT[f * nMels + m] - mean[m]
                variance[m] += d * d
            }
        }
        // Bessel-corrected variance
        let denom = Float(max(nFrames - 1, 1))
        for m in 0 ..< nMels { variance[m] /= denom }
        let eps: Float = 1e-5
        for f in 0 ..< nFrames {
            for m in 0 ..< nMels {
                let std = sqrt(variance[m])
                melT[f * nMels + m] = (melT[f * nMels + m] - mean[m]) / (std + eps)
            }
        }
    }

    var arr = MLXArray(melT)
    arr = arr.reshaped([1, nFrames, nMels])
    return arr
}

/// Load a 16 kHz mono Float32 array from an audio file.
/// Resamples + downmixes if necessary.
public func nemotronOmniLoadAudioFile(
    _ url: URL,
    targetSampleRate: Double = 16000
) throws -> [Float] {
    let file = try AVAudioFile(forReading: url)
    let inputFormat = file.processingFormat

    guard let outFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: targetSampleRate,
        channels: 1,
        interleaved: false)
    else {
        throw NSError(
            domain: "NemotronHOmni", code: -1,
            userInfo: [NSLocalizedDescriptionKey: "Failed to create output audio format"])
    }

    // Read full file
    let frameCount = AVAudioFrameCount(file.length)
    guard frameCount > 0 else { return [] }
    guard let inBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: frameCount) else {
        throw NSError(
            domain: "NemotronHOmni", code: -2,
            userInfo: [NSLocalizedDescriptionKey: "Failed to alloc input buffer"])
    }
    try file.read(into: inBuffer)

    // Fast path: already 16 kHz mono float32 PCM
    if abs(inputFormat.sampleRate - targetSampleRate) < 0.5
        && inputFormat.channelCount == 1
        && inputFormat.commonFormat == .pcmFormatFloat32
        && !inputFormat.isInterleaved
    {
        let n = Int(inBuffer.frameLength)
        let chData = inBuffer.floatChannelData!
        return Array(UnsafeBufferPointer(start: chData[0], count: n))
    }

    // Convert via AVAudioConverter
    guard let converter = AVAudioConverter(from: inputFormat, to: outFormat) else {
        throw NSError(
            domain: "NemotronHOmni", code: -3,
            userInfo: [NSLocalizedDescriptionKey: "Failed to create AVAudioConverter"])
    }

    let outFrames = AVAudioFrameCount(
        Double(inBuffer.frameLength) * targetSampleRate / inputFormat.sampleRate + 1024)
    guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outFrames) else {
        throw NSError(
            domain: "NemotronHOmni", code: -4,
            userInfo: [NSLocalizedDescriptionKey: "Failed to alloc output buffer"])
    }

    var consumed = false
    var error: NSError?
    let status = converter.convert(to: outBuffer, error: &error) { _, outStatus in
        if consumed {
            outStatus.pointee = .endOfStream
            return nil
        }
        consumed = true
        outStatus.pointee = .haveData
        return inBuffer
    }

    if status == .error {
        throw error ?? NSError(
            domain: "NemotronHOmni", code: -5,
            userInfo: [NSLocalizedDescriptionKey: "AVAudioConverter failed"])
    }

    let n = Int(outBuffer.frameLength)
    let chData = outBuffer.floatChannelData!
    return Array(UnsafeBufferPointer(start: chData[0], count: n))
}

// MARK: - Video preprocessing (frame extraction + EVS)

/// Extract uniformly-spaced frames from a video file using the same async
/// `AVAssetImageGenerator.images(for:)` API as Qwen 2/2.5/3/3.5/3.6 VL.
/// Reuses `MediaProcessing.asCIImageSequence` so the frame-decode pipeline
/// is shared across every VLM in vmlx-swift-lm.
///
/// Returns up to `targetFrames` frames. Frames are full-resolution CIImages
/// (caller post-processes them via `nemotronOmniPreprocessVideo`).
public func nemotronOmniExtractVideoFrames(
    _ url: URL,
    targetFrames: Int = 32
) async throws -> [CIImage] {
    let asset = AVURLAsset(url: url)
    let duration = try await asset.load(.duration)
    let durationSeconds = duration.seconds
    guard durationSeconds.isFinite, durationSeconds > 0 else {
        throw NSError(
            domain: "NemotronHOmni", code: -10,
            userInfo: [NSLocalizedDescriptionKey: "Invalid video duration"])
    }
    // Translate target frame count into samplesPerSecond. The shared
    // `asCIImageSequence` uses linspace(0, durationValue, count: fps*duration)
    // — we invert: samplesPerSecond = targetFrames / duration.
    let samplesPerSecond = max(1, Int(round(Double(targetFrames) / durationSeconds)))
    return try await MediaProcessing.asCIImageSequence(
        asset, samplesPerSecond: samplesPerSecond,
    )
}

/// Native Swift video preprocessor — full Nemotron-3-Nano-Omni pipeline:
///   1. Frame extraction (uniform sample via `MediaProcessing.asCIImageSequence`)
///   2. Pad N to a multiple of `videoTemporalPatchDim=2` by repeating last frame
///   3. Bicubic resize each frame to (imageSize, imageSize) via CIImage transform
///   4. RGB Float32 extraction via vImage
///   5. CLIP normalize per channel
///   6. Stack T frames into channel dim → (N/T, T*3, H, W) MLXArray
///
/// Returns the MLXArray that can be fed directly to `RADIOVisionModel(x, video: true)`.
///
/// - Parameters:
///   - url: video file URL
///   - imageSize: per-frame target size (default 512, matches force_image_size)
///   - targetFrames: how many frames to sample (default 32)
///   - videoTemporalPatchDim: T (default 2 — RADIO video_embedder accepts T*3*P*P input)
public func nemotronOmniPreprocessVideo(
    url: URL,
    imageSize: Int = 512,
    targetFrames: Int = 32,
    videoTemporalPatchDim: Int = 2
) async throws -> MLXArray {
    var frames = try await nemotronOmniExtractVideoFrames(
        url, targetFrames: targetFrames,
    )
    if frames.isEmpty {
        throw NSError(
            domain: "NemotronHOmni", code: -11,
            userInfo: [NSLocalizedDescriptionKey: "No frames decoded from video"])
    }

    // Pad to a multiple of T by repeating the last frame
    while frames.count % videoTemporalPatchDim != 0 {
        frames.append(frames.last!)
    }
    let nFrames = frames.count
    let nGroups = nFrames / videoTemporalPatchDim
    let H = imageSize
    let W = imageSize

    // Resize + normalize each frame to a (3, H, W) Float32 array.
    // Layout: per-frame (3, H, W) row-major — same as PyTorch convention.
    var stacked = [Float](repeating: 0, count: nFrames * 3 * H * W)
    let perFrame = 3 * H * W
    for (i, frame) in frames.enumerated() {
        let resized = nemotronOmniResizeAndNormalize(frame, target: imageSize)
        for j in 0 ..< perFrame {
            stacked[i * perFrame + j] = resized[j]
        }
    }

    // Build (N, 3, H, W) MLXArray, then reshape to (N/T, T*3, H, W).
    let pixelValues = MLXArray(stacked, [nFrames, 3, H, W])
    return pixelValues.reshaped([nGroups, videoTemporalPatchDim * 3, H, W])
}

/// Bicubic resize a CIImage to (target, target) and normalize via CLIP
/// mean/std → returns a contiguous (3*target*target,) Float32 buffer in
/// (3, H, W) order.
private func nemotronOmniResizeAndNormalize(_ image: CIImage, target: Int) -> [Float] {
    // Resize via CIImage affine transform with Lanczos interpolation
    // (closest match to PyTorch BICUBIC for our use-case).
    let extent = image.extent
    let scaleX = CGFloat(target) / extent.width
    let scaleY = CGFloat(target) / extent.height
    let resized = image
        .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        .cropped(to: CGRect(x: 0, y: 0, width: target, height: target))

    // Render to RGBA8 via CIContext, then strip alpha + normalize
    let ctx = imagePreprocessContext
    let bitsPerComponent = 8
    let bytesPerRow = 4 * target
    var buffer = [UInt8](repeating: 0, count: 4 * target * target)
    let cgContext = CGContext(
        data: &buffer, width: target, height: target,
        bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue,
    )!
    if let cg = ctx.createCGImage(resized, from: CGRect(x: 0, y: 0, width: target, height: target)) {
        cgContext.draw(cg, in: CGRect(x: 0, y: 0, width: target, height: target))
    }

    // Convert (H, W, 4) RGBA8 → (3, H, W) Float32 with CLIP normalization
    let mean: [Float] = NEMOTRON_OMNI_CLIP_MEAN
    let std: [Float] = NEMOTRON_OMNI_CLIP_STD
    var out = [Float](repeating: 0, count: 3 * target * target)
    let plane = target * target
    for y in 0 ..< target {
        for x in 0 ..< target {
            let i = (y * target + x) * 4
            let r = Float(buffer[i]) / 255.0
            let g = Float(buffer[i + 1]) / 255.0
            let b = Float(buffer[i + 2]) / 255.0
            let pix = y * target + x
            out[0 * plane + pix] = (r - mean[0]) / std[0]
            out[1 * plane + pix] = (g - mean[1]) / std[1]
            out[2 * plane + pix] = (b - mean[2]) / std[2]
        }
    }
    return out
}

/// Apply Efficient Video Sampling (EVS) at the *embedding* level.
/// Drops `pruningRate` of redundant tokens by computing cosine similarity
/// between consecutive temporal-group embeddings at the same spatial position.
/// Mirrors the broad strokes of `compute_evs_retention_mask` in video_processor.py.
///
/// - Parameters:
///   - feats: (nGroups, tokensPerGroup, hidden) MLXArray
///   - pruningRate: fraction of tokens to drop
public func nemotronOmniApplyEVS(
    _ feats: MLXArray,
    pruningRate: Float = 0.7
) -> MLXArray {
    let nGroups = feats.dim(0)
    let tokensPerGroup = feats.dim(1)
    let hidden = feats.dim(2)

    if nGroups < 2 {
        return feats
    }

    // Cosine similarity between consecutive groups, per token position.
    let g0 = feats[0 ..< (nGroups - 1)] // (G-1, P, D)
    let g1 = feats[1 ..< nGroups]
    let dot = (g0 * g1).sum(axis: -1) // (G-1, P)
    let n0 = MLX.sqrt((g0 * g0).sum(axis: -1) + 1e-8)
    let n1 = MLX.sqrt((g1 * g1).sum(axis: -1) + 1e-8)
    let cos = dot / (n0 * n1) // (G-1, P)

    // Keep first group entirely. For subsequent groups, drop tokens with
    // highest similarity to corresponding token in prior group, until
    // `pruningRate` of total dropped.
    let totalTokens = nGroups * tokensPerGroup
    let dropTarget = Int(Float(totalTokens) * pruningRate)

    // Convert cos to flat index list sorted by similarity desc, restricted
    // to non-first-group tokens.
    let cosFlat = cos.reshaped([(nGroups - 1) * tokensPerGroup])
    let cosArray = cosFlat.asArray(Float.self)
    let sortedIdx = (0 ..< cosArray.count).sorted { cosArray[$0] > cosArray[$1] }

    // Build keep mask of size nGroups*tokensPerGroup, default true.
    var keep = [Bool](repeating: true, count: totalTokens)
    var dropped = 0
    for relIdx in sortedIdx {
        if dropped >= dropTarget { break }
        // relIdx in cosFlat corresponds to (group=1+relIdx/P, token=relIdx%P)
        let group = 1 + relIdx / tokensPerGroup
        let tokIn = relIdx % tokensPerGroup
        let absIdx = group * tokensPerGroup + tokIn
        if keep[absIdx] {
            keep[absIdx] = false
            dropped += 1
        }
    }

    // Gather kept indices and produce (1, kept, hidden) tensor.
    let keptIdx = (0 ..< totalTokens).filter { keep[$0] }
    if keptIdx.isEmpty {
        return feats[0 ..< 1]
    }
    let flat = feats.reshaped([totalTokens, hidden])
    let idxArr = MLXArray(keptIdx.map { Int32($0) })
    let gathered = flat.take(idxArr, axis: 0)
    return gathered.reshaped([1, keptIdx.count, hidden])
}
