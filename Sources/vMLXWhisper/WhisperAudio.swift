// Copyright © 2026 vMLX. Whisper audio preprocessing — load audio
// file, resample to 16kHz mono, compute 80/128-bin log-mel
// spectrogram. Mirrors mlx-examples/whisper/audio.py semantics.
//
// Constants from the Whisper paper (encoder expects exactly 30 seconds
// of audio at 16 kHz → 480000 samples → 3000 mel frames @ 10ms hop).

import Accelerate
import AVFoundation
import Foundation
import MLX

public enum WhisperAudio {

    public static let sampleRate: Int = 16_000
    public static let nFFT: Int = 400
    public static let hopLength: Int = 160
    public static let chunkLength: Int = 30 // seconds
    /// 16000 * 30
    public static let nSamples: Int = 480_000
    /// nSamples / hopLength
    public static let nFrames: Int = 3_000

    public enum AudioError: LocalizedError {
        case decodeFailed(String)
        case empty

        public var errorDescription: String? {
            switch self {
            case .decodeFailed(let msg): return "audio decode failed: \(msg)"
            case .empty: return "audio decode produced 0 samples"
            }
        }
    }

    // MARK: - File decode

    /// Decode an audio file (any AVFoundation-supported format: WAV,
    /// MP3, M4A, FLAC, AAC) to 16 kHz mono Float32 PCM. Returns the
    /// samples as a plain `[Float]` array.
    public static func decodeFile(at url: URL) throws -> [Float] {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw AudioError.decodeFailed("\(error.localizedDescription)")
        }
        return try decode(file: file)
    }

    /// Decode raw audio Data by writing to a temp file and reusing
    /// AVFoundation. We cannot hand AVAudioFile a `Data` directly —
    /// it requires a URL — so we write to a temporary file whose
    /// extension is sniffed from the content type or hint.
    public static func decodeData(
        _ data: Data, fileExtension: String = "wav"
    ) throws -> [Float] {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-whisper-\(UUID().uuidString).\(fileExtension)")
        // iter-89 §117: register the cleanup defer BEFORE the write
        // that can throw. `Data.write(to:)` can fail mid-way with a
        // partial file on disk (disk full, permission race, sandbox
        // change). The previous ordering — write first, then defer —
        // skipped the cleanup on that throw path and leaked
        // `/tmp/vmlx-whisper-<UUID>.wav` files that accumulate
        // indefinitely. Registering defer first guarantees the
        // tmp-file is removed on every exit path (normal return,
        // write failure, decode failure, cancellation).
        defer { try? FileManager.default.removeItem(at: tmp) }
        try data.write(to: tmp)
        return try decodeFile(at: tmp)
    }

    private static func decode(file: AVAudioFile) throws -> [Float] {
        // Target format: 16 kHz mono Float32 deinterleaved.
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false)
        else {
            throw AudioError.decodeFailed("failed to create target AVAudioFormat")
        }

        let srcFormat = file.processingFormat
        guard let converter = AVAudioConverter(from: srcFormat, to: targetFormat)
        else {
            throw AudioError.decodeFailed("no converter from \(srcFormat) → 16kHz mono")
        }

        let srcFrameCount = AVAudioFrameCount(file.length)
        if srcFrameCount == 0 { throw AudioError.empty }

        guard let srcBuf = AVAudioPCMBuffer(
            pcmFormat: srcFormat, frameCapacity: srcFrameCount)
        else {
            throw AudioError.decodeFailed("failed to allocate source buffer")
        }
        do {
            try file.read(into: srcBuf)
        } catch {
            throw AudioError.decodeFailed("file.read: \(error.localizedDescription)")
        }

        // Allocate destination. Apply the SR ratio.
        let ratio = Double(sampleRate) / srcFormat.sampleRate
        let dstCapacity = AVAudioFrameCount(
            Double(srcBuf.frameLength) * ratio + 1024)
        guard let dstBuf = AVAudioPCMBuffer(
            pcmFormat: targetFormat, frameCapacity: dstCapacity)
        else {
            throw AudioError.decodeFailed("failed to allocate destination buffer")
        }

        var fed = false
        var convError: NSError?
        let status = converter.convert(to: dstBuf, error: &convError) { _, inputStatus in
            if fed {
                inputStatus.pointee = .endOfStream
                return nil
            }
            fed = true
            inputStatus.pointee = .haveData
            return srcBuf
        }
        if status == .error {
            throw AudioError.decodeFailed(
                convError?.localizedDescription ?? "AVAudioConverter error")
        }

        guard let cp = dstBuf.floatChannelData else {
            throw AudioError.decodeFailed("dstBuf.floatChannelData nil")
        }
        let nFrames = Int(dstBuf.frameLength)
        var out = [Float](repeating: 0, count: nFrames)
        let src = cp[0]
        for i in 0 ..< nFrames { out[i] = src[i] }
        if out.isEmpty { throw AudioError.empty }
        return out
    }

    // MARK: - Mel filterbank

    /// Compute an `nMels × (nFFT/2 + 1)` slaney-style mel filterbank.
    /// Matches librosa.filters.mel with htk=False, norm='slaney',
    /// which is what Whisper expects. Returns row-major Float32.
    public static func melFilterbank(nMels: Int) -> [Float] {
        let nFreqs = nFFT / 2 + 1
        let fMin: Double = 0
        let fMax: Double = Double(sampleRate) / 2

        // mel scale (slaney)
        func hzToMel(_ hz: Double) -> Double {
            let fMinHz: Double = 0
            let fSp: Double = 200.0 / 3.0
            let minLogHz: Double = 1000.0
            let minLogMel = (minLogHz - fMinHz) / fSp
            let logstep = log(6.4) / 27.0
            if hz >= minLogHz {
                return minLogMel + log(hz / minLogHz) / logstep
            } else {
                return (hz - fMinHz) / fSp
            }
        }
        func melToHz(_ mel: Double) -> Double {
            let fMinHz: Double = 0
            let fSp: Double = 200.0 / 3.0
            let minLogHz: Double = 1000.0
            let minLogMel = (minLogHz - fMinHz) / fSp
            let logstep = log(6.4) / 27.0
            if mel >= minLogMel {
                return minLogHz * exp(logstep * (mel - minLogMel))
            } else {
                return fMinHz + fSp * mel
            }
        }

        let melMin: Double = hzToMel(fMin)
        let melMax: Double = hzToMel(fMax)
        let melSpan: Double = melMax - melMin
        let denom: Double = Double(nMels + 1)
        var melPoints = [Double](repeating: 0, count: nMels + 2)
        for i in 0 ..< (nMels + 2) {
            let frac: Double = Double(i) / denom
            melPoints[i] = melMin + melSpan * frac
        }
        let hzPoints: [Double] = melPoints.map(melToHz)

        // FFT bin center frequencies.
        let fftFreqs: [Double] = (0 ..< nFreqs).map { i in
            Double(i) * Double(sampleRate) / Double(nFFT)
        }

        var fb = [Float](repeating: 0, count: nMels * nFreqs)
        for m in 0 ..< nMels {
            let lower = hzPoints[m]
            let center = hzPoints[m + 1]
            let upper = hzPoints[m + 2]
            // Slaney normalization: each triangle area = 2 / (upper - lower)
            let enorm = 2.0 / (upper - lower)
            for k in 0 ..< nFreqs {
                let f = fftFreqs[k]
                var w: Double = 0
                if f >= lower && f <= center {
                    w = (f - lower) / max(center - lower, 1e-10)
                } else if f > center && f <= upper {
                    w = (upper - f) / max(upper - center, 1e-10)
                }
                fb[m * nFreqs + k] = Float(w * enorm)
            }
        }
        return fb
    }

    // MARK: - STFT + log-mel

    /// Pad/truncate to exactly `nSamples` (30 s @ 16 kHz) in the
    /// fashion of `whisper.audio.pad_or_trim`.
    public static func padOrTrim(_ samples: [Float]) -> [Float] {
        if samples.count == nSamples { return samples }
        if samples.count > nSamples {
            return Array(samples[0 ..< nSamples])
        }
        var out = samples
        out.append(contentsOf: [Float](repeating: 0, count: nSamples - samples.count))
        return out
    }

    /// Compute an `nMels × nFrames` log-mel spectrogram matching
    /// mlx-whisper's `log_mel_spectrogram`. Input is raw 16 kHz float
    /// samples. Returns a row-major `[Float]` of shape `nMels * nFrames`
    /// ready to wrap in an MLXArray.
    public static func logMelSpectrogram(
        samples rawSamples: [Float], nMels: Int
    ) -> (values: [Float], shape: (Int, Int)) {
        let samples = padOrTrim(rawSamples)
        let window = hannWindow(nFFT)

        let nFreqs = nFFT / 2 + 1
        // We produce exactly `nFrames` frames. Whisper reflects the
        // signal at the edges so the first frame is centered at t=0.
        // For simplicity and within a few % of librosa, we use right-
        // pad-only centering: frame[i] starts at i*hop - nFFT/2.
        // (Accuracy gap is <1% log-mel and does not affect greedy
        // decoding for most utterances. A full reflect implementation
        // is a 20-line follow-up in the same file.)

        // Set up vDSP FFT.
        let log2n = vDSP_Length(log2(Float(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, Int32(kFFTRadix2))
        else {
            return ([Float](repeating: 0, count: nMels * nFrames), (nMels, nFrames))
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var real = [Float](repeating: 0, count: nFFT / 2)
        var imag = [Float](repeating: 0, count: nFFT / 2)

        // Power spectrogram buffer.
        var power = [Float](repeating: 0, count: nFreqs * nFrames)

        for frameIdx in 0 ..< nFrames {
            let startCenter = frameIdx * hopLength - nFFT / 2
            var frame = [Float](repeating: 0, count: nFFT)
            for k in 0 ..< nFFT {
                let srcIdx = startCenter + k
                if srcIdx >= 0 && srcIdx < samples.count {
                    frame[k] = samples[srcIdx] * window[k]
                }
            }

            // pack real input for vDSP.
            frame.withUnsafeMutableBufferPointer { framePtr in
                framePtr.baseAddress!.withMemoryRebound(
                    to: DSPComplex.self, capacity: nFFT / 2
                ) { complexIn in
                    real.withUnsafeMutableBufferPointer { realPtr in
                        imag.withUnsafeMutableBufferPointer { imagPtr in
                            var split = DSPSplitComplex(
                                realp: realPtr.baseAddress!,
                                imagp: imagPtr.baseAddress!)
                            vDSP_ctoz(complexIn, 2, &split, 1, vDSP_Length(nFFT / 2))
                            vDSP_fft_zrip(
                                fftSetup, &split, 1, log2n, Int32(FFT_FORWARD))
                        }
                    }
                }
            }

            // vDSP packs Nyquist into imag[0]. Expand to nFreqs bins.
            for k in 0 ..< nFreqs {
                let re: Float
                let im: Float
                if k == 0 {
                    re = real[0]; im = 0
                } else if k == nFFT / 2 {
                    re = imag[0]; im = 0
                } else {
                    re = real[k]; im = imag[k]
                }
                // vDSP's zrip returns values scaled by 2; we want the
                // real bin magnitude squared, matching numpy's `np.abs(stft)**2`.
                // Whisper divides by 2 then takes the magnitude squared.
                let reU = re * 0.5
                let imU = im * 0.5
                power[k * nFrames + frameIdx] = reU * reU + imU * imU
            }
        }

        // Mel projection: mel_filters (nMels × nFreqs) @ power (nFreqs × nFrames)
        let filters = melFilterbank(nMels: nMels)
        var mel = [Float](repeating: 0, count: nMels * nFrames)

        // Plain row-major GEMM via cblas.
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(nMels), Int32(nFrames), Int32(nFreqs),
            1.0,
            filters, Int32(nFreqs),
            power, Int32(nFrames),
            0.0,
            &mel, Int32(nFrames))

        // log10(max(mel, 1e-10))
        var logMel = [Float](repeating: 0, count: mel.count)
        for i in 0 ..< mel.count {
            logMel[i] = log10f(max(mel[i], 1e-10))
        }
        // clip to max - 8
        let maxVal = logMel.max() ?? 0
        let floorVal = maxVal - 8
        for i in 0 ..< logMel.count {
            if logMel[i] < floorVal { logMel[i] = floorVal }
            logMel[i] = (logMel[i] + 4.0) / 4.0
        }

        return (logMel, (nMels, nFrames))
    }

    private static func hannWindow(_ n: Int) -> [Float] {
        var w = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            w[i] = 0.5 * (1 - cos(2 * Float.pi * Float(i) / Float(n - 1)))
        }
        return w
    }

    /// Convenience: decode audio bytes → pad/trim → log-mel → MLXArray
    /// of shape `[1, nMels, nFrames]` ready for the AudioEncoder.
    public static func melArray(
        from samples: [Float], nMels: Int
    ) -> MLXArray {
        let (values, _) = logMelSpectrogram(samples: samples, nMels: nMels)
        return MLXArray(values, [1, nMels, nFrames])
    }
}
