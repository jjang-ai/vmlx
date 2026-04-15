// vMLXTTS — Text-to-speech module for the vMLX Swift rewrite.
//
// Goal: replace the `/v1/audio/speech` 501 stub with a real WAV-producing
// pipeline that (a) parses the OpenAI request shape, (b) returns real
// audio bytes, and (c) is structured so a neural backend (Kokoro,
// XTTS, Piper) can be slotted in without touching the route layer.
//
// CURRENT STATE
// -------------
// The neural Kokoro port is a non-trivial multi-day effort (StyleTTS
// text encoder + duration predictor + HiFi-GAN vocoder + eSpeak G2P
// OR a bundled English phoneme table). It is scaffolded below but not
// wired. To keep the endpoint from returning 501 we ship a deterministic
// "PlaceholderSynth" backend that produces a real envelope-shaped
// waveform (one short tone burst per word) at 24 kHz / 16-bit PCM.
// The response header `X-vMLX-TTS-Backend: placeholder-tone` makes it
// trivial for clients to detect this is not a neural model.
//
// Once Kokoro (or an alternative) is ported, `KokoroBackend.synthesize`
// replaces the placeholder call in `TTSEngine.synthesize`.

import Foundation

public enum TTSResponseFormat: String, Sendable {
    case wav
    case mp3
    case flac
    case opus
    case pcm

    public var contentType: String {
        switch self {
        case .wav:  return "audio/wav"
        case .mp3:  return "audio/mpeg"
        case .flac: return "audio/flac"
        case .opus: return "audio/ogg"
        case .pcm:  return "audio/pcm"
        }
    }
}

public struct TTSRequest: Sendable {
    public var model: String
    public var input: String
    public var voice: String
    public var format: TTSResponseFormat
    public var speed: Double   // 0.25 ... 4.0, OpenAI spec

    public init(model: String,
                input: String,
                voice: String = "default",
                format: TTSResponseFormat = .wav,
                speed: Double = 1.0) {
        self.model = model
        self.input = input
        self.voice = voice
        self.format = format
        self.speed = max(0.25, min(4.0, speed))
    }
}

public struct TTSResult: Sendable {
    public var audio: Data
    public var contentType: String
    public var backend: String       // "kokoro", "placeholder-tone", ...
    public var sampleRate: Int
    public var durationSec: Double
}

public enum TTSError: Error, CustomStringConvertible {
    case missingInput
    case unsupportedFormat(String)
    case modelNotPorted(String)
    case encoderUnavailable(String)

    public var description: String {
        switch self {
        case .missingInput:
            return "TTS request missing 'input' field"
        case .unsupportedFormat(let f):
            return "TTS response_format '\(f)' not supported; use wav/mp3/flac/opus/pcm"
        case .modelNotPorted(let m):
            return "TTS model '\(m)' neural backend not yet ported; see vMLXTTS/Kokoro/README"
        case .encoderUnavailable(let f):
            return "TTS encoder for format '\(f)' unavailable on this platform"
        }
    }
}

/// Public entry point — parse an OpenAI-shaped request dictionary
/// and produce a `TTSResult` containing raw audio bytes.
public struct TTSEngine: Sendable {

    public init() {}

    public func synthesize(request: [String: Any]) throws -> TTSResult {
        guard let input = request["input"] as? String, !input.isEmpty else {
            throw TTSError.missingInput
        }
        let model = (request["model"] as? String) ?? "vmlx-tts-placeholder"
        let voice = (request["voice"] as? String) ?? "default"
        let fmtRaw = (request["response_format"] as? String) ?? "wav"
        guard let fmt = TTSResponseFormat(rawValue: fmtRaw.lowercased()) else {
            throw TTSError.unsupportedFormat(fmtRaw)
        }
        let speed = (request["speed"] as? Double)
            ?? (request["speed"] as? Int).map(Double.init)
            ?? 1.0

        let req = TTSRequest(model: model,
                             input: input,
                             voice: voice,
                             format: fmt,
                             speed: speed)
        return try synthesize(req)
    }

    public func synthesize(_ req: TTSRequest) throws -> TTSResult {
        // Route to neural backend when it exists; fall back to
        // placeholder until Kokoro lands.
        let pcm = PlaceholderSynth.renderPCM(text: req.input, speed: req.speed)
        let sampleRate = PlaceholderSynth.sampleRate

        let audioData: Data
        let effectiveFormat: TTSResponseFormat
        let backendLabel: String
        switch req.format {
        case .wav:
            audioData = WavEncoder.encode(pcm: pcm, sampleRate: sampleRate)
            effectiveFormat = .wav
            backendLabel = "placeholder-tone"
        case .pcm:
            audioData = WavEncoder.rawPCMLE(pcm)
            effectiveFormat = .pcm
            backendLabel = "placeholder-tone"
        case .mp3, .flac, .opus:
            // Graceful fallback: mp3/flac/opus encoders aren't wired
            // yet (AVFoundation AVAudioConverter port deferred with
            // the rest of the neural TTS backend). Rather than fail
            // the whole request, emit WAV bytes — every browser and
            // ffmpeg-based client transcodes WAV on playback — and
            // tag the backend label so callers who care can detect
            // the degradation. The `effectiveFormat` field carried
            // back to the HTTP route overrides the Content-Type
            // header to `audio/wav` so the response stays honest.
            //
            // This preserves API-level success (the caller gets
            // playable audio instead of a 500) and matches the
            // Python server's behavior on unsupported encoders.
            FileHandle.standardError.write(Data(
                "[vmlx][tts] format=\(req.format.rawValue) not yet encoded natively; falling back to WAV\n".utf8))
            audioData = WavEncoder.encode(pcm: pcm, sampleRate: sampleRate)
            effectiveFormat = .wav
            backendLabel = "placeholder-tone (wav fallback from \(req.format.rawValue))"
        }

        let duration = Double(pcm.count) / Double(sampleRate)
        return TTSResult(audio: audioData,
                         contentType: effectiveFormat.contentType,
                         backend: backendLabel,
                         sampleRate: sampleRate,
                         durationSec: duration)
    }
}
