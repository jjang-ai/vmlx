// PlaceholderSynth — deterministic non-neural fallback so
// `/v1/audio/speech` returns real audio bytes even before the
// Kokoro port lands.
//
// This is NOT speech synthesis. It produces an envelope-shaped
// tone burst per word at a pitch derived from the word's hash,
// with short silences between words and longer silences at
// punctuation. It exists only so API clients don't get a 501
// and so integration tests can exercise the full request path.
//
// Replace with `KokoroBackend.renderPCM(...)` once the neural
// port is done. The `TTSEngine` handoff is a single call site.

import Foundation

public enum PlaceholderSynth {

    public static let sampleRate: Int = 24_000

    public static func renderPCM(text: String, speed: Double) -> [Int16] {
        let tokens = tokenize(text)
        if tokens.isEmpty { return [] }

        let speed = max(0.25, min(4.0, speed))
        // ~180 ms per word base, scaled by speed.
        let wordMs = Int(180.0 / speed)
        let gapMs = Int(60.0 / speed)
        let punctGapMs = Int(220.0 / speed)

        var out: [Int16] = []
        out.reserveCapacity(tokens.count * (wordMs + gapMs) * sampleRate / 1000)

        for tok in tokens {
            switch tok {
            case .word(let w):
                let freq = 140.0 + Double(stableHash(w) % 220)  // 140-360 Hz
                appendTone(into: &out, freq: freq, ms: wordMs)
                appendSilence(into: &out, ms: gapMs)
            case .punct:
                appendSilence(into: &out, ms: punctGapMs)
            }
        }
        return out
    }

    // MARK: - Tone + silence

    private static func appendTone(into buf: inout [Int16],
                                   freq: Double,
                                   ms: Int) {
        let n = ms * sampleRate / 1000
        if n <= 0 { return }
        let twoPiF = 2.0 * Double.pi * freq / Double(sampleRate)
        let attack = max(1, n / 20)
        let release = max(1, n / 10)
        let amp: Double = 0.18 * Double(Int16.max)
        for i in 0..<n {
            var env: Double = 1.0
            if i < attack { env = Double(i) / Double(attack) }
            else if i > n - release { env = Double(n - i) / Double(release) }
            let s = sin(Double(i) * twoPiF) * amp * env
            buf.append(Int16(max(-32768.0, min(32767.0, s))))
        }
    }

    private static func appendSilence(into buf: inout [Int16], ms: Int) {
        let n = ms * sampleRate / 1000
        if n <= 0 { return }
        buf.append(contentsOf: repeatElement(Int16(0), count: n))
    }

    // MARK: - Tokenization

    private enum Tok {
        case word(String)
        case punct
    }

    private static func tokenize(_ text: String) -> [Tok] {
        var out: [Tok] = []
        var buf = ""
        for ch in text {
            if ch.isLetter || ch.isNumber {
                buf.append(ch)
            } else {
                if !buf.isEmpty { out.append(.word(buf)); buf = "" }
                if ch == "." || ch == "," || ch == "!" || ch == "?" || ch == ";" || ch == ":" {
                    out.append(.punct)
                }
            }
        }
        if !buf.isEmpty { out.append(.word(buf)) }
        return out
    }

    private static func stableHash(_ s: String) -> Int {
        // Deterministic across runs (Swift's String.hashValue is salted).
        var h: UInt64 = 1469598103934665603
        for b in s.utf8 {
            h ^= UInt64(b)
            h &*= 1099511628211
        }
        return Int(h & 0x7FFFFFFF)
    }
}
