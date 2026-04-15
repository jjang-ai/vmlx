// Minimal WAV (RIFF) encoder — 16-bit little-endian PCM, mono.
// Self-contained so vMLXTTS has zero external audio dependencies.

import Foundation

public enum WavEncoder {

    /// Encode a mono 16-bit PCM sample buffer as a WAV file.
    public static func encode(pcm: [Int16], sampleRate: Int) -> Data {
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = UInt32(pcm.count * 2)
        let chunkSize = 36 + dataSize

        var out = Data()
        out.append("RIFF".data(using: .ascii)!)
        out.append(le(chunkSize))
        out.append("WAVE".data(using: .ascii)!)

        out.append("fmt ".data(using: .ascii)!)
        out.append(le(UInt32(16)))              // PCM chunk size
        out.append(le(UInt16(1)))                // PCM format
        out.append(le(numChannels))
        out.append(le(UInt32(sampleRate)))
        out.append(le(byteRate))
        out.append(le(blockAlign))
        out.append(le(bitsPerSample))

        out.append("data".data(using: .ascii)!)
        out.append(le(dataSize))
        out.append(rawPCMLE(pcm))
        return out
    }

    /// Raw little-endian 16-bit PCM payload (no WAV header).
    public static func rawPCMLE(_ pcm: [Int16]) -> Data {
        var bytes = Data(capacity: pcm.count * 2)
        for sample in pcm {
            let u = UInt16(bitPattern: sample)
            bytes.append(UInt8(u & 0xFF))
            bytes.append(UInt8((u >> 8) & 0xFF))
        }
        return bytes
    }

    private static func le(_ v: UInt32) -> Data {
        var x = v.littleEndian
        return Data(bytes: &x, count: 4)
    }
    private static func le(_ v: UInt16) -> Data {
        var x = v.littleEndian
        return Data(bytes: &x, count: 2)
    }
}
