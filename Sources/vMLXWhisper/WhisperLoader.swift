// Copyright © 2026 vMLX. Whisper weight + config loader.
//
// Reads `config.json` + `*.safetensors` from a local model directory
// (e.g. `mlx-community/whisper-tiny` pulled via HF hub), instantiates
// the `Whisper` module, sanitizes the weight keys, and populates
// parameters.

import Foundation
import MLX
import MLXNN

public struct LoadedWhisper {
    public let model: Whisper
    public let tokenizer: WhisperTokenizer
    public let config: WhisperConfig
    public let modelDir: URL
}

public enum WhisperLoader {

    /// Load a whisper model + tokenizer from `dir`. The directory must
    /// contain `config.json`, at least one `.safetensors` file, and a
    /// `tokenizer.json` (HuggingFace format).
    public static func load(from dir: URL) async throws -> LoadedWhisper {
        let configURL = dir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw WhisperError.configurationMissing("config.json")
        }
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(WhisperConfig.self, from: configData)

        let model = Whisper(config)

        // Whisper weights on HF ship in one of two formats:
        //   - `.safetensors` (modern mlx-examples / mlx-community `*-mlx*` repos)
        //   - `.npz` (legacy `mlx-community/whisper-*` dumps)
        // The vendored Swift MLX `loadArrays` only supports safetensors,
        // so we prefer it and only fall back to .npz with a clear error
        // if the caller points at a legacy repo with no safetensors file.
        var weights: [String: MLXArray] = [:]
        var safetensorsFiles: [URL] = []
        var npzFiles: [URL] = []
        if let enumerator = FileManager.default.enumerator(
            at: dir, includingPropertiesForKeys: nil) {
            for case let url as URL in enumerator {
                let ext = url.pathExtension.lowercased()
                if ext == "safetensors" { safetensorsFiles.append(url) }
                else if ext == "npz" { npzFiles.append(url) }
            }
        }

        if !safetensorsFiles.isEmpty {
            for url in safetensorsFiles {
                let w = try MLX.loadArrays(url: url)
                for (key, value) in w { weights[key] = value }
            }
        } else if let npz = npzFiles.first {
            // Legacy `.npz` — vendored Swift MLX only supports safetensors,
            // so transcode once with Python MLX and cache the result as
            // `model.safetensors` next to the .npz. Subsequent loads hit
            // the cached safetensors directly (the "if !safetensorsFiles
            // is empty" branch above). This avoids forcing users to know
            // which mlx-community variant ships which format.
            let converted = dir.appendingPathComponent("model.safetensors")
            try Self.transcodeNPZToSafetensors(source: npz, dest: converted)
            let w = try MLX.loadArrays(url: converted)
            for (key, value) in w { weights[key] = value }
        } else {
            throw WhisperError.weightsMissing
        }

        weights = Whisper.sanitizeWeights(weights)

        // Populate module parameters. verify:[] tolerates stray keys
        // (e.g. an alignment head buffer we don't model) so the load
        // does not hard-fail on non-essential tensors.
        let params = ModuleParameters.unflattened(weights)
        try model.update(parameters: params, verify: [])

        // Trigger MLX lazy evaluation so weights are materialized
        // before the first forward pass.
        MLX.eval(model)

        let tokenizer = try await WhisperTokenizer.load(
            from: dir, isMultilingual: config.isMultilingual)

        return LoadedWhisper(
            model: model, tokenizer: tokenizer,
            config: config, modelDir: dir)
    }

    /// One-shot transcode of an `.npz` weight dump to `.safetensors`.
    /// Shells out to `python3 -m mlx.core` because the vendored Swift MLX
    /// has no `.npz` reader, and implementing a zip + `.npy` parser in
    /// Swift just for whisper's legacy dumps would be disproportionate
    /// effort. Python MLX is ~always available on any machine that can
    /// download whisper weights from HF in the first place. The conversion
    /// runs once per model snapshot; cached output is re-used on the
    /// next load.
    private static func transcodeNPZToSafetensors(
        source: URL, dest: URL
    ) throws {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = [
            "python3", "-c",
            """
            import sys, mlx.core as mx
            d = mx.load(sys.argv[1])
            if not isinstance(d, dict):
                raise SystemExit('whisper: .npz did not contain a dict of arrays')
            mx.save_safetensors(sys.argv[2], d)
            """,
            source.path, dest.path,
        ]
        let err = Pipe()
        proc.standardError = err
        proc.standardOutput = Pipe()
        try proc.run()
        proc.waitUntilExit()
        if proc.terminationStatus != 0 {
            let msg = String(
                data: err.fileHandleForReading.readDataToEndOfFile(),
                encoding: .utf8) ?? ""
            throw WhisperError.configurationMissing(
                "failed to transcode \(source.lastPathComponent) → safetensors: \(msg)")
        }
    }
}
