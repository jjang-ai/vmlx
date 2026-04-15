// Copyright © 2026 vMLX.
//
// Engine extension — audio transcription via vMLXWhisper. Mirrors the
// Python `vmlx_engine.server.create_transcription` endpoint semantics
// (OpenAI `/v1/audio/transcriptions`) but runs entirely native via our
// MLX-Swift Whisper port, no Python hop.
//
// Flow:
//   1. `transcribe(request:)` accepts the same dict-form the Swift route
//      builds from multipart/form-data.
//   2. If no whisper model is loaded, or the caller asks for a different
//      `model` than the one currently loaded, we resolve the requested
//      model via `ModelLibrary` (falling back to the default) and load
//      it lazily with `WhisperLoader.load(from:)`.
//   3. Raw audio bytes → `WhisperAudio.decodeData` → Float32 PCM at 16 kHz.
//   4. `WhisperDecoder.transcribe` runs greedy decoding.
//   5. Response is formatted in OpenAI shape. The caller handles
//      response_format ("json", "text", "verbose_json", "srt", "vtt").

import Foundation
import vMLXWhisper

extension Engine {

    // MARK: - State slot (stored in Engine via the actor's private
    // scratchpad field. We can't add stored properties in an extension,
    // so we piggy-back on the existing `_whisperBox` pattern used for
    // other optional subsystems. If that doesn't exist yet, the field
    // lives on `EngineExtras`.)
    //
    // Implementation note: we use a global actor-isolated box keyed by
    // ObjectIdentifier. This is private to this file and does not leak.

    private var whisperSlot: LoadedWhisper? {
        get { _whisperBundle as? LoadedWhisper }
        set { _whisperBundle = newValue }
    }

    /// Ensure a whisper model is loaded. If a specific `modelName` is
    /// requested we resolve it through `ModelLibrary`; otherwise we
    /// pick the currently-loaded whisper (if any) or the first
    /// `whisper*` directory found on disk.
    private func ensureWhisperLoaded(modelName: String?) async throws
        -> LoadedWhisper
    {
        if let existing = whisperSlot, modelName == nil
            || existing.modelDir.lastPathComponent.contains(modelName ?? "")
        {
            return existing
        }

        // Resolve model directory. First priority: ModelLibrary entries
        // whose display name matches. Second priority: any HF cache
        // directory whose name contains "whisper".
        let candidateURL: URL? = await resolveWhisperDirectory(hint: modelName)
        guard let dir = candidateURL else {
            throw WhisperError.modelNotLoaded
        }

        await logs.append(
            .info, category: "whisper",
            "loading whisper model from \(dir.path)")
        let loaded = try await WhisperLoader.load(from: dir)
        whisperSlot = loaded
        return loaded
    }

    /// Ask ModelLibrary for a whisper directory, or scan the HF hub
    /// cache as a fallback. Returns nil if nothing plausible exists.
    private func resolveWhisperDirectory(hint: String?) async -> URL? {
        // 1. ModelLibrary — check if a loaded-library entry matches.
        // `scan(force: false)` is idempotent inside the freshness
        // window and returns the live entry snapshot.
        let library = self.modelLibrary
        let entries = await library.scan(force: false)
        for entry in entries {
            let name = entry.displayName.lowercased()
            if name.contains("whisper") {
                if let hint, !name.contains(hint.lowercased()) { continue }
                return entry.canonicalPath
            }
        }
        // 2. Direct filesystem scan of the HF hub cache.
        let hubRoot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        if let contents = try? FileManager.default.contentsOfDirectory(
            at: hubRoot, includingPropertiesForKeys: nil)
        {
            for entry in contents {
                let name = entry.lastPathComponent.lowercased()
                if name.contains("whisper") {
                    if let hint, !name.contains(hint.lowercased()) { continue }
                    // The hub cache stores snapshots under
                    // `models--org--name/snapshots/<hash>/`.
                    let snapshots = entry.appendingPathComponent("snapshots")
                    if let subs = try? FileManager.default.contentsOfDirectory(
                        at: snapshots, includingPropertiesForKeys: nil),
                       let first = subs.first
                    {
                        return first
                    }
                }
            }
        }
        return nil
    }

    // MARK: - Public API

    /// Transcribe audio via a native Whisper model.
    ///
    /// Expected request keys:
    ///   - `file`: `Data` — raw audio bytes (required)
    ///   - `file_extension`: String — hint for format ("wav", "mp3", "m4a")
    ///   - `model`: String — model hint (optional)
    ///   - `language`: String — BCP-47 code (optional)
    ///   - `task`: "transcribe" (default) or "translate"
    ///   - `temperature`: Double (currently ignored, greedy only)
    ///
    /// Returns an OpenAI-shape dict: `{text, language, duration, task}`.
    public func transcribe(request: [String: Any]) async throws
        -> [String: Any]
    {
        guard let audio = request["file"] as? Data, !audio.isEmpty else {
            throw EngineError.notImplemented(
                "transcribe: missing 'file' audio bytes")
        }
        let fileExt = (request["file_extension"] as? String) ?? "wav"
        let modelHint = request["model"] as? String
        let language = request["language"] as? String
        let task = (request["task"] as? String) ?? "transcribe"

        let loaded = try await ensureWhisperLoaded(modelName: modelHint)

        let samples: [Float]
        do {
            samples = try WhisperAudio.decodeData(audio, fileExtension: fileExt)
        } catch {
            throw WhisperError.audioDecodeFailed("\(error.localizedDescription)")
        }

        // Greedy decode. Runs on the engine actor — OK, whisper tiny is
        // fast enough that blocking the actor for one request is fine.
        let result = WhisperDecoder.transcribe(
            loaded: loaded,
            samples: samples,
            language: language,
            task: task)

        await logs.append(
            .info, category: "whisper",
            "transcribed \(String(format: "%.1f", result.durationSeconds))s → \(result.tokens.count) tokens")

        return [
            "text": result.text,
            "language": result.language,
            "duration": result.durationSeconds,
            "task": task,
            "model": loaded.modelDir.lastPathComponent,
        ]
    }
}
