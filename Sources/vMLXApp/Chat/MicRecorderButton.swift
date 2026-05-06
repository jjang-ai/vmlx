// SPDX-License-Identifier: Apache-2.0
//
// MicRecorderButton — push-to-talk dictation in the Swift chat composer.
//
// Why: docs/AUDIT-MULTIMODAL-UI.md (2026-04-30) flagged the Swift app as
// having ZERO microphone UI even though `EngineTranscribe.swift` had
// shipped a working /v1/audio/transcriptions HTTP route for weeks. The
// Electron panel has had this since 2026-Q1 (VoiceChat.tsx). Closing
// the parity gap so a Swift-app user can dictate into chat without
// dropping to the panel.
//
// Flow:
//   1. Tap → request mic permission → AVAudioRecorder starts to a temp
//      .wav file (16 kHz mono — matches what WhisperAudio.decodeData
//      expects so we avoid an extra resample hop).
//   2. Tap again → stop recording → POST multipart/form-data
//      `/v1/audio/transcriptions` against the in-process Hummingbird
//      server (same way the Electron panel calls the Python server).
//   3. Append the transcribed text to `vm.inputText` with a leading
//      space, matching VoiceChat.tsx's onTranscription handler.
//
// Permissions: macOS apps need `NSMicrophoneUsageDescription` in
// Info.plist. The CLI binary build of vMLX does not have an Info.plist;
// we surface a graceful "permission denied" message in that case so the
// user knows to either grant access via System Settings → Privacy &
// Security → Microphone or run from a properly-bundled .app.
import AVFoundation
import Foundation
import SwiftUI
import vMLXTheme

@MainActor
struct MicRecorderButton: View {
    @Bindable var vm: ChatViewModel
    let appLocale: AppLocale

    @State private var recorder: AVAudioRecorder?
    @State private var isRecording = false
    @State private var isTranscribing = false
    @State private var lastError: String?

    var body: some View {
        Button(action: toggleRecording) {
            Image(systemName: iconName)
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(iconColor)
                .frame(width: 28, height: 28)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(backgroundColor)
                )
                .overlay(
                    Group {
                        if isTranscribing {
                            ProgressView()
                                .controlSize(.small)
                                .tint(Theme.Colors.accent)
                        }
                    }
                )
        }
        .buttonStyle(.plain)
        .disabled(isTranscribing)
        .accessibilityLabel(isRecording ? "Stop recording" : "Record voice message")
        .accessibilityHint(isRecording
            ? "Stops recording and transcribes via the loaded Whisper model"
            : "Starts a microphone recording. Tap again to stop and transcribe.")
        .help(helpText)
    }

    // MARK: - UI tokens

    private var iconName: String {
        if isTranscribing { return "waveform" }
        return isRecording ? "stop.circle.fill" : "mic.fill"
    }

    private var iconColor: Color {
        isRecording ? Theme.Colors.danger : Theme.Colors.accent
    }

    private var backgroundColor: Color {
        isRecording ? Theme.Colors.danger.opacity(0.18) : Theme.Colors.surfaceHi
    }

    private var helpText: String {
        if let lastError {
            return lastError
        }
        return isRecording
            ? "Stop recording and transcribe"
            : "Record a voice message (uses the loaded Whisper STT model)"
    }

    // MARK: - Recording

    private func toggleRecording() {
        if isRecording { stopAndTranscribe() } else { startRecording() }
    }

    private func startRecording() {
        lastError = nil
        // AVAudioRecorder in 16 kHz mono PCM16 → matches Whisper's expected
        // sample rate so the server side skips a resample hop. Smaller file
        // means faster multipart upload and faster decode, since the in-
        // process Hummingbird endpoint reads the bytes directly into a
        // Float32 PCM array.
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-mic-\(UUID().uuidString).wav")
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        do {
            let r = try AVAudioRecorder(url: temp, settings: settings)
            r.isMeteringEnabled = true
            guard r.prepareToRecord() else {
                lastError = "AVAudioRecorder.prepareToRecord returned false"
                return
            }
            guard r.record() else {
                lastError = "Mic permission denied or recording failed to start"
                return
            }
            recorder = r
            isRecording = true
        } catch {
            lastError = "Failed to start recording: \(error.localizedDescription)"
        }
    }

    private func stopAndTranscribe() {
        guard let r = recorder else { return }
        let url = r.url
        r.stop()
        recorder = nil
        isRecording = false

        Task { await transcribe(url: url) }
    }

    // MARK: - Transcribe

    private func transcribe(url: URL) async {
        isTranscribing = true
        defer { isTranscribing = false }

        do {
            let data = try Data(contentsOf: url)
            // Multipart/form-data POST to the in-process Hummingbird server
            // — mirrors how the Python panel's audio.ts main-process IPC
            // handler builds the request. Server endpoint is registered
            // by `EngineTranscribe.swift` and accepts file= + model=.
            // The gateway port is the user's app-wide setting; we read it
            // from the same global store the rest of the chat path uses.
            // P0 audit fix — must use the BOUND gateway port, not the
            // configured/requested value. Auto-bump (gateway port taken
            // by Ollama, etc.) makes the bound port differ from the
            // configured one; previously this posted into the wrong
            // process and 404'd silently. `gatewayBoundPort()` returns
            // nil when the gateway is .disabled or .failed — bail with
            // a user-actionable message rather than hammer 127.0.0.1
            // on the wrong port and produce "HTTP -1".
            let port: Int = await { @MainActor in vm.gatewayBoundPort() }()
                ?? Int(0)
            guard port > 0 else {
                lastError = "Gateway is not running — enable it in Tray → Server, then retry."
                return
            }
            let endpointURL = URL(string: "http://127.0.0.1:\(port)/v1/audio/transcriptions")!

            let boundary = "Boundary-\(UUID().uuidString)"
            var body = Data()
            func appendField(_ name: String, _ value: String) {
                body.append("--\(boundary)\r\n".data(using: .utf8)!)
                body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
                body.append("\(value)\r\n".data(using: .utf8)!)
            }
            appendField("model", "whisper-large-v3-turbo")
            appendField("response_format", "json")
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"file\"; filename=\"recording.wav\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
            body.append(data)
            body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

            var req = URLRequest(url: endpointURL)
            req.httpMethod = "POST"
            req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
            req.timeoutInterval = 120

            let (respData, response) = try await URLSession.shared.upload(for: req, from: body)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                let snippet = String(data: respData.prefix(200), encoding: .utf8) ?? ""
                lastError = "Transcription failed: HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1) — \(snippet)"
                return
            }
            // OpenAI shape: {"text": "..."}
            if let json = try JSONSerialization.jsonObject(with: respData) as? [String: Any],
               let text = json["text"] as? String {
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    let pad = vm.inputText.isEmpty ? "" : (vm.inputText.hasSuffix(" ") ? "" : " ")
                    vm.inputText += pad + trimmed
                }
            }
            // Best-effort cleanup of the temp recording file.
            try? FileManager.default.removeItem(at: url)
        } catch {
            lastError = "Transcription error: \(error.localizedDescription)"
        }
    }
}
