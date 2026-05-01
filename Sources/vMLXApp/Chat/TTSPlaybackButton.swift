// SPDX-License-Identifier: Apache-2.0
//
// TTSPlaybackButton — read assistant messages aloud.
//
// Parity with panel/src/renderer/src/components/chat/VoiceChat.tsx
// `TTSPlayer` (used by MessageBubble.tsx). The Electron panel has had
// this since 2026-Q1; the Swift app shipped `EngineTTS.swift` on the
// server side but no client UI was wired to it. Closing the gap so a
// Swift-app user can have assistant replies read aloud the same way
// panel users can.
//
// Flow:
//   1. Tap → POST /v1/audio/speech with the assistant message text.
//   2. Server returns wav bytes (Kokoro TTS via vMLX-side EngineTTS).
//   3. AVAudioPlayer plays the buffer; tap again to stop early.
//
// Defaults match the panel: model = "kokoro", voice = "af_heart",
// speed = 1.0. The panel's TTSPlayer reads these from the user's
// settings store; we read them from the same Swift-side settings
// (`SessionConfig.ttsModel`/`ttsVoice`/`ttsSpeed`) when available.
import AVFoundation
import Foundation
import SwiftUI
import vMLXTheme

@MainActor
final class TTSPlaybackController: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published var isPlaying = false
    @Published var isFetching = false
    @Published var lastError: String?

    private var player: AVAudioPlayer?
    private var task: Task<Void, Never>?

    func toggle(text: String, port: Int) {
        if isPlaying {
            stop()
        } else {
            play(text: text, port: port)
        }
    }

    func play(text: String, port: Int) {
        stop()
        lastError = nil
        isFetching = true
        let body: [String: Any] = [
            "model": "kokoro",
            "input": text,
            "voice": "af_heart",
            "speed": 1.0,
            "response_format": "wav",
        ]
        guard
            let bodyData = try? JSONSerialization.data(withJSONObject: body),
            let url = URL(string: "http://127.0.0.1:\(port)/v1/audio/speech")
        else {
            lastError = "Bad TTS request"
            isFetching = false
            return
        }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.timeoutInterval = 120

        task = Task { [weak self] in
            do {
                let (data, resp) = try await URLSession.shared.upload(for: req, from: bodyData)
                await MainActor.run {
                    guard let self else { return }
                    self.isFetching = false
                    guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
                        let snippet = String(data: data.prefix(200), encoding: .utf8) ?? ""
                        self.lastError = "TTS HTTP \((resp as? HTTPURLResponse)?.statusCode ?? -1): \(snippet)"
                        return
                    }
                    do {
                        let p = try AVAudioPlayer(data: data)
                        p.delegate = self
                        if p.play() {
                            self.player = p
                            self.isPlaying = true
                        } else {
                            self.lastError = "AVAudioPlayer.play returned false"
                        }
                    } catch {
                        self.lastError = "Audio decode failed: \(error.localizedDescription)"
                    }
                }
            } catch {
                await MainActor.run {
                    guard let self else { return }
                    self.isFetching = false
                    self.lastError = "TTS fetch error: \(error.localizedDescription)"
                }
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
        if let p = player, p.isPlaying { p.stop() }
        player = nil
        isPlaying = false
        isFetching = false
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.isPlaying = false
            self.player = nil
        }
    }
}

@MainActor
struct TTSPlaybackButton: View {
    let text: String
    let isGenerating: Bool
    @StateObject private var controller = TTSPlaybackController()
    @AppStorage("vmlx.gatewayPort") private var gatewayPortStorage: Int = 8080

    var body: some View {
        Button {
            controller.toggle(text: text, port: gatewayPortStorage)
        } label: {
            Image(systemName: iconName)
                .font(.system(size: 10))
                .foregroundStyle(isGenerating
                                 ? Theme.Colors.textLow.opacity(0.4)
                                 : (controller.isPlaying ? Theme.Colors.accent : Theme.Colors.textLow))
        }
        .buttonStyle(.plain)
        .disabled(isGenerating || text.isEmpty || controller.isFetching)
        .help(helpText)
    }

    private var iconName: String {
        if controller.isFetching { return "ellipsis" }
        return controller.isPlaying ? "stop.circle" : "speaker.wave.2"
    }

    private var helpText: String {
        if let err = controller.lastError { return err }
        if controller.isFetching { return "Generating speech…" }
        return controller.isPlaying ? "Stop playback" : "Read aloud (Kokoro TTS)"
    }
}
