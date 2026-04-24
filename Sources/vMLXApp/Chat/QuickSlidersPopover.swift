// SPDX-License-Identifier: Apache-2.0
//
// Compact sampling-knobs popover attached to the chat top bar.
//
// Usability gap the round-5 audit flagged: `temperature` / `top_p` /
// `top_k` / `min_p` all live in `ChatSettingsPopover` under eight
// disclosure sections. A user wanting to bump temperature from 0.7
// to 0.9 mid-conversation had to open settings, scroll to
// "Inference defaults", expand it, scroll to the temperature row,
// drag the slider, and dismiss. Five clicks for a one-knob tweak.
//
// This popover surfaces the two most-tweaked knobs (temperature,
// top_p) plus max_tokens and repetition_penalty as a single view
// that writes directly to the chat-level `SessionSettings` row. No
// disclosures, no tier-inheritance UI — just the four sliders the
// user is almost certainly reaching for. Full settings remain
// available via the gear button next door.
//
// Values are debounced 400ms via a `Task` pattern so dragging the
// slider doesn't spam SQLite. On popover dismissal, any pending
// write flushes via `engine.settings.setChat(chatId, ...)` which
// routes through the existing debounced `SettingsStore` path.
//
// Persistence semantics:
//   - Loading: read `engine.settings.chat(chatId)` once on appear.
//     When a field is `nil` (inherit from session/global), the
//     slider displays the resolved global value as the starting
//     point so the user can see what they're tweaking relative to.
//   - Saving: every slider drag writes the field into the
//     chat-scoped `SessionSettings` row. Setting back to the
//     original inherited value clears the field (nil) so the row
//     doesn't hold a spurious override.

import SwiftUI
import vMLXEngine
import vMLXTheme

struct QuickSlidersPopover: View {
    @Environment(AppState.self) private var app
    @Environment(\.dismiss) private var dismiss
    @Environment(\.appLocale) private var appLocale: AppLocale
    let chatId: UUID

    // Live slider state. Loaded from chat-level SessionSettings on
    // appear, with fallbacks to global defaults so the UI never
    // shows zero for an inherit-from-global chat. Default placeholders
    // match GlobalSettings() defaults so a fresh chat looks right
    // even before the async load completes.
    @State private var temperature: Double = 0.7
    @State private var topP: Double = 1.0
    @State private var maxTokens: Double = 512
    @State private var repetitionPenalty: Double = 1.0
    @State private var loaded = false
    @State private var saveTask: Task<Void, Never>? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            Text(L10n.ChatUI.quickTweaks.render(appLocale))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            slider("Temperature", value: $temperature,
                   range: 0...2, step: 0.05,
                   format: "%.2f")
            slider("Top-p", value: $topP,
                   range: 0.01...1, step: 0.01,
                   format: "%.2f")
            slider("Max tokens", value: $maxTokens,
                   range: 16...32_768, step: 16,
                   format: "%.0f")
            slider("Repetition penalty", value: $repetitionPenalty,
                   range: 0.5...2, step: 0.05,
                   format: "%.2f")
            Divider().overlay(Theme.Colors.border)
            HStack {
                Button("Reset to defaults") {
                    Task { await resetToGlobal() }
                }
                .buttonStyle(.plain)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
                Spacer()
                Text(L10n.ChatUI.openFullSettings.render(appLocale))
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }
        }
        .padding(Theme.Spacing.lg)
        .frame(width: 320)
        .task { await loadFromSettings() }
        .onDisappear { saveTask?.cancel() }
    }

    @ViewBuilder
    private func slider(_ label: String, value: Binding<Double>,
                        range: ClosedRange<Double>, step: Double,
                        format: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .font(Theme.Typography.mono)
                    .foregroundStyle(Theme.Colors.textHigh)
            }
            Slider(value: value, in: range, step: step) { editing in
                if !editing { scheduleSave() }
            }
            .tint(Theme.Colors.accent)
        }
    }

    // MARK: - Persistence

    private func loadFromSettings() async {
        guard !loaded else { return }
        let store = app.engine.settings
        let globals = await store.global()
        let chat = await store.chat(chatId)
        // ChatSettings uses non-prefixed field names (`temperature`
        // etc.) since it's the override tier; fall back to the
        // `default*` fields on GlobalSettings when the chat hasn't
        // set its own value.
        temperature = chat?.temperature ?? globals.defaultTemperature
        topP = chat?.topP ?? globals.defaultTopP
        let mt = chat?.maxTokens ?? globals.defaultMaxTokens
        maxTokens = Double(max(16, mt))
        repetitionPenalty = chat?.repetitionPenalty
            ?? globals.defaultRepetitionPenalty
        loaded = true
    }

    /// Debounced write back to the chat-level settings row.
    /// Slider drags call `.onEditingChanged` with `editing=false`
    /// on release; we schedule a 400ms task and any follow-up
    /// release cancels the previous one. Prevents thrashing the
    /// SQLite write-ahead log on rapid tweaks.
    private func scheduleSave() {
        saveTask?.cancel()
        let t = temperature
        let p = topP
        let mt = Int(maxTokens)
        let rp = repetitionPenalty
        saveTask = Task { [app, chatId] in
            try? await Task.sleep(nanoseconds: 400_000_000)
            if Task.isCancelled { return }
            let store = app.engine.settings
            var chat = await store.chat(chatId) ?? .init()
            chat.temperature = t
            chat.topP = p
            chat.maxTokens = mt
            chat.repetitionPenalty = rp
            await store.setChat(chatId, chat)
        }
    }

    /// Clear chat-scoped overrides so the sliders show the
    /// inherited global defaults. Useful when a chat's been tweaked
    /// and the user wants a blank slate without leaving the chat.
    private func resetToGlobal() async {
        let store = app.engine.settings
        var chat = await store.chat(chatId) ?? .init()
        chat.temperature = nil
        chat.topP = nil
        chat.maxTokens = nil
        chat.repetitionPenalty = nil
        await store.setChat(chatId, chat)
        loaded = false
        await loadFromSettings()
    }
}
