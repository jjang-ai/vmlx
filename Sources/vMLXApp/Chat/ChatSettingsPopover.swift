// SPDX-License-Identifier: Apache-2.0
//
// ChatSettingsPopover — Phase 3 chat settings UI with 4-tier inheritance
// preview. Anchored to a gear button in the chat header (see ChatScreen.swift).
//
// Design goals:
//   1. Every control maps to a ChatSettings field. Writes go through
//      `settings.setChat(chatId, _:)` so they persist via SettingsStore's
//      debounced writer.
//   2. When a field is nil at the chat tier, show "(from session)" /
//      "(from global)" inline and a tiny colored dot — both signals so
//      colorblind users still see the tier.
//   3. Per-field reset button clears only that field to nil.
//      "Reset all" at the bottom clears every chat-tier override.
//   4. Fixed 480x640 frame — everything scrolls inside collapsible sections.
//   5. The "Effective values" footer shows the 4 most-important resolved
//      fields so the user can confirm the stack at a glance.
//
// Safety: the popover snapshots `ChatSettings` into a local `@State draft` on
// open and writes back through `SettingsStore` each time the user changes a
// field. The resolved preview is re-fetched on every mutation so the footer
// and tier chips stay in sync.

import SwiftUI
import vMLXEngine
import vMLXTheme
import AppKit

// MARK: - Popover

struct ChatSettingsPopover: View {
    @Environment(AppState.self) private var app
    let chatId: UUID
    let sessionId: UUID?

    @State private var draft: ChatSettings = ChatSettings()
    @State private var resolved: ResolvedSettings? = nil
    @State private var loaded = false

    // Section collapse flags. Sampling + Reasoning are open by default.
    @State private var openSampling = true
    @State private var openReasoning = true
    @State private var openSystem = false
    @State private var openStops = false
    @State private var openTools = false
    @State private var openBuiltinTools = false
    @State private var openWorkdir = false
    @State private var openMCP = false
    @State private var openWireApi = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().background(Theme.Colors.border)
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                    samplingSection
                    reasoningSection
                    systemPromptSection
                    stopSequencesSection
                    toolsSection
                    builtinToolsSection
                    workdirSection
                    mcpSection
                    wireApiSection
                }
                .padding(Theme.Spacing.md)
            }
            Divider().background(Theme.Colors.border)
            effectiveValuesFooter
        }
        .frame(width: 480, height: 640)
        .background(Theme.Colors.background)
        .task(id: chatId) { await load() }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Image(systemName: "slider.horizontal.3")
                .foregroundStyle(Theme.Colors.accent)
            Text("Chat Settings")
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            Spacer()
            Button("Reset all") { resetAll() }
                .buttonStyle(.borderless)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.danger)
        }
        .padding(.horizontal, Theme.Spacing.md)
        .padding(.vertical, Theme.Spacing.sm)
        .background(Theme.Colors.surface)
    }

    // MARK: - Sections

    private var samplingSection: some View {
        DisclosureGroup(isExpanded: $openSampling) {
            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                doubleRow("Temperature", value: Binding(
                    get: { draft.temperature },
                    set: { draft.temperature = $0; writeBack() }
                ), range: 0...2, step: 0.05, traceKey: "defaultTemperature",
                   fallback: { resolved?.settings.defaultTemperature ?? 0 })
                doubleRow("Top P", value: Binding(
                    get: { draft.topP },
                    set: { draft.topP = $0; writeBack() }
                ), range: 0...1, step: 0.01, traceKey: "defaultTopP",
                   fallback: { resolved?.settings.defaultTopP ?? 0 })
                intRow("Top K", value: Binding(
                    get: { draft.topK },
                    set: { draft.topK = $0; writeBack() }
                ), range: 0...200, traceKey: "defaultTopK",
                   fallback: { resolved?.settings.defaultTopK ?? 0 })
                doubleRow("Min P", value: Binding(
                    get: { draft.minP },
                    set: { draft.minP = $0; writeBack() }
                ), range: 0...1, step: 0.01, traceKey: "defaultMinP",
                   fallback: { resolved?.settings.defaultMinP ?? 0 })
                doubleRow("Repetition Penalty", value: Binding(
                    get: { draft.repetitionPenalty },
                    set: { draft.repetitionPenalty = $0; writeBack() }
                ), range: 0.5...2.0, step: 0.01, traceKey: "defaultRepetitionPenalty",
                   fallback: { resolved?.settings.defaultRepetitionPenalty ?? 0 })
                intRow("Max Tokens", value: Binding(
                    get: { draft.maxTokens },
                    set: { draft.maxTokens = $0; writeBack() }
                ), range: 1...131072, traceKey: "defaultMaxTokens",
                   fallback: { resolved?.settings.defaultMaxTokens ?? 0 })
            }
            .padding(.top, Theme.Spacing.xs)
        } label: { sectionLabel("Sampling — this chat (overrides global)") }
        .tint(Theme.Colors.textMid)
    }

    private var reasoningSection: some View {
        DisclosureGroup(isExpanded: $openReasoning) {
            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                // enableThinking tri-state
                HStack {
                    Text("Enable Thinking").font(Theme.Typography.body)
                        .foregroundStyle(Theme.Colors.textHigh)
                    Spacer()
                    Picker("", selection: Binding<Int>(
                        get: {
                            if draft.enableThinking == nil { return 0 }
                            return draft.enableThinking! ? 1 : 2
                        },
                        set: { v in
                            draft.enableThinking = (v == 0) ? nil : (v == 1)
                            writeBack()
                        }
                    )) {
                        Text("Inherit").tag(0)
                        Text("On").tag(1)
                        Text("Off").tag(2)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 180)
                }
                tierChipRow(traceKey: "defaultEnableThinking",
                            isChatSet: draft.enableThinking != nil) {
                    draft.enableThinking = nil; writeBack()
                }

                HStack {
                    Text("Reasoning Effort").font(Theme.Typography.body)
                        .foregroundStyle(Theme.Colors.textHigh)
                    Spacer()
                    Picker("", selection: Binding<String>(
                        get: { draft.reasoningEffort ?? "__inherit__" },
                        set: { v in
                            draft.reasoningEffort = (v == "__inherit__") ? nil : v
                            writeBack()
                        }
                    )) {
                        Text("Inherit").tag("__inherit__")
                        Text("None").tag("none")
                        Text("Low").tag("low")
                        Text("Medium").tag("medium")
                        Text("High").tag("high")
                    }
                    .pickerStyle(.menu)
                    .frame(width: 180)
                }
                tierChipRow(traceKey: "defaultReasoningParser",
                            isChatSet: draft.reasoningEffort != nil) {
                    draft.reasoningEffort = nil; writeBack()
                }
            }
            .padding(.top, Theme.Spacing.xs)
        } label: { sectionLabel("Reasoning") }
        .tint(Theme.Colors.textMid)
    }

    private var systemPromptSection: some View {
        DisclosureGroup(isExpanded: $openSystem) {
            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                TextEditor(text: Binding(
                    get: { draft.systemPrompt ?? "" },
                    set: { v in
                        draft.systemPrompt = v.isEmpty ? nil : v
                        writeBack()
                    }
                ))
                .font(Theme.Typography.mono)
                .frame(minHeight: 100, maxHeight: 160)
                .padding(Theme.Spacing.xs)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.surface)
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.md)
                                .stroke(Theme.Colors.border, lineWidth: 1)
                        )
                )
                tierChipRow(traceKey: "defaultSystemPrompt",
                            isChatSet: draft.systemPrompt != nil) {
                    draft.systemPrompt = nil; writeBack()
                }
            }
            .padding(.top, Theme.Spacing.xs)
        } label: { sectionLabel("System Prompt") }
        .tint(Theme.Colors.textMid)
    }

    private var stopSequencesSection: some View {
        DisclosureGroup(isExpanded: $openStops) {
            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                let stops = draft.stopSequences ?? []
                ForEach(Array(stops.enumerated()), id: \.offset) { idx, seq in
                    HStack {
                        TextField("stop", text: Binding(
                            get: { seq },
                            set: { v in
                                var arr = draft.stopSequences ?? []
                                if idx < arr.count { arr[idx] = v }
                                draft.stopSequences = arr.isEmpty ? nil : arr
                                writeBack()
                            }
                        ))
                        .textFieldStyle(.roundedBorder)
                        .font(Theme.Typography.mono)
                        Button {
                            var arr = draft.stopSequences ?? []
                            if idx < arr.count { arr.remove(at: idx) }
                            draft.stopSequences = arr.isEmpty ? nil : arr
                            writeBack()
                        } label: {
                            Image(systemName: "minus.circle")
                                .foregroundStyle(Theme.Colors.danger)
                        }
                        .buttonStyle(.borderless)
                    }
                }
                Button {
                    var arr = draft.stopSequences ?? []
                    arr.append("")
                    draft.stopSequences = arr
                    writeBack()
                } label: {
                    Label("Add stop", systemImage: "plus.circle")
                        .font(Theme.Typography.caption)
                }
                .buttonStyle(.borderless)
                .foregroundStyle(Theme.Colors.accent)
                tierChipRow(traceKey: "stopSequences",
                            isChatSet: draft.stopSequences != nil) {
                    draft.stopSequences = nil; writeBack()
                }
            }
            .padding(.top, Theme.Spacing.xs)
        } label: { sectionLabel("Stop Sequences") }
        .tint(Theme.Colors.textMid)
    }

    private var toolsSection: some View {
        DisclosureGroup(isExpanded: $openTools) {
            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                HStack {
                    Text("Tool Choice").font(Theme.Typography.body)
                        .foregroundStyle(Theme.Colors.textHigh)
                    Spacer()
                    Picker("", selection: Binding<String>(
                        get: { draft.toolChoice ?? "__inherit__" },
                        set: { v in
                            draft.toolChoice = (v == "__inherit__") ? nil : v
                            writeBack()
                        }
                    )) {
                        Text("Inherit").tag("__inherit__")
                        Text("Auto").tag("auto")
                        Text("None").tag("none")
                        Text("Required").tag("required")
                    }
                    .pickerStyle(.menu)
                    .frame(width: 180)
                }
                tierChipRow(traceKey: "toolChoice",
                            isChatSet: draft.toolChoice != nil) {
                    draft.toolChoice = nil; writeBack()
                }
                intRow("Max Tool Iterations", value: Binding(
                    get: { draft.maxToolIterations },
                    set: { draft.maxToolIterations = $0; writeBack() }
                ), range: 1...32, traceKey: "maxToolIterations",
                   fallback: { 8 })
                toggleRow("Built-in Tools Enabled",
                          value: Binding(
                            get: { draft.builtinToolsEnabled },
                            set: { draft.builtinToolsEnabled = $0; writeBack() }
                          ),
                          traceKey: "builtinToolsEnabled")
                toggleRow("Hide Tool Status",
                          value: Binding(
                            get: { draft.hideToolStatus },
                            set: { draft.hideToolStatus = $0; writeBack() }
                          ),
                          traceKey: "hideToolStatusDefault")
            }
            .padding(.top, Theme.Spacing.xs)
        } label: { sectionLabel("Tools") }
        .tint(Theme.Colors.textMid)
    }

    private var builtinToolsSection: some View {
        DisclosureGroup(isExpanded: $openBuiltinTools) {
            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                toggleRow("Web Search", value: Binding(
                    get: { draft.webSearchEnabled },
                    set: { draft.webSearchEnabled = $0; writeBack() }
                ), traceKey: "webSearchEnabled")
                toggleRow("Fetch URL", value: Binding(
                    get: { draft.fetchUrlEnabled },
                    set: { draft.fetchUrlEnabled = $0; writeBack() }
                ), traceKey: "fetchUrlEnabled")
                toggleRow("File Tools", value: Binding(
                    get: { draft.fileToolsEnabled },
                    set: { draft.fileToolsEnabled = $0; writeBack() }
                ), traceKey: "fileToolsEnabled")
                toggleRow("Search Tools", value: Binding(
                    get: { draft.searchToolsEnabled },
                    set: { draft.searchToolsEnabled = $0; writeBack() }
                ), traceKey: "searchToolsEnabled")
                toggleRow("Shell", value: Binding(
                    get: { draft.shellEnabled },
                    set: { draft.shellEnabled = $0; writeBack() }
                ), traceKey: "shellEnabled")
                toggleRow("Git", value: Binding(
                    get: { draft.gitEnabled },
                    set: { draft.gitEnabled = $0; writeBack() }
                ), traceKey: "gitEnabled")
                toggleRow("Utility Tools", value: Binding(
                    get: { draft.utilityToolsEnabled },
                    set: { draft.utilityToolsEnabled = $0; writeBack() }
                ), traceKey: "utilityToolsEnabled")
                toggleRow("Brave Search", value: Binding(
                    get: { draft.braveSearchEnabled },
                    set: { draft.braveSearchEnabled = $0; writeBack() }
                ), traceKey: "braveSearchEnabled")
            }
            .padding(.top, Theme.Spacing.xs)
        } label: { sectionLabel("Built-in Tools") }
        .tint(Theme.Colors.textMid)
    }

    private var workdirSection: some View {
        DisclosureGroup(isExpanded: $openWorkdir) {
            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                HStack {
                    TextField("/path/to/workdir", text: Binding(
                        get: { draft.workingDirectory ?? "" },
                        set: { v in
                            draft.workingDirectory = v.isEmpty ? nil : v
                            writeBack()
                        }
                    ))
                    .textFieldStyle(.roundedBorder)
                    .font(Theme.Typography.mono)
                    Button {
                        let panel = NSOpenPanel()
                        panel.canChooseFiles = false
                        panel.canChooseDirectories = true
                        panel.allowsMultipleSelection = false
                        if panel.runModal() == .OK, let url = panel.url {
                            draft.workingDirectory = url.path
                            writeBack()
                        }
                    } label: {
                        Image(systemName: "folder")
                    }
                    .buttonStyle(.borderless)
                    .foregroundStyle(Theme.Colors.accent)
                }
                tierChipRow(traceKey: "workingDirectory",
                            isChatSet: draft.workingDirectory != nil) {
                    draft.workingDirectory = nil; writeBack()
                }
            }
            .padding(.top, Theme.Spacing.xs)
        } label: { sectionLabel("Working Directory") }
        .tint(Theme.Colors.textMid)
    }

    private var mcpSection: some View {
        DisclosureGroup(isExpanded: $openMCP) {
            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                toggleRow("MCP Enabled", value: Binding(
                    get: { draft.mcpEnabled },
                    set: { draft.mcpEnabled = $0; writeBack() }
                ), traceKey: "mcpEnabled")
            }
            .padding(.top, Theme.Spacing.xs)
        } label: { sectionLabel("MCP") }
        .tint(Theme.Colors.textMid)
    }

    private var wireApiSection: some View {
        DisclosureGroup(isExpanded: $openWireApi) {
            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                HStack {
                    Text("Wire API").font(Theme.Typography.body)
                        .foregroundStyle(Theme.Colors.textHigh)
                    Spacer()
                    Picker("", selection: Binding<String>(
                        get: { draft.wireApi ?? "__inherit__" },
                        set: { v in
                            draft.wireApi = (v == "__inherit__") ? nil : v
                            writeBack()
                        }
                    )) {
                        Text("Inherit").tag("__inherit__")
                        Text("Auto").tag("auto")
                        Text("OpenAI").tag("openai")
                        Text("Anthropic").tag("anthropic")
                        Text("Ollama").tag("ollama")
                    }
                    .pickerStyle(.menu)
                    .frame(width: 180)
                }
                tierChipRow(traceKey: "wireApi",
                            isChatSet: draft.wireApi != nil) {
                    draft.wireApi = nil; writeBack()
                }
            }
            .padding(.top, Theme.Spacing.xs)
        } label: { sectionLabel("Wire API") }
        .tint(Theme.Colors.textMid)
    }

    // MARK: - Footer

    private var effectiveValuesFooter: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            Text("Effective values")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            if let r = resolved {
                HStack(spacing: Theme.Spacing.md) {
                    effChip("temp", value: String(format: "%.2f", r.settings.defaultTemperature),
                            tier: r.resolutionTrace["defaultTemperature"] ?? .global)
                    effChip("topP", value: String(format: "%.2f", r.settings.defaultTopP),
                            tier: r.resolutionTrace["defaultTopP"] ?? .global)
                    effChip("think",
                            value: r.settings.defaultEnableThinking.map { $0 ? "on" : "off" } ?? "nil",
                            tier: r.resolutionTrace["defaultEnableThinking"] ?? .global)
                    effChip("sys",
                            value: (r.settings.defaultSystemPrompt?.isEmpty == false) ? "set" : "—",
                            tier: r.resolutionTrace["defaultSystemPrompt"] ?? .global)
                }
            } else {
                Text("Loading…").font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }
        }
        .padding(.horizontal, Theme.Spacing.md)
        .padding(.vertical, Theme.Spacing.sm)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.Colors.surface)
    }

    // MARK: - Row helpers

    private func sectionLabel(_ text: String) -> some View {
        Text(text)
            .font(Theme.Typography.bodyHi)
            .foregroundStyle(Theme.Colors.textHigh)
    }

    private func doubleRow(_ title: String,
                           value: Binding<Double?>,
                           range: ClosedRange<Double>,
                           step: Double,
                           traceKey: String,
                           fallback: @escaping () -> Double) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title).font(Theme.Typography.body)
                    .foregroundStyle(Theme.Colors.textHigh)
                Spacer()
                Text(String(format: "%.2f", value.wrappedValue ?? fallback()))
                    .font(Theme.Typography.mono)
                    .foregroundStyle(Theme.Colors.textMid)
                    .frame(width: 52, alignment: .trailing)
            }
            Slider(value: Binding(
                get: { value.wrappedValue ?? fallback() },
                set: { value.wrappedValue = $0 }
            ), in: range, step: step)
            .tint(Theme.Colors.accent)
            tierChipRow(traceKey: traceKey,
                        isChatSet: value.wrappedValue != nil) {
                value.wrappedValue = nil
            }
        }
    }

    private func intRow(_ title: String,
                        value: Binding<Int?>,
                        range: ClosedRange<Int>,
                        traceKey: String,
                        fallback: @escaping () -> Int) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title).font(Theme.Typography.body)
                    .foregroundStyle(Theme.Colors.textHigh)
                Spacer()
                Stepper(value: Binding(
                    get: { value.wrappedValue ?? fallback() },
                    set: { value.wrappedValue = $0 }
                ), in: range) {
                    Text("\(value.wrappedValue ?? fallback())")
                        .font(Theme.Typography.mono)
                        .foregroundStyle(Theme.Colors.textMid)
                        .frame(width: 64, alignment: .trailing)
                }
            }
            tierChipRow(traceKey: traceKey,
                        isChatSet: value.wrappedValue != nil) {
                value.wrappedValue = nil
            }
        }
    }

    private func toggleRow(_ title: String,
                           value: Binding<Bool?>,
                           traceKey: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title).font(Theme.Typography.body)
                    .foregroundStyle(Theme.Colors.textHigh)
                Spacer()
                Picker("", selection: Binding<Int>(
                    get: {
                        if value.wrappedValue == nil { return 0 }
                        return value.wrappedValue! ? 1 : 2
                    },
                    set: { v in
                        value.wrappedValue = (v == 0) ? nil : (v == 1)
                    }
                )) {
                    Text("Inherit").tag(0)
                    Text("On").tag(1)
                    Text("Off").tag(2)
                }
                .pickerStyle(.segmented)
                .frame(width: 180)
                .labelsHidden()
            }
            tierChipRow(traceKey: traceKey,
                        isChatSet: value.wrappedValue != nil) {
                value.wrappedValue = nil
            }
        }
    }

    /// Small "(from X)" chip + colored dot row + reset button.
    /// - colored dot: the tier source so colorblind users still see the signal
    /// - "(from X)" label: same info, textual
    /// - Reset button: clears the chat-tier value for this field
    private func tierChipRow(traceKey: String,
                             isChatSet: Bool,
                             reset: @escaping () -> Void) -> some View {
        let tier = resolved?.resolutionTrace[traceKey] ?? .global
        let effective: SettingsTier = isChatSet ? .chat : tier
        return HStack(spacing: Theme.Spacing.xs) {
            Circle()
                .fill(Self.tierColor(effective))
                .frame(width: 6, height: 6)
            Text("(from \(effective.rawValue))")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            Spacer()
            if isChatSet {
                Button("Reset") { reset(); writeBack() }
                    .buttonStyle(.borderless)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
            }
        }
    }

    private func effChip(_ label: String, value: String, tier: SettingsTier) -> some View {
        HStack(spacing: 4) {
            Circle()
                .fill(Self.tierColor(tier))
                .frame(width: 5, height: 5)
            Text("\(label):\(value)")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
        }
    }

    /// Tier → color mapping. Matches the task spec:
    ///   request → (none shown — request is ephemeral and never reaches the popover)
    ///   chat    → accent
    ///   session → warning
    ///   global  → textLow
    ///   builtin → textLow (falls through to global visually)
    static func tierColor(_ tier: SettingsTier) -> Color {
        switch tier {
        case .request: return Theme.Colors.accentHi
        case .chat:    return Theme.Colors.accent
        case .session: return Theme.Colors.warning
        case .global:  return Theme.Colors.textLow
        case .builtin: return Theme.Colors.textLow
        }
    }

    // MARK: - Persistence

    private func load() async {
        let current = await app.engine.settings.chat(chatId) ?? ChatSettings()
        let res = await app.engine.settings.resolved(
            sessionId: sessionId, chatId: chatId, request: nil
        )
        await MainActor.run {
            self.draft = current
            self.resolved = res
            self.loaded = true
        }
    }

    private func writeBack() {
        let snap = draft
        let cid = chatId
        let sid = sessionId
        Task {
            await app.engine.settings.setChat(cid, snap)
            let res = await app.engine.settings.resolved(
                sessionId: sid, chatId: cid, request: nil
            )
            await MainActor.run { self.resolved = res }
        }
    }

    private func resetAll() {
        draft = ChatSettings()
        writeBack()
    }
}
