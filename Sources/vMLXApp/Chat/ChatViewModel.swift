import Foundation
import SwiftUI
import vMLXEngine

/// Chat mode view-model. One per window.
/// Bridges the SQLite session store, the `Engine` actor, and the SwiftUI views.
@Observable
@MainActor
final class ChatViewModel {
    var sessions: [ChatSession] = []
    var activeSessionId: UUID? = nil
    var messages: [ChatMessage] = []
    var searchQuery: String = ""
    var isGenerating: Bool = false
    var reasoningEnabled: Bool = false
    var pendingImages: [Data] = []
    var inputText: String = ""
    var bannerMessage: String? = nil

    /// Per-session scrollback of user prompts for the Up-arrow input recall
    /// shortcut. Only user messages go here; `historyCursor` tracks the
    /// current recall index (nil = not recalling).
    var inputHistory: [String] = []
    var historyCursor: Int? = nil

    /// Stack of recently-closed session ids so Cmd-Shift-T can reopen the
    /// last-closed chat. We snapshot the full session row at close-time so
    /// a reopen doesn't rely on SQLite still having the row (Cmd-W just
    /// removes from the sidebar — it doesn't touch the DB).
    private(set) var recentlyClosed: [ChatSession] = []

    private weak var app: AppState?
    private var streamTask: Task<Void, Never>? = nil

    /// Optional server-session id — when set, `send()` targets the engine
    /// owned by that specific server session instead of `app.engine`. Used
    /// by future multi-session chat pinning; default nil = use active engine.
    var serverSessionId: UUID? = nil

    /// Set to `true` by `stop()` so the cancellation thrown from the
    /// streaming task is recognized as user-intentional and renders as
    /// "[stopped]" instead of a red error banner. Mirrors Electron's
    /// `intentionalStopRef` in `ChatInterface.tsx`.
    private var intentionalStop: Bool = false

    /// Convenience alias used by Chat views — `isStreaming` reads better in
    /// view code, and the audit/UX-AUDIT items 9/13 spec their guards by name.
    var isStreaming: Bool { isGenerating }

    /// Pure chunk-application helper. Extracted so the reasoning-OFF
    /// fallthrough rule (§15 of NO-REGRESSION-CHECKLIST.md) can be unit
    /// tested without standing up the engine actor. Mirrors the inline body
    /// of `send()`'s for-await loop.
    nonisolated static func applyChunk(_ chunk: StreamChunk,
                                       to message: inout ChatMessage,
                                       reasoningEnabled: Bool) {
        if let c = chunk.content { message.content += c }
        if let r = chunk.reasoning {
            message.reasoning = (message.reasoning ?? "") + r
            // CRITICAL §15: when reasoning is OFF, route reasoning deltas
            // through to visible content so the bubble doesn't render blank.
            // Mirrors server.py:_stream_chat_completions
            //   emit_content = delta_msg.content || delta_msg.reasoning
            // This regresses every couple sessions — DO NOT remove.
            if !reasoningEnabled {
                message.content += r
            }
        }
        if let u = chunk.usage { message.usage = u }

        // Tool calls → persisted JSON + inline card rendering.
        if let calls = chunk.toolCalls, !calls.isEmpty {
            let encoded: [[String: Any]] = calls.map { tc in
                [
                    "id": tc.id,
                    "type": tc.type,
                    "function": [
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    ],
                ]
            }
            if let data = try? JSONSerialization.data(withJSONObject: encoded),
               let json = String(data: data, encoding: .utf8) {
                message.toolCallsJSON = json
            }
            // Seed any missing statuses as .pending so the card renders
            // immediately — the next ToolStatus chunk will upgrade them.
            for tc in calls where message.toolStatuses[tc.id] == nil {
                message.toolStatuses[tc.id] = .pending
            }
        }
        if let status = chunk.toolStatus {
            let phase: ToolCallStatus
            switch status.phase {
            case .started: phase = .pending
            case .running: phase = .running
            case .done:    phase = .done
            case .error:   phase = .error
            }
            message.toolStatuses[status.toolCallId] = phase
        }
    }

    func attach(_ app: AppState) {
        self.app = app
        reload()
        if activeSessionId == nil, let first = sessions.first {
            activeSessionId = first.id
            messages = Database.shared.messages(for: first.id)
        } else if sessions.isEmpty {
            newSession()
        }
    }

    var filteredSessions: [ChatSession] {
        guard !searchQuery.isEmpty else { return sessions }
        let q = searchQuery.lowercased()
        return sessions.filter { $0.title.lowercased().contains(q) }
    }

    func reload() {
        sessions = Database.shared.allSessions()
    }

    func newSession() {
        let s = ChatSession()
        Database.shared.upsertSession(s)
        sessions.insert(s, at: 0)
        activeSessionId = s.id
        messages = []
    }

    func selectSession(_ id: UUID) {
        // If we're mid-stream on the old chat, cancel cleanly so the
        // running Task doesn't write deltas into the new chat's
        // messages array. The assistant message on the old chat stays
        // in SQLite with whatever it had collected so far.
        if isGenerating {
            streamTask?.cancel()
            streamTask = nil
            isGenerating = false
        }
        activeSessionId = id
        messages = Database.shared.messages(for: id)
    }

    /// Cmd-W: remove a chat from the sidebar without touching the DB.
    /// Pushes the row onto `recentlyClosed` so Cmd-Shift-T can undo.
    func closeSession(_ id: UUID) {
        guard let idx = sessions.firstIndex(where: { $0.id == id }) else { return }
        let closed = sessions.remove(at: idx)
        recentlyClosed.append(closed)
        if activeSessionId == id {
            activeSessionId = sessions.first?.id
            messages = activeSessionId.map { Database.shared.messages(for: $0) } ?? []
        }
    }

    /// Cmd-Shift-T: reopen the most-recently closed chat.
    func reopenLastClosed() {
        guard let s = recentlyClosed.popLast() else { return }
        // Only re-insert if it isn't already present (closed + undeleted
        // could race with another insert).
        if !sessions.contains(where: { $0.id == s.id }) {
            sessions.insert(s, at: 0)
        }
        selectSession(s.id)
    }

    /// Up-arrow input recall. Returns true if recall fired (so the caller
    /// can mark the key event handled); false when there's no history or
    /// the user has typed something.
    @discardableResult
    func recallPreviousInput() -> Bool {
        guard inputText.isEmpty, !inputHistory.isEmpty else { return false }
        let nextIdx: Int
        if let cur = historyCursor {
            nextIdx = max(0, cur - 1)
        } else {
            nextIdx = inputHistory.count - 1
        }
        historyCursor = nextIdx
        inputText = inputHistory[nextIdx]
        return true
    }

    func deleteSession(_ id: UUID) {
        Database.shared.deleteSession(id)
        sessions.removeAll { $0.id == id }
        if activeSessionId == id {
            activeSessionId = sessions.first?.id
            messages = activeSessionId.map { Database.shared.messages(for: $0) } ?? []
        }
    }

    /// Rename a chat session in-place. Persists immediately so the new
    /// title survives relaunch. Empty strings are coerced to "Untitled
    /// chat" so the sidebar never renders a blank row. Trims whitespace
    /// to avoid leading/trailing spaces from quick edits.
    func renameSession(_ id: UUID, to newTitle: String) {
        guard let idx = sessions.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        let safe = trimmed.isEmpty ? "Untitled chat" : trimmed
        var s = sessions[idx]
        s.title = safe
        s.updatedAt = Date()
        sessions[idx] = s
        Database.shared.upsertSession(s)
    }

    /// Wipe every chat from SQLite and the sidebar in one go. Used by the
    /// "Clear all chats" footer button — the caller must already have
    /// shown a confirmation dialog. After the wipe a fresh empty session
    /// is created so the user lands in a usable state instead of a
    /// dead-end empty list.
    func clearAllSessions() {
        for s in sessions {
            Database.shared.deleteSession(s.id)
        }
        sessions.removeAll()
        recentlyClosed.removeAll()
        activeSessionId = nil
        messages = []
        // Land in a fresh empty session so the chat screen isn't blank.
        newSession()
    }

    func deleteMessage(_ id: UUID) {
        Database.shared.deleteMessage(id)
        messages.removeAll { $0.id == id }
    }

    func editMessage(_ id: UUID, newContent: String) {
        guard let idx = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[idx].content = newContent
        Database.shared.upsertMessage(messages[idx])
    }

    /// Regenerate: drop everything from `messageId` forward, resend the prior user turn.
    func regenerate(from messageId: UUID) {
        guard let sessionId = activeSessionId else { return }
        guard let idx = messages.firstIndex(where: { $0.id == messageId }) else { return }
        let anchor = messages[idx]
        Database.shared.deleteMessages(after: anchor.createdAt, in: sessionId)
        messages.removeSubrange(idx...)
        send()
    }

    func send() {
        guard let sessionId = activeSessionId else { return }
        // No-model guard: if no model is loaded, surface a banner with a
        // "Open Server tab" action and abort. Without this the request fans
        // out with an empty model string and the user sees a confusing stub
        // error. Mirrors Electron's `ChatInterface.tsx` `handleSend` guard.
        if let appRef = app {
            let hasModel = (appRef.selectedModelPath != nil) &&
                           appRef.engineState != .stopped &&
                           !isErrorState(appRef.engineState)
            if !hasModel {
                bannerMessage = "Load a model in the Server tab first"
                appRef.flashBanner(bannerMessage ?? "")
                // Switch user straight to Server mode so the CTA is one click.
                appRef.mode = .server
                return
            }
        }
        let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            let user = ChatMessage(
                sessionId: sessionId,
                role: .user,
                content: trimmed,
                imageData: pendingImages
            )
            messages.append(user)
            Database.shared.upsertMessage(user)
            // Push onto input history for Up-arrow recall.
            if inputHistory.last != trimmed {
                inputHistory.append(trimmed)
                if inputHistory.count > 100 { inputHistory.removeFirst() }
            }
            historyCursor = nil
            inputText = ""
            pendingImages = []

            // Bump session updatedAt so the sidebar reorders this chat
            // to the top (allSessions ORDERs BY updated_at DESC).
            if let idx = sessions.firstIndex(where: { $0.id == sessionId }) {
                var bumped = sessions[idx]
                bumped.updatedAt = Date()
                // First message in a new session → derive the title
                // from the user text (first ~40 chars, trimmed at word).
                if bumped.title == "New Chat" || bumped.title.isEmpty {
                    let candidate = String(trimmed.prefix(40))
                    bumped.title = candidate.isEmpty ? "New Chat" : candidate
                }
                sessions[idx] = bumped
                // Move to top.
                sessions.remove(at: idx)
                sessions.insert(bumped, at: 0)
                Database.shared.upsertSession(bumped)
            }
        }

        // Placeholder assistant message we'll stream into.
        var assistant = ChatMessage(sessionId: sessionId, role: .assistant, isStreaming: true)
        messages.append(assistant)
        Database.shared.upsertMessage(assistant)
        isGenerating = true

        streamTask?.cancel()
        let app = self.app
        let engine: Engine? = {
            if let sid = serverSessionId { return app?.engine(for: sid) }
            return app?.engine
        }()
        // Build request messages. If a user turn carries attached images
        // (`imageData`), wrap them as OpenAI-style `image_url` parts with
        // base64 data URLs so `Engine.stream` → `Stream.extractImages`
        // can forward them to `UserInput.images` for the VLM processor.
        let reqMessages: [ChatRequest.Message] = messages.dropLast().map { m in
            let content: ChatRequest.ContentValue
            if m.role == .user && !m.imageData.isEmpty {
                var parts: [ChatRequest.ContentPart] = []
                if !m.content.isEmpty {
                    parts.append(ChatRequest.ContentPart(
                        type: "text", text: m.content))
                }
                for data in m.imageData {
                    let b64 = data.base64EncodedString()
                    let dataURL = "data:image/png;base64,\(b64)"
                    parts.append(ChatRequest.ContentPart(
                        type: "image_url",
                        imageUrl: .init(url: dataURL)))
                }
                content = .parts(parts)
            } else {
                content = .string(m.content)
            }
            return ChatRequest.Message(
                role: m.role.rawValue,
                content: content,
                name: nil,
                toolCalls: nil,
                toolCallId: nil
            )
        }
        let reasoning = reasoningEnabled
        let fallbackModelPath = app?.selectedModelPath?.path ?? ""
        let chatId = sessionId

        let assistantId = assistant.id
        streamTask = Task { [weak self] in
            guard let engine = engine else { return }
            // Pull the 4-tier-resolved settings snapshot for this chat. The
            // session-level `modelAlias` (when set) wins over whatever the
            // selected model path is, mirroring vmlx-engine HTTP dispatch.
            let resolved = await engine.settings.resolved(
                sessionId: nil, chatId: chatId, request: nil
            )
            let r = resolved.settings
            let chatOverrides = await engine.settings.chat(chatId)
            let modelField = chatOverrides?.modelAlias ?? fallbackModelPath

            // Tool execution: if the chat has any tool flag enabled we
            // pass BashTool + MCP through to the engine. Stream.swift
            // already handles multi-turn tool dispatch in-process via
            // executeToolCall, so we don't need an outer loop here —
            // tool_call / tool_status chunks stream back naturally and
            // applyChunk renders them as InlineToolCallCard UI.
            //
            // Default is OFF: users opt in via ChatSettingsPopover →
            // "Tools". Enabling "Shell tool" turns on bash specifically;
            // enabling "Builtin tools" is the umbrella flag that also
            // pulls in MCP / file / search tools when the engine wires
            // them up.
            let shellOn = chatOverrides?.shellEnabled ?? false
            let builtinOn = chatOverrides?.builtinToolsEnabled ?? false
            let toolsEnabled = shellOn || builtinOn
            var toolList: [ChatRequest.Tool]? = chatOverrides?.tools
            if toolsEnabled, shellOn {
                var merged = toolList ?? []
                if !merged.contains(where: { $0.function.name == "bash" }) {
                    merged.append(BashTool.openAISchema)
                }
                toolList = merged
            }
            let toolChoiceValue: ChatRequest.ToolChoice? = {
                guard toolsEnabled else { return nil }
                if let raw = chatOverrides?.toolChoice {
                    switch raw {
                    case "none":     return ChatRequest.ToolChoice.none
                    case "required": return .required
                    default:         return .auto
                    }
                }
                return .auto
            }()

            let req = ChatRequest(
                model: modelField,
                messages: reqMessages,
                stream: true,
                maxTokens: r.defaultMaxTokens,
                temperature: r.defaultTemperature,
                topP: r.defaultTopP,
                topK: r.defaultTopK,
                minP: r.defaultMinP,
                repetitionPenalty: r.defaultRepetitionPenalty,
                stop: nil,
                seed: nil,
                enableThinking: r.defaultEnableThinking ?? reasoning,
                reasoningEffort: nil,
                tools: toolList,
                toolChoice: toolChoiceValue
            )
            let stream = await engine.stream(request: req)
            do {
                for try await chunk in stream {
                    try Task.checkCancellation()
                    await MainActor.run {
                        guard let self,
                              let i = self.messages.firstIndex(where: { $0.id == assistantId })
                        else { return }
                        Self.applyChunk(chunk,
                                        to: &self.messages[i],
                                        reasoningEnabled: self.reasoningEnabled)
                        Database.shared.upsertMessage(self.messages[i])
                    }
                }
            } catch let err as EngineError {
                await MainActor.run {
                    guard let self else { return }
                    if self.intentionalStop {
                        self.intentionalStop = false
                        self.finishOk(assistantId)
                    } else {
                        self.finishWithError(assistantId, err.description)
                    }
                }
                return
            } catch {
                await MainActor.run {
                    guard let self else { return }
                    if self.intentionalStop {
                        self.intentionalStop = false
                        self.finishOk(assistantId)
                    } else {
                        self.finishWithError(assistantId, "\(error)")
                    }
                }
                return
            }
            await MainActor.run { self?.finishOk(assistantId) }
        }

        assistant.isStreaming = false
    }

    func stop() {
        intentionalStop = true
        streamTask?.cancel()
        // Also reach into the Engine actor to cancel the in-flight
        // generation directly. `streamTask.cancel()` alone won't
        // interrupt a blocking prefill (vmlx-swift-lm runs prefill
        // synchronously inside TokenIterator.init), so we bypass the
        // AsyncStream layer via `Engine.cancelStream()`.
        if let appRef = app {
            let engine: Engine = {
                if let sid = serverSessionId { return appRef.engine(for: sid) }
                return appRef.engine
            }()
            Task { await engine.cancelStream() }
        }
        isGenerating = false
        if let sid = activeSessionId, let last = messages.last, last.role == .assistant {
            var m = messages[messages.count - 1]
            if !m.content.hasSuffix("[stopped]") {
                if !m.content.isEmpty && !m.content.hasSuffix("\n") { m.content += "\n" }
                m.content += "[stopped]"
            }
            m.isStreaming = false
            messages[messages.count - 1] = m
            Database.shared.upsertMessage(m)
            _ = sid
        }
    }

    private func isErrorState(_ s: EngineState) -> Bool {
        if case .error = s { return true }
        return false
    }

    private func finishOk(_ id: UUID) {
        isGenerating = false
        if let i = messages.firstIndex(where: { $0.id == id }) {
            messages[i].isStreaming = false
            Database.shared.upsertMessage(messages[i])
        }
    }

    private func finishWithError(_ id: UUID, _ msg: String) {
        isGenerating = false
        if let i = messages.firstIndex(where: { $0.id == id }) {
            if messages[i].content.isEmpty {
                messages[i].content = "[engine error] \(msg)"
            }
            messages[i].isStreaming = false
            Database.shared.upsertMessage(messages[i])
        }
        bannerMessage = "Engine not yet wired: \(msg)"
        app?.flashBanner(bannerMessage ?? "")
    }
}
