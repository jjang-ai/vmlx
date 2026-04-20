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
    /// Chat-level "hide tool status" preference, hydrated from
    /// ChatSettings.hideToolStatus when the active chat switches.
    /// MessageList passes this down to each MessageBubble which
    /// suppresses InlineToolCallCard when true. Pre-iter-39 the popover
    /// toggle persisted to SQLite but no reader consumed it, so
    /// flipping it had no observable effect.
    var hideToolStatus: Bool = false
    var pendingImages: [Data] = []
    /// Pending video attachments keyed by absolute file:// URL. Unlike
    /// images we don't inline the bytes: a 30 s 1080p clip is ~30 MB
    /// and would bloat SQLite. The engine path (ChatRequest →
    /// video_url ContentPart) consumes the URL directly; InputBar
    /// shows a thumbnail via AVAssetImageGenerator.
    var pendingVideos: [URL] = []
    var inputText: String = ""
    var bannerMessage: String? = nil

    /// Per-session draft stash: when the user switches away mid-compose
    /// (half-typed message, attached images/videos), we snapshot the
    /// pending state under the outgoing sessionId here. Switching back
    /// restores it so the user doesn't lose their in-progress turn.
    /// Cleared when `send()` actually dispatches the draft (iter-22).
    private struct ChatDraft {
        var inputText: String
        var pendingImages: [Data]
        var pendingVideos: [URL]
    }
    private var drafts: [UUID: ChatDraft] = [:]

    /// iter-110 §136: snapshot cache for ChatSettings captured at
    /// `deleteSession` / `clearAllSessions` time, read by the matching
    /// undo closure to restore per-chat reasoning_effort / systemPrompt /
    /// tools / etc. Populated by a MainActor-scheduled Task that runs
    /// immediately after the delete call site, so by the time the user
    /// clicks Undo (minimum human reaction ~100ms, the one-hop actor
    /// read is sub-millisecond) the box is filled. Race window is
    /// effectively zero in practice; worst case an ultra-fast undo
    /// loses settings customizations (session survives, defaults kick
    /// in) — strictly better than the prior behavior where every
    /// deleted chat leaked its ChatSettings row forever.
    private var deletedChatSettings: [UUID: ChatSettings] = [:]

    // iter-112 §138: `inputHistory` + `historyCursor` +
    // `recallPreviousInput()` REMOVED. They were a dead parallel
    // Up-arrow history dict that polluted across sessions (unlike
    // InputBar's `userHistory()` which derives per-chat from
    // `messages.filter(.user)`). The screen-level ChatScreen handler
    // that read them has also been removed — InputBar's TextField-
    // focused handler is the sole Up-arrow recall path now.

    /// Stack of recently-closed session ids so Cmd-Shift-T can reopen the
    /// last-closed chat. We snapshot the full session row at close-time so
    /// a reopen doesn't rely on SQLite still having the row (Cmd-W just
    /// removes from the sidebar — it doesn't touch the DB).
    private(set) var recentlyClosed: [ChatSession] = []

    /// Tiny in-memory undo stack for destructive actions (delete-session,
    /// delete-message, clear-all). Each entry pairs a human-readable
    /// label with a closure that restores the prior state. Cmd-Z pops
    /// the top action and invokes its closure.
    ///
    /// Bounded to the last 20 actions — an undo-all-the-way-back use
    /// case for a chat app is unnecessary, and holding snapshots of
    /// every deleted chat forever is a memory leak waiting to happen.
    struct UndoAction {
        let label: String
        let restore: () -> Void
    }
    private(set) var undoStack: [UndoAction] = []
    private let undoStackCap = 20

    /// Register an undoable action. Keeps the stack bounded; oldest
    /// entries fall off the bottom once cap is reached.
    private func pushUndo(_ label: String, restore: @escaping () -> Void) {
        undoStack.append(UndoAction(label: label, restore: restore))
        if undoStack.count > undoStackCap {
            undoStack.removeFirst(undoStack.count - undoStackCap)
        }
    }

    /// Pop and run the most recent undoable action. Returns the label
    /// that was undone so the caller (menu command, banner) can
    /// surface "Undid: delete chat …" feedback. Returns nil when the
    /// stack is empty.
    @discardableResult
    func undo() -> String? {
        guard let action = undoStack.popLast() else { return nil }
        action.restore()
        return action.label
    }

    /// Non-consuming peek used by the Edit menu command to render
    /// "Undo Delete chat …" (live-tracked via the menu item label).
    /// Returns nil when the stack is empty so SwiftUI can disable the
    /// menu item.
    var topUndoLabel: String? { undoStack.last?.label }

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
        // 2026-04-18 iter-22: save the outgoing chat's compose state
        // so it survives a round-trip. Pre-fix, switching away mid-
        // compose and back discarded the user's half-typed text +
        // attached images/videos, OR worse: kept them attached and
        // accidentally sent them to the new chat. Now the draft is
        // keyed per-session.
        if let outgoing = activeSessionId {
            if !inputText.isEmpty || !pendingImages.isEmpty || !pendingVideos.isEmpty {
                drafts[outgoing] = ChatDraft(
                    inputText: inputText,
                    pendingImages: pendingImages,
                    pendingVideos: pendingVideos)
            } else {
                drafts.removeValue(forKey: outgoing)
            }
        }
        activeSessionId = id
        messages = Database.shared.messages(for: id)
        // Hydrate chat-level UI preferences from SettingsStore so the
        // popover toggles take effect immediately on chat switch.
        //   • `hideToolStatus` drives InlineToolCallCard rendering in
        //     MessageList → MessageBubble.
        //   • `workingDirectory` seeds the engine's `terminalCwd` so
        //     a subsequent bash tool call defaults to that directory
        //     instead of the process cwd. Before iter-43 the field
        //     was persisted but never propagated, so users setting a
        //     per-chat workdir saw no effect unless the model
        //     explicitly filled `cwd` in the tool-call arguments.
        if let engine = app?.engine {
            Task { @MainActor [weak self] in
                let chat = await engine.settings.chat(id)
                self?.hideToolStatus = chat?.hideToolStatus ?? false
                if let wd = chat?.workingDirectory, !wd.isEmpty {
                    await engine.setTerminalCwd(URL(fileURLWithPath: wd))
                }
            }
        }
        // Restore (or zero) the incoming chat's draft.
        if let draft = drafts[id] {
            inputText = draft.inputText
            pendingImages = draft.pendingImages
            pendingVideos = draft.pendingVideos
        } else {
            inputText = ""
            pendingImages = []
            pendingVideos = []
        }
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

    // iter-112 §138: `recallPreviousInput()` deleted — see inputHistory
    // comment above. InputBar owns Up-arrow recall via userHistory().

    func deleteSession(_ id: UUID) {
        // Snapshot BEFORE the delete so we can restore on undo. Messages
        // are fetched fresh from SQLite rather than relying on `messages`
        // (which only holds the currently-active session) — otherwise
        // undo-deleting a background chat would resurrect it empty.
        guard let snapshot = sessions.first(where: { $0.id == id }) else { return }
        let snapshotMessages = Database.shared.messages(for: id)
        let priorActive = activeSessionId
        // **iter-70 (§99)**: snapshot the draft BEFORE the dict drop so
        // an undo can restore the user's in-progress compose state.
        // Previously `deleteSession` only cleaned SQLite + `sessions`;
        // the draft entry in `drafts[id]` survived as an orphan
        // (harmless — keyed by UUID so no cross-chat clobber — but a
        // slow memory leak if the user deletes lots of chats without
        // ever re-opening). Clearing the stash here means undo must
        // also re-seat it, handled below.
        let snapshotDraft = drafts[id]
        Database.shared.deleteSession(id)
        // iter-110 §136: close the ChatSettings row leak on permanent
        // delete. Snapshot + delete async; undo closure reads the
        // snapshot back and re-setChat. Race window is sub-millisecond
        // (one actor hop vs. human reaction time) so in practice the
        // box is populated by the time undo fires.
        if let engine = app?.engine {
            let chatIdCopy = id
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                if let snap = await engine.settings.chat(chatIdCopy) {
                    self.deletedChatSettings[chatIdCopy] = snap
                }
                await engine.settings.deleteChat(chatIdCopy)
            }
        }
        sessions.removeAll { $0.id == id }
        drafts.removeValue(forKey: id)
        if activeSessionId == id {
            activeSessionId = sessions.first?.id
            messages = activeSessionId.map { Database.shared.messages(for: $0) } ?? []
        }
        pushUndo("Delete chat \"\(snapshot.title)\"") { [weak self] in
            guard let self else { return }
            // Iter-27: batch under a single transaction — same
            // rationale as clearAllSessions undo above. A 500-message
            // delete-chat undo pre-fix was 500 sync fsyncs.
            Database.shared.withTransaction {
                Database.shared.upsertSession(snapshot)
                for m in snapshotMessages { Database.shared.upsertMessage(m) }
            }
            // iter-110 §136: restore ChatSettings snapshot captured at
            // delete time. If the capture Task hadn't completed before
            // this undo fires (tiny race window), the dict lookup
            // returns nil and the session restores with session/global
            // default settings — acceptable degradation.
            if let engine = self.app?.engine,
               let snap = self.deletedChatSettings.removeValue(forKey: id)
            {
                Task { await engine.settings.setChat(id, snap) }
            }
            self.sessions = Database.shared.allSessions()
            if let d = snapshotDraft {
                self.drafts[id] = d
            }
            if priorActive == id {
                self.activeSessionId = id
                self.messages = snapshotMessages
                // Active-chat resurrection must also restore live compose
                // state — not just the drafts dict — otherwise the user
                // undoes a delete and lands on a blank composer even
                // though their draft is back in `drafts[id]`.
                if let d = snapshotDraft {
                    self.inputText = d.inputText
                    self.pendingImages = d.pendingImages
                    self.pendingVideos = d.pendingVideos
                }
            }
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
        // Snapshot every session + its full message history so undo can
        // resurrect the whole set. Wipe runs first so the fresh session
        // `newSession()` creates at the end doesn't appear in the
        // snapshot (otherwise undo would also recreate + then delete it).
        let snapshotSessions = sessions
        var snapshotMessages: [UUID: [ChatMessage]] = [:]
        for s in sessions {
            snapshotMessages[s.id] = Database.shared.messages(for: s.id)
        }
        // iter-70 (§99): snapshot every stashed draft so undo can restore
        // in-progress compose state alongside the sessions/messages.
        // Same orphan-leak rationale as `deleteSession` — without the
        // clear, drafts for wiped chats survive in memory (harmless
        // cross-chat wise because they're UUID-keyed, but a slow leak
        // if the user does repeated clear cycles).
        let snapshotDrafts = drafts
        // iter-110 §136: capture IDs we're about to delete so the
        // async ChatSettings-cleanup task can loop through them. The
        // same snapshot box (`deletedChatSettings`) is populated
        // one-by-one; undo drains the whole dict.
        let idsToCleanSettings = sessions.map(\.id)
        for s in sessions {
            Database.shared.deleteSession(s.id)
        }
        sessions.removeAll()
        recentlyClosed.removeAll()
        drafts.removeAll(keepingCapacity: false)
        activeSessionId = nil
        messages = []
        if let engine = app?.engine {
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                for chatId in idsToCleanSettings {
                    if let snap = await engine.settings.chat(chatId) {
                        self.deletedChatSettings[chatId] = snap
                    }
                    await engine.settings.deleteChat(chatId)
                }
            }
        }
        // Land in a fresh empty session so the chat screen isn't blank.
        newSession()
        pushUndo("Clear all chats (\(snapshotSessions.count))") { [weak self] in
            guard let self else { return }
            // Iter-27: batch the bulk-restore under a single SQLite
            // transaction. Pre-fix a 20-chat × 50-msg clear→undo cycle
            // was ~1000 synchronous fsync calls (each upsertSession /
            // upsertMessage committed independently in WAL mode) which
            // stalled the Main Actor for multiple seconds on rotational
            // storage. With one txn this is sub-100ms on SSD, a couple
            // hundred ms on rotational.
            Database.shared.withTransaction {
                for s in snapshotSessions {
                    Database.shared.upsertSession(s)
                    for m in snapshotMessages[s.id] ?? [] {
                        Database.shared.upsertMessage(m)
                    }
                }
            }
            self.sessions = Database.shared.allSessions()
            // iter-110 §136: drain the ChatSettings snapshot dict back
            // into the engine. Same caveat as deleteSession undo —
            // settings that hadn't been captured yet (undo fires
            // before the capture Task ran) stay empty; restored
            // sessions get session/global defaults in that edge case.
            if let engine = self.app?.engine {
                let drained = self.deletedChatSettings
                self.deletedChatSettings.removeAll()
                if !drained.isEmpty {
                    Task {
                        for (chatId, snap) in drained {
                            await engine.settings.setChat(chatId, snap)
                        }
                    }
                }
            }
            // Re-seat every snapshotted draft so undo fully restores
            // the pre-wipe compose state across all chats.
            for (id, d) in snapshotDrafts {
                self.drafts[id] = d
            }
            if let first = self.sessions.first {
                self.activeSessionId = first.id
                self.messages = Database.shared.messages(for: first.id)
                if let d = snapshotDrafts[first.id] {
                    self.inputText = d.inputText
                    self.pendingImages = d.pendingImages
                    self.pendingVideos = d.pendingVideos
                }
            }
        }
    }

    func deleteMessage(_ id: UUID) {
        // Snapshot + index BEFORE the delete so undo re-inserts at the
        // original position. `upsertMessage` with the same id puts the
        // row back in SQLite; we re-insert in-memory at the captured
        // index to preserve ordering.
        guard let idx = messages.firstIndex(where: { $0.id == id }) else { return }
        // iter-111 §137: if the user deletes the currently-streaming
        // assistant (always the LAST message while `isGenerating`),
        // cancel the stream task first. Without this the engine keeps
        // generating tokens that silently drop at the firstIndex
        // lookup in the stream loop (line ~975) since the message no
        // longer exists — wasting engine cycles until the stream
        // naturally completes. We flip `intentionalStop` so the
        // cancellation surfaces as a clean stop rather than a red
        // error banner elsewhere. Earlier-message deletes don't touch
        // the stream; the engine's prompt context was already consumed
        // at prefill so they can continue cleanly.
        if isGenerating && idx == messages.count - 1 {
            intentionalStop = true
            streamTask?.cancel()
        }
        let snapshot = messages[idx]
        Database.shared.deleteMessage(id)
        messages.remove(at: idx)
        pushUndo("Delete message") { [weak self] in
            guard let self else { return }
            Database.shared.upsertMessage(snapshot)
            let clamped = min(idx, self.messages.count)
            self.messages.insert(snapshot, at: clamped)
        }
    }

    func editMessage(_ id: UUID, newContent: String) {
        // iter-109 §135: VM-level guard. UI disables the pencil during
        // streaming, but a non-UI caller (scripting, future keyboard
        // shortcut, programmatic test harness) would otherwise mutate
        // the in-flight assistant's prompt mid-stream. Mirrors the
        // branchSession guard (§134) for consistency across
        // destructive chat-mutation verbs.
        guard !isGenerating else {
            app?.flashBanner("Stop the current response before editing")
            return
        }
        guard let idx = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[idx].content = newContent
        Database.shared.upsertMessage(messages[idx])
    }

    /// Regenerate: drop everything from `messageId` forward, resend the prior user turn.
    func regenerate(from messageId: UUID) {
        // iter-109 §135: VM-level guard. Same reasoning as editMessage —
        // UI disables while streaming but a non-UI caller could race the
        // in-flight assistant placeholder that streaming is writing
        // into. `send()` down the call chain does cancel `streamTask`
        // anyway, but the pre-cancel `removeSubrange` can race with the
        // token-append path observing stale indices. Reject early and
        // flash for parity with branchSession.
        guard !isGenerating else {
            app?.flashBanner("Stop the current response before regenerating")
            return
        }
        guard let sessionId = activeSessionId else { return }
        guard let idx = messages.firstIndex(where: { $0.id == messageId }) else { return }
        let anchor = messages[idx]
        Database.shared.deleteMessages(after: anchor.createdAt, in: sessionId)
        messages.removeSubrange(idx...)
        send()
    }

    /// Branch: fork this chat from `messageId` into a new session. Copies
    /// every message strictly BEFORE `messageId` into a fresh ChatSession
    /// (with new UUIDs to avoid primary-key collisions in SQLite) and
    /// opens that session. The original chat is unchanged — the user can
    /// return to it any time.
    ///
    /// Production use case: "what if I'd answered differently mid-way
    /// through". Common in Claude/ChatGPT web UIs. 2026-04-18 iter-9.
    func branchSession(from messageId: UUID) {
        // VM-level guards — complement the MessageBubble disable state
        // so non-UI callers (future scripting / keyboard shortcut) can't
        // branch into an inconsistent state.
        guard !isGenerating else {
            app?.flashBanner("Stop the current response before branching")
            return
        }
        guard let sourceId = activeSessionId else { return }
        guard let source = sessions.first(where: { $0.id == sourceId }) else { return }
        guard let idx = messages.firstIndex(where: { $0.id == messageId }) else { return }

        // Slice [0, idx) — everything STRICTLY before the anchor.
        let copied = Array(messages.prefix(idx))
        guard !copied.isEmpty else {
            // Can't branch from the first message — that's identical to
            // a fresh session. Flash-banner the user so they know why
            // nothing happened.
            app?.flashBanner("Can't branch from the first message — use New Chat instead")
            return
        }

        // Build the fork with a fresh UUID + a title that tells the
        // user which chat this branched from without being noisy.
        let forkTitle: String = {
            let base = source.title.trimmingCharacters(in: .whitespacesAndNewlines)
            let trimmed = base.isEmpty ? "chat" : base
            return "↱ \(trimmed)"
        }()
        let now = Date()
        let fork = ChatSession(
            id: UUID(), title: forkTitle,
            modelPath: source.modelPath,
            createdAt: now, updatedAt: now
        )
        // Iter-27: wrap fork creation + message copy in a single
        // SQLite transaction. Pre-fix a 500-message branch blocked
        // the Main Actor for ~hundreds of ms on rotational storage.
        // Re-id each copied message so SQLite primary keys don't
        // collide and cascading deletes on the original chat don't
        // take the fork's rows with them. Preserve ordering via
        // incremental microsecond bumps on createdAt.
        Database.shared.withTransaction {
            Database.shared.upsertSession(fork)
            for (offset, original) in copied.enumerated() {
                var m = original
                m.id = UUID()
                m.sessionId = fork.id
                m.createdAt = now.addingTimeInterval(TimeInterval(offset) * 1e-6)
                Database.shared.upsertMessage(m)
            }
        }

        // iter-108 §134: carry per-chat ChatSettings from source → fork.
        // Without this the fork silently inherits session/global defaults
        // for reasoning_effort / systemPrompt / tools / stopSequences /
        // toolChoice / workingDirectory / hideToolStatus / etc. User
        // complaint shape: "I branched my thinking chat and it stopped
        // thinking", or "where did my tools go". Messages are copied
        // (transaction above), but the settings row is keyed by chat
        // UUID so the fresh UUID has no row until we explicitly seed
        // it. `setChat` debounces → the fork's settings land on disk
        // in the same quit-flush drain as normal setting writes.
        if let engine = app?.engine {
            let forkId = fork.id
            let sourceIdCopy = sourceId
            Task {
                if let srcChat = await engine.settings.chat(sourceIdCopy) {
                    await engine.settings.setChat(forkId, srcChat)
                }
            }
        }

        // Open the fork so the user lands inside it immediately.
        sessions.insert(fork, at: 0)
        selectSession(fork.id)
        pushUndo("Branch \"\(fork.title)\"") { [weak self] in
            guard let self else { return }
            Database.shared.deleteSession(fork.id)
            // iter-108 §134: drop the ChatSettings row we just seeded on
            // the source-to-fork copy. Without this the undo would leave
            // an orphan chat_overrides row that persists beyond the
            // deleted session — cheap but accumulates for users who
            // repeatedly branch+undo.
            if let engine = self.app?.engine {
                let forkId = fork.id
                Task { await engine.settings.deleteChat(forkId) }
            }
            self.sessions.removeAll(where: { $0.id == fork.id })
            if self.activeSessionId == fork.id {
                self.activeSessionId = sourceId
                self.messages = Database.shared.messages(for: sourceId)
            }
        }
    }

    func send() {
        guard let sessionId = activeSessionId else { return }
        // No-model guard. User complaint: "model loading does not work" —
        // root cause was that this guard hard-required `selectedModelPath
        // != nil`, but `selectedModelPath` is ONLY set by the cmd+k quick
        // picker and the onboarding wizard. Starting a session from the
        // Chat ▶ button OR from the Server tab's Start Session button
        // never touched it, so even after a successful load the guard
        // bounced users back to Server with "Load a model in the Server
        // tab first" — which looked like loading had failed.
        //
        // Post-Gateway-multiplexer (UI-9), the right question is NOT
        // "is the globally-selected model loaded" but "is ANY session
        // running we can route to" — the gateway picks the right engine
        // from the request's `model` field. So pass if:
        //   - selectedModelPath is set AND its engine is live, OR
        //   - at least one session is running/in-standby (any model).
        // Falls through to the actual request if the chat carries a
        // `modelAlias` that matches a running session.
        if let appRef = app {
            let globalHasModel = (appRef.selectedModelPath != nil) &&
                                 appRef.engineState != .stopped &&
                                 !isErrorState(appRef.engineState)
            let anySessionLive = appRef.sessions.contains {
                switch $0.state {
                case .running, .loading, .standby: return true
                case .stopped, .error: return false
                }
            }
            if !globalHasModel && !anySessionLive {
                bannerMessage = "Load a model first — hit ▶ next to the model picker above, or use the Server tab"
                appRef.flashBanner(bannerMessage ?? "")
                // Do NOT auto-switch to Server tab anymore — the Chat
                // page now has its own start controls (picker ▶ button +
                // per-row menu). Bouncing back to Server was the old
                // workaround for not having those controls.
                return
            }
        }
        let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            let user = ChatMessage(
                sessionId: sessionId,
                role: .user,
                content: trimmed,
                imageData: pendingImages,
                videoPaths: pendingVideos.map { $0.absoluteString }
            )
            messages.append(user)
            Database.shared.upsertMessage(user)
            // iter-112 §138: the send-time `inputHistory.append(trimmed)`
            // + `historyCursor = nil` block was the only writer of the
            // dead parallel history store that §138 removed. InputBar
            // derives its history from `messages.filter(.user)` so the
            // sent turn is already tracked there (line above).
            inputText = ""
            pendingImages = []
            pendingVideos = []
            // iter-22: dispatched the draft — clear its stashed copy
            // under the active session so a selectSession() round-trip
            // doesn't silently resurrect the already-sent content.
            drafts.removeValue(forKey: sessionId)

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
            let hasMedia = m.role == .user &&
                (!m.imageData.isEmpty || !m.videoPaths.isEmpty)
            if hasMedia {
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
                // Video attachments: pass the absolute `file://` URL
                // through as a `video_url` ContentPart. `Stream.extractVideos`
                // (iter-0 carry-over) stages the file via AVURLAsset and
                // hands it to the VLM processor as a UserInput.Video.
                for path in m.videoPaths {
                    parts.append(ChatRequest.ContentPart(
                        type: "video_url",
                        videoUrl: .init(url: path)))
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
        // Resolve the per-session SessionSettings BEFORE dispatch so we
        // can fork between local Engine and remote endpoint. Reading
        // SessionSettings here (rather than inside the streamTask) keeps
        // the 4-tier resolution cheap and avoids two extra actor hops.
        // `serverSid` is captured into the streamTask below so the
        // settings resolution + remote-endpoint dispatch can look up
        // the per-session SessionSettings blob. MainActor hop would be
        // wasteful here — we pass the raw UUID and let the Task do the
        // engine.settings.session(_:) lookup asynchronously.
        let serverSid = serverSessionId
        streamTask = Task { [weak self] in
            guard let engine = engine else { return }
            // Pull the 4-tier-resolved settings snapshot for this chat. The
            // session-level `modelAlias` (when set) wins over whatever the
            // selected model path is, mirroring vmlx-engine HTTP dispatch.
            let resolved = await engine.settings.resolved(
                sessionId: serverSid, chatId: chatId, request: nil
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

            // Chat-level sampling overrides that the 4-tier resolver
            // doesn't synthesize into `r` (which returns a flat
            // GlobalSettings). ChatSettingsPopover persists both of
            // these, but ChatViewModel used to hardcode them to nil on
            // the outbound ChatRequest — meaning the UI knobs had no
            // observable effect on the in-app chat path.
            //   • reasoningEffort: none/low/medium/high clamps the
            //     engine's reasoning window for Mistral 4 / DeepSeek /
            //     GLM. Global default is "none" for parity with the
            //     Python app's v1.3.x behavior.
            //   • stopSequences: extra strings that halt decode early;
            //     used by tool-calling clients + user-visible "Stop on
            //     </end>" workflows.
            let reasoningEffortOut = chatOverrides?.reasoningEffort
            let stopSequencesOut = chatOverrides?.stopSequences

            // System prompt injection. The 4-tier resolver already picks
            // the highest-priority non-nil systemPrompt (chat → session
            // → global) into `r.defaultSystemPrompt`. Before iter-34 no
            // consumer read it on the in-app chat path — users setting
            // a per-chat system prompt via ChatSettingsPopover got
            // silently ignored. Now we prepend it as a `system` message
            // IFF the transcript doesn't already start with one. The
            // dedup is deliberate: an explicit system message in the
            // visible chat transcript is the user directly authoring
            // instructions and must win over the settings-tier default.
            let resolvedSystemPrompt = r.defaultSystemPrompt ?? ""
            let hasExistingSystemMsg = reqMessages.first?.role.lowercased() == "system"
            let effectiveMessages: [ChatRequest.Message] = {
                guard !resolvedSystemPrompt.isEmpty, !hasExistingSystemMsg else {
                    return reqMessages
                }
                let sys = ChatRequest.Message(
                    role: "system",
                    content: .string(resolvedSystemPrompt),
                    name: nil,
                    toolCalls: nil,
                    toolCallId: nil
                )
                return [sys] + reqMessages
            }()

            var req = ChatRequest(
                model: modelField,
                messages: effectiveMessages,
                stream: true,
                maxTokens: r.defaultMaxTokens,
                temperature: r.defaultTemperature,
                topP: r.defaultTopP,
                topK: r.defaultTopK,
                minP: r.defaultMinP,
                repetitionPenalty: r.defaultRepetitionPenalty,
                stop: stopSequencesOut,
                seed: nil,
                enableThinking: r.defaultEnableThinking ?? reasoning,
                reasoningEffort: reasoningEffortOut,
                tools: toolList,
                toolChoice: toolChoiceValue
            )
            // Tool-loop ceiling: `ChatRequest.init` doesn't take
            // maxToolIterations (historical — the init predates the
            // tool-loop feature). Before iter-37, leaving it nil meant
            // Stream.swift fell back to its internal default (10) and
            // the ChatSettingsPopover slider (1–32) was purely
            // cosmetic. Post-init assignment wires the user's value
            // through; Stream.swift reads `request.maxToolIterations`
            // first when deciding the loop budget.
            if let mti = chatOverrides?.maxToolIterations, mti > 0 {
                req.maxToolIterations = mti
            }

            // JIT wake: if this engine is in soft/deep standby (idle timer
            // fired, user manually slept it, or wake was skipped), bring
            // it back online before dispatching. HTTP routes do this at
            // every entry point (OpenAIRoutes, Ollama, Anthropic,
            // Gateway); the in-app Chat path used to skip it, which made
            // the first send after idle-sleep error with "Engine
            // unloaded". wakeFromStandby is a no-op for `.running`.
            await engine.wakeFromStandby()

            // Remote-session dispatch: if this chat is bound to a server
            // session whose SessionSettings carries a remoteURL, it
            // targets an external OpenAI/Ollama/Anthropic-compatible
            // server instead of the local engine. RemoteEngineClient
            // yields the same StreamChunk type so the downstream
            // applyChunk + isStreaming flow stays identical.
            let sessionForRemote: SessionSettings? = await {
                guard let sid = serverSid else { return nil }
                return await engine.settings.session(sid)
            }()
            let stream: AsyncThrowingStream<StreamChunk, Error>
            if let s = sessionForRemote, s.isRemote,
               let remoteURLString = s.remoteURL,
               let remoteURL = URL(string: remoteURLString)
            {
                let kind = RemoteEngineClient.Kind(
                    rawOrDefault: s.remoteProtocol)
                let remoteModelName = s.remoteModelName ?? modelField
                let client = RemoteEngineClient(
                    endpoint: remoteURL,
                    kind: kind,
                    apiKey: s.remoteAPIKey,
                    modelName: remoteModelName
                )
                // Track on the engine so stop() can cancel through the
                // existing cancellation pathway.
                await engine.attachRemoteClient(client)
                stream = await client.stream(request: req)
            } else {
                stream = await engine.stream(request: req)
            }
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
        // Audit 2026-04-16 UX: was "Engine not yet wired: …" — leftover
        // scaffold string visible to end users on any stream failure.
        bannerMessage = "Generation failed: \(msg)"
        app?.flashBanner(bannerMessage ?? "")
    }
}
