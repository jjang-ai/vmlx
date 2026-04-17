// SPDX-License-Identifier: Apache-2.0
//
// SettingsStore — 4-tier resolver + debounced persistence.
//
// Resolution order (highest wins):
//     request override  →  chat  →  session  →  global  →  builtin
//
// The store keeps the three tiers in memory and writes them back to SettingsDB
// through a 500ms debounce so rapid UI slider scrubs coalesce into a single
// disk write.

import Foundation

public actor SettingsStore {

    // MARK: - Storage

    private let db: SettingsDB
    private var globalCache: GlobalSettings
    private var sessionCache: [UUID: SessionSettings] = [:]
    private var chatCache: [UUID: ChatSettings] = [:]

    // Debounce handles, one per key.
    private var globalSaveTask: Task<Void, Never>?
    private var sessionSaveTasks: [UUID: Task<Void, Never>] = [:]
    private var chatSaveTasks: [UUID: Task<Void, Never>] = [:]

    // Test hook: caller can shorten the debounce in unit tests.
    private let debounceNanos: UInt64

    // Subscribers
    private var listeners: [UUID: AsyncStream<SettingsChange>.Continuation] = [:]

    // MARK: - Init

    public init(database: SettingsDB, debounceMs: Int = 500) {
        self.db = database
        self.debounceNanos = UInt64(max(0, debounceMs)) * 1_000_000
        let decoder = JSONDecoder()
        if let data = database.getGlobal(),
           let g = try? decoder.decode(GlobalSettings.self, from: data) {
            self.globalCache = g
        } else {
            self.globalCache = GlobalSettings()
            // First-run: seed the DB so user_version is bumped and defaults
            // are visible to any external inspector.
            if let data = try? JSONEncoder().encode(globalCache) {
                database.setGlobal(data)
            }
        }
        // Sessions + chats: lazy on first access (avoid scanning the whole
        // table on every launch). We still preload IDs for diagnostics.
        _ = database.allSessionIDs()
        _ = database.allChatIDs()
    }

    // MARK: - Accessors

    public func global() -> GlobalSettings {
        globalCache
    }

    public func session(_ id: UUID) -> SessionSettings? {
        if let s = sessionCache[id] { return s }
        guard let data = db.getSession(id) else { return nil }
        let decoder = JSONDecoder()
        if let s = try? decoder.decode(SessionSettings.self, from: data) {
            sessionCache[id] = s
            return s
        }
        return nil
    }

    /// Enumerate every session id that has a stored settings blob.
    /// Used by `AppState.hydrateSessionsFromSettings()` on app launch
    /// to rebuild the in-memory Session list so users come back to
    /// the sessions they left open before quitting.
    public func allSessionIDs() -> [UUID] {
        db.allSessionIDs()
    }

    public func chat(_ id: UUID) -> ChatSettings? {
        if let c = chatCache[id] { return c }
        guard let data = db.getChat(id) else { return nil }
        let decoder = JSONDecoder()
        if let c = try? decoder.decode(ChatSettings.self, from: data) {
            chatCache[id] = c
            return c
        }
        return nil
    }

    // MARK: - Setters (debounced)

    public func setGlobal(_ s: GlobalSettings) {
        globalCache = s
        globalSaveTask?.cancel()
        globalSaveTask = Task { [debounceNanos] in
            if debounceNanos > 0 {
                try? await Task.sleep(nanoseconds: debounceNanos)
            }
            if Task.isCancelled { return }
            await self.flushGlobal()
        }
        broadcast(.global)
    }

    public func setSession(_ id: UUID, _ s: SessionSettings) {
        sessionCache[id] = s
        sessionSaveTasks[id]?.cancel()
        sessionSaveTasks[id] = Task { [debounceNanos] in
            if debounceNanos > 0 {
                try? await Task.sleep(nanoseconds: debounceNanos)
            }
            if Task.isCancelled { return }
            await self.flushSession(id)
        }
        broadcast(.session(id))
    }

    public func setChat(_ id: UUID, _ s: ChatSettings) {
        chatCache[id] = s
        chatSaveTasks[id]?.cancel()
        chatSaveTasks[id] = Task { [debounceNanos] in
            if debounceNanos > 0 {
                try? await Task.sleep(nanoseconds: debounceNanos)
            }
            if Task.isCancelled { return }
            await self.flushChat(id)
        }
        broadcast(.chat(id))
    }

    public func deleteSession(_ id: UUID) {
        sessionCache.removeValue(forKey: id)
        sessionSaveTasks[id]?.cancel()
        sessionSaveTasks.removeValue(forKey: id)
        db.deleteSession(id)
        broadcast(.session(id))
    }

    public func deleteChat(_ id: UUID) {
        chatCache.removeValue(forKey: id)
        chatSaveTasks[id]?.cancel()
        chatSaveTasks.removeValue(forKey: id)
        db.deleteChat(id)
        broadcast(.chat(id))
    }

    /// Force any pending debounced writes to flush immediately. Useful for
    /// app-will-terminate handlers and for unit tests.
    public func flushPending() async {
        if let t = globalSaveTask { t.cancel(); await flushGlobal() }
        for (id, t) in sessionSaveTasks {
            t.cancel()
            await flushSession(id)
        }
        for (id, t) in chatSaveTasks {
            t.cancel()
            await flushChat(id)
        }
    }

    private func flushGlobal() {
        guard let data = try? JSONEncoder().encode(globalCache) else { return }
        db.setGlobal(data)
        globalSaveTask = nil
    }

    private func flushSession(_ id: UUID) {
        guard let s = sessionCache[id],
              let data = try? JSONEncoder().encode(s) else { return }
        db.setSession(id, data)
        sessionSaveTasks.removeValue(forKey: id)
    }

    private func flushChat(_ id: UUID) {
        guard let c = chatCache[id],
              let data = try? JSONEncoder().encode(c) else { return }
        db.setChat(id, data)
        chatSaveTasks.removeValue(forKey: id)
    }

    // MARK: - Subscription

    public func subscribe() -> AsyncStream<SettingsChange> {
        let id = UUID()
        return AsyncStream { continuation in
            self.register(id: id, continuation: continuation)
            continuation.onTermination = { @Sendable _ in
                Task { await self.unregister(id: id) }
            }
        }
    }

    private func register(id: UUID, continuation: AsyncStream<SettingsChange>.Continuation) {
        listeners[id] = continuation
    }

    private func unregister(id: UUID) {
        listeners.removeValue(forKey: id)
    }

    private func broadcast(_ change: SettingsChange) {
        for (_, c) in listeners {
            c.yield(change)
        }
    }

    // MARK: - Resolver (4-tier)

    /// Walk request -> chat -> session -> global -> builtin, first non-nil wins
    /// for each field. Returns a fully-populated `ResolvedSettings` plus a
    /// field-keyed trace for debug panels.
    public func resolved(
        sessionId: UUID? = nil,
        chatId: UUID? = nil,
        request: RequestOverride? = nil
    ) -> ResolvedSettings {
        let g = globalCache
        let s = sessionId.flatMap { session($0) }
        let c = chatId.flatMap { chat($0) }
        let r = request

        var out = g   // start from the full global snapshot
        var trace: [String: SettingsTier] = [:]

        // Inference overrides (request > chat > session > global)
        if let v = r?.temperature ?? c?.temperature ?? s?.defaultTemperature {
            out.defaultTemperature = v
            trace["defaultTemperature"] = Self.whichInf(r?.temperature, c?.temperature, s?.defaultTemperature)
        } else {
            trace["defaultTemperature"] = .global
        }
        if let v = r?.topP ?? c?.topP ?? s?.defaultTopP {
            out.defaultTopP = v
            trace["defaultTopP"] = Self.whichInf(r?.topP, c?.topP, s?.defaultTopP)
        } else { trace["defaultTopP"] = .global }
        if let v = r?.topK ?? c?.topK ?? s?.defaultTopK {
            out.defaultTopK = v
            trace["defaultTopK"] = Self.whichInf(r?.topK, c?.topK, s?.defaultTopK)
        } else { trace["defaultTopK"] = .global }
        if let v = r?.minP ?? c?.minP ?? s?.defaultMinP {
            out.defaultMinP = v
            trace["defaultMinP"] = Self.whichInf(r?.minP, c?.minP, s?.defaultMinP)
        } else { trace["defaultMinP"] = .global }
        if let v = r?.repetitionPenalty ?? c?.repetitionPenalty ?? s?.defaultRepetitionPenalty {
            out.defaultRepetitionPenalty = v
            trace["defaultRepetitionPenalty"] = Self.whichInf(
                r?.repetitionPenalty, c?.repetitionPenalty, s?.defaultRepetitionPenalty)
        } else { trace["defaultRepetitionPenalty"] = .global }
        if let v = r?.maxTokens ?? c?.maxTokens ?? s?.defaultMaxTokens {
            out.defaultMaxTokens = v
            trace["defaultMaxTokens"] = Self.whichInf(r?.maxTokens, c?.maxTokens, s?.defaultMaxTokens)
        } else { trace["defaultMaxTokens"] = .global }

        // enableThinking — tri-state (Bool?); both layers are Optional, so we
        // have to pick explicitly.
        if let v = r?.enableThinking {
            out.defaultEnableThinking = v
            trace["defaultEnableThinking"] = .request
        } else if let v = c?.enableThinking {
            out.defaultEnableThinking = v
            trace["defaultEnableThinking"] = .chat
        } else if let v = s?.defaultEnableThinking {
            out.defaultEnableThinking = v
            trace["defaultEnableThinking"] = .session
        } else {
            trace["defaultEnableThinking"] = .global
        }

        // System prompt — if request set it, use that; else chat; else session
        // override; else whatever's in global (which is also String?).
        if let v = r?.systemPrompt {
            out.defaultSystemPrompt = v
            trace["defaultSystemPrompt"] = .request
        } else if let v = c?.systemPrompt {
            out.defaultSystemPrompt = v
            trace["defaultSystemPrompt"] = .chat
        } else if let v = s?.defaultSystemPrompt {
            out.defaultSystemPrompt = v
            trace["defaultSystemPrompt"] = .session
        } else {
            trace["defaultSystemPrompt"] = .global
        }

        // Session-only engine-load overrides (request/chat have no opinion).
        if let v = s?.engineKind { out.engineKind = v; trace["engineKind"] = .session } else { trace["engineKind"] = .global }
        if let v = s?.maxNumSeqs { out.maxNumSeqs = v; trace["maxNumSeqs"] = .session } else { trace["maxNumSeqs"] = .global }
        if let v = s?.prefillStepSize { out.prefillStepSize = v; trace["prefillStepSize"] = .session } else { trace["prefillStepSize"] = .global }
        if let v = s?.memoryCachePercent { out.memoryCachePercent = v; trace["memoryCachePercent"] = .session } else { trace["memoryCachePercent"] = .global }
        if let v = s?.memoryCacheTTLMinutes { out.memoryCacheTTLMinutes = v; trace["memoryCacheTTLMinutes"] = .session } else { trace["memoryCacheTTLMinutes"] = .global }
        if let v = s?.maxCacheBlocks { out.maxCacheBlocks = v; trace["maxCacheBlocks"] = .session } else { trace["maxCacheBlocks"] = .global }
        if let v = s?.enableTurboQuant { out.enableTurboQuant = v; trace["enableTurboQuant"] = .session } else { trace["enableTurboQuant"] = .global }
        if let v = s?.turboQuantBits { out.turboQuantBits = v; trace["turboQuantBits"] = .session } else { trace["turboQuantBits"] = .global }
        if let v = s?.enableJANG { out.enableJANG = v; trace["enableJANG"] = .session } else { trace["enableJANG"] = .global }
        if let v = s?.enablePrefixCache { out.enablePrefixCache = v; trace["enablePrefixCache"] = .session } else { trace["enablePrefixCache"] = .global }
        if let v = s?.enableSSMCompanion { out.enableSSMCompanion = v; trace["enableSSMCompanion"] = .session } else { trace["enableSSMCompanion"] = .global }
        if let v = s?.enableBlockDiskCache { out.enableBlockDiskCache = v; trace["enableBlockDiskCache"] = .session } else { trace["enableBlockDiskCache"] = .global }
        if let v = s?.enableDiskCache { out.enableDiskCache = v; trace["enableDiskCache"] = .session } else { trace["enableDiskCache"] = .global }
        if let v = s?.diskCacheDir { out.diskCacheDir = v; trace["diskCacheDir"] = .session } else { trace["diskCacheDir"] = .global }
        if let v = s?.diskCacheMaxGB { out.diskCacheMaxGB = v; trace["diskCacheMaxGB"] = .session } else { trace["diskCacheMaxGB"] = .global }
        if let v = s?.kvCacheQuantization { out.kvCacheQuantization = v; trace["kvCacheQuantization"] = .session } else { trace["kvCacheQuantization"] = .global }
        if let v = s?.flashMoe { out.flashMoe = v; trace["flashMoe"] = .session } else { trace["flashMoe"] = .global }
        if let v = s?.flashMoeSlotBank { out.flashMoeSlotBank = v; trace["flashMoeSlotBank"] = .session } else { trace["flashMoeSlotBank"] = .global }
        if let v = s?.flashMoePrefetch { out.flashMoePrefetch = v; trace["flashMoePrefetch"] = .session } else { trace["flashMoePrefetch"] = .global }
        if let v = s?.flashMoeIoSplit { out.flashMoeIoSplit = v; trace["flashMoeIoSplit"] = .session } else { trace["flashMoeIoSplit"] = .global }
        if let v = s?.smelt { out.smelt = v; trace["smelt"] = .session } else { trace["smelt"] = .global }
        if let v = s?.smeltExperts { out.smeltExperts = v; trace["smeltExperts"] = .session } else { trace["smeltExperts"] = .global }
        if let v = s?.smeltMode { out.smeltMode = v; trace["smeltMode"] = .session } else { trace["smeltMode"] = .global }
        if let v = s?.dflash { out.dflash = v; trace["dflash"] = .session } else { trace["dflash"] = .global }
        if let v = s?.dflashDrafterPath { out.dflashDrafterPath = v; trace["dflashDrafterPath"] = .session } else { trace["dflashDrafterPath"] = .global }
        if let v = s?.dflashBlockSize { out.dflashBlockSize = v; trace["dflashBlockSize"] = .session } else { trace["dflashBlockSize"] = .global }
        if let v = s?.dflashTopK { out.dflashTopK = v; trace["dflashTopK"] = .session } else { trace["dflashTopK"] = .global }
        if let v = s?.dflashNumPaths { out.dflashNumPaths = v; trace["dflashNumPaths"] = .session } else { trace["dflashNumPaths"] = .global }
        if let v = s?.dflashTapLayers { out.dflashTapLayers = v; trace["dflashTapLayers"] = .session } else { trace["dflashTapLayers"] = .global }
        if let v = s?.dflashTargetHiddenDim { out.dflashTargetHiddenDim = v; trace["dflashTargetHiddenDim"] = .session } else { trace["dflashTargetHiddenDim"] = .global }
        if let v = s?.distributed { out.distributed = v; trace["distributed"] = .session } else { trace["distributed"] = .global }
        if let v = s?.distributedHost { out.distributedHost = v; trace["distributedHost"] = .session } else { trace["distributedHost"] = .global }
        if let v = s?.distributedPort { out.distributedPort = v; trace["distributedPort"] = .session } else { trace["distributedPort"] = .global }
        if let v = s?.distributedMode { out.distributedMode = v; trace["distributedMode"] = .session } else { trace["distributedMode"] = .global }
        if let v = s?.clusterSecret { out.clusterSecret = v; trace["clusterSecret"] = .session } else { trace["clusterSecret"] = .global }
        if let v = s?.workerNodes { out.workerNodes = v; trace["workerNodes"] = .session } else { trace["workerNodes"] = .global }

        if let v = s?.defaultReasoningParser { out.defaultReasoningParser = v; trace["defaultReasoningParser"] = .session } else { trace["defaultReasoningParser"] = .global }
        if let v = s?.defaultToolParser { out.defaultToolParser = v; trace["defaultToolParser"] = .session } else { trace["defaultToolParser"] = .global }
        if let v = s?.enableAutoToolChoice { out.enableAutoToolChoice = v; trace["enableAutoToolChoice"] = .session } else { trace["enableAutoToolChoice"] = .global }
        if let v = s?.chatTemplate { out.chatTemplate = v; trace["chatTemplate"] = .session } else { trace["chatTemplate"] = .global }
        if let v = s?.chatTemplateKwargs { out.chatTemplateKwargs = v; trace["chatTemplateKwargs"] = .session } else { trace["chatTemplateKwargs"] = .global }

        // Server binding
        if let v = s?.host { out.defaultHost = v; trace["defaultHost"] = .session } else { trace["defaultHost"] = .global }
        if let v = s?.port { out.defaultPort = v; trace["defaultPort"] = .session } else { trace["defaultPort"] = .global }
        if let v = s?.lan { out.defaultLAN = v; trace["defaultLAN"] = .session } else { trace["defaultLAN"] = .global }
        if let v = s?.quantizationOverride { out.quantizationOverride = v; trace["quantizationOverride"] = .session } else { trace["quantizationOverride"] = .global }

        // Idle
        if let v = s?.idleEnabled { out.idleEnabled = v; trace["idleEnabled"] = .session } else { trace["idleEnabled"] = .global }
        if let v = s?.idleSoftSec { out.idleSoftSec = v; trace["idleSoftSec"] = .session } else { trace["idleSoftSec"] = .global }
        if let v = s?.idleDeepSec { out.idleDeepSec = v; trace["idleDeepSec"] = .session } else { trace["idleDeepSec"] = .global }

        // Auth
        if let v = s?.apiKey { out.apiKey = v; trace["apiKey"] = .session } else { trace["apiKey"] = .global }
        if let v = s?.adminToken { out.adminToken = v; trace["adminToken"] = .session } else { trace["adminToken"] = .global }
        if let v = s?.corsOrigins { out.corsOrigins = v; trace["corsOrigins"] = .session } else { trace["corsOrigins"] = .global }

        // MCP
        if let v = s?.mcpConfigPath { out.mcpConfigPath = v; trace["mcpConfigPath"] = .session } else { trace["mcpConfigPath"] = .global }
        if let v = s?.mcpServers { out.mcpServers = v; trace["mcpServers"] = .session } else { trace["mcpServers"] = .global }

        // Audit 2026-04-16 UX #3: synchronize `enableTurboQuant` and
        // `kvCacheQuantization`. The SessionConfigForm picker writes
        // `kvCacheQuantization` ("none"/"q4"/"q8"/"turboquant") while
        // `Stream.buildGenerateParameters` reads the `enableTurboQuant`
        // Bool. These two could disagree, leaving TQ on when the picker
        // said "none" or TQ off when the picker said "turboquant". Canonical
        // source of truth is the string; derive the Bool from it.
        // If the string is anything else (q4/q8/none), TQ is off.
        out.enableTurboQuant = (out.kvCacheQuantization.lowercased() == "turboquant")
        return ResolvedSettings(settings: out, trace: trace)
    }

    // MARK: - Helper

    /// Classify which tier contributed a value in the request→chat→session chain.
    /// Called only after we've already confirmed at least one of them is non-nil.
    private static func whichInf<T>(_ r: T?, _ c: T?, _ s: T?) -> SettingsTier {
        if r != nil { return .request }
        if c != nil { return .chat }
        if s != nil { return .session }
        return .global
    }
}

// MARK: - Engine.LoadOptions integration

extension Engine.LoadOptions {
    /// Populate a LoadOptions from a resolved settings snapshot. The caller
    /// is responsible for supplying the modelPath, since that's a per-load
    /// decision not a settings field.
    public init(modelPath: URL, from resolved: ResolvedSettings) {
        self.init(modelPath: modelPath)
        let r = resolved.settings
        self.kind = (r.engineKind == .simple) ? .simple : .batched
        self.maxNumSeqs = r.maxNumSeqs
        self.prefillStepSize = r.prefillStepSize
        self.maxCacheBlocks = r.maxCacheBlocks
        self.enableTurboQuant = r.enableTurboQuant
        self.enableJANG = r.enableJANG
        self.enablePrefixCache = r.enablePrefixCache
        self.enableSSMCompanion = r.enableSSMCompanion
        self.defaultEnableThinking = r.defaultEnableThinking
        self.idleSoftSec = r.idleSoftSec
        self.idleDeepSec = r.idleDeepSec
        self.idleEnabled = r.idleEnabled
        // Cache stack
        self.enableMemoryCache = r.enableMemoryCache
        self.enableDiskCache = r.enableDiskCache
        self.diskCacheDir = r.diskCacheDir
        self.diskCacheMaxGB = r.diskCacheMaxGB
        self.enableBlockDiskCache = r.enableBlockDiskCache
        self.blockDiskCacheDir = r.blockDiskCacheDir
        self.blockDiskCacheMaxGB = r.blockDiskCacheMaxGB
        self.kvCacheQuantization = r.kvCacheQuantization
        self.kvCacheGroupSize = r.kvCacheGroupSize
        self.turboQuantBits = r.turboQuantBits
        self.enableSSMReDerive = r.enableSSMReDerive
        // Smelt + Flash MoE
        self.smelt = r.smelt
        self.smeltExperts = r.smeltExperts
        self.smeltMode = r.smeltMode
        self.flashMoe = r.flashMoe
        self.flashMoeSlotBank = r.flashMoeSlotBank
        self.flashMoePrefetch = r.flashMoePrefetch
        self.flashMoeIoSplit = r.flashMoeIoSplit
        // DFlash
        self.dflash = r.dflash
        self.dflashDrafterPath = r.dflashDrafterPath
        self.dflashBlockSize = r.dflashBlockSize
        self.dflashTopK = r.dflashTopK
        self.dflashNumPaths = r.dflashNumPaths
        self.dflashTapLayers = r.dflashTapLayers
        self.dflashTargetHiddenDim = r.dflashTargetHiddenDim
        // Parser overrides
        self.defaultToolParser = r.defaultToolParser
        self.defaultReasoningParser = r.defaultReasoningParser
    }
}
