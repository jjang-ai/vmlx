import Foundation

// MARK: - ModelRegistry
//
// Canonical name → loader function map. Mirrors the Python
// `SUPPORTED_MODELS` + `EDIT_MODELS` + `_NAME_TO_CLASS` tables from
// `vmlx_engine/image_gen.py`.
//
// New models register themselves via `ModelRegistry.register(_:)` in
// their own file — no central edit needed to add a variant.

public struct ModelEntry: Sendable {
    public let name: String              // canonical key, e.g. "flux1-schnell"
    public let displayName: String       // human-friendly, e.g. "FLUX.1 Schnell"
    public let kind: ModelKind
    public let defaultSteps: Int
    public let defaultGuidance: Float
    public let supportsLoRA: Bool
    /// Loader — constructs the concrete model from a local weights dir +
    /// optional quantization bits. Called from inside `FluxEngine` actor.
    public let loader: @Sendable (URL, Int?) async throws -> any FluxModel

    public init(
        name: String,
        displayName: String,
        kind: ModelKind,
        defaultSteps: Int,
        defaultGuidance: Float,
        supportsLoRA: Bool = false,
        loader: @Sendable @escaping (URL, Int?) async throws -> any FluxModel
    ) {
        self.name = name
        self.displayName = displayName
        self.kind = kind
        self.defaultSteps = defaultSteps
        self.defaultGuidance = defaultGuidance
        self.supportsLoRA = supportsLoRA
        self.loader = loader
    }
}

public enum ModelRegistry {
    nonisolated(unsafe) private static var entries: [String: ModelEntry] = [:]
    private static let lock = NSLock()

    /// Register a model. Call from the model's own file at module-load
    /// time (via a `static let _register: Void = { ModelRegistry.register(...) }()`
    /// idiom) so the registry stays decentralized.
    public static func register(_ entry: ModelEntry) {
        lock.lock()
        defer { lock.unlock() }
        entries[entry.name] = entry
    }

    /// Canonical-name lookup.
    public static func lookup(name: String) -> ModelEntry? {
        lock.lock()
        defer { lock.unlock() }
        return entries[name]
    }

    /// Fuzzy lookup — normalizes case, strips HF org prefix, strips
    /// `-<N>bit` quantization suffix, then does a canonical lookup.
    /// Also strips `.` and `_` separators since users may write
    /// `FLUX.1-schnell` (display form) or `flux1-schnell` (canonical).
    /// Mirrors the Python normalize-then-resolve flow.
    public static func lookupFuzzy(name: String) -> ModelEntry? {
        var key = name.lowercased()
        if let slash = key.lastIndex(of: "/") {
            key = String(key[key.index(after: slash)...])
        }
        // Strip `-4bit` / `-8bit` / `-3bit` suffix.
        if let match = key.range(of: "-\\d+bit$", options: .regularExpression) {
            key.removeSubrange(match)
        }
        // Direct hit first.
        if let entry = lookup(name: key) { return entry }
        // Collapse `flux.1-schnell` → `flux1-schnell`.
        let collapsed = key.replacingOccurrences(of: ".", with: "")
                           .replacingOccurrences(of: "_", with: "-")
        return lookup(name: collapsed)
    }

    /// Enumerate all registered models (for UI listing).
    public static func all() -> [ModelEntry] {
        lock.lock()
        defer { lock.unlock() }
        return Array(entries.values).sorted { $0.name < $1.name }
    }

    /// Filter by capability.
    public static func all(kind: ModelKind) -> [ModelEntry] {
        all().filter { $0.kind == kind }
    }
}
