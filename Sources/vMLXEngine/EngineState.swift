import Foundation

/// Lifecycle state of the engine. Mirrors the `sessionStatus` enum the Electron
/// app uses across `ChatInterface.tsx`, `SessionCard.tsx`, `SessionView.tsx`.
///
/// The UI binds to `Engine.stateStream` and renders the right banner for each
/// state. Adding a new state here requires updating the banner switch in
/// `vMLXApp/Chat/ChatScreen.swift` AND the status pill in
/// `vMLXApp/Server/ServerScreen.swift` — see SWIFT-NO-REGRESSION-CHECKLIST.
public enum EngineState: Sendable, Equatable {
    case stopped
    case loading(LoadProgress)
    case running
    case standby(StandbyDepth)
    case error(String)

    public enum StandbyDepth: String, Sendable, Equatable {
        case soft   // weights still in unified memory, fast wake
        case deep   // weights unloaded, slow wake
    }
}

/// Determinate progress emitted while the engine loads a model. Mirrors the
/// `session:loadProgress` event the Electron main process derives from engine
/// stdout via the regex table at `panel/src/main/sessions.ts:140-198`.
///
/// Phases match the Python loader pipeline:
///   downloading → reading shards → quantizing → applying weights → warming up
public struct LoadProgress: Sendable, Equatable {
    public var phase: Phase
    /// 0.0 ... 1.0. May be `nil` when the phase has no determinate progress yet.
    public var fraction: Double?
    /// Free-form label shown under the bar, e.g. "shard 4/12" or
    /// "applying JANG repack (layer 23/40)".
    public var label: String

    public enum Phase: String, Sendable, Equatable {
        case downloading
        case reading
        case quantizing
        case applying
        case warmup
        case finalizing
    }

    public init(phase: Phase, fraction: Double? = nil, label: String = "") {
        self.phase = phase
        self.fraction = fraction
        self.label = label
    }

    /// Convenience: a 0-progress placeholder for the start of a phase.
    public static func startingPhase(_ phase: Phase, label: String = "") -> LoadProgress {
        LoadProgress(phase: phase, fraction: 0, label: label)
    }
}

/// One stream output from `Engine.load`. Each event is either a progress
/// update or terminal `.done` / `.failed`.
public enum LoadEvent: Sendable {
    case progress(LoadProgress)
    case done
    case failed(String)
}
