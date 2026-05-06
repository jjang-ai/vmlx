// SPDX-License-Identifier: Apache-2.0
//
// WakeNoticeBox — actor-isolated buffer for wake-from-standby notices
// captured during `Engine.wakeFromStandby(emitNotice:)`.
//
// Routes that emit SSE responses can want to prepend a leading comment
// line ("Waking model from deep sleep — replaying load…") so callers
// staring at a hung HTTP connection during a 5–30 s deep wake see
// "something is happening" instead of a silent stall.
//
// The Engine fires `emitNotice` BEFORE spawning the wake task. The route
// then reads the buffered notices via `snapshot()` and passes them into
// `SSEEncoder.chatCompletionStream(leadingComments:)` (or the equivalent
// helper on Anthropic / Ollama). For non-streaming routes we still call
// the closure (it's harmless), but the snapshot is discarded — there's
// no place to surface a pre-message in a one-shot JSON response.

import Foundation

/// Sendable, actor-isolated string buffer. Used to capture the wake
/// notices fired by `Engine.wakeFromStandby(emitNotice:)`.
public actor WakeNoticeBox {
    private var notices: [String] = []

    public init() {}

    public func append(_ msg: String) {
        notices.append(msg)
    }

    /// Return the captured notices and reset the buffer. Cheap to call
    /// even when empty — most requests don't trigger a wake.
    public func snapshot() -> [String] {
        let snap = notices
        notices.removeAll()
        return snap
    }
}
