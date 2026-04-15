import Foundation
import vMLXEngine

/// Persisted chat session.
struct ChatSession: Identifiable, Codable, Hashable {
    var id: UUID
    var title: String
    var modelPath: String?
    var createdAt: Date
    var updatedAt: Date

    init(id: UUID = UUID(),
         title: String = "New chat",
         modelPath: String? = nil,
         createdAt: Date = .now,
         updatedAt: Date = .now) {
        self.id = id
        self.title = title
        self.modelPath = modelPath
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

/// Tool-call lifecycle state for the inline tool-call card UI. Mirrors
/// `StreamChunk.ToolStatus.Phase` but as a persisted enum on the chat
/// message so the card can re-render its pill after a reload.
enum ToolCallStatus: String, Codable, Hashable {
    case pending, running, done, error
}

/// Inline tool-call metadata surfaced in chat bubbles. Parsed from
/// `ChatMessage.toolCallsJSON` so persisted rows survive restart and the
/// renderer doesn't have to hand-decode JSON.
struct InlineToolCall: Identifiable, Hashable {
    var id: String        // tool_call.id
    var name: String      // function name
    var arguments: String // JSON args string (may be partial mid-stream)
    var status: ToolCallStatus
    var output: String?   // stdout/stderr once done
    var exitCode: Int?
}

/// Persisted chat message.
struct ChatMessage: Identifiable, Codable, Hashable {
    enum Role: String, Codable { case system, user, assistant, tool }

    var id: UUID
    var sessionId: UUID
    var role: Role
    var content: String
    var reasoning: String?
    var imageData: [Data]        // inline base64-decoded images
    var toolCallsJSON: String?   // raw tool_calls array JSON
    /// Lifecycle phase keyed by `tool_call.id`. Updated from streaming
    /// `StreamChunk.ToolStatus` events. Persisted alongside the raw
    /// tool-calls JSON so InlineToolCallCard can re-render its pill after
    /// a reload.
    var toolStatuses: [String: ToolCallStatus] = [:]
    var createdAt: Date
    var isStreaming: Bool

    /// Per-message metrics surfaced by the metrics strip under each assistant
    /// turn. Transient — not persisted to SQLite (matches Electron behavior:
    /// metrics live only for the duration of the in-memory message). Manually
    /// excluded from Codable + Hashable to avoid a schema migration.
    var usage: StreamChunk.Usage? = nil

    enum CodingKeys: String, CodingKey {
        case id, sessionId, role, content, reasoning, imageData, toolCallsJSON
        case toolStatuses, createdAt, isStreaming
    }

    static func == (lhs: ChatMessage, rhs: ChatMessage) -> Bool {
        lhs.id == rhs.id && lhs.sessionId == rhs.sessionId && lhs.role == rhs.role &&
        lhs.content == rhs.content && lhs.reasoning == rhs.reasoning &&
        lhs.imageData == rhs.imageData && lhs.toolCallsJSON == rhs.toolCallsJSON &&
        lhs.toolStatuses == rhs.toolStatuses &&
        lhs.createdAt == rhs.createdAt && lhs.isStreaming == rhs.isStreaming
    }
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(content)
        hasher.combine(isStreaming)
    }

    init(id: UUID = UUID(),
         sessionId: UUID,
         role: Role,
         content: String = "",
         reasoning: String? = nil,
         imageData: [Data] = [],
         toolCallsJSON: String? = nil,
         toolStatuses: [String: ToolCallStatus] = [:],
         createdAt: Date = .now,
         isStreaming: Bool = false) {
        self.id = id
        self.sessionId = sessionId
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.imageData = imageData
        self.toolCallsJSON = toolCallsJSON
        self.toolStatuses = toolStatuses
        self.createdAt = createdAt
        self.isStreaming = isStreaming
    }

    /// Decoded tool-call list for inline cards. Empty when nothing is set.
    var inlineToolCalls: [InlineToolCall] {
        guard let json = toolCallsJSON,
              let data = json.data(using: .utf8),
              let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else { return [] }
        return arr.compactMap { raw in
            guard let id = raw["id"] as? String else { return nil }
            let fn = (raw["function"] as? [String: Any]) ?? [:]
            let name = (fn["name"] as? String) ?? "function"
            let args: String
            if let s = fn["arguments"] as? String {
                args = s
            } else if let obj = fn["arguments"] {
                args = (try? String(data: JSONSerialization.data(withJSONObject: obj), encoding: .utf8)) ?? ""
            } else {
                args = ""
            }
            let status = toolStatuses[id] ?? .pending
            return InlineToolCall(
                id: id, name: name, arguments: args,
                status: status, output: nil, exitCode: nil
            )
        }
    }
}
