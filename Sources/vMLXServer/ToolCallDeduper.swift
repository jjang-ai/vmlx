import Foundation
import vMLXEngine

enum ToolCallDeduper {
    private static let separator = "\u{1f}"

    static func key(_ call: ChatRequest.ToolCall) -> String {
        [
            call.id,
            call.type,
            call.function.name,
            call.function.arguments,
        ].joined(separator: separator)
    }

    @discardableResult
    static func insert(_ call: ChatRequest.ToolCall, seen: inout Set<String>) -> Bool {
        seen.insert(key(call)).inserted
    }

    @discardableResult
    static func appendUnique(
        _ calls: [ChatRequest.ToolCall],
        to output: inout [ChatRequest.ToolCall],
        seen: inout Set<String>
    ) -> [ChatRequest.ToolCall] {
        var added: [ChatRequest.ToolCall] = []
        for call in calls where insert(call, seen: &seen) {
            output.append(call)
            added.append(call)
        }
        return added
    }
}
