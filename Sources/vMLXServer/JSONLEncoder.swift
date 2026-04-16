import Foundation
import Hummingbird
import NIOCore
import vMLXEngine

/// Ollama-compatible NDJSON streaming encoder.
///
/// Wire format (mirrors `vmlx_engine/api/ollama_adapter.py::openai_chat_chunk_to_ollama_ndjson`):
///
///     {"model":"<name>","created_at":"2026-04-13T00:00:00.000Z","message":{"role":"assistant","content":"Hi"},"done":false}\n
///     ...
///     {"model":"<name>","created_at":"...","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop",
///      "total_duration":0,"prompt_eval_count":N,"eval_count":M}\n
public enum JSONLEncoder {

    public static func ollamaChatStream(
        model: String,
        upstream: AsyncThrowingStream<StreamChunk, Error>
    ) -> ResponseBody {
        ResponseBody { writer in
            let allocator = ByteBufferAllocator()
            let createdAt = Self.iso8601Now()

            var finishReason: String? = nil
            var lastUsage: StreamChunk.Usage? = nil
            // Tool-call buffer: VS Code Copilot and Open WebUI read
            // tool_calls ONLY from the final done:true chunk with
            // done_reason="tool_calls". Inline tool_calls on delta chunks
            // are ignored by those clients. Mirrors the stateful wrapper
            // from vmlx_engine/api/ollama_adapter.py added in v1.3.50.
            var toolCallBuffer: [[String: Any]] = []

            do {
                for try await chunk in upstream {
                    var message: [String: Any] = ["role": "assistant", "content": chunk.content ?? ""]
                    if let reasoning = chunk.reasoning, !reasoning.isEmpty {
                        message["thinking"] = reasoning
                    }
                    if let tcs = chunk.toolCalls, !tcs.isEmpty {
                        let encoded = tcs.map { tc -> [String: Any] in
                            let args = Self.parseJSONObject(tc.function.arguments) ?? [:]
                            return [
                                "function": [
                                    "name": tc.function.name,
                                    "arguments": args,
                                ] as [String: Any]
                            ]
                        }
                        // Keep delta emission for clients that read inline,
                        // AND buffer for the final chunk.
                        message["tool_calls"] = encoded
                        toolCallBuffer.append(contentsOf: encoded)
                    }
                    let obj: [String: Any] = [
                        "model": model,
                        "created_at": createdAt,
                        "message": message,
                        "done": false,
                    ]
                    try await writer.write(allocator.buffer(string: SSEEncoder.asciiJSON(obj) + "\n"))
                    if let fr = chunk.finishReason { finishReason = fr }
                    if let u = chunk.usage { lastUsage = u }
                }
            } catch {
                let errObj: [String: Any] = [
                    "model": model,
                    "created_at": createdAt,
                    "error": "\(error)",
                    "done": true,
                ]
                try await writer.write(allocator.buffer(string: SSEEncoder.asciiJSON(errObj) + "\n"))
                try await writer.finish(nil)
                return
            }

            // Final chunk: splice buffered tool_calls into message and
            // upgrade done_reason to "tool_calls" when any were seen.
            // This is the shape Copilot/Open WebUI dispatch on.
            var finalMessage: [String: Any] = ["role": "assistant", "content": ""]
            var doneReason = finishReason ?? "stop"
            if !toolCallBuffer.isEmpty {
                finalMessage["tool_calls"] = toolCallBuffer
                doneReason = "tool_calls"
            }
            var finalObj: [String: Any] = [
                "model": model,
                "created_at": createdAt,
                "message": finalMessage,
                "done": true,
                "done_reason": doneReason,
            ]
            if let u = lastUsage {
                finalObj["prompt_eval_count"] = u.promptTokens
                finalObj["eval_count"] = u.completionTokens
            }
            try await writer.write(allocator.buffer(string: SSEEncoder.asciiJSON(finalObj) + "\n"))
            try await writer.finish(nil)
        }
    }

    /// NDJSON encoder for Ollama's `/api/generate` (text-completion).
    ///
    /// Same wire shape as `ollamaChatStream` but emits `response:"..."`
    /// per token instead of a nested `message:{role,content}`. Used by
    /// Ollama CLI `ollama generate`.
    public static func ollamaGenerateStream(
        model: String,
        upstream: AsyncThrowingStream<StreamChunk, Error>
    ) -> ResponseBody {
        ResponseBody { writer in
            let allocator = ByteBufferAllocator()
            let createdAt = Self.iso8601Now()
            var finishReason: String? = nil
            var lastUsage: StreamChunk.Usage? = nil

            do {
                for try await chunk in upstream {
                    let response = chunk.content ?? ""
                    var obj: [String: Any] = [
                        "model": model,
                        "created_at": createdAt,
                        "response": response,
                        "done": false,
                    ]
                    if let reasoning = chunk.reasoning, !reasoning.isEmpty {
                        obj["thinking"] = reasoning
                    }
                    try await writer.write(allocator.buffer(string: SSEEncoder.asciiJSON(obj) + "\n"))
                    if let fr = chunk.finishReason { finishReason = fr }
                    if let u = chunk.usage { lastUsage = u }
                }
            } catch {
                let errObj: [String: Any] = [
                    "model": model,
                    "created_at": createdAt,
                    "error": "\(error)",
                    "done": true,
                ]
                try await writer.write(allocator.buffer(string: SSEEncoder.asciiJSON(errObj) + "\n"))
                try await writer.finish(nil)
                return
            }

            var finalObj: [String: Any] = [
                "model": model,
                "created_at": createdAt,
                "response": "",
                "done": true,
                "done_reason": finishReason ?? "stop",
            ]
            if let u = lastUsage {
                finalObj["prompt_eval_count"] = u.promptTokens
                finalObj["eval_count"] = u.completionTokens
            }
            try await writer.write(allocator.buffer(string: SSEEncoder.asciiJSON(finalObj) + "\n"))
            try await writer.finish(nil)
        }
    }

    public static func iso8601Now() -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: Date())
    }

    static func parseJSONObject(_ s: String) -> [String: Any]? {
        guard let data = s.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return obj
    }
}

extension ByteBufferAllocator {
    fileprivate func buffer(string: String) -> ByteBuffer {
        var b = self.buffer(capacity: string.utf8.count)
        b.writeString(string)
        return b
    }
}
