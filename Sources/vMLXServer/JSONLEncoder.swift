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

            // 2026-04-18 heartbeat — Ollama NDJSON clients can't parse
            // SSE comments, but they DO tolerate empty-content "ping"
            // objects that look like any other delta (Ollama itself
            // emits these during reasoning). When upstream is silent
            // for > interval, emit a `{"message":{"content":""},"done":false}`
            // line so the TCP connection stays warm through a 20-40 s
            // thinking-model prefill + reasoning phase. Same env
            // override as SSE (`VMLX_SSE_HEARTBEAT_SEC`).
            let merged = sseMergeWithHeartbeat(
                upstream: upstream, interval: sseHeartbeatInterval)
            do {
                for try await event in merged {
                    switch event {
                    case .heartbeat:
                        let hb: [String: Any] = [
                            "model": model,
                            "created_at": createdAt,
                            "message": ["role": "assistant", "content": ""] as [String: Any],
                            "done": false,
                        ]
                        try await writer.write(
                            allocator.buffer(string: SSEEncoder.asciiJSON(hb) + "\n"))
                        continue
                    case .chunk(let chunk):
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
            Self.applyOllamaTimings(into: &finalObj, usage: lastUsage)
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

            let merged = sseMergeWithHeartbeat(
                upstream: upstream, interval: sseHeartbeatInterval)
            do {
                for try await event in merged {
                    switch event {
                    case .heartbeat:
                        let hb: [String: Any] = [
                            "model": model,
                            "created_at": createdAt,
                            "response": "",
                            "done": false,
                        ]
                        try await writer.write(
                            allocator.buffer(string: SSEEncoder.asciiJSON(hb) + "\n"))
                        continue
                    case .chunk(let chunk):
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
            Self.applyOllamaTimings(into: &finalObj, usage: lastUsage)
            try await writer.write(allocator.buffer(string: SSEEncoder.asciiJSON(finalObj) + "\n"))
            try await writer.finish(nil)
        }
    }

    public static func iso8601Now() -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: Date())
    }

    /// iter-64: shared timing-envelope emitter used by both the chat
    /// and generate streaming encoders. Non-streaming `/api/generate`
    /// already landed this mapping inline (§93). Mirrors that logic
    /// so SSE and NDJSON clients see the same four nanosecond fields
    /// on the final `done:true` chunk — required for LangChain,
    /// Copilot, Open WebUI, OllamaJS latency UIs.
    static func applyOllamaTimings(
        into obj: inout [String: Any],
        usage: StreamChunk.Usage?
    ) {
        guard let u = usage else { return }
        obj["prompt_eval_count"] = u.promptTokens
        obj["eval_count"] = u.completionTokens
        if let totalMs = u.totalMs {
            obj["total_duration"] = Int64(totalMs * 1_000_000)
        }
        if let prefillMs = u.prefillMs {
            obj["prompt_eval_duration"] = Int64(prefillMs * 1_000_000)
            if let totalMs = u.totalMs {
                let evalMs = max(0, totalMs - prefillMs)
                obj["eval_duration"] = Int64(evalMs * 1_000_000)
            }
        }
        obj["load_duration"] = 0
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
