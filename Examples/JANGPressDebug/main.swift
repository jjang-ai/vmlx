// vMLX Examples — JANGPressDebug
//
// Debug harness for the iter-13 finding that `engine.stream` returned
// 0 chunks. Loads a bundle, runs ONE simple request, and prints
// every StreamChunk verbatim — content, reasoning, toolCalls,
// toolStatus — so we can see where the tokens went.
//
// USAGE
//   swift run JANGPressDebug <bundle> [thinking=0|1]

import Foundation
import vMLXEngine
import vMLXLMCommon
import RuntimeShared

let bundlePath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "/Users/eric/.mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ"
let enableThinking = CommandLine.arguments.count > 2
    ? (CommandLine.arguments[2] == "1") : false

let bundleURL = URL(fileURLWithPath: bundlePath)
guard FileManager.default.fileExists(atPath: bundleURL.path) else {
    print("bundle not found"); exit(2)
}

print("=== JANGPressDebug: \(bundleURL.lastPathComponent) (thinking=\(enableThinking)) ===\n")

let cacheDir = FileManager.default.temporaryDirectory
    .appendingPathComponent("jangpress-debug")

var opts = RuntimeShared.makeLoadOptions(
    bundle: bundleURL, cacheDir: cacheDir,
    kvCacheBits: 4, kvCacheGroupSize: 64, slidingWindowMode: "long")
// JangPress OFF for the baseline debug
opts.enableJangPress = false

print("Loading...")
let engine = Engine()
try await RuntimeShared.awaitLoad(engine, options: opts)
print("Loaded.\n")

// Build a request EXACTLY how the existing example does it.
let req = ChatRequest(
    model: "in-process",
    messages: [ChatRequest.Message(role: "user", content: .string("What is 2+2? Reply with just the number."))],
    maxTokens: 32,
    temperature: 0.0,
    enableThinking: enableThinking,
    includeReasoning: true)

print("Streaming a single request...")
print("Request: maxTokens=\(req.maxTokens ?? -1) thinking=\(req.enableThinking ?? false)\n")

var chunkIdx = 0
var totalContent = ""
var totalReasoning = ""
let stream = await engine.stream(request: req)
for try await chunk in stream {
    chunkIdx += 1
    let c = chunk.content
    let r = chunk.reasoning
    let tc = chunk.toolCalls?.count ?? 0
    let tcd = chunk.toolCallDelta != nil ? "yes" : "no"
    let ts = chunk.toolStatus != nil ? "yes" : "no"
    let fr = chunk.finishReason ?? "nil"
    let usage = chunk.usage.map { u -> String in
        "p=\(u.promptTokens) c=\(u.completionTokens) cached=\(u.cachedTokens) tps=\(u.tokensPerSecond.map { String(format: "%.1f", $0) } ?? "nil") partial=\(u.isPartial)"
    } ?? "nil"
    let lp = chunk.logprobs?.count ?? 0

    print("[\(chunkIdx)] content=\(c.map { "\"\($0.prefix(60))\"" } ?? "nil")")
    print("    reasoning=\(r.map { "\"\($0.prefix(60))\"" } ?? "nil")")
    print("    toolCalls=\(tc)  toolCallDelta=\(tcd)  toolStatus=\(ts)  logprobs=\(lp)")
    print("    finishReason=\(fr)")
    print("    usage=\(usage)")

    if let c = c { totalContent += c }
    if let r = r { totalReasoning += r }
}

print("\n=== TOTALS ===")
print("Total chunks streamed: \(chunkIdx)")
print("content (\(totalContent.count) chars): \"\(totalContent)\"")
print("reasoning (\(totalReasoning.count) chars): \"\(totalReasoning.prefix(200))\"")
