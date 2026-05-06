// vMLX Examples — LagunaRuntime
//
// Drives a Laguna-XS.2 (Poolside agentic-coding MoE, model_type=laguna)
// JANGTQ bundle through the vMLX runtime stack with full cache support.
//
// Why this example exists:
//   1. Laguna has hybrid SWA + full-attention layers with PER-LAYER
//      head count (48 full / 64 SWA) and dual RoPE flavors (YaRN on
//      full, default on SWA). The runtime auto-wires those via the
//      Swift `Laguna` model in `vMLXLLM/Models/`. This example
//      verifies that wiring with a real bundle.
//   2. Laguna uses the GLM-thinking-v5 chat template — `<think>`
//      family. The ReasoningParser handles the `laguna` prefix
//      (see `ReasoningParser.swift`).
//   3. Tool-call format is GLM-4 style (`<tool_call>...</tool_call>`).
//      Routed via `ToolCallFormat.glm4` for `laguna` model_type.
//   4. TurboQuant KV compression engages on the SWA layers; full-attn
//      layers stay float since their cache is bounded by sliding
//      window size rather than the model's full context.
//
// Build & run:
//   swift run --package-path /Users/eric/vmlx/swift LagunaRuntime \
//       /Users/eric/.mlxstudio/models/_bundles/Laguna-XS.2-JANGTQ
//
// Notes on the cache cascade for Laguna:
//   • L1 disk cache stores the post-prefill float KV per turn.
//   • BlockDiskCache storage exists but is not in the live fetch/store path.
//     Prompt-level DiskCache is the persistent tier today.
//   • SSM re-derive does not fire for Laguna (no Mamba layers) but
//     the path stays enabled to keep config consistent.

import Foundation
import vMLXEngine
import vMLXLMCommon
import RuntimeShared
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

func flushAndExit(_ code: Int32) -> Never {
    fflush(nil)
    exit(code)
}

func say(_ line: String) {
    FileHandle.standardError.write(Data((line + "\n").utf8))
}

let bundlePath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : (FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".mlxstudio/models/_bundles/Laguna-XS.2-JANGTQ").path)

let bundle = URL(fileURLWithPath: bundlePath)
guard FileManager.default.fileExists(atPath: bundle.path) else {
    print("bundle not found: \(bundle.path)")
    exit(2)
}

let cacheDir = FileManager.default.temporaryDirectory
    .appendingPathComponent("vmlx-examples-laguna")
let slidingWindowMode = ProcessInfo.processInfo.environment["VMLX_LAGUNA_SLIDING_MODE"]
    ?? "auto"

print("=== Laguna runtime example ===")
print("bundle: \(bundle.lastPathComponent)")
print("cache:  \(cacheDir.path)")

let opts = RuntimeShared.makeLoadOptions(
    bundle: bundle,
    cacheDir: cacheDir,
    kvCacheBits: 4,
    kvCacheGroupSize: 64,
    slidingWindowMode: slidingWindowMode
)
RuntimeShared.reportLoadOptions(opts)

let engine = Engine()
let t0 = Date()
try await RuntimeShared.awaitLoad(engine, options: opts)
print("[runtime] loaded in \(Int(Date().timeIntervalSince(t0))) s")

if ProcessInfo.processInfo.environment["VMLX_LAGUNA_LOOP"] == "1" {
    say("\n--- long file-tree loop probe / real Engine.stream path ---")
    let env = ProcessInfo.processInfo.environment
    let prompt = try lagunaLoopPrompt(env: env)
    say("[loop.prompt] chars=\(prompt.count)")
    let maxTokens = Int(env["VMLX_LAGUNA_MAX_TOKENS"] ?? "512") ?? 512
    let temperature = Double(env["VMLX_LAGUNA_TEMP"] ?? "0.6") ?? 0.6
    let topP = Double(env["VMLX_LAGUNA_TOP_P"] ?? "0.95") ?? 0.95
    let rep = Double(env["VMLX_LAGUNA_REP"] ?? "1.15") ?? 1.15

    func runLoopCase(_ thinking: Bool) async throws {
        say("[loop.\(thinking ? "thinking_on" : "thinking_off")] starting")
        let req = ChatRequest(
            model: "in-process",
            messages: [RuntimeShared.userMsg(prompt)],
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: rep,
            enableThinking: thinking,
            includeReasoning: true)
        let start = Date()
        let (reasoning, content, tools) = try await RuntimeShared.drainStream(
            engine, req, printContent: false, printReasoningTick: false)
        say("[loop.\(thinking ? "thinking_on" : "thinking_off")] stream drained")
        let elapsed = Date().timeIntervalSince(start)
        let combined = content.isEmpty ? reasoning : content
        say(String(format:
            "[loop.%@] elapsed=%.2fs content=%d reasoning=%d tools=%d loop=%@",
            thinking ? "thinking_on" : "thinking_off",
            elapsed, content.count, reasoning.count, tools.count,
            lagunaLoopDetected(combined) ? "YES" : "NO"))
        let preview = combined.count > 600
            ? String(combined.prefix(600)) + "..."
            : combined
        say(preview.replacingOccurrences(of: "\n", with: "\\n"))
    }

    try await runLoopCase(false)
    try await runLoopCase(true)
    say("\n=== laguna loop probe done ===")
    flushAndExit(0)
}

if ProcessInfo.processInfo.environment["VMLX_LAGUNA_BENCH"] == "1" {
    print("\n--- decode bench / real Engine.stream path ---")
    let bench = await engine.benchmark(suite: .decode256)
    for try await event in bench {
        switch event {
        case .progress(let fraction, let label):
            print("[bench.progress] \(String(format: "%.0f", fraction * 100))% \(label)")
        case .done(let report):
            print("[bench.done] tokensPerSec=\(String(format: "%.2f", report.tokensPerSec)) generationTps=\(String(format: "%.2f", report.generationTps ?? 0)) ttftMs=\(String(format: "%.0f", report.ttftMs)) totalMs=\(String(format: "%.0f", report.totalMs)) notes=\(report.notes)")
        case .failed(let msg):
            print("[bench.failed] \(msg)")
            flushAndExit(3)
        }
    }
    print("\n=== bench done ===")
    flushAndExit(0)
}

if ProcessInfo.processInfo.environment["VMLX_LAGUNA_QUICK"] == "1" {
    print("\n--- quick hi / thinking off ---")
    let off = RuntimeShared.makeRequest(
        [RuntimeShared.userMsg("hi")],
        maxTokens: 96,
        temperature: 0.0,
        enableThinking: false
    )
    let (offReasoning, offContent, _) = try await RuntimeShared.drainStream(engine, off)
    print("[quick.off.reasoning.count] \(offReasoning.count)")
    print("[quick.off.content] \(offContent)")
    RuntimeShared.assertNoLeak(offContent)

    print("\n--- quick hi / thinking on ---")
    let on = RuntimeShared.makeRequest(
        [RuntimeShared.userMsg("hi")],
        maxTokens: 160,
        temperature: 0.6,
        topP: 0.95,
        enableThinking: true
    )
    let (onReasoning, onContent, _) = try await RuntimeShared.drainStream(
        engine, on, printContent: true, printReasoningTick: true)
    print("[quick.on.reasoning.count] \(onReasoning.count)")
    print("[quick.on.content] \(onContent)")
    RuntimeShared.assertNoLeak(onContent)
    print("\n=== quick done ===")
    flushAndExit(0)
}

// MARK: 1 — Plain code completion (FIM-style with chat template)
do {
    print("\n--- code completion ---")
    let prompt = """
    Refactor the following Python function into a one-liner using a list comprehension and the `**` power operator:

    def sum_even_squares(xs):
        total = 0
        for x in xs:
            if x % 2 == 0:
                total += x ** 2
        return total
    """
    let req = RuntimeShared.makeRequest(
        [RuntimeShared.userMsg(prompt)],
        maxTokens: 256, temperature: 0.0, enableThinking: false
    )
    let (_, content, _) = try await RuntimeShared.drainStream(engine, req)
    RuntimeShared.assertNoLeak(content)
    await RuntimeShared.reportCacheStats(engine)
}

// MARK: 2 — Thinking-mode ON
do {
    print("\n--- thinking on ---")
    let prompt = "I have a bug: my Python `binary_search` returns the wrong index for the last element. Walk through what could be wrong, then give the fixed code."
    let req = RuntimeShared.makeRequest(
        [RuntimeShared.userMsg(prompt)],
        maxTokens: 1024, temperature: 0.6, topP: 0.95, enableThinking: true
    )
    let (reasoning, content, _) = try await RuntimeShared.drainStream(
        engine, req, printContent: false, printReasoningTick: true)
    print("\nreasoning: \(reasoning.count) ch")
    print("content:   \(content.prefix(200))")
    RuntimeShared.assertNoLeak(content)
    await RuntimeShared.reportCacheStats(engine)
}

// MARK: 3 — Tool calling (GLM-4 style)
do {
    print("\n--- tool calling ---")
    let sys = """
    You have access to one tool: read_file(path: string) -> string.
    Always emit a tool call when the user asks to read a file.
    Use GLM-4 syntax: <tool_call>{"name":"read_file","arguments":{"path":"..."}}</tool_call>
    """
    let req = RuntimeShared.makeRequest(
        [RuntimeShared.systemMsg(sys),
         RuntimeShared.userMsg("Read the README at ./project/README.md and tell me what it says.")],
        maxTokens: 256, temperature: 0.0
    )
    let (_, content, tools) = try await RuntimeShared.drainStream(engine, req, printContent: false)
    print("[TOOLS PARSED] \(tools)")
    print("content:    \(content.prefix(120))")
    if tools.isEmpty {
        print("[hint] If no tool calls fired, prompt may need stronger few-shot.")
    }
}

// MARK: 4 — Multi-turn over a small "agentic" plan
do {
    print("\n--- multi-turn agentic ---")
    var msgs: [ChatRequest.Message] = [
        RuntimeShared.systemMsg("You are a senior engineer. Plan in 3 steps."),
    ]
    let asks = [
        "Outline a unit test plan for a function that sorts a list of dicts by `priority` then `created_at`.",
        "Refine step 1 to use pytest parametrize syntax.",
        "Now write the actual test file.",
    ]
    for ask in asks {
        msgs.append(RuntimeShared.userMsg(ask))
        let req = RuntimeShared.makeRequest(msgs, maxTokens: 384, temperature: 0.0)
        let (_, ans, _) = try await RuntimeShared.drainStream(engine, req, printContent: false)
        print("\n>>> \(ask)\n<<< \(ans.prefix(200))…")
        msgs.append(RuntimeShared.assistantMsg(ans))
    }
    await RuntimeShared.reportCacheStats(engine)
    print("[note] blockDisk.wired=false today; prompt-level DiskCache is the persistent tier.")
}

print("\n=== done ===")

private func lagunaLoopPrompt(env: [String: String]) throws -> String {
    if let promptFile = env["VMLX_LAGUNA_PROMPT_FILE"], !promptFile.isEmpty {
        return try String(contentsOfFile: promptFile, encoding: .utf8)
    }
    let treePath = env["VMLX_LAGUNA_TREE_PATH"] ?? "/Users/eric/vmlx-swift-lm"
    let root = URL(fileURLWithPath: treePath).standardizedFileURL
    let limit = Int(env["VMLX_LAGUNA_TREE_LIMIT"] ?? "420") ?? 420
    var entries: [String] = []
    if let enumerator = FileManager.default.enumerator(
        at: root,
        includingPropertiesForKeys: [.isDirectoryKey],
        options: [.skipsHiddenFiles, .skipsPackageDescendants])
    {
        for case let url as URL in enumerator {
            let path = url.standardizedFileURL.path
            guard path.hasPrefix(root.path) else { continue }
            var rel = String(path.dropFirst(root.path.count))
            if rel.hasPrefix("/") { rel.removeFirst() }
            if rel.isEmpty { continue }
            if rel.hasPrefix(".build/") || rel.hasPrefix("DerivedData/") {
                continue
            }
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey])
            let isDir = values?.isDirectory ?? false
            entries.append("- \(rel)\(isDir ? "/" : "")")
            if entries.count >= max(1, limit) { break }
        }
    }
    entries.sort()
    return """
    For each file in this folder, summarize in one line what it does.

    file_tree path: \(root.path)
    \(entries.joined(separator: "\n"))
    """
}

private func lagunaLoopDetected(_ text: String) -> Bool {
    let normalized = text
        .lowercased()
        .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        .trimmingCharacters(in: .whitespacesAndNewlines)
    let words = normalized.split(separator: " ").map(String.init)
    guard words.count >= 18 else { return false }
    let maxWidth = min(32, words.count / 3)
    if maxWidth >= 6 {
        for width in 6...maxWidth {
            let tail = Array(words.suffix(width * 3))
            if Array(tail[0..<width]) == Array(tail[width..<(width * 2)]),
               Array(tail[width..<(width * 2)]) == Array(tail[(width * 2)..<(width * 3)])
            {
                return true
            }
        }
    }
    return false
}
