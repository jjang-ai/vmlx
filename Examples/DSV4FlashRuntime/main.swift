// vMLX Examples — DSV4FlashRuntime
//
// Drives a DeepSeek-V4-Flash JANGTQ bundle through the full vMLX
// runtime stack:
//
//   1. Load via `Engine.load` with prefix cache + prompt-level disk cache
//      + SSM re-derive turned ON.
//   2. Three modes — `chat`, `think`, `think_max` — through the
//      DeepseekV4ChatEncoder + ReasoningParser + ToolCallProcessor
//      (DSML format).
//   3. After each turn, print cache stats so the reader can confirm
//      prefix/disk hits and cache-family status.
//   4. Verify reasoning / content split has zero tag leak.
//
// Build & run:
//   swift run --package-path /Users/eric/vmlx/swift DSV4FlashRuntime \
//       /Users/eric/.mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ
//
// Architecture notes:
//   • DSV4-Flash runs HSA + CSA + SWA tri-mode attention. The Engine
//     auto-wires `DSV4LayerCache` for `compress_ratio>0` HSA/CSA layers
//     and plain `RotatingKVCache` for SWA boundary layers. Python's
//     PoolQuantizedV4Cache path is intentionally not active in Swift; the
//     legacy Python app currently forces it off because it bypassed the
//     DeepseekV4Cache isinstance gate.
//   • Reasoning + tool-call dispatch is wired in `ReasoningParser` /
//     `ToolCallFormat` for `deepseek_v4` model_type.
//   • SSM re-derive does NOT fire for DSV4 (no Mamba layers). It's
//     enabled in LoadOptions so the runtime stays consistent across
//     model families; it short-circuits cleanly for non-hybrid
//     architectures.

import Foundation
import vMLXEngine
import vMLXLMCommon
import RuntimeShared

let bundlePath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : (FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ").path)

let bundle = URL(fileURLWithPath: bundlePath)
guard FileManager.default.fileExists(atPath: bundle.path) else {
    print("bundle not found: \(bundle.path)")
    exit(2)
}

let cacheDir = FileManager.default.temporaryDirectory
    .appendingPathComponent("vmlx-examples-dsv4")

print("=== DSV4-Flash runtime example ===")
print("bundle: \(bundle.lastPathComponent)")
print("cache:  \(cacheDir.path)")

let opts = RuntimeShared.makeLoadOptions(
    bundle: bundle,
    cacheDir: cacheDir,
    kvCacheBits: 4,
    kvCacheGroupSize: 64,
    slidingWindowMode: "long"
)
RuntimeShared.reportLoadOptions(opts)

let engine = Engine()
let t0 = Date()
try await RuntimeShared.awaitLoad(engine, options: opts)
print("[runtime] loaded in \(Int(Date().timeIntervalSince(t0))) s")

if ProcessInfo.processInfo.environment["VMLX_DSV4_QUICK"] == "1" {
    let env = ProcessInfo.processInfo.environment
    let chatMax = Int(env["VMLX_DSV4_CHAT_MAX_TOKENS"] ?? "48") ?? 48
    let thinkMax = Int(env["VMLX_DSV4_THINK_MAX_TOKENS"] ?? "384") ?? 384
    let chatTemperature = Double(env["VMLX_DSV4_CHAT_TEMPERATURE"] ?? "0.0") ?? 0.0
    let chatTopP = Double(env["VMLX_DSV4_CHAT_TOP_P"] ?? "")

    print("\n--- quick mode=chat ---")
    let chatReq = RuntimeShared.makeRequest(
        [RuntimeShared.userMsg("What is the capital of France? Answer in one word.")],
        maxTokens: chatMax, temperature: chatTemperature, topP: chatTopP,
        enableThinking: false
    )
    let (chatReasoning, chatContent, _) = try await RuntimeShared.drainStream(
        engine, chatReq, printContent: false)
    print("[quick.chat.reasoning] \(chatReasoning.count)")
    print("[quick.chat.content] \(chatContent.prefix(160))")
    RuntimeShared.assertNoLeak(chatContent, label: "dsv4 quick chat content")
    await RuntimeShared.reportCacheStats(engine)

    print("\n--- quick mode=think ---")
    let thinkReq = RuntimeShared.makeRequest(
        [RuntimeShared.userMsg("What is 17 + 28? Think briefly first, then answer.")],
        maxTokens: thinkMax, temperature: 0.6, topP: 0.95,
        enableThinking: true
    )
    let (thinkReasoning, thinkContent, _) = try await RuntimeShared.drainStream(
        engine, thinkReq, printContent: false, printReasoningTick: true)
    print("[quick.think.reasoning] \(thinkReasoning.count)")
    print("[quick.think.content] \(thinkContent.prefix(160))")
    RuntimeShared.assertNoLeak(thinkContent, label: "dsv4 quick think content")
    await RuntimeShared.reportCacheStats(engine)

    print("\n=== quick done ===")
    exit(0)
}

// MARK: Mode 1 — chat (no reasoning)
do {
    print("\n--- mode=chat ---")
    let req = RuntimeShared.makeRequest(
        [RuntimeShared.userMsg("What is the capital of France? Answer in one word.")],
        maxTokens: 32, temperature: 0.0, enableThinking: false
    )
    let (_, content, _) = try await RuntimeShared.drainStream(engine, req)
    RuntimeShared.assertNoLeak(content)
    await RuntimeShared.reportCacheStats(engine)
}

// MARK: Mode 2 — think (Think High)
do {
    print("\n--- mode=think (Think High) ---")
    let req = RuntimeShared.makeRequest(
        [RuntimeShared.userMsg("What is 17 + 28?  Think briefly first, then answer.")],
        maxTokens: 1024, temperature: 0.6, topP: 0.95, enableThinking: true
    )
    let (reasoning, content, _) = try await RuntimeShared.drainStream(
        engine, req, printContent: false, printReasoningTick: true)
    print("reasoning=\(reasoning.count) ch  content=\(content.count) ch")
    print("answer: \(content.prefix(120))")
    RuntimeShared.assertNoLeak(content)
    await RuntimeShared.reportCacheStats(engine)
}

// MARK: Mode 3 — think_max (paper-budget reasoning)
do {
    print("\n--- mode=think_max ---")
    let prompt = """
    Three boxes: A has 2R/3B balls, B has 1R/4B, C has 4R/1B. Pick a random box uniformly, \
    then 2 balls without replacement. Both red. P(C | both red)? Show full Bayes derivation.
    """
    let req = RuntimeShared.makeRequest(
        [RuntimeShared.userMsg(prompt)],
        maxTokens: 4096, temperature: 1.0, topP: 0.95,
        enableThinking: true, reasoningEffort: "max"
    )
    let (_, content, _) = try await RuntimeShared.drainStream(
        engine, req, printContent: false, printReasoningTick: true)
    print("answer: \(content.prefix(200))")
    RuntimeShared.assertNoLeak(content)
    await RuntimeShared.reportCacheStats(engine)
}

// MARK: Mode 4 — DSML tool-call demo
do {
    print("\n--- mode=chat with tools (DSML) ---")
    let sys = """
    You can call tools using DSML format. Tool: get_time(timezone: string).
    Emit <｜DSML｜invoke name="get_time"><｜DSML｜parameter name="timezone" string="true">UTC</｜DSML｜parameter></｜DSML｜invoke> when asked the time.
    """
    let req = RuntimeShared.makeRequest(
        [RuntimeShared.systemMsg(sys), RuntimeShared.userMsg("What time is it in UTC?")],
        maxTokens: 256, temperature: 0.0
    )
    let (_, content, tools) = try await RuntimeShared.drainStream(engine, req)
    print("[TOOLS PARSED] \(tools)")
    RuntimeShared.assertNoLeak(content)
}

// MARK: Multi-turn — should hit memory cache, then L1, then L2 across runs
do {
    print("\n--- multi-turn cache reuse ---")
    var msgs: [ChatRequest.Message] = []
    let asks = [
        "Tell me one fact about the Eiffel Tower in 12 words.",
        "Now one about the Louvre, also 12 words.",
        "And one about Notre-Dame.",
    ]
    for ask in asks {
        msgs.append(RuntimeShared.userMsg(ask))
        let req = RuntimeShared.makeRequest(msgs, maxTokens: 64, temperature: 0.0)
        let (_, ans, _) = try await RuntimeShared.drainStream(engine, req, printContent: false)
        print(">>> \(ask)\n<<< \(ans.prefix(100))…")
        msgs.append(RuntimeShared.assistantMsg(ans))
    }
    await RuntimeShared.reportCacheStats(engine)
    print("[note] L1 hit on turn 2 = prefix cache reuse working;")
    print("       blockDisk.wired=false today; prompt-level DiskCache is the persistent tier.")
}

print("\n=== done ===")
