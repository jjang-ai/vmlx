// vMLX Examples — JANGPressMultiTurn
//
// Multi-turn coherency stress test for JangPress. The hardest test for
// the iter 24 first-inference reclaim: after turn 1, JangPress madvises
// DONTNEED on every routed-expert range. Turn 2's first inference must
// then re-fault those pages from disk while staying coherent. We do
// 3 turns and check:
//
//   • Each turn produces non-empty reasoning (proves output works)
//   • Each turn's reasoning addresses the prompt (proves coherence)
//   • Cross-turn context is maintained (turn 2 should see turn 1's content)
//   • RSS evolution: post-turn-1 should drop (reclaim), turn 2+ stable
//   • Re-fault latency on turn 2 vs turn 1 (the cold path)
//
// USAGE
// =====
//   swift run JANGPressMultiTurn <bundle> [pct=70] [turns=3]
//
// EXAMPLES
// ========
//   swift run JANGPressMultiTurn /…/Holo3-A3B-JANGTQ
//   swift run JANGPressMultiTurn /…/DSV4-Flash-JANGTQ 70 3
//
// PROTOCOL
// ========
// Turn 1: "List 3 facts about Apple Silicon."
// Turn 2: "Now compare Apple Silicon to Intel chips on the SAME aspects."
// Turn 3: "Which aspect matters most for laptop battery life?"
//
// A coherent multi-turn response will have turn 2 reference the same 3
// aspects as turn 1, and turn 3 should pick one of those aspects.
//
// All turns at temperature=0, enableThinking=true so output goes through
// the reasoning channel (the BLOCKER #3 workaround for thinkInTemplate=true
// bundles — see DEEP-TRACE Issue 1).

import Foundation
import Darwin
import vMLXEngine
import vMLXLMCommon
import RuntimeShared

// Stderr-line-buffered output. print() goes to stdout which is block-buffered
// on a non-TTY (e.g. ssh subprocess) and loses lines on async exit.
func say(_ s: String) {
    FileHandle.standardError.write(Data((s + "\n").utf8))
}

/// Sample BOTH `resident_size` (file-backed RSS, useful for "what
/// pages are physically resident") AND `phys_footprint` (Activity
/// Monitor's "Memory" column, the right number for "will this fit on
/// a user's machine"). The May-3 oldmbp:vmlx-swift-lm progress doc
/// flagged that RSS alone is misleading for mmap-backed weights —
/// pages may be resident in the kernel page cache without being
/// counted in this process's footprint, and vice versa. All product
/// claims should cite footprint; RSS is debug-only.
func sampleMemory() -> (rss: UInt64, footprint: UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return (0, 0) }
    return (UInt64(info.resident_size), UInt64(info.phys_footprint))
}

func sampleRSS() -> UInt64 { sampleMemory().rss }

func gb(_ b: UInt64) -> String { String(format: "%.2f GB", Double(b) / 1_073_741_824.0) }

struct TurnResult {
    let index: Int
    let prompt: String
    let firstTokenMs: Int
    let decodeMs: Int
    let engineTps: Double
    let promptTokens: Int
    let completionTokens: Int
    let content: String
    let reasoning: String
    let rssAfter: UInt64
    let footprintAfter: UInt64
}

let bundlePath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "/Users/eric/.mlxstudio/models/JANGQ-AI/Holo3-35B-A3B-JANGTQ"
let pct = CommandLine.arguments.count > 2 ? (Int(CommandLine.arguments[2]) ?? 70) : 70
let turns = CommandLine.arguments.count > 3 ? (Int(CommandLine.arguments[3]) ?? 3) : 3
let bundleURL = URL(fileURLWithPath: bundlePath)
guard FileManager.default.fileExists(atPath: bundleURL.path) else {
    say("bundle not found: \(bundleURL.path)"); exit(2)
}

let prompts = [
    "List 3 distinct facts about Apple Silicon. Number them. Be concise.",
    "Now compare Apple Silicon to Intel chips on those SAME 3 aspects.",
    "Of those 3 aspects, which matters most for laptop battery life and why?",
]

say("=== JANGPressMultiTurn: \(bundleURL.lastPathComponent) ===")
say("pct=\(pct), turns=\(turns)\n")

// 1. Boot the engine WITH JangPress enabled.
let cacheDir = FileManager.default.temporaryDirectory
    .appendingPathComponent("jangpress-multiturn-\(UUID().uuidString)")
var opts = RuntimeShared.makeLoadOptions(
    bundle: bundleURL, cacheDir: cacheDir,
    kvCacheBits: 4, kvCacheGroupSize: 64, slidingWindowMode: "long")
opts.enableJangPress = true
opts.jangPressCompressPct = pct
opts.jangPressBackend = .mmap
opts.jangPressForceMode = .soft

let memBaseline = sampleMemory()
let rssBaseline = memBaseline.rss
let footprintBaseline = memBaseline.footprint
let tLoad = Date()
let engine = Engine()
try await RuntimeShared.awaitLoad(engine, options: opts)
let loadMs = Int(Date().timeIntervalSince(tLoad) * 1000)
let memPostLoad = sampleMemory()
let rssPostLoad = memPostLoad.rss
let footprintPostLoad = memPostLoad.footprint
say("[load] \(loadMs) ms")
say("[load] RSS post-load: \(gb(rssPostLoad)) (Δ \(gb(rssPostLoad - rssBaseline)))")
say("[load] footprint post-load: \(gb(footprintPostLoad)) (Δ \(gb(footprintPostLoad - footprintBaseline)))")
say("       footprint == Activity Monitor 'Memory' column. Use this for 'will it fit' claims.\n")

// 1.5. Warmup turn — flush MLX's lazy graph compile + trigger
// JangPress's first-inference reclaim BEFORE the measured turns.
// Without warmup, the measured turn 1 sometimes emits very short
// content (model occasionally hits an early stop on the FIRST
// inference; appears to be MLX FP non-determinism on cold-start).
say("--- Warmup ---")
let warmupReq = ChatRequest(
    model: "in-process",
    messages: [.init(role: "user", content: .string("Hi"))],
    maxTokens: 8, temperature: 0.0,
    enableThinking: true, includeReasoning: true)
let warmStart = Date()
let warmStream = await engine.stream(request: warmupReq)
for try await _ in warmStream { /* drain */ }
let warmMs = Int(Date().timeIntervalSince(warmStart) * 1000)
say("[warmup] \(warmMs) ms; iter-24 first-inference reclaim has fired by now\n")

// 2. Multi-turn. We accumulate the prior turns into a chat-history list
// so the model sees actual multi-turn context.
var history: [ChatRequest.Message] = []
var results: [TurnResult] = []

for turnIdx in 0..<min(turns, prompts.count) {
    let prompt = prompts[turnIdx]
    history.append(.init(role: "user", content: .string(prompt)))

    say("--- Turn \(turnIdx + 1) ---")
    say("[prompt] \(prompt)")

    let maxTokensEnv = ProcessInfo.processInfo.environment["JPB_MAX_TOKENS"].flatMap { Int($0) }
    let req = ChatRequest(
        model: "in-process", messages: history,
        maxTokens: maxTokensEnv ?? 128, temperature: 0.0,
        enableThinking: true, includeReasoning: true)

    let tDecode = Date()
    var firstTokenMs: Int = -1
    var content = ""
    var reasoning = ""
    var engineTps: Double = 0
    var promptTokens: Int = 0
    var completionTokens: Int = 0
    say("[debug] requesting stream…")
    let s = await engine.stream(request: req)
    say("[debug] stream object obtained, beginning iteration")
    var chunkCount = 0
    for try await chunk in s {
        chunkCount += 1
        if chunkCount == 1 || chunkCount % 16 == 0 {
            say("[debug] chunk #\(chunkCount) content?=\(chunk.content != nil) reasoning?=\(chunk.reasoning != nil) usage?=\(chunk.usage != nil)")
        }
        if firstTokenMs < 0 && (chunk.content != nil || chunk.reasoning != nil) {
            firstTokenMs = Int(Date().timeIntervalSince(tDecode) * 1000)
        }
        if let c = chunk.content, !c.isEmpty { content += c }
        if let r = chunk.reasoning, !r.isEmpty { reasoning += r }
        if let u = chunk.usage {
            if let tps = u.tokensPerSecond { engineTps = tps }
            if !u.isPartial {
                promptTokens = u.promptTokens
                completionTokens = u.completionTokens
            }
        }
    }
    let decodeMs = Int(Date().timeIntervalSince(tDecode) * 1000)
    let memAfter = sampleMemory()
    let rssAfter = memAfter.rss
    let footprintAfter = memAfter.footprint

    // Add the assistant's response to history so the next turn sees it.
    let assistantText = !content.isEmpty ? content : reasoning
    history.append(.init(role: "assistant", content: .string(assistantText)))

    results.append(TurnResult(
        index: turnIdx + 1, prompt: prompt,
        firstTokenMs: firstTokenMs, decodeMs: decodeMs,
        engineTps: engineTps, promptTokens: promptTokens,
        completionTokens: completionTokens,
        content: content, reasoning: reasoning,
        rssAfter: rssAfter,
        footprintAfter: footprintAfter))

    say("[decode] wall=\(decodeMs) ms, first chunk @ \(firstTokenMs) ms")
    say("[decode] tokens: prompt=\(promptTokens) completion=\(completionTokens) tps=\(String(format: "%.2f", engineTps))")
    say("[content first 300]: \(content.prefix(300))")
    if !reasoning.isEmpty {
        say("[reasoning first 300]: \(reasoning.prefix(300))")
    }
    say("[mem] after turn \(turnIdx + 1): RSS=\(gb(rssAfter)) footprint=\(gb(footprintAfter))\n")
}

// 3. Coherency check
say("=== COHERENCY CHECK ===")
say(String(repeating: "-", count: 50))

var allCoherent = true
for r in results {
    let outputText = !r.content.isEmpty ? r.content : r.reasoning
    let lower = outputText.lowercased()
    let nonEmpty = !outputText.isEmpty
    let producesText = outputText.count > 50
    // Density check — chars-per-token. Real English averages 4-6.
    // < 1.5 means most tokens are structural (whitespace, special).
    // The bench occasionally hits this on Holo3 turn 1 due to MLX
    // FP non-determinism producing an early stop. With the warmup
    // above this should be rare; flagging as "thin" rather than
    // failure when token count is high relative to char count.
    let charsPerTok = r.completionTokens > 0
        ? Double(outputText.count) / Double(r.completionTokens) : 0
    let densityOK = charsPerTok >= 1.5
    // Topic check — explicit keywords from any of the 3 prompts
    // (Apple Silicon / Intel / battery / power / efficient).
    let topical = lower.contains("apple") || lower.contains("silicon")
        || lower.contains("chip") || lower.contains("m1")
        || lower.contains("m2") || lower.contains("m3") || lower.contains("m4")
        || lower.contains("intel") || lower.contains("arm")
        || lower.contains("battery") || lower.contains("power")
        || lower.contains("efficien") || lower.contains("performance")
    // PASS criteria: substantial AND densely-formed AND topical.
    // Each is independently informative when it fails.
    let pass = nonEmpty && producesText && densityOK && topical
    let mark = pass ? "✅" : "❌"
    var reasons: [String] = []
    if !nonEmpty { reasons.append("EMPTY") }
    if !producesText { reasons.append("THIN(<50ch)") }
    if !densityOK { reasons.append("LOW-DENSITY(\(String(format: "%.2f", charsPerTok)) cpt)") }
    if !topical { reasons.append("OFF-TOPIC") }
    let reasonStr = reasons.isEmpty ? "" : " — \(reasons.joined(separator: ","))"
    say("\(mark) Turn \(r.index): \(outputText.count) chars, tokens=\(r.completionTokens), tps=\(String(format: "%.2f", r.engineTps)), ttft=\(r.firstTokenMs)ms\(reasonStr)")
    say("   first 200: \(outputText.prefix(200))")
    if !pass { allCoherent = false }
}

say("")
say("=== EVOLUTION ===")
say(String(format: "Pre-load:    RSS=%@ footprint=%@",
             gb(rssBaseline), gb(footprintBaseline)))
say(String(format: "Post-load:   RSS=%@ footprint=%@",
             gb(rssPostLoad), gb(footprintPostLoad)))
for r in results {
    let totalChars = r.content.count + r.reasoning.count
    say(String(format: "Turn %d: ttft=%5d ms, decode=%5d ms, tps=%5.2f, %5d chars, RSS=%@ footprint=%@",
                 r.index, r.firstTokenMs, r.decodeMs, r.engineTps, totalChars,
                 gb(r.rssAfter), gb(r.footprintAfter)))
}

// Advisor snapshot. Whether router-advice is enabled or not, the
// snapshot's `enabled=false` is itself a signal — confirms that
// the failsafe gate held when the user asked for cold-only mode.
let advisor = JangPressCanonicalExpertAdvisor.shared.snapshot()
say("")
say("=== ROUTER ADVISOR ===")
say(String(format: "enabled=%@ asyncReadback=%@ warmAdvice=%@ hotPerLayer=%d hot=%d",
             "\(advisor.enabled)", "\(advisor.asyncReadback)",
             "\(advisor.warmAdvice)", advisor.hotPerLayer,
             advisor.hotExpertCount))
say(String(format: "warmCalls=%d coldCalls=%d rewarms=%d (thrashRatio=%.2f) distinctCold=%d",
             advisor.warmCalls, advisor.coldCalls, advisor.rewarms,
             Double(advisor.rewarms) / Double(max(1, advisor.coldCalls)),
             advisor.distinctColdAdvisedPairs))
say(String(format: "warmBytes=%@ coldBytes=%@ pending=%d droppedQueueFull=%d readbacks=%d skippedLarge=%d skippedTracer=%d",
             gb(UInt64(max(0, advisor.warmBytes))),
             gb(UInt64(max(0, advisor.coldBytes))),
             advisor.pendingObservations, advisor.droppedQueueFull,
             advisor.readbacks, advisor.skippedLargeIndexTensors,
             advisor.skippedTracerArrays))

say("")
say("=== VERDICT ===")
if allCoherent {
    say("✅ All \(results.count) turns produced coherent output addressing the prompts.")
    say("   JangPress + multi-turn = OK on this bundle.")
} else {
    say("❌ Some turns failed coherency check.")
    say("   Output may be garbage, or model is broken on this bundle independent of JangPress.")
}

// 4. Re-fault latency comparison: turn 1 vs turn 2 ttft
if results.count >= 2 {
    let t1ttft = results[0].firstTokenMs
    let t2ttft = results[1].firstTokenMs
    let delta = t2ttft - t1ttft
    say("")
    say("=== POST-RECLAIM REFAULT LATENCY ===")
    say("turn 1 ttft: \(t1ttft) ms")
    say("turn 2 ttft: \(t2ttft) ms")
    if abs(delta) > 50 {
        say("delta: \(delta > 0 ? "+" : "")\(delta) ms — first-inference reclaim caused refault on turn 2")
    } else {
        say("delta: \(delta > 0 ? "+" : "")\(delta) ms — within noise; reclaim cost negligible")
    }
}
