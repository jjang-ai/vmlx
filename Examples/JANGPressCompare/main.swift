// vMLX Examples — JANGPressCompare
//
// Side-by-side measurement: same prompt, two runs (JangPress off
// vs on at compressPct=70). Records load time, prefill time, decode
// tok/s, RSS at every phase, and full output text. Diffs the two
// outputs to verify coherency (output should match because temp=0).
//
// USAGE
//   swift run JANGPressCompare <bundle> [compressPct=70]

import Foundation
import Darwin
import vMLXEngine
import vMLXLMCommon
import RuntimeShared

func sampleRSS() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? UInt64(info.resident_size) : 0
}

func gb(_ b: UInt64) -> String { String(format: "%.2f GB", Double(b) / 1_073_741_824.0) }

struct RunResult {
    let label: String
    let loadMs: Int
    let firstTokenMs: Int
    let decodeMs: Int
    let chunks: Int
    let content: String
    let reasoning: String
    let tokensCharsTotal: Int
    let tps: Double
    let rssBaseline: UInt64
    let rssPostLoad: UInt64
    let rssPostDecode: UInt64
    let emberBackend: String
}

func runOnce(label: String, bundleURL: URL, enableJP: Bool, pct: Int, forceMode: Bool = false) async throws -> RunResult {
    let modeStr = enableJP ? "ON pct=\(pct) force=\(forceMode)" : "OFF"
    print("\n========== RUN: \(label) (jangPress=\(modeStr)) ==========")

    let rssBaseline = sampleRSS()
    let cacheDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("jangpress-compare-\(label)")

    var opts = RuntimeShared.makeLoadOptions(
        bundle: bundleURL, cacheDir: cacheDir,
        kvCacheBits: 4, kvCacheGroupSize: 64, slidingWindowMode: "long")
    opts.enableJangPress = enableJP
    opts.jangPressCompressPct = pct
    opts.jangPressBackend = .mmap
    // iter 18 trace: prefetch toggle so we can isolate the source of
    // OFF→ON sub-1% drift at T=0. Controlled via env var.
    let prefetchOn = ProcessInfo.processInfo.environment["JANGPRESS_PREFETCH"] != "0"
    opts.jangPressEnablePrefetch = prefetchOn
    opts.jangPressForceMode = forceMode ? .force : .soft

    let tLoad = Date()
    let engine = Engine()
    try await RuntimeShared.awaitLoad(engine, options: opts)
    let loadMs = Int(Date().timeIntervalSince(tLoad) * 1000)
    let rssPostLoad = sampleRSS()
    print("[load] \(loadMs) ms, RSS post-load: \(gb(rssPostLoad))")

    // ONE warmup turn (consistency with production)
    let warmReq = ChatRequest(
        model: "in-process",
        messages: [.init(role: "user", content: .string("Hi"))],
        maxTokens: 4, temperature: 0.0,
        enableThinking: false, includeReasoning: true)
    _ = try await {
        var c = ""
        let s = await engine.stream(request: warmReq)
        for try await chunk in s {
            if let x = chunk.content { c += x }
            if let x = chunk.reasoning { c += x }
        }
        return c
    }()

    // The actual measurement turn — fixed prompt, same seed.
    //
    // enableThinking=true: needed for thinking-aware bundles (DSV4, Holo3,
    // MiniMax, Qwen3.x, Nemotron-H, Laguna — all have thinkInTemplate=true
    // in ModelCapabilities). With enableThinking=false, BLOCKER #3 in
    // Stream.swift §1588 SUPPRESSES all reasoning output (template stamps
    // `<think>` regardless of user setting), and content stays empty until
    // `</think>` arrives — which doesn't fit in 64 tokens. Result: 0 chunks.
    //
    // With enableThinking=true the parser routes content to chunk.reasoning
    // (still deterministic at T=0 → coherency check still works).
    let maxTok = Int(ProcessInfo.processInfo.environment["MAX_TOKENS"] ?? "64") ?? 64
    let req = ChatRequest(
        model: "in-process",
        messages: [.init(role: "user", content: .string("List 5 facts about Apple Silicon. Be concise."))],
        maxTokens: maxTok, temperature: 0.0,
        enableThinking: true, includeReasoning: true)

    let tDecode = Date()
    var firstTokenMs: Int = -1
    var content = ""
    var reasoning = ""
    var chunks = 0
    var engineTps: Double = 0          // engine-reported decode tok/s
    var promptTokens: Int = 0
    var completionTokens: Int = 0
    var ttftMs: Double = 0
    var prefillMs: Double = 0
    let s = await engine.stream(request: req)
    for try await chunk in s {
        if firstTokenMs < 0 && (chunk.content != nil || chunk.reasoning != nil) {
            firstTokenMs = Int(Date().timeIntervalSince(tDecode) * 1000)
        }
        if let c = chunk.content, !c.isEmpty { content += c; chunks += 1 }
        if let r = chunk.reasoning, !r.isEmpty { reasoning += r }
        // Capture engine-reported metrics from any usage chunk —
        // final non-partial usage has the authoritative numbers.
        if let u = chunk.usage {
            if let tps = u.tokensPerSecond { engineTps = tps }
            if let ttft = u.ttftMs { ttftMs = ttft }
            if let pf = u.prefillMs { prefillMs = pf }
            if !u.isPartial {
                promptTokens = u.promptTokens
                completionTokens = u.completionTokens
            }
        }
    }
    let decodeMs = Int(Date().timeIntervalSince(tDecode) * 1000)
    let rssPostDecode = sampleRSS()

    let totalChars = content.count + reasoning.count
    let approxTokens = max(chunks, completionTokens, totalChars / 3)
    // Wall-clock tok/s (total tokens / total time INCLUDING prefill).
    let wallTps = Double(approxTokens) / max(Double(decodeMs) / 1000.0, 0.001)
    // Engine-reported tps is decode-only — apples-to-apples for compression cost.
    let tps = engineTps > 0 ? engineTps : wallTps

    let stats = try? await engine.cacheStats()
    let backend: String = (stats?["jangPress"] as? [String: Any])?["backend"] as? String ?? "unknown"

    print("[decode] wall=\(decodeMs) ms, first chunk @ \(firstTokenMs) ms, ttft=\(Int(ttftMs)) ms, prefill=\(Int(prefillMs)) ms")
    print("[decode] chunks=\(chunks) content_chars=\(content.count) reasoning_chars=\(reasoning.count)")
    print("[decode] tokens: prompt=\(promptTokens) completion=\(completionTokens) approx=\(approxTokens)")
    print("[decode] engine tok/s=\(String(format: "%.2f", engineTps))   wall tok/s=\(String(format: "%.2f", wallTps))")
    print("[decode] backend=\(backend)")
    print("[content first 200]: \(content.prefix(200))")
    if !reasoning.isEmpty {
        print("[reasoning first 200]: \(reasoning.prefix(200))")
    }
    print("[rss] baseline=\(gb(rssBaseline)) post-load=\(gb(rssPostLoad)) post-decode=\(gb(rssPostDecode))")

    return RunResult(
        label: label, loadMs: loadMs, firstTokenMs: firstTokenMs, decodeMs: decodeMs,
        chunks: chunks, content: content, reasoning: reasoning,
        tokensCharsTotal: totalChars, tps: tps,
        rssBaseline: rssBaseline, rssPostLoad: rssPostLoad, rssPostDecode: rssPostDecode,
        emberBackend: backend)
}

// MARK: - main

let bundlePath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "/Users/eric/.mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ"
let pct = CommandLine.arguments.count > 2 ? (Int(CommandLine.arguments[2]) ?? 70) : 70

let bundleURL = URL(fileURLWithPath: bundlePath)
guard FileManager.default.fileExists(atPath: bundleURL.path) else {
    print("bundle not found: \(bundleURL.path)"); exit(2)
}

print("=== JANGPressCompare: \(bundleURL.lastPathComponent) ===")

// Run mode: pass MODE=off or MODE=on as env var to do a SINGLE run.
// Apples-to-apples comparison requires two separate processes (the
// 2-engines-in-one-process path has a 3× slowdown artifact unrelated
// to JangPress — verified empirically by running BOTH runs OFF in
// the same process and seeing the same 3× slowdown).
//
// Default mode runs both in same process anyway for quick smoke
// (will be biased — use shell wrapper for real numbers).
let mode = ProcessInfo.processInfo.environment["MODE"] ?? "both"

let off: RunResult
let on: RunResult
switch mode {
case "off":
    off = try await runOnce(label: "OFF", bundleURL: bundleURL, enableJP: false, pct: 0)
    on = off    // print same data twice, summary code skips comparison if labels match
case "on":
    on = try await runOnce(label: "ON", bundleURL: bundleURL, enableJP: true, pct: pct, forceMode: false)
    off = on
case "on-force":
    on = try await runOnce(label: "ON-FORCE", bundleURL: bundleURL, enableJP: true, pct: pct, forceMode: true)
    off = on
case "control":
    off = try await runOnce(label: "OFF",  bundleURL: bundleURL, enableJP: false, pct: 0)
    on  = try await runOnce(label: "OFF2", bundleURL: bundleURL, enableJP: false, pct: 0)
case "csv":
    // Single-row CSV: tag,enableJP,pct,force,loadMs,decodeMs,engineTps,rssPostLoadMB,rssPostDecodeMB,reasoningChars
    // Driven by env vars: JP_ENABLE=0|1, JP_PCT=N, JP_FORCE=0|1, JP_TAG=name
    let envEn = (ProcessInfo.processInfo.environment["JP_ENABLE"] ?? "0") == "1"
    let envPct = Int(ProcessInfo.processInfo.environment["JP_PCT"] ?? "0") ?? 0
    let envFm = (ProcessInfo.processInfo.environment["JP_FORCE"] ?? "0") == "1"
    let envTag = ProcessInfo.processInfo.environment["JP_TAG"] ?? "csv"
    let r = try await runOnce(label: envTag, bundleURL: bundleURL,
                              enableJP: envEn, pct: envPct, forceMode: envFm)
    let row = "\(envTag),\(envEn),\(envPct),\(envFm),\(r.loadMs),\(r.decodeMs),\(String(format: "%.2f", r.tps)),\(r.rssPostLoad / 1_048_576),\(r.rssPostDecode / 1_048_576),\(r.reasoning.count)"
    print("CSVROW:" + row)
    off = r; on = r
default:
    off = try await runOnce(label: "OFF", bundleURL: bundleURL, enableJP: false, pct: 0)
    on  = try await runOnce(label: "ON",  bundleURL: bundleURL, enableJP: true,  pct: pct, forceMode: false)
}

print("\n========== COMPARISON ==========")
print(String(format: "  load:        OFF=%5d ms   ON=%5d ms   Δ=%+d ms",
             off.loadMs, on.loadMs, on.loadMs - off.loadMs))
print(String(format: "  first chunk: OFF=%5d ms   ON=%5d ms   Δ=%+d ms",
             off.firstTokenMs, on.firstTokenMs, on.firstTokenMs - off.firstTokenMs))
print(String(format: "  decode:      OFF=%5d ms   ON=%5d ms   Δ=%+d ms",
             off.decodeMs, on.decodeMs, on.decodeMs - off.decodeMs))
print(String(format: "  tok/s:       OFF=%.1f       ON=%.1f       Δ=%+.1f%%",
             off.tps, on.tps,
             ((on.tps - off.tps) / max(off.tps, 0.001)) * 100))
print(String(format: "  RSS post-decode: OFF=%@   ON=%@   Δ=%+.2f GB",
             gb(off.rssPostDecode), gb(on.rssPostDecode),
             (Double(on.rssPostDecode) - Double(off.rssPostDecode)) / 1_073_741_824.0))

print("\n=== COHERENCY CHECK ===")
if off.content == on.content {
    print("✅ content output IDENTICAL (\(off.content.count) chars)")
} else {
    print("❌ content output DIFFERS")
    print("  OFF (\(off.content.count) chars): \(off.content.prefix(120))")
    print("   ON (\(on.content.count) chars): \(on.content.prefix(120))")
}
if off.reasoning == on.reasoning {
    print("✅ reasoning output IDENTICAL (\(off.reasoning.count) chars)")
} else {
    print("❌ reasoning output DIFFERS")
    print("  OFF (\(off.reasoning.count) chars): \(off.reasoning.prefix(120))")
    print("   ON (\(on.reasoning.count) chars): \(on.reasoning.prefix(120))")
}
