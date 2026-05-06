// vMLX Examples — JANGPressE2E
//
// End-to-end Engine + JANGPress measurement. Loads a JANGTQ bundle
// via the production `Engine.load` path with `enableJangPress=true`,
// runs a real inference through `Engine.stream`, samples task RSS
// (resident_size from mach_task_info) at every phase, and reports
// the empirical answer to "what does JANGPress cost on tok/s + RSS".
//
// Phases sampled:
//   1. baseline (process startup, before Engine)
//   2. post-load (Engine.load completed, MLX has weights resident)
//   3. post-warmup (one short generation, caches primed)
//   4. post-decode (full inference, JANGPress's mmap pages
//                   touched alongside MLX's weights)
//   5. post-quiesce (after waiting for ember controller's quiesce
//                    timeout — should see RSS drop if pressure)
//
// Usage:
//   swift run JANGPressE2E /path/to/bundle [compressPct]
//
// On M4 Max with DSV4-Flash-JANGTQ:
//   swift run JANGPressE2E ~/.mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ 70

import Foundation
import Darwin
import vMLXEngine
import vMLXLMCommon
import RuntimeShared

// MARK: - RSS sampling

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

func gb(_ bytes: UInt64) -> String {
    String(format: "%.2f GB", Double(bytes) / 1024.0 / 1024.0 / 1024.0)
}

// MARK: - main

let bundlePath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "/Users/eric/.mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ"
let compressPct = CommandLine.arguments.count > 2
    ? (Int(CommandLine.arguments[2]) ?? 70) : 70

let bundleURL = URL(fileURLWithPath: bundlePath)
guard FileManager.default.fileExists(atPath: bundleURL.path) else {
    print("bundle not found: \(bundleURL.path)"); exit(2)
}

print("=== JANGPressE2E: \(bundleURL.lastPathComponent) ===")
print("compressPct: \(compressPct)\n")

let rssBaseline = sampleRSS()
print("[phase 1] RSS baseline:           \(gb(rssBaseline))")

// MARK: Phase 2 — Engine.load with JANGPress on
let cacheDir = FileManager.default.temporaryDirectory
    .appendingPathComponent("jangember-e2e")

var opts = RuntimeShared.makeLoadOptions(
    bundle: bundleURL, cacheDir: cacheDir,
    kvCacheBits: 4, kvCacheGroupSize: 64, slidingWindowMode: "long")
opts.enableJangPress = true
opts.jangPressCompressPct = compressPct
opts.jangPressBackend = .mmap
opts.jangPressEnablePrefetch = true

print("[phase 2] loading via Engine.load (.mmap backend, pct=\(compressPct))...")
let engine = Engine()
let tLoad = Date()
try await RuntimeShared.awaitLoad(engine, options: opts)
let loadMs = Int(Date().timeIntervalSince(tLoad) * 1000)
let rssPostLoad = sampleRSS()
print("[phase 2] load wall:              \(loadMs) ms")
print("[phase 2] RSS post-load:          \(gb(rssPostLoad))   (Δ \(gb(rssPostLoad &- rssBaseline)))")

// MARK: Phase 3 — warmup (one short generation)
print("\n[phase 3] warmup inference (short prompt)...")
let warmupReq = RuntimeShared.makeRequest(
    [RuntimeShared.userMsg("Say hi.")],
    maxTokens: 8, temperature: 0.0, enableThinking: false)
let tWarm = Date()
let (_, warmContent, _) = try await RuntimeShared.drainStream(engine, warmupReq, printContent: false)
let warmMs = Int(Date().timeIntervalSince(tWarm) * 1000)
let rssPostWarm = sampleRSS()
print("[phase 3] warmup wall:            \(warmMs) ms")
print("[phase 3] warmup output:          \(warmContent.prefix(80))")
print("[phase 3] RSS post-warmup:        \(gb(rssPostWarm))   (Δ vs load \(Int64(rssPostWarm) - Int64(rssPostLoad)) bytes)")

// MARK: Phase 4 — real inference, measure tok/s by streaming chunks
print("\n[phase 4] real inference (32-token decode)...")
let decodeReq = RuntimeShared.makeRequest(
    [RuntimeShared.userMsg("Write one sentence about Apple Silicon.")],
    maxTokens: 32, temperature: 0.0, enableThinking: false)
let tDecode = Date()
var decodeContent = ""
var nChunks = 0
let stream = await engine.stream(request: decodeReq)
for try await chunk in stream {
    if let c = chunk.content, !c.isEmpty {
        decodeContent += c
        nChunks += 1
    }
}
let decodeMs = Date().timeIntervalSince(tDecode)
// Approximate tokens: chunk count is a lower bound, character count / 3
// is a rough upper bound (typical bf16 tokenizer ratio).
let approxTokens = max(nChunks, decodeContent.count / 3)
let tps = Double(approxTokens) / max(decodeMs, 0.001)
let rssPostDecode = sampleRSS()
print("[phase 4] decode wall:            \(Int(decodeMs * 1000)) ms")
print("[phase 4] decode output:          \(decodeContent.prefix(180))")
print("[phase 4] content chars:          \(decodeContent.count)")
print("[phase 4] chunks streamed:        \(nChunks)")
print("[phase 4] approx tokens:          \(approxTokens)")
print("[phase 4] approx tok/s:           \(String(format: "%.1f", tps))")
print("[phase 4] RSS post-decode:        \(gb(rssPostDecode))   (Δ vs warm \(Int64(rssPostDecode) - Int64(rssPostWarm)) bytes)")

// MARK: Phase 5 — wait past quiesce, see if controller fires
let quiesceWait = 35   // s — controller's quiesceTimeoutMs is 30s by default
print("\n[phase 5] waiting \(quiesceWait) s for ember quiesce timeout...")
sleep(UInt32(quiesceWait))
let rssPostQuiesce = sampleRSS()
let savedBytes = Int64(rssPostDecode) - Int64(rssPostQuiesce)
let savedMB = Double(savedBytes) / 1024.0 / 1024.0
print("[phase 5] RSS post-quiesce:       \(gb(rssPostQuiesce))   (Δ vs decode \(Int64(savedBytes / 1024 / 1024)) MB reclaimed)")

// MARK: Cache stats
let stats = try await engine.cacheStats()
if let ember = stats["jangPress"] as? [String: Any] {
    print("\n[cache] jangPress stats:")
    for (k, v) in ember { print("  \(k): \(v)") }
}

// MARK: Summary
print("\n=== SUMMARY ===")
print(String(format: "  Load wall:               %d ms", loadMs))
print(String(format: "  Warmup wall:             %d ms", warmMs))
print(String(format: "  Decode wall:             %d ms (~%d tokens, ~%.1f tok/s)", Int(decodeMs * 1000), approxTokens, tps))
print(String(format: "  RSS baseline:            %@", gb(rssBaseline)))
print(String(format: "  RSS post-load:           %@   (+%@ for model)", gb(rssPostLoad), gb(rssPostLoad - rssBaseline)))
print(String(format: "  RSS post-decode:         %@", gb(rssPostDecode)))
print(String(format: "  RSS post-quiesce:        %@   (Δ %.2f GB reclaimed)", gb(rssPostQuiesce), savedMB / 1024.0))
print("\nThis measurement reflects the production code path: Engine.load,")
print("Engine.stream, JangPressController quiesce-time compaction.")
