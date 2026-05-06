import Foundation
import vMLXEngine
import vMLXLMCommon
import RuntimeShared

func envBool(_ key: String, default defaultValue: Bool = false) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[key]?.lowercased() else {
        return defaultValue
    }
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

func envInt(_ key: String, default defaultValue: Int) -> Int {
    ProcessInfo.processInfo.environment[key].flatMap(Int.init) ?? defaultValue
}

func envString(_ key: String, default defaultValue: String) -> String {
    ProcessInfo.processInfo.environment[key] ?? defaultValue
}

func envCSV(_ key: String) -> Set<String> {
    guard let raw = ProcessInfo.processInfo.environment[key], !raw.isEmpty else {
        return []
    }
    return Set(raw
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
        .filter { !$0.isEmpty })
}

func envList(_ key: String) -> [String] {
    guard let raw = ProcessInfo.processInfo.environment[key], !raw.isEmpty else {
        return []
    }
    return raw
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
}

func envPathList(_ pluralKey: String, _ singularKey: String) -> [String] {
    var values = envList(pluralKey)
    if let single = ProcessInfo.processInfo.environment[singularKey],
       !single.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    {
        values.append(single.trimmingCharacters(in: .whitespacesAndNewlines))
    }
    return values
}

func applyBlockDiskOverride(to opts: inout Engine.LoadOptions) {
    let env = ProcessInfo.processInfo.environment

    if let raw = env["CACHE_PROBE_BLOCK_DISK"], !raw.isEmpty {
        switch raw.lowercased() {
        case "0", "false", "no", "off":
            opts.enableBlockDiskCache = false
        case "1", "true", "yes", "on":
            opts.enableBlockDiskCache = true
        default:
            // Back-compat with the first live probe harness: a non-boolean
            // value is a directory path, not "false".
            opts.enableBlockDiskCache = true
            opts.blockDiskCacheDir = raw
        }
    }

    if let rawDir = env["CACHE_PROBE_BLOCK_DISK_DIR"], !rawDir.isEmpty {
        opts.enableBlockDiskCache = true
        opts.blockDiskCacheDir = rawDir
    }
}

func oneLine(_ s: String, limit: Int = envInt("CACHE_PROBE_CONTENT_LIMIT", default: 220)) -> String {
    let collapsed = s.replacingOccurrences(of: "\n", with: "\\n")
    return collapsed.count > limit ? String(collapsed.prefix(limit)) + "..." : collapsed
}

func stat(_ stats: [String: Any], _ section: String, _ key: String) -> Any {
    (stats[section] as? [String: Any])?[key] ?? "nil"
}

func dict(_ stats: [String: Any], _ section: String) -> [String: Any] {
    stats[section] as? [String: Any] ?? [:]
}

func intValue(_ value: Any?) -> Int {
    if let v = value as? Int { return v }
    if let v = value as? Int64 { return Int(v) }
    if let v = value as? NSNumber { return v.intValue }
    if let v = value as? String { return Int(v) ?? 0 }
    return 0
}

func doubleValue(_ value: Any?) -> Double {
    if let v = value as? Double { return v }
    if let v = value as? Float { return Double(v) }
    if let v = value as? Int { return Double(v) }
    if let v = value as? NSNumber { return v.doubleValue }
    if let v = value as? String { return Double(v) ?? 0 }
    return 0
}

func boolValue(_ value: Any?) -> Bool {
    if let v = value as? Bool { return v }
    if let v = value as? NSNumber { return v.boolValue }
    if let v = value as? String {
        let s = v.lowercased()
        return s == "1" || s == "true" || s == "yes" || s == "on"
    }
    return false
}

func statInt(_ stats: [String: Any], _ section: String, _ key: String) -> Int {
    intValue(dict(stats, section)[key])
}

func nestedInt(_ stats: [String: Any], _ section: String, _ child: String, _ key: String) -> Int {
    intValue((dict(stats, section)[child] as? [String: Any])?[key])
}

func nestedBool(_ stats: [String: Any], _ section: String, _ child: String, _ key: String) -> Bool {
    boolValue((dict(stats, section)[child] as? [String: Any])?[key])
}

func deltaInt(_ before: [String: Any], _ after: [String: Any], _ section: String, _ key: String) -> Int {
    statInt(after, section, key) - statInt(before, section, key)
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    print("usage: CacheMatrixProbe <bundle>")
    exit(2)
}

let bundle = URL(fileURLWithPath: args[1])
guard FileManager.default.fileExists(atPath: bundle.path) else {
    print("bundle not found: \(bundle.path)")
    exit(2)
}

let mode = envString("CACHE_PROBE_MODE", default: "all").lowercased()
let cacheRoot = URL(fileURLWithPath: envString(
    "CACHE_PROBE_CACHE_DIR",
    default: FileManager.default.temporaryDirectory
        .appendingPathComponent("vmlx-cache-matrix-\(bundle.lastPathComponent)-\(mode)")
        .path))
try? FileManager.default.removeItem(at: cacheRoot)
try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)

let basePrompt = envString("CACHE_PROBE_PROMPT", default: String(repeating:
    "This is a cache validation prompt with stable repeated context about Apple Silicon, memory hierarchy, routers, sliding windows, and SSM state. ",
    count: 10) + "Answer with exactly one short sentence.")
let secondPrompt = envString("CACHE_PROBE_SECOND_PROMPT", default: basePrompt)
let imagePaths = envPathList("CACHE_PROBE_IMAGES", "CACHE_PROBE_IMAGE")
let videoPaths = envPathList("CACHE_PROBE_VIDEOS", "CACHE_PROBE_VIDEO")
let audioPaths = envPathList("CACHE_PROBE_AUDIOS", "CACHE_PROBE_AUDIO")
let inputAudioPaths = envPathList("CACHE_PROBE_INPUT_AUDIOS", "CACHE_PROBE_INPUT_AUDIO")
let turn2ImagePaths = envPathList("CACHE_PROBE_TURN2_IMAGES", "CACHE_PROBE_TURN2_IMAGE")
let turn2VideoPaths = envPathList("CACHE_PROBE_TURN2_VIDEOS", "CACHE_PROBE_TURN2_VIDEO")
let turn2AudioPaths = envPathList("CACHE_PROBE_TURN2_AUDIOS", "CACHE_PROBE_TURN2_AUDIO")
let turn2InputAudioPaths = envPathList("CACHE_PROBE_TURN2_INPUT_AUDIOS", "CACHE_PROBE_TURN2_INPUT_AUDIO")
let replayTurn1InTurn2 = envBool("CACHE_PROBE_MULTITURN_HISTORY", default: false)
let turn1ExpectedTerms = envList("CACHE_PROBE_TURN1_EXPECT")
let turn2ExpectedTerms = envList("CACHE_PROBE_TURN2_EXPECT")
let maxTokens = envInt("CACHE_PROBE_MAX_TOKENS", default: 24)
let slidingMode = envString("CACHE_PROBE_SLIDING_MODE", default: "auto")
let required = envCSV("CACHE_PROBE_REQUIRE")
let minCompletionTokens = envInt("CACHE_PROBE_MIN_COMPLETION_TOKENS", default: 1)
let minContentChars = envInt("CACHE_PROBE_MIN_CONTENT_CHARS", default: 1)
let minTurn2GenerationTps = Double(envString("CACHE_PROBE_MIN_TURN2_TPS", default: "0")) ?? 0
let routerDrainMs = envInt("CACHE_PROBE_DRAIN_ROUTER_ADVICE_MS", default: 0)
let thinkingBudget = envInt("CACHE_PROBE_THINKING_BUDGET", default: 0)
let reasoningEffort = envString("CACHE_PROBE_REASONING_EFFORT", default: "")
let failOnLeak = envBool("CACHE_PROBE_FAIL_ON_LEAK", default: true)
let failWhitespaceOnly = envBool("CACHE_PROBE_FAIL_WHITESPACE_ONLY", default: true)

var opts = RuntimeShared.makeLoadOptions(
    bundle: bundle,
    cacheDir: cacheRoot,
    kvCacheBits: envInt("CACHE_PROBE_TQ_BITS", default: 4),
    kvCacheGroupSize: envInt("CACHE_PROBE_TQ_GROUP", default: 64),
    slidingWindowMode: slidingMode)
opts.pagedCacheBlockSize = envInt("CACHE_PROBE_BLOCK_SIZE", default: opts.pagedCacheBlockSize)
opts.maxCacheBlocks = envInt("CACHE_PROBE_MAX_BLOCKS", default: opts.maxCacheBlocks)

switch mode {
case "disk":
    opts.enableMemoryCache = false
    opts.enablePrefixCache = false
    opts.usePagedCache = false
    opts.enableDiskCache = true
case "paged":
    opts.enableMemoryCache = false
    opts.enablePrefixCache = true
    opts.usePagedCache = true
    opts.enableDiskCache = false
case "blockdisk":
    opts.enableMemoryCache = false
    opts.enablePrefixCache = true
    opts.usePagedCache = true
    opts.enableDiskCache = false
    opts.enableBlockDiskCache = true
case "memory":
    opts.enableMemoryCache = true
    opts.enablePrefixCache = false
    opts.usePagedCache = false
    opts.enableDiskCache = false
default:
    break
}

opts.enableTurboQuant = envBool("CACHE_PROBE_TQ", default: opts.enableTurboQuant)
opts.kvCacheQuantization = opts.enableTurboQuant ? "turboquant" : "none"
opts.enableJangPress = envBool("CACHE_PROBE_JANGPRESS", default: false)
opts.jangPressBackend = .mmap
opts.jangPressCompressPct = envInt("CACHE_PROBE_JANGPRESS_PCT", default: 70)
opts.enableJangPressRouterAdvice = envBool("CACHE_PROBE_ROUTER_ADVICE", default: false)
applyBlockDiskOverride(to: &opts)

print("=== CacheMatrixProbe ===")
print("bundle=\(bundle.path)")
print("mode=\(mode) cacheDir=\(cacheRoot.path)")
RuntimeShared.reportLoadOptions(opts)
print("JangPress=\(opts.enableJangPress) routerAdvice=\(opts.enableJangPressRouterAdvice) sliding=\(opts.slidingWindowMode)")
print("Thinking=\(envBool("CACHE_PROBE_THINKING", default: false)) reasoningEffort=\(reasoningEffort.isEmpty ? "nil" : reasoningEffort) thinkingBudget=\(thinkingBudget > 0 ? String(thinkingBudget) : "nil")")
print("Media turn1 images=\(imagePaths.count) videos=\(videoPaths.count) audios=\(audioPaths.count) inputAudios=\(inputAudioPaths.count) turn2 images=\(turn2ImagePaths.count) videos=\(turn2VideoPaths.count) audios=\(turn2AudioPaths.count) inputAudios=\(turn2InputAudioPaths.count) history=\(replayTurn1InTurn2)")
if !required.isEmpty {
    print("[requirements] \(required.sorted().joined(separator: ",")) minCompletionTokens=\(minCompletionTokens) minContentChars=\(minContentChars) minTurn2Tps=\(minTurn2GenerationTps)")
}

let engine = Engine()
let tLoad = Date()
try await RuntimeShared.awaitLoad(engine, options: opts)
print("[load] seconds=\(String(format: "%.2f", Date().timeIntervalSince(tLoad)))")

struct TurnProbe {
    var label: String
    var result: RuntimeShared.DrainResult
    var elapsed: TimeInterval
}

func runTurn(_ label: String, messages: [ChatRequest.Message]) async throws -> TurnProbe {
    let req = RuntimeShared.makeRequest(
        messages,
        maxTokens: maxTokens,
        temperature: 0.0,
        enableThinking: envBool("CACHE_PROBE_THINKING", default: false),
        reasoningEffort: reasoningEffort.isEmpty ? nil : reasoningEffort,
        thinkingBudget: thinkingBudget > 0 ? thinkingBudget : nil)
    let t = Date()
    let result = try await RuntimeShared.drainStreamDetailed(
        engine, req, printContent: false, printReasoningTick: false)
    let elapsed = Date().timeIntervalSince(t)
    let usage = result.finalUsage ?? result.lastUsage
    let tps = usage?.tokensPerSecond.map { String(format: "%.2f", $0) } ?? "nil"
    let promptTps = usage?.promptTokensPerSecond.map { String(format: "%.2f", $0) } ?? "nil"
    let ttft = usage?.ttftMs.map { String(format: "%.0f", $0) } ?? "nil"
    let total = usage?.totalMs.map { String(format: "%.0f", $0) } ?? "nil"
    print("[\(label)] seconds=\(String(format: "%.2f", elapsed)) contentChars=\(result.content.count) reasoningChars=\(result.reasoning.count) contentChunks=\(result.contentChunkCount) reasoningChunks=\(result.reasoningChunkCount) partialUsage=\(result.partialUsageCount) finish=\(result.finishReason ?? "nil")")
    print("[\(label).usage] prompt=\(usage?.promptTokens ?? 0) completion=\(usage?.completionTokens ?? 0) cached=\(usage?.cachedTokens ?? 0) tps=\(tps) promptTps=\(promptTps) ttftMs=\(ttft) totalMs=\(total) cacheDetail=\(usage?.cacheDetail ?? "nil")")
    print("[\(label).content] \(oneLine(result.content.isEmpty ? result.reasoning : result.content))")
    return TurnProbe(label: label, result: result, elapsed: elapsed)
}

func userTurn(
    prompt: String,
    images: [String],
    videos: [String],
    audios: [String],
    inputAudios: [String]
) -> ChatRequest.Message {
    if images.isEmpty && videos.isEmpty && audios.isEmpty && inputAudios.isEmpty {
        return RuntimeShared.userMsg(prompt)
    }
    return RuntimeShared.userMediaMsg(
        prompt,
        imagePaths: images,
        videoPaths: videos,
        audioPaths: audios,
        inputAudioPaths: inputAudios)
}

func visibleOrFallback(_ turn: TurnProbe) -> String {
    turn.result.content.isEmpty ? turn.result.reasoning : turn.result.content
}

let before = try await engine.cacheStats()
print("[stats.before] paged.h=\(stat(before, "paged", "hitCount")) paged.m=\(stat(before, "paged", "missCount")) disk.h=\(stat(before, "disk", "hitCount")) disk.m=\(stat(before, "disk", "missCount")) disk.store=\(stat(before, "disk", "storeCount")) disk.tqEntries=\(stat(before, "disk", "turboQuantEntryCount")) disk.tqKeys=\(stat(before, "disk", "turboQuantTensorKeyCount")) block.h=\(stat(before, "blockDisk", "hitCount")) block.m=\(stat(before, "blockDisk", "missCount")) block.store=\(stat(before, "blockDisk", "storeCount")) mem.h=\(stat(before, "memory", "hitCount")) mem.m=\(stat(before, "memory", "missCount"))")

let turn1Message = userTurn(
    prompt: basePrompt,
    images: imagePaths,
    videos: videoPaths,
    audios: audioPaths,
    inputAudios: inputAudioPaths)
let turn1 = try await runTurn("turn1", messages: [turn1Message])
let after1 = try await engine.cacheStats()
print("[stats.after1] paged.h=\(stat(after1, "paged", "hitCount")) paged.m=\(stat(after1, "paged", "missCount")) disk.h=\(stat(after1, "disk", "hitCount")) disk.m=\(stat(after1, "disk", "missCount")) disk.store=\(stat(after1, "disk", "storeCount")) disk.entries=\(stat(after1, "disk", "entryCount")) disk.tqEntries=\(stat(after1, "disk", "turboQuantEntryCount")) disk.tqKeys=\(stat(after1, "disk", "turboQuantTensorKeyCount")) block.h=\(stat(after1, "blockDisk", "hitCount")) block.m=\(stat(after1, "blockDisk", "missCount")) block.store=\(stat(after1, "blockDisk", "storeCount")) block.entries=\(stat(after1, "blockDisk", "entryCount")) mem.h=\(stat(after1, "memory", "hitCount")) mem.m=\(stat(after1, "memory", "missCount")) ssm.h=\(stat(after1, "ssmCompanion", "hitCount")) ssm.m=\(stat(after1, "ssmCompanion", "missCount"))")

if mode == "blockdisk" || envBool("CACHE_PROBE_CLEAR_BEFORE_TURN2", default: false) {
    print("[probe] clearing in-memory caches before turn2; persistent BlockDiskCache is retained")
    await engine.clearCaches()
}

let turn2Message = userTurn(
    prompt: secondPrompt,
    images: turn2ImagePaths,
    videos: turn2VideoPaths,
    audios: turn2AudioPaths,
    inputAudios: turn2InputAudioPaths)
let turn2Messages: [ChatRequest.Message] = replayTurn1InTurn2
    ? [
        turn1Message,
        RuntimeShared.assistantMsg(visibleOrFallback(turn1)),
        turn2Message,
    ]
    : [turn2Message]
let turn2 = try await runTurn("turn2", messages: turn2Messages)
if routerDrainMs > 0 {
    print("[probe] waiting \(routerDrainMs)ms for async router advice drain")
    try await Task.sleep(nanoseconds: UInt64(routerDrainMs) * 1_000_000)
}
let after2 = try await engine.cacheStats()
print("[stats.after2] paged.h=\(stat(after2, "paged", "hitCount")) paged.m=\(stat(after2, "paged", "missCount")) disk.h=\(stat(after2, "disk", "hitCount")) disk.m=\(stat(after2, "disk", "missCount")) disk.store=\(stat(after2, "disk", "storeCount")) disk.entries=\(stat(after2, "disk", "entryCount")) disk.tqEntries=\(stat(after2, "disk", "turboQuantEntryCount")) disk.tqKeys=\(stat(after2, "disk", "turboQuantTensorKeyCount")) block.h=\(stat(after2, "blockDisk", "hitCount")) block.m=\(stat(after2, "blockDisk", "missCount")) block.store=\(stat(after2, "blockDisk", "storeCount")) block.entries=\(stat(after2, "blockDisk", "entryCount")) mem.h=\(stat(after2, "memory", "hitCount")) mem.m=\(stat(after2, "memory", "missCount")) ssm.h=\(stat(after2, "ssmCompanion", "hitCount")) ssm.m=\(stat(after2, "ssmCompanion", "missCount"))")

if let arch = after2["architecture"] as? [String: Any] {
    print("[architecture] \(arch)")
}
if let jp = after2["jangPress"] as? [String: Any] {
    print("[jangPress] \(jp)")
}

let diskHit = deltaInt(after1, after2, "disk", "hitCount") > 0
let pagedHit = deltaInt(after1, after2, "paged", "hitCount") > 0
let memHit = deltaInt(after1, after2, "memory", "hitCount") > 0
let blockHit = deltaInt(after1, after2, "blockDisk", "hitCount") > 0
let ssmHit = deltaInt(after1, after2, "ssmCompanion", "hitCount") > 0
let diskStore = statInt(after1, "disk", "storeCount") > statInt(before, "disk", "storeCount")
let blockStore = statInt(after1, "blockDisk", "storeCount") > statInt(before, "blockDisk", "storeCount")
let jp = dict(after2, "jangPress")
let routerAdvisor = jp["routerAdvisor"] as? [String: Any] ?? [:]
let canonicalMmap = boolValue(jp["canonicalStorageReplaced"])
let routesObserved = intValue(jp["totalRoutesObserved"])
let willNeedBytes = intValue(jp["canonicalWillNeedBytes"])
let dontNeedBytes = intValue(jp["canonicalDontNeedBytes"])
let advisorReadbacks = intValue(routerAdvisor["readbacks"])
let advisorWarmBytes = intValue(routerAdvisor["warmBytes"])
let advisorColdBytes = intValue(routerAdvisor["coldBytes"])
let advisorEnabled = boolValue(routerAdvisor["enabled"])
let routeEvidence = routesObserved > 0 || advisorReadbacks > 0
let willNeedEvidence = willNeedBytes > 0 || advisorWarmBytes > 0

func validateTurn(_ turn: TurnProbe, failures: inout [String]) {
    let usage = turn.result.finalUsage ?? turn.result.lastUsage
    let visible = turn.result.content
    let fallbackVisible = visible.isEmpty ? turn.result.reasoning : visible
    if minCompletionTokens > 0 && (usage?.completionTokens ?? 0) < minCompletionTokens {
        failures.append("\(turn.label) completion tokens \(usage?.completionTokens ?? 0) < \(minCompletionTokens)")
    }
    if minContentChars > 0 && visible.count < minContentChars {
        failures.append("\(turn.label) visible content chars \(visible.count) < \(minContentChars)")
    }
    if failWhitespaceOnly && !fallbackVisible.isEmpty
        && fallbackVisible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    {
        failures.append("\(turn.label) output is whitespace-only")
    }
    if failOnLeak {
        for tag in RuntimeShared.leakTags where visible.contains(tag) {
            failures.append("\(turn.label) visible content leaked tag \(tag)")
        }
    }
}

func validateExpectedTerms(_ terms: [String], in turn: TurnProbe, failures: inout [String]) {
    guard !terms.isEmpty else { return }
    let haystack = visibleOrFallback(turn).lowercased()
    for term in terms {
        let needle = term.lowercased()
        if !haystack.contains(needle) {
            failures.append("\(turn.label) expected term missing: \(term)")
        }
    }
}

var failures: [String] = []
validateTurn(turn1, failures: &failures)
validateTurn(turn2, failures: &failures)
validateExpectedTerms(turn1ExpectedTerms, in: turn1, failures: &failures)
validateExpectedTerms(turn2ExpectedTerms, in: turn2, failures: &failures)

for requirement in required {
    switch requirement {
    case "paged":
        if !pagedHit { failures.append("required paged hit did not occur on turn2") }
    case "disk":
        if !diskHit { failures.append("required DiskCache hit did not occur on turn2") }
    case "blockdisk", "block":
        if !blockHit { failures.append("required BlockDisk hit did not occur on turn2") }
    case "memory", "mem":
        if !memHit { failures.append("required memory-cache hit did not occur on turn2") }
    case "ssm":
        if !ssmHit { failures.append("required SSM companion hit did not occur on turn2") }
    case "disk-store":
        if !diskStore { failures.append("required DiskCache store did not occur on turn1") }
    case "blockdisk-store", "block-store":
        if !blockStore { failures.append("required BlockDisk store did not occur on turn1") }
    case "content":
        if turn2.result.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            failures.append("required non-empty visible content did not occur on turn2")
        }
    case "tokens":
        let tokens = (turn2.result.finalUsage ?? turn2.result.lastUsage)?.completionTokens ?? 0
        if tokens < minCompletionTokens {
            failures.append("required turn2 completion token floor not met: \(tokens) < \(minCompletionTokens)")
        }
    case "image", "images":
        if imagePaths.isEmpty && turn2ImagePaths.isEmpty {
            failures.append("required image path was not configured")
        }
    case "video", "videos":
        if videoPaths.isEmpty && turn2VideoPaths.isEmpty {
            failures.append("required video path was not configured")
        }
    case "audio", "audios":
        if audioPaths.isEmpty && inputAudioPaths.isEmpty
            && turn2AudioPaths.isEmpty && turn2InputAudioPaths.isEmpty
        {
            failures.append("required audio path was not configured")
        }
    case "multiturn-history", "history":
        if !replayTurn1InTurn2 {
            failures.append("required multi-turn history replay was not enabled")
        }
    case "speed", "tps":
        let tps = (turn2.result.finalUsage ?? turn2.result.lastUsage)?.tokensPerSecond ?? 0
        if tps < minTurn2GenerationTps {
            failures.append("required turn2 tps floor not met: \(String(format: "%.2f", tps)) < \(minTurn2GenerationTps)")
        }
    case "jangpress-canonical", "jp-canonical":
        if !canonicalMmap { failures.append("required JangPress canonical mmap storage was not active") }
    case "jangpress-routes", "jp-routes":
        if !routeEvidence { failures.append("required JangPress route evidence was not observed") }
    case "jangpress-controller-routes", "jp-controller-routes":
        if routesObserved <= 0 { failures.append("required JangPress controller routes were not observed") }
    case "jangpress-willneed", "jp-willneed":
        if !willNeedEvidence { failures.append("required JangPress canonical/router WILLNEED bytes stayed zero") }
    case "jangpress-controller-willneed", "jp-controller-willneed":
        if willNeedBytes <= 0 { failures.append("required JangPress controller canonical WILLNEED bytes stayed zero") }
    case "jangpress-dontneed", "jp-dontneed":
        if dontNeedBytes <= 0 { failures.append("required JangPress canonical DONTNEED bytes stayed zero") }
    case "jangpress-advisor", "jp-advisor":
        if !advisorEnabled { failures.append("required JangPress router advisor was not enabled") }
    case "jangpress-readbacks", "jp-readbacks":
        if advisorReadbacks <= 0 { failures.append("required JangPress router advisor readbacks stayed zero") }
    case "jangpress-warm-bytes", "jp-warm-bytes":
        if advisorWarmBytes <= 0 { failures.append("required JangPress router advisor warm bytes stayed zero") }
    case "jangpress-cold-bytes", "jp-cold-bytes":
        if advisorColdBytes <= 0 { failures.append("required JangPress router advisor cold bytes stayed zero") }
    default:
        failures.append("unknown CACHE_PROBE_REQUIRE token: \(requirement)")
    }
}

print("[verdict] diskHit=\(diskHit) diskStore=\(diskStore) blockDiskHit=\(blockHit) blockDiskStore=\(blockStore) pagedHit=\(pagedHit) memoryHit=\(memHit) ssmHit=\(ssmHit) canonicalMmap=\(canonicalMmap) routeEvidence=\(routeEvidence) willNeedEvidence=\(willNeedEvidence) controllerRoutes=\(routesObserved) controllerWillNeedBytes=\(willNeedBytes) dontNeedBytes=\(dontNeedBytes) advisorEnabled=\(advisorEnabled) advisorReadbacks=\(advisorReadbacks) advisorWarmBytes=\(advisorWarmBytes) advisorColdBytes=\(advisorColdBytes)")

if failures.isEmpty {
    print("[probe.pass] requirements satisfied")
} else {
    for failure in failures {
        print("[probe.fail] \(failure)")
    }
    exit(1)
}
