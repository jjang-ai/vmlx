// vMLX Examples — JANGPressRSSBench
//
// Measures the ACTUAL RSS (resident set size) delta when JANGPress
// flips from "all routed experts hot" to "all routed experts cold".
// This is the empirical answer to "does this fucking work" for the
// memory-saving claim.
//
// EXPERIMENT
// ==========
// 1. Open the bundle's safetensors shards via JangPressMmapTier.
//    Pages are lazy — RSS doesn't grow yet.
// 2. Force-fault every routed-expert page by issuing WILLNEED on
//    every layer-expert pair, then touching the pages once. This
//    is the "cold-start, all hot" baseline.
// 3. Sample RSS via mach_task_info(TASK_VM_INFO).
// 4. Issue DONTNEED on every routed-expert range — kernel reclaims.
// 5. Sample RSS again.
// 6. Report the delta. A working pipeline gives `before > after`
//    by approximately `total_routed_bytes`.
//
// USAGE
// =====
//   swift run --package-path /Users/eric/vmlx/swift JANGPressRSSBench \
//       /Volumes/EricsLLMDrive/jangq-ai/JANGQ-AI/Laguna-XS.2-JANGTQ
//
// CAVEATS
// =======
// macOS aggressively page-caches mmap'd files even when MADV_DONTNEED
// is issued — actual reclamation depends on kernel pressure heuristics.
// On a 128 GB host with no other workload, the kernel may keep our
// pages resident as opportunistic cache. Run on a constrained box or
// alongside a memory balloon for cleaner numbers.

import Foundation
import Darwin
import vMLXLMCommon

// MARK: - mach RSS sampling
//
// macOS exposes two relevant counters via `task_info(TASK_VM_INFO)`:
//
//   • `phys_footprint`  — anonymous + dirty file-backed pages.
//                         Doesn't count clean file-backed mmap pages.
//                         This is what `Activity Monitor` calls
//                         "Memory" for a process.
//
//   • `resident_size`   — ALL resident pages including clean file
//                         mappings. What `top RSIZE` shows. The
//                         JANGPress comparison we care about
//                         (mmap'd file-backed vs MLX's anonymous
//                         copy) shows up HERE — not phys_footprint.
//
// We sample both so the user can see exactly which axis moves under
// DONTNEED. For the file-backed mmap path: `resident_size` should
// drop, `phys_footprint` shouldn't change much.

struct VmSample {
    let physFootprint: UInt64
    let residentSize: UInt64
}

func sampleVM() -> VmSample {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    if kr != KERN_SUCCESS { return VmSample(physFootprint: 0, residentSize: 0) }
    return VmSample(physFootprint: UInt64(info.phys_footprint),
                    residentSize: UInt64(info.resident_size))
}

func currentResidentBytes() -> UInt64 {
    sampleVM().residentSize
}

func mb(_ bytes: UInt64) -> String {
    String(format: "%.1f MB", Double(bytes) / 1024.0 / 1024.0)
}

// MARK: - main

let bundlePath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "/Volumes/EricsLLMDrive/jangq-ai/JANGQ-AI/Laguna-XS.2-JANGTQ"

let bundleURL = URL(fileURLWithPath: bundlePath)
guard FileManager.default.fileExists(atPath: bundleURL.path) else {
    print("bundle not found: \(bundleURL.path)")
    exit(2)
}

print("=== JANGPressRSSBench: \(bundleURL.lastPathComponent) ===\n")

let rssBaseline = currentResidentBytes()
print("RSS baseline (before mmap):  \(mb(rssBaseline))")

// Phase 1 — open shards via JangPressMmapTier
let tier = try JangPressMmapTier(
    config: .init(bundleURL: bundleURL, hotPercent: 100, startCold: false))
let stats = tier.snapshot()
print("Tier built: \(stats.shardCount) shards, \(stats.expertCount) experts, \(stats.totalRoutedBytes / 1024 / 1024) MB routed")

let rssAfterMmap = currentResidentBytes()
print("RSS after mmap (lazy):       \(mb(rssAfterMmap))   (Δ \(mb(rssAfterMmap &- rssBaseline)))")

// Phase 2 — force-fault every routed-expert page by touching them
print("\n--- forcing all routed-expert pages resident ---")
let touchStart = Date()
var touchedBytes: UInt64 = 0
for (_, ranges) in tier.experts {
    for part in ranges.parts {
        guard let shard = tier.shards[part.shard] else { continue }
        let buf = shard.bytes(in: part.range)
        // Stride through pages forcing a load
        let pageSize = Int(getpagesize())
        var sum: UInt64 = 0
        for i in stride(from: 0, to: buf.count, by: pageSize) {
            sum &+= UInt64(buf[i])
        }
        touchedBytes += UInt64(buf.count)
        // Avoid optimizer eliding the read
        if sum == 0xDEADBEEF { print("(unreachable)") }
    }
}
print("touched: \(touchedBytes / 1024 / 1024) MB in \(Int(Date().timeIntervalSince(touchStart) * 1000)) ms")

let rssAllHot = currentResidentBytes()
print("RSS all-routed hot:          \(mb(rssAllHot))   (Δ vs mmap \(mb(rssAllHot &- rssAfterMmap)))")

// Phase 3a — DONTNEED on every routed-expert range (soft hint).
// macOS treats MADV_DONTNEED as a hint and frequently ignores it
// when free RAM is abundant. Documented; included here for contrast.
print("\n--- phase 3a: soft hint (madvise DONTNEED) ---")
let dontStart = Date()
let layers = Array(stats.byLayer.keys).sorted()
for layer in layers {
    let count = stats.byLayer[layer] ?? 0
    if count > 0 {
        tier.release(layer: layer, experts: Array(0..<count))
    }
}
print("DONTNEED pass: \(Int(Date().timeIntervalSince(dontStart) * 1000)) ms")
sleep(2)
let rssAfterDontNeed = currentResidentBytes()
print("RSS after DONTNEED:          \(mb(rssAfterDontNeed))   (Δ vs all-hot \(Int64(rssAfterDontNeed) - Int64(rssAllHot)) bytes)")

// Phase 3b — force-invalidate via msync(MS_INVALIDATE). This is the
// Darwin-specific path that actually drops file-backed clean pages.
// Stronger signal to the kernel than DONTNEED.
print("\n--- phase 3b: force invalidate (msync MS_INVALIDATE) ---")
let invStart = Date()
for layer in layers {
    let count = stats.byLayer[layer] ?? 0
    if count > 0 {
        tier.forceRelease(layer: layer, experts: Array(0..<count))
    }
}
print("forceInvalidate pass: \(Int(Date().timeIntervalSince(invStart) * 1000)) ms")
sleep(2)
let rssAllCold = currentResidentBytes()
print("RSS after forceInvalidate:   \(mb(rssAllCold))   (Δ vs all-hot \(Int64(rssAllCold) - Int64(rssAllHot)) bytes)")

// Phase 4 — re-acquire one expert; latency tells us if pages need refault
print("\n--- re-acquire warmth check ---")
let warmStart = Date()
if let firstLayer = layers.first {
    tier.acquire(layer: firstLayer, experts: [0])
}
print("first re-acquire wall: \(Int(Date().timeIntervalSince(warmStart) * 1_000_000)) µs")
let rssAfterReacq = currentResidentBytes()
print("RSS after re-acquire:        \(mb(rssAfterReacq))")

// Phase 5 — production-realistic partial release.
//
// Real production state has a hot-set (the routinely-routed experts)
// + a cold tail. We simulate this by:
//   1. Re-touching all experts to fault them back
//   2. forceReleasing only the bottom 70% of layers' experts
//   3. Measuring RSS — should drop by ~70% of routed mass, not 100%
print("\n--- phase 5: partial release (top 30% kept hot, bottom 70% forced) ---")
// Re-warm everything first.
for (_, ranges) in tier.experts {
    for part in ranges.parts {
        guard let shard = tier.shards[part.shard] else { continue }
        let buf = shard.bytes(in: part.range)
        let pageSize = Int(getpagesize())
        var s: UInt64 = 0
        for i in stride(from: 0, to: buf.count, by: pageSize) {
            s &+= UInt64(buf[i])
        }
        if s == 0xDEADBEEF { print("(unreachable)") }
    }
}
let rssAllHotAgain = currentResidentBytes()
print("RSS all-hot (re-warmed):     \(mb(rssAllHotAgain))")

// Decide which 70 % of layers go cold. Keep first 30 % hot.
let keepHot = max(1, Int(Double(layers.count) * 0.30))
let coldLayers = Array(layers.dropFirst(keepHot))
let partStart = Date()
for layer in coldLayers {
    let count = stats.byLayer[layer] ?? 0
    if count > 0 {
        tier.forceRelease(layer: layer, experts: Array(0..<count))
    }
}
print("partial forceRelease pass: \(Int(Date().timeIntervalSince(partStart) * 1000)) ms over \(coldLayers.count) of \(layers.count) layers")
sleep(2)
let rssPartial = currentResidentBytes()
let partialReclaimed = Int64(rssAllHotAgain) - Int64(rssPartial)
print("RSS after partial release:   \(mb(rssPartial))   (Δ \(partialReclaimed / 1024 / 1024) MB reclaimed)")
print("Partial reclaim rate:        \(Double(partialReclaimed) / Double(stats.totalRoutedBytes) * 100) % of total routed mass")

// Summary
print("\n=== summary ===")
let savedBytes = Int64(rssAllHot) - Int64(rssAllCold)
let savedMB = Double(savedBytes) / 1024.0 / 1024.0
let totalRoutedMB = Double(stats.totalRoutedBytes) / 1024.0 / 1024.0
let reclaimPct = totalRoutedMB > 0 ? Double(savedBytes) / Double(stats.totalRoutedBytes) * 100.0 : 0.0
print(String(format: "  total routed mass:   %.1f MB", totalRoutedMB))
print(String(format: "  RSS reclaimed:       %.1f MB (%.1f %% of routed mass)", savedMB, reclaimPct))
print(String(format: "  RSS baseline:        %@", mb(rssBaseline)))
print(String(format: "  RSS all hot:         %@", mb(rssAllHot)))
print(String(format: "  RSS all cold:        %@", mb(rssAllCold)))
print(String(format: "  RSS after re-acq:    %@", mb(rssAfterReacq)))

if savedBytes > Int64(stats.totalRoutedBytes) / 4 {
    print("\nVERDICT: kernel honored DONTNEED on a substantial fraction of pages.")
} else if savedBytes > 0 {
    print("\nVERDICT: partial reclaim — kernel kept some pages opportunistically.")
} else {
    print("\nVERDICT: no measurable reclaim. Likely free RAM was abundant; kernel kept pages as page cache. Re-run on a constrained system or with a balloon.")
}
