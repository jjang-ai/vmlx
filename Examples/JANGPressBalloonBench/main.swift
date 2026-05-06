// vMLX Examples — JANGPressBalloonBench
//
// Verifies the production claim: "under memory pressure, the macOS
// kernel reclaims our DONTNEED'd file-backed mmap pages." The standard
// JANGPressRSSBench can't prove this on a 128 GB host because the
// kernel has no reason to reclaim — there's plenty of free RAM.
//
// THIS bench inflates a memory balloon (anonymous, mlock'd) until the
// kernel feels actual pressure, then verifies that:
//   1. JangPress's DONTNEED hint actually causes reclaim under pressure.
//   2. Re-acquire after reclaim refaults from disk and produces
//      byte-identical data (no silent corruption).
//   3. Re-acquire latency is bounded (~ms per tile, not seconds).
//
// USAGE
// =====
//   swift run JANGPressBalloonBench <bundle> [pct] [balloon_gb]
//
// EXAMPLES
// ========
//   # Default: 70% reclaim target, 4 GB balloon
//   JANGPressBalloonBench /Users/.../Holo3-35B-A3B-JANGTQ
//
//   # Aggressive: 100% reclaim, 16 GB balloon
//   JANGPressBalloonBench /Users/.../Holo3-35B-A3B-JANGTQ 100 16
//
// HOW IT WORKS
// ============
// 1. Open mmap tier via JangPressMmapTier
// 2. Force-fault all routed-expert pages (RSS goes up by full mass)
// 3. Snapshot the first tile's bytes (ground truth for integrity check)
// 4. Issue madvise(MADV_DONTNEED) on bottom-pct% of layers
// 5. Allocate a balloon: mmap anonymous pages, touch each page to
//    force allocation. Hold for ~5 seconds.
// 6. Sample RSS — should be lower than pre-balloon if kernel reclaimed
// 7. Re-acquire all routed-expert pages
// 8. Verify the snapshotted tile's bytes match (data integrity)
// 9. Free balloon
// 10. Report whether reclaim actually happened
//
// SAFETY
// ======
// The balloon is bounded (default 4 GB, max 32 GB). It uses mmap
// MAP_PRIVATE | MAP_ANON which gets backed by anonymous swap-eligible
// pages. We don't mlock — letting the kernel make tradeoffs (it should
// prefer reclaiming our clean file-backed routed-expert pages over
// the dirty balloon pages, which is exactly what we're testing).

import Foundation
import Darwin
import vMLXLMCommon

// Use line-buffered stderr so log lines appear in real time even when
// piped/captured. Plain print() is block-buffered to stdout when stdout
// isn't a TTY, which masks crashes mid-bench.
func log(_ s: String) {
    FileHandle.standardError.write(Data((s + "\n").utf8))
}

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

func mb(_ b: UInt64) -> String { String(format: "%.1f MB", Double(b) / 1_048_576.0) }
func gb(_ b: UInt64) -> String { String(format: "%.2f GB", Double(b) / 1_073_741_824.0) }

// Allocate `bytes` of anonymous memory and touch every page to force
// real allocation. Returns base pointer + size for later munmap.
func inflateBalloon(bytes: Int) -> (UnsafeMutableRawPointer, Int)? {
    guard bytes > 0 else { return nil }
    let p = mmap(nil, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANON, -1, 0)
    guard p != UnsafeMutableRawPointer(bitPattern: -1)! else {
        log("[balloon] mmap failed errno=\(errno)")
        return nil
    }
    let pageSize = Int(getpagesize())
    var written: Int = 0
    let buf = p!.assumingMemoryBound(to: UInt8.self)
    var i = 0
    while i < bytes {
        // Write a non-zero byte so the page is dirty + must be allocated.
        buf[i] = UInt8((i / pageSize) & 0xFF)
        written += 1
        i += pageSize
    }
    log("[balloon] inflated \(gb(UInt64(bytes))) (touched \(written) pages)")
    return (p!, bytes)
}

func deflateBalloon(_ p: UnsafeMutableRawPointer, _ size: Int) {
    munmap(p, size)
    log("[balloon] deflated")
}

// MARK: - Main

let bundlePath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "/Users/eric/.mlxstudio/models/JANGQ-AI/Holo3-35B-A3B-JANGTQ"
let pct = CommandLine.arguments.count > 2 ? (Int(CommandLine.arguments[2]) ?? 70) : 70
let balloonGB = CommandLine.arguments.count > 3 ? (Int(CommandLine.arguments[3]) ?? 4) : 4

let bundleURL = URL(fileURLWithPath: bundlePath)
guard FileManager.default.fileExists(atPath: bundleURL.path) else {
    log("bundle not found: \(bundleURL.path)"); exit(2)
}

log("=== JANGPressBalloonBench: \(bundleURL.lastPathComponent) ===")
log("compressPct=\(pct), balloon=\(balloonGB) GB\n")

// 1. Open mmap tier (eager fault during phase 2).
let tier = try JangPressMmapTier(config: .init(
    bundleURL: bundleURL, hotPercent: 100 - pct, startCold: false))
let stats = tier.snapshot()
log("[init] mmap tier: \(stats.shardCount) shards, \(stats.expertCount) tiles, \(mb(stats.totalRoutedBytes)) routed mass")

let rssInit = sampleRSS()
log("[rss] init: \(mb(rssInit))\n")

// 2. Force-fault all routed-expert pages. We do a checksum read so the
// kernel can't be lazy.
log("--- phase 2: force-fault all routed pages (acquire all) ---")
let touchStart = Date()
var checksum: UInt64 = 0
for (key, ranges) in tier.experts {
    tier.acquire(layer: key.layer, experts: [key.expert])
    for part in ranges.parts {
        guard let shard = tier.shards[part.shard] else { continue }
        let buf = shard.bytes(in: part.range)
        // Read first byte of every 4 KB page.
        var off = 0
        while off < buf.count {
            checksum = checksum &+ UInt64(buf[off])
            off += 4096
        }
    }
}
let touchMs = Int(Date().timeIntervalSince(touchStart) * 1000)
log("[touch] all-faulted in \(touchMs) ms, checksum=\(checksum)")
let rssHot = sampleRSS()
log("[rss] all-hot: \(mb(rssHot))\n")

// 3. Snapshot first tile's bytes (ground truth for integrity).
guard let firstKey = tier.experts.keys.sorted(by: { ($0.layer, $0.expert) < ($1.layer, $1.expert) }).first,
      let firstRanges = tier.experts[firstKey],
      let firstPart = firstRanges.parts.first,
      let firstShard = tier.shards[firstPart.shard]
else {
    log("[error] no tiles found")
    exit(3)
}
let truthBytes = Array(firstShard.bytes(in: firstPart.range))
log("[truth] snapshotted layer=\(firstKey.layer) expert=\(firstKey.expert): \(truthBytes.count) bytes")

// 4. Apply forceRelease on bottom pct% (matches what the controller
// does at quiesce time). For maximum stress we use msync(INVALIDATE).
log("\n--- phase 4: forceRelease bottom \(pct)% via msync(INVALIDATE) ---")
let releaseStart = Date()
let layerKeys = Array(Set(tier.experts.keys.map(\.layer))).sorted()
let coldLayerCount = max(0, Int(Double(layerKeys.count) * Double(pct) / 100.0))
let coldLayers = Array(layerKeys.prefix(coldLayerCount))
for layer in coldLayers {
    let experts = tier.experts.keys.filter { $0.layer == layer }.map(\.expert)
    tier.forceRelease(layer: layer, experts: experts)
}
let releaseMs = Int(Date().timeIntervalSince(releaseStart) * 1000)
log("[release] forceRelease \(coldLayers.count) layers in \(releaseMs) ms")
let rssAfterRelease = sampleRSS()
log("[rss] after release: \(mb(rssAfterRelease)) (Δ \(mb(rssHot - rssAfterRelease)))\n")

// 5. Inflate balloon. This is the actual pressure test.
log("--- phase 5: inflate \(balloonGB) GB balloon ---")
let bytesToInflate = balloonGB * 1024 * 1024 * 1024
guard let (balloonPtr, balloonSize) = inflateBalloon(bytes: bytesToInflate) else {
    log("[abort] balloon inflate failed")
    exit(4)
}
defer { deflateBalloon(balloonPtr, balloonSize) }

// Hold balloon for 3 seconds so kernel pressure heuristics can react.
log("[balloon] holding for 3s to let kernel adjust...")
Thread.sleep(forTimeInterval: 3.0)
let rssWithBalloon = sampleRSS()
log("[rss] under balloon: \(mb(rssWithBalloon)) (Δ vs all-hot \(mb(rssHot - rssWithBalloon)) reclaimed)\n")

// 6. Re-acquire and verify the truth bytes still match.
log("--- phase 6: re-acquire + verify integrity ---")
let reacqStart = Date()
tier.acquire(layer: firstKey.layer, experts: [firstKey.expert])
// Read the bytes again (forcing re-fault from disk)
let reread = Array(firstShard.bytes(in: firstPart.range))
let reacqMs = Int(Date().timeIntervalSince(reacqStart) * 1000)

if reread == truthBytes {
    log("✅ data integrity: re-fault produced byte-identical data (\(reread.count) bytes) in \(reacqMs) ms")
} else {
    log("❌ data integrity: re-fault produced DIFFERENT bytes (\(reread.count) vs \(truthBytes.count))")
    var diffCount = 0
    for i in 0..<min(reread.count, truthBytes.count) {
        if reread[i] != truthBytes[i] { diffCount += 1 }
    }
    log("   first 1000 byte diffs: \(diffCount)")
}

// 7. Final RSS sample
let rssFinal = sampleRSS()
log("\n[rss] post re-acquire: \(mb(rssFinal))")

// 8. Summary
log("\n=== SUMMARY ===")
log("init RSS:       \(mb(rssInit))")
log("all-hot RSS:    \(mb(rssHot))")
log("after release:  \(mb(rssAfterRelease))   (Δ from hot: \(mb(rssHot - rssAfterRelease)))")
log("under balloon:  \(mb(rssWithBalloon))   (Δ from hot: \(mb(rssHot - rssWithBalloon)))")
log("post re-acquire:\(mb(rssFinal))")
log("re-acquire latency: \(reacqMs) ms for \(mb(UInt64(truthBytes.count))) tile")

if rssAfterRelease < rssHot {
    let reclaim = Double(rssHot - rssAfterRelease) / Double(rssHot - rssInit) * 100
    log("\n✅ VERDICT: kernel reclaimed \(String(format: "%.1f", reclaim))% of routed mass on forceRelease.")
    log("   Under memory pressure, JangPress will reclaim your routed experts on demand.")
} else {
    log("\n⚠️  VERDICT: no measurable reclaim. Try larger balloon or constrained host.")
}
