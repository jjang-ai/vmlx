import SwiftUI
import vMLXEngine
import vMLXTheme

/// Live cache stats panel. Polls `Engine.cacheStats()` every 2s while the
/// Server tab is visible. Three collapsible sections:
///
/// 1. Paged (L1 in-memory) — utilization bar + hit rate + block config.
/// 2. Disk (L2 on-disk)     — size bar + entry count + hit rate + path.
/// 3. SSM Companion         — entry count + hybrid pill (hybrid models only).
///
/// Footer has a "Clear caches" action that calls `Engine.clearCaches()`.
struct CachePanel: View {
    @Environment(AppState.self) private var app
    @Environment(\.appLocale) private var appLocale: AppLocale
    @State private var stats: [String: Any] = [:]
    @State private var loaded: Bool = false
    @State private var pollTask: Task<Void, Never>? = nil
    @State private var archExpanded: Bool = true
    @State private var pagedExpanded: Bool = true
    @State private var memoryExpanded: Bool = true
    @State private var diskExpanded: Bool = true
    @State private var ssmExpanded: Bool = true
    @State private var clearInFlight: Bool = false
    @State private var warmInFlight: Bool = false
    @State private var warmStatus: String? = nil
    @State private var showClearConfirm: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            Text(L10n.ServerUI.cache.render(appLocale))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)

            if !loaded {
                Text(L10n.ServerUI.noModelLoadedSidebar.render(appLocale))
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                    .padding(.vertical, Theme.Spacing.sm)
            } else {
                DisclosureGroup(isExpanded: $archExpanded) {
                    architectureSection
                } label: {
                    HStack(spacing: Theme.Spacing.sm) {
                        Text(L10n.ServerUI.modelArchitecture.render(appLocale))
                            .font(Theme.Typography.body)
                            .foregroundStyle(Theme.Colors.textHigh)
                        if archBool("hybridSSMActive") { hybridPill }
                        if archBool("slidingWindowActive") { swaPill }
                        if archBool("turboQuantActive") { tqPill }
                    }
                }
                .tint(Theme.Colors.textMid)

                Divider().overlay(Theme.Colors.border)

                DisclosureGroup(isExpanded: $pagedExpanded) {
                    pagedSection
                } label: {
                    sectionLabel("Paged (L1 in-memory)", enabled: paged("enabled"))
                }
                .tint(Theme.Colors.textMid)

                Divider().overlay(Theme.Colors.border)

                DisclosureGroup(isExpanded: $memoryExpanded) {
                    memorySection
                } label: {
                    sectionLabel("Memory (L1.5 byte-budgeted)", enabled: memory("enabled"))
                }
                .tint(Theme.Colors.textMid)

                Divider().overlay(Theme.Colors.border)

                DisclosureGroup(isExpanded: $diskExpanded) {
                    diskSection
                } label: {
                    sectionLabel("Disk (L2 on-disk)", enabled: disk("enabled"))
                }
                .tint(Theme.Colors.textMid)

                Divider().overlay(Theme.Colors.border)

                DisclosureGroup(isExpanded: $ssmExpanded) {
                    ssmSection
                } label: {
                    HStack(spacing: Theme.Spacing.sm) {
                        sectionLabel("SSM Companion", enabled: ssm("enabled"))
                        if (ssm("enabled") as? Bool) == true {
                            hybridPill
                        }
                    }
                }
                .tint(Theme.Colors.textMid)

                Divider().overlay(Theme.Colors.border)

                footer
            }
        }
        .padding(Theme.Spacing.lg)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .fill(Theme.Colors.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.lg)
                        .stroke(Theme.Colors.border, lineWidth: 1)
                )
        )
        .task(id: app.engineState) { startPolling() }
        .onDisappear { pollTask?.cancel() }
    }

    // MARK: - Sections

    /// Per-layer cache breakdown surfaced from `Engine.cacheStats().architecture`.
    /// Ground truth: walked from the loaded model's cache array, not heuristics.
    /// Layer kinds: `rotating` = sliding-window attention, `mamba` = SSM,
    /// `turboQuant` = TurboQuant KV, `quantized` = Quantized KV, `kvSimple` =
    /// vanilla KV, `other` = unrecognized custom cache.
    @ViewBuilder
    private var architectureSection: some View {
        let total = archInt("total")
        let kv = archInt("kvSimple")
        let rot = archInt("rotating")
        let tq = archInt("turboQuant")
        let q = archInt("quantized")
        let m = archInt("mamba")
        let other = archInt("other")

        if total == 0 {
            Text(L10n.ServerUI.loadModelForCacheBreakdown.render(appLocale))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
                .padding(.top, Theme.Spacing.sm)
        } else {
            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                LazyVGrid(columns: [
                    GridItem(.flexible()),
                    GridItem(.flexible()),
                ], spacing: Theme.Spacing.sm) {
                    statCell("Total layers", "\(total)")
                    statCell("KV simple", "\(kv)")
                    statCell("Sliding window", "\(rot)")
                    statCell("Hybrid SSM", "\(m)")
                    statCell("TurboQuant KV", "\(tq)")
                    statCell("Quantized KV", "\(q)")
                    if other > 0 {
                        statCell("Other", "\(other)")
                    }
                }
                .padding(.top, 2)
            }
            .padding(.top, Theme.Spacing.sm)
        }
    }

    @ViewBuilder
    private var pagedSection: some View {
        if (paged("enabled") as? Bool) == true {
            let inUse = pagedInt("blocksInUse")
            let maxBlocks = pagedInt("maxBlocks")
            let usable = max(maxBlocks - 1, 1)
            let hits = pagedInt("hitCount")
            let misses = pagedInt("missCount")
            let hitRate = pagedDouble("hitRate")
            let blockSize = pagedInt("blockSize")
            let evictions = pagedInt("evictions")

            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                ProgressBar(
                    label: "Blocks",
                    detail: "\(inUse) / \(usable)",
                    fraction: Double(inUse) / Double(usable),
                    tint: Theme.Colors.accent
                )
                LazyVGrid(columns: [
                    GridItem(.flexible()),
                    GridItem(.flexible()),
                ], spacing: Theme.Spacing.sm) {
                    statCell("Hit rate", String(format: "%.1f%%", hitRate * 100.0))
                    statCell("Hits / Misses", "\(hits) / \(misses)")
                    statCell("Block size", "\(blockSize)")
                    statCell("Evictions", "\(evictions)")
                }
                .padding(.top, 2)
            }
            .padding(.top, Theme.Spacing.sm)
        } else {
            disabledHint(.paged)
        }
    }

    @ViewBuilder
    private var memorySection: some View {
        if (memory("enabled") as? Bool) == true {
            let currentMB = memoryDouble("currentMemoryMB")
            let maxMB = memoryDouble("maxMemoryMB")
            let entryCount = memoryInt("entryCount")
            let hits = memoryInt("hitCount")
            let misses = memoryInt("missCount")
            let hitRate = memoryDouble("hitRate")
            let evictions = memoryInt("evictions")
            let util = memoryDouble("utilizationPct")

            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                ProgressBar(
                    label: "Memory",
                    detail: String(format: "%.1f / %.1f MB", currentMB, maxMB),
                    fraction: maxMB > 0 ? currentMB / maxMB : 0,
                    tint: Theme.Colors.accent
                )
                LazyVGrid(columns: [
                    GridItem(.flexible()),
                    GridItem(.flexible()),
                ], spacing: Theme.Spacing.sm) {
                    statCell("Hit rate", String(format: "%.1f%%", hitRate * 100.0))
                    statCell("Hits / Misses", "\(hits) / \(misses)")
                    statCell("Entries", "\(entryCount)")
                    statCell("Evictions", "\(evictions)")
                    statCell("Utilization", String(format: "%.1f%%", util))
                }
                .padding(.top, 2)
            }
            .padding(.top, Theme.Spacing.sm)
        } else {
            disabledHint(.memory)
        }
    }

    @ViewBuilder
    private var diskSection: some View {
        if (disk("enabled") as? Bool) == true {
            let currentGB = diskDouble("currentGB")
            let maxGB = diskDouble("maxGB")
            let entryCount = diskInt("entryCount")
            let hits = diskInt("hitCount")
            let misses = diskInt("missCount")
            let hitRate = diskDouble("hitRate")
            let directory = diskString("directory")

            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                ProgressBar(
                    label: "Size",
                    detail: String(format: "%.2f / %.1f GB", currentGB, maxGB),
                    fraction: maxGB > 0 ? currentGB / maxGB : 0.0,
                    tint: Theme.Colors.success
                )
                LazyVGrid(columns: [
                    GridItem(.flexible()),
                    GridItem(.flexible()),
                ], spacing: Theme.Spacing.sm) {
                    statCell("Entries", "\(entryCount)")
                    statCell("Hit rate", String(format: "%.1f%%", hitRate * 100.0))
                    statCell("Hits / Misses", "\(hits) / \(misses)")
                    statCell("Stores", "\(diskInt("storeCount"))")
                }
                if !directory.isEmpty {
                    Text(directory)
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .padding(.top, 2)
                }
            }
            .padding(.top, Theme.Spacing.sm)
        } else {
            disabledHint(.disk)
        }
    }

    @ViewBuilder
    private var ssmSection: some View {
        if (ssm("enabled") as? Bool) == true {
            let hits = ssmInt("hitCount")
            let misses = ssmInt("missCount")
            let hitRate = ssmDouble("hitRate")
            let maxEntries = ssmInt("maxEntries")
            // iter-40: re-derive watcher activity. Fires when a
            // thinking-template turn triggers a fresh prompt-only
            // forward pass because the post-generation SSM state is
            // contaminated. Zero for non-thinking models.
            let reDerives = ssmInt("reDerives")

            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                LazyVGrid(columns: [
                    GridItem(.flexible()),
                    GridItem(.flexible()),
                ], spacing: Theme.Spacing.sm) {
                    statCell("Max entries", "\(maxEntries)")
                    statCell("Hit rate", String(format: "%.1f%%", hitRate * 100.0))
                    statCell("Hits / Misses", "\(hits) / \(misses)")
                    statCell("Re-derives", "\(reDerives)")
                }
            }
            .padding(.top, Theme.Spacing.sm)
        } else {
            Text(L10n.ServerUI.hybridInactive.render(appLocale))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
                .padding(.top, Theme.Spacing.sm)
        }
    }

    // MARK: - Chrome

    private var footer: some View {
        HStack {
            if let status = warmStatus {
                Text(status)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
            }
            Spacer()

            // Warm button: replays the last 5 user messages through the
            // engine at max_tokens=1 so the prefix cache populates
            // naturally. Useful after a clear, or after switching models
            // to repopulate common system prompts. Disabled while a
            // clear is in flight to avoid racing with empty state.
            Button {
                Task { await warmFromRecent() }
            } label: {
                HStack(spacing: 4) {
                    if warmInFlight {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: "flame.fill")
                    }
                    Text(warmInFlight ? "Warming…" : "Warm from recent")
                        .font(Theme.Typography.caption)
                }
                .padding(.horizontal, Theme.Spacing.md)
                .padding(.vertical, Theme.Spacing.xs)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.surfaceHi)
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.md)
                                .stroke(Theme.Colors.border, lineWidth: 1)
                        )
                )
                .foregroundStyle(Theme.Colors.textHigh)
            }
            .buttonStyle(.plain)
            .disabled(warmInFlight || clearInFlight)
            .help(L10n.Tooltip.cacheWarm5.render(appLocale))

            Button {
                showClearConfirm = true
            } label: {
                HStack(spacing: 4) {
                    if clearInFlight {
                        ProgressView()
                            .controlSize(.small)
                    }
                    Text(clearInFlight ? "Clearing…" : "Clear caches")
                        .font(Theme.Typography.caption)
                }
                .padding(.horizontal, Theme.Spacing.md)
                .padding(.vertical, Theme.Spacing.xs)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.surfaceHi)
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.md)
                                .stroke(Theme.Colors.border, lineWidth: 1)
                        )
                )
                .foregroundStyle(Theme.Colors.textHigh)
            }
            .buttonStyle(.plain)
            .disabled(clearInFlight)
            .confirmationDialog(
                "Clear all caches?",
                isPresented: $showClearConfirm,
                titleVisibility: .visible
            ) {
                Button(L10n.Misc.clearCaches.render(appLocale), role: .destructive) {
                    Task { await clearCaches() }
                }
                Button(L10n.Common.cancel.render(appLocale), role: .cancel) { }
            } message: {
                Text(L10n.ServerUI.dropCacheHelp.render(appLocale))
            }
        }
    }

    private var hybridPill: some View {
        pill(text: "hybrid", tint: Theme.Colors.accent, textTint: Theme.Colors.accentHi)
    }

    private var swaPill: some View {
        pill(text: "SWA", tint: Theme.Colors.success, textTint: Theme.Colors.success)
    }

    private var tqPill: some View {
        pill(text: "TQ", tint: Theme.Colors.accent, textTint: Theme.Colors.accentHi)
    }

    @ViewBuilder
    private func pill(text: String, tint: Color, textTint: Color) -> some View {
        Text(text)
            .font(Theme.Typography.caption)
            .foregroundStyle(textTint)
            .padding(.horizontal, 6)
            .padding(.vertical, 1)
            .background(
                Capsule().fill(tint.opacity(0.15))
            )
            .overlay(Capsule().stroke(tint.opacity(0.4), lineWidth: 1))
    }

    @ViewBuilder
    private func sectionLabel(_ title: String, enabled: Any?) -> some View {
        let on = (enabled as? Bool) ?? false
        HStack(spacing: Theme.Spacing.sm) {
            Text(title)
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textHigh)
            Text(on ? "on" : "off")
                .font(Theme.Typography.caption)
                .foregroundStyle(on ? Theme.Colors.success : Theme.Colors.textLow)
        }
    }

    /// Per-tier "off" explanation. Each tier has different reasons
    /// for being disabled — memory cache defaults off, disk is
    /// user-toggled, SSM is auto-enabled by hybrid detection. The
    /// audit flagged "Disabled" without explanation as a
    /// discoverability gap.
    @ViewBuilder
    private func disabledHint(_ tier: CacheTier) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(L10n.ServerUI.disabled.render(appLocale))
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            Text(tier.disabledHelp)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 1)
        }
        .padding(.top, Theme.Spacing.sm)
    }

    private enum CacheTier {
        case paged, memory, disk, ssm
        var disabledHelp: String {
            switch self {
            case .paged:
                return "Enable `Use paged cache` in Server settings to let long multi-turn prompts reuse previous-turn KV state in-memory."
            case .memory:
                return "Enable `Memory cache (L1.5)` in Server settings for byte-budgeted whole-prompt reuse. Useful for long sessions with RAM to spare."
            case .disk:
                return "Enable `Disk cache (L2)` in Server settings to persist KV state across restarts. Default on for new installs."
            case .ssm:
                return "SSM companion tier is auto-enabled for hybrid models (Nemotron-H, Qwen3.5-A3B, Jamba, FalconH1, Mamba variants). Off for pure-attention models."
            }
        }
    }

    @ViewBuilder
    private func statCell(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            Text(value)
                .font(Theme.Typography.mono)
                .foregroundStyle(Theme.Colors.textHigh)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(Theme.Spacing.sm)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.md)
                .fill(Theme.Colors.surfaceHi)
        )
    }

    // MARK: - Polling + actions

    private func startPolling() {
        pollTask?.cancel()
        pollTask = Task { @MainActor in
            while !Task.isCancelled {
                await refresh()
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    private func refresh() async {
        guard let engine = app.activeEngine else {
            loaded = false
            return
        }
        do {
            let s = try await engine.cacheStats()
            stats = s
            loaded = (s["loaded"] as? Bool) ?? false
        } catch {
            loaded = false
        }
    }

    private func clearCaches() async {
        guard let engine = app.activeEngine else { return }
        clearInFlight = true
        await engine.clearCaches()
        clearInFlight = false
        await refresh()
    }

    /// Warm the prefix cache by replaying the 5 most recent user messages
    /// through the engine at max_tokens=1. Mirrors what the
    /// `POST /v1/cache/warm` admin endpoint does so the UI affordance
    /// and the HTTP API stay in lockstep.
    private func warmFromRecent() async {
        guard let engine = app.activeEngine else {
            warmStatus = "Load a model first"
            return
        }
        warmInFlight = true
        defer { warmInFlight = false }

        let recent = Database.shared
            .recentUserPrompts(limit: 5)
            .filter { !$0.isEmpty }
        guard !recent.isEmpty else {
            warmStatus = "No recent prompts to warm from"
            return
        }
        do {
            let modelName = app.selectedModelPath?.lastPathComponent ?? ""
            let warmed = try await engine.cacheWarm(
                prompts: recent, model: modelName
            )
            warmStatus = "Warmed \(warmed) prompt\(warmed == 1 ? "" : "s")"
            await refresh()
        } catch {
            warmStatus = "Warm failed: \(error)"
        }
    }

    // MARK: - Value extractors

    private func paged(_ key: String) -> Any? { (stats["paged"] as? [String: Any])?[key] }
    private func memory(_ key: String) -> Any? { (stats["memory"] as? [String: Any])?[key] }
    private func disk(_ key: String) -> Any? { (stats["disk"] as? [String: Any])?[key] }
    private func ssm(_ key: String) -> Any? { (stats["ssmCompanion"] as? [String: Any])?[key] }
    private func arch(_ key: String) -> Any? { (stats["architecture"] as? [String: Any])?[key] }
    private func archInt(_ key: String) -> Int { (arch(key) as? Int) ?? 0 }
    private func archBool(_ key: String) -> Bool { (arch(key) as? Bool) ?? false }

    private func pagedInt(_ key: String) -> Int { (paged(key) as? Int) ?? 0 }
    private func pagedDouble(_ key: String) -> Double { (paged(key) as? Double) ?? 0 }
    private func memoryInt(_ key: String) -> Int { (memory(key) as? Int) ?? 0 }
    private func memoryDouble(_ key: String) -> Double { (memory(key) as? Double) ?? 0 }
    private func diskInt(_ key: String) -> Int { (disk(key) as? Int) ?? 0 }
    private func diskDouble(_ key: String) -> Double { (disk(key) as? Double) ?? 0 }
    private func diskString(_ key: String) -> String { (disk(key) as? String) ?? "" }
    private func ssmInt(_ key: String) -> Int { (ssm(key) as? Int) ?? 0 }
    private func ssmDouble(_ key: String) -> Double { (ssm(key) as? Double) ?? 0 }
}

// MARK: - ProgressBar

private struct ProgressBar: View {
    let label: String
    let detail: String
    let fraction: Double
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                Spacer()
                Text(detail)
                    .font(Theme.Typography.mono)
                    .foregroundStyle(Theme.Colors.textMid)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Theme.Colors.surfaceHi)
                    RoundedRectangle(cornerRadius: 2)
                        .fill(tint)
                        .frame(width: geo.size.width * clamp(fraction))
                }
            }
            .frame(height: 6)
        }
    }

    private func clamp(_ x: Double) -> CGFloat {
        CGFloat(min(max(x, 0.0), 1.0))
    }
}
