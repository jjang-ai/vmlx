import SwiftUI
import Foundation
import vMLXTheme
#if canImport(AppKit)
import AppKit
#endif

/// Top-of-window banner that shows up when a newer vMLX release is
/// available on GitHub. Wired at the top of `RootView` so it stacks above
/// `DownloadStatusBar`.
///
/// Phase 3 ships with a **stub version service** — the real fetcher against
/// `https://api.github.com/repos/jjang-ai/mlxstudio/releases/latest` slots
/// into `UpdateAvailableService.check()` later. UI is done now so there's
/// nothing to wire on release day.
struct UpdateAvailableBanner: View {
    @State private var latest: UpdateInfo? = nil

    var body: some View {
        Group {
            if let info = latest {
                HStack(spacing: Theme.Spacing.sm) {
                    Image(systemName: "arrow.down.circle.fill")
                        .foregroundStyle(Theme.Colors.accent)
                    Text("vMLX v\(info.version) available — download")
                        .font(Theme.Typography.bodyHi)
                        .foregroundStyle(Theme.Colors.textHigh)
                    Spacer()
                    Button("Download") {
                        if let url = URL(string: info.htmlURL) {
                            #if canImport(AppKit)
                            NSWorkspace.shared.open(url)
                            #endif
                        }
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Theme.Colors.accent)
                    Button {
                        latest = nil
                    } label: {
                        Image(systemName: "xmark")
                            .foregroundStyle(Theme.Colors.textLow)
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, Theme.Spacing.lg)
                .padding(.vertical, Theme.Spacing.sm)
                .background(Theme.Colors.surfaceHi)
            }
        }
        .task {
            latest = await UpdateAvailableService.shared.check()
            // Re-check every 6 hours while the window is alive.
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(6 * 3600) * 1_000_000_000)
                if Task.isCancelled { break }
                if let next = await UpdateAvailableService.shared.check() {
                    latest = next
                }
            }
        }
    }
}

struct UpdateInfo: Equatable, Sendable {
    let version: String
    let htmlURL: String
}

/// Real GitHub Releases fetcher. Hits
/// `https://api.github.com/repos/jjang-ai/mlxstudio/releases/latest` once per
/// hour at most (UserDefaults-backed throttle) and returns an `UpdateInfo`
/// when `tag_name` (with a leading `v` stripped) is newer than the bundle's
/// `CFBundleShortVersionString`. Comparison is pure-Swift semver via
/// `compareVersions` — no third-party deps.
actor UpdateAvailableService {
    static let shared = UpdateAvailableService()

    private let releasesURL = URL(
        string: "https://api.github.com/repos/jjang-ai/mlxstudio/releases/latest"
    )!
    private let throttle: TimeInterval = 3600  // 1 hour
    private let lastCheckKey = "vmlx.updateCheck.lastCheckAt"
    private let cachedInfoKey = "vmlx.updateCheck.cachedInfo"

    func check() async -> UpdateInfo? {
        // Throttle: bail early if the last hit happened within `throttle`.
        let defaults = UserDefaults.standard
        if let last = defaults.object(forKey: lastCheckKey) as? Date,
           Date().timeIntervalSince(last) < throttle {
            return cachedInfo()
        }

        var req = URLRequest(url: releasesURL)
        req.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        req.setValue("vMLX-swift", forHTTPHeaderField: "User-Agent")
        req.timeoutInterval = 10

        do {
            let (data, response) = try await URLSession.shared.data(for: req)
            defaults.set(Date(), forKey: lastCheckKey)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                return cachedInfo()
            }
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let rawTag = json["tag_name"] as? String else {
                return cachedInfo()
            }
            let latestVersion = rawTag.trimmingCharacters(in: CharacterSet(charactersIn: "vV "))
            let htmlURL = (json["html_url"] as? String)
                ?? "https://github.com/jjang-ai/mlxstudio/releases/latest"

            let current = Self.currentVersion()
            if Self.compareVersions(latestVersion, current) == .orderedDescending {
                let info = UpdateInfo(version: latestVersion, htmlURL: htmlURL)
                cacheInfo(info)
                return info
            }
            // Up to date — clear any stale cached info.
            defaults.removeObject(forKey: cachedInfoKey)
            return nil
        } catch {
            return cachedInfo()
        }
    }

    // MARK: - Cache + bundle version

    private func cachedInfo() -> UpdateInfo? {
        guard let dict = UserDefaults.standard.dictionary(forKey: cachedInfoKey),
              let v = dict["version"] as? String,
              let u = dict["url"] as? String else { return nil }
        return UpdateInfo(version: v, htmlURL: u)
    }

    private func cacheInfo(_ info: UpdateInfo) {
        UserDefaults.standard.set(
            ["version": info.version, "url": info.htmlURL],
            forKey: cachedInfoKey
        )
    }

    static func currentVersion() -> String {
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String)
            ?? "0.0.0"
    }

    // MARK: - Semver compare

    /// Compare two dot-separated version strings component-wise. Non-numeric
    /// segments fall back to lexical compare. Missing components are treated
    /// as zero (so `1.3` == `1.3.0`).
    static func compareVersions(_ a: String, _ b: String) -> ComparisonResult {
        func parts(_ s: String) -> [Int] {
            let cleaned = s.trimmingCharacters(in: CharacterSet(charactersIn: "vV "))
            return cleaned.split(separator: ".").map { Int($0) ?? 0 }
        }
        let la = parts(a)
        let lb = parts(b)
        let n = max(la.count, lb.count)
        for i in 0..<n {
            let ai = i < la.count ? la[i] : 0
            let bi = i < lb.count ? lb[i] : 0
            if ai < bi { return .orderedAscending }
            if ai > bi { return .orderedDescending }
        }
        return .orderedSame
    }
}
