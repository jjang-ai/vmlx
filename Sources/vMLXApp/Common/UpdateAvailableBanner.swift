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
/// Calls `UpdateAvailableService.check()` on appear and every 6 hours
/// while the window is alive. The service hits
/// `https://api.github.com/repos/jjang-ai/mlxstudio/releases/latest`,
/// throttled to one network request per hour via UserDefaults, and
/// compares the returned `tag_name` (with a leading `v` stripped)
/// against this bundle's `CFBundleShortVersionString` using the
/// pure-Swift `compareVersions` semver helper. Banner appears only
/// when remote > local; network failures return the last cached
/// `UpdateInfo` so the UI isn't brittle on flaky links.
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
            // **iter-83 (§111)** — validate the release URL before
            // trusting it. `NSWorkspace.shared.open(url)` is executed
            // verbatim when the user hits the Download button, and it
            // will happily dispatch ANY scheme (javascript:, file:,
            // vmlx:// custom scheme, ftp:, etc.). A compromised
            // GitHub API response or a MITM on the update-check
            // request could swap `html_url` for a phishing page or a
            // local-file trigger. Require https:// + our own repo
            // domain; fall back to a hardcoded safe URL otherwise.
            let htmlURL = Self.validatedReleaseURL(
                json["html_url"] as? String
            ) ?? "https://github.com/jjang-ai/mlxstudio/releases/latest"

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

    // MARK: - iter-83 §111 release URL validation

    /// Verify that the release URL coming back from the GitHub API
    /// points at our release page on `github.com` with an `https`
    /// scheme. Rejects any other scheme (js:, file:, data:, custom
    /// URL schemes), hosts (phishing lookalikes), or malformed URLs.
    /// Returns the original string when safe; `nil` when rejected so
    /// the caller can fall back to a hardcoded URL.
    internal static func validatedReleaseURL(_ raw: String?) -> String? {
        guard let raw, !raw.isEmpty,
              let components = URLComponents(string: raw),
              components.scheme?.lowercased() == "https",
              let host = components.host?.lowercased()
        else {
            return nil
        }
        // Only accept the release host — github.com (renders the HTML
        // release page) or api.github.com (JSON API). Everything else
        // including GitHub Pages sub-domains is out.
        let allowedHosts: Set<String> = ["github.com", "api.github.com"]
        guard allowedHosts.contains(host) else { return nil }
        return raw
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
