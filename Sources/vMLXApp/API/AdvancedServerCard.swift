import SwiftUI
import vMLXEngine
import vMLXTheme

/// O6 §292 — Advanced server settings card.
///
/// Surfaces CORS `allowedOrigins`, per-IP `rateLimit`, and TLS
/// `sslKeyFile` / `sslCertFile` — all three live in GlobalSettings but
/// previously had no UI; users had to hand-edit SQLite or run the CLI
/// with the right flags. All four fields are **per-session** in the
/// §276 apply-timing matrix (read by HTTPServerActor.start), so every
/// input is badged "Restart required" and the Apply button explicitly
/// stops + starts the running session to pick up the new values.
struct AdvancedServerCard: View {

    @Environment(AppState.self) private var app

    @State private var expanded: Bool = false
    @State private var corsOriginsText: String = ""
    @State private var rateLimitInt: Int = 0
    @State private var sslKeyFile: String = ""
    @State private var sslCertFile: String = ""
    @State private var dirty: Bool = false
    @State private var applying: Bool = false
    @State private var lastApplyMessage: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            header
            if expanded {
                Divider().opacity(0.3)
                corsRow
                rateLimitRow
                tlsRows
                applyRow
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
        .task { await loadInitial() }
    }

    // MARK: — sections

    private var header: some View {
        HStack {
            Button(action: { expanded.toggle() }) {
                HStack(spacing: 6) {
                    Image(systemName: expanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 10))
                        .foregroundStyle(Theme.Colors.textLow)
                    Text("ADVANCED — CORS, RATE LIMIT, TLS")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                }
            }
            .buttonStyle(.plain)
            Spacer()
            if !expanded {
                Text(summaryLabel)
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.Colors.textLow)
            } else {
                restartBadge
            }
        }
    }

    private var summaryLabel: String {
        var parts: [String] = []
        if !corsOriginsText.isEmpty && corsOriginsText != "*" {
            parts.append("CORS: \(corsOriginsText.split(separator: ",").count) origins")
        }
        if rateLimitInt > 0 {
            parts.append("\(rateLimitInt) req/min/IP")
        }
        if !sslKeyFile.isEmpty && !sslCertFile.isEmpty {
            parts.append("TLS on")
        }
        return parts.isEmpty ? "defaults" : parts.joined(separator: "  ·  ")
    }

    private var restartBadge: some View {
        HStack(spacing: 4) {
            Image(systemName: "arrow.clockwise")
                .font(.system(size: 9))
            Text("Restart required to apply")
                .font(.system(size: 10, weight: .medium))
        }
        .foregroundStyle(Theme.Colors.warning)
    }

    // CORS
    private var corsRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("CORS allowed origins")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            TextField("*",
                      text: $corsOriginsText,
                      prompt: Text("* for any origin, or comma-separated list"))
                .textFieldStyle(.plain)
                .font(Theme.Typography.body)
                .padding(Theme.Spacing.sm)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.surfaceHi)
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.md)
                                .stroke(Theme.Colors.border, lineWidth: 1)
                        )
                )
                .onChange(of: corsOriginsText) { _, _ in dirty = true }
            Text("Comma-separated. * means all. Single exact origin maps to Access-Control-Allow-Origin: <origin>. Multiple non-wildcard entries enable origin-echo mode.")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // Rate limit
    private var rateLimitRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Rate limit (requests / minute / IP)")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                Spacer()
                Text(rateLimitInt == 0 ? "disabled" : "\(rateLimitInt) req/min")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(rateLimitInt == 0 ? Theme.Colors.textLow : Theme.Colors.accent)
            }
            HStack {
                Slider(value: Binding(
                    get: { Double(rateLimitInt) },
                    set: { v in
                        let newVal = Int(v)
                        if newVal != rateLimitInt {
                            rateLimitInt = newVal
                            dirty = true
                        }
                    }), in: 0...600, step: 10)
                Button("0") { rateLimitInt = 0; dirty = true }
                    .buttonStyle(.plain)
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.Colors.textLow)
            }
        }
    }

    // TLS
    private var tlsRows: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            Text("TLS (HTTPS) — set BOTH to enable, leave blank for HTTP")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            filePathRow(label: "Private key (PEM)",
                        placeholder: "/path/to/server.key",
                        binding: $sslKeyFile)
            filePathRow(label: "Certificate (PEM)",
                        placeholder: "/path/to/server.crt",
                        binding: $sslCertFile)
            if !sslKeyFile.isEmpty || !sslCertFile.isEmpty {
                tlsStatusRow
            }
        }
    }

    private func filePathRow(label: String, placeholder: String,
                             binding: Binding<String>) -> some View {
        HStack(spacing: Theme.Spacing.sm) {
            TextField(placeholder, text: binding)
                .textFieldStyle(.plain)
                .font(.system(size: 11, design: .monospaced))
                .padding(Theme.Spacing.sm)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.surfaceHi)
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.md)
                                .stroke(Theme.Colors.border, lineWidth: 1)
                        )
                )
                .onChange(of: binding.wrappedValue) { _, _ in dirty = true }
            Button("Browse…") { browseFile(assignTo: binding) }
                .buttonStyle(.plain)
                .font(.system(size: 10))
                .foregroundStyle(Theme.Colors.accent)
            Text(label)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
        }
    }

    /// File-existence status — users frequently typo paths or forget
    /// absolute vs relative. Shows ✓ if the file exists + is readable,
    /// otherwise an explicit warning. Applies to both TLS files.
    @ViewBuilder
    private var tlsStatusRow: some View {
        let keyOK = !sslKeyFile.isEmpty && FileManager.default.isReadableFile(atPath: sslKeyFile)
        let certOK = !sslCertFile.isEmpty && FileManager.default.isReadableFile(atPath: sslCertFile)
        let bothSet = !sslKeyFile.isEmpty && !sslCertFile.isEmpty
        let oneSet = (!sslKeyFile.isEmpty) != (!sslCertFile.isEmpty)
        HStack(spacing: 6) {
            if oneSet {
                Image(systemName: "exclamationmark.triangle")
                    .foregroundStyle(Theme.Colors.warning)
                Text("Both key and cert must be set to enable TLS")
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.Colors.warning)
            } else if bothSet && (!keyOK || !certOK) {
                Image(systemName: "exclamationmark.circle")
                    .foregroundStyle(Theme.Colors.danger)
                Text("One of the TLS files is missing or unreadable")
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.Colors.danger)
            } else if bothSet && keyOK && certOK {
                Image(systemName: "checkmark.circle")
                    .foregroundStyle(Theme.Colors.success)
                Text("TLS will use https:// on next restart")
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.Colors.success)
            }
        }
    }

    // Apply row
    private var applyRow: some View {
        HStack(spacing: Theme.Spacing.md) {
            Button(applying ? "Applying..." : "Save & restart running sessions") {
                Task { await apply() }
            }
            .buttonStyle(.plain)
            .disabled(!dirty || applying)
            .foregroundStyle(dirty ? Theme.Colors.accent : Theme.Colors.textLow)
            .font(Theme.Typography.bodyHi)
            if !lastApplyMessage.isEmpty {
                Text(lastApplyMessage)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(lastApplyMessage.hasPrefix("✓")
                                     ? Theme.Colors.success
                                     : Theme.Colors.warning)
            }
            Spacer()
        }
    }

    // MARK: — actions

    private func loadInitial() async {
        let g = await app.engine.settings.global()
        corsOriginsText = g.corsOrigins.joined(separator: ", ")
        rateLimitInt = g.rateLimit
        sslKeyFile = g.sslKeyFile
        sslCertFile = g.sslCertFile
        dirty = false
    }

    /// Persist to GlobalSettings and stop+restart every running HTTP
    /// session. CORS/rate-limit/TLS are all per-session (§276 apply-timing)
    /// so a running listener keeps the OLD values until it rebinds.
    /// This button does the rebind safely.
    private func apply() async {
        applying = true
        defer { applying = false }
        var g = await app.engine.settings.global()
        let origins = corsOriginsText
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        g.corsOrigins = origins.isEmpty ? ["*"] : origins
        g.rateLimit = rateLimitInt
        g.sslKeyFile = sslKeyFile
        g.sslCertFile = sslCertFile
        await app.engine.applySettings(g)

        // Restart each session's HTTPServerActor so the new values take
        // effect. iter-86 §120 already exposes restart via Tray; we hop
        // through the same path here. Skip if no running servers.
        var restarted = 0
        for (id, srv) in app.httpServers {
            guard await srv.isRunning else { continue }
            let host = await srv.host
            let port = await srv.port
            await srv.stop()
            do {
                try await srv.start(
                    host: host,
                    port: port,
                    apiKey: g.apiKey,
                    adminToken: g.adminToken,
                    logLevel: .info,
                    allowedOrigins: g.corsOrigins,
                    rateLimitPerMinute: g.rateLimit,
                    tlsKeyPath: g.sslKeyFile.isEmpty ? nil : g.sslKeyFile,
                    tlsCertPath: g.sslCertFile.isEmpty ? nil : g.sslCertFile
                )
                restarted += 1
            } catch {
                lastApplyMessage = "⚠ restart failed for session \(id.uuidString.prefix(4)): \(error)"
                dirty = false
                return
            }
        }
        dirty = false
        lastApplyMessage = restarted == 0
            ? "✓ saved (no running sessions to restart)"
            : "✓ saved and restarted \(restarted) session(s)"
    }

    // MARK: — file picker

    private func browseFile(assignTo binding: Binding<String>) {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = []  // any file
        panel.title = "Select TLS file"
        panel.prompt = "Select"
        if panel.runModal() == .OK, let url = panel.url {
            binding.wrappedValue = url.path
            dirty = true
        }
        #endif
    }
}

#if canImport(AppKit)
import AppKit
#endif
