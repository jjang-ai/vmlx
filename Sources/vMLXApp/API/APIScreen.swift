import SwiftUI
import vMLXEngine
import vMLXTheme

/// Full-parity API screen. Replaces the scaffold from Phase 2 with:
///   * live endpoint (host/port/lan from SettingsStore)
///   * port validation + collision warning
///   * LAN 127.0.0.1 ↔ 0.0.0.0 toggle
///   * running-sessions list (bound to AppState.sessions)
///   * format-tabbed code snippets (openai/anthropic/ollama/curl/python/ts)
///   * persisted keys via APIKeyManager (SQLite)
///   * bearer-auth wiring into SettingsStore.apiKey
///   * phone-scannable QR code when LAN is enabled
struct APIScreen: View {
    @Environment(AppState.self) private var app

    @State private var keys: [Database.APIKeyRow] = []
    @State private var newKeyLabel: String = ""
    @State private var bearerRequired: Bool = false
    @State private var snippetFormat: SnippetFormat = .curl

    /// ID of the key pending revoke confirmation. Non-nil drives the
    /// confirmation dialog; tapping Revoke flushes it via `revoke(id:)`.
    @State private var pendingRevokeId: String? = nil

    // Live-edited endpoint settings (debounced into SettingsStore on change).
    @State private var hostBinding: String = "127.0.0.1"
    @State private var portBinding: Int = 8000
    @State private var lanBinding: Bool = false

    enum SnippetFormat: String, CaseIterable, Identifiable {
        case openai, anthropic, ollama, curl, python, typescript
        var id: String { rawValue }
        var label: String {
            switch self {
            case .openai:     return "OpenAI"
            case .anthropic:  return "Anthropic"
            case .ollama:     return "Ollama"
            case .curl:       return "curl"
            case .python:     return "Python"
            case .typescript: return "TypeScript"
            }
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.xl) {
                Text("API")
                    .font(Theme.Typography.title)
                    .foregroundStyle(Theme.Colors.textHigh)

                endpointCard
                sessionsCard
                keysCard
                HuggingFaceTokenCard()
                RequestLogPanel()
                MCPPanel()
                snippetsCard
            }
            .padding(Theme.Spacing.xl)
        }
        .background(Theme.Colors.background)
        .task { await loadInitial() }
        .task {
            for await snap in APIKeyManager.shared.subscribe() {
                keys = snap
            }
        }
    }

    // MARK: - derived

    /// Client-facing endpoint URL. `0.0.0.0` never works for a client — maps
    /// to `127.0.0.1` per `panel/src/renderer/src/lib/connectHost.ts`.
    private var endpointURL: String {
        let h = connectHost(hostBinding)
        return "http://\(h):\(portBinding)"
    }

    /// Machine-visible LAN URL (when LAN binding is on). This is what the QR
    /// code encodes so a phone on the same network can reach the server.
    ///
    /// Resolves the machine's actual LAN IP (first non-loopback IPv4
    /// interface address) so scanning the QR hits the right host instead
    /// of `0.0.0.0` which is meaningless to a client. Falls back to the
    /// bound host if the interface enumeration fails.
    private var lanURL: String? {
        guard lanBinding else { return nil }
        let host = Self.resolveLANAddress() ?? hostBinding
        return "http://\(host):\(portBinding)"
    }

    /// Walks `getifaddrs` looking for the first non-loopback IPv4 interface
    /// that is up and running. Returns the dotted string form, or `nil`
    /// if none is available (airplane mode, no active NIC).
    ///
    /// Prefers `en0` (built-in Wi-Fi) then `en1+` (USB-C/Thunderbolt
    /// Ethernet) so the QR shows the physically nearest adapter when
    /// multiple networks are active.
    private static func resolveLANAddress() -> String? {
        var head: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&head) == 0, let first = head else { return nil }
        defer { freeifaddrs(first) }

        var candidates: [(iface: String, ip: String)] = []
        var ptr: UnsafeMutablePointer<ifaddrs>? = first
        while let cur = ptr {
            defer { ptr = cur.pointee.ifa_next }
            guard let addr = cur.pointee.ifa_addr else { continue }
            let family = addr.pointee.sa_family
            guard family == UInt8(AF_INET) else { continue }
            let iface = String(cString: cur.pointee.ifa_name)
            // Skip loopback + link-local.
            guard iface != "lo0", !iface.hasPrefix("llw") else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            let res = getnameinfo(
                addr, socklen_t(addr.pointee.sa_len),
                &host, socklen_t(host.count),
                nil, 0, NI_NUMERICHOST
            )
            guard res == 0 else { continue }
            let ip = String(cString: host)
            guard !ip.hasPrefix("127."), !ip.hasPrefix("169.254.") else { continue }
            candidates.append((iface, ip))
        }

        // Prefer en0 (Wi-Fi on most Macs), then en1+ (Ethernet), then anything.
        if let wifi = candidates.first(where: { $0.iface == "en0" }) {
            return wifi.ip
        }
        if let eth = candidates.first(where: { $0.iface.hasPrefix("en") }) {
            return eth.ip
        }
        return candidates.first?.ip
    }

    private var portInUse: Bool {
        app.sessions.contains(where: { $0.port == portBinding && $0.isActiveGroup })
    }

    private var portValid: Bool {
        (1024...65535).contains(portBinding)
    }

    // MARK: - cards

    private var endpointCard: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            Text("ENDPOINT")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            HStack {
                Text(endpointURL)
                    .font(Theme.Typography.mono)
                    .foregroundStyle(Theme.Colors.textHigh)
                    .textSelection(.enabled)
                Spacer()
                Button("Copy") { copyToPasteboard(endpointURL) }
                    .buttonStyle(.plain)
                    .font(Theme.Typography.bodyHi)
                    .foregroundStyle(Theme.Colors.accent)
            }

            HStack(spacing: Theme.Spacing.md) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Port")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                    HStack(spacing: Theme.Spacing.xs) {
                        TextField("", value: $portBinding, format: .number.grouping(.never))
                            .textFieldStyle(.plain)
                            .font(Theme.Typography.mono)
                            .foregroundStyle(Theme.Colors.textHigh)
                            .frame(width: 80)
                            .padding(Theme.Spacing.xs)
                            .background(
                                RoundedRectangle(cornerRadius: Theme.Radius.sm)
                                    .fill(Theme.Colors.surfaceHi)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: Theme.Radius.sm)
                                            .stroke(portValid ? Theme.Colors.border : Theme.Colors.danger, lineWidth: 1)
                                    )
                            )
                        Stepper("", value: $portBinding, in: 1024...65535, step: 1)
                            .labelsHidden()
                    }
                }
                Toggle("LAN (0.0.0.0)", isOn: $lanBinding)
                    .toggleStyle(.switch)
                    .tint(Theme.Colors.accent)
                    .font(Theme.Typography.body)
                    .foregroundStyle(Theme.Colors.textHigh)
                    .onChange(of: lanBinding) { _, newValue in
                        hostBinding = newValue ? "0.0.0.0" : "127.0.0.1"
                        persistEndpoint()
                    }
                Spacer()
            }
            if !portValid {
                Text("Port must be between 1024 and 65535")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.danger)
            } else if portInUse {
                Text("Port \(portBinding) is already in use by another session")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.warning)
            }

            Toggle("Require Bearer auth", isOn: $bearerRequired)
                .toggleStyle(.switch)
                .tint(Theme.Colors.accent)
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textHigh)
                .onChange(of: bearerRequired) { _, _ in persistBearer() }

            if lanBinding, let lan = lanURL {
                HStack(alignment: .top, spacing: Theme.Spacing.md) {
                    VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                        Text("PHONE / LAN")
                            .font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Colors.textLow)
                        Text(lan)
                            .font(Theme.Typography.mono)
                            .foregroundStyle(Theme.Colors.textMid)
                            .textSelection(.enabled)
                        Text("Scan from another device on your network")
                            .font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Colors.textLow)
                    }
                    Spacer()
                    QRCodeView(text: lan, size: 120)
                }
                .padding(.top, Theme.Spacing.sm)
            }
        }
        .padding(Theme.Spacing.lg)
        .background(card)
        .onChange(of: portBinding) { _, _ in persistEndpoint() }
    }

    private var sessionsCard: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            Text("RUNNING SESSIONS")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            if app.sessions.isEmpty {
                Text("No running sessions. Start a model from the Server tab.")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            } else {
                ForEach(app.sessions) { s in
                    HStack(spacing: Theme.Spacing.sm) {
                        Circle()
                            .fill(dotColor(for: s.state))
                            .frame(width: 6, height: 6)
                        Text(s.displayName)
                            .font(Theme.Typography.bodyHi)
                            .foregroundStyle(Theme.Colors.textHigh)
                        Text("\(s.host):\(s.port)")
                            .font(Theme.Typography.mono)
                            .foregroundStyle(Theme.Colors.textMid)
                        Spacer()
                        Text(stateLabel(s.state))
                            .font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Colors.textLow)
                    }
                    .padding(.vertical, Theme.Spacing.xs)
                }
            }
        }
        .padding(Theme.Spacing.lg)
        .background(card)
    }

    private var keysCard: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            Text("API KEYS")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            keysCardGenerateRow
            keysCardList
        }
        .padding(Theme.Spacing.lg)
        .background(card)
        .confirmationDialog(
            "Revoke this API key?",
            isPresented: revokeDialogBinding,
            titleVisibility: .visible,
            actions: { revokeDialogActions },
            message: { revokeDialogMessage }
        )
    }

    private var keysCardGenerateRow: some View {
        HStack(spacing: Theme.Spacing.sm) {
            TextField("Key label", text: $newKeyLabel)
                .textFieldStyle(.plain)
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textHigh)
                .padding(Theme.Spacing.sm)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.surfaceHi)
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.md)
                                .stroke(Theme.Colors.border, lineWidth: 1)
                        )
                )
            Button("Generate") {
                _ = APIKeyManager.shared.generate(label: newKeyLabel)
                newKeyLabel = ""
            }
            .buttonStyle(.plain)
            .font(Theme.Typography.bodyHi)
            .foregroundStyle(Theme.Colors.accent)
            .disabled(newKeyLabel.trimmingCharacters(in: .whitespaces).isEmpty)
        }
    }

    @ViewBuilder
    private var keysCardList: some View {
        if keys.isEmpty {
            EmptyStateView(
                systemImage: "key",
                title: "No API keys yet",
                caption: "Generate a key above to use with OpenAI / Anthropic / Ollama clients."
            )
            .frame(height: 140)
        } else {
            ForEach(keys) { k in
                keyRow(k)
            }
        }
    }

    private func keyRow(_ k: Database.APIKeyRow) -> some View {
        HStack(spacing: Theme.Spacing.sm) {
            Text(k.label)
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            Text(k.value)
                .font(Theme.Typography.mono)
                .foregroundStyle(Theme.Colors.textMid)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Button("Copy") { copyToPasteboard(k.value) }
                .buttonStyle(.plain)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.accent)
            Button {
                pendingRevokeId = k.id
            } label: {
                Image(systemName: "trash")
                    .foregroundStyle(Theme.Colors.textLow)
            }
            .buttonStyle(.plain)
            .help("Revoke API key")
        }
        .padding(Theme.Spacing.sm)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.md)
                .fill(Theme.Colors.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .stroke(Theme.Colors.border, lineWidth: 1)
                )
        )
    }

    private var revokeDialogBinding: Binding<Bool> {
        Binding(
            get: { pendingRevokeId != nil },
            set: { if !$0 { pendingRevokeId = nil } }
        )
    }

    @ViewBuilder
    private var revokeDialogActions: some View {
        Button("Revoke", role: .destructive) {
            if let id = pendingRevokeId {
                APIKeyManager.shared.revoke(id: id)
            }
            pendingRevokeId = nil
        }
        Button("Cancel", role: .cancel) { pendingRevokeId = nil }
    }

    private var revokeDialogMessage: Text {
        Text("Any client using this key will immediately lose access. This can't be undone — a new key will have a different value.")
    }

    private var snippetsCard: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            HStack {
                Text("CODE SNIPPETS")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                Spacer()
                Picker("", selection: $snippetFormat) {
                    ForEach(SnippetFormat.allCases) { f in
                        Text(f.label).tag(f)
                    }
                }
                .labelsHidden()
                .frame(width: 160)
            }
            CodeBlockView(language: snippetFormat.rawValue, code: currentSnippet)
        }
        .padding(Theme.Spacing.lg)
        .background(card)
    }

    // MARK: - snippet strings

    private var currentSnippet: String {
        let ep = endpointURL
        let key = bearerRequired ? "$VMLX_API_KEY" : "not-needed"
        switch snippetFormat {
        case .curl:
            let authLine = bearerRequired ? "  -H \"Authorization: Bearer \(key)\" \\\n" : ""
            return """
            curl \(ep)/v1/chat/completions \\
              -H "Content-Type: application/json" \\
            \(authLine)  -d '{
                "model": "auto",
                "messages": [{"role":"user","content":"hello"}]
              }'
            """
        case .openai:
            return """
            from openai import OpenAI
            client = OpenAI(base_url="\(ep)/v1", api_key="\(key)")
            resp = client.chat.completions.create(
                model="auto",
                messages=[{"role":"user","content":"hello"}],
            )
            print(resp.choices[0].message.content)
            """
        case .anthropic:
            return """
            from anthropic import Anthropic
            client = Anthropic(base_url="\(ep)", api_key="\(key)")
            resp = client.messages.create(
                model="auto",
                max_tokens=1024,
                messages=[{"role":"user","content":"hello"}],
            )
            print(resp.content[0].text)
            """
        case .ollama:
            return """
            curl \(ep)/api/chat \\
              -d '{
                "model": "auto",
                "messages": [{"role":"user","content":"hello"}],
                "stream": false
              }'
            """
        case .python:
            let auth = bearerRequired ? "    \"Authorization\": f\"Bearer \(key)\",\n" : ""
            return """
            import requests
            headers = {
                "Content-Type": "application/json",
            \(auth)}
            r = requests.post(
                "\(ep)/v1/chat/completions",
                headers=headers,
                json={"model":"auto","messages":[{"role":"user","content":"hello"}]},
            )
            print(r.json()["choices"][0]["message"]["content"])
            """
        case .typescript:
            let auth = bearerRequired ? "    Authorization: `Bearer ${process.env.VMLX_API_KEY}`,\n" : ""
            return """
            const r = await fetch("\(ep)/v1/chat/completions", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
            \(auth)  },
              body: JSON.stringify({
                model: "auto",
                messages: [{ role: "user", content: "hello" }],
              }),
            });
            const j = await r.json();
            console.log(j.choices[0].message.content);
            """
        }
    }

    // MARK: - persistence

    private func loadInitial() async {
        let g = await app.engine.settings.global()
        hostBinding = g.defaultHost
        portBinding = g.defaultPort
        lanBinding = (g.defaultHost == "0.0.0.0")
        bearerRequired = (g.apiKey != nil && !(g.apiKey ?? "").isEmpty)
        keys = APIKeyManager.shared.list()
    }

    private func persistEndpoint() {
        Task {
            var g = await app.engine.settings.global()
            g.defaultHost = hostBinding
            g.defaultPort = portBinding
            await app.engine.applySettings(g)
        }
    }

    private func persistBearer() {
        Task {
            var g = await app.engine.settings.global()
            if bearerRequired {
                // Use the most recently generated key (or generate one).
                let list = APIKeyManager.shared.list()
                let value = list.first?.value ?? APIKeyManager.shared.generate(label: "Default").value
                g.apiKey = value
            } else {
                g.apiKey = nil
            }
            await app.engine.applySettings(g)
        }
    }

    // MARK: - styling helpers

    private var card: some View {
        RoundedRectangle(cornerRadius: Theme.Radius.lg)
            .fill(Theme.Colors.surface)
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.lg)
                    .stroke(Theme.Colors.border, lineWidth: 1)
            )
    }

    private func copyToPasteboard(_ s: String) {
        #if canImport(AppKit)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(s, forType: .string)
        #endif
    }

    private func connectHost(_ host: String) -> String {
        host == "0.0.0.0" ? "127.0.0.1" : host
    }

    private func dotColor(for s: EngineState) -> Color {
        switch s {
        case .running: return Theme.Colors.success
        case .loading: return Theme.Colors.accent
        case .standby: return Theme.Colors.warning
        case .error:   return Theme.Colors.danger
        case .stopped: return Theme.Colors.textLow
        }
    }

    private func stateLabel(_ s: EngineState) -> String {
        switch s {
        case .running:        return "running"
        case .loading:        return "loading"
        case .standby(.soft): return "light sleep"
        case .standby(.deep): return "deep sleep"
        case .error:          return "error"
        case .stopped:        return "stopped"
        }
    }
}
