import SwiftUI
import AppKit
import vMLXEngine
import vMLXTheme

/// Live view of the engine's MCP server registry with a file picker for
/// the `mcp.json` config and per-server start/stop + per-tool inspect.
///
/// Reads `engine.mcp.listServers()` + `listTools()` on a 2-second poll so
/// the view stays fresh without needing an explicit subscription protocol
/// on the actor. When the user points at a new config path we persist it
/// to SessionSettings and call Engine.reloadMCPConfig — the engine does
/// the (potentially slow) startServer round-trips in the background and
/// the next poll surfaces the new state.
struct MCPPanel: View {
    @Environment(AppState.self) private var app

    @State private var servers: [MCPServerStatus] = []
    @State private var tools: [MCPTool] = []
    @State private var configPath: String = ""
    @State private var lastError: String? = nil
    @State private var pollTask: Task<Void, Never>? = nil
    @State private var reloading: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            header
            configRow
            if let err = lastError {
                errorBanner(err)
            }
            if servers.isEmpty {
                emptyState
            } else {
                serverList
                if !tools.isEmpty {
                    Divider().background(Theme.Colors.border)
                    toolList
                }
            }
        }
        .padding(Theme.Spacing.lg)
        .background(Theme.Colors.surface)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.lg))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .stroke(Theme.Colors.border, lineWidth: 1)
        )
        .task { await start() }
        .onDisappear {
            pollTask?.cancel()
            pollTask = nil
        }
    }

    // MARK: - Sub-views

    private var header: some View {
        HStack {
            Image(systemName: "puzzlepiece.extension")
                .foregroundStyle(Theme.Colors.accent)
            Text("MCP servers")
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)
            Spacer()
            Text("\(servers.count) configured · \(tools.count) tools")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
        }
    }

    private var configRow: some View {
        HStack(spacing: Theme.Spacing.sm) {
            TextField("Path to mcp.json — leave empty to use default",
                      text: $configPath)
                .textFieldStyle(.plain)
                .font(Theme.Typography.mono)
                .padding(.horizontal, Theme.Spacing.sm)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.sm)
                        .fill(Theme.Colors.surfaceHi)
                )
            Button("Browse…") { pickConfigPath() }
                .buttonStyle(.borderless)
            Button {
                Task { await reload() }
            } label: {
                if reloading {
                    ProgressView().controlSize(.small)
                } else {
                    Text("Reload")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(reloading)
        }
    }

    private var serverList: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            ForEach(servers, id: \.name) { status in
                serverRow(status)
            }
        }
    }

    private func serverRow(_ status: MCPServerStatus) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: Theme.Spacing.sm) {
                Circle()
                    .fill(stateColor(status.state))
                    .frame(width: 8, height: 8)
                Text(status.name)
                    .font(Theme.Typography.mono)
                    .foregroundStyle(Theme.Colors.textHigh)
                Text("(\(status.transport.rawValue))")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                Spacer()
                Text("\(status.toolsCount) tools")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
                // Per-server tool-call timeout — makes the "how long until
                // a hung tool aborts" answer legible without opening the
                // underlying `mcp.json`. Default 30s matches Python.
                Text("timeout \(Int(status.timeoutSeconds))s")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                Text(status.state.rawValue)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(stateColor(status.state))
                    .padding(.horizontal, Theme.Spacing.sm)
                    .padding(.vertical, 2)
                    .background(
                        Capsule()
                            .fill(stateColor(status.state).opacity(0.12))
                    )
                Button {
                    Task { await toggleServer(status) }
                } label: {
                    Image(systemName: status.state == .connected
                          ? "stop.circle" : "play.circle")
                        .foregroundStyle(Theme.Colors.accent)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(status.state == .connected
                    ? "Stop MCP server \(status.name)"
                    : "Start MCP server \(status.name)")
                .help(status.state == .connected ? "Stop server" : "Start server")
            }
            // Error row — only shown on `.error` state. Previously the
            // panel only showed the state-pill color-coded red with no
            // actual error text; the connection error was buried in the
            // engine logs.
            if let err = status.error, status.state == .error {
                Text(err)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.danger)
                    .lineLimit(2)
                    .truncationMode(.tail)
                    .padding(.leading, 18)  // align under server name
            }
        }
        .padding(.vertical, 4)
    }

    private var toolList: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("AVAILABLE TOOLS")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            ForEach(tools, id: \.fullName) { tool in
                HStack(alignment: .top, spacing: Theme.Spacing.sm) {
                    Text(tool.fullName)
                        .font(Theme.Typography.mono)
                        .foregroundStyle(Theme.Colors.textHigh)
                        .frame(width: 220, alignment: .leading)
                    Text(tool.description)
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textMid)
                        .lineLimit(2)
                    Spacer()
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: Theme.Spacing.sm) {
            Image(systemName: "puzzlepiece.extension")
                .font(.system(size: 28))
                .foregroundStyle(Theme.Colors.textLow)
            Text("No MCP servers configured")
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textMid)
            Text("Point at an mcp.json above and hit Reload. See the MCP spec for config format.")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, Theme.Spacing.lg)
    }

    private func errorBanner(_ msg: String) -> some View {
        HStack(alignment: .top, spacing: Theme.Spacing.sm) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Theme.Colors.danger)
            Text(msg)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
            Button {
                lastError = nil
            } label: {
                Image(systemName: "xmark")
                    .foregroundStyle(Theme.Colors.textLow)
            }
            .buttonStyle(.plain)
        }
        .padding(Theme.Spacing.sm)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.sm)
                .fill(Theme.Colors.danger.opacity(0.1))
        )
    }

    // MARK: - Actions

    private func start() async {
        // Seed the path field with the persisted session setting so the
        // user sees what's currently active.
        if let sid = app.selectedServerSessionId,
           let s = await app.engine.settings.session(sid),
           let persisted = s.mcpConfigPath
        {
            configPath = persisted
        }
        pollTask?.cancel()
        pollTask = Task { @MainActor in
            while !Task.isCancelled {
                await refreshOnce()
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    private func refreshOnce() async {
        let mcp = await app.engine.mcp
        servers = await mcp.listServers()
        tools = await mcp.listTools()
    }

    private func reload() async {
        reloading = true
        defer { reloading = false }
        lastError = nil
        let path = configPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let url: URL? = path.isEmpty ? nil : URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        do {
            try await app.engine.reloadMCPConfig(path: url)
            // Persist to SessionSettings so next relaunch picks it up.
            if let sid = app.selectedServerSessionId {
                var s = (await app.engine.settings.session(sid))
                    ?? SessionSettings(modelPath: URL(fileURLWithPath: "/"))
                s.mcpConfigPath = path.isEmpty ? nil : path
                await app.engine.settings.setSession(sid, s)
            }
            await refreshOnce()
        } catch {
            lastError = "\(error)"
        }
    }

    private func toggleServer(_ status: MCPServerStatus) async {
        lastError = nil
        let mcp = await app.engine.mcp
        do {
            switch status.state {
            case .connected:
                try await mcp.stopServer(status.name)
            default:
                try await mcp.startServer(status.name)
            }
            await refreshOnce()
        } catch {
            lastError = "\(status.name): \(error)"
        }
    }

    private func pickConfigPath() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = []
        panel.allowsOtherFileTypes = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.message = "Pick an mcp.json config"
        if panel.runModal() == .OK, let url = panel.url {
            configPath = url.path
        }
    }

    private func stateColor(_ state: MCPServerState) -> Color {
        switch state {
        case .connected:    return Theme.Colors.success
        case .connecting:   return Theme.Colors.warning
        case .disconnected: return Theme.Colors.textLow
        case .error:        return Theme.Colors.danger
        }
    }
}
