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
    // CRUD sheet state: `editorDraft` is non-nil iff the add/edit sheet
    // is on screen. A fresh draft (from "Add server") has an empty
    // name; an edit draft is pre-populated from the saved config.
    @State private var editorDraft: MCPServerDraft? = nil
    @State private var confirmDelete: String? = nil

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
        .sheet(item: $editorDraft) { draft in
            MCPServerEditor(
                draft: draft,
                existingNames: Set(servers.map { $0.name }),
                onSave: { saved in
                    editorDraft = nil
                    Task { await upsert(saved) }
                },
                onCancel: { editorDraft = nil }
            )
            .frame(minWidth: 540, minHeight: 420)
        }
        .alert("Remove MCP server?",
               isPresented: Binding(
                get: { confirmDelete != nil },
                set: { if !$0 { confirmDelete = nil } })
        ) {
            Button("Cancel", role: .cancel) { confirmDelete = nil }
            Button("Remove", role: .destructive) {
                if let name = confirmDelete {
                    confirmDelete = nil
                    Task { await remove(name) }
                }
            }
        } message: {
            Text("\"\(confirmDelete ?? "")\" will be removed from mcp.json and stopped if running. This action is not undoable from the app.")
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
            Button {
                editorDraft = MCPServerDraft.newDraft()
            } label: {
                Label("Add server", systemImage: "plus")
            }
            .buttonStyle(.bordered)
            .help("Add a new MCP server to mcp.json")
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
            // §340 — one-click import from Claude Desktop / Cursor /
            // Windsurf / Zed clipboard format. Reads the current
            // system clipboard, parses the `mcpServers` block, and
            // upserts each entry via the same path as manual add.
            // Much faster than "Browse" when the user already has
            // a working config somewhere.
            Button("Paste JSON") {
                Task { await pasteJSONImport() }
            }
            .buttonStyle(.borderless)
            .help("Paste a Claude-Desktop-style mcp.json block from the clipboard and import every server.")
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
                Button {
                    Task { await beginEdit(status.name) }
                } label: {
                    Image(systemName: "pencil")
                        .foregroundStyle(Theme.Colors.textMid)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Edit MCP server \(status.name)")
                .help("Edit configuration")
                Button {
                    confirmDelete = status.name
                } label: {
                    Image(systemName: "trash")
                        .foregroundStyle(Theme.Colors.danger)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Remove MCP server \(status.name)")
                .help("Remove from mcp.json")
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

    // Resolve the config path that `upsertMCPServer` / `removeMCPServer`
    // should write to. Prefer the explicit path the user typed/picked;
    // fall back to nil so the engine lands on the default search path.
    private func resolvedConfigURL() -> URL? {
        let p = configPath.trimmingCharacters(in: .whitespacesAndNewlines)
        return p.isEmpty ? nil : URL(fileURLWithPath: (p as NSString).expandingTildeInPath)
    }

    /// Pull the current config off the engine so we can pre-fill the
    /// editor with everything the user already saved (command, args,
    /// env, url, enabled, timeout). Falls back to a minimal stdio draft
    /// when the manager doesn't know the name — shouldn't happen in
    /// practice because the row is rendered from `listServers()`.
    private func beginEdit(_ name: String) async {
        let cfg = await app.engine.mcp.currentConfig()
        if let existing = cfg.servers[name] {
            editorDraft = MCPServerDraft(existing: existing)
        } else {
            editorDraft = MCPServerDraft.newDraft(name: name)
        }
    }

    private func upsert(_ server: MCPServerConfig) async {
        lastError = nil
        do {
            try await app.engine.upsertMCPServer(server, configPath: resolvedConfigURL())
            await refreshOnce()
        } catch {
            lastError = "Save failed: \(error)"
        }
    }

    /// §340 — one-click import from the clipboard. Reads the current
    /// pasteboard, runs it through `MCPClipboardImport.parse`, and
    /// upserts every valid server. Surfaces a banner with
    /// imported/skipped counts so users know what landed.
    private func pasteJSONImport() async {
        lastError = nil
        #if canImport(AppKit)
        guard let raw = NSPasteboard.general.string(forType: .string),
              !raw.isEmpty
        else {
            lastError = "Clipboard is empty — copy your mcp.json block first."
            return
        }
        do {
            let result = try MCPClipboardImport.parse(json: raw)
            guard result.hasAnyImported else {
                let skips = result.skipped.map { "\($0.name): \($0.reason)" }
                    .joined(separator: "; ")
                lastError = "No importable servers. \(skips.isEmpty ? "" : "Skipped — \(skips)")"
                return
            }
            var failed: [String] = []
            for cfg in result.servers {
                do {
                    try await app.engine.upsertMCPServer(
                        cfg, configPath: resolvedConfigURL())
                } catch {
                    failed.append("\(cfg.name): \(error)")
                }
            }
            await refreshOnce()
            var summary = "Imported \(result.servers.count - failed.count) of \(result.totalParsed) server(s)."
            if !result.skipped.isEmpty {
                let skips = result.skipped.map { $0.name }.joined(separator: ", ")
                summary += " Skipped shape-invalid: \(skips)."
            }
            if !failed.isEmpty {
                summary += " Save errors: \(failed.joined(separator: "; "))."
            }
            lastError = summary  // reused as info banner — the red
                                 // styling is acceptable since we
                                 // still surface both successes and
                                 // failures in one line.
        } catch {
            lastError = "Import failed: \(error)"
        }
        #else
        lastError = "Clipboard import requires macOS AppKit."
        #endif
    }

    private func remove(_ name: String) async {
        lastError = nil
        do {
            try await app.engine.removeMCPServer(name, configPath: resolvedConfigURL())
            await refreshOnce()
        } catch {
            lastError = "Remove failed: \(error)"
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

// MARK: - Editor draft + sheet

/// Mutable, text-friendly mirror of `MCPServerConfig` that the form
/// binds to. Array/dict fields (`args`, `env`) are edited as newline-
/// separated text and split on save — a full KV editor was overkill
/// for the 1–8 entries a typical server needs.
///
/// `id` matches the server name when editing an existing row; fresh
/// drafts start with a UUID so the `.sheet(item:)` presentation key
/// doesn't collide with a stale name.
struct MCPServerDraft: Identifiable, Equatable {
    var id: String
    var isNew: Bool
    var name: String
    var transport: MCPTransport
    var command: String
    var argsText: String   // newline-separated
    var envText: String    // newline-separated KEY=VALUE
    var url: String
    var enabled: Bool
    var timeoutSeconds: Double
    var skipSecurityValidation: Bool

    static func newDraft(name: String = "") -> MCPServerDraft {
        MCPServerDraft(
            id: UUID().uuidString,
            isNew: true,
            name: name,
            transport: .stdio,
            command: "",
            argsText: "",
            envText: "",
            url: "",
            enabled: true,
            timeoutSeconds: 30,
            skipSecurityValidation: false
        )
    }

    init(existing: MCPServerConfig) {
        self.id = existing.name
        self.isNew = false
        self.name = existing.name
        self.transport = existing.transport
        self.command = existing.command ?? ""
        self.argsText = (existing.args ?? []).joined(separator: "\n")
        self.envText = (existing.env ?? [:])
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: "\n")
        self.url = existing.url ?? ""
        self.enabled = existing.enabled
        self.timeoutSeconds = existing.timeout
        self.skipSecurityValidation = existing.skipSecurityValidation
    }

    init(
        id: String, isNew: Bool, name: String, transport: MCPTransport,
        command: String, argsText: String, envText: String, url: String,
        enabled: Bool, timeoutSeconds: Double, skipSecurityValidation: Bool
    ) {
        self.id = id
        self.isNew = isNew
        self.name = name
        self.transport = transport
        self.command = command
        self.argsText = argsText
        self.envText = envText
        self.url = url
        self.enabled = enabled
        self.timeoutSeconds = timeoutSeconds
        self.skipSecurityValidation = skipSecurityValidation
    }

    /// Build an `MCPServerConfig` from the draft, validating required
    /// fields per transport. Returns `(config, nil)` on success or
    /// `(nil, "reason")` on failure — surfaced inline in the sheet.
    func build() -> (MCPServerConfig?, String?) {
        let trimmedName = name.trimmingCharacters(in: .whitespaces)
        if trimmedName.isEmpty { return (nil, "Name is required") }
        let args = argsText
            .split(whereSeparator: { $0.isNewline })
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        var env: [String: String] = [:]
        for line in envText.split(whereSeparator: { $0.isNewline }) {
            let raw = String(line).trimmingCharacters(in: .whitespaces)
            if raw.isEmpty { continue }
            guard let eq = raw.firstIndex(of: "=") else {
                return (nil, "Env line missing '=': \(raw)")
            }
            let key = String(raw[..<eq]).trimmingCharacters(in: .whitespaces)
            let value = String(raw[raw.index(after: eq)...])
            if key.isEmpty { return (nil, "Env key empty in line: \(raw)") }
            env[key] = value
        }
        let cfg = MCPServerConfig(
            name: trimmedName,
            transport: transport,
            command: transport == .stdio ? command.trimmingCharacters(in: .whitespaces) : nil,
            args: args.isEmpty ? nil : args,
            env: env.isEmpty ? nil : env,
            url: transport == .sse ? url.trimmingCharacters(in: .whitespaces) : nil,
            enabled: enabled,
            timeout: max(1, timeoutSeconds),
            skipSecurityValidation: skipSecurityValidation
        )
        if let reason = cfg.validateShape() { return (nil, reason) }
        return (cfg, nil)
    }
}

struct MCPServerEditor: View {
    @State var draft: MCPServerDraft
    let existingNames: Set<String>
    let onSave: (MCPServerConfig) -> Void
    let onCancel: () -> Void
    @State private var error: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            Text(draft.isNew ? "Add MCP server" : "Edit \(draft.name)")
                .font(Theme.Typography.bodyHi)
                .foregroundStyle(Theme.Colors.textHigh)

            Form {
                Section {
                    TextField("Name", text: $draft.name)
                        .disabled(!draft.isNew)  // renaming changes the dict key
                        .help(draft.isNew
                            ? "Unique identifier — also the prefix on every tool name (server__tool)"
                            : "Renaming requires remove+add — use a separate step")
                    Picker("Transport", selection: $draft.transport) {
                        Text("stdio (subprocess)").tag(MCPTransport.stdio)
                        Text("sse (remote URL)").tag(MCPTransport.sse)
                    }
                    .pickerStyle(.segmented)
                }
                if draft.transport == .stdio {
                    Section("stdio") {
                        TextField("Command", text: $draft.command)
                            .help("Absolute path or PATH-resolvable executable")
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Args (one per line)")
                                .font(Theme.Typography.caption)
                                .foregroundStyle(Theme.Colors.textLow)
                            TextEditor(text: $draft.argsText)
                                .frame(minHeight: 60, maxHeight: 120)
                                .font(Theme.Typography.mono)
                        }
                    }
                } else {
                    Section("sse") {
                        TextField("URL", text: $draft.url)
                            .help("Full https:// URL of the MCP SSE endpoint")
                    }
                }
                Section("Environment") {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("KEY=VALUE per line")
                            .font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Colors.textLow)
                        TextEditor(text: $draft.envText)
                            .frame(minHeight: 60, maxHeight: 120)
                            .font(Theme.Typography.mono)
                    }
                }
                Section("Runtime") {
                    Toggle("Enabled", isOn: $draft.enabled)
                    HStack {
                        Text("Timeout")
                        Slider(value: $draft.timeoutSeconds, in: 1...300, step: 1) {
                            Text("Timeout seconds")
                        }
                        Text("\(Int(draft.timeoutSeconds))s")
                            .monospacedDigit()
                            .frame(width: 44, alignment: .trailing)
                    }
                    Toggle("Skip security validation (dev only)",
                           isOn: $draft.skipSecurityValidation)
                }
            }

            if let err = error {
                Text(err)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.danger)
            }

            HStack {
                Spacer()
                Button("Cancel", action: onCancel)
                    .keyboardShortcut(.cancelAction)
                Button(draft.isNew ? "Add" : "Save") {
                    if draft.isNew && existingNames.contains(
                        draft.name.trimmingCharacters(in: .whitespaces)
                    ) {
                        error = "A server named '\(draft.name)' already exists."
                        return
                    }
                    let (cfg, reason) = draft.build()
                    if let reason {
                        error = reason
                    } else if let cfg {
                        onSave(cfg)
                    }
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(Theme.Spacing.lg)
    }
}
