import SwiftUI
import vMLXEngine
import vMLXTheme

/// Thin Server screen shell. Top half = multi-session dashboard; bottom half =
/// the selected session's detail view with its config form, logs, performance,
/// cache, and benchmark tabs. Both halves share `AppState.sessions` and
/// `AppState.selectedServerSessionId`.
struct ServerScreen: View {
    @Environment(AppState.self) private var app

    var body: some View {
        @Bindable var s = app
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.xl) {
                SessionDashboard(selection: Binding(
                    get: { s.selectedServerSessionId ?? s.sessions.first?.id },
                    set: { s.selectedServerSessionId = $0 }
                ))

                if let selected = currentSelection {
                    SessionView(session: selected)
                } else {
                    // No session yet — surface the Model Directories panel
                    // directly so first-launch users can configure where to
                    // scan for models BEFORE they hit the empty model picker.
                    ModelDirectoriesPanel()
                }
            }
            .padding(Theme.Spacing.xl)
        }
        .background(Theme.Colors.background)
        .onAppear {
            // Keep the global engineState mirrored into the selected session's
            // state so SessionCard / SessionView paint correctly. The rest of
            // the fields (pid, latency, load progress) come from AppState.
            syncSelected()
        }
        .onChange(of: app.engineState) { _, _ in syncSelected() }
        .onChange(of: app.loadProgress) { _, _ in syncSelected() }
    }

    private var currentSelection: Session? {
        let id = app.selectedServerSessionId ?? app.sessions.first?.id
        return app.sessions.first(where: { $0.id == id })
    }

    /// Mirror AppState.engineState + loadProgress into the selected session's
    /// copy so the single-session Phase-3 world stays in sync. When
    /// multi-engine lands each Session will feed off its own Engine actor
    /// instead of the shared one.
    private func syncSelected() {
        guard let id = app.selectedServerSessionId ?? app.sessions.first?.id else { return }
        guard let idx = app.sessions.firstIndex(where: { $0.id == id }) else { return }
        app.sessions[idx].state = app.engineState
        app.sessions[idx].loadProgress = app.loadProgress
    }
}
