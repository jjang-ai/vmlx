import SwiftUI
import vMLXEngine
import vMLXTheme

/// Thin Server screen shell. Top half = multi-session dashboard; bottom half =
/// the selected session's detail view with its config form, logs, performance,
/// cache, and benchmark tabs. Both halves share `AppState.sessions` and
/// `AppState.selectedServerSessionId`.
struct ServerScreen: View {
    @Environment(AppState.self) private var app
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        @Bindable var s = app
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.xl) {
                SessionDashboard(selection: Binding(
                    get: { s.selectedServerSessionId ?? s.sessions.first?.id },
                    set: { selectSession($0) }
                ))

                if let selected = currentSelection {
                    SessionView(session: selected)
                } else {
                    // S2 §308: first-run users see a hero "Load your first
                    // model" CTA rather than just the directories panel,
                    // which buries the action one step. The directories
                    // panel still renders below so users who need custom
                    // scan paths can set them inline without leaving the
                    // screen.
                    EmptyStateView(
                        systemImage: "square.stack.3d.up",
                        title: "Load your first model",
                        caption: "Pick a local model to spin up a session. vMLX will start a HTTP listener, load weights, and be ready for chat + API.",
                        cta: ("Open Downloads", {
                            openWindow(id: "downloads")
                        })
                    )
                    .frame(minHeight: 260)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.lg)
                            .fill(Theme.Colors.surface)
                            .overlay(
                                RoundedRectangle(cornerRadius: Theme.Radius.lg)
                                    .stroke(Theme.Colors.border, lineWidth: 1)
                            )
                    )
                    ModelDirectoriesPanel()
                }
            }
            .padding(Theme.Spacing.xl)
        }
        .background(Theme.Colors.background)
        .onAppear {
            if app.selectedServerSessionId == nil {
                selectSession(app.sessions.first?.id)
            }
        }
    }

    private var currentSelection: Session? {
        let id = app.selectedServerSessionId ?? app.sessions.first?.id
        return app.sessions.first(where: { $0.id == id })
    }

    /// Selection is the only thing ServerScreen should mutate. Session
    /// state/load progress are owned by AppState.observePerSessionEngine;
    /// mirroring the global `engineState` back into the selected row reverts
    /// the app to the old single-engine model and can overwrite a just-created
    /// or loading row with stale state.
    private func selectSession(_ id: UUID?) {
        guard app.selectedServerSessionId != id else { return }
        app.selectedServerSessionId = id
        if let id {
            app.engineState = app.engine(for: id).state
            app.loadProgress = app.sessions.first(where: { $0.id == id })?.loadProgress
        } else {
            app.engineState = .stopped
            app.loadProgress = nil
        }
        app.rebindEngineObserver()
    }
}
