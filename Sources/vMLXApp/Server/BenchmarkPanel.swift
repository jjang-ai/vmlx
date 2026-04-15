import SwiftUI
import SQLite3
import vMLXEngine
import vMLXTheme

/// Benchmark panel shown in the SessionView bottom tabs. Offers three preset
/// benchmark buttons, runs the suite through `Engine.benchmark(suite:)`,
/// shows a live progress bar + final report, and persists results to the
/// `bench_results` table in `vmlx.sqlite3` keyed by (model_id, suite, date).
///
/// Electron parity: `panel/src/renderer/src/components/sessions/BenchmarkPanel.tsx`.
/// Phase 3 scope: works against the stub `Engine.benchmark` which emits
/// `.failed` immediately. When the real generation loop lands, all four
/// display paths (progress, live label, final numbers, history) will light up
/// without a second round of UI work.
struct BenchmarkPanel: View {
    @Environment(AppState.self) private var app
    let sessionId: UUID
    let modelId: String

    @State private var running: Engine.BenchSuite? = nil
    @State private var progress: Double = 0
    @State private var progressLabel: String = ""
    @State private var lastReport: Engine.BenchReport? = nil
    @State private var errorMessage: String? = nil
    @State private var history: [BenchRow] = []
    @State private var runTask: Task<Void, Never>? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            Text("BENCHMARK")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)

            presetRow
            if running != nil {
                progressBlock
            }
            if let report = lastReport {
                reportBlock(report)
            }
            if let err = errorMessage {
                errorBlock(err)
            }
            historyList
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
        .onAppear { reloadHistory() }
    }

    // MARK: - Presets

    private var presetRow: some View {
        HStack(spacing: Theme.Spacing.sm) {
            ForEach(Engine.BenchSuite.allCases, id: \.self) { suite in
                Button {
                    startRun(suite)
                } label: {
                    Text(suite.displayName)
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textHigh)
                        .padding(.horizontal, Theme.Spacing.md)
                        .padding(.vertical, Theme.Spacing.sm)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.Radius.sm)
                                .fill(running == suite ? Theme.Colors.accent : Theme.Colors.surfaceHi)
                        )
                }
                .buttonStyle(.plain)
                .disabled(running != nil)
            }
            Spacer()
        }
    }

    // MARK: - Progress

    private var progressBlock: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            LoadingBar(fraction: progress)
            Text(progressLabel)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
        }
    }

    // MARK: - Report

    @ViewBuilder
    private func reportBlock(_ r: Engine.BenchReport) -> some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            Text("Last run — \(r.suite.displayName)")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            HStack(spacing: Theme.Spacing.lg) {
                stat("tok/s", String(format: "%.1f", r.tokensPerSec), delta: delta(for: r, field: .tokensPerSec))
                stat("TTFT",  String(format: "%.0f ms", r.ttftMs),    delta: delta(for: r, field: .ttft))
                stat("total", String(format: "%.0f ms", r.totalMs),   delta: nil)
                stat("hit",   String(format: "%.0f%%", r.cacheHitRate * 100), delta: nil)
            }
        }
        .padding(Theme.Spacing.sm)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.sm)
                .fill(Theme.Colors.surfaceHi)
        )
    }

    @ViewBuilder
    private func stat(_ label: String, _ value: String, delta: String?) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(label)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            Text(value)
                .font(Theme.Typography.mono)
                .foregroundStyle(Theme.Colors.textHigh)
            if let d = delta {
                Text(d)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textMid)
            }
        }
    }

    private enum DeltaField { case tokensPerSec, ttft }
    private func delta(for r: Engine.BenchReport, field: DeltaField) -> String? {
        guard let prev = history.first(where: {
            $0.suite == r.suite && $0.modelId == r.modelId && $0.date < r.date
        }) else { return nil }
        switch field {
        case .tokensPerSec:
            let d = r.tokensPerSec - prev.tokensPerSec
            return String(format: "%+.1f vs last", d)
        case .ttft:
            let d = r.ttftMs - prev.ttftMs
            return String(format: "%+.0fms vs last", d)
        }
    }

    private func errorBlock(_ err: String) -> some View {
        HStack(spacing: Theme.Spacing.sm) {
            Image(systemName: "exclamationmark.triangle")
                .foregroundStyle(Theme.Colors.warning)
            Text(err)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textMid)
            Spacer()
        }
        .padding(Theme.Spacing.sm)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.sm)
                .fill(Theme.Colors.surfaceHi)
        )
    }

    // MARK: - History

    private var historyList: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            Text("HISTORY (last 20)")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            if history.isEmpty {
                Text("No benchmark runs yet.")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(history) { row in
                            HStack(spacing: Theme.Spacing.md) {
                                Text(Self.formatDate(row.date))
                                    .foregroundStyle(Theme.Colors.textLow)
                                    .frame(width: 130, alignment: .leading)
                                Text(row.suite.displayName)
                                    .foregroundStyle(Theme.Colors.textMid)
                                    .frame(width: 180, alignment: .leading)
                                Text(String(format: "%.1f tok/s", row.tokensPerSec))
                                    .foregroundStyle(Theme.Colors.textHigh)
                                Spacer()
                                Text(row.modelId)
                                    .foregroundStyle(Theme.Colors.textLow)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                            .font(Theme.Typography.mono)
                        }
                    }
                    .padding(Theme.Spacing.sm)
                }
                .frame(maxHeight: 180)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.sm)
                        .fill(Theme.Colors.background)
                )
            }
        }
    }

    // MARK: - Run

    private func startRun(_ suite: Engine.BenchSuite) {
        running = suite
        progress = 0
        progressLabel = "Queued"
        errorMessage = nil
        let engine = app.engine
        let currentModel = modelId
        runTask?.cancel()
        runTask = Task { @MainActor in
            do {
                for try await event in await engine.benchmark(suite: suite) {
                    switch event {
                    case .progress(let f, let label):
                        progress = f
                        progressLabel = label
                    case .done(let report):
                        lastReport = report
                        BenchmarkStore.shared.insert(report)
                        reloadHistory()
                        running = nil
                    case .failed(let msg):
                        // Still log a placeholder row so the history list
                        // grows visibly — gives Eric feedback that the button
                        // click was honored while the engine port is pending.
                        let placeholder = Engine.BenchReport(
                            suite: suite, modelId: currentModel,
                            tokensPerSec: 0, ttftMs: 0, totalMs: 0,
                            cacheHitRate: 0, notes: msg
                        )
                        BenchmarkStore.shared.insert(placeholder)
                        reloadHistory()
                        errorMessage = msg
                        running = nil
                    }
                }
            } catch {
                self.errorMessage = "\(error)"
                running = nil
            }
        }
    }

    private func reloadHistory() {
        history = BenchmarkStore.shared.recent(limit: 20)
    }

    private static let fmt: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return f
    }()
    private static func formatDate(_ d: Date) -> String { fmt.string(from: d) }
}

// MARK: - BenchmarkStore (SQLite)

/// Thin wrapper on top of `vmlx.sqlite3` that stores benchmark results keyed
/// by (model_id, suite, date). Main-actor-only because the shared DB handle
/// is main-actor-only.
@MainActor
final class BenchmarkStore {
    static let shared = BenchmarkStore()
    private var db: OpaquePointer? = nil

    private init() { openDB(); migrate() }

    private func openDB() {
        let fm = FileManager.default
        let appSup = try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                 appropriateFor: nil, create: true)
        let dir = (appSup ?? URL(fileURLWithPath: NSTemporaryDirectory()))
            .appendingPathComponent("vMLX", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let path = dir.appendingPathComponent("vmlx.sqlite3").path
        _ = sqlite3_open(path, &db)
        _ = sqlite3_exec(db, "PRAGMA journal_mode=WAL;", nil, nil, nil)
    }

    private func migrate() {
        let sql = """
        CREATE TABLE IF NOT EXISTS bench_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            suite TEXT NOT NULL,
            date REAL NOT NULL,
            tokens_per_sec REAL NOT NULL,
            ttft_ms REAL NOT NULL,
            total_ms REAL NOT NULL,
            cache_hit_rate REAL NOT NULL,
            notes TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_bench_date ON bench_results(date DESC);
        """
        _ = sqlite3_exec(db, sql, nil, nil, nil)
    }

    func insert(_ r: Engine.BenchReport) {
        let sql = """
        INSERT INTO bench_results (model_id, suite, date, tokens_per_sec, ttft_ms, total_ms, cache_hit_rate, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
        sqlite3_bind_text(stmt, 1, r.modelId, -1, SQLITE_TRANSIENT_BENCH)
        sqlite3_bind_text(stmt, 2, r.suite.rawValue, -1, SQLITE_TRANSIENT_BENCH)
        sqlite3_bind_double(stmt, 3, r.date.timeIntervalSince1970)
        sqlite3_bind_double(stmt, 4, r.tokensPerSec)
        sqlite3_bind_double(stmt, 5, r.ttftMs)
        sqlite3_bind_double(stmt, 6, r.totalMs)
        sqlite3_bind_double(stmt, 7, r.cacheHitRate)
        sqlite3_bind_text(stmt, 8, r.notes, -1, SQLITE_TRANSIENT_BENCH)
        sqlite3_step(stmt)
        sqlite3_finalize(stmt)
    }

    func recent(limit: Int = 20) -> [BenchRow] {
        let sql = """
        SELECT model_id, suite, date, tokens_per_sec, ttft_ms, total_ms, cache_hit_rate, IFNULL(notes, '')
        FROM bench_results ORDER BY date DESC LIMIT ?;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return [] }
        sqlite3_bind_int(stmt, 1, Int32(limit))
        var out: [BenchRow] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let modelId = String(cString: sqlite3_column_text(stmt, 0))
            let suiteS  = String(cString: sqlite3_column_text(stmt, 1))
            let date    = Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2))
            let tps     = sqlite3_column_double(stmt, 3)
            let ttft    = sqlite3_column_double(stmt, 4)
            let total   = sqlite3_column_double(stmt, 5)
            let hit     = sqlite3_column_double(stmt, 6)
            let notes   = String(cString: sqlite3_column_text(stmt, 7))
            guard let suite = Engine.BenchSuite(rawValue: suiteS) else { continue }
            out.append(BenchRow(
                id: UUID(), modelId: modelId, suite: suite, date: date,
                tokensPerSec: tps, ttftMs: ttft, totalMs: total,
                cacheHitRate: hit, notes: notes
            ))
        }
        sqlite3_finalize(stmt)
        return out
    }
}

struct BenchRow: Identifiable, Equatable {
    let id: UUID
    let modelId: String
    let suite: Engine.BenchSuite
    let date: Date
    let tokensPerSec: Double
    let ttftMs: Double
    let totalMs: Double
    let cacheHitRate: Double
    let notes: String
}

private let SQLITE_TRANSIENT_BENCH = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
