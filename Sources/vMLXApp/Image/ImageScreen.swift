// SPDX-License-Identifier: Apache-2.0
//
// ImageScreen — full-parity Image tab for the Swift rewrite.
//
// IMAGE CHECKLIST VERIFICATION (feedback_image_checklist.md)
// ==========================================================
// Every item below is either (a) honored by this file / its siblings in
// Sources/vMLXApp/Image/ or (b) explicitly deferred with a reason. Checked
// on every image code change per Eric's "EVERY time" rule.
//
// SERVER TAB — Pre-startup (CreateSession)
//   [✓] Image Gen model — simplified config ....... ServerScreen (owned there;
//                                                    imageMode=generate passed
//                                                    via SessionSettings)
//   [✓] Image Edit model — simplified config ....... same, imageMode=edit
//   [✓] Text model — full config .................. unchanged
//   [✓] Auto-detection .............................. ModelLibrary.modality
//
// SERVER TAB — After model loaded (SessionView)
//   [~] Image Gen "Open Image Generator" button ..... Server screen delegates
//                                                     to AppState.mode=.image
//   [~] Image Edit "Open Image Editor" button ....... same path
//   [✓] Chat/cache/bench/embed/perf buttons hidden .. Server screen concerns
//   [~] Sidebar "Open Image Tab" for image .......... Sidebar already routes
//                                                     via AppState.mode
//   [✓] Logs always accessible ...................... LogsPanel unchanged
//   [✓] Stop/Cancel always accessible ............... ImageTopBar Stop button
//
// IMAGE TAB (this file + siblings)
//   [✓] Model picker: Gen and Edit SEPARATE ......... ImageModelPicker.swift
//   [✓] Dropdown shows download status per model .... ImageModelPicker green dot
//   [✓] Edit: source upload + strength + Edit button  ImagePromptBar.swift
//   [✓] Gen: no upload, Generate button ............. ImagePromptBar branches
//   [✓] Gallery grid ................................ ImageGallery LazyVGrid
//                                                     adaptive minimum 220
//   [✓] History: Gen/Edit badges on ALL sessions .... ImageHistory.swift
//   [✓] Logs work before server starts .............. LogsPanel independent
//   [✓] Cancel during generation/editing ............ Top bar + gen state view
//
// API PAGE
//   [✓] Gen server: /v1/images/generations + snippets APIScreen owns this
//   [✓] Edit server: /v1/images/edits + snippets .... same
//   [✓] Correct model name / base URL / API key ..... same
//
// CHAT TAB
//   [✓] Image sessions FILTERED OUT of model picker . ChatViewModel filters by
//                                                     modality; not modified
//                                                     here
//
// DOWNLOADS
//   [✓] Popup opens on ANY download start ........... DownloadManager auto-open
//   [✓] "View Downloads" button in picker ........... ImageModelPicker row
//                                                     Download button
//   [✓] Models show availability per DB lookup ...... ImageModelPicker entryFor
//   [✓] HF auth token properly used ................. DownloadManager
//
// Redo buttons always visible (MEMORY note) .......... ImageGallery ImageCard
// NO regex for model detection ....................... explicit ImageScreen.Tab
//                                                     + ImageCatalogModel.kind
// Download popup always visible ...................... DownloadManager path

import SwiftUI
import vMLXEngine
import vMLXTheme

struct ImageScreen: View {
    @Environment(AppState.self) private var appState

    enum Tab: String, CaseIterable, Identifiable {
        case generate = "Generate"
        case edit = "Edit"
        var id: String { rawValue }
    }

    enum Status: Equatable {
        case idle
        case generating
        case editing
        case error(String)

        var isActive: Bool {
            switch self { case .generating, .editing: return true; default: return false }
        }
        var label: String {
            switch self {
            case .idle:       return "Idle"
            case .generating: return "Generating"
            case .editing:    return "Editing"
            case .error:      return "Error"
            }
        }
        var dotColor: Color {
            switch self {
            case .idle:       return Theme.Colors.textLow
            case .generating: return Theme.Colors.accent
            case .editing:    return Theme.Colors.accent
            case .error:      return Theme.Colors.danger
            }
        }
    }

    // MARK: - View state
    @State private var tab: Tab = .generate
    @State private var selected: ImageCatalogModel? = nil
    @State private var prompt: String = ""
    @State private var sourceImage: Data? = nil
    @State private var maskImage: Data? = nil
    @State private var images: [GeneratedImage] = []
    @State private var history: [ImageGenerationRecord] = []
    @State private var settings = ImageGenSettings()
    @State private var status: Status = .idle
    @State private var elapsed: Int = 0
    @State private var currentStep: Int = 0
    @State private var errorBanner: ImageErrorBanner? = nil
    @State private var showSettings = false
    @State private var tickerTask: Task<Void, Never>? = nil
    @State private var jobTask: Task<Void, Never>? = nil

    private let historyStore = ImageHistoryStore.shared

    var body: some View {
        HStack(spacing: 0) {
            ImageHistory(
                records: history,
                onRecall: { r in recall(r) },
                onDelete: { r in
                    _ = historyStore.delete(r.id)
                    history.removeAll { $0.id == r.id }
                }
            )
            Divider().background(Theme.Colors.border)

            VStack(spacing: 0) {
                ImageTopBar(
                    selectedModel: selected,
                    status: status,
                    elapsedSeconds: elapsed,
                    currentStep: currentStep,
                    totalSteps: settings.steps,
                    onStop: stop,
                    onOpenSettings: { showSettings.toggle() }
                )
                Divider().background(Theme.Colors.border)

                HStack(alignment: .top, spacing: Theme.Spacing.lg) {
                    ImageModelPicker(selected: $selected, mode: $tab)
                        .frame(width: 320)
                        .padding(Theme.Spacing.lg)

                    ImageGallery(
                        images: images,
                        isGenerating: status.isActive,
                        currentStep: currentStep,
                        totalSteps: settings.steps,
                        elapsedSeconds: elapsed,
                        preview: nil,
                        errorBanner: errorBanner,
                        onRedo: { img in recallFromGenerated(img) },
                        onDelete: { img in images.removeAll { $0.id == img.id } },
                        onStop: stop,
                        onDismissError: { errorBanner = nil }
                    )
                }

                ImagePromptBar(
                    prompt: $prompt,
                    sourceImage: $sourceImage,
                    maskImage: $maskImage,
                    strength: Binding(
                        get: { settings.strength },
                        set: { settings.strength = $0 }
                    ),
                    mode: tab,
                    canSubmit: canSubmit,
                    onSubmit: submit,
                    onDownloadNeeded: nil
                )
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(Theme.Colors.background)
        .popover(isPresented: $showSettings) {
            ImageSettingsDrawer(
                settings: $settings,
                mode: tab,
                onClose: { showSettings = false },
                onPersist: { s in Task { await persistDefaults(s) } }
            )
        }
        .task { await initialLoad() }
    }

    // MARK: - Computed

    private var canSubmit: Bool {
        guard selected != nil else { return false }
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return false }
        if tab == .edit && sourceImage == nil { return false }
        if status.isActive { return false }
        return true
    }

    // MARK: - Submit / stop

    private func submit() {
        guard canSubmit, let model = selected else { return }
        errorBanner = nil
        status = (tab == .edit) ? .editing : .generating
        currentStep = 0
        elapsed = 0
        let jobId = UUID()

        let settingsJSON: String = {
            guard let data = try? JSONEncoder().encode(settings) else { return "{}" }
            return String(data: data, encoding: .utf8) ?? "{}"
        }()
        let record = ImageGenerationRecord(
            id: jobId,
            modelAlias: model.displayName,
            prompt: prompt,
            sourceImagePath: sourceImage != nil ? "<inline>" : nil,
            maskPath: maskImage != nil ? "<inline>" : nil,
            settingsJSON: settingsJSON,
            outputPath: nil,
            status: .pending
        )
        _ = historyStore.upsert(record)
        history.insert(record, at: 0)

        startTicker()

        let currentTab = tab
        let currentPrompt = prompt
        let currentSource = sourceImage
        let currentMask = maskImage
        let currentStrength = settings.strength
        let currentSettings = settings
        let currentDisplay = model.displayName

        jobTask = Task {
            do {
                let url: URL
                if currentTab == .edit, let src = currentSource {
                    url = try await appState.engine.editImage(
                        prompt: currentPrompt,
                        model: currentDisplay,
                        source: src,
                        mask: currentMask,
                        strength: currentStrength,
                        settings: currentSettings
                    )
                } else {
                    url = try await appState.engine.generateImage(
                        prompt: currentPrompt,
                        model: currentDisplay,
                        settings: currentSettings
                    )
                }
                await MainActor.run {
                    if let data = try? Data(contentsOf: url) {
                        images.insert(
                            GeneratedImage(data: data, prompt: currentPrompt, createdAt: .now),
                            at: 0
                        )
                    }
                    finish(jobId: jobId, outputPath: url.path, status: .completed)
                }
            } catch {
                await MainActor.run {
                    let raw = String(describing: error)
                    let hfAuth = raw.contains("401") || raw.contains("403")

                    // Friendly rewrite for the "image gen backend is still
                    // being ported" case. Every Flux concrete model
                    // currently throws `notImplemented` from its
                    // `.generate()` method — surface a clearer message
                    // pointing at the architecture doc rather than the
                    // raw stack trace.
                    let text: String
                    if raw.contains("FluxBackend") && raw.contains("not implemented") {
                        // §358 — honest scaffold status. Previous message
                        // pointed users at a README that doesn't exist.
                        // Name the ONE working model and admit the rest
                        // aren't ported yet.
                        text = """
                        Image generation for this model isn't ported yet. \
                        The only runnable image model on vMLX Swift today \
                        is Z-Image Turbo — pick it in the Model picker and \
                        try again. Flux Schnell / Flux Dev / other pipelines \
                        are scaffolded (scheduler, VAE, DiT, loaders in \
                        place) but their sampler loops are not wired. \
                        Track ongoing porting at docs/audit/OPEN-FIX-LIST.md.
                        """
                    } else {
                        text = raw
                    }
                    errorBanner = ImageErrorBanner(message: text, hfAuth: hfAuth)
                    finish(jobId: jobId, outputPath: nil, status: .failed)
                }
            }
        }
    }

    private func stop() {
        jobTask?.cancel()
        jobTask = nil
        tickerTask?.cancel()
        tickerTask = nil
        status = .idle
    }

    private func finish(jobId: UUID, outputPath: String?, status endStatus: ImageGenerationRecord.Status) {
        tickerTask?.cancel()
        tickerTask = nil
        status = .idle
        if let idx = history.firstIndex(where: { $0.id == jobId }) {
            history[idx].outputPath = outputPath
            history[idx].durationMs = elapsed * 1000
            history[idx].status = endStatus
            _ = historyStore.upsert(history[idx])
        }
    }

    private func startTicker() {
        tickerTask?.cancel()
        tickerTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                await MainActor.run {
                    if status.isActive {
                        elapsed += 1
                        if currentStep < settings.steps - 1 {
                            currentStep += 1
                        }
                    }
                }
            }
        }
    }

    // MARK: - History / recall

    private func initialLoad() async {
        let loaded = historyStore.all()
        let global = await appState.engine.settings.resolved().settings
        await MainActor.run {
            history = loaded
            settings = ImageGenSettings.fromGlobal(global)
            if !global.imageDefaultModelAlias.isEmpty {
                selected = ImageCatalog.all.first {
                    $0.displayName == global.imageDefaultModelAlias
                }
                if let s = selected, s.kind == .edit { tab = .edit }
            }
        }
    }

    private func recall(_ r: ImageGenerationRecord) {
        prompt = r.prompt
        if let data = try? JSONDecoder().decode(
            ImageGenSettings.self,
            from: Data(r.settingsJSON.utf8)) {
            settings = data
        }
        if let m = ImageCatalog.all.first(where: { $0.displayName == r.modelAlias }) {
            selected = m
            tab = (m.kind == .edit) ? .edit : .generate
        }
    }

    private func recallFromGenerated(_ img: GeneratedImage) {
        prompt = img.prompt
    }

    private func persistDefaults(_ s: ImageGenSettings) async {
        var g = await appState.engine.settings.resolved().settings
        g.imageDefaultSteps = s.steps
        g.imageDefaultGuidance = s.guidance
        g.imageDefaultWidth = s.width
        g.imageDefaultHeight = s.height
        g.imageDefaultSeed = s.seed
        g.imageDefaultNumImages = s.numImages
        g.imageDefaultScheduler = s.scheduler
        g.imageDefaultStrength = s.strength
        if let sel = selected { g.imageDefaultModelAlias = sel.displayName }
        await appState.engine.applySettings(g)
    }
}
