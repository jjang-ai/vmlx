import Foundation
@preconcurrency import Tokenizers
@preconcurrency import vMLXLMCommon

// MARK: - TransformersTokenizerLoader
//
// Hand-written replacement for the `#huggingFaceTokenizerLoader()` +
// `#adaptHuggingFaceTokenizer(_:)` macro pair that lived in
// `vmlx-swift-lm/Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift`.
//
// When we vendored vmlx-swift-lm into vmlx we dropped the macro target
// because swift-syntax macro resolution was flaky on the current
// toolchain (6.2.3) and the macro only had two call sites — this file
// and `#adaptHuggingFaceTokenizer(upstream)` inside the loader.
//
// The bridge wraps swift-transformers' `Tokenizers.Tokenizer` into
// `vMLXLMCommon.Tokenizer`. Every method forwards 1:1 to the upstream
// implementation, with two deliberate quirks:
//   1. `decode(tokenIds:)` in vMLXLMCommon maps to `decode(tokens:)` in
//      swift-transformers (name mismatch only, same semantics).
//   2. `applyChatTemplate` catches swift-transformers'
//      `TokenizerError.missingChatTemplate` and rethrows as
//      `vMLXLMCommon.TokenizerError.missingChatTemplate` so downstream
//      chat template fallback logic catches the right type.

/// Loads a tokenizer from a local model directory via swift-transformers'
/// `AutoTokenizer.from(modelFolder:)` and wraps it as an
/// `vMLXLMCommon.TokenizerLoader`.
public struct TransformersTokenizerLoader: vMLXLMCommon.TokenizerLoader {
    public init() {}

    public func load(from directory: URL) async throws -> any vMLXLMCommon.Tokenizer {
        let forceLagunaTemplateFallback =
            Self.shouldForceLagunaTemplateFallback(in: directory)
        let forceMistralTemplateFallback =
            Self.shouldForceMistralTemplateFallback(in: directory)
        let forceBailingTemplateFallback =
            Self.shouldForceBailingTemplateFallback(in: directory)
        let mistralDefaultSystemMessage =
            Self.loadMistralDefaultSystemMessage(in: directory)
        do {
            let upstream = try await AutoTokenizer.from(modelFolder: directory)
            return TransformersTokenizerBridge(
                upstream,
                forceLagunaTemplateFallback: forceLagunaTemplateFallback,
                forceMistralTemplateFallback: forceMistralTemplateFallback,
                forceBailingTemplateFallback: forceBailingTemplateFallback,
                mistralDefaultSystemMessage: mistralDefaultSystemMessage)
        } catch Tokenizers.TokenizerError.unsupportedTokenizer(let name) {
            // swift-transformers allowlists a fixed set of tokenizer classes
            // (Tokenizer.swift:103-119). HuggingFace model repos and some
            // mlx-community quants ship `tokenizer_config.json` with newer
            // or custom class names (e.g. "TokenizersBackend", "Qwen3Tokenizer",
            // "Gemma3Tokenizer") that break this lookup. The underlying
            // `tokenizer.json` is almost always a standard BPE/Unigram
            // payload that any of the allowlisted tokenizers can load.
            //
            // Retry by materializing a shadow copy of the model dir with a
            // patched `tokenizer_config.json` that substitutes a safe class
            // name. File contents are symlinked (not copied) to avoid
            // duplicating multi-GB safetensors.
            #if DEBUG
            print("[vMLX] tokenizer class '\(name)' unknown; retrying with substituted class name")
            #endif
            let fallbackClass = Self.fallbackTokenizerClass(for: name, in: directory)
            let shadow = try Self.makeShadowWithPatchedTokenizerClass(
                sourceDir: directory,
                newClass: fallbackClass)
            // iter-98 §125: `AutoTokenizer.from(modelFolder:)` reads the
            // shadow's `tokenizer_config.json` + `tokenizer.json` once and
            // returns a tokenizer object that no longer references the
            // folder. The shadow dir was otherwise orphaned under
            // `/var/folders/.../T/vmlx-tokenizer-shadow-<UUID>` for the
            // process lifetime, accumulating every time a JANG model
            // with a custom tokenizer class name got loaded (e.g.
            // "TokenizersBackend", "Qwen3Tokenizer"). Nuke it via defer
            // once the tokenizer is hydrated. `try?` on the remove
            // because failure to clean up is a log-only concern — the
            // bridge has the weights it needs by this point.
            defer { try? FileManager.default.removeItem(at: shadow) }
            let upstream = try await AutoTokenizer.from(modelFolder: shadow)
            return TransformersTokenizerBridge(
                upstream,
                forceLagunaTemplateFallback: forceLagunaTemplateFallback,
                forceMistralTemplateFallback: forceMistralTemplateFallback,
                forceBailingTemplateFallback: forceBailingTemplateFallback,
                mistralDefaultSystemMessage: mistralDefaultSystemMessage)
        }
    }

    /// Laguna bundles store `tokenizer_config.json::chat_template` as an
    /// include wrapper (`{% include 'chat_template.jinja' %}`). Depending on
    /// the Jinja runtime this can fail, or worse, "succeed" while rendering an
    /// empty prompt because the include loader has no model-directory context.
    /// In either case we need to bypass upstream Jinja and use the native
    /// Laguna fallback renderer.
    static func shouldForceLagunaTemplateFallback(in directory: URL) -> Bool {
        guard
            let cfgData = try? Data(contentsOf:
                directory.appendingPathComponent("config.json")),
            let cfg = try? JSONSerialization.jsonObject(with: cfgData) as? [String: Any],
            let modelType = cfg["model_type"] as? String,
            modelType.lowercased() == "laguna",
            let tokData = try? Data(contentsOf:
                directory.appendingPathComponent("tokenizer_config.json")),
            let tok = try? JSONSerialization.jsonObject(with: tokData) as? [String: Any],
            let template = tok["chat_template"] as? String
        else { return false }
        let lower = template.lowercased()
        return lower.contains("include") && lower.contains("chat_template.jinja")
    }

    /// Mistral-Medium-3.5 bundles can ship `chat_template.jinja` beside an
    /// otherwise-valid tokenizer, but omit `tokenizer_config.json::
    /// chat_template`. swift-transformers then reports
    /// `missingChatTemplate`, and the generic LLM processor falls back to
    /// plain text. For Mistral this is catastrophic: prompts must be wrapped
    /// in `[SYSTEM_PROMPT]`, `[MODEL_SETTINGS]`, and `[INST]` tokens.
    ///
    /// Force the built-in Mistral renderer for this bundle shape. This also
    /// covers the parser-limited case where native Jinja uses constructs the
    /// vendored parser cannot handle.
    static func shouldForceMistralTemplateFallback(in directory: URL) -> Bool {
        guard
            let cfgData = try? Data(contentsOf:
                directory.appendingPathComponent("config.json")),
            let cfg = try? JSONSerialization.jsonObject(with: cfgData) as? [String: Any]
        else { return false }

        let modelType = ((cfg["model_type"] as? String) ?? "").lowercased()
        let textModelType = (((cfg["text_config"] as? [String: Any])?["model_type"] as? String)
            ?? "").lowercased()
        let isMistralFamily =
            modelType.contains("mistral")
            || modelType.contains("ministral")
            || textModelType.contains("mistral")
            || textModelType.contains("ministral")
        guard isMistralFamily else { return false }

        let hasSidecarTemplate = FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("chat_template.jinja").path)

        let tokenizerTemplate: String? = {
            guard
                let tokData = try? Data(contentsOf:
                    directory.appendingPathComponent("tokenizer_config.json")),
                let tok = try? JSONSerialization.jsonObject(with: tokData) as? [String: Any]
            else { return nil }
            return tok["chat_template"] as? String
        }()

        if tokenizerTemplate == nil && hasSidecarTemplate { return true }
        if let template = tokenizerTemplate {
            let lower = template.lowercased()
            // Same unsupported shape as the native Mistral 3.5 sidecar:
            // loop_messages + [{...}] plus model-settings/tool sections.
            return lower.contains("loop_messages +")
                || lower.contains("[model_settings]")
                || lower.contains("[available_tools]")
        }
        return false
    }

    /// BailingHybrid / Ling templates use Jinja namespace mutation, reverse
    /// slicing, and tool-call object rewrites. Those are outside the subset
    /// that has historically been reliable in the vendored Jinja runtime, so
    /// render the native role-tag format directly for this model type.
    static func shouldForceBailingTemplateFallback(in directory: URL) -> Bool {
        guard
            let cfgData = try? Data(contentsOf:
                directory.appendingPathComponent("config.json")),
            let cfg = try? JSONSerialization.jsonObject(with: cfgData) as? [String: Any]
        else { return false }
        let modelType = ((cfg["model_type"] as? String) ?? "").lowercased()
        let textModelType = (((cfg["text_config"] as? [String: Any])?["model_type"] as? String)
            ?? "").lowercased()
        return modelType == "bailing_hybrid"
            || modelType == "bailing_moe_v2_5"
            || textModelType == "bailing_hybrid"
            || textModelType == "bailing_moe_v2_5"
    }

    static func loadMistralDefaultSystemMessage(in directory: URL) -> String? {
        guard shouldForceMistralTemplateFallback(in: directory) else { return nil }
        let url = directory.appendingPathComponent("SYSTEM_PROMPT.txt")
        guard var text = try? String(contentsOf: url, encoding: .utf8) else {
            return nil
        }
        // Keep this deterministic and close to the bundled template. The
        // native template's default is 2026-04-29 / 2026-04-28; replacing
        // the placeholder with real dates avoids surfacing `{today}` in the
        // prompt when SYSTEM_PROMPT.txt is the source.
        text = text.replacingOccurrences(of: "{today}", with: "03-05-2026")
        text = text.replacingOccurrences(of: "{yesterday}", with: "02-05-2026")
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Pick a safe tokenizer class for a repo whose `tokenizer_config.json`
    /// carries an unrecognized name. Mirrors the Osaurus/JANG converter
    /// compatibility map for the common `TokenizersBackend` case, and keeps
    /// `VMLX_TOKENIZER_CLASS_OVERRIDE=<target>` as an escape hatch for
    /// newly-published tokenizers that need a different concrete class before
    /// the map is updated.
    static func fallbackTokenizerClass(for original: String, in directory: URL) -> String {
        if let override = ProcessInfo.processInfo.environment["VMLX_TOKENIZER_CLASS_OVERRIDE"],
           !override.isEmpty
        {
            return override
        }

        // Probe config.json model_type if present.
        if let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            let mt = ((obj["model_type"] as? String)
                ?? ((obj["text_config"] as? [String: Any])?["model_type"] as? String)
                ?? "").lowercased()
            if mt.contains("qwen") { return "Qwen2Tokenizer" }
            if mt.contains("llama") { return "LlamaTokenizer" }
            if mt.contains("mistral") || mt.contains("ministral") { return "LlamaTokenizer" }
            if mt.contains("gemma") { return "GemmaTokenizer" }
            if mt.contains("nemotron") || mt.contains("minimax") { return "GPT2Tokenizer" }
        }
        _ = original  // silence warning
        return "PreTrainedTokenizer"
    }

    /// Build a shadow directory that symlinks every file from `sourceDir`
    /// EXCEPT `tokenizer_config.json`, which is copied with its
    /// `tokenizer_class` field overwritten. The caller is responsible for
    /// keeping the returned URL alive for the duration of the tokenizer load.
    /// We stash it under `FileManager.default.temporaryDirectory` with a UUID
    /// suffix so concurrent loads don't collide.
    static func makeShadowWithPatchedTokenizerClass(
        sourceDir: URL,
        newClass: String
    ) throws -> URL {
        let fm = FileManager.default
        let shadow = fm.temporaryDirectory
            .appendingPathComponent("vmlx-tokenizer-shadow-\(UUID().uuidString)")
        try fm.createDirectory(at: shadow, withIntermediateDirectories: true)

        let contents = try fm.contentsOfDirectory(at: sourceDir, includingPropertiesForKeys: nil)
        for src in contents {
            let dst = shadow.appendingPathComponent(src.lastPathComponent)
            if src.lastPathComponent == "tokenizer_config.json" { continue }
            try fm.createSymbolicLink(at: dst, withDestinationURL: src)
        }

        let configURL = sourceDir.appendingPathComponent("tokenizer_config.json")
        guard let data = try? Data(contentsOf: configURL),
              var obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            // No tokenizer_config.json — create a minimal one with the class.
            let minimal = ["tokenizer_class": newClass] as [String: Any]
            let out = try JSONSerialization.data(withJSONObject: minimal, options: [])
            try out.write(to: shadow.appendingPathComponent("tokenizer_config.json"))
            return shadow
        }
        obj["tokenizer_class"] = newClass
        let patched = try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted])
        try patched.write(to: shadow.appendingPathComponent("tokenizer_config.json"))
        return shadow
    }
}

/// Bridge struct — `Tokenizers.Tokenizer` → `vMLXLMCommon.Tokenizer`.
/// `@unchecked Sendable` because swift-transformers' protocol isn't
/// marked Sendable but the underlying implementations are immutable
/// after `from(modelFolder:)` returns.
struct TransformersTokenizerBridge: vMLXLMCommon.Tokenizer, @unchecked Sendable {
    private let upstream: any Tokenizers.Tokenizer
    private let forceLagunaTemplateFallback: Bool
    private let forceMistralTemplateFallback: Bool
    private let forceBailingTemplateFallback: Bool
    private let mistralDefaultSystemMessage: String?

    init(
        _ upstream: any Tokenizers.Tokenizer,
        forceLagunaTemplateFallback: Bool = false,
        forceMistralTemplateFallback: Bool = false,
        forceBailingTemplateFallback: Bool = false,
        mistralDefaultSystemMessage: String? = nil
    ) {
        self.upstream = upstream
        self.forceLagunaTemplateFallback = forceLagunaTemplateFallback
        self.forceMistralTemplateFallback = forceMistralTemplateFallback
        self.forceBailingTemplateFallback = forceBailingTemplateFallback
        self.mistralDefaultSystemMessage = mistralDefaultSystemMessage
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    // swift-transformers uses `decode(tokens:)`; we expose it under
    // the vMLXLMCommon name `decode(tokenIds:)`.
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        // iter-60: honor a caller-supplied chat-template override. The
        // engine sneaks the override through as a reserved
        // `__chat_template_override__` key in additionalContext so we
        // don't have to change the vMLXLMCommon.Tokenizer protocol.
        // When present + non-empty, swap to the swift-transformers
        // overload that takes `chatTemplate: .literal(...)`. This wires
        // up `--chat-template` (CLI) and `settings.chatTemplate`
        // (session/global) which pre-iter-60 were parsed + persisted
        // but never consumed (see §88 warning in Stream.swift).
        let overrideKey = "__chat_template_override__"
        var cleanContext = additionalContext
        let overrideTemplate: String? = {
            guard let raw = additionalContext?[overrideKey] as? String,
                  !raw.isEmpty
            else { return nil }
            cleanContext?.removeValue(forKey: overrideKey)
            return raw
        }()
        let addGen = ChatTemplateFallback.bool(
            in: cleanContext,
            key: "add_generation_prompt",
            default: true)
        let reasoningEffort = cleanContext?["reasoning_effort"] as? String
        let enableThinking = ChatTemplateFallback.bool(
            in: cleanContext,
            key: "enable_thinking",
            default: false)
        if overrideTemplate == nil
            && !enableThinking
            && upstream.bosToken == "]~!b["
            && upstream.eosToken == "[e~["
            && upstream.convertTokenToId("]~b]") != nil
            && upstream.convertTokenToId("[e~[") != nil
        {
            let rendered = ChatTemplateFallback.renderMiniMaxM2(
                messages: messages,
                addGenerationPrompt: addGen,
                tools: tools,
                enableThinking: enableThinking)
            if ProcessInfo.processInfo.environment["VMLX_TEMPLATE_TRACE"] == "1" {
                let preview = rendered.prefix(2000)
                FileHandle.standardError.write(Data(
                    "[template-trace] minimax_m2 native correction enable_thinking=\(enableThinking) add_generation_prompt=\(addGen)\n\(preview)\n[template-trace] end minimax_m2 prompt\n".utf8))
            }
            return upstream.encode(text: rendered, addSpecialTokens: false)
        }
        if overrideTemplate == nil && forceLagunaTemplateFallback {
            let rendered = ChatTemplateFallback.renderLaguna(
                messages: messages,
                addGenerationPrompt: addGen,
                bosToken: upstream.bosToken,
                tools: tools,
                enableThinking: enableThinking)
            if ProcessInfo.processInfo.environment["VMLX_TEMPLATE_TRACE"] == "1" {
                let preview = rendered.prefix(2000)
                FileHandle.standardError.write(Data(
                    "[template-trace] laguna native fallback enable_thinking=\(enableThinking) add_generation_prompt=\(addGen)\n\(preview)\n[template-trace] end laguna prompt\n".utf8))
            }
            return upstream.encode(text: rendered, addSpecialTokens: false)
        }
        if overrideTemplate == nil && forceMistralTemplateFallback {
            let rendered = ChatTemplateFallback.renderMistral(
                messages: messages,
                addGenerationPrompt: addGen,
                bosToken: upstream.bosToken,
                tools: tools,
                reasoningEffort: reasoningEffort,
                hasModelSettings: upstream.convertTokenToId("[MODEL_SETTINGS]") != nil,
                hasSystemPromptTokens:
                    upstream.convertTokenToId("[SYSTEM_PROMPT]") != nil
                    && upstream.convertTokenToId("[/SYSTEM_PROMPT]") != nil,
                defaultSystemMessage: mistralDefaultSystemMessage)
            if ProcessInfo.processInfo.environment["VMLX_TEMPLATE_TRACE"] == "1" {
                let preview = rendered.prefix(2000)
                FileHandle.standardError.write(Data(
                    "[template-trace] mistral native fallback enable_thinking=\(enableThinking) add_generation_prompt=\(addGen)\n\(preview)\n[template-trace] end mistral prompt\n".utf8))
            }
            return upstream.encode(text: rendered, addSpecialTokens: false)
        }
        if overrideTemplate == nil && forceBailingTemplateFallback {
            let rendered = ChatTemplateFallback.renderBailingHybrid(
                messages: messages,
                addGenerationPrompt: addGen,
                tools: tools,
                enableThinking: enableThinking)
            if ProcessInfo.processInfo.environment["VMLX_TEMPLATE_TRACE"] == "1" {
                let preview = rendered.prefix(2000)
                FileHandle.standardError.write(Data(
                    "[template-trace] bailing_hybrid native fallback enable_thinking=\(enableThinking) add_generation_prompt=\(addGen)\n\(preview)\n[template-trace] end bailing_hybrid prompt\n".utf8))
            }
            return upstream.encode(text: rendered, addSpecialTokens: false)
        }
        do {
            if let template = overrideTemplate {
                return try upstream.applyChatTemplate(
                    messages: messages,
                    chatTemplate: .literal(template),
                    addGenerationPrompt: addGen,
                    truncation: false,
                    maxLength: nil,
                    tools: tools,
                    additionalContext: cleanContext)
            }
            return try upstream.applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: cleanContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            // No template at all — let the upstream "missing template"
            // branch rethrow so downstream code can pick a default.
            throw vMLXLMCommon.TokenizerError.missingChatTemplate
        } catch {
            // Template exists but swift-jinja failed to parse or render it.
            // This hits on Gemma 4 family templates (and any other HF model
            // whose chat_template.jinja uses constructs outside the subset
            // supported by github.com/johnmai-dev/Jinja). Rather than 500
            // every /v1/chat/completions, fall back to a built-in ChatML
            // renderer so serve stays usable.
            //
            // The rendered string is then encoded via the real tokenizer
            // (so BOS/EOS/special tokens are preserved), giving output
            // that is not ideal but is at least structurally valid.
            if ProcessInfo.processInfo.environment["VMLX_TEMPLATE_TRACE"] == "1" {
                FileHandle.standardError.write(Data(
                    "[template-trace] fallback fired: \(error) (msgs=\(messages.count))\n".utf8))
            }
            #if DEBUG
            print("[vMLX] applyChatTemplate fallback: upstream Jinja failed with \(error). Using built-in family renderer.")
            #endif
            // Mistral renderer needs `reasoning_effort` from
            // additionalContext (Stream.swift:816 stamps it via
            // templateExtras). Without this hint the renderer emits
            // `[MODEL_SETTINGS]{"reasoning_effort": "none"}` for every
            // request — wrong for the 99% of users on default settings.
            let rendered = ChatTemplateFallback.render(
                messages: messages,
                addGenerationPrompt: addGen,
                bosToken: upstream.bosToken,
                tokenExists: { upstream.convertTokenToId($0) != nil },
                tools: tools,
                reasoningEffort: reasoningEffort,
                enableThinking: enableThinking)
            return upstream.encode(text: rendered, addSpecialTokens: false)
        }
    }
}

// MARK: - Fallback ChatML renderer
//
// Last-resort safety net invoked when swift-jinja can't parse or render
// a model's chat_template.jinja. ChatML is not every model's preferred
// format, but every tokenizer we ship CAN encode it, and it keeps the
// server responding so users aren't blocked by upstream Jinja bugs.
//
// Emitted format (per message):
//   <|im_start|>{role}\n{content}<|im_end|>\n
// Plus a trailing `<|im_start|>assistant\n` when addGenerationPrompt is true.
// Prefixed with {bos_token} if the tokenizer has one.
enum ChatTemplateFallback {
    /// Dispatch to the best-fitting renderer for the loaded tokenizer.
    ///
    /// We sniff the tokenizer's vocab for family-specific role tags and
    /// pick the matching format. This avoids the problem where a generic
    /// ChatML prompt fed to e.g. a Gemma model makes the model immediately
    /// emit EOS (because `<|im_start|>` is not in Gemma's training
    /// distribution at all) and therefore produces empty responses.
    ///
    /// Supported families (in priority order):
    ///   - **Gemma 4** — `<|turn>role\n...<turn|>\n` with role `user` or
    ///     `model`. The `<|turn>` / `<turn|>` tokens are native in
    ///     Gemma 4's vocab (ids 105/106 in the e2b/e4b/26B/31B lineup).
    ///   - **Gemma 2 / Gemma 3** — `<start_of_turn>role\n...<end_of_turn>\n`.
    ///   - **Llama 3** — `<|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>`
    ///   - Fallback: ChatML — `<|im_start|>role\n...<|im_end|>\n`
    static func render(
        messages: [[String: any Sendable]],
        addGenerationPrompt: Bool,
        bosToken: String?,
        tokenExists: (String) -> Bool,
        tools: [[String: any Sendable]]? = nil,
        reasoningEffort: String? = nil,
        enableThinking: Bool = false
    ) -> String {
        // Laguna / Poolside family. The public bundles ship
        // `tokenizer_config.json::chat_template` as only:
        //
        //   {% include 'chat_template.jinja' %}
        //
        // swift-transformers' Jinja renderer does not resolve that include
        // against the model directory, so the bridge falls into this fallback
        // path. Before this branch, Laguna fell all the way through to ChatML
        // (`<|im_start|>...`) even though the model was trained on
        // `<system>/<user>/<assistant>` turns, causing visible prompt/template
        // garbage in the app. Detect the native Laguna marker set and render
        // the template shape directly.
        if tokenExists("<assistant>")
            && tokenExists("</assistant>")
            && tokenExists("<think>")
            && tokenExists("</think>")
            && bosToken == "〈|EOS|〉"
        {
            return renderLaguna(
                messages: messages,
                addGenerationPrompt: addGenerationPrompt,
                bosToken: bosToken,
                tools: tools,
                enableThinking: enableThinking)
        }

        // BailingHybrid / Ling native role-tag format. Keep this before the
        // generic ChatML fallback; `<role>...<|role_end|>` is the trained
        // grammar and generic `<|im_start|>` prompts are out of distribution.
        if tokenExists("<role>")
            && tokenExists("</role>")
            && tokenExists("<|role_end|>")
            && tokenExists("<think>")
            && tokenExists("</think>")
        {
            return renderBailingHybrid(
                messages: messages,
                addGenerationPrompt: addGenerationPrompt,
                tools: tools,
                enableThinking: enableThinking)
        }

        // Nemotron-H / Omni. These tokenizers include Mistral-style
        // `[INST]` specials for compatibility, so this must run before the
        // Mistral sniff below. The native template is ChatML-like but load
        // bearingly stamps `<think></think>` when thinking is disabled.
        if tokenExists("<|im_start|>")
            && tokenExists("<|im_end|>")
            && tokenExists("<think>")
            && tokenExists("</think>")
            && tokenExists("<tool_call>")
            && tokenExists("<tool_response>")
            && tokenExists("[AVAILABLE_TOOLS]")
        {
            return renderNemotron(
                messages: messages,
                addGenerationPrompt: addGenerationPrompt,
                bosToken: bosToken,
                tools: tools,
                enableThinking: enableThinking)
        }

        // Mistral family — sniff `[INST]` special token. This is the
        // cleanest signal across every Mistral 3 / Mistral 3.5 /
        // Mistral 4 / ministral3 distribution (per
        // vmlx-swift-lm/ChatTemplateFallbacks.swift). Native Mistral 3.5
        // template uses `for message in loop_messages + [{...}]` which
        // our vendored johnmai-dev/Jinja 1.3.0 parser can't handle —
        // falling through to ChatML emits `<|im_start|>` markers that
        // Mistral was never trained on, producing incoherent output for
        // BOTH MXFP4 AND JANGTQ Mistral bundles. The osaurus-ai
        // swift-jinja fork (revision 58d21aa) patches the lexer for
        // this; we use a content-equivalent Mistral renderer instead.
        if tokenExists("[INST]") && tokenExists("[/INST]") {
            return renderMistral(
                messages: messages,
                addGenerationPrompt: addGenerationPrompt,
                bosToken: bosToken,
                tools: tools,
                reasoningEffort: reasoningEffort,
                hasModelSettings: tokenExists("[MODEL_SETTINGS]"),
                hasSystemPromptTokens:
                    tokenExists("[SYSTEM_PROMPT]") && tokenExists("[/SYSTEM_PROMPT]"))
        }
        if tokenExists("<|turn>") && tokenExists("<turn|>") {
            return renderGemma4(
                messages: messages,
                addGenerationPrompt: addGenerationPrompt,
                bosToken: bosToken,
                enableThinking: enableThinking)
        }
        if tokenExists("<start_of_turn>") && tokenExists("<end_of_turn>") {
            return renderGemma(
                messages: messages,
                addGenerationPrompt: addGenerationPrompt,
                bosToken: bosToken)
        }
        if tokenExists("<|start_header_id|>") && tokenExists("<|end_header_id|>") {
            return renderLlama3(
                messages: messages,
                addGenerationPrompt: addGenerationPrompt,
                bosToken: bosToken)
        }
        return renderChatML(
            messages: messages,
            addGenerationPrompt: addGenerationPrompt,
            bosToken: bosToken)
    }

    /// BailingHybrid / Ling role-tag chat format. Mirrors the bundled
    /// `tokenizer_config.json::chat_template` for the production paths:
    /// system/tool preamble, `<role>HUMAN</role>` user turns,
    /// `<role>ASSISTANT</role>` assistant turns, grouped observation turns,
    /// and a final assistant rail for generation.
    static func renderBailingHybrid(
        messages: [[String: any Sendable]],
        addGenerationPrompt: Bool,
        tools: [[String: any Sendable]]?,
        enableThinking: Bool
    ) -> String {
        let thinkingOption = enableThinking ? "on" : "off"
        var out = "<role>SYSTEM</role>"
        let firstRole = messages.first?["role"] as? String
        let firstSystem = (firstRole == "system" || firstRole == "developer")
            ? stringifyContent(messages.first?["content"])
            : nil

        if let tools, !tools.isEmpty {
            if let firstSystem { out += firstSystem + "\n" }
            out += "# Tools\n\n"
            out += "You may call one or more functions to assist with the user query.\n\n"
            out += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in tools {
                out += "\n" + jsonString(tool)
            }
            out += "\n</tools>\n\n"
            out += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            out += "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n"
            out += "detailed thinking \(thinkingOption)<|role_end|>"
        } else if let firstSystem {
            if firstSystem.contains("detailed thinking on")
                || firstSystem.contains("detailed thinking off")
            {
                out += firstSystem + "<|role_end|>"
            } else {
                out += firstSystem + "\n"
                out += "detailed thinking \(thinkingOption)<|role_end|>"
            }
        } else {
            out += "detailed thinking \(thinkingOption)<|role_end|>"
        }

        var lastQueryIndex = messages.count - 1
        for idx in stride(from: messages.count - 1, through: 0, by: -1) {
            let message = messages[idx]
            guard (message["role"] as? String) == "user" else { continue }
            let content = stringifyContent(message["content"])
            if !(content.hasPrefix("<tool_response>")
                && content.hasSuffix("</tool_response>"))
            {
                lastQueryIndex = idx
                break
            }
        }

        for (idx, message) in messages.enumerated() {
            let role = (message["role"] as? String) ?? "user"
            if idx == 0 && (role == "system" || role == "developer") {
                continue
            }
            let content = stringifyContent(message["content"])
            switch role {
            case "user":
                out += "<role>HUMAN</role>" + content + "<|role_end|>"
            case "system", "developer":
                out += "<role>SYSTEM</role>" + content + "<|role_end|>"
            case "assistant":
                var visible = content
                var reasoning = (message["reasoning_content"] as? String)
                    ?? (message["reasoning"] as? String)
                    ?? ""
                if let end = visible.range(of: "</think>") {
                    if reasoning.isEmpty {
                        let before = String(visible[..<end.lowerBound])
                        if let start = before.range(of: "<think>") {
                            reasoning = String(before[start.upperBound...])
                        } else {
                            reasoning = before
                        }
                    }
                    visible = String(visible[end.upperBound...])
                }
                out += "<role>ASSISTANT</role>"
                let trimmedReasoning = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
                if idx > lastQueryIndex && !trimmedReasoning.isEmpty {
                    out += "\n<think>\n"
                    out += trimmedReasoning
                    out += "\n</think>\n\n"
                    out += visible.trimmingCharacters(in: .whitespacesAndNewlines)
                } else {
                    out += visible
                }
                if let toolCalls = message["tool_calls"] as? [[String: any Sendable]],
                   !toolCalls.isEmpty
                {
                    for call in toolCalls {
                        out += "\n" + renderBailingToolCall(call)
                    }
                }
                out += "<|role_end|>"
            case "tool":
                let previousRole = idx > 0 ? messages[idx - 1]["role"] as? String : nil
                let nextRole = idx + 1 < messages.count
                    ? messages[idx + 1]["role"] as? String
                    : nil
                if previousRole != "tool" {
                    out += "<role>OBSERVATION</role>"
                }
                out += "\n<tool_response>\n"
                out += content
                out += "\n</tool_response>\n"
                if nextRole != "tool" {
                    out += "<|role_end|>"
                }
            default:
                break
            }
        }

        if addGenerationPrompt {
            out += "<role>ASSISTANT</role>"
        }
        return out
    }

    /// Laguna / Poolside chat format. This mirrors the bundled
    /// `chat_template.jinja` for the common server path and replaces the
    /// broken `{% include 'chat_template.jinja' %}` render fallback.
    ///
    /// Format:
    ///   〈|EOS|〉
    ///   <system>\n\n...\n</system>\n
    ///   <user>\n...\n</user>\n
    ///   <assistant>\n</think> or <think>...
    static func renderLaguna(
        messages: [[String: any Sendable]],
        addGenerationPrompt: Bool,
        bosToken: String?,
        tools: [[String: any Sendable]]?,
        enableThinking: Bool
    ) -> String {
        let defaultSystem =
            "You are a helpful, conversationally-fluent assistant made by Poolside. " +
            "You are here to be helpful to users through natural language conversations."

        var out = ""
        if let bos = bosToken, !bos.isEmpty { out += bos }

        var systemMessage = defaultSystem
        var body: [[String: any Sendable]] = []
        for (idx, message) in messages.enumerated() {
            let role = (message["role"] as? String) ?? "user"
            if idx == 0 && role == "system" {
                systemMessage = stringifyContent(message["content"])
            } else {
                body.append(message)
            }
        }

        let hasSystem = !systemMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let hasTools = !(tools?.isEmpty ?? true)
        if hasSystem || hasTools {
            out += "<system>\n"
            if hasSystem {
                out += "\n"
                out += systemMessage.trimmingCharacters(in: .newlines)
            }
            if let tools, !tools.isEmpty {
                out += "\n\n### Tools\n\n"
                out += "You may call functions to assist with the user query.\n"
                out += "All available function signatures are listed below:\n"
                out += "<available_tools>\n"
                for tool in tools {
                    if let data = try? JSONSerialization.data(
                        withJSONObject: tool,
                        options: [.fragmentsAllowed, .sortedKeys]),
                       let json = String(data: data, encoding: .utf8)
                    {
                        out += json + "\n"
                    }
                }
                out += "</available_tools>\n\n"
                if enableThinking {
                    out += "Wrap your thinking in '<think>', '</think>' tags, followed by a function call. "
                }
                out += "For each function call, return an unescaped XML-like object with function name and arguments within '<tool_call>' and '</tool_call>' tags."
            }
            out += "\n</system>\n"
        }

        for message in body {
            let role = (message["role"] as? String) ?? "user"
            var content = stringifyContent(message["content"])
            switch role {
            case "user":
                out += "<user>\n\(content)\n</user>\n"
            case "assistant":
                out += "<assistant>\n"
                var reasoning = ""
                if let r = message["reasoning"] as? String {
                    reasoning = r
                } else if let r = message["reasoning_content"] as? String {
                    reasoning = r
                }
                if let end = content.range(of: "</think>") {
                    if reasoning.isEmpty {
                        let before = String(content[..<end.lowerBound])
                        if let start = before.range(of: "<think>") {
                            reasoning = String(before[start.upperBound...])
                        } else {
                            reasoning = before
                        }
                        reasoning = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                    content = String(content[end.upperBound...])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                }
                if !reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    out += "<think>\n"
                    out += reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
                    out += "\n</think>\n"
                } else {
                    out += "</think>\n"
                }
                let trimmedContent = content.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedContent.isEmpty {
                    out += trimmedContent + "\n"
                }
                out += "</assistant>\n"
            case "tool":
                out += "<tool_response>\n\(content)\n</tool_response>\n"
            case "system", "developer":
                out += "<system>\n\(content)\n</system>\n"
            default:
                break
            }
        }

        if addGenerationPrompt {
            out += "<assistant>\n"
            out += enableThinking ? "<think>" : "</think>"
        }
        return out
    }

    /// MiniMax-M2 chat format. The public M2/M2.7 templates are close to
    /// correct, but their thinking-off generation prompt leaves the model on
    /// a bare assistant rail. In short direct-answer probes the model often
    /// spends the whole budget emitting structural/special tokens. The
    /// reference Swift runtime corrects this by stamping a closed empty think
    /// block when `enable_thinking=false`, while preserving the native open
    /// `<think>` prefill for thinking-on requests.
    static func renderMiniMaxM2(
        messages: [[String: any Sendable]],
        addGenerationPrompt: Bool,
        tools: [[String: any Sendable]]?,
        enableThinking: Bool
    ) -> String {
        let defaultSystem =
            "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."

        var systemMessage = defaultSystem
        var body: [[String: any Sendable]] = []
        for (idx, message) in messages.enumerated() {
            let role = (message["role"] as? String) ?? "user"
            if idx == 0 && role == "system" {
                let content = stringifyContent(message["content"])
                if !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    systemMessage = content
                }
            } else {
                body.append(message)
            }
        }

        var out = "]~!b[]~b]system\n"
        out += systemMessage
        if let tools, !tools.isEmpty {
            out += "\n\n# Tools\n"
            out += "You may call one or more tools to assist with the user query.\n"
            out += "Here are the tools available in JSONSchema format:\n"
            out += "<tools>\n"
            for tool in tools {
                if let data = try? JSONSerialization.data(
                    withJSONObject: tool,
                    options: [.fragmentsAllowed, .sortedKeys]),
                   let json = String(data: data, encoding: .utf8)
                {
                    out += "<tool>\(json)</tool>\n"
                }
            }
            out += "</tools>\n"
            out += "When making tool calls, use XML format to invoke tools and pass parameters:\n"
            out += "<minimax:tool_call>\n"
            out += "<invoke name=\"tool-name\"><parameter name=\"param-key\">param-value</parameter></invoke>\n"
            out += "</minimax:tool_call>"
        }
        out += "[e~[\n"

        var lastUserIndex = -1
        for (idx, message) in body.enumerated() {
            if (message["role"] as? String) == "user" {
                lastUserIndex = idx
            }
        }

        for (idx, message) in body.enumerated() {
            let role = (message["role"] as? String) ?? "user"
            switch role {
            case "assistant":
                var content = stringifyContent(message["content"])
                var reasoning = (message["reasoning_content"] as? String)
                    ?? (message["reasoning"] as? String)
                    ?? ""
                if let end = content.range(of: "</think>") {
                    if reasoning.isEmpty {
                        let before = String(content[..<end.lowerBound])
                        if let start = before.range(of: "<think>") {
                            reasoning = String(before[start.upperBound...])
                        } else {
                            reasoning = before
                        }
                    }
                    content = String(content[end.upperBound...])
                }
                out += "]~b]ai\n"
                let trimmedReasoning = reasoning
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedReasoning.isEmpty && idx > lastUserIndex {
                    out += "<think>\n\(trimmedReasoning)\n</think>\n\n"
                }
                let trimmedContent = content
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedContent.isEmpty { out += trimmedContent }
                out += "[e~[\n"
            case "tool":
                out += "]~b]tool\n"
                out += "<response>"
                out += stringifyContent(message["content"])
                out += "</response>[e~[\n"
            case "user":
                out += "]~b]user\n"
                out += stringifyContent(message["content"])
                out += "[e~[\n"
            case "system", "developer":
                // The native template only treats the first system message
                // specially. Later system/developer turns are folded onto the
                // user rail so the text is not silently dropped.
                out += "]~b]user\n"
                out += stringifyContent(message["content"])
                out += "[e~[\n"
            default:
                break
            }
        }

        if addGenerationPrompt {
            out += "]~b]ai\n"
            out += enableThinking ? "<think>\n" : "<think>\n</think>\n\n"
        }
        return out
    }

    /// Nemotron-H / Omni chat format. The bundled template is
    /// `<|im_start|>role\n...<|im_end|>` plus a generation prompt that
    /// opens `<think>` when thinking is enabled and emits `<think></think>`
    /// when disabled. Generic ChatML lacks that stamp; Mistral fallback is
    /// also wrong because Nemotron tokenizers include `[INST]` compatibility
    /// tokens despite training on the ChatML-like rail.
    static func renderNemotron(
        messages: [[String: any Sendable]],
        addGenerationPrompt: Bool,
        bosToken: String?,
        tools: [[String: any Sendable]]?,
        enableThinking: Bool
    ) -> String {
        var out = ""
        if let bos = bosToken, !bos.isEmpty { out += bos }

        var body: [[String: any Sendable]] = []
        var systemContent = ""
        for message in messages {
            let role = (message["role"] as? String) ?? "user"
            if role == "system" || role == "developer" {
                let content = sanitizeNemotronContent(stringifyContent(message["content"]))
                systemContent += (systemContent.isEmpty ? "" : "\n") + content
            } else {
                body.append(message)
            }
        }

        if !systemContent.isEmpty || !(tools?.isEmpty ?? true) {
            out += "<|im_start|>system\n"
            out += systemContent
            if let tools, !tools.isEmpty {
                if !systemContent.isEmpty { out += "\n\n" }
                out += "# Tools\n\nYou have access to the following functions:\n\n<tools>"
                for tool in tools {
                    if let data = try? JSONSerialization.data(
                        withJSONObject: tool,
                        options: [.fragmentsAllowed, .sortedKeys]),
                       let json = String(data: data, encoding: .utf8)
                    {
                        out += "\n<function>\(json)</function>"
                    }
                }
                out += "\n</tools>"
            }
            out += "<|im_end|>\n"
        }

        for message in body {
            let role = (message["role"] as? String) ?? "user"
            switch role {
            case "assistant":
                var content = stringifyContent(message["content"])
                if let reasoning = (message["reasoning_content"] as? String)
                    ?? (message["reasoning"] as? String),
                   !reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                {
                    content = "<think>\n"
                        + reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
                        + "\n</think>\n"
                        + content
                } else if !content.contains("<think>") && !content.contains("</think>") {
                    content = "<think></think>" + content
                }
                out += "<|im_start|>assistant\n"
                out += content.trimmingCharacters(in: .whitespacesAndNewlines)
                out += "<|im_end|>\n"
            case "tool":
                out += "<|im_start|>user\n<tool_response>\n"
                out += stringifyContent(message["content"])
                out += "\n</tool_response>\n<|im_end|>\n"
            case "user", "system", "developer":
                let content = sanitizeNemotronContent(stringifyContent(message["content"]))
                let emitRole = role == "developer" ? "system" : role
                out += "<|im_start|>\(emitRole)\n"
                out += content.trimmingCharacters(in: .whitespacesAndNewlines)
                out += "<|im_end|>\n"
            default:
                let content = sanitizeNemotronContent(stringifyContent(message["content"]))
                out += "<|im_start|>\(role)\n"
                out += content.trimmingCharacters(in: .whitespacesAndNewlines)
                out += "<|im_end|>\n"
            }
        }

        if addGenerationPrompt {
            out += enableThinking
                ? "<|im_start|>assistant\n<think>\n"
                : "<|im_start|>assistant\n<think></think>"
        }
        return out
    }

    private static func sanitizeNemotronContent(_ content: String) -> String {
        content
            .replacingOccurrences(of: "</think>", with: "<_end_think>")
            .replacingOccurrences(of: "/think", with: "")
            .replacingOccurrences(of: "/no_think", with: "")
            .replacingOccurrences(of: "<_end_think>", with: "</think>")
    }

    /// Mistral family format — Mistral 3 / Mistral 3.5 / Mistral 4 /
    /// ministral3. The native template uses:
    ///   `[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]` for system role
    ///   `[MODEL_SETTINGS]{"reasoning_effort":"..."}[/MODEL_SETTINGS]` for Mistral 3.5+ reasoning gate
    ///   `[AVAILABLE_TOOLS][...json...][/AVAILABLE_TOOLS]` for tools declaration
    ///   `[INST]...[/INST]` for user turns
    ///   `assistant text` (raw) followed by `eos_token` for assistant turns
    ///   `[TOOL_CALLS][...json...][/TOOL_CALLS]` for assistant tool calls
    ///   `[TOOL_RESULTS]...[/TOOL_RESULTS]` for tool replies
    ///
    /// Mirrors `vmlx-swift-lm/ChatTemplateFallbacks._mistral3Minimal_DEPRECATED`
    /// — that template is what kept Mistral 3.5 coherent before the
    /// osaurus-ai swift-jinja fork landed the for-iterable `+` patch.
    /// We don't have the patched parser, so we use this renderer
    /// directly. Output is bit-equivalent to a successful native render
    /// for the common case (no consecutive same-role messages).
    static func renderMistral(
        messages: [[String: any Sendable]],
        addGenerationPrompt: Bool,
        bosToken: String?,
        tools: [[String: any Sendable]]?,
        reasoningEffort: String?,
        hasModelSettings: Bool,
        hasSystemPromptTokens: Bool,
        defaultSystemMessage: String? = nil
    ) -> String {
        var out = ""
        if let bos = bosToken, !bos.isEmpty { out += bos }

        // System message → `[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]` when the
        // tokenizer has those special tokens (Mistral 3.5+); otherwise
        // fold into the first `[INST]`.
        var systemContent = ""
        var body: [[String: any Sendable]] = []
        for m in messages {
            let role = (m["role"] as? String) ?? "user"
            if role == "system" || role == "developer" {
                let c = stringifyContent(m["content"])
                systemContent += (systemContent.isEmpty ? "" : "\n") + c
            } else {
                body.append(m)
            }
        }
        if systemContent.isEmpty,
           let defaultSystemMessage,
           !defaultSystemMessage.isEmpty,
           hasSystemPromptTokens
        {
            systemContent = defaultSystemMessage
        }
        if !systemContent.isEmpty && hasSystemPromptTokens {
            out += "[SYSTEM_PROMPT]\(systemContent)[/SYSTEM_PROMPT]"
        }

        // Mistral 3.5+ — `[MODEL_SETTINGS]{"reasoning_effort":"..."}[/MODEL_SETTINGS]`
        // gates the model's reasoning behavior. The `effort` value comes
        // from `templateExtras["reasoning_effort"]` set by Stream.swift
        // line 815-817 (BLOCKER #5 sampling cascade work). When the
        // tokenizer lacks `[MODEL_SETTINGS]` (legacy Mistral 3) skip
        // this block — emitting unknown special tokens degrades the
        // output.
        if hasModelSettings {
            let effort = (reasoningEffort?.isEmpty == false) ? reasoningEffort! : "none"
            out += "[MODEL_SETTINGS]{\"reasoning_effort\": \"\(effort)\"}[/MODEL_SETTINGS]"
        }

        // Tools declaration — `[AVAILABLE_TOOLS][...][/AVAILABLE_TOOLS]`.
        if let tools = tools, !tools.isEmpty {
            // JSON-serialize the tools array. Mistral expects each tool
            // entry to be a `{"type":"function","function":{...}}` dict
            // — the same shape the OpenAI ChatCompletion tools field
            // already uses, so we can pass through as-is.
            if let data = try? JSONSerialization.data(
                    withJSONObject: tools,
                    options: [.fragmentsAllowed, .sortedKeys]),
               let json = String(data: data, encoding: .utf8) {
                out += "[AVAILABLE_TOOLS]\(json)[/AVAILABLE_TOOLS]"
            }
        }

        for m in body {
            let role = (m["role"] as? String) ?? "user"
            let content = stringifyContent(m["content"])
            switch role {
            case "user":
                // First [INST] absorbs system content if no SYSTEM_PROMPT
                // tokens. (Legacy Mistral 3 — pre-3.5 — doesn't have
                // `[SYSTEM_PROMPT]`.)
                var instBody = content
                if !systemContent.isEmpty && !hasSystemPromptTokens {
                    instBody = systemContent + "\n\n" + instBody
                    systemContent = ""
                }
                out += "[INST]\(instBody)[/INST]"
            case "assistant":
                // Tool-call assistant messages: `[TOOL_CALLS]...[/TOOL_CALLS]`.
                if let toolCalls = m["tool_calls"] as? [[String: any Sendable]],
                   !toolCalls.isEmpty {
                    if let data = try? JSONSerialization.data(
                            withJSONObject: toolCalls,
                            options: [.fragmentsAllowed, .sortedKeys]),
                       let json = String(data: data, encoding: .utf8) {
                        out += "[TOOL_CALLS]\(json)[/TOOL_CALLS]"
                    }
                } else {
                    // Plain assistant turn — content followed by eos_token.
                    // We don't append eos here; the encoder adds it via
                    // `add_special_tokens` at encode time only when the
                    // template explicitly requests it. Append the literal
                    // `</s>` form per Mistral convention; downstream
                    // tokenizer maps it to the EOS id.
                    out += "\(content)</s>"
                }
            case "tool":
                out += "[TOOL_RESULTS]\(content)[/TOOL_RESULTS]"
            default:
                break
            }
        }

        // No explicit "model turn opener" in Mistral — generation
        // begins immediately after the last `[/INST]` or `[/TOOL_RESULTS]`.
        // The `addGenerationPrompt` flag is satisfied by the body
        // ending in those markers naturally.
        _ = addGenerationPrompt
        return out
    }

    /// Gemma 4 format. Roles: `user`, `model` (not `assistant`). System
    /// messages are prepended to the first user turn — Gemma 4 has no
    /// system role in the turn grammar, only a per-template `<|turn>system`
    /// header that the HF chat_template synthesizes.
    static func renderGemma4(
        messages: [[String: any Sendable]],
        addGenerationPrompt: Bool,
        bosToken: String?,
        enableThinking: Bool = false
    ) -> String {
        var out = ""
        if let bos = bosToken, !bos.isEmpty { out += bos }
        // Collect any leading system message(s) and emit them as their own
        // `<|turn>system\n...<turn|>` block, matching the upstream template.
        var pendingSystem = ""
        var body: [[String: any Sendable]] = []
        for message in messages {
            let role = (message["role"] as? String) ?? "user"
            if role == "system" || role == "developer" {
                let c = stringifyContent(message["content"])
                pendingSystem += (pendingSystem.isEmpty ? "" : "\n") + c
            } else {
                body.append(message)
            }
        }
        if enableThinking {
            pendingSystem = pendingSystem.isEmpty ? "<|think|>" : "<|think|>\n\(pendingSystem)"
        }
        if !pendingSystem.isEmpty {
            out += "<|turn>system\n\(pendingSystem)<turn|>\n"
        }
        for message in body {
            let rawRole = (message["role"] as? String) ?? "user"
            let role = rawRole == "assistant" ? "model" : "user"
            let content = stringifyContent(message["content"])
            out += "<|turn>\(role)\n\(content)<turn|>\n"
        }
        if addGenerationPrompt {
            out += "<|turn>model\n"
            if !enableThinking {
                out += "<|channel>thought\n<channel|>"
            }
        }
        return out
    }

    /// Gemma 2 / Gemma 3 format (pre-Gemma-4). Roles: `user`, `model`.
    /// System messages are prepended to the first user turn.
    static func renderGemma(
        messages: [[String: any Sendable]],
        addGenerationPrompt: Bool,
        bosToken: String?
    ) -> String {
        var out = ""
        if let bos = bosToken, !bos.isEmpty { out += bos }
        var pendingSystem = ""
        for message in messages {
            let rawRole = (message["role"] as? String) ?? "user"
            let content = stringifyContent(message["content"])
            if rawRole == "system" || rawRole == "developer" {
                pendingSystem += (pendingSystem.isEmpty ? "" : "\n") + content
                continue
            }
            let role = rawRole == "assistant" ? "model" : "user"
            var body = content
            if role == "user" && !pendingSystem.isEmpty {
                body = pendingSystem + "\n\n" + body
                pendingSystem = ""
            }
            out += "<start_of_turn>\(role)\n\(body)<end_of_turn>\n"
        }
        // Stray system message with no following user turn — still inject it.
        if !pendingSystem.isEmpty {
            out += "<start_of_turn>user\n\(pendingSystem)<end_of_turn>\n"
        }
        if addGenerationPrompt {
            out += "<start_of_turn>model\n"
        }
        return out
    }

    /// Llama 3 format using the header-id tokens. System messages land as
    /// their own turn at the start.
    static func renderLlama3(
        messages: [[String: any Sendable]],
        addGenerationPrompt: Bool,
        bosToken: String?
    ) -> String {
        var out = ""
        if let bos = bosToken, !bos.isEmpty { out += bos }
        for message in messages {
            let role = (message["role"] as? String) ?? "user"
            let content = stringifyContent(message["content"])
            out += "<|start_header_id|>\(role)<|end_header_id|>\n\n\(content)<|eot_id|>"
        }
        if addGenerationPrompt {
            out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        }
        return out
    }

    static func renderChatML(
        messages: [[String: any Sendable]],
        addGenerationPrompt: Bool,
        bosToken: String?
    ) -> String {
        var out = ""
        if let bos = bosToken, !bos.isEmpty {
            out += bos
        }
        for message in messages {
            let role = (message["role"] as? String) ?? "user"
            let content = stringifyContent(message["content"])
            out += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
        }
        if addGenerationPrompt {
            out += "<|im_start|>assistant\n"
        }
        return out
    }

    /// OpenAI-style messages sometimes carry content as an array of parts
    /// (text + image_url + etc). Flatten to text for the fallback; non-text
    /// parts are replaced by a placeholder so the user's intent is at least
    /// visible in the prompt.
    static func stringifyContent(_ raw: Any?) -> String {
        if let s = raw as? String { return s }
        if let parts = raw as? [[String: Any]] {
            var buf = ""
            for p in parts {
                if let text = p["text"] as? String {
                    buf += text
                } else if let type = p["type"] as? String {
                    buf += "[\(type)]"
                }
            }
            return buf
        }
        return ""
    }

    private static func jsonString(_ value: Any) -> String {
        if let data = try? JSONSerialization.data(
            withJSONObject: value,
            options: [.fragmentsAllowed, .sortedKeys]),
           let json = String(data: data, encoding: .utf8)
        {
            return json
        }
        return "{}"
    }

    private static func renderBailingToolCall(_ call: [String: any Sendable]) -> String {
        let fn = call["function"] as? [String: any Sendable]
        let name = (fn?["name"] as? String)
            ?? (call["name"] as? String)
            ?? ""
        let rawArguments: Any = fn?["arguments"] ?? call["arguments"] ?? [String: Any]()
        let arguments: String
        if let s = rawArguments as? String {
            let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
            arguments = trimmed.isEmpty ? "{}" : trimmed
        } else {
            arguments = jsonString(rawArguments)
        }
        return "<tool_call>\n{\"name\": \(jsonString(name)), \"arguments\": \(arguments)}\n</tool_call>"
    }

    /// Read a Bool out of the additionalContext dict that swift-transformers
    /// passes through to Jinja. Defaults if absent or unparseable.
    static func bool(
        in context: [String: any Sendable]?,
        key: String,
        default def: Bool
    ) -> Bool {
        guard let ctx = context, let v = ctx[key] else { return def }
        if let b = v as? Bool { return b }
        if let n = v as? Int { return n != 0 }
        if let s = v as? String { return s.lowercased() == "true" || s == "1" }
        return def
    }
}
