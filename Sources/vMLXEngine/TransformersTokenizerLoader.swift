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
        do {
            let upstream = try await AutoTokenizer.from(modelFolder: directory)
            return TransformersTokenizerBridge(upstream)
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
            let shadow = try Self.makeShadowWithPatchedTokenizerClass(
                sourceDir: directory,
                newClass: Self.fallbackTokenizerClass(for: name, in: directory))
            let upstream = try await AutoTokenizer.from(modelFolder: shadow)
            return TransformersTokenizerBridge(upstream)
        }
    }

    /// Pick a safe tokenizer class for a repo whose `tokenizer_config.json`
    /// carries an unrecognized name. Prefer Qwen2Tokenizer for Qwen-family
    /// models (identified by vocab size + `im_start`/`im_end` tokens) and
    /// generic PreTrainedTokenizer otherwise.
    static func fallbackTokenizerClass(for original: String, in directory: URL) -> String {
        // Probe config.json model_type if present.
        if let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            let mt = (obj["model_type"] as? String)
                ?? ((obj["text_config"] as? [String: Any])?["model_type"] as? String)
                ?? ""
            if mt.contains("qwen") { return "Qwen2Tokenizer" }
            if mt.contains("llama") { return "LlamaTokenizer" }
            if mt.contains("gemma") { return "GemmaTokenizer" }
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

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
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
        do {
            if let template = overrideTemplate {
                return try upstream.applyChatTemplate(
                    messages: messages,
                    chatTemplate: .literal(template),
                    addGenerationPrompt: ChatTemplateFallback.bool(
                        in: cleanContext,
                        key: "add_generation_prompt",
                        default: true),
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
            #if DEBUG
            print("[vMLX] applyChatTemplate fallback: upstream Jinja failed with \(error). Using built-in ChatML renderer.")
            #endif
            let addGen = ChatTemplateFallback.bool(
                in: additionalContext,
                key: "add_generation_prompt",
                default: true)
            let rendered = ChatTemplateFallback.render(
                messages: messages,
                addGenerationPrompt: addGen,
                bosToken: upstream.bosToken,
                tokenExists: { upstream.convertTokenToId($0) != nil })
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
        tokenExists: (String) -> Bool
    ) -> String {
        if tokenExists("<|turn>") && tokenExists("<turn|>") {
            return renderGemma4(
                messages: messages,
                addGenerationPrompt: addGenerationPrompt,
                bosToken: bosToken)
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

    /// Gemma 4 format. Roles: `user`, `model` (not `assistant`). System
    /// messages are prepended to the first user turn — Gemma 4 has no
    /// system role in the turn grammar, only a per-template `<|turn>system`
    /// header that the HF chat_template synthesizes.
    static func renderGemma4(
        messages: [[String: any Sendable]],
        addGenerationPrompt: Bool,
        bosToken: String?
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
