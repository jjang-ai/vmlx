// SPDX-License-Identifier: Apache-2.0
// Name → parser factory mapping. Mirrors the `reasoning_parser` / `tool_parser`
// strings used in vmlx_engine/model_configs.py so Swift callers can look up
// parsers by the same keys the Python engine uses.

import Foundation

public enum ReasoningParserRegistry {
    public static func make(_ name: String) -> ReasoningParser? {
        switch name.lowercased() {
        case "qwen3":                         return Qwen3ReasoningParser()
        case "deepseek_r1":                   return DeepSeekR1ReasoningParser()
        case "mistral":                       return MistralReasoningParser()
        case "gemma4":                        return Gemma4ReasoningParser()
        case "openai_gptoss", "gptoss", "gpt_oss",
             "harmony", "glm47_flash":        return GptOssReasoningParser()
        // v1.3.53 parity: `auto` is the CLI / config shorthand for
        // "pick a reasonable default when the model_type isn't in
        // the allowlist." The safe fallback is DeepSeek R1's
        // `<think>...</think>` parser — every allowlisted thinking
        // model either uses DeepSeek R1 syntax or a superset (Qwen3
        // adds `<think/>` empty-tag support but the DeepSeek parser
        // handles the common case). Before this round, `auto` fell
        // through to `default` and returned nil → reasoning blobs
        // landed in `content` instead of `reasoning_content`.
        case "auto":                          return DeepSeekR1ReasoningParser()
        default:                              return nil
        }
    }

    public static var registered: [String] {
        ["qwen3", "deepseek_r1", "mistral", "gemma4", "openai_gptoss", "auto"]
    }
}

public enum ToolCallParserRegistry {
    public static func make(_ name: String) -> ToolCallParser? {
        switch name.lowercased() {
        case "hermes", "nous":
            return HermesToolCallParser()
        case "qwen", "qwen3":
            return QwenToolCallParser()
        case "llama", "llama3", "llama4":
            return LlamaToolCallParser()
        case "mistral":
            return MistralToolCallParser()
        case "deepseek", "deepseek_v3", "deepseek_r1":
            return DeepSeekToolCallParser()
        case "kimi", "kimi_k2", "moonshot":
            return KimiToolCallParser()
        case "granite", "granite3":
            return GraniteToolCallParser()
        case "nemotron", "nemotron3":
            return NemotronToolCallParser()
        case "step3p5", "stepfun":
            return Step3p5ToolCallParser()
        case "xlam":
            return XlamToolCallParser()
        case "functionary", "meetkai":
            return FunctionaryToolCallParser()
        case "glm47", "glm4":
            return Glm47ToolCallParser()
        case "minimax", "minimax_m2":
            return MiniMaxToolCallParser()
        case "gemma4":
            return Gemma4ToolCallParser()
        case "native", "json":
            return NativeToolCallParser()
        default:
            return nil
        }
    }

    public static var registered: [String] {
        [
            "hermes", "qwen", "llama", "mistral", "deepseek", "kimi",
            "granite", "nemotron", "step3p5", "xlam", "functionary",
            "glm47", "minimax", "gemma4", "native",
        ]
    }
}
