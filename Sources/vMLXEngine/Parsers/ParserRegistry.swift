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
        // Qwen family aliases covering JANG capability-stamp + vLLM
        // ecosystem names + Qwen 3.6 short/long forms (2026-04-16
        // parser alias audit). `laguna` ships poolside's 33B/3B
        // agentic-coding MoE which uses Qwen2-flavored tokenizer +
        // qwen-style tool format — its silver-tier registry row
        // declares toolParser: "qwen", so route the literal "laguna"
        // family name here too. (User audit 2026-05-01: missing this
        // alias caused tool_call leak into delta.content.)
        case "qwen", "qwen3", "qwen3_5", "qwen35", "qwen3_6", "qwen36",
             "qwen3_coder", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text",
             "laguna":
            return QwenToolCallParser()
        case "llama", "llama3", "llama4":
            return LlamaToolCallParser()
        // Mistral family — include `mistral4` variant (JANG stamp uses
        // it for MLA-backed Mistral 4). Before this, Mistral 4 stamped
        // models silently lost tool parsing. `ministral3` covers the
        // Mistral-Medium-3.5 inner text model_type — silver registry
        // registers it as a separate family but it uses the same
        // mistral [TOOL_CALLS] / native tool format.
        case "mistral", "mistral3", "mistral4", "ministral3":
            return MistralToolCallParser()
        case "deepseek", "deepseek_v3", "deepseek_r1", "deepseek_v32":
            return DeepSeekToolCallParser()
        // DeepSeek V4 (Flash / Pro) — DSML envelope with ｜DSML｜-framed
        // invoke/parameter blocks. The `deepseek_v4` model_type alias
        // routes here via the silver table; clients may also stamp
        // `tool_parser: "dsml"` directly in jang_config.
        case "dsml", "deepseek_v4":
            return DSMLToolCallParser()
        case "kimi", "kimi_k2", "moonshot":
            return KimiToolCallParser()
        case "granite", "granite3", "granitemoehybrid":
            return GraniteToolCallParser()
        // Nemotron family — include `nemotron_h` (JANG-stamp family
        // name for hybrid SSM Nemotron).
        case "nemotron", "nemotron3", "nemotron_h":
            return NemotronToolCallParser()
        case "step3p5", "stepfun":
            return Step3p5ToolCallParser()
        case "xlam":
            return XlamToolCallParser()
        case "functionary", "meetkai":
            return FunctionaryToolCallParser()
        // GLM 4 family — `glm5` / `glm_moe_dsa` intentionally NOT here:
        // the JANG stamp for GLM-5.1 bundles writes `tool_parser: "deepseek"`
        // (GLM-5.1 uses DeepSeek's `[TOOL_CALLS]` JSON envelope), which
        // routes through the "deepseek" alias above. Only GLM-4 and its
        // MoE variant use the native glm47 tool parser.
        case "glm47", "glm4", "glm4_moe":
            return Glm47ToolCallParser()
        // MiniMax family — include `minimax_m2_5` (JANG-stamp for the
        // M2.5 variant; was previously only aliased as `minimax_m2`).
        case "minimax", "minimax_m2", "minimax_m2_5":
            return MiniMaxToolCallParser()
        // Gemma family — accept bare `gemma` + `gemma3n` (JANG-stamp
        // uses short form for some bundles, gemma3n is a Gemma 3n
        // variant that uses Gemma 4's hermes-style tool format).
        case "gemma", "gemma4", "gemma3", "gemma3n":
            return Gemma4ToolCallParser()
        case "native", "json":
            return NativeToolCallParser()
        default:
            return nil
        }
    }

    public static var registered: [String] {
        [
            "hermes", "qwen", "llama", "mistral", "mistral4", "deepseek",
            "dsml", "deepseek_v4", "kimi", "granite", "nemotron",
            "nemotron_h", "step3p5", "xlam", "functionary", "glm47",
            "glm4", "glm4_moe", "glm5", "minimax", "minimax_m2",
            "minimax_m2_5", "gemma", "gemma4", "native",
        ]
    }
}
