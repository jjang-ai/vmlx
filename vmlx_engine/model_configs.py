# SPDX-License-Identifier: Apache-2.0
"""
Model family configurations for vmlx-engine.

Defines configuration profiles for all supported model families including
cache types, tokenizer settings, tool parsers, and architecture hints
indexed strictly by Hugging Face `model_type`.

Must stay in sync with panel/src/main/model-config-registry.ts (TypeScript side).
"""

from .model_config_registry import ModelConfig, ModelConfigRegistry

HARMONY_CHAT_TEMPLATE = """\
{%- if tools %}
    {{- '<|start|>system<|message|>' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call>\\nYou may also use the native format: to=<function-name> code{<args-json-object>}\\nYou MUST call tools when the task requires external information or actions. Do not just describe what you would do — actually call the tool.<|end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|start|>system<|message|>' + messages[0].content + '<|end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if message.role == "user" or (message.role == "system" and not loop.first) %}
        {{- '<|start|>' + message.role + '<|message|>' + message.content + '<|end|>\\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|start|>assistant<|message|>' }}
        {%- if message.content %}
            {{- message.content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and message.content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|start|>user<|message|>' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start|>assistant<|message|>' }}
{%- endif %}
"""


def register_all(registry=None):
    if registry is None:
        registry = ModelConfigRegistry()

    existing = {c.family_name for c in registry._configs}

    def _register(config):
        if config.family_name not in existing:
            registry.register(config)
            existing.add(config.family_name)

    # ── Qwen family ──

    # Note: qwen3_5 / qwen3_5_moe model_types are shared between text and VL variants.
    # VL detection relies on config.json vision_config presence (authoritative check),
    # NOT the registry's is_mllm flag. Keep is_mllm=False here.
    _register(
        ModelConfig(
            family_name="qwen3_5",
            model_types=["qwen3_5"],
            cache_type="kv",
            eos_tokens=["<|im_end|>"],
            tool_parser="qwen",
            reasoning_parser="qwen3",
            think_in_template=True,
            is_mllm=False,
            priority=4,
        )
    )

    _register(
        ModelConfig(
            family_name="qwen3_5_moe",
            # qwen3_5_moe_text is the inner text_config.model_type for Qwen3.6-35B-A3B
            # (VLM wrapper: architectures=Qwen3_5MoeForConditionalGeneration). Without
            # registering it, the text_config disambiguator in the registry misses it
            # and falls back to default — no reasoning parser, no tool parser.
            model_types=["qwen3_5_moe", "qwen3_5_moe_text"],
            cache_type="kv",
            eos_tokens=["<|im_end|>"],
            tool_parser="qwen",
            reasoning_parser="qwen3",
            think_in_template=True,
            is_mllm=False,
            priority=4,
        )
    )

    _register(
        ModelConfig(
            family_name="qwen3",
            model_types=["qwen3"],
            cache_type="kv",
            eos_tokens=["<|im_end|>"],
            tool_parser="qwen",
            reasoning_parser="qwen3",
            think_in_template=True,
            supports_native_tools=True,
            priority=10,
        )
    )

    _register(
        ModelConfig(
            family_name="qwen3_moe",
            model_types=["qwen3_moe"],
            cache_type="kv",
            eos_tokens=["<|im_end|>"],
            tool_parser="qwen",
            reasoning_parser="qwen3",
            think_in_template=True,
            supports_native_tools=True,
            priority=5,
        )
    )

    _register(
        ModelConfig(
            family_name="qwen3_vl",
            model_types=["qwen3_vl", "qwen3_vl_moe"],
            cache_type="kv",
            eos_tokens=["<|im_end|>"],
            tool_parser="qwen",
            reasoning_parser="qwen3",
            think_in_template=True,
            is_mllm=True,
            priority=5,
        )
    )

    _register(
        ModelConfig(
            family_name="qwen3_next",
            model_types=["qwen3_next"],
            cache_type="mamba",
            eos_tokens=["<|im_end|>"],
            tool_parser="qwen",
            reasoning_parser="qwen3",
            think_in_template=True,
            priority=1,
        )
    )

    _register(
        ModelConfig(
            family_name="qwen2",
            model_types=["qwen2", "qwen2_moe", "qwen"],
            cache_type="kv",
            eos_tokens=["<|im_end|>"],
            tool_parser="qwen",
            supports_native_tools=True,
            priority=20,
        )
    )

    _register(
        ModelConfig(
            family_name="qwen2_vl",
            model_types=["qwen2_vl", "qwen2_5_vl"],
            cache_type="kv",
            eos_tokens=["<|im_end|>"],
            tool_parser="qwen",
            is_mllm=True,
            priority=10,
        )
    )

    _register(
        ModelConfig(
            family_name="qwen_mamba",
            model_types=["qwen_mamba"],
            cache_type="mamba",
            eos_tokens=["<|im_end|>"],
            tool_parser="qwen",
            priority=5,
        )
    )

    # ── Llama family ──

    _register(
        ModelConfig(
            family_name="llama4",
            model_types=["llama4"],
            cache_type="kv",
            tool_parser="llama",
            supports_native_tools=True,
            preserve_native_tool_format=True,
            priority=5,
        )
    )

    _register(
        ModelConfig(
            family_name="llama",
            model_types=["llama"],
            cache_type="kv",
            tool_parser="llama",
            supports_native_tools=True,
            preserve_native_tool_format=True,
            priority=20,
        )
    )

    # ── Mistral family ──

    _register(
        ModelConfig(
            family_name="devstral",
            model_types=["devstral"],
            cache_type="kv",
            tool_parser="mistral",
            supports_native_tools=True,
            preserve_native_tool_format=True,
            priority=5,
        )
    )

    _register(
        ModelConfig(
            family_name="codestral",
            model_types=["codestral"],
            cache_type="kv",
            tool_parser="mistral",
            supports_native_tools=True,
            preserve_native_tool_format=True,
            priority=5,
        )
    )

    _register(
        ModelConfig(
            family_name="pixtral",
            model_types=["pixtral"],
            cache_type="kv",
            tool_parser="mistral",
            supports_native_tools=True,
            preserve_native_tool_format=True,
            is_mllm=True,
            priority=5,
        )
    )

    _register(
        ModelConfig(
            family_name="mistral",
            model_types=["mistral", "mixtral"],
            cache_type="kv",
            tool_parser="mistral",
            supports_native_tools=True,
            preserve_native_tool_format=True,
            priority=20,
        )
    )

    # Mistral 4 (MLA + MoE) — original vMLX integration by Jinho Jang (eric@jangq.ai)
    # MLA attention with kv_lora_rank=256, q_lora_rank=1024 requires:
    #   - kv_b_proj split into embed_q/unembed_out (JANG loader)
    #   - bfloat16 for MoE expert overflow prevention
    #   - KV cache quantization disabled (MLA stores compressed latents)
    #   - Prefix cache head validation returns 1 (not num_key_value_heads)
    #   - reasoning_effort "none"/"high" via [MODEL_SETTINGS] template block
    #   - VLM wrapper (mistral3) with _Mistral4VLMBackbone dispatch
    # This took significant original engineering. Please credit if you adapt.
    _register(
        ModelConfig(
            family_name="mistral4",
            model_types=["mistral4"],
            cache_type="kv",
            tool_parser="mistral",
            reasoning_parser="mistral",
            think_in_template=False,
            supports_native_tools=True,
            preserve_native_tool_format=True,
            priority=30,
        )
    )

    # Mistral 3 VLM wrapper (pixtral, mistral3) — also used by Mistral 4 as VLM envelope
    # text_config.model_type lookup in registry resolves mistral3→mistral4 for MLA models
    _register(
        ModelConfig(
            family_name="mistral3",
            model_types=["mistral3"],
            cache_type="kv",
            is_mllm=True,
            tool_parser="mistral",
            reasoning_parser=None,  # Only mistral4 has reasoning; detected via text_config
            think_in_template=False,
            supports_native_tools=True,
            preserve_native_tool_format=True,
            priority=10,
        )
    )

    # ── DeepSeek family ──

    _register(
        ModelConfig(
            family_name="deepseek_vl",
            model_types=["deepseek_vl", "deepseek_vl2", "deepseek_vl_v2"],
            cache_type="kv",
            tool_parser="deepseek",
            is_mllm=True,
            priority=5,
        )
    )

    _register(
        ModelConfig(
            family_name="deepseek",
            model_types=["deepseek_v2", "deepseek_v3", "deepseek2", "deepseek"],
            cache_type="kv",
            tool_parser="deepseek",
            reasoning_parser="deepseek_r1",
            priority=20,
        )
    )

    # GLM-5.1 (glm_moe_dsa): inherits deepseek_v32 architecture (MLA + MoE).
    # Same fp32-SDPA fix applies. Uses DeepSeek R1 reasoning parser since
    # GLM-5.1 emits <think>...</think> tags (same as DeepSeek V3.2).
    _register(
        ModelConfig(
            family_name="glm5",
            model_types=["glm_moe_dsa"],
            cache_type="kv",
            tool_parser="deepseek",
            reasoning_parser="deepseek_r1",
            think_in_template=True,
            priority=20,
        )
    )

    # Kimi K2.6 (kimi_k25) — DeepseekV3 text backbone + MoonViT 27-block
    # vision tower + PatchMergerMLP projector. Text wrapper class name
    # in HuggingFace is Kimi_K25ForConditionalGeneration which registers
    # model_type="kimi_k25"; mlx_vlm's kimi_vl module handles both the
    # Kimi-VL-A3B (Moonlight) and K2.6 variants (identical vision +
    # projector architecture), so load routes through the kimi_k25 →
    # kimi_vl remap installed by jang_tools.load_jangtq_kimi_vlm.
    #
    # Tool/reasoning parsers: Kimi K2 uses TS-style tool calls (kimi_k2
    # parser in mlx_lm.tool_parsers) with <think> reasoning tags shared
    # with DeepSeek R1. Bundle is 2-bit MXTQ quantized — JANGTQ loader
    # auto-detects via .tq_packed suffix. Cache type: "kv" at engine
    # boundary (MLA is internal to DeepseekV3Attention, cache surface
    # is standard KV).
    _register(
        ModelConfig(
            family_name="kimi_k25",
            model_types=["kimi_k25"],
            cache_type="kv",
            # "kimi" is the canonical parser name in VALID_TOOL_PARSERS
            # (tests/test_model_config_registry.py); KimiToolParser registers
            # the aliases ["kimi", "kimi_k2", "moonshot"] at import so this
            # resolves to the same class as "kimi_k2".
            tool_parser="kimi",
            reasoning_parser="deepseek_r1",
            think_in_template=True,
            is_mllm=True,
            priority=20,
        )
    )

    # DeepSeek V4-Flash (deepseek_v4) — 284B total / ~13B active, MLA with
    # head_dim=512, 256 routed experts top-6 + 1 shared, sqrtsoftplus routing
    # with hash layers for first 3 blocks, mHC (Manifold Hyper-Connections)
    # hc_mult=4, attention sink per head, inverse RoPE on output, swiglu_limit=10,
    # 1M context via YaRN factor=16 from 65k, sliding window 128, compressor+
    # indexer for compressed global context. Runtime lives in
    # jang_tools.dsv4.mlx_model (~1128 LOC) and is registered into mlx_lm
    # via jang_tools.dsv4.mlx_register at load time. Not VLM.
    #
    # Three reasoning modes per research/DSV4-RUNTIME-ARCHITECTURE.md §4:
    #   - chat          (instruct, thinking suppressed via trailing </think>)
    #   - thinking      (reasoning_effort=high)
    #   - thinking max  (reasoning_effort=max, extra system hint)
    # Multi-turn: jang_config.chat.reasoning.drop_earlier_reasoning=true →
    # strip prior <think>...</think> blocks from history when building next
    # prompt. Our deepseek_r1 reasoning parser handles the <think> tags;
    # think_in_template=True so the empty-tag suppression trick works.
    #
    # Tool calls use DSML format: <｜DSML｜invoke name="fn">...<｜DSML｜parameter
    # name="p" string="true">val</｜DSML｜parameter></｜DSML｜invoke>.
    # Parser name "dsml" registered in tool_parsers (see DSV4 test_chat.py).
    #
    # Cache type "kv" at engine boundary. Internally DSV4 uses a custom
    # DeepseekV4Cache wrapping a RotatingKVCache(max_size=sliding_window=128,
    # keep=0) for local attention + compressor/indexer state buffers for
    # cross-window pooling. Loader constructs it via make_cache(); engine
    # just drives update_and_fetch like standard KV.
    #
    # Bundle formats (per §5 cheat sheet): JANG_2L (107 GB), JANGTQ2 (74 GB
    # recommended prod default), JANGTQ4 (173 GB highest fidelity), JANG4
    # (173 GB uniform 4-bit). Do NOT use JANGTQ4-HP (mxfp4+bf16 unstable).
    _register(
        ModelConfig(
            family_name="deepseek_v4",
            model_types=["deepseek_v4"],
            cache_type="kv",
            tool_parser="dsml",
            reasoning_parser="deepseek_r1",
            think_in_template=True,
            priority=20,
        )
    )

    # ── GLM family (CRITICAL: different reasoning parsers per variant) ──

    # GPT-OSS: Harmony <|channel|> protocol reasoning
    _register(
        ModelConfig(
            family_name="gpt_oss",
            model_types=["gpt_oss"],
            cache_type="kv",
            tool_parser="glm47",
            reasoning_parser="openai_gptoss",
            chat_template_custom=HARMONY_CHAT_TEMPLATE,
            preserve_native_tool_format=True,
            priority=3,
        )
    )

    # GLM-4.7 Flash (MoE): also uses Harmony/openai_gptoss reasoning, NOT deepseek_r1
    _register(
        ModelConfig(
            family_name="glm4_moe",
            model_types=["glm4_moe", "glm4_moe_lite"],
            cache_type="kv",
            tool_parser="glm47",
            reasoning_parser="openai_gptoss",
            chat_template_custom=HARMONY_CHAT_TEMPLATE,
            preserve_native_tool_format=True,
            priority=3,
        )
    )

    # GLM-Z1: reasoning model using Harmony channel protocol (same as GPT-OSS/GLM-4.7).
    # Uses openai_gptoss parser for <|channel|>analysis reasoning, NOT deepseek_r1.
    # The Harmony template does not inject <think>, so think_in_template must be False.
    # (Shares model_type "glm4" with base GLM-4, disambiguated by name in lookup())
    _register(
        ModelConfig(
            family_name="glm_z1",
            model_types=[],  # No unique model_type — disambiguated by name in lookup()
            cache_type="kv",
            tool_parser="glm47",
            reasoning_parser="openai_gptoss",
            chat_template_custom=HARMONY_CHAT_TEMPLATE,
            preserve_native_tool_format=True,
            priority=2,
        )
    )

    # GLM-4 / ChatGLM: base model (tools only, no reasoning)
    _register(
        ModelConfig(
            family_name="chatglm",
            model_types=["chatglm", "glm4", "glm"],
            cache_type="kv",
            tool_parser="glm47",
            chat_template_custom=HARMONY_CHAT_TEMPLATE,
            preserve_native_tool_format=True,
            priority=20,
        )
    )

    # ── StepFun family ──

    # Step-1V is a vision-language model
    _register(
        ModelConfig(
            family_name="step_vl",
            model_types=["step1v"],
            cache_type="kv",
            tool_parser="step3p5",
            reasoning_parser="qwen3",
            think_in_template=True,
            is_mllm=True,
            priority=5,
        )
    )

    _register(
        ModelConfig(
            family_name="step",
            model_types=["step3p5", "step"],
            cache_type="kv",
            tool_parser="step3p5",
            reasoning_parser="qwen3",
            think_in_template=True,
            priority=10,
        )
    )

    # ── Gemma family ──

    _register(
        ModelConfig(
            family_name="gemma4",
            model_types=["gemma4"],
            cache_type="kv",
            tool_parser="gemma4",
            reasoning_parser="gemma4",  # qwen3 parser is more robust with quantized models
            eos_tokens=["<eos>", "<turn|>"],
            special_tokens_to_clean=["<turn|>", "<|turn>", "<|channel>", "<channel|>"],
            is_mllm=True,
            architecture_hints={"inject_pixel_values": True},
            priority=5,
        )
    )

    _register(
        ModelConfig(
            family_name="gemma4_text",
            model_types=["gemma4_text"],
            cache_type="kv",
            tool_parser="gemma4",
            reasoning_parser="gemma4",  # qwen3 parser is more robust with quantized models
            eos_tokens=["<eos>", "<turn|>"],
            special_tokens_to_clean=["<turn|>", "<|turn>", "<|channel>", "<channel|>"],
            priority=4,
        )
    )

    # Gemma 3 / Gemma 3n — Google's documented function-calling format is
    # a `tool_code` Python-code-block, NOT hermes JSON. Was previously
    # misconfigured with tool_parser="hermes" which silently failed to
    # extract calls, leaving clients with raw `` ```tool_code\nfunc(..)\n``` ``
    # in the content. Also needs `<end_of_turn>` in eos_tokens —
    # without it Gemma 3/3n enters an infinite loop emitting the
    # token after the first tool call.
    _register(
        ModelConfig(
            family_name="gemma3",
            model_types=["gemma3"],
            cache_type="kv",
            tool_parser="gemma3",
            reasoning_parser=None,
            eos_tokens=["<end_of_turn>", "<eos>"],
            is_mllm=True,
            architecture_hints={"inject_pixel_values": True},
            priority=10,
        )
    )

    _register(
        ModelConfig(
            family_name="gemma3_text",
            model_types=["gemma3_text"],
            cache_type="kv",
            tool_parser="gemma3",
            reasoning_parser=None,
            eos_tokens=["<end_of_turn>", "<eos>"],
            priority=8,
        )
    )

    # Gemma 3n (E2B / E4B) — tiny text+vision+audio Gemma. Uses the same
    # `tool_code` block format as Gemma 3. Previously unregistered,
    # so fresh users of gemma-3n-E2B-it-4bit etc. got family="unknown"
    # and no tool/reasoning parser auto-applied.
    _register(
        ModelConfig(
            family_name="gemma3n",
            model_types=["gemma3n"],
            cache_type="kv",
            tool_parser="gemma3",
            reasoning_parser=None,
            eos_tokens=["<end_of_turn>", "<eos>"],
            is_mllm=True,
            priority=10,
        )
    )
    _register(
        ModelConfig(
            family_name="gemma3n_text",
            model_types=["gemma3n_text"],
            cache_type="kv",
            tool_parser="gemma3",
            reasoning_parser=None,
            eos_tokens=["<end_of_turn>", "<eos>"],
            priority=8,
        )
    )

    _register(
        ModelConfig(
            family_name="gemma",
            model_types=["gemma", "gemma2"],
            cache_type="kv",
            priority=30,
        )
    )

    # MedGemma: multimodal medical model (uses gemma2 model_type,
    # disambiguated by name matching in lookup())
    _register(
        ModelConfig(
            family_name="medgemma",
            model_types=[],  # No unique model_type — uses gemma2, disambiguated by name
            cache_type="kv",
            is_mllm=True,
            architecture_hints={"inject_pixel_values": True},
            priority=3,
        )
    )

    _register(
        ModelConfig(
            family_name="paligemma",
            model_types=["paligemma", "paligemma2"],
            cache_type="kv",
            is_mllm=True,
            priority=15,
        )
    )

    # ── Phi family ──

    _register(
        ModelConfig(
            family_name="phi4_reasoning",
            model_types=["phi4_reasoning"],
            cache_type="kv",
            tool_parser="hermes",
            reasoning_parser="deepseek_r1",
            priority=2,
        )
    )

    _register(
        ModelConfig(
            family_name="phi4_multimodal",
            model_types=["phi4mm"],
            cache_type="kv",
            is_mllm=True,
            priority=2,
        )
    )

    _register(
        ModelConfig(
            family_name="phi4",
            model_types=["phi4", "phi4flash"],
            cache_type="kv",
            tool_parser="hermes",
            priority=10,
        )
    )

    _register(
        ModelConfig(
            family_name="phi3_v",
            model_types=["phi3v"],
            cache_type="kv",
            tool_parser="llama",
            is_mllm=True,
            priority=8,
        )
    )

    _register(
        ModelConfig(
            family_name="phi3",
            model_types=["phi3", "phi3small", "phi"],
            cache_type="kv",
            tool_parser="llama",
            priority=20,
        )
    )

    # ── Hermes (NousResearch) ──

    _register(
        ModelConfig(
            family_name="hermes",
            model_types=["hermes"],
            cache_type="kv",
            tool_parser="hermes",
            priority=30,
        )
    )

    # ── Nemotron (NVIDIA) ──

    _register(
        ModelConfig(
            family_name="nemotron",
            model_types=["nemotron"],
            cache_type="kv",
            eos_tokens=["<|im_end|>"],
            tool_parser="nemotron",
            reasoning_parser="deepseek_r1",
            think_in_template=True,
            tokenizer_fallback=True,
            priority=10,
        )
    )

    _register(
        ModelConfig(
            family_name="nemotron_h",
            model_types=["nemotron_h"],
            cache_type="hybrid",
            eos_tokens=["<|im_end|>"],
            tool_parser="nemotron",
            reasoning_parser="deepseek_r1",
            think_in_template=True,
            tokenizer_fallback=True,
            priority=10,
        )
    )

    # ── Cohere ──

    _register(
        ModelConfig(
            family_name="cohere",
            model_types=["cohere", "cohere2"],
            cache_type="kv",
            priority=20,
        )
    )

    # ── IBM Granite ──

    _register(
        ModelConfig(
            family_name="granite",
            model_types=["granite", "granite_moe"],
            cache_type="kv",
            tool_parser="granite",
            priority=20,
        )
    )

    # ── MiniMax ──

    _register(
        ModelConfig(
            family_name="minimax",
            model_types=["minimax", "minimax_m2", "minimax_m2_5"],
            cache_type="kv",
            tool_parser="minimax",
            reasoning_parser="qwen3",
            think_in_template=True,
            priority=20,
        )
    )

    # ── xLAM (Salesforce) — no unique model_type, usually Llama-based ──

    # ── Kimi/Moonshot ──

    _register(
        ModelConfig(
            family_name="kimi",
            model_types=["kimi_k2"],
            cache_type="kv",
            tool_parser="kimi",
            reasoning_parser="deepseek_r1",
            priority=20,
        )
    )

    # ── InternLM ──

    _register(
        ModelConfig(
            family_name="internlm",
            model_types=["internlm", "internlm2", "internlm3"],
            cache_type="kv",
            priority=20,
        )
    )

    # ── EXAONE ──

    _register(
        ModelConfig(
            family_name="exaone",
            model_types=["exaone", "exaone3"],
            cache_type="kv",
            priority=20,
        )
    )

    # ── OLMo ──

    _register(
        ModelConfig(
            family_name="olmo",
            model_types=["olmo", "olmo2"],
            cache_type="kv",
            priority=20,
        )
    )

    # ── VLM / MLLM models ──

    _register(
        ModelConfig(
            family_name="llava",
            model_types=["llava", "llava_next"],
            cache_type="kv",
            is_mllm=True,
            priority=20,
        )
    )

    _register(
        ModelConfig(
            family_name="idefics",
            model_types=["idefics2", "idefics3"],
            cache_type="kv",
            is_mllm=True,
            priority=15,
        )
    )

    _register(
        ModelConfig(
            family_name="cogvlm",
            model_types=["cogvlm", "cogvlm2"],
            cache_type="kv",
            is_mllm=True,
            priority=20,
        )
    )

    _register(
        ModelConfig(
            family_name="florence",
            model_types=["florence2"],
            cache_type="kv",
            is_mllm=True,
            priority=20,
        )
    )

    _register(
        ModelConfig(
            family_name="got_ocr",
            model_types=["got_ocr2"],
            cache_type="kv",
            is_mllm=True,
            priority=15,
            description="GOT-OCR2 (General OCR Theory) — document/scene OCR",
        )
    )

    _register(
        ModelConfig(
            family_name="molmo",
            model_types=["molmo"],
            cache_type="kv",
            is_mllm=True,
            priority=20,
        )
    )

    _register(
        ModelConfig(
            family_name="minicpm_v",
            model_types=["minicpmv"],
            cache_type="kv",
            is_mllm=True,
            priority=20,
        )
    )

    _register(
        ModelConfig(
            family_name="smolvlm",
            model_types=["smolvlm"],
            cache_type="kv",
            is_mllm=True,
            priority=20,
        )
    )

    _register(
        ModelConfig(
            family_name="internvl",
            model_types=["internvl_chat"],
            cache_type="kv",
            is_mllm=True,
            priority=15,
        )
    )

    _register(
        ModelConfig(
            family_name="internlm_xcomposer",
            model_types=["internlm_xcomposer2"],
            cache_type="kv",
            is_mllm=True,
            priority=8,
        )
    )

    # ── SSM / Mamba ──

    _register(
        ModelConfig(
            family_name="falcon_mamba",
            model_types=["falcon_mamba"],
            cache_type="mamba",
            priority=5,
        )
    )

    _register(
        ModelConfig(
            family_name="mamba",
            model_types=["mamba", "mamba2", "codestral_mamba"],
            cache_type="mamba",
            priority=30,
        )
    )

    _register(
        ModelConfig(
            family_name="rwkv",
            model_types=["rwkv", "rwkv5", "rwkv6"],
            cache_type="mamba",
            priority=30,
        )
    )

    # ── Hybrid SSM ──

    _register(
        ModelConfig(
            family_name="jamba",
            model_types=["jamba"],
            cache_type="hybrid",
            priority=10,
        )
    )

    # ── Others ──

    _register(
        ModelConfig(
            family_name="starcoder",
            model_types=["starcoder2"],
            cache_type="kv",
            priority=30,
        )
    )

    _register(
        ModelConfig(
            family_name="stablelm",
            model_types=["stablelm"],
            cache_type="kv",
            priority=30,
        )
    )

    _register(
        ModelConfig(
            family_name="baichuan",
            model_types=["baichuan"],
            cache_type="kv",
            priority=30,
        )
    )
