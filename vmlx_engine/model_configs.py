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
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags. ALWAYS use this exact format:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call>\\nDo NOT use any other tool call format such as to=name code{} syntax.<|end|>\\n" }}
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
    _register(ModelConfig(
        family_name="qwen3_5",
        model_types=["qwen3_5"],
        cache_type="kv",
        eos_tokens=["<|im_end|>"],
        tool_parser="qwen",
        reasoning_parser="qwen3",
        think_in_template=True,
        is_mllm=False,
        priority=4,
    ))

    _register(ModelConfig(
        family_name="qwen3_5_moe",
        model_types=["qwen3_5_moe"],
        cache_type="kv",
        eos_tokens=["<|im_end|>"],
        tool_parser="qwen",
        reasoning_parser="qwen3",
        think_in_template=True,
        is_mllm=False,
        priority=4,
    ))

    _register(ModelConfig(
        family_name="qwen3",
        model_types=["qwen3"],
        cache_type="kv",
        eos_tokens=["<|im_end|>"],
        tool_parser="qwen",
        reasoning_parser="qwen3",
        think_in_template=True,
        supports_native_tools=True,
        priority=10,
    ))

    _register(ModelConfig(
        family_name="qwen3_moe",
        model_types=["qwen3_moe"],
        cache_type="kv",
        eos_tokens=["<|im_end|>"],
        tool_parser="qwen",
        reasoning_parser="qwen3",
        think_in_template=True,
        supports_native_tools=True,
        priority=5,
    ))

    _register(ModelConfig(
        family_name="qwen3_vl",
        model_types=["qwen3_vl", "qwen3_vl_moe"],
        cache_type="kv",
        eos_tokens=["<|im_end|>"],
        tool_parser="qwen",
        reasoning_parser="qwen3",
        think_in_template=True,
        is_mllm=True,
        priority=5,
    ))

    _register(ModelConfig(
        family_name="qwen3_next",
        model_types=["qwen3_next"],
        cache_type="mamba",
        eos_tokens=["<|im_end|>"],
        tool_parser="nemotron",
        priority=1,
    ))

    _register(ModelConfig(
        family_name="qwen2",
        model_types=["qwen2", "qwen2_moe", "qwen"],
        cache_type="kv",
        eos_tokens=["<|im_end|>"],
        tool_parser="qwen",
        supports_native_tools=True,
        priority=20,
    ))

    _register(ModelConfig(
        family_name="qwen2_vl",
        model_types=["qwen2_vl", "qwen2_5_vl"],
        cache_type="kv",
        eos_tokens=["<|im_end|>"],
        tool_parser="qwen",
        is_mllm=True,
        priority=10,
    ))

    _register(ModelConfig(
        family_name="qwen_mamba",
        model_types=["qwen_mamba"],
        cache_type="mamba",
        eos_tokens=["<|im_end|>"],
        tool_parser="qwen",
        priority=5,
    ))

    # ── Llama family ──

    _register(ModelConfig(
        family_name="llama4",
        model_types=["llama4"],
        cache_type="kv",
        tool_parser="llama",
        supports_native_tools=True,
        preserve_native_tool_format=True,
        priority=5,
    ))

    _register(ModelConfig(
        family_name="llama",
        model_types=["llama"],
        cache_type="kv",
        tool_parser="llama",
        supports_native_tools=True,
        preserve_native_tool_format=True,
        priority=20,
    ))

    # ── Mistral family ──

    _register(ModelConfig(
        family_name="devstral",
        model_types=["devstral"],
        cache_type="kv",
        tool_parser="mistral",
        supports_native_tools=True,
        preserve_native_tool_format=True,
        priority=5,
    ))

    _register(ModelConfig(
        family_name="codestral",
        model_types=["codestral"],
        cache_type="kv",
        tool_parser="mistral",
        supports_native_tools=True,
        preserve_native_tool_format=True,
        priority=5,
    ))

    _register(ModelConfig(
        family_name="pixtral",
        model_types=["pixtral"],
        cache_type="kv",
        tool_parser="mistral",
        supports_native_tools=True,
        preserve_native_tool_format=True,
        is_mllm=True,
        priority=5,
    ))

    _register(ModelConfig(
        family_name="mistral",
        model_types=["mistral", "mixtral"],
        cache_type="kv",
        tool_parser="mistral",
        supports_native_tools=True,
        preserve_native_tool_format=True,
        priority=20,
    ))

    # ── DeepSeek family ──

    _register(ModelConfig(
        family_name="deepseek_vl",
        model_types=["deepseek_vl", "deepseek_vl2", "deepseek_vl_v2"],
        cache_type="kv",
        tool_parser="deepseek",
        is_mllm=True,
        priority=5,
    ))

    _register(ModelConfig(
        family_name="deepseek",
        model_types=["deepseek_v2", "deepseek_v3", "deepseek2", "deepseek"],
        cache_type="kv",
        tool_parser="deepseek",
        reasoning_parser="deepseek_r1",
        priority=20,
    ))

    # ── GLM family (CRITICAL: different reasoning parsers per variant) ──

    # GPT-OSS: Harmony <|channel|> protocol reasoning
    _register(ModelConfig(
        family_name="gpt_oss",
        model_types=["gpt_oss"],
        cache_type="kv",
        tool_parser="glm47",
        reasoning_parser="openai_gptoss",
        chat_template_custom=HARMONY_CHAT_TEMPLATE,
        priority=3,
    ))

    # GLM-4.7 Flash (MoE): also uses Harmony/openai_gptoss reasoning, NOT deepseek_r1
    _register(ModelConfig(
        family_name="glm4_moe",
        model_types=["glm4_moe", "glm4_moe_lite"],
        cache_type="kv",
        tool_parser="glm47",
        reasoning_parser="openai_gptoss",
        chat_template_custom=HARMONY_CHAT_TEMPLATE,
        priority=3,
    ))

    # GLM-Z1: reasoning model using deepseek_r1 (shares model_type "glm4" with base GLM-4,
    # but detected by name matching in lookup() — see model_config_registry.py)
    _register(ModelConfig(
        family_name="glm_z1",
        model_types=[],  # No unique model_type — disambiguated by name in lookup()
        cache_type="kv",
        tool_parser="glm47",
        reasoning_parser="deepseek_r1",
        think_in_template=True,
        chat_template_custom=HARMONY_CHAT_TEMPLATE,
        priority=2,
    ))

    # GLM-4 / ChatGLM: base model (tools only, no reasoning)
    _register(ModelConfig(
        family_name="chatglm",
        model_types=["chatglm", "glm4", "glm"],
        cache_type="kv",
        tool_parser="glm47",
        chat_template_custom=HARMONY_CHAT_TEMPLATE,
        priority=20,
    ))

    # ── StepFun family ──

    # Step-1V is a vision-language model
    _register(ModelConfig(
        family_name="step_vl",
        model_types=["step1v"],
        cache_type="kv",
        tool_parser="step3p5",
        reasoning_parser="qwen3",
        think_in_template=True,
        is_mllm=True,
        priority=5,
    ))

    _register(ModelConfig(
        family_name="step",
        model_types=["step3p5", "step"],
        cache_type="kv",
        tool_parser="step3p5",
        reasoning_parser="qwen3",
        think_in_template=True,
        priority=10,
    ))

    # ── Gemma family ──

    _register(ModelConfig(
        family_name="gemma3",
        model_types=["gemma3"],
        cache_type="kv",
        tool_parser="hermes",
        reasoning_parser="deepseek_r1",
        is_mllm=True,
        priority=10,
    ))

    _register(ModelConfig(
        family_name="gemma3_text",
        model_types=["gemma3_text"],
        cache_type="kv",
        tool_parser="hermes",
        reasoning_parser="deepseek_r1",
        priority=8,
    ))

    _register(ModelConfig(
        family_name="gemma",
        model_types=["gemma", "gemma2"],
        cache_type="kv",
        priority=30,
    ))

    # MedGemma: multimodal medical model (uses gemma2 model_type,
    # disambiguated by name matching in lookup())
    _register(ModelConfig(
        family_name="medgemma",
        model_types=[],  # No unique model_type — uses gemma2, disambiguated by name
        cache_type="kv",
        is_mllm=True,
        priority=3,
    ))

    _register(ModelConfig(
        family_name="paligemma",
        model_types=["paligemma", "paligemma2"],
        cache_type="kv",
        is_mllm=True,
        priority=15,
    ))

    # ── Phi family ──

    _register(ModelConfig(
        family_name="phi4_reasoning",
        model_types=["phi4_reasoning"],
        cache_type="kv",
        tool_parser="hermes",
        reasoning_parser="deepseek_r1",
        priority=2,
    ))

    _register(ModelConfig(
        family_name="phi4_multimodal",
        model_types=["phi4mm"],
        cache_type="kv",
        is_mllm=True,
        priority=2,
    ))

    _register(ModelConfig(
        family_name="phi4",
        model_types=["phi4", "phi4flash"],
        cache_type="kv",
        tool_parser="hermes",
        priority=10,
    ))

    _register(ModelConfig(
        family_name="phi3_v",
        model_types=["phi3v"],
        cache_type="kv",
        tool_parser="llama",
        is_mllm=True,
        priority=8,
    ))

    _register(ModelConfig(
        family_name="phi3",
        model_types=["phi3", "phi3small", "phi"],
        cache_type="kv",
        tool_parser="llama",
        priority=20,
    ))

    # ── Hermes (NousResearch) ──

    _register(ModelConfig(
        family_name="hermes",
        model_types=["hermes"],
        cache_type="kv",
        tool_parser="hermes",
        priority=30,
    ))

    # ── Nemotron (NVIDIA) ──

    _register(ModelConfig(
        family_name="nemotron",
        model_types=["nemotron", "nemotron_h"],
        cache_type="hybrid",
        tool_parser="nemotron",
        reasoning_parser="deepseek_r1",
        tokenizer_fallback=True,
        priority=10,
    ))

    # ── Cohere ──

    _register(ModelConfig(
        family_name="cohere",
        model_types=["cohere", "cohere2"],
        cache_type="kv",
        priority=20,
    ))

    # ── IBM Granite ──

    _register(ModelConfig(
        family_name="granite",
        model_types=["granite", "granite_moe"],
        cache_type="kv",
        tool_parser="granite",
        priority=20,
    ))

    # ── MiniMax ──

    _register(ModelConfig(
        family_name="minimax",
        model_types=["minimax", "minimax_m2", "minimax_m2_5"],
        cache_type="kv",
        tool_parser="minimax",
        reasoning_parser="qwen3",
        think_in_template=True,
        priority=20,
    ))

    # ── xLAM (Salesforce) — no unique model_type, usually Llama-based ──

    # ── Kimi/Moonshot ──

    _register(ModelConfig(
        family_name="kimi",
        model_types=["kimi_k2"],
        cache_type="kv",
        tool_parser="kimi",
        priority=20,
    ))

    # ── InternLM ──

    _register(ModelConfig(
        family_name="internlm",
        model_types=["internlm", "internlm2", "internlm3"],
        cache_type="kv",
        priority=20,
    ))

    # ── EXAONE ──

    _register(ModelConfig(
        family_name="exaone",
        model_types=["exaone", "exaone3"],
        cache_type="kv",
        priority=20,
    ))

    # ── OLMo ──

    _register(ModelConfig(
        family_name="olmo",
        model_types=["olmo", "olmo2"],
        cache_type="kv",
        priority=20,
    ))

    # ── VLM / MLLM models ──

    _register(ModelConfig(
        family_name="llava",
        model_types=["llava", "llava_next"],
        cache_type="kv",
        is_mllm=True,
        priority=20,
    ))

    _register(ModelConfig(
        family_name="idefics",
        model_types=["idefics2", "idefics3"],
        cache_type="kv",
        is_mllm=True,
        priority=15,
    ))

    _register(ModelConfig(
        family_name="cogvlm",
        model_types=["cogvlm", "cogvlm2"],
        cache_type="kv",
        is_mllm=True,
        priority=20,
    ))

    _register(ModelConfig(
        family_name="florence",
        model_types=["florence2"],
        cache_type="kv",
        is_mllm=True,
        priority=20,
    ))

    _register(ModelConfig(
        family_name="molmo",
        model_types=["molmo"],
        cache_type="kv",
        is_mllm=True,
        priority=20,
    ))

    _register(ModelConfig(
        family_name="minicpm_v",
        model_types=["minicpmv"],
        cache_type="kv",
        is_mllm=True,
        priority=20,
    ))

    _register(ModelConfig(
        family_name="smolvlm",
        model_types=["smolvlm"],
        cache_type="kv",
        is_mllm=True,
        priority=20,
    ))

    _register(ModelConfig(
        family_name="internvl",
        model_types=["internvl_chat"],
        cache_type="kv",
        is_mllm=True,
        priority=15,
    ))

    _register(ModelConfig(
        family_name="internlm_xcomposer",
        model_types=["internlm_xcomposer2"],
        cache_type="kv",
        is_mllm=True,
        priority=8,
    ))

    # ── SSM / Mamba ──

    _register(ModelConfig(
        family_name="falcon_mamba",
        model_types=["falcon_mamba"],
        cache_type="mamba",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="mamba",
        model_types=["mamba", "mamba2", "codestral_mamba"],
        cache_type="mamba",
        priority=30,
    ))

    _register(ModelConfig(
        family_name="rwkv",
        model_types=["rwkv", "rwkv5", "rwkv6"],
        cache_type="mamba",
        priority=30,
    ))

    # ── Hybrid SSM ──

    _register(ModelConfig(
        family_name="jamba",
        model_types=["jamba"],
        cache_type="hybrid",
        priority=10,
    ))

    # ── Others ──

    _register(ModelConfig(
        family_name="starcoder",
        model_types=["starcoder2"],
        cache_type="kv",
        priority=30,
    ))

    _register(ModelConfig(
        family_name="stablelm",
        model_types=["stablelm"],
        cache_type="kv",
        priority=30,
    ))

    _register(ModelConfig(
        family_name="baichuan",
        model_types=["baichuan"],
        cache_type="kv",
        priority=30,
    ))
