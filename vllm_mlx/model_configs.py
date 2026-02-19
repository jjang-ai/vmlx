# SPDX-License-Identifier: Apache-2.0
"""
Model family configurations for vllm-mlx.

Defines configuration profiles for all supported model families including
cache types, tokenizer settings, tool parsers, and architecture hints.

All configurations are auto-registered on import via register_all().
"""

from .model_config_registry import ModelConfig, ModelConfigRegistry

# =============================================================================
# Harmony Protocol Chat Template
# =============================================================================
# GPT-OSS and GLM-4.7 Flash models use the Harmony protocol with special tokens:
#   <|start|> (200006), <|end|> (200007), <|channel|> (200005), <|message|> (200008)
# The shipped chat_template.jinja uses <|im_start|>/<|im_end|> which are NOT in
# the tokenizer vocabulary, causing them to shatter into 5-7 sub-tokens. This
# template uses the correct Harmony tokens that the model was trained on.
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
    """Register all model configs. Safe to call multiple times."""
    if registry is None:
        registry = ModelConfigRegistry()

    existing = {c.family_name for c in registry._configs}

    def _register(config):
        if config.family_name not in existing:
            registry.register(config)
            existing.add(config.family_name)

    # =========================================================================
    # Qwen Family
    # =========================================================================

    _register(ModelConfig(
        family_name="qwen3-next",
        pattern=r"(?i)qwen[\-_.]?3[\-_.]?(?:coder[\-_.]?)?next",
        cache_type="mamba",
        eos_tokens=["<|im_end|>"],
        tool_parser="qwen",
        reasoning_parser="qwen3",
        think_in_template=True,
        supports_native_tools=True,
        description="Qwen 3 Coder Next (hybrid Mamba architecture)",
        priority=1,
    ))

    _register(ModelConfig(
        family_name="qwen3-vl",
        pattern=r"(?i)qwen[\-_.]?3.*VL",
        cache_type="kv",
        eos_tokens=["<|im_end|>"],
        tool_parser="qwen",
        reasoning_parser="qwen3",
        think_in_template=True,
        is_mllm=True,
        description="Qwen 3 Vision-Language models",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="qwen3-moe",
        pattern=r"(?i)qwen[\-_.]?3.*(?:moe|MoE)",
        cache_type="kv",
        eos_tokens=["<|im_end|>"],
        tool_parser="qwen",
        reasoning_parser="qwen3",
        think_in_template=True,
        supports_native_tools=True,
        description="Qwen 3 MoE models",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="qwen3",
        pattern=r"(?i)qwen[\-_.]?3",
        cache_type="kv",
        eos_tokens=["<|im_end|>"],
        tool_parser="qwen",
        reasoning_parser="qwen3",
        think_in_template=True,
        supports_native_tools=True,
        description="Qwen 3 series",
        priority=10,
    ))

    _register(ModelConfig(
        family_name="qwen2-vl",
        pattern=r"(?i)qwen[\-_.]?2.*VL",
        cache_type="kv",
        tool_parser="qwen",
        is_mllm=True,
        description="Qwen 2 Vision-Language models",
        priority=10,
    ))

    _register(ModelConfig(
        family_name="qwen2",
        pattern=r"(?i)qwen[\-_.]?2",
        cache_type="kv",
        tool_parser="qwen",
        description="Qwen 2 series",
        priority=20,
    ))

    _register(ModelConfig(
        family_name="qwen-mamba",
        pattern=r"(?i)qwen.*mamba",
        cache_type="mamba",
        tool_parser="qwen",
        description="Qwen Mamba variant",
        priority=5,
    ))

    # =========================================================================
    # Llama Family
    # =========================================================================

    _register(ModelConfig(
        family_name="llama4",
        pattern=r"(?i)llama[\-_.]?4",
        cache_type="kv",
        tool_parser="llama",
        supports_native_tools=True,
        preserve_native_tool_format=True,
        description="Llama 4 series",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="llama3",
        pattern=r"(?i)llama[\-_.]?3",
        cache_type="kv",
        tool_parser="llama",
        supports_native_tools=True,
        preserve_native_tool_format=True,
        description="Llama 3.x series",
        priority=10,
    ))

    _register(ModelConfig(
        family_name="llama",
        pattern=r"(?i)llama",
        cache_type="kv",
        tool_parser="llama",
        description="Llama models (generic)",
        priority=50,
    ))

    # =========================================================================
    # Mistral/Mixtral/Devstral/Codestral Family
    # =========================================================================

    _register(ModelConfig(
        family_name="devstral",
        pattern=r"(?i)devstral",
        cache_type="kv",
        tool_parser="mistral",
        supports_native_tools=True,
        preserve_native_tool_format=True,
        description="Devstral coding models (Mistral)",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="codestral",
        pattern=r"(?i)codestral",
        cache_type="kv",
        tool_parser="mistral",
        supports_native_tools=True,
        preserve_native_tool_format=True,
        description="Codestral coding models (Mistral)",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="pixtral",
        pattern=r"(?i)pixtral",
        cache_type="kv",
        tool_parser="mistral",
        is_mllm=True,
        supports_native_tools=True,
        preserve_native_tool_format=True,
        description="Pixtral vision models",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="mixtral",
        pattern=r"(?i)mixtral",
        cache_type="kv",
        tool_parser="mistral",
        supports_native_tools=True,
        preserve_native_tool_format=True,
        description="Mixtral MoE models",
        priority=10,
    ))

    _register(ModelConfig(
        family_name="mistral",
        pattern=r"(?i)mistral",
        cache_type="kv",
        tool_parser="mistral",
        supports_native_tools=True,
        preserve_native_tool_format=True,
        description="Mistral models",
        priority=20,
    ))

    # =========================================================================
    # DeepSeek Family
    # =========================================================================

    _register(ModelConfig(
        family_name="deepseek-vl",
        pattern=r"(?i)deepseek[\-_.]?vl",
        cache_type="kv",
        tool_parser="deepseek",
        is_mllm=True,
        description="DeepSeek-VL vision-language models",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="deepseek-r1",
        pattern=r"(?i)deepseek[\-_.]?r1",
        cache_type="kv",
        tool_parser="deepseek",
        reasoning_parser="deepseek_r1",
        think_in_template=True,
        description="DeepSeek R1 reasoning model",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="deepseek-v3",
        pattern=r"(?i)deepseek[\-_.]?v3",
        cache_type="kv",
        tool_parser="deepseek",
        supports_native_tools=True,
        preserve_native_tool_format=True,
        description="DeepSeek V3 series",
        priority=10,
    ))

    _register(ModelConfig(
        family_name="deepseek-v2",
        pattern=r"(?i)deepseek[\-_.]?v2",
        cache_type="kv",
        tool_parser="deepseek",
        description="DeepSeek V2 series",
        priority=10,
    ))

    _register(ModelConfig(
        family_name="deepseek",
        pattern=r"(?i)deepseek",
        cache_type="kv",
        tool_parser="deepseek",
        description="DeepSeek models (generic)",
        priority=50,
    ))

    # =========================================================================
    # Phi Family (Microsoft)
    # =========================================================================

    _register(ModelConfig(
        family_name="phi4-multimodal",
        pattern=r"(?i)phi[\-_.]?4.*(?:vision|multimodal|vlm)",
        cache_type="kv",
        is_mllm=True,
        description="Microsoft Phi-4 multimodal",
        priority=3,
    ))

    # Phi-4 Reasoning models use <think>...</think> tags (same format as DeepSeek R1)
    _register(ModelConfig(
        family_name="phi4-reasoning",
        pattern=r"(?i)phi[\-_.]?4.*(?:reason|think)",
        cache_type="kv",
        tool_parser="hermes",
        reasoning_parser="deepseek_r1",
        think_in_template=True,
        supports_native_tools=True,
        description="Microsoft Phi-4 Reasoning",
        priority=2,
    ))

    _register(ModelConfig(
        family_name="phi4",
        pattern=r"(?i)phi[\-_.]?4",
        cache_type="kv",
        tool_parser="hermes",
        supports_native_tools=True,
        description="Microsoft Phi-4",
        priority=10,
    ))

    _register(ModelConfig(
        family_name="phi3-vision",
        pattern=r"(?i)phi[\-_.]?3[\-_.]?(?:vision|v)",
        cache_type="kv",
        is_mllm=True,
        description="Microsoft Phi-3 Vision",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="phi3-moe",
        pattern=r"(?i)phi[\-_.]?3.*(?:moe|MoE)",
        cache_type="kv",
        description="Microsoft Phi-3 MoE",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="phi3",
        pattern=r"(?i)phi[\-_.]?3",
        cache_type="kv",
        description="Microsoft Phi-3",
        priority=20,
    ))

    # =========================================================================
    # Gemma Family (Google)
    # =========================================================================

    _register(ModelConfig(
        family_name="medgemma",
        pattern=r"(?i)medgemma",
        cache_type="kv",
        is_mllm=True,
        architecture_hints={"inject_pixel_values": True},
        description="Google MedGemma (medical multimodal)",
        priority=3,
    ))

    _register(ModelConfig(
        family_name="paligemma",
        pattern=r"(?i)paligemma",
        cache_type="kv",
        is_mllm=True,
        description="Google PaliGemma",
        priority=5,
    ))

    # gemma3-text: text-only Gemma 3 (detected via config.json model_type="gemma3_text")
    # Shares parsers with gemma3 but is NOT multimodal — supports batching, paged cache, KV quant
    _register(ModelConfig(
        family_name="gemma3-text",
        pattern=r"(?i)gemma[\-_.]?3[\-_.]?text",
        cache_type="kv",
        tool_parser="hermes",
        reasoning_parser="deepseek_r1",
        think_in_template=True,
        supports_native_tools=True,
        is_mllm=False,
        description="Google Gemma 3 (text-only, tool calling, thinking)",
        priority=8,
    ))

    _register(ModelConfig(
        family_name="gemma3",
        pattern=r"(?i)gemma[\-_.]?3",
        cache_type="kv",
        tool_parser="hermes",
        reasoning_parser="deepseek_r1",
        think_in_template=True,
        supports_native_tools=True,
        is_mllm=True,
        architecture_hints={"inject_pixel_values": True},
        description="Google Gemma 3 (multimodal, tool calling, thinking)",
        priority=10,
    ))

    _register(ModelConfig(
        family_name="gemma2",
        pattern=r"(?i)gemma[\-_.]?2",
        cache_type="kv",
        description="Google Gemma 2",
        priority=20,
    ))

    _register(ModelConfig(
        family_name="gemma",
        pattern=r"(?i)gemma",
        cache_type="kv",
        description="Google Gemma (generic)",
        priority=50,
    ))

    # =========================================================================
    # NVIDIA Nemotron Family
    # =========================================================================

    _register(ModelConfig(
        family_name="nemotron",
        pattern=r"(?i)nemotron|NVIDIA[\-_]Nemotron",
        cache_type="hybrid",
        tokenizer_fallback=True,
        tool_parser="nemotron",
        description="NVIDIA Nemotron (hybrid Mamba+Attention)",
        priority=10,
    ))

    # =========================================================================
    # Cohere Family
    # =========================================================================

    _register(ModelConfig(
        family_name="command-r-plus",
        pattern=r"(?i)command[\-_.]?r[\-_.]?(?:plus|\+)",
        cache_type="kv",
        special_tokens_to_clean=["<|END_OF_TURN|>", "<|START_OF_TURN|>"],
        description="Cohere Command-R+",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="command-r",
        pattern=r"(?i)command[\-_.]?r",
        cache_type="kv",
        special_tokens_to_clean=["<|END_OF_TURN|>", "<|START_OF_TURN|>"],
        description="Cohere Command-R",
        priority=10,
    ))

    _register(ModelConfig(
        family_name="aya",
        pattern=r"(?i)aya[\-_.]",
        cache_type="kv",
        special_tokens_to_clean=["<|END_OF_TURN|>", "<|START_OF_TURN|>"],
        description="Cohere Aya models",
        priority=20,
    ))

    # =========================================================================
    # Yi Family (01.AI)
    # =========================================================================

    _register(ModelConfig(
        family_name="yi-vl",
        pattern=r"(?i)yi[\-_.].*(?:vl|vision)",
        cache_type="kv",
        is_mllm=True,
        description="Yi Vision-Language models",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="yi",
        pattern=r"(?i)(?:^|/)yi[\-_.]",
        cache_type="kv",
        description="01.AI Yi models",
        priority=30,
    ))

    # =========================================================================
    # Falcon Family (TII)
    # =========================================================================

    _register(ModelConfig(
        family_name="falcon-mamba",
        pattern=r"(?i)falcon[\-_.]?mamba",
        cache_type="mamba",
        description="Falcon Mamba (SSM variant)",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="falcon",
        pattern=r"(?i)falcon",
        cache_type="kv",
        description="TII Falcon models",
        priority=30,
    ))

    # =========================================================================
    # Jamba Family (AI21) - Hybrid SSM
    # =========================================================================

    _register(ModelConfig(
        family_name="jamba",
        pattern=r"(?i)jamba",
        cache_type="hybrid",
        description="AI21 Jamba (hybrid Mamba+Attention)",
        priority=20,
    ))

    # =========================================================================
    # Mamba (Pure SSM)
    # =========================================================================

    _register(ModelConfig(
        family_name="mamba",
        pattern=r"(?i)(?:^|/)mamba[\-_.]",
        cache_type="mamba",
        description="Mamba state space models",
        priority=30,
    ))

    # =========================================================================
    # StableLM / Stability AI
    # =========================================================================

    _register(ModelConfig(
        family_name="stablelm",
        pattern=r"(?i)stable[\-_.]?lm",
        cache_type="kv",
        description="Stability AI StableLM",
        priority=30,
    ))

    _register(ModelConfig(
        family_name="starcoder",
        pattern=r"(?i)starcoder",
        cache_type="kv",
        description="StarCoder code models",
        priority=30,
    ))

    # =========================================================================
    # RWKV
    # =========================================================================

    _register(ModelConfig(
        family_name="rwkv",
        pattern=r"(?i)rwkv",
        cache_type="mamba",
        description="RWKV recurrent models",
        priority=30,
    ))

    # =========================================================================
    # GLM Family (Zhipu)
    # =========================================================================

    # GLM-4.7 Flash uses Harmony/GPT-OSS protocol: <|channel|>analysis/final
    # NOT <think> tags. think_in_template=False because the template uses
    # <|channel|> markers instead.
    # chat_template_custom overrides the shipped ChatML template which uses
    # <|im_start|>/<|im_end|> — tokens NOT in the tokenizer vocabulary.
    _register(ModelConfig(
        family_name="glm47-flash",
        pattern=r"(?i)glm[\-_.]?4[\-_.]?7[\-_.]?flash",
        cache_type="kv",
        eos_tokens=["<|end|>"],
        tool_parser="glm47",
        reasoning_parser="openai_gptoss",
        think_in_template=False,
        supports_native_tools=True,
        chat_template_custom=HARMONY_CHAT_TEMPLATE,
        description="GLM-4.7 Flash (MoE, Harmony protocol reasoning)",
        priority=3,
    ))

    # GPT-OSS models (20B, 120B) — same Harmony protocol as GLM-4.7 Flash
    # chat_template_custom: shipped template uses <|im_start|> which isn't in
    # the tokenizer — causes token shattering and garbled output.
    _register(ModelConfig(
        family_name="gpt-oss",
        pattern=r"(?i)gpt[\-_.]?oss",
        cache_type="kv",
        eos_tokens=["<|end|>"],
        tool_parser="glm47",
        reasoning_parser="openai_gptoss",
        think_in_template=False,
        supports_native_tools=True,
        chat_template_custom=HARMONY_CHAT_TEMPLATE,
        description="GPT-OSS (Harmony protocol reasoning)",
        priority=3,
    ))

    # GLM-4.7 / GLM-Z1 uses <think> tags (like DeepSeek R1), NOT Harmony protocol
    _register(ModelConfig(
        family_name="glm47",
        pattern=r"(?i)glm[\-_.]?(?:4\.7|4[\-_]7|z1)",
        cache_type="kv",
        tool_parser="glm47",
        reasoning_parser="deepseek_r1",
        think_in_template=True,
        supports_native_tools=True,
        description="GLM-4.7 / GLM-Z1 (<think> reasoning)",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="glm4",
        pattern=r"(?i)glm[\-_.]?4",
        cache_type="kv",
        tool_parser="glm47",
        supports_native_tools=True,
        description="Zhipu GLM-4 (tools only, no reasoning)",
        priority=20,
    ))

    # =========================================================================
    # Granite Family (IBM)
    # =========================================================================

    _register(ModelConfig(
        family_name="granite3",
        pattern=r"(?i)granite[\-_.]?3",
        cache_type="kv",
        tool_parser="granite",
        supports_native_tools=True,
        description="IBM Granite 3",
        priority=10,
    ))

    _register(ModelConfig(
        family_name="granite",
        pattern=r"(?i)granite",
        cache_type="kv",
        tool_parser="granite",
        supports_native_tools=True,
        description="IBM Granite models",
        priority=30,
    ))

    # =========================================================================
    # Additional MLLM Models
    # =========================================================================

    _register(ModelConfig(
        family_name="llava",
        pattern=r"(?i)llava",
        cache_type="kv",
        is_mllm=True,
        description="LLaVA vision-language models",
        priority=20,
    ))

    _register(ModelConfig(
        family_name="idefics",
        pattern=r"(?i)idefics",
        cache_type="kv",
        is_mllm=True,
        description="Idefics vision-language models",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="molmo",
        pattern=r"(?i)molmo",
        cache_type="kv",
        is_mllm=True,
        description="Molmo multimodal models",
        priority=20,
    ))

    _register(ModelConfig(
        family_name="cogvlm",
        pattern=r"(?i)cogvlm",
        cache_type="kv",
        is_mllm=True,
        description="CogVLM vision-language models",
        priority=20,
    ))

    _register(ModelConfig(
        family_name="internvl",
        pattern=r"(?i)internvl",
        cache_type="kv",
        is_mllm=True,
        description="InternVL vision-language models",
        priority=20,
    ))

    _register(ModelConfig(
        family_name="minicpm-v",
        pattern=r"(?i)minicpm[\-_.]?v",
        cache_type="kv",
        is_mllm=True,
        description="MiniCPM-V vision models",
        priority=20,
    ))

    _register(ModelConfig(
        family_name="florence",
        pattern=r"(?i)florence",
        cache_type="kv",
        is_mllm=True,
        description="Florence vision models",
        priority=20,
    ))

    _register(ModelConfig(
        family_name="smolvlm",
        pattern=r"(?i)smol[\-_.]?vlm",
        cache_type="kv",
        is_mllm=True,
        description="SmolVLM compact vision-language models",
        priority=20,
    ))

    # =========================================================================
    # Tool-Calling Specialist Models
    # =========================================================================

    _register(ModelConfig(
        family_name="functionary",
        pattern=r"(?i)functionary|meetkai",
        cache_type="kv",
        tool_parser="functionary",
        supports_native_tools=True,
        description="MeetKai Functionary models",
        priority=20,
    ))

    _register(ModelConfig(
        family_name="xlam",
        pattern=r"(?i)xlam",
        cache_type="kv",
        tool_parser="xlam",
        supports_native_tools=True,
        description="Salesforce xLAM models",
        priority=20,
    ))

    _register(ModelConfig(
        family_name="kimi-k2",
        pattern=r"(?i)kimi[\-_.]?k2",
        cache_type="kv",
        tool_parser="kimi",
        supports_native_tools=True,
        description="Kimi K2 (MoE, agentic)",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="kimi",
        pattern=r"(?i)kimi|moonshot",
        cache_type="kv",
        tool_parser="kimi",
        supports_native_tools=True,
        description="Kimi/Moonshot models",
        priority=20,
    ))

    _register(ModelConfig(
        family_name="hermes",
        pattern=r"(?i)hermes",
        cache_type="kv",
        tool_parser="hermes",
        supports_native_tools=True,
        description="Hermes/NousResearch models",
        priority=30,
    ))

    # =========================================================================
    # Solar Family (Upstage)
    # =========================================================================

    _register(ModelConfig(
        family_name="solar-pro",
        pattern=r"(?i)solar[\-_.]?pro",
        cache_type="kv",
        description="Upstage Solar Pro models",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="solar",
        pattern=r"(?i)solar",
        cache_type="kv",
        description="Upstage Solar models",
        priority=30,
    ))

    # =========================================================================
    # Exaone Family (LG AI Research)
    # =========================================================================

    _register(ModelConfig(
        family_name="exaone",
        pattern=r"(?i)exaone",
        cache_type="kv",
        description="LG AI Research EXAONE models",
        priority=20,
    ))

    # =========================================================================
    # OLMo Family (AI2)
    # =========================================================================

    _register(ModelConfig(
        family_name="olmo",
        pattern=r"(?i)olmo",
        cache_type="kv",
        description="AI2 OLMo models",
        priority=30,
    ))

    # =========================================================================
    # OpenELM (Apple)
    # =========================================================================

    _register(ModelConfig(
        family_name="openelm",
        pattern=r"(?i)open[\-_.]?elm",
        cache_type="kv",
        description="Apple OpenELM models",
        priority=30,
    ))

    # =========================================================================
    # InternLM Family (Shanghai AI Lab)
    # =========================================================================

    _register(ModelConfig(
        family_name="internlm-xcomposer",
        pattern=r"(?i)internlm[\-_.]?xcomposer",
        cache_type="kv",
        is_mllm=True,
        description="InternLM-XComposer multimodal models",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="internlm",
        pattern=r"(?i)internlm",
        cache_type="kv",
        description="Shanghai AI Lab InternLM models",
        priority=30,
    ))

    # =========================================================================
    # MiniCPM (OpenBMB)
    # =========================================================================

    _register(ModelConfig(
        family_name="minicpm",
        pattern=r"(?i)minicpm(?![\-_.]?v)",
        cache_type="kv",
        description="OpenBMB MiniCPM text models",
        priority=30,
    ))

    # =========================================================================
    # Step Family (StepFun)
    # =========================================================================

    _register(ModelConfig(
        family_name="step-3.5-flash",
        pattern=r"(?i)step[\-_.]?3[\-_.]?5[\-_.]?flash",
        cache_type="kv",
        tool_parser="step3p5",
        reasoning_parser="qwen3",
        think_in_template=True,
        supports_native_tools=True,
        description="StepFun Step-3.5-Flash (MoE, think-in-template)",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="step",
        pattern=r"(?i)(?:^|/)step[\-_.]",
        cache_type="kv",
        tool_parser="step3p5",
        reasoning_parser="qwen3",
        think_in_template=True,
        supports_native_tools=True,
        description="StepFun Step models",
        priority=30,
    ))

    # =========================================================================
    # MiniMax Family
    # =========================================================================

    _register(ModelConfig(
        family_name="minimax-m2.5",
        pattern=r"(?i)minimax[\-_.]?m2[\-_.]?5",
        cache_type="kv",
        tool_parser="minimax",
        reasoning_parser="qwen3",
        think_in_template=True,
        eos_tokens=["[e~["],
        supports_native_tools=True,
        preserve_native_tool_format=True,
        description="MiniMax M2.5 (230B MoE, 10B active, agentic)",
        priority=5,
    ))

    _register(ModelConfig(
        family_name="minimax-m2",
        pattern=r"(?i)minimax[\-_.]?m2(?![\-_.]?5)",
        cache_type="kv",
        tool_parser="minimax",
        reasoning_parser="qwen3",
        think_in_template=True,
        eos_tokens=["[e~["],
        supports_native_tools=True,
        preserve_native_tool_format=True,
        description="MiniMax M2 family",
        priority=10,
    ))


# Auto-register on import
register_all()

