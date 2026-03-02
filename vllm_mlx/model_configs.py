# SPDX-License-Identifier: Apache-2.0
"""
Model family configurations for vllm-mlx.

Defines configuration profiles for all supported model families including
cache types, tokenizer settings, tool parsers, and architecture hints
indexed strictly by Hugging Face `model_type`.
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

    _register(ModelConfig(
        family_name="qwen2",
        model_types=["qwen2", "qwen2_moe"],
        cache_type="kv",
        eos_tokens=["<|im_end|>"],
        tool_parser="qwen",
        reasoning_parser="qwen3",
        think_in_template=True,
        supports_native_tools=True,
    ))
    
    _register(ModelConfig(
        family_name="qwen2_vl",
        model_types=["qwen2_vl", "qwen2_5_vl"],
        cache_type="kv",
        eos_tokens=["<|im_end|>"],
        tool_parser="qwen",
        reasoning_parser="qwen3",
        think_in_template=True,
        is_mllm=True,
    ))

    _register(ModelConfig(
        family_name="llama",
        model_types=["llama"],
        cache_type="kv",
        tool_parser="llama",
        supports_native_tools=True,
        preserve_native_tool_format=True,
    ))

    _register(ModelConfig(
        family_name="mistral",
        model_types=["mistral", "mixtral"],
        cache_type="kv",
        tool_parser="mistral",
        supports_native_tools=True,
        preserve_native_tool_format=True,
    ))

    _register(ModelConfig(
        family_name="deepseek",
        model_types=["deepseek_v2", "deepseek_v3", "deepseek"],
        cache_type="kv",
        tool_parser="deepseek",
        reasoning_parser="deepseek_r1",
    ))

    _register(ModelConfig(
        family_name="phi3",
        model_types=["phi3"],
        cache_type="kv",
        tool_parser="llama",
    ))
    
    _register(ModelConfig(
        family_name="phi3_v",
        model_types=["phi3_v"],
        cache_type="kv",
        tool_parser="llama",
        is_mllm=True,
    ))

    _register(ModelConfig(
        family_name="gemma",
        model_types=["gemma", "gemma2"],
        cache_type="kv",
    ))

    _register(ModelConfig(
        family_name="paligemma",
        model_types=["paligemma"],
        cache_type="kv",
        is_mllm=True,
    ))

    _register(ModelConfig(
        family_name="chatglm",
        model_types=["chatglm"],
        cache_type="kv",
        tool_parser="glm47",
        chat_template_custom=HARMONY_CHAT_TEMPLATE,
    ))

    _register(ModelConfig(
        family_name="mamba",
        model_types=["mamba"],
        cache_type="mamba",
    ))

    _register(ModelConfig(
        family_name="internlm",
        model_types=["internlm", "internlm2", "internlm3"],
        cache_type="kv",
    ))

    _register(ModelConfig(
        family_name="jamba",
        model_types=["jamba"],
        cache_type="hybrid",
    ))
