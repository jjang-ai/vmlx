#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI for vmlx-engine.

Commands:
    vmlx-engine serve <model> --port 8000    Start OpenAI-compatible server
    vmlx-engine bench <model>                Run benchmark

Usage:
    vmlx-engine serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
    vmlx-engine serve /path/to/model-JANG-2.5bit                            # JANG format
    vmlx-engine bench mlx-community/Llama-3.2-1B-Instruct-4bit --num-prompts 10
"""

import argparse
import os
import sys


def serve_command(args):
    """Start the OpenAI-compatible server."""
    import logging

    import uvicorn

    # Import unified server
    from . import server
    from .scheduler import SchedulerConfig
    from .server import RateLimiter, app, load_model

    logger = logging.getLogger(__name__)

    # Unified server configuration — explicit args ONLY
    # If the user passes --tool-call-parser qwen, we use qwen.
    # If it's "auto", we auto-detect via model_config_registry.
    # If it's "none" or omitted, tool calling is disabled.

    server._api_key = args.api_key or os.environ.get("VLLM_API_KEY")

    # Plumb --omni-backend → VMLX_OMNI_BACKEND so OmniMultimodalDispatcher
    # picks the right path on first request. Explicit CLI arg wins over
    # whatever was in the environment so the panel/UI can override.
    if getattr(args, "omni_backend", None):
        os.environ["VMLX_OMNI_BACKEND"] = args.omni_backend
        logger.info(
            "Nemotron-Omni backend pinned to %s via --omni-backend",
            args.omni_backend,
        )
    if args.timeout <= 0:
        print(f"Error: --timeout must be positive, got {args.timeout}")
        sys.exit(1)
    server._default_timeout = args.timeout
    if args.rate_limit > 0:
        server._rate_limiter = RateLimiter(
            requests_per_minute=args.rate_limit, enabled=True
        )

    # Early detection: image vs text model
    from pathlib import Path
    import json as _json
    model_dir = Path(args.model)
    _is_image = False

    # Named mflux models (not filesystem paths) — detect by known names
    # Keep in sync with SUPPORTED_MODELS + EDIT_MODELS in image_gen.py
    from .image_gen import SUPPORTED_MODELS as _IMG_SUPPORTED, EDIT_MODELS as _EDIT_SUPPORTED
    MFLUX_NAMED_MODELS = set(_IMG_SUPPORTED.keys()) | set(_EDIT_SUPPORTED.keys()) | {
        'krea-dev', 'dev-krea', 'qwen', 'fibo', 'fibo-lite',  # Additional mflux models
    }
    if args.model.lower() in MFLUX_NAMED_MODELS:
        _is_image = True

    if not _is_image and (model_dir / "model_index.json").exists():
        try:
            idx = _json.loads((model_dir / "model_index.json").read_text())
            _is_image = "_diffusers_version" in idx
        except Exception:
            _is_image = True
    elif (model_dir / "transformer").is_dir() and (model_dir / "text_encoder").is_dir():
        # mflux-quantized models: transformer/ + text_encoder/ without model_index.json
        _is_image = True
    elif (model_dir / "transformer").is_dir() and (model_dir / "vae").is_dir():
        # Diffusion model with transformer + vae subdirs
        _is_image = True

    # Configure tool calling
    if args.enable_auto_tool_choice and args.tool_call_parser and args.tool_call_parser != "none":
        server._enable_auto_tool_choice = True
        # "auto" → resolved below by model_config_registry auto-apply (lines 86-108)
        server._tool_call_parser = None if args.tool_call_parser == "auto" else args.tool_call_parser
    else:
        server._enable_auto_tool_choice = False
        server._tool_call_parser = None

    # Configure generation defaults (validate ranges)
    if getattr(args, 'default_temperature', None) is not None:
        if not (0 <= args.default_temperature <= 2):
            print(f"Error: --default-temperature must be between 0 and 2, got {args.default_temperature}")
            sys.exit(1)
        server._default_temperature = args.default_temperature
    if getattr(args, 'default_top_p', None) is not None:
        if not (0 < args.default_top_p <= 1):
            print(f"Error: --default-top-p must be between 0 (exclusive) and 1, got {args.default_top_p}")
            sys.exit(1)
        server._default_top_p = args.default_top_p
    if getattr(args, 'default_repetition_penalty', None) is not None:
        if not (0.5 <= args.default_repetition_penalty <= 2.0):
            print(
                f"Error: --default-repetition-penalty must be between 0.5 and 2.0, "
                f"got {args.default_repetition_penalty}"
            )
            sys.exit(1)
        server._default_repetition_penalty = args.default_repetition_penalty

    # Apply default enable_thinking
    _det = getattr(args, 'default_enable_thinking', None)
    if _det is not None:
        server._default_enable_thinking = _det == "true"

    # Apply custom chat template override
    if getattr(args, 'chat_template', None):
        server._custom_chat_template = args.chat_template

    # Parse --chat-template-kwargs JSON and apply server-wide defaults
    if getattr(args, 'chat_template_kwargs', None) is not None:
        import json as _json
        try:
            ct_kwargs = _json.loads(args.chat_template_kwargs)
            if not isinstance(ct_kwargs, dict):
                print("Error: --chat-template-kwargs must be a JSON object")
                sys.exit(1)
        except _json.JSONDecodeError as e:
            print(f"Error: --chat-template-kwargs is not valid JSON: {e}")
            sys.exit(1)
        # Extract enable_thinking into the dedicated server default
        # Guard against bool("false") == True trap — require actual JSON boolean
        if "enable_thinking" in ct_kwargs:
            val = ct_kwargs["enable_thinking"]
            if not isinstance(val, bool):
                print(f"Error: enable_thinking must be a JSON boolean (true/false), got: {val!r}")
                sys.exit(1)
            server._default_enable_thinking = val
        # Store full kwargs for forwarding to chat templates
        server._default_chat_template_kwargs = ct_kwargs

    # Configure reasoning parser (strictly explicit; "auto" and "none"
    # fall through to registry/template auto-detection further down).
    parser_name = getattr(args, 'reasoning_parser', None)
    _user_requested_auto = parser_name == "auto"
    if parser_name in ("auto", "none", None) or not parser_name:
        parser_name = None

    if parser_name:
        try:
            from .reasoning import get_parser
            parser_cls = get_parser(parser_name)
            server._reasoning_parser = parser_cls()
            logger.info(f"Reasoning parser enabled: {parser_name}")
        except KeyError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except ImportError as e:
            print(f"Error: Failed to import reasoning module: {e}")
            sys.exit(1)
        except Exception as e:
            print(
                f"Error: Failed to initialize reasoning parser "
                f"'{parser_name}': {e}"
            )
            sys.exit(1)
    else:
        server._reasoning_parser = None

    # Auto-apply tool/reasoning parsers from model config registry when CLI
    # flags were not explicitly set.  This lets known models "just work" for
    # direct CLI users who don't pass --tool-call-parser / --reasoning-parser.
    _registry_thinking_model = False
    try:
        from .model_config_registry import get_model_config_registry
        _mc = get_model_config_registry().lookup(args.model)
        if _mc.family_name != "unknown":
            # Auto-apply tool parser
            if not server._tool_call_parser and _mc.tool_parser:
                server._tool_call_parser = _mc.tool_parser
                server._enable_auto_tool_choice = True
                logger.info(f"Auto-configured tool parser from registry: {_mc.tool_parser}")
            # Auto-apply reasoning parser
            if not server._reasoning_parser and _mc.reasoning_parser:
                try:
                    from .reasoning import get_parser
                    _rp_cls = get_parser(_mc.reasoning_parser)
                    server._reasoning_parser = _rp_cls()
                    logger.info(f"Auto-configured reasoning parser from registry: {_mc.reasoning_parser}")
                except Exception as e:
                    logger.warning(f"Failed to auto-configure reasoning parser '{_mc.reasoning_parser}': {e}")
            if getattr(_mc, "think_in_template", False):
                _registry_thinking_model = True
    except Exception as e:
        logger.debug(f"Registry auto-apply skipped: {e}")

    # Thinking-template detection & warning (mlxstudio user report: Raymond
    # Wong, GLM-5.1-JANG_1L).
    #
    # Thinking models (GLM-5.1, DeepSeek-R1, Qwen3 Thinking, Mistral 4
    # reasoning, etc.) inject a `<think>` sentinel into the assistant turn
    # of their chat template. If no reasoning parser is wired up — either
    # by an explicit `--reasoning-parser` flag or via the registry
    # auto-apply above — the model's raw chain of thought streams into
    # the `content` field and looks like rubbish to anyone using the
    # server as a drop-in OpenAI endpoint for concise tasks like title
    # generation.
    #
    # Detection priority:
    #   1. Registry entry says `think_in_template=True` → thinking model.
    #   2. Registry entry exists but `think_in_template=False` → TRUST
    #      the registry. Some templates (Gemma 4, Qwen3) define an
    #      `enable_thinking` jinja variable that's gated off by default
    #      — those are NOT thinking-by-default and a naive template
    #      probe would false-positive on them.
    #   3. Registry has no entry (family=="unknown") → probe the model's
    #      chat_template.jinja or tokenizer_config.json for a literal
    #      `<think>` sentinel. Catches any new thinking model family
    #      we haven't catalogued yet.
    _chat_template_has_think = False
    _registry_known = False
    try:
        from .model_config_registry import get_model_config_registry as _get_reg
        _mc_check = _get_reg().lookup(args.model)
        _registry_known = _mc_check.family_name != "unknown"
    except Exception:
        _registry_known = False

    if not _registry_known:
        try:
            from pathlib import Path as _P
            _mdir = _P(args.model)
            if _mdir.is_dir():
                _tpl_path = _mdir / "chat_template.jinja"
                if _tpl_path.exists():
                    _tpl_text = _tpl_path.read_text(errors="replace")
                    # Only flag literal `<think>` opening tag — the
                    # sentinel that thinking-by-default templates inject
                    # into the assistant turn. Avoid matching
                    # `<|think|>` (Gemma 4 channel marker) or
                    # `strip_thinking` macro references.
                    if "<think>" in _tpl_text:
                        _chat_template_has_think = True
                if not _chat_template_has_think:
                    _tcfg = _mdir / "tokenizer_config.json"
                    if _tcfg.exists():
                        import json as _json
                        _cfg = _json.loads(_tcfg.read_text())
                        _inline = _cfg.get("chat_template") or ""
                        if isinstance(_inline, list):
                            _inline = " ".join(
                                t.get("template", "") if isinstance(t, dict) else str(t)
                                for t in _inline
                            )
                        if "<think>" in _inline:
                            _chat_template_has_think = True
        except Exception as _detect_e:
            logger.debug(f"Thinking-template detection skipped: {_detect_e}")

    _is_thinking_model = _registry_thinking_model or _chat_template_has_think
    _thinking_default = getattr(server, "_default_enable_thinking", None)
    _thinking_off = _thinking_default is False

    # If the user asked for --reasoning-parser auto AND neither the
    # registry nor anything else picked one, fall back to deepseek_r1 as
    # the safe default for any model whose template contains <think>.
    # DeepSeek R1's parser is the most lenient (accepts missing
    # <think> sentinels, handles partial blocks) and works for GLM-5.1,
    # DeepSeek-R1, Nemotron-R, Phi-4 Reasoning, and most unknown think-
    # tag models. Explicit user requests (`--reasoning-parser qwen3`
    # etc.) still win.
    if (
        _user_requested_auto
        and _is_thinking_model
        and not server._reasoning_parser
    ):
        try:
            from .reasoning import get_parser
            server._reasoning_parser = get_parser("deepseek_r1")()
            logger.info(
                "Reasoning parser auto-detected from chat template "
                "(<think> sentinel found) → deepseek_r1"
            )
        except Exception as _auto_e:
            logger.warning(f"--reasoning-parser auto fallback skipped: {_auto_e}")

    if _is_thinking_model and not server._reasoning_parser and not _thinking_off:
        # Prominent loud warning — this is the exact condition that made
        # Raymond Wong think GLM-5.1-JANG_1L was broken. Tell the user
        # what's going on and give them two one-line fixes.
        _src = "registry" if _registry_thinking_model else "chat_template"
        print("=" * 60, file=sys.stderr)
        print("THINKING MODEL WARNING", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(
            f"  This model is a THINKING model (detected via {_src}).",
            file=sys.stderr,
        )
        print(
            "  No reasoning parser is active and thinking is not disabled,",
            file=sys.stderr,
        )
        print(
            "  so the model's chain-of-thought WILL stream into the",
            file=sys.stderr,
        )
        print(
            "  `content` field of every response. Clients expecting a",
            file=sys.stderr,
        )
        print(
            "  terse answer (title generators, tool selectors, etc.) will",
            file=sys.stderr,
        )
        print("  see what looks like rubbish.", file=sys.stderr)
        print("", file=sys.stderr)
        print("  Pick one:", file=sys.stderr)
        print(
            "    A) add  --reasoning-parser auto  to extract <think> into",
            file=sys.stderr,
        )
        print(
            "       a separate reasoning_content field, or",
            file=sys.stderr,
        )
        print(
            "    B) add  --default-enable-thinking false  to skip",
            file=sys.stderr,
        )
        print(
            "       reasoning entirely and go straight to the answer.",
            file=sys.stderr,
        )
        print("=" * 60, file=sys.stderr)
        logger.warning(
            "Thinking model loaded without a reasoning parser — output "
            "will include raw <think> blocks in content. See the "
            "THINKING MODEL WARNING printed above for one-line fixes."
        )

    # Security summary at startup
    print("=" * 60)
    print("SECURITY CONFIGURATION")
    print("=" * 60)
    if args.api_key:
        print("  Authentication: ENABLED (API key required)")
    else:
        print("  Authentication: DISABLED - Use --api-key to enable")
    if args.rate_limit > 0:
        print(f"  Rate limiting: ENABLED ({args.rate_limit} req/min)")
    else:
        print("  Rate limiting: DISABLED - Use --rate-limit to enable")
    print(f"  Request timeout: {args.timeout}s")
    if args.enable_auto_tool_choice:
        print(f"  Tool calling: ENABLED (parser: {args.tool_call_parser})")
    else:
        print("  Tool calling: Use --enable-auto-tool-choice to enable")
    if server._reasoning_parser:
        parser_display = type(server._reasoning_parser).__name__
        _thinking_suffix = ""
        if _is_thinking_model:
            _thinking_suffix = "  [thinking model detected]"
        print(f"  Reasoning: ENABLED (parser: {parser_display}){_thinking_suffix}")
    elif getattr(args, 'reasoning_parser', None):
        print(f"  Reasoning: requested '{args.reasoning_parser}' but no parser matched")
    elif _is_thinking_model and _thinking_off:
        print(
            "  Reasoning: thinking-model detected, default_enable_thinking=false "
            "(model will skip reasoning)"
        )
    elif _is_thinking_model:
        print(
            "  Reasoning: THINKING MODEL DETECTED but no parser active — "
            "see warning above, output will include <think> blocks"
        )
    else:
        print("  Reasoning: Use --reasoning-parser to enable")
    spec_model = getattr(args, 'speculative_model', None)
    if spec_model:
        print(f"  Speculative decoding: ENABLED")
        print(f"    Draft model: {spec_model}")
        print(f"    Draft tokens per step: {getattr(args, 'num_draft_tokens', 3)}")
    else:
        print("  Speculative decoding: Use --speculative-model to enable")
    if getattr(args, 'chat_template', None):
        print(f"  Chat template: CUSTOM ({len(args.chat_template)} chars)")
    if getattr(args, 'chat_template_kwargs', None):
        print(f"  Chat template kwargs: {args.chat_template_kwargs}")
    print("=" * 60)

    print(f"Loading model: {args.model}")
    if getattr(args, 'served_model_name', None):
        print(f"Served model name: {args.served_model_name}")
    print(f"Default max tokens: {args.max_tokens}")

    # Store MCP config path for FastAPI startup
    if args.mcp_config:
        print(f"MCP config: {args.mcp_config}")
        os.environ["VLLM_MLX_MCP_CONFIG"] = args.mcp_config

    # Pre-load embedding model if specified
    if args.embedding_model:
        print(f"Pre-loading embedding model: {args.embedding_model}")
        server.load_embedding_model(args.embedding_model, lock=True)
        print(f"Embedding model loaded: {args.embedding_model}")

    # Smelt mode info
    if getattr(args, 'smelt', False):
        pct = getattr(args, 'smelt_experts', 50)
        print(f"\n  Smelt mode: loading {pct}% of MoE experts per layer\n")
        # Set server module smelt globals IMMEDIATELY so any downstream
        # is_mllm_model() call (e.g., the speculative decoding VLM check
        # below) sees smelt as active. Without this, is_mllm_model() runs
        # detection based on config.json alone and returns True for VLM
        # architectures, which would then emit a misleading speculative-
        # decoding warning AND let force_mllm slip through to the loader.
        from . import server as _server_module
        _server_module._smelt_enabled = True
        _server_module._smelt_experts = pct
        # Smelt ⊕ VLM mutual exclusion.
        #
        # Smelt's partial-expert loader swaps experts in/out of GPU RAM to
        # reduce footprint on MoE models. The vision encoder and multimodal
        # projector, however, are NOT expert modules — they're always
        # resident and depend on the full mlx_vlm model wrapper. When smelt
        # is active the loader takes a different code path (`smelt_load`)
        # that does not set up the vision tower state dict the same way as
        # `load_jang_vlm_model`. In practice that means image input on a
        # smelt-loaded VLM silently produces garbage logits (vision_tower
        # weights present but not wired through the expert swapper) and the
        # model answers as if no image was attached.
        #
        # Rather than letting the user hit that silent failure, force
        # text-only mode: override --is-mllm off and tell is_mllm_model()
        # to skip VLM detection. If the user explicitly passed --is-mllm
        # with --smelt we warn loudly so they know the flag was dropped.
        if getattr(args, 'is_mllm', False):
            print(
                "  WARNING: --is-mllm is incompatible with --smelt. Smelt's "
                "partial-expert loader does not wire the vision tower, so "
                "image input would produce garbage output. Disabling VLM "
                "mode for this session — text-only inference only."
            )
            args.is_mllm = False
        # Sentinel consumed by is_mllm_model() via server._smelt_enabled —
        # the loader also reads this flag to pick the smelt_load code path.
        setattr(args, '_smelt_forces_text_only', True)

    # Flash MoE mutual exclusion guards
    if getattr(args, 'flash_moe', False):
        if getattr(args, 'smelt', False):
            print("ERROR: --flash-moe and --smelt are mutually exclusive.")
            print("  Both modify MoE expert layers. Use one or the other:")
            print("    --smelt: loads partial experts at startup (cache-biased routing)")
            print("    --flash-moe: streams ALL experts from SSD on-demand (slot-bank cache)")
            sys.exit(1)
        if getattr(args, 'distributed', False):
            print("ERROR: --flash-moe and --distributed are mutually exclusive.")
            print("  Flash MoE patches local model layers but distributed workers")
            print("  have their own model copies. Use --smelt for distributed MoE.")
            sys.exit(1)

    # Distributed compute mutual exclusion guards (Phase 1 conservative set).
    # Distributed pipeline parallelism splits model layers across workers;
    # features that patch or trace local layer objects cannot coexist safely.
    if getattr(args, 'distributed', False):
        if getattr(args, 'enable_jit', False):
            print("ERROR: --distributed and --enable-jit are mutually exclusive.")
            print("  JIT (mx.compile) traces layer objects on the coordinator, but")
            print("  workers own distinct layer ranges — the compiled graph would be")
            print("  wrong for every worker except whichever happens to match the trace.")
            print("  Start distributed without JIT, or run single-node with JIT.")
            sys.exit(1)
        if getattr(args, 'smelt', False):
            print("ERROR: --distributed and --smelt are mutually exclusive (experimental).")
            print("  Distributed + Smelt has not been validated end-to-end. Each worker")
            print("  would need to independently smelt its layer range, and layer_assign")
            print("  is not yet MoE-smelt-aware. Lift this guard once Phase 2 testing")
            print("  confirms the combination works.")
            sys.exit(1)
        if getattr(args, 'speculative_model', None):
            print("ERROR: --distributed and --speculative-model are mutually exclusive.")
            print("  Speculative decoding requires the draft model to be co-located with")
            print("  the target model for low-latency drafting; distributing both would")
            print("  negate the speedup. Run speculative decoding single-node.")
            sys.exit(1)
        # Tensor parallelism is stubbed for Phase 2. The `tensor_parallel.py`
        # module exists as scaffolding (ColumnParallelLinear, RowParallelLinear,
        # shard_model_tp) but no path in the codebase wires it up. Refusing
        # here prevents users from silently getting wrong behavior.
        if getattr(args, 'distributed_mode', 'pipeline') == 'tensor':
            print("ERROR: --distributed-mode tensor is not yet implemented.")
            print("  Tensor parallelism would shard each layer's weight matrices")
            print("  across nodes (column- and row-parallel linears with all-reduce).")
            print("  The vmlx_engine.distributed.tensor_parallel module exists as")
            print("  scaffolding only — no forward pass is wired. Phase 2 work item.")
            print("  Use --distributed-mode pipeline for now (this IS the default).")
            sys.exit(1)

    # Build scheduler config for batched mode
    scheduler_config = None
    if args.continuous_batching:
        # Handle prefix cache flags
        enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache

        # Validate flag combinations BEFORE building config
        if not args.use_paged_cache and args.enable_block_disk_cache:
            print("  WARNING: --enable-block-disk-cache requires --use-paged-cache, disabling disk cache")
            args.enable_block_disk_cache = False

        scheduler_config = SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            prefill_batch_size=args.prefill_batch_size,
            prefill_step_size=getattr(args, 'prefill_step_size', 2048),
            completion_batch_size=args.completion_batch_size,
            enable_prefix_cache=enable_prefix_cache,
            prefix_cache_size=args.prefix_cache_size,
            prefix_cache_max_bytes=getattr(args, 'prefix_cache_max_bytes', None),
            use_state_machine_stops=not getattr(args, 'no_state_machine_stops', False),
            # Memory-aware cache options
            use_memory_aware_cache=not args.no_memory_aware_cache,
            cache_memory_mb=args.cache_memory_mb,
            cache_memory_percent=args.cache_memory_percent,
            cache_ttl_minutes=getattr(args, 'cache_ttl_minutes', 0),
            # Paged cache options
            use_paged_cache=args.use_paged_cache,
            paged_cache_block_size=args.paged_cache_block_size,
            max_cache_blocks=args.max_cache_blocks,
            # KV cache quantization
            kv_cache_quantization=args.kv_cache_quantization,
            kv_cache_group_size=args.kv_cache_group_size,
            # Disk cache
            enable_disk_cache=args.enable_disk_cache,
            disk_cache_dir=args.disk_cache_dir,
            disk_cache_max_gb=args.disk_cache_max_gb,
            model_path=args.model,
            # Loader fingerprint inputs (F6 + A4 Concern #1) — feed smelt
            # state into the trie cache key so divergent loader configs
            # never share K/V entries.
            smelt_enabled=getattr(args, 'smelt', False),
            smelt_pct=(
                float(getattr(args, 'smelt_experts', 50))
                if getattr(args, 'smelt', False)
                else None
            ),
            # Block-level disk cache (L2 for paged cache)
            enable_block_disk_cache=args.enable_block_disk_cache,
            block_disk_cache_dir=args.block_disk_cache_dir,
            block_disk_cache_max_gb=args.block_disk_cache_max_gb,
            # Prompt Lookup Decoding
            pld_enabled=args.enable_pld,
            pld_summary_interval=args.pld_summary_interval,
        )

        print("Mode: Continuous batching (for multiple concurrent users)")
        print(f"Stream interval: {args.stream_interval} tokens")
        if args.use_paged_cache:
            print(
                f"Paged cache: block_size={args.paged_cache_block_size}, max_blocks={args.max_cache_blocks}"
            )
            if args.enable_block_disk_cache:
                print(f"Block disk cache: max={args.block_disk_cache_max_gb}GB")
        elif enable_prefix_cache and not args.no_memory_aware_cache:
            cache_info = (
                f"{args.cache_memory_mb}MB"
                if args.cache_memory_mb
                else f"{args.cache_memory_percent*100:.0f}% of RAM"
            )
            print(f"Memory-aware cache: {cache_info}")
        elif enable_prefix_cache:
            print(f"Prefix cache: max_entries={args.prefix_cache_size}")
        if args.kv_cache_quantization != "none":
            print(f"KV cache quantization: {args.kv_cache_quantization} (group_size={args.kv_cache_group_size})")
    else:
        print("Mode: Simple (maximum throughput)")
        # Warn about settings that require continuous batching
        ignored = []
        if args.kv_cache_quantization != "none":
            ignored.append(f"--kv-cache-quantization {args.kv_cache_quantization}")
        if args.use_paged_cache:
            ignored.append("--use-paged-cache")
        if args.enable_prefix_cache and not args.disable_prefix_cache:
            ignored.append("--enable-prefix-cache")
        if ignored:
            print(f"  NOTE: These settings require --continuous-batching and will be ignored: {', '.join(ignored)}")

    # Load speculative decoding draft model if configured
    spec_model = getattr(args, 'speculative_model', None)
    if spec_model:
        from .speculative import SpeculativeConfig, load_draft_model
        spec_config = SpeculativeConfig(
            model=spec_model,
            num_tokens=getattr(args, 'num_draft_tokens', 3),
        )
        load_draft_model(spec_config)
        print(f"Draft model loaded: {spec_model}")

        # Warn about incompatible combinations
        if getattr(args, 'continuous_batching', False):
            print(
                "  ⚠️  WARNING: Speculative decoding is incompatible with --continuous-batching.")
            print(
                "     The draft model will only be used in SimpleEngine (non-batched) mode.")
            print(
                "     BatchedEngine requests will use standard (non-speculative) generation.")

        # Check if target model is MLLM
        from .api.utils import is_mllm_model
        is_mllm = is_mllm_model(args.model, force_mllm=getattr(args, 'is_mllm', False))
        if is_mllm:
            print(
                "  ⚠️  WARNING: Speculative decoding is incompatible with multimodal (VLM) models.")
            print(
                "     The draft model will be ignored for VLM requests (mlx-vlm has no spec decoding).")

    # Configure log level
    log_level = getattr(args, 'log_level', 'INFO').upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), force=True)
    server._log_level = log_level

    # Configure CORS
    allowed_origins = getattr(args, 'allowed_origins', '*')
    server._allowed_origins = allowed_origins

    # Configure JIT compilation
    server._enable_jit = getattr(args, 'enable_jit', False)

    # Validate port range BEFORE loading model (loading can take 30s+)
    if args.port < 0 or args.port > 65535:
        print(f"Error: --port must be between 0 and 65535, got {args.port}")
        sys.exit(1)

    if _is_image:
        # Image model path — load via mflux, serve /v1/images/generations only
        logger.info(f"Detected image model: {args.model}")
        server._model_type = "image"
        # Use served_model_name (original model ID like "flux-kontext") if provided,
        # otherwise extract from path (gives directory name like "flux-kontext-dev-4bit")
        _served = getattr(args, 'served_model_name', None)
        server._model_name = _served or args.model.rstrip("/").split("/")[-1]
        server._model_path = args.model
        server._served_model_name = _served

        # Load image model at startup (not lazy)
        try:
            from .image_gen import ImageGenEngine
        except ImportError:
            print("Error: mflux not installed. Image generation requires mflux.")
            print("Install with: pip install mflux")
            sys.exit(1)

        # Resolve quantize: explicit flag > directory name detection > None
        _image_quantize = getattr(args, 'image_quantize', None)
        if _image_quantize is None:
            # Check both model_name and actual model path directory name
            # (--served-model-name may be a short alias like "flux-kontext" without "-4bit")
            _names_to_check = [server._model_name.lower()]
            _path_dir_name = args.model.rstrip("/").split("/")[-1].lower()
            if _path_dir_name != _names_to_check[0]:
                _names_to_check.append(_path_dir_name)
            for _check_name in _names_to_check:
                for bits in [3, 4, 5, 6, 8]:
                    if f"-{bits}bit" in _check_name or f"_{bits}bit" in _check_name:
                        _image_quantize = bits
                        logger.info(f"Detected {bits}-bit quantization from model name")
                        break
                if _image_quantize is not None:
                    break
        # Store globally so /v1/images/edits can use it for lazy model loading
        server._image_quantize = _image_quantize

        try:
            server._image_gen = ImageGenEngine()
            _mflux_class = getattr(args, 'mflux_class', None)
            # Unified load — handles both gen and edit models via mflux_class dispatch
            server._image_gen.load(
                model_name=server._model_name,
                model_path=args.model,
                quantize=_image_quantize,
                mflux_class=_mflux_class,
                mflux_name=server._model_name,
            )
        except Exception as e:
            print(f"Error: Failed to load image model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Text model path — existing flow
        server._model_type = "text"
        load_model(
            args.model,
            use_batching=args.continuous_batching,
            scheduler_config=scheduler_config,
            stream_interval=args.stream_interval if args.continuous_batching else 1,
            max_tokens=args.max_tokens,
            served_model_name=getattr(args, 'served_model_name', None),
            force_mllm=getattr(args, 'is_mllm', False),
            smelt=getattr(args, 'smelt', False),
            smelt_experts=getattr(args, 'smelt_experts', 50),
            distributed=getattr(args, 'distributed', False),
            distributed_mode=getattr(args, 'distributed_mode', 'pipeline'),
            cluster_secret=getattr(args, 'cluster_secret', '') or os.environ.get('VMLX_CLUSTER_SECRET', ''),
            worker_nodes=getattr(args, 'worker_nodes', None),
            flash_moe=getattr(args, 'flash_moe', False),
            flash_moe_slot_bank=getattr(args, 'flash_moe_slot_bank', 64),
            flash_moe_prefetch=getattr(args, 'flash_moe_prefetch', 'none'),
            flash_moe_io_split=getattr(args, 'flash_moe_io_split', 4),
        )
        # Save speculative config for deep sleep/wake reload
        server._cli_args['speculative_model'] = getattr(args, 'speculative_model', None)
        server._cli_args['num_draft_tokens'] = getattr(args, 'num_draft_tokens', 3)

    # Configure CORS middleware
    from fastapi.middleware.cors import CORSMiddleware
    origins = [o.strip() for o in allowed_origins.split(',') if o.strip()]
    # CORS spec: allow_credentials=True is invalid with origins=["*"]
    # Only enable credentials when specific origins are listed
    has_wildcard = '*' in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=not has_wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security warning: 0.0.0.0 exposes server to the network
    if args.host == '0.0.0.0' and not server._api_key:
        print()
        print("=" * 60)
        print("  WARNING: Server binding to 0.0.0.0 (all interfaces)")
        print("  with no API key set. Any device on your network can")
        print("  access this server. Set --api-key or change --host")
        print("  to 127.0.0.1 for local-only access.")
        print("=" * 60)
        print()

    # Start server
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=log_level.lower())


def bench_command(args):
    """Run benchmark."""
    import asyncio
    import time

    from mlx_lm import load

    from .engine_core import AsyncEngineCore, EngineConfig
    from .request import SamplingParams
    from .scheduler import SchedulerConfig

    # Smelt mode: set server globals
    if getattr(args, 'smelt', False):
        from . import server as _server_module
        _server_module._smelt_enabled = True
        _server_module._smelt_experts = getattr(args, 'smelt_experts', 50)

    # Handle prefix cache flags
    enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache

    # Flash MoE guards for bench
    if getattr(args, 'flash_moe', False):
        if getattr(args, 'smelt', False):
            print("ERROR: --flash-moe and --smelt are mutually exclusive (bench).")
            sys.exit(1)

    async def run_benchmark():
        print(f"Loading model: {args.model}")
        from .utils.tokenizer import load_model_with_fallback
        model, tokenizer = load_model_with_fallback(args.model)

        # Apply Flash MoE if requested
        if getattr(args, 'flash_moe', False):
            print(f"Flash MoE: enabled (slot_bank={getattr(args, 'flash_moe_slot_bank', 64)})")
            try:
                from .utils.smelt_loader import ExpertIndex
                from .utils.flash_moe_loader import FlashMoEExpertLoader, SlotBankCache
                from .models.flash_moe_integration import apply_flash_moe, free_expert_weights
                from .api.utils import resolve_to_local_path

                resolved = resolve_to_local_path(args.model)
                ei = ExpertIndex.build(resolved)
                if ei.num_moe_layers > 0:
                    cache = SlotBankCache(max_slots=getattr(args, 'flash_moe_slot_bank', 64))
                    loader = FlashMoEExpertLoader(ei, cache, io_workers=4)
                    # Unwrap MLLMModelWrapper if present (bench doesn't use it but be defensive)
                    raw = getattr(model, '_model', None) or getattr(model, 'model', None) or model
                    patched = apply_flash_moe(raw, loader)
                    if patched > 0:
                        freed = free_expert_weights(raw)
                        print(f"Flash MoE: {patched} layers patched, {freed/1e9:.1f}GB freed")
            except Exception as e:
                print(f"Flash MoE setup failed: {e}")

        scheduler_config = SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            prefill_batch_size=args.prefill_batch_size,
            prefill_step_size=getattr(args, 'prefill_step_size', 2048),
            completion_batch_size=args.completion_batch_size,
            enable_prefix_cache=enable_prefix_cache,
            prefix_cache_size=args.prefix_cache_size,
            prefix_cache_max_bytes=getattr(args, 'prefix_cache_max_bytes', None),
            use_state_machine_stops=not getattr(args, 'no_state_machine_stops', False),
            # Memory-aware cache options
            use_memory_aware_cache=not args.no_memory_aware_cache,
            cache_memory_mb=args.cache_memory_mb,
            cache_memory_percent=args.cache_memory_percent,
            cache_ttl_minutes=getattr(args, 'cache_ttl_minutes', 0),
            # Paged cache options
            use_paged_cache=args.use_paged_cache,
            paged_cache_block_size=args.paged_cache_block_size,
            max_cache_blocks=args.max_cache_blocks,
            # KV cache quantization
            kv_cache_quantization=args.kv_cache_quantization,
            kv_cache_group_size=args.kv_cache_group_size,
            model_path=args.model,
            # Disk cache (prompt-level L2)
            enable_disk_cache=getattr(args, 'enable_disk_cache', False),
            disk_cache_dir=getattr(args, 'disk_cache_dir', None),
            disk_cache_max_gb=getattr(args, 'disk_cache_max_gb', 10.0),
            # Block disk cache (L2 for paged cache)
            enable_block_disk_cache=getattr(args, 'enable_block_disk_cache', False),
            block_disk_cache_dir=getattr(args, 'block_disk_cache_dir', None),
            block_disk_cache_max_gb=getattr(args, 'block_disk_cache_max_gb', 10.0),
            # Loader fingerprint inputs (F6 + A4 Concern #1)
            smelt_enabled=getattr(args, 'smelt', False),
            smelt_pct=(
                float(getattr(args, 'smelt_experts', 50))
                if getattr(args, 'smelt', False)
                else None
            ),
        )
        engine_config = EngineConfig(
            model_name=args.model,
            scheduler_config=scheduler_config,
        )

        if args.use_paged_cache:
            print(
                f"Paged cache: block_size={args.paged_cache_block_size}, max_blocks={args.max_cache_blocks}"
            )

        # Generate prompts
        prompts = [
            f"Write a short poem about {topic}."
            for topic in [
                "nature",
                "love",
                "technology",
                "space",
                "music",
                "art",
                "science",
                "history",
                "food",
                "travel",
            ][: args.num_prompts]
        ]

        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.7,
        )

        print(
            f"\nRunning benchmark with {len(prompts)} prompts, max_tokens={args.max_tokens}"
        )
        print("-" * 50)

        total_prompt_tokens = 0
        total_completion_tokens = 0

        async with AsyncEngineCore(model, tokenizer, engine_config) as engine:
            await asyncio.sleep(0.1)  # Warm up

            start_time = time.perf_counter()

            # Add all requests
            request_ids = []
            for prompt in prompts:
                rid = await engine.add_request(prompt, params)
                request_ids.append(rid)

            # Collect all outputs
            async def get_output(rid):
                async for out in engine.stream_outputs(rid, timeout=120):
                    if out.finished:
                        return out
                return None

            results = await asyncio.gather(*[get_output(r) for r in request_ids])

            total_time = time.perf_counter() - start_time

        # Calculate stats
        for r in results:
            if r:
                total_prompt_tokens += r.prompt_tokens
                total_completion_tokens += r.completion_tokens

        total_tokens = total_prompt_tokens + total_completion_tokens

        print("\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Prompts/second: {len(prompts)/total_time:.2f}")
        print(f"  Total prompt tokens: {total_prompt_tokens}")
        print(f"  Total completion tokens: {total_completion_tokens}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Tokens/second: {total_completion_tokens/total_time:.2f}")
        print(f"  Throughput: {total_tokens/total_time:.2f} tok/s")

    asyncio.run(run_benchmark())


def bench_detok_command(args):
    """Benchmark streaming detokenizer optimization."""
    import statistics
    import time

    from mlx_lm import load
    from mlx_lm.generate import generate

    print("=" * 70)
    print(" Streaming Detokenizer Benchmark")
    print("=" * 70)
    print()

    print(f"Loading model: {args.model}")
    from .utils.tokenizer import load_model_with_fallback
    model, tokenizer = load_model_with_fallback(args.model)

    # Generate tokens for benchmark
    prompt = "Write a detailed explanation of how machine learning works and its applications in modern technology."
    print(f"Generating tokens with prompt: {prompt[:50]}...")

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=2000,
        verbose=False,
    )

    prompt_tokens = tokenizer.encode(prompt)
    all_tokens = tokenizer.encode(output)
    generated_tokens = all_tokens[len(prompt_tokens) :]
    print(f"Generated {len(generated_tokens)} tokens for benchmark")
    print()

    iterations = args.iterations

    # Benchmark naive decode (old method)
    print("Benchmarking Naive Decode (OLD method)...")
    naive_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for t in generated_tokens:
            _ = tokenizer.decode([t])
        elapsed = time.perf_counter() - start
        naive_times.append(elapsed)

    naive_mean = statistics.mean(naive_times) * 1000

    # Benchmark streaming decode (new method)
    print("Benchmarking Streaming Detokenizer (NEW method)...")
    streaming_times = []
    detok_class = tokenizer._detokenizer_class
    for _ in range(iterations):
        detok = detok_class(tokenizer)
        detok.reset()
        start = time.perf_counter()
        for t in generated_tokens:
            detok.add_token(t)
            _ = detok.last_segment
        detok.finalize()
        elapsed = time.perf_counter() - start
        streaming_times.append(elapsed)

    streaming_mean = statistics.mean(streaming_times) * 1000

    # Results
    speedup = naive_mean / streaming_mean
    time_saved = naive_mean - streaming_mean

    print()
    print("=" * 70)
    print(f" RESULTS: {len(generated_tokens)} tokens, {iterations} iterations")
    print("=" * 70)
    print(f"{'Method':<25} {'Time':>12} {'Speedup':>10}")
    print("-" * 70)
    print(f"{'Naive decode():':<25} {naive_mean:>10.2f}ms {'1.00x':>10}")
    print(f"{'Streaming detokenizer:':<25} {streaming_mean:>10.2f}ms {speedup:>9.2f}x")
    print("-" * 70)
    print(f"{'Time saved per request:':<25} {time_saved:>10.2f}ms")
    print(
        f"{'Per-token savings:':<25} {(time_saved/len(generated_tokens)*1000):>10.1f}µs"
    )
    print()

    # Verify correctness (strip for BPE edge cases with leading/trailing spaces)
    print("Verifying correctness...")
    detok = detok_class(tokenizer)
    detok.reset()
    for t in generated_tokens:
        detok.add_token(t)
    detok.finalize()

    batch_result = tokenizer.decode(generated_tokens)
    # BPE tokenizers may have minor edge case differences with spaces
    # Compare stripped versions for functional correctness
    streaming_stripped = detok.text.strip()
    batch_stripped = batch_result.strip()
    if streaming_stripped == batch_stripped:
        print("  ✓ Streaming output matches batch decode")
    elif streaming_stripped in batch_stripped or batch_stripped in streaming_stripped:
        print("  ✓ Streaming output matches (minor BPE edge case)")
    else:
        # Check if most of the content matches (BPE edge cases at boundaries)
        common_len = min(len(streaming_stripped), len(batch_stripped)) - 10
        if (
            common_len > 0
            and streaming_stripped[:common_len] == batch_stripped[:common_len]
        ):
            print("  ✓ Streaming output matches (BPE boundary difference)")
        else:
            print("  ✗ MISMATCH! Results differ")
            print(f"    Streaming: {repr(detok.text[:100])}...")
            print(f"    Batch: {repr(batch_result[:100])}...")


def _check_macos_compat():
    """mlxstudio#90 — bundled MLX is compiled with `minos 26.0`. macOS < 26
    crashes on import with a cryptic
        Symbol not found: __ZNSt13exception_ptr31__from_native_exception_pointerEPv
    libc++ symbol that landed in macOS 14.5+. Surface a clear actionable
    error before the import.
    """
    import platform, sys
    if sys.platform != "darwin":
        return
    try:
        ver = tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2])
    except Exception:
        return
    if ver < (14, 5):
        sys.stderr.write(
            f"vmlx requires macOS >= 14.5 (you are on {platform.mac_ver()[0]}).\n"
            "The bundled MLX runtime depends on libc++ symbols added in "
            "macOS 14.5; older systems will hit:\n"
            "  Symbol not found: __ZNSt13exception_ptr31__from_native_"
            "exception_pointerEPv\n"
            "Upgrade to macOS 14.5+ (Sonoma) or 15.x (Sequoia) and retry.\n"
        )
        sys.exit(2)


def _check_no_duplicate_mlx():
    """vmlx#120 / mlxstudio#101 — `nanobind error: refusing to add duplicate
    key "cpu" to enumeration "mlx.core.DeviceType"` happens when the MLX C
    extension is loaded twice in one process. Most common cause: two distinct
    `mlx` packages on sys.path (e.g. bundled python + a stale user-site
    install). The Electron panel sets PYTHONNOUSERSITE=1 and `-s`, but if
    the user pip-installed `mlx` into the bundled python directly OR has
    a non-standard PYTHONPATH, both copies show up. Detect and abort with
    a clear message.
    """
    import importlib.util, sys
    try:
        # Look for ALL mlx packages on the path. Spec.submodule_search_locations
        # gives every directory contributing to the namespace.
        spec = importlib.util.find_spec("mlx")
        if spec is None:
            return
        locations = list(spec.submodule_search_locations or [])
        # Also walk sys.path for shadowed copies.
        seen = []
        for p in sys.path:
            try:
                cand = __import__("os").path.join(p, "mlx")
                if __import__("os").path.isdir(cand) and cand not in locations:
                    locations.append(cand)
                    seen.append(cand)
            except Exception:
                pass
        if len(set(locations)) > 1:
            sys.stderr.write(
                "vmlx detected MULTIPLE `mlx` package locations on sys.path. "
                "Loading two MLX C extensions in one process triggers:\n"
                "  Critical nanobind error: refusing to add duplicate key \"cpu\" "
                "to enumeration \"mlx.core.DeviceType\"\n"
                "Locations found:\n"
            )
            for p in set(locations):
                sys.stderr.write(f"  - {p}\n")
            sys.stderr.write(
                "\nFix:\n"
                "  1. Uninstall any extra mlx wheels: `pip uninstall mlx mlx-lm`\n"
                "  2. Use only the bundled python (vmlx-engine spawns with -s + "
                "PYTHONNOUSERSITE=1)\n"
                "  3. Unset PYTHONPATH if it's pointing into a second site-packages\n"
            )
            sys.exit(2)
    except Exception:
        pass


def main():
    _check_macos_compat()
    _check_no_duplicate_mlx()
    parser = argparse.ArgumentParser(
        description="vmlx-engine: Apple Silicon MLX backend for vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vmlx-engine serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
  vmlx-engine bench mlx-community/Llama-3.2-1B-Instruct-4bit --num-prompts 10
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible server")
    serve_parser.add_argument(
        "model", type=str,
        help="HuggingFace model name or local path to serve. "
             "Example: mlx-community/Llama-3.2-3B-Instruct-4bit",
    )
    serve_parser.add_argument(
        "--served-model-name", type=str, default=None,
        help="Custom model name exposed via /v1/models API. Clients use this name in requests. "
             "Useful for aliasing long model paths. Default: auto-extracted from model path.",
    )
    serve_parser.add_argument(
        "--image-quantize", type=int, default=None, choices=[3, 4, 5, 6, 8],
        help="Quantization bits for image models (mflux). "
             "Lower = less memory, faster. 4-bit recommended. (default: auto-detect from model name)",
    )
    serve_parser.add_argument(
        "--image-mode", type=str, default=None, choices=["generate", "edit"],
        help="Image model mode. 'generate' for text-to-image, 'edit' for image editing "
             "(Kontext, Fill, Qwen-Image-Edit). (default: auto-detect from model name)",
    )
    serve_parser.add_argument(
        "--mflux-class", type=str, default=None,
        help="Explicit mflux model class name (e.g., Flux1, Flux2Klein, ZImage, QwenImageEdit). "
             "Used by the image loader to pick the correct Python class. "
             "If not set, falls back to name-based lookup.",
    )
    serve_parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Network interface to bind. '0.0.0.0' = all interfaces (LAN access). "
             "'127.0.0.1' = localhost only (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000,
        help="TCP port for the API server (default: 8000). Example: --port 8092",
    )
    serve_parser.add_argument(
        "--max-num-seqs", type=int, default=256,
        help="Maximum number of requests that can be processed simultaneously. Higher values "
             "use more memory but support more concurrent users. Requires --continuous-batching. "
             "(default: 256)",
    )
    serve_parser.add_argument(
        "--prefill-batch-size", type=int, default=8,
        help="How many new prompts to process at once during the 'prefill' phase "
             "(reading the input). Higher = faster throughput but more memory spikes. "
             "Requires --continuous-batching. (default: 8)",
    )
    serve_parser.add_argument(
        "--prefill-step-size", type=int, default=2048,
        help="Maximum tokens per prefill chunk. Smaller values reduce peak Metal memory "
             "during prefill — required for very large MoE models (e.g. 128-expert Qwen3.5-397B) "
             "that hit Metal single-buffer OOM at long contexts. Try 512 or 256 if prefill "
             "crashes. Requires --continuous-batching. (default: 2048)",
    )
    serve_parser.add_argument(
        "--completion-batch-size", type=int, default=32,
        help="How many responses to generate tokens for simultaneously during the 'decode' phase. "
             "Higher = more throughput for concurrent requests but more memory. "
             "Requires --continuous-batching. (default: 32)",
    )
    serve_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Cache computed KV states for prompt prefixes so repeated system prompts "
             "or conversation history don't need to be recomputed. Saves significant time "
             "on multi-turn conversations. Requires --continuous-batching. (default: enabled)",
    )
    serve_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Turn off prefix caching. Useful for debugging or if memory is very limited.",
    )
    serve_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Maximum number of cached prompt prefixes (legacy entry-count mode only, "
             "ignored when memory-aware cache is active). (default: 100)",
    )
    serve_parser.add_argument(
        "--prefix-cache-max-bytes",
        type=int,
        default=None,
        help="Optional global byte budget for the prefix cache. When set, eviction "
             "fires when total cached bytes exceed this. Eviction priority is "
             "assistant → user → system, so shared system prompts persist across "
             "users/sessions. None = unlimited (entry-count only). (default: None)",
    )
    serve_parser.add_argument(
        "--no-state-machine-stops",
        action="store_true",
        help="Disable token-level reasoning/stop detection via SequenceStateMachine "
             "(Phase 3c) and fall back to the legacy substring '<think>' scan. "
             "Useful for rollback if a regression appears. (default: state machine ON)",
    )
    # Memory-aware cache options
    serve_parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Fixed memory budget for prefix cache in megabytes. When set, cached prompts "
             "are evicted based on actual memory usage rather than entry count. "
             "Example: --cache-memory-mb 4096 for 4GB. Default: auto-detect (~30%% of available RAM).",
    )
    serve_parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=0.30,
        help="Fraction of available unified memory to use for prefix cache when auto-detecting. "
             "Only used when --cache-memory-mb is not set. Value is a decimal: 0.30 = 30%%. "
             "Example: 0.50 for half of RAM. (default: 0.30)",
    )
    serve_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Fall back to simple entry-count-based cache eviction instead of memory-aware. "
             "Not recommended — memory-aware cache is better for most workloads.",
    )
    serve_parser.add_argument(
        "--cache-ttl-minutes",
        type=float,
        default=0,
        help="Automatically evict cache entries not accessed within this many minutes. "
             "Useful for long-running servers to prevent stale caches from consuming memory. "
             "0 = never expire, entries only evicted when cache is full. (default: 0)",
    )
    serve_parser.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="How many tokens to generate before sending a streaming update to the client. "
             "1 = send every token (smoothest typing effect). Higher values batch tokens "
             "for slightly better throughput. Requires --continuous-batching. (default: 1)",
    )
    serve_parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Default maximum number of tokens the model will generate per request. "
             "Can be overridden per-request via the 'max_tokens' API parameter. "
             "Higher values allow longer responses but use more memory. (default: 32768)",
    )
    serve_parser.add_argument(
        "--continuous-batching",
        action="store_true",
        help="Process multiple requests simultaneously using continuous batching. "
             "Required for: prefix caching, paged cache, KV quantization, and concurrent users. "
             "Without this, requests are processed one at a time (faster for single user). "
             "Example: vmlx-engine serve model --continuous-batching",
    )
    # Paged cache options
    serve_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use block-based (paged) KV cache management. Splits cached prompts into "
             "fixed-size blocks that can be shared across requests with common prefixes. "
             "Reduces memory fragmentation and improves cache utilization for multi-user "
             "workloads. Requires --continuous-batching.",
    )
    serve_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Number of tokens per cache block when using paged cache. Smaller blocks = "
             "more granular sharing but higher overhead. Larger blocks = less overhead but "
             "waste space on short prompts. (default: 64)",
    )
    serve_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks to keep in memory. Each block holds KV states "
             "for --paged-cache-block-size tokens. Total cache capacity = blocks × block_size. "
             "(default: 1000, i.e. 64,000 tokens with default block size)",
    )
    # KV cache quantization
    serve_parser.add_argument(
        "--kv-cache-quantization",
        type=str,
        default="none",
        choices=["none", "q4", "q8"],
        help="Compress stored KV cache to reduce unified memory usage by 2-4x. "
             "q8 = 8-bit (minimal quality loss, ~2x savings). "
             "q4 = 4-bit (slight quality loss, ~4x savings). "
             "Cache is stored compressed but decompressed for generation (no inference slowdown). "
             "Requires --continuous-batching. (default: none)",
    )
    serve_parser.add_argument(
        "--kv-cache-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization. Smaller = better accuracy but larger "
             "metadata overhead. Only used when --kv-cache-quantization is q4 or q8. (default: 64)",
    )
    # Disk cache options
    serve_parser.add_argument(
        "--enable-disk-cache",
        action="store_true",
        help="Persist prompt KV caches to SSD so they survive server restarts. Acts as "
             "an L2 cache: on L1 (in-memory) miss, checks disk before recomputing. "
             "Great for system prompts and repeated conversations. "
             "Requires --continuous-batching.",
    )
    serve_parser.add_argument(
        "--disk-cache-dir",
        type=str,
        default=None,
        help="Directory to store disk cache files. Each cached prompt becomes a .safetensors file. "
             "(default: ~/.cache/vmlx-engine/prompt-cache)",
    )
    serve_parser.add_argument(
        "--disk-cache-max-gb",
        type=float,
        default=10.0,
        help="Maximum total size of disk cache in gigabytes. Oldest entries are evicted "
             "when the limit is exceeded. 0 = unlimited. (default: 10)",
    )
    # Block-level disk cache (L2 for paged cache)
    serve_parser.add_argument(
        "--enable-block-disk-cache",
        action="store_true",
        help="Persist individual paged cache blocks to SSD. When a block is evicted from "
             "memory, it's saved to disk and reloaded on the next hit instead of recomputing. "
             "Requires --use-paged-cache. Great for large multi-user workloads.",
    )
    serve_parser.add_argument(
        "--block-disk-cache-dir",
        type=str,
        default=None,
        help="Directory for block disk cache files. (default: ~/.cache/vmlx-engine/block-cache/<model_hash>)",
    )
    serve_parser.add_argument(
        "--block-disk-cache-max-gb",
        type=float,
        default=10.0,
        help="Maximum total size of block disk cache in GB. 0 = unlimited. (default: 10)",
    )
    # Smelt mode (partial expert loading for MoE models)
    serve_parser.add_argument(
        "--smelt",
        action="store_true",
        help="Enable Smelt mode: load backbone + N%% of MoE experts from SSD. "
             "Reduces RAM by ~50%% while maintaining ~97%% baseline speed via "
             "cache-biased routing and native SwitchGLU kernels.",
    )
    serve_parser.add_argument(
        "--smelt-experts",
        type=int,
        default=50,
        help="Percentage of experts to load per MoE layer (10-100). "
             "Lower = less RAM, more routing bias. (default: 50)",
    )
    # Flash MoE SSD streaming (on-demand expert loading)
    serve_parser.add_argument(
        "--flash-moe",
        action="store_true",
        default=False,
        help="Enable Flash MoE: stream expert weights from SSD on-demand. "
             "Enables massive MoE models (35B-397B) to run with limited RAM "
             "by keeping only active experts in a slot-bank cache.",
    )
    serve_parser.add_argument(
        "--flash-moe-slot-bank",
        type=int,
        default=64,
        help="Number of expert weight sets to cache in RAM (default: 64). "
             "Higher = more cache hits but more RAM usage.",
    )
    serve_parser.add_argument(
        "--flash-moe-prefetch",
        type=str,
        choices=["none", "temporal"],
        default="none",
        help="Expert prefetching strategy (default: none). "
             "'temporal' prefetches recently-used experts.",
    )
    serve_parser.add_argument(
        "--flash-moe-io-split",
        type=int,
        default=4,
        help="Number of parallel I/O threads for expert loading (default: 4).",
    )
    # Distributed compute options
    serve_parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Enable distributed inference (coordinator mode). Discovers worker "
             "nodes via Bonjour and splits the model across them using pipeline "
             "parallelism. Workers must be running vmlx-worker on the same network.",
    )
    serve_parser.add_argument(
        "--cluster-secret",
        type=str,
        default="",
        help="Shared secret for authenticating worker nodes. All workers in the "
             "cluster must use the same secret.",
    )
    serve_parser.add_argument(
        "--distributed-mode",
        type=str,
        choices=["pipeline", "tensor"],
        default="pipeline",
        help="Parallelism mode: 'pipeline' splits layers across nodes (default), "
             "'tensor' splits weights within layers (requires high bandwidth).",
    )
    serve_parser.add_argument(
        "--worker-nodes",
        type=str,
        default=None,
        help="Comma-separated list of worker node addresses (ip:port). "
             "Use instead of Bonjour auto-discovery. Example: 192.168.1.10:9100,192.168.1.11:9100",
    )
    # MCP options
    serve_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP (Model Context Protocol) config file (JSON/YAML). Enables the model "
             "to call external tools (web search, code execution, etc.) via MCP servers. "
             "Tools appear in /v1/mcp/tools and can be used in chat completions with tool_choice.",
    )
    # Security options
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Require this API key for all requests. Clients must send it as "
             "'Authorization: Bearer <key>'. Without this, anyone with network access "
             "can use your model. Example: --api-key sk-my-secret-key",
    )
    serve_parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Maximum requests per minute per client IP. Prevents abuse from a single "
             "client overwhelming the server. 0 = no limit. Example: --rate-limit 60",
    )
    serve_parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Maximum time in seconds for a single request before it's cancelled. "
             "Long prompts or high max_tokens may need higher values. (default: 300 = 5 minutes)",
    )
    # API Gateway routing & lifecycles
    serve_parser.add_argument(
        "--inference-endpoints",
        type=str,
        default=None,
        help="Comma-separated list of API endpoints that should be treated as actual inference endpoints. "
             "Requests to these endpoints will keep the server awake and reset the idle timeout. "
             "If not set, uses a default list covering OpenAI, MLX, and Ollama endpoints.",
    )
    serve_parser.add_argument(
        "--wake-timeout",
        type=int,
        default=300,
        help="Maximum time in seconds to wait for a model to reload from deep sleep during a JIT wake. "
             "Larger models may need higher values to finish loading. (default: 300)",
    )
    # Tool calling options
    serve_parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        help="Let the model decide when to call tools (function calling). Must be combined "
             "with --tool-call-parser to specify how the model formats tool calls. "
             "Example: --enable-auto-tool-choice --tool-call-parser qwen",
    )
    serve_parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        choices=[
            "auto",
            "none",
            # Primary names
            "mistral",
            "qwen",
            "llama",
            "hermes",
            "deepseek",
            "kimi",
            "granite",
            "nemotron",
            "minimax",
            "xlam",
            "functionary",
            "glm47",
            "step3p5",
            # DeepSeek V4 DSML format (<｜DSML｜invoke name="…">)
            "dsml",
            "deepseek_v4",
            # Aliases (map to same parsers)
            "generic",
            "qwen3",
            "llama3",
            "llama4",
            "nous",
            "deepseek_v3",
            "deepseek_r1",
            "kimi_k2",
            "moonshot",
            "granite3",
            "nemotron3",
            "minimax_m2",
            "meetkai",
            "stepfun",
            "glm4",
            "gemma4",
        ],
        help="Which format to use for parsing tool calls from model output. Must match your "
             "model's training format. Common choices: 'qwen' for Qwen models, 'llama' for "
             "Llama 3+, 'hermes' for Hermes/NousResearch, 'mistral' for Mistral. "
             "Use 'auto' to detect from model name. Requires --enable-auto-tool-choice.",
    )
    # Reasoning parser options - choices loaded dynamically from registry
    from .reasoning import list_parsers

    reasoning_choices = list_parsers()
    serve_parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        choices=["auto", "none"] + reasoning_choices,
        help="Extract reasoning/thinking content from model output. Reasoning models like "
             "Qwen3 and DeepSeek-R1 wrap their thinking in special tags. This parser extracts "
             "that content into a separate 'reasoning_content' field in the API response. "
             "'auto' = detect from model name. 'none' = explicitly disable. "
             f"Parsers: {', '.join(reasoning_choices)}.",
    )
    serve_parser.add_argument(
        "--is-mllm",
        action="store_true",
        help="Force the model to load as a multimodal (vision) model even if auto-detection "
             "doesn't recognize it. Use this for VLM models that aren't in the built-in registry. "
             "Auto-detection checks config.json for 'vision_config'.",
    )
    # Nemotron-3-Nano-Omni multimodal backend selector. Exposes the
    # VMLX_OMNI_BACKEND env var as a first-class CLI flag so the panel +
    # Swift settings UI can flip between the bit-exact Stage-1 PyTorch
    # bridge (default, correct) and the ~15–21× faster Stage-2 native
    # MLX path (RADIO bilinear pos_embed + Parakeet Conformer in MLX,
    # zero PyTorch in the hot path). User reported the JANGQ-AI banner
    # speed claims (1.4 s image / 2.1 s 20-s audio / 3.6 s 8-frame
    # video / 82 tok/s decode) — those are Stage-2 numbers; Stage-1 is
    # 15-21× slower on encoders. Default stays Stage-1 until Wave-4
    # rel-pos parity validation completes; users can opt in here.
    serve_parser.add_argument(
        "--omni-backend",
        type=str,
        default=None,
        choices=["stage1", "stage2", "pytorch", "mlx"],
        help="Nemotron-Omni multimodal backend. 'stage1'/'pytorch' = bit-exact "
             "PyTorch+MPS encoder bridge (default, slower). 'stage2'/'mlx' = "
             "native MLX RADIO + Parakeet (~15-21x faster encoders, ~82 tok/s "
             "decode on M4 Max — the JANGQ-AI banner numbers). Stage-2 "
             "default-off pending Wave-4 quality validation; opt in for "
             "benchmarking. Equivalent to setting VMLX_OMNI_BACKEND env var.",
    )
    # Embedding model option
    serve_parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Pre-load a separate embedding model at startup for /v1/embeddings endpoint. "
             "Runs alongside the main chat model. Example: --embedding-model "
             "mlx-community/embeddinggemma-300m-6bit",
    )
    serve_parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="Server-wide default temperature for generation. Controls randomness: "
             "0.0 = deterministic, 1.0 = creative. Overridden by per-request 'temperature'. "
             "If not set, uses 0.7 as fallback.",
    )
    serve_parser.add_argument(
        "--default-top-p",
        type=float,
        default=None,
        help="Server-wide default top-p (nucleus sampling) for generation. Only considers "
             "tokens whose cumulative probability ≤ this value: 0.9 = use top 90%% of probability mass. "
             "Lower = more focused, higher = more diverse. Overridden by per-request 'top_p'. "
             "If not set, uses model default.",
    )
    # vmlx#75: reporter's `vmlx-engine serve ... --default-repetition-penalty
    # 1.10` failed with `unrecognized arguments` on older binaries. The flag
    # is registered here and validated in serve_command() (0.5..2.0 range
    # matches user's 1.10 value). Paired with mlxstudio#76 diagnostic hint
    # for cases where an older vmlx-engine shadows this one on PATH.
    serve_parser.add_argument(
        "--default-repetition-penalty",
        type=float,
        default=None,
        help="Server-wide default repetition penalty for generation. 1.0 = no penalty, "
             "1.1 = mild anti-loop (recommended — prevents Gemma 4 word-loops and 2-bit "
             "quant dash-loops), higher = less repetition at the cost of fluency. "
             "Overridden by per-request 'repetition_penalty'. If not set, no penalty is "
             "applied server-wide.",
    )
    serve_parser.add_argument(
        "--default-enable-thinking",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Server-wide default for thinking/reasoning mode. 'true' enables thinking, "
             "'false' disables it. Overridden by per-request 'enable_thinking'. "
             "If not set, uses model's default behavior.",
    )
    serve_parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Custom Jinja2 chat template string to override the model's built-in template. "
             "Useful for models whose templates are incompatible with certain clients "
             "(e.g., JetBrains AI Chat). The template receives 'messages' and 'add_generation_prompt' variables.",
    )
    serve_parser.add_argument(
        "--chat-template-kwargs",
        type=str,
        default=None,
        help='Server-wide default kwargs passed to the chat template. JSON string, e.g. '
             '\'{"enable_thinking": false}\'. Per-request chat_template_kwargs override these. '
             "Common use: disable thinking for reasoning models like Qwen3.",
    )
    # JIT compilation
    serve_parser.add_argument(
        "--enable-jit",
        action="store_true",
        default=False,
        help="Enable JIT compilation (mx.compile) on the model forward pass. "
             "This fuses Metal operations for faster inference after a one-time warmup. "
             "May not work with all models. Falls back gracefully if compilation fails.",
    )

    # Speculative decoding options
    serve_parser.add_argument(
        "--speculative-model",
        type=str,
        default=None,
        help="Path or HuggingFace name of a small draft model for speculative decoding. "
             "The draft model proposes tokens that the main model verifies in a single "
             "forward pass, giving 20-90%% speedup with zero quality loss. Must use the "
             "same tokenizer as the main model. Example: --speculative-model mlx-community/Llama-3.2-1B-Instruct-4bit",
    )
    serve_parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=3,
        help="Number of tokens the draft model proposes per speculative decoding step. "
             "Higher values = more potential speedup but lower acceptance rate. "
             "Typical sweet spot is 2-5. (default: 3)",
    )

    # Prompt Lookup Decoding (PLD)
    serve_parser.add_argument(
        "--enable-pld",
        action="store_true",
        default=False,
        help="Enable Prompt Lookup Decoding speculative acceleration. "
             "PLD verifies n-gram draft tokens in a single forward pass, giving ~5-8x speedup "
             "on long structured or repetitive output (code, JSON, summarisation). "
             "Net-neutral or negative on short novel prompts. Check 'eff tok/pass' in the "
             "[PLD:3b1f] log summary to evaluate impact for your workload.",
    )
    serve_parser.add_argument(
        "--pld-summary-interval",
        type=int,
        default=487,
        help="Log a PLD effectiveness summary every N spec-decode tokens. "
             "The summary reports accept rate, full/zero %%, and eff tok/pass "
             "(>1.0 = net gain over baseline). (default: 487)",
    )
    # Logging
    serve_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Server log verbosity. DEBUG shows all details, ERROR shows only errors. (default: INFO)",
    )
    # CORS
    serve_parser.add_argument(
        "--allowed-origins",
        type=str,
        default="*",
        help="Comma-separated list of allowed CORS origins for browser-based API consumers. "
             "Use * to allow all origins. Example: http://localhost:3000,https://myapp.com (default: *)",
    )
    # Bench command
    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument("model", type=str, help="Model to benchmark")
    bench_parser.add_argument(
        "--num-prompts", type=int, default=10, help="Number of prompts"
    )
    bench_parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens per prompt"
    )
    bench_parser.add_argument(
        "--max-num-seqs", type=int, default=32, help="Max concurrent sequences"
    )
    bench_parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Prefill batch size"
    )
    bench_parser.add_argument(
        "--prefill-step-size", type=int, default=2048,
        help="Tokens per prefill chunk — reduce for large MoE models hitting Metal OOM (default: 2048)"
    )
    bench_parser.add_argument(
        "--completion-batch-size", type=int, default=16, help="Completion batch size"
    )
    bench_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching (default: enabled)",
    )
    bench_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    bench_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100, legacy mode only)",
    )
    # Memory-aware cache options (recommended for large models)
    bench_parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Cache memory limit in MB (default: auto-detect ~30%% of RAM)",
    )
    bench_parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=0.30,
        help="Fraction of available RAM for cache if auto-detecting (default: 0.30)",
    )
    bench_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache, use legacy entry-count based cache",
    )
    bench_parser.add_argument(
        "--cache-ttl-minutes",
        type=float,
        default=0,
        help="Cache entry time-to-live in minutes. 0=disabled (default: 0)",
    )
    # Paged cache options (experimental)
    bench_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use paged KV cache for memory efficiency (experimental)",
    )
    bench_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    bench_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )
    # KV cache quantization
    bench_parser.add_argument(
        "--kv-cache-quantization",
        type=str,
        default="none",
        choices=["none", "q4", "q8"],
        help="Quantize KV cache to reduce GPU memory (~2-4x). q8=8-bit, q4=4-bit (default: none)",
    )
    bench_parser.add_argument(
        "--kv-cache-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    # Disk cache options (bench)
    bench_parser.add_argument(
        "--enable-disk-cache",
        action="store_true",
        help="Save prompt caches to disk for reuse across runs",
    )
    bench_parser.add_argument(
        "--disk-cache-dir",
        type=str,
        default=None,
        help="Directory for disk cache files (default: ~/.cache/vmlx-engine/prompt-cache)",
    )
    bench_parser.add_argument(
        "--disk-cache-max-gb",
        type=float,
        default=10.0,
        help="Maximum disk cache size in GB (default: 10)",
    )
    bench_parser.add_argument(
        "--enable-block-disk-cache",
        action="store_true",
        help="Enable block-level disk persistence (requires --use-paged-cache)",
    )
    bench_parser.add_argument(
        "--block-disk-cache-dir",
        type=str,
        default=None,
        help="Directory for block disk cache",
    )
    bench_parser.add_argument(
        "--block-disk-cache-max-gb",
        type=float,
        default=10.0,
        help="Maximum block disk cache size in GB (default: 10)",
    )
    bench_parser.add_argument(
        "--smelt",
        action="store_true",
        help="Enable Smelt mode for benchmarking with partial expert loading.",
    )
    bench_parser.add_argument(
        "--smelt-experts",
        type=int,
        default=50,
        help="Percentage of experts to load per MoE layer (default: 50)",
    )
    bench_parser.add_argument(
        "--flash-moe",
        action="store_true",
        default=False,
        help="Enable Flash MoE SSD streaming for benchmarking.",
    )
    bench_parser.add_argument(
        "--flash-moe-slot-bank",
        type=int,
        default=64,
        help="Flash MoE slot bank size (default: 64)",
    )

    # Detokenizer benchmark
    detok_parser = subparsers.add_parser(
        "bench-detok", help="Benchmark streaming detokenizer optimization"
    )
    detok_parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default="mlx-community/Qwen3-0.6B-8bit",
        help="Model to use for tokenizer (default: mlx-community/Qwen3-0.6B-8bit)",
    )
    detok_parser.add_argument(
        "--iterations", type=int, default=5, help="Benchmark iterations (default: 5)"
    )

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert HuggingFace model to quantized MLX or JANG format"
    )
    convert_parser.add_argument(
        "model", type=str,
        help="HuggingFace model name or local path to convert",
    )
    convert_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory (default: auto-generated from model name)",
    )

    # MLX uniform quantization options
    convert_parser.add_argument(
        "--bits", "-b", type=int, default=None, choices=[2, 3, 4, 6, 8],
        help="MLX uniform quantization bit-width (mutually exclusive with --jang-profile)",
    )
    convert_parser.add_argument(
        "--group-size", type=int, default=64,
        help="Quantization group size (default: 64)",
    )
    convert_parser.add_argument(
        "--mode", type=str, default="default",
        choices=["default", "NF4"],
        help="Quantization mode (default: default)",
    )

    # JANG adaptive quantization options
    convert_parser.add_argument(
        "--jang-profile", "-j", type=str, default=None,
        help="JANG adaptive profile (e.g. JANG_2S, JANG_3M, JANG_4L). "
             "Automatically assigns different bit widths to attention vs MLP tensors.",
    )
    convert_parser.add_argument(
        "--jang-method", type=str, default="mse",
        choices=["mse", "rtn", "mse-all"],
        help="JANG quantization method: mse (MSE for attention, RTN for MLP), "
             "rtn (fast), mse-all (MSE everywhere, slow). Default: mse",
    )
    # Advanced JANG options (beginners can ignore these)
    convert_parser.add_argument(
        "--calibration-method", type=str, default="weights",
        choices=["weights", "activations"],
        help="How to score tensor importance. 'weights' is fast (default). "
             "'activations' gives better quality at 2-3 bit but requires more time. "
             "Most users should leave this at the default.",
    )
    convert_parser.add_argument(
        "--imatrix-path", type=str, default=None,
        help="Path to a pre-computed importance matrix (.safetensors). "
             "Skips calibration step. Advanced: use if you've pre-calibrated with custom data.",
    )
    convert_parser.add_argument(
        "--use-awq", action="store_true", default=False,
        help="Enable AWQ (Activation-Aware Weighting) scaling for better quality. "
             "Experimental. Adds activation norm collection step before quantization.",
    )
    convert_parser.add_argument(
        "--awq-alpha", type=float, default=0.25,
        help="AWQ scaling factor (0.0-1.0). Only used when --use-awq is enabled. "
             "Higher values increase activation awareness. Default: 0.25.",
    )

    convert_parser.add_argument(
        "--dtype", type=str, default=None,
        help="Non-quantized layer dtype (e.g. float16, bfloat16)",
    )
    convert_parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output directory",
    )
    convert_parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip post-conversion smoke test",
    )
    convert_parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Trust remote code from HuggingFace",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Display model metadata"
    )
    info_parser.add_argument(
        "model", type=str,
        help="Model path or HuggingFace name to inspect",
    )

    # List command
    list_parser = subparsers.add_parser(
        "list", help="List models in a directory"
    )
    list_parser.add_argument(
        "directory", type=str,
        help="Directory to scan for models",
    )

    # Doctor command
    doctor_parser = subparsers.add_parser(
        "doctor", help="Run diagnostics on a model directory"
    )
    doctor_parser.add_argument(
        "model", type=str,
        help="Model path or HuggingFace name to diagnose",
    )
    doctor_parser.add_argument(
        "--no-inference", action="store_true",
        help="Skip inference smoke test",
    )

    # mlxstudio#76: parse_args() exits with code 2 on unrecognized args,
    # which is opaque to end users running via the panel's process spawner.
    # When a user's vmlx-engine binary is shadowed by an older install on
    # PATH, flags like --smelt or --default-repetition-penalty (added in
    # newer releases) get rejected. The panel logs "unrecognized arguments"
    # but the user has no obvious path to diagnose. Catch SystemExit(2)
    # from argparse and emit a one-line hint telling them to check for
    # a stale binary; then re-raise so argparse's own message still shows.
    try:
        args = parser.parse_args()
    except SystemExit as _argp_exit:
        if _argp_exit.code == 2:
            # Argparse already wrote "error: unrecognized arguments: ..." to
            # stderr. Append a diagnostic tail so users can act on it.
            try:
                import sys as _sys
                from . import __version__ as _vmlx_ver
            except Exception:
                _vmlx_ver = "unknown"
            _sys.stderr.write(
                f"\n"
                f"hint: vmlx_engine version is {_vmlx_ver}. "
                f"If this flag was recently added and you expect it to "
                f"work, verify no older `vmlx-engine` is shadowing the "
                f"bundled one on your PATH:\n"
                f"    which -a vmlx-engine\n"
                f"    python -c 'import vmlx_engine; print(vmlx_engine.__file__)'\n"
                f"    pip show vmlx\n"
                f"See mlxstudio#76.\n"
            )
        raise

    if args.command == "serve":
        serve_command(args)
    elif args.command == "bench":
        bench_command(args)
    elif args.command == "bench-detok":
        bench_detok_command(args)
    elif args.command == "convert":
        from .commands.convert import convert_command
        convert_command(args)
    elif args.command == "info":
        from .commands.info import info_command
        info_command(args)
    elif args.command == "list":
        from .commands.list import list_command
        list_command(args)
    elif args.command == "doctor":
        from .commands.doctor import doctor_command
        doctor_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
