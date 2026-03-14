# SPDX-License-Identifier: Apache-2.0
"""Model diagnostics command for vmlx-engine.

Checks model health:
- Config validation (required fields, architecture detection)
- Weight file integrity (missing/orphaned tensors, file corruption)
- Quick inference test (loads model, generates tokens)
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger("vmlx_engine")


def doctor_command(args) -> None:
    """
    Run diagnostics on a model directory.

    Checks:
    1. Config.json validity and required fields
    2. Weight file integrity (safetensors keys)
    3. Architecture-specific validation
    4. Optional inference smoke test
    """
    from ..utils.model_inspector import (
        format_model_info,
        inspect_model,
        resolve_model_path,
    )

    model_input = args.model
    issues = []
    warnings = []

    # --- 1. Resolve model path ---
    print(f"Examining: {model_input}")
    try:
        model_path = resolve_model_path(model_input)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # --- 2. Inspect model ---
    try:
        info = inspect_model(model_path)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print()
    print(format_model_info(info))
    print()

    # --- 3. Config checks ---
    print("Checking config...")
    config_issues = _check_config(info)
    for issue in config_issues:
        issues.append(f"Config: {issue}")
    if not config_issues:
        print("  Config: OK")

    # --- 4. Weight checks ---
    print("Checking weights...")
    weight_issues, weight_warnings = _check_weights(model_path, info)
    for issue in weight_issues:
        issues.append(f"Weights: {issue}")
    for warn in weight_warnings:
        warnings.append(f"Weights: {warn}")
    if not weight_issues and not weight_warnings:
        print("  Weights: OK")

    # --- 5. Architecture-specific checks ---
    print("Checking architecture...")
    arch_issues = _check_architecture(info)
    for issue in arch_issues:
        issues.append(f"Architecture: {issue}")
    if not arch_issues:
        print("  Architecture: OK")

    # --- 6. Inference test ---
    if not args.no_inference:
        print("Running inference test...")
        success, message = _run_inference_test(model_path)
        if success:
            print(f"  Inference: PASS - {message}")
        else:
            issues.append(f"Inference: {message}")
    else:
        print("  Inference: skipped (--no-inference)")

    # --- Summary ---
    print()
    print("=" * 60)
    if issues:
        print(f"ISSUES FOUND: {len(issues)}")
        for issue in issues:
            print(f"  FAIL: {issue}")
    if warnings:
        print(f"WARNINGS: {len(warnings)}")
        for warn in warnings:
            print(f"  WARN: {warn}")
    if not issues and not warnings:
        print("ALL CHECKS PASSED")
        print("Model appears healthy.")
    elif not issues:
        print("No critical issues found.")
    print("=" * 60)

    if issues:
        sys.exit(1)


def _check_config(info) -> list[str]:
    """Validate config.json has required fields."""
    issues = []

    required_fields = {
        "model_type": info.model_type,
        "hidden_size": info.hidden_size,
        "num_hidden_layers": info.num_layers,
        "vocab_size": info.vocab_size,
    }

    for field_name, value in required_fields.items():
        if not value:
            issues.append(f"Missing or zero: {field_name}")

    if info.model_type == "unknown":
        issues.append("model_type not set in config.json")

    if not info.config.get("architectures"):
        issues.append("'architectures' field missing — model loader may fail")

    return issues


def _check_weights(model_path: str, info) -> tuple[list[str], list[str]]:
    """
    Check weight file integrity.

    Returns:
        (issues, warnings) — issues are failures, warnings are informational
    """
    issues = []
    warnings = []
    path = Path(model_path)

    if not info.weight_files:
        issues.append("No .safetensors files found")
        return issues, warnings

    # Check for weight index
    index_path = path / "model.safetensors.index.json"
    has_index = index_path.exists()

    if len(info.weight_files) > 1 and not has_index:
        warnings.append(
            f"Multiple weight files ({len(info.weight_files)}) "
            f"but no model.safetensors.index.json"
        )

    # Try to enumerate tensor keys
    try:
        from safetensors import safe_open

        all_keys = set()
        for wf in info.weight_files:
            try:
                with safe_open(str(path / wf), framework="numpy") as f:
                    all_keys.update(f.keys())
            except Exception as e:
                issues.append(f"Failed to read {wf}: {e}")

        if all_keys:
            print(f"  Found {len(all_keys)} tensors across {len(info.weight_files)} files")

            # Check for expected patterns
            has_embed = any("embed_tokens" in k or "wte" in k or "embedding.weight" in k for k in all_keys)
            has_lm_head = any("lm_head" in k for k in all_keys)
            has_layers = any("layers.0." in k or "blocks.0." in k or "h.0." in k for k in all_keys)

            if not has_embed:
                warnings.append("No embedding layer found (embed_tokens/wte)")
            if not has_lm_head:
                tie = info.config.get("tie_word_embeddings", False)
                if not tie:
                    warnings.append("No lm_head found and tie_word_embeddings is False")
            if not has_layers:
                issues.append("No layer weights found (layers.0.*/blocks.0.*)")

            # Check layer continuity (handles layers.X, h.X, blocks.X)
            layer_indices = set()
            layer_key_names = {"layers", "h", "blocks"}
            for key in all_keys:
                parts = key.split(".")
                for i, part in enumerate(parts):
                    if part in layer_key_names and i + 1 < len(parts) and parts[i + 1].isdigit():
                        layer_indices.add(int(parts[i + 1]))

            if layer_indices:
                expected = set(range(max(layer_indices) + 1))
                missing_layers = expected - layer_indices
                if missing_layers:
                    issues.append(
                        f"Missing layer weights for layers: "
                        f"{sorted(missing_layers)[:10]}"
                        f"{'...' if len(missing_layers) > 10 else ''}"
                    )
                if max(layer_indices) + 1 != info.num_layers:
                    warnings.append(
                        f"Config says {info.num_layers} layers but "
                        f"weights have {max(layer_indices) + 1}"
                    )

            # LatentMoE-specific checks
            if info.needs_latent_moe:
                latent_keys = [k for k in all_keys if "latent_proj" in k]
                if not latent_keys:
                    issues.append(
                        "Model needs LatentMoE but no fc1_latent_proj/fc2_latent_proj "
                        "weights found. The model may not have been converted correctly."
                    )
                else:
                    print(f"  LatentMoE weights: {len(latent_keys)} tensors found")

    except ImportError:
        warnings.append(
            "safetensors not installed — cannot verify weight integrity. "
            "Install with: pip install safetensors"
        )

    return issues, warnings


def _check_architecture(info) -> list[str]:
    """Architecture-specific validation."""
    issues = []

    # Check for known problematic configurations
    if info.is_moe and info.n_routed_experts and info.n_routed_experts > 256:
        if not info.is_quantized:
            issues.append(
                f"Large MoE with {info.n_routed_experts} experts is not quantized. "
                f"This will require significant memory."
            )

    if info.needs_latent_moe and not info.moe_latent_size:
        issues.append("Model type suggests LatentMoE but moe_latent_size is missing")

    # Check hybrid pattern consistency
    if info.hybrid_pattern:
        pattern_len = len(info.hybrid_pattern)
        if pattern_len != info.num_layers:
            issues.append(
                f"hybrid_override_pattern length ({pattern_len}) doesn't match "
                f"num_hidden_layers ({info.num_layers})"
            )
        invalid_chars = set(info.hybrid_pattern) - {"M", "E", "*"}
        if invalid_chars:
            issues.append(
                f"hybrid_override_pattern has invalid characters: {invalid_chars}"
            )

    return issues


def _run_inference_test(model_path: str) -> tuple[bool, str]:
    """Load model and generate a few tokens."""
    try:
        from ..utils.nemotron_latent_moe import ensure_latent_moe_support
        ensure_latent_moe_support(model_path)

        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = load(model_path)

        import mlx.core as mx
        from mlx_lm.generate import generate_step

        prompt_text = "Hello, world!"
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt = mx.array(prompt_tokens)

        sampler = make_sampler(temp=0.0)
        generated = []
        for step in generate_step(
            prompt=prompt,
            model=model,
            max_tokens=10,
            sampler=sampler,
        ):
            # generate_step yields (token_id, logprobs) tuples
            token = step[0] if isinstance(step, tuple) else step
            generated.append(int(token))

        if not generated:
            return False, "Generated no tokens"

        output = tokenizer.decode(generated)
        if not output.strip():
            return False, "Generated empty/whitespace output"

        return True, f"Generated {len(generated)} tokens: '{output.strip()[:60]}'"

    except Exception as e:
        return False, f"Inference failed: {e}"
