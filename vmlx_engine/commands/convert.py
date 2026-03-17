# SPDX-License-Identifier: Apache-2.0
"""Model conversion command for vmlx-engine.

Converts HuggingFace models to quantized MLX format with:
- Automatic LatentMoE patching for Nemotron-H models
- Pre-flight memory checks
- Post-conversion verification via smoke test
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("vmlx_engine")


def convert_command(args: argparse.Namespace) -> None:
    """
    Convert a HuggingFace model to quantized MLX or JANG format.

    Flow:
    1. Inspect model metadata (config.json)
    2. Pre-flight memory check
    3. Apply LatentMoE patch if needed
    4. Run mlx_lm.convert.convert() or jang-tools convert
    5. Post-conversion smoke test (unless --skip-verify)
    """
    # Route to JANG conversion if --jang-profile is specified
    if args.jang_profile:
        return _jang_convert_command(args)

    if not args.bits:
        print("Error: Either --bits (MLX uniform) or --jang-profile (JANG adaptive) is required.")
        sys.exit(1)

    from ..utils.model_inspector import (
        available_memory_gb,
        estimate_conversion_memory_gb,
        format_model_info,
        inspect_model,
        resolve_model_path,
    )
    from ..utils.nemotron_latent_moe import ensure_latent_moe_support

    model_input = args.model

    # --- 1. Resolve model path ---
    print(f"Resolving model: {model_input}")
    try:
        model_path = resolve_model_path(model_input)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # --- 1b. Check for GGUF format (not supported) ---
    model_dir = Path(model_path)
    if model_dir.is_dir():
        gguf_files = list(model_dir.glob("*.gguf")) + list(model_dir.glob("*.gguf.part"))
        safetensors_files = list(model_dir.glob("*.safetensors"))
        if gguf_files and not safetensors_files:
            print(f"\nError: This model is in GGUF format, which cannot be converted by vmlx-engine.")
            print(f"  Found: {', '.join(f.name for f in gguf_files[:3])}")
            print(f"\nGGUF models must first be converted to HuggingFace safetensors format")
            print(f"before they can be quantized with vmlx. Use a tool like")
            print(f"'convert-gguf-to-hf' or download the original HuggingFace model instead.")
            sys.exit(1)
    elif str(model_path).lower().endswith('.gguf'):
        print(f"\nError: Single GGUF files cannot be converted. Provide a HuggingFace model directory instead.")
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

    # --- 3. Compute output path ---
    output_path = args.output
    if not output_path:
        output_path = _default_output_name(model_input, args.bits)

    output_dir = Path(output_path)
    if output_dir.exists():
        if not args.force:
            print(f"Error: Output directory already exists: {output_path}")
            print("Use --force to overwrite, or --output to specify a different path.")
            sys.exit(1)
        else:
            import shutil
            print(f"Removing existing output directory: {output_path}")
            shutil.rmtree(output_dir)

    # --- 4. Pre-flight memory check ---
    _preflight_check(info, args.bits)

    # --- 5. Apply LatentMoE patch ---
    if info.needs_latent_moe:
        print("Applying LatentMoE patch for Nemotron-H...")
        ensure_latent_moe_support(model_path)
        print("  LatentMoE patch active")

    print()

    # --- 6. Run conversion ---
    print("=" * 60)
    print(f"Converting: {info.architecture}")
    print(f"  Source: {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Quantization: {args.bits}-bit (group_size={args.group_size}, mode={args.mode})")
    if args.dtype:
        print(f"  Non-quantized dtype: {args.dtype}")
    print("=" * 60)
    print()

    start_time = time.time()

    try:
        _run_conversion(
            hf_path=model_path,
            mlx_path=str(output_dir),
            q_bits=args.bits,
            q_group_size=args.group_size,
            q_mode=args.mode if args.mode != "default" else None,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nError: Conversion failed after {elapsed:.1f}s: {e}")
        print()
        print("Tips:")
        print("  - Check available memory (conversion needs source + target weights)")
        print(f"  - Run 'vmlx-engine doctor {model_input}' to diagnose issues")
        print("  - Ensure the model directory has config.json and .safetensors files")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\nConversion completed in {elapsed:.1f}s")

    # Show output size
    output_files = list(output_dir.glob("*.safetensors"))
    output_size = sum(f.stat().st_size for f in output_files) / (1024**3)
    print(f"Output size: {output_size:.1f} GB ({len(output_files)} files)")
    print(f"Output path: {output_dir.resolve()}")

    # --- 7. Post-conversion smoke test ---
    if not args.skip_verify:
        print()
        print("Running verification smoke test...")
        success, message = _smoke_test(str(output_dir))
        if success:
            print(f"  PASS: {message}")
            print()
            print(f"Model ready! Load with:")
            print(f"  vmlx-engine serve {output_dir}")
        else:
            print(f"  FAIL: {message}")
            print()
            print("The model converted but may produce incorrect output.")
            print(f"Run 'vmlx-engine doctor {output_dir}' for detailed diagnostics.")
            sys.exit(1)
    else:
        print()
        print(f"Verification skipped. To verify later:")
        print(f"  vmlx-engine doctor {output_dir}")


def _default_output_name(model_input: str, bits: int) -> str:
    """
    Generate default output directory name.

    Examples:
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16" → "NVIDIA-Nemotron-3-Super-120B-A12B-BF16-vmlx-4bit"
        "/path/to/model" → "model-vmlx-4bit"
    """
    # Extract model name from path or HF ID
    name = model_input.rstrip("/")
    if "/" in name:
        name = name.split("/")[-1]

    # Remove common suffixes that would be redundant
    for suffix in ["-BF16", "-bf16", "-FP16", "-fp16", "-FP32", "-fp32"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    return f"{name}-vmlx-{bits}bit"


def _preflight_check(info, bits: int) -> None:
    """Print memory estimates and warn if conversion may fail."""
    from ..utils.model_inspector import (
        available_memory_gb,
        estimate_conversion_memory_gb,
        total_memory_gb,
    )

    needed = estimate_conversion_memory_gb(info, bits)
    available = available_memory_gb()
    total = total_memory_gb()

    print(f"Memory estimate:")
    print(f"  Conversion needs: ~{needed:.1f} GB")
    print(f"  Available: {available:.1f} GB / {total:.0f} GB total")

    if needed > total:
        print()
        print(f"  WARNING: This model may be too large for your system.")
        print(f"  Consider a lower bit-width or a machine with more memory.")
        print()
    elif needed > available:
        print()
        print(f"  WARNING: Conversion needs more than currently available memory.")
        print(f"  Close other applications or expect heavy swap usage.")
        print()
    elif needed > available * 0.7:
        print()
        print(f"  WARNING: Conversion will use most of your available memory.")
        print(f"  Close other applications to free up memory.")
        print()
    else:
        print(f"  Status: OK")


def _run_conversion(
    hf_path: str,
    mlx_path: str,
    q_bits: int,
    q_group_size: int,
    q_mode: str | None,
    dtype: str | None,
    trust_remote_code: bool,
) -> None:
    """Run mlx_lm.convert.convert() with the LatentMoE patch active."""
    from mlx_lm.convert import convert

    convert(
        hf_path=hf_path,
        mlx_path=mlx_path,
        quantize=True,
        q_group_size=q_group_size,
        q_bits=q_bits,
        q_mode=q_mode,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )


def _smoke_test(model_path: str) -> tuple[bool, str]:
    """
    Load the converted model and generate a few tokens to verify it works.

    Returns:
        (success, message) tuple
    """
    try:
        from ..utils.nemotron_latent_moe import ensure_latent_moe_support

        # Patch again for the converted model (it has the same config.json)
        ensure_latent_moe_support(model_path)

        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = load(model_path)

        # Generate a few tokens with a simple prompt
        import mlx.core as mx
        from mlx_lm.generate import generate_step

        prompt_text = "The capital of France is"
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt = mx.array(prompt_tokens)

        sampler = make_sampler(temp=0.0)
        generated = []
        for step in generate_step(
            prompt=prompt,
            model=model,
            max_tokens=5,
            sampler=sampler,
        ):
            # generate_step yields (token_id, logprobs) tuples
            token = step[0] if isinstance(step, tuple) else step
            generated.append(int(token))

        if not generated:
            return False, "Model loaded but generated no tokens"

        output_text = tokenizer.decode(generated)

        # Basic sanity: check that output contains recognizable text
        if len(output_text.strip()) == 0:
            return False, "Model generated empty output"

        return True, f"Generated: '{prompt_text}{output_text.rstrip()}'"

    except Exception as e:
        return False, f"Smoke test failed: {e}"


def _jang_smoke_test(model_path: str) -> tuple[bool, str]:
    """Load a JANG model and generate a few tokens to verify it works."""
    try:
        from ..utils.jang_loader import load_jang_model
        model, tokenizer = load_jang_model(model_path)

        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.generate import generate_step

        prompt_text = "The capital of France is"
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt = mx.array(prompt_tokens)

        sampler = make_sampler(temp=0.0)
        generated = []
        for step in generate_step(prompt=prompt, model=model, max_tokens=5, sampler=sampler):
            token = step[0] if isinstance(step, tuple) else step
            generated.append(int(token))

        if not generated:
            return False, "Model loaded but generated no tokens"

        output_text = tokenizer.decode(generated)
        if len(output_text.strip()) == 0:
            return False, "Model generated empty output"

        return True, f"Generated: '{prompt_text}{output_text.rstrip()}'"
    except Exception as e:
        return False, f"JANG smoke test failed: {e}"


def _jang_convert_command(args: argparse.Namespace) -> None:
    """Convert a HuggingFace model to JANG adaptive mixed-precision format."""
    from ..utils.model_inspector import resolve_model_path

    model_input = args.model

    try:
        from jang_tools.allocate import JANG_PROFILES, profile_for_bits
    except ImportError:
        print("\nError: jang-tools not installed. Install with: pip install jang-tools")
        sys.exit(1)

    # Accept profile name, bit number, or custom CUSTOM_C_I_P format
    raw_profile = args.jang_profile
    custom_bits = None
    if raw_profile.upper().startswith('CUSTOM_'):
        # Custom mix: CUSTOM_8_4_3 → critical=8, important=4, compress=3
        parts = raw_profile.split('_')[1:]
        if len(parts) == 3:
            try:
                custom_bits = tuple(int(x) for x in parts)
                # Register as a temporary profile
                profile = f"CUSTOM_{custom_bits[0]}_{custom_bits[1]}_{custom_bits[2]}"
                JANG_PROFILES[profile] = custom_bits
                print(f"Custom mix: CRITICAL={custom_bits[0]}b IMPORTANT={custom_bits[1]}b COMPRESS={custom_bits[2]}b")
            except ValueError:
                print(f"Error: Invalid custom profile format: {raw_profile}")
                sys.exit(1)
        else:
            print(f"Error: Custom profile must be CUSTOM_C_I_P (e.g., CUSTOM_8_4_3)")
            sys.exit(1)
    elif raw_profile.isdigit():
        # User passed a number like "2" or "4"
        profile = profile_for_bits(int(raw_profile))
        # Fallback: jang-tools may return profiles not in JANG_PROFILES (e.g., JANG_3K)
        if profile not in JANG_PROFILES:
            _fallback_map = {1: "JANG_1L", 2: "JANG_2S", 3: "JANG_3S", 4: "JANG_4S",
                             5: "JANG_5M", 6: "JANG_6M", 7: "JANG_7L", 8: "JANG_8L"}
            profile = _fallback_map.get(int(raw_profile), profile)
        print(f"Bit target {raw_profile} → using profile {profile}")
    else:
        profile = raw_profile.upper()

    # Resolve model path
    print(f"Resolving model: {model_input}")
    try:
        model_path = resolve_model_path(model_input)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    if profile not in JANG_PROFILES:
        print(f"\nError: Unknown JANG profile '{profile}'.")
        print(f"Available: {', '.join(sorted(JANG_PROFILES.keys()))}")
        print(f"Or use a number 1-8 for automatic profile selection.")
        sys.exit(1)

    # Output path
    output_path = args.output
    if not output_path:
        name = model_input.rstrip("/").split("/")[-1]
        for suffix in ["-BF16", "-bf16", "-FP16", "-fp16", "-FP32", "-fp32"]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        output_path = f"{name}-{profile}"

    output_dir = Path(output_path)
    if output_dir.exists():
        if not args.force:
            print(f"Error: Output directory already exists: {output_path}")
            print("Use --force to overwrite.")
            sys.exit(1)
        else:
            import shutil
            shutil.rmtree(output_dir)

    # Show pre-conversion estimate
    crit, imp, comp = JANG_PROFILES[profile]

    # Estimate size and check memory
    est_str = ""
    try:
        from ..utils.model_inspector import inspect_model, available_memory_gb, total_memory_gb
        info = inspect_model(model_path)
        param_b = info.param_count_billions or 0
        if param_b > 0:
            from jang_tools.allocate import estimate_size_gb
            est = estimate_size_gb(int(param_b * 1e9), profile)
            est_str = f"  Estimated output: ~{est['total_gb']} GB ({est['avg_bits_approx']}b avg)"
    except Exception:
        info = None

    # Memory warning (conversion needs source weights + quantized output in RAM)
    # Use profile-aware multiplier: source + (target_bits/16 * source) + 0.3 overhead
    # e.g. 2-bit target: 1 + 0.125 + 0.3 = 1.425x, 8-bit: 1 + 0.5 + 0.3 = 1.8x
    try:
        available = available_memory_gb()
        total = total_memory_gb()
        source_gb = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors")) / (1024**3)
        # Determine target bits from the COMPRESS tier (smallest, most layers)
        target_bits = comp
        multiplier = 1.0 + target_bits / 16.0 + 0.3
        needed = source_gb * multiplier
        print(f"Memory estimate:")
        print(f"  Conversion needs: ~{needed:.1f} GB (profile {profile}: {target_bits}-bit target, {multiplier:.2f}x)")
        print(f"  Available: {available:.1f} GB / {total:.0f} GB total")
        if needed > total:
            print(f"\n  WARNING: This model may be too large for your system.")
        elif needed > available:
            print(f"\n  WARNING: Conversion needs more than currently available memory.")
            print(f"  Close other applications or expect heavy swap usage.")
        else:
            print(f"  Status: OK")
    except Exception:
        pass

    # Resolve advanced options (must be before the summary printout)
    calibration_method = getattr(args, 'calibration_method', 'weights')
    imatrix_path = getattr(args, 'imatrix_path', None)
    use_awq = getattr(args, 'use_awq', False)

    print()
    print("=" * 60)
    print(f"  JANG Convert — {profile}")
    print(f"  CRITICAL={crit}b  IMPORTANT={imp}b  COMPRESS={comp}b")
    print(f"  Source: {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Method: {args.jang_method}")
    if calibration_method != 'weights':
        print(f"  Calibration: {calibration_method}")
    if imatrix_path:
        print(f"  Importance matrix: {imatrix_path}")
    if use_awq:
        print(f"  AWQ scaling: enabled")
    if est_str:
        print(est_str)
    print("=" * 60)
    print()

    start_time = time.time()

    try:
        from jang_tools.convert import convert_model
        result = convert_model(
            model_path=str(model_path),
            output_path=str(output_dir),
            target_bits=comp,  # target is the COMPRESS tier bits
            profile=profile,
            quantization_method=args.jang_method,
            calibration_method=calibration_method,
            imatrix_path=imatrix_path,
            use_awq=use_awq,
        )
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nError: JANG conversion failed after {elapsed:.1f}s: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\nJANG conversion completed in {elapsed:.1f}s")
    print(f"  Profile: {profile}")
    print(f"  Actual bits: {result['actual_bits']}")
    print(f"  Weight size: {result['total_weight_gb']} GB")
    print(f"  Output: {output_dir.resolve()}")

    # Show output size
    output_files = list(output_dir.glob("*.safetensors"))
    output_size = sum(f.stat().st_size for f in output_files) / (1024**3)
    print(f"  Disk size: {output_size:.1f} GB ({len(output_files)} files)")

    # Post-conversion smoke test (verify the model loads and generates)
    if not args.skip_verify:
        print()
        print("Running JANG verification smoke test...")
        success, message = _jang_smoke_test(str(output_dir))
        if success:
            print(f"  PASS: {message}")
        else:
            print(f"  WARN: {message}")
            print("  The model converted but may need verification.")
            print(f"  Run 'vmlx-engine doctor {output_dir}' for diagnostics.")
    else:
        print()
        print(f"Verification skipped. To verify later:")
        print(f"  vmlx-engine doctor {output_dir}")

    print()
    print(f"Load with:")
    print(f"  vmlx-engine serve {output_dir}")
