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
    Convert a HuggingFace model to quantized MLX format.

    Flow:
    1. Inspect model metadata (config.json)
    2. Pre-flight memory check
    3. Apply LatentMoE patch if needed
    4. Run mlx_lm.convert.convert()
    5. Post-conversion smoke test (unless --skip-verify)
    """
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
