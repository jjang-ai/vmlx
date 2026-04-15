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
        print("\nError: jang package not installed. Install with: pip install jang")
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

    # Accept both fixed 3-tier profiles (JANG_2S, etc.) and K-quant dynamic profiles (JANG_3K, etc.)
    try:
        from jang_tools.allocate import is_k_quant, JANG_K_TARGETS
        _is_valid = profile in JANG_PROFILES or is_k_quant(profile)
    except ImportError:
        _is_valid = profile in JANG_PROFILES
    if not _is_valid:
        print(f"\nError: Unknown JANG profile '{profile}'.")
        all_profiles = sorted(set(list(JANG_PROFILES.keys()) + list(JANG_K_TARGETS.keys())) if 'JANG_K_TARGETS' in dir() else JANG_PROFILES.keys())
        print(f"Available: {', '.join(all_profiles)}")
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
    if profile in JANG_PROFILES:
        crit, imp, comp = JANG_PROFILES[profile]
    else:
        # K-quant dynamic profile — use target bits as the compress tier
        try:
            from jang_tools.allocate import JANG_K_TARGETS
            comp = int(JANG_K_TARGETS.get(profile, int(raw_profile) if raw_profile.isdigit() else 3))
        except Exception:
            comp = int(raw_profile) if raw_profile.isdigit() else 3
        crit, imp = comp, comp  # K-quant uses dynamic per-layer, not fixed 3-tier

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
    awq_alpha = getattr(args, 'awq_alpha', 0.25)

    print()
    print("=" * 60)
    print(f"  JANG Convert — {profile}")
    if profile in JANG_PROFILES:
        print(f"  CRITICAL={crit}b  IMPORTANT={imp}b  COMPRESS={comp}b")
    else:
        print(f"  K-Quant dynamic allocation — target ~{comp}b average")
    print(f"  Source: {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Method: {args.jang_method}")
    if calibration_method != 'weights':
        print(f"  Calibration: {calibration_method}")
    if imatrix_path:
        print(f"  Importance matrix: {imatrix_path}")
    if use_awq:
        print(f"  AWQ scaling: enabled (alpha={awq_alpha})")
    if est_str:
        print(est_str)
    print("=" * 60)
    print()

    # Pre-flight checks for mlxstudio#74 (Hemanth Pai report).
    # The user's quantize tracker hit 100% then "Conversion failed" with
    # only a leaked-semaphore warning visible — meaning the real error
    # was lost. These checks fail fast with actionable messages BEFORE
    # spending hours on quantization.
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # 1. Filesystem sanity — refuse exFAT/FAT32 since safetensors
        #    shards default to 5 GB and FAT-family caps single files at 4 GB
        try:
            import subprocess as _sp
            _di = _sp.run(
                ["diskutil", "info", str(output_dir)],
                capture_output=True, text=True, timeout=5,
            )
            _fs_line = ""
            for _l in (_di.stdout or "").splitlines():
                if "File System Personality" in _l or "Type (Bundle)" in _l:
                    _fs_line = _l.split(":", 1)[1].strip().lower()
                    break
            if _fs_line and any(bad in _fs_line for bad in ("exfat", "ms-dos", "fat32", "msdos")):
                print(
                    f"\nError: Output directory '{output_path}' is on a "
                    f"{_fs_line.upper()} filesystem. JANG shards default to "
                    f"5 GB but FAT-family filesystems cap single files at 4 "
                    f"GB. Move the output directory to an APFS or HFS+ "
                    f"volume (or reformat the external drive)."
                )
                sys.exit(1)
            if _fs_line:
                print(f"  Output filesystem: {_fs_line}")
        except (FileNotFoundError, _sp.TimeoutExpired):
            pass  # diskutil not available or slow — skip
        except Exception as _fs_e:
            logger.debug(f"Filesystem check skipped: {_fs_e}")

        # 2. Disk space sanity — need at least ~target_bits/16 * source_size
        try:
            _src_bytes = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors"))
            _src_gb = _src_bytes / (1024 ** 3)
            _need_gb = _src_gb * (comp / 16.0) * 1.2  # 20% slack for index + tokenizer
            import shutil as _sh
            _free_bytes = _sh.disk_usage(str(output_dir)).free
            _free_gb = _free_bytes / (1024 ** 3)
            print(f"  Disk space: {_free_gb:.1f} GB free, ~{_need_gb:.1f} GB needed for output")
            if _free_gb < _need_gb:
                print(
                    f"\nError: Insufficient disk space. Output needs ~"
                    f"{_need_gb:.1f} GB but only {_free_gb:.1f} GB free at "
                    f"'{output_path}'. Free up space or pick a different "
                    f"output directory."
                )
                sys.exit(1)
        except Exception as _ds_e:
            logger.debug(f"Disk space check skipped: {_ds_e}")

        # 3. Test write a small file to make sure output is writable
        _probe = output_dir / ".jang_write_probe"
        try:
            _probe.write_bytes(b"probe")
            _probe.unlink()
        except OSError as _wp_e:
            print(
                f"\nError: Cannot write to output directory '{output_path}': "
                f"{_wp_e}. Check permissions or pick a different output path."
            )
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as _pf_e:
        logger.debug(f"Pre-flight checks raised: {_pf_e}")

    # ──────────────────────────────────────────────────────────────────
    # mlxstudio#74: error visibility hardening. The previous handler
    # only printed traceback at DEBUG level, so users who didn't set
    # `--log-level DEBUG` lost the real error and saw only a generic
    # "JANG conversion failed: <oneline>" + the multiprocessing
    # resource_tracker semaphore warning.
    #
    # Now: ALWAYS print full traceback to stderr, AND mirror it to
    # `<output_dir>/convert_error.log` so the error survives any
    # subsequent process kill / panel UI truncation.
    # ──────────────────────────────────────────────────────────────────
    start_time = time.time()
    _conv_error_log = output_dir / "convert_error.log"

    try:
        from jang_tools.convert import convert_model
        result = convert_model(
            model_path=str(model_path),
            output_path=str(output_dir),
            target_bits=comp,  # COMPRESS tier bits (fixed profile) or K target average (K-quant profile)
            profile=profile,
            quantization_method=args.jang_method,
            calibration_method=calibration_method,
            imatrix_path=imatrix_path,
            use_awq=use_awq,
            awq_alpha=awq_alpha,
        )
    except SystemExit:
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        import traceback as _tb
        tb_text = _tb.format_exc()
        # 1. Loud stderr — always, regardless of log level
        sys.stderr.write("\n" + "=" * 60 + "\n")
        sys.stderr.write(f"JANG CONVERSION FAILED after {elapsed:.1f}s\n")
        sys.stderr.write(f"Error: {type(e).__name__}: {e}\n")
        sys.stderr.write("=" * 60 + "\n")
        sys.stderr.write(tb_text)
        sys.stderr.write("=" * 60 + "\n")
        sys.stderr.flush()
        # 2. Mirror to convert_error.log inside the output dir so the
        #    error survives any subprocess kill or UI clip.
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(_conv_error_log, "w") as f:
                f.write(f"JANG conversion failed after {elapsed:.1f}s\n")
                f.write(f"Source: {model_path}\n")
                f.write(f"Output: {output_dir}\n")
                f.write(f"Profile: {profile}\n")
                f.write(f"Method: {args.jang_method}\n")
                f.write(f"Calibration: {calibration_method}\n")
                f.write(f"Error: {type(e).__name__}: {e}\n")
                f.write("=" * 60 + "\n")
                f.write(tb_text)
            print(f"\n  Error log written to: {_conv_error_log}", file=sys.stderr)
        except Exception as _le:
            sys.stderr.write(f"  (could not write convert_error.log: {_le})\n")
        # 3. Stdout copy too in case the panel only buffers one stream
        print(f"\nError: JANG conversion failed after {elapsed:.1f}s: {type(e).__name__}: {e}")
        print(f"See {_conv_error_log} for the full traceback.")
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
