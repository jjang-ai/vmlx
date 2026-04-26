#!/usr/bin/env python3
"""
Test full inference with codebook VQ model.

This test:
1. Loads the Qwen3.5-397B-CODEBOOK-TEST model
2. Patches SwitchLinear to use codebook loader
3. Runs a forward pass
4. Measures performance
"""

import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_model_loading():
    """Test loading the codebook VQ model."""
    logger.info("=" * 60)
    logger.info("TEST: Model Loading")
    logger.info("=" * 60)

    try:
        from vmlx_engine.utils.jang_loader import load_jang_model

        model_path = Path.home() / ".mlxstudio/models/Qwen3.5-397B-CODEBOOK-TEST"
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return False

        logger.info(f"Loading model from: {model_path}")

        start = time.time()
        model, tokenizer = load_jang_model(str(model_path), lazy=True)
        load_time = time.time() - start

        logger.info(f"Model loaded in {load_time:.1f}s")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Tokenizer: {type(tokenizer)}")

        # Check model structure
        logger.info(f"Base model: {type(model._base_model)}")

        return True, model, tokenizer

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None, None


def test_patch_switch_linear(model):
    """Test patching SwitchLinear layers."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST: Patch SwitchLinear")
    logger.info("=" * 60)

    try:
        from vmlx_engine.models.codebook_moe_integration import (
            find_switch_linears,
            patch_switch_linear,
        )
        from vmlx_engine.models.codebook_expert_loader import CodebookExpertLoader
        import json

        # Find SwitchLinears
        switch_linears = find_switch_linears(model)
        logger.info(f"Found {len(switch_linears)} SwitchLinear layers")

        for sl in switch_linears[:5]:  # Show first 5
            logger.info(
                f"  {sl['name']}: layer={sl['layer_idx']}, type={sl['tensor_type']}"
            )

        if len(switch_linears) > 5:
            logger.info(f"  ... and {len(switch_linears) - 5} more")

        # Create expert loader
        model_path = Path.home() / ".mlxstudio/models/Qwen3.5-397B-CODEBOOK-TEST"
        jang_config = json.loads((model_path / "jang_config.json").read_text())

        class MockModel:
            pass

        expert_loader = CodebookExpertLoader(
            base_model=MockModel(),
            model_path=str(model_path),
            jang_config=jang_config,
            lazy=True,
        )

        # Patch model
        patcher = patch_switch_linear(model, expert_loader, verbose=True)
        logger.info(
            f"Patched {len(patcher._patched_classes)} classes, {patcher._instances_patched} instances"
        )

        return True, patcher

    except Exception as e:
        logger.error(f"Patch failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_forward_pass(model, tokenizer):
    """Test a forward pass."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST: Forward Pass")
    logger.info("=" * 60)

    try:
        import mlx.core as mx

        # Create test input
        prompt = "Hello, world!"
        tokens = tokenizer.encode(prompt)
        logger.info(f"Input: '{prompt}'")
        logger.info(f"Tokens: {len(tokens)}")

        # Convert to MLX array
        input_ids = mx.array([tokens])

        # Forward pass
        logger.info("Running forward pass...")
        start = time.time()

        # Model is wrapped, need to call base model
        output = model._base_model(input_ids)

        elapsed = time.time() - start
        logger.info(f"Forward pass completed in {elapsed * 1000:.2f}ms")
        logger.info(f"Output shape: {output.shape}")

        return True, output

    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def main():
    logger.info("")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 15 + "CODEBOOK VQ INFERENCE TEST" + " " * 14 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("")

    results = {}

    # Test 1: Model loading
    success, model, tokenizer = test_model_loading()
    results["model_loading"] = success

    if not success:
        logger.error("Model loading failed, skipping remaining tests")
        return 1

    # Test 2: Patch SwitchLinear
    success, patcher = test_patch_switch_linear(model)
    results["patch_switch_linear"] = success

    if not success:
        logger.warning("SwitchLinear patching failed, continuing with original model")

    # Test 3: Forward pass
    success, output = test_forward_pass(model, tokenizer)
    results["forward_pass"] = success

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")

    all_passed = all(results.values())
    logger.info("")
    if all_passed:
        logger.info("All tests passed!")
    else:
        logger.warning("Some tests failed!")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
