#!/usr/bin/env python3
"""
Test harness for codebook VQ kernels and expert loader.

Tests:
1. Metal kernel compilation
2. 2D codebook matmul (simple codebook VQ)
3. 3D MoE codebook matmul (selected experts)
4. Correctness vs reference implementation
5. Performance comparison

Usage:
    python -m vmlx_engine.tests.test_codebook_kernels
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

import numpy as np


def test_metal_kernel_compilation():
    """Test that Metal kernels compile successfully."""
    logger.info("=" * 60)
    logger.info("TEST 1: Metal Kernel Compilation")
    logger.info("=" * 60)

    try:
        from vmlx_engine.metal.kernel_manager import CodebookKernelManager

        km = CodebookKernelManager()
        stats = km.get_stats()

        logger.info(f"  Metal available: {stats['metal_available']}")
        logger.info(f"  Kernels compiled: {stats['compiled_kernels']}")
        logger.info(f"  Is available: {stats['is_available']}")

        if km.is_available:
            logger.info("✓ Metal kernels compiled successfully")
            return True, km
        else:
            logger.warning("✗ Metal not available, will use MLX fallback")
            return False, km

    except Exception as e:
        logger.error(f"✗ Kernel compilation failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_2d_codebook_matmul(km, codebook_file):
    """Test 2D codebook matmul using a single expert's indices."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 2: 2D Codebook Matmul (Single Expert)")
    logger.info("=" * 60)

    try:
        import mlx.core as mx
        from safetensors import safe_open

        # Load codebook data
        with safe_open(str(codebook_file), framework="numpy") as f:
            codebook_np = f.get_tensor("codebook")
            indices_np = f.get_tensor("indices")

        codebook = mx.array(codebook_np)
        indices = mx.array(indices_np)

        logger.info(f"  Codebook shape: {codebook.shape}")
        logger.info(
            f"  Indices shape: {indices.shape} (3D: [n_experts, out_dim, n_groups])"
        )

        # Extract single expert's indices for 2D test
        n_experts, out_dim, n_groups = indices.shape
        group_size = 8
        in_dim = n_groups * group_size

        # Use first expert's indices for 2D test
        expert_0_indices = indices[0]  # [out_dim, n_groups]

        logger.info(f"  Using expert 0 indices: {expert_0_indices.shape}")
        logger.info(f"  out_dim={out_dim}, n_groups={n_groups}, in_dim={in_dim}")

        # Create test input
        batch_size = 4
        x = mx.random.normal((batch_size, in_dim)).astype(mx.float16)

        # Reference implementation (MLX)
        t0 = time.time()
        flat_idx = expert_0_indices.reshape(-1)
        flat_weights = codebook[flat_idx]
        W = flat_weights.reshape(out_dim, in_dim)
        ref_output = x @ W.T
        ref_time = time.time() - t0

        logger.info(f"  Reference implementation: {ref_time * 1000:.2f}ms")
        logger.info(f"  Reference output shape: {ref_output.shape}")

        # Kernel implementation
        if km and km.is_available and "codebook_matvec_2d" in km._kernels:
            t0 = time.time()
            kernel_output = km.codebook_matvec_2d(
                x,
                codebook,
                expert_0_indices,
                out_dim=out_dim,
                n_groups=n_groups,
                group_size=group_size,
            )
            kernel_time = time.time() - t0

            logger.info(f"  Kernel implementation: {kernel_time * 1000:.2f}ms")
            logger.info(f"  Kernel output shape: {kernel_output.shape}")

            # Compare outputs
            diff = mx.abs(ref_output - kernel_output).astype(mx.float32)
            max_diff = float(mx.max(diff))
            mean_diff = float(mx.mean(diff))

            logger.info(f"  Max difference: {max_diff}")
            logger.info(f"  Mean difference: {mean_diff}")

            if max_diff < 0.1:
                logger.info("✓ 2D codebook matmul: CORRECT")
                return True
            else:
                logger.warning("✗ 2D codebook matmul: OUTPUT MISMATCH")
                return False
        else:
            logger.info("  Skipping kernel test (Metal not available)")
            return True

    except Exception as e:
        logger.error(f"✗ 2D test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_3d_moe_matmul(km, codebook_file):
    """Test 3D MoE codebook matmul with selected experts."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 3: 3D MoE Codebook Matmul (Top-K Selection)")
    logger.info("=" * 60)

    try:
        import mlx.core as mx
        from safetensors import safe_open

        # Load codebook data
        with safe_open(str(codebook_file), framework="numpy") as f:
            codebook_np = f.get_tensor("codebook")
            indices_np = f.get_tensor("indices")

        codebook = mx.array(codebook_np)
        indices = mx.array(indices_np)

        n_experts, out_dim, n_groups = indices.shape
        group_size = 8
        in_dim = n_groups * group_size
        batch_size = 4
        k = 8  # Top-8 experts

        logger.info(f"  Indices shape: {indices.shape} [n_experts, out_dim, n_groups]")
        logger.info(f"  n_experts={n_experts}, out_dim={out_dim}, n_groups={n_groups}")
        logger.info(f"  batch_size={batch_size}, k={k}")

        # Select top-8 experts (simulated routing)
        np.random.seed(42)
        selected_expert_ids = np.random.choice(n_experts, size=k, replace=False).astype(
            np.uint16
        )
        logger.info(f"  Selected expert IDs: {selected_expert_ids}")

        # Create test input
        x = mx.random.normal((batch_size, in_dim)).astype(mx.float16)

        # Reference implementation
        t0 = time.time()
        ref_outputs = []
        for i in range(k):
            expert_id = int(selected_expert_ids[i])
            expert_indices = indices[expert_id]  # [out_dim, n_groups]

            flat_idx = expert_indices.reshape(-1)
            flat_weights = codebook[flat_idx]
            W = flat_weights.reshape(out_dim, in_dim)

            out = x @ W.T  # [batch, out_dim]
            ref_outputs.append(out)

        ref_output = mx.stack(ref_outputs, axis=1)  # [batch, k, out_dim]
        ref_time = time.time() - t0

        logger.info(f"  Reference implementation: {ref_time * 1000:.2f}ms")
        logger.info(f"  Reference output shape: {ref_output.shape}")

        # Kernel implementation
        if km and km.is_available and "codebook_moe_matvec" in km._kernels:
            t0 = time.time()
            kernel_output = km.codebook_moe_matvec(
                x=x,
                codebook=codebook,
                indices=indices,
                selected_expert_ids=selected_expert_ids,
                out_dim=out_dim,
                n_groups=n_groups,
                group_size=group_size,
            )
            kernel_time = time.time() - t0

            logger.info(f"  Kernel implementation: {kernel_time * 1000:.2f}ms")
            logger.info(f"  Kernel output shape: {kernel_output.shape}")

            # Compare outputs
            diff = mx.abs(ref_output - kernel_output).astype(mx.float32)
            max_diff = float(mx.max(diff))
            mean_diff = float(mx.mean(diff))

            logger.info(f"  Max difference: {max_diff}")
            logger.info(f"  Mean difference: {mean_diff}")

            if max_diff < 0.1:
                logger.info("✓ 3D MoE codebook matmul: CORRECT")
                return True
            else:
                logger.warning("✗ 3D MoE codebook matmul: OUTPUT MISMATCH")
                return False
        else:
            logger.info("  Skipping kernel test (Metal not available)")
            return True

    except Exception as e:
        logger.error(f"✗ 3D MoE test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_expert_loader():
    """Test the CodebookExpertLoader."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 4: CodebookExpertLoader")
    logger.info("=" * 60)

    try:
        from vmlx_engine.models.codebook_expert_loader import CodebookExpertLoader

        model_path = Path.home() / ".mlxstudio/models/Qwen3.5-397B-CODEBOOK-TEST"
        if not model_path.exists():
            logger.warning(f"  Model path not found: {model_path}")
            return True

        jang_config_path = model_path / "jang_config.json"
        if not jang_config_path.exists():
            logger.warning(f"  jang_config.json not found: {jang_config_path}")
            return True

        import json

        jang_config = json.loads(jang_config_path.read_text())

        logger.info(f"  Model path: {model_path}")
        logger.info(f"  JANG config loaded")

        class MockModel:
            pass

        base_model = MockModel()

        loader = CodebookExpertLoader(
            base_model=base_model,
            model_path=str(model_path),
            jang_config=jang_config,
            lazy=True,
        )

        logger.info(f"  Codebook files indexed: {len(loader._codebook_files)}")
        logger.info(f"  Stats: {loader.get_stats()}")

        test_layer = 0
        test_type = "gate_proj"
        if (test_layer, test_type) in loader._codebook_files:
            codebook, indices = loader.load_codebook_layer(test_layer, test_type)
            logger.info(f"  Loaded layer {test_layer}/{test_type}:")
            logger.info(f"    codebook shape: {codebook.shape}")
            logger.info(f"    indices shape: {indices.shape}")

            selected = [0, 1, 2, 3, 4, 5, 6, 7]
            for exp_id in selected:
                exp_indices = loader.get_expert_indices_slice(
                    test_layer, test_type, exp_id
                )
                logger.info(f"    Expert {exp_id} indices shape: {exp_indices.shape}")

        logger.info("✓ CodebookExpertLoader: INITIALIZED")
        return True

    except Exception as e:
        logger.error(f"✗ Expert loader test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance_comparison(km, codebook_file):
    """Compare performance of naive vs fused approaches."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 5: Performance Comparison")
    logger.info("=" * 60)

    try:
        import mlx.core as mx
        from safetensors import safe_open

        with safe_open(str(codebook_file), framework="numpy") as f:
            codebook_np = f.get_tensor("codebook")
            indices_np = f.get_tensor("indices")

        codebook = mx.array(codebook_np)
        indices = mx.array(indices_np)

        n_experts, out_dim, n_groups = indices.shape
        group_size = 8
        in_dim = n_groups * group_size
        batch_size = 1
        k = 8

        logger.info(f"  Configuration:")
        logger.info(
            f"    n_experts={n_experts}, out_dim={out_dim}, n_groups={n_groups}"
        )
        logger.info(f"    batch_size={batch_size}, k={k}")

        np.random.seed(42)
        selected_expert_ids = np.random.choice(n_experts, size=k, replace=False).astype(
            np.uint16
        )

        x = mx.random.normal((batch_size, in_dim)).astype(mx.float16)

        # Naive: reconstruct ALL experts, then select
        t0 = time.time()
        flat_idx = indices.reshape(-1)
        flat_weights = codebook[flat_idx]
        W_all = flat_weights.reshape(n_experts, out_dim, in_dim)

        naive_output = []
        for i in range(k):
            expert_id = int(selected_expert_ids[i])
            naive_output.append(x @ W_all[expert_id].T)
        naive_output = mx.stack(naive_output, axis=1)
        naive_time = time.time() - t0

        logger.info(
            f"  Naive (reconstruct all {n_experts} experts): {naive_time * 1000:.2f}ms"
        )

        # Fused: select first, compute second
        if km and km.is_available and "codebook_moe_matvec" in km._kernels:
            t0 = time.time()
            fused_output = km.codebook_moe_matvec(
                x=x,
                codebook=codebook,
                indices=indices,
                selected_expert_ids=selected_expert_ids,
                out_dim=out_dim,
                n_groups=n_groups,
                group_size=group_size,
            )
            fused_time = time.time() - t0

            logger.info(
                f"  Fused (select first, compute second): {fused_time * 1000:.2f}ms"
            )

            speedup = naive_time / fused_time if fused_time > 0 else float("inf")
            logger.info(f"  Speedup: {speedup:.1f}x")

            naive_mem = n_experts * out_dim * in_dim * 2
            fused_mem = k * out_dim * in_dim * 2
            logger.info(
                f"  Memory: {naive_mem / 1e9:.2f}GB (naive) vs {fused_mem / 1e6:.2f}MB (fused)"
            )

        logger.info("✓ Performance test complete")
        return True

    except Exception as e:
        logger.error(f"✗ Performance test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    logger.info("")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 15 + "CODEBOOK VQ KERNEL TESTS" + " " * 19 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("")

    possible_paths = [
        Path.home() / ".mlxstudio/models/Qwen3.5-397B-CODEBOOK-TEST",
    ]

    codebook_file = None
    for p in possible_paths:
        if p.exists():
            files = list(p.glob("codebook-layer-000-gate_proj.safetensors"))
            if files:
                codebook_file = files[0]
                break

    if codebook_file:
        logger.info(f"Using codebook file: {codebook_file}")
    else:
        logger.warning("No codebook file found, some tests will be skipped")

    results = {}

    available, km = test_metal_kernel_compilation()
    results["kernel_compilation"] = available

    if codebook_file:
        results["2d_matmul"] = test_2d_codebook_matmul(km, codebook_file)
        results["3d_moe_matmul"] = test_3d_moe_matmul(km, codebook_file)
        results["performance"] = test_performance_comparison(km, codebook_file)
    else:
        logger.warning("Skipping 2D/3D/performance tests (no codebook file)")

    results["expert_loader"] = test_expert_loader()

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
