# SPDX-License-Identifier: Apache-2.0
"""Opt-in parity checks for MiniMax-M3 VL native image preprocessing.

The reference MiniMax processor imports torch, so this test is intentionally
guarded out of the default suite. Run it before changing the native
preprocessor:

    VMLINUX_RUN_M3_VL_PROCESSOR_PARITY=1 \
    VMLINUX_M3_VL_PARITY_MODEL=/path/to/MiniMax-M3-VL \
    pytest tests/test_minimax_m3_vl_processor_parity.py
"""

from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("VMLINUX_RUN_M3_VL_PROCESSOR_PARITY") != "1",
    reason=(
        "MiniMax-M3 VL processor parity imports the HF torch image processor; "
        "set VMLINUX_RUN_M3_VL_PROCESSOR_PARITY=1 and "
        "VMLINUX_M3_VL_PARITY_MODEL to run it."
    ),
)


def _synthetic_image(width: int, height: int):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (21, 36, 57))
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (
            max(0, width // 12),
            max(0, height // 8),
            min(width - 1, width // 12 + max(10, width // 4)),
            min(height - 1, height // 8 + max(10, height // 4)),
        ),
        fill=(230, 20, 20),
    )
    draw.rectangle(
        (
            max(0, width // 2),
            max(0, height // 3),
            min(width - 1, width // 2 + max(10, width // 5)),
            min(height - 1, height // 3 + max(10, height // 5)),
        ),
        fill=(20, 50, 230),
    )
    draw.text((max(0, width // 8), max(0, height - 28)), "M3", fill=(245, 245, 245))
    return image


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (112, 112),
        (113, 251),
        (640, 360),
        (896, 512),
        (47, 999),
    ],
)
def test_native_minimax_m3_vl_image_preprocess_matches_hf_processor(width, height):
    import numpy as np
    from transformers import AutoImageProcessor

    from vmlx_engine.models.minimax_m3.minimax_m3_vl import preprocess_image

    model_path = os.environ.get("VMLINUX_M3_VL_PARITY_MODEL")
    if not model_path:
        pytest.skip("VMLINUX_M3_VL_PARITY_MODEL is required")

    image = _synthetic_image(width, height)
    native_pixels, native_grid = preprocess_image(image)
    processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    hf = processor(images=[image], return_tensors="np")

    native_pixels_np = np.asarray(native_pixels.tolist(), dtype=np.float32)
    native_grid_np = np.asarray(native_grid.tolist(), dtype=np.int64)
    hf_pixels_np = np.asarray(hf["pixel_values"], dtype=np.float32)
    hf_grid_np = np.asarray(hf["image_grid_thw"], dtype=np.int64)

    assert np.array_equal(native_grid_np, hf_grid_np)
    np.testing.assert_allclose(native_pixels_np, hf_pixels_np, rtol=1e-5, atol=1e-5)
