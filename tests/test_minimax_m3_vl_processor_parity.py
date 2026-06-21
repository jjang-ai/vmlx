# SPDX-License-Identifier: Apache-2.0
"""Opt-in parity checks for MiniMax-M3 VL native image preprocessing.

The reference MiniMax processor's resize/patchify math uses torch, so this test
is intentionally guarded out of the default suite. Run it before changing the
native preprocessor:

    VMLINUX_RUN_M3_VL_PROCESSOR_PARITY=1 \
    pytest tests/test_minimax_m3_vl_processor_parity.py
"""

from __future__ import annotations

import math
import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("VMLINUX_RUN_M3_VL_PROCESSOR_PARITY") != "1",
    reason=(
        "MiniMax-M3 VL processor parity imports torch; set "
        "VMLINUX_RUN_M3_VL_PROCESSOR_PARITY=1 to run it."
    ),
)

MAX_RATIO = 200
IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 451584,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError("aspect ratio too extreme")
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


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


def _reference_preprocess_image(
    image,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    max_pixels: int = 451584,
):
    """Torch reference from bundled MiniMaxM3VLImageProcessor._preprocess."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    image = image.convert("RGB")
    tensor = (
        torch.from_numpy(np.array(image))
        .contiguous()
        .permute(2, 0, 1)
        .contiguous()
    )
    height, width = tensor.shape[-2:]
    resized_height, resized_width = _smart_resize(
        height,
        width,
        factor=patch_size * merge_size,
        max_pixels=max_pixels,
    )
    tensor = F.interpolate(
        tensor.unsqueeze(0).to(dtype=torch.float32),
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0)

    patches = tensor.to(dtype=torch.float32) * (1 / 255)
    mean = torch.tensor(IMAGE_MEAN, dtype=patches.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=patches.dtype).view(3, 1, 1)
    patches = (patches - mean) / std
    patches = patches.unsqueeze(0).unsqueeze(0)

    if patches.shape[1] % temporal_patch_size != 0:
        repeats = patches[:, -1:].repeat(
            1,
            temporal_patch_size - (patches.shape[1] % temporal_patch_size),
            1,
            1,
            1,
        )
        patches = torch.cat([patches, repeats], dim=1)

    batch_size, grid_t, channel = patches.shape[:3]
    grid_t = grid_t // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    patches = patches.view(
        batch_size,
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flatten_patches = patches.reshape(
        batch_size * grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )
    grid = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
    return flatten_patches.detach().cpu().numpy().astype(np.float32), grid


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
def test_native_minimax_m3_vl_image_preprocess_matches_torch_reference(width, height):
    import numpy as np

    from vmlx_engine.models.minimax_m3.minimax_m3_vl import preprocess_image

    image = _synthetic_image(width, height)
    native_pixels, native_grid = preprocess_image(image)
    ref_pixels_np, ref_grid_np = _reference_preprocess_image(image)

    native_pixels_np = np.asarray(native_pixels.tolist(), dtype=np.float32)
    native_grid_np = np.asarray(native_grid.tolist(), dtype=np.int64)

    assert np.array_equal(native_grid_np, ref_grid_np)
    np.testing.assert_allclose(native_pixels_np, ref_pixels_np, rtol=1e-5, atol=1e-5)
