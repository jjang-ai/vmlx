# SPDX-License-Identifier: Apache-2.0
"""Normalize mflux paint masks without expanding the selected region."""
from PIL import Image


def normalize_paint_mask(image: Image.Image) -> Image.Image:
    """White/opaque means edit, black/transparent means keep.

    An opaque canvas PNG stores selection in luminance, not its constant
    alpha channel. With transparency, opacity carries the painted selection;
    hidden RGB values must not turn transparent background into an edit.
    This is the mflux paint-mask contract, not an inverted-alpha mask.
    """
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    if "A" in image.getbands():
        alpha = image.getchannel("A")
        if alpha.getextrema() != (255, 255):
            return alpha.copy()
    return image.convert("L")
