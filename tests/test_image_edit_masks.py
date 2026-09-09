import pytest
from PIL import Image
from vmlx_engine.image_masks import normalize_paint_mask


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "LA"])
def test_opaque_canvas_preserves_exact_selected_region(mode):
    source = Image.new("L", (16, 16), 0)
    for x in range(3, 8):
        for y in range(4, 10):
            source.putpixel((x, y), 255)
    assert normalize_paint_mask(source.convert(mode)).tobytes() == source.tobytes()


@pytest.mark.parametrize("mode", ["RGBA", "LA"])
def test_opaque_black_canvas_is_empty(mode):
    assert normalize_paint_mask(Image.new("L", (16, 16), 0).convert(mode)).getextrema() == (0, 0)


@pytest.mark.parametrize("rgb", [0, 255])
def test_alpha_paint_ignores_hidden_rgb(rgb):
    image = Image.new("RGBA", (16, 16), (rgb, rgb, rgb, 0))
    image.putpixel((3, 4), (0, 0, 0, 255))
    result = normalize_paint_mask(image)
    assert sum(v > 0 for v in result.getdata()) == 1
    assert result.getpixel((3, 4)) == 255


def test_transparent_la_and_palette_masks():
    image = Image.new("LA", (16, 16), (255, 0))
    image.putpixel((2, 3), (0, 255))
    assert sum(v > 0 for v in normalize_paint_mask(image).getdata()) == 1
    palette = Image.new("P", (16, 16), 0)
    palette.info["transparency"] = 0
    palette.putpixel((2, 3), 1)
    assert sum(v > 0 for v in normalize_paint_mask(palette).getdata()) == 1
