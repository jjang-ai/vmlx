"""Real MLX vision grid regressions without loading model weights."""
import numpy as np
import pytest


def _tower():
    import vmlx_engine  # installs runtime compatibility patches
    from mlx_vlm.models.gemma4.config import VisionConfig
    from mlx_vlm.models.gemma4.vision import VisionModel

    return VisionModel(VisionConfig(
        hidden_size=4, num_hidden_layers=0, num_attention_heads=1,
        num_key_value_heads=1, head_dim=4, intermediate_size=8,
        position_embedding_size=128, default_output_length=280,
        patch_size=16, pooling_kernel_size=3,
    ))


@pytest.mark.parametrize("side,tokens", [(768, 256), (1104, 529), (1584, 1089)])
def test_processed_image_grid_reaches_pooler_and_restores_default(side, tokens):
    import mlx.core as mx

    model = _tower()
    output = model(mx.ones((1, 3, side, side)))
    mx.eval(output)
    assert output.shape == (1, tokens, 4)
    assert bool(mx.all(mx.isfinite(output)))
    assert (model.max_patches, model.default_output_length,
            model.pooler.default_output_length) == (2520, 280, 280)
    # A following native-default request must retain its normal geometry.
    assert model(mx.ones((1, 3, 768, 768))).shape == (1, 256, 4)


def test_mixed_sizes_preserve_each_image_budget():
    import mlx.core as mx

    model = _tower()
    output = model([
        np.ones((3, 768, 768), dtype=np.float32),
        mx.ones((3, 1104, 1104)),
    ])
    mx.eval(output)
    assert output.shape == (1, 256 + 529, 4)
    assert model.max_patches == 2520


def test_encoder_failure_restores_budget():
    import mlx.core as mx

    model = _tower()

    def fail(*args):
        raise RuntimeError("encoder failed")

    model.encoder = fail
    with pytest.raises(RuntimeError, match="encoder failed"):
        model(mx.ones((1, 3, 1104, 1104)))
    assert (model.max_patches, model.default_output_length,
            model.pooler.default_output_length) == (2520, 280, 280)
