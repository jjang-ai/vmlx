import pytest

mx = pytest.importorskip('mlx.core')
nn = pytest.importorskip('mlx.nn')
from mlx.utils import tree_flatten
from vmlx_engine.image_quantized_encoder import restore_packed_encoder


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 128)
        self.proj = nn.Linear(128, 64)
        self.norm = nn.RMSNorm(64)

    def __call__(self, ids):
        return self.norm(self.proj(self.embed(ids)))


def saved_encoder():
    good = TinyEncoder()
    nn.quantize(good, class_predicate=lambda path, module:
                {'bits': 6, 'group_size': 32} if path == 'embed' else
                {'bits': 8, 'group_size': 64} if path == 'proj' else False)
    bad = TinyEncoder()
    bad.update(good.parameters(), strict=False)
    return good, bad


def test_restores_mixed_stored_bits_and_groups_without_value_changes():
    good, bad = saved_encoder()
    before = dict(tree_flatten(good.parameters()))
    result, count = restore_packed_encoder(bad, good.parameters)
    assert count == 2
    assert (result.embed.bits, result.embed.group_size) == (6, 32)
    assert (result.proj.bits, result.proj.group_size) == (8, 64)
    after = dict(tree_flatten(result.parameters()))
    assert before.keys() == after.keys()
    for key in before:
        assert before[key].dtype == after[key].dtype
        assert mx.array_equal(before[key], after[key]).item()
    ids = mx.array([[1, 4, 7]])
    assert mx.array_equal(good(ids), result(ids)).item()


def test_full_precision_and_already_quantized_are_unchanged():
    original = TinyEncoder()
    def unused():
        raise AssertionError('Unchanged encoder must not reread weights')
    assert restore_packed_encoder(original, unused) == (original, 0)
    good, _ = saved_encoder()
    assert restore_packed_encoder(good, unused) == (good, 0)


@pytest.mark.parametrize('damage', ['missing_scales', 'bad_columns', 'scale_dtype'])
def test_rejects_malformed_packed_layers(damage):
    good, bad = saved_encoder()
    if damage == 'missing_scales':
        del good.embed['scales']
    elif damage == 'bad_columns':
        good.embed.weight = good.embed.weight[:, :-1]
    else:
        good.embed.scales = good.embed.scales.astype(mx.uint32)
    with pytest.raises(ValueError, match='encoder'):
        restore_packed_encoder(bad, good.parameters)
