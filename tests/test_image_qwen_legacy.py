from types import SimpleNamespace
import pytest
mx=pytest.importorskip('mlx.core')
nn=pytest.importorskip('mlx.nn')
from mlx.utils import tree_unflatten
from vmlx_engine.image_qwen_legacy import restore_legacy_modulation


def fixture():
    block=SimpleNamespace(img_mod_linear=nn.Linear(128,768),txt_mod_linear=nn.Linear(128,768))
    target=SimpleNamespace(transformer_blocks=[block],norm_out=SimpleNamespace(
        embedding_dim=128,linear=nn.Linear(128,256,bias=False)))
    rows=[]; expected={}
    for lane,bits,group in [('img',6,32),('txt',8,64)]:
        linear=nn.QuantizedLinear(128,768,bits=bits,group_size=group)
        expected[lane]=linear
        rows.extend((f'transformer_blocks.0.{lane}_norm1.mod_linear.{k}',v) for k,v in linear.parameters().items())
    bias=mx.ones((256,))*0.2
    rows.append(('norm_out.linear.bias',bias))
    return target,rows,expected,bias


def test_legacy_mapping_preserves_each_packed_layer_and_output_bias():
    target,rows,expected,bias=fixture()
    assert restore_legacy_modulation(target,tree_unflatten(rows))==2
    x=mx.ones((1,128))
    for lane,original in expected.items():
        actual=getattr(target.transformer_blocks[0],lane+'_mod_linear')
        assert mx.array_equal(actual(x),original(x)).item()
        for key,v in original.parameters().items():
            assert mx.array_equal(actual.parameters()[key],v).item()
    assert mx.array_equal(target.norm_out.linear.bias,bias).item()


def test_modern_layout_is_not_changed():
    target,_,_,_=fixture()
    original=target.transformer_blocks[0].img_mod_linear
    assert restore_legacy_modulation(target,{'transformer_blocks':[]})==0
    assert target.transformer_blocks[0].img_mod_linear is original


def test_ambiguous_layout_rejected_without_partial_publication():
    target,rows,_,_=fixture()
    original=target.transformer_blocks[0].img_mod_linear
    rows.append(('transformer_blocks.0.img_mod_linear.weight',original.weight))
    with pytest.raises(ValueError,match='Ambiguous'):
        restore_legacy_modulation(target,tree_unflatten(rows))
    assert target.transformer_blocks[0].img_mod_linear is original


def test_missing_output_bias_does_not_publish_replacements():
    target,rows,_,_=fixture()
    original=target.transformer_blocks[0].img_mod_linear
    with pytest.raises(ValueError,match='output modulation bias'):
        restore_legacy_modulation(target,tree_unflatten(rows[:-1]))
    assert target.transformer_blocks[0].img_mod_linear is original
