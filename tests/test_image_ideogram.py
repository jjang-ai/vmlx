"""Native Ideogram presets remain distinct from custom constant guidance."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from PIL import Image
from vmlx_engine.image_gen import ImageGenEngine, SUPPORTED_MODELS, _normalize_for_lookup

@pytest.fixture
def engine():
    e = ImageGenEngine()
    e._model = SimpleNamespace(prompt_cache={})
    e._loaded = True
    e._model_name = 'ideogram-4-fp8'
    e._mflux_class = 'Ideogram4'
    e._generate_with_trace = MagicMock(return_value=(SimpleNamespace(image=Image.new('RGB', (256,256))), MagicMock(job_id='test')))
    return e

@pytest.mark.parametrize('preset,steps', [(None,20),('V4_DEFAULT_20',20),('V4_TURBO_12',12),('V4_QUALITY_48',48)])
def test_native_schedule_not_replaced_by_numeric_defaults(engine,preset,steps):
    result=engine.generate('caption',width=256,height=256,seed=42,preset=preset)
    sent=engine._generate_with_trace.call_args.kwargs
    assert sent['num_inference_steps'] is None and sent['guidance'] is None
    assert sent['preset']==preset and result.steps==steps

@pytest.mark.parametrize('guidance', [None,0,3.5,7])
def test_explicit_custom_parameters_preserved(engine,guidance):
    engine.generate('caption',width=256,height=256,seed=42,steps=9,guidance=guidance)
    sent=engine._generate_with_trace.call_args.kwargs
    assert sent['num_inference_steps']==9 and sent['guidance']==guidance

@pytest.mark.parametrize('kwargs', [dict(preset='bad'),dict(preset='V4_DEFAULT_20',steps=20),dict(preset='V4_DEFAULT_20',guidance=7),dict(guidance=7),dict(width=257),dict(height=128),dict(image_path='foo',image_strength=.8)])
def test_invalid_or_ambiguous_native_controls_fail_before_generation(engine,kwargs):
    args=dict(prompt='caption',width=256,height=256);args.update(kwargs)
    with pytest.raises(ValueError):engine.generate(**args)
    engine._generate_with_trace.assert_not_called()

def test_prompt_feature_cache_keeps_only_current_prompt_size(engine):
    engine.generate('first',width=256,height=256)
    engine._model.prompt_cache['native-feature']='retained'
    engine.generate('first',width=256,height=256,seed=3)
    assert engine._model.prompt_cache=={'native-feature':'retained'}
    engine.generate('second',width=256,height=256)
    assert engine._model.prompt_cache=={}
    engine._model.prompt_cache['feature']='old-size'
    engine.generate('second',width=512,height=256)
    assert engine._model.prompt_cache=={}

@pytest.mark.parametrize('name',['ideogram-4-mflux-q4','ideogram-4-mflux-q8','ideogram-4-fp8'])
def test_saved_bundle_names(name):
    assert SUPPORTED_MODELS[_normalize_for_lookup(name)]=='ideogram-4-fp8'

def test_other_adapter_rejects_preset(engine):
    engine._mflux_class='ZImage'
    with pytest.raises(ValueError,match='supported Ideogram'):engine.generate('caption',preset='V4_DEFAULT_20')
