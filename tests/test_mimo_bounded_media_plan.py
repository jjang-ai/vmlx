"""CPU-only native MiMo media planner: memory and protected-state boundaries."""
import ast
import os
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'vmlx_engine'


def owners():
    ns = {'PrefillAdmissionError': RuntimeError, '_TIGHT_PROJECTED_STEP_CAP': 1024}
    for filename, names in [
        ('utils/prefill_admission.py', {'max_prefill_chunk_tokens', 'replace_chunk_transient_observation'}),
        ('mllm_batch_generator.py', {'_media_chunk_boundaries', '_bounded_mimo_media_plan', '_native_media_clean_boundary', '_media_forward', '_media_placeholder_runs'}),
    ]:
        tree = ast.parse((ROOT / filename).read_text())
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *funcs], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), filename, 'exec'), ns)
    return ns


class BoundedMiMoMediaPlanTests(unittest.TestCase):
    def setUp(self):
        self.ns = owners()

    def plan(self, runs=(), clean=(6232,), active=104395439750, limit=115448725504, step=2048):
        return self.ns['_bounded_mimo_media_plan'](6236, step, 64, 0, active, limit, runs, clean)

    def test_observed_geometry_is_bounded_and_keeps_native_checkpoint(self):
        request = SimpleNamespace(_original_token_ids=[0]*6233, _cached_tokens=0)
        cache = [object()]
        owner = SimpleNamespace(model=SimpleNamespace(_mimo_v26_runtime=True), _media_prefix_cache_allowed=lambda *a: True)
        boundary = self.ns['_native_media_clean_boundary'](owner, request, 6236, cache)
        step, bounds = self.plan(clean=request._media_clean_capture_boundaries)
        self.assertEqual(boundary, 6232)
        self.assertEqual(step, 1024)
        self.assertEqual(bounds, [1024,2048,3072,4096,5120,6144,6232,6236])
        self.assertEqual(len(cache), 1)
        self.assertTrue(request._media_clean_snapshot_allowed)

    def test_protected_runs_are_kept_whole(self):
        runs = [(900,1120), (1800,2400), (3500,3900)]
        step, bounds = self.plan(runs=runs)
        self.assertTrue(all(not a < end < b for end in bounds for a,b in runs))
        self.assertLessEqual(max(b-a for a,b in zip([0]+bounds[:-1], bounds)), step)
        self.assertIn(6232, bounds)

    def test_oversized_run_and_split_checkpoint_decline(self):
        for runs, clean in [([(900,2500)], (6232,)), ([(6100,6235)], (6232,))]:
            with self.subTest(runs=runs), self.assertRaises(RuntimeError):
                self.plan(runs=runs, clean=clean)

    def test_missing_or_exhausted_headroom_declines(self):
        for active, limit in [(0,10), (10,0), (10,10), (20,10)]:
            with self.subTest(active=active,limit=limit), self.assertRaises(RuntimeError):
                self.plan(active=active,limit=limit)

    def test_projection_and_configured_smaller_step_are_respected(self):
        self.assertEqual(self.plan(step=256)[0], 256)
        # 256MiB free -> quarter headroom / (64 heads *6236 context *4 bytes).
        step, bounds = self.plan(active=100000000000, limit=100000000000+256*1024**2)
        self.assertLess(step, 1024)
        self.assertEqual(bounds[-1], 6236)

    def test_actual_media_owner_never_falls_back_when_tight_native(self):
        for cause in ('disabled', 'missing_cache', 'missing_embedding_api'):
            with self.subTest(cause=cause):
                calls = []
                class Model:
                    _mimo_v26_runtime = True
                    def __call__(self, *args, **kwargs):
                        calls.append('unsafe_forward')
                owner = SimpleNamespace(model=Model(), _tight_memory_prefill_drain=True,
                                        language_model=object())
                self.ns.update(os=os, _raise_if_prefill_cancelled=lambda req: None,
                               _media_embed_kwarg_name=lambda lm: None)
                with patch.dict(os.environ, {'VMLX_DISABLE_MEDIA_CHUNKED_PREFILL': '1' if cause == 'disabled' else '0'}):
                    with self.assertRaises(RuntimeError):
                        self.ns['_media_forward'](owner, object(), object(), 6236,
                                                  None if cause == 'missing_cache' else [], {})
                self.assertEqual(calls, [])

    def test_non_native_fallback_remains_available(self):
        calls = []
        class Model:
            _mimo_v26_runtime = False
            def __call__(self, *args, **kwargs):
                calls.append('forward')
                return 'result'
        owner = SimpleNamespace(model=Model(), _tight_memory_prefill_drain=True)
        self.ns.update(os=os, _raise_if_prefill_cancelled=lambda req: None)
        with patch.dict(os.environ, {'VMLX_DISABLE_MEDIA_CHUNKED_PREFILL': '0'}):
            result = self.ns['_media_forward'](owner, object(), object(), 6236, None, {})
        self.assertEqual(result, 'result')
        self.assertEqual(calls, ['forward'])

    def test_actual_owner_keeps_large_transient_after_short_span(self):
        tree = ast.parse((ROOT/'mllm_batch_generator.py').read_text())
        owner = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_media_forward')
        block = next(n for n in ast.walk(owner) if isinstance(n, ast.If)
                     and isinstance(n.test, ast.BoolOp)
                     and ast.unparse(n.test) == 'bounded_glm or bounded_mimo'
                     and any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'peak' for t in x.targets) for x in n.body))
        ns = dict(self.ns, bounded_glm=False, bounded_mimo=True, active=1000,
                  limit=10000, observed_transient=8000, observed_context=4096,
                  observed_width=1024, start=4096, end=4100,
                  request=SimpleNamespace(request_id='r', _cached_tokens=0),
                  mx=SimpleNamespace(get_peak_memory=lambda: 1100),
                  logger=SimpleNamespace(info=lambda *a: None))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), 'owning_observation', 'exec'), ns)
        self.assertEqual((ns['observed_transient'], ns['observed_context'], ns['observed_width']), (8000,4096,1024))

    def test_complete_media_owner_sequences_state_checkpoint_and_logits(self):
        for mode in ('complete', 'cancel', 'admission'):
            with self.subTest(mode=mode):
                ns = owners()
                events = []
                failure = RuntimeError(mode)
                cache = [object()]
                class Array:
                    ndim = 3
                    shape = (1, 6236, 4096)
                    def __getitem__(self, item):
                        return self
                    def tolist(self):
                        return [0] * 6236
                class Logits:
                    def __getitem__(self, item):
                        events.append('logit_slice')
                        return self
                class LM:
                    model_type = 'mimo_v2'
                    def __call__(self, ids, inputs_embeds=None, cache=None):
                        events.append('forward')
                        return SimpleNamespace(logits=Logits())
                class Model:
                    _mimo_v26_runtime = True
                    def get_input_embeddings(self, *a, **k):
                        return SimpleNamespace(inputs_embeds=Array())
                    def __call__(self, *a, **k):
                        raise AssertionError('unbounded fallback')
                request = SimpleNamespace(request_id='r', _cached_tokens=0,
                                          _original_token_ids=[0]*6233)
                owner = SimpleNamespace(model=Model(), language_model=LM(),
                    _tight_memory_prefill_drain=True, prefill_step_size=2048,
                    _media_prefill_chunk_tokens=lambda n: 4096,
                    _media_placeholder_token_ids=lambda: {99},
                    _media_prefix_cache_allowed=lambda *a: True)
                owner._native_media_clean_boundary = lambda req,n,c: ns['_native_media_clean_boundary'](owner,req,n,c)
                def capture(req, actual_cache):
                    self.assertIs(actual_cache, cache)
                    self.assertEqual(req._prefill_tokens_done, 6232)
                    self.assertEqual(events[-1], 'state')
                    events.append('checkpoint')
                owner._maybe_capture_mixed_swa_boundary = capture
                def cancel(req):
                    if mode == 'cancel' and getattr(req, '_prefill_tokens_done', 0) > 0:
                        raise failure
                def valve(*args, **kwargs):
                    events.append('guard')
                    if mode == 'admission':
                        raise failure
                ns.update(os=os, _raise_if_prefill_cancelled=cancel,
                    _media_embed_kwarg_name=lambda lm: 'inputs_embeds',
                    _named_params=lambda fn: {'inputs_embeds','cache'},
                    _infer_attention_heads_for_hybrid_oom_guard=lambda lm: 64,
                    get_effective_metal_working_set_bytes=lambda mx: (104000000000,115000000000),
                    hybrid_chunk_valve_check=valve, prefill_valve_min_margin_bytes=lambda: 0,
                    _materialize_prefill_cache_state=lambda c: events.append('state'),
                    _diag_fingerprints_enabled=lambda: False,
                    _HYBRID_PREFILL_MEM_TRACE=False, _MEDIA_PREFILL_CHUNK_MIN_SEQ=8192,
                    logger=SimpleNamespace(info=lambda *a: None),
                    mx=SimpleNamespace(eval=lambda x: events.append('logit_eval' if isinstance(x,Logits) else 'embedding_eval'),
                        clear_cache=lambda: events.append('clear'), reset_peak_memory=lambda: None,
                        get_peak_memory=lambda: 104000001000))
                with patch.dict(os.environ, {'VMLX_DISABLE_MEDIA_CHUNKED_PREFILL':'0'}):
                    if mode == 'complete':
                        result = ns['_media_forward'](owner, request, Array(),6236,cache,{})
                        self.assertIsInstance(result, Logits)
                        self.assertEqual(events.count('forward'),8)
                        self.assertEqual(events.count('state'),8)
                        self.assertEqual(events.count('checkpoint'),1)
                        self.assertEqual(events[-3:], ['logit_slice','logit_eval','clear'])
                        self.assertEqual(events[0], 'embedding_eval')
                    else:
                        with self.assertRaises(RuntimeError) as caught:
                            ns['_media_forward'](owner,request,Array(),6236,cache,{})
                        self.assertIs(caught.exception,failure)
                        self.assertEqual(events.count('forward'),1 if mode=='cancel' else 0)
                        self.assertNotIn('checkpoint',events)
                for i,event in enumerate(events):
                    if event=='forward':
                        self.assertEqual(events[i-1],'guard')
                        self.assertEqual(events[i+1],'state')

    def test_oversized_merged_media_owner_sequences_state_checkpoint_and_logits(self):
        for mode in ('complete', 'cancel', 'admission'):
            with self.subTest(mode=mode):
                ns = owners()
                events = []
                failure = RuntimeError(mode)
                cache = [object()]
                class Array:
                    ndim = 3
                    shape = (1, 6236, 4096)
                    def __getitem__(self, item):
                        return self
                    def tolist(self):
                        return [0]*100 + [99]*1876 + [0]*(6236-1976)
                class Logits:
                    def __getitem__(self, item):
                        events.append('logit_slice')
                        return self
                class LM:
                    model_type = 'mimo_v2'
                    def __call__(self, ids, inputs_embeds=None, cache=None):
                        events.append('forward')
                        return SimpleNamespace(logits=Logits())
                class Model:
                    _mimo_v26_runtime = True
                    def get_input_embeddings(self, *a, **k):
                        return SimpleNamespace(inputs_embeds=Array())
                    def __call__(self, *a, **k):
                        raise AssertionError('unbounded fallback')
                request = SimpleNamespace(request_id='r', _cached_tokens=0,
                                          _original_token_ids=[0]*6233)
                owner = SimpleNamespace(model=Model(), language_model=LM(),
                    _tight_memory_prefill_drain=True, prefill_step_size=2048,
                    _media_prefill_chunk_tokens=lambda n: 4096,
                    _media_placeholder_token_ids=lambda: {99},
                    _media_prefix_cache_allowed=lambda *a: True)
                owner._native_media_clean_boundary = lambda req,n,c: ns['_native_media_clean_boundary'](owner,req,n,c)
                def capture(req, actual_cache):
                    self.assertIs(actual_cache, cache)
                    self.assertEqual(req._prefill_tokens_done, 6232)
                    self.assertEqual(events[-1], 'state')
                    events.append('checkpoint')
                owner._maybe_capture_mixed_swa_boundary = capture
                def cancel(req):
                    if mode == 'cancel' and getattr(req, '_prefill_tokens_done', 0) > 0:
                        raise failure
                def valve(*args, **kwargs):
                    events.append('guard')
                    if mode == 'admission':
                        raise failure
                ns.update(os=os, _raise_if_prefill_cancelled=cancel,
                    _media_embed_kwarg_name=lambda lm: 'inputs_embeds',
                    _named_params=lambda fn: {'inputs_embeds','cache'},
                    _infer_attention_heads_for_hybrid_oom_guard=lambda lm: 64,
                    get_effective_metal_working_set_bytes=lambda mx: (104000000000,115000000000),
                    hybrid_chunk_valve_check=valve, prefill_valve_min_margin_bytes=lambda: 0,
                    _materialize_prefill_cache_state=lambda c: events.append('state'),
                    _diag_fingerprints_enabled=lambda: False,
                    _HYBRID_PREFILL_MEM_TRACE=False, _MEDIA_PREFILL_CHUNK_MIN_SEQ=8192,
                    logger=SimpleNamespace(info=lambda *a: None),
                    mx=SimpleNamespace(eval=lambda x: events.append('logit_eval' if isinstance(x,Logits) else 'embedding_eval'),
                        clear_cache=lambda: events.append('clear'), reset_peak_memory=lambda: None,
                        get_peak_memory=lambda: 104000001000))
                with patch.dict(os.environ, {'VMLX_DISABLE_MEDIA_CHUNKED_PREFILL':'0'}):
                    if mode == 'complete':
                        result = ns['_media_forward'](owner, request, Array(),6236,cache,{})
                        self.assertIsInstance(result, Logits)
                        self.assertEqual(events.count('forward'),8)
                        self.assertGreater(request._prefill_tokens_done,1976)
                        self.assertEqual(events.count('state'),8)
                        self.assertEqual(events.count('checkpoint'),1)
                        self.assertEqual(events[-3:], ['logit_slice','logit_eval','clear'])
                        self.assertEqual(events[0], 'embedding_eval')
                    else:
                        with self.assertRaises(RuntimeError) as caught:
                            ns['_media_forward'](owner,request,Array(),6236,cache,{})
                        self.assertIs(caught.exception,failure)
                        self.assertEqual(events.count('forward'),1 if mode=='cancel' else 0)
                        self.assertNotIn('checkpoint',events)
                for i,event in enumerate(events):
                    if event=='forward':
                        self.assertEqual(events[i-1],'guard')
                        self.assertEqual(events[i+1],'state')


class OversizedMergedMiMoTests(unittest.TestCase):
    def setUp(self):
        self.ns = owners()

    def plan(self, runs, clean=(2500,), allow=True):
        return self.ns['_bounded_mimo_media_plan'](2506,2048,64,0,104000000000,115000000000,runs,clean,allow_oversized_merged_runs=allow)
    def test_default_remains_fail_closed(self):
        with self.assertRaises(RuntimeError):
            self.plan([(100,1976)], allow=False)
    def test_real_300second_count_geometry_splits_without_gaps(self):
        step,bounds=self.plan([(100,1976)])
        self.assertEqual(step,1024)
        self.assertTrue(any(100<end<1976 for end in bounds))
        self.assertEqual(bounds[-1],2506)
        self.assertIn(2500,bounds)
        self.assertTrue(all(0<end-start<=step for start,end in zip([0]+bounds[:-1],bounds)))
    def test_short_run_partition_is_identical(self):
        for runs in [[],[(900,1120)],[(900,1120),(1800,2400)]]:
            with self.subTest(runs=runs):
                self.assertEqual(self.plan(runs,allow=True),self.plan(runs,allow=False))
    def test_short_neighbor_stays_whole_while_long_run_splits(self):
        runs=[(50,200),(300,1600),(1950,2150)]
        step,bounds=self.plan(runs)
        for a,b in [runs[0],runs[2]]:
            self.assertTrue(all(not a<end<b for end in bounds))
        self.assertTrue(any(300<end<1600 for end in bounds))
    def test_checkpoint_inside_even_long_run_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.plan([(100,1976)],clean=(1500,2500))


if __name__ == '__main__':
    unittest.main()
