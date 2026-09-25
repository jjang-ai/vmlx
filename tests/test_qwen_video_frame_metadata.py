"""CPU-only regressions for sampled video metadata and ordered Qwen templates.

Execute owning functions from source without importing MLX or loading a model.
Decoder/planner stubs isolate metadata transport and frame selection.
"""
import ast
import math
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(os.environ.get("VMLX_VIDEO_METADATA_SOURCE_ROOT", Path(__file__).resolve().parents[1]))

def function(path, name, env):
    tree = ast.parse((ROOT / path).read_text())
    node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), name, 'exec'), env)
    return env[name]

class Indices(list):

    def round(self):
        return Indices((round(x) for x in self))

    def astype(self, _):
        return self

class NP:

    @staticmethod
    def linspace(a, b, n):
        return Indices((a + i * (b - a) / (n - 1) for i in range(n)))
    isfinite = staticmethod(math.isfinite)

class Capture:

    def __init__(self, fps=2):
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        return 9 if key == 1 else self.fps

    def set(self, key, index):
        self.index = index

    def read(self):
        return (self.index != 4, self.index)

    def release(self):
        self.released = True

class TestVideoFrameMetadataHelpers(unittest.TestCase):

    def sample(self, meta, fps=2):
        cap = Capture(fps)
        cv = types.ModuleType('cv2')
        cv.VideoCapture = lambda _: cap
        cv.CAP_PROP_FRAME_COUNT = 1
        cv.CAP_PROP_FPS = 2
        cv.CAP_PROP_POS_FRAMES = 3
        cv.COLOR_BGR2RGB = 4
        cv.cvtColor = lambda x, _: x
        env = {'DEFAULT_FPS': 2, 'MAX_FRAMES': 8, 'np': NP, 'logger': types.SimpleNamespace(info=lambda *a: None), 'smart_nframes': lambda **kw: 5}
        fn = function('vmlx_engine/models/mllm.py', 'extract_video_frames_smart', env)
        with patch.dict(sys.modules, {'cv2': cv}):
            result = fn('fixture', **({'return_metadata': True} if meta else {}))
        self.assertTrue(cap.released)
        return result

    def test_successful_indices_and_times(self):
        frames, meta = self.sample(True)
        self.assertEqual(frames, [0, 2, 6, 8])
        self.assertEqual([m['frame_index'] for m in meta], frames)
        self.assertEqual([m['timestamp_seconds'] for m in meta], [0, 1, 3, 4])

    def test_legacy_return_unchanged(self):
        self.assertEqual(self.sample(False), [0, 2, 6, 8])

    def test_unknown_fps_never_invents_times(self):
        self.assertTrue(all((m['timestamp_seconds'] is None for m in self.sample(True, 0)[1])))

    def build(self, messages):
        util = types.ModuleType('mlx_vlm.prompt_utils')
        util.extract_text_from_content = lambda c: c if isinstance(c, str) else ''.join((p.get('text', '') for p in c))
        util.get_message_json = lambda family, text, role, **kw: {'role': role, 'content': 'LEGACY:' + text}
        env = {}
        function('vmlx_engine/engine/batched.py', '_ordered_qwen_frame_fallback_message', env)
        fn = function('vmlx_engine/engine/batched.py', '_build_messages_preserving_image_owner', env)
        with patch.dict(sys.modules, {'mlx_vlm': types.ModuleType('mlx_vlm'), 'mlx_vlm.prompt_utils': util}):
            return fn({'model_type': 'qwen3_5'}, messages, 2)

    def test_fallback_keeps_adjacency(self):
        msg = {'role': 'user', '_vmlx_qwen_video_frame_metadata': True, 'content': [{'type': 'text', 'text': 'clip1 frame1 t0'}, {'type': 'image_url', 'image_url': {'url': 'a'}}, {'type': 'text', 'text': 'clip1 frame2 t1'}, {'type': 'image_url', 'image_url': {'url': 'b'}}, {'type': 'text', 'text': 'endclip'}]}
        result = self.build([msg])[0]
        self.assertEqual([p['type'] for p in result['content']], ['text', 'image', 'text', 'image', 'text'])
        self.assertEqual(result['content'][2]['text'], 'clip1 frame2 t1')
        self.assertNotIn('_vmlx_qwen_video_frame_metadata', result)

    def test_mixed_native_video_and_fallback_adjacency(self):
        env = {}
        fn = function('vmlx_engine/engine/batched.py', '_ordered_qwen_frame_fallback_message', env)
        parts = [{'type': 'text', 'text': 'clip1frame1'}, {'type': 'image_url'}, {'type': 'text', 'text': 'endclip1'}, {'type': 'video_url'}]
        self.assertEqual([p['type'] for p in fn({'content': parts})['content']], ['text', 'image', 'text', 'video'])

    def test_other_messages_and_tool_metadata_unchanged(self):
        tool = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'a'}], 'reasoning_content': 'native'}
        plain = {'role': 'user', 'content': [{'type': 'text', 'text': 'plain'}]}
        result = self.build([plain, tool])
        self.assertEqual(result[0]['content'], 'LEGACY:plain')
        self.assertIs(result[1], tool)
import types, sys, unittest, os
from unittest.mock import patch
NS = types.SimpleNamespace

class Controls:

    def __init__(self, **kw):
        pass

    def effective_fps(self):
        return 2

    def effective_max_frames(self):
        return 8

    def fallback_bounds(self, **kw):
        return NS(max_long_edge=768, max_pixels=100, resize=None)

    def cache_key_fragment(self):
        return 'fps2max8'

class MetadataError(ValueError):
    pass

class TestQwenVideoFrameFallback(unittest.TestCase):

    def run_fallback(self, messages, family='qwen3_5', badmeta=False, badpaths=False, decodefail=False):
        calls = []
        bounds = []
        m = types.ModuleType('vmlx_engine.models.mllm')
        m.DEFAULT_FPS = 2
        m.MAX_FRAMES = 8

        def sample(path, **kw):
            calls.append((path, kw))
            if decodefail and path == 'second':
                raise RuntimeError('decoder failed')
            frames = [NS(shape=(10, 10, 3), name=f'{path}-{i}') for i in [0, 2, 6, 8]]
            meta = [{'frame_index': i, 'timestamp_seconds': i / 2} for i in [0, 2, 6, 8]]
            return (frames, meta[:-1] if badmeta else meta) if kw.get('return_metadata') else frames
        m.extract_video_frames_smart = sample
        m.process_video_input = lambda x: x
        m.save_frames_to_temp = lambda fs: [f.name for f in fs][1:] if badpaths else [f.name for f in fs]
        v = types.ModuleType('vmlx_engine.video_controls')
        v.VideoControls = Controls
        v.image_pixel_floor = lambda p: 1
        v.image_token_pixels = lambda p: 1
        v.plan_fallback_frames = lambda controls, **kw: NS(num_frames=min(2, kw['frame_cap']), frame_cap=kw['frame_cap'], per_frame_max_pixels=100, expected_tokens_per_frame=1, expected_total=2, budget=None, met=None, reason='test')
        v.subsample_frames_evenly = lambda seq, keep: [seq[0], seq[-1]][:keep]
        v.fallback_plan_diagnostics = lambda *a, **kw: []
        d = types.ModuleType('vmlx_engine.request_diagnostics')
        d.record_for = lambda *a: None

        def bound(frames, **kw):
            bounds.append(kw)
            return frames
        env = {'__package__': 'vmlx_engine.engine', 'Any': object, 'os': os, 'logger': NS(info=lambda *a: None, warning=lambda *a: None, debug=lambda *a: None), '_bound_video_fallback_frames': bound, 'MediaControlsUnmeetableError': type('MediaControlsUnmeetableError', (Exception,), {}), '_VideoFrameMetadataError': MetadataError}
        fn = function('vmlx_engine/engine/batched.py', '_video_frame_fallback_messages', env)
        owner = NS(_model_family_name=lambda: family, _mllm_scheduler=NS(config=NS(max_images_per_request=5)), _count_video_and_image_parts=lambda messages: (2, 1))
        modules = {'vmlx_engine.models.mllm': m, 'vmlx_engine.video_controls': v, 'vmlx_engine.request_diagnostics': d}
        with patch.dict(sys.modules, modules):
            result = fn(owner, messages)
        return (result, calls, bounds)

    def fixture(self):
        return [{'role': 'user', 'content': [{'type': 'text', 'text': 'before'}, {'type': 'image_url', 'image_url': {'url': 'standalone'}}, {'type': 'video_url', 'video_url': {'url': 'first'}}, {'type': 'text', 'text': 'between'}, {'type': 'video_url', 'video_url': {'url': 'second'}}]}]

    def test_two_clips_standalone_bounds_and_selected_time(self):
        result, calls, bounds = self.run_fallback(self.fixture())
        parts = result[0]['content']
        labels = [p['text'] for p in parts if p['type'] == 'text']
        urls = [p['image_url']['url'] for p in parts if p['type'] == 'image_url']
        self.assertEqual(urls, ['standalone', 'first-0', 'first-8', 'second-0', 'second-8'])
        self.assertTrue(any(('Video 2, frame 2/2, source frame 8, nominal time 4.000s' in x for x in labels)))
        self.assertEqual(parts[0]['text'], 'before')
        self.assertIn('between', labels)
        self.assertEqual(len(bounds), 2)
        self.assertTrue(all((x['max_pixels'] == 100 for x in bounds)))

    def test_metadata_mismatch_rejects_not_native_fallback(self):
        with self.assertRaises(MetadataError):
            self.run_fallback(self.fixture(), badmeta=True)

    def test_saved_paths_mismatch_rejects(self):
        with self.assertRaises(MetadataError):
            self.run_fallback(self.fixture(), badpaths=True)

    def test_decode_failure_retains_original_video_without_false_labels(self):
        result, _, _ = self.run_fallback(self.fixture(), decodefail=True)
        parts = result[0]['content']
        self.assertEqual(parts[-1], self.fixture()[0]['content'][-1])
        self.assertFalse(any(('Video 2' in p.get('text', '') for p in parts)))

    def test_native_qwen4_unchanged_no_decode(self):
        messages = self.fixture()
        result, calls, bounds = self.run_fallback(messages, family='qwen4_exp')
        self.assertIs(result, messages)
        self.assertEqual(calls, [])
import types, sys, unittest
from unittest.mock import patch

class MetadataError(ValueError):
    pass

class TestQwenMetadataNormalization(unittest.TestCase):

    def normalize(self, messages, extracted):
        env = {'__package__': 'vmlx_engine.engine', 'mllm_model_type': 'qwen3_5', 'num_images': 1, 'num_videos': 1, 'num_audio': 0, '_VideoFrameMetadataError': MetadataError}
        function('vmlx_engine/engine/batched.py', '_ordered_qwen_frame_fallback_message', env)
        fn = function('vmlx_engine/engine/batched.py', '_normalize_processor_messages', env)
        module = types.ModuleType('vmlx_engine.models.mllm')
        module.MLXMultimodalLM = types.SimpleNamespace(_extract_multimodal_messages=lambda messages: (extracted, [], [], []))
        with patch.dict(sys.modules, {'vmlx_engine.models.mllm': module}):
            return fn(messages)

    def marked(self):
        return {'role': 'user', 'name': 'caller', 'reasoning_content': 'retained-field', 'tool_call_id': 'original-id', '_vmlx_qwen_video_frame_metadata': True, 'content': [{'type': 'text', 'text': 'frame1'}, {'type': 'image_url', 'image_url': {'url': 'image'}}, {'type': 'video_url', 'video_url': {'url': 'native'}}]}

    def test_alignment_error_escapes_broad_catch(self):
        with self.assertRaises(MetadataError):
            self.normalize([self.marked()], [])

    def test_unknown_part_escapes_broad_catch(self):
        msg = self.marked()
        msg['content'].append({'type': 'unsupported'})
        with self.assertRaises(MetadataError):
            self.normalize([msg], [{'role': 'user', 'content': 'flattened'}])

    def test_nondict_part_is_metadata_error(self):
        msg = self.marked()
        msg['content'].append('not a native content part')
        with self.assertRaises(MetadataError):
            self.normalize([msg], [{'role': 'user', 'content': 'flattened'}])

    def test_fields_preserved_and_parts_ordered(self):
        msg = self.marked()
        result = self.normalize([msg], [{'role': 'user', 'content': 'flattened'}])[0]
        for key in ('role', 'name', 'reasoning_content', 'tool_call_id'):
            self.assertEqual(result[key], msg[key])
        self.assertNotIn('_vmlx_qwen_video_frame_metadata', result)
        self.assertEqual([p['type'] for p in result['content']], ['text', 'image', 'video'])

    def test_unmarked_path_unchanged(self):
        existing = [{'role': 'user', 'content': 'old'}]
        self.assertIs(self.normalize([{'role': 'user', 'content': 'original'}], existing), existing)

if __name__ == "__main__":
    unittest.main()
