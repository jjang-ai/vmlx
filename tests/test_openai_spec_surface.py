# SPDX-License-Identifier: Apache-2.0
"""OpenAI spec surface compatibility tests."""

import inspect

import vmlx_engine.server as server


def test_openai_non_v1_aliases_exist():
    source = inspect.getsource(server)
    for route in (
        '/chat/completions',
        '/responses',
        '/responses/{response_id}',
        '/responses/{response_id}/cancel',
        '/responses/{response_id}/input_items',
        '/completions',
        '/embeddings',
        '/models',
        '/files',
        '/files/{file_id}',
        '/files/{file_id}/content',
        '/audio/transcriptions',
        '/audio/translations',
        '/audio/speech',
        '/audio/voices',
        '/images/generations',
        '/images/edits',
        '/images/variations',
        '/moderations',
        '/realtime/sessions',
        '/realtime/client_secrets',
        '/realtime/transcription_sessions',
    ):
        assert route in source


def test_openai_cloud_surface_fallback_prefixes_exist():
    source = inspect.getsource(server)
    for prefix in (
        "assistants",
        "threads",
        "vector_stores",
        "uploads",
        "organization",
        "fine_tuning",
        "chatkit",
    ):
        assert f'"{prefix}"' in source


def test_openai_v1_fallback_supports_all_write_methods():
    source = inspect.getsource(server)
    for route in (
        '/v1/{prefix}/{rest:path}',
    ):
        assert route in source
    for fn_name in (
        'openai_v1_core_get_fallback',
        'openai_v1_core_post_fallback',
        'openai_v1_core_put_fallback',
        'openai_v1_core_patch_fallback',
        'openai_v1_core_delete_fallback',
    ):
        assert f'def {fn_name}' in source


def test_inference_tracking_includes_new_local_apis():
    source = inspect.getsource(server)
    for path in (
        '/lmstudio/v1/chat',
        '/deepgram/vl/listen',
        '/deepgram/vl/speak',
        '/deepgram/vl/read',
        '/responses',
        '/chat/completions',
    ):
        assert path in source
