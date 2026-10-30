# SPDX-License-Identifier: Apache-2.0
"""glm5_next (GLM-5.3-Flash) vendored text runtime — synthetic contract tests.

Real-bundle load/generation receipts live in the private campaign evidence;
these tests pin the source contract on a tiny random model: registration,
config parsing, cache layout, prefill/decode path agreement (the chunked
KDA prefill vs split-recurrent must be BIT-exact — same fp32 recurrence),
the dense-DSA bound refusal, sanitize drops, and registry detection.
"""

import json
import random

import mlx.core as mx
import pytest

from vmlx_engine.models.glm5_next.register import (
    register_glm5_next_runtime,
    register_glm5_next_vlm_runtime,
)


TINY_CFG = {
    "model_type": "glm5_next",
    "text_config": {
        "model_type": "glm5_next_text",
        "hidden_size": 64,
        "num_hidden_layers": 5,
        "vocab_size": 512,
        "num_attention_heads": 4,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "v_head_dim": 16,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "intermediate_size": 96,
        "linear_attn_config": {
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
        "qk_rope_head_dim": 0,
        "index_topk": 200,
        "num_nextn_predict_layers": 1,
        "index_share_for_mtp_iteration": True,
    },
}


@pytest.fixture(scope="module")
def glm5():
    register_glm5_next_runtime()
    import mlx_lm.models.glm5_next as g

    return g


@pytest.fixture(scope="module")
def tiny_model(glm5):
    from vmlx_engine.text_mtp import (
        draft_head_loading_enabled,
        set_draft_head_loading,
    )

    previous = draft_head_loading_enabled()
    set_draft_head_loading(False)
    try:
        model = glm5.Model(glm5.ModelArgs.from_dict(TINY_CFG))
    finally:
        set_draft_head_loading(previous)
    mx.eval(model.parameters())
    return model


class TestRegistration:
    def test_registers_under_mlx_lm_namespace(self, glm5):
        import sys

        assert "mlx_lm.models.glm5_next" in sys.modules
        assert glm5.Model is not None and glm5.ModelArgs is not None

    def test_args_from_nested_text_config(self, glm5):
        args = glm5.ModelArgs.from_dict(TINY_CFG)
        assert args.model_type == "glm5_next"
        assert args.num_hidden_layers == 5
        assert args.linear_num_heads == 4
        assert args.index_topk == 200
        assert args.num_nextn_predict_layers == 1
        assert args.index_share_for_mtp_iteration is True

    def test_compact_mla_defaults_on_with_explicit_legacy_opt_out(
        self, glm5, monkeypatch
    ):
        monkeypatch.delenv("VMLINUX_GLM5_MLA_ABSORB", raising=False)
        monkeypatch.delenv("VMLX_GLM5_MLA_ABSORB", raising=False)
        assert glm5.glm5_mla_absorb_enabled() is True

        monkeypatch.setenv("VMLX_GLM5_MLA_ABSORB", "0")
        assert glm5.glm5_mla_absorb_enabled() is False

        monkeypatch.setenv("VMLINUX_GLM5_MLA_ABSORB", "true")
        assert glm5.glm5_mla_absorb_enabled() is True

    def test_rejects_rope_and_grouped_router(self, glm5):
        bad = json.loads(json.dumps(TINY_CFG))
        bad["text_config"]["qk_rope_head_dim"] = 64
        with pytest.raises(ValueError):
            glm5.ModelArgs.from_dict(bad)
        bad2 = json.loads(json.dumps(TINY_CFG))
        bad2["text_config"]["n_group"] = 8
        with pytest.raises(ValueError):
            glm5.ModelArgs.from_dict(bad2)


class TestCacheAndForward:
    def test_cache_layout(self, tiny_model, glm5):
        cache = tiny_model.make_cache()
        kinds = [type(c) for c in cache]
        assert kinds == [glm5.Glm5KDACache, glm5.Glm5KDACache,
                         glm5.Glm5KDACache, glm5.Glm5MLACache,
                         glm5.Glm5KDACache]
        assert len(cache[0].cache) == 4  # conv_q, conv_k, conv_v, S
        assert len(cache[3].cache) == 4  # keys, values, packed, DSA pool keys

    def test_cache_extract_preserves_empty_slots_and_concrete_types(self, glm5):
        """BatchGenerator's terminal epilogue extracts one request cache.

        GLM mixed-state slots are populated lazily, so valid ``None`` values
        must survive that extraction instead of reaching the upstream
        ``ArraysCache.extract`` implementation, which blindly subscripts
        every slot.  Keeping the concrete classes also preserves KDA rollback
        and MLA/DSA metadata for the scheduler-owned result.
        """

        kda = glm5.Glm5KDACache()
        kda.cache[glm5.KDA_CONV_Q] = mx.arange(12).reshape(2, 2, 3)
        kda.cache[glm5.KDA_STATE] = mx.arange(16).reshape(2, 2, 2, 2)

        extracted_kda = kda.extract(1)
        assert isinstance(extracted_kda, glm5.Glm5KDACache)
        assert extracted_kda.cache[glm5.KDA_CONV_K] is None
        assert extracted_kda.cache[glm5.KDA_CONV_V] is None
        assert extracted_kda.cache[glm5.KDA_CONV_Q].shape == (1, 2, 3)
        assert extracted_kda.cache[glm5.KDA_STATE].shape == (1, 2, 2, 2)

        mla = glm5.Glm5MLACache(kpool=7)
        mla.cache[glm5.MLA_KEYS] = mx.arange(24).reshape(2, 1, 3, 4)
        mla.cache[glm5.MLA_PACKED] = mx.arange(48).reshape(2, 3, 8)

        extracted_mla = mla.extract(0)
        assert isinstance(extracted_mla, glm5.Glm5MLACache)
        assert extracted_mla.kpool == 7
        assert extracted_mla.cache[glm5.MLA_VALUES] is None
        assert extracted_mla.cache[glm5.MLA_POOL_KEYS] is None
        assert extracted_mla.cache[glm5.MLA_KEYS].shape == (1, 1, 3, 4)
        assert extracted_mla.cache[glm5.MLA_PACKED].shape == (1, 3, 8)

    def test_cache_merge_preserves_typed_batch_and_mla_kpool(self, tiny_model, glm5):
        rows = []
        for tokens in ([1, 2, 3, 4], [5, 6, 7, 8]):
            cache = tiny_model.make_cache()
            tiny_model(mx.array([tokens]), cache=cache)
            rows.append(cache)

        merged_kda = glm5.Glm5KDACache.merge([rows[0][0], rows[1][0]])
        merged_mla = glm5.Glm5MLACache.merge([rows[0][3], rows[1][3]])
        assert isinstance(merged_kda, glm5.Glm5KDACache)
        assert isinstance(merged_mla, glm5.Glm5MLACache)
        assert merged_kda.cache[glm5.KDA_STATE].shape[0] == 2
        assert merged_mla.cache[glm5.MLA_KEYS].shape[0] == 2
        assert merged_mla.kpool == rows[0][3].kpool == rows[1][3].kpool
        assert merged_mla.extract(1).offset == rows[1][3].offset == 4

        wrong_kpool = glm5.Glm5MLACache(kpool=merged_mla.kpool + 1)
        with pytest.raises(ValueError, match="disagree on DSA kpool"):
            glm5.Glm5MLACache.merge([rows[0][3], wrong_kpool])

    def test_typed_cache_clone_preserves_every_native_state_without_aliasing(
        self, tiny_model, glm5
    ):
        from vmlx_engine.models.glm5_next.glm5_next import (
            clone_glm5_next_layer_cache,
        )

        live = tiny_model.make_cache()
        tiny_model(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=live)
        mx.eval(*[value for layer in live for value in layer.state if value is not None])

        def copy_array(value):
            copied = value + mx.zeros_like(value)
            mx.eval(copied)
            return copied

        cloned = [
            clone_glm5_next_layer_cache(layer, copy_fn=copy_array)
            for layer in live
        ]
        assert [type(layer) for layer in cloned] == [type(layer) for layer in live]
        assert cloned[3].kpool == live[3].kpool
        assert cloned[3].offset == live[3].offset == 8
        for source, copy in zip(live, cloned):
            assert copy.meta_state == source.meta_state
            for original, duplicate in zip(source.state, copy.state):
                if original is None:
                    assert duplicate is None
                else:
                    assert duplicate is not original
                    assert duplicate.shape == original.shape
                    assert duplicate.dtype == original.dtype
                    if original.size:
                        assert float(mx.max(mx.abs(original - duplicate))) == 0.0

    def test_typed_cache_metadata_rejects_partial_or_misaligned_state(self, glm5):
        with pytest.raises(ValueError, match="typed schema"):
            glm5.Glm5KDACache.from_state([None] * 4, ())
        with pytest.raises(ValueError, match="partial recurrent"):
            glm5.Glm5KDACache.from_state(
                [mx.zeros((1, 3, 4)), None, None, None],
                ("glm5_next_native_v1", "kda"),
            )
        with pytest.raises(ValueError, match="typed offset"):
            glm5.Glm5MLACache.from_state(
                [
                    mx.zeros((1, 2, 3, 4)),
                    mx.zeros((1, 2, 3, 4)),
                    mx.zeros((1, 3, 8)),
                    None,
                ],
                ("glm5_next_native_v1", "mla", "4", "4"),
            )

    def test_absorbed_mla_cache_schema_trim_and_clone_are_isolated(self, glm5):
        from vmlx_engine.models.glm5_next.glm5_next import (
            clone_glm5_next_layer_cache,
        )

        cache = glm5.Glm5MLACache(kpool=4, absorbed=True)
        cache.update_latent(mx.arange(160).reshape(1, 1, 10, 16))
        cache.update_packed(mx.arange(80).reshape(1, 10, 8))
        cache.update_pool_keys(mx.arange(8).reshape(1, 2, 4))
        mx.eval(*cache.state)

        clone = clone_glm5_next_layer_cache(
            cache,
            copy_fn=lambda value: value + mx.zeros_like(value),
        )
        assert clone.absorbed is True
        assert clone.meta_state == (
            "glm5_next_native_v2",
            "mla_absorbed",
            "4",
            "10",
        )
        assert clone.offset == 10

        cache.update_latent(mx.zeros((1, 1, 2, 16)))
        cache.update_packed(mx.zeros((1, 2, 8)))
        cache.update_pool_keys(mx.zeros((1, 1, 4)))
        assert cache.offset == 12
        assert clone.offset == 10
        assert clone.pool_keys.shape == (1, 2, 4)

        assert cache.trim(3) == 3
        assert cache.offset == 9
        assert cache.cache[glm5.MLA_LATENT_PACKED].shape[1] == 9
        assert cache.pool_keys.shape[1] == 2

        with pytest.raises(ValueError, match="latent must be"):
            glm5.Glm5MLACache.from_state(
                [
                    mx.zeros((1, 2, 4, 8)),
                    mx.zeros((1, 2, 4, 8)),
                    mx.zeros((1, 4, 8)),
                    mx.zeros((1, 1, 8)),
                ],
                ("glm5_next_native_v2", "mla_absorbed", "4", "4"),
            )

    def test_absorbed_mla_capacity_append_matches_concatenate_reference(
        self, glm5
    ):
        """Decode appends reuse bounded buffers without changing logical bytes."""

        cache = glm5.Glm5MLACache(kpool=4, absorbed=True)
        latent_parts = [mx.arange(128).reshape(1, 1, 8, 16)]
        packed_parts = [mx.arange(64).reshape(1, 8, 8)]
        pool_parts = [mx.arange(8).reshape(1, 2, 4)]
        cache.update_latent(latent_parts[0])
        cache.update_packed(packed_parts[0])
        cache.update_pool_keys(pool_parts[0])

        # The first decode append grows to the bounded architecture block.
        latent_parts.append(mx.full((1, 1, 4, 16), 901))
        packed_parts.append(mx.full((1, 4, 8), 902))
        pool_parts.append(mx.full((1, 1, 4), 903))
        cache.update_latent(latent_parts[-1])
        cache.update_packed(packed_parts[-1])
        cache.update_pool_keys(pool_parts[-1])
        latent_buffer = cache._capacity_buffers[glm5.MLA_LATENT]
        packed_buffer = cache._capacity_buffers[glm5.MLA_LATENT_PACKED]
        pool_buffer = cache._capacity_buffers[glm5.MLA_LATENT_POOL_KEYS]
        assert latent_buffer.shape[2] == 2048
        assert packed_buffer.shape[1] == 2048
        assert pool_buffer.shape[1] == 512

        # Further decode-sized appends stay in those same physical buffers.
        for index, value in enumerate(range(904, 912), start=1):
            latent = mx.full((1, 1, 1, 16), value)
            packed = mx.full((1, 1, 8), value + 100)
            latent_parts.append(latent)
            packed_parts.append(packed)
            cache.update_latent(latent)
            cache.update_packed(packed)
            if index % 4 == 0:
                pool = mx.full((1, 1, 4), value + 200)
                pool_parts.append(pool)
                cache.update_pool_keys(pool)
        assert cache._capacity_buffers[glm5.MLA_LATENT] is latent_buffer
        assert cache._capacity_buffers[glm5.MLA_LATENT_PACKED] is packed_buffer

        expected_latent = mx.concatenate(latent_parts, axis=2)
        expected_packed = mx.concatenate(packed_parts, axis=1)
        expected_pool = mx.concatenate(pool_parts, axis=1)
        mx.eval(*cache.state, expected_latent, expected_packed, expected_pool)
        assert bool(
            mx.array_equal(cache.cache[glm5.MLA_LATENT], expected_latent).item()
        )
        assert bool(
            mx.array_equal(
                cache.cache[glm5.MLA_LATENT_PACKED], expected_packed
            ).item()
        )
        assert bool(mx.array_equal(cache.pool_keys, expected_pool).item())
        assert cache.offset == expected_latent.shape[2]

        logical_nbytes = sum(value.nbytes for value in cache.state)
        assert cache.nbytes > logical_nbytes

    def test_absorbed_mla_identity_clone_is_copy_on_write_after_trim(
        self, glm5
    ):
        """An N-1 wrapper may share bytes but never observe later tail writes."""

        from vmlx_engine.models.glm5_next.glm5_next import (
            clone_glm5_next_layer_cache,
        )

        cache = glm5.Glm5MLACache(kpool=4, absorbed=True)
        cache.update_latent(mx.arange(160).reshape(1, 1, 10, 16))
        cache.update_packed(mx.arange(80).reshape(1, 10, 8))
        cache.update_pool_keys(mx.arange(8).reshape(1, 2, 4))
        mx.eval(*cache.state)

        boundary = clone_glm5_next_layer_cache(
            cache,
            copy_fn=lambda value: value,
        )
        boundary_state = [mx.array(value) for value in boundary.state]
        mx.eval(*boundary_state)
        assert (
            cache._capacity_buffers[glm5.MLA_LATENT]
            is boundary._capacity_buffers[glm5.MLA_LATENT]
        )

        cache.update_latent(mx.full((1, 1, 2, 16), 777))
        cache.update_packed(mx.full((1, 2, 8), 778))
        cache.update_pool_keys(mx.full((1, 1, 4), 779))
        mx.eval(*cache.state)
        assert (
            cache._capacity_buffers[glm5.MLA_LATENT]
            is not boundary._capacity_buffers[glm5.MLA_LATENT]
        )
        for expected, actual in zip(boundary_state, boundary.state):
            assert bool(mx.array_equal(expected, actual).item())

        # Reusing slots exposed by a rollback also detaches before writing.
        assert boundary.trim(3) == 3
        boundary.update_latent(mx.full((1, 1, 1, 16), 880))
        boundary.update_packed(mx.full((1, 1, 8), 881))
        mx.eval(*boundary.state)
        assert boundary.offset == 8
        assert cache.offset == 12
        assert bool(
            mx.array_equal(cache.cache[glm5.MLA_LATENT][..., :10, :], boundary_state[0]).item()
        )

    def test_absorbed_mla_persistence_excludes_spare_capacity(self, glm5):
        cache = glm5.Glm5MLACache(kpool=4, absorbed=True)
        cache.update_latent(mx.zeros((1, 1, 8, 16)))
        cache.update_packed(mx.zeros((1, 8, 8)))
        cache.update_pool_keys(mx.zeros((1, 2, 4)))
        cache.update_latent(mx.ones((1, 1, 1, 16)))
        cache.update_packed(mx.ones((1, 1, 8)))
        mx.eval(*cache.state)

        assert cache._capacity_buffers[glm5.MLA_LATENT].shape[2] == 2048
        assert cache.state[glm5.MLA_LATENT].shape[2] == 9
        restored = glm5.Glm5MLACache.from_state(cache.state, cache.meta_state)
        assert restored.offset == 9
        assert restored._capacity_buffers[glm5.MLA_LATENT].shape[2] == 9
        assert restored._capacity_buffers[glm5.MLA_LATENT_PACKED].shape[1] == 9
        assert restored._capacity_buffers[glm5.MLA_LATENT_POOL_KEYS].shape[1] == 2
        for expected, actual in zip(cache.state, restored.state):
            assert bool(mx.array_equal(expected, actual).item())

    def test_live_validator_accepts_glm_typed_metadata_and_rejects_drift(
        self, tiny_model, glm5
    ):
        """Generic live validation must delegate GLM's non-numeric meta schema."""
        from vmlx_engine.cache_record_validator import validate_live_cache

        live = tiny_model.make_cache()
        tiny_model(mx.array([[1, 2, 3, 4]]), cache=live)
        mx.eval(
            *[
                value
                for layer in live
                for value in layer.state
                if value is not None
            ]
        )
        ok, reason, nbytes = validate_live_cache(
            live,
            expected_num_layers=5,
            source="test:glm5-native",
        )
        assert ok is True, reason
        assert nbytes > 0

        bad_kda = {
            "class_name": "Glm5KDACache",
            "state": live[0].state,
            "meta_state": ("wrong_schema", "kda"),
        }
        ok, reason, _ = validate_live_cache([bad_kda], source="test:glm5-bad-kda")
        assert ok is False
        assert "typed schema" in reason

        bad_mla = {
            "class_name": "Glm5MLACache",
            "state": live[3].state,
            "meta_state": (*live[3].meta_state[:3], str(live[3].offset + 1)),
        }
        ok, reason, _ = validate_live_cache([bad_mla], source="test:glm5-bad-mla")
        assert ok is False
        assert "lengths do not match" in reason

    def test_single_batch_snapshot_is_exact_n_minus_one(self, tiny_model, glm5):
        from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

        generator = SingleBatchGenerator(tiny_model, max_tokens=1)
        generator.insert([[10, 11, 12, 13]])
        prompt_responses, generation_responses = generator.next()

        assert generation_responses == []
        response = prompt_responses[0]
        snapshot = response.prompt_cache_snapshot
        assert snapshot is not None
        assert isinstance(snapshot[0], glm5.Glm5KDACache)
        assert isinstance(snapshot[3], glm5.Glm5MLACache)
        assert snapshot[3].offset == 3
        assert response.prompt_cache[3].offset >= 4
        assert snapshot[0].cache[glm5.KDA_STATE] is not response.prompt_cache[0].cache[
            glm5.KDA_STATE
        ]

    def test_disk_cache_roundtrip_restores_both_typed_cache_classes(
        self, tiny_model, glm5, tmp_path
    ):
        from vmlx_engine.disk_cache import DiskCacheManager
        from vmlx_engine.models.glm5_next.glm5_next import (
            clone_glm5_next_layer_cache,
        )

        # This row owns the legacy materialized-v1 safetensors contract; the
        # following row independently owns compact-v2 persistence.
        live = [
            glm5.Glm5KDACache()
            if layer.is_linear
            else glm5.Glm5MLACache(
                tiny_model.args.index_kpool,
                absorbed=False,
            )
            for layer in tiny_model.model.layers
        ]
        tiny_model(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=live)
        assert live[3].cache[glm5.MLA_POOL_KEYS] is None
        payload = [
            type(layer).from_state(
                [
                    None if value is None else value.astype(mx.bfloat16)
                    for value in layer.state
                ],
                layer.meta_state,
            )
            for layer in live
        ]
        mx.eval(
            *[
                value
                for layer in payload
                for value in layer.state
                if value is not None
            ]
        )
        cache_dir = tmp_path / "glm5-l2"
        required = ("Glm5KDACache", "Glm5MLACache")
        writer = DiskCacheManager(
            str(cache_dir),
            expected_num_layers=5,
            required_cache_classes=required,
        )
        cache_tokens = list(range(9))
        assert writer.store(cache_tokens, payload)
        assert writer.flush_pending_writes(cache_tokens) is True
        writer.shutdown()

        cache_file = next(cache_dir.glob("*.safetensors"))
        raw_arrays, metadata = mx.load(str(cache_file), return_metadata=True)
        bitcast = json.loads(metadata["1.bitcast_dtypes"])
        assert bitcast
        assert all(raw_arrays[key].dtype == mx.uint16 for key in bitcast)

        reader = DiskCacheManager(
            str(cache_dir),
            expected_num_layers=5,
            required_cache_classes=required,
        )
        try:
            restored = reader.fetch(list(range(9)))
            assert restored is not None
            assert {type(layer).__name__ for layer in restored} == set(required)
            assert restored[3].offset == 8
            assert restored[3].cache[glm5.MLA_POOL_KEYS].shape[1] == 0
            assert restored[0].cache[glm5.KDA_STATE].dtype == mx.bfloat16
            for expected_layer, restored_layer in zip(payload, restored):
                assert restored_layer.meta_state == expected_layer.meta_state
                for expected, actual in zip(
                    expected_layer.state,
                    restored_layer.state,
                ):
                    assert actual.shape == expected.shape
                    assert actual.dtype == expected.dtype
                    if expected.size:
                        assert float(mx.max(mx.abs(actual - expected))) == 0.0
        finally:
            reader.shutdown()

    def test_absorbed_mla_disk_roundtrip_preserves_v2_state(
        self, tiny_model, glm5, tmp_path
    ):
        from vmlx_engine.disk_cache import DiskCacheManager

        live = [
            glm5.Glm5KDACache()
            if layer.is_linear
            else glm5.Glm5MLACache(
                tiny_model.args.index_kpool,
                absorbed=True,
            )
            for layer in tiny_model.model.layers
        ]
        tiny_model(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=live)
        mx.eval(
            *[
                value
                for layer in live
                for value in layer.state
                if value is not None
            ]
        )
        assert live[3].meta_state[:2] == (
            "glm5_next_native_v2",
            "mla_absorbed",
        )
        assert live[3].offset == 8

        cache_dir = tmp_path / "glm5-absorbed-l2"
        required = ("Glm5KDACache", "Glm5MLACache")
        writer = DiskCacheManager(
            str(cache_dir),
            expected_num_layers=5,
            required_cache_classes=required,
        )
        cache_tokens = list(range(9))
        assert writer.store(cache_tokens, live)
        assert writer.flush_pending_writes(cache_tokens) is True
        writer.shutdown()

        reader = DiskCacheManager(
            str(cache_dir),
            expected_num_layers=5,
            required_cache_classes=required,
        )
        try:
            restored = reader.fetch(cache_tokens)
            assert restored is not None
            restored_mla = restored[3]
            assert restored_mla.absorbed is True
            assert restored_mla.meta_state == live[3].meta_state
            for expected, actual in zip(live[3].state, restored_mla.state):
                assert actual.shape == expected.shape
                assert actual.dtype == expected.dtype
                if expected.size:
                    assert bool(mx.array_equal(expected, actual).item())
        finally:
            reader.shutdown()

    def test_disk_cache_refuses_incomplete_typed_glm_payload(
        self, glm5, tmp_path
    ):
        from vmlx_engine.disk_cache import DiskCacheManager

        manager = DiskCacheManager(
            str(tmp_path / "glm5-incomplete-l2"),
            required_cache_classes=("Glm5KDACache", "Glm5MLACache"),
        )
        try:
            assert manager.store([1, 2], [glm5.Glm5KDACache()]) is False
        finally:
            manager.shutdown()

    def test_memory_cache_hit_is_typed_isolated_and_not_reverse_truncatable(
        self, tiny_model, glm5
    ):
        from vmlx_engine.memory_cache import (
            MemoryAwarePrefixCache,
            MemoryCacheConfig,
        )

        stored = tiny_model.make_cache()
        tiny_model(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=stored)
        prefix = MemoryAwarePrefixCache(
            model=tiny_model,
            config=MemoryCacheConfig(max_memory_mb=512),
            model_path="/tmp/glm5-next-typed-cache-test",
        )
        tokens = list(range(8))
        assert prefix.store(tokens, stored)

        exact, remaining = prefix.fetch(tokens)
        assert exact is not None
        assert remaining == []
        assert [type(layer) for layer in exact] == [type(layer) for layer in stored]
        assert exact[3].offset == stored[3].offset == 8
        assert exact[0] is not stored[0]
        assert exact[3] is not stored[3]
        assert exact[0].cache[glm5.KDA_STATE] is stored[0].cache[glm5.KDA_STATE]
        assert exact[3].cache[glm5.MLA_KEYS] is stored[3].cache[glm5.MLA_KEYS]

        tiny_model(mx.array([[9]]), cache=exact)
        assert exact[3].offset == 9
        assert stored[3].offset == 8
        assert exact[0].cache[glm5.KDA_STATE] is not stored[0].cache[glm5.KDA_STATE]
        assert exact[3].cache[glm5.MLA_KEYS] is not stored[3].cache[glm5.MLA_KEYS]

        forward, remaining = prefix.fetch(tokens + [8])
        assert forward is not None
        assert remaining == [8]
        assert forward[3].offset == 8
        assert forward[0].cache[glm5.KDA_STATE] is stored[0].cache[glm5.KDA_STATE]

        reverse, remaining = prefix.fetch(tokens[:-1])
        assert reverse is None
        assert remaining == tokens[:-1]

    def test_prefill_and_decode_shapes(self, tiny_model):
        cache = tiny_model.make_cache()
        out = tiny_model(mx.array([[1, 2, 3, 4, 5, 6, 7]]), cache=cache)
        assert out.shape == (1, 7, 512)
        out2 = tiny_model(mx.array([[8]]), cache=cache)
        assert out2.shape == (1, 1, 512)
        # KDA state is fp32 and fixed-shape after decode
        assert cache[0].cache[3].dtype == mx.float32
        assert cache[0].cache[3].shape == (1, 4, 16, 16)

    @pytest.mark.parametrize("sparse", [False, True])
    def test_absorbed_mla_matches_materialized_prefill_and_decode(
        self, tiny_model, glm5, sparse
    ):
        attention = tiny_model.model.layers[3].self_attn
        old_attention_topk = attention.index_topk
        old_indexer_topk = attention.indexer.topk
        if sparse:
            attention.index_topk = 4
            attention.indexer.topk = 4

        def make_cache(absorbed):
            return [
                glm5.Glm5KDACache()
                if layer.is_linear
                else glm5.Glm5MLACache(
                    tiny_model.args.index_kpool,
                    absorbed=absorbed,
                )
                for layer in tiny_model.model.layers
            ]

        try:
            tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
            materialized = make_cache(False)
            absorbed = make_cache(True)
            materialized_logits = tiny_model(tokens, cache=materialized)
            absorbed_logits = tiny_model(tokens, cache=absorbed)
            materialized_decode = tiny_model(
                mx.array([[13]]), cache=materialized
            )
            absorbed_decode = tiny_model(mx.array([[13]]), cache=absorbed)
            mx.eval(
                materialized_logits,
                absorbed_logits,
                materialized_decode,
                absorbed_decode,
            )

            assert float(mx.max(mx.abs(
                materialized_logits - absorbed_logits
            ))) < 0.005
            assert int(mx.argmax(materialized_logits[0, -1])) == int(
                mx.argmax(absorbed_logits[0, -1])
            )
            assert float(mx.max(mx.abs(
                materialized_decode - absorbed_decode
            ))) < 0.005
            assert int(mx.argmax(materialized_decode[0, -1])) == int(
                mx.argmax(absorbed_decode[0, -1])
            )
            assert materialized[3].offset == absorbed[3].offset == 13
            if sparse:
                assert materialized[3].pool_keys.shape == (1, 3, 128)
                assert absorbed[3].pool_keys.shape == (1, 3, 128)
        finally:
            attention.index_topk = old_attention_topk
            attention.indexer.topk = old_indexer_topk

    def test_mla_pool_cache_trim_keeps_only_complete_pools(self, glm5):
        cache = glm5.Glm5MLACache(kpool=4, absorbed=False)
        cache.cache[glm5.MLA_KEYS] = mx.zeros((1, 2, 10, 8))
        cache.cache[glm5.MLA_VALUES] = mx.zeros((1, 2, 10, 8))
        cache.cache[glm5.MLA_PACKED] = mx.zeros((1, 10, 16))
        cache.cache[glm5.MLA_POOL_KEYS] = mx.zeros((1, 2, 8))

        assert cache.trim(3) == 3
        assert cache.offset == 7
        assert cache.cache[glm5.MLA_PACKED].shape[1] == 7
        assert cache.cache[glm5.MLA_POOL_KEYS].shape[1] == 1

    def test_chunked_prefill_bit_exact_vs_split_recurrent(self, tiny_model):
        """T=100 goes through kda_chunked; 50+50 goes through kda_recurrent
        twice. Same fp32 recurrence — logits and state must be BIT-exact."""
        random.seed(7)
        toks = [random.randint(1, 500) for _ in range(100)]
        ids = mx.array([toks])
        c1 = tiny_model.make_cache()
        o1 = tiny_model(ids, cache=c1)
        c2 = tiny_model.make_cache()
        tiny_model(ids[:, :50], cache=c2)
        o2 = tiny_model(ids[:, 50:], cache=c2)
        mx.eval(o1, o2)
        assert float(mx.abs(o1[0, -1] - o2[0, -1]).max()) == 0.0
        assert float(mx.abs(c1[0].cache[3] - c2[0].cache[3]).max()) == 0.0
        # MLA KV offsets agree
        assert c1[3].offset == c2[3].offset == 100

    def test_context_bound_refuses_loudly(self, glm5):
        cfg = json.loads(json.dumps(TINY_CFG))
        cfg["text_config"]["max_position_embeddings"] = 64
        model = glm5.Model(glm5.ModelArgs.from_dict(cfg))
        mx.eval(model.parameters())
        with pytest.raises(ValueError, match="context limit"):
            model(mx.array([[1] * 65]))

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    def test_kda_qkv_conv_fusion_matches_stock(self, dtype):
        from vmlx_engine.metal.kda_conv_decode import (
            glm5_kda_conv_decode,
            glm5_kda_conv_status,
        )
        from vmlx_engine.models.glm5_next.kda import short_conv

        channels = 64
        kernel_size = 4
        arrays = [
            (mx.random.normal((1, 1, channels)) * 0.2).astype(dtype)
            for _ in range(3)
        ]
        states = [
            (mx.random.normal((1, kernel_size - 1, channels)) * 0.2).astype(
                dtype
            )
            for _ in range(3)
        ]
        weights = [
            (mx.random.normal((channels, kernel_size)) * 0.2).astype(dtype)
            for _ in range(3)
        ]
        reference = [
            short_conv(array, weight, state)
            for array, weight, state in zip(arrays, weights, states)
        ]
        candidate = glm5_kda_conv_decode(
            *arrays,
            *states,
            *weights,
            enabled=True,
        )
        assert candidate is not None
        assert glm5_kda_conv_status()["observed_calls"] == 1
        mx.eval(*candidate, *(value for pair in reference for value in pair))
        for index, (expected_out, expected_state) in enumerate(reference):
            actual_out = candidate[index]
            actual_state = candidate[index + 3]
            assert mx.array_equal(actual_state, expected_state)
            assert mx.allclose(
                actual_out,
                expected_out,
                rtol=8e-3,
                atol=8e-3,
            )

    def test_kda_qkv_conv_fusion_refuses_prefill(self):
        from vmlx_engine.metal.kda_conv_decode import glm5_kda_conv_decode

        arrays = [mx.zeros((1, 2, 64), dtype=mx.float16) for _ in range(3)]
        states = [mx.zeros((1, 3, 64), dtype=mx.float16) for _ in range(3)]
        weights = [mx.zeros((64, 4), dtype=mx.float16) for _ in range(3)]
        assert glm5_kda_conv_decode(
            *arrays,
            *states,
            *weights,
            enabled=True,
        ) is None

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
    def test_kda_step_fusion_matches_stock(self, dtype):
        from vmlx_engine.metal.kda_step_decode import (
            glm5_kda_step_decode,
            glm5_kda_step_status,
        )
        from vmlx_engine.models.glm5_next.kda import kda_step

        heads, key_dim, value_dim = 4, 16, 16
        q = (mx.random.normal((1, heads, key_dim)) * 0.2).astype(dtype)
        k = (mx.random.normal((1, heads, key_dim)) * 0.2).astype(dtype)
        q = q.astype(mx.float32)
        q = q * mx.rsqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        q = q.astype(dtype)
        k = k.astype(mx.float32)
        k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        k = k.astype(dtype)
        v = (mx.random.normal((1, heads, value_dim)) * 0.2).astype(dtype)
        g = (-mx.abs(mx.random.normal((1, heads, key_dim))) * 0.05).astype(
            dtype
        )
        beta = mx.sigmoid(mx.random.normal((1, heads))).astype(dtype)
        state = (mx.random.normal((1, heads, key_dim, value_dim)) * 0.1).astype(
            mx.float32
        )

        reference = kda_step(q, k, v, g, beta, state)
        candidate = glm5_kda_step_decode(
            q,
            k,
            v,
            g,
            beta,
            state,
            enabled=True,
        )
        assert candidate is not None
        assert glm5_kda_step_status()["observed_calls"] == 1
        mx.eval(*reference, *candidate)
        assert mx.allclose(candidate[0], reference[0], rtol=2e-4, atol=2e-5)
        assert mx.allclose(candidate[1], reference[1], rtol=2e-4, atol=2e-5)

    def test_kda_step_fusion_refuses_non_fp32_state(self):
        from vmlx_engine.metal.kda_step_decode import glm5_kda_step_decode

        q = mx.zeros((1, 4, 16), dtype=mx.float16)
        v = mx.zeros((1, 4, 16), dtype=mx.float16)
        beta = mx.zeros((1, 4), dtype=mx.float16)
        state = mx.zeros((1, 4, 16, 16), dtype=mx.float16)
        assert glm5_kda_step_decode(
            q, q, v, q, beta, state, enabled=True
        ) is None

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
    def test_mhc_decode_fusion_matches_stock(self, dtype, glm5):
        from vmlx_engine.metal.glm5_mhc_decode import (
            glm5_mhc_decode,
            glm5_mhc_status,
        )

        args = glm5.ModelArgs.from_dict(TINY_CFG)
        module = glm5.HyperConnection(args)
        module.hc_fn = (mx.random.normal(module.hc_fn.shape) * 0.01).astype(dtype)
        module.hc_base = (mx.random.normal(module.hc_base.shape) * 0.01).astype(
            mx.float32
        )
        streams = (mx.random.normal((1, 1, args.hc_mult, 64)) * 0.1).astype(
            dtype
        )

        module._fused_decode = False
        reference = module(streams)
        candidate = glm5_mhc_decode(
            streams,
            module.hc_fn,
            module.hc_base,
            module.hc_scale,
            rms_eps=module.rms_eps,
            sink_eps=module.eps,
            iterations=module.iters,
            enabled=True,
            verify_enabled=True,
        )
        assert candidate is not None
        assert glm5_mhc_status()["observed_calls"] == 1
        mx.eval(*reference, *candidate)
        assert mx.allclose(candidate[0], reference[0], rtol=2e-4, atol=2e-5)
        assert mx.allclose(candidate[1], reference[1], rtol=2e-4, atol=2e-5)
        assert mx.allclose(candidate[2], reference[2], rtol=2e-4, atol=2e-5)

    @pytest.mark.parametrize("rows", [2, 3, 4])
    def test_mhc_decode_fusion_matches_mtp_verify_slabs(self, rows, glm5):
        from vmlx_engine.metal.glm5_mhc_decode import glm5_mhc_decode

        # Seed: the fused-vs-stock tolerance is tight enough that unseeded
        # inputs made this order-dependent across the suite (passed 3/3 in
        # isolation, failed only after other tests advanced the global RNG).
        mx.random.seed(1733 + rows)
        args = glm5.ModelArgs.from_dict(TINY_CFG)
        module = glm5.HyperConnection(args)
        module.hc_fn = (mx.random.normal(module.hc_fn.shape) * 0.01).astype(
            mx.bfloat16
        )
        module.hc_base = (mx.random.normal(module.hc_base.shape) * 0.01).astype(
            mx.float32
        )
        streams = (
            mx.random.normal((1, rows, args.hc_mult, args.hidden_size)) * 0.1
        ).astype(mx.bfloat16)
        module._fused_decode = False
        reference = module(streams)
        candidate = glm5_mhc_decode(
            streams,
            module.hc_fn,
            module.hc_base,
            module.hc_scale,
            rms_eps=module.rms_eps,
            sink_eps=module.eps,
            iterations=module.iters,
            enabled=True,
            verify_enabled=True,
        )
        assert candidate is not None
        mx.eval(*reference, *candidate)
        assert mx.allclose(candidate[0], reference[0], rtol=2e-4, atol=2e-5)
        assert mx.allclose(candidate[1], reference[1], rtol=2e-4, atol=2e-5)
        max_abs = float(
            mx.max(
                mx.abs(
                    candidate[2].astype(mx.float32)
                    - reference[2].astype(mx.float32)
                )
            ).item()
        )
        max_ref = max(
            float(mx.max(mx.abs(reference[2].astype(mx.float32))).item()),
            1e-9,
        )
        # The direct Metal reduction and MLX matmul use different accumulation
        # orders. Bound the BF16 collapsed stream to the same one-storage-step
        # envelope as the already-qualified single-token kernel.
        assert max_abs / max_ref <= 6e-3

    def test_mhc_decode_fusion_refuses_long_prefill(self, glm5):
        from vmlx_engine.metal.glm5_mhc_decode import glm5_mhc_decode

        args = glm5.ModelArgs.from_dict(TINY_CFG)
        module = glm5.HyperConnection(args)
        streams = mx.zeros((1, 5, args.hc_mult, args.hidden_size))
        assert glm5_mhc_decode(
            streams,
            module.hc_fn,
            module.hc_base,
            module.hc_scale,
            rms_eps=module.rms_eps,
            sink_eps=module.eps,
            iterations=module.iters,
            enabled=True,
        ) is None

    def test_mhc_decode_verify_slab_defaults_to_fused_with_opt_out(
        self, monkeypatch, glm5
    ):
        from vmlx_engine.metal.glm5_mhc_decode import glm5_mhc_decode

        monkeypatch.delenv("VMLX_GLM5_FUSED_MHC_VERIFY", raising=False)
        monkeypatch.delenv("VMLINUX_GLM5_FUSED_MHC_VERIFY", raising=False)
        args = glm5.ModelArgs.from_dict(TINY_CFG)
        module = glm5.HyperConnection(args)
        streams = mx.zeros((1, 4, args.hc_mult, args.hidden_size))
        kwargs = dict(
            rms_eps=module.rms_eps,
            sink_eps=module.eps,
            iterations=module.iters,
            enabled=True,
        )
        assert glm5_mhc_decode(
            streams,
            module.hc_fn,
            module.hc_base,
            module.hc_scale,
            **kwargs,
        ) is not None

        monkeypatch.setenv("VMLX_GLM5_FUSED_MHC_VERIFY", "0")
        assert glm5_mhc_decode(
            streams,
            module.hc_fn,
            module.hc_base,
            module.hc_scale,
            **kwargs,
        ) is None

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
    def test_hc_place_decode_fusion_matches_stock(self, dtype, glm5):
        mx.random.seed(113)
        post = mx.sigmoid(mx.random.normal((1, 1, 4))).astype(mx.float32) * 2
        comb = mx.softmax(mx.random.normal((1, 1, 4, 4)), axis=-1).astype(
            mx.float32
        )
        out = (mx.random.normal((1, 1, 64)) * 0.1).astype(dtype)
        residual = (mx.random.normal((1, 1, 4, 64)) * 0.1).astype(dtype)
        reference = glm5.hc_place(post, comb, out, residual)
        candidate = glm5.hc_place(
            post, comb, out, residual, fused_decode=True
        )
        from vmlx_engine.metal.glm5_hc_place_decode import glm5_hc_place_status

        assert glm5_hc_place_status()["observed_calls"] == 1
        mx.eval(reference, candidate)
        max_abs = float(
            mx.max(
                mx.abs(
                    candidate.astype(mx.float32)
                    - reference.astype(mx.float32)
                )
            ).item()
        )
        max_ref = max(
            float(mx.max(mx.abs(reference.astype(mx.float32))).item()), 1e-9
        )
        # MLX's 4x4 matmul and the direct four-term kernel use different
        # accumulation order. Production-shape probes bound BF16 to one final
        # storage step; F16/F32 remain below 0.1% relative.
        relative_limit = 6e-3 if dtype == mx.bfloat16 else 1e-3
        assert max_abs / max_ref <= relative_limit

    @pytest.mark.parametrize("rows", [2, 3, 4])
    def test_hc_place_decode_fusion_matches_mtp_verify_slabs(
        self, rows, monkeypatch, glm5
    ):
        monkeypatch.setenv("VMLX_GLM5_FUSED_HC_PLACE_VERIFY", "1")
        post = mx.sigmoid(mx.random.normal((1, rows, 4))).astype(mx.float32) * 2
        comb = mx.softmax(mx.random.normal((1, rows, 4, 4)), axis=-1).astype(
            mx.float32
        )
        out = (mx.random.normal((1, rows, 64)) * 0.1).astype(mx.bfloat16)
        residual = (mx.random.normal((1, rows, 4, 64)) * 0.1).astype(
            mx.bfloat16
        )
        reference = glm5.hc_place(post, comb, out, residual)
        candidate = glm5.hc_place(
            post, comb, out, residual, fused_decode=True
        )
        mx.eval(reference, candidate)
        assert mx.allclose(candidate, reference, rtol=6e-3, atol=2e-3)

    def test_hc_place_verify_slab_defaults_to_fused_with_opt_out(
        self, monkeypatch, glm5
    ):
        from vmlx_engine.metal.glm5_hc_place_decode import (
            glm5_hc_place_decode,
        )

        monkeypatch.delenv("VMLX_GLM5_FUSED_HC_PLACE_VERIFY", raising=False)
        monkeypatch.delenv(
            "VMLINUX_GLM5_FUSED_HC_PLACE_VERIFY", raising=False
        )
        post = mx.ones((1, 4, 4), dtype=mx.float32)
        comb = mx.ones((1, 4, 4, 4), dtype=mx.float32)
        out = mx.zeros((1, 4, 64), dtype=mx.bfloat16)
        residual = mx.zeros((1, 4, 4, 64), dtype=mx.bfloat16)
        assert glm5_hc_place_decode(
            post, comb, out, residual, enabled=True
        ) is not None

        monkeypatch.setenv("VMLX_GLM5_FUSED_HC_PLACE_VERIFY", "0")
        assert glm5_hc_place_decode(
            post, comb, out, residual, enabled=True
        ) is None

    def test_hc_place_decode_fusion_refuses_long_prefill(self, glm5):
        post = mx.ones((1, 5, 4), dtype=mx.float32)
        comb = mx.ones((1, 5, 4, 4), dtype=mx.float32)
        out = mx.zeros((1, 5, 64), dtype=mx.bfloat16)
        residual = mx.zeros((1, 5, 4, 64), dtype=mx.bfloat16)
        reference = glm5.hc_place(post, comb, out, residual)
        candidate = glm5.hc_place(
            post, comb, out, residual, fused_decode=True
        )
        mx.eval(reference, candidate)
        assert mx.array_equal(candidate, reference)

    def test_kda_quantized_qkv_group_is_exact_and_releases_sources(self, glm5):
        args = glm5.ModelArgs.from_dict(TINY_CFG)
        attn = glm5.KDAAttention(args)
        attn.q_proj = attn.q_proj.to_quantized(group_size=32, bits=4)
        attn.k_proj = attn.k_proj.to_quantized(group_size=32, bits=4)
        attn.v_proj = attn.v_proj.to_quantized(group_size=32, bits=4)
        x = (mx.arange(128, dtype=mx.float32) / 127.0).reshape(1, 2, 64)
        reference = (attn.q_proj(x), attn.k_proj(x), attn.v_proj(x))

        assert attn.prepare_runtime() is True
        candidate = attn._project_qkv(x)
        mx.eval(*reference, *candidate)

        assert attn.q_proj is attn.k_proj is attn.v_proj is None
        for expected, actual in zip(reference, candidate):
            assert float(mx.abs(expected - actual).max()) == 0.0

    def test_dense_quantized_gate_up_group_is_exact_and_releases_sources(self, glm5):
        args = glm5.ModelArgs.from_dict(TINY_CFG)
        mlp = glm5.DenseMLP(args, 96)
        mlp.gate_proj = mlp.gate_proj.to_quantized(group_size=32, bits=4)
        mlp.up_proj = mlp.up_proj.to_quantized(group_size=32, bits=4)
        x = (mx.arange(128, dtype=mx.float32) / 127.0).reshape(1, 2, 64)
        reference = (mlp.gate_proj(x), mlp.up_proj(x))

        assert mlp.prepare_runtime() is True
        candidate = mlp.gate_up_group(x)
        mx.eval(*reference, *candidate)

        assert mlp.gate_proj is mlp.up_proj is None
        for expected, actual in zip(reference, candidate):
            assert float(mx.abs(expected - actual).max()) == 0.0

    def test_kda_fused_gated_norm_matches_reference_at_mtp_width(self, glm5):
        args = glm5.ModelArgs.from_dict(TINY_CFG)
        attn = glm5.KDAAttention(args)
        attn._fused_gated_norm = True
        x = (mx.arange(4 * 64, dtype=mx.float32) / 127.0).reshape(1, 4, 64)

        fused = attn(x)
        attn._fused_gated_norm = False
        reference = attn(x)
        mx.eval(fused, reference)

        assert fused.shape == reference.shape
        assert float(mx.abs(fused - reference).max()) < 2e-2

    def test_sanitize_drops_visual_and_mtp_keeps_indexer(self, tiny_model):
        weights = {
            "visual.blocks.0.attn.proj.weight": mx.zeros((1,)),
            "model.layers.5.mlp.gate.weight": mx.zeros((1,)),  # MTP layer (== num_hidden_layers)
            "model.layers.3.self_attn.indexer.wk.weight": mx.zeros((1,)),
            "model.layers.0.self_attn.q_proj.weight": mx.zeros((1,)),
            "lm_head.weight": mx.zeros((1,)),
        }
        kept = tiny_model.sanitize(weights)
        assert set(kept) == {
            "model.layers.3.self_attn.indexer.wk.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "lm_head.weight",
        }


class TestSparseDSA:
    """The sparse path's two oracles: full-budget sparse ≡ dense, and
    causality (a query's logits are invariant to future-token changes)."""

    @pytest.fixture(autouse=True)
    def _materialized_sparse_oracle(self, monkeypatch):
        """Keep these gathered-K/V oracles on their owning legacy route.

        Compact-v2 dense/sparse prefill and decode equivalence is covered by
        ``test_absorbed_mla_matches_materialized_prefill_and_decode``.  This
        class intentionally pins the independent materialized implementation.
        """

        monkeypatch.delenv("VMLINUX_GLM5_MLA_ABSORB", raising=False)
        monkeypatch.setenv("VMLX_GLM5_MLA_ABSORB", "0")

    def _model(self, glm5, index_topk):
        cfg = json.loads(json.dumps(TINY_CFG))
        cfg["text_config"]["index_topk"] = index_topk
        cfg["text_config"]["max_position_embeddings"] = 4096
        model = glm5.Model(glm5.ModelArgs.from_dict(cfg))
        mx.eval(model.parameters())
        return model

    def _toks(self, n, seed=11):
        random.seed(seed)
        return [random.randint(1, 500) for _ in range(n)]

    def test_full_budget_sparse_matches_dense(self, glm5):
        """With the selection budget covering every pool + tail, the sparse
        path must reproduce dense causal attention (same math, mask-built)."""
        toks = self._toks(100)
        dense = self._model(glm5, index_topk=4096)   # dense bypass active
        out_d = dense(mx.array([toks]))
        sparse = self._model(glm5, index_topk=4096)
        sparse.update(dense.parameters())            # identical weights
        # Force the sparse machinery while keeping a full selection budget.
        for layer in sparse.model.layers:
            if not layer.is_linear:
                layer.self_attn.index_topk = 16      # switch beyond 16 tokens
                layer.self_attn.indexer.topk = 4096  # budget selects ALL pools
        out_s = sparse(mx.array([toks]))
        mx.eval(out_d, out_s)
        d = float(mx.abs(out_d[0, -1] - out_s[0, -1]).max())
        assert d < 1e-4, f"full-budget sparse diverged from dense: {d}"

    def test_full_budget_sparse_decode_matches_dense_decode(self, glm5):
        toks = self._toks(80)
        dense = self._model(glm5, index_topk=4096)
        c1 = dense.make_cache()
        dense(mx.array([toks]), cache=c1)
        out_d = dense(mx.array([[7]]), cache=c1)

        sparse = self._model(glm5, index_topk=4096)
        sparse.update(dense.parameters())
        for layer in sparse.model.layers:
            if not layer.is_linear:
                layer.self_attn.index_topk = 16
                layer.self_attn.indexer.topk = 4096
        c2 = sparse.make_cache()
        sparse(mx.array([toks]), cache=c2)
        assert c2[3].cache[3].shape[1] == len(toks) // 4
        out_s = sparse(mx.array([[7]]), cache=c2)
        mx.eval(out_d, out_s)
        d = float(mx.abs(out_d[0, -1] - out_s[0, -1]).max())
        assert d < 1e-4, f"full-budget sparse decode diverged: {d}"

    def test_sparse_selection_is_causal(self, glm5):
        """With a REAL (small) selection budget active, logits at position p
        must not change when a token AFTER p changes."""
        model = self._model(glm5, index_topk=16)     # sparse beyond 16 tokens
        base = self._toks(60)
        alt = list(base)
        alt[-1] = (alt[-1] + 7) % 500 + 1
        out_a = model(mx.array([base]))
        out_b = model(mx.array([alt]))
        mx.eval(out_a, out_b)
        # position -2 saw identical prefixes in both runs
        d = float(mx.abs(out_a[0, -2] - out_b[0, -2]).max())
        assert d == 0.0, f"future token leaked into past logits: {d}"

    def test_sparse_prefill_matches_sparse_incremental(self, glm5):
        """One-shot sparse prefill vs prefill+decode: the SELECTED index set
        for the final query must be identical (indexer packed-state
        accumulation across calls), and logits must agree within the
        random-init mHC amplification floor.

        The logits are NOT expected bit-exact: decode uses the gathered-K/V
        SDPA while prefill uses the masked full-width SDPA — same math,
        different fp32 summation order — and random-init mHC amplifies that
        reordering noise ~x40 at the output (see 01-CAMPAIGN-RECORD: gates on
        raw end-to-end deltas measure the noise floor, not correctness).
        """
        model = self._model(glm5, index_topk=16)
        toks = self._toks(50)
        out_full = model(mx.array([toks]))
        cache = model.make_cache()
        model(mx.array([toks[:-1]]), cache=cache)
        out_inc = model(mx.array([toks[-1:]]), cache=cache)
        mx.eval(out_full, out_inc)

        # Exact selection-set equality for the last query on an MLA layer.
        attn = model.model.layers[3].self_attn
        packed = cache[3].cache[2]
        T = packed.shape[1]
        q_pos = mx.array([T - 1])
        x_last = mx.zeros((1, 1, 64))  # positions only drive visibility/tail
        # Rebuild both selections from the same packed history at the same
        # position: one-shot vs incremental share `packed` by construction,
        # so equality here proves accumulation produced the same history.
        idx_a, val_a = attn.indexer.topk_indices(x_last, mx.zeros((1, 1, 32)), packed, q_pos)
        pool_keys = cache[3].cache[3]
        idx_b, val_b = attn.indexer.topk_indices(
            x_last,
            mx.zeros((1, 1, 32)),
            packed,
            q_pos,
            pool_keys=pool_keys,
        )
        sel_a = sorted(int(i) for i, v in zip(idx_a[0, 0].tolist(), val_a[0, 0].tolist()) if v)
        sel_b = sorted(int(i) for i, v in zip(idx_b[0, 0].tolist(), val_b[0, 0].tolist()) if v)
        assert sel_a == sel_b and len(sel_a) > 0

        d = float(mx.abs(out_full[0, -1] - out_inc[0, -1]).max())
        assert d < 2e-2, f"sparse incremental diverged beyond noise floor: {d}"


class TestNativeMTP:
    def _active_model(self, glm5):
        from vmlx_engine.text_mtp import set_draft_head_loading

        set_draft_head_loading(True)
        try:
            model = glm5.Model(glm5.ModelArgs.from_dict(TINY_CFG))
            mx.eval(model.parameters())
            return model
        finally:
            set_draft_head_loading(False)

    def test_layer45_head_attaches_and_runs(self, glm5):
        model = self._active_model(glm5)
        assert hasattr(model, "mtp")
        ids = mx.array([[1, 2, 3, 4]])
        logits, hidden = model(
            ids, cache=model.make_cache(), return_hidden=True
        )
        mtp_cache = model.make_mtp_cache()
        draft_logits, draft_hidden = model.mtp_forward(
            hidden[:, -1:, :], ids[:, -1:], mtp_cache, return_hidden=True
        )
        mx.eval(logits, hidden, draft_logits, draft_hidden)
        assert logits.shape == (1, 4, 512)
        assert hidden.shape == (1, 4, 64)
        assert draft_logits.shape == (1, 1, 512)
        assert draft_hidden.shape == (1, 1, 64)
        assert len(mtp_cache) == 1
        assert isinstance(mtp_cache[0], glm5.Glm5MLACache)

    def test_text_prompt_priming_folds_glm_history_once(
        self, glm5, monkeypatch
    ):
        from mlx_lm.generate import BatchGenerator

        from vmlx_engine.text_mtp import (
            install,
            draft_head_loading_enabled,
            set_draft_head_loading,
        )

        assert install() is True
        monkeypatch.setenv("VMLX_GLM5_MTP_PROMPT_PRIMING", "1")
        monkeypatch.setenv("VMLX_GLM5_ALIGNED_MTP_HEAD_CACHE", "1")
        monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "3")
        previous = draft_head_loading_enabled()
        generator = None
        try:
            set_draft_head_loading(True)
            # Construct before the request policy is enabled. The request plan,
            # not a stale model-construction snapshot of the environment, owns
            # whether prompt capture runs.
            monkeypatch.delenv("VMLX_GLM5_MTP_PROMPT_PRIMING", raising=False)
            model = glm5.Model(glm5.ModelArgs.from_dict(TINY_CFG))
            model.eval()
            mx.eval(model.parameters())
            monkeypatch.setenv("VMLX_GLM5_MTP_PROMPT_PRIMING", "1")
            generator = BatchGenerator(
                model,
                max_tokens=8,
                completion_batch_size=1,
                prefill_batch_size=1,
            )
            assert type(generator._prompt_batch.model).__name__ == "Model"
            prompt = [3, 5, 7, 11, 13, 17, 19]
            generator.insert([prompt])
            state = None
            for _ in range(4):
                generator.next()
                state = getattr(
                    generator._generation_batch, "_vmlx_text_mtp_lane", None
                )
                if state is not None:
                    break
            assert state is not None
            assert state.stats.prompt_prime_source == "cold_prompt"
            assert state.stats.prompt_primed_pairs == len(prompt)
            assert state.stats.mtp_head_cache_policy == "glm_aligned"
            # The proxy is scoped to PromptProcessingBatch.prompt(); the
            # generation batch retains the real model and its native MTP API.
            assert generator._generation_batch.model is model
        finally:
            if generator is not None:
                generator.close()
            set_draft_head_loading(previous)

    def test_vectorized_verify_generation_matches_ar_at_all_depths(
        self, glm5, monkeypatch
    ):
        from mlx_lm.generate import BatchGenerator

        from vmlx_engine.text_mtp import (
            install,
            draft_head_loading_enabled,
            set_draft_head_loading,
        )

        assert install() is True
        monkeypatch.setenv("VMLX_GLM5_VECTOR_KDA_VERIFY", "1")
        monkeypatch.setenv("VMLX_NATIVE_MTP_ADAPTIVE_DEPTH", "0")
        previous = draft_head_loading_enabled()

        def generate(*, attach_mtp: bool, depth: int | None) -> list[int]:
            if depth is None:
                monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH", raising=False)
            else:
                monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", str(depth))
            mx.random.seed(7713)
            set_draft_head_loading(attach_mtp)
            model = glm5.Model(glm5.ModelArgs.from_dict(TINY_CFG))
            model.eval()
            mx.eval(model.parameters())
            generator = BatchGenerator(model, max_tokens=32)
            generator.insert([[3, 5, 7, 11, 13, 17, 19]])
            tokens = []
            while len(tokens) < 32:
                _prompt, responses = generator.next()
                for response in responses:
                    tokens.append(int(response.token))
                    if response.finish_reason is not None:
                        generator.close()
                        return tokens
            generator.close()
            return tokens

        try:
            baseline = generate(attach_mtp=False, depth=None)
            assert len(baseline) == 32
            for aligned in ("0", "1"):
                monkeypatch.setenv(
                    "VMLX_GLM5_ALIGNED_MTP_HEAD_CACHE", aligned
                )
                for depth in (1, 2, 3):
                    assert generate(attach_mtp=True, depth=depth) == baseline
        finally:
            set_draft_head_loading(previous)

    def test_mtp_batch_generator_carries_exact_n_minus_one_snapshot(
        self, glm5
    ):
        """Native MTP uses mlx-lm BatchGenerator, not SingleBatchGenerator.

        The final prompt token advances path-dependent KDA state, so the
        reusable cache must be cloned before that token enters GenerationBatch
        and must survive until the terminal generation response.
        """
        from mlx_lm.generate import BatchGenerator

        from vmlx_engine.text_mtp import install

        assert install() is True
        model = self._active_model(glm5)
        generator = BatchGenerator(model, max_tokens=1)
        generator.insert([[10, 11, 12, 13]])

        prompt_responses, generation_responses = generator.next()
        assert prompt_responses and generation_responses == []
        prompt_responses, generation_responses = generator.next()
        assert prompt_responses[0].end_of_prompt is True
        assert generation_responses == []

        prompt_responses, generation_responses = generator.next()
        assert prompt_responses == []
        assert len(generation_responses) == 1
        response = generation_responses[0]
        assert response.finish_reason == "length"
        snapshot = response.prompt_cache_snapshot
        assert snapshot is not None
        assert isinstance(snapshot[0], glm5.Glm5KDACache)
        assert isinstance(snapshot[3], glm5.Glm5MLACache)
        assert snapshot[3].offset == 3
        assert response.prompt_cache[3].offset >= 4
        assert snapshot[0].cache[glm5.KDA_STATE] is not response.prompt_cache[0].cache[
            glm5.KDA_STATE
        ]
        control = model.make_cache()
        model(mx.array([[10, 11, 12]]), cache=control)
        mx.eval(
            *[
                value
                for layer in control
                for value in layer.state
                if value is not None
            ]
        )
        for expected_layer, snapshot_layer in zip(control, snapshot):
            for expected, actual in zip(expected_layer.state, snapshot_layer.state):
                if expected is None:
                    assert actual is None
                else:
                    assert bool(mx.array_equal(expected, actual).item())
        assert not hasattr(generator._generation_batch, "_vmlx_prompt_cache_snapshots")
        generator.close()

    def test_mtp_batch_generator_hands_boundary_to_callback_without_retaining_it(
        self, glm5
    ):
        """Production persists GLM's N-1 state before allocating the final step.

        The callback may inspect/persist the extracted typed state, but the
        BatchGenerator must not keep a second Metal owner or attach a terminal
        snapshot after the callback returns.
        """
        from mlx_lm.generate import BatchGenerator

        from vmlx_engine.text_mtp import install

        assert install() is True
        model = self._active_model(glm5)
        generator = BatchGenerator(model, max_tokens=1)
        boundaries = []

        def persist(uid, cache):
            boundaries.append(
                {
                    "uid": uid,
                    "classes": [type(layer).__name__ for layer in cache],
                    "mla_offset": cache[3].offset,
                }
            )
            return True

        generator._vmlx_prompt_boundary_callback = persist
        generator.insert([[10, 11, 12, 13]])

        prompt_responses, generation_responses = generator.next()
        assert prompt_responses and generation_responses == []
        prompt_responses, generation_responses = generator.next()
        assert prompt_responses[0].end_of_prompt is True
        assert generation_responses == []
        assert boundaries == [
            {
                "uid": 0,
                "classes": [
                    "Glm5KDACache",
                    "Glm5KDACache",
                    "Glm5KDACache",
                    "Glm5MLACache",
                    "Glm5KDACache",
                ],
                "mla_offset": 3,
            }
        ]
        assert not hasattr(generator._generation_batch, "_vmlx_prompt_cache_snapshots")

        prompt_responses, generation_responses = generator.next()
        assert prompt_responses == []
        assert len(generation_responses) == 1
        assert generation_responses[0].finish_reason == "length"
        assert not hasattr(generation_responses[0], "prompt_cache_snapshot")
        generator.close()

    def test_scheduler_persists_glm_boundary_and_flushes_before_final_token(self):
        """The scheduler binds the full prompt key to the exact N-1 payload."""
        from types import SimpleNamespace

        from vmlx_engine.scheduler import Scheduler

        calls = []

        class FakeDisk:
            def store(self, tokens, cache, *, cache_type):
                calls.append(("store", list(tokens), cache, cache_type))
                return True

            def flush_pending_writes(self, tokens, *, cache_extra_keys=None):
                calls.append(("flush", list(tokens), cache_extra_keys))
                return True

        request = SimpleNamespace(
            prompt_token_ids=[10, 11, 12, 13],
            _bypass_prefix_cache=False,
            _cache_extra_keys=None,
            _segment_boundaries=[(2, "user")],
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.uid_to_request_id = {7: "request-7"}
        scheduler.running = {"request-7": request}
        scheduler.disk_cache = FakeDisk()
        prompt_cache = [object(), object()]

        assert scheduler._store_glm_prompt_boundary(7, prompt_cache) is True
        assert calls == [
            ("store", [10, 11, 12, 13], prompt_cache, "user"),
            ("flush", [10, 11, 12, 13], None),
        ]
        assert request._glm_prompt_boundary_disk_store is True
        assert request._glm_prompt_boundary_cache_layers == 2

    def test_active_sanitize_remaps_appended_layer_to_mtp(self, glm5):
        model = self._active_model(glm5)
        weights = {
            "model.layers.5.enorm.weight": mx.ones((64,)),
            "model.layers.5.shared_head.norm.weight": mx.ones((64,)),
            "model.layers.4.self_attn.q_proj.weight": mx.ones((1,)),
            "visual.blocks.0.weight": mx.ones((1,)),
        }
        kept = model.sanitize(weights)
        assert set(kept) == {
            "mtp.enorm.weight",
            "mtp.shared_head.norm.weight",
            "model.layers.4.self_attn.q_proj.weight",
        }

    @pytest.mark.parametrize("vectorized_verify", [False, True])
    def test_depth3_partial_rejection_restores_accepted_prefix(
        self, glm5, monkeypatch, vectorized_verify
    ):
        from vmlx_engine.text_mtp.lane import rewind_backbone

        monkeypatch.setenv(
            "VMLX_GLM5_VECTOR_KDA_VERIFY",
            "1" if vectorized_verify else "0",
        )
        model = self._active_model(glm5)
        prefix = mx.array([[1, 2, 3, 4, 5, 6]])
        control = model.make_cache()
        sequential = model.make_cache()
        verify = model.make_cache()
        model(prefix, cache=control)
        model(prefix, cache=sequential)
        model(prefix, cache=verify)

        # A D2 verify advances [confirmed, accepted draft, rejected draft].
        # Rolling one token back must retain the first two positions.
        model(mx.array([[7]]), cache=control)
        model(mx.array([[8]]), cache=control)
        sequential_logits = mx.concatenate(
            [
                model(mx.array([[token]]), cache=sequential)
                for token in (7, 8, 9)
            ],
            axis=1,
        )
        verify_logits = model(
            mx.array([[7, 8, 9]]),
            cache=verify,
            return_hidden=True,
            n_confirmed=1,
        )[0]
        mx.eval(sequential_logits, verify_logits)
        assert float(mx.abs(sequential_logits - verify_logits).max()) < 3e-2
        assert rewind_backbone(verify, 1) is True
        assert control[3].offset == verify[3].offset == 8

        for expected_cache, actual_cache in zip(control, verify):
            for expected, actual in zip(expected_cache.cache, actual_cache.cache):
                if expected is None or actual is None:
                    assert expected is actual
                    continue
                mx.eval(expected, actual)
                delta = float(mx.abs(expected - actual).max())
                assert delta < 3e-2, delta

    def test_depth_policy_accepts_glm_partial_rollback_cache(self, glm5, monkeypatch):
        from types import SimpleNamespace

        from vmlx_engine.text_mtp.lane_policy import resolved_depth

        model = self._active_model(glm5)
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "3")
        assert resolved_depth(SimpleNamespace(prompt_cache=model.make_cache())) == 3

    def test_bundle_inspection_marks_appended_layer_runtime_ready_when_explicit(
        self, tmp_path, monkeypatch
    ):
        from vmlx_engine.native_mtp import inspect_native_mtp_bundle

        monkeypatch.delenv("VMLINUX_NATIVE_MTP", raising=False)
        monkeypatch.delenv("VMLX_NATIVE_MTP", raising=False)
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "3")
        (tmp_path / "config.json").write_text(json.dumps(TINY_CFG))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.layers.5.enorm.weight": "model-1.safetensors",
                        "model.layers.5.self_attn.q_a_proj.weight": "model-1.safetensors",
                    }
                }
            )
        )
        status = inspect_native_mtp_bundle(tmp_path)
        assert status["artifact_available"] is True
        assert status["runtime_supported"] is True
        assert status["runtime_available"] is True
        assert status["runtime_default_mode"] == "off"
        assert status["index_mtp_layer_count"] == 1
        assert status["status"] == "native_runtime_ready"


class TestRegistryDetection:
    def test_family_row(self, tmp_path):
        from vmlx_engine.model_config_registry import get_model_config_registry

        (tmp_path / "config.json").write_text(json.dumps(TINY_CFG))
        registry = get_model_config_registry()
        registry.clear_cache()
        conf = registry.lookup(str(tmp_path))
        assert conf.family_name == "glm5_next"
        assert conf.cache_type == "hybrid"
        assert conf.cache_subtype == "glm5_next_native_v2"
        assert conf.tool_parser == "glm47"
        assert conf.reasoning_parser == "deepseek_r1"
        assert conf.think_in_template is True
        assert conf.supported_reasoning_efforts == ["low", "high", "max"]

    def test_text_route_override(self, tmp_path):
        from vmlx_engine.api.utils import is_mllm_model

        cfg = json.loads(json.dumps(TINY_CFG))
        cfg["vision_config"] = {"model_type": "glm5_next"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        assert is_mllm_model(str(tmp_path)) is False
        assert is_mllm_model(str(tmp_path), force_mllm=True) is False

    def test_indexed_visual_bundle_routes_mllm(self, tmp_path):
        from vmlx_engine.api.utils import is_mllm_model

        cfg = json.loads(json.dumps(TINY_CFG))
        cfg["vision_config"] = {"model_type": "glm5_next_vision"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {"weight_map": {"visual.patch_embed.proj.weight": "model-1.safetensors"}}
            )
        )
        assert is_mllm_model(str(tmp_path)) is True


class TestVisionRuntime:
    def test_registers_under_mlx_vlm_namespace(self):
        assert register_glm5_next_vlm_runtime() is True
        import mlx_vlm.models.glm5_next as glm_vlm

        assert glm_vlm.Model is not None
        assert glm_vlm.VisionModel is not None
        assert glm_vlm.ImageProcessor is not None

    def test_processor_from_pretrained_forwards_use_fast_once(
        self, tmp_path, monkeypatch
    ):
        from types import SimpleNamespace

        from vmlx_engine.models.glm5_next import processing

        calls = []
        tokenizer = SimpleNamespace(
            convert_tokens_to_ids=lambda token: {
                "<|image|>": 100,
                "<|video|>": 101,
            }[token]
        )

        def fake_tokenizer_from_pretrained(model_path, **kwargs):
            calls.append((model_path, kwargs))
            return tokenizer

        def fake_processor_init(
            self, image_processor=None, tokenizer=None, **kwargs
        ):
            self.image_processor = image_processor
            self.tokenizer = tokenizer

        monkeypatch.setattr(
            processing.AutoTokenizer,
            "from_pretrained",
            fake_tokenizer_from_pretrained,
        )
        monkeypatch.setattr(
            processing.GlmOcrProcessor,
            "__init__",
            fake_processor_init,
        )
        monkeypatch.setattr(processing, "load_chat_template", lambda *args: None)

        processor = processing.Glm5NextProcessor.from_pretrained(
            tmp_path,
            use_fast=False,
        )

        assert processor.tokenizer is tokenizer
        assert calls == [(tmp_path, {"use_fast": False})]

    def test_jang_quantized_paths_cover_text_vision_and_native_mtp(self):
        from vmlx_engine.utils.jang_loader import _vlm_quant_module_path_candidates

        assert "model.layers.3.self_attn.q_a_proj" in _vlm_quant_module_path_candidates(
            "language_model.model.layers.3.self_attn.q_a_proj", "glm5_next"
        )
        assert "visual.blocks.0.attn.qkv" in _vlm_quant_module_path_candidates(
            "vision_tower.blocks.0.attn.qkv", "glm5_next"
        )
        assert "model.layers.45.self_attn.q_a_proj" in _vlm_quant_module_path_candidates(
            "language_model.mtp.self_attn.q_a_proj", "glm5_next"
        )

    def test_image_processor_emits_exact_patch_contract(self):
        import numpy as np

        from vmlx_engine.models.glm5_next.processing import Glm5NextImageProcessor

        processor = Glm5NextImageProcessor(
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
            min_image_tokens=1,
            max_image_tokens=16,
        )
        result = processor(np.zeros((28, 28, 3), dtype=np.uint8))
        assert result["pixel_values"].shape == (4, 3 * 2 * 14 * 14)
        assert result["image_grid_thw"].tolist() == [[1, 2, 2]]

    def test_tiny_vision_features_replace_only_image_placeholders(self):
        from vmlx_engine.models.glm5_next.config import ModelConfig, VisionConfig
        from vmlx_engine.models.glm5_next.glm5_next import ModelArgs
        from vmlx_engine.models.glm5_next.vlm import Model

        text = ModelArgs.from_dict(TINY_CFG)
        vision = VisionConfig(
            depth=1,
            hidden_size=64,
            intermediate_size=96,
            num_heads=4,
            patch_size=2,
            image_size=4,
            out_hidden_size=64,
            projection_intermediate_size=128,
            spatial_merge_size=1,
            temporal_patch_size=2,
        )
        config = ModelConfig(
            text_config=text,
            vision_config=vision,
            image_token_id=500,
            video_token_id=501,
        )
        model = Model(config)
        mx.eval(model.parameters())
        input_ids = mx.array([[500, 500, 500, 500, 7]])
        pixel_values = mx.zeros((4, 3 * 2 * 2 * 2))
        result = model.get_input_embeddings(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=mx.array([[1, 2, 2]]),
        )
        assert result.inputs_embeds.shape == (1, 5, 64)

        with pytest.raises(ValueError, match="placeholders disagree"):
            model.get_input_embeddings(
                mx.array([[500, 7]]),
                pixel_values=pixel_values,
                image_grid_thw=mx.array([[1, 2, 2]]),
            )

    def test_sanitize_keeps_visual_and_native_mtp_weights(self, monkeypatch):
        from vmlx_engine.models.glm5_next.config import ModelConfig, VisionConfig
        from vmlx_engine.models.glm5_next.glm5_next import ModelArgs
        from vmlx_engine.models.glm5_next.vlm import Model
        from vmlx_engine import text_mtp

        monkeypatch.setattr(text_mtp, "draft_head_loading_enabled", lambda: True)
        text = ModelArgs.from_dict(TINY_CFG)
        config = ModelConfig(text_config=text, vision_config=VisionConfig(depth=0))
        model = Model(config)
        weights = model.sanitize(
            {
                "visual.patch_embed.proj.bias": mx.zeros((1024,)),
                "model.layers.5.enorm.weight": mx.zeros((64,)),
                "model.embed_tokens.weight": mx.zeros((512, 64)),
            }
        )
        assert "vision_tower.patch_embed.proj.bias" in weights
        assert "language_model.mtp.enorm.weight" in weights
        assert "language_model.model.embed_tokens.weight" in weights


class TestPublicLoaderMtpHandoff:
    @staticmethod
    def _prepare_loader(monkeypatch, tmp_path, *, mtp_status, model):
        import mlx_lm
        import numpy as np
        from safetensors.numpy import save_file

        from vmlx_engine import mlx_memory, native_mtp
        from vmlx_engine.models.glm5_next import register as glm_register
        from vmlx_engine.utils import jang_loader, nanbeige_runtime, tokenizer

        cfg = json.loads(json.dumps(TINY_CFG))
        cfg["jang_config"] = {
            "format": "jang_v2",
            "family": "glm5_next",
            "mtp": {"mtp_mode": "preserved_enabled", "num_layers": 1},
        }
        cfg["quantization"] = {
            "group_size": 64,
            "bits": 4,
            "model.layers.5.mlp.switch_mlp.gate_proj": {
                "group_size": 64,
                "bits": 2,
            },
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        save_file(
            {"weight": np.array([1], dtype=np.int8)},
            str(tmp_path / "model.safetensors"),
        )

        events = []
        load_kwargs = {}
        monkeypatch.setattr(tokenizer, "_register_mimo_v2_runtime_for_mlx_lm", lambda: False)
        monkeypatch.setattr(tokenizer, "_needs_tokenizer_fallback", lambda _path: False)
        monkeypatch.setattr(tokenizer, "_inject_chat_template_if_missing", lambda *_a, **_k: None)
        monkeypatch.setattr(jang_loader, "is_jang_model", lambda _path: False)
        monkeypatch.setattr(nanbeige_runtime, "ensure_nanbeige_runtime_registered", lambda _path: None)
        monkeypatch.setattr(nanbeige_runtime, "validate_nanbeige_loop_cache_contract", lambda *_a: None)
        monkeypatch.setattr(mlx_memory, "maybe_harmonize_quant_metadata_dtypes", lambda *_a, **_k: None)
        monkeypatch.setattr(
            tokenizer,
            "_warm_glm5_next_first_forward",
            lambda _model: events.append("warm"),
        )
        monkeypatch.setattr(glm_register, "register_glm5_next_runtime", lambda: events.append("register") or True)
        monkeypatch.setattr(glm_register, "glm5_next_runtime_available", lambda: True)
        monkeypatch.setattr(
            native_mtp,
            "maybe_apply_native_mtp",
            lambda *_a, **_k: events.append("activate") or dict(mtp_status),
        )
        def fake_load(*_args, **kwargs):
            events.append("load")
            load_kwargs.update(kwargs)
            return model, object()

        monkeypatch.setattr(mlx_lm, "load", fake_load)
        return tokenizer, events, load_kwargs

    def test_glm_mtp_activation_precedes_generic_model_construction(
        self, monkeypatch, tmp_path
    ):
        class _MtpModel:
            mtp = object()

            def mtp_forward(self):
                pass

            def make_mtp_cache(self):
                pass

        tokenizer, events, load_kwargs = self._prepare_loader(
            monkeypatch,
            tmp_path,
            mtp_status={"runtime_active": True, "status": "native_runtime_ready"},
            model=_MtpModel(),
        )

        tokenizer.load_model_with_fallback(str(tmp_path), skip_turboquant=True)

        assert events == ["register", "activate", "load", "warm"]
        assert load_kwargs["model_config"]["quantization"][
            "mtp.mlp.switch_mlp.gate_proj"
        ] == {"group_size": 64, "bits": 2}

    def test_glm_mtp_bundle_fails_if_generic_load_drops_attached_head(
        self, monkeypatch, tmp_path
    ):
        tokenizer, events, _load_kwargs = self._prepare_loader(
            monkeypatch,
            tmp_path,
            mtp_status={"runtime_active": True, "status": "native_runtime_ready"},
            model=object(),
        )

        with pytest.raises(RuntimeError, match="no attached draft head"):
            tokenizer.load_model_with_fallback(str(tmp_path), skip_turboquant=True)

        assert events == ["register", "activate", "load"]

    def test_glm_ar_bundle_does_not_require_an_mtp_head(
        self, monkeypatch, tmp_path
    ):
        model = object()
        tokenizer, events, _load_kwargs = self._prepare_loader(
            monkeypatch,
            tmp_path,
            mtp_status={"runtime_active": False, "status": "not_configured"},
            model=model,
        )

        loaded, _ = tokenizer.load_model_with_fallback(
            str(tmp_path), skip_turboquant=True
        )

        assert loaded is model
        assert events == ["register", "activate", "load", "warm"]

    def test_glm_startup_warmup_uses_disposable_native_cache(self):
        from vmlx_engine.utils.tokenizer import _warm_glm5_next_first_forward

        events = []
        cache = [object()]

        class _Model:
            def make_cache(self):
                events.append("cache")
                return cache

            def __call__(self, inputs, *, cache):
                events.append(("forward", inputs.shape, cache))
                return mx.array([[[1.0]]])

        _warm_glm5_next_first_forward(_Model())

        assert events == ["cache", ("forward", (1, 1), cache)]

    def test_glm_startup_warmup_prepares_acceleration_before_forward(self):
        from vmlx_engine.utils.tokenizer import _warm_glm5_next_first_forward

        events = []

        class _Model:
            def prepare_acceleration(self):
                events.append("prepare")
                return {
                    "base_launches_removed_per_forward": 113,
                    "mtp_launches_removed_per_forward": 1,
                }

            def make_cache(self):
                events.append("cache")
                return []

            def __call__(self, inputs, *, cache):
                events.append(("forward", inputs.shape, cache))
                return mx.array([[[1.0]]])

        _warm_glm5_next_first_forward(_Model())

        assert events == ["prepare", "cache", ("forward", (1, 1), [])]


class TestGlmHealthTruth:
    def test_native_cache_health_reports_fail_closed_effective_state(self):
        from types import SimpleNamespace

        from vmlx_engine.server import _native_cache_status

        scheduler = SimpleNamespace(
            _model_type_for_runtime="glm5_next",
            _glm5_next_cache_unsupported=True,
            _prefix_cache_requested=True,
            _prompt_disk_cache_requested=True,
            _block_disk_cache_requested=True,
            config=SimpleNamespace(
                enable_prefix_cache=True,
                enable_disk_cache=True,
                enable_block_disk_cache=True,
            ),
            block_aware_cache=object(),
            paged_cache_manager=SimpleNamespace(_disk_store=object()),
            disk_cache=object(),
        )

        status = _native_cache_status(scheduler, family="glm5_next")

        assert status["schema"] == "glm5_next_native_v1"
        assert status["schema_implemented"] is False
        assert status["prefix_configured"] is True
        assert status["prompt_disk_l2_configured"] is True
        assert status["block_disk_l2_configured"] is True
        assert status["prefix"] is False
        assert status["paged"] is False
        assert status["prompt_disk_l2"] is False
        assert status["block_disk_l2"] is False
        assert status["cache_store_policy"]["every_request_recomputes_full_prefix"] is True

    def test_native_cache_health_reports_typed_exact_boundary_state(self):
        from types import SimpleNamespace

        from vmlx_engine.server import _native_cache_status

        scheduler = SimpleNamespace(
            _model_type_for_runtime="glm5_next",
            _uses_glm5_next_cache=True,
            _glm5_next_cache_unsupported=False,
            _prefix_cache_requested=True,
            _prompt_disk_cache_requested=True,
            _block_disk_cache_requested=True,
            config=SimpleNamespace(
                enable_prefix_cache=True,
                enable_disk_cache=True,
                enable_block_disk_cache=False,
            ),
            memory_aware_cache=object(),
            disk_cache=object(),
            block_aware_cache=None,
            paged_cache_manager=None,
        )

        status = _native_cache_status(scheduler, family="glm5_next")

        assert status["schema"] == "glm5_next_native_v1"
        assert status["schema_implemented"] is True
        assert status["prefix"] is True
        assert status["paged"] is False
        assert status["prompt_disk_l2"] is True
        assert status["block_disk_l2"] is False
        assert status["cache_store_policy"]["prompt_boundary"] == "exact_n_minus_one"

    def test_native_cache_health_reports_absorbed_runtime_schema(
        self, tiny_model, glm5
    ):
        from types import SimpleNamespace

        from vmlx_engine.server import _native_cache_status
        from vmlx_engine.utils.cache_types import describe_runtime_cache_layout

        absorbed_cache = [
            glm5.Glm5KDACache()
            if layer.is_linear
            else glm5.Glm5MLACache(
                tiny_model.args.index_kpool,
                absorbed=True,
            )
            for layer in tiny_model.model.layers
        ]
        runtime_layout = describe_runtime_cache_layout(
            absorbed_cache,
            model=tiny_model,
        )
        scheduler = SimpleNamespace(
            _model_type_for_runtime="glm5_next",
            _uses_glm5_next_cache=True,
            _glm5_next_cache_unsupported=False,
            _prefix_cache_requested=True,
            _prompt_disk_cache_requested=True,
            _block_disk_cache_requested=False,
            _runtime_cache_layout=runtime_layout,
            config=SimpleNamespace(
                enable_prefix_cache=True,
                enable_disk_cache=True,
                enable_block_disk_cache=False,
            ),
            memory_aware_cache=object(),
            disk_cache=object(),
            block_aware_cache=None,
            paged_cache_manager=None,
        )

        status = _native_cache_status(scheduler, family="glm5_next")

        assert status["schema"] == "glm5_next_native_v2"
        assert status["cache_subtype"] == "glm5_next_native_v2"
        assert "mla_latent" in status["components"]
        assert "mla_kv" not in status["components"]
        assert ["glm5_next_native_v2", "mla_absorbed"] in status[
            "runtime_cache_slot_schema_tags"
        ]

    def test_weight_index_counts_appended_glm_mtp_layer(self, tmp_path):
        from vmlx_engine.server import _bundle_weight_index_status

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "glm5_next",
                    "text_config": {
                        "model_type": "glm5_next_text",
                        "num_hidden_layers": 45,
                    },
                }
            )
        )
        keys = {
            "model.layers.44.self_attn.q_proj.weight": "a.safetensors",
            "model.layers.45.enorm.weight": "b.safetensors",
            "model.layers.45.eh_proj.weight": "b.safetensors",
            "model.layers.45.self_attn.q_a_proj.weight": "b.safetensors",
            "model.layers.45.shared_head.norm.weight": "b.safetensors",
        }
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": keys})
        )

        status = _bundle_weight_index_status(str(tmp_path))

        assert status is not None
        assert status["mtp_tensor_count"] == 4


class TestDsaBf16State:
    def test_default_stays_fp32_and_env_narrows_state(self, monkeypatch, glm5):
        args = glm5.ModelArgs.from_dict(TINY_CFG)

        monkeypatch.delenv("VMLX_GLM5_DSA_BF16", raising=False)
        indexer = glm5.Glm5NextIndexer(args)
        x = mx.random.normal((1, 8, args.hidden_size)).astype(mx.bfloat16)
        packed = indexer.packed_states(x)
        assert packed.dtype == mx.float32
        pools = indexer.compress_pool_keys(
            packed[:, : (8 // args.index_kpool) * args.index_kpool, :]
        )
        assert pools.dtype == mx.float32

        monkeypatch.setenv("VMLX_GLM5_DSA_BF16", "1")
        narrow = glm5.Glm5NextIndexer(args)
        narrow.update(indexer.parameters())
        packed16 = narrow.packed_states(x)
        assert packed16.dtype == mx.bfloat16
        pools16 = narrow.compress_pool_keys(
            packed16[:, : (8 // args.index_kpool) * args.index_kpool, :]
        )
        assert pools16.dtype == mx.bfloat16
        # Same math, narrowed storage: values agree within bf16 rounding.
        mx.eval(pools, pools16)
        assert mx.allclose(
            pools16.astype(mx.float32), pools, rtol=2e-2, atol=2e-2
        )
