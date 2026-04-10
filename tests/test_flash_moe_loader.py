"""Tests for Flash MoE SSD expert streaming."""
import pytest


class TestFlashMoEConfig:
    """Verify FlashMoEConfig dataclass validation."""

    def test_default_config(self):
        from vmlx_engine.flash_moe_config import FlashMoEConfig
        cfg = FlashMoEConfig()
        assert cfg.enabled is False
        assert cfg.slot_bank_size == 64
        assert cfg.prefetch == "none"
        assert cfg.cache_io_split == 4

    def test_enabled_config(self):
        from vmlx_engine.flash_moe_config import FlashMoEConfig
        cfg = FlashMoEConfig(enabled=True, slot_bank_size=128)
        assert cfg.enabled is True
        assert cfg.slot_bank_size == 128

    def test_invalid_slot_bank(self):
        from vmlx_engine.flash_moe_config import FlashMoEConfig
        with pytest.raises(ValueError, match="slot_bank_size"):
            FlashMoEConfig(slot_bank_size=0)

    def test_invalid_prefetch(self):
        from vmlx_engine.flash_moe_config import FlashMoEConfig
        with pytest.raises(ValueError, match="prefetch"):
            FlashMoEConfig(prefetch="invalid")

    def test_invalid_io_split(self):
        from vmlx_engine.flash_moe_config import FlashMoEConfig
        with pytest.raises(ValueError, match="cache_io_split"):
            FlashMoEConfig(cache_io_split=0)

    def test_from_dict(self):
        from vmlx_engine.flash_moe_config import FlashMoEConfig
        cfg = FlashMoEConfig.from_dict({
            "enabled": True,
            "slot_bank": 128,  # alias for slot_bank_size
            "prefetch": "temporal",
        })
        assert cfg.enabled is True
        assert cfg.slot_bank_size == 128
        assert cfg.prefetch == "temporal"

    def test_to_dict(self):
        from vmlx_engine.flash_moe_config import FlashMoEConfig
        cfg = FlashMoEConfig(enabled=True, slot_bank_size=32)
        d = cfg.to_dict()
        assert d["enabled"] is True
        assert d["slot_bank_size"] == 32


class TestSlotBankCache:
    """Verify LRU cache behavior."""

    def test_basic_put_get(self):
        import mlx.core as mx
        from vmlx_engine.utils.flash_moe_loader import SlotBankCache, ExpertWeightSet
        cache = SlotBankCache(max_slots=4)

        ews = ExpertWeightSet(layer_idx=0, expert_idx=0, tensors={
            "gate_proj": {"weight": mx.zeros((128, 64), dtype=mx.uint32)}
        })
        cache.put(ews)

        result = cache.get(0, 0)
        assert result is not None
        assert result.layer_idx == 0
        assert result.expert_idx == 0

    def test_cache_miss(self):
        from vmlx_engine.utils.flash_moe_loader import SlotBankCache
        cache = SlotBankCache(max_slots=4)
        assert cache.get(0, 0) is None

    def test_lru_eviction(self):
        import mlx.core as mx
        from vmlx_engine.utils.flash_moe_loader import SlotBankCache, ExpertWeightSet
        cache = SlotBankCache(max_slots=2)

        def make_ews(layer, expert):
            return ExpertWeightSet(layer_idx=layer, expert_idx=expert, tensors={
                "up_proj": {"weight": mx.zeros((10,))}
            })

        cache.put(make_ews(0, 0))
        cache.put(make_ews(0, 1))
        # Cache full: [0:0, 0:1]

        # Access 0:0 to make it recently used
        cache.get(0, 0)

        # Add 0:2 — should evict 0:1 (LRU)
        cache.put(make_ews(0, 2))

        assert cache.get(0, 0) is not None  # still present (recently used)
        assert cache.get(0, 1) is None  # evicted
        assert cache.get(0, 2) is not None  # just added

    def test_stats(self):
        import mlx.core as mx
        from vmlx_engine.utils.flash_moe_loader import SlotBankCache, ExpertWeightSet
        cache = SlotBankCache(max_slots=4)

        ews = ExpertWeightSet(layer_idx=0, expert_idx=0, tensors={})
        cache.put(ews)
        cache.get(0, 0)  # hit
        cache.get(0, 1)  # miss

        stats = cache.stats()
        assert stats["slots_used"] == 1
        assert stats["max_slots"] == 4
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_clear(self):
        import mlx.core as mx
        from vmlx_engine.utils.flash_moe_loader import SlotBankCache, ExpertWeightSet
        cache = SlotBankCache(max_slots=4)

        cache.put(ExpertWeightSet(layer_idx=0, expert_idx=0, tensors={}))
        cache.clear()

        assert cache.get(0, 0) is None
        stats = cache.stats()
        assert stats["slots_used"] == 0
        assert stats["hits"] == 0

    def test_update_existing(self):
        import mlx.core as mx
        from vmlx_engine.utils.flash_moe_loader import SlotBankCache, ExpertWeightSet
        cache = SlotBankCache(max_slots=2)

        ews1 = ExpertWeightSet(layer_idx=0, expert_idx=0, tensors={"a": {}})
        ews2 = ExpertWeightSet(layer_idx=0, expert_idx=0, tensors={"b": {}})
        cache.put(ews1)
        cache.put(ews2)

        result = cache.get(0, 0)
        assert "b" in result.tensors  # updated to ews2
        assert cache.stats()["slots_used"] == 1  # no double-count


class TestExpertWeightSet:
    """Verify ExpertWeightSet data structure."""

    def test_total_bytes(self):
        import mlx.core as mx
        from vmlx_engine.utils.flash_moe_loader import ExpertWeightSet
        ews = ExpertWeightSet(
            layer_idx=0, expert_idx=0,
            tensors={
                "gate_proj": {
                    "weight": mx.zeros((128, 64), dtype=mx.float32),
                    "scales": mx.zeros((128, 4), dtype=mx.float16),
                },
            },
        )
        total = ews.total_bytes
        # 128*64*4 + 128*4*2 = 32768 + 1024 = 33792
        assert total == 33792

    def test_empty_tensors(self):
        from vmlx_engine.utils.flash_moe_loader import ExpertWeightSet
        ews = ExpertWeightSet(layer_idx=5, expert_idx=3, tensors={})
        assert ews.total_bytes == 0
        assert ews.layer_idx == 5
        assert ews.expert_idx == 3


class TestFlashMoEExpertLoader:
    """Verify loader with mock expert index."""

    def test_loader_stats(self):
        from vmlx_engine.utils.flash_moe_loader import (
            FlashMoEExpertLoader, SlotBankCache,
        )
        from vmlx_engine.utils.smelt_loader import ExpertIndex
        cache = SlotBankCache(max_slots=4)
        ei = ExpertIndex()  # empty index
        loader = FlashMoEExpertLoader(ei, cache, io_workers=2)
        stats = loader.stats()
        assert stats["slots_used"] == 0
        assert stats["disk_loads"] == 0
        assert stats["io_workers"] == 2

    def test_load_nonexistent_layer(self):
        from vmlx_engine.utils.flash_moe_loader import (
            FlashMoEExpertLoader, SlotBankCache,
        )
        from vmlx_engine.utils.smelt_loader import ExpertIndex
        cache = SlotBankCache(max_slots=4)
        ei = ExpertIndex()
        loader = FlashMoEExpertLoader(ei, cache)

        result = loader.load_expert(0, 0)
        assert result is None  # layer 0 not in index

    def test_load_experts_parallel_empty(self):
        from vmlx_engine.utils.flash_moe_loader import (
            FlashMoEExpertLoader, SlotBankCache,
        )
        from vmlx_engine.utils.smelt_loader import ExpertIndex
        cache = SlotBankCache(max_slots=4)
        ei = ExpertIndex()
        loader = FlashMoEExpertLoader(ei, cache)

        result = loader.load_experts_parallel(0, [0, 1, 2])
        assert result == {}

    def test_shutdown(self):
        from vmlx_engine.utils.flash_moe_loader import (
            FlashMoEExpertLoader, SlotBankCache,
        )
        from vmlx_engine.utils.smelt_loader import ExpertIndex
        cache = SlotBankCache(max_slots=4)
        ei = ExpertIndex()
        loader = FlashMoEExpertLoader(ei, cache)
        loader.shutdown()
        assert cache.stats()["slots_used"] == 0


class TestFlashMoEIntegration:
    """Verify model integration module imports and basic structure."""

    def test_import_flash_moe_block(self):
        from vmlx_engine.models.flash_moe_integration import FlashMoEBlock
        assert FlashMoEBlock is not None

    def test_import_apply_flash_moe(self):
        from vmlx_engine.models.flash_moe_integration import apply_flash_moe
        assert callable(apply_flash_moe)

    def test_import_free_expert_weights(self):
        from vmlx_engine.models.flash_moe_integration import free_expert_weights
        assert callable(free_expert_weights)
