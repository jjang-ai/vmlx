# SPDX-License-Identifier: Apache-2.0
"""
Tests for Continuous Batching with Hybrid/Mamba Models.

These tests verify that the system correctly identifies Hybrid/Mamba architectures
(those returning MambaCache/ArraysCache from make_cache()) and appropriately
configures their caching strategies, avoiding Memory-Aware cache where incompatible.
"""

from unittest.mock import MagicMock, patch

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


class KVCache:
    pass


class MambaCache:
    pass


class ArraysCache:
    pass


@pytest.fixture
def mock_standard_model():
    model = MagicMock()
    model.make_cache = MagicMock(return_value=[KVCache(), KVCache()])
    return model


@pytest.fixture
def mock_hybrid_model():
    model = MagicMock()
    model.make_cache = MagicMock(return_value=[KVCache(), MambaCache()])
    return model


@pytest.fixture
def mock_pure_mamba_model():
    model = MagicMock()
    model.make_cache = MagicMock(return_value=[MambaCache(), ArraysCache()])
    return model


class TestHybridBatching:

    def test_hybrid_detection(
        self, mock_standard_model, mock_hybrid_model, mock_pure_mamba_model
    ):
        """Test that _is_hybrid_model correctly identifies non-standard caches."""
        from vmlx_engine.scheduler import Scheduler

        # Standard KV-only
        assert Scheduler._is_hybrid_model(mock_standard_model) is False

        # Mixed KV and Mamba
        assert Scheduler._is_hybrid_model(mock_hybrid_model) is True

        # Pure Mamba/SSM
        assert Scheduler._is_hybrid_model(mock_pure_mamba_model) is True

        # Model with no make_cache
        mock_no_cache = MagicMock(spec=[])
        assert Scheduler._is_hybrid_model(mock_no_cache) is False

    @patch("vmlx_engine.scheduler.Scheduler._is_hybrid_model")
    def test_hybrid_forces_legacy_cache(
        self, mock_is_hybrid, mock_hybrid_model
    ):
        """
        Test that a hybrid model bypasses Memory-Aware cache sizing and
        routes to either Legacy Cache or Paged Cache, depending on settings.
        """
        mock_is_hybrid.return_value = True

        from vmlx_engine.scheduler import Scheduler, SchedulerConfig
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        mock_tokenizer = MagicMock(spec=TokenizerWrapper)
        
        # Scenario 1: Memory-Aware caching requested (the default for simple continuous batching)
        config = SchedulerConfig(
            max_num_seqs=4,
            use_memory_aware_cache=True,  # Default
            use_paged_cache=False
        )

        with patch("vmlx_engine.scheduler.logger") as mock_logger:
            # We must trap model properties required inside __init__
            mock_hybrid_model.config = MagicMock()
            
            scheduler = Scheduler(mock_hybrid_model, mock_tokenizer, config)
            
            # Since memory_aware_cache requires KV caching, hybrid model should force it to False
            # and fall back to Legacy KV caching approach (which doesn't dynamically size physical chunks)
            assert scheduler.config.use_memory_aware_cache is False
            
            # The system warns the user
            mock_logger.info.assert_any_call(
                "Non-standard cache model detected (MambaCache/hybrid layers). "
                "Auto-switching to paged cache for correct cache reuse."
            )

    @patch("vmlx_engine.scheduler.Scheduler._is_hybrid_model")
    def test_hybrid_allows_paged_cache(
        self, mock_is_hybrid, mock_hybrid_model
    ):
        """Test that paged cache is permitted with hybrid models."""
        mock_is_hybrid.return_value = True

        from vmlx_engine.scheduler import Scheduler, SchedulerConfig
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        mock_tokenizer = MagicMock(spec=TokenizerWrapper)

        # User explicitly requested Paged Cache
        config = SchedulerConfig(
            max_num_seqs=4,
            use_paged_cache=True
        )

        mock_hybrid_model.config = MagicMock()
        
        with patch("vmlx_engine.scheduler.logger") as mock_logger:
            scheduler = Scheduler(mock_hybrid_model, mock_tokenizer, config)
            
            # It retains paged cache because paged cache implements custom Mamba block mappings
            assert scheduler.config.use_paged_cache is True
            assert scheduler.block_aware_cache is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
