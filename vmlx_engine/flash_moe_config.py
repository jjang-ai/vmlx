# SPDX-License-Identifier: Apache-2.0
"""Flash MoE configuration for SSD expert streaming."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FlashMoEConfig:
    """Configuration for Flash MoE SSD expert streaming.

    Enables massive MoE models (35B-397B) to run on Macs with limited RAM
    by streaming expert weights from SSD on-demand, keeping only active
    experts in a slot-bank cache.

    Attributes:
        enabled: Whether Flash MoE is active. Default False (opt-in).
        slot_bank_size: Maximum number of expert weight sets cached in RAM.
            Higher = more cache hits but more RAM. Typical: 64-128.
        prefetch: Prefetching strategy for experts.
            "none" — no prefetching, load on demand only.
            "temporal" — prefetch experts that were recently used (LRU warmup).
        cache_io_split: Number of parallel I/O operations for expert loading.
            Higher = faster loading but more I/O pressure. Default 4.
        expert_index_path: Optional path to a pre-built expert index JSON.
            If None, the index is built on first load from safetensors headers.
    """

    enabled: bool = False
    slot_bank_size: int = 64
    prefetch: str = "none"
    cache_io_split: int = 4
    expert_index_path: Optional[str] = None

    def __post_init__(self):
        if self.slot_bank_size < 1:
            raise ValueError(f"slot_bank_size must be >= 1, got {self.slot_bank_size}")
        if self.prefetch not in ("none", "temporal"):
            raise ValueError(
                f"prefetch must be 'none' or 'temporal', got '{self.prefetch}'"
            )
        if self.cache_io_split < 1:
            raise ValueError(
                f"cache_io_split must be >= 1, got {self.cache_io_split}"
            )

    @classmethod
    def from_dict(cls, d: dict) -> "FlashMoEConfig":
        """Create from a dictionary (e.g., from config file or API request)."""
        return cls(
            enabled=d.get("enabled", False),
            slot_bank_size=d.get("slot_bank_size", d.get("slot_bank", 64)),
            prefetch=d.get("prefetch", "none"),
            cache_io_split=d.get("cache_io_split", 4),
            expert_index_path=d.get("expert_index_path"),
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "slot_bank_size": self.slot_bank_size,
            "prefetch": self.prefetch,
            "cache_io_split": self.cache_io_split,
            "expert_index_path": self.expert_index_path,
        }
