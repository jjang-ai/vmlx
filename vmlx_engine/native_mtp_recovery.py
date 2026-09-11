# SPDX-License-Identifier: Apache-2.0
"""Request-owned retry scheduling after a measured MTP performance handoff.

No model/cache objects are retained here. Correctness failures must not create
this record. Timing samples come from completed productive stock decoder steps,
not from hypothetical acceptance rates or the cost of an MTP verify call.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any


INITIAL_COOLDOWN_TOKENS = 128
MAX_COOLDOWN_TOKENS = 4096
MIN_AR_SAMPLES = 8


@dataclass
class NativeMTPRecovery:
    uid: Any
    depth_ceiling: int
    adaptive: bool
    cooldown: int = INITIAL_COOLDOWN_TOKENS
    remaining: int = INITIAL_COOLDOWN_TOKENS
    attempts: int = 0
    failed_probes: int = 0
    standard_tokens: int = 0
    standard_wall_ms: float = 0.0
    # Bounded, recent samples. Skip the first step after every handoff so the
    # pipeline transition is not mistaken for steady stock decoding cost.
    samples: deque = field(default_factory=lambda: deque(maxlen=16))
    skip_sample: bool = True
    completed_mtp: dict = field(default_factory=dict)
    parked_stats: Any = field(default=None, repr=False)

    def observe_standard(self, elapsed_ms: float) -> None:
        self.standard_tokens += 1
        self.remaining = max(0, self.remaining - 1)
        if math.isfinite(elapsed_ms) and elapsed_ms > 0:
            self.standard_wall_ms += elapsed_ms
            if not self.skip_sample:
                self.samples.append(elapsed_ms)
        self.skip_sample = False

    @property
    def ar_ms(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def ready(self) -> bool:
        return self.remaining == 0 and len(self.samples) >= MIN_AR_SAMPLES

    def park(self, *, failed_probe: bool) -> None:
        if failed_probe:
            self.failed_probes += 1
            self.cooldown = min(MAX_COOLDOWN_TOKENS, self.cooldown * 2)
        else:
            self.cooldown = INITIAL_COOLDOWN_TOKENS
        self.remaining = self.cooldown
        self.samples.clear()
        self.skip_sample = True

    def snapshot(self) -> dict:
        return {
            "attempts": self.attempts,
            "failed_probes": self.failed_probes,
            "cooldown_tokens": self.cooldown,
            "remaining_ar_tokens": self.remaining,
            "productive_ar_tokens": self.standard_tokens,
            "productive_ar_wall_ms": self.standard_wall_ms,
            "measured_ar_ms_per_token": self.ar_ms,
            "completed_mtp_phases": dict(self.completed_mtp),
        }
