# SPDX-License-Identifier: Apache-2.0
"""vMLX native multi-token-prediction (MTP) speculative-decoding runtime.

Phase 1 (current): math primitives only — see ``speculative.py``. The math is
NumPy-only so it can be exercised by a correctness oracle without loading any
model weights. Per-architecture backends and scheduler integration arrive in
later phases.

Algorithm reference: Leviathan et al. 2023 (arxiv:2211.17192) +
Chen et al. 2023 (arxiv:2302.01318).
"""

from .speculative import (
    SpeculativeDecision,
    acceptance_probability,
    residual_distribution,
    sample_from_distribution,
    speculative_output_marginal,
    verify_one_token,
)

__all__ = [
    "SpeculativeDecision",
    "acceptance_probability",
    "residual_distribution",
    "sample_from_distribution",
    "speculative_output_marginal",
    "verify_one_token",
]
