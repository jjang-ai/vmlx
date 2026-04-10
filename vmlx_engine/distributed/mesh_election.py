# SPDX-License-Identifier: Apache-2.0
"""Coordinator election based on capability scoring.

Unlike Raft/Paxos consensus, this is a simple one-round election:
all nodes broadcast their capability score, highest score wins.
Re-election only triggers when the coordinator disconnects.

Capability score = RAM(40%) + GPU_cores(30%) + bandwidth(20%) + uptime(10%)

This is simpler than exo's seniority-based multi-round election because
vMLX meshes are small (2-10 nodes) and coordinator role is heavy
(embedding + lm_head + API + tokenizer), so the most capable node
should always be coordinator.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ELECTION_TIMEOUT = 5.0  # seconds to wait for all votes


class ElectionCandidate:
    """A vote/candidate in an election round."""
    def __init__(self, node_id: str, hostname: str, capability_score: float, ram_gb: int = 0, gpu_cores: int = 0):
        self.node_id = node_id
        self.hostname = hostname
        self.capability_score = capability_score
        self.ram_gb = ram_gb
        self.gpu_cores = gpu_cores
        self.voted_at = time.time()

    def __repr__(self):
        return f"<Candidate {self.hostname} score={self.capability_score:.1f}>"


class ElectionResult:
    """Result of a coordinator election."""
    def __init__(self, coordinator_id: str, hostname: str, score: float, candidates: List[ElectionCandidate]):
        self.coordinator_id = coordinator_id
        self.hostname = hostname
        self.score = score
        self.candidates = candidates
        self.elected_at = time.time()

    def __repr__(self):
        return f"<ElectionResult coordinator={self.hostname} score={self.score:.1f} candidates={len(self.candidates)}>"


class MeshElection:
    """Manages coordinator election for the mesh.

    Election flow:
    1. Trigger: coordinator lost OR mesh formation
    2. All nodes broadcast VOTE with their capability score
    3. Wait ELECTION_TIMEOUT for all votes
    4. Highest score wins (ties broken by node_id sort)
    5. Winner broadcasts ELECTED
    6. All nodes accept the result
    """

    def __init__(
        self,
        local_node_id: str,
        local_hostname: str,
        local_score: float,
        local_ram_gb: int = 0,
        local_gpu_cores: int = 0,
        on_elected: Optional[Callable[[ElectionResult], None]] = None,
        forced_coordinator_id: Optional[str] = None,
    ):
        self.local_node_id = local_node_id
        self.local_hostname = local_hostname
        self.local_score = local_score
        self.local_ram_gb = local_ram_gb
        self.local_gpu_cores = local_gpu_cores
        self._on_elected = on_elected
        self._forced_coordinator_id = forced_coordinator_id

        self._candidates: Dict[str, ElectionCandidate] = {}
        self._election_in_progress = False
        self._election_event = asyncio.Event()
        self._result: Optional[ElectionResult] = None

    async def start_election(self) -> ElectionResult:
        """Run an election round. Returns the result."""
        if self._forced_coordinator_id:
            # Manual override — skip election
            result = ElectionResult(
                coordinator_id=self._forced_coordinator_id,
                hostname="(forced)",
                score=float("inf"),
                candidates=[],
            )
            self._result = result
            if self._on_elected:
                self._on_elected(result)
            return result

        logger.info("Election started — collecting votes for %ds", ELECTION_TIMEOUT)
        self._election_in_progress = True
        # Keep any pre-registered votes (from discovered peers) — only add ours
        self.receive_vote(
            self.local_node_id, self.local_hostname, self.local_score,
            self.local_ram_gb, self.local_gpu_cores,
        )

        # Wait for additional votes from network (peers that weren't pre-registered)
        await asyncio.sleep(ELECTION_TIMEOUT)

        # Determine winner
        candidates = sorted(
            self._candidates.values(),
            key=lambda c: (c.capability_score, c.node_id),
            reverse=True,
        )

        winner = candidates[0]
        result = ElectionResult(
            coordinator_id=winner.node_id,
            hostname=winner.hostname,
            score=winner.capability_score,
            candidates=candidates,
        )

        self._result = result
        self._election_in_progress = False
        self._election_event.set()

        logger.info(
            "Election complete: %s wins (score=%.1f, %d candidates)",
            winner.hostname, winner.capability_score, len(candidates),
        )
        for i, c in enumerate(candidates):
            logger.info(
                "  #%d %s: score=%.1f (%dGB RAM, %d GPU cores)",
                i + 1, c.hostname, c.capability_score, c.ram_gb, c.gpu_cores,
            )

        if self._on_elected:
            self._on_elected(result)

        return result

    def receive_vote(self, node_id: str, hostname: str, score: float, ram_gb: int = 0, gpu_cores: int = 0):
        """Receive a vote from a peer during an election."""
        self._candidates[node_id] = ElectionCandidate(
            node_id=node_id, hostname=hostname, capability_score=score,
            ram_gb=ram_gb, gpu_cores=gpu_cores,
        )
        logger.debug("Vote received: %s (score=%.1f)", hostname, score)

    @property
    def is_in_progress(self) -> bool:
        return self._election_in_progress

    @property
    def result(self) -> Optional[ElectionResult]:
        return self._result

    async def wait_for_result(self, timeout: float = ELECTION_TIMEOUT + 2) -> Optional[ElectionResult]:
        """Wait for an ongoing election to complete."""
        try:
            await asyncio.wait_for(self._election_event.wait(), timeout=timeout)
            return self._result
        except asyncio.TimeoutError:
            return None
