"""Sequence state machine for stop / reasoning / tool tag detection.

This module vendors the Aho-Corasick `SequenceStateMachine` from
`mlx_lm.generate` (mlx-lm 0.31.2) so vMLX can use it on the current pinned
version (`mlx-lm>=0.30.2`). The verbatim upstream classes are kept under their
upstream names so a future bump can switch the import without changing call
sites.

vMLX additions on top of the upstream module:

* `advance_from(state, tokens)` — skip-scan a trusted token prefix (one that
  came back from a prompt-cache hit) and return the post-prefix state without
  re-matching the cached tokens. Required by the trie-based prefix cache
  introduced by Agent 1's `PrefixCacheManager` cache_type LRU.

* `make_state_machine(parser_registry, model_key, stop_words, ...)` — factory
  that builds a state machine from vMLX's parser registry entries instead of
  upstream's tokenizer-extension API. Pulls reasoning + tool tag tokens from
  the per-parser config and builds the `normal -> reasoning|tool -> normal`
  transition table that mirrors `mlx_lm.server._make_state_machine`.

* `_FACTORY_CACHE` — LRU keyed by
  `(model_key, reasoning_parser_id, tool_parser_id, tuple(stop_words),
  initial_state)` so the factory does not rebuild the trie on every request.

The state machine itself is purely combinatorial — no MLX, no torch, no
threading state. Aho-Corasick advance is O(1) amortized per token; building
the trie is O(total_pattern_length).
"""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Upstream-vendored core (mlx-lm 0.31.2 generate.py:896-1001)
# ---------------------------------------------------------------------------


def _build_trie(sequences):
    """Build an Aho-Corasick trie from the provided sequences.

    See https://en.wikipedia.org/wiki/Aho-Corasick_algorithm.

    Vendored verbatim from mlx-lm 0.31.2 generate.py:896-931.
    """
    trie: Dict[Any, Any] = {}
    for idx, seq in enumerate(sequences):
        node = trie
        try:
            for tok in seq:
                node = node.setdefault(tok, {})
            node["__match__"] = (tuple(seq), idx)
        except TypeError:
            node = node.setdefault(seq, {})
            node["__match__"] = ((seq,), idx)

    queue = deque()
    for key, child in trie.items():
        if key == "__match__":
            continue
        child["__fail__"] = trie
        queue.append(child)
    while queue:
        parent = queue.popleft()
        for key, child in parent.items():
            if key in ("__fail__", "__match__"):
                continue
            queue.append(child)
            fail = parent["__fail__"]
            while key not in fail and fail is not trie:
                fail = fail["__fail__"]
            child["__fail__"] = fail[key] if key in fail else trie
            if "__match__" not in child and "__match__" in child["__fail__"]:
                child["__match__"] = child["__fail__"]["__match__"]
    return trie


def _step_trie(node, trie, x):
    """One step in the Aho-Corasick trie. Vendored from mlx-lm 0.31.2."""
    while x not in node and node is not trie:
        node = node["__fail__"]
    if x in node:
        node = node[x]
    return node


class SequenceStateMachine:
    """A state machine that uses one Aho-Corasick trie per state to efficiently
    track state across a generated sequence.

    Vendored from mlx-lm 0.31.2 generate.py:943-1001 with two vMLX additions:
    `advance_from` (instance method) and `current_state` (instance method).

    Transitions are provided as `state -> [(sequence, new_state), ...]`.
    `new_state == None` means generation halts when this sequence matches
    (used for EOS / stop sequences). Other state names are jumped to.

    Example::

        sm = SequenceStateMachine(
            transitions={
                "normal": [
                    (think_start_tokens, "reasoning"),
                    (tool_start_tokens, "tool"),
                    (eos, None),
                ],
                "reasoning": [
                    (think_end_tokens, "normal"),
                    (eos, None),
                ],
                "tool": [
                    (tool_end_tokens, "normal"),
                    (eos, None),
                ],
            },
            initial="normal",
        )
    """

    def __init__(self, transitions: Optional[Dict[str, List[Tuple[Sequence[int], Optional[str]]]]] = None, initial: str = "normal"):
        transitions = transitions or {}
        self._initial = initial
        self._states: Dict[str, Tuple[Dict[Any, Any], Tuple[Optional[str], ...]]] = {}
        for src, edges in transitions.items():
            # vMLX guard: empty edges list (a "terminal" state with no
            # outgoing transitions) is legal but `zip(*[])` would crash. Use
            # an empty trie + empty destination tuple — match() will then
            # never find a `__match__` and the state stays put forever.
            if not edges:
                self._states[src] = (_build_trie([]), ())
                continue
            sequences, dst = zip(*edges)
            self._states[src] = (_build_trie(sequences), dst)
        if self._initial not in self._states:
            self._states[self._initial] = (_build_trie([]), ())

    def __deepcopy__(self, memo):
        new = object.__new__(SequenceStateMachine)
        new._initial = self._initial
        new._states = self._states
        return new

    def make_state(self):
        """Return a fresh per-sequence state tuple."""
        return (self._initial, self._states[self._initial][0], self._states)

    @staticmethod
    def match(state, x):
        """Advance the state machine by one token. Vendored verbatim.

        Returns ``((next_state_name, next_node, states), matched_seq_or_None,
        current_state_name_or_None)``. ``current_state_name`` is ``None``
        when a halting sequence (EOS / stop) just matched — the caller should
        finalize the sequence.
        """
        s, n, states = state
        n = _step_trie(n, states[s][0], x)

        seq = None
        match = n.get("__match__")
        if match is not None:
            seq = match[0]
            s = states[s][1][match[1]]
            n = states[s][0] if s is not None else None

        return (s, n, states), seq, s

    # ------------------------------------------------------------------
    # vMLX-specific additions
    # ------------------------------------------------------------------

    def advance_from(self, state, tokens: Iterable[int]):
        """Advance the state machine across a trusted token prefix.

        Used after a prefix-cache hit (Agent 1's `PrefixCacheManager`): the
        cached tokens are known-clean, but the state machine still needs to
        end up in the correct ``normal | reasoning | tool`` state so that
        subsequent tokens are matched against the right transition trie.

        Returns the new state tuple. If the prefix happens to contain a
        halting sequence (EOS or stop), the caller can detect it via the
        returned ``current_state is None`` — but that should not happen for
        a properly-trimmed cache hit (Agent 1's trie strips beyond such
        boundaries before insertion).
        """
        for x in tokens:
            state, _seq, current = SequenceStateMachine.match(state, x)
            if current is None:
                # halting sequence inside a "trusted" prefix — return as-is
                # so the caller can decide whether to bail out
                return state
        return state

    @staticmethod
    def current_state(state) -> Optional[str]:
        """Read the current state name out of a state tuple."""
        return state[0]


# ---------------------------------------------------------------------------
# vMLX factory: build state machines from the parser registry
# ---------------------------------------------------------------------------


# Module-level LRU. Bounded so a long-running server with many models doesn't
# leak. Eviction is FIFO; rebuild cost is small (sub-millisecond for typical
# parser tag sets).
_FACTORY_CACHE: "OrderedDict[Tuple, SequenceStateMachine]" = OrderedDict()
_FACTORY_CACHE_MAX = 100


def _to_token_seq(item) -> Tuple[int, ...]:
    """Normalize a tag entry to a tuple of ints.

    Accepts: a single int (becomes a 1-tuple), a list/tuple of ints, or a
    string (raises — caller should pre-tokenize). Strings raise because
    matching against tokenized output requires the parser to have already
    encoded its tags through the model's tokenizer.
    """
    if isinstance(item, int):
        return (item,)
    if isinstance(item, (list, tuple)):
        return tuple(int(x) for x in item)
    raise TypeError(
        f"state_machine: tag must be int or List[int] (token IDs); got {type(item).__name__}. "
        "Tokenize tag strings with the model tokenizer before passing them in."
    )


def make_state_machine(
    *,
    model_key: str,
    reasoning_parser_id: Optional[str] = None,
    tool_parser_id: Optional[str] = None,
    reasoning_start_tokens: Optional[Iterable[Sequence[int]]] = None,
    reasoning_end_tokens: Optional[Iterable[Sequence[int]]] = None,
    tool_start_tokens: Optional[Iterable[Sequence[int]]] = None,
    tool_end_tokens: Optional[Iterable[Sequence[int]]] = None,
    eos_tokens: Optional[Iterable[Sequence[int]]] = None,
    stop_token_sequences: Optional[Iterable[Sequence[int]]] = None,
    initial_state: str = "normal",
) -> SequenceStateMachine:
    """Build (or fetch from cache) a `SequenceStateMachine` for one model.

    The factory mirrors upstream `mlx_lm.server._make_state_machine` but pulls
    its tag inputs from explicit kwargs (vMLX has multiple parser sources:
    `tool_parsers/`, `reasoning/`, model-config registry). Callers in the
    scheduler are expected to assemble the right token sequences from the
    parser registry before invoking this factory.

    All token-sequence inputs may be ``None`` (treated as empty). Empty
    transition sets are still valid — they produce a state machine that only
    matches stop sequences (or, if those are empty too, never halts).

    Cache key includes the parser ids so models with the same tags but
    different parser registry entries don't collide.
    """
    rs_tokens = tuple(_to_token_seq(s) for s in (reasoning_start_tokens or ()))
    re_tokens = tuple(_to_token_seq(s) for s in (reasoning_end_tokens or ()))
    ts_tokens = tuple(_to_token_seq(s) for s in (tool_start_tokens or ()))
    te_tokens = tuple(_to_token_seq(s) for s in (tool_end_tokens or ()))
    eos_seqs = tuple(_to_token_seq(s) for s in (eos_tokens or ()))
    stop_seqs = tuple(_to_token_seq(s) for s in (stop_token_sequences or ()))

    cache_key = (
        model_key,
        reasoning_parser_id or "",
        tool_parser_id or "",
        rs_tokens,
        re_tokens,
        ts_tokens,
        te_tokens,
        eos_seqs,
        stop_seqs,
        initial_state,
    )

    cached = _FACTORY_CACHE.get(cache_key)
    if cached is not None:
        _FACTORY_CACHE.move_to_end(cache_key)
        return cached

    transitions: Dict[str, List[Tuple[Sequence[int], Optional[str]]]] = {}

    normal_edges: List[Tuple[Sequence[int], Optional[str]]] = []
    for seq in rs_tokens:
        normal_edges.append((seq, "reasoning"))
    for seq in ts_tokens:
        normal_edges.append((seq, "tool"))
    for seq in eos_seqs:
        normal_edges.append((seq, None))
    for seq in stop_seqs:
        normal_edges.append((seq, None))
    transitions["normal"] = normal_edges

    if rs_tokens or re_tokens:
        reasoning_edges: List[Tuple[Sequence[int], Optional[str]]] = []
        for seq in re_tokens:
            reasoning_edges.append((seq, "normal"))
        for seq in eos_seqs:
            reasoning_edges.append((seq, None))
        for seq in stop_seqs:
            reasoning_edges.append((seq, None))
        transitions["reasoning"] = reasoning_edges

    if ts_tokens or te_tokens:
        tool_edges: List[Tuple[Sequence[int], Optional[str]]] = []
        for seq in te_tokens:
            tool_edges.append((seq, "normal"))
        for seq in eos_seqs:
            tool_edges.append((seq, None))
        for seq in stop_seqs:
            tool_edges.append((seq, None))
        transitions["tool"] = tool_edges

    sm = SequenceStateMachine(transitions=transitions, initial=initial_state)

    _FACTORY_CACHE[cache_key] = sm
    _FACTORY_CACHE.move_to_end(cache_key)
    while len(_FACTORY_CACHE) > _FACTORY_CACHE_MAX:
        _FACTORY_CACHE.popitem(last=False)

    return sm


def reset_factory_cache() -> None:
    """Clear the factory LRU. Used by tests and by `engine.shutdown()`."""
    _FACTORY_CACHE.clear()


__all__ = [
    "SequenceStateMachine",
    "make_state_machine",
    "reset_factory_cache",
]
