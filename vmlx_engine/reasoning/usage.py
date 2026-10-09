# SPDX-License-Identifier: Apache-2.0
"""Count generated token IDs inside explicit reasoning delimiters.

This never re-tokenizes displayed text or charges prompt-injected markers.
Parsers without token-delimiter contracts return None rather than a fabricated
zero. Call once on completed output, outside the decode loop.
"""

def count_reasoning_tokens(token_ids, parser, tokenizer):
    if token_ids is None or parser is None:
        return None
    # Accounting is optional metadata: a parser without the token-delimiter contract (duck-typed parsers, test
    # doubles) or a tag lookup that fails must omit the count, never abort the stream that is reporting it.
    tag_fn = getattr(parser, "reasoning_tag_token_seqs", None)
    if not callable(tag_fn):
        return None
    try:
        tags = tag_fn(tokenizer) or {}
    except Exception:
        return None
    starts, ends = tags.get('start') or [], tags.get('end') or []
    if not starts or not ends:
        return None
    starts = sorted((tuple(s) for s in starts if s), key=len, reverse=True)
    ends = sorted((tuple(s) for s in ends if s), key=len, reverse=True)
    max_end_length = max(map(len, ends), default=0)
    ids = list(token_ids)
    inside = bool(getattr(parser, '_think_in_prompt', False))
    eos = set(getattr(tokenizer, "eos_token_ids", ()) or ())
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos.add(tokenizer.eos_token_id)
    count = i = 0
    while i < len(ids):
        match = next((s for s in starts if ids[i] == s[0] and tuple(ids[i:i+len(s)]) == s), None)
        if match is not None:
            inside = True
            i += len(match)
            continue
        match = next((s for s in ends if ids[i] == s[0] and tuple(ids[i:i+len(s)]) == s), None)
        if match is not None:
            inside = False
            i += len(match)
            continue
        if inside and ids[i] not in eos:
            # A truncated closing delimiter is control syntax, not reasoning.
            if len(ids) - i < max_end_length:
                tail = tuple(ids[i:])
                if any(len(tail) < len(s) and s[:len(tail)] == tail for s in ends):
                    break
            count += 1
        i += 1
    return count
