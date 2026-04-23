# SPDX-License-Identifier: Apache-2.0
"""VL helpers — ad-hoc generation paths for one-shot scripts.

Production traffic flows through vMLX's scheduler stack
(``MLLMScheduler`` → ``MLLMBatchGenerator``). This module exists for
ad-hoc scripts that want to drive a loaded VLM directly the way
``jang_tools.kimi_prune.generate_vl`` does — no scheduler, no batching,
just tokenize+vision+chunked-prefill+decode.

Matches the module path prescribed in
``research/KIMI-K2.6-VMLX-INTEGRATION.md`` §1.4.
"""
