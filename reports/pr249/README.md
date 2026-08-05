# PR #249 evidence

## v1.6.24 integration evidence (current, sealing revision `680add2d4`)

Sealed on the target hardware (Apple M4 Max, 128 GB, macOS 26.5.1) against
the immutable 102-shard DeepSeek-V4-Flash-0731-JANG bundle
(manifest `fc68310e3b81…`), one fresh process per profile, identical
`bench/dsv4_performance_suite.py` protocol and controls:

| file | profile |
|---|---|
| `dsv4-native-v1624-680add2d4.json` | N — native baseline, all candidate paths off |
| `dsv4-lmhead-v1624-680add2d4.json` | H — exact lm_head cache only |
| `dsv4-rope-v1624-680add2d4.json` | R — RoPE table sharing only |
| `dsv4-optimized-v1624-680add2d4.json` | HR — shipping profile |
| `dsv4-admission-denied-v1624-680add2d4.json` | D — cache admission forced to decline; proves the fail-closed path stays native |
| `dsv4-acceptance-v1624-680add2d4.json` | gate evaluation (exactness, throughput, tail, prefill, memory, composition, fail-closed) |

Headlines: token-ID SHA-256 identical to native in **every case of every
profile**, including 1,000- and 8,000-token streams; HR median decode
1.20–1.45× native across the matrix with prefill ≥ native; the persistent
FP32 head (2.118 GB on this bundle) is deducted byte-for-byte from the MLX
allocator-cache ceiling and released exactly on unload.

Caveat recorded in the acceptance report: this host exhibits a bimodal
low-power GPU state (~800 MHz) that intermittently strikes long
single-repeat samples in ANY profile, including pure-native runs; per-case
medians with repeats are used for throughput gates, and the anomaly is
documented as pre-existing host behavior, not a property of this change.

## Historical evidence (pre-1.6.23 base)

`dsv4-native-15e9420d.json` and `dsv4-optimized-15e9420d.json` were sealed
against source revision `15e9420db804cfa81167acaceae13476d856a7b5`, whose
merge base predates vMLX 1.6.23/1.6.24 (answer-budget reservation, projected
byte output caps, and the Metal live-buffer ceiling). They document the
original candidate measurement and are retained for provenance only.
