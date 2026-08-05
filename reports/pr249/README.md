# PR #249 evidence

## Historical evidence (pre-1.6.23 base)

`dsv4-native-15e9420d.json` and `dsv4-optimized-15e9420d.json` were sealed
against source revision `15e9420db804cfa81167acaceae13476d856a7b5`, whose
merge base predates vMLX 1.6.23/1.6.24 (answer-budget reservation, projected
byte output caps, and the Metal live-buffer ceiling). They document the
original candidate measurement and are retained for provenance only.

They are **not** acceptance evidence for the v1.6.24 integration. The
integration is accepted or rejected solely on new reports sealed against the
`agent/pr249-v1.6.24-integration` revisions, produced by
`bench/dsv4_performance_suite.py` per
`docs/benchmarks/dsv4-performance-acceptance.md`.
