# Nemotron Omni media sessions

The native Omni media dispatcher accepts image, audio-input and video-frame
requests through Chat Completions, Responses and Messages. Actual modalities
are derived from the bundle components and reported in `/health` under
`omni_multimodal`. Audio output and native media tool combinations require
separate support; do not infer them from text-only tool support.

With Block Disk Cache enabled, Stage-1 media requests store their native
attention and recurrent state under the configured block-cache directory.
These snapshots share the aggregate SSD limit with ordinary text-cache blocks
and other model namespaces. Changing the cache directory selects a different
pool. Disabling Block Disk Cache disables native snapshot reuse as well.

The request's terminal response waits for its snapshot publication. A failed
write returns an error rather than claiming a durable cache entry. After the
boundary, reusable KV/SSM payloads leave the native session; a later request
restores an exact matching conversation snapshot or rebuilds the supplied
history. Different conversations have separate snapshots. This does not imply
arbitrary suffix reuse or token-level truncation of recurrent state.

Snapshots preserve the model's actual tensor representation. A JANG weight bit
width is not a KV-cache bit width. Full-precision recurrent state may be needed
alongside smaller attention tensors; the disk writer does not apply an extra
lossy codec. `/health` reports the active native session cache policy and its
most recently used snapshot, while the shared cache statistics account for the
aggregate pool.

The existing Cache TTL control applies to RAM caching, not these SSD snapshots.
SSD entries are retained until capacity eviction or an explicit cache clear.
Old snapshots in the former `omni-session` directory are not imported or reused
by the managed pool.
