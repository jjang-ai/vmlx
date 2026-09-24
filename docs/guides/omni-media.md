# Nemotron Omni media sessions

The native Omni media dispatcher accepts image, audio-input and video-frame
requests through Chat Completions, Responses and Messages. Actual modalities
are derived from bundle components, including indexed tensor headers and
projector shapes, and reported in `/health` under `omni_multimodal`. Audio
output is unsupported. Stage-1 supports function tools; Stage-2 does not share
that tool or SSD-checkpoint contract.

`/v1/capabilities` separates `request_routes.text` from
`request_routes.native_media` when the native route is enabled. Media anywhere
in the supplied conversation selects the native route. Its controls and cache
contract differ from text generation. **Force Off** (`--text-only`) rejects
media, including media in restored conversation history. Health retains the
artifact's modalities separately from the disabled runtime route.

Native video sampling honors request-local `video_fps` and `video_max_frames`,
including the values saved in MLX Studio's server settings. If omitted, the
native defaults are 1 FPS and 4 frames (overridable through
`VMLINUX_OMNI_VIDEO_FPS` and `VMLINUX_OMNI_VIDEO_MAX_FRAMES`). For bundles with trained temporal projection
weights, the runtime validates their shapes and uses native temporal video
encoding. Repeated frames are retained, and the frame ceiling counts sampled
frames before any internal temporal padding. Different sampling policies have
separate frame and conversation-cache identities.

The RADIO path does not enforce pixel, explicit-dimension or video-token
budgets; requests containing those video controls return an error. Sampling
failure also returns an error instead of switching to fixed native defaults.

The native media route currently supports temperature, top-p, the total output
limit, the thinking toggle and the documented temporal video controls. It
rejects structured-output constraints, custom stop strings, seeds, non-neutral
extended sampling controls and explicit image pixel/token budgets. These
constraints must not silently disappear when switching from a text request to
a media request.
Unexpected native dispatch failures return an error through every API adapter;
they never retry the same media request through a text-only fallback.

Stage-1 accepts `tool_choice: "auto"` or `"none"`. Function arguments must be
JSON objects and satisfy their supplied schemas. Schema references must resolve
offline; external schema retrieval is unsupported. Tool-result history must
contain complete, ID-matched batches. Executable calls are delivered only after
the whole generated batch passes validation and the native SSD write fence
finishes. Malformed model output is rejected, not repaired into a guessed call.

Thinking is controlled by `enable_thinking` and the bundle's native template.
Stage-1 also accepts `chat_template_kwargs.reasoning_budget` (a nonnegative
integer soft hint) and `chat_template_kwargs.truncate_history_thinking` (a
boolean). A soft hint is not an enforced token cap. Native media rejects
`max_thinking_tokens` and the equivalent protocol budget fields; use the total
output limit. Text-route budget support remains available in its own capability
record.

With Block Disk Cache enabled, Stage-1 media requests store their native
attention and recurrent state under the configured block-cache directory.
These snapshots share the aggregate SSD limit with ordinary text-cache blocks
and other model namespaces. Changing the cache directory selects a different
pool. Disabling Block Disk Cache disables native snapshot reuse as well.

The request's terminal response waits for its snapshot publication. A failed
write returns an error rather than claiming a durable cache entry. After the
boundary, reusable KV/SSM payloads leave the native session; a later request
restores the longest validated causal-prefix checkpoint and prefills only the
remainder, or rebuilds the supplied history on a miss. Checkpoints include
complete tool-result boundaries. A subsequent tool request can start as soon as
the preceding response ends; no client-side sleep is required. Different
conversations have separate snapshots. This does not imply
arbitrary suffix reuse or token-level truncation of recurrent state.

Stage-1 prompt usage includes both the restored native attention offset and
the newly prefetched tokens, including media placeholders. Chat Completions
reports reused tokens in `prompt_tokens_details.cached_tokens`; Responses
preserves them in `input_tokens_details.cached_tokens`. A cache bypass reports
zero reused tokens. These counts describe native decoder state, not cached
encoder features or a media-file cache.
Decoded uploads and sampled video frames live in a private per-request temporary
directory. They are removed when the native turn returns, fails, or is
cooperatively cancelled. Local input files are never removed. These temporary
inputs are separate from persistent SSD model-state snapshots; a process crash
can still leave temporary files for the operating system to clean up.
Messages reports the uncached portion as `input_tokens` and reused tokens
separately as `cache_read_input_tokens`; add them to recover the full input.

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
