# MiMo V2.5 JANGTQ2 Direct Color A/B - 2026-06-10

Model:

- `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2`
- served as `mimo-v25-jangtq2-direct-color-ab-20260610`
- source `.venv`, `--is-mllm`, continuous batching, paged cache, block-disk L2
- `VMLINUX_DISABLE_MIMO_V2_COMPILED_ROUTER=1` for this classifier run

Runtime proof from server log:

- `BatchedEngine loaded ... (mllm=True)`
- MiMo preserved media runtime auto-enabled: `vision=True audio=True`
- bound preserved media tensors: `visual=364`, `audio_encoder=75`, `speech_embeddings=20`
- media diag on image turns: `engine_is_mllm=true`, `registry_is_mllm=true`, `types={"image_url":1}`
- image turns skipped media prompt cache store as intended

Same-process color A/B outputs:

| input | output |
| --- | --- |
| text only, no image | `Blue.` |
| solid red 128x128 PNG | `White.` |
| solid green 128x128 PNG | `White.` |
| solid blue 128x128 PNG | `White.` |
| solid white 128x128 PNG | `White.` |
| solid black 128x128 PNG | `White.` |

Classification:

- The current source no longer fails at the old `mimo_v2_preserved_text_runtime` text-only gate.
- The route processes image inputs, but simple color semantics are not image-conditioned enough to clear `vl_image`.
- This is not a red-channel-only swap: all image colors collapse to `White.` in this prompt, while the no-image control returns `Blue.`.
- Do not clear this with prompt tuning, regex changes, parser cleanup, JSON repair, or forced output rewrites.
- Next useful work is a first-logit/runtime comparison against the local Torch `modeling_mimo_v2.py` visual path or an artifact/runtime contract proving whether the MLX adapter, JANGTQ artifact, or language-side splicing is the first divergence.
