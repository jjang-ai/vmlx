#!/usr/bin/env python3
"""No-heavy MiMo V2 MLLM inputs_embeds interface proof.

This proves the runtime bridge that media-expanded MiMo prompts need before a
live media E2E can be release-cleared:

* the locally registered mlx-vlm MiMo module forwards provided inputs_embeds
  through the JANG text runtime wrapper;
* image/video/audio placeholder tokens are replaced by supplied modal
  embeddings before the language model forward.

It intentionally uses a fake tiny text runtime and supplied modal embeddings,
so it does not load the 79G/105G MiMo bundles.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = Path(
    "build/current-mimo-v2-mllm-inputs-embeds-interface-proof-20260609.json"
)
MODULE_NAME = "mlx_vlm.models.mimo_v2"


def _array_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def _register_fake_runtime(tmp_dir: Path):
    from vmlx_engine.models import mllm

    model_dir = tmp_dir / "MiMo-V2.5-JANG_2L"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"mimo_v2","vision_config":{},"audio_config":{}}\n',
        encoding="utf-8",
    )

    class FakeTextConfig:
        model_type = "mimo_v2"

        @classmethod
        def from_dict(cls, _params):
            return cls()

    class FakeTextModel:
        def __init__(self, _config):
            self.calls = []
            self.layers = []
            self.model = SimpleNamespace(embed_tokens=self._embed_tokens)

        def _embed_tokens(self, input_ids):
            ids = mx.array(input_ids)
            if ids.ndim == 1:
                ids = ids[None, :]
            return mx.zeros(tuple(ids.shape) + (16,))

        def __call__(self, input_ids=None, *, inputs_embeds=None, cache=None, mask=None, **kwargs):
            self.calls.append(
                {
                    "input_ids_shape": _array_shape(input_ids),
                    "inputs_embeds_shape": _array_shape(inputs_embeds),
                    "cache_present": cache is not None,
                    "mask_present": mask is not None,
                    "kwargs": sorted(kwargs.keys()),
                }
            )
            return SimpleNamespace(logits=inputs_embeds)

        def make_cache(self):
            return []

        def sanitize(self, weights):
            return weights

        def load_weights(self, weights, strict=True):
            self.loaded_weights = dict(weights)
            return None

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "jang_tools.mimo_v2.mlx_model":
            return SimpleNamespace(Model=FakeTextModel, ModelArgs=FakeTextConfig)
        return real_import_module(name, package)

    original_import_module = mllm.importlib.import_module
    sys.modules.pop(MODULE_NAME, None)
    mllm.importlib.import_module = fake_import_module
    try:
        mllm._register_local_mlx_vlm_runtime_if_needed(model_dir)
    finally:
        mllm.importlib.import_module = original_import_module
    return sys.modules[MODULE_NAME]


def _nonzero_vector(value: Any) -> bool:
    return any(abs(float(item)) > 0.0 for item in value)


def _run_checks() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="mimo-inputs-embeds-") as tmp:
        module = _register_fake_runtime(Path(tmp))

        model = module.Model(
            module.ModelConfig.from_dict(
                {
                    "model_type": "mimo_v2",
                    "multimodal_status": "media_runtime_enabled",
                    "vision_config": {
                        "hidden_size": 4,
                        "out_hidden_size": 16,
                        "patch_size": 2,
                        "temporal_patch_size": 1,
                        "in_channels": 1,
                        "spatial_merge_size": 1,
                        "depth": 1,
                        "num_heads": 2,
                        "num_key_value_heads": 1,
                        "intermediate_size": 8,
                        "fullatt_block_indexes": [0],
                    },
                    "audio_config": {
                        "audio_channels": 1,
                        "group_size": 1,
                        "input_local_attn_heads": 1,
                        "input_local_dim": 4,
                        "input_local_head_dim": 4,
                        "input_local_intermediate_size": 8,
                        "input_local_layers": 1,
                        "out_hidden_size": 16,
                        "speech_vocab_size": 8,
                    },
                    "processor_config": {
                        "image_token_id": 151655,
                        "video_token_id": 151656,
                        "audio_token_id": 151669,
                    },
                }
            )
        )

        supplied = mx.ones((1, 3, 16))
        direct_output = model(mx.array([[1, 2, 3]]), inputs_embeds=supplied, cache=[])
        direct_calls = model.language_model.inner.calls
        checks["registered_module_present"] = MODULE_NAME in sys.modules
        checks["direct_inputs_embeds_forwarded"] = (
            _array_shape(direct_output.logits) == [1, 3, 16]
            and bool(direct_calls)
            and direct_calls[-1]["inputs_embeds_shape"] == [1, 3, 16]
        )
        details["direct_forward_call"] = direct_calls[-1] if direct_calls else None

        input_ids = mx.array([[11, 151655, 22, 151656, 33, 151669, 44]])
        modal_output = model(
            input_ids,
            image_embeds=mx.ones((1, 16)),
            video_embeds=mx.ones((1, 16)) * 2,
            audio_embeds=mx.ones((1, 16)) * 3,
        )
        logits = modal_output.logits.tolist()[0]
        checks["image_embedding_spliced"] = _nonzero_vector(logits[1]) and logits[1][0] == 1.0
        checks["video_embedding_spliced"] = _nonzero_vector(logits[3]) and logits[3][0] == 2.0
        checks["audio_embedding_spliced"] = _nonzero_vector(logits[5]) and logits[5][0] == 3.0
        checks["text_tokens_preserved_as_embeddings"] = (
            logits[0] == [0.0] * 16
            and logits[2] == [0.0] * 16
            and logits[4] == [0.0] * 16
            and logits[6] == [0.0] * 16
        )
        details["modal_logits_shape"] = _array_shape(modal_output.logits)
        details["modal_positions"] = {
            "image": logits[1][:4],
            "video": logits[3][:4],
            "audio": logits[5][:4],
            "text_head": logits[0][:4],
        }

        sys.modules.pop(MODULE_NAME, None)

    status = "pass" if all(checks.values()) else "open"
    return {
        "status": status,
        "generated_at": int(time.time()),
        "checks": checks,
        "details": details,
        "release_boundary": (
            "This clears only the MiMo MLLM inputs_embeds interface proof. It does "
            "not clear live image/video/audio E2E, artifact exactness, decode speed, "
            "JANG_2L memory-safe live rows, or installed-app UI parity."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    result = _run_checks()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.out)
    print(f"status={result['status']}")
    for name, ok in result["checks"].items():
        print(f"{'PASS' if ok else 'OPEN'} {name}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
