# SPDX-License-Identifier: Apache-2.0
"""Unsupported encoder reranker architectures are refused at load, not scored.

Issue #398: BGE / XLM-RoBERTa cross-encoder checkpoints (a `roberta.` prefix
and a `classifier` head) reach the strict mlx-embeddings weight load, which
rejects every tensor and surfaces as an HTTP 500 listing the checkpoint
weights. The load must refuse architectures whose classification head
mlx-embeddings cannot provide, and the endpoint must answer a clear 4xx.

The bundles below are tiny local fixtures: a config.json plus a safetensors
file holding two checkpoint-shaped keys. No download and no real weights.
"""

import json
import platform
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)

# (architecture, model_type) pairs whose classification head mlx-embeddings
# does not provide. ModernBertForSequenceClassification is the one encoder
# architecture it scores (see tests/test_reranker_encoder.py).
UNSUPPORTED = [
    ("XLMRobertaForSequenceClassification", "xlm-roberta"),
    ("BertForSequenceClassification", "bert"),
    ("RobertaForSequenceClassification", "roberta"),
]

TINY_ENCODER_CONFIG = {
    "hidden_size": 4,
    "num_hidden_layers": 1,
    "intermediate_size": 8,
    "num_attention_heads": 2,
    "max_position_embeddings": 16,
    "vocab_size": 32,
}


def _write_bundle(tmp_path, config, projector=False, write_config=True):
    """A local model bundle: config.json + two checkpoint-shaped weights."""
    import numpy as np
    from safetensors.numpy import save_file

    if write_config:
        (tmp_path / "config.json").write_text(json.dumps(config))
    save_file(
        {
            "roberta.embeddings.LayerNorm.weight": np.zeros(4, dtype=np.float32),
            "classifier.out_proj.weight": np.zeros((1, 4), dtype=np.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    if projector:
        save_file(
            {
                "linear1.weight": np.zeros((4, 4), dtype=np.float32),
                "linear2.weight": np.zeros((4, 4), dtype=np.float32),
            },
            str(tmp_path / "projector.safetensors"),
        )
    return str(tmp_path)


@pytest.fixture
def client():
    """Create a TestClient with auth disabled and no reranker loaded."""
    from unittest.mock import patch

    from vmlx_engine.server import app

    with patch("vmlx_engine.server._api_key", None), \
         patch("vmlx_engine.server._reranker", None):
        from starlette.testclient import TestClient
        yield TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("architecture, model_type", UNSUPPORTED)
def test_load_refuses_unsupported_classification(tmp_path, architecture, model_type):
    """The load refuses the architecture instead of listing its weights."""
    from vmlx_engine.reranker import Reranker, UnsupportedRerankerError

    config = dict(TINY_ENCODER_CONFIG, model_type=model_type,
                  architectures=[architecture])
    path = _write_bundle(tmp_path, config)

    with pytest.raises(UnsupportedRerankerError) as excinfo:
        Reranker(path).rerank("query", ["document"])

    message = str(excinfo.value)
    assert architecture in message
    assert path in message
    assert "parameters not in model" not in message


@pytest.mark.parametrize("architecture, model_type", UNSUPPORTED)
def test_endpoint_refuses_unsupported_classification_with_400(
    client, tmp_path, architecture, model_type
):
    """/v1/rerank answers 400 naming the architecture, with no weight list."""
    config = dict(TINY_ENCODER_CONFIG, model_type=model_type,
                  architectures=[architecture])
    path = _write_bundle(tmp_path, config)

    resp = client.post("/v1/rerank", json={
        "query": "What is the capital of France?",
        "documents": ["Bananas are yellow.", "Paris is the capital of France."],
        "model": path,
    })

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    # Positive twin: the architecture and model are named.
    assert architecture in detail
    assert path in detail
    # Absence: the checkpoint weight list never reaches the client.
    assert "parameters not in model" not in detail
    assert "classifier.out_proj.weight" not in detail


def test_modernbert_classification_passes_the_gate(tmp_path):
    """ModernBERT classification is not refused; it reaches the weight load."""
    from vmlx_engine.reranker import Reranker

    config = dict(
        TINY_ENCODER_CONFIG,
        model_type="modernbert",
        architectures=["ModernBertForSequenceClassification"],
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=32,
        vocab_size=16,
        id2label={"0": "LABEL_0"},
        classifier_pooling="mean",
    )
    path = _write_bundle(tmp_path, config)

    # The dummy keys are not ModernBERT weights, so the strict load reports
    # them — proving the architecture gate let this checkpoint through.
    with pytest.raises(ValueError, match="parameters not in model"):
        Reranker(path).rerank("query", ["document"])


def test_causal_architecture_passes_the_gate(tmp_path):
    """A CausalLM config is not refused by the encoder architecture gate."""
    from vmlx_engine.reranker import Reranker

    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 16,
        "vocab_size": 16,
    }
    path = _write_bundle(tmp_path, config)

    with pytest.raises(ValueError, match="parameters not in model"):
        Reranker(path).rerank("query", ["document"])


def test_late_interaction_detection_precedes_the_gate(tmp_path):
    """A projector bundle takes the late-interaction path, not the refusal."""
    from vmlx_engine.reranker import Reranker

    config = dict(TINY_ENCODER_CONFIG, model_type="xlm-roberta",
                  architectures=["XLMRobertaForSequenceClassification"])
    path = _write_bundle(tmp_path, config, projector=True)

    # mlx_lm cannot load an xlm-roberta config; whatever it raises, it must
    # not be the encoder architecture refusal.
    with pytest.raises(Exception) as excinfo:
        Reranker(path).rerank("query", ["document"])
    assert "Unsupported reranker architecture" not in str(excinfo.value)


def test_missing_config_falls_through_to_the_loader(tmp_path):
    """Without a config.json the existing loader error is preserved."""
    from vmlx_engine.reranker import Reranker

    path = _write_bundle(tmp_path, {}, write_config=False)

    with pytest.raises(FileNotFoundError):
        Reranker(path).rerank("query", ["document"])