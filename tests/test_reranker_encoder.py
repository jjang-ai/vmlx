# SPDX-License-Identifier: Apache-2.0
"""Encoder reranking consumes classification scores, not embedding vectors."""

import math
import platform
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


@pytest.fixture
def reranker():
    import numpy as np

    from vmlx_engine.reranker import Reranker

    ranker = Reranker("local-test-model")
    ranker._backend = "encoder"
    ranker._tokenizer = lambda *args, **kwargs: {
        "input_ids": np.array([[1, 2, 3]]),
        "attention_mask": np.array([[1, 1, 1]]),
    }
    return ranker


@pytest.mark.parametrize(
    "regression, biases, expected",
    [
        (True, [-2.0], -2.0),
        (False, [-2.0], 1 / (1 + math.exp(2))),
        (False, [0.0, 2.0], 1 / (1 + math.exp(-2))),
    ],
    ids=["regression", "binary-classification", "two-class-classification"],
)
def test_modernbert_classification_scores(reranker, regression, biases, expected):
    import mlx.core as mx
    from mlx_embeddings.models.modernbert import Model, ModelArgs

    model = Model(
        ModelArgs(
            model_type="modernbert",
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            intermediate_size=16,
            num_attention_heads=2,
            max_position_embeddings=32,
            architectures=["ModernBertForSequenceClassification"],
            is_regression=regression,
            id2label={str(i): f"LABEL_{i}" for i in range(len(biases))},
        )
    )
    model.classifier.weight = mx.zeros_like(model.classifier.weight)
    model.classifier.bias = mx.array(biases)
    model.eval()
    reranker._model = model

    results = reranker.rerank("query", ["document"], return_documents=True)

    assert len(results) == 1
    assert results[0].index == 0
    assert results[0].document == "document"
    assert results[0].relevance_score == pytest.approx(expected)


@pytest.mark.parametrize(
    "output_kind", ["logits", "dict-logits", "dict-scores", "tuple", "array"]
)
def test_existing_encoder_score_outputs(reranker, output_kind):
    import mlx.core as mx

    logits = mx.array([[1.0, 3.0]])
    outputs = {
        "logits": SimpleNamespace(logits=logits),
        "dict-logits": {"logits": logits},
        "dict-scores": {"scores": logits},
        "tuple": (logits,),
        "array": logits,
    }
    reranker._model = lambda **kwargs: outputs[output_kind]

    assert reranker._score_encoder("query", ["document"]) == [3.0]


@pytest.mark.parametrize(
    "architecture",
    [
        "ModernBertModel",
        "ModernBertForMaskedLM",
        "BertModel",
        "XLMRobertaForSequenceClassification",
    ],
)
def test_embedding_outputs_are_not_relevance_scores(reranker, architecture):
    import mlx.core as mx
    from mlx_embeddings.models.base import BaseModelOutput

    class EmbeddingModel:
        config = SimpleNamespace(architectures=[architecture])

        def __call__(self, **kwargs):
            return BaseModelOutput(
                pooler_output=mx.array([[0.2, 0.8]]),
                text_embeds=mx.array([[0.2, 0.8]]),
            )

    reranker._model = EmbeddingModel()

    with pytest.raises(ValueError, match=rf"classification scores.*{architecture}"):
        reranker.rerank("query", ["document"])
