# SPDX-License-Identifier: Apache-2.0
"""
Reranker engine for /v1/rerank endpoint.

Supports two scoring backends:
1. Encoder models (ModernBERT, XLM-RoBERTa, BGE) — sequence classification
   scores via forward pass through cross-encoder architecture.
2. CausalLM models (Qwen3-Reranker) — yes/no logit scoring via generate.

Usage:
    reranker = Reranker(model_path)
    results = reranker.rerank(query, documents, top_n=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """A single reranking result."""
    index: int
    relevance_score: float
    document: Optional[str] = None


class Reranker:
    """Cross-encoder reranker with automatic backend detection."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._backend: str | None = None  # "encoder" or "causal"

    def _load(self):
        """Lazy-load model and detect backend type."""
        if self._model is not None:
            return

        import mlx.core as mx

        # Try encoder model first (cross-encoder / sequence classification)
        try:
            from mlx_embeddings import load as load_embeddings
            self._model, self._tokenizer = load_embeddings(self.model_path)
            self._backend = "encoder"
            logger.info(f"Loaded encoder reranker: {self.model_path}")
            return
        except ImportError:
            pass

        # Fall back to CausalLM (e.g., Qwen3-Reranker)
        try:
            from mlx_lm import load as load_lm
            self._model, self._tokenizer = load_lm(self.model_path)
            self._backend = "causal"
            logger.info(f"Loaded CausalLM reranker: {self.model_path}")
            return
        except Exception as e:
            raise RuntimeError(
                f"Could not load reranker model {self.model_path}: {e}"
            ) from e

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        return_documents: bool = False,
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_n: Return only top N results (default: all).
            return_documents: Include document text in results.

        Returns:
            List of RerankResult sorted by relevance_score descending.
        """
        self._load()

        if self._backend == "encoder":
            scores = self._score_encoder(query, documents)
        elif self._backend == "causal":
            scores = self._score_causal(query, documents)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

        # Build results
        results = []
        for i, score in enumerate(scores):
            results.append(RerankResult(
                index=i,
                relevance_score=float(score),
                document=documents[i] if return_documents else None,
            ))

        # Sort by score descending
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        if top_n is not None:
            results = results[:top_n]

        return results

    def _score_encoder(self, query: str, documents: list[str]) -> list[float]:
        """Score using encoder cross-encoder (sequence classification)."""
        import mlx.core as mx

        scores = []
        for doc in documents:
            # Cross-encoder: concatenate query and document
            inputs = self._tokenizer(
                query, doc,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            input_ids = mx.array(inputs["input_ids"])
            attention_mask = mx.array(inputs["attention_mask"])

            # Forward pass
            output = self._model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract relevance score
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, dict):
                logits = output.get("logits", output.get("scores"))
            elif isinstance(output, (list, tuple)):
                logits = output[0]
            else:
                logits = output

            # For binary classification, take positive class score
            if logits.shape[-1] == 1:
                score = float(logits[0, 0])
            elif logits.shape[-1] >= 2:
                score = float(logits[0, 1])  # Positive class
            else:
                score = float(logits.reshape(-1)[0])

            scores.append(score)

        return scores

    def _score_causal(self, query: str, documents: list[str]) -> list[float]:
        """Score using CausalLM yes/no logit comparison (e.g., Qwen3-Reranker)."""
        import mlx.core as mx

        scores = []

        # Get token IDs for "Yes" and "No" (capital — required by Qwen3-Reranker)
        yes_ids = self._tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = self._tokenizer.encode("No", add_special_tokens=False)
        if not yes_ids or not no_ids:
            raise RuntimeError("Could not find 'Yes'/'No' tokens in tokenizer vocabulary")
        yes_id = yes_ids[-1]
        no_id = no_ids[-1]

        for doc in documents:
            # Build reranker prompt
            prompt = (
                f"Given the query: {query}\n\n"
                f"Is the following document relevant? Answer yes or no.\n\n"
                f"Document: {doc}\n\n"
                f"Answer:"
            )

            # Tokenize
            if hasattr(self._tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                input_ids = self._tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                input_ids = mx.array([input_ids])
            else:
                inputs = self._tokenizer(prompt, return_tensors="np")
                input_ids = mx.array(inputs["input_ids"])

            # Forward pass to get logits
            output = self._model(input_ids)
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, (list, tuple)):
                logits = output[0]
            else:
                logits = output

            # Get logits for last token position
            last_logits = logits[0, -1]

            # Score = log(P(yes)) - log(P(no)) via logit difference
            yes_logit = float(last_logits[yes_id])
            no_logit = float(last_logits[no_id])
            score = yes_logit - no_logit

            scores.append(score)

        return scores

    def unload(self):
        """Unload model to free memory."""
        self._model = None
        self._tokenizer = None
        self._backend = None
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_memory_cache"):
                mx.clear_memory_cache()
        except Exception:
            pass
