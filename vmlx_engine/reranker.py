# SPDX-License-Identifier: Apache-2.0
"""
Reranker engine for /v1/rerank endpoint.

Supports three scoring backends:
1. Late-interaction models (jina-reranker-v3) — listwise reranking via
   special token embeddings + MLP projector + cosine similarity.
2. Encoder models (ModernBERT, XLM-RoBERTa, BGE) — sequence classification
   scores via forward pass through cross-encoder architecture.
3. CausalLM models (Qwen3-Reranker) — yes/no logit scoring via generate.

Usage:
    reranker = Reranker(model_path)
    results = reranker.rerank(query, documents, top_n=5)
"""

from __future__ import annotations

import logging
import os
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
        self._projector = None
        self._backend: str | None = None  # "late_interaction", "encoder", or "causal"
        # Special token IDs for late-interaction models
        self._doc_embed_token_id: int | None = None
        self._query_embed_token_id: int | None = None

    def _is_late_interaction_model(self, model_path: str) -> bool:
        """Check if model is a late-interaction reranker (e.g., jina-reranker-v3).

        Detection: presence of projector.safetensors in the model directory.
        """
        # Resolve HF cache path or local path
        resolved = self._resolve_model_path(model_path)
        if resolved and os.path.isfile(os.path.join(resolved, "projector.safetensors")):
            return True
        return False

    @staticmethod
    def _resolve_model_path(model_path: str) -> str | None:
        """Resolve a HF model ID or local path to a directory."""
        # If it's already a directory, use it
        if os.path.isdir(model_path):
            return model_path

        # Try HF cache
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(model_path, local_files_only=True)
        except Exception:
            pass

        # Try common HF cache locations
        for cache_base in [
            os.path.expanduser("~/.cache/huggingface/hub"),
            "/tmp/huggingface",
        ]:
            candidate = os.path.join(cache_base, model_path)
            if os.path.isdir(candidate):
                return candidate

        return None

    def _load(self):
        """Lazy-load model and detect backend type."""
        if self._model is not None:
            return

        import mlx.core as mx

        # 1. Check for late-interaction model (jina-reranker-v3 style)
        if self._is_late_interaction_model(self.model_path):
            self._load_late_interaction()
            return

        # 2. Try encoder model (cross-encoder / sequence classification)
        try:
            from mlx_embeddings import load as load_embeddings
            self._model, self._tokenizer = load_embeddings(self.model_path)
            self._backend = "encoder"
            logger.info(f"Loaded encoder reranker: {self.model_path}")
            return
        except ImportError:
            pass

        # 3. Fall back to CausalLM (e.g., Qwen3-Reranker)
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

    def _load_late_interaction(self):
        """Load a late-interaction reranker (jina-reranker-v3 style).

        These models use:
        - A CausalLM backbone (Qwen3) for hidden state extraction
        - An MLP projector to map hidden states to embedding space
        - Special tokens to mark query/document embedding positions
        - Cosine similarity for scoring
        """
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load as load_lm

        self._model, self._tokenizer = load_lm(self.model_path)

        # Load projector weights
        resolved = self._resolve_model_path(self.model_path)
        projector_path = os.path.join(resolved, "projector.safetensors")

        from safetensors import safe_open

        # Detect projector architecture from weights
        with safe_open(projector_path, framework="numpy") as f:
            keys = list(f.keys())
            w0 = f.get_tensor("linear1.weight")
            w2 = f.get_tensor("linear2.weight")
            in_dim, out_mid = w0.shape[0], w0.shape[1]
            _, out_dim = w2.shape

        # Build MLP projector dynamically
        class MLPProjector(nn.Module):
            def __init__(self, in_features, mid_features, out_features):
                super().__init__()
                self.linear1 = nn.Linear(in_features, mid_features, bias=False)
                self.linear2 = nn.Linear(mid_features, out_features, bias=False)

            def __call__(self, x):
                x = self.linear1(x)
                x = nn.relu(x)
                x = self.linear2(x)
                return x

        self._projector = MLPProjector(in_dim, out_mid, out_dim)
        with safe_open(projector_path, framework="numpy") as f:
            self._projector.linear1.weight = mx.array(f.get_tensor("linear1.weight"))
            self._projector.linear2.weight = mx.array(f.get_tensor("linear2.weight"))

        # Detect special token IDs from tokenizer or config
        self._doc_embed_token_id = self._find_special_token_id("embed_token", 151670)
        self._query_embed_token_id = self._find_special_token_id("rerank_token", 151671)

        self._backend = "late_interaction"
        logger.info(
            f"Loaded late-interaction reranker: {self.model_path} "
            f"(projector {in_dim}→{out_mid}→{out_dim})"
        )

    def _find_special_token_id(self, token_name: str, default_id: int) -> int:
        """Find special token ID from tokenizer, falling back to default."""
        # Try various token formats
        for pattern in [f"<|{token_name}|>", f"<{token_name}>", token_name]:
            try:
                ids = self._tokenizer.encode(pattern, add_special_tokens=False)
                if ids:
                    return ids[0] if len(ids) == 1 else ids[-1]
            except Exception:
                pass

        # Try tokenizer's added_tokens
        tokenizer = getattr(self._tokenizer, "_tokenizer", self._tokenizer)
        if hasattr(tokenizer, "added_tokens_encoder"):
            for tok_str, tok_id in tokenizer.added_tokens_encoder.items():
                if token_name in tok_str:
                    return tok_id

        logger.warning(
            f"Special token '{token_name}' not found in tokenizer, "
            f"using default ID {default_id}"
        )
        return default_id

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

        if self._backend == "late_interaction":
            scores = self._score_late_interaction(query, documents)
        elif self._backend == "encoder":
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

    def _score_late_interaction(
        self, query: str, documents: list[str]
    ) -> list[float]:
        """Score using late-interaction (jina-reranker-v3 style).

        All documents are processed in a single forward pass:
        1. Format query + all docs into a listwise prompt with special tokens
        2. Extract hidden states at special token positions
        3. Project through MLP projector
        4. Compute cosine similarity between query and each doc embedding
        """
        import mlx.core as mx
        import numpy as np

        special_tokens = {
            "query_embed_token": "<|rerank_token|>",
            "doc_embed_token": "<|embed_token|>",
        }

        # Build the listwise prompt
        prompt = self._format_late_interaction_prompt(
            query, documents, special_tokens
        )

        # Tokenize
        input_ids = self._tokenizer.encode(prompt)

        # Forward pass through the backbone to get hidden states
        hidden_states = self._model.model(mx.array([input_ids]))

        # Remove batch dimension: [seq_len, hidden_size]
        hidden_states = hidden_states[0]

        input_ids_np = np.array(input_ids)

        # Find positions of special tokens
        query_positions = np.where(
            input_ids_np == self._query_embed_token_id
        )[0]
        doc_positions = np.where(
            input_ids_np == self._doc_embed_token_id
        )[0]

        if len(query_positions) == 0:
            raise RuntimeError(
                "Query embed token not found in tokenized input. "
                f"Token ID {self._query_embed_token_id} missing."
            )
        if len(doc_positions) == 0:
            raise RuntimeError(
                "Document embed tokens not found in tokenized input. "
                f"Token ID {self._doc_embed_token_id} missing."
            )

        # Extract embeddings at special token positions
        query_hidden = mx.expand_dims(
            hidden_states[int(query_positions[0])], axis=0
        )  # [1, hidden_size]
        doc_hidden = mx.stack(
            [hidden_states[int(pos)] for pos in doc_positions]
        )  # [num_docs, hidden_size]

        # Project through MLP
        query_embeds = self._projector(query_hidden)  # [1, embed_dim]
        doc_embeds = self._projector(doc_hidden)  # [num_docs, embed_dim]

        # Cosine similarity
        query_expanded = mx.broadcast_to(
            query_embeds, doc_embeds.shape
        )  # [num_docs, embed_dim]

        scores = mx.sum(doc_embeds * query_expanded, axis=-1) / (
            mx.sqrt(mx.sum(doc_embeds * doc_embeds, axis=-1))
            * mx.sqrt(mx.sum(query_expanded * query_expanded, axis=-1))
        )  # [num_docs]

        mx.eval(scores)
        return scores.tolist()

    @staticmethod
    def _format_late_interaction_prompt(
        query: str,
        docs: list[str],
        special_tokens: dict[str, str],
        instruction: str | None = None,
    ) -> str:
        """Format query and documents into a listwise reranking prompt.

        Follows the jina-reranker-v3 prompt format with special embed tokens
        for extracting query and document representations.
        """
        doc_emb_token = special_tokens["doc_embed_token"]
        query_emb_token = special_tokens["query_embed_token"]

        prefix = (
            "<|im_start|>system\n"
            "You are a search relevance expert who can determine a ranking "
            "of the passages based on how relevant they are to the query. "
            "If the query is a question, how relevant a passage is depends "
            "on how well it answers the question. If not, try to analyze the "
            "intent of the query and assess how well each passage satisfies "
            "the intent. If an instruction is provided, you should follow "
            "the instruction when determining the ranking."
            "<|im_end|>\n<|im_start|>user\n"
        )
        # Suppress reasoning (thinking) for efficiency
        suffix = (
            "<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

        body = (
            f"I will provide you with {len(docs)} passages, each indicated "
            f"by a numerical identifier. Rank the passages based on their "
            f"relevance to query: {query}\n"
        )

        if instruction:
            body += f"<instruct>\n{instruction}\n</instruct>\n"

        doc_parts = [
            f'<passage id="{i}">\n{doc}{doc_emb_token}\n</passage>'
            for i, doc in enumerate(docs)
        ]
        body += "\n".join(doc_parts) + "\n"
        body += f"<query>\n{query}{query_emb_token}\n</query>"

        return prefix + body + suffix

    def _score_encoder(self, query: str, documents: list[str]) -> list[float]:
        """Score using encoder cross-encoder (sequence classification)."""
        import mlx.core as mx

        # Unwrap TokenizerWrapper (mlx_embeddings wraps the HF tokenizer)
        tokenizer = getattr(self._tokenizer, "_tokenizer", self._tokenizer)

        scores = []
        for doc in documents:
            # Cross-encoder: concatenate query and document
            inputs = tokenizer(
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
        self._projector = None
        self._backend = None
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_memory_cache"):
                mx.clear_memory_cache()
        except Exception:
            pass
