"""High-performance PyTorch inference engine with micro-batching and LRU caching.

Integrates custom CrossEncoderReranker with LRUCache (from Step 2).
Provides sub-millisecond cached responses and dynamic micro-batched neural inference.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.core.dsa.lru_cache import LRUCache
from src.models.checkpoint import ModelCheckpointManager
from src.models.dataset import SimpleTokenizer
from src.models.device import DeviceManager, get_device
from src.models.reranker import CrossEncoderReranker


@dataclass
class RerankResult:
    """Structured result for a reranked candidate passage."""

    document: str
    score: float
    original_index: int
    metadata: Optional[Dict[str, Any]] = None


class CrossEncoderInferenceEngine:
    """Production serving engine for Cross-Encoder neural reranking.

    Features:
      - Embedded LRUCache eliminating redundant forward passes for repeated queries
      - Vectorized micro-batching for maximum GPU tensor core utilization
      - Zero-overhead inference execution under torch.inference_mode()
      - Telemetry reporting inference latency and cache hit ratios
    """

    def __init__(
        self,
        model: CrossEncoderReranker,
        tokenizer: SimpleTokenizer,
        device: Optional[torch.device] = None,
        cache_capacity: int = 1024,
        max_seq_len: int = 64,
    ) -> None:
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Integrated LRU Cache (query, document) -> float relevance score
        self.cache: LRUCache[Tuple[str, str], float] = LRUCache(capacity=cache_capacity)

        # Performance telemetry
        self.total_queries: int = 0
        self.total_documents_scored: int = 0
        self.total_inference_time_ms: float = 0.0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        cache_capacity: int = 1024,
    ) -> "CrossEncoderInferenceEngine":
        """Factory method to instantiate the inference engine directly from a saved checkpoint."""
        model, tokenizer, meta = ModelCheckpointManager.load_checkpoint(
            filepath=checkpoint_path,
            device=device,
        )
        max_len = model.config.max_seq_len
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            cache_capacity=cache_capacity,
            max_seq_len=max_len,
        )

    def predict_score(self, query: str, document: str) -> float:
        """Predict continuous relevance score for a single query-document pair.

        Utilizes LRUCache to return in O(1) time if previously scored.
        """
        cache_key = (query.strip().lower(), document.strip().lower())
        cached_score = self.cache.get(cache_key)
        if cached_score is not None:
            return cached_score

        # Uncached: run forward pass
        encoded = self.tokenizer.encode_pair(query, document, max_seq_len=self.max_seq_len)
        input_ids = torch.tensor([encoded[0]], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([encoded[1]], dtype=torch.long, device=self.device)
        token_type_ids = torch.tensor([encoded[2]], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            logit = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            score = float(torch.sigmoid(logit).item())

        self.cache.put(cache_key, score)
        return score

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        batch_size: int = 16,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RerankResult]:
        """Rerank a collection of documents against a query.

        Args:
            query: The user research inquiry.
            documents: List of retrieved text passages.
            top_k: Optional cutoff to return only the highest-scoring K documents.
            batch_size: Micro-batch size for neural forward pass.
            metadata_list: Optional metadata associated with each document.

        Returns:
            List of RerankResult sorted by relevance score descending.
        """
        if not documents:
            return []

        start_time = time.perf_counter()
        clean_q = query.strip().lower()
        num_docs = len(documents)

        scores: List[Optional[float]] = [None] * num_docs
        uncached_indices: List[int] = []

        # Step 1: Check LRU Cache for each document
        for idx, doc in enumerate(documents):
            key = (clean_q, doc.strip().lower())
            cached_val = self.cache.get(key)
            if cached_val is not None:
                scores[idx] = cached_val
            else:
                uncached_indices.append(idx)

        # Step 2: Micro-batch forward pass for uncached documents
        if uncached_indices:
            for b_start in range(0, len(uncached_indices), batch_size):
                b_indices = uncached_indices[b_start : b_start + batch_size]
                b_docs = [documents[i] for i in b_indices]

                # Encode pairs
                encoded_pairs = [
                    self.tokenizer.encode_pair(query, doc, max_seq_len=self.max_seq_len)
                    for doc in b_docs
                ]

                # Dynamic padding to max sequence length within this micro-batch
                max_len = max(len(p[0]) for p in encoded_pairs)
                b_size = len(b_docs)

                b_ids = torch.full((b_size, max_len), fill_value=SimpleTokenizer.PAD_ID, dtype=torch.long)
                b_mask = torch.zeros((b_size, max_len), dtype=torch.long)
                b_types = torch.zeros((b_size, max_len), dtype=torch.long)

                for row_idx, (p_ids, p_mask, p_types) in enumerate(encoded_pairs):
                    cur_len = len(p_ids)
                    b_ids[row_idx, :cur_len] = torch.tensor(p_ids, dtype=torch.long)
                    b_mask[row_idx, :cur_len] = torch.tensor(p_mask, dtype=torch.long)
                    b_types[row_idx, :cur_len] = torch.tensor(p_types, dtype=torch.long)

                # Move to compute accelerator
                b_ids = b_ids.to(self.device)
                b_mask = b_mask.to(self.device)
                b_types = b_types.to(self.device)

                with torch.inference_mode():
                    logits = self.model(b_ids, attention_mask=b_mask, token_type_ids=b_types)
                    b_scores = torch.sigmoid(logits).view(-1).tolist()

                # Store scores and populate LRU Cache
                for orig_idx, pred_score in zip(b_indices, b_scores):
                    scores[orig_idx] = pred_score
                    self.cache.put((clean_q, documents[orig_idx].strip().lower()), pred_score)

        # Step 3: Package, sort descending, and slice top-K
        results: List[RerankResult] = []
        for idx, (doc, s) in enumerate(zip(documents, scores)):
            meta = metadata_list[idx] if metadata_list and idx < len(metadata_list) else None
            results.append(
                RerankResult(
                    document=doc,
                    score=s if s is not None else 0.0,
                    original_index=idx,
                    metadata=meta,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        # Update telemetry
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.total_queries += 1
        self.total_documents_scored += num_docs
        self.total_inference_time_ms += elapsed_ms

        return results

    @property
    def telemetry(self) -> Dict[str, Any]:
        """Return runtime performance statistics."""
        avg_ms_per_query = (
            self.total_inference_time_ms / self.total_queries
            if self.total_queries > 0
            else 0.0
        )
        return {
            "total_queries": self.total_queries,
            "total_documents_scored": self.total_documents_scored,
            "cache_stats": self.cache.stats,
            "average_latency_ms": round(avg_ms_per_query, 2),
        }
