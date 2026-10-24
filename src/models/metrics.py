"""Information Retrieval and Reranking evaluation metrics.

Implements standard benchmark metrics:
  - MRR (Mean Reciprocal Rank)
  - NDCG@K (Normalized Discounted Cumulative Gain at rank K)
  - Hit Rate@K (Success at rank K)
  - MAP (Mean Average Precision)
"""

import math
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np


def compute_reciprocal_rank(
    predicted_scores: Sequence[float],
    ground_truth_relevance: Sequence[Union[int, float]],
) -> float:
    """Compute Reciprocal Rank (RR) for a single ranked candidate list.

    Formula:
        RR = 1 / rank_of_first_relevant_document

    Returns:
        float: 1 / rank if a relevant document (relevance > 0) is found, else 0.0.
    """
    if len(predicted_scores) != len(ground_truth_relevance):
        raise ValueError("Length mismatch between predictions and ground truth.")

    # Sort indices by predicted scores descending
    ranked_indices = np.argsort(predicted_scores)[::-1]

    for rank_idx, doc_idx in enumerate(ranked_indices, start=1):
        if ground_truth_relevance[doc_idx] > 0:
            return 1.0 / rank_idx

    return 0.0


def compute_dcg_at_k(
    relevance_labels: Sequence[Union[int, float]],
    k: int = 10,
) -> float:
    """Compute Discounted Cumulative Gain at rank K.

    Formula:
        DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    """
    dcg = 0.0
    limit = min(k, len(relevance_labels))

    for i in range(limit):
        rel = relevance_labels[i]
        # Gain discount factor: log2(rank + 1)
        dcg += (math.pow(2.0, rel) - 1.0) / math.log2(i + 2.0)

    return dcg


def compute_ndcg_at_k(
    predicted_scores: Sequence[float],
    ground_truth_relevance: Sequence[Union[int, float]],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K.

    Formula:
        NDCG@K = DCG@K / IDCG@K

    Where IDCG@K is the ideal DCG achieved by perfectly ordering relevance.
    """
    if len(predicted_scores) != len(ground_truth_relevance):
        raise ValueError("Length mismatch between predictions and ground truth.")

    # Sort documents by predicted scores descending
    ranked_indices = np.argsort(predicted_scores)[::-1]
    sorted_labels = [ground_truth_relevance[idx] for idx in ranked_indices]

    # Calculate actual DCG@K
    actual_dcg = compute_dcg_at_k(sorted_labels, k=k)

    # Calculate Ideal DCG@K (sorted by ground truth relevance descending)
    ideal_labels = sorted(ground_truth_relevance, reverse=True)
    ideal_dcg = compute_dcg_at_k(ideal_labels, k=k)

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def compute_hit_rate_at_k(
    predicted_scores: Sequence[float],
    ground_truth_relevance: Sequence[Union[int, float]],
    k: int = 5,
) -> float:
    """Check if at least one relevant document appears in the top-K.

    Returns:
        1.0 if top-K contains any document with relevance > 0, else 0.0.
    """
    if len(predicted_scores) != len(ground_truth_relevance):
        raise ValueError("Length mismatch between predictions and ground truth.")

    ranked_indices = np.argsort(predicted_scores)[::-1][:k]
    for idx in ranked_indices:
        if ground_truth_relevance[idx] > 0:
            return 1.0
    return 0.0


class RankingEvaluator:
    """Aggregates ranking performance metrics across a full evaluation dataset."""

    def __init__(self, k_thresholds: Tuple[int, ...] = (1, 3, 5, 10)) -> None:
        self.k_thresholds = k_thresholds
        self.query_records: List[Dict[str, float]] = []

    def evaluate_query(
        self,
        predicted_scores: Sequence[float],
        ground_truth_relevance: Sequence[Union[int, float]],
        query_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute and store metrics for a single query."""
        rr = compute_reciprocal_rank(predicted_scores, ground_truth_relevance)

        record: Dict[str, float] = {"mrr": rr}

        for k in self.k_thresholds:
            record[f"ndcg@{k}"] = compute_ndcg_at_k(predicted_scores, ground_truth_relevance, k=k)
            record[f"hit_rate@{k}"] = compute_hit_rate_at_k(predicted_scores, ground_truth_relevance, k=k)

        self.query_records.append(record)
        return record

    def compute_summary(self) -> Dict[str, float]:
        """Compute average metrics across all evaluated queries."""
        if not self.query_records:
            return {}

        total_queries = len(self.query_records)
        summary: Dict[str, float] = {"total_queries": float(total_queries)}

        # Aggregate all keys across records
        all_keys = self.query_records[0].keys()
        for key in all_keys:
            vals = [rec[key] for rec in self.query_records]
            summary[f"mean_{key}"] = round(float(np.mean(vals)), 4)

        return summary

    def clear(self) -> None:
        """Reset accumulated evaluation records."""
        self.query_records.clear()
