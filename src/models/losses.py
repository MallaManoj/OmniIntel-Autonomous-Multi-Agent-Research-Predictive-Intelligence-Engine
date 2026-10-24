"""Custom PyTorch loss functions and Hard Negative Mining for semantic reranking.

Implements Triplet Margin Ranking Loss and In-Batch Hard Negative Mining.
Forces the deep model to enforce a strict geometric separation margin between
relevant documents and misleading/confusing negative documents.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginRankingLoss(nn.Module):
    """Pairwise Triplet Margin Loss for neural ranking and rerankers.

    Mathematical formulation:
        L(s_pos, s_neg) = max(0, margin - (s_pos - s_neg))

    Where:
        s_pos = Predicted relevance score for (query, positive_document)
        s_neg = Predicted relevance score for (query, negative_document)
        margin = Minimum target separation between positive and negative scores

    Attributes:
        margin: Target score separation gap (default: 1.0).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError(f"Margin must be positive, got {margin}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")

        self.margin: float = margin
        self.reduction: str = reduction

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the margin ranking loss.

        Args:
            pos_scores: Tensor of shape [batch_size, 1] or [batch_size].
            neg_scores: Tensor of shape [batch_size, 1] or [batch_size].

        Returns:
            Computed scalar loss (or tensor if reduction='none').
        """
        # Ensure flat 1D tensors
        p = pos_scores.view(-1)
        n = neg_scores.view(-1)

        if p.shape != n.shape:
            raise ValueError(
                f"Shape mismatch: pos_scores {p.shape} vs neg_scores {n.shape}"
            )

        # Margin penalty: loss > 0 whenever (p - n) < margin
        raw_loss = torch.clamp(self.margin - (p - n), min=0.0)

        if self.reduction == "mean":
            return raw_loss.mean()
        elif self.reduction == "sum":
            return raw_loss.sum()
        return raw_loss

    def compute_telemetry(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> dict[str, float]:
        """Compute training diagnostics without calculating gradients.

        Returns:
            Dictionary with:
              - 'active_triplet_ratio': Fraction of triplets violating margin [0.0 - 1.0]
              - 'average_score_gap': Average difference (s_pos - s_neg)
              - 'accuracy': Fraction of triplets where s_pos > s_neg
        """
        with torch.no_grad():
            p = pos_scores.view(-1)
            n = neg_scores.view(-1)
            diff = p - n

            active_mask = diff < self.margin
            correct_mask = diff > 0.0

            total = float(len(diff))
            return {
                "active_triplet_ratio": float(active_mask.sum().item() / total) if total > 0 else 0.0,
                "average_score_gap": float(diff.mean().item()) if total > 0 else 0.0,
                "ranking_accuracy": float(correct_mask.sum().item() / total) if total > 0 else 0.0,
            }


class InBatchHardNegativeMiner:
    """Mines the most difficult negative candidates within a batch to prevent gradient saturation.

    Identifies negative passages that have high predicted scores (confusing to the model),
    focusing gradient updates on the most informative training samples.
    """

    @staticmethod
    def mine_hardest_negatives(
        pos_scores: torch.Tensor,
        neg_candidates_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select the highest-scoring (hardest) negative for each query in the batch.

        Args:
            pos_scores: Tensor [batch_size, 1] containing scores of true positives.
            neg_candidates_scores: Tensor [batch_size, num_negatives] containing candidate scores.

        Returns:
            Tuple of (pos_scores, hardest_neg_scores [batch_size, 1]).
        """
        if neg_candidates_scores.dim() != 2:
            raise ValueError("neg_candidates_scores must be a 2D tensor [batch_size, num_negatives]")

        # Pick the maximum scoring negative per row (the hardest imposter)
        hardest_neg, _ = torch.max(neg_candidates_scores, dim=1, keepdim=True)
        return pos_scores, hardest_neg

    @staticmethod
    def mine_semi_hard_negatives(
        pos_scores: torch.Tensor,
        neg_candidates_scores: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Mine semi-hard negatives: s_neg where (s_pos - margin) < s_neg < s_pos.

        Semi-hard negatives provide cleaner, more stable gradients than extreme outliers.
        """
        batch_size = pos_scores.shape[0]
        selected_negs = []

        for i in range(batch_size):
            p = pos_scores[i].item()
            negs = neg_candidates_scores[i]

            # Filter for semi-hard condition
            semi_hard_mask = (negs < p) & (negs > (p - margin))
            candidates = negs[semi_hard_mask]

            if len(candidates) > 0:
                # Take highest among semi-hard
                selected_negs.append(candidates.max().unsqueeze(0))
            else:
                # Fallback to hardest negative overall
                selected_negs.append(negs.max().unsqueeze(0))

        return torch.stack(selected_negs, dim=0)


class ListNetRankingLoss(nn.Module):
    """Listwise Cross-Entropy Ranking Loss (ListNet) for top-K list reranking.

    Transforms candidate scores into a probability distribution via Softmax and
    computes Cross-Entropy with true target relevance distribution.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        predicted_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Args:

        predicted_logits: Tensor [batch_size, num_candidates] of predicted scores.
        target_labels: Tensor [batch_size, num_candidates] of ground truth relevance (e.g. 0 to 3).
        """
        # Convert target labels and predictions to probability distributions
        p_target = F.softmax(target_labels / self.temperature, dim=-1)
        log_p_pred = F.log_softmax(predicted_logits / self.temperature, dim=-1)

        # Cross-Entropy: - sum(P_target * log(P_pred))
        loss = -torch.sum(p_target * log_p_pred, dim=-1)
        return loss.mean()
