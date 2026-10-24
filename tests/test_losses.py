"""Unit tests for custom ranking losses and Hard Negative Mining.

Compatible with both standard library `unittest` and `pytest`.
"""

import unittest
from src.models.device import get_device
from src.models.losses import (
    InBatchHardNegativeMiner,
    ListNetRankingLoss,
    TripletMarginRankingLoss,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for Loss tests")
class TestRankingLosses(unittest.TestCase):
    """Test suite verifying TripletMarginRankingLoss, ListNet, and Hard Negative Mining."""

    def setUp(self) -> None:
        self.device = get_device()
        self.loss_fn = TripletMarginRankingLoss(margin=1.0)

    def test_loss_zero_when_margin_satisfied(self) -> None:
        """When (pos - neg) >= margin, loss must be exactly 0.0."""
        pos = torch.tensor([[5.0], [4.0]], device=self.device)
        neg = torch.tensor([[3.0], [2.0]], device=self.device)  # gap is 2.0 >= margin 1.0

        loss = self.loss_fn(pos, neg)
        self.assertEqual(loss.item(), 0.0)

    def test_loss_positive_when_margin_violated(self) -> None:
        """When (pos - neg) < margin, loss must penalize proportional to violation."""
        pos = torch.tensor([[3.0]], device=self.device)
        neg = torch.tensor([[2.5]], device=self.device)  # gap is 0.5 < margin 1.0 -> loss should be 0.5

        loss = self.loss_fn(pos, neg)
        self.assertAlmostEqual(loss.item(), 0.5, places=4)

    def test_loss_backward_gradients(self) -> None:
        """Verify backpropagation produces valid non-zero gradients on scores."""
        pos = torch.tensor([[2.0]], requires_grad=True, device=self.device)
        neg = torch.tensor([[2.0]], requires_grad=True, device=self.device)

        loss = self.loss_fn(pos, neg)
        loss.backward()

        self.assertIsNotNone(pos.grad)
        self.assertIsNotNone(neg.grad)
        # Gradient of (margin - (p - n)) w.r.t p is -1.0; w.r.t n is +1.0
        self.assertAlmostEqual(pos.grad.item(), -1.0, places=4)
        self.assertAlmostEqual(neg.grad.item(), 1.0, places=4)

    def test_loss_telemetry_metrics(self) -> None:
        """compute_telemetry should return accurate diagnostics."""
        pos = torch.tensor([[3.0], [1.0]], device=self.device)
        neg = torch.tensor([[1.0], [2.0]], device=self.device)
        # Pair 1: gap = 2.0 (>= margin 1.0, not active, correct)
        # Pair 2: gap = -1.0 (< margin 1.0, active, incorrect)

        stats = self.loss_fn.compute_telemetry(pos, neg)
        self.assertEqual(stats["active_triplet_ratio"], 0.5)
        self.assertEqual(stats["ranking_accuracy"], 0.5)
        self.assertAlmostEqual(stats["average_score_gap"], 0.5)

    def test_invalid_parameters_error(self) -> None:
        """Loss should reject non-positive margins or invalid reductions."""
        with self.assertRaises(ValueError):
            TripletMarginRankingLoss(margin=0.0)
        with self.assertRaises(ValueError):
            TripletMarginRankingLoss(margin=-1.0)
        with self.assertRaises(ValueError):
            TripletMarginRankingLoss(reduction="invalid_red")

    def test_in_batch_hardest_negative_mining(self) -> None:
        """Hardest negative miner must pick the candidate with highest predicted score."""
        pos = torch.tensor([[4.0], [5.0]], device=self.device)
        neg_candidates = torch.tensor([
            [1.0, 3.8, 2.0],  # Hardest negative is 3.8
            [4.9, 0.5, 1.2],  # Hardest negative is 4.9
        ], device=self.device)

        _, hardest = InBatchHardNegativeMiner.mine_hardest_negatives(pos, neg_candidates)
        self.assertEqual(hardest.shape, (2, 1))
        self.assertAlmostEqual(hardest[0].item(), 3.8, places=4)
        self.assertAlmostEqual(hardest[1].item(), 4.9, places=4)

    def test_listnet_ranking_loss(self) -> None:
        """ListNet ranking loss should compute non-negative cross-entropy and backpropagate."""
        listnet = ListNetRankingLoss(temperature=1.0)
        preds = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True, device=self.device)
        targets = torch.tensor([[3.0, 1.0, 0.0]], device=self.device)

        loss = listnet(preds, targets)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(preds.grad)


if __name__ == "__main__":
    unittest.main()
