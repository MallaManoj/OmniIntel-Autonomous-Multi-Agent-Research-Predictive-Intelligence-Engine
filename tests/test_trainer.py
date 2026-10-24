"""Unit tests for CrossEncoderTrainer, optimization, and training telemetry.

Compatible with both standard library `unittest` and `pytest`.
"""

import unittest
from torch.utils.data import DataLoader

from src.models.dataset import (
    SimpleTokenizer,
    TripletRerankingDataset,
    TripletSample,
    collate_triplet_batch,
)
from src.models.device import get_device
from src.models.losses import TripletMarginRankingLoss
from src.models.reranker import CrossEncoderConfig, CrossEncoderReranker
from src.models.trainer import CrossEncoderTrainer, TrainingConfig

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for Trainer tests")
class TestCrossEncoderTrainer(unittest.TestCase):
    """Test suite verifying optimizer weight decay groups, scheduler, and training epochs."""

    def setUp(self) -> None:
        self.device = get_device()
        self.tokenizer = SimpleTokenizer(vocab=["revenue", "growth", "sec", "loss", "profit"])
        self.model_config = CrossEncoderConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=32,
            embedding_dim=32,
            num_heads=2,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.1,
        )
        self.model = CrossEncoderReranker(self.model_config).to(self.device)
        self.train_config = TrainingConfig(
            learning_rate=1e-3,
            weight_decay=0.01,
            max_grad_norm=1.0,
            epochs=2,
            warmup_ratio=0.1,
        )
        self.trainer = CrossEncoderTrainer(
            model=self.model,
            config=self.train_config,
            loss_fn=TripletMarginRankingLoss(margin=1.0),
            device=self.device,
        )

        # Synthetic samples for test training
        self.samples = [
            TripletSample("revenue growth", "profit increased by 20%", "loss increased by 50%"),
            TripletSample("sec filing", "annual 10-k financial report", "marketing press release"),
            TripletSample("profit margin", "net operating profit growth", "operating expense surge"),
            TripletSample("quarter earnings", "quarterly earnings exceeded targets", "product recall warning"),
        ]
        self.dataset = TripletRerankingDataset(self.samples, self.tokenizer, max_seq_len=32)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_triplet_batch,
        )

    def test_optimizer_parameter_grouping(self) -> None:
        """Optimizer must separate 2D weights (decay) from 1D biases/norms (no decay)."""
        groups = self.trainer.optimizer.param_groups
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["weight_decay"], self.train_config.weight_decay)
        self.assertEqual(groups[1]["weight_decay"], 0.0)

        # Ensure all trainable parameters are covered
        total_optim_params = sum(len(g["params"]) for g in groups)
        trainable_model_params = len([p for p in self.model.parameters() if p.requires_grad])
        self.assertEqual(total_optim_params, trainable_model_params)

    def test_train_epoch(self) -> None:
        """train_epoch must return loss, active triplets, and updated learning rate."""
        # Initialize scheduler
        self.trainer.scheduler = self.trainer._setup_scheduler(len(self.dataloader) * self.train_config.epochs)

        metrics = self.trainer.train_epoch(self.dataloader)
        self.assertIn("train_loss", metrics)
        self.assertIn("active_triplets", metrics)
        self.assertIn("learning_rate", metrics)
        self.assertGreaterEqual(metrics["train_loss"], 0.0)
        self.assertEqual(self.trainer.current_epoch, 0)
        self.assertEqual(self.trainer.global_step, len(self.dataloader))

    def test_evaluate(self) -> None:
        """evaluate must return val_loss, val_accuracy, and val_score_gap."""
        metrics = self.trainer.evaluate(self.dataloader)
        self.assertIn("val_loss", metrics)
        self.assertIn("val_accuracy", metrics)
        self.assertIn("val_score_gap", metrics)
        self.assertGreaterEqual(metrics["val_loss"], 0.0)
        self.assertTrue(0.0 <= metrics["val_accuracy"] <= 1.0)

    def test_full_fit(self) -> None:
        """fit must train across epochs and record history."""
        history = self.trainer.fit(self.dataloader, val_loader=self.dataloader)
        self.assertEqual(len(history), self.train_config.epochs)
        for record in history:
            self.assertIn("train_loss", record)
            self.assertIn("val_loss", record)
            self.assertIn("val_accuracy", record)


if __name__ == "__main__":
    unittest.main()
