"""Production-grade PyTorch Trainer for Cross-Encoder Neural Rerankers.

Features:
  - AdamW optimizer with decoupled weight decay parameter grouping
  - Learning rate warmup with linear decay scheduler
  - Gradient norm clipping to prevent attention divergence
  - Metric telemetry and early stopping to prevent overfitting
"""

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.models.dataset import TripletBatch
from src.models.device import get_device
from src.models.losses import TripletMarginRankingLoss
from src.models.reranker import CrossEncoderReranker


class TrainingConfig:
    """Hyperparameters and settings for model optimization."""

    def __init__(
        self,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        epochs: int = 3,
        warmup_ratio: float = 0.1,
        patience: int = 3,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir


class CrossEncoderTrainer:
    """End-to-end Trainer managing the optimization lifecycle of CrossEncoderReranker."""

    def __init__(
        self,
        model: CrossEncoderReranker,
        config: Optional[TrainingConfig] = None,
        loss_fn: Optional[TripletMarginRankingLoss] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.config = config or TrainingConfig()
        self.loss_fn = loss_fn or TripletMarginRankingLoss(margin=1.0)

        # Build AdamW with decoupled weight decay for 2D weights vs 1D biases/LayerNorms
        self.optimizer = self._build_optimizer()
        self.scheduler: Optional[LambdaLR] = None

        # Tracking state
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self.history: List[Dict[str, float]] = []

    def _build_optimizer(self) -> AdamW:
        """Separate parameters into decay and non-decay groups (biases and LayerNorm)."""
        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Apply weight decay only to 2D+ weights (matrices), not 1D vectors (biases, LayerNorm)
            if param.dim() >= 2 and not any(nd in name for nd in ["bias", "LayerNorm", "norm"]):
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return AdamW(optim_groups, lr=self.config.learning_rate, betas=(0.9, 0.98), eps=1e-8)

    def _setup_scheduler(self, total_steps: int) -> LambdaLR:
        """Construct linear warmup and linear decay learning rate scheduler."""
        warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute one complete training epoch over the training DataLoader."""
        self.model.train()
        total_loss: float = 0.0
        total_active_triplets: float = 0.0
        total_steps = 0

        for batch_data in dataloader:
            if not isinstance(batch_data, TripletBatch):
                raise TypeError(f"Expected TripletBatch, got {type(batch_data)}")

            batch: TripletBatch = batch_data.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass on Positive Pair
            pos_scores = self.model(
                input_ids=batch.pos_input_ids,
                attention_mask=batch.pos_attention_mask,
                token_type_ids=batch.pos_token_type_ids,
            )

            # Forward pass on Negative Pair
            neg_scores = self.model(
                input_ids=batch.neg_input_ids,
                attention_mask=batch.neg_attention_mask,
                token_type_ids=batch.neg_token_type_ids,
            )

            # Compute Triplet Margin Loss
            loss = self.loss_fn(pos_scores, neg_scores)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.max_grad_norm,
            )

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            # Record telemetry
            telemetry = self.loss_fn.compute_telemetry(pos_scores, neg_scores)
            total_loss += loss.item()
            total_active_triplets += telemetry["active_triplet_ratio"]
            total_steps += 1
            self.global_step += 1

        avg_loss = total_loss / max(1, total_steps)
        avg_active = total_active_triplets / max(1, total_steps)
        current_lr = self.optimizer.param_groups[0]["lr"]

        return {
            "train_loss": avg_loss,
            "active_triplets": avg_active,
            "learning_rate": current_lr,
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate validation metrics without computing gradients."""
        self.model.eval()
        total_loss: float = 0.0
        total_accuracy: float = 0.0
        total_gap: float = 0.0
        steps = 0

        with torch.no_grad():
            for batch_data in dataloader:
                batch: TripletBatch = batch_data.to(self.device)

                pos_scores = self.model(
                    input_ids=batch.pos_input_ids,
                    attention_mask=batch.pos_attention_mask,
                    token_type_ids=batch.pos_token_type_ids,
                )
                neg_scores = self.model(
                    input_ids=batch.neg_input_ids,
                    attention_mask=batch.neg_attention_mask,
                    token_type_ids=batch.neg_token_type_ids,
                )

                loss = self.loss_fn(pos_scores, neg_scores)
                telemetry = self.loss_fn.compute_telemetry(pos_scores, neg_scores)

                total_loss += loss.item()
                total_accuracy += telemetry["ranking_accuracy"]
                total_gap += telemetry["average_score_gap"]
                steps += 1

        return {
            "val_loss": total_loss / max(1, steps),
            "val_accuracy": total_accuracy / max(1, steps),
            "val_score_gap": total_gap / max(1, steps),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> List[Dict[str, float]]:
        """Run the full training procedure across all configured epochs with early stopping."""
        total_steps = len(train_loader) * self.config.epochs
        self.scheduler = self._setup_scheduler(total_steps)
        no_improvement_count = 0

        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(train_loader)

            epoch_record: Dict[str, float] = {
                "epoch": float(epoch),
                **train_metrics,
            }

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                epoch_record.update(val_metrics)
                current_val_loss = val_metrics["val_loss"]

                # Check for improvement
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= self.config.patience:
                        epoch_record["early_stopped"] = 1.0
                        self.history.append(epoch_record)
                        break

            self.history.append(epoch_record)

        return self.history
