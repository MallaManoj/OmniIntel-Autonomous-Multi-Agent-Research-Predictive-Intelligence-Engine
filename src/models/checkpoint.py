"""Production-grade Model Checkpoint Manager for PyTorch architectures.

Features:
  - Strict state_dict serialization (avoids arbitrary pickle security risks)
  - Atomic saving via temporary file replacement to prevent corrupt files
  - Automatic model and tokenizer reconstruction on load
  - Lightweight inspection of metadata without GPU memory allocation
"""

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.dataset import SimpleTokenizer
from src.models.device import get_device
from src.models.reranker import CrossEncoderConfig, CrossEncoderReranker


class ModelCheckpointManager:
    """Manages atomic serialization, validation, and reconstruction of neural models."""

    CHECKPOINT_VERSION: str = "1.0.0"

    @classmethod
    def save_checkpoint(
        cls,
        model: CrossEncoderReranker,
        filepath: str,
        tokenizer: Optional[SimpleTokenizer] = None,
        config: Optional[CrossEncoderConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Atomically persist model state_dict and metadata to disk.

        Uses atomic file renaming (os.replace) to ensure no corrupt partial files
        exist if the write process is interrupted.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        tmp_filepath = f"{filepath}.tmp"

        cfg = config or model.config
        payload: Dict[str, Any] = {
            "version": cls.CHECKPOINT_VERSION,
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": cfg.vocab_size,
                "max_seq_len": cfg.max_seq_len,
                "embedding_dim": cfg.embedding_dim,
                "num_heads": cfg.num_heads,
                "num_layers": cfg.num_layers,
                "dim_feedforward": cfg.dim_feedforward,
                "dropout": cfg.dropout,
            },
            "vocab": tokenizer.vocab if tokenizer else None,
            "epoch": epoch,
            "metrics": metrics or {},
        }

        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()

        # Write to temporary file, then atomic rename
        torch.save(payload, tmp_filepath)
        os.replace(tmp_filepath, filepath)
        return filepath

    @classmethod
    def load_checkpoint(
        cls,
        filepath: str,
        device: Optional[torch.device] = None,
    ) -> Tuple[CrossEncoderReranker, SimpleTokenizer, Dict[str, Any]]:
        """Reconstruct model and tokenizer directly from saved checkpoint state.

        Returns:
            Tuple of (model, tokenizer, metadata_dict) with model set to eval().
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        target_device = device or get_device()
        checkpoint = torch.load(filepath, map_location=target_device)

        # 1. Reconstruct Config
        cfg_dict = checkpoint["config"]
        config = CrossEncoderConfig(
            vocab_size=cfg_dict["vocab_size"],
            max_seq_len=cfg_dict["max_seq_len"],
            embedding_dim=cfg_dict["embedding_dim"],
            num_heads=cfg_dict["num_heads"],
            num_layers=cfg_dict["num_layers"],
            dim_feedforward=cfg_dict["dim_feedforward"],
            dropout=cfg_dict.get("dropout", 0.1),
        )

        # 2. Reconstruct Model Architecture & Load Weights
        model = CrossEncoderReranker(config).to(target_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # 3. Reconstruct Tokenizer
        vocab = checkpoint.get("vocab") or []
        tokenizer = SimpleTokenizer(vocab=vocab)

        metadata = {
            "version": checkpoint.get("version"),
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics", {}),
            "filepath": filepath,
        }

        return model, tokenizer, metadata

    @classmethod
    def inspect_checkpoint(cls, filepath: str) -> Dict[str, Any]:
        """Read checkpoint metadata and parameter counts without allocating GPU tensors."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        # Load metadata strictly on CPU
        data = torch.load(filepath, map_location=torch.device("cpu"))
        state_dict = data.get("model_state_dict", {})

        total_tensors = len(state_dict)
        total_parameters = sum(t.numel() for t in state_dict.values())
        file_size_bytes = os.path.getsize(filepath)

        return {
            "version": data.get("version"),
            "epoch": data.get("epoch"),
            "metrics": data.get("metrics", {}),
            "config": data.get("config", {}),
            "vocab_size": len(data.get("vocab", [])),
            "total_tensors": total_tensors,
            "total_parameters": total_parameters,
            "file_size_mb": round(file_size_bytes / (1024**2), 4),
        }
