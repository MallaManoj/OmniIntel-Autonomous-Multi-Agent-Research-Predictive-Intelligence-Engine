"""Unit tests for ModelCheckpointManager and atomic checkpoint persistence.

Compatible with both standard library `unittest` and `pytest`.
"""

import os
import shutil
import tempfile
import unittest

from src.models.checkpoint import ModelCheckpointManager
from src.models.dataset import SimpleTokenizer
from src.models.device import get_device
from src.models.reranker import CrossEncoderConfig, CrossEncoderReranker

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for Checkpoint tests")
class TestModelCheckpointManager(unittest.TestCase):
    """Test suite verifying atomic save, load, inspection, and weight fidelity."""

    def setUp(self) -> None:
        self.device = get_device()
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, "test_model.pt")

        self.tokenizer = SimpleTokenizer(vocab=["revenue", "apple", "profit", "quarter"])
        self.config = CrossEncoderConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=16,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
            dim_feedforward=32,
        )
        self.model = CrossEncoderReranker(self.config).to(self.device)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_roundtrip(self) -> None:
        """Loaded model must match original configuration and parameter values."""
        ModelCheckpointManager.save_checkpoint(
            model=self.model,
            filepath=self.checkpoint_path,
            tokenizer=self.tokenizer,
            config=self.config,
            epoch=2,
            metrics={"mrr": 1.0, "ndcg": 0.95},
        )
        self.assertTrue(os.path.exists(self.checkpoint_path))

        # Reload model
        loaded_model, loaded_tokenizer, meta = ModelCheckpointManager.load_checkpoint(
            filepath=self.checkpoint_path,
            device=self.device,
        )

        self.assertFalse(loaded_model.training)  # Must be in eval() mode
        self.assertEqual(loaded_model.config.vocab_size, self.config.vocab_size)
        self.assertEqual(loaded_tokenizer.vocab, self.tokenizer.vocab)
        self.assertEqual(meta["epoch"], 2)
        self.assertAlmostEqual(meta["metrics"]["mrr"], 1.0)

        # Check numerical fidelity: weights must be identical
        for (n1, p1), (n2, p2) in zip(self.model.named_parameters(), loaded_model.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.allclose(p1, p2, atol=1e-6))

    def test_reloaded_model_forward_pass_identical(self) -> None:
        """Loaded model must produce bit-for-bit identical forward output logits."""
        ModelCheckpointManager.save_checkpoint(
            model=self.model,
            filepath=self.checkpoint_path,
            tokenizer=self.tokenizer,
        )
        loaded_model, _, _ = ModelCheckpointManager.load_checkpoint(
            filepath=self.checkpoint_path,
            device=self.device,
        )

        sample_ids = torch.randint(0, self.config.vocab_size, (2, 8), device=self.device)
        self.model.eval()

        with torch.no_grad():
            orig_out = self.model(sample_ids)
            loaded_out = loaded_model(sample_ids)

        self.assertTrue(torch.allclose(orig_out, loaded_out, atol=1e-5))

    def test_inspect_checkpoint(self) -> None:
        """inspect_checkpoint must read metadata and stats without loading model into memory."""
        ModelCheckpointManager.save_checkpoint(
            model=self.model,
            filepath=self.checkpoint_path,
            tokenizer=self.tokenizer,
            epoch=5,
            metrics={"mrr": 0.88},
        )

        info = ModelCheckpointManager.inspect_checkpoint(self.checkpoint_path)
        self.assertEqual(info["epoch"], 5)
        self.assertEqual(info["metrics"]["mrr"], 0.88)
        self.assertGreater(info["total_parameters"], 0)
        self.assertGreater(info["file_size_mb"], 0.0)

    def test_missing_file_raises_error(self) -> None:
        """Loading or inspecting non-existent path must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            ModelCheckpointManager.load_checkpoint("non_existent_file.pt")
        with self.assertRaises(FileNotFoundError):
            ModelCheckpointManager.inspect_checkpoint("non_existent_file.pt")


if __name__ == "__main__":
    unittest.main()
