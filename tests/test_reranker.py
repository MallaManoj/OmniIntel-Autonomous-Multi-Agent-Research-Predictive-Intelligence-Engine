"""Unit tests for CrossEncoderReranker neural network architecture.

Compatible with both standard library `unittest` and `pytest`.
"""

import unittest
from src.models.device import get_device
from src.models.reranker import CrossEncoderConfig, CrossEncoderReranker

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for CrossEncoder tests")
class TestCrossEncoderReranker(unittest.TestCase):
    """Test suite verifying neural forward/backward passes and architectural constraints."""

    def setUp(self) -> None:
        self.device = get_device()
        self.config = CrossEncoderConfig(
            vocab_size=500,
            max_seq_len=32,
            embedding_dim=32,
            num_heads=2,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.1,
        )
        self.model = CrossEncoderReranker(self.config).to(self.device)

    def test_parameter_counts(self) -> None:
        """Verify model registers trainable parameters."""
        self.assertGreater(self.model.total_parameters, 0)
        self.assertEqual(self.model.total_parameters, self.model.trainable_parameters)

    def test_forward_output_dimensions(self) -> None:
        """Forward pass must produce [batch_size, 1] logits."""
        batch_size = 3
        seq_len = 16
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=self.device)
        logits = self.model(input_ids)

        self.assertEqual(logits.shape, (batch_size, 1))
        self.assertFalse(torch.isnan(logits).any())

    def test_exceeding_max_sequence_length(self) -> None:
        """Passing sequence longer than max_seq_len must raise ValueError."""
        long_seq_len = self.config.max_seq_len + 10
        input_ids = torch.randint(0, self.config.vocab_size, (2, long_seq_len), device=self.device)
        with self.assertRaises(ValueError):
            self.model(input_ids)

    def test_attention_mask_and_token_types(self) -> None:
        """Verify forward pass with explicit padding mask and segment IDs."""
        batch_size = 2
        seq_len = 20
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=self.device)
        mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=self.device)
        mask[:, 15:] = 0  # Padding
        token_types = torch.zeros((batch_size, seq_len), dtype=torch.long, device=self.device)
        token_types[:, 10:] = 1  # Passage tokens

        logits = self.model(input_ids, attention_mask=mask, token_type_ids=token_types)
        self.assertEqual(logits.shape, (batch_size, 1))

    def test_score_sigmoid_bounds(self) -> None:
        """score() must produce probabilities bounded in [0.0, 1.0]."""
        input_ids = torch.randint(0, self.config.vocab_size, (4, 16), device=self.device)
        scores = self.model.score(input_ids)

        self.assertEqual(scores.shape, (4, 1))
        self.assertTrue((scores >= 0.0).all())
        self.assertTrue((scores <= 1.0).all())

    def test_backward_gradient_computation(self) -> None:
        """Verify backpropagation produces valid non-zero gradients."""
        input_ids = torch.randint(0, self.config.vocab_size, (2, 16), device=self.device)
        target = torch.tensor([[1.0], [0.0]], device=self.device)

        logits = self.model(input_ids)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()

        # Ensure embedding and scorer head have computed gradients
        self.assertIsNotNone(self.model.token_embeddings.weight.grad)
        self.assertIsNotNone(self.model.scorer_head[0].weight.grad)
        self.assertFalse(torch.isnan(self.model.scorer_head[0].weight.grad).any())


if __name__ == "__main__":
    unittest.main()
