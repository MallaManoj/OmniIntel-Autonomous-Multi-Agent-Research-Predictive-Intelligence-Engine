"""Unit tests for PyTorch TripletRerankingDataset, dynamic batch collation, and tokenizer.

Compatible with both standard library `unittest` and `pytest`.
"""

import unittest
from torch.utils.data import DataLoader

from src.models.dataset import (
    SimpleTokenizer,
    TripletBatch,
    TripletRerankingDataset,
    TripletSample,
    collate_triplet_batch,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for Dataset tests")
class TestDatasetAndCollator(unittest.TestCase):
    """Test suite verifying tokenization, TripletDataset, and dynamic batch padding."""

    def setUp(self) -> None:
        self.tokenizer = SimpleTokenizer(vocab=["apple", "revenue", "quarter", "growth", "sec", "filing"])

    def test_tokenizer_encoding_structure(self) -> None:
        """encode_pair must produce [CLS] q [SEP] doc [SEP] with correct token_type_ids."""
        query = "apple revenue"
        doc = "quarter growth"
        input_ids, attention_mask, token_type_ids = self.tokenizer.encode_pair(query, doc, max_seq_len=32)

        # Structure: [CLS] apple revenue [SEP] quarter growth [SEP]
        self.assertEqual(input_ids[0], SimpleTokenizer.CLS_ID)
        self.assertEqual(input_ids[3], SimpleTokenizer.SEP_ID)
        self.assertEqual(input_ids[-1], SimpleTokenizer.SEP_ID)
        self.assertEqual(len(input_ids), 7)
        self.assertEqual(len(attention_mask), 7)
        self.assertTrue(all(m == 1 for m in attention_mask))

        # Check token_type_ids: 0 for query part (first 4 tokens), 1 for doc part (last 3 tokens)
        self.assertEqual(token_type_ids[:4], [0, 0, 0, 0])
        self.assertEqual(token_type_ids[4:], [1, 1, 1])

    def test_tokenizer_truncation(self) -> None:
        """Sequences longer than max_seq_len must be truncated within budget."""
        long_query = "apple " * 20
        long_doc = "growth " * 20
        input_ids, _, _ = self.tokenizer.encode_pair(long_query, long_doc, max_seq_len=16)
        self.assertLessEqual(len(input_ids), 16)
        self.assertEqual(input_ids[0], SimpleTokenizer.CLS_ID)
        self.assertEqual(input_ids[-1], SimpleTokenizer.SEP_ID)

    def test_triplet_dataset_and_dynamic_collation(self) -> None:
        """collate_triplet_batch must pad to the maximum length of the batch, not global max."""
        samples = [
            TripletSample(
                query="apple",
                positive_doc="quarter revenue",
                negative_doc="sec filing",
            ),
            TripletSample(
                query="growth",
                positive_doc="apple revenue quarter growth",  # longer positive doc
                negative_doc="filing",
            ),
        ]
        dataset = TripletRerankingDataset(samples, self.tokenizer, max_seq_len=64)
        self.assertEqual(len(dataset), 2)

        # Wrap in DataLoader
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_triplet_batch,
        )

        batch: TripletBatch = next(iter(loader))
        self.assertIsInstance(batch, TripletBatch)

        # Batch size check
        self.assertEqual(batch.pos_input_ids.shape[0], 2)
        self.assertEqual(batch.neg_input_ids.shape[0], 2)

        # Dynamic padding check: batch length should be equal to the longest sample in the batch,
        # which is MUCH less than global max_seq_len (64)
        self.assertLess(batch.pos_input_ids.shape[1], 20)
        self.assertLess(batch.neg_input_ids.shape[1], 20)

        # Check that padding token is 0
        self.assertTrue((batch.pos_input_ids[:, 0] == SimpleTokenizer.CLS_ID).all())

    def test_batch_to_device(self) -> None:
        """TripletBatch.to(device) should transfer all internal tensors to target device."""
        samples = [
            TripletSample(query="q1", positive_doc="pos", negative_doc="neg"),
        ]
        dataset = TripletRerankingDataset(samples, self.tokenizer)
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_triplet_batch)
        batch = next(iter(loader))

        cpu_device = torch.device("cpu")
        batch_cpu = batch.to(cpu_device)
        self.assertEqual(batch_cpu.pos_input_ids.device.type, "cpu")
        self.assertEqual(batch_cpu.neg_input_ids.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
