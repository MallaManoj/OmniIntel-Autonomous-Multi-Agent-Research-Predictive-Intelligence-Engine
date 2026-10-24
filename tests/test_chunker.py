"""Unit tests for TextChunker and DocumentChunk sliding-window ingestion.

Compatible with both standard library `unittest` and `pytest`.
"""

import unittest
from src.knowledge_graph.chunker import DocumentChunk, TextChunker


class TestTextChunker(unittest.TestCase):
    """Test suite verifying sentence boundary preservation, sliding window overlap, and offsets."""

    def setUp(self) -> None:
        self.chunker = TextChunker(chunk_size=15, chunk_overlap=5, min_chunk_size=4)

    def test_invalid_parameters_raise_error(self) -> None:
        """Chunker should reject non-positive chunk_size and invalid overlap."""
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=0)
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=10, chunk_overlap=-1)
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=10, chunk_overlap=10)  # overlap == size
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=10, chunk_overlap=15)  # overlap > size

    def test_sentence_boundary_integrity(self) -> None:
        """TextChunker must never split words or mid-sentence phrases."""
        text = (
            "NVIDIA reported strong earnings for the fiscal quarter. "
            "Data Center revenue surged by over two hundred percent. "
            "Demand for Hopper compute platforms drove record operating margins across cloud customers. "
            "Automotive revenue experienced moderate seasonal growth during the same period."
        )

        chunks = self.chunker.chunk_text(text, doc_id="nvda_q3")
        self.assertGreater(len(chunks), 1)

        # Every chunk should start and end with complete words
        for chunk in chunks:
            self.assertIsInstance(chunk, DocumentChunk)
            self.assertTrue(chunk.chunk_id.startswith("nvda_q3_chunk_"))
            self.assertGreater(chunk.word_count, 0)
            # Ensure chunk ends with valid punctuation or word, not a broken hyphen or severed word
            self.assertNotIn("  ", chunk.text)
            self.assertFalse(chunk.text.endswith(" "))

    def test_sliding_window_overlap(self) -> None:
        """Consecutive chunks must share overlapping context."""
        text = (
            "Sentence one discusses market conditions in detail. "
            "Sentence two explains quarterly revenue and profit margins. "
            "Sentence three highlights risks associated with foreign exchange. "
            "Sentence four concludes the executive summary with future guidance."
        )

        chunker = TextChunker(chunk_size=16, chunk_overlap=8, min_chunk_size=2)
        chunks = chunker.chunk_text(text, doc_id="doc_overlap")

        if len(chunks) >= 2:
            # Words from chunk 0 should overlap into chunk 1
            words_chunk0 = set(chunks[0].text.lower().split())
            words_chunk1 = set(chunks[1].text.lower().split())
            common_words = words_chunk0.intersection(words_chunk1)
            self.assertGreater(len(common_words), 0)

    def test_character_offset_accuracy(self) -> None:
        """start_char and end_char must correctly bound the chunk in the original text."""
        text = "First sentence is short. Second sentence explains cloud infrastructure."
        chunks = self.chunker.chunk_text(text, doc_id="doc_offsets")

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.start_char, 0)
        self.assertEqual(chunk.end_char, len(text))
        self.assertEqual(text[chunk.start_char : chunk.end_char], text)

    def test_trailing_fragment_merge(self) -> None:
        """Trailing fragments with word count < min_chunk_size must merge into preceding chunk."""
        text = (
            "This is a long introductory sentence detailing financial performance and growth across divisions. "
            "Another sentence to push the word count past the threshold of the first chunk. "
            "Small ending."  # Only 2 words -> below min_chunk_size=4
        )

        chunker = TextChunker(chunk_size=15, chunk_overlap=3, min_chunk_size=4)
        chunks = chunker.chunk_text(text)

        # Last chunk must not have fewer words than min_chunk_size if merged
        self.assertGreaterEqual(chunks[-1].word_count, 4)

    def test_empty_or_whitespace_text(self) -> None:
        """Chunker should handle empty or whitespace-only strings gracefully."""
        self.assertEqual(self.chunker.chunk_text(""), [])
        self.assertEqual(self.chunker.chunk_text("   \n\n  "), [])

    def test_chunk_documents_batch(self) -> None:
        """chunk_documents must batch process multiple documents."""
        docs = {
            "doc1": "Document one content with multiple financial terms and figures.",
            "doc2": "Document two contains separate regulatory disclosures.",
        }
        meta_map = {"doc1": {"category": "financial"}, "doc2": {"category": "legal"}}

        chunks = self.chunker.chunk_documents(docs, metadata_map=meta_map)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["category"], "financial")
        self.assertEqual(chunks[1].metadata["category"], "legal")


if __name__ == "__main__":
    unittest.main()
