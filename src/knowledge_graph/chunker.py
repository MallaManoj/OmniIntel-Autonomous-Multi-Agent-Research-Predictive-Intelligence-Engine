"""Semantic text chunking and sliding window document ingestion.

Splits unstructured documents into coherent chunks respecting sentence boundaries,
maintains sliding window overlap to preserve semantic context across chunk edges,
and tracks character offsets for source citation and provenance.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DocumentChunk:
    """An atomic chunk of text extracted from a parent document.

    Attributes:
        chunk_id: Unique identifier for this chunk (e.g. 'doc_123_chunk_002').
        doc_id: Identifier of the source document.
        chunk_index: Zero-indexed position of this chunk within the document.
        text: The text content of the chunk.
        start_char: Starting character offset in the original document.
        end_char: Ending character offset in the original document.
        word_count: Total word count in this chunk.
        metadata: Optional dictionary with parent document metadata (author, title, etc.).
    """

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    word_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Production-grade semantic text chunker with sentence-boundary preservation.

    Complexity:
        Time Complexity: O(N) where N is total characters in document.
        Space Complexity: O(N) for sentence tokens and chunks.
    """

    def __init__(
        self,
        chunk_size: int = 100,
        chunk_overlap: int = 20,
        min_chunk_size: int = 10,
    ) -> None:
        """Args:

        chunk_size: Target maximum words per chunk.
        chunk_overlap: Number of overlapping words between consecutive chunks.
        min_chunk_size: Minimum word count threshold below which trailing chunks are merged.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Regex for sentence boundary detection: matches ., !, ?, or paragraph breaks
        self._sentence_regex = re.compile(r"(?<=[.!?])\s+|\n{2,}")

    def _split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences while tracking exact character offsets.

        Returns:
            List of tuples: (sentence_text, start_char, end_char)
        """
        if not text:
            return []

        sentences: List[Tuple[str, int, int]] = []
        last_end = 0

        # Find boundaries using regex splits
        for match in self._sentence_regex.finditer(text):
            sent_end = match.start()
            sent_text = text[last_end:sent_end].strip()

            if sent_text:
                # Find exact start after leading whitespace
                actual_start = text.find(sent_text, last_end)
                actual_end = actual_start + len(sent_text)
                sentences.append((sent_text, actual_start, actual_end))

            last_end = match.end()

        # Final trailing sentence
        if last_end < len(text):
            trailing_text = text[last_end:].strip()
            if trailing_text:
                actual_start = text.find(trailing_text, last_end)
                actual_end = actual_start + len(trailing_text)
                sentences.append((trailing_text, actual_start, actual_end))

        # Fallback if regex found no sentence boundaries
        if not sentences and text.strip():
            stripped = text.strip()
            s = text.find(stripped)
            sentences.append((stripped, s, s + len(stripped)))

        return sentences

    def chunk_text(
        self,
        text: str,
        doc_id: str = "doc_0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Chunk a single text string into sliding-window DocumentChunk instances.

        Respects sentence boundaries: sentences are never cut in half.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        meta = metadata or {}
        chunks: List[DocumentChunk] = []

        current_sentences: List[Tuple[str, int, int]] = []
        current_word_count = 0

        for sent_text, s_start, s_end in sentences:
            sent_words = len(sent_text.split())

            # If adding this sentence exceeds chunk_size and we already have content
            if current_word_count + sent_words > self.chunk_size and current_sentences:
                # Create and finalize current chunk
                chunk_str = " ".join(s[0] for s in current_sentences)
                chunk_start = current_sentences[0][1]
                chunk_end = current_sentences[-1][2]

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}_chunk_{len(chunks):03d}",
                        doc_id=doc_id,
                        chunk_index=len(chunks),
                        text=chunk_str,
                        start_char=chunk_start,
                        end_char=chunk_end,
                        word_count=current_word_count,
                        metadata=meta,
                    )
                )

                # Slide window back by chunk_overlap words
                overlap_sentences: List[Tuple[str, int, int]] = []
                overlap_words = 0

                for s_item in reversed(current_sentences):
                    words_in_s = len(s_item[0].split())
                    if overlap_words + words_in_s <= self.chunk_overlap or not overlap_sentences:
                        overlap_sentences.insert(0, s_item)
                        overlap_words += words_in_s
                    else:
                        break

                current_sentences = overlap_sentences
                current_word_count = overlap_words

            current_sentences.append((sent_text, s_start, s_end))
            current_word_count += sent_words

        # Final trailing chunk
        if current_sentences:
            chunk_str = " ".join(s[0] for s in current_sentences)
            chunk_start = current_sentences[0][1]
            chunk_end = current_sentences[-1][2]

            # If the trailing chunk is too small and we already have chunks, merge with last chunk
            if current_word_count < self.min_chunk_size and chunks:
                prev_chunk = chunks[-1]
                merged_text = prev_chunk.text + " " + chunk_str
                prev_chunk.text = merged_text
                prev_chunk.end_char = chunk_end
                prev_chunk.word_count = len(merged_text.split())
            else:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}_chunk_{len(chunks):03d}",
                        doc_id=doc_id,
                        chunk_index=len(chunks),
                        text=chunk_str,
                        start_char=chunk_start,
                        end_char=chunk_end,
                        word_count=current_word_count,
                        metadata=meta,
                    )
                )

        return chunks

    def chunk_documents(
        self,
        documents: Dict[str, str],
        metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[DocumentChunk]:
        """Process a collection of documents mapping doc_id -> raw_text."""
        all_chunks: List[DocumentChunk] = []
        for doc_id, text in documents.items():
            meta = metadata_map.get(doc_id) if metadata_map else None
            chunks = self.chunk_text(text, doc_id=doc_id, metadata=meta)
            all_chunks.extend(chunks)
        return all_chunks
