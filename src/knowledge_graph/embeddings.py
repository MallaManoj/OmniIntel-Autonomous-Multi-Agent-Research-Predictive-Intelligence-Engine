"""Dense embedding generation engine with L2 vector normalization.

Maps text passages into fixed-dimensional continuous vector representations.
Enforces strict L2 normalization so that Cosine Similarity simplifies to a fast
vector dot product, enabling SIMD and BLAS matrix multiplication acceleration.
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.knowledge_graph.chunker import DocumentChunk
from src.models.dataset import SimpleTokenizer
from src.models.device import get_device
from src.models.reranker import TransformerEncoderBlock


class DenseEmbeddingModel(nn.Module):
    """PyTorch Neural Network for producing dense text embeddings with Mean Pooling.

    Architecture:
      Token + Position Embeddings -> Transformer Encoder -> Masked Mean Pooling -> Dense Projection -> L2 Normalization.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        max_seq_len: int = 64,
        embedding_dim: int = 64,
        num_heads: int = 2,
        dim_feedforward: int = 128,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.encoder = TransformerEncoderBlock(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
        )
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            input_ids: Tensor [batch_size, seq_len]
            attention_mask: Tensor [batch_size, seq_len] (1 = real token, 0 = pad)

        Returns:
            torch.Tensor [batch_size, embedding_dim] of unit-length L2-normalized embeddings.
        """
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.layer_norm(x)

        # MultiHeadAttention expects key_padding_mask where True = padding
        key_padding_mask = attention_mask == 0
        hidden_states = self.encoder(x, key_padding_mask=key_padding_mask)

        # Masked Mean Pooling: average hidden states over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        projected = self.projection(mean_pooled)

        # L2 Normalization: ensures ||v||_2 = 1.0
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized


class EmbeddingEngine:
    """Production serving engine for generating normalized dense embeddings.

    Converts raw text or DocumentChunk instances into unit-norm numpy vectors.
    """

    def __init__(
        self,
        model: Optional[DenseEmbeddingModel] = None,
        tokenizer: Optional[SimpleTokenizer] = None,
        embedding_dim: int = 64,
        max_seq_len: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or get_device()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer or SimpleTokenizer()

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = DenseEmbeddingModel(
                vocab_size=max(1000, self.tokenizer.vocab_size),
                max_seq_len=self.max_seq_len,
                embedding_dim=self.embedding_dim,
            ).to(self.device)

        self.model.eval()

        # Telemetry
        self.total_embeddings_generated: int = 0
        self.total_compute_time_ms: float = 0.0

    def embed_text(self, text: str) -> np.ndarray:
        """Generate a single unit-length dense embedding vector for a string.

        Returns:
            np.ndarray of shape [embedding_dim] with L2 norm = 1.0.
        """
        vectors = self.embed_batch([text])
        return vectors[0]

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Batch process texts into normalized dense embedding vectors.

        Returns:
            np.ndarray of shape [N, embedding_dim].
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        start_time = time.perf_counter()
        all_embeddings: List[np.ndarray] = []

        for b_start in range(0, len(texts), batch_size):
            b_texts = texts[b_start : b_start + batch_size]
            tokenized_seqs: List[List[int]] = []

            for text in b_texts:
                tokens = [SimpleTokenizer.CLS_ID] + self.tokenizer.tokenize_text(text)[: self.max_seq_len - 2] + [SimpleTokenizer.SEP_ID]
                tokenized_seqs.append(tokens)

            # Dynamic padding within batch
            batch_max_len = max(len(s) for s in tokenized_seqs)
            b_ids = torch.full((len(b_texts), batch_max_len), fill_value=SimpleTokenizer.PAD_ID, dtype=torch.long)
            b_mask = torch.zeros((len(b_texts), batch_max_len), dtype=torch.long)

            for i, seq in enumerate(tokenized_seqs):
                b_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                b_mask[i, : len(seq)] = 1

            b_ids = b_ids.to(self.device)
            b_mask = b_mask.to(self.device)

            with torch.inference_mode():
                norm_embeds = self.model(b_ids, b_mask)
                all_embeddings.append(norm_embeds.cpu().numpy())

        res = np.vstack(all_embeddings)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.total_embeddings_generated += len(texts)
        self.total_compute_time_ms += elapsed_ms

        return res

    def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 16,
    ) -> List[Tuple[DocumentChunk, np.ndarray]]:
        """Encode a collection of DocumentChunks into (chunk, embedding) pairs."""
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_batch(texts, batch_size=batch_size)

        return [(chunk, embeddings[i]) for i, chunk in enumerate(chunks)]

    @property
    def telemetry(self) -> Dict[str, Any]:
        """Return runtime statistics."""
        avg_ms = (
            self.total_compute_time_ms / self.total_embeddings_generated
            if self.total_embeddings_generated > 0
            else 0.0
        )
        return {
            "total_embeddings_generated": self.total_embeddings_generated,
            "embedding_dimension": self.embedding_dim,
            "average_latency_per_text_ms": round(avg_ms, 3),
        }
