"""Custom PyTorch Cross-Encoder Neural Network Architecture for semantic reranking.

Processes concatenated query-document pairs with full cross-attention to capture
token-level interactions that standard dual/bi-encoders miss.
Subclasses torch.nn.Module with modular transformer blocks and a classification head.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEncoderConfig:
    """Hyperparameter configuration for CrossEncoderReranker."""

    def __init__(
        self,
        vocab_size: int = 30000,
        max_seq_len: int = 128,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout


class TransformerEncoderBlock(nn.Module):
    """Bidirectional Self-Attention Block with Pre-LayerNorm and residual connections."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        # Pre-LayerNorm architecture (more stable training dynamics than post-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:

        x: Tensor of shape [batch_size, seq_len, d_model]
        key_padding_mask: BoolTensor [batch_size, seq_len] where True indicates padding tokens.
        """
        # Multi-Head Self-Attention with residual connection
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attn_out)

        # Feed-Forward with residual connection
        x = x + self.ffn(self.norm2(x))
        return x


class CrossEncoderReranker(nn.Module):
    """Deep Learning Cross-Encoder model for passage reranking.

    Input shape:
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len] (1 = real token, 0 = padding)
        token_type_ids: Optional [batch_size, seq_len] (0 = query, 1 = passage)

    Output shape:
        logits: [batch_size, 1] (Raw relevance score logit)
    """

    def __init__(self, config: Optional[CrossEncoderConfig] = None) -> None:
        super().__init__()
        self.config = config or CrossEncoderConfig()

        # Token, Position, and Segment Embeddings
        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.position_embeddings = nn.Embedding(self.config.max_seq_len, self.config.embedding_dim)
        self.segment_embeddings = nn.Embedding(2, self.config.embedding_dim)

        self.embedding_norm = nn.LayerNorm(self.config.embedding_dim)
        self.dropout = nn.Dropout(self.config.dropout)

        # Stacked Transformer Encoder Layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=self.config.embedding_dim,
                nhead=self.config.num_heads,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(self.config.embedding_dim)

        # Cross-Encoder Classification Head (MLP)
        self.scorer_head = nn.Sequential(
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim // 2),
            nn.LayerNorm(self.config.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.embedding_dim // 2, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with Truncated Normal distribution (BERT standard)."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @property
    def total_parameters(self) -> int:
        """Return total number of parameters in the network."""
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute forward pass.

        Args:
            input_ids: Tensor [batch_size, seq_len] of token IDs.
            attention_mask: Tensor [batch_size, seq_len] with 1 for tokens, 0 for pad.
            token_type_ids: Tensor [batch_size, seq_len] with 0 for query, 1 for doc.

        Returns:
            torch.Tensor of shape [batch_size, 1] containing relevance logits.
        """
        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        # Position indices: [0, 1, 2, ..., seq_len - 1]
        positions = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Default segment IDs to 0 if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Combine embeddings: token + positional + segment
        x = (
            self.token_embeddings(input_ids)
            + self.position_embeddings(positions)
            + self.segment_embeddings(token_type_ids)
        )
        x = self.dropout(self.embedding_norm(x))

        # Convert attention_mask (1=valid, 0=pad) to key_padding_mask (True=pad, False=valid)
        key_padding_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        # Pass through Transformer encoder blocks
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)

        # Pooling: Extract representation of the first token ([CLS] token at index 0)
        cls_representation = x[:, 0, :]  # Shape: [batch_size, embedding_dim]

        # Pass through regression head to obtain scalar relevance logit
        logits = self.scorer_head(cls_representation)  # Shape: [batch_size, 1]
        return logits

    def score(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute normalized relevance probability in range [0.0, 1.0] via sigmoid."""
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, token_type_ids)
            return torch.sigmoid(logits)
