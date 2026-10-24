"""PyTorch Dataset, dynamic batch collator, and tokenizer for Cross-Encoder training.

Implements TripletRerankingDataset and dynamic padding collation.
Minimizes compute overhead by dynamically padding batches to the longest sequence
in each mini-batch rather than a fixed global maximum length.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class SimpleTokenizer:
    """Lightweight rule-based tokenizer with special tokens and vocabulary mapping.

    Special Token IDs:
        [PAD] = 0
        [UNK] = 1
        [CLS] = 2
        [SEP] = 3
    """

    PAD_ID: int = 0
    UNK_ID: int = 1
    CLS_ID: int = 2
    SEP_ID: int = 3

    def __init__(self, vocab: Optional[List[str]] = None) -> None:
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        self.vocab: List[str] = list(self.special_tokens)

        if vocab:
            for w in vocab:
                if w not in self.vocab:
                    self.vocab.append(w)

        self.token_to_id: dict[str, int] = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token: dict[int, str] = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def add_words(self, words: List[str]) -> None:
        """Expand vocabulary with new domain terms."""
        for word in words:
            w = word.lower().strip()
            if w and w not in self.token_to_id:
                new_idx = len(self.vocab)
                self.vocab.append(w)
                self.token_to_id[w] = new_idx
                self.id_to_token[new_idx] = w

    def tokenize_text(self, text: str) -> List[int]:
        """Convert a text string into list of token IDs."""
        tokens: List[int] = []
        for word in text.lower().replace(",", " ").replace(".", " ").split():
            tokens.append(self.token_to_id.get(word, self.UNK_ID))
        return tokens

    def encode_pair(
        self,
        query: str,
        document: str,
        max_seq_len: int = 128,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Encode a query-document pair into Cross-Encoder format:

        [CLS] query_tokens [SEP] doc_tokens [SEP]

        Returns:
            Tuple of (input_ids, attention_mask, token_type_ids)
        """
        q_tokens = self.tokenize_text(query)
        d_tokens = self.tokenize_text(document)

        # Budget: max_seq_len minus 3 special tokens ([CLS], [SEP], [SEP])
        available_budget = max_seq_len - 3
        if len(q_tokens) + len(d_tokens) > available_budget:
            # Preserve query, truncate document if needed
            max_q = min(len(q_tokens), available_budget // 2)
            max_d = available_budget - max_q
            q_tokens = q_tokens[:max_q]
            d_tokens = d_tokens[:max_d]

        input_ids = [self.CLS_ID] + q_tokens + [self.SEP_ID] + d_tokens + [self.SEP_ID]
        # Query tokens = segment 0, Doc tokens = segment 1
        token_type_ids = [0] * (len(q_tokens) + 2) + [1] * (len(d_tokens) + 1)
        attention_mask = [1] * len(input_ids)

        return input_ids, attention_mask, token_type_ids


@dataclass
class TripletSample:
    """Single training sample containing a query, relevant document, and negative document."""

    query: str
    positive_doc: str
    negative_doc: str


@dataclass
class EncodedPair:
    """Tensor-ready token representation for a query-document sequence."""

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]


@dataclass
class TripletBatch:
    """Collation container for a mini-batch of positive and negative pairs."""

    pos_input_ids: torch.Tensor
    pos_attention_mask: torch.Tensor
    pos_token_type_ids: torch.Tensor
    neg_input_ids: torch.Tensor
    neg_attention_mask: torch.Tensor
    neg_token_type_ids: torch.Tensor

    def to(self, device: torch.device) -> "TripletBatch":
        """Move all batch tensors to target compute device."""
        return TripletBatch(
            pos_input_ids=self.pos_input_ids.to(device),
            pos_attention_mask=self.pos_attention_mask.to(device),
            pos_token_type_ids=self.pos_token_type_ids.to(device),
            neg_input_ids=self.neg_input_ids.to(device),
            neg_attention_mask=self.neg_attention_mask.to(device),
            neg_token_type_ids=self.neg_token_type_ids.to(device),
        )


class TripletRerankingDataset(Dataset):
    """PyTorch Dataset managing query-positive-negative triplets for ranking models."""

    def __init__(
        self,
        triplets: List[TripletSample],
        tokenizer: SimpleTokenizer,
        max_seq_len: int = 128,
    ) -> None:
        self.triplets: List[TripletSample] = triplets
        self.tokenizer: SimpleTokenizer = tokenizer
        self.max_seq_len: int = max_seq_len

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[EncodedPair, EncodedPair]:
        item = self.triplets[idx]

        # Encode Positive Pair: (query, positive_doc)
        pos_ids, pos_mask, pos_types = self.tokenizer.encode_pair(
            query=item.query,
            document=item.positive_doc,
            max_seq_len=self.max_seq_len,
        )

        # Encode Negative Pair: (query, negative_doc)
        neg_ids, neg_mask, neg_types = self.tokenizer.encode_pair(
            query=item.query,
            document=item.negative_doc,
            max_seq_len=self.max_seq_len,
        )

        return (
            EncodedPair(pos_ids, pos_mask, pos_types),
            EncodedPair(neg_ids, neg_mask, neg_types),
        )


def collate_triplet_batch(
    batch: List[Tuple[EncodedPair, EncodedPair]],
    pad_id: int = SimpleTokenizer.PAD_ID,
) -> TripletBatch:
    """Collate function that dynamically pads sequences to the maximum length in this mini-batch.

    Avoids padding to global max_seq_len, reducing Transformer quadratic attention compute.
    """
    pos_pairs = [item[0] for item in batch]
    neg_pairs = [item[1] for item in batch]

    def _pad_encoded_pairs(pairs: List[EncodedPair]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(pairs)
        max_len = max(len(p.input_ids) for p in pairs)

        input_ids = torch.full((batch_size, max_len), fill_value=pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        token_type_ids = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, pair in enumerate(pairs):
            seq_len = len(pair.input_ids)
            input_ids[i, :seq_len] = torch.tensor(pair.input_ids, dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(pair.attention_mask, dtype=torch.long)
            token_type_ids[i, :seq_len] = torch.tensor(pair.token_type_ids, dtype=torch.long)

        return input_ids, attention_mask, token_type_ids

    pos_ids, pos_mask, pos_types = _pad_encoded_pairs(pos_pairs)
    neg_ids, neg_mask, neg_types = _pad_encoded_pairs(neg_pairs)

    return TripletBatch(
        pos_input_ids=pos_ids,
        pos_attention_mask=pos_mask,
        pos_token_type_ids=pos_types,
        neg_input_ids=neg_ids,
        neg_attention_mask=neg_mask,
        neg_token_type_ids=neg_types,
    )
