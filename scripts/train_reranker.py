#!/usr/bin/env bash
"""Domain Training and Fine-Tuning Pipeline for OmniIntel Cross-Encoder Reranker.

Generates structured domain training data, fine-tunes the Pre-LN Transformer
using Triplet Margin Loss, evaluates MRR and NDCG@K, and persists model checkpoints.
"""

import os
import sys
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.checkpoint import ModelCheckpointManager
from src.models.dataset import (
    SimpleTokenizer,
    TripletBatch,
    TripletRerankingDataset,
    TripletSample,
    collate_triplet_batch,
)
from src.models.device import DeviceManager, get_device
from src.models.losses import TripletMarginRankingLoss
from src.models.metrics import RankingEvaluator
from src.models.reranker import CrossEncoderConfig, CrossEncoderReranker
from src.models.trainer import CrossEncoderTrainer, TrainingConfig


def get_curated_financial_dataset() -> Tuple[List[TripletSample], List[TripletSample]]:
    """Return realistic domain triplets for enterprise & financial intelligence training.

    Each triplet contains:
      - Query: Target financial search inquiry
      - Positive Doc: Relevant context answering query
      - Negative Doc: Plausible hard negative (same company/topic, but wrong division or metric)
    """
    train_triplets = [
        TripletSample(
            query="nvidia data center revenue growth",
            positive_doc="nvidia reported data center revenue of 14.5 billion up 279 percent year-over-year driven by hgx compute systems",
            negative_doc="nvidia gaming revenue was 2.86 billion reflecting seasonal trends in geforce rtx consumer graphics cards",
        ),
        TripletSample(
            query="apple iphone sales decline",
            positive_doc="apple iphone sales fell 2.4 percent to 45.96 billion due to assembly supply chain delays in asian manufacturing plants",
            negative_doc="apple services revenue reached an all-time record of 22.3 billion up 16 percent annually driven by app store subscriptions",
        ),
        TripletSample(
            query="amd ai accelerator competition",
            positive_doc="amd launched the instinct mi300x accelerator targeting enterprise ai inference and competing directly with nvidia h100",
            negative_doc="amd epyc server cpu shipments grew steadily across enterprise cloud hyperscaler data center deployments",
        ),
        TripletSample(
            query="federal reserve interest rate hike",
            positive_doc="the federal open market committee raised the benchmark federal funds rate by 25 basis points to combat inflation",
            negative_doc="the federal reserve balance sheet expanded during quantitative easing in previous economic monetary cycles",
        ),
        TripletSample(
            query="microsoft azure cloud growth rate",
            positive_doc="microsoft intelligent cloud revenue grew 19 percent led by azure and other cloud services growth of 29 percent",
            negative_doc="microsoft productivity and business processes revenue grew 13 percent driven by office commercial seat expansions",
        ),
        TripletSample(
            query="tesla automotive gross margin price cuts",
            positive_doc="tesla gaap automotive gross margin excluding regulatory credits dropped to 16.3 percent following aggressive global price cuts",
            negative_doc="tesla energy storage deployments increased 125 percent to 4.0 gwh in the quarter with record gross profit",
        ),
        TripletSample(
            query="amazon aws operating profit",
            positive_doc="amazon web services segment operating income surged to 7.0 billion compared to operating income of 5.4 billion year-over-year",
            negative_doc="amazon north america retail operating margin improved by 120 basis points due to regional fulfillment warehouse optimization",
        ),
        TripletSample(
            query="google search advertising revenue",
            positive_doc="google search and other advertising revenues increased to 44.03 billion compared to 39.54 billion in the prior period",
            negative_doc="google cloud segment revenue rose 28 percent to 8.41 billion with accelerating operating margin profitability",
        ),
    ]

    val_triplets = [
        TripletSample(
            query="semiconductor chip fabrication foundry",
            positive_doc="tsmc advanced 3nm and 5nm node wafer fabrication capacity was fully booked by major silicon design customers",
            negative_doc="intel personal computer client computing group revenue fell 12 percent due to inventory adjustments",
        ),
        TripletSample(
            query="inflation consumer price index",
            positive_doc="the consumer price index increased 3.1 percent on an annual basis exceeding consensus economist projections",
            negative_doc="unemployment claims held steady near historic lows as labor market demand remained resilient across sectors",
        ),
    ]

    return train_triplets, val_triplets


def main() -> None:
    print("=" * 65)
    print("      OmniIntel: PyTorch Cross-Encoder Fine-Tuning Pipeline     ")
    print("=" * 65)

    device = get_device()
    info = DeviceManager.get_device_info(device)
    print(f"Device: {info.get('selected_device')} ({info.get('gpu_name', 'CPU')})")
    print(f"PyTorch Version: {info.get('torch_version')}\n")

    # 1. Load domain triplets
    train_triplets, val_triplets = get_curated_financial_dataset()
    print(f"Loaded {len(train_triplets)} training triplets and {len(val_triplets)} validation triplets.")

    # 2. Build domain vocabulary
    all_text = []
    for t in train_triplets + val_triplets:
        all_text.extend([t.query, t.positive_doc, t.negative_doc])

    words = set()
    for text in all_text:
        words.update(text.lower().replace(",", " ").replace(".", " ").split())

    tokenizer = SimpleTokenizer(vocab=sorted(list(words)))
    print(f"Constructed domain vocabulary: {tokenizer.vocab_size} unique tokens.")

    # 3. Create Datasets and DataLoaders
    train_dataset = TripletRerankingDataset(train_triplets, tokenizer, max_seq_len=64)
    val_dataset = TripletRerankingDataset(val_triplets, tokenizer, max_seq_len=64)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_triplet_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_triplet_batch,
    )

    # 4. Initialize Cross-Encoder Architecture
    model_config = CrossEncoderConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )
    model = CrossEncoderReranker(model_config).to(device)
    print(f"Initialized Cross-Encoder: {model.total_parameters:,} parameters.")

    # 5. Pre-training Evaluation (Baseline Zero-Shot Performance)
    evaluator = RankingEvaluator(k_thresholds=(1, 3))
    print("\n--- Evaluating Pre-Training (Untrained Baseline) ---")
    model.eval()
    with torch.no_grad():
        for sample in val_triplets:
            q_pos = tokenizer.encode_pair(sample.query, sample.positive_doc, max_seq_len=64)
            q_neg = tokenizer.encode_pair(sample.query, sample.negative_doc, max_seq_len=64)

            pos_tensor = torch.tensor([q_pos[0]], device=device)
            neg_tensor = torch.tensor([q_neg[0]], device=device)

            pos_score = model(pos_tensor).item()
            neg_score = model(neg_tensor).item()

            evaluator.evaluate_query(
                predicted_scores=[pos_score, neg_score],
                ground_truth_relevance=[1, 0],
            )

    pre_metrics = evaluator.compute_summary()
    print(f"Baseline MRR: {pre_metrics.get('mean_mrr'):.4f} | Baseline NDCG@3: {pre_metrics.get('mean_ndcg@3'):.4f}")

    # 6. Execute Training Loop
    train_config = TrainingConfig(
        learning_rate=5e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        epochs=4,
        warmup_ratio=0.1,
    )
    trainer = CrossEncoderTrainer(
        model=model,
        config=train_config,
        loss_fn=TripletMarginRankingLoss(margin=1.0),
        device=device,
    )

    print("\n--- Starting Training Optimization Loop ---")
    history = trainer.fit(train_loader, val_loader=val_loader)

    for record in history:
        print(
            f"Epoch {int(record['epoch'])}: "
            f"Train Loss = {record['train_loss']:.4f} | "
            f"Val Loss = {record['val_loss']:.4f} | "
            f"Val Accuracy = {record['val_accuracy'] * 100:.1f}% | "
            f"LR = {record['learning_rate']:.6f}"
        )

    # 7. Post-training Evaluation (Trained Performance)
    evaluator.clear()
    model.eval()
    with torch.no_grad():
        for sample in val_triplets:
            q_pos = tokenizer.encode_pair(sample.query, sample.positive_doc, max_seq_len=64)
            q_neg = tokenizer.encode_pair(sample.query, sample.negative_doc, max_seq_len=64)

            pos_tensor = torch.tensor([q_pos[0]], device=device)
            neg_tensor = torch.tensor([q_neg[0]], device=device)

            pos_score = model(pos_tensor).item()
            neg_score = model(neg_tensor).item()

            evaluator.evaluate_query(
                predicted_scores=[pos_score, neg_score],
                ground_truth_relevance=[1, 0],
            )

    post_metrics = evaluator.compute_summary()
    print("\n" + "=" * 65)
    print("             Post-Training Fine-Tuned Performance            ")
    print("=" * 65)
    print(f"Fine-Tuned MRR:      {post_metrics.get('mean_mrr'):.4f}")
    print(f"Fine-Tuned NDCG@3:   {post_metrics.get('mean_ndcg@3'):.4f}")
    print(f"Fine-Tuned HitRate@1:{post_metrics.get('mean_hit_rate@1'):.4f}")

    # 8. Save Checkpoint
    checkpoint_path = os.path.join(train_config.checkpoint_dir, "reranker_best.pt")
    ModelCheckpointManager.save_checkpoint(
        model=model,
        filepath=checkpoint_path,
        tokenizer=tokenizer,
        config=model_config,
        optimizer=trainer.optimizer,
        epoch=trainer.current_epoch,
        metrics=post_metrics,
    )
    print(f"✓ Saved trained model checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
