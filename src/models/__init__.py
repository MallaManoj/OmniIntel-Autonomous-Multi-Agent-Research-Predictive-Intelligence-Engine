"""Custom PyTorch models, losses, and inference engines."""

from src.models.checkpoint import ModelCheckpointManager
from src.models.dataset import (
    EncodedPair,
    SimpleTokenizer,
    TripletBatch,
    TripletRerankingDataset,
    TripletSample,
    collate_triplet_batch,
)
from src.models.device import DeviceManager, get_device
from src.models.inference import CrossEncoderInferenceEngine, RerankResult
from src.models.losses import (
    InBatchHardNegativeMiner,
    ListNetRankingLoss,
    TripletMarginRankingLoss,
)
from src.models.metrics import (
    RankingEvaluator,
    compute_dcg_at_k,
    compute_hit_rate_at_k,
    compute_ndcg_at_k,
    compute_reciprocal_rank,
)
from src.models.reranker import (
    CrossEncoderConfig,
    CrossEncoderReranker,
    TransformerEncoderBlock,
)
from src.models.trainer import CrossEncoderTrainer, TrainingConfig

__all__ = [
    "DeviceManager",
    "get_device",
    "CrossEncoderConfig",
    "CrossEncoderReranker",
    "TransformerEncoderBlock",
    "TripletMarginRankingLoss",
    "InBatchHardNegativeMiner",
    "ListNetRankingLoss",
    "SimpleTokenizer",
    "TripletSample",
    "EncodedPair",
    "TripletBatch",
    "TripletRerankingDataset",
    "collate_triplet_batch",
    "TrainingConfig",
    "CrossEncoderTrainer",
    "compute_reciprocal_rank",
    "compute_dcg_at_k",
    "compute_ndcg_at_k",
    "compute_hit_rate_at_k",
    "RankingEvaluator",
    "ModelCheckpointManager",
    "CrossEncoderInferenceEngine",
    "RerankResult",
]
