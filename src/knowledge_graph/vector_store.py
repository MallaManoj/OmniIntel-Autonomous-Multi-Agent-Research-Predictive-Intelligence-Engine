"""In-Memory Vector Store with Cosine Similarity and Min-Heap Top-K Retrieval.

Stores dense L2-normalized vector embeddings alongside document metadata.
Executes similarity scoring via BLAS-accelerated matrix-vector multiplication
and extracts top-K nearest neighbors using an O(N log K) bounded Min-Heap.
"""

from dataclasses import dataclass, field
import heapq
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class VectorRecord:
    """Represents a stored vector entry with unique ID and payload metadata."""

    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.asarray(self.vector, dtype=np.float32)
        else:
            self.vector = self.vector.astype(np.float32, copy=False)

        if self.vector.ndim != 1:
            raise ValueError(f"Vector for id '{self.id}' must be 1-dimensional, got shape {self.vector.shape}")


@dataclass
class VectorSearchResult:
    """Result of a top-K nearest-neighbor similarity search."""

    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[np.ndarray] = None


class VectorStore:
    """High-performance in-memory vector index with bounded Min-Heap top-K retrieval.

    Complexity:
      - Add / Upsert: O(1) amortized
      - Lookup / Delete: O(1)
      - Similarity Scoring: O(N * D) via SIMD/BLAS matrix-vector product
      - Top-K Selection: O(N log K) using a size-K Min-Heap (vs O(N log N) full sort)
    """

    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
    ) -> None:
        """Args:
            dimension: Fixed length of embedding vectors.
            metric: Similarity metric ("cosine", "dot_product", or "euclidean").
        """
        if dimension <= 0:
            raise ValueError(f"Vector dimension must be positive, got {dimension}")
        if metric not in ("cosine", "dot_product", "euclidean"):
            raise ValueError(f"Unsupported metric '{metric}'. Choose 'cosine', 'dot_product', or 'euclidean'.")

        self.dimension = dimension
        self.metric = metric

        # Storage
        self._records: Dict[str, VectorRecord] = {}
        self._matrix: Optional[np.ndarray] = None
        self._row_to_id: List[str] = []
        self._id_to_row: Dict[str, int] = {}
        self._is_dirty: bool = False

        # Telemetry
        self._total_searches: int = 0
        self._total_search_time_ms: float = 0.0

    def __len__(self) -> int:
        return len(self._records)

    @property
    def count(self) -> int:
        """Total number of stored vectors."""
        return len(self._records)

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """Enforce L2 unit-norm constraint ||v||_2 = 1.0."""
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return vec
        return vec / norm

    def add(
        self,
        id: str,
        vector: Union[np.ndarray, List[float]],
        metadata: Optional[Dict[str, Any]] = None,
        normalize: bool = True,
    ) -> None:
        """Insert or update a single vector record.

        Args:
            id: Unique string identifier.
            vector: 1D array of length equal to self.dimension.
            metadata: Associated metadata dictionary.
            normalize: If True and metric is cosine, normalize to unit length.
        """
        if not id or not isinstance(id, str):
            raise ValueError("Record ID must be a non-empty string.")

        vec = np.asarray(vector, dtype=np.float32)
        if vec.shape != (self.dimension,):
            raise ValueError(
                f"Vector shape mismatch: expected ({self.dimension},), got {vec.shape}"
            )

        if normalize and self.metric == "cosine":
            vec = self._normalize_vector(vec)

        self._records[id] = VectorRecord(id=id, vector=vec, metadata=metadata or {})
        self._is_dirty = True

    def add_batch(
        self,
        ids: List[str],
        vectors: Union[np.ndarray, List[List[float]]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        normalize: bool = True,
    ) -> None:
        """Insert a batch of vector records in bulk.

        Args:
            ids: List of unique string identifiers.
            vectors: 2D array of shape [N, dimension].
            metadatas: Optional list of metadata dicts corresponding to ids.
            normalize: If True and metric is cosine, normalize all rows to unit length.
        """
        if len(ids) == 0:
            return

        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dimension:
            raise ValueError(
                f"Vectors shape mismatch: expected (N, {self.dimension}), got {vecs.shape}"
            )
        if len(ids) != vecs.shape[0]:
            raise ValueError(
                f"Number of IDs ({len(ids)}) does not match number of vectors ({vecs.shape[0]})"
            )

        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError(
                f"Number of metadatas ({len(metadatas)}) does not match number of IDs ({len(ids)})"
            )

        if normalize and self.metric == "cosine":
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-9)
            vecs = vecs / norms

        for i, doc_id in enumerate(ids):
            if not doc_id or not isinstance(doc_id, str):
                raise ValueError(f"Record ID at index {i} must be a non-empty string.")
            meta = metadatas[i] if metadatas is not None else {}
            self._records[doc_id] = VectorRecord(id=doc_id, vector=vecs[i], metadata=meta)

        self._is_dirty = True

    def get(self, id: str) -> Optional[VectorRecord]:
        """Retrieve a record by ID in O(1) time."""
        return self._records.get(id)

    def delete(self, id: str) -> bool:
        """Remove a record by ID.

        Returns True if removed, False if not found.
        """
        if id in self._records:
            del self._records[id]
            self._is_dirty = True
            return True
        return False

    def clear(self) -> None:
        """Clear all stored vectors and reset memory structures."""
        self._records.clear()
        self._matrix = None
        self._row_to_id.clear()
        self._id_to_row.clear()
        self._is_dirty = False

    def _sync_matrix(self) -> None:
        """Reconstruct contiguous 2D NumPy matrix from active records for fast vectorization."""
        if not self._is_dirty and self._matrix is not None:
            return

        n = len(self._records)
        if n == 0:
            self._matrix = np.empty((0, self.dimension), dtype=np.float32)
            self._row_to_id = []
            self._id_to_row = {}
            self._is_dirty = False
            return

        self._row_to_id = list(self._records.keys())
        self._id_to_row = {doc_id: i for i, doc_id in enumerate(self._row_to_id)}
        self._matrix = np.empty((n, self.dimension), dtype=np.float32)

        for i, doc_id in enumerate(self._row_to_id):
            self._matrix[i] = self._records[doc_id].vector

        self._is_dirty = False

    def search(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        return_vectors: bool = False,
    ) -> List[VectorSearchResult]:
        """Retrieve the top-K nearest neighbors using BLAS dot products and a Min-Heap.

        Algorithm:
          1. BLAS GEMV: Computes dot products for all N items in O(N * D).
          2. Min-Heap Filter: Maintains a bounded heap of size K. For each candidate:
             - If heap size < K: push (score, id, metadata).
             - If score > heap[0].score: heapreplace (pop smallest and insert candidate) in O(log K).
             Overall selection complexity: O(N log K), avoiding an O(N log N) full sort.

        Args:
            query_vector: 1D query array of length self.dimension.
            top_k: Number of highest-scoring results to return.
            score_threshold: Minimum similarity score required to qualify.
            filter_fn: Optional predicate taking record metadata dict and returning bool.
            return_vectors: If True, attaches the stored vector to each result.

        Returns:
            List of VectorSearchResult instances ordered by score descending.
        """
        if top_k <= 0:
            return []

        start_time = time.perf_counter()

        q = np.asarray(query_vector, dtype=np.float32)
        if q.shape != (self.dimension,):
            raise ValueError(
                f"Query vector shape mismatch: expected ({self.dimension},), got {q.shape}"
            )

        if self.count == 0:
            return []

        self._sync_matrix()
        assert self._matrix is not None

        if self.metric == "cosine":
            q = self._normalize_vector(q)
            # Cosine similarity on unit vectors simplifies to dot product: S = M @ q
            scores = self._matrix.dot(q)
        elif self.metric == "dot_product":
            scores = self._matrix.dot(q)
        elif self.metric == "euclidean":
            # Negative Euclidean distance so higher is closer
            dists = np.linalg.norm(self._matrix - q, axis=1)
            scores = -dists
        else:
            raise ValueError(f"Unknown metric {self.metric}")

        # Top-K selection using a bounded Min-Heap of capacity K
        # Heap elements: (score, insertion_index, row_index)
        heap: List[Tuple[float, int, int]] = []
        entry_idx = 0

        for row_idx, score_val in enumerate(scores):
            score = float(score_val)

            if score_threshold is not None and score < score_threshold:
                continue

            doc_id = self._row_to_id[row_idx]
            meta = self._records[doc_id].metadata

            if filter_fn is not None and not filter_fn(meta):
                continue

            if len(heap) < top_k:
                heapq.heappush(heap, (score, entry_idx, row_idx))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, entry_idx, row_idx))

            entry_idx += 1

        # Extract elements and sort descending
        results: List[VectorSearchResult] = []
        while heap:
            score, _, row_idx = heapq.heappop(heap)
            doc_id = self._row_to_id[row_idx]
            rec = self._records[doc_id]
            results.append(
                VectorSearchResult(
                    id=doc_id,
                    score=score,
                    metadata=rec.metadata,
                    vector=rec.vector if return_vectors else None,
                )
            )

        # Heap pops smallest first, so reverse to obtain descending order
        results.reverse()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._total_searches += 1
        self._total_search_time_ms += elapsed_ms

        return results

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist index vectors and metadata to disk using compressed NumPy format."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._sync_matrix()
        assert self._matrix is not None

        ids = np.array(self._row_to_id, dtype=object)
        metadatas = [self._records[doc_id].metadata for doc_id in self._row_to_id]
        meta_json = np.array(json.dumps(metadatas), dtype=object)
        header = np.array(
            json.dumps({"dimension": self.dimension, "metric": self.metric}),
            dtype=object,
        )

        np.savez_compressed(
            path,
            matrix=self._matrix,
            ids=ids,
            metadata=meta_json,
            header=header,
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "VectorStore":
        """Load a persisted VectorStore from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"VectorStore index file not found: {path}")

        data = np.load(path, allow_pickle=True)
        header = json.loads(str(data["header"]))
        store = cls(dimension=header["dimension"], metric=header["metric"])

        matrix = data["matrix"]
        ids = data["ids"].tolist()
        metadatas = json.loads(str(data["metadata"]))

        store.add_batch(ids=ids, vectors=matrix, metadatas=metadatas, normalize=False)
        return store

    @property
    def telemetry(self) -> Dict[str, Any]:
        """Return runtime index statistics and search latency."""
        avg_latency = (
            self._total_search_time_ms / self._total_searches
            if self._total_searches > 0
            else 0.0
        )
        return {
            "total_records": self.count,
            "dimension": self.dimension,
            "metric": self.metric,
            "total_searches": self._total_searches,
            "average_search_latency_ms": round(avg_latency, 4),
        }
