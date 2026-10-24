"""Hybrid Retrieval Engine fusing Dense Vector Search, Sparse Lexical Matching, and Knowledge Graph Paths.

Executes multi-channel retrieval across:
  1. Dense Semantic Embeddings (SIMD dot-product cosine similarity)
  2. Sparse Lexical Indexing (TF-IDF + Prefix Trie exact entity scanning)
  3. Structured Knowledge Graph Expansion (BFS K-hop & associative reasoning paths)

Fuses results using Reciprocal Rank Fusion (RRF) or Convex Linear Combination,
and extracts top candidates using an O(N log K) bounded Min-Heap.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import heapq
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from src.core.dsa.trie import PrefixTrie
from src.knowledge_graph.chunker import DocumentChunk, TextChunker
from src.knowledge_graph.embeddings import EmbeddingEngine
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.schema import Entity, FactTriple, slugify_id
from src.knowledge_graph.traversal import GraphTraversalEngine, PathResult
from src.knowledge_graph.vector_store import VectorStore


@dataclass
class HybridSearchConfig:
    """Hyperparameters governing hybrid retrieval weights and fusion strategies."""

    dense_weight: float = 0.5
    sparse_weight: float = 0.3
    graph_weight: float = 0.2
    rrf_k: int = 60
    fusion_method: str = "rrf"  # "rrf" or "linear"
    top_k: int = 5
    graph_max_hops: int = 2
    min_score_threshold: Optional[float] = None


@dataclass
class HybridSearchResult:
    """Unified retrieval result with provenance breakdown across all modalities."""

    doc_id: str
    text: str
    score: float
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    graph_rank: Optional[int] = None
    graph_score: Optional[float] = None
    matched_entities: List[str] = field(default_factory=list)
    graph_paths: List[PathResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridSearchEngine:
    """Production Hybrid Retrieval Engine combining Dense, Sparse, and Graph intelligence."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_engine: EmbeddingEngine,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        chunker: Optional[TextChunker] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.traversal_engine = GraphTraversalEngine(self.knowledge_graph)
        self.chunker = chunker or TextChunker(chunk_size=100, chunk_overlap=20)

        # Entity recognition Trie
        self.trie: PrefixTrie[str] = PrefixTrie(case_sensitive=False)

        # Document storage
        self._doc_texts: Dict[str, str] = {}
        self._doc_metadatas: Dict[str, Dict[str, Any]] = {}
        self._doc_entities: Dict[str, Set[str]] = defaultdict(set)

        # Sparse Inverted Index for lexical retrieval: token -> {doc_id: count}
        self._inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._doc_lengths: Dict[str, int] = {}
        self._total_documents: int = 0

        # Synchronize Trie with all known Knowledge Graph entities
        self._sync_trie_with_graph()

        # Telemetry
        self._total_queries: int = 0
        self._total_query_time_ms: float = 0.0

    def _sync_trie_with_graph(self) -> None:
        """Register all entities and surface forms from the Knowledge Graph into the Prefix Trie."""
        for entity in self.knowledge_graph._nodes.values():
            for form in entity.all_surface_forms():
                self.trie.insert(form, value=entity.entity_id)

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace and alphanumeric tokenizer."""
        return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())

    def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunks: Optional[List[DocumentChunk]] = None,
    ) -> List[DocumentChunk]:
        """Ingest, chunk, embed, and index a document across Dense, Sparse, and Graph stores.

        Returns:
            List of generated DocumentChunk instances.
        """
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        if not text.strip():
            raise ValueError("text cannot be empty")

        meta = metadata or {}
        self._doc_texts[doc_id] = text
        self._doc_metadatas[doc_id] = meta

        # 1. Sparse Indexing (Token frequencies)
        tokens = self._tokenize(text)
        token_counts = Counter(tokens)
        self._doc_lengths[doc_id] = len(tokens)
        self._total_documents = len(self._doc_texts)

        for token, count in token_counts.items():
            self._inverted_index[token][doc_id] = count

        # 2. Entity Scanning via Prefix Trie
        entity_matches = self.trie.extract_entities(text)
        for _, _, _, entity_slug in entity_matches:
            if entity_slug:
                self._doc_entities[doc_id].add(entity_slug)

        # 3. Dense Chunking and Embedding
        doc_chunks = chunks or self.chunker.chunk_text(text, doc_id=doc_id)
        if doc_chunks:
            chunk_pairs = self.embedding_engine.embed_chunks(doc_chunks)
            chunk_ids = [c.chunk_id for c, _ in chunk_pairs]
            chunk_vectors = np.vstack([vec for _, vec in chunk_pairs])
            chunk_metas = [
                {
                    "doc_id": doc_id,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    **meta,
                }
                for c, _ in chunk_pairs
            ]
            self.vector_store.add_batch(chunk_ids, chunk_vectors, chunk_metas)

        return doc_chunks

    def _compute_sparse_scores(self, query_tokens: List[str]) -> Dict[str, float]:
        """Compute TF-IDF lexical matching scores for each document against query tokens."""
        scores: Dict[str, float] = defaultdict(float)
        n_docs = max(1, self._total_documents)

        for token in set(query_tokens):
            if token not in self._inverted_index:
                continue

            postings = self._inverted_index[token]
            df = len(postings)
            # Standard IDF: log(1 + (N - df + 0.5) / (df + 0.5))
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

            for doc_id, tf in postings.items():
                doc_len = self._doc_lengths.get(doc_id, 1)
                norm_tf = tf / math.sqrt(doc_len)
                scores[doc_id] += norm_tf * idf

        return dict(scores)

    def _compute_graph_scores(
        self,
        query: str,
        max_hops: int = 2,
    ) -> Tuple[Dict[str, float], List[PathResult], List[str]]:
        """Identify query entities, expand K-hop subgraphs, and score documents citing connected concepts."""
        entity_matches = self.trie.extract_entities(query)
        detected_entity_ids = list(dict.fromkeys([slug for _, _, _, slug in entity_matches if slug]))

        if not detected_entity_ids:
            return {}, [], []

        graph_scores: Dict[str, float] = defaultdict(float)
        found_paths: List[PathResult] = []

        # Find paths between pairs of detected query entities
        if len(detected_entity_ids) >= 2:
            for i in range(len(detected_entity_ids)):
                for j in range(i + 1, len(detected_entity_ids)):
                    path = self.traversal_engine.find_most_reliable_path(
                        detected_entity_ids[i],
                        detected_entity_ids[j],
                        max_hops=max_hops,
                    )
                    if path and path.found:
                        found_paths.append(path)

        # Expand K-hop subgraphs from detected entities to find connected entity context
        expanded_entity_weights: Dict[str, float] = defaultdict(float)

        for root_id in detected_entity_ids:
            subgraph = self.traversal_engine.bfs_k_hop(root_id, max_hops=max_hops)
            if subgraph:
                for ent in subgraph.entities:
                    expanded_entity_weights[ent.entity_id] += 1.0

        # Score documents containing the query or expanded entities
        for doc_id, doc_entities in self._doc_entities.items():
            score = 0.0
            for ent_id in doc_entities:
                if ent_id in detected_entity_ids:
                    score += 2.0  # Exact entity mention bonus
                elif ent_id in expanded_entity_weights:
                    score += 1.0  # Connected graph neighbor bonus
                    # Trace reasoning path connecting root query entity to document entity
                    for root_id in detected_entity_ids:
                        path = self.traversal_engine.find_most_reliable_path(
                            root_id, ent_id, max_hops=max_hops
                        )
                        if path and path.found and path.hop_count > 0:
                            found_paths.append(path)
            if score > 0.0:
                graph_scores[doc_id] = score

        return dict(graph_scores), found_paths, detected_entity_ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        config: Optional[HybridSearchConfig] = None,
    ) -> List[HybridSearchResult]:
        """Perform unified multi-channel hybrid retrieval with rank/score fusion.

        Pipeline:
          1. Dense Vector Search: query -> embedding -> SIMD dot product.
          2. Sparse Lexical Search: TF-IDF on token inverted index.
          3. Knowledge Graph Expansion: Trie entity scanning + BFS multi-hop expansion.
          4. Fusion: Reciprocal Rank Fusion (RRF) or Convex Linear Normalization.
          5. Bounded Min-Heap: Top-K selection in O(N log K).
        """
        if not query.strip():
            return []

        cfg = config or HybridSearchConfig(top_k=top_k)
        start_time = time.perf_counter()

        # Channel 1: Dense Retrieval
        dense_results: Dict[str, float] = {}
        query_vec = self.embedding_engine.embed_text(query)
        if self.vector_store.count > 0:
            raw_dense = self.vector_store.search(query_vec, top_k=top_k * 4)
            for hit in raw_dense:
                doc_id = hit.metadata.get("doc_id", hit.id)
                # Keep maximum score if document has multiple chunks
                if doc_id not in dense_results or hit.score > dense_results[doc_id]:
                    dense_results[doc_id] = hit.score

        # Channel 2: Sparse Lexical Retrieval
        query_tokens = self._tokenize(query)
        sparse_results = self._compute_sparse_scores(query_tokens)

        # Channel 3: Knowledge Graph Expansion
        graph_results, graph_paths, detected_entities = self._compute_graph_scores(
            query, max_hops=cfg.graph_max_hops
        )

        # Union of candidate document IDs across all channels
        all_candidate_ids = set(dense_results.keys()) | set(sparse_results.keys()) | set(graph_results.keys())
        if not all_candidate_ids:
            return []

        # Build ranked lists for RRF (1-based ranking)
        sorted_dense = sorted(dense_results.items(), key=lambda x: x[1], reverse=True)
        dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_dense)}

        sorted_sparse = sorted(sparse_results.items(), key=lambda x: x[1], reverse=True)
        sparse_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_sparse)}

        sorted_graph = sorted(graph_results.items(), key=lambda x: x[1], reverse=True)
        graph_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_graph)}

        # Fusion computation
        fused_scores: Dict[str, float] = {}

        if cfg.fusion_method == "rrf":
            # Reciprocal Rank Fusion: score = sum( w / (k + rank) )
            for doc_id in all_candidate_ids:
                rrf_score = 0.0
                if doc_id in dense_ranks:
                    rrf_score += cfg.dense_weight * (1.0 / (cfg.rrf_k + dense_ranks[doc_id]))
                if doc_id in sparse_ranks:
                    rrf_score += cfg.sparse_weight * (1.0 / (cfg.rrf_k + sparse_ranks[doc_id]))
                if doc_id in graph_ranks:
                    rrf_score += cfg.graph_weight * (1.0 / (cfg.rrf_k + graph_ranks[doc_id]))
                fused_scores[doc_id] = rrf_score
        else:
            # Linear Score Combination with calibrated scale mapping
            # 1. Cosine similarity [-1.0, 1.0] maps smoothly to [0.0, 1.0]
            norm_dense = {k: max(0.0, min(1.0, (v + 1.0) / 2.0)) for k, v in dense_results.items()}

            # 2. Non-negative sparse and graph scores scale relative to maximum active score
            max_sp = max(sparse_results.values()) if sparse_results else 1.0
            norm_sparse = {k: v / max_sp for k, v in sparse_results.items()} if max_sp > 0 else {}

            max_gr = max(graph_results.values()) if graph_results else 1.0
            norm_graph = {k: v / max_gr for k, v in graph_results.items()} if max_gr > 0 else {}

            for doc_id in all_candidate_ids:
                fused = (
                    cfg.dense_weight * norm_dense.get(doc_id, 0.0)
                    + cfg.sparse_weight * norm_sparse.get(doc_id, 0.0)
                    + cfg.graph_weight * norm_graph.get(doc_id, 0.0)
                )
                fused_scores[doc_id] = fused

        # Top-K selection via bounded Min-Heap
        heap: List[Tuple[float, int, str]] = []
        entry_idx = 0

        for doc_id, final_score in fused_scores.items():
            if cfg.min_score_threshold is not None and final_score < cfg.min_score_threshold:
                continue

            if len(heap) < cfg.top_k:
                heapq.heappush(heap, (final_score, entry_idx, doc_id))
            elif final_score > heap[0][0]:
                heapq.heapreplace(heap, (final_score, entry_idx, doc_id))
            entry_idx += 1

        # Extract sorted descending
        results: List[HybridSearchResult] = []
        while heap:
            score_val, _, doc_id = heapq.heappop(heap)
            text = self._doc_texts.get(doc_id, "")
            meta = self._doc_metadatas.get(doc_id, {})
            results.append(
                HybridSearchResult(
                    doc_id=doc_id,
                    text=text,
                    score=round(score_val, 6),
                    dense_rank=dense_ranks.get(doc_id),
                    dense_score=round(dense_results[doc_id], 4) if doc_id in dense_results else None,
                    sparse_rank=sparse_ranks.get(doc_id),
                    sparse_score=round(sparse_results[doc_id], 4) if doc_id in sparse_results else None,
                    graph_rank=graph_ranks.get(doc_id),
                    graph_score=round(graph_results[doc_id], 4) if doc_id in graph_results else None,
                    matched_entities=list(self._doc_entities.get(doc_id, set())),
                    graph_paths=graph_paths,
                    metadata=meta,
                )
            )
        results.reverse()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._total_queries += 1
        self._total_query_time_ms += elapsed_ms

        return results

    @property
    def telemetry(self) -> Dict[str, Any]:
        """Runtime query statistics."""
        avg_lat = (
            self._total_query_time_ms / self._total_queries
            if self._total_queries > 0
            else 0.0
        )
        return {
            "total_documents": self._total_documents,
            "total_queries": self._total_queries,
            "average_latency_ms": round(avg_lat, 3),
            "vector_store_records": self.vector_store.count,
            "graph_entities": self.knowledge_graph.num_entities,
            "trie_words": len(self.trie),
        }
