"""Multi-hop Knowledge Graph Traversal Algorithms for Associative Entity Reasoning.

Implements BFS K-hop neighborhood expansion, Unweighted Shortest Path,
Dijkstra's Most Reliable Path using -log(confidence) weights, and
All-Paths DFS chain discovery with natural language reasoning explanations.
"""

from collections import deque
from dataclasses import dataclass, field
import heapq
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.schema import (
    Entity,
    FactTriple,
    RelationType,
    SubgraphResult,
    slugify_id,
)


@dataclass
class PathResult:
    """Represents a structured reasoning path connecting two entities through graph edges."""

    source_id: str
    target_id: str
    entities: List[Entity] = field(default_factory=list)
    triples: List[FactTriple] = field(default_factory=list)
    hop_count: int = 0
    total_weight: float = 0.0
    confidence_score: float = 1.0
    found: bool = True

    def to_explanation(self) -> str:
        """Render the traversal path into a step-by-step reasoning chain."""
        if not self.found or not self.entities:
            return f"No valid reasoning path found between '{self.source_id}' and '{self.target_id}'."

        if len(self.entities) == 1:
            return f"Entity '{self.entities[0].name}' is self-referential (0 hops)."

        chain_parts: List[str] = [self.entities[0].name]
        for i, triple in enumerate(self.triples):
            next_entity = self.entities[i + 1]
            chain_parts.append(f"--[{triple.relation.relation_name} (conf: {triple.confidence:.2f})]--> {next_entity.name}")

        summary = " ".join(chain_parts)
        return f"Reasoning Chain ({self.hop_count} hops, aggregate confidence: {self.confidence_score:.3f}):\n  {summary}"


class GraphTraversalEngine:
    """Production graph traversal and associative multi-hop reasoning engine.

    Algorithms:
      - BFS K-Hop Neighborhood Expansion: O(|V| + |E|)
      - Shortest Path (BFS): O(|V| + |E|)
      - Most Reliable Path (Dijkstra on -log(confidence)): O(|E| + |V| log |V|)
      - All Simple Paths (DFS with depth bound): Bounded O(B^K)
      - Common Neighbors Intersection: O(deg(A) + deg(B))
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def bfs_k_hop(
        self,
        source_id: str,
        max_hops: int = 2,
        min_confidence: float = 0.0,
        relation_types: Optional[List[Union[RelationType, str]]] = None,
        direction: str = "both",
    ) -> Optional[SubgraphResult]:
        """Expand and extract all entities and factual triples reachable within max_hops.

        Time Complexity: O(|V| + |E|)
        Space Complexity: O(|V|)
        """
        root = self.graph.get_entity(source_id)
        if root is None:
            return None

        allowed_relations: Optional[Set[str]] = None
        if relation_types:
            allowed_relations = {
                r.value if isinstance(r, RelationType) else str(r) for r in relation_types
            }

        visited_nodes: Set[str] = {root.entity_id}
        collected_entities: Dict[str, Entity] = {root.entity_id: root}
        collected_triples: Dict[str, FactTriple] = {}
        paths: List[List[str]] = []

        queue: deque[Tuple[str, int]] = deque([(root.entity_id, 0)])

        while queue:
            curr_id, current_hop = queue.popleft()

            if current_hop >= max_hops:
                continue

            neighbors = self.graph.get_neighbors(curr_id, direction=direction)
            for neighbor_entity, triple in neighbors:
                if triple.confidence < min_confidence:
                    continue

                if allowed_relations and triple.relation.relation_name not in allowed_relations:
                    continue

                nid = neighbor_entity.entity_id
                collected_entities[nid] = neighbor_entity
                collected_triples[triple.triple_id] = triple
                paths.append([curr_id, nid])

                if nid not in visited_nodes:
                    visited_nodes.add(nid)
                    queue.append((nid, current_hop + 1))

        return SubgraphResult(
            root_entity=root,
            entities=list(collected_entities.values()),
            triples=list(collected_triples.values()),
            paths=paths,
        )

    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 5,
        direction: str = "both",
    ) -> Optional[PathResult]:
        """Find the path with the minimum number of edge hops between two entities using BFS.

        Time Complexity: O(|V| + |E|)
        """
        src = slugify_id(source_id)
        tgt = slugify_id(target_id)

        source_node = self.graph.get_entity(src)
        target_node = self.graph.get_entity(tgt)
        if source_node is None or target_node is None:
            return None

        if src == tgt:
            return PathResult(
                source_id=src,
                target_id=tgt,
                entities=[source_node],
                triples=[],
                hop_count=0,
                total_weight=0.0,
                confidence_score=1.0,
                found=True,
            )

        # BFS queue stores: (current_node_id, current_hop)
        queue: deque[Tuple[str, int]] = deque([(src, 0)])
        # parent maps: node_id -> (predecessor_node_id, connecting_FactTriple)
        predecessors: Dict[str, Tuple[str, FactTriple]] = {}
        visited: Set[str] = {src}

        path_found = False

        while queue:
            curr_id, hops = queue.popleft()

            if hops >= max_hops:
                continue

            for neighbor, triple in self.graph.get_neighbors(curr_id, direction=direction):
                nid = neighbor.entity_id
                if nid not in visited:
                    visited.add(nid)
                    predecessors[nid] = (curr_id, triple)

                    if nid == tgt:
                        path_found = True
                        break

                    queue.append((nid, hops + 1))

            if path_found:
                break

        if not path_found:
            return PathResult(source_id=src, target_id=tgt, found=False)

        # Reconstruct path backwards from target
        curr = tgt
        triples_rev: List[FactTriple] = []
        entities_rev: List[Entity] = [target_node]

        while curr != src:
            prev_node_id, triple = predecessors[curr]
            triples_rev.append(triple)
            prev_entity = self.graph.get_entity(prev_node_id)
            if prev_entity is not None:
                entities_rev.append(prev_entity)
            curr = prev_node_id

        entities = list(reversed(entities_rev))
        triples = list(reversed(triples_rev))

        total_weight = sum(t.relation.weight for t in triples)
        confidence = 1.0
        for t in triples:
            confidence *= t.confidence

        return PathResult(
            source_id=src,
            target_id=tgt,
            entities=entities,
            triples=triples,
            hop_count=len(triples),
            total_weight=round(total_weight, 4),
            confidence_score=round(confidence, 6),
            found=True,
        )

    def find_most_reliable_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 5,
        direction: str = "both",
    ) -> Optional[PathResult]:
        """Find the reasoning path that maximizes aggregate confidence using Dijkstra's algorithm.

        Mathematical Formulation:
          Maximizing confidence product: max Product(c_i) for c_i in [0, 1]
          is equivalent to minimizing: min Sum( -log(max(1e-9, c_i)) )
          Since -log(c_i) >= 0, Dijkstra's algorithm guarantees finding the globally optimal path.

        Time Complexity: O(|E| + |V| log |V|)
        """
        src = slugify_id(source_id)
        tgt = slugify_id(target_id)

        source_node = self.graph.get_entity(src)
        target_node = self.graph.get_entity(tgt)
        if source_node is None or target_node is None:
            return None

        if src == tgt:
            return PathResult(
                source_id=src,
                target_id=tgt,
                entities=[source_node],
                triples=[],
                hop_count=0,
                confidence_score=1.0,
                found=True,
            )

        # Min-Heap stores: (accumulated_penalty, hops, current_node_id)
        # penalty = sum( -log(confidence) )
        heap: List[Tuple[float, int, str]] = [(0.0, 0, src)]
        best_penalties: Dict[str, float] = {src: 0.0}
        predecessors: Dict[str, Tuple[str, FactTriple]] = {}

        path_found = False

        while heap:
            curr_penalty, hops, curr_id = heapq.heappop(heap)

            if curr_id == tgt:
                path_found = True
                break

            if curr_penalty > best_penalties.get(curr_id, float("inf")):
                continue

            if hops >= max_hops:
                continue

            for neighbor, triple in self.graph.get_neighbors(curr_id, direction=direction):
                nid = neighbor.entity_id

                # -log(confidence) weight mapping
                c = max(1e-6, min(1.0, triple.confidence))
                edge_penalty = -math.log(c)
                new_penalty = curr_penalty + edge_penalty

                if new_penalty < best_penalties.get(nid, float("inf")):
                    best_penalties[nid] = new_penalty
                    predecessors[nid] = (curr_id, triple)
                    heapq.heappush(heap, (new_penalty, hops + 1, nid))

        if not path_found:
            return PathResult(source_id=src, target_id=tgt, found=False)

        # Reconstruct path backwards
        curr = tgt
        triples_rev: List[FactTriple] = []
        entities_rev: List[Entity] = [target_node]

        while curr != src:
            prev_id, triple = predecessors[curr]
            triples_rev.append(triple)
            prev_entity = self.graph.get_entity(prev_id)
            if prev_entity is not None:
                entities_rev.append(prev_entity)
            curr = prev_id

        entities = list(reversed(entities_rev))
        triples = list(reversed(triples_rev))

        total_weight = sum(t.relation.weight for t in triples)
        confidence = math.exp(-best_penalties[tgt])

        return PathResult(
            source_id=src,
            target_id=tgt,
            entities=entities,
            triples=triples,
            hop_count=len(triples),
            total_weight=round(total_weight, 4),
            confidence_score=round(confidence, 6),
            found=True,
        )

    def find_all_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
        limit: int = 10,
        direction: str = "both",
    ) -> List[PathResult]:
        """Discover all simple reasoning chains between source and target up to max_hops via bounded DFS.

        Time Complexity: O(deg^max_hops)
        """
        src = slugify_id(source_id)
        tgt = slugify_id(target_id)

        source_node = self.graph.get_entity(src)
        target_node = self.graph.get_entity(tgt)
        if source_node is None or target_node is None:
            return []

        all_results: List[PathResult] = []

        def dfs(
            curr_id: str,
            visited: Set[str],
            curr_entities: List[Entity],
            curr_triples: List[FactTriple],
        ) -> None:
            if len(all_results) >= limit:
                return

            if curr_id == tgt and curr_triples:
                tot_weight = sum(t.relation.weight for t in curr_triples)
                conf = 1.0
                for t in curr_triples:
                    conf *= t.confidence
                all_results.append(
                    PathResult(
                        source_id=src,
                        target_id=tgt,
                        entities=list(curr_entities),
                        triples=list(curr_triples),
                        hop_count=len(curr_triples),
                        total_weight=round(tot_weight, 4),
                        confidence_score=round(conf, 6),
                        found=True,
                    )
                )
                return

            if len(curr_triples) >= max_hops:
                return

            for neighbor, triple in self.graph.get_neighbors(curr_id, direction=direction):
                nid = neighbor.entity_id
                if nid not in visited:
                    visited.add(nid)
                    curr_entities.append(neighbor)
                    curr_triples.append(triple)

                    dfs(nid, visited, curr_entities, curr_triples)

                    curr_triples.pop()
                    curr_entities.pop()
                    visited.remove(nid)

        dfs(src, {src}, [source_node], [])
        return all_results

    def find_common_neighbors(
        self,
        entity_a: str,
        entity_b: str,
        direction: str = "both",
    ) -> List[Entity]:
        """Find all intermediate entities connected to both entity_a and entity_b.

        Time Complexity: O(deg(A) + deg(B))
        """
        ea = slugify_id(entity_a)
        eb = slugify_id(entity_b)

        if not self.graph.has_entity(ea) or not self.graph.has_entity(eb):
            return []

        neighbors_a = {n[0].entity_id: n[0] for n in self.graph.get_neighbors(ea, direction=direction)}
        neighbors_b = {n[0].entity_id: n[0] for n in self.graph.get_neighbors(eb, direction=direction)}

        common_ids = set(neighbors_a.keys()).intersection(neighbors_b.keys())
        # Exclude endpoints
        common_ids.discard(ea)
        common_ids.discard(eb)

        return [neighbors_a[cid] for cid in common_ids]
