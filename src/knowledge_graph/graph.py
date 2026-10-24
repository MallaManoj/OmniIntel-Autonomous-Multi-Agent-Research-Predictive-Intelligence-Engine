"""In-Memory Knowledge Graph with Adjacency List & Multi-Index Representations.

Supports directed multi-relational graphs, provenance tracking, fast entity resolution,
and bidirectional neighbor traversal in O(deg(u)) time.
"""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.knowledge_graph.schema import (
    Entity,
    EntityType,
    FactTriple,
    Relation,
    RelationType,
    SubgraphResult,
    slugify_id,
)


class KnowledgeGraph:
    """Production in-memory Knowledge Graph backed by double adjacency lists and inverted indices.

    Complexity:
      - Add Entity / Node: O(1)
      - Add FactTriple / Edge: O(1)
      - Lookup Entity: O(1)
      - Get Incident Neighbors: O(deg(u))
      - Entity Resolution (Alias -> Entity): O(1)
      - Delete Entity: O(deg(u))
    """

    def __init__(self) -> None:
        # Node storage: entity_id -> Entity
        self._nodes: Dict[str, Entity] = {}

        # Adjacency lists: source_id -> { target_id -> [FactTriple] }
        self._outgoing: Dict[str, Dict[str, List[FactTriple]]] = defaultdict(lambda: defaultdict(list))
        self._incoming: Dict[str, Dict[str, List[FactTriple]]] = defaultdict(lambda: defaultdict(list))

        # Triple storage: triple_id -> FactTriple
        self._triples: Dict[str, FactTriple] = {}

        # Secondary Indices for high-speed filtering
        self._relation_index: Dict[str, Set[str]] = defaultdict(set)
        self._type_index: Dict[EntityType, Set[str]] = defaultdict(set)
        self._alias_index: Dict[str, str] = {}  # alias / surface form -> entity_id

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def num_entities(self) -> int:
        """Total number of unique entities (nodes) in the graph."""
        return len(self._nodes)

    @property
    def num_triples(self) -> int:
        """Total number of factual edges (triples) in the graph."""
        return len(self._triples)

    @property
    def density(self) -> float:
        """Directed graph density = |E| / (|V| * (|V| - 1)). Returns 0.0 if |V| < 2."""
        v = self.num_entities
        if v < 2:
            return 0.0
        e = self.num_triples
        return round(e / (v * (v - 1)), 6)

    def add_entity(self, entity: Entity) -> None:
        """Insert or update an entity node in the Knowledge Graph.

        Time Complexity: O(1) amortized
        """
        self._nodes[entity.entity_id] = entity
        self._type_index[entity.entity_type].add(entity.entity_id)

        # Index all surface forms (canonical name + aliases) for entity resolution
        for form in entity.all_surface_forms():
            self._alias_index[form] = entity.entity_id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by its canonical slug ID in O(1) time."""
        return self._nodes.get(slugify_id(entity_id))

    def has_entity(self, entity_id: str) -> bool:
        """Check whether an entity exists in the graph."""
        return slugify_id(entity_id) in self._nodes

    def resolve_entity(self, query: str) -> Optional[Entity]:
        """Resolve a natural language entity mention or alias to its canonical Entity.

        Time Complexity: O(1)
        """
        query_clean = query.strip().lower()
        if not query_clean:
            return None

        # 1. Direct surface form / alias match
        if query_clean in self._alias_index:
            target_id = self._alias_index[query_clean]
            return self._nodes.get(target_id)

        # 2. Slugified ID fallback
        slug = slugify_id(query_clean)
        return self._nodes.get(slug)

    def add_triple(self, triple: FactTriple) -> None:
        """Insert a factual relationship edge connecting subject and object.

        Automatically inserts subject and object entities if not already present.
        Time Complexity: O(1)
        """
        sub_id = triple.subject.entity_id
        obj_id = triple.object.entity_id

        # Ensure entities exist
        if sub_id not in self._nodes:
            self.add_entity(triple.subject)
        if obj_id not in self._nodes:
            self.add_entity(triple.object)

        self._triples[triple.triple_id] = triple

        # Register in forward and reverse adjacency lists
        self._outgoing[sub_id][obj_id].append(triple)
        self._incoming[obj_id][sub_id].append(triple)

        rel_name = triple.relation.relation_name
        self._relation_index[rel_name].add(triple.triple_id)

        # If undirected relation, create symmetric edge representation
        if not triple.relation.directed:
            self._outgoing[obj_id][sub_id].append(triple)
            self._incoming[sub_id][obj_id].append(triple)

    def get_triple(self, triple_id: str) -> Optional[FactTriple]:
        """Retrieve a specific fact triple by its ID in O(1) time."""
        return self._triples.get(triple_id)

    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "out",
        relation_type: Optional[Union[RelationType, str]] = None,
    ) -> List[Tuple[Entity, FactTriple]]:
        """Retrieve incident neighbors and connecting fact triples.

        Args:
            entity_id: Canonical entity slug.
            direction: "out" (outgoing edges), "in" (incoming edges), or "both".
            relation_type: Optional filter for a specific relation type.

        Returns:
            List of (NeighborEntity, FactTriple) tuples.
        """
        eid = slugify_id(entity_id)
        if eid not in self._nodes:
            return []

        rel_filter = (
            relation_type.value if isinstance(relation_type, RelationType)
            else str(relation_type) if relation_type is not None
            else None
        )

        results: List[Tuple[Entity, FactTriple]] = []

        # Outgoing edges: u -> v
        if direction in ("out", "both"):
            for target_id, triples in self._outgoing.get(eid, {}).items():
                target_node = self._nodes[target_id]
                for t in triples:
                    if rel_filter is None or t.relation.relation_name == rel_filter:
                        results.append((target_node, t))

        # Incoming edges: v -> u
        if direction in ("in", "both"):
            for source_id, triples in self._incoming.get(eid, {}).items():
                source_node = self._nodes[source_id]
                for t in triples:
                    if rel_filter is None or t.relation.relation_name == rel_filter:
                        # Avoid duplicates if undirected edge was already counted in "out"
                        if direction == "both" and not t.relation.directed and source_id in self._outgoing.get(eid, {}):
                            continue
                        results.append((source_node, t))

        return results

    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Retrieve all entity nodes classified under a specific EntityType."""
        eids = self._type_index.get(entity_type, set())
        return [self._nodes[eid] for eid in eids if eid in self._nodes]

    def out_degree(self, entity_id: str) -> int:
        """Number of outgoing edges from entity."""
        eid = slugify_id(entity_id)
        return sum(len(triples) for triples in self._outgoing.get(eid, {}).values())

    def in_degree(self, entity_id: str) -> int:
        """Number of incoming edges into entity."""
        eid = slugify_id(entity_id)
        return sum(len(triples) for triples in self._incoming.get(eid, {}).values())

    def degree(self, entity_id: str) -> int:
        """Total degree (in-degree + out-degree) of an entity node."""
        return self.out_degree(entity_id) + self.in_degree(entity_id)

    def remove_triple(self, triple_id: str) -> bool:
        """Remove a fact triple from the Knowledge Graph.

        Returns True if deleted, False if not found.
        """
        if triple_id not in self._triples:
            return False

        triple = self._triples.pop(triple_id)
        sub_id = triple.subject.entity_id
        obj_id = triple.object.entity_id
        rel_name = triple.relation.relation_name

        # Clean from relation index
        if rel_name in self._relation_index:
            self._relation_index[rel_name].discard(triple_id)

        # Clean from outgoing adjacency
        if sub_id in self._outgoing and obj_id in self._outgoing[sub_id]:
            self._outgoing[sub_id][obj_id] = [t for t in self._outgoing[sub_id][obj_id] if t.triple_id != triple_id]
            if not self._outgoing[sub_id][obj_id]:
                del self._outgoing[sub_id][obj_id]

        # Clean from incoming adjacency
        if obj_id in self._incoming and sub_id in self._incoming[obj_id]:
            self._incoming[obj_id][sub_id] = [t for t in self._incoming[obj_id][sub_id] if t.triple_id != triple_id]
            if not self._incoming[obj_id][sub_id]:
                del self._incoming[obj_id][sub_id]

        # Clean undirected symmetric references
        if not triple.relation.directed:
            if obj_id in self._outgoing and sub_id in self._outgoing[obj_id]:
                self._outgoing[obj_id][sub_id] = [t for t in self._outgoing[obj_id][sub_id] if t.triple_id != triple_id]
                if not self._outgoing[obj_id][sub_id]:
                    del self._outgoing[obj_id][sub_id]
            if sub_id in self._incoming and obj_id in self._incoming[sub_id]:
                self._incoming[sub_id][obj_id] = [t for t in self._incoming[sub_id][obj_id] if t.triple_id != triple_id]
                if not self._incoming[sub_id][obj_id]:
                    del self._incoming[sub_id][obj_id]

        return True

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and all its incident edges from the Knowledge Graph.

        Returns True if removed, False if entity did not exist.
        """
        eid = slugify_id(entity_id)
        if eid not in self._nodes:
            return False

        entity = self._nodes.pop(eid)

        # Remove from type index
        self._type_index[entity.entity_type].discard(eid)

        # Remove from alias index
        for form in entity.all_surface_forms():
            if self._alias_index.get(form) == eid:
                del self._alias_index[form]

        # Collect and delete all incident triples
        incident_triple_ids: Set[str] = set()

        for triples in self._outgoing.get(eid, {}).values():
            for t in triples:
                incident_triple_ids.add(t.triple_id)

        for triples in self._incoming.get(eid, {}).values():
            for t in triples:
                incident_triple_ids.add(t.triple_id)

        for tid in incident_triple_ids:
            self.remove_triple(tid)

        if eid in self._outgoing:
            del self._outgoing[eid]
        if eid in self._incoming:
            del self._incoming[eid]

        return True

    def get_subgraph(self, entity_id: str, max_hops: int = 1) -> Optional[SubgraphResult]:
        """Extract local egocentric subgraph centered at entity_id up to max_hops."""
        root = self.get_entity(entity_id)
        if root is None:
            return None

        visited_nodes: Set[str] = {root.entity_id}
        collected_entities: Dict[str, Entity] = {root.entity_id: root}
        collected_triples: Dict[str, FactTriple] = {}
        paths: List[List[str]] = []

        current_frontier: List[str] = [root.entity_id]

        for _ in range(max_hops):
            next_frontier: List[str] = []
            for curr_id in current_frontier:
                neighbors = self.get_neighbors(curr_id, direction="both")
                for neighbor_entity, triple in neighbors:
                    nid = neighbor_entity.entity_id
                    collected_entities[nid] = neighbor_entity
                    collected_triples[triple.triple_id] = triple
                    paths.append([curr_id, nid])

                    if nid not in visited_nodes:
                        visited_nodes.add(nid)
                        next_frontier.append(nid)

            current_frontier = next_frontier
            if not current_frontier:
                break

        return SubgraphResult(
            root_entity=root,
            entities=list(collected_entities.values()),
            triples=list(collected_triples.values()),
            paths=paths,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete Knowledge Graph into a JSON-compatible dictionary."""
        return {
            "entities": [entity.model_dump() for entity in self._nodes.values()],
            "triples": [triple.model_dump() for triple in self._triples.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Reconstruct a KnowledgeGraph from a serialized dictionary."""
        kg = cls()
        # Ingest entities
        for ent_dict in data.get("entities", []):
            kg.add_entity(Entity.model_validate(ent_dict))

        # Ingest triples
        for trip_dict in data.get("triples", []):
            kg.add_triple(FactTriple.model_validate(trip_dict))

        return kg

    def get_all_entities(self) -> List[Entity]:
        """Retrieve all registered entities in the graph."""
        return list(self._nodes.values())

    def get_all_triples(self) -> List[FactTriple]:
        """Retrieve all factual relationship triples in the graph."""
        return list(self._triples.values())

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Persist graph structure to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> "KnowledgeGraph":
        """Load a persisted KnowledgeGraph from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge Graph file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
