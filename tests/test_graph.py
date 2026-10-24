"""Unit tests for KnowledgeGraph double adjacency list and indexing.

Validates node addition, automatic entity extraction on edge insertion,
degree calculations, entity resolution via aliases, edge removal,
subgraph extraction, and JSON persistence round-trips.
Compatible with standard library `unittest` and `pytest`.
"""

import tempfile
import unittest
from pathlib import Path

from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.schema import (
    Entity,
    EntityType,
    FactTriple,
    Relation,
    RelationType,
)


class TestKnowledgeGraph(unittest.TestCase):
    """Test suite verifying in-memory Knowledge Graph data structure and operations."""

    def setUp(self) -> None:
        self.kg = KnowledgeGraph()

        self.e_nvidia = Entity(
            entity_id="nvidia",
            name="NVIDIA Corporation",
            entity_type=EntityType.ORGANIZATION,
            aliases=["NVDA", "nvidia"],
        )
        self.e_jensen = Entity(
            entity_id="jensen_huang",
            name="Jensen Huang",
            entity_type=EntityType.PERSON,
            aliases=["Jensen"],
        )
        self.e_blackwell = Entity(
            entity_id="blackwell_b200",
            name="Blackwell B200",
            entity_type=EntityType.PRODUCT,
            aliases=["B200"],
        )
        self.e_tsmc = Entity(
            entity_id="tsmc",
            name="TSMC",
            entity_type=EntityType.ORGANIZATION,
            aliases=["Taiwan Semiconductor"],
        )

    def test_empty_graph(self) -> None:
        """New graph must be empty with zero density."""
        self.assertEqual(len(self.kg), 0)
        self.assertEqual(self.kg.num_entities, 0)
        self.assertEqual(self.kg.num_triples, 0)
        self.assertEqual(self.kg.density, 0.0)

    def test_add_and_get_entity(self) -> None:
        """Adding entity must index node and allow retrieval via canonical or raw ID."""
        self.kg.add_entity(self.e_nvidia)
        self.assertEqual(self.kg.num_entities, 1)
        self.assertTrue(self.kg.has_entity("nvidia"))
        self.assertTrue(self.kg.has_entity("  NVIDIA  "))

        ent = self.kg.get_entity("nvidia")
        self.assertIsNotNone(ent)
        self.assertEqual(ent.name, "NVIDIA Corporation")

    def test_resolve_entity_via_aliases(self) -> None:
        """resolve_entity must look up entities using alias dictionary."""
        self.kg.add_entity(self.e_nvidia)
        self.kg.add_entity(self.e_jensen)

        # Canonical lookup
        self.assertEqual(self.kg.resolve_entity("nvidia"), self.e_nvidia)

        # Alias lookup
        self.assertEqual(self.kg.resolve_entity("nvda"), self.e_nvidia)
        self.assertEqual(self.kg.resolve_entity("NVDA"), self.e_nvidia)
        self.assertEqual(self.kg.resolve_entity("Jensen"), self.e_jensen)

        # Nonexistent entity
        self.assertIsNone(self.kg.resolve_entity("Unknown Corp"))

    def test_add_triple_and_degrees(self) -> None:
        """Adding a triple auto-registers nodes and correctly updates degrees."""
        rel_leads = Relation(relation_id="r1", relation_type=RelationType.LEADS)
        triple = FactTriple(
            triple_id="t1",
            subject=self.e_jensen,
            relation=rel_leads,
            object=self.e_nvidia,
        )

        self.kg.add_triple(triple)
        self.assertEqual(self.kg.num_entities, 2)
        self.assertEqual(self.kg.num_triples, 1)

        # Jensen -> NVIDIA
        self.assertEqual(self.kg.out_degree("jensen_huang"), 1)
        self.assertEqual(self.kg.in_degree("jensen_huang"), 0)
        self.assertEqual(self.kg.degree("jensen_huang"), 1)

        self.assertEqual(self.kg.out_degree("nvidia"), 0)
        self.assertEqual(self.kg.in_degree("nvidia"), 1)
        self.assertEqual(self.kg.degree("nvidia"), 1)

    def test_get_neighbors_and_filtering(self) -> None:
        """get_neighbors must support directionality and relation filtering."""
        t1 = FactTriple(
            triple_id="t1",
            subject=self.e_jensen,
            relation=Relation(relation_id="r1", relation_type=RelationType.LEADS),
            object=self.e_nvidia,
        )
        t2 = FactTriple(
            triple_id="t2",
            subject=self.e_nvidia,
            relation=Relation(relation_id="r2", relation_type=RelationType.DEVELOPS),
            object=self.e_blackwell,
        )
        t3 = FactTriple(
            triple_id="t3",
            subject=self.e_nvidia,
            relation=Relation(relation_id="r3", relation_type=RelationType.PARTNERS_WITH),
            object=self.e_tsmc,
        )

        self.kg.add_triple(t1)
        self.kg.add_triple(t2)
        self.kg.add_triple(t3)

        # NVIDIA outgoing neighbors
        out_neighbors = self.kg.get_neighbors("nvidia", direction="out")
        self.assertEqual(len(out_neighbors), 2)
        target_ids = {n[0].entity_id for n in out_neighbors}
        self.assertEqual(target_ids, {"blackwell_b200", "tsmc"})

        # NVIDIA incoming neighbors
        in_neighbors = self.kg.get_neighbors("nvidia", direction="in")
        self.assertEqual(len(in_neighbors), 1)
        self.assertEqual(in_neighbors[0][0].entity_id, "jensen_huang")

        # Relation filtering
        partners = self.kg.get_neighbors("nvidia", direction="out", relation_type=RelationType.PARTNERS_WITH)
        self.assertEqual(len(partners), 1)
        self.assertEqual(partners[0][0].entity_id, "tsmc")

    def test_undirected_relations(self) -> None:
        """Undirected relations should allow bidirectional traversal."""
        rel_undirected = Relation(
            relation_id="r_peer",
            relation_type=RelationType.COMPETES_WITH,
            directed=False,
        )
        e_amd = Entity(entity_id="amd", name="AMD", entity_type=EntityType.ORGANIZATION)
        triple = FactTriple(
            triple_id="t_undir",
            subject=self.e_nvidia,
            relation=rel_undirected,
            object=e_amd,
        )

        self.kg.add_triple(triple)
        # Should be accessible in outgoing neighbors from both sides
        nvda_out = self.kg.get_neighbors("nvidia", direction="out")
        amd_out = self.kg.get_neighbors("amd", direction="out")

        self.assertTrue(any(n[0].entity_id == "amd" for n in nvda_out))
        self.assertTrue(any(n[0].entity_id == "nvidia" for n in amd_out))

    def test_get_entities_by_type(self) -> None:
        """get_entities_by_type must retrieve all matching entity nodes."""
        self.kg.add_entity(self.e_nvidia)
        self.kg.add_entity(self.e_jensen)
        self.kg.add_entity(self.e_tsmc)

        orgs = self.kg.get_entities_by_type(EntityType.ORGANIZATION)
        self.assertEqual(len(orgs), 2)
        org_ids = {e.entity_id for e in orgs}
        self.assertEqual(org_ids, {"nvidia", "tsmc"})

        persons = self.kg.get_entities_by_type(EntityType.PERSON)
        self.assertEqual(len(persons), 1)
        self.assertEqual(persons[0].entity_id, "jensen_huang")

    def test_remove_triple(self) -> None:
        """remove_triple must detach edge while preserving entity nodes."""
        triple = FactTriple(
            triple_id="t1",
            subject=self.e_jensen,
            relation=Relation(relation_id="r1", relation_type=RelationType.LEADS),
            object=self.e_nvidia,
        )
        self.kg.add_triple(triple)
        self.assertEqual(self.kg.num_triples, 1)

        removed = self.kg.remove_triple("t1")
        self.assertTrue(removed)
        self.assertEqual(self.kg.num_triples, 0)
        self.assertEqual(self.kg.num_entities, 2)  # Nodes still exist
        self.assertEqual(self.kg.out_degree("jensen_huang"), 0)
        self.assertEqual(self.kg.in_degree("nvidia"), 0)

        # Removing nonexistent triple returns False
        self.assertFalse(self.kg.remove_triple("t1"))

    def test_remove_entity_cascades_edges(self) -> None:
        """remove_entity must delete node and purge all incident triples."""
        t1 = FactTriple(
            triple_id="t1",
            subject=self.e_jensen,
            relation=Relation(relation_id="r1", relation_type=RelationType.LEADS),
            object=self.e_nvidia,
        )
        t2 = FactTriple(
            triple_id="t2",
            subject=self.e_nvidia,
            relation=Relation(relation_id="r2", relation_type=RelationType.DEVELOPS),
            object=self.e_blackwell,
        )
        self.kg.add_triple(t1)
        self.kg.add_triple(t2)
        self.assertEqual(self.kg.num_entities, 3)
        self.assertEqual(self.kg.num_triples, 2)

        # Delete central hub node NVIDIA
        self.assertTrue(self.kg.remove_entity("nvidia"))
        self.assertEqual(self.kg.num_entities, 2)
        self.assertEqual(self.kg.num_triples, 0)  # Both triples purged
        self.assertIsNone(self.kg.resolve_entity("nvda"))  # Alias purged

    def test_subgraph_extraction(self) -> None:
        """get_subgraph must return egocentric subgraph around a root entity."""
        t1 = FactTriple(
            triple_id="t1",
            subject=self.e_jensen,
            relation=Relation(relation_id="r1", relation_type=RelationType.LEADS),
            object=self.e_nvidia,
        )
        t2 = FactTriple(
            triple_id="t2",
            subject=self.e_nvidia,
            relation=Relation(relation_id="r2", relation_type=RelationType.DEVELOPS),
            object=self.e_blackwell,
        )
        self.kg.add_triple(t1)
        self.kg.add_triple(t2)

        # 1-hop subgraph around NVIDIA
        subgraph = self.kg.get_subgraph("nvidia", max_hops=1)
        self.assertIsNotNone(subgraph)
        self.assertEqual(subgraph.root_entity.entity_id, "nvidia")
        self.assertEqual(len(subgraph.entities), 3)  # nvidia, jensen, blackwell
        self.assertEqual(len(subgraph.triples), 2)

    def test_save_and_load_json(self) -> None:
        """KnowledgeGraph must accurately persist to and load from JSON files."""
        triple = FactTriple(
            triple_id="t1",
            subject=self.e_jensen,
            relation=Relation(relation_id="r1", relation_type=RelationType.LEADS),
            object=self.e_nvidia,
        )
        self.kg.add_triple(triple)

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_kg.json"
            self.kg.save_json(file_path)
            self.assertTrue(file_path.exists())

            loaded_kg = KnowledgeGraph.load_json(file_path)
            self.assertEqual(loaded_kg.num_entities, 2)
            self.assertEqual(loaded_kg.num_triples, 1)
            self.assertTrue(loaded_kg.has_entity("nvidia"))
            self.assertEqual(loaded_kg.resolve_entity("nvda"), self.e_nvidia)


if __name__ == "__main__":
    unittest.main()
