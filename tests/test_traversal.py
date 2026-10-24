"""Unit tests for GraphTraversalEngine multi-hop algorithms.

Validates BFS K-hop neighborhood expansion, Unweighted Shortest Path,
Dijkstra's Most Reliable Path using -log(confidence) weights,
All-Paths DFS chain discovery, common neighbor intersection, and explanation rendering.
Compatible with standard library `unittest` and `pytest`.
"""

import unittest

from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.schema import (
    Entity,
    EntityType,
    FactTriple,
    Relation,
    RelationType,
)
from src.knowledge_graph.traversal import GraphTraversalEngine, PathResult


class TestGraphTraversalEngine(unittest.TestCase):
    """Test suite covering multi-hop graph algorithms and reasoning paths."""

    def setUp(self) -> None:
        self.kg = KnowledgeGraph()
        self.engine = GraphTraversalEngine(self.kg)

        # Entity definitions
        self.e_jensen = Entity(entity_id="jensen", name="Jensen Huang", entity_type=EntityType.PERSON)
        self.e_nvidia = Entity(entity_id="nvidia", name="NVIDIA", entity_type=EntityType.ORGANIZATION)
        self.e_tsmc = Entity(entity_id="tsmc", name="TSMC", entity_type=EntityType.ORGANIZATION)
        self.e_apple = Entity(entity_id="apple", name="Apple", entity_type=EntityType.ORGANIZATION)
        self.e_asml = Entity(entity_id="asml", name="ASML", entity_type=EntityType.ORGANIZATION)
        self.e_isolated = Entity(entity_id="isolated", name="Isolated Node", entity_type=EntityType.CONCEPT)

        # Build Graph Topology:
        # Jensen --[LEADS, 1.0]--> NVIDIA --[PARTNERS_WITH, 0.95]--> TSMC --[PARTNERS_WITH, 0.95]--> Apple
        # TSMC --[BUYS_FROM, 0.90]--> ASML
        # Jensen --[ADVISES, 0.40]--> Apple (direct low-confidence shortcut)

        self.kg.add_triple(
            FactTriple(
                triple_id="t1",
                subject=self.e_jensen,
                relation=Relation(relation_id="r1", relation_type=RelationType.LEADS, weight=1.0),
                object=self.e_nvidia,
                confidence=1.0,
            )
        )
        self.kg.add_triple(
            FactTriple(
                triple_id="t2",
                subject=self.e_nvidia,
                relation=Relation(relation_id="r2", relation_type=RelationType.PARTNERS_WITH, weight=1.0),
                object=self.e_tsmc,
                confidence=0.95,
            )
        )
        self.kg.add_triple(
            FactTriple(
                triple_id="t3",
                subject=self.e_tsmc,
                relation=Relation(relation_id="r3", relation_type=RelationType.PARTNERS_WITH, weight=1.0),
                object=self.e_apple,
                confidence=0.95,
            )
        )
        self.kg.add_triple(
            FactTriple(
                triple_id="t4",
                subject=self.e_tsmc,
                relation=Relation(relation_id="r4", relation_type="BUYS_FROM", weight=1.0),
                object=self.e_asml,
                confidence=0.90,
            )
        )
        self.kg.add_triple(
            FactTriple(
                triple_id="t_shortcut",
                subject=self.e_jensen,
                relation=Relation(relation_id="r_short", relation_type="ADVISES", weight=1.0),
                object=self.e_apple,
                confidence=0.40,  # Unreliable shortcut
            )
        )
        self.kg.add_entity(self.e_isolated)

    def test_bfs_k_hop(self) -> None:
        """bfs_k_hop must retrieve nodes within max_hops bounded frontier."""
        # 1-hop from Jensen: NVIDIA and Apple
        subgraph_1 = self.engine.bfs_k_hop("jensen", max_hops=1)
        self.assertIsNotNone(subgraph_1)
        self.assertEqual(len(subgraph_1.entities), 3)  # jensen, nvidia, apple

        # 2-hop from Jensen: Jensen, NVIDIA, Apple, TSMC
        subgraph_2 = self.engine.bfs_k_hop("jensen", max_hops=2)
        self.assertIsNotNone(subgraph_2)
        e_ids = {e.entity_id for e in subgraph_2.entities}
        self.assertEqual(e_ids, {"jensen", "nvidia", "apple", "tsmc"})

    def test_bfs_k_hop_filters(self) -> None:
        """bfs_k_hop must filter by confidence threshold and relation type."""
        # Filter with min_confidence=0.80 should prune the 0.40 shortcut to Apple
        subgraph = self.engine.bfs_k_hop("jensen", max_hops=1, min_confidence=0.80)
        self.assertIsNotNone(subgraph)
        e_ids = {e.entity_id for e in subgraph.entities}
        self.assertEqual(e_ids, {"jensen", "nvidia"})

    def test_find_shortest_path_unweighted(self) -> None:
        """find_shortest_path must find the path with fewest edge hops."""
        # Jensen -> Apple has a 1-hop shortcut (confidence 0.40)
        path = self.engine.find_shortest_path("jensen", "apple")
        self.assertIsNotNone(path)
        self.assertTrue(path.found)
        self.assertEqual(path.hop_count, 1)
        self.assertEqual([e.entity_id for e in path.entities], ["jensen", "apple"])

        # Jensen -> TSMC has 2 hops: Jensen -> NVIDIA -> TSMC
        path_tsmc = self.engine.find_shortest_path("jensen", "tsmc", direction="out")
        self.assertIsNotNone(path_tsmc)
        self.assertEqual(path_tsmc.hop_count, 2)
        self.assertEqual([e.entity_id for e in path_tsmc.entities], ["jensen", "nvidia", "tsmc"])

    def test_find_shortest_path_disconnected(self) -> None:
        """Path between disconnected nodes must report found=False."""
        path = self.engine.find_shortest_path("jensen", "isolated")
        self.assertIsNotNone(path)
        self.assertFalse(path.found)

    def test_dijkstra_most_reliable_path(self) -> None:
        """Dijkstra must prioritize higher aggregate confidence over raw hop count."""
        # Shortcut: Jensen -> Apple is 1 hop with confidence = 0.40.
        # Multi-hop: Jensen -> NVIDIA -> TSMC -> Apple is 3 hops with confidence = 1.0 * 0.95 * 0.95 = 0.9025.
        # Shortest path picks 1 hop; Most reliable path must pick the 3-hop chain with 0.9025 confidence!
        reliable_path = self.engine.find_most_reliable_path("jensen", "apple", direction="out")
        self.assertIsNotNone(reliable_path)
        self.assertTrue(reliable_path.found)
        self.assertEqual(reliable_path.hop_count, 3)
        self.assertEqual([e.entity_id for e in reliable_path.entities], ["jensen", "nvidia", "tsmc", "apple"])
        self.assertAlmostEqual(reliable_path.confidence_score, 0.9025, places=3)

    def test_find_all_paths(self) -> None:
        """find_all_paths must discover all distinct reasoning chains up to max_hops."""
        # Two paths from Jensen to Apple:
        # Path 1: Jensen -> Apple (1 hop)
        # Path 2: Jensen -> NVIDIA -> TSMC -> Apple (3 hops)
        paths = self.engine.find_all_paths("jensen", "apple", max_hops=3, direction="out")
        self.assertEqual(len(paths), 2)
        hop_counts = {p.hop_count for p in paths}
        self.assertEqual(hop_counts, {1, 3})

    def test_find_common_neighbors(self) -> None:
        """find_common_neighbors must find intersecting entities between two nodes."""
        # Both NVIDIA and Apple are connected to TSMC and Jensen
        common_nvda_apple = self.engine.find_common_neighbors("nvidia", "apple")
        self.assertEqual(len(common_nvda_apple), 2)
        common_ids = {e.entity_id for e in common_nvda_apple}
        self.assertEqual(common_ids, {"tsmc", "jensen"})

        # Only TSMC is connected to both NVIDIA and ASML
        common_nvda_asml = self.engine.find_common_neighbors("nvidia", "asml")
        self.assertEqual(len(common_nvda_asml), 1)
        self.assertEqual(common_nvda_asml[0].entity_id, "tsmc")

    def test_path_explanation(self) -> None:
        """to_explanation must produce a human-readable chain-of-thought string."""
        path = self.engine.find_shortest_path("jensen", "tsmc", direction="out")
        self.assertIsNotNone(path)
        explanation = path.to_explanation()
        self.assertIn("Reasoning Chain (2 hops", explanation)
        self.assertIn("Jensen Huang --[LEADS", explanation)
        self.assertIn("NVIDIA --[PARTNERS_WITH", explanation)
        self.assertIn("TSMC", explanation)


if __name__ == "__main__":
    unittest.main()
