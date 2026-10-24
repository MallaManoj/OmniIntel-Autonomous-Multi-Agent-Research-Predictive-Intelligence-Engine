"""Unit tests for Knowledge Graph schema definitions and Pydantic v2 validation.

Validates slugification, entity alias normalization, relation edge attributes,
fact triple Cypher generation, confidence scoring, and serialization.
Compatible with standard library `unittest` and `pytest`.
"""

import json
import unittest

from pydantic import ValidationError

from src.knowledge_graph.schema import (
    Entity,
    EntityType,
    FactTriple,
    GraphQuery,
    Relation,
    RelationType,
    SubgraphResult,
    slugify_id,
)


class TestKnowledgeGraphSchema(unittest.TestCase):
    """Test suite covering Entity, Relation, FactTriple and Subgraph schemas."""

    def test_slugify_id(self) -> None:
        """Slugify must convert text to clean lowercase underscore slugs."""
        self.assertEqual(slugify_id("NVIDIA Corp."), "nvidia_corp")
        self.assertEqual(slugify_id("  Jensen Huang  "), "jensen_huang")
        self.assertEqual(slugify_id("OpenAI / ChatGPT-4"), "openai_chatgpt_4")
        self.assertEqual(slugify_id("H100 & B200 Tensor Core"), "h100_b200_tensor_core")

    def test_entity_creation_and_validation(self) -> None:
        """Entity creation must normalize slug, deduplicate aliases, and sort them."""
        entity = Entity(
            entity_id="  NVIDIA Corp.  ",
            name="NVIDIA Corporation",
            entity_type=EntityType.ORGANIZATION,
            aliases=["NVDA", "nvidia", " Nvidia ", "nvda"],
            properties={"ticker": "NVDA", "exchange": "NASDAQ"},
            confidence=0.98,
        )

        self.assertEqual(entity.entity_id, "nvidia_corp")
        self.assertEqual(entity.name, "NVIDIA Corporation")
        self.assertEqual(entity.entity_type, EntityType.ORGANIZATION)
        # Deduplicated and alphabetically sorted in lowercase
        self.assertEqual(entity.aliases, ["nvda", "nvidia"])
        self.assertEqual(entity.properties["ticker"], "NVDA")
        self.assertEqual(entity.confidence, 0.98)

        # all_surface_forms includes lowercased canonical name + aliases
        forms = entity.all_surface_forms()
        self.assertIn("nvidia corporation", forms)
        self.assertIn("nvidia", forms)
        self.assertIn("nvda", forms)

    def test_entity_invalid_inputs(self) -> None:
        """Entity must reject invalid confidence, empty name, or empty slug."""
        with self.assertRaises(ValidationError):
            Entity(
                entity_id="valid_id",
                name="",  # empty name
                entity_type=EntityType.COMPANY if hasattr(EntityType, "COMPANY") else EntityType.ORGANIZATION,
            )

        with self.assertRaises(ValidationError):
            Entity(
                entity_id="   ",  # invalid slug
                name="Valid Name",
                entity_type=EntityType.ORGANIZATION,
            )

        with self.assertRaises(ValidationError):
            Entity(
                entity_id="valid_id",
                name="Valid Name",
                entity_type=EntityType.ORGANIZATION,
                confidence=1.5,  # must be <= 1.0
            )

    def test_relation_creation_and_validation(self) -> None:
        """Relation must validate relation_id and positive weights."""
        rel = Relation(
            relation_id="rel_101",
            relation_type=RelationType.DEVELOPS,
            weight=1.5,
            directed=True,
            properties={"product_line": "Blackwell"},
        )
        self.assertEqual(rel.relation_id, "rel_101")
        self.assertEqual(rel.relation_name, "DEVELOPS")
        self.assertEqual(rel.weight, 1.5)
        self.assertTrue(rel.directed)

        # Custom string relation type
        custom_rel = Relation(
            relation_id="rel_custom",
            relation_type="PIONEERED",
            weight=1.0,
        )
        self.assertEqual(custom_rel.relation_name, "PIONEERED")

    def test_relation_invalid_weight(self) -> None:
        """Edge weight must be strictly positive (> 0.0)."""
        with self.assertRaises(ValidationError):
            Relation(
                relation_id="rel_zero",
                relation_type=RelationType.ACQUIRED,
                weight=0.0,
            )
        with self.assertRaises(ValidationError):
            Relation(
                relation_id="rel_neg",
                relation_type=RelationType.ACQUIRED,
                weight=-1.0,
            )

    def test_fact_triple_properties_and_cypher(self) -> None:
        """FactTriple must correctly link subject, relation, and object with Cypher export."""
        subject = Entity(
            entity_id="jensen_huang",
            name="Jensen Huang",
            entity_type=EntityType.PERSON,
        )
        relation = Relation(
            relation_id="leads_rel",
            relation_type=RelationType.LEADS,
            weight=1.0,
            directed=True,
        )
        obj = Entity(
            entity_id="nvidia",
            name="NVIDIA Corporation",
            entity_type=EntityType.ORGANIZATION,
        )

        triple = FactTriple(
            triple_id="triple_001",
            subject=subject,
            relation=relation,
            object=obj,
            evidence_text="Jensen Huang co-founded and leads NVIDIA as Chief Executive Officer.",
            source_doc_id="sec_10k_2024",
            confidence=0.99,
        )

        self.assertEqual(triple.subject_id, "jensen_huang")
        self.assertEqual(triple.object_id, "nvidia")
        self.assertEqual(triple.to_tuple(), ("Jensen Huang", "LEADS", "NVIDIA Corporation"))

        # Cypher string export
        cypher = triple.to_cypher()
        self.assertIn("(:PERSON {name: 'Jensen Huang', id: 'jensen_huang'})", cypher)
        self.assertIn("-[:LEADS {weight: 1.0}]->", cypher)
        self.assertIn("(:ORGANIZATION {name: 'NVIDIA Corporation', id: 'nvidia'})", cypher)

        # Human-readable text format
        text_rep = triple.to_text()
        self.assertIn("Jensen Huang [LEADS] NVIDIA Corporation", text_rep)
        self.assertIn("0.99", text_rep)

    def test_graph_query_validation(self) -> None:
        """GraphQuery must validate max_hops bounds [1, 10] and confidence [0.0, 1.0]."""
        query = GraphQuery(
            source_entity_id="nvidia",
            max_hops=3,
            relation_types=[RelationType.COMPETES_WITH, RelationType.PARTNERS_WITH],
            min_confidence=0.7,
        )
        self.assertEqual(query.source_entity_id, "nvidia")
        self.assertEqual(query.max_hops, 3)

        with self.assertRaises(ValidationError):
            GraphQuery(source_entity_id="nvidia", max_hops=0)  # max_hops < 1
        with self.assertRaises(ValidationError):
            GraphQuery(source_entity_id="nvidia", max_hops=15)  # max_hops > 10

    def test_subgraph_result_summary(self) -> None:
        """SubgraphResult must format multi-line textual summary for agents."""
        root = Entity(entity_id="alphabet", name="Alphabet Inc.", entity_type=EntityType.ORGANIZATION)
        google = Entity(entity_id="google", name="Google LLC", entity_type=EntityType.ORGANIZATION)
        rel = Relation(relation_id="r1", relation_type=RelationType.SUBSIDIARY_OF)
        triple = FactTriple(triple_id="t1", subject=google, relation=rel, object=root)

        result = SubgraphResult(
            root_entity=root,
            entities=[root, google],
            triples=[triple],
            paths=[["google", "alphabet"]],
        )

        summary = result.summary()
        self.assertIn("Subgraph rooted at 'Alphabet Inc.'", summary)
        self.assertIn("Google LLC [SUBSIDIARY_OF] Alphabet Inc.", summary)

    def test_pydantic_json_serialization_roundtrip(self) -> None:
        """Entity and FactTriple must serialize to and deserialize from JSON cleanly."""
        entity = Entity(
            entity_id="amd",
            name="Advanced Micro Devices",
            entity_type=EntityType.ORGANIZATION,
            aliases=["AMD"],
            properties={"founding_year": 1969},
        )

        json_str = entity.model_dump_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["entity_id"], "amd")
        self.assertEqual(parsed["name"], "Advanced Micro Devices")

        # Reload from dict
        reloaded = Entity.model_validate_json(json_str)
        self.assertEqual(reloaded, entity)


if __name__ == "__main__":
    unittest.main()
