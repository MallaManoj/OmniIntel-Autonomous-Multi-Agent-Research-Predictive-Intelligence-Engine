"""Unit tests verifying Streamlit UI helper utilities, DAG Graphviz generator, and subgraph renderers."""

from pathlib import Path
import unittest

from src.ui.dashboard import build_dag_dot_graph, build_subgraph_dot_graph


class TestUIDashboardComponents(unittest.TestCase):
    """Test suite validating visual DAG builders, DOT graph generation, and UI helpers."""

    def test_build_dag_dot_graph_empty_input(self) -> None:
        """Empty plan dictionary should return fallback DOT diagram."""
        dot = build_dag_dot_graph({})
        self.assertIn("digraph", dot)
        self.assertIn("No DAG Schedule Available", dot)

    def test_build_dag_dot_graph_structured_plan(self) -> None:
        """Structured plan dictionary should render valid Graphviz DAG with task nodes and edges."""
        sample_plan = {
            "complexity": "COMPLEX",
            "total_tasks": 4,
            "total_stages": 3,
            "batches": [["task_0"], ["task_1", "task_2"], ["task_3"]],
            "tasks": {
                "task_0": {
                    "name": "Decompose Directives",
                    "agent_role": "PlannerAgent",
                    "dependencies": [],
                },
                "task_1": {
                    "name": "Retrieve SEC Filings",
                    "agent_role": "ResearchAgent",
                    "dependencies": ["task_0"],
                },
                "task_2": {
                    "name": "Compute Financial Ratios",
                    "agent_role": "QuantAgent",
                    "dependencies": ["task_0"],
                },
                "task_3": {
                    "name": "Audit & Ground Claims",
                    "agent_role": "CriticAgent",
                    "dependencies": ["task_1", "task_2"],
                },
            },
        }

        dot = build_dag_dot_graph(sample_plan)
        self.assertIn("digraph KahnExecutionDAG", dot)
        self.assertIn("Decompose Directives", dot)
        self.assertIn("PlannerAgent", dot)
        self.assertIn("ResearchAgent", dot)
        self.assertIn("QuantAgent", dot)
        self.assertIn("CriticAgent", dot)

        # Verify directed edges representing dependencies
        self.assertIn('"task_0" -> "task_1";', dot)
        self.assertIn('"task_0" -> "task_2";', dot)
        self.assertIn('"task_1" -> "task_3";', dot)
        self.assertIn('"task_2" -> "task_3";', dot)

    def test_build_subgraph_dot_graph_empty_input(self) -> None:
        """Empty subgraph dictionary should return fallback DOT diagram."""
        dot = build_subgraph_dot_graph({})
        self.assertIn("digraph", dot)
        self.assertIn("No Subgraph Available", dot)

    def test_build_subgraph_dot_graph_structured(self) -> None:
        """Structured subgraph should render egocentric nodes and labeled relationship edges."""
        sample_subgraph = {
            "root_entity_id": "nvidia",
            "entities": [
                {"entity_id": "nvidia", "name": "NVIDIA", "entity_type": "ORGANIZATION"},
                {"entity_id": "tsmc", "name": "TSMC", "entity_type": "ORGANIZATION"},
            ],
            "triples": [
                {
                    "subject": {"entity_id": "nvidia"},
                    "object": {"entity_id": "tsmc"},
                    "relation": {"relation_name": "MANUFACTURED_BY"},
                    "confidence": 0.95,
                }
            ],
        }

        dot = build_subgraph_dot_graph(sample_subgraph)
        self.assertIn("digraph EgocentricTopology", dot)
        self.assertIn('"nvidia"', dot)
        self.assertIn('"tsmc"', dot)
        self.assertIn("MANUFACTURED_BY", dot)
        self.assertIn("0.95", dot)

    def test_dashboard_script_syntax_and_compilation(self) -> None:
        """src/ui/dashboard.py must compile without AST syntax errors."""
        ui_path = Path(__file__).resolve().parent.parent / "src" / "ui" / "dashboard.py"
        self.assertTrue(ui_path.is_file())
        content = ui_path.read_text(encoding="utf-8")
        compiled = compile(content, str(ui_path), "exec")
        self.assertIsNotNone(compiled)


if __name__ == "__main__":
    unittest.main()
