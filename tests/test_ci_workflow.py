"""Unit tests verifying GitHub Actions CI/CD pipeline configuration and workflow integrity."""

from pathlib import Path
import unittest
import yaml


class TestCIWorkflowConfiguration(unittest.TestCase):
    """Test suite validating .github/workflows/ci.yml syntax, job dependencies, and triggers."""

    def setUp(self) -> None:
        self.root_dir = Path(__file__).resolve().parent.parent
        self.workflow_file = self.root_dir / ".github" / "workflows" / "ci.yml"

    def test_workflow_file_exists_and_valid_yaml(self) -> None:
        """ci.yml must exist and be valid parseable YAML."""
        self.assertTrue(self.workflow_file.is_file(), "ci.yml workflow file is missing")
        with open(self.workflow_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("name"), "OmniIntel CI/CD Pipeline")

    def test_workflow_triggers_and_concurrency(self) -> None:
        """Workflow must trigger on push/PR and define cancel-in-progress concurrency."""
        with open(self.workflow_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Triggers (handle YAML 1.1 boolean key 'on' / True)
        triggers = data.get("on") or data.get(True) or {}
        self.assertIn("push", triggers)
        self.assertIn("pull_request", triggers)
        self.assertIn("workflow_dispatch", triggers)

        # Concurrency
        conc = data.get("concurrency", {})
        self.assertTrue(conc.get("cancel-in-progress"))
        self.assertIn("group", conc)

    def test_job_topology_and_dependencies(self) -> None:
        """Workflow must define quality, test matrix, docker build, and compose verification jobs."""
        with open(self.workflow_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        jobs = data.get("jobs", {})
        expected_jobs = [
            "quality-and-lint",
            "test-and-coverage",
            "docker-build-and-verify",
            "docker-compose-validate",
        ]
        for job_name in expected_jobs:
            self.assertIn(job_name, jobs, f"Job {job_name} missing from CI workflow")

        # Check job dependencies (DAG)
        self.assertEqual(jobs["test-and-coverage"].get("needs"), "quality-and-lint")
        docker_needs = jobs["docker-build-and-verify"].get("needs", [])
        self.assertIn("quality-and-lint", docker_needs)
        self.assertIn("test-and-coverage", docker_needs)

    def test_python_matrix_strategy(self) -> None:
        """test-and-coverage job must test across Python 3.11 and 3.12."""
        with open(self.workflow_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        test_job = data["jobs"]["test-and-coverage"]
        strategy = test_job.get("strategy", {})
        matrix = strategy.get("matrix", {})
        versions = matrix.get("python-version", [])

        self.assertIn("3.11", versions)
        self.assertIn("3.12", versions)


if __name__ == "__main__":
    unittest.main()
