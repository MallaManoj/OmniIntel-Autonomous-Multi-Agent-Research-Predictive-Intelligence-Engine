"""Unit tests verifying production Docker Compose multi-service orchestration stack."""

from pathlib import Path
import unittest
import yaml


class TestDockerComposeOrchestration(unittest.TestCase):
    """Test suite validating docker-compose.yml configuration, dependencies, and network isolation."""

    def setUp(self) -> None:
        self.root_dir = Path(__file__).resolve().parent.parent
        self.compose_file = self.root_dir / "docker-compose.yml"
        self.ui_file = self.root_dir / "src" / "ui" / "dashboard.py"

    def test_compose_file_exists_and_valid_yaml(self) -> None:
        """docker-compose.yml must exist and be valid parseable YAML."""
        self.assertTrue(self.compose_file.is_file(), "docker-compose.yml is missing")
        with open(self.compose_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertIsInstance(data, dict)
        self.assertIn("services", data)
        self.assertIn("networks", data)
        self.assertIn("volumes", data)

    def test_api_service_configuration(self) -> None:
        """omniintel-api service must expose port 8000, declare healthcheck, and bind volumes."""
        with open(self.compose_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        services = data.get("services", {})
        self.assertIn("omniintel-api", services)
        api = services["omniintel-api"]

        # Port mapping
        self.assertIn("8000:8000", api.get("ports", []))

        # Healthcheck configuration
        hc = api.get("healthcheck", {})
        self.assertTrue(any("/health" in str(cmd) for cmd in hc.get("test", [])))
        self.assertIn("interval", hc)
        self.assertIn("retries", hc)

        # Volumes for persistent indices and model checkpoints
        vols = api.get("volumes", [])
        self.assertTrue(any("omniintel-data" in str(v) for v in vols))
        self.assertTrue(any("omniintel-checkpoints" in str(v) for v in vols))

        # Network attachment
        self.assertIn("omniintel-net", api.get("networks", []))

    def test_ui_service_configuration(self) -> None:
        """omniintel-ui service must depend on healthy API and expose port 8501."""
        with open(self.compose_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        services = data.get("services", {})
        self.assertIn("omniintel-ui", services)
        ui = services["omniintel-ui"]

        # Port mapping
        self.assertIn("8501:8501", ui.get("ports", []))

        # Dependency with health condition
        dep = ui.get("depends_on", {})
        self.assertIn("omniintel-api", dep)
        self.assertEqual(dep["omniintel-api"]["condition"], "service_healthy")

        # Network and Environment
        self.assertIn("omniintel-net", ui.get("networks", []))
        env_vars = ui.get("environment", [])
        self.assertTrue(any("omniintel-api:8000" in str(e) for e in env_vars))

    def test_network_and_volume_declarations(self) -> None:
        """Compose file must declare dedicated bridge network and local storage volumes."""
        with open(self.compose_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Bridge network
        nets = data.get("networks", {})
        self.assertIn("omniintel-net", nets)
        self.assertEqual(nets["omniintel-net"].get("driver"), "bridge")

        # Storage volumes
        vols = data.get("volumes", {})
        self.assertIn("omniintel-data", vols)
        self.assertIn("omniintel-checkpoints", vols)

    def test_ui_dashboard_file_integrity(self) -> None:
        """src/ui/dashboard.py must exist and compile cleanly."""
        self.assertTrue(self.ui_file.is_file(), "src/ui/dashboard.py is missing")
        content = self.ui_file.read_text(encoding="utf-8")
        compile(content, str(self.ui_file), "exec")
        self.assertIn("OmniIntel", content)


if __name__ == "__main__":
    unittest.main()
