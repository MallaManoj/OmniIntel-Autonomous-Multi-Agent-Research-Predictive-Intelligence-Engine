"""Unit tests verifying production multi-stage Dockerfile and .dockerignore configuration."""

import os
from pathlib import Path
import unittest


class TestDockerConfiguration(unittest.TestCase):
    """Test suite validating Dockerfile syntax, multi-stage structure, and security controls."""

    def setUp(self) -> None:
        self.root_dir = Path(__file__).resolve().parent.parent
        self.dockerfile_path = self.root_dir / "Dockerfile"
        self.dockerignore_path = self.root_dir / ".dockerignore"
        self.requirements_path = self.root_dir / "requirements.txt"

    def test_dockerfile_existence_and_multistage(self) -> None:
        """Dockerfile must exist and implement multi-stage build pattern."""
        self.assertTrue(self.dockerfile_path.is_file(), "Dockerfile missing from root")
        content = self.dockerfile_path.read_text(encoding="utf-8")

        # Multi-stage targets
        self.assertIn("FROM python:", content)
        self.assertIn("AS builder", content)
        self.assertIn("AS runner", content)

    def test_dockerfile_security_and_non_root_user(self) -> None:
        """Dockerfile must create an unprivileged user (never run as root) for CIS compliance."""
        content = self.dockerfile_path.read_text(encoding="utf-8")

        # Non-root user creation and switch
        self.assertTrue(
            "useradd" in content or "adduser" in content,
            "Dockerfile should create an unprivileged system user",
        )
        self.assertIn("USER omniintel", content)
        self.assertIn("chown", content)

    def test_dockerfile_healthcheck_and_ports(self) -> None:
        """Dockerfile must declare healthcheck and port exposure."""
        content = self.dockerfile_path.read_text(encoding="utf-8")

        self.assertIn("EXPOSE 8000", content)
        self.assertIn("HEALTHCHECK", content)
        self.assertIn("/health", content)

    def test_dockerfile_runtime_entrypoint(self) -> None:
        """Dockerfile CMD entrypoint must run production uvicorn ASGI server."""
        content = self.dockerfile_path.read_text(encoding="utf-8")

        self.assertIn("uvicorn", content)
        self.assertIn("src.api.app:app", content)
        self.assertIn("0.0.0.0", content)
        self.assertIn("8000", content)

    def test_dockerignore_rules(self) -> None:
        """Dockerignore must exclude virtual environments, git history, and caches."""
        self.assertTrue(self.dockerignore_path.is_file(), ".dockerignore missing from root")
        content = self.dockerignore_path.read_text(encoding="utf-8")

        required_patterns = [
            ".venv",
            "__pycache__",
            ".git",
            "*.py[cod]",
            ".env",
        ]
        for pattern in required_patterns:
            self.assertIn(pattern, content, f"Pattern {pattern} missing from .dockerignore")

    def test_requirements_file_production_deps(self) -> None:
        """requirements.txt must specify core production dependencies."""
        self.assertTrue(self.requirements_path.is_file())
        reqs = self.requirements_path.read_text(encoding="utf-8")

        self.assertIn("torch", reqs)
        self.assertIn("fastapi", reqs)
        self.assertIn("uvicorn", reqs)
        self.assertIn("websockets", reqs)
        self.assertIn("pydantic", reqs)


if __name__ == "__main__":
    unittest.main()
