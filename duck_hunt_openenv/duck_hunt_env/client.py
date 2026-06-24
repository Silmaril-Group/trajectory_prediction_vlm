"""Duck Hunt OpenEnv Client"""

import requests
from typing import Self

from .models import ShootAction, DuckHuntObservation


class DuckHuntEnv:
    """Client wrapper for Duck Hunt OpenEnv."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    @classmethod
    def from_docker_image(cls, image_name: str, port: int = 8000) -> Self:
        """Connect to containerized environment."""
        # Assumes container is running and mapped to localhost
        return cls(f"http://localhost:{port}")

    @classmethod
    def from_local(cls, host: str = "localhost", port: int = 8000) -> Self:
        """Connect to locally running server."""
        return cls(f"http://{host}:{port}")

    def health(self) -> dict:
        """Check server health."""
        response = self._session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def reset(self) -> DuckHuntObservation:
        """Reset the environment and return initial observation."""
        response = self._session.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return DuckHuntObservation.from_dict(response.json())

    def step(self, action: ShootAction) -> DuckHuntObservation:
        """Execute action and return observation."""
        response = self._session.post(
            f"{self.base_url}/step",
            json=action.to_dict(),
        )
        response.raise_for_status()
        return DuckHuntObservation.from_dict(response.json())

    def close(self):
        """Close the client session."""
        self._session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
