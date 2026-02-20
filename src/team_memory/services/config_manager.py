"""Configuration manager — runtime modification and YAML persistence.

Provides read/write access to the application configuration at runtime,
with the ability to persist changes back to config.yaml. Interface is
designed to support future DB-backed configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from team_memory.config import Settings

logger = logging.getLogger("team_memory.config_manager")


class ConfigManager:
    """Manages runtime configuration with optional YAML persistence."""

    def __init__(self, settings: Settings, config_path: str | None = None):
        self._settings = settings
        self._config_path = config_path or self._detect_config_path()

    def get(self, section: str | None = None) -> dict[str, Any]:
        """Read configuration as a dict.

        Args:
            section: Optional top-level section name (e.g. "search", "cache").
                     If None, returns the full configuration.
        """
        full = self._settings.model_dump()
        if section is None:
            return full
        return full.get(section, {})

    def update(self, section: str, values: dict[str, Any]) -> dict[str, Any]:
        """Update a configuration section at runtime.

        Modifies the in-memory Settings object. Call ``persist()`` to
        write changes to YAML.

        Returns:
            The updated section dict.
        """
        if not hasattr(self._settings, section):
            raise KeyError(f"Unknown config section: {section}")

        sub = getattr(self._settings, section)
        if hasattr(sub, "model_copy"):
            updated = sub.model_copy(update=values)
            setattr(self._settings, section, updated)
        else:
            for k, v in values.items():
                if hasattr(sub, k):
                    setattr(sub, k, v)

        logger.info("Config section '%s' updated at runtime", section)
        return self.get(section)

    def persist(self) -> Path:
        """Write the current settings to YAML."""
        path = Path(self._config_path)
        data = self._settings.model_dump(exclude_defaults=False)

        _remove_none_values(data)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info("Config persisted to %s", path)
        return path

    @staticmethod
    def _detect_config_path() -> str:
        """Try common config file locations."""
        candidates = [
            "config.yaml",
            "config.yml",
            "config.production.yaml",
        ]
        for name in candidates:
            if Path(name).exists():
                return name
        return "config.yaml"


def _remove_none_values(d: dict) -> None:
    """Recursively strip None values from a dict for cleaner YAML output."""
    keys_to_remove = []
    for k, v in d.items():
        if v is None:
            keys_to_remove.append(k)
        elif isinstance(v, dict):
            _remove_none_values(v)
    for k in keys_to_remove:
        del d[k]
