"""Factory to create ArchitectureProvider from config.

Returns the provider instance for the configured backend (gitnexus | builtin).
When provider is builtin, returns None (not implemented yet).
"""

from __future__ import annotations

from team_memory.architecture.base import ArchitectureProvider
from team_memory.architecture.gitnexus_provider import GitNexusProvider
from team_memory.config import ArchitectureConfig


def get_provider(arch_config: ArchitectureConfig) -> ArchitectureProvider | None:
    """Create an ArchitectureProvider from architecture config.

    Args:
        arch_config: Settings.architecture (provider, gitnexus, builtin).

    Returns:
        Provider instance for gitnexus; None for builtin (reserved).
    """
    if arch_config.provider == "gitnexus":
        return GitNexusProvider(arch_config.gitnexus)
    if arch_config.provider == "builtin":
        # Reserved; no BuiltinProvider implementation yet.
        return None
    return None
