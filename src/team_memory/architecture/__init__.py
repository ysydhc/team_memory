"""Architecture (code graph) provider abstraction and factory."""

from team_memory.architecture.base import ArchitectureProvider
from team_memory.architecture.factory import get_provider
from team_memory.architecture.gitnexus_provider import GitNexusProvider

__all__ = ["ArchitectureProvider", "get_provider", "GitNexusProvider"]
