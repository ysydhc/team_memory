"""Extension framework — auto-load Python modules from an extensions directory.

Extensions are Python files in the ``extensions/`` directory (or a custom path).
Each file must define a ``register(ctx)`` function that receives an
:class:`ExtensionContext` with references to core services.

Example extension (``extensions/example_webhook_logger.py``)::

    def register(ctx):
        async def on_created(payload):
            ctx.logger.info("扩展: 新经验创建 %s", payload.get("id"))
        ctx.event_bus.on("experience.created", on_created)

Extension loading is best-effort: a failure in one extension is logged
but does not prevent the application from starting.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from team_memory.config import Settings
    from team_memory.schemas import SchemaRegistry
    from team_memory.services.event_bus import EventBus

logger = logging.getLogger("team_memory.extensions")


@dataclass
class ExtensionContext:
    """Context object passed to every extension's ``register()`` function.

    Attributes:
        event_bus: The global event bus (subscribe / emit events).
        schema_registry: Read-only schema registry for type metadata.
        settings: Application settings (read-only reference).
        logger: A pre-configured logger extensions can use.
    """

    event_bus: "EventBus | None" = None
    schema_registry: "SchemaRegistry | None" = None
    settings: "Settings | None" = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("team_memory.ext"))
    _loaded_extensions: list[str] = field(default_factory=list)


def load_extensions(
    ctx: ExtensionContext,
    ext_dir: str | Path = "extensions",
) -> int:
    """Scan ``ext_dir`` for ``*.py`` files and call ``register(ctx)`` in each.

    Args:
        ctx: The extension context to inject.
        ext_dir: Directory to scan (relative or absolute). Defaults to ``extensions/``.

    Returns:
        Number of successfully loaded extensions.
    """
    ext_path = Path(ext_dir)
    if not ext_path.is_dir():
        logger.debug("Extensions directory '%s' does not exist — skipping.", ext_dir)
        return 0

    loaded = 0
    for py_file in sorted(ext_path.glob("*.py")):
        if py_file.name.startswith("_"):
            continue  # skip __init__.py and private modules
        name = py_file.stem
        module_name = f"team_memory_ext_{name}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                logger.warning("Cannot create module spec for %s", py_file)
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            register_fn = getattr(module, "register", None)
            if register_fn is None:
                logger.warning(
                    "Extension '%s' has no register() function — skipping.",
                    py_file.name,
                )
                continue

            register_fn(ctx)
            ctx._loaded_extensions.append(name)
            loaded += 1
            logger.info("Loaded extension: %s", py_file.name)

        except Exception:
            logger.warning(
                "Failed to load extension '%s'", py_file.name, exc_info=True
            )

    if loaded > 0:
        logger.info("Extensions loaded: %d of %d", loaded, len(list(ext_path.glob("*.py"))))

    return loaded
