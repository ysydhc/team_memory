"""Example extension — logs experience lifecycle events.

This file demonstrates how to write a team_memory extension.
Place any ``*.py`` file in the ``extensions/`` directory and define a
``register(ctx)`` function.  The framework calls it automatically on startup.

To disable an extension without deleting it, rename it to start with ``_``
(e.g. ``_example_webhook_logger.py``).
"""


def register(ctx):
    """Register event handlers on the global EventBus.

    Parameters
    ----------
    ctx : team_memory.extensions.ExtensionContext
        Provides ``event_bus``, ``schema_registry``, ``settings``, and ``logger``.
    """
    if ctx.event_bus is None:
        ctx.logger.debug("EventBus not available — example extension skipped.")
        return

    async def on_experience_created(payload):
        exp_id = payload.get("id", "unknown")
        title = payload.get("title", "")
        ctx.logger.info("[example-ext] 新经验已创建: %s — %s", exp_id, title)

    async def on_experience_published(payload):
        exp_id = payload.get("id", "unknown")
        ctx.logger.info("[example-ext] 经验已发布: %s", exp_id)

    ctx.event_bus.on("experience.created", on_experience_created)
    ctx.event_bus.on("experience.published", on_experience_published)

    ctx.logger.info("[example-ext] 示例扩展已加载 — 监听 created / published 事件")
