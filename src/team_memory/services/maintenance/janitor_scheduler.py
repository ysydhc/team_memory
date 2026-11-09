"""Janitor scheduler service for automated background cleanup.

Provides scheduled execution of MemoryJanitor operations in the Web process
using asyncio tasks for non-blocking background maintenance.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from team_memory.services.maintenance.janitor import MemoryJanitor

logger = logging.getLogger("team_memory.services.maintenance.janitor_scheduler")

DEFAULT_JANITOR_STATE_FILE = Path.home() / ".cache" / "tm-pipeline" / "last_janitor.txt"


class JanitorScheduler:
    """Background scheduler for automated memory janitor operations.

    Runs MemoryJanitor operations on a configurable interval in the Web/Daemon process
    using asyncio tasks to avoid blocking the main application.
    """

    def __init__(
        self,
        janitor: MemoryJanitor,
        config: Any = None,
        *,
        state_file: Path | str | None = None,
        initial_delay_seconds: float = 15.0,
    ):
        """Initialize scheduler with janitor instance and configuration.

        Args:
            janitor: MemoryJanitor instance to run operations
            config: Configuration object with scheduler settings
            state_file: Optional path to persist last-run ISO timestamp across restarts
            initial_delay_seconds: Delay before running an overdue janitor pass on startup
        """
        self._janitor = janitor
        self._config = config
        self._task: asyncio.Task | None = None
        self._running = False
        self._state_file: Path | None = Path(state_file) if state_file else None
        self._initial_delay_seconds = initial_delay_seconds

    def enable_state_persistence(
        self,
        state_file: Path | str | None = None,
        initial_delay_seconds: float = 15.0,
    ) -> JanitorScheduler:
        """Enable persistent last-run timestamp tracking across restarts."""
        self._state_file = Path(state_file) if state_file else DEFAULT_JANITOR_STATE_FILE
        self._initial_delay_seconds = initial_delay_seconds
        return self

    def _read_last_run(self) -> datetime | None:
        """Read last janitor run timestamp from state_file if enabled."""
        if self._state_file is None:
            return None
        try:
            if not self._state_file.exists():
                return None
            raw = self._state_file.read_text(encoding="utf-8").strip()
            if not raw:
                return None
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception as e:
            logger.warning("Failed to read last_janitor state file: %s", e)
            return None

    def _write_last_run(self, ts: datetime | None = None) -> None:
        """Write current UTC timestamp to state_file if enabled."""
        if self._state_file is None:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            now = ts or datetime.now(timezone.utc)
            self._state_file.write_text(now.isoformat(), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write last_janitor state file: %s", e)

    def _compute_next_wait_seconds(self, interval_seconds: float) -> float:
        """Compute how long to sleep before the next janitor execution."""
        if self._state_file is None:
            return interval_seconds
        last_run = self._read_last_run()
        if last_run is None:
            return min(self._initial_delay_seconds, interval_seconds)
        elapsed = (datetime.now(timezone.utc) - last_run).total_seconds()
        if elapsed >= interval_seconds:
            return min(self._initial_delay_seconds, interval_seconds)
        return max(5.0, interval_seconds - elapsed)

    def _get_config(self, key: str, default=None):
        """Get configuration value with fallback to default."""
        if self._config is None:
            return default
        try:
            return getattr(self._config, key)
        except AttributeError:
            return default

    async def start(self) -> None:
        """Start the background janitor scheduler.

        Creates an asyncio task that runs the janitor loop in the background.
        Safe to call multiple times - will not create duplicate tasks.
        """
        if self._running:
            logger.warning("Janitor scheduler is already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())

        interval_hours = self._get_config("interval_hours", 24)
        logger.info("Janitor scheduler started with %d hour interval", interval_hours)

    async def stop(self) -> None:
        """Stop the background janitor scheduler.

        Sets the running flag to False and cancels the background task.
        Waits for the task to complete cancellation.
        """
        if not self._running:
            logger.warning("Janitor scheduler is not running")
            return

        logger.info("Stopping janitor scheduler...")
        self._running = False

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("Janitor scheduler task cancelled successfully")

        self._task = None
        logger.info("Janitor scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop that runs janitor operations periodically.

        Sleeps for the configured interval, then runs all janitor operations.
        Continues until _running is set to False.
        """
        interval_hours = self._get_config("interval_hours", 24)
        interval_seconds = interval_hours * 3600

        logger.info("Janitor scheduler loop started (interval: %d hours)", interval_hours)

        try:
            while self._running:
                wait_seconds = self._compute_next_wait_seconds(interval_seconds)
                await asyncio.sleep(wait_seconds)

                if not self._running:
                    break

                if self._state_file is not None:
                    last_run = self._read_last_run()
                    if last_run is not None:
                        elapsed = (datetime.now(timezone.utc) - last_run).total_seconds()
                        if elapsed < interval_seconds * 0.9:
                            continue

                # Run janitor operations
                logger.info("Running scheduled janitor operations...")
                try:
                    # Run for all projects (project=None)
                    result = await self._janitor.run_all(project=None)
                    self._write_last_run()

                    # Log summary of operations
                    operations = result.get("operations", {})
                    summary_parts = []

                    if "score_decay" in operations:
                        decay_count = operations["score_decay"].get("updated_count", 0)
                        summary_parts.append(f"decay: {decay_count}")

                    if "outdated_sweep" in operations:
                        sweep_found = operations["outdated_sweep"].get("found_count", 0)
                        sweep_deleted = operations["outdated_sweep"].get("deleted_count", 0)
                        summary_parts.append(f"outdated: {sweep_found} found, {sweep_deleted} deleted")

                    if "soft_deleted_purge" in operations:
                        purge_count = operations["soft_deleted_purge"].get("purged_count", 0)
                        summary_parts.append(f"purged: {purge_count}")

                    if "draft_expiration" in operations:
                        draft_count = operations["draft_expiration"].get("expired_count", 0)
                        summary_parts.append(f"drafts: {draft_count}")

                    if "personal_memory_pruning" in operations:
                        memory_count = operations["personal_memory_pruning"].get("pruned_count", 0)
                        dedup_static = operations["personal_memory_pruning"].get("deduped_static_count", 0)
                        summary_parts.append(
                            f"personal: {memory_count}"
                            + (f" (dedup_static: {dedup_static})" if dedup_static else "")
                        )

                    if "archive_maintenance" in operations:
                        arch = operations["archive_maintenance"]
                        summary_parts.append(
                            f"archives: norm={arch.get('normalized_types_count', 0)},"
                            f"dedup={arch.get('deduped_count', 0)},"
                            f"promoted={arch.get('promoted_drafts_count', 0)}"
                        )

                    summary = ", ".join(summary_parts) if summary_parts else "no changes"
                    logger.info("Scheduled janitor run completed: %s", summary)

                except Exception as e:
                    logger.error("Scheduled janitor run failed: %s", e, exc_info=True)
                    if self._state_file is not None:
                        await asyncio.sleep(min(300.0, interval_seconds))

        except asyncio.CancelledError:
            logger.debug("Janitor scheduler loop cancelled")
            raise
        except Exception as e:
            logger.error("Unexpected error in janitor scheduler loop: %s", e, exc_info=True)
        finally:
            logger.info("Janitor scheduler loop ended")

    def is_running(self) -> bool:
        """Check if the scheduler is currently running.

        Returns:
            True if the scheduler is running, False otherwise
        """
        return self._running and self._task is not None and not self._task.done()
