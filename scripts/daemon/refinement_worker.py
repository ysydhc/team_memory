"""Background worker that refines drafts via LLM and updates Experience records.

Scans the DraftBuffer for drafts with status='needs_refinement',
calls an LLM (via LiteLLM proxy or Ollama) to extract structured fields,
publishes the draft, then updates the published Experience with refined data.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from daemon.config import DaemonConfig, RefinementSettings
from daemon.draft_buffer import DraftBuffer
from daemon.tm_sink import TMSink
from team_memory.services.llm_parser import LLMParseError, parse_content_openai

logger = logging.getLogger("daemon.refinement_worker")

# Maximum refinement attempts before giving up on a draft
_MAX_REFINE_ATTEMPTS = 3


class RefinementWorker:
    """Periodic background worker for LLM-driven draft refinement.

    Usage::

        worker = RefinementWorker(config, buf, sink)
        task = asyncio.create_task(worker.run())
        ...
        worker.stop()
        await task
    """

    def __init__(
        self,
        config: DaemonConfig,
        buf: DraftBuffer,
        sink: TMSink,
    ) -> None:
        self._cfg: RefinementSettings = config.refinement
        self._buf = buf
        self._sink = sink
        self._running = False
        self._task: asyncio.Task | None = None
        # In-memory retry counter: draft_id → attempt count
        self._retry_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> asyncio.Task:
        """Start the background worker loop."""
        if not self._cfg.enabled:
            logger.info("RefinementWorker is disabled in config")
            # Return a no-op task so callers can always await it
            self._task = asyncio.create_task(self._noop())
            return self._task

        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "RefinementWorker started (interval=%ss, model=%s, provider=%s)",
            self._cfg.scan_interval_seconds,
            self._cfg.model,
            self._cfg.provider,
        )
        return self._task

    async def _noop(self) -> None:
        """No-op task when worker is disabled."""
        pass

    def stop(self) -> None:
        """Signal the worker to stop gracefully."""
        self._running = False
        if self._task is not None:
            self._task.cancel()

    async def _loop(self) -> None:
        """Main loop: sleep, scan, refine, repeat."""
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                logger.debug("RefinementWorker loop cancelled")
                raise
            except Exception:
                logger.exception("RefinementWorker tick failed")

            try:
                await asyncio.sleep(self._cfg.scan_interval_seconds)
            except asyncio.CancelledError:
                logger.debug("RefinementWorker sleep cancelled")
                raise

    # ------------------------------------------------------------------
    # Per-tick logic
    # ------------------------------------------------------------------

    @staticmethod
    def _is_learning_card(title: str, description: str = "") -> bool:
        """Check if a draft is a learning card that doesn't need solution extraction.

        Detects learning content by:
        1. Title prefixes (卡片-, 主题-, Layer, 学习计划, 客户端 -)
        2. Content patterns (学习目标, 背景, 前置知识, Layer N, 扫盲)
        3. Absence of problem/solution structure
        """
        # Title-based detection
        skip_prefixes = ("卡片-", "主题-", "Layer ", "学习计划", "客户端 -", "Client -")
        if any(title.startswith(p) for p in skip_prefixes):
            return True

        # Content-based detection
        text = f"{title} {description}"[:500]
        learning_signals = ("学习目标", "Learning Goal", "前置知识", "学习路径",
                          "Layer 0", "Layer 1", "Layer 2", "Layer 3", "Layer 4",
                          "扫盲", "知识地图", "学习计划", "学习进度")
        if any(s in text for s in learning_signals):
            return True

        return False

    async def _tick(self) -> None:
        """Process all drafts needing refinement."""
        drafts = await self._buf.get_needs_refinement()
        if not drafts:
            return

        logger.info("RefinementWorker: %d draft(s) need refinement", len(drafts))
        for draft in drafts:
            draft_id: str = draft["id"]
            title: str = draft.get("title", "")
            description: str = draft.get("description", "")

            # Skip learning cards — they don't need solution extraction
            if self._is_learning_card(title, description):
                logger.info("Skipping learning card: %s", title[:50])
                await self._buf.mark_published(draft_id)
                continue

            # Check retry limit
            attempts = self._retry_counts.get(draft_id, 0)
            if attempts >= _MAX_REFINE_ATTEMPTS:
                logger.warning(
                    "Draft %s exceeded %d refine attempts, marking as refinement_failed",
                    draft_id,
                    _MAX_REFINE_ATTEMPTS,
                )
                await self._buf.mark_published(draft_id)  # Remove from refinement queue
                self._retry_counts.pop(draft_id, None)
                continue

            self._retry_counts[draft_id] = attempts + 1

            try:
                await self._refine_one(draft)
                # Success: clear retry count
                self._retry_counts.pop(draft_id, None)
            except Exception:
                logger.exception("Failed to refine draft %s (attempt %d/%d)",
                                 draft_id, attempts + 1, _MAX_REFINE_ATTEMPTS)
                if self._cfg.fallback_on_failure:
                    await self._fallback_publish(draft)

    async def _refine_one(self, draft: dict[str, Any]) -> None:
        """Refine a single draft via LLM and update the Experience.

        Steps:
        1. Create/save a draft in TM (skip_dedup=True), getting an experience_id.
           If dedup still triggers, use the candidate's id to continue.
        2. Call LLM to extract structured fields from draft content.
        3. Publish the draft (status → published).
        4. Update the Experience with refined fields.
        5. Mark local draft as published.
        """
        draft_id: str = draft["id"]
        content: str = draft.get("content", "")
        project: str = draft.get("project", "default")
        title: str = draft.get("title", f"Draft {draft_id[:8]}")
        conversation_id: str = draft.get("conversation_id", "")

        # Step 1: Save draft to TM — skip dedup since convergence already confirmed
        save_result = await self._sink.draft_save(
            title=title,
            content=content,
            project=project,
            conversation_id=conversation_id,
            skip_dedup=True,
        )
        logger.info("draft_save result for local draft %s: %s", draft_id, save_result)

        # Early-exit on explicit error (e.g. embedding failure)
        if save_result.get("error") and save_result.get("code") != "duplicate_detected":
            raise RuntimeError(
                f"draft_save failed for local draft {draft_id}: {save_result.get('message')}"
            )

        experience_id = save_result.get("id")

        # Handle dedup response — use candidate id if available
        if not experience_id and save_result.get("code") == "duplicate_detected":
            candidates = save_result.get("candidates", [])
            if candidates:
                experience_id = candidates[0].get("id")
                logger.info(
                    "Dedup detected, using candidate experience_id=%s for local draft %s",
                    experience_id,
                    draft_id,
                )

        if not experience_id:
            logger.error("draft_save returned no id for local draft %s: %s", draft_id, save_result)
            raise RuntimeError("draft_save returned no experience_id")

        # Step 2: LLM refinement
        structured = await self._call_llm(content)

        # Step 3: Publish the draft (status → published)
        publish_result = await self._sink.draft_publish(
            draft_id=experience_id,
        )
        if publish_result.get("error"):
            logger.error(
                "Draft publish failed for experience %s: %s",
                experience_id,
                publish_result.get("message"),
            )
            raise RuntimeError(f"draft_publish failed: {publish_result.get('message')}")

        # Step 4: Update Experience with refined fields
        update_result = await self._sink.update_experience(
            experience_id=experience_id,
            title=structured.get("title"),
            problem=structured.get("problem"),
            solution=structured.get("solution"),
            tags=structured.get("tags"),
            experience_type=structured.get("experience_type"),
        )
        if update_result.get("error"):
            logger.error(
                "Experience update failed for %s: %s",
                experience_id,
                update_result.get("message"),
            )
            raise RuntimeError(f"update_experience failed: {update_result.get('message')}")

        # Step 5: mark local draft published
        await self._buf.mark_published(draft_id)
        logger.info(
            "Refined and published draft %s → experience %s (title=%s)",
            draft_id,
            experience_id,
            structured.get("title", "")[:40],
        )

    async def _call_llm(self, content: str) -> dict[str, Any]:
        """Call LLM to extract structured fields from draft content.

        Supports "litellm" (OpenAI-compatible) and "ollama" providers.
        Retries up to max_retries on transient failures.
        """
        last_error: Exception | None = None
        max_retries = max(0, self._cfg.max_retries)

        for attempt in range(max_retries + 1):
            try:
                if self._cfg.provider == "litellm":
                    api_key = os.environ.get(self._cfg.api_key_env, "")
                    return await parse_content_openai(
                        content,
                        model=self._cfg.model,
                        base_url=self._cfg.base_url,
                        api_key=api_key or None,
                        timeout=self._cfg.timeout,
                        max_input_chars=self._cfg.max_input_chars,
                    )
                elif self._cfg.provider == "ollama":
                    # Fallback to existing Ollama parse_content
                    from team_memory.services.llm_parser import parse_content
                    from team_memory.config.llm import LLMConfig

                    llm_config = LLMConfig(
                        model=self._cfg.model,
                        base_url=self._cfg.base_url,
                    )
                    return await parse_content(
                        content,
                        llm_config=llm_config,
                        max_input_chars=self._cfg.max_input_chars,
                    )
                else:
                    raise ValueError(f"Unsupported refinement provider: {self._cfg.provider}")
            except LLMParseError as e:
                last_error = e
                logger.warning(
                    "LLM refinement attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))

        # All retries exhausted
        raise last_error or LLMParseError("LLM refinement failed after all retries")

    async def _fallback_publish(self, draft: dict[str, Any]) -> None:
        """Fallback: publish the draft without LLM refinement.

        Ensures no data is lost when LLM calls fail.
        """
        draft_id: str = draft["id"]
        content: str = draft.get("content", "")
        project: str = draft.get("project", "default")
        title: str = draft.get("title", f"Draft {draft_id[:8]}")
        conversation_id: str = draft.get("conversation_id", "")

        try:
            # Save to TM — skip dedup
            save_result = await self._sink.draft_save(
                title=title,
                content=content,
                project=project,
                conversation_id=conversation_id,
                skip_dedup=True,
            )
            experience_id = save_result.get("id")

            # Handle explicit error (e.g. embedding failure)
            if save_result.get("error") and save_result.get("code") != "duplicate_detected":
                logger.error(
                    "Fallback: draft_save error for %s: %s", draft_id, save_result.get("message")
                )
                return

            # Handle dedup — use candidate id
            if not experience_id and save_result.get("code") == "duplicate_detected":
                candidates = save_result.get("candidates", [])
                if candidates:
                    experience_id = candidates[0].get("id")

            if not experience_id:
                logger.error("Fallback: draft_save returned no id for %s: %s", draft_id, save_result)
                return

            # Publish
            result = await self._sink.draft_publish(draft_id=experience_id)
            if result.get("error"):
                logger.error(
                    "Fallback publish failed for %s: %s",
                    experience_id,
                    result.get("message"),
                )
                return
            await self._buf.mark_published(draft_id)
            self._retry_counts.pop(draft_id, None)
            logger.warning(
                "Fallback published draft %s without LLM refinement (experience_id=%s)",
                draft_id,
                experience_id,
            )
        except Exception:
            logger.exception("Fallback publish failed for draft %s", draft_id)
