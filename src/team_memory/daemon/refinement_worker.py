"""Background refinement worker for the TM Daemon.

Scans the ConversationBuffer for sessions with status='needs_refinement',
calls an LLM (via LiteLLM proxy or Ollama) to extract structured fields,
publishes the draft, then updates the published Experience with refined data.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from team_memory.daemon.config import DaemonConfig, RefinementSettings
from team_memory.daemon.conversation_buffer import ConversationBuffer
from team_memory.daemon.tm_sink import TMSink
from team_memory.services.llm_parser import LLMParseError, _extract_json, call_llm_openai

logger = logging.getLogger("daemon.refinement_worker")

# Maximum refinement attempts before giving up on a draft
_MAX_REFINE_ATTEMPTS = 3

# Minimum character lengths for structural quality checks
_MIN_TITLE_LEN = 5
_MIN_PROBLEM_LEN = 50
_MIN_SOLUTION_LEN = 50

# ---------------------------------------------------------------------------
# Quality Evaluation Prompt — asks LLM to score a refined experience
# ---------------------------------------------------------------------------
_QUALITY_EVAL_PROMPT = """\
Evaluate the following technical experience on a 0-100 scale based on:
- Specificity: Does it describe a concrete problem with clear context?
- Actionability: Can someone follow the solution steps?
- Completeness: Are problem context and solution steps both well-described?
- Uniqueness: Is this a specific incident, not generic advice?

Title: {title}
Problem: {problem}
Solution: {solution}
Type: {experience_type}

Return ONLY a JSON object: {{"score": <integer 0-100>}}
"""

# Mapping from refinement experience_type → canonical EXPERIENCE_TYPES
_REFINEMENT_TYPE_MAP = {
    "problem_solution": "bugfix",
    "workaround": "best_practice",
    "configuration": "feature",
    "insight": "learning",
}

# ---------------------------------------------------------------------------
# Refinement Prompt — targeted at technical coding/debugging/deployment
# conversations, NOT generic text.
# Uses {{ }} for literal braces in .format() template strings.
# ---------------------------------------------------------------------------
_REFINEMENT_PROMPT = """\
你是一个技术经验提取助手。你的任务是从项目「{project}」的技术对话记录中，\
提取结构化的开发经验。

**输入说明**: 输入内容是 AI 助手在编程、调试、部署等技术任务中积累的回复记录，\
其中包含问题分析、方案尝试、错误排查、配置修改等实际操作过程。

**核心要求**: 每条经验必须包含明确的「问题→解决方案」结构——
- 问题: 遇到了什么故障 / 需要实现什么功能 / 碰到了什么困难
- 解决方案: 具体采取了什么行动来解决问题，关键步骤是什么

**以下情况不算经验，不要提取**:
- 简单问答（如「什么是 Docker？」→ 仅知识陈述，无问题解决过程）
- 仅有任务描述但无解决方案（如「需要优化性能」→ 缺少行动和结果）
- 通用知识或事实（如「Python 是解释型语言」→ 无具体场景）
- 琐碎操作（如「执行了 npm install」「重启了服务」→ 无诊断或决策过程）
- 重复或空泛内容

**质量要求**:
- title 必须具体，包含关键技术词，禁止使用泛泛的标题如「调试」「修复」「配置」等。\
好的标题示例：「Docker 容器内 PostgreSQL 连接 host.docker.internal 超时解决」
- description 必须解释问题的上下文和具体表现，不少于 50 字
- solution 必须是可执行的操作步骤或配置变更，不少于 50 字，保留关键命令、路径、参数
- tags 提取 3-8 个技术关键词，小写英文，包含相关技术栈和问题域

**输出格式** — 严格返回以下 JSON（不要用 markdown 代码块包裹）:
{{
  "title": "具体的问题+解决方案标题",
  "description": "问题的详细描述：现象、上下文、影响范围",
  "solution": "具体的解决步骤：操作命令、配置修改、代码变更要点",
  "tags": ["tag1", "tag2", "tag3"],
  "experience_type": "problem_solution"
}}

experience_type 必须是以下之一:
- problem_solution: 故障排查、bug 修复、问题解决（最常见）
- workaround: 绕过方案、临时解决方案、妥协处理
- configuration: 环境配置、依赖安装、部署设置、参数调优
- insight: 从实践中获得的技术洞察、最佳实践总结、架构经验

**反例（不要输出这样的结果）**:
❌ title: "调试" → 太泛，缺少具体问题
❌ title: "修复了一个bug" → 没说什么 bug
❌ description: "遇到了问题" → 没说具体什么问题
❌ solution: "修改了配置" → 没说改了什么配置、改成什么值
❌ experience_type: "bug" → 不在允许列表中

**正例**:
✅ title: "Nginx 反向代理 WebSocket 502 错误修复"
✅ description: "生产环境中 WebSocket 连接通过 Nginx 反向代理时返回 502 Bad Gateway，\
排查发现 Nginx 默认不支持 WebSocket 升级请求，需要在 location 块中添加 Upgrade 头处理"
✅ solution: "在 Nginx 配置的 location 块中添加 proxy_set_header Upgrade $http_upgrade; \
和 proxy_set_header Connection 'upgrade'; 并设置 proxy_http_version 1.1，reload 后恢复正常"
✅ tags: ["nginx", "websocket", "reverse-proxy", "502", "configuration"]
✅ experience_type: "problem_solution"

如果输入内容中没有符合上述标准的经验（即没有包含明确问题解决过程的内容），\
返回:
{{"title": "", "description": "", "solution": "", "tags": [], "experience_type": ""}}

{similar_context}
对话记录:
{content}
"""


async def _refine_via_openai(
    content: str,
    *,
    project: str = "default",
    model: str = "qwen-plus",
    base_url: str = "http://localhost:4000/v1",
    api_key: str | None = None,
    timeout: float = 30.0,
    max_input_chars: int = 6000,
    similar_context: str = "",
) -> dict:
    """Call LLM with the custom refinement prompt and normalize the result.

    Uses _REFINEMENT_PROMPT instead of the generic parse_single prompt,
    and maps the output fields (description→problem, experience_type mapping)
    to match the downstream update_experience API.
    """
    system_prompt = _REFINEMENT_PROMPT.format(
        project=project,
        content=content[:max_input_chars],
        similar_context=similar_context,
    )
    messages = [
        {"role": "system", "content": "You are a technical experience extraction assistant. Output valid JSON only."},
        {"role": "user", "content": system_prompt},
    ]

    llm_text = await call_llm_openai(
        messages,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        temperature=0.2,
    )

    parsed = _extract_json(llm_text)
    return _normalize_refinement_result(parsed)


def _normalize_refinement_result(parsed: dict) -> dict:
    """Normalize the refinement prompt output to match update_experience fields.

    Maps:
      - description → problem (for update_experience API)
      - experience_type: problem_solution → bugfix, workaround → best_practice,
        configuration → feature, insight → learning
    """
    from team_memory.schemas import EXPERIENCE_TYPES

    title = str(parsed.get("title", "")).strip()
    description = str(parsed.get("description", "")).strip()
    solution = str(parsed.get("solution", "")).strip() or None
    tags = parsed.get("tags", [])
    exp_type = str(parsed.get("experience_type", "")).strip()

    # Map refinement-specific types to canonical EXPERIENCE_TYPES
    if exp_type in _REFINEMENT_TYPE_MAP:
        exp_type = _REFINEMENT_TYPE_MAP[exp_type]

    # Validate against known types
    if exp_type not in EXPERIENCE_TYPES:
        exp_type = "general"

    # Normalize tags
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip().lower() for t in tags if t]

    return {
        "title": title,
        "problem": description,  # Map description → problem for downstream API
        "description": description,  # Keep both for flexibility
        "solution": solution,
        "tags": tags,
        "experience_type": exp_type,
    }


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
        buf: ConversationBuffer,
        sink: TMSink,
        queue: Any = None,
        metrics: Any = None,
        health_check: Any = None,
    ) -> None:
        self._cfg: RefinementSettings = config.refinement
        self._buf = buf
        self._sink = sink
        self._queue = queue
        self._metrics = metrics
        self._health_check = health_check
        self._running = False
        self._task: asyncio.Task | None = None

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

    async def _mark_idle_sessions(self) -> None:
        """Auto-mark active sessions that have been idle too long as needs_refinement."""
        idle_minutes = getattr(self._cfg, "idle_timeout_minutes", 30)
        marked = await self._buf.mark_idle_sessions_for_refinement(idle_minutes)
        if marked:
            logger.info(
                "RefinementWorker: marked %d idle session(s) for refinement (idle>%dm)",
                marked, idle_minutes,
            )

    async def _tick(self) -> None:
        """Process all drafts needing refinement.

        Also auto-marks idle active sessions as needs_refinement.
        A session is considered idle if it has had no new messages
        for idle_timeout_minutes (default 30).
        """
        # Step 1: auto-mark idle active sessions
        await self._mark_idle_sessions()

        # Step 1: Process queue retries FIRST (积压优先)
        if self._queue:
            due_ops = self._queue.get_due_ops()
            for op in due_ops:
                op_id = op["id"]
                try:
                    payload = op["payload"]
                    draft = {
                        "id": payload.get("draft_id") or payload.get("session_id") or op["session_id"],
                        "content": payload.get("content", ""),
                        "project": payload.get("project", "default"),
                        "title": payload.get("title", f"Draft {op['session_id'][:8]}"),
                        "conversation_id": payload.get("conversation_id", ""),
                    }
                    await self._refine_one(draft)
                    self._queue.dequeue(op_id)
                    logger.info("Queue retry succeeded for %s", op_id)
                except Exception as retry_err:
                    retries = op["retries"] + 1
                    if retries >= self._queue.MAX_RETRIES:
                        self._queue.mark_failed(op_id)
                        logger.error("Queue op %s permanently failed after %d retries", op_id, retries)
                        if self._cfg.fallback_on_failure:
                            await self._fallback_publish_from_payload(op["payload"], op["session_id"])
                    else:
                        backoff_idx = min(retries - 1, len(self._queue.BACKOFF_SECONDS) - 1)
                        backoff = self._queue.BACKOFF_SECONDS[backoff_idx]
                        self._queue.bump_retry(op_id, str(retry_err)[:500], backoff)

        # Snapshot cross-DB metrics
        if self._metrics:
            await self._metrics.cross_db_snapshot()

        # In-process health check
        if self._health_check:
            try:
                await self._health_check.check_all()
            except Exception:
                logger.exception("HealthCheck failed (non-critical)")

        # Step 2: Process new needs_refinement sessions
        all_drafts = await self._buf.get_needs_refinement()
        if not all_drafts:
            return

        # Group by project for isolated processing
        # 按 project 分组，避免跨项目去重污染
        projects = list({d.get("project", "default") for d in all_drafts})
        logger.info("RefinementWorker: %d draft(s) across %d project(s) need refinement",
                    len(all_drafts), len(projects))

        for project in projects:
            drafts = await self._buf.get_needs_refinement(project=project)
            logger.info("RefinementWorker: processing project=%s (%d draft(s))", project, len(drafts))
            for draft in drafts:
                draft_id: str = draft["id"]
                title: str = draft.get("title", "")
                description: str = draft.get("description", "")

                # Skip learning cards — they don't need solution extraction
                # 跳过学习卡片，不需要解决方案提取
                if self._is_learning_card(title, description):
                    logger.info("Skipping learning card: %s", title[:50])
                    await self._buf.mark_published(draft_id)
                    continue

                # Check for crash-recovery: skip_llm bypasses refine
                if draft.get("skip_llm") and draft.get("refined_payload"):
                    payload = draft["refined_payload"]
                    logger.info("skip_llm: replaying publish for %s", draft_id)
                    await self._sink.draft_publish(draft_id=payload["experience_id"])
                    await self._sink.update_experience(
                        payload["experience_id"], payload["structured"]
                    )
                    await self._buf.mark_published(draft_id)
                    continue

                # Mark as refining before processing / 处理前标记 refining
                await self._buf.mark_refining(draft_id)

                try:
                    await self._refine_one(draft)
                except Exception:
                    from team_memory.daemon.structured_logger import StructuredLogger
                    sl = StructuredLogger(logger)
                    sl.error("refine_fail", session=draft_id, project=draft.get("project","default"))
                    logger.exception("Failed to refine draft %s (traceback above)", draft_id)
                    if self._queue:
                        self._queue.enqueue(
                            op_type="refine",
                            session_id=draft_id,
                            payload={
                                "draft_id": draft_id,
                                "content": draft.get("content", ""),
                                "project": draft.get("project", "default"),
                                "title": title,
                                "conversation_id": draft.get("conversation_id", ""),
                            },
                        )
                    elif self._cfg.fallback_on_failure:
                        await self._fallback_publish(draft)

    async def _refine_one(self, draft: dict[str, Any]) -> None:
        """Refine a single draft via LLM and update the Experience.

        Steps:
        0. Pre-refinement dedup check: search for similar experiences.
           - If cosine > dedup_similarity_threshold → merge into existing.
           - If cosine in [dedup_context_threshold, dedup_similarity_threshold)
             → include as context in the LLM prompt to avoid duplication.
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

        # Step 0: Pre-refinement dedup check (single recall for both dedup + context)
        similar_context = ""
        dedup_hit, context_hits = await self._dedup_search(content, project)
        if dedup_hit is not None:
            hit_id, hit_title, hit_score = dedup_hit
            # High similarity: merge into existing experience instead of creating new
            logger.info(
                "found similar experience: %s (similarity=%.4f), merging draft %s into %s",
                hit_title, hit_score, draft_id, hit_id,
            )
            try:
                await self._sink.increment_used_count(hit_id)
            except Exception:
                logger.warning("Failed to increment used_count for experience %s", hit_id, exc_info=True)
            await self._buf.mark_published(draft_id)
            logger.info(
                "Merged draft %s into existing experience %s (dedup, similarity=%.4f)",
                draft_id, hit_id, hit_score,
            )
            return

        # Medium similarity: build context string for LLM prompt
        if context_hits:
            lines = ["已有相关经验（请勿重复）："]
            for hit in context_hits:
                hit_title = hit.get("title", "Untitled")
                hit_desc = hit.get("problem") or hit.get("description", "")
                lines.append(f"- {hit_title}: {hit_desc[:100]}")
            similar_context = "\n".join(lines)
            logger.info(
                "Including %d similar experience(s) as context for draft %s",
                len(context_hits), draft_id,
            )

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
        structured = await self._call_llm(content, project=project, similar_context=similar_context)

        # Persist LLM output for crash recovery (before quality gate)
        await self._buf.mark_publishing_in_progress(
            draft_id,
            {"experience_id": experience_id, "structured": structured},
        )

        # Map "description" from new prompt output to "problem" for update_experience
        if "description" in structured and "problem" not in structured:
            structured["problem"] = structured["description"]

        # ---- Quality Gate ----
        if self._cfg.quality_gate_enabled:
            passed, reason = self._structural_check(structured)
            if not passed:
                logger.info(
                    "Quality gate discarded draft %s (experience %s): %s",
                    draft_id, experience_id, reason,
                )
                if self._metrics:
                    self._metrics.record(
                        "quality_rejected", 1,
                        {"project": project, "reason": "structure_failed"},
                    )
                await self._sink.draft_publish(draft_id=experience_id)
                await self._buf.mark_published(draft_id)
                return

            score = await self._quality_score(structured)
            if score < self._cfg.quality_min_score:
                logger.info(
                    "Quality gate discarded draft %s (experience %s): "
                    "quality score %d < %d",
                    draft_id, experience_id, score,
                    self._cfg.quality_min_score,
                )
                if self._metrics:
                    self._metrics.record(
                        "quality_score", float(score),
                        {"project": project, "result": "rejected"},
                    )
                    self._metrics.record(
                        "quality_rejected", 1,
                        {"project": project, "reason": "below_threshold"},
                    )
                await self._sink.draft_publish(draft_id=experience_id)
                await self._buf.mark_published(draft_id)
                return

            # Gate passed — record score
            if self._metrics:
                self._metrics.record(
                    "quality_score", float(score),
                    {"project": project, "result": "passed"},
                )

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

        # Step 5: title quality check + mark published
        if self._metrics:
            title_str = structured.get("title", "")
            is_good = bool(title_str) and not title_str.startswith("Draft ") and len(title_str) > 3
            self._metrics.record(
                "title_quality",
                1.0 if is_good else 0.0,
                {"project": project},
            )
        await self._buf.mark_published(draft_id)
        from team_memory.daemon.structured_logger import StructuredLogger
        sl = StructuredLogger(logger)
        sl.info("refine_done", session=draft_id, project=project,
                experience_id=experience_id, title=structured.get("title","")[:40])

    # ------------------------------------------------------------------
    # Pre-refinement dedup helpers
    # ------------------------------------------------------------------

    async def _dedup_search(
        self, content: str, project: str
    ) -> tuple[tuple[str, str, float] | None, list[dict]]:
        """Single recall for both dedup check and context extraction.

        Returns (dedup_hit, context_hits) where:
        - dedup_hit: (experience_id, title, similarity) if similarity >
          dedup_similarity_threshold, else None.
        - context_hits: list of up to 3 results with similarity in
          [dedup_context_threshold, dedup_similarity_threshold].

        If recall() fails or returns empty, returns (None, []) so normal
        refinement proceeds without blocking.
        """
        if not content.strip():
            return None, []
        try:
            results = await self._sink.recall(
                query=content[:500],
                project=project,
                max_results=5,
            )
        except Exception:
            logger.warning("Dedup recall failed, proceeding with normal refinement", exc_info=True)
            return None, []

        if not results:
            return None, []

        dedup_threshold = self._cfg.dedup_similarity_threshold
        ctx_threshold = self._cfg.dedup_context_threshold

        dedup_hit = None
        context_hits: list[dict] = []

        for hit in results:
            score = hit.get("similarity", 0.0)
            # Check for high-similarity duplicate (take the first/best match)
            if dedup_hit is None and score > dedup_threshold:
                hit_id = hit.get("id") or hit.get("group_id", "")
                hit_title = hit.get("title", "Untitled")
                if hit_id:
                    dedup_hit = (hit_id, hit_title, score)
                    continue  # Don't also add to context
            # Check for medium-similarity context
            if ctx_threshold <= score <= dedup_threshold:
                context_hits.append(hit)
                if len(context_hits) >= 3:
                    pass  # Keep scanning for dedup_hit

        return dedup_hit, context_hits

    # ------------------------------------------------------------------
    # Quality gate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _structural_check(structured: dict[str, Any]) -> tuple[bool, str]:
        """Validate structural quality of a refined experience.

        Returns (passed, reason) where reason is empty on success.
        """
        title = str(structured.get("title", "")).strip()
        problem = str(structured.get("problem", "")).strip()
        solution = str(structured.get("solution", "") or "").strip()
        exp_type = str(structured.get("experience_type", "")).strip()

        if not title or len(title) < _MIN_TITLE_LEN:
            return False, "discarded: title too short"
        if not problem or len(problem) < _MIN_PROBLEM_LEN:
            return False, "discarded: problem too short"
        if not solution or len(solution) < _MIN_SOLUTION_LEN:
            return False, "discarded: solution too short"
        if not exp_type:
            return False, "discarded: experience_type empty"

        return True, ""

    async def _quality_score(self, structured: dict[str, Any]) -> int:
        """Ask LLM to score a refined experience on 0-100.

        On any error, returns 100 (pass) so that LLM failures don't block
        experiences that already passed structural checks.
        """
        try:
            prompt = _QUALITY_EVAL_PROMPT.format(
                title=structured.get("title", ""),
                problem=structured.get("problem", ""),
                solution=structured.get("solution", ""),
                experience_type=structured.get("experience_type", ""),
            )
            messages = [
                {"role": "system", "content": "You are a quality evaluator. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ]
            llm_text = await call_llm_openai(
                messages,
                model=self._cfg.model,
                base_url=self._cfg.base_url,
                api_key=os.environ.get(self._cfg.api_key_env, "") or None,
                timeout=self._cfg.timeout,
                temperature=0.1,
            )
            parsed = _extract_json(llm_text)
            score = int(parsed.get("score", 100))
            # Clamp to 0-100
            return max(0, min(100, score))
        except Exception:
            logger.warning("Quality scoring LLM call failed, defaulting to pass", exc_info=True)
            return 100

    async def _call_llm(self, content: str, project: str = "default", similar_context: str = "") -> dict[str, Any]:
        """Call LLM to extract structured fields from draft content.

        Uses the custom _REFINEMENT_PROMPT for technical conversation extraction.
        Supports "litellm" (OpenAI-compatible) and "ollama" providers.
        Retries up to max_retries on transient failures.
        """
        last_error: Exception | None = None
        max_retries = max(0, self._cfg.max_retries)

        for attempt in range(max_retries + 1):
            try:
                if self._cfg.provider == "litellm":
                    api_key = os.environ.get(self._cfg.api_key_env, "")
                    return await _refine_via_openai(
                        content,
                        project=project,
                        model=self._cfg.model,
                        base_url=self._cfg.base_url,
                        api_key=api_key or None,
                        timeout=self._cfg.timeout,
                        max_input_chars=self._cfg.max_input_chars,
                        similar_context=similar_context,
                    )
                elif self._cfg.provider == "ollama":
                    # Fallback to existing Ollama parse_content
                    from team_memory.config.llm import LLMConfig
                    from team_memory.services.llm_parser import parse_content

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

    async def _fallback_publish_from_payload(
        self, payload: dict[str, Any], session_id: str
    ) -> None:
        """Fallback publish from a queue payload / 从队列 payload 恢复并发布。"""
        draft = {
            "id": payload.get("draft_id") or session_id,
            "content": payload.get("content", ""),
            "project": payload.get("project", "default"),
            "title": payload.get("title", f"Draft {session_id[:8]}"),
            "conversation_id": payload.get("conversation_id", ""),
        }
        await self._fallback_publish(draft)

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
            # Clear stale refined_payload after fallback
            try:
                await self._buf._db.execute(
                    "UPDATE sessions SET refined_payload=NULL WHERE id=?",
                    (draft_id,),
                )
                await self._buf._db.commit()
            except Exception:
                pass
            logger.warning(
                "Fallback published draft %s without LLM refinement (experience_id=%s)",
                draft_id,
                experience_id,
            )
        except Exception:
            logger.exception("Fallback publish failed for draft %s", draft_id)
