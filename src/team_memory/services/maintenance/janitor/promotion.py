"""L2 -> L3 experience promotion and Markdown export for MemoryJanitor."""

from __future__ import annotations

import logging
import os
import subprocess as _default_subprocess
import sys
from datetime import datetime, timezone

from sqlalchemy import and_, or_, select

from team_memory.services.maintenance.janitor.helpers import _is_eligible_for_promotion
from team_memory.services.maintenance.promotion_compiler import (
    PromotionCompiler as _DefaultPromotionCompiler,
)
from team_memory.storage.models import Experience

logger = logging.getLogger("team_memory.services.maintenance.janitor")


def _get_promotion_compiler_cls():
    pkg = sys.modules.get("team_memory.services.maintenance.janitor")
    return getattr(pkg, "PromotionCompiler", _DefaultPromotionCompiler)


def _get_subprocess_mod():
    pkg = sys.modules.get("team_memory.services.maintenance.janitor")
    return getattr(pkg, "subprocess", _default_subprocess)


class JanitorPromotionMixin:
    """Mixin providing experience promotion and Markdown writing."""

    async def run_promotion(self, project: str | None = None) -> dict:
        """Promote published experiences to 'promoted' status based on thresholds.

        Two promotion paths:
        1. recall_count threshold: experiences with recall_count >= threshold get promoted
        2. group_key threshold: group_keys with count >= threshold cause all
           published experiences in that group to be promoted

        Only experiences with exp_status="published" are eligible; already-promoted
        experiences are excluded to prevent double promotion.

        After identifying candidates, Markdown is compiled via PromotionCompiler
        and optionally written to Obsidian output directories (git add + commit).

        Args:
            project: Project to limit promotion to, None for all projects

        Returns:
            Dict with promotion statistics
        """
        recall_count_threshold = self._get_config("promotion_use_count_threshold", 3)
        require_used = bool(self._get_config("promotion_require_used", False))
        group_key_threshold = self._get_config("promotion_group_key_threshold", 5)
        output_dirs: dict[str, str] = self._get_config("promotion_output_dirs", {})

        promoted_by_recall_count = 0
        promoted_by_group = 0

        async with self._session() as session:
            # 1. Promote by recall_count threshold
            count_condition = (
                Experience.used_count >= recall_count_threshold
                if require_used
                else or_(
                    Experience.used_count >= recall_count_threshold,
                    Experience.recall_count >= recall_count_threshold,
                )
            )
            recall_count_query = select(Experience).where(
                and_(
                    Experience.is_deleted == False,  # noqa: E712
                    Experience.exp_status == "published",
                    Experience.valid_until.is_(None),
                    Experience.negative_feedback_count == 0,
                    count_condition,
                )
            )
            if project:
                recall_count_query = recall_count_query.where(Experience.project == project)

            result = await session.execute(recall_count_query)
            recall_count_experiences = [
                e
                for e in result.scalars().all()
                if _is_eligible_for_promotion(e, recall_count_threshold, require_used)
            ]

            # 2. Promote by group_key threshold
            # Find group_keys with enough published experiences
            from sqlalchemy import func

            group_count_query = (
                select(Experience.group_key, func.count(Experience.id).label("cnt"))
                .where(
                    and_(
                        Experience.is_deleted == False,  # noqa: E712
                        Experience.exp_status == "published",
                        Experience.valid_until.is_(None),
                        Experience.negative_feedback_count == 0,
                        Experience.group_key.isnot(None),
                    )
                )
                .group_by(Experience.group_key)
                .having(func.count(Experience.id) >= group_key_threshold)
            )
            if project:
                group_count_query = group_count_query.where(Experience.project == project)

            group_result = await session.execute(group_count_query)
            qualifying_groups = [row[0] for row in group_result.all()]

            group_experiences: list[Experience] = []
            if qualifying_groups:
                # Get all published experiences in qualifying groups
                group_exp_query = select(Experience).where(
                    and_(
                        Experience.is_deleted == False,  # noqa: E712
                        Experience.exp_status == "published",
                        Experience.valid_until.is_(None),
                        Experience.negative_feedback_count == 0,
                        Experience.group_key.in_(qualifying_groups),
                    )
                )
                if project:
                    group_exp_query = group_exp_query.where(Experience.project == project)

                group_exp_result = await session.execute(group_exp_query)
                group_experiences = [
                    e
                    for e in group_exp_result.scalars().all()
                    if _is_eligible_for_promotion(e, 0, False)
                ]

            # ----------------------------------------------------------
            # 3. Compile Markdown + write files BEFORE status change
            # ----------------------------------------------------------
            compiler = _get_promotion_compiler_cls()()
            promoted_ids: set[str] = set()

            # 3a. recall_count promotions — one Markdown per experience
            for exp in recall_count_experiences:
                exp_dict = exp.to_dict()
                markdown = await compiler.compile([exp_dict], exp.group_key)
                await self._write_promoted(exp, markdown, output_dirs)
                promoted_ids.add(str(exp.id))
                promoted_by_recall_count += 1
                logger.debug(
                    "Promoted experience by recall_count: %s (recall_count=%d)",
                    exp.id,
                    exp.recall_count,
                )

            # 3b. group_key promotions — one Markdown per group
            # Group experiences by group_key for batch compilation
            by_group: dict[str, list[Experience]] = {}
            for exp in group_experiences:
                if str(exp.id) in promoted_ids:
                    continue  # avoid double compilation
                by_group.setdefault(exp.group_key, []).append(exp)

            for gk, exps in by_group.items():
                exp_dicts = [e.to_dict() for e in exps]
                markdown = await compiler.compile(exp_dicts, gk)
                for exp in exps:
                    await self._write_promoted(exp, markdown, output_dirs)
                    promoted_ids.add(str(exp.id))
                    promoted_by_group += 1
                    logger.debug(
                        "Promoted experience by group_key: %s (group_key=%s)",
                        exp.id,
                        exp.group_key,
                    )

            # ----------------------------------------------------------
            # 4. Mark all promoted experiences
            # ----------------------------------------------------------
            for exp in recall_count_experiences:
                exp.exp_status = "promoted"

            for exp in group_experiences:
                if str(exp.id) not in set(str(e.id) for e in recall_count_experiences):
                    exp.exp_status = "promoted"

            await session.commit()

        total = promoted_by_recall_count + promoted_by_group

        logger.info(
            "Promotion completed: %d by recall_count, %d by group_key, %d total",
            promoted_by_recall_count,
            promoted_by_group,
            total,
        )

        return {
            "promoted_by_recall_count": promoted_by_recall_count,
            "promoted_by_used_count": promoted_by_recall_count,
            "promoted_by_group": promoted_by_group,
            "total": total,
            "recall_count_threshold": recall_count_threshold,
            "group_key_threshold": group_key_threshold,
            "project": project,
        }

    async def _write_promoted(self, exp: Experience, markdown: str, output_dirs: dict[str, str]) -> None:
        """Write compiled Markdown to the Obsidian output directory and git add/commit.

        Args:
            exp: The promoted Experience ORM object.
            markdown: The compiled Markdown string.
            output_dirs: project → output directory mapping.
        """
        project = getattr(exp, "project", "default") or "default"
        output_dir = output_dirs.get(project)
        if not output_dir:
            return

        os.makedirs(output_dir, exist_ok=True)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        group_key = exp.group_key or "misc"
        filename = f"promoted-{today}-{group_key}.md"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            f.write(markdown)

        # git add + commit (best effort)
        subproc = _get_subprocess_mod()
        try:
            subproc.run(
                ["git", "add", filepath],
                cwd=output_dir,
                capture_output=True,
                timeout=10,
            )
            subproc.run(
                ["git", "commit", "-m", f"promoted: {group_key}"],
                cwd=output_dir,
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass  # best effort
