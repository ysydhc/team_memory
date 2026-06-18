from __future__ import annotations

import asyncio
import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg

from team_memory.daemon.quality.arena.evaluator import ArenaEvaluator
from team_memory.daemon.quality.arena.meta_judge import MetaJudge, MetaVerdict
from team_memory.daemon.quality.arena.models import (
    EpochRecord,
    PromptEntry,
    RoundRecord,
    TaskEval,
)
from team_memory.daemon.quality.arena.prompt_generator import PromptGenerator
from team_memory.daemon.quality.arena.storage import ArenaStorage, update_elo

logger = logging.getLogger("arena.runner")

ROUNDS_PER_EPOCH = 10
PROTECTION_ROUNDS = 5
SAMPLE_SIZE = 3


class ArenaRunner:
    """Orchestrates arena competition rounds and production evaluation."""

    def __init__(
        self,
        dsn: str,
        storage: ArenaStorage | None = None,
        evaluator: ArenaEvaluator | None = None,
        meta_judge: MetaJudge | None = None,
        generator: PromptGenerator | None = None,
    ) -> None:
        self._raw_dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        self._async_dsn = dsn if "+asyncpg" in dsn else dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
        self._storage = storage or ArenaStorage()
        self._evaluator = evaluator or ArenaEvaluator()
        self._meta_judge = meta_judge or MetaJudge()
        self._generator = generator or PromptGenerator()
        self._pool: asyncpg.Pool | None = None

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None or self._pool._closed:
            self._pool = await asyncpg.create_pool(self._raw_dsn, min_size=1, max_size=3)
        return self._pool

    async def ensure_initialized(self) -> None:
        """If no prompts exist, generate seed prompts via the generator."""
        active = self._storage.get_active_prompts()
        if active:
            return

        logger.info("No active prompts found — generating seed prompts")
        seeds = await self._generator.generate_seed_prompts()
        self._storage.save_prompts(seeds)
        logger.info("Generated %d seed prompts", len(seeds))

    async def run_round(self) -> RoundRecord | None:
        """Execute one arena round. Returns RoundRecord or None if not enough data."""
        await self.ensure_initialized()

        active = self._storage.get_active_prompts()
        if len(active) < 2:
            logger.warning("Need at least 2 active prompts, have %d", len(active))
            return None

        tasks_raw = await self._sample_tasks(SAMPLE_SIZE)
        if not tasks_raw:
            logger.info("No suitable tasks in response buffer for arena round")
            return None

        prompt_a, prompt_b = self._pick_pair(active)

        from team_memory.services.maintenance.batch_ops import BatchOpsService

        batch_svc = BatchOpsService(db_url=self._async_dsn)

        task_evals: list[TaskEval] = []
        query_contexts: list[tuple[str, list[str], str]] = []

        eval_tasks = []
        for entry in tasks_raw:
            eval_tasks.append(self._evaluate_task(entry, prompt_a, prompt_b, batch_svc))

        eval_results = await asyncio.gather(*eval_tasks, return_exceptions=True)

        for result in eval_results:
            if isinstance(result, Exception):
                logger.debug("Task evaluation raised exception: %s", result)
                continue
            if result is None:
                continue
            task_eval, qc = result
            task_evals.append(task_eval)
            query_contexts.append(qc)

        if not task_evals:
            logger.warning("All task evaluations failed — aborting round")
            return None

        verdict = await self._meta_judge.judge(tasks=task_evals, query_contexts=query_contexts)
        verdict = self._enforce_collapse_guard(verdict, task_evals, prompt_a, prompt_b)

        prompts = self._storage.load_prompts()
        prompts = self._apply_elo_update(prompts, prompt_a, prompt_b, verdict)
        self._storage.save_prompts(prompts)

        epoch = self._storage.get_current_epoch()
        record = RoundRecord(
            round_id=uuid.uuid4().hex[:12],
            epoch=epoch,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_a_id=prompt_a.id,
            prompt_b_id=prompt_b.id,
            tasks=task_evals,
            meta_verdict=verdict.winner,
            meta_reasoning=verdict.reasoning,
        )
        self._storage.append_round(record)

        logger.info(
            "Round %s complete: %s vs %s → %s (%d tasks)",
            record.round_id,
            prompt_a.id,
            prompt_b.id,
            verdict.winner,
            len(task_evals),
        )

        rounds_in_epoch = self._count_rounds_in_current_epoch()
        if rounds_in_epoch >= ROUNDS_PER_EPOCH:
            await self._handle_epoch()

        return record

    async def _evaluate_task(
        self,
        entry: dict[str, Any],
        prompt_a: PromptEntry,
        prompt_b: PromptEntry,
        batch_svc: Any,
    ) -> tuple[TaskEval, tuple[str, list[str], str]] | None:
        """Evaluate a single task with both prompts. Returns None on failure."""
        query = entry["query"]
        agent_response = entry["agent_response"]
        result_ids = entry["result_ids"]

        contexts = await batch_svc.load_experience_contexts(result_ids)
        if not contexts:
            logger.debug("No contexts loaded for entry %s", entry["id"][:8])
            return None

        eval_a, eval_b = await asyncio.gather(
            self._evaluator.evaluate(
                prompt_template=prompt_a.text,
                query=query,
                contexts=contexts,
                agent_response=agent_response,
            ),
            self._evaluator.evaluate(
                prompt_template=prompt_b.text,
                query=query,
                contexts=contexts,
                agent_response=agent_response,
            ),
        )

        if eval_a.relevance < 0 and eval_b.relevance < 0:
            logger.debug("Both evaluations failed for entry %s", entry["id"][:8])
            return None

        task_eval = TaskEval(
            buffer_entry_id=entry["id"],
            query=query,
            context_count=len(contexts),
            eval_a=eval_a if eval_a.relevance >= 0 else None,
            eval_b=eval_b if eval_b.relevance >= 0 else None,
        )

        return task_eval, (query, contexts, agent_response)

    async def _handle_epoch(self) -> None:
        """End-of-epoch: retire weakest prompt (if past protection period), generate new one."""
        active = self._storage.get_active_prompts()
        if len(active) < 2:
            return

        rounds = self._storage.load_rounds()
        epoch = self._storage.get_current_epoch()

        # Find the prompt with lowest ELO that has participated in enough rounds
        candidates_for_retirement: list[PromptEntry] = []
        for prompt in active:
            rounds_participated = sum(1 for r in rounds if r.prompt_a_id == prompt.id or r.prompt_b_id == prompt.id)
            if rounds_participated >= PROTECTION_ROUNDS:
                candidates_for_retirement.append(prompt)

        retired_prompt: PromptEntry | None = None
        if candidates_for_retirement:
            retired_prompt = min(candidates_for_retirement, key=lambda p: p.elo)

        champion = self._storage.get_champion()
        if champion is None:
            return

        # Don't retire the champion
        if retired_prompt and retired_prompt.id == champion.id:
            retired_prompt = None

        new_prompt: PromptEntry | None = None
        try:
            retired_prompts = [p for p in self._storage.load_prompts() if p.retired]
            recent_rounds = rounds[-10:]
            new_prompt = await self._generator.generate_evolved_prompt(
                champion=champion,
                retired_prompts=retired_prompts,
                recent_rounds=recent_rounds,
                parent_id=champion.id,
            )
        except Exception as e:
            logger.error("Failed to generate evolved prompt: %s", e)

        prompts = self._storage.load_prompts()

        if retired_prompt:
            prompts = [
                PromptEntry(
                    id=p.id,
                    version=p.version,
                    text=p.text,
                    elo=p.elo,
                    wins=p.wins,
                    losses=p.losses,
                    draws=p.draws,
                    created_at=p.created_at,
                    retired=True if p.id == retired_prompt.id else p.retired,
                    parent_id=p.parent_id,
                )
                for p in prompts
            ]

        if new_prompt:
            prompts.append(new_prompt)

        self._storage.save_prompts(prompts)

        epoch_record = EpochRecord(
            epoch=epoch,
            rounds_completed=self._count_rounds_in_current_epoch(),
            champion_id=champion.id,
            champion_elo=champion.elo,
            retired_id=retired_prompt.id if retired_prompt else None,
            new_prompt_id=new_prompt.id if new_prompt else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._storage.append_epoch(epoch_record)

        logger.info(
            "Epoch %d complete: champion=%s (ELO %.0f), retired=%s, new=%s",
            epoch,
            champion.id,
            champion.elo,
            retired_prompt.id if retired_prompt else "none",
            new_prompt.id if new_prompt else "none",
        )

    async def run_production_eval(self, batch_size: int = 20) -> dict[str, Any]:
        """Use champion prompt to evaluate pending buffer entries.

        Replaces faithfulness_batch. Writes relevance_score + helpfulness_score
        to search_logs.

        Returns summary dict with counts.
        """
        await self.ensure_initialized()

        champion = self._storage.get_champion()
        if champion is None:
            logger.warning("No champion prompt available for production eval")
            return {"evaluated": 0, "skipped": 0, "errors": 0, "error": "no_champion"}

        from team_memory.daemon.buffers.response_buffer import ResponseBuffer
        from team_memory.services.maintenance.batch_ops import BatchOpsService

        buffer = ResponseBuffer(dsn=self._raw_dsn)
        batch_svc = BatchOpsService(db_url=self._async_dsn)

        try:
            entries = await buffer.fetch_pending(limit=batch_size)
            if not entries:
                return {"evaluated": 0, "skipped": 0, "errors": 0}

            evaluated = 0
            skipped = 0
            errors = 0
            scores: list[float] = []

            for entry in entries:
                entry_id = entry["id"]
                result_ids = entry["result_ids"]
                search_log_id = entry.get("search_log_id")

                if not result_ids:
                    await buffer.mark_evaluated(entry_id, faithfulness_score=0.0, judge_reasoning="skipped: no results")
                    skipped += 1
                    continue

                try:
                    contexts = await batch_svc.load_experience_contexts(result_ids)
                    if not contexts:
                        await buffer.mark_evaluated(
                            entry_id,
                            faithfulness_score=0.0,
                            judge_reasoning="skipped: no valid contexts",
                        )
                        skipped += 1
                        continue

                    result = await self._evaluator.evaluate(
                        prompt_template=champion.text,
                        query=entry["query"],
                        contexts=contexts,
                        agent_response=entry["agent_response"],
                    )

                    if result.relevance < 0:
                        await buffer.mark_evaluated(
                            entry_id,
                            faithfulness_score=-1.0,
                            judge_reasoning=f"eval_error: {result.reasoning}",
                        )
                        errors += 1
                        continue

                    faithfulness_score = (result.relevance + result.helpfulness) / 2.0
                    await buffer.mark_evaluated(
                        entry_id,
                        faithfulness_score=faithfulness_score,
                        judge_reasoning=result.reasoning[:2000],
                    )

                    if search_log_id:
                        await self._write_arena_scores_to_search_log(
                            search_log_id, result.relevance, result.helpfulness
                        )

                    evaluated += 1
                    scores.append(faithfulness_score)

                    logger.debug(
                        "Evaluated %s: relevance=%.2f helpfulness=%.2f",
                        entry_id[:8],
                        result.relevance,
                        result.helpfulness,
                    )

                except Exception as e:
                    logger.error("Production eval failed for entry %s: %s", entry_id[:8], e)
                    errors += 1

            avg_score = sum(scores) / len(scores) if scores else 0.0
            logger.info(
                "Production eval complete: evaluated=%d skipped=%d errors=%d avg=%.3f",
                evaluated,
                skipped,
                errors,
                avg_score,
            )

            return {
                "evaluated": evaluated,
                "skipped": skipped,
                "errors": errors,
                "avg_score": round(avg_score, 3),
                "champion_id": champion.id,
                "champion_elo": champion.elo,
            }

        finally:
            await buffer.close()

    async def _sample_tasks(self, count: int) -> list[dict[str, Any]]:
        """Sample random tasks from response_buffer that have valid result_ids.

        Samples from ALL entries (not just pending) — arena comparison needs
        real data regardless of evaluation status. Joins with experiences to
        ensure at least one referenced experience still exists.
        """
        import json as _json

        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT rb.id, rb.query, rb.agent_response, rb.result_ids,
                           rb.search_log_id, rb.created_at
                    FROM response_buffer rb
                    WHERE rb.result_ids IS NOT NULL
                      AND EXISTS (
                          SELECT 1 FROM experiences e
                          WHERE e.id::text IN (
                              SELECT jsonb_array_elements(rb.result_ids::jsonb)->>'id'
                          )
                      )
                    ORDER BY random()
                    LIMIT $1
                    """,
                    count,
                )
            return [
                {
                    "id": str(r["id"]),
                    "query": r["query"],
                    "agent_response": r["agent_response"],
                    "result_ids": _json.loads(r["result_ids"]) if r["result_ids"] else None,
                    "search_log_id": str(r["search_log_id"]) if r["search_log_id"] else None,
                    "created_at": r["created_at"],
                }
                for r in rows
                if r["result_ids"]
            ]
        except Exception as e:
            logger.warning("Failed to sample tasks: %s", e)
            return []

    async def _write_arena_scores_to_search_log(self, search_log_id: str, relevance: float, helpfulness: float) -> None:
        """Write relevance_score and helpfulness_score to search_logs table."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE search_logs
                    SET relevance_score = $2,
                        helpfulness_score = $3,
                        was_used = CASE WHEN $3::float >= 0.5 THEN TRUE ELSE was_used END,
                        judgment_source = 'arena',
                        updated_at = $4
                    WHERE id = $1
                    """,
                    uuid.UUID(search_log_id),
                    relevance,
                    helpfulness,
                    datetime.now(timezone.utc),
                )
        except Exception as e:
            logger.debug("Failed to write arena scores to search_log %s: %s", search_log_id[:8], e)

    @staticmethod
    def _enforce_collapse_guard(
        verdict: MetaVerdict,
        task_evals: list[TaskEval],
        prompt_a: PromptEntry,
        prompt_b: PromptEntry,
    ) -> MetaVerdict:
        """Override verdict to draw if the winner shows collapse-to-zero pattern.

        A prompt that scores 0.0 on ALL tasks cannot truly differentiate quality —
        it's a degenerate strategy. If meta-judge still awards it the win, downgrade
        to draw so it doesn't gain ELO from gaming.
        """
        if verdict.winner == "draw":
            return verdict

        winner_side = "a" if verdict.winner == "a_wins" else "b"
        winner_scores: list[float] = []
        opponent_scores: list[float] = []
        for t in task_evals:
            ev_winner = t.eval_a if winner_side == "a" else t.eval_b
            ev_opponent = t.eval_b if winner_side == "a" else t.eval_a
            if ev_winner:
                winner_scores.extend([ev_winner.relevance, ev_winner.helpfulness])
            if ev_opponent:
                opponent_scores.extend([ev_opponent.relevance, ev_opponent.helpfulness])

        if not winner_scores:
            return verdict

        all_zero = all(s <= 0.05 for s in winner_scores)
        opponent_has_nonzero = any(s > 0.3 for s in opponent_scores)

        if all_zero and opponent_has_nonzero:
            winner_id = prompt_a.id if winner_side == "a" else prompt_b.id
            logger.info(
                "Collapse guard: overriding %s → draw (winner %s scored all zeros, opponent has signal)",
                verdict.winner,
                winner_id[:12],
            )
            return MetaVerdict(
                winner="draw",
                reasoning=f"collapse_guard: {verdict.reasoning}",
            )

        return verdict

    def _pick_pair(self, prompts: list[PromptEntry]) -> tuple[PromptEntry, PromptEntry]:
        """Pick 2 prompts for a round — prefer ELO-close pairing."""
        if len(prompts) == 2:
            return prompts[0], prompts[1]

        sorted_prompts = sorted(prompts, key=lambda p: p.elo)
        idx = random.randint(0, len(sorted_prompts) - 2)
        return sorted_prompts[idx], sorted_prompts[idx + 1]

    def _count_rounds_in_current_epoch(self) -> int:
        """Count rounds played since the last epoch boundary."""
        epochs = self._storage.load_epochs()
        rounds = self._storage.load_rounds()
        if not epochs:
            return len(rounds)
        last_epoch_ts = epochs[-1].timestamp
        return sum(1 for r in rounds if r.timestamp > last_epoch_ts)

    def _apply_elo_update(
        self,
        prompts: list[PromptEntry],
        prompt_a: PromptEntry,
        prompt_b: PromptEntry,
        verdict: MetaVerdict,
    ) -> list[PromptEntry]:
        """Apply ELO update to prompt list based on verdict. Returns new list."""
        a_elo = prompt_a.elo
        b_elo = prompt_b.elo

        if verdict.winner == "a_wins":
            new_a_elo, new_b_elo = update_elo(a_elo, b_elo)
        elif verdict.winner == "b_wins":
            new_b_elo, new_a_elo = update_elo(b_elo, a_elo)
        else:
            new_a_elo, new_b_elo = update_elo(a_elo, b_elo, draw=True)

        updated: list[PromptEntry] = []
        for p in prompts:
            if p.id == prompt_a.id:
                updated.append(
                    PromptEntry(
                        id=p.id,
                        version=p.version,
                        text=p.text,
                        elo=new_a_elo,
                        wins=p.wins + (1 if verdict.winner == "a_wins" else 0),
                        losses=p.losses + (1 if verdict.winner == "b_wins" else 0),
                        draws=p.draws + (1 if verdict.winner == "draw" else 0),
                        created_at=p.created_at,
                        retired=p.retired,
                        parent_id=p.parent_id,
                    )
                )
            elif p.id == prompt_b.id:
                updated.append(
                    PromptEntry(
                        id=p.id,
                        version=p.version,
                        text=p.text,
                        elo=new_b_elo,
                        wins=p.wins + (1 if verdict.winner == "b_wins" else 0),
                        losses=p.losses + (1 if verdict.winner == "a_wins" else 0),
                        draws=p.draws + (1 if verdict.winner == "draw" else 0),
                        created_at=p.created_at,
                        retired=p.retired,
                        parent_id=p.parent_id,
                    )
                )
            else:
                updated.append(p)

        return updated

    async def close(self) -> None:
        """Cleanup resources."""
        if self._pool and not self._pool._closed:
            await self._pool.close()
