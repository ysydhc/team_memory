#!/opt/homebrew/bin/python3.13
"""Formal verification runner for TeamMemory Daemon Session Refinement Lifecycle.

Runs:
1. TLA+ model checking (TLC 1.7.4) on RefinementLifecycle.tla (and RefinementLifecycleFixed.tla).
2. Live trace replay and invariant verification against the real Python implementation:
   - team_memory.daemon.refinement.refinement_worker.RefinementWorker
   - team_memory.daemon.buffers.conversation_buffer.ConversationBuffer
   - team_memory.daemon.buffers.pending_queue.PendingQueue

Usage:
  /opt/homebrew/bin/python3.13 .agents/skills/verify-refinement-lifecycle/scripts/verify_refinement_lifecycle.py
  /opt/homebrew/bin/python3.13 \\
      .agents/skills/verify-refinement-lifecycle/scripts/verify_refinement_lifecycle.py --strict
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

SKILL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SKILL_DIR.parents[2]
for _p in [REPO_ROOT / "src", *REPO_ROOT.glob(".venv/lib/python*/site-packages")]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import aiosqlite as _aiosqlite  # noqa: F401
except ModuleNotFoundError:
    class _AsyncCursor:
        def __init__(self, cur: sqlite3.Cursor) -> None:
            self._cur = cur
            self.rowcount = cur.rowcount
            self.lastrowid = cur.lastrowid

        async def fetchone(self) -> Any:
            return self._cur.fetchone()

        async def fetchall(self) -> list[Any]:
            return self._cur.fetchall()

    class _AsyncConn:
        def __init__(self, path: str) -> None:
            self._conn = sqlite3.connect(path)

        @property
        def row_factory(self) -> Any:
            return self._conn.row_factory

        @row_factory.setter
        def row_factory(self, val: Any) -> None:
            self._conn.row_factory = val

        async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> _AsyncCursor:
            return _AsyncCursor(self._conn.execute(sql, params))

        async def execute_fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[Any]:
            return self._conn.execute(sql, params).fetchall()

        async def commit(self) -> None:
            self._conn.commit()

        async def close(self) -> None:
            self._conn.close()

    async def _connect(path: str, **_kw: Any) -> _AsyncConn:
        return _AsyncConn(path)

    _shim = types.ModuleType("aiosqlite")
    _shim.connect = _connect  # type: ignore[attr-defined]
    _shim.Row = sqlite3.Row  # type: ignore[attr-defined]
    _shim.Connection = _AsyncConn  # type: ignore[attr-defined]
    sys.modules["aiosqlite"] = _shim

SPEC_FILE = SKILL_DIR / "RefinementLifecycle.tla"
FIXED_SPEC_FILE = SKILL_DIR / "RefinementLifecycleFixed.tla"

KNOWN_FINDINGS: set[str] = set()

ALL_INVARIANTS = [
    "TypeOK",
    "AtMostOnePendingQueueEntry",
    "NoStrandedRefiningSession",
    "RejectedDraftNeverPublished",
    "AtMostOnePublishedExperience",
    "ImporterPreservesPublishingSession",
]


def _find_java() -> str:
    env_java = os.environ.get("JAVA")
    if env_java and Path(env_java).is_file():
        return env_java
    candidates = [
        "/opt/homebrew/opt/java/bin/java",
        "/opt/homebrew/opt/openjdk/bin/java",
        "/usr/local/opt/openjdk/bin/java",
    ]
    for c in candidates:
        if Path(c).is_file():
            return c
    which_java = shutil.which("java")
    if which_java:
        return which_java
    raise RuntimeError("Java runtime not found. Set JAVA=/path/to/java.")


def _find_tla2tools() -> Path:
    env_jar = os.environ.get("TLA2TOOLS")
    if env_jar and Path(env_jar).is_file():
        return Path(env_jar)
    candidates = [
        SKILL_DIR / "tools" / "tla2tools.jar",
        Path.home() / ".gemini" / "config" / "skills" / "statechart" / "tla2tools.jar",
        Path("/tmp/tla2tools.jar"),
        Path.home() / ".cache" / "tm-verify" / "tla2tools-1.7.4.jar",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise RuntimeError(
        "tla2tools.jar not found. Set TLA2TOOLS=/path/to/tla2tools.jar or place it at /tmp/tla2tools.jar."
    )


def run_tlc_invariant(java_bin: str, jar_path: Path, spec_path: Path, invariant: str) -> tuple[bool, str]:
    """Run TLC on spec_path checking a single INVARIANT. Returns (passed, output)."""
    with tempfile.TemporaryDirectory(prefix=f"tlc-{invariant}-") as work_dir:
        work = Path(work_dir)
        local_spec = work / spec_path.name
        shutil.copy2(spec_path, local_spec)
        cfg_path = work / f"{spec_path.stem}.cfg"
        cfg_path.write_text(f"SPECIFICATION Spec\nINVARIANT {invariant}\n", encoding="utf-8")

        cmd = [
            java_bin,
            "-XX:+UseParallelGC",
            "-cp",
            str(jar_path),
            "tlc2.TLC",
            "-workers",
            "1",
            "-deadlock",
            "-cleanup",
            "-metadir",
            str(work / "meta"),
            "-config",
            str(cfg_path),
            str(local_spec),
        ]
        proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True, check=False)
        out = proc.stdout + "\n" + proc.stderr
        passed = proc.returncode == 0 and "Model checking completed. No error has been found." in out
        return passed, out


class ObservableSink:
    """In-memory TMSink spy recording observable draft_save and draft_publish calls."""

    def __init__(self, fail_publish_times: int = 0) -> None:
        self.saved_drafts: list[dict[str, Any]] = []
        self.published_draft_ids: list[str] = []
        self.updated_experiences: list[dict[str, Any]] = []
        self.fail_publish_times = fail_publish_times
        self._publish_attempts = 0

    async def recall(self, query: str, project: str = "default", max_results: int = 5) -> list[dict[str, Any]]:
        return []

    async def draft_save(
        self,
        title: str,
        content: str,
        project: str = "default",
        conversation_id: str = "",
        skip_dedup: bool = False,
    ) -> dict[str, Any]:
        exp_id = f"exp-{len(self.saved_drafts) + 1}"
        self.saved_drafts.append(
            {
                "id": exp_id,
                "title": title,
                "content": content,
                "project": project,
                "conversation_id": conversation_id,
                "skip_dedup": skip_dedup,
            }
        )
        return {"id": exp_id, "status": "draft"}

    async def draft_publish(self, draft_id: str) -> dict[str, Any]:
        self._publish_attempts += 1
        if self._publish_attempts <= self.fail_publish_times:
            return {"error": True, "message": "simulated transient PG publish failure"}
        self.published_draft_ids.append(draft_id)
        return {"id": draft_id, "status": "published"}

    async def update_experience(self, experience_id: str, **kwargs: Any) -> dict[str, Any]:
        self.updated_experiences.append({"experience_id": experience_id, **kwargs})
        return {"id": experience_id, "status": "updated"}


async def verify_live_code() -> tuple[dict[str, tuple[bool, str]], dict[str, tuple[bool, str]]]:
    """Run live code property checks and counterexample trace replays against real Python classes.

    Returns:
      (guarded_properties, finding_repros)
      where each dict maps invariant_name -> (holds_in_code, detail_message).
    """
    from team_memory.daemon.buffers.conversation_buffer import ConversationBuffer
    from team_memory.daemon.buffers.pending_queue import PendingQueue
    from team_memory.daemon.refinement.refinement_worker import RefinementWorker
    from team_memory.daemon.runtime.config import DaemonConfig
    from team_memory.daemon.runtime.config import RefinementSettings as RefinementConfig

    guarded: dict[str, tuple[bool, str]] = {}
    findings: dict[str, tuple[bool, str]] = {}

    # ------------------------------------------------------------------
    # Guarded Property 1: AtMostOnePendingQueueEntry
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="tm-verify-g1-") as tmp:
        db_path = str(Path(tmp) / "daemon.db")
        async with ConversationBuffer(db_path=db_path) as buf:
            queue = PendingQueue(db_path=db_path)
            sink = ObservableSink()
            cfg = RefinementConfig(enabled=True, max_retries=0, fallback_on_failure=False)
            worker = RefinementWorker(config=DaemonConfig(refinement=cfg), buf=buf, sink=sink, queue=queue)

            session_id = "sess-dedup-check"
            await buf.add_message(session_id, "assistant", "Resolved deadlock by lock ordering.", project="team_memory")
            await buf.mark_needs_refinement(session_id)

            # Enqueue twice for the same session_id via worker failure + direct retry enqueue
            with (
                patch(
                    "team_memory.daemon.refinement.refinement_worker._llm_api_reachable",
                    return_value=True,
                ),
                patch(
                    "team_memory.daemon.refinement.refinement_worker._refine_via_openai",
                    side_effect=RuntimeError("LLM parse failure"),
                ),
            ):
                await worker._tick()
                # Attempt a second enqueue for the same active session_id
                queue.enqueue("refine", session_id, {"draft_id": session_id, "content": "retry"})

            active_count = queue._db.execute(
                "SELECT COUNT(*) FROM pending_ops WHERE session_id = ? AND status != 'failed'",
                (session_id,),
            ).fetchone()[0]

        if active_count == 1:
            guarded["AtMostOnePendingQueueEntry"] = (
                True,
                "PendingQueue.enqueue deduplicated active retry ops for session_id (active_count=1).",
            )
        else:
            guarded["AtMostOnePendingQueueEntry"] = (
                False,
                f"VIOLATION: PendingQueue allowed {active_count} active retry ops for the same session_id.",
            )

    # ------------------------------------------------------------------
    # Guarded Property 2: CrashRecoveryAndSkipLLMReplay
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="tm-verify-g2-") as tmp:
        db_path = str(Path(tmp) / "daemon.db")
        async with ConversationBuffer(db_path=db_path) as buf:
            sink = ObservableSink()
            cfg = RefinementConfig(enabled=True, quality_gate_enabled=False)
            worker = RefinementWorker(config=DaemonConfig(refinement=cfg), buf=buf, sink=sink, queue=None)

            # Part A: reset_refining_sessions restores status='needs_refinement'
            s_ref = "sess-crash-refining"
            await buf.add_message(s_ref, "assistant", "Fixed migration issue.", project="team_memory")
            await buf.mark_refining(s_ref)
            await buf.reset_refining_sessions()
            s_ref_row = await buf.get_session(s_ref)
            await buf.close_session(s_ref, "published")

            # Part B: mark_publishing_in_progress + reset_publishing_sessions replays via skip_llm without calling LLM
            s_pub = "sess-crash-publishing"
            await buf.add_message(s_pub, "assistant", "Fixed connection pool.", project="team_memory")
            await buf.mark_publishing_in_progress(
                s_pub,
                {
                    "experience_id": "exp-saved-99",
                    "structured": {
                        "title": "Connection Pool Fix",
                        "problem": "Pool exhausted under load",
                        "solution": "Set max_overflow and pool_pre_ping",
                        "tags": ["db"],
                        "experience_type": "bugfix",
                    },
                },
            )
            await buf.reset_publishing_sessions()
            needs = await buf.get_needs_refinement()
            llm_called = False

            async def _unexpected_llm(*args: Any, **kwargs: Any) -> dict[str, Any]:
                nonlocal llm_called
                llm_called = True
                raise AssertionError("LLM should not be called on skip_llm replay")

            worker._call_llm = _unexpected_llm  # type: ignore[method-assign]
            await worker._tick()
            s_pub_row = await buf.get_session(s_pub)

        ref_ok = s_ref_row is not None and s_ref_row.get("status") == "needs_refinement"
        pub_ok = (
            any(d["id"] == s_pub and d.get("skip_llm") is True for d in needs)
            and not llm_called
            and "exp-saved-99" in sink.published_draft_ids
            and s_pub_row is not None
            and s_pub_row.get("status") == "published"
        )
        if ref_ok and pub_ok:
            guarded["CrashRecoveryAndSkipLLMReplay"] = (
                True,
                "reset_refining_sessions and reset_publishing_sessions + skip_llm replay preserved state cleanly.",
            )
        else:
            guarded["CrashRecoveryAndSkipLLMReplay"] = (
                False,
                f"VIOLATION: ref_ok={ref_ok} (row={s_ref_row}), pub_ok={pub_ok} "
                f"(published={sink.published_draft_ids}, row={s_pub_row}).",
            )

    # ------------------------------------------------------------------
    # Finding 1 Replay: NoStrandedRefiningSession
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="tm-verify-f1-") as tmp:
        db_path = str(Path(tmp) / "daemon.db")
        async with ConversationBuffer(db_path=db_path) as buf:
            queue = PendingQueue(db_path=db_path)
            sink = ObservableSink()
            cfg = RefinementConfig(enabled=True, quality_gate_enabled=False)
            worker = RefinementWorker(config=DaemonConfig(refinement=cfg), buf=buf, sink=sink, queue=queue)

            s_id = "sess-f1-llm-unreachable"
            await buf.add_message(
                s_id,
                "assistant",
                "Detailed debugging notes for pgvector index.",
                project="team_memory",
            )
            await buf.mark_needs_refinement(s_id)

            with patch(
                "team_memory.daemon.refinement.refinement_worker._llm_api_reachable",
                return_value=False,
            ):
                await worker._tick()

            s_row = await buf.get_session(s_id)
            needs_after = await buf.get_needs_refinement()
            due_ops = queue.get_due_ops()

        status_after = s_row.get("status") if s_row else None
        holds = status_after == "needs_refinement" or len(due_ops) > 0
        findings["NoStrandedRefiningSession"] = (
            holds,
            f"After _tick() with _llm_api_reachable=False: session.status={status_after!r}, "
            f"get_needs_refinement()={len(needs_after)}, queue.due_ops={len(due_ops)}, "
            f"sink.saved_drafts={len(sink.saved_drafts)}.",
        )

    # ------------------------------------------------------------------
    # Finding 2 Replay: RejectedDraftNeverPublished
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="tm-verify-f2-") as tmp:
        db_path = str(Path(tmp) / "daemon.db")
        async with ConversationBuffer(db_path=db_path) as buf:
            sink = ObservableSink()
            cfg = RefinementConfig(enabled=True, quality_gate_enabled=True, quality_min_score=70)
            worker = RefinementWorker(config=DaemonConfig(refinement=cfg), buf=buf, sink=sink, queue=None)

            s_id = "sess-f2-quality-rejected"
            await buf.add_message(s_id, "assistant", "Short trivial message.", project="team_memory")
            await buf.mark_needs_refinement(s_id)

            # Return structurally invalid LLM output (missing problem & solution -> rejected by _structural_check)
            with (
                patch(
                    "team_memory.daemon.refinement.refinement_worker._llm_api_reachable",
                    return_value=True,
                ),
                patch(
                    "team_memory.daemon.refinement.refinement_worker._refine_via_openai",
                    return_value={"title": "Trivial", "problem": "", "solution": ""},
                ),
            ):
                await worker._tick()

        holds = len(sink.published_draft_ids) == 0
        findings["RejectedDraftNeverPublished"] = (
            holds,
            f"After _structural_check rejected the draft: sink.published_draft_ids={sink.published_draft_ids}.",
        )

    # ------------------------------------------------------------------
    # Finding 3 Replay: AtMostOnePublishedExperience
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="tm-verify-f3-") as tmp:
        db_path = str(Path(tmp) / "daemon.db")
        async with ConversationBuffer(db_path=db_path) as buf:
            queue = PendingQueue(db_path=db_path)
            # First draft_publish fails (Step 2 publish fails -> enqueues in PendingQueue)
            sink = ObservableSink(fail_publish_times=2)
            cfg = RefinementConfig(enabled=True, quality_gate_enabled=False, fallback_on_failure=False)
            worker = RefinementWorker(config=DaemonConfig(refinement=cfg), buf=buf, sink=sink, queue=queue)

            s_id = "sess-f3-double-publish"
            await buf.add_message(s_id, "assistant", "Root cause and fix for asyncpg timeout.", project="team_memory")
            await buf.mark_needs_refinement(s_id)

            valid_llm_output = {
                "title": "Fix asyncpg pool timeout",
                "problem": "Connections leaked under cancellation",
                "solution": "Release connection in finally block",
                "tags": ["asyncpg"],
                "experience_type": "bugfix",
            }

            with (
                patch(
                    "team_memory.daemon.refinement.refinement_worker._llm_api_reachable",
                    return_value=True,
                ),
                patch(
                    "team_memory.daemon.refinement.refinement_worker._refine_via_openai",
                    return_value=valid_llm_output,
                ),
            ):
                # Tick 1 (Step 2): LLM succeeds, mark_publishing_in_progress saves refined_payload,
                # draft_publish fails (#1), worker enqueues op into PendingQueue.
                await worker._tick()

                # Tick 2 (Step 1): PendingQueue retry runs, draft_publish fails (#2),
                # bump_retry pushes next_retry_at +30s while session.status remains 'publishing'.
                await worker._tick()

                # Daemon restart crash recovery while PendingQueue op is backed off:
                # resets 'publishing' session to 'needs_refinement' without clearing PendingQueue.
                await buf.reset_refining_sessions()
                await buf.reset_publishing_sessions()

                # Tick 3 (Step 2): PendingQueue is backed off, so Step 2 (get_needs_refinement)
                # sees skip_llm=True and publishes Experience #1!
                await worker._tick()

                # Backoff timer expires (simulate 30s elapsed in PendingQueue)
                queue._db.execute("UPDATE pending_ops SET next_retry_at = '2000-01-01T00:00:00+00:00'")
                queue._db.commit()

                # Tick 4 (Step 1): PendingQueue runs again and publishes Experience #2 for the same session!
                await worker._tick()

        holds = len(sink.published_draft_ids) <= 1
        findings["AtMostOnePublishedExperience"] = (
            holds,
            f"After publish failure + restart recovery + queue retry: "
            f"sink.published_draft_ids={sink.published_draft_ids} (count={len(sink.published_draft_ids)}).",
        )

    # ------------------------------------------------------------------
    # Finding 4 Replay: ImporterPreservesPublishingSession
    # ------------------------------------------------------------------
    from team_memory.daemon.refinement.chatgpt_importer import import_chatgpt_sessions

    with tempfile.TemporaryDirectory(prefix="tm-verify-f4-") as tmp:
        db_path = str(Path(tmp) / "daemon.db")
        async with ConversationBuffer(db_path=db_path) as buf:
            s_id = "chatgpt-conv-pub-inflight"
            await buf.add_message(
                s_id,
                "assistant",
                "Root cause analysis for alembic migration lock timeout.",
                project="team_memory",
            )
            await buf.mark_publishing_in_progress(
                s_id,
                {
                    "experience_id": "exp-inflight-1",
                    "structured": {
                        "title": "Alembic Lock Timeout Fix",
                        "problem": "Migration hung waiting on table lock",
                        "solution": "Set lock_timeout before DDL statements",
                        "tags": ["alembic", "postgres"],
                        "experience_type": "bugfix",
                    },
                },
            )
            raw_convs = [
                {
                    "id": "conv-pub-inflight",
                    "title": "Alembic migration lock timeout debugging",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Why does alembic upgrade head hang on table lock in postgres?",
                        },
                        {
                            "role": "assistant",
                            "content": (
                                "Root cause: concurrent open transaction holds AccessShareLock while "
                                "ALTER TABLE requests AccessExclusiveLock.\n"
                                "Solution: run `SET lock_timeout = '5s'` inside env.py and retry:\n"
                                "```python\n"
                                "op.execute(\"SET lock_timeout = '5s'\")\n"
                                "```"
                            ),
                        },
                    ],
                }
            ]
            await import_chatgpt_sessions(
                buf=buf,
                raw_conversations=raw_convs,
                project="team_memory",
                dry_run=False,
            )
            s_row = await buf.get_session(s_id)

        status_after = s_row.get("status") if s_row else None
        holds = status_after == "publishing"
        findings["ImporterPreservesPublishingSession"] = (
            holds,
            f"After import_chatgpt_sessions re-ran on a status='publishing' session: "
            f"session.status={status_after!r} (expected 'publishing').",
        )

    # ------------------------------------------------------------------
    # Finding 5 Replay: ToolObservationTrajectoryContract
    # ------------------------------------------------------------------
    from team_memory.daemon.quality.convergence_detector import ConvergenceDetector
    from team_memory.daemon.refinement.pipeline import process_tool_observation

    with tempfile.TemporaryDirectory(prefix="tm-verify-f5-") as tmp:
        db_path = str(Path(tmp) / "daemon.db")
        async with ConversationBuffer(db_path=db_path) as buf:
            sink = ObservableSink()
            cfg = RefinementConfig(enabled=True, quality_gate_enabled=False)
            daemon_cfg = DaemonConfig(refinement=cfg)
            worker = RefinementWorker(config=daemon_cfg, buf=buf, sink=sink, queue=None)

            s_id = "sess-f5-tool-obs"
            await buf.add_message(s_id, "assistant", "Fixed failing test in server.py.", project="team_memory")

            # Part A: Write path via process_tool_observation -> buf.add_tool_observation
            write_err: str | None = None
            try:
                await process_tool_observation(
                    input_data={
                        "conversation_id": s_id,
                        "tool_name": "Bash",
                        "command": "pytest tests/test_server.py -v",
                        "target_file": "tests/test_server.py",
                        "exit_code": 0,
                        "succeeded": True,
                        "project": "team_memory",
                    },
                    config=daemon_cfg,
                    buf=buf,
                )
            except TypeError as exc:
                write_err = str(exc)

            # Part B: Read path via buf.get_tool_observations -> _refine_one & ConvergenceDetector
            await buf.add_tool_observation(
                s_id,
                "Bash",
                command_summary="pytest tests/test_server.py -v",
                target_files=["tests/test_server.py"],
                exit_code=1,
                error_excerpt="AssertionError",
                succeeded=False,
                project="team_memory",
            )
            await buf.add_tool_observation(
                s_id,
                "Bash",
                command_summary="pytest tests/test_server.py -v",
                target_files=["tests/test_server.py"],
                exit_code=0,
                succeeded=True,
                project="team_memory",
            )
            stored_obs = await buf.get_tool_observations(s_id)
            detector = ConvergenceDetector()
            e2p_converged = detector.check_error_to_pass_resolution(stored_obs)

            await buf.mark_needs_refinement(s_id)
            captured_llm_content: list[str] = []

            def _capture_llm(content: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
                captured_llm_content.append(content)
                return {
                    "title": "Test fix",
                    "problem": "Test failed",
                    "solution": "Updated assertion",
                    "tags": ["test"],
                    "experience_type": "bugfix",
                }

            with (
                patch(
                    "team_memory.daemon.refinement.refinement_worker._llm_api_reachable",
                    return_value=True,
                ),
                patch(
                    "team_memory.daemon.refinement.refinement_worker._refine_via_openai",
                    side_effect=_capture_llm,
                ),
            ):
                await worker._tick()

        llm_prompt_text = captured_llm_content[0] if captured_llm_content else ""
        read_preserved = "pytest tests/test_server.py -v" in llm_prompt_text
        holds = (write_err is None) and read_preserved and e2p_converged
        findings["ToolObservationTrajectoryContract"] = (
            holds,
            f"write_err={write_err!r}, command_in_llm_prompt={read_preserved}, "
            f"error_to_pass_converged={e2p_converged}.",
        )

    # ------------------------------------------------------------------
    # Finding 6 Replay: IngestRoutesSignatureContract
    # ------------------------------------------------------------------
    import ast
    import inspect

    from team_memory.daemon.observe.agy_transcript_watcher import import_agy_sessions

    hooks_path = REPO_ROOT / "src" / "team_memory" / "daemon" / "runtime" / "routes" / "hooks.py"
    hooks_tree = ast.parse(hooks_path.read_text(encoding="utf-8"), filename=str(hooks_path))
    chatgpt_params = set(inspect.signature(import_chatgpt_sessions).parameters.keys())
    agy_params = set(inspect.signature(import_agy_sessions).parameters.keys())

    unknown_chatgpt_kwargs: set[str] = set()
    unknown_agy_kwargs: set[str] = set()
    for node in ast.walk(hooks_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
            if node.func.id == "import_chatgpt_sessions":
                unknown_chatgpt_kwargs |= kw_names - chatgpt_params
            elif node.func.id == "import_agy_sessions":
                unknown_agy_kwargs |= kw_names - agy_params

    holds = not unknown_chatgpt_kwargs and not unknown_agy_kwargs
    findings["IngestRoutesSignatureContract"] = (
        holds,
        f"hooks.py passes unknown kwargs to import_chatgpt_sessions={sorted(unknown_chatgpt_kwargs)} "
        f"and import_agy_sessions={sorted(unknown_agy_kwargs)}.",
    )

    return guarded, findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify TeamMemory Daemon Session Refinement Lifecycle.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any known finding is still violated (default exits non-zero only on regression).",
    )
    args = parser.parse_args()

    java_bin = _find_java()
    jar_path = _find_tla2tools()

    print("=== 1. TLA+ Model Checking (TLC 1.7.4) ===")
    print(f"Spec: {SPEC_FILE}")
    tlc_results: dict[str, bool] = {}
    for inv in ALL_INVARIANTS:
        passed, out = run_tlc_invariant(java_bin, jar_path, SPEC_FILE, inv)
        tlc_results[inv] = passed
        if passed:
            status = "PASS"
        elif inv in KNOWN_FINDINGS:
            status = "FAIL (known finding)"
        else:
            status = "FAIL (REGRESSION)"
        m = re.search(r"(\d+) states generated, (\d+) distinct states found", out)
        states_info = f" [{m.group(2)} distinct states]" if m else ""
        print(f"  - {inv:<32} {status}{states_info}")

    print("\n=== 2. Proposed Fix Model Check (RefinementLifecycleFixed.tla) ===")
    fixed_all_pass = True
    for inv in ALL_INVARIANTS:
        passed, _ = run_tlc_invariant(java_bin, jar_path, FIXED_SPEC_FILE, inv)
        fixed_all_pass = fixed_all_pass and passed
        print(f"  - {inv:<32} {'PASS' if passed else 'FAIL'}")

    print("\n=== 3. Live Python Code Invariant & Trace Replay ===")
    guarded, findings = asyncio.run(verify_live_code())
    for name, (ok, msg) in guarded.items():
        print(f"  - [Guarded] {name:<28} {'PASS' if ok else 'FAIL (REGRESSION)'} — {msg}")
    for name, (ok, msg) in findings.items():
        if ok:
            label = "PASS (fixed!)"
        elif name in KNOWN_FINDINGS:
            label = "FAIL (known finding reproduced)"
        else:
            label = "FAIL (REGRESSION)"
        print(f"  - [Finding] {name:<28} {label} — {msg}")

    regressions: list[str] = []
    for inv in ALL_INVARIANTS:
        if inv not in KNOWN_FINDINGS and not tlc_results[inv]:
            regressions.append(f"TLA+ invariant {inv} broke")
    if not fixed_all_pass:
        regressions.append("RefinementLifecycleFixed.tla failed an invariant")
    for name, (ok, _) in guarded.items():
        if not ok:
            regressions.append(f"Live code guarded property {name} broke")
    for name, (ok, _) in findings.items():
        if name not in KNOWN_FINDINGS and not ok:
            regressions.append(f"Live code property {name} broke")

    if args.strict:
        for inv in KNOWN_FINDINGS:
            if not tlc_results[inv]:
                regressions.append(f"TLA+ finding {inv} still violated")
        for name, (ok, _) in findings.items():
            if not ok:
                regressions.append(f"Live code finding {name} still violated")

    if regressions:
        print("\nVERIFICATION FAILED:")
        for r in regressions:
            print(f"  * {r}")
        return 1

    print(f"\nVERIFICATION OK: All guarded invariants hold and all {len(findings)} known findings are tracked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
