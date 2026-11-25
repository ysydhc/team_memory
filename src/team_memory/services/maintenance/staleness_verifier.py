"""Code-Grounded Staleness Verifier (Defense 2).

Verifies `evidence_anchors` (`file_paths`, `symbols`, `commit_sha`) of active
experiences against the actual Git repository working tree so that deleted,
renamed, or heavily refactored code automatically marks dependent experiences
as `possibly_stale` without ever harming anchor-free conceptual/architectural
experiences.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import and_, select

from team_memory.storage.database import get_session
from team_memory.storage.models import Experience

logger = logging.getLogger("team_memory.staleness_verifier")


@dataclass
class AnchorCheckResult:
    """Outcome of checking one experience's code anchors against a repo."""

    experience_id: str
    title: str
    old_status: str
    new_status: str
    changed: bool
    reasons: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    missing_symbols: list[str] = field(default_factory=list)


class StalenessVerifier:
    """Verifies code-grounded evidence anchors against a local repository."""

    def __init__(
        self,
        db_url: str,
        *,
        project_repo_map: dict[str, str] | None = None,
        churn_lines_with_symbols: int = 80,
        churn_lines_without_symbols: int = 150,
        on_mutation: Callable[[], Any] | None = None,
    ) -> None:
        self._db_url = db_url
        self._project_repo_map = dict(project_repo_map or {})
        self._churn_lines_with_symbols = max(1, int(churn_lines_with_symbols))
        self._churn_lines_without_symbols = max(1, int(churn_lines_without_symbols))
        self._on_mutation = on_mutation

    @staticmethod
    def _build_repo_file_index(repo_root: Path) -> list[str]:
        """Collect relative file paths inside repo_root (via git ls-files or walk)."""
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo_root), "ls-files"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if files:
                return files
        except Exception:
            pass

        rel_files: list[str] = []
        skip_dirs = {".git", "node_modules", ".venv", "__pycache__", ".pytest_cache", "dist", "build"}
        for p in repo_root.rglob("*"):
            if any(part in skip_dirs for part in p.parts):
                continue
            if p.is_file():
                try:
                    rel_files.append(str(p.relative_to(repo_root)).replace("\\", "/"))
                except ValueError:
                    pass
        return rel_files

    @staticmethod
    def resolve_anchor_path(
        anchor_path: str,
        repo_root: Path,
        repo_files: list[str],
    ) -> Path | None:
        """Resolve an anchor file_path inside repo_root with safe suffix matching.

        Lookup order:
          1. Exact relative path `repo_root / fp`
          2. `repo_root / "src" / fp`
          3. Unique suffix match in `repo_files` (only when exactly 1 file matches)
        """
        norm = anchor_path.strip().lstrip("./").replace("\\", "/")
        if not norm:
            return None

        # Strip repo_root prefix if an absolute path belonging to this repo was stored
        root_str = str(repo_root.resolve()).replace("\\", "/").rstrip("/")
        if norm.startswith(root_str + "/"):
            norm = norm[len(root_str) + 1 :]

        direct = repo_root / norm
        if direct.is_file():
            return direct

        src_prefixed = repo_root / "src" / norm
        if src_prefixed.is_file():
            return src_prefixed

        suffix = "/" + norm
        matches = [rf for rf in repo_files if rf == norm or rf.endswith(suffix)]
        if len(matches) == 1:
            candidate = repo_root / matches[0]
            if candidate.is_file():
                return candidate

        return None

    @staticmethod
    def _git_file_churn_lines(
        repo_root: Path,
        rel_path: str,
        *,
        since_sha: str | None = None,
        since_dt: datetime | None = None,
    ) -> int:
        """Count total added + deleted lines for rel_path since commit_sha or timestamp."""
        cmd = ["git", "-C", str(repo_root), "log", "--numstat", "--pretty=format:"]
        if since_sha:
            cmd.append(f"{since_sha}..HEAD")
        elif since_dt is not None:
            cmd.append(f"--since={since_dt.isoformat()}")
        else:
            return 0
        cmd.extend(["--", rel_path])

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
            if proc.returncode != 0:
                return 0
        except Exception:
            return 0

        total = 0
        for line in proc.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    added = int(parts[0]) if parts[0].isdigit() else 0
                    deleted = int(parts[1]) if parts[1].isdigit() else 0
                    total += added + deleted
                except ValueError:
                    pass
        return total

    def evaluate_anchors(
        self,
        *,
        experience_id: str,
        title: str,
        current_status: str,
        anchors: dict[str, Any] | None,
        repo_root: Path,
        repo_files: list[str],
        since_dt: datetime | None = None,
        deleted_files_hint: set[str] | None = None,
        churn_cache: dict[tuple[str, str], int] | None = None,
    ) -> AnchorCheckResult:
        """Evaluate one experience's code anchors against repo_root (pure local check)."""
        if not isinstance(anchors, dict):
            return AnchorCheckResult(
                experience_id=experience_id,
                title=title,
                old_status=current_status,
                new_status=current_status,
                changed=False,
            )

        file_paths = [str(p).strip() for p in (anchors.get("file_paths") or []) if str(p).strip()]
        if not file_paths:
            # Anchor-free conceptual/architectural experience: never mark stale.
            return AnchorCheckResult(
                experience_id=experience_id,
                title=title,
                old_status=current_status,
                new_status=current_status,
                changed=False,
            )

        symbols = [str(s).strip() for s in (anchors.get("symbols") or []) if str(s).strip()]
        commit_sha = str(anchors.get("commit_sha") or "").strip() or None
        deleted_hint = deleted_files_hint or set()

        missing_files: list[str] = []
        resolved_files: list[Path] = []
        reasons: list[str] = []

        for fp in file_paths:
            norm = fp.lstrip("./").replace("\\", "/")
            if norm in deleted_hint:
                missing_files.append(fp)
                continue
            resolved = self.resolve_anchor_path(fp, repo_root, repo_files)
            if resolved is None:
                missing_files.append(fp)
            else:
                resolved_files.append(resolved)

        if missing_files:
            reasons.append(f"missing_files: {', '.join(missing_files[:5])}")
            new_status = "possibly_stale" if current_status == "fresh" else current_status
            return AnchorCheckResult(
                experience_id=experience_id,
                title=title,
                old_status=current_status,
                new_status=new_status,
                changed=(new_status != current_status),
                reasons=reasons,
                missing_files=missing_files,
            )

        # All anchored files exist — check for heavy refactor / missing anchored symbols.
        heavy_churn_files: list[tuple[Path, int]] = []
        since_key = commit_sha or (since_dt.strftime("%Y-%m-%d") if since_dt is not None else "")
        for rf in resolved_files:
            try:
                rel = str(rf.relative_to(repo_root)).replace("\\", "/")
            except ValueError:
                continue
            cache_key = (rel, since_key)
            if churn_cache is not None and cache_key in churn_cache:
                churn = churn_cache[cache_key]
            else:
                churn = self._git_file_churn_lines(
                    repo_root,
                    rel,
                    since_sha=commit_sha,
                    since_dt=since_dt,
                )
                if churn_cache is not None:
                    churn_cache[cache_key] = churn
            threshold = (
                self._churn_lines_with_symbols if symbols else self._churn_lines_without_symbols
            )
            if churn >= threshold:
                heavy_churn_files.append((rf, churn))

        if heavy_churn_files:
            if symbols:
                # Symbol-level verification with word-boundary matching to avoid substring false negatives.
                combined_source = ""
                for rf in resolved_files:
                    try:
                        combined_source += "\n" + rf.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        pass

                def _symbol_present(sym: str) -> bool:
                    short = sym.split(".")[-1]
                    for token in (sym, short):
                        pat = rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"
                        if token and re.search(pat, combined_source):
                            return True
                    return False

                missing_syms = [sym for sym in symbols if not _symbol_present(sym)]
                if missing_syms:
                    reasons.append(f"missing_symbols_after_churn: {', '.join(missing_syms[:5])}")
                    new_status = "possibly_stale" if current_status == "fresh" else current_status
                    return AnchorCheckResult(
                        experience_id=experience_id,
                        title=title,
                        old_status=current_status,
                        new_status=new_status,
                        changed=(new_status != current_status),
                        reasons=reasons,
                        missing_symbols=missing_syms,
                    )
            else:
                reasons.append(
                    f"heavy_refactor: {heavy_churn_files[0][0].name} ({heavy_churn_files[0][1]} lines)"
                )
                new_status = "possibly_stale" if current_status == "fresh" else current_status
                return AnchorCheckResult(
                    experience_id=experience_id,
                    title=title,
                    old_status=current_status,
                    new_status=new_status,
                    changed=(new_status != current_status),
                    reasons=reasons,
                )

        return AnchorCheckResult(
            experience_id=experience_id,
            title=title,
            old_status=current_status,
            new_status="fresh",
            changed=(current_status != "fresh"),
            reasons=["anchors_verified"],
        )

    async def verify_project(
        self,
        *,
        project: str,
        repo_path: str | Path | None = None,
        dry_run: bool = False,
        deleted_files: list[str] | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Verify all anchored active experiences for a single project (tenant RLS aware)."""
        raw_repo = repo_path or self._project_repo_map.get(project)
        if not raw_repo:
            return {
                "project": project,
                "skipped": True,
                "reason": "no_repo_mapping",
                "checked": 0,
                "marked_possibly_stale": 0,
                "reverified_fresh": 0,
                "skipped_no_anchors": 0,
                "details": [],
            }

        repo_root = Path(raw_repo).expanduser().resolve()
        if not repo_root.is_dir():
            return {
                "project": project,
                "skipped": True,
                "reason": f"repo_not_found: {repo_root}",
                "checked": 0,
                "marked_possibly_stale": 0,
                "reverified_fresh": 0,
                "skipped_no_anchors": 0,
                "details": [],
            }

        repo_files = self._build_repo_file_index(repo_root)
        deleted_hint = {
            str(f).strip().lstrip("./").replace("\\", "/") for f in (deleted_files or []) if f
        }
        now = datetime.now(timezone.utc)

        checked = 0
        marked_stale = 0
        reverified_fresh = 0
        skipped_no_anchors = 0
        details: list[dict[str, Any]] = []
        any_db_update = False
        churn_cache: dict[tuple[str, str], int] = {}

        async with get_session(self._db_url) as session:
            stmt = (
                select(Experience)
                .where(
                    and_(
                        Experience.project == project,
                        Experience.is_deleted == False,  # noqa: E712
                        Experience.valid_until.is_(None),
                        Experience.exp_status.in_(["published", "promoted"]),
                    )
                )
                .limit(limit)
            )
            res = await session.execute(stmt)
            experiences = res.scalars().all()

            for exp in experiences:
                anchors = getattr(exp, "evidence_anchors", None)
                if not isinstance(anchors, dict) or not anchors.get("file_paths"):
                    skipped_no_anchors += 1
                    continue

                checked += 1
                current_status = getattr(exp, "staleness_status", "fresh") or "fresh"
                since_dt = getattr(exp, "last_verified_at", None) or getattr(exp, "updated_at", None)

                outcome = self.evaluate_anchors(
                    experience_id=str(exp.id),
                    title=exp.title or "",
                    current_status=current_status,
                    anchors=anchors,
                    repo_root=repo_root,
                    repo_files=repo_files,
                    since_dt=since_dt,
                    deleted_files_hint=deleted_hint,
                    churn_cache=churn_cache,
                )

                if outcome.new_status == "possibly_stale" and outcome.changed:
                    marked_stale += 1
                    details.append(
                        {
                            "id": outcome.experience_id,
                            "title": outcome.title[:100],
                            "old_status": outcome.old_status,
                            "new_status": outcome.new_status,
                            "reasons": outcome.reasons,
                        }
                    )
                    if not dry_run:
                        exp.staleness_status = "possibly_stale"
                        exp.updated_at = now
                        any_db_update = True
                elif outcome.new_status == "fresh":
                    if outcome.changed:
                        reverified_fresh += 1
                    if not dry_run:
                        if outcome.changed:
                            exp.staleness_status = "fresh"
                        exp.last_verified_at = now
                        any_db_update = True

            if any_db_update and not dry_run:
                await session.commit()

        if any_db_update and not dry_run and self._on_mutation is not None:
            try:
                maybe_coro = self._on_mutation()
                if hasattr(maybe_coro, "__await__"):
                    await maybe_coro
            except Exception:
                logger.debug("on_mutation callback failed after staleness verification", exc_info=True)

        return {
            "project": project,
            "repo_path": str(repo_root),
            "dry_run": dry_run,
            "checked": checked,
            "marked_possibly_stale": marked_stale,
            "reverified_fresh": reverified_fresh,
            "skipped_no_anchors": skipped_no_anchors,
            "details": details,
        }
