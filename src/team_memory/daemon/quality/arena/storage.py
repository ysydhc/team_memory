from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from team_memory.daemon.quality.arena.models import (
    EpochRecord,
    EvalResult,
    PromptEntry,
    RoundRecord,
    TaskEval,
)
from team_memory.project_resolver import find_project_root


def update_elo(winner_elo: float, loser_elo: float, k: float = 32.0, draw: bool = False) -> tuple[float, float]:
    """Standard ELO update. Returns (new_winner_elo, new_loser_elo)."""
    expected_winner = 1.0 / (1.0 + 10.0 ** ((loser_elo - winner_elo) / 400.0))
    expected_loser = 1.0 - expected_winner

    if draw:
        new_winner = winner_elo + k * (0.5 - expected_winner)
        new_loser = loser_elo + k * (0.5 - expected_loser)
    else:
        new_winner = winner_elo + k * (1.0 - expected_winner)
        new_loser = loser_elo + k * (0.0 - expected_loser)

    return new_winner, new_loser


class ArenaStorage:
    def __init__(self, data_dir: str | None = None) -> None:
        if data_dir is None:
            root = find_project_root()
            self._data_dir = Path(root) / "data" / "arena"
        else:
            self._data_dir = Path(data_dir)

    @property
    def _prompts_path(self) -> Path:
        return self._data_dir / "prompts.json"

    @property
    def _rounds_path(self) -> Path:
        return self._data_dir / "rounds.json"

    @property
    def _epochs_path(self) -> Path:
        return self._data_dir / "epochs.json"

    def _ensure_dir(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_write(self, path: Path, data: dict[str, Any]) -> None:
        self._ensure_dir()
        fd, tmp_path = tempfile.mkstemp(dir=str(self._data_dir), suffix=".tmp", prefix=path.stem)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp_path, str(path))
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def load_prompts(self) -> list[PromptEntry]:
        data = self._read_json(self._prompts_path)
        return [PromptEntry(**entry) for entry in data.get("prompts", [])]

    def save_prompts(self, prompts: list[PromptEntry]) -> None:
        self._atomic_write(
            self._prompts_path,
            {"prompts": [dataclasses.asdict(p) for p in prompts]},
        )

    def get_active_prompts(self) -> list[PromptEntry]:
        return [p for p in self.load_prompts() if not p.retired]

    def get_champion(self) -> PromptEntry | None:
        active = self.get_active_prompts()
        if not active:
            return None
        return max(active, key=lambda p: p.elo)

    # ------------------------------------------------------------------
    # Rounds
    # ------------------------------------------------------------------

    def load_rounds(self) -> list[RoundRecord]:
        data = self._read_json(self._rounds_path)
        raw_rounds = data.get("rounds", [])
        rounds: list[RoundRecord] = []
        for r in raw_rounds:
            tasks = [
                TaskEval(
                    buffer_entry_id=t["buffer_entry_id"],
                    query=t["query"],
                    context_count=t["context_count"],
                    eval_a=EvalResult(**t["eval_a"]) if t.get("eval_a") else None,
                    eval_b=EvalResult(**t["eval_b"]) if t.get("eval_b") else None,
                )
                for t in r.get("tasks", [])
            ]
            rounds.append(
                RoundRecord(
                    round_id=r["round_id"],
                    epoch=r["epoch"],
                    timestamp=r["timestamp"],
                    prompt_a_id=r["prompt_a_id"],
                    prompt_b_id=r["prompt_b_id"],
                    tasks=tasks,
                    meta_verdict=r.get("meta_verdict", ""),
                    meta_reasoning=r.get("meta_reasoning", ""),
                )
            )
        return rounds

    def append_round(self, record: RoundRecord) -> None:
        data = self._read_json(self._rounds_path)
        rounds = data.get("rounds", [])
        rounds.append(dataclasses.asdict(record))
        self._atomic_write(self._rounds_path, {"rounds": rounds})

    # ------------------------------------------------------------------
    # Epochs
    # ------------------------------------------------------------------

    def load_epochs(self) -> list[EpochRecord]:
        data = self._read_json(self._epochs_path)
        return [EpochRecord(**entry) for entry in data.get("epochs", [])]

    def append_epoch(self, record: EpochRecord) -> None:
        data = self._read_json(self._epochs_path)
        epochs = data.get("epochs", [])
        epochs.append(dataclasses.asdict(record))
        self._atomic_write(self._epochs_path, {"epochs": epochs})

    def get_current_epoch(self) -> int:
        return len(self.load_epochs()) + 1

    def is_converged(self, threshold: int = 30) -> bool:
        """True if the current champion has been unbeaten for threshold rounds."""
        champion = self.get_champion()
        if champion is None:
            return False

        rounds = self.load_rounds()
        if not rounds:
            return False

        recent = rounds[-threshold:]
        if len(recent) < threshold:
            return False

        for r in recent:
            if champion.id not in (r.prompt_a_id, r.prompt_b_id):
                continue
            if r.meta_verdict == "a_wins" and r.prompt_b_id == champion.id:
                return False
            if r.meta_verdict == "b_wins" and r.prompt_a_id == champion.id:
                return False

        return True
