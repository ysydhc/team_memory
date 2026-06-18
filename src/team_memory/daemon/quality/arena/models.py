from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalResult:
    relevance: float
    helpfulness: float
    reasoning: str = ""


@dataclass
class TaskEval:
    buffer_entry_id: str
    query: str
    context_count: int
    eval_a: EvalResult | None = None
    eval_b: EvalResult | None = None


@dataclass
class PromptEntry:
    id: str
    version: int
    text: str
    elo: float = 1500.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    created_at: str = ""
    retired: bool = False
    parent_id: str | None = None


@dataclass
class RoundRecord:
    round_id: str
    epoch: int
    timestamp: str
    prompt_a_id: str
    prompt_b_id: str
    tasks: list[TaskEval] = field(default_factory=list)
    meta_verdict: str = ""
    meta_reasoning: str = ""


@dataclass
class EpochRecord:
    epoch: int
    rounds_completed: int
    champion_id: str
    champion_elo: float
    retired_id: str | None = None
    new_prompt_id: str | None = None
    timestamp: str = ""
