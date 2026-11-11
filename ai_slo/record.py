"""One structured record per agent task.

The article's design rule: compute SLOs from a stable record written by the
application at the end of each task, not from the trace backend.

Each task is one JSON object on one line (JSONL).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

# Terminal states. The article says there are eight but does not list them;
# these are my reconstruction from its examples (see PLAN.md).
TERMINAL_STATES = frozenset(
    {
        "completed",
        "failed",
        "timeout",
        "cancelled",
        "refused",
        "handed_to_human",
        "budget_exceeded",
        "max_steps",
    }
)

# Outcomes of a single tool call.
TOOL_OK = "ok"
TOOL_FAILURE = "tool_failure"  # the tool server is broken: a dependency incident
TOOL_AGENT_INVALID = "agent_invalid_call"  # the agent sent bad arguments: a ticket

TOOL_OUTCOMES = frozenset({TOOL_OK, TOOL_FAILURE, TOOL_AGENT_INVALID})


@dataclass
class ToolCall:
    name: str
    duration_ms: float
    outcome: str = TOOL_OK

    def __post_init__(self) -> None:
        if self.outcome not in TOOL_OUTCOMES:
            raise ValueError(f"unknown tool outcome: {self.outcome!r}")


@dataclass
class ModelCall:
    duration_ms: float
    error: bool = False
    # Ollama reports these even for non-streaming requests. Their sum
    # approximates time to first token (the article's approach).
    load_duration_ms: Optional[float] = None
    prompt_eval_duration_ms: Optional[float] = None


@dataclass
class MemoryRead:
    observed_version: Optional[int]
    latest_acked_version: Optional[int]


@dataclass
class TaskRecord:
    task_id: str
    task_class: str
    duration_ms: float
    terminal_state: str
    steps_taken: int = 0
    human_intervened: bool = False
    infrastructure_error: bool = False  # the invocation failed for an infra reason
    total_tokens: int = 0  # root-level total, as the article's token budget needs
    tool_calls: list[ToolCall] = field(default_factory=list)
    model_calls: list[ModelCall] = field(default_factory=list)
    memory_reads: list[MemoryRead] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.terminal_state not in TERMINAL_STATES:
            raise ValueError(f"unknown terminal state: {self.terminal_state!r}")

    @classmethod
    def from_dict(cls, data: dict) -> "TaskRecord":
        data = dict(data)
        data["tool_calls"] = [ToolCall(**t) for t in data.get("tool_calls", [])]
        data["model_calls"] = [ModelCall(**m) for m in data.get("model_calls", [])]
        data["memory_reads"] = [MemoryRead(**m) for m in data.get("memory_reads", [])]
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def load_records(lines: Iterable[str]) -> list[TaskRecord]:
    """Parse JSONL text. Blank lines are skipped."""
    records = []
    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(TaskRecord.from_dict(json.loads(line)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc
    return records
