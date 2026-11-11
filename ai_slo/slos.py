"""SLO definitions.

Every SLI is a ratio: good events / valid events (the article's definition).
Each SLO extracts a list of booleans (True = good) from one task record.
Events that are not valid for an SLO are simply not returned.

Targets and thresholds are my placeholders; the article gives no numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .record import TOOL_AGENT_INVALID, TOOL_FAILURE, TaskRecord


@dataclass(frozen=True)
class Thresholds:
    tool_latency_ms: float = 2_000
    model_latency_ms: float = 5_000
    invocation_latency_ms: float = 60_000
    token_budget: int = 20_000


@dataclass(frozen=True)
class SLO:
    name: str
    group: str  # "requests" | "pipeline" | "storage" (Google SRE style grouping)
    action: str  # "page" | "ticket", as the article suggests per group
    target: float  # fraction, e.g. 0.99
    description: str
    extract: Callable[[TaskRecord, Thresholds], list[bool]]


def _invocation_availability(t: TaskRecord, _: Thresholds) -> list[bool]:
    return [not t.infrastructure_error]


def _invocation_latency(t: TaskRecord, th: Thresholds) -> list[bool]:
    return [t.duration_ms <= th.invocation_latency_ms]


def _tool_availability(t: TaskRecord, _: Thresholds) -> list[bool]:
    return [c.outcome != TOOL_FAILURE for c in t.tool_calls]


def _tool_call_validity(t: TaskRecord, _: Thresholds) -> list[bool]:
    return [c.outcome != TOOL_AGENT_INVALID for c in t.tool_calls]


def _tool_latency(t: TaskRecord, th: Thresholds) -> list[bool]:
    return [c.duration_ms <= th.tool_latency_ms for c in t.tool_calls]


def _model_availability(t: TaskRecord, _: Thresholds) -> list[bool]:
    return [not m.error for m in t.model_calls]


def _model_latency(t: TaskRecord, th: Thresholds) -> list[bool]:
    # Time to first token ~= load_duration + prompt_eval_duration (Ollama).
    # Calls without both fields cannot be measured, so they are not valid events.
    out = []
    for m in t.model_calls:
        if m.load_duration_ms is None or m.prompt_eval_duration_ms is None:
            continue
        ttft = m.load_duration_ms + m.prompt_eval_duration_ms
        out.append(ttft <= th.model_latency_ms)
    return out


def _task_completion(t: TaskRecord, _: Thresholds) -> list[bool]:
    return [t.terminal_state == "completed"]


def _autonomous_completion(t: TaskRecord, _: Thresholds) -> list[bool]:
    return [t.terminal_state == "completed" and not t.human_intervened]


def _token_budget(t: TaskRecord, th: Thresholds) -> list[bool]:
    return [t.total_tokens <= th.token_budget]


def _memory_consistency(t: TaskRecord, _: Thresholds) -> list[bool]:
    # Good when the read saw the latest version acknowledged before the read.
    out = []
    for r in t.memory_reads:
        if r.observed_version is None or r.latest_acked_version is None:
            continue
        out.append(r.observed_version >= r.latest_acked_version)
    return out


SLOS: tuple[SLO, ...] = (
    SLO("invocation_availability", "requests", "page", 0.99,
        "Task invocation was not ended by an infrastructure error.", _invocation_availability),
    SLO("invocation_latency", "requests", "page", 0.95,
        "Task finished within the invocation latency threshold.", _invocation_latency),
    SLO("tool_availability", "requests", "page", 0.99,
        "Tool call did not fail because the tool server failed.", _tool_availability),
    SLO("tool_call_validity", "requests", "page", 0.99,
        "Tool call was not rejected for invalid arguments sent by the agent.", _tool_call_validity),
    SLO("tool_latency", "requests", "page", 0.95,
        "Tool call finished within the tool latency threshold.", _tool_latency),
    SLO("model_availability", "requests", "page", 0.99,
        "Model call completed without error.", _model_availability),
    SLO("model_latency", "requests", "page", 0.95,
        "Approximate time to first token (load + prompt eval) within threshold.", _model_latency),
    SLO("task_completion", "pipeline", "ticket", 0.95,
        "Task reached the 'completed' terminal state.", _task_completion),
    SLO("autonomous_completion", "pipeline", "ticket", 0.90,
        "Task completed without human intervention.", _autonomous_completion),
    SLO("token_budget", "pipeline", "ticket", 0.95,
        "Task total token count stayed within budget.", _token_budget),
    SLO("memory_consistency", "storage", "page", 0.999,
        "Memory read returned the latest acknowledged version.", _memory_consistency),
)
