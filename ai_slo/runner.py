"""Run agent tasks against a model and emit one TaskRecord per task.

The loop is written by hand (not a framework) so every span and record comes
from code in this repo, as the article recommends.

Tool outcomes (see record.py):
  - unknown tool name, or arguments that do not match the handler  -> agent_invalid_call
  - the handler raises                                               -> tool_failure
"""

from __future__ import annotations

import inspect
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from . import attributes as A
from .ollama_client import OllamaClient, OllamaError
from .record import TOOL_AGENT_INVALID, TOOL_FAILURE, TOOL_OK, ModelCall, TaskRecord, ToolCall
from opentelemetry import trace

from .tracing import get_tracer, mark_error, set_common_attrs, set_messages


@dataclass
class AgentTool:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict[str, Any]  # JSON schema for the arguments

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Task:
    prompt: str
    task_class: str


SYSTEM_PROMPT = (
    "You are a careful assistant. Use the provided tools when they help. "
    "When you have the final answer, reply with it in plain text and no tool call."
)


def _invoke_tool(tool: AgentTool, arguments: Any) -> tuple[str, Any]:
    """Return (outcome, result_or_error_text)."""
    if not isinstance(arguments, dict):
        return TOOL_AGENT_INVALID, "arguments must be a JSON object"
    # Check the call against the handler's signature *before* running it, so
    # bad arguments are the agent's fault, not the tool's.
    try:
        inspect.signature(tool.fn).bind(**arguments)
    except TypeError as exc:
        return TOOL_AGENT_INVALID, f"invalid arguments: {exc}"
    try:
        return TOOL_OK, tool.fn(**arguments)
    except Exception as exc:  # the tool itself broke
        return TOOL_FAILURE, f"tool error: {exc}"


def _run_loop(
    client: OllamaClient,
    task: Task,
    tools: Sequence[AgentTool],
    max_steps: int,
    token_budget: int,
    task_id: Optional[str],
) -> TaskRecord:
    tracer = get_tracer()
    by_name = {t.name: t for t in tools}
    specs = [t.spec() for t in tools]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]

    started = time.perf_counter()
    record = TaskRecord(
        task_id=task_id or uuid.uuid4().hex[:8],
        task_class=task.task_class,
        duration_ms=0.0,
        terminal_state="failed",
    )

    last_text = {"value": ""}  # latest assistant text, used as the task's output

    def finish(state: str) -> TaskRecord:
        record.terminal_state = state
        record.duration_ms = (time.perf_counter() - started) * 1000
        # finish() runs inside the invoke_agent span, so this sets the root's output.
        trace.get_current_span().set_attribute("output.value", last_text["value"] or state)
        return record

    while True:
        if record.steps_taken >= max_steps:
            return finish("max_steps")

        t0 = time.perf_counter()
        with tracer.start_as_current_span("chat") as cspan:
            set_common_attrs(cspan, "LLM", A.OP_CHAT)
            cspan.set_attribute("gen_ai.request.model", str(getattr(client, "model", "unknown")))
            set_messages(cspan, "llm.input_messages", messages)
            try:
                result = client.chat(messages, tools=specs)
            except OllamaError as exc:
                mark_error(cspan, str(exc))
                cspan.record_exception(exc)
                record.model_calls.append(ModelCall(
                    duration_ms=(time.perf_counter() - t0) * 1000, error=True))
                record.infrastructure_error = True
                return finish("failed")
            set_messages(cspan, "llm.output_messages", [result.message])
            cspan.set_attribute("output.value", result.content or json.dumps(result.tool_calls))
            cspan.set_attribute(A.GEN_AI_USAGE_INPUT_TOKENS, result.prompt_tokens)
            cspan.set_attribute("gen_ai.usage.output_tokens", result.completion_tokens)
            if result.load_duration_ms is not None:
                cspan.set_attribute("ollama.load_duration_ms", result.load_duration_ms)
            if result.prompt_eval_duration_ms is not None:
                cspan.set_attribute("ollama.prompt_eval_duration_ms", result.prompt_eval_duration_ms)

        record.model_calls.append(ModelCall(
            duration_ms=(time.perf_counter() - t0) * 1000,
            load_duration_ms=result.load_duration_ms,
            prompt_eval_duration_ms=result.prompt_eval_duration_ms,
        ))
        record.steps_taken += 1
        record.total_tokens += result.prompt_tokens + result.completion_tokens

        if record.total_tokens > token_budget:
            return finish("budget_exceeded")

        messages.append(result.message)
        if result.content.strip():
            last_text["value"] = result.content
        if not result.tool_calls:
            return finish("completed" if result.content.strip() else "failed")

        for call in result.tool_calls:
            fn_part = call.get("function", {})
            name = fn_part.get("name", "")
            args = fn_part.get("arguments", {})
            if isinstance(args, str):  # some models send arguments as a JSON string
                try:
                    args = json.loads(args)
                except ValueError:
                    args = None
            tool = by_name.get(name)
            t0 = time.perf_counter()
            with tracer.start_as_current_span("execute_tool") as tspan:
                set_common_attrs(tspan, "TOOL", A.OP_EXECUTE_TOOL)
                tspan.set_attribute(A.GEN_AI_TOOL_NAME, name or "<unnamed>")
                tspan.set_attribute("input.value", json.dumps(args, default=str))
                if tool is None:
                    outcome, payload = TOOL_AGENT_INVALID, f"unknown tool: {name}"
                else:
                    outcome, payload = _invoke_tool(tool, args)
                tspan.set_attribute("output.value", str(payload))
                tspan.set_attribute("agent.tool_outcome", outcome)
                if outcome == TOOL_AGENT_INVALID:
                    tspan.set_attribute(A.AGENT_TOOL_REJECTION_REASON, str(payload))
                elif outcome == TOOL_FAILURE:
                    mark_error(tspan, str(payload))
            record.tool_calls.append(ToolCall(
                name=name or "<unnamed>",
                duration_ms=(time.perf_counter() - t0) * 1000,
                outcome=outcome,
            ))
            messages.append({
                "role": "tool",
                "name": name,
                "content": payload if isinstance(payload, str) else json.dumps(payload),
            })


def run_task(
    client: OllamaClient,
    task: Task,
    tools: Sequence[AgentTool],
    max_steps: int = 6,
    token_budget: int = 20_000,
    task_id: Optional[str] = None,
) -> TaskRecord:
    """Run one task inside an invoke_agent root span and return its record."""
    tracer = get_tracer()
    with tracer.start_as_current_span("invoke_agent") as span:
        set_common_attrs(span, "AGENT", A.OP_INVOKE_AGENT)
        span.set_attribute("input.value", task.prompt)
        span.set_attribute(A.AGENT_TASK_CLASS, task.task_class)
        rec = _run_loop(client, task, tools, max_steps, token_budget, task_id)
        span.set_attribute(A.GEN_AI_CONVERSATION_ID, rec.task_id)
        span.set_attribute(A.AGENT_TERMINAL_STATE, rec.terminal_state)
        span.set_attribute(A.AGENT_STEPS_TAKEN, rec.steps_taken)
        span.set_attribute(A.AGENT_HUMAN_INTERVENTION, rec.human_intervened)
        span.set_attribute("agent.total_tokens", rec.total_tokens)  # root-level total, per the article
        if rec.terminal_state == "failed":
            mark_error(span, "task failed")
        return rec
