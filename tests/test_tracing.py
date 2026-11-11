"""Span tree and attributes produced by a task, checked with an in-memory exporter."""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ai_slo.demo_tools import Scratchpad, default_tools
from ai_slo.ollama_client import ChatResult
from ai_slo.runner import Task, run_task

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)  # set once for this test module


class ScriptedClient:
    model = "test-model"

    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, tools=None):
        return self.script.pop(0)


def _call(name, arguments):
    return ChatResult(
        message={"role": "assistant", "content": "",
                 "tool_calls": [{"function": {"name": name, "arguments": arguments}}]},
        total_duration_ms=1, load_duration_ms=20.0, prompt_eval_duration_ms=30.0,
        prompt_tokens=10, completion_tokens=5,
    )


def _text(content):
    return ChatResult(
        message={"role": "assistant", "content": content},
        total_duration_ms=1, load_duration_ms=5.0, prompt_eval_duration_ms=6.0,
        prompt_tokens=7, completion_tokens=3,
    )


def _spans_for_run():
    _exporter.clear()
    client = ScriptedClient([_call("add", {"a": 1, "b": 2}), _text("3")])
    run_task(client, Task("add them", "arithmetic"), default_tools(inject_failure=False))
    return _exporter.get_finished_spans()


def test_one_root_span_with_chat_and_tool_children():
    spans = _spans_for_run()
    roots = [s for s in spans if s.parent is None]
    assert [s.name for s in roots] == ["invoke_agent"]
    root = roots[0]
    children = [s for s in spans if s.parent and s.parent.span_id == root.context.span_id]
    names = sorted(s.name for s in children)
    assert names == ["chat", "chat", "execute_tool"]


def test_root_span_carries_task_attributes():
    root = next(s for s in _spans_for_run() if s.parent is None)
    attrs = root.attributes
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["openinference.span.kind"] == "AGENT"
    assert attrs["agent.terminal_state"] == "completed"
    assert attrs["agent.steps_taken"] == 2
    assert attrs["agent.task_class"] == "arithmetic"
    assert attrs["agent.total_tokens"] == 25  # (10+5) + (7+3)


def test_tool_span_has_tool_name_and_outcome():
    tool = next(s for s in _spans_for_run() if s.name == "execute_tool")
    assert tool.attributes["gen_ai.tool.name"] == "add"
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool.attributes["agent.tool_outcome"] == "ok"


def test_chat_span_has_token_and_ollama_timing_attributes():
    chats = [s for s in _spans_for_run() if s.name == "chat"]
    first = chats[0].attributes
    assert first["gen_ai.usage.input_tokens"] == 10
    assert first["ollama.load_duration_ms"] == 20.0
    assert first["ollama.prompt_eval_duration_ms"] == 30.0
    assert first["gen_ai.request.model"] == "test-model"


def test_tool_failure_marks_span_as_error():
    _exporter.clear()
    client = ScriptedClient([_call("web_lookup", {"query": "x"}), _text("sorry")])
    run_task(client, Task("look", "research"), default_tools(inject_failure=True))
    tool = next(s for s in _exporter.get_finished_spans() if s.name == "execute_tool")
    assert tool.attributes["agent.tool_outcome"] == "tool_failure"
    assert tool.status.status_code == trace.StatusCode.ERROR


def test_invalid_call_records_rejection_reason():
    _exporter.clear()
    client = ScriptedClient([_call("add", {"a": 1}), _text("done")])
    run_task(client, Task("bad args", "arithmetic"), default_tools(inject_failure=False))
    tool = next(s for s in _exporter.get_finished_spans() if s.name == "execute_tool")
    assert tool.attributes["agent.tool_outcome"] == "agent_invalid_call"
    assert "invalid arguments" in tool.attributes["agent.tool_rejection_reason"]


def test_memory_spans_nest_under_the_tool_span():
    _exporter.clear()
    pad = Scratchpad(inject_stale=True)
    pad.write("k", "a")
    pad.write("k", "b")
    pad.read("k")  # stale on the first read when injection is on
    spans = {s.name: s for s in _exporter.get_finished_spans()}
    read = [s for s in _exporter.get_finished_spans() if s.name == "memory.read"][0]
    assert read.attributes["memory.observed_version"] == 1
    assert read.attributes["memory.latest_acked_version"] == 2
    assert "memory.write" in spans
