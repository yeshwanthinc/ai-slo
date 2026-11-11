"""Runner and Ollama client tests. No network: the model is a scripted fake."""

import pytest
import requests

from ai_slo.demo_tools import default_tools
from ai_slo.ollama_client import ChatResult, OllamaClient, OllamaError, parse_chat_response
from ai_slo.runner import AgentTool, Task, run_task


class FakeClient:
    """Stands in for OllamaClient. Each script item is a ChatResult or an exception."""

    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, tools=None):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _text(content, prompt=10, completion=5):
    return ChatResult(
        message={"role": "assistant", "content": content},
        total_duration_ms=100, load_duration_ms=20, prompt_eval_duration_ms=30,
        prompt_tokens=prompt, completion_tokens=completion,
    )


def _call(name, arguments, prompt=10, completion=5):
    return ChatResult(
        message={"role": "assistant", "content": "",
                 "tool_calls": [{"function": {"name": name, "arguments": arguments}}]},
        total_duration_ms=100, load_duration_ms=20, prompt_eval_duration_ms=30,
        prompt_tokens=prompt, completion_tokens=completion,
    )


TASK = Task("some prompt", "test")


def test_happy_path_tool_then_answer():
    client = FakeClient([_call("add", {"a": 1, "b": 2}), _text("3")])
    rec = run_task(client, TASK, default_tools(inject_failure=False))
    assert rec.terminal_state == "completed"
    assert rec.steps_taken == 2
    assert [t.outcome for t in rec.tool_calls] == ["ok"]
    assert len(rec.model_calls) == 2
    assert rec.model_calls[0].load_duration_ms == 20
    assert rec.model_calls[0].prompt_eval_duration_ms == 30
    assert rec.total_tokens == 30  # 2 turns x (10 prompt + 5 completion)


def test_unknown_tool_is_agent_invalid_call():
    client = FakeClient([_call("nope", {}), _text("done")])
    rec = run_task(client, TASK, default_tools(inject_failure=False))
    assert rec.tool_calls[0].outcome == "agent_invalid_call"
    assert rec.terminal_state == "completed"


def test_missing_argument_is_agent_invalid_call():
    client = FakeClient([_call("add", {"a": 1}), _text("done")])
    rec = run_task(client, TASK, default_tools(inject_failure=False))
    assert rec.tool_calls[0].outcome == "agent_invalid_call"


def test_extra_argument_is_agent_invalid_call():
    client = FakeClient([_call("add", {"a": 1, "b": 2, "c": 3}), _text("done")])
    rec = run_task(client, TASK, default_tools(inject_failure=False))
    assert rec.tool_calls[0].outcome == "agent_invalid_call"


def test_broken_tool_is_tool_failure():
    client = FakeClient([_call("web_lookup", {"query": "x"}), _text("sorry")])
    rec = run_task(client, TASK, default_tools(inject_failure=True))
    assert rec.tool_calls[0].outcome == "tool_failure"


def test_string_arguments_are_parsed():
    client = FakeClient([_call("add", '{"a": 2, "b": 2}'), _text("4")])
    rec = run_task(client, TASK, default_tools(inject_failure=False))
    assert rec.tool_calls[0].outcome == "ok"


def test_model_error_marks_infrastructure_error():
    client = FakeClient([OllamaError("down")])
    rec = run_task(client, TASK, default_tools(inject_failure=False))
    assert rec.terminal_state == "failed"
    assert rec.infrastructure_error is True
    assert rec.model_calls[0].error is True


def test_empty_answer_is_failed_not_completed():
    client = FakeClient([_text("   ")])
    rec = run_task(client, TASK, default_tools(inject_failure=False))
    assert rec.terminal_state == "failed"


def test_max_steps_stops_a_looping_agent():
    tool = AgentTool("add", "", lambda a, b: a + b, {})
    client = FakeClient([_call("add", {"a": 1, "b": 1}) for _ in range(10)])
    rec = run_task(client, TASK, [tool], max_steps=3)
    assert rec.terminal_state == "max_steps"
    assert rec.steps_taken == 3


def test_token_budget_stops_the_task():
    client = FakeClient([_text("answer", prompt=600, completion=600)])
    rec = run_task(client, TASK, default_tools(inject_failure=False), token_budget=1000)
    assert rec.terminal_state == "budget_exceeded"


def test_parse_chat_response_converts_ns_to_ms():
    result = parse_chat_response({
        "message": {"role": "assistant", "content": "hi"},
        "total_duration": 5_000_000_000,
        "load_duration": 1_500_000_000,
        "prompt_eval_duration": 2_000_000_000,
        "prompt_eval_count": 12,
        "eval_count": 7,
    })
    assert result.total_duration_ms == 5000
    assert result.load_duration_ms == 1500
    assert result.prompt_eval_duration_ms == 2000
    assert result.prompt_tokens == 12 and result.completion_tokens == 7


def test_parse_chat_response_rejects_bad_body():
    with pytest.raises(OllamaError):
        parse_chat_response({"error": "model not found"})


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


class _Session:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def post(self, url, json, timeout):
        self.calls.append((url, json))
        if self.exc:
            raise self.exc
        return self.resp

    def get(self, url, timeout):
        self.calls.append((url, None))
        if self.exc:
            raise self.exc
        return self.resp


def test_client_unreachable_raises_ollama_error():
    sess = _Session(exc=requests.ConnectionError("refused"))
    client = OllamaClient(model="m", host="http://127.0.0.1:1", session=sess)
    with pytest.raises(OllamaError, match="cannot reach"):
        client.chat([{"role": "user", "content": "hi"}])


def test_client_http_error_raises_ollama_error():
    sess = _Session(resp=_Resp(404, {"error": "model 'm' not found"}))
    client = OllamaClient(model="m", host="http://x", session=sess)
    with pytest.raises(OllamaError, match="HTTP 404"):
        client.chat([])


def test_client_posts_non_streaming_chat_with_tools():
    body = {"message": {"role": "assistant", "content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}
    sess = _Session(resp=_Resp(200, body))
    client = OllamaClient(model="llama-x", host="http://host:11434/", session=sess)
    result = client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    url, payload = sess.calls[0]
    assert url == "http://host:11434/api/chat"
    assert payload["stream"] is False
    assert payload["model"] == "llama-x"
    assert payload["tools"] == [{"type": "function"}]
    assert result.content == "ok"


def test_model_is_required():
    with pytest.raises(ValueError):
        OllamaClient(model="")


def test_check_passes_when_model_installed_with_tag():
    sess = _Session(resp=_Resp(200, {"models": [{"name": "llama3.2:latest"}]}))
    client = OllamaClient(model="llama3.2", host="http://x", session=sess)
    assert client.check() == ["llama3.2:latest"]


def test_check_fails_when_model_missing():
    sess = _Session(resp=_Resp(200, {"models": [{"name": "qwen:7b"}]}))
    client = OllamaClient(model="llama3.2", host="http://x", session=sess)
    with pytest.raises(OllamaError, match="ollama pull llama3.2"):
        client.check()
