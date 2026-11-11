import json
from pathlib import Path

import pytest

from ai_slo import (
    MemoryRead, ModelCall, SLOS, TaskRecord, ToolCall, Thresholds,
    evaluate, load_records,
)
from ai_slo.cli import main

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_records.jsonl"


def _by_name(results):
    return {r.name: r for r in results}


def _task(**kw):
    base = dict(task_id="x", task_class="c", duration_ms=1000, terminal_state="completed")
    base.update(kw)
    return TaskRecord(**base)


# Expected values below were worked out by hand from examples/sample_records.jsonl.
EXPECTED = {
    "invocation_availability": (4, 5),
    "invocation_latency": (5, 5),
    "tool_availability": (5, 6),
    "tool_call_validity": (5, 6),
    "tool_latency": (4, 6),
    "model_availability": (4, 5),
    "model_latency": (3, 4),
    "task_completion": (2, 5),
    "autonomous_completion": (1, 5),
    "token_budget": (4, 5),
    "memory_consistency": (3, 4),
}


def test_sample_file_matches_hand_computed_counts():
    records = load_records(SAMPLE.read_text(encoding="utf-8").splitlines())
    results = _by_name(evaluate(records))
    assert set(results) == set(EXPECTED)
    for name, (good, valid) in EXPECTED.items():
        assert (results[name].good, results[name].valid) == (good, valid), name


def test_every_slo_has_a_group_and_target():
    groups = {s.group for s in SLOS}
    assert groups == {"requests", "pipeline", "storage"}
    assert all(0 < s.target <= 1 for s in SLOS)


def test_ratio_is_none_without_valid_events():
    results = _by_name(evaluate([_task()]))
    # No tool or model calls, no memory reads: nothing to measure.
    assert results["tool_latency"].ratio is None
    assert results["tool_latency"].meets_target is None
    assert results["memory_consistency"].ratio is None


def test_empty_input_does_not_crash():
    results = evaluate([])
    assert all(r.valid == 0 and r.ratio is None for r in results)


def test_tool_failure_and_agent_error_are_separated():
    failing_tool = _task(tool_calls=[ToolCall("a", 10, "tool_failure")])
    bad_args = _task(tool_calls=[ToolCall("a", 10, "agent_invalid_call")])
    r1 = _by_name(evaluate([failing_tool]))
    r2 = _by_name(evaluate([bad_args]))
    assert r1["tool_availability"].good == 0
    assert r1["tool_call_validity"].good == 1
    assert r2["tool_availability"].good == 1
    assert r2["tool_call_validity"].good == 0


def test_model_latency_uses_load_plus_prompt_eval():
    rec = _task(model_calls=[ModelCall(100, load_duration_ms=3000, prompt_eval_duration_ms=2500)])
    res = _by_name(evaluate([rec], thresholds=Thresholds(model_latency_ms=5000)))["model_latency"]
    assert (res.good, res.valid) == (0, 1)  # 5500 ms > 5000 ms


def test_memory_consistency_requires_latest_version():
    stale = _task(memory_reads=[MemoryRead(observed_version=1, latest_acked_version=2)])
    fresh = _task(memory_reads=[MemoryRead(observed_version=2, latest_acked_version=2)])
    assert _by_name(evaluate([stale]))["memory_consistency"].good == 0
    assert _by_name(evaluate([fresh]))["memory_consistency"].good == 1


def test_token_budget_threshold_is_configurable():
    rec = _task(total_tokens=5000)
    tight = _by_name(evaluate([rec], thresholds=Thresholds(token_budget=1000)))
    loose = _by_name(evaluate([rec], thresholds=Thresholds(token_budget=10000)))
    assert tight["token_budget"].good == 0
    assert loose["token_budget"].good == 1


def test_unknown_terminal_state_rejected():
    with pytest.raises(ValueError):
        _task(terminal_state="done-ish")


def test_unknown_tool_outcome_rejected():
    with pytest.raises(ValueError):
        ToolCall("a", 1, "exploded")


def test_load_records_reports_line_number():
    with pytest.raises(ValueError, match="line 2"):
        load_records(['{"task_id": "a", "task_class": "c", "duration_ms": 1, "terminal_state": "completed"}', "not json"])


def test_load_records_skips_blank_lines():
    recs = load_records(["", '{"task_id": "a", "task_class": "c", "duration_ms": 1, "terminal_state": "completed"}', "  "])
    assert len(recs) == 1


def test_cli_table_runs(capsys):
    assert main(["report", str(SAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "records: 5" in out
    assert "task_completion" in out


def test_cli_json_output(capsys):
    assert main(["report", str(SAMPLE), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    completion = next(d for d in data if d["name"] == "task_completion")
    assert completion["good"] == 2 and completion["valid"] == 5


def test_cli_bad_file_returns_error(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{nope", encoding="utf-8")
    assert main(["report", str(bad)]) == 2
    assert "error" in capsys.readouterr().err
