# Plan: AI agent SLO calculator

Source: "I Built 15 SLOs for an AI Agent. Here's What OpenTelemetry Could Measure" (Tanmay Deshpande, Medium, Data Science Collective). Read via the browser on 2026-11-11.

## What the article says (and what it does not)

- Each SLI is a ratio: `good events / valid events`.
- Indicators are grouped the way Google SRE groups them:
  - **Requests** (tool calls, model calls): failures can page someone.
  - **Pipeline** (whole agent task): failures are better as tickets.
  - **Storage** (working memory): failures page, because stale reads give wrong answers that look normal.
- OpenTelemetry already gives trace IDs, span status, timestamps, and latency. The article calls invocation availability and invocation latency "basically free" and tool/model availability and latency "partly free".
- Everything that describes a whole task needs custom attributes. Named in the article:
  - `agent.terminal_state` (8 possible values in the article, not listed)
  - `agent.steps_taken`
  - task class
  - human intervention
  - memory version information (for freshness)
  - `agent.tool_rejection_reason` (separates tool failure from agent error)
  - total token count on the root task
- Model latency: time to first token can be approximated as `load_duration + prompt_eval_duration` (Ollama), with no streaming needed.
- Design rule: compute SLOs from one structured record per task, written by the application, not from the trace backend.
- Standard names to use where they exist: `gen_ai.operation.name`, `gen_ai.usage.input_tokens`, `gen_ai.tool.name`, `gen_ai.memory.*`.

**Gap:** The article says there are 15 indicators, but the text I could read does not list all 15. I will not invent the missing ones. This plan implements the indicators the article describes explicitly and marks the rest as open.

## Scope

Python 3.10+, standard library only, with pytest for tests. No OpenTelemetry SDK dependency: this project works on the task record, which the agent writes at the end of each task. Span and attribute names are defined as constants so the agent can set them on spans later.

### Files

```
ai_slo/
  __init__.py
  attributes.py   # OTel attribute and span-name constants from the article
  record.py       # TaskRecord dataclasses + JSON load/dump (one JSON object per line)
  slos.py         # SLO definitions: name, group, SRE category, event filter, target
  calculator.py   # computes good/valid/ratio per SLO and checks targets
  cli.py          # python -m ai_slo report <records.jsonl>
tests/
  test_record.py
  test_calculator.py
examples/
  sample_records.jsonl   # hand-written records, covering each failure mode
README.md
PLAN.md
```

### Indicators implemented

| SLO | Group (SRE) | Good / valid | Source in the article |
|---|---|---|---|
| invocation_availability | Requests | valid = every invocation; good = invocation not ended by an infrastructure error | basically free |
| invocation_latency | Requests | good = duration ≤ threshold | basically free |
| tool_availability | Requests | valid = tool calls; good = not `tool_failure` | rejection-reason split |
| tool_call_validity | Requests | valid = tool calls; good = not rejected for agent-side invalid args | rejection-reason split |
| tool_latency | Requests | good = tool span duration ≤ threshold | partly free |
| model_availability | Requests | valid = model calls; good = no model error | partly free |
| model_latency | Requests | good = `load_duration + prompt_eval_duration` ≤ threshold | Ollama approximation |
| task_completion | Pipeline (ticket) | valid = tasks; good = `terminal_state == completed` | terminal state |
| autonomous_completion | Pipeline (ticket) | valid = tasks; good = completed without human intervention | human intervention |
| token_budget | Pipeline (ticket) | valid = tasks; good = total tokens ≤ budget | root-span token total |
| memory_consistency | Storage (page) | valid = memory reads; good = read version equals latest acknowledged version | memory version info |

Plus: `task_correctness` is left out because the article gives no way to measure correctness. It needs an external label, which will be a future addition.

### Decisions I am making (not in the article)

- **Targets:** The article gives no numbers. Defaults are placeholders: 99.0% for requests and storage, 95.0% for pipeline. Latency thresholds: tool 2 s, model 5 s, invocation 60 s. All are set in `slos.py` and easy to change.
- **Record format:** One JSON object per line. Fields are listed in `record.py`. The record has the root-level total tokens, the terminal state, and per-call entries for tools, models, and memory.
- **Terminal states:** The article says there are 8 but does not list them. I'll define the 8 that match its examples (`completed`, `failed`, `timeout`, `cancelled`, `refused`, `handed_to_human`, plus `budget_exceeded` and `max_steps`). Mark them as my guess in the README.
- **Memory consistency:** A read is good when its version equals the latest acknowledged write version seen before it in the same task.
- **Tests:** Run on the sample records, with one assertion per SLO and checks for empty input (no valid events → ratio is `None`, not a crash).

### Steps

1. Write this plan (done).
2. `attributes.py`, `record.py`, `slos.py`, `calculator.py`, `cli.py`.
3. `examples/sample_records.jsonl` covering each failure mode.
4. Tests, run with pytest.
5. Run the CLI on the sample file and check the output.
6. README: indicator table, what is my choice vs. from the article, and open gaps (the missing indicators, the 8 terminal states, targets).

### Open questions for the user

- The full list of 15 indicators and the 8 terminal states are not in the text I read. If you can paste the rest of the article or the author's list, I'll update `slos.py`.
- Are the placeholder targets acceptable, or do you have real ones?
