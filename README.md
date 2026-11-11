# ai-slo

An SLO calculator for AI agents, following the approach in the Medium article
["I Built 15 SLOs for an AI Agent. Here's What OpenTelemetry Could Measure"](https://medium.com/data-science-collective/i-built-15-slos-for-an-ai-agent-heres-what-opentelemetry-could-measure-dcbacf4c3e9e) (Tanmay Deshpande).

The plan and the reasoning behind each decision are in [PLAN.md](PLAN.md).

## Approach

- Every SLI is a ratio: `good events / valid events`.
- SLOs are computed from **one JSON record per task**, written by the agent at the end of the task, not from the trace backend.
- Indicators are grouped as in Google SRE: **requests** (tool and model calls, page), **pipeline** (whole task, ticket), and **storage** (working memory, page).
- Standard OpenTelemetry GenAI names are used where they exist. Custom names are defined in `ai_slo/attributes.py`.

## Usage

Python 3.10+, with `requests`. Tests need `pytest`.

```bash
python3 -m ai_slo report examples/sample_records.jsonl
python3 -m ai_slo report examples/sample_records.jsonl --json
python3 -m pytest -q
```

### Running the demo agent on a local Ollama model

The `run` command runs the demo tasks through a hand-written agent loop against a
local Ollama server and writes one record per task. It checks first that the
server is up and the model is installed, and stops with an error if not.

```bash
ollama serve                          # if it is not already running
ollama pull <model>                   # e.g. a small model you have locally
python3 -m ai_slo run --model <model> --out examples/ollama_records.jsonl
python3 -m ai_slo report examples/ollama_records.jsonl
```

Options: `--host` (default `$OLLAMA_HOST` or `http://localhost:11434`), `--tasks N`,
`--max-steps`, `--token-budget`, `--no-failure` (leave out the broken `web_lookup`
tool that injects a tool failure).

The demo runs the article's 30-task count across five groups: arithmetic, lookup,
research, memory, and direct answers. Its tools are local Python functions, not MCP
servers as in the article, and it injects two failures (the article injected five):

- `web_lookup` always fails, so the model's tool calls show up as `tool_failure`.
- The working-memory store serves a stale version on every other read of a key that
  has been overwritten. This produces the `memory_consistency` misses.

Both failures are injected on purpose, so the misses they cause are expected. Use
`--no-failure` and `--no-stale` to turn them off. Last run: qwen2.5:7b completed
30/30 tasks. The output is in `examples/ollama_records.jsonl`, and the run log is in
`examples/ollama_run.log`.

### Record format

One JSON object per line. See `ai_slo/record.py` for the full schema. Example:

```json
{"task_id": "t1", "task_class": "lookup", "duration_ms": 4200,
 "terminal_state": "completed", "steps_taken": 3, "total_tokens": 1800,
 "tool_calls": [{"name": "search", "duration_ms": 300, "outcome": "ok"}],
 "model_calls": [{"duration_ms": 900, "load_duration_ms": 200, "prompt_eval_duration_ms": 400}],
 "memory_reads": [{"observed_version": 4, "latest_acked_version": 4}]}
```

- `tool_calls[].outcome`: `ok`, `tool_failure` (the tool server failed: a dependency incident), or `agent_invalid_call` (the agent sent bad arguments: a ticket).
- `model_calls[].load_duration_ms` and `prompt_eval_duration_ms`: Ollama fields. Their sum approximates time to first token.
- `memory_reads`: the version observed and the latest acknowledged version at the time of the read.

## Indicators implemented

Implemented in `ai_slo/slos.py`. Targets and thresholds are **placeholders** of mine; the article gives no numbers.

| SLO | Group | Action | Target | Good event |
|---|---|---|---|---|
| invocation_availability | requests | page | 99.0% | task not ended by an infrastructure error |
| invocation_latency | requests | page | 95.0% | task ≤ 60 s |
| tool_availability | requests | page | 99.0% | tool call did not fail on the tool server |
| tool_call_validity | requests | page | 99.0% | tool call not rejected for agent-side arguments |
| tool_latency | requests | page | 95.0% | tool call ≤ 2 s |
| model_availability | requests | page | 99.0% | model call completed without error |
| model_latency | requests | page | 95.0% | load + prompt eval ≤ 5 s |
| task_completion | pipeline | ticket | 95.0% | terminal state `completed` |
| autonomous_completion | pipeline | ticket | 90.0% | completed without human intervention |
| token_budget | pipeline | ticket | 95.0% | total tokens ≤ 20,000 |
| memory_consistency | storage | page | 99.9% | read saw the latest acknowledged version |

## What is from the article and what is my choice

From the article:
- the ratio definition, the Google SRE grouping, and the page/ticket split;
- the attributes the article says OpenTelemetry does not provide (`agent.terminal_state`, `agent.steps_taken`, task class, human intervention, `agent.tool_rejection_reason`, memory version);
- the Ollama load + prompt-eval approximation for time to first token;
- the rule of computing SLOs from a stable record instead of the trace backend.

My choices (not in the article):
- all targets and thresholds;
- the eight terminal states (the article says there are eight but does not list them; I guessed from its examples: `completed`, `failed`, `timeout`, `cancelled`, `refused`, `handed_to_human`, `budget_exceeded`, `max_steps`);
- the record schema and the JSONL format;
- memory consistency defined as `observed_version >= latest_acked_version`;
- the sample data in `examples/`.

## Open gaps

- **Only 11 of the 15 indicators are implemented.** The text of the article I could read did not list all 15, so I did not invent the other four. Add them to `ai_slo/slos.py` once you have the list.
- **Task correctness is not measured.** The article gives no way to measure it; it needs an external label.
- **Tool-failure attribution** depends on the tool server returning `tool_rejection_reason`-style information. The article says this may need changes to the tools, and this project only reads the outcome.
- **The OpenTelemetry instrumentation** that writes these records is not included. This project defines the names and the record, not the agent.
- The article notes that GenAI semantic conventions are still changing (e.g. `gen_ai.task.status`). Revisit `attributes.py` when they settle.
