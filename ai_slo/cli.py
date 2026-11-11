"""Command line: python -m ai_slo report <records.jsonl> [--json]"""

from __future__ import annotations

import argparse
import json
import sys

from .calculator import SLOResult, evaluate
from .record import load_records


def _fmt_ratio(r: SLOResult) -> str:
    return "n/a" if r.ratio is None else f"{r.ratio * 100:.2f}%"


def _render_table(results: list[SLOResult]) -> str:
    header = f"{'SLO':<24}{'group':<10}{'action':<8}{'target':>8}{'sli':>10}{'good/valid':>14}  status"
    lines = [header, "-" * len(header)]
    for r in results:
        if r.meets_target is None:
            status = "no data"
        else:
            status = "ok" if r.meets_target else f"MISS ({r.action})"
        lines.append(
            f"{r.name:<24}{r.group:<10}{r.action:<8}{r.target * 100:>7.1f}%"
            f"{_fmt_ratio(r):>10}{f'{r.good}/{r.valid}':>14}  {status}"
        )
    return "\n".join(lines)


def _run_demo(args: argparse.Namespace) -> int:
    from .demo_tools import DEMO_TASKS, Scratchpad, default_tools
    from .ollama_client import OllamaClient
    from .runner import Task, run_task

    client = OllamaClient(model=args.model, host=args.host, timeout=args.timeout)
    client.check()  # fail fast: an unreachable server should not become 'failed' records
    print(f"Ollama OK at {client.host}, model {client.model!r}", file=sys.stderr)
    tasks = DEMO_TASKS[: args.tasks] if args.tasks else DEMO_TASKS
    records = []
    for i, (prompt, task_class) in enumerate(tasks, start=1):
        # Fresh working memory per task, so tasks do not see each other's keys.
        pad = Scratchpad(inject_stale=not args.no_stale)
        tools = default_tools(inject_failure=not args.no_failure, scratchpad=pad)
        print(f"[{i}/{len(tasks)}] {task_class}: {prompt}", file=sys.stderr)
        rec = run_task(client, Task(prompt, task_class), tools,
                       max_steps=args.max_steps, token_budget=args.token_budget)
        rec.memory_reads = pad.drain_reads()
        print(f"    -> {rec.terminal_state} in {rec.steps_taken} steps", file=sys.stderr)
        records.append(rec)

    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict()) + "\n")
    print(f"wrote {len(records)} records to {args.out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai_slo")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="compute SLIs from a JSONL record file")
    rep.add_argument("path", help="JSONL file with one task record per line")
    rep.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    run = sub.add_parser("run", help="run demo tasks against a local Ollama model")
    run.add_argument("--model", required=True, help="model name as shown by `ollama list`")
    run.add_argument("--host", default=None, help="Ollama URL (default: $OLLAMA_HOST or localhost:11434)")
    run.add_argument("--out", required=True, help="output JSONL path")
    run.add_argument("--tasks", type=int, default=0, help="run only the first N tasks (0 = all)")
    run.add_argument("--max-steps", type=int, default=6)
    run.add_argument("--token-budget", type=int, default=20_000)
    run.add_argument("--timeout", type=float, default=300.0)
    run.add_argument("--no-failure", action="store_true", help="do not inject the broken web_lookup tool")
    run.add_argument("--no-stale", action="store_true", help="do not inject stale working-memory reads")
    args = parser.parse_args(argv)

    if args.command == "run":
        from .ollama_client import OllamaError
        try:
            return _run_demo(args)
        except (OllamaError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    try:
        with open(args.path, encoding="utf-8") as fh:
            records = load_records(fh)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results = evaluate(records)
    if args.json:
        print(json.dumps(
            [
                {
                    "name": r.name, "group": r.group, "action": r.action,
                    "target": r.target, "good": r.good, "valid": r.valid,
                    "ratio": r.ratio, "meets_target": r.meets_target,
                }
                for r in results
            ],
            indent=2,
        ))
    else:
        print(f"records: {len(records)}")
        print(_render_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
