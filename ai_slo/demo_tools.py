"""Tools for the demo agent, plus injected failures.

Injected failures (the article injected five; these are two):
  - web_lookup always fails                      -> tool_failure
  - the scratchpad serves stale reads (optional) -> memory_consistency miss
"""

from __future__ import annotations

from .record import MemoryRead
from .runner import AgentTool

_CAPITALS = {
    "france": "Paris", "japan": "Tokyo", "kenya": "Nairobi", "brazil": "Brasilia",
    "peru": "Lima", "canada": "Ottawa",
}


def add(a: float, b: float) -> float:
    return a + b


def capital_of(country: str) -> str:
    # An unknown country is a normal answer, not a tool outage, so return it
    # instead of raising (raising would be recorded as a tool_failure).
    return _CAPITALS.get(country.strip().lower(), f"no entry for {country!r}")


def broken_lookup(query: str) -> str:
    """Injected failure: the backing service is down."""
    raise ConnectionError("lookup service unavailable")


class Scratchpad:
    """Versioned key/value store used as the agent's working memory.

    Every write gets the next version for its key. A read records what version
    it observed and what the latest acknowledged version was, which is exactly
    what the memory_consistency SLO needs.

    With inject_stale=True, every other read of a key that has two or more
    versions returns the previous version (a stale cache).
    """

    def __init__(self, inject_stale: bool = False) -> None:
        self.inject_stale = inject_stale
        self._values: dict[str, list[str]] = {}
        self._reads: list[MemoryRead] = []
        self._stale_counter = 0

    def write(self, key: str, value: str) -> dict:
        versions = self._values.setdefault(key, [])
        versions.append(value)
        return {"key": key, "stored": True, "version": len(versions)}

    def read(self, key: str) -> dict:
        versions = self._values.get(key)
        if not versions:
            return {"key": key, "found": False}  # nothing to be consistent with
        latest = len(versions)
        observed = latest
        if self.inject_stale and latest >= 2:
            self._stale_counter += 1
            if self._stale_counter % 2 == 1:
                observed = latest - 1
        self._reads.append(MemoryRead(observed_version=observed, latest_acked_version=latest))
        return {"key": key, "found": True, "value": versions[observed - 1], "version": observed}

    def drain_reads(self) -> list[MemoryRead]:
        reads, self._reads = self._reads, []
        return reads


def default_tools(
    inject_failure: bool = True,
    scratchpad: Scratchpad | None = None,
) -> list[AgentTool]:
    tools = [
        AgentTool(
            "add", "Add two numbers.", add,
            {"type": "object",
             "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
             "required": ["a", "b"]},
        ),
        AgentTool(
            "capital_of", "Return the capital city of a country.", capital_of,
            {"type": "object",
             "properties": {"country": {"type": "string"}},
             "required": ["country"]},
        ),
    ]
    if scratchpad is not None:
        tools += [
            AgentTool(
                "memory_write", "Store a text value under a key in working memory.",
                scratchpad.write,
                {"type": "object",
                 "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                 "required": ["key", "value"]},
            ),
            AgentTool(
                "memory_read", "Read the value stored under a key in working memory.",
                scratchpad.read,
                {"type": "object",
                 "properties": {"key": {"type": "string"}},
                 "required": ["key"]},
            ),
        ]
    if inject_failure:
        tools.append(AgentTool(
            "web_lookup", "Search the web for a query.", broken_lookup,
            {"type": "object",
             "properties": {"query": {"type": "string"}},
             "required": ["query"]},
        ))
    return tools


def _demo_tasks() -> list[tuple[str, str]]:
    tasks: list[tuple[str, str]] = []
    tasks += [
        ("What is 17 plus 25? Use the add tool.", "arithmetic"),
        ("Add 3.5 and 4.25 with the add tool, then tell me the result.", "arithmetic"),
        ("What is 1024 plus 976? Use the add tool.", "arithmetic"),
        ("Use the add tool to compute 0.1 plus 0.2.", "arithmetic"),
        ("Add -8 and 3 with the add tool and report the sum.", "arithmetic"),
        ("What is 99 plus 1? Use the add tool.", "arithmetic"),
    ]
    tasks += [
        ("What is the capital of Japan? Use the capital_of tool.", "lookup"),
        ("What is the capital of Kenya? Use the capital_of tool.", "lookup"),
        ("What is the capital of Brazil? Use the capital_of tool.", "lookup"),
        ("What is the capital of Peru? Use the capital_of tool.", "lookup"),
        ("What is the capital of Canada? Use the capital_of tool.", "lookup"),
        ("What is the capital of Atlantis? Use the capital_of tool.", "lookup"),
        ("What is the capital of Germany? Use the capital_of tool.", "lookup"),
    ]
    tasks += [
        ("Search the web for the population of Brazil.", "research"),
        ("Find the latest news about OpenTelemetry using web_lookup.", "research"),
        ("Use web_lookup to find the tallest mountain in Africa.", "research"),
        ("Search the web for the founding year of the Medium platform.", "research"),
        ("Look up the boiling point of water with web_lookup.", "research"),
    ]
    tasks += [
        ("Save 'blue' under key 'color', then read key 'color' back and tell me it.", "memory"),
        ("Store 'Paris' under key 'city' and read it back.", "memory"),
        ("Save 'alpha' under key 'code', then save 'beta' under key 'code', then read 'code' and tell me its value.", "memory"),
        ("Save 'v1' under key 'ref', then 'v2' under key 'ref', then read 'ref' and report it.", "memory"),
        ("Save 'red' under key 'shade', then 'green' under key 'shade', then read 'shade' and tell me the value.", "memory"),
        ("Save 'one' under key 'num', then 'two' under key 'num', then 'three' under key 'num', then read 'num'.", "memory"),
        ("Read key 'missing' and tell me what you find.", "memory"),
        ("Save 'ok' under key 'status', then read 'status'.", "memory"),
    ]
    tasks += [
        ("What is the capital of France? Answer directly without tools.", "direct"),
        ("What color is the sky on a clear day? Answer directly, no tools.", "direct"),
        ("Say hello in Spanish. Answer directly, no tools.", "direct"),
        ("What is 2 times 3? Answer directly, no tools.", "direct"),
    ]
    return tasks


DEMO_TASKS: list[tuple[str, str]] = _demo_tasks()
