"""Demo tools: task list shape and the versioned scratchpad."""

from ai_slo.demo_tools import DEMO_TASKS, Scratchpad, default_tools


def test_demo_has_the_articles_30_tasks():
    assert len(DEMO_TASKS) == 30


def test_demo_task_classes_cover_each_group():
    classes = {c for _, c in DEMO_TASKS}
    assert classes == {"arithmetic", "lookup", "research", "memory", "direct"}


def test_scratchpad_records_consistent_reads():
    pad = Scratchpad()
    pad.write("k", "a")
    pad.write("k", "b")
    assert pad.read("k")["value"] == "b"
    reads = pad.drain_reads()
    assert len(reads) == 1
    assert reads[0].observed_version == reads[0].latest_acked_version == 2


def test_scratchpad_missing_key_records_no_read():
    pad = Scratchpad()
    assert pad.read("nope") == {"key": "nope", "found": False}
    assert pad.drain_reads() == []


def test_scratchpad_injects_stale_reads_on_alternate_reads():
    pad = Scratchpad(inject_stale=True)
    pad.write("k", "old")
    pad.write("k", "new")
    first = pad.read("k")    # stale: returns the previous version
    second = pad.read("k")   # fresh
    assert first["value"] == "old" and first["version"] == 1
    assert second["value"] == "new" and second["version"] == 2
    reads = pad.drain_reads()
    assert [r.observed_version >= r.latest_acked_version for r in reads] == [False, True]


def test_scratchpad_drain_resets_reads():
    pad = Scratchpad()
    pad.write("k", "v")
    pad.read("k")
    pad.drain_reads()
    assert pad.drain_reads() == []


def test_memory_tools_only_added_with_scratchpad():
    names = {t.name for t in default_tools(inject_failure=False)}
    assert "memory_read" not in names
    names = {t.name for t in default_tools(inject_failure=False, scratchpad=Scratchpad())}
    assert {"memory_read", "memory_write"} <= names
