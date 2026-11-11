"""Compute SLI ratios and compare them to targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .record import TaskRecord
from .slos import SLO, SLOS, Thresholds


@dataclass(frozen=True)
class SLOResult:
    name: str
    group: str
    action: str
    target: float
    good: int
    valid: int

    @property
    def ratio(self) -> Optional[float]:
        """None when there were no valid events (nothing to measure)."""
        return self.good / self.valid if self.valid else None

    @property
    def meets_target(self) -> Optional[bool]:
        return None if self.ratio is None else self.ratio >= self.target


def evaluate(
    records: Iterable[TaskRecord],
    slos: Sequence[SLO] = SLOS,
    thresholds: Thresholds = Thresholds(),
) -> list[SLOResult]:
    records = list(records)
    results = []
    for slo in slos:
        good = valid = 0
        for record in records:
            events = slo.extract(record, thresholds)
            valid += len(events)
            good += sum(1 for e in events if e)
        results.append(
            SLOResult(slo.name, slo.group, slo.action, slo.target, good, valid)
        )
    return results
