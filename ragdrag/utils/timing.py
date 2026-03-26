"""Latency measurement utilities for RAG fingerprinting.

Used by R1 techniques (RD-0101) to detect RAG presence through
response timing analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TimingResult:
    """Result of a timed HTTP request."""

    url: str
    query: str
    elapsed_ms: float
    status_code: int
    response_text: str = ""
    response_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class TimingStats:
    """Aggregate timing statistics from multiple requests."""

    results: list[TimingResult] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.results)

    @property
    def mean_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.elapsed_ms for r in self.results) / len(self.results)

    @property
    def min_ms(self) -> float:
        if not self.results:
            return 0.0
        return min(r.elapsed_ms for r in self.results)

    @property
    def max_ms(self) -> float:
        if not self.results:
            return 0.0
        return max(r.elapsed_ms for r in self.results)

    @property
    def delta_ms(self) -> float:
        return self.max_ms - self.min_ms

    def add(self, result: TimingResult) -> None:
        self.results.append(result)

    def to_dict(self, include_responses: bool = False) -> dict:
        d: dict = {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "delta_ms": round(self.delta_ms, 2),
        }
        if include_responses:
            d["queries"] = [
                {
                    "query": r.query,
                    "elapsed_ms": round(r.elapsed_ms, 2),
                    "status_code": r.status_code,
                    "response_text": r.response_text[:2000],
                    "response_headers": r.response_headers,
                }
                for r in self.results
            ]
        return d


def measure_elapsed(start: float) -> float:
    """Return elapsed milliseconds since start."""
    return (time.monotonic() - start) * 1000
