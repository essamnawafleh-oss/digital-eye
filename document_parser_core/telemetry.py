"""Timing and memory diagnostics. Nothing is sent anywhere; results go into the output."""

from __future__ import annotations

import sys
import time
import tracemalloc
from contextlib import contextmanager

from .schemas import StageRecord


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process since start (not per call)."""
    try:
        import resource
    except ImportError:  # Windows
        try:
            import psutil
            mem = psutil.Process().memory_info()
            return round(getattr(mem, "peak_wset", mem.rss) / 2**20, 1)
        except Exception:
            return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (2**20 if sys.platform == "darwin" else 2**10), 1)


class Telemetry:
    def __init__(self, trace_python_memory: bool = False):
        self.stages: list[StageRecord] = []
        self._t0 = time.perf_counter()
        self._trace = trace_python_memory
        if trace_python_memory and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_trace = True
        else:
            self._started_trace = False

    @contextmanager
    def stage(self, level: int, name: str, provider: str | None = None, pages: list[int] | None = None):
        rec = StageRecord(level=level, stage=name, provider=provider, pages=pages or [], outcome="ok")
        t = time.perf_counter()
        try:
            yield rec
        except Exception as exc:
            rec.outcome = "failed"
            rec.detail = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            rec.duration_ms = round((time.perf_counter() - t) * 1000, 2)
            self.stages.append(rec)

    def record(self, level: int, name: str, outcome: str, provider: str | None = None, detail: str | None = None, pages=None):
        self.stages.append(StageRecord(level=level, stage=name, provider=provider, outcome=outcome, detail=detail, pages=pages or []))

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 2)

    def python_peak_mb(self) -> float | None:
        if not self._trace or not tracemalloc.is_tracing():
            return None
        peak = tracemalloc.get_traced_memory()[1] / 2**20
        if self._started_trace:
            tracemalloc.stop()
        return round(peak, 2)
