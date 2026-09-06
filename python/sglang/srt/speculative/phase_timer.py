# Lightweight speculative-decoding phase profiler (diagnostic).
#
# Enabled with SGL_PHASE_TIMING=1. Records CUDA-event GPU durations and CPU
# wall durations around the coarse spec-decode phases (draft / verify target
# forward / verify post-processing / draft_extend) without any per-phase
# synchronization. Aggregates are logged every SGL_PHASE_TIMING_EVERY steps
# (default 100) after a single torch.cuda.synchronize().
#
# NOTE: diagnostic-only; zero overhead when SGL_PHASE_TIMING != 1.

import logging
import os
import time
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)

def _env(name: str, default: str) -> str:
    return os.environ.get(f"SGLANG_{name}", os.environ.get(f"SGL_{name}", default))


_ENABLED = _env("PHASE_TIMING", "0") == "1"
_LOG_EVERY = int(_env("PHASE_TIMING_EVERY", "100"))


class _Span:
    __slots__ = ("timer", "name", "start_ev", "end_ev", "cpu_start")

    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        if not self.timer.enabled:
            return self
        self.start_ev = torch.cuda.Event(enable_timing=True)
        self.end_ev = torch.cuda.Event(enable_timing=True)
        self.start_ev.record()
        self.cpu_start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if not self.timer.enabled:
            return False
        self.end_ev.record()
        self.timer._spans.append(
            (self.name, self.start_ev, self.end_ev, self.cpu_start, time.perf_counter())
        )
        return False


class PhaseTimer:
    def __init__(self):
        self.enabled = _ENABLED
        self._spans = []
        self._count = 0
        self._gpu_ms = defaultdict(float)
        self._cpu_ms = defaultdict(float)
        self._hits = defaultdict(int)

    def span(self, name):
        return _Span(self, name)

    # JOURNAL 12.113: cpu-only marks for segments with no event objects
    # (scheduler-loop / result-processing). Timestamp-only, same overhead
    # class as the existing spans. Printed by flush alongside span stats.
    def mark_begin(self):
        if not self.enabled:
            return None
        return time.perf_counter()

    def mark_end(self, name, t0):
        if not self.enabled or t0 is None:
            return
        self._cpu_ms[name] += (time.perf_counter() - t0) * 1000.0
        self._hits[name] += 1

    def step_done(self):
        if not self.enabled:
            return
        self._count += 1
        if self._count % _LOG_EVERY == 0:
            self.flush()

    def flush(self):
        if not self._spans and not self._cpu_ms:
            self._count = 0
            return
        torch.cuda.synchronize()
        for name, s, e, c0, c1 in self._spans:
            self._gpu_ms[name] += s.elapsed_time(e)
            self._cpu_ms[name] += (c1 - c0) * 1000.0
            self._hits[name] += 1
        self._spans.clear()
        n = max(1, self._count)
        parts = [
            f"{name}: gpu {self._gpu_ms[name] / max(1, self._hits[name]):.2f} ms"
            f" (cpu {self._cpu_ms[name] / max(1, self._hits[name]):.2f} ms, n={self._hits[name]})"
            for name in sorted(set(self._gpu_ms) | set(self._cpu_ms))
        ]
        logger.info("PHASE-TIMING over %d steps | %s", n, " | ".join(parts))
        self._gpu_ms.clear()
        self._cpu_ms.clear()
        self._hits.clear()
        self._count = 0


_phase_timer = PhaseTimer()


def get_phase_timer():
    return _phase_timer
