"""Per-seed PHASE TIMERS for discovery (2026-09-05, H100 launch prep).

    from observability.phases import phase, reset_phases, snapshot_phases
    with phase("seed.fit"): ...

Accumulates wall-clock per named phase in a process-global table that the
discovery window resets per (seed, method) and snapshots into that seed's
task-metrics row as {"phases": {name: {"s": total_seconds, "n": calls}}}.
So the production run's own metrics files answer "where did the time go",
per seed and therefore per depth — no separate profiling run needed.

Accuracy note: CUDA is asynchronous. Coarse phases (probes / fit / eval /
analyses) end on host reads and are accurate as-is. The step-level fit
phases (fit.fwd / fit.bwd / fit.opt) are launch-time only unless
PHASE_SYNC=1, which synchronises at every phase boundary — use that on a
profiling shard, not the production run (each sync idles the GPU briefly
while the host queues the next kernels).
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Dict, Iterator

import torch

_SYNC = os.environ.get("PHASE_SYNC") == "1"
_ACC: Dict[str, list] = {}


def _maybe_sync() -> None:
    if _SYNC and torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def phase(name: str) -> Iterator[None]:
    _maybe_sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _maybe_sync()
        e = _ACC.setdefault(name, [0.0, 0])
        e[0] += time.perf_counter() - t0
        e[1] += 1


def reset_phases() -> None:
    _ACC.clear()


def snapshot_phases() -> Dict[str, Dict[str, float]]:
    return {k: {"s": round(v[0], 4), "n": v[1]} for k, v in _ACC.items()}
