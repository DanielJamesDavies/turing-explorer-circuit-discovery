import time
from contextlib import contextmanager
from typing import Optional


class TimerResult:
    """Holds the elapsed time after a Timer context exits."""
    __slots__ = ("elapsed_s",)

    def __init__(self):
        self.elapsed_s: float = 0.0

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000.0

    def __repr__(self) -> str:
        return f"TimerResult({self.elapsed_ms:.1f} ms)"


@contextmanager
def timer(label: Optional[str] = None, into: Optional[dict] = None):
    """
    Context manager for timing a block of code.

    Usage:
        with timer() as t:
            do_work()
        print(t.elapsed_ms)

        # Or accumulate into a dict:
        timing = {}
        with timer("phase_1", into=timing):
            do_work()
        # timing == {"phase_1": <elapsed_seconds>}
    """
    result = TimerResult()
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_s = time.perf_counter() - t0
        if into is not None and label is not None:
            into[label] = result.elapsed_s
