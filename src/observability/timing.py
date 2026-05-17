import time
from contextlib import contextmanager
from typing import Optional

import torch


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


def format_duration(seconds: float) -> str:
    """Return a compact human-readable duration."""
    if seconds < 1.0:
        return f"{seconds * 1000.0:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes, rem = divmod(seconds, 60.0)
    return f"{int(minutes)}m {rem:.1f}s"


def _cpu_rss_mb() -> Optional[float]:
    """Best-effort current process RSS in MB without requiring psutil."""
    try:
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / 1024**2
    except Exception:
        pass

    try:
        import os
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes.
        if os.name == "posix" and rss > 10**7:
            return rss / 1024**2
        return rss / 1024
    except Exception:
        return None


def resource_snapshot() -> str:
    """Return a compact CPU/GPU memory snapshot for logs."""
    parts: list[str] = []
    rss_mb = _cpu_rss_mb()
    if rss_mb is not None:
        parts.append(f"cpu_rss={rss_mb:.0f} MB")

    if torch.cuda.is_available():
        try:
            dev = torch.cuda.current_device()
            alloc = torch.cuda.memory_allocated(dev) / 1024**3
            reserved = torch.cuda.memory_reserved(dev) / 1024**3
            peak_alloc = torch.cuda.max_memory_allocated(dev) / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved(dev) / 1024**3
            parts.append(
                "cuda:{dev} alloc={alloc:.2f} GB reserved={reserved:.2f} GB "
                "peak_alloc={peak_alloc:.2f} GB peak_reserved={peak_reserved:.2f} GB".format(
                    dev=dev,
                    alloc=alloc,
                    reserved=reserved,
                    peak_alloc=peak_alloc,
                    peak_reserved=peak_reserved,
                )
            )
        except Exception:
            pass

    return " | ".join(parts) if parts else "resources=unavailable"


@contextmanager
def phase_timer(label: str):
    """
    Log wall time and best-effort resource usage around a pipeline phase.

    CUDA peak counters are reset at phase start so the reported peak covers the
    current phase, not the entire process lifetime.
    """
    print(f"[timing] START {label} | {resource_snapshot()}")
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        elapsed = time.perf_counter() - t0
        print(f"[timing] END   {label} | {format_duration(elapsed)} | {resource_snapshot()}")
