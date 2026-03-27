import time
from contextlib import contextmanager

class Observability:
    def __init__(self):
        self.forward_passes = 0
        self.total_forward_time = 0.0
        
        # Per-discovery-attempt tracking
        self.attempt_forward_passes = 0
        self.attempt_start_time = 0.0
        
    def start_attempt(self):
        self.attempt_forward_passes = 0
        self.attempt_start_time = time.perf_counter()
        
    def stop_attempt(self) -> float:
        return time.perf_counter() - self.attempt_start_time

    @contextmanager
    def track_forward(self):
        t0 = time.perf_counter()
        self.forward_passes += 1
        self.attempt_forward_passes += 1
        try:
            yield
        finally:
            self.total_forward_time += (time.perf_counter() - t0)

obs = Observability()
