import os
import time
from datetime import datetime
from utils.observability import obs


class CircuitLogger:
    """
    Per-seed, per-method text logger for circuit discovery.

    Records node/edge counts at each named stage, final evaluation scores,
    and the reason a circuit was accepted or rejected.  The log file is
    always written to disk inside a ``finally`` block, so every attempt
    leaves a trace regardless of outcome or exception.

    Output directory: ``outputs/discovery_logs/``
    Filename pattern: ``<method>__comp<N>_lat<M>.txt``
    """

    _LOG_DIR = "outputs/discovery_logs"

    def __init__(self, seed_comp: int, seed_latent: int, method_name: str) -> None:
        os.makedirs(self._LOG_DIR, exist_ok=True)
        filename = f"{method_name}__comp{seed_comp}_lat{seed_latent}.txt"
        self.path = os.path.join(self._LOG_DIR, filename)
        self._lines: list[str] = []
        self._start_time = time.perf_counter()
        self._last_stage_time = self._start_time
        self._enabled = True
        obs.start_attempt()
        
        self._w(f"=== {method_name}  |  comp={seed_comp}  lat={seed_latent} ===")
        self._w(f"    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._w("")

    # ------------------------------------------------------------------
    # Low-level writer
    # ------------------------------------------------------------------

    def _w(self, line: str) -> None:
        self._lines.append(line)

    # ------------------------------------------------------------------
    # Structured log helpers
    # ------------------------------------------------------------------

    def header(
        self,
        seed_layer: int,
        seed_kind: str,
        seed_latent: int,
        n_pos: int,
        n_neg: int,
    ) -> None:
        """Log seed identity and probe dataset sizes."""
        self._w(f"Seed   layer={seed_layer}  kind={seed_kind}  latent={seed_latent}")
        self._w(f"Probe  n_pos={n_pos}  n_neg={n_neg}")
        self._w("")

    def stage(
        self,
        label: str,
        n_nodes: int,
        n_edges: int,
        note: str = "",
    ) -> None:
        """Log node/edge counts at a named stage of the algorithm, with timing."""
        now = time.perf_counter()
        stage_dt = (now - self._last_stage_time) * 1000
        self._last_stage_time = now
        
        tail = f"  ({note})" if note else ""
        self._w(f"  [{label:<25}]  nodes={n_nodes:<4}  edges={n_edges:<4}  | {stage_dt:>8.1f} ms {tail}")

    def note(self, text: str) -> None:
        """Append a free-form informational line."""
        self._w(f"  {text}")

    def eval(
        self,
        faithfulness: float,
        sufficiency: float,
        completeness: float,
    ) -> None:
        """Log the three evaluation scores with timing."""
        now = time.perf_counter()
        stage_dt = (now - self._last_stage_time) * 1000
        self._last_stage_time = now

        self._w("")
        self._w(
            f"  EVAL  faithfulness={faithfulness:.4f}"
            f"  sufficiency={sufficiency:.4f}"
            f"  completeness={completeness:.4f}"
            f"  | {stage_dt:>8.1f} ms"
        )

    def reject(self, reason: str) -> None:
        """Mark this attempt as rejected and record the reason."""
        self._w(f"  REJECTED — {reason}")

    def accept(self, n_nodes: int, n_edges: int) -> None:
        """Mark this attempt as accepted."""
        self._w(f"  ACCEPTED  nodes={n_nodes}  edges={n_edges}")

    def cancel(self) -> None:
        """Prevents the log from being saved to disk. Use for trivial rejections."""
        self._enabled = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Flush all buffered lines to the log file. Always call from finally."""
        total_dt = obs.stop_attempt()
        if not self._enabled:
            return

        self._w("")
        self._w(f"Total time: {total_dt:.2f} s")
        self._w(f"Total forward passes: {obs.attempt_forward_passes}")
        if obs.attempt_forward_passes > 0:
            self._w(f"Average forward time: {total_dt / obs.attempt_forward_passes * 1000:.1f} ms")
        
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("\n".join(self._lines) + "\n")
        except Exception:
            pass  # never let logging crash the pipeline
