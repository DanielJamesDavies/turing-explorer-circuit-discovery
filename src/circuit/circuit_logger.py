import os
from datetime import datetime


class CircuitLogger:
    """
    Per-seed, per-method text logger for circuit discovery.

    Records node/edge counts at each named stage, final evaluation scores,
    and the reason a circuit was accepted or rejected.  The log file is
    always written to disk inside a ``finally`` block, so every attempt
    leaves a trace regardless of outcome or exception.

    Output directory: ``outputs/discovery_logs/``
    Filename pattern: ``<method>__comp<N>_lat<M>.txt``

    Typical usage inside a ``discover()`` method::

        logger = CircuitLogger(seed_comp_idx, seed_latent_idx, "logit_attribution")
        try:
            ...
            logger.stage("logit attribution", len(included), 0,
                         note=f"{n_considered} candidates, threshold={self.logit_threshold}")
            ...
            logger.eval(final_f, final_s, final_c)
            if final_f < self.min_faithfulness:
                logger.reject(f"faithfulness {final_f:.4f} < {self.min_faithfulness}")
                return None
            logger.accept(len(circuit.nodes), len(circuit.edges))
            return circuit
        finally:
            logger.save()
    """

    _LOG_DIR = "outputs/discovery_logs"

    def __init__(self, seed_comp: int, seed_latent: int, method_name: str) -> None:
        os.makedirs(self._LOG_DIR, exist_ok=True)
        filename = f"{method_name}__comp{seed_comp}_lat{seed_latent}.txt"
        self.path = os.path.join(self._LOG_DIR, filename)
        self._lines: list[str] = []
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
        """Log node/edge counts at a named stage of the algorithm."""
        tail = f"  ({note})" if note else ""
        self._w(f"  [{label}]  nodes={n_nodes}  edges={n_edges}{tail}")

    def note(self, text: str) -> None:
        """Append a free-form informational line."""
        self._w(f"  {text}")

    def eval(
        self,
        faithfulness: float,
        sufficiency: float,
        completeness: float,
    ) -> None:
        """Log the three evaluation scores."""
        self._w("")
        self._w(
            f"  EVAL  faithfulness={faithfulness:.4f}"
            f"  sufficiency={sufficiency:.4f}"
            f"  completeness={completeness:.4f}"
        )

    def reject(self, reason: str) -> None:
        """Mark this attempt as rejected and record the reason."""
        self._w(f"  REJECTED — {reason}")

    def accept(self, n_nodes: int, n_edges: int) -> None:
        """Mark this attempt as accepted."""
        self._w(f"  ACCEPTED  nodes={n_nodes}  edges={n_edges}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Flush all buffered lines to the log file. Always call from finally."""
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("\n".join(self._lines) + "\n")
        except Exception:
            pass  # never let logging crash the pipeline
