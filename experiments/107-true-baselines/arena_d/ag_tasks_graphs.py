"""Arena D pod stage (plan D8): ag_tasks.py run, UNCHANGED, plus every attribution graph saved next to its per-prompt JSON.

Why: the D1 scoring that has no runner yet (README "D1 scoring extension") needs the full graphs after the pod: their
completeness / replacement on graphs pruned to a WCM-PLT circuit's node set, and our tests on AG node sets at other
thresholds. Saving them now means nothing is regenerated after the pod.

How (no edit to ag_tasks.py or circuit-tracer): ag_tasks.one_prompt imports `circuit_tracer.attribute` at call time; this
wrapper swaps that module attribute, for the duration of each one_prompt call, for one that calls the real attribute()
with the same arguments and saves the returned Graph (Graph.to_pt, their format) to
  <out>/<task>/<split>/p<id>.graph.pt
before ag_tasks prunes and scores it. The adjacency matrix is stored as float16 (AG_GRAPH_DTYPE=float16, default; ~0.1 GB
per 7-token prompt instead of ~0.2 GB; kept float32 when a value exceeds the float16 range) and is restored to its
original dtype before ag_tasks uses the graph, so ag_tasks' outputs are those of a plain run. AG_GRAPH_DTYPE=float32
keeps it exact. One task per process (the pod stage runs one per task).

  ct107: ~/venvs/ct107/bin/python -B -I ag_tasks_graphs.py run --tasks rc --split held ... (ag_tasks.py's arguments)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ag_tasks as AT  # noqa: E402

GRAPH_DTYPE = os.environ.get("AG_GRAPH_DTYPE", "float16")


def save_graph(g, path: Path) -> None:
    import torch
    adj = g.adjacency_matrix
    if GRAPH_DTYPE == "float16" and bool(torch.isfinite(adj).all()) and float(adj.abs().max()) < 6.0e4:
        g.adjacency_matrix = adj.detach().to("cpu", torch.float16)
    else:
        g.adjacency_matrix = adj.detach().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        g.to_pt(str(tmp))
        os.replace(tmp, path)
    finally:
        g.adjacency_matrix = adj


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = AT.parse_args(argv)
    if a.cmd != "run":
        return AT.main(argv)
    tasks = ["tiny"] if a.fixture else [t for t in a.tasks.split(",") if t]
    if len(tasks) != 1:
        raise SystemExit("ag_tasks_graphs.py: one task per process (got %s)" % tasks)
    task = tasks[0]
    orig = AT.one_prompt

    def one_prompt(torch, model, ids, row, args, dev):
        import circuit_tracer
        real = circuit_tracer.attribute
        path = Path(args.out) / task / args.split / ("p%04d.graph.pt" % int(row["id"]))

        def attribute(*pa, **kw):
            g = real(*pa, **kw)
            save_graph(g, path)
            return g

        circuit_tracer.attribute = attribute
        try:
            return orig(torch, model, ids, row, args, dev)
        finally:
            circuit_tracer.attribute = real

    AT.one_prompt = one_prompt
    return AT.main(argv)


if __name__ == "__main__":
    sys.exit(main())
