"""Items 2 (off-manifold check) and 3 (per-context spread of faithfulness): what the archives store.

Prints (and writes out/run2026_coef_stored.json):
  - the metadata keys of the circuit members (all three models' main circuits, every node scanned);
  - the eval-row keys that hold a list (the only candidates for per-context reads), main held strong rows;
  - the ctx record keys (TuringLLM ctx_fresh) that hold activations.
  python -I -X utf8 experiments/113-coefficients-shape/c23_stored.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import lib113 as C  # noqa: E402

L = C.L


def main():
    out = {}
    for model in C.MODELS:
        d = C.circuit_dir(model, "main")
        keys = set()
        for p in sorted(d.glob("*.pt"))[:50]:
            c = torch.load(p, map_location="cpu", weights_only=False)
            for n in c.nodes.values():
                keys |= set(n.metadata)
            keys |= {"circuit." + k for k in c.metadata}
        rows = L.read_rows(L.find(model, "out", "main"), "eval")
        lists = sorted({k for r in rows[:2000] for k, v in r.items() if isinstance(v, (list, dict))})
        out[model] = dict(node_and_circuit_keys=sorted(keys), eval_list_or_dict_keys=lists)
        print(model, json.dumps(out[model], indent=1))
    ctx = next(iter(C.L.find("turingllm", "work", "ctx_fresh")[0].glob("*.pt")))
    o = torch.load(ctx, map_location="cpu", weights_only=False)
    rec = next(iter(o.values()))
    out["turingllm_ctx_record"] = {k: (sorted(v) if isinstance(v, dict) else type(v).__name__) for k, v in rec.items()}
    print(json.dumps(out["turingllm_ctx_record"], indent=1))
    (C.OUT / (C.PREFIX + "stored.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
