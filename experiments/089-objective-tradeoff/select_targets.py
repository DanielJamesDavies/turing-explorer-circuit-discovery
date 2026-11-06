"""089: pick ~8 failing production circuits (seed 20261025) for the objective trade-off test. No GPU.

Failure pattern (062 main eval, held == "strong", not vacuous): Z (free0_tk) in [0.8, 1.5], necessity
(phi_sup_blind_tk) >= 0.9, A (freeM_topk_tk) and C (freeN_topk_tk) both < 0.8. 6 MLP / residual targets at L5-11,
2 attention targets with clean rank < 64 (062 spec rows, pi == "Z"), so near-threshold Top-K failures are excluded.
Writes targets.txt (one key per line) and prints the picked rows.
"""
import glob
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
MAIN = HERE.parent / "062-h100-protocol-v1" / "out_full" / "out" / "main"


def rows(pattern):
    for f in sorted(glob.glob(str(MAIN / pattern))):
        for line in open(f):
            try:
                yield json.loads(line)
            except Exception:  # noqa: BLE001
                continue


def main():
    ev = {}
    for r in rows("eval.shard*.jsonl"):
        if r.get("held") == "strong" and "error" not in r and "skip" not in r:
            ev[r["seed"]] = r
    rank = {r["seed"]: r.get("rank_clean") for r in rows("spec.shard*.jsonl") if r.get("pi") == "Z"}

    def fails(r):
        z, a, c, nec = (r.get(k) for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk"))
        if None in (z, a, c, nec) or r.get("vacuous_tk"):
            return False
        return 0.8 <= z <= 1.5 and nec >= 0.9 and a < 0.8 and c < 0.8

    pool = [r for r in ev.values() if fails(r)]
    deep = sorted(r["seed"] for r in pool if r["kind"] in ("mlp", "resid") and 5 <= int(r["layer"]) <= 11
                  and (HERE.parent / "062-h100-protocol-v1" / "out_full" / "out" / "ctx" / ("%s.pt" % r["seed"])).exists())
    attn = sorted(r["seed"] for r in pool if r["kind"] == "attn" and rank.get(r["seed"]) is not None
                  and rank[r["seed"]] < 64)
    print("failing (Z ok, nec ok, A and C < 0.8): %d | L5-11 mlp/resid %d | attn rank<64 %d" % (
        len(pool), len(deep), len(attn)))
    rng = random.Random(20261025)
    pick = rng.sample(deep, 6) + rng.sample(attn, 2)
    for k in pick:
        r = ev[k]
        print("%-16s n %4d  Z %.2f  A %.2f  C %.2f  nec %.2f  rank %s" % (
            k, r["n"], r["free0_tk"], r["freeM_topk_tk"], r["freeN_topk_tk"], r["phi_sup_blind_tk"], rank.get(k)))
    (HERE / "targets.txt").write_text("\n".join(pick) + "\n")


if __name__ == "__main__":
    main()
