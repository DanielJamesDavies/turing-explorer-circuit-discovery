"""090: pick the ~20 targets for the sparsity-preserving-fill training test (seed 20261025). No GPU.

Groups (062 production main eval, held == "strong", not vacuous, context record and circuit on disk):
  fail89   the 8 failing circuits of 089 (089/targets.txt)
  attn     4 deep (L5-11) attention targets whose DENSE A >= 0.8 but RULE A < 0.5 (freeM_dense_tk vs freeM_topk_tk):
           2 near-threshold (clean rank >= 64 in 062 spec rows, pi == "Z") and 2 not (clean rank < 64)
  control  8 PASSING production circuits (rule pass: Z, A, C in [0.8, 1.5], necessity >= 0.9), one per band x kind
           cell for 8 of the 9 cells (bands L0-4 / L5-7 / L8-11, kinds attn / mlp / resid; the dropped cell is drawn)
Writes targets.json ({group: [keys]}) and targets.txt (all keys, group order), and prints the picked rows.
"""
import glob
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
EXP = HERE.parent
OUT62 = EXP / "062-h100-protocol-v1" / "out_full" / "out"
MAIN = OUT62 / "main"
SEED = 20261025
BANDS = ("L0-4", "L5-7", "L8-11")
KINDS = ("attn", "mlp", "resid")


def band(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def rows(pattern):
    for f in sorted(glob.glob(str(MAIN / pattern))):
        for line in open(f):
            try:
                yield json.loads(line)
            except Exception:  # noqa: BLE001
                continue


def passes(r):
    v = [r.get(k) for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")]
    nec = r.get("phi_sup_blind_tk")
    return None not in v and nec is not None and all(0.8 <= x <= 1.5 for x in v) and nec >= 0.9


def on_disk(key):
    return (OUT62 / "ctx" / ("%s.pt" % key)).exists() and (MAIN / "circuits" / ("%s.pt" % key)).exists()


def main():
    ev = {}
    for r in rows("eval.shard*.jsonl"):
        if r.get("held") == "strong" and "error" not in r and "skip" not in r and not r.get("vacuous_tk"):
            ev[r["seed"]] = r
    rank = {r["seed"]: r.get("rank_clean") for r in rows("spec.shard*.jsonl") if r.get("pi") == "Z"}
    fail89 = [t for t in (EXP / "089-objective-tradeoff" / "targets.txt").read_text().split() if t]
    taken = set(fail89)
    rng = random.Random(SEED)

    def deep_attn(r):
        a_d, a_r = r.get("freeM_dense_tk"), r.get("freeM_topk_tk")
        return (r["kind"] == "attn" and 5 <= int(r["layer"]) <= 11 and a_d is not None and a_r is not None
                and a_d >= 0.8 and a_r < 0.5 and rank.get(r["seed"]) is not None)
    pool = sorted(k for k, r in ev.items() if deep_attn(r) and k not in taken)
    near = [k for k in pool if rank[k] >= 64]
    far = [k for k in pool if rank[k] < 64]
    print("deep attention, dense A >= 0.8 and rule A < 0.5: %d (near-threshold %d, not %d)" % (
        len(pool), len(near), len(far)))
    near_p = [k for k in rng.sample(near, len(near)) if on_disk(k)][:2]
    far_p = [k for k in rng.sample(far, len(far)) if on_disk(k)][:2]
    attn = near_p + far_p
    taken |= set(attn)

    cells = [(b, k) for b in BANDS for k in KINDS]
    rng.shuffle(cells)
    cells = cells[:8]
    control = []
    for b, kd in cells:
        cand = sorted(k for k, r in ev.items() if passes(r) and r["kind"] == kd and band(int(r["layer"])) == b
                      and k not in taken)
        pick = next((k for k in rng.sample(cand, len(cand)) if on_disk(k)), None)
        print("control cell %-5s %-5s: %d passing, picked %s" % (b, kd, len(cand), pick))
        if pick:
            control.append(pick)
            taken.add(pick)

    groups = {"fail89": fail89, "attn": attn, "control": control}
    sub = {k: ("near" if k in near_p else "far") for k in attn}
    for g, ks in groups.items():
        for k in ks:
            r = ev.get(k, {})
            f = lambda x: "-" if x is None else "%.2f" % x
            print("%-8s %-16s %-5s n %4s  Z %s  A %s  C %s  A dense %s  nec %s  rank %s" % (
                g, k, sub.get(k, ""), r.get("n"), f(r.get("free0_tk")), f(r.get("freeM_topk_tk")),
                f(r.get("freeN_topk_tk")), f(r.get("freeM_dense_tk")), f(r.get("phi_sup_blind_tk")), rank.get(k)))
    (HERE / "targets.json").write_text(json.dumps(dict(groups, attn_sub=sub), indent=1) + "\n")
    (HERE / "targets.txt").write_text("\n".join(k for ks in groups.values() for k in ks) + "\n")


if __name__ == "__main__":
    main()
