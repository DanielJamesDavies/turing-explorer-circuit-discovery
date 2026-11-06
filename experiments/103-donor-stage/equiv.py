"""103 equivalence: the native donor test (src/eval/donor_test.py) against experiment 100's rows.

Compares, per (target, arm) present in both: the per-draw anchor means of the empty circuit and of the fit / a1 /
pinned variants (100 out/main/targets/<key>.<arm>.json), the donors drawn, the natural anchor reads, and the pooled
recoveries (100 out/main/rows.jsonl). Writes out/equiv/compare.json and prints a table. No GPU.

  python experiments/103-donor-stage/equiv.py [--native experiments/103-donor-stage/out/equiv]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E100 = HERE.parent / "100-scrub-frontier" / "out" / "main"


def load_jsonl(p):
    out = []
    for f in sorted(p.parent.glob(p.name)):
        for line in open(f):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def maxdiff(a, b):
    if a is None or b is None:
        return 0.0 if a is b else float("inf")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0) if len(a) == len(b) else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", type=Path, default=HERE / "out" / "equiv")
    a = ap.parse_args()
    rows = {(r["seed"], r["arm"]): r for r in load_jsonl(a.native / "rows*.jsonl") if "error" not in r and not r.get("skip")}
    raws = {(r["seed"], r["arm"]): r for r in load_jsonl(a.native / "raw*.jsonl")}
    ref = {(r["seed"], r["arm"]): r for r in load_jsonl(E100 / "rows.jsonl")}
    res = []
    for k, r in sorted(rows.items()):
        if k not in ref:
            continue
        q = ref[k]
        t = json.loads((E100 / "targets" / ("%s.%s.json" % k)).read_text())
        raw = raws[k]
        d = dict(seed=k[0], arm=k[1], n=r["n"], n_100=q["n"])
        d["natural_maxdiff"] = maxdiff(raw["natural"], [round(v, 6) for v in t["natural_raw"]])
        donors_same = True
        for ds in ("A", "C", "R"):
            e100 = [x for x in t["empty"] if x["background"] == ds]
            d["empty_%s_maxdiff" % ds] = maxdiff(r["empty_means"][ds], [x["mean"] for x in e100])
            donors_same &= raw["donor_indices"][ds] == [x["donor_indices"] for x in e100]
            for var in ("fit", "a1", "pinned"):
                v100 = [x["mean"] for x in t["variants"][var] if x["background"] == ds]
                d["%s_%s_maxdiff" % (var, ds)] = maxdiff(r["means"][var][ds], v100)
                ra, rb = r["recovery"][var][ds], q["recovery"][var][ds]
                d["rec_%s_%s" % (var, ds)] = [ra, rb]
                d["rec_%s_%s_diff" % (var, ds)] = (0.0 if ra is None and rb is None else
                                                   (abs(ra - rb) if ra is not None and rb is not None else float("inf")))
        d["donors_identical"] = donors_same
        d["pool_sha256"] = r.get("pool_sha256")
        diffs = [v for kk, v in d.items() if kk.endswith("maxdiff") or kk.endswith("_diff")]
        d["max_abs_diff_all"] = max(diffs)
        d["bit_identical"] = donors_same and d["max_abs_diff_all"] == 0.0
        res.append(d)
    pool_check = None
    if (a.native / "random_pool.pt").exists():
        import torch
        mine = torch.load(a.native / "random_pool.pt", weights_only=False, map_location="cpu")
        p99 = torch.load(HERE.parent / "099-recursive-causal-scrubbing" / "out" / "panel18" / "random_pool.pt",
                         weights_only=False, map_location="cpu")
        p73 = torch.load(HERE.parent / "073-strict-eval" / "out" / "random_pool.pt", weights_only=False,
                         map_location="cpu")
        pool_check = dict(native_built_n=int(mine["tokens"].shape[0]), n_099=int(p99["tokens"].shape[0]),
                          equals_099_panel18_pool=bool(torch.equal(mine["tokens"].cpu(), p99["tokens"].cpu())),
                          rebuilt_073_pool_equals_073_file=(mine.get("n_drawn") == int(p73["tokens"].shape[0]) and bool(
                              torch.equal(mine["tokens"].cpu(), p73["tokens"].cpu()[mine["retained_indices"]]))),
                          n_records=mine.get("n_records"))
        print("pool:", pool_check)
    out = a.native / "compare.json"
    out.write_text(json.dumps(dict(pool=pool_check, rows=res), indent=1) + "\n")
    print("%-14s %-10s %5s %6s %9s %13s  rec fit A / C / R (native | 100)" % ("target", "arm", "n", "donors", "max diff",
                                                                             "bit-identical"))
    for d in res:
        print("%-14s %-10s %5d %6s %9.2g %13s  %s" % (
            d["seed"], d["arm"], d["n"], d["donors_identical"], d["max_abs_diff_all"], d["bit_identical"],
            " / ".join("%.4f|%.4f" % tuple(x if x is not None else float("nan") for x in d["rec_fit_%s" % ds])
                       for ds in ("A", "C", "R"))))
    print("%d / %d (target, arm) bit-identical" % (sum(d["bit_identical"] for d in res), len(res)))


if __name__ == "__main__":
    main()
