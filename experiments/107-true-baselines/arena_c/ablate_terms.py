"""Arena C diagnostic: which v3 term destabilises a behavioural fit? Refit one (task, λ, means) cell with the v3 terms
switched on / off and score each fit the same way as the run (wcm_marks.run).

  ./.venv/bin/python experiments/107-true-baselines/arena_c/ablate_terms.py --task simple --lam 1e-5 --means pooled

Variants: v3 (as run), no_rank (rank_weight 0), no_R (resample_weight 0), v2 (neither). Rows go to
out/ablate_terms/rows.jsonl, one per variant (resumable: a scored variant is skipped).
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wcm_marks as WM  # noqa: E402

VARIANTS = {"v3": {}, "no_rank": dict(rank_weight=0.0), "no_R": dict(resample_weight=0.0),
            "v2": dict(rank_weight=0.0, resample_weight=0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="simple")
    ap.add_argument("--lam", type=float, default=1e-5)
    ap.add_argument("--means", default="pooled")
    ap.add_argument("--sites", default="noembed")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--save", default=None,
                    help="(2026-11-02) fit each variant through wcm_marks.run into <save>/<variant>, so the circuit "
                         "JSON and its eval row are kept (resumable; for donor_tasks.py). rows.jsonl is not touched")
    a = ap.parse_args()
    if a.save:
        eng = WM.make_engine(a.sites, means=a.means)
        base = dict(WM.V3_LOGIT)
        try:
            for v in a.variants.split(","):
                WM.V3_LOGIT.clear()
                WM.V3_LOGIT.update(base, **VARIANTS[v])
                WM.run([a.task], [a.lam], "sfc", a.steps, a.sites, out=Path(a.save) / v, eng=eng, means=a.means)
        finally:
            WM.V3_LOGIT.clear()
            WM.V3_LOGIT.update(base)
        return
    out = HERE / "out" / "ablate_terms"
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    if (out / "rows.jsonl").exists():
        done = {(r["task"], r["lam"], r["means"], r["variant"]) for r in map(json.loads, open(out / "rows.jsonl"))}
    C, _ = WM._import_081()
    T = WM.build_task(a.task, "sfc")
    eng = WM.make_engine(a.sites, means=a.means)
    B = eng.batch(T)
    base = dict(WM.V3_LOGIT)
    for v in a.variants.split(","):
        key = (a.task, a.lam, a.means, v)
        if key in done:
            print("skip", key)
            continue
        WM.V3_LOGIT.clear()
        WM.V3_LOGIT.update(base, **VARIANTS[v])
        alphas, prov, t_fit, peak = eng.fit(B, "W", a.lam, C.PROD_LR, a.steps)
        sc = eng.score(B, alphas)
        g = lambda k: sc.get("task_" + k)   # noqa: E731
        n = sum(len(d) for d in alphas.values())
        cl = WM.curve_summary(prov)
        row = dict(task=a.task, lam=a.lam, means=a.means, variant=v, n=n, Z=g("free0"), A=g("freeM_topk"),
                   C=g("freeN_topk"), nec=g("nec"), Z_a1=g("free0_a1"), t_fit=round(t_fit, 1),
                   loss_final=prov.get("loss_final"),
                   curve_last={k: s["last"] for k, s in cl.items() if s and k in ("rank", "loss") or k.startswith("terms.")
                               and s},
                   wcm=dict(WM.V3_LOGIT))
        with open(out / "rows.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print("%-8s n %5d  Z %5.2f A %5.2f C %5.2f nec %5.2f  a1Z %5.2f  rank_last %s  (%.0fs)" % (
            v, n, row["Z"], row["A"], row["C"], row["nec"], row["Z_a1"],
            None if row["curve_last"].get("rank") is None else round(row["curve_last"]["rank"], 3), t_fit), flush=True)
    WM.V3_LOGIT.clear()
    WM.V3_LOGIT.update(base)


if __name__ == "__main__":
    main()
