"""Latent pilot parts 3-4: the 6 attention targets (targets_attn.txt) under each rank / censoring arm on top of
geometric x3.15, λ 1e-3. Per target and arm: production pass (102's rule), size, Z / A / C / necessity on the
activation read, Z on the pre-activation read, and from the training curve (arms run after 2026-11-02 17:30 only) the
share of anchors with the target outside its site's Top-K under the zero ablation, mean over the last 50 steps.

  PYTHONIOENCODING=utf-8 python experiments/107-true-baselines/latent_pilot/compare_attn.py
"""
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ["geo_x3.15_l1e-3", "geo_x3.15_l1e-3_cut32", "geo_x3.15_l1e-3_cut64", "geo_x3.15_l1e-3_censor0.05",
        "geo_x3.15_l1e-3_rankzero10", "geo_x3.15_l1e-3_rival0.1", "geo_x3.15_l1e-3_censor0.05_rival0.1"]
BAND = (0.75, 1.5)


def jl(p):
    out = []
    for f in sorted(glob.glob(str(p))):
        for line in open(f):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def passes(e):
    if e is None or e.get("vacuous_tk"):
        return None
    v = [e.get(k) for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")]
    nec = e.get("phi_sup_blind_tk")
    return bool(all(isinstance(x, (int, float)) and BAND[0] <= x <= BAND[1] for x in v)
                and isinstance(nec, (int, float)) and nec >= 0.9)


def main():
    keys = [k for k in (HERE / "targets_attn.txt").read_text().split() if k]
    f = lambda x: "-" if not isinstance(x, (int, float)) else "%.2f" % x  # noqa: E731
    for a in ARMS:
        d = HERE / "out" / a / "main"
        ev = {r["seed"]: r for r in jl(d / "eval.shard*.jsonl") if r.get("held") == "strong"}
        st = {r["seed"]: r for r in jl(d / "status.shard*.jsonl") if r.get("ok")}
        tr = {r["seed"]: r for r in jl(d / "train.shard*.jsonl")}
        if not ev:
            print("== %s: (no rows yet)" % a)
            continue
        npass = sum(bool(passes(ev.get(k))) for k in keys)
        print("== %s: %d/%d pass" % (a, npass, sum(k in ev for k in keys)))
        for k in keys:
            e = ev.get(k)
            if e is None:
                continue
            cens = ((tr.get(k) or {}).get("curve") or {}).get("censored") or {}
            cz = cens.get("zero")
            cz = (sum(cz[-50:]) / len(cz[-50:])) if cz else None
            print("  %-14s %-4s n %5s | Z/A/C %s/%s/%s nec %s | Z pre %s | train: censored under Z (last 50) %s" % (
                k, "PASS" if passes(e) else "fail", (st.get(k) or {}).get("n", "-"), f(e.get("free0_tk")),
                f(e.get("freeM_topk_tk")), f(e.get("freeN_topk_tk")), f(e.get("phi_sup_blind_tk")),
                f(e.get("free0_pre")), f(cz)))


if __name__ == "__main__":
    main()
