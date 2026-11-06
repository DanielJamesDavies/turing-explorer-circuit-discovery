"""101: compare the native-R driver smoke (run_smoke.sh) with the circuits it should reproduce, and report the fit-time
ratio. CPU only, read-only on every input; writes out/compare.json and prints a summary.

  v2cr (native, WCM_VERSION=v2cr)  vs  100 Run A/C v2cr_1e-3 (095 rs2 patch, same 099 records, same src engine)
  v2   (WCM_VERSION=v2)            vs  099's production circuits (which 100 reused after a bit-identical refit check)
Per target: circuit identity (members + amplitudes, exact), Jaccard, max |alpha diff|, the per-step training loss curve
(CURVE_STRIDE 20) and every numeric production score of the held-out-strongest row (max |diff|); then the fit-time
ratio v2cr / v2 from the smoke's own status rows (same process type, same GPU, same session).

  PYTHONPATH=src python experiments/101-r-native/compare.py
"""
import json
import math
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = HERE / "out"
E100 = EXP / "100-scrub-frontier" / "out" / "main"
CIRC99 = EXP / "099-recursive-causal-scrubbing" / "out" / "panel18" / "circuits"
REF = {"v2cr": dict(circ=lambda k: E100 / "circuits" / "v2cr_1e-3" / ("%s.pt" % k), arm="v2cr_1e-3"),
       "v2": dict(circ=lambda k: CIRC99 / ("%s.wcm.pt" % k), arm="wcm_1e-3")}


RUN_STATS = ("secs", "peak_mem_mb")          # the scorer's wall time / memory: not scores


def load_jsonl(p):
    out = []
    for path in sorted(p.parent.glob(p.name)):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return out


def sig(c):
    out = []
    for n in c.nodes.values():
        if n.metadata.get("role") == "seed":
            continue
        f = n.metadata["feature_id"]
        out.append((int(f.layer), str(f.kind), int(f.index), float(n.metadata.get("amplitude", 1.0))))
    return sorted(out)


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and not math.isinf(v)


def main():
    rows100 = {(r["seed"], r["arm"]): r for r in load_jsonl(E100 / "rows.jsonl") if "error" not in r}
    train100 = {(r.get("seed"), r.get("arm")): r for r in load_jsonl(E100 / "train.jsonl")
                if r.get("kind") == "circuit"}
    res = {"targets": {}, "fit_time": {}}
    keys = [k for k in (HERE / "targets_smoke.txt").read_text().split() if k]
    for ver in ("v2cr", "v2"):
        d = OUT / ("smoke_%s" % ver) / "main"
        status = {r["seed"]: r for r in load_jsonl(d / "status.shard*.jsonl") if r.get("ok")}
        evals = {r["seed"]: r for r in load_jsonl(d / "eval.shard*.jsonl") if r.get("held") == "strong"}
        trains = {r["seed"]: r for r in load_jsonl(d / "train.shard*.jsonl")}
        for k in keys:
            t = res["targets"].setdefault(k, {})
            e = dict(done=k in status, t_fit=status.get(k, {}).get("t_fit"), n=status.get(k, {}).get("n"))
            t[ver] = e
            cpath = d / "circuits" / ("%s.pt" % k)
            if not cpath.exists():
                continue
            c = torch.load(cpath, weights_only=False)
            rp = REF[ver]["circ"](k)
            if rp.exists():
                ref = torch.load(rp, weights_only=False)
                a, b = sig(c), sig(ref)
                ka, kb = {x[:3] for x in a}, {x[:3] for x in b}
                e.update(identical=a == b, n_ref=len(b), jaccard=len(ka & kb) / max(len(ka | kb), 1),
                         max_alpha_diff=(max((abs(x[3] - y[3]) for x, y in zip(a, b)), default=0.0)
                                         if ka == kb else None))
            r100 = rows100.get((k, REF[ver]["arm"]))
            if r100 is not None and k in evals:
                p_ref, p_new = r100["prod"], evals[k]
                diffs = {m: abs(p_new[m] - p_ref[m]) for m in p_ref
                         if num(p_ref.get(m)) and num(p_new.get(m)) and m not in RUN_STATS}
                e.update(n_scores=len(diffs), max_score_diff=max(diffs.values()) if diffs else None,
                         score_diff_keys=sorted(m for m, v in diffs.items() if v > 0)[:10],
                         pass_inputs={m: (p_new.get(m), p_ref.get(m)) for m in
                                      ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk")})
                e["t_fit_ref"] = (r100.get("fit") or {}).get("t_fit")
            tr = trains.get(k)
            t_ref = train100.get((k, REF[ver]["arm"]))
            if tr is not None:
                e.update(term_names=sorted((tr.get("curve") or {}).get("terms", {}).keys()),
                         resample_draws=tr.get("resample_draws"), resample_fill_calls=tr.get("resample_fill_calls"))
                if t_ref is not None:
                    e["same_loss_curve"] = (tr.get("curve") or {}).get("loss") == (t_ref.get("curve") or {}).get("loss")
    ratios = [t["v2cr"]["t_fit"] / t["v2"]["t_fit"] for t in res["targets"].values()
              if t.get("v2cr", {}).get("t_fit") and t.get("v2", {}).get("t_fit")]
    res["fit_time"] = dict(per_target={k: (t.get("v2cr", {}).get("t_fit"), t.get("v2", {}).get("t_fit"))
                                       for k, t in res["targets"].items()},
                           ratio_median=sorted(ratios)[len(ratios) // 2] if ratios else None,
                           ratio_total=(sum(t["v2cr"]["t_fit"] for t in res["targets"].values() if t.get("v2cr", {}).get("t_fit") and t.get("v2", {}).get("t_fit"))
                                        / max(sum(t["v2"]["t_fit"] for t in res["targets"].values() if t.get("v2cr", {}).get("t_fit") and t.get("v2", {}).get("t_fit")), 1e-9))
                           if ratios else None)
    # the same ratio from the uncontended historical runs on this GPU, all 18 panel targets: 100's v2cr fits (095 patch,
    # same engine code path) over 099's v2 fits (the circuits 100 reused), as 100's rows record them
    hist = {}
    for (k, arm), r in rows100.items():
        if arm in ("v2cr_1e-3", "wcm_1e-3") and (r.get("fit") or {}).get("t_fit"):
            hist.setdefault(k, {})[arm] = float(r["fit"]["t_fit"])
    hr = sorted(v["v2cr_1e-3"] / v["wcm_1e-3"] for v in hist.values() if len(v) == 2)
    both = [v for v in hist.values() if len(v) == 2]
    res["fit_time"]["historical_100_vs_099"] = dict(
        n=len(hr), ratio_median=hr[len(hr) // 2] if hr else None, ratio_min=hr[0] if hr else None,
        ratio_max=hr[-1] if hr else None,
        ratio_total=(sum(v["v2cr_1e-3"] for v in both) / sum(v["wcm_1e-3"] for v in both)) if both else None)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare.json").write_text(json.dumps(res, indent=1, default=str))
    for k, t in res["targets"].items():
        for ver in ("v2cr", "v2"):
            e = t.get(ver, {})
            print("%-14s %-4s done %-5s n %-5s identical %-5s jacc %s max|da| %s curve %s max|dscore| %s t_fit %s (ref %s)" % (
                k, ver, e.get("done"), e.get("n"), e.get("identical"), e.get("jaccard"), e.get("max_alpha_diff"),
                e.get("same_loss_curve"), e.get("max_score_diff"), e.get("t_fit"), e.get("t_fit_ref")))
    print("fit time v2cr / v2:", res["fit_time"])


if __name__ == "__main__":
    main()
