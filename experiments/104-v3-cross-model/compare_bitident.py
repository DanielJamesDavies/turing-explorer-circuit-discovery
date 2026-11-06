"""104 part A: compare the native WCM_VERSION=v3 driver fits (run_bitident.sh -> out/bitident) with exp 102's
r1.0_rk1e-1 circuits (102 out/check/circuits/r1.0_rk1e-1, p102 engine on the frozen learned_mask copy b9a6142b).
CPU only, read-only on every input; writes out/bitident_compare.json and prints a summary.

Per target: circuit identity (members + amplitudes, exact), Jaccard, max |alpha diff|, the training loss curve and every
per-term curve (CURVE_STRIDE 20 in both), and every numeric production score of the held-out-strongest row vs 102's
row["prod"] (max |diff|, the scorer's wall time / memory excluded).

  PYTHONPATH=src python experiments/104-v3-cross-model/compare_bitident.py
"""
import json
import math
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
E102 = HERE.parent / "102-pass-improvement" / "out" / "check"
ARM = "r1.0_rk1e-1"
RUN_STATS = ("secs", "peak_mem_mb")


def load_jsonl(pattern_path):
    out = []
    for path in sorted(pattern_path.parent.glob(pattern_path.name)):
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
    keys = [k for k in (HERE / "targets_bitident.txt").read_text().split() if k]
    d = HERE / "out" / "bitident" / "main"
    rows102 = {r["seed"]: r for r in load_jsonl(E102 / "rows.jsonl") if r.get("arm") == ARM and "error" not in r}
    train102 = {r["seed"]: r for r in load_jsonl(E102 / "train.jsonl") if r.get("arm102") == ARM}
    status = {r["seed"]: r for r in load_jsonl(d / "status.shard*.jsonl") if r.get("ok")}
    evals = {r["seed"]: r for r in load_jsonl(d / "eval.shard*.jsonl") if r.get("held") == "strong"}
    trains = {r["seed"]: r for r in load_jsonl(d / "train.shard*.jsonl")}
    res = {}
    for k in keys:
        e = dict(done=k in status, t_fit=status.get(k, {}).get("t_fit"), n=status.get(k, {}).get("n"),
                 t_fit_102=(rows102.get(k) or {}).get("t_fit"))
        res[k] = e
        cp = d / "circuits" / ("%s.pt" % k)
        if not cp.exists():
            continue
        a, b = sig(torch.load(cp, weights_only=False)), sig(torch.load(E102 / "circuits" / ARM / ("%s.pt" % k),
                                                                         weights_only=False))
        ka, kb = {x[:3] for x in a}, {x[:3] for x in b}
        e.update(identical=a == b, members_identical=ka == kb, n_ref=len(b),
                 jaccard=len(ka & kb) / max(len(ka | kb), 1),
                 max_alpha_diff=(max((abs(x[3] - y[3]) for x, y in zip(a, b)), default=0.0) if ka == kb else None))
        tr, t2 = trains.get(k), train102.get(k)
        if tr is not None and t2 is not None:
            c1, c2 = tr.get("curve") or {}, t2.get("curve") or {}
            e["same_loss_curve"] = c1.get("loss") == c2.get("loss")
            e["same_term_curves"] = c1.get("terms") == c2.get("terms")
            e["term_names"] = sorted((c1.get("terms") or {}).keys())
            e["resample_weight"] = tr.get("resample_weight")
            e["same_loss_final"] = tr.get("loss_final") == t2.get("loss_final")
        p_new, r2 = evals.get(k), rows102.get(k)
        if p_new is not None and r2 is not None:
            p_ref = r2["prod"]
            diffs = {m: abs(p_new[m] - p_ref[m]) for m in p_ref
                     if num(p_ref.get(m)) and num(p_new.get(m)) and m not in RUN_STATS}
            e.update(n_scores=len(diffs), max_score_diff=max(diffs.values()) if diffs else None,
                     score_diff_keys=sorted(m for m, v in diffs.items() if v > 0)[:10],
                     wcm_version=p_new.get("wcm_version"))
    out = HERE / "out" / "bitident_compare.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    for k, e in res.items():
        print("%-14s done %-5s n %-5s (102 %s) identical %-5s jacc %s max|da| %s curve %s terms %s max|dscore| %s "
              "(%s scores) t_fit %s (102 %s)" % (
                  k, e.get("done"), e.get("n"), e.get("n_ref"), e.get("identical"), e.get("jaccard"),
                  e.get("max_alpha_diff"), e.get("same_loss_curve"), e.get("same_term_curves"), e.get("max_score_diff"),
                  e.get("n_scores"), e.get("t_fit"), e.get("t_fit_102")))


if __name__ == "__main__":
    main()
