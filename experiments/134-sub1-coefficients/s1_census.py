"""134 step 1 (CPU): members with fitted coefficient alpha < 1.

  (a) census of the production members (113's parquet cache): share below 1 / 0.9 / 0.5 by model x member kind,
      member depth (layer distance to the target), target band, and gate m;
  (b) the off-term pilot (131): per target, v4 vs o4 (and o1, o0.25) coefficient distributions; members shared by both
      arms, matched on (layer, kind, index): how many move below 1 under the off term, and the paired alpha change.

Writes out/s1_census.md. Python, repo root:
  PYTHONPATH=src .venv/bin/python -B experiments/134-sub1-coefficients/s1_census.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
M113 = ROOT / "experiments" / "113-coefficients-shape" / "out"
C131 = ROOT / "experiments" / "131-off-term" / "out" / "circuits"
MODELS = ("turingllm", "pythia-70m", "gpt2-small")


def shares(a):
    a = np.asarray(a)
    return dict(n=len(a), lt1=np.mean(a < 1), lt09=np.mean(a < 0.9), lt05=np.mean(a < 0.5), med=np.median(a))


def table(df, by, title):
    rows = ["", "### " + title, "", "| %s | members | median alpha | < 1 | < 0.9 | < 0.5 |" % " | ".join(by),
            "|" + "---|" * (len(by) + 5)]
    for key, g in df.groupby(list(by), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        s = shares(g["alpha"])
        rows.append("| %s | %d | %.3f | %.1f%% | %.1f%% | %.1f%% |" % (
            " | ".join(str(k) for k in key), s["n"], s["med"], 100 * s["lt1"], 100 * s["lt09"], 100 * s["lt05"]))
    return rows


def band(layer, model):
    if model == "pythia-70m":
        return "L0-1" if layer <= 1 else "L2-3" if layer <= 3 else "L4-5"
    return "L0-4" if layer <= 4 else "L5-7" if layer <= 7 else "L8-11"


def read_circuit(path):
    import torch
    c = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for n in c.nodes.values():
        md = n.metadata
        if md.get("role") == "seed":
            continue
        f = md["feature_id"]
        out[(int(f.layer), str(f.kind), int(f.index))] = (float(md.get("amplitude", np.nan)),
                                                          float(md.get("attribution_score", np.nan)))
    return out


def main():
    lines = ["# 134 step 1: members with alpha < 1 (CPU)", ""]
    # (a) production census
    for model in MODELS:
        df = pd.read_parquet(M113 / ("members_%s_main.parquet" % model))
        df["dist"] = (df["t_layer"] - df["layer"]).clip(lower=0)
        df["dist_bin"] = pd.cut(df["dist"], [-1, 0, 1, 3, 5, 99], labels=["0", "1", "2-3", "4-5", "6+"])
        df["t_band"] = [band(l, model) for l in df["t_layer"]]
        df["m_bin"] = pd.cut(df["m"], [-np.inf, 0.6, 0.8, 0.95, np.inf], labels=["<0.6", "0.6-0.8", "0.8-0.95", ">=0.95"])
        s = shares(df["alpha"])
        lines += ["", "## %s production (%d members, %d circuits): < 1 %.1f%%, < 0.9 %.1f%%, < 0.5 %.1f%%"
                  % (model, s["n"], df["seed"].nunique(), 100 * s["lt1"], 100 * s["lt09"], 100 * s["lt05"])]
        lines += table(df, ["kind"], "by member kind")
        lines += table(df, ["dist_bin"], "by layer distance to the target")
        lines += table(df, ["t_band"], "by target band")
        lines += table(df, ["t_kind"], "by target kind")
        lines += table(df, ["m_bin"], "by mask gate m")
        # per-circuit: share of a circuit's members below 0.9
        pc = df.groupby("seed")["alpha"].apply(lambda a: np.mean(a < 0.9))
        lines.append("")
        lines.append("Per circuit share < 0.9: median %.1f%%, p10 %.1f%%, p90 %.1f%%"
                     % tuple(100 * np.percentile(pc, q) for q in (50, 10, 90)))

    # (b) the off-term pilot
    lines += ["", "## 131 off-term pilot (GPT-2 small): v4 vs off arms", "",
              "| target | arm | n | < 1 | < 0.9 | < 0.5 | median | shared with v4 | shared: v4 >= 1 -> arm < 1 | shared: median d alpha | shared: d alpha < -0.1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    v4dir = C131 / "v4"
    agg = {a: [] for a in ("o0.25", "o1", "o4")}
    for p in sorted(v4dir.glob("*.pt")):
        base = read_circuit(p)
        ab = np.array([v[0] for v in base.values()])
        s = shares(ab)
        lines.append("| %s | v4 | %d | %.1f%% | %.1f%% | %.1f%% | %.3f | | | | |" % (
            p.stem, s["n"], 100 * s["lt1"], 100 * s["lt09"], 100 * s["lt05"], s["med"]))
        for arm in ("o0.25", "o1", "o4"):
            q = C131 / arm / p.name
            if not q.exists():
                continue
            c = read_circuit(q)
            a = np.array([v[0] for v in c.values()])
            s = shares(a)
            sh = sorted(set(base) & set(c))
            d = np.array([c[k][0] - base[k][0] for k in sh])
            flip = sum(1 for k in sh if base[k][0] >= 1 and c[k][0] < 1)
            agg[arm].append(dict(target=p.stem, d=d, flip=flip, n_sh=len(sh), lt1_v4=np.mean(ab < 1), lt1=s["lt1"],
                                 lt05_v4=np.mean(ab < 0.5), lt05=s["lt05"]))
            lines.append("| | %s | %d | %.1f%% | %.1f%% | %.1f%% | %.3f | %d | %d | %+.3f | %.1f%% |" % (
                arm, s["n"], 100 * s["lt1"], 100 * s["lt09"], 100 * s["lt05"], s["med"], len(sh), flip,
                np.median(d) if len(d) else np.nan, 100 * np.mean(d < -0.1) if len(d) else np.nan))
    lines += ["", "Summary over targets (paired, arm minus v4):", ""]
    for arm, rs in agg.items():
        if not rs:
            continue
        dlt1 = [r["lt1"] - r["lt1_v4"] for r in rs]
        dlt05 = [r["lt05"] - r["lt05_v4"] for r in rs]
        dall = np.concatenate([r["d"] for r in rs])
        lines.append("- %s: %d targets; share < 1 change median %+.1f pts [%d up / %d down]; share < 0.5 change median "
                     "%+.1f pts [%d up / %d down]; shared members' alpha change median %+.3f, %.1f%% fall by > 0.1, "
                     "%.1f%% rise by > 0.1" % (
                         arm, len(rs), 100 * np.median(dlt1), sum(x > 0 for x in dlt1), sum(x < 0 for x in dlt1),
                         100 * np.median(dlt05), sum(x > 0 for x in dlt05), sum(x < 0 for x in dlt05),
                         np.median(dall), 100 * np.mean(dall < -0.1), 100 * np.mean(dall > 0.1)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s1_census.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
