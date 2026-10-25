"""DAN-64 follow-up (no refitting): does a circuit's C faithfulness depend on WHICH part of its contrast background
the C mean is taken from?

Each circuit fitted at lambda 1e-3 (default arms: random_silent, then close_silent) is re-scored on held-out strongest
with the C ablation mean computed from one third of the random set's 48 training contexts, ranked by similarity to
the target's top contexts (set_geometry.py's contexts.csv): near (most similar 16), mid, far. Reference row "all" =
the random set itself, as in the sweep's battery, scored in the same process. Everything else is fixed: the circuit,
its coefficients, the activating contexts, and the random set's 16 held-out contexts (sufficiency to induce).

The scorer takes the contrast set's first 48 rows as the C-mean set, so a third is passed three times over (its mean is
exactly the third's mean) followed by the 16 held-out rows.

  OUT=experiments/071-contrast-sweep/out PYTHONPATH=src python experiments/071-contrast-sweep/thirds_rescore.py
  env: ARMS (random_silent,close_silent)  LAM (0.001)  REPORT=1 (summary only; no GPU)
Writes OUT/thirds/thirds.jsonl (resumable, one row per target x arm x third) and OUT/thirds/report.md.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).parent
OUT = Path(os.environ.get("OUT", str(HERE / "out")))
os.environ["OUT"] = str(OUT)                         # driver.Runner reads its ctx records from OUT/ctx
os.environ["EVAL_SPLIT"] = "explicit"
ARMS = [a for a in os.environ.get("ARMS", "random_silent,close_silent").split(",") if a]
LAM = os.environ.get("LAM", "0.001")
TH = OUT / "thirds"
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]
THIRDS = ("near", "mid", "far", "all")


def band(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def main():
    import pandas as pd
    import torch
    sys.path.insert(0, str(HERE.parent / "062-h100-protocol-v1"))
    import driver
    sims = pd.read_csv(OUT / "geometry" / "contexts.csv")
    sims = sims[(sims.set == "random") & (sims.split == "train")]
    sim_of = {(r.seed, int(r.seq_id)): r.sim for r in sims.itertuples()}
    TH.mkdir(parents=True, exist_ok=True)
    path = TH / "thirds.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "error" not in r:
                done.add((r["seed"], r["arm"], r["third"]))
    keys = [t for t in (HERE / "targets.txt").read_text().split() if t]
    R = driver.Runner()
    fh = open(path, "a")
    t0 = time.time()
    for arm in ARMS:
        for n_, key in enumerate(keys):
            if all((key, arm, t) in done for t in THIRDS):
                continue
            try:
                c = torch.load(OUT / "circuits" / arm / LAM / ("%s.pt" % key), weights_only=False)
                st = torch.load(OUT / "sets" / "random_silent" / ("%s.pt" % key), weights_only=False)
                rec = R.contexts(key)
                n_tr = int(st["n_train"])
                toks, ids = st["tokens"].cpu().long(), [int(x) for x in st["ids"]]
                s = [sim_of.get((key, ids[j]), float("nan")) for j in range(n_tr)]
                order = sorted(range(n_tr), key=lambda j: (s[j] != s[j], -s[j] if s[j] == s[j] else 0))
                third = n_tr // 3
                parts = {"near": order[:third], "mid": order[third:2 * third], "far": order[2 * third:3 * third]}
                held = toks[n_tr:]
                for name in THIRDS:
                    if (key, arm, name) in done:
                        continue
                    if name == "all":
                        neg = toks
                        sim_med = float(pd.Series(s).median())
                    else:
                        idx = parts[name]
                        rep = toks[idx].repeat(n_tr // len(idx) + 1, 1)[:n_tr]        # the third, repeated to 48 rows
                        neg = torch.cat([rep, held])
                        sim_med = float(pd.Series([s[j] for j in idx]).median())
                    row = R.score(c, dict(rec, neg=neg), "strong")
                    row = {k: v for k, v in row.items() if isinstance(v, (int, float, str, bool)) or v is None}
                    row.update(seed=key, arm=arm, third=name, sim_med=sim_med, n_nan_sim=sum(x != x for x in s))
                    fh.write(json.dumps(row) + "\n"); fh.flush()
            except Exception as e:  # noqa: BLE001
                fh.write(json.dumps(dict(seed=key, arm=arm, third="error", error="%s: %s" % (type(e).__name__,
                                                                                            str(e)[:300]))) + "\n")
                fh.flush()
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print("%-14s %3d/%d %-16s %.0fs" % (arm, n_ + 1, len(keys), key, time.time() - t0), flush=True)
    print("DONE", flush=True)


def report():
    import numpy as np
    import pandas as pd
    d = pd.DataFrame([json.loads(l) for l in open(TH / "thirds.jsonl")])
    d = d[d.third.isin(THIRDS)].drop_duplicates(["seed", "arm", "third"], keep="last")
    d = d[~d["vacuous_tk"].fillna(False).astype(bool)]
    ok = d[HEAD].notna().all(axis=1)
    d["passed"] = ok & ((d[HEAD] >= 0.8) & (d[HEAD] <= 1.5)).all(axis=1) & (d["phi_sup_blind_tk"] >= 0.9)
    d["band"] = d.seed.map(lambda k: band(int(k.split(".")[0])))
    out = ["# Near / mid / far C-mean re-score (DAN-64 follow-up, no refit)", "",
           "C mean from one third of the random set's 48 training contexts, ranked by similarity to the target's top "
           "contexts; `all` = the whole random set. Same circuits (lambda %s), same activating and held-out sets." % LAM,
           ""]
    for arm in [a for a in ARMS if a in set(d.arm)]:
        x = d[d.arm == arm]
        complete = x.groupby("seed").third.nunique()
        x = x[x.seed.isin(complete[complete == len(THIRDS)].index)]
        out += ["## trained on %s (%d targets with all four rows)" % (arm, x.seed.nunique()), "", "```"]
        g = x.groupby("third").agg(sim=("sim_med", "median"), passed=("passed", "mean"),
                                   C=("freeN_topk_tk", "median"), Z=("free0_tk", "median"),
                                   A=("freeM_topk_tk", "median"), nec=("phi_sup_blind_tk", "median"))
        out += [g.loc[[t for t in THIRDS if t in g.index]].round(3).to_string(), "```", "", "By band (pass rate):",
                "```", x.pivot_table(index="band", columns="third", values="passed", aggfunc="mean")
                .reindex(columns=list(THIRDS)).round(3).to_string(), "```"]
        p = x.pivot_table(index="seed", columns="third", values="passed", aggfunc="first").dropna()
        rng = np.random.default_rng(0)
        for a_, b_ in (("near", "far"), ("near", "all"), ("far", "all")):
            diff = (p[a_].astype(float) - p[b_].astype(float)).values
            boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
            out.append("- pass %s - %s: %+.3f [%+.3f, %+.3f] (paired bootstrap, n = %d)" % (
                a_, b_, diff.mean(), np.quantile(boot, .025), np.quantile(boot, .975), len(diff)))
        out.append("")
    txt = "\n".join(out)
    (TH / "report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    report() if os.environ.get("REPORT") else main()
