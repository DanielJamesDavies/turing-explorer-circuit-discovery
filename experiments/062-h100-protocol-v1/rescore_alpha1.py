"""DAN-20: the same nodes at natural scale. Re-score full-run circuits with every coefficient set to 1 (alpha = 1),
on the held-out strongest contexts, with the same scorer and context injection as the production eval. The fitted-alpha
scores are the run's own eval rows, so the two are paired per target.

Sample: stratified, N_PER (50) targets per (layer, site kind), drawn with a fixed seed from the targets that have a
circuit. Resumable: one jsonl row per target in OUT_A1/alpha1.jsonl.

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src python experiments/062-h100-protocol-v1/rescore_alpha1.py
  env: N_PER (50)  OUT_A1 (default experiments/062-h100-protocol-v1/out_alpha1)
"""
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).parent
os.environ.setdefault("OUT", str(HERE / "out_full" / "out"))
sys.path.insert(0, str(HERE))
import driver  # noqa: E402  (reads OUT at import)

N_PER = int(os.environ.get("N_PER", 50))
OUT_A1 = Path(os.environ.get("OUT_A1", str(HERE / "out_alpha1")))


def sample():
    have = sorted(p.stem for p in (driver.OUT / "main" / "circuits").glob("*.pt"))
    cells = defaultdict(list)
    for k in have:
        l, kind, _ = k.split(".")
        cells[(int(l), kind)].append(k)
    rng = random.Random(20261002)
    out = []
    for cell in sorted(cells):
        ks = cells[cell]
        out += rng.sample(ks, min(N_PER, len(ks)))
    return out


def ones(c):
    """Every member at alpha = 1, in the scorer's override format {(layer, kind): {index: alpha}}."""
    out = defaultdict(dict)
    for n in c.nodes.values():
        md = n.metadata
        if md.get("role") == "seed":
            continue
        f = md["feature_id"]
        out[(int(f.layer), str(f.kind))][int(f.index)] = 1.0
    return dict(out)


def main():
    OUT_A1.mkdir(parents=True, exist_ok=True)
    path = OUT_A1 / "alpha1.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                done.add(json.loads(line)["seed"])
            except Exception:  # noqa: BLE001
                pass
    keys = [k for k in sample() if k not in done]
    print("alpha = 1 re-scoring: %d targets to do (%d done)" % (len(keys), len(done)), flush=True)
    R = driver.Runner()
    fh = open(path, "a")
    t0 = time.time()
    for n, key in enumerate(keys):
        row = dict(seed=key)
        try:
            c = torch.load(driver.OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
            if c is None:
                row["skip"] = "rejected"
            else:
                rec = R.contexts(key)
                R.H.patch_eval_contexts(R.G, rec, "strong")
                r = R.V.score_circuit(c, {}, override_alphas=ones(c), skip_roles=True)
                row.update({k: v for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None})
                row["held"] = "strong"
        except Exception as e:  # noqa: BLE001
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        fh.write(json.dumps(row) + "\n"); fh.flush()
        if (n + 1) % 25 == 0:
            el = time.time() - t0
            print("  %d/%d  %.1fs per target, ~%.0f min left" % (n + 1, len(keys), el / (n + 1),
                                                                el / (n + 1) * (len(keys) - n - 1) / 60), flush=True)
    print("DONE", flush=True)


def report():
    """REPORT=1: pair each alpha = 1 row with the run's fitted-alpha eval row (held-out strongest) and summarise by
    depth band and site kind: pass rate under the DAN-8 rule, median Z / A / C, necessity. No GPU."""
    import glob
    import pandas as pd
    HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]

    def passes(df):
        inb = ((df[HEAD] >= 0.8) & (df[HEAD] <= 1.5)).all(axis=1)
        return inb & (df.phi_sup_blind_tk >= 0.9)

    a1 = pd.DataFrame([json.loads(l) for l in open(OUT_A1 / "alpha1.jsonl")])
    a1 = a1[a1.get("error", pd.Series(index=a1.index, dtype=object)).isna()]
    a1 = a1[a1.get("skip", pd.Series(index=a1.index, dtype=object)).isna()].drop_duplicates("seed", keep="last")
    ev = []
    for f in glob.glob(str(driver.OUT / "main" / "eval.shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("held") == "strong" and "error" not in r and r.get("seed") in set(a1.seed):
                ev.append(r)
    fit = pd.DataFrame(ev).drop_duplicates("seed", keep="last")
    vac = lambda d: d.get("vacuous_tk", pd.Series(False, index=d.index)).fillna(False).astype(bool)
    m = a1.set_index("seed")[HEAD + ["phi_sup_blind_tk", "vacuous_tk"]].join(
        fit.set_index("seed")[HEAD + ["phi_sup_blind_tk", "vacuous_tk", "n"]], lsuffix="_a1", rsuffix="_fit", how="inner")
    m = m[~(m.vacuous_tk_a1.fillna(False).astype(bool) | m.vacuous_tk_fit.fillna(False).astype(bool))]
    get = lambda sfx: m[[h + sfx for h in HEAD] + ["phi_sup_blind_tk" + sfx]].rename(columns=lambda c: c.replace(sfx, ""))
    m["pass_fit"], m["pass_a1"] = passes(get("_fit")), passes(get("_a1"))
    m["layer"] = [int(k.split(".")[0]) for k in m.index]
    m["kind"] = [k.split(".")[1] for k in m.index]
    m["band"] = m.layer.map(lambda l: "L0-4" if l <= 4 else ("L5-7" if l <= 7 else "L8-11"))
    print("paired targets: %d" % len(m))
    for by in ("band", "kind"):
        g = m.groupby(by).agg(n=("n", "median"), pass_fit=("pass_fit", "mean"), pass_a1=("pass_a1", "mean"),
                              Z_fit=("free0_tk_fit", "median"), Z_a1=("free0_tk_a1", "median"),
                              A_fit=("freeM_topk_tk_fit", "median"), A_a1=("freeM_topk_tk_a1", "median"),
                              C_fit=("freeN_topk_tk_fit", "median"), C_a1=("freeN_topk_tk_a1", "median"))
        print(g.round(2).to_string())
    print("all: pass fitted %.1f%% -> alpha = 1 %.1f%%; median Z %.2f -> %.2f, A %.2f -> %.2f, C %.2f -> %.2f" % (
        100 * m.pass_fit.mean(), 100 * m.pass_a1.mean(), m.free0_tk_fit.median(), m.free0_tk_a1.median(),
        m.freeM_topk_tk_fit.median(), m.freeM_topk_tk_a1.median(), m.freeN_topk_tk_fit.median(),
        m.freeN_topk_tk_a1.median()))


if __name__ == "__main__":
    report() if os.environ.get("REPORT") else main()
