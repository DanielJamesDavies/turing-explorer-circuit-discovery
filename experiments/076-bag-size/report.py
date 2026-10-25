"""076 report (CPU): causal truncation results (out/rows.jsonl) and the all-15k magnitude-share concentration from the
explorer bundle (reading_member.share = |alpha| x anchor x decoder norm / circuit total; NOT causal, a write-size proxy).

  PYTHONPATH=src python experiments/076-bag-size/report.py      -> out/report.md
"""
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT76 = Path(os.environ.get("OUT76", str(HERE / "out")))
BUNDLE = HERE.parent.parent / "outputs" / "explorer_bundle" / "2026-09-27_062-full"
BANDS = ["L0-4", "L5-7", "L8-11"]
E62 = HERE.parent / "062-h100-protocol-v1"


def band(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def bundle_concentration():
    con = sqlite3.connect("file:%s?mode=ro" % (BUNDLE / "reading.sqlite"), uri=True)
    ex = sqlite3.connect("file:%s?mode=ro" % (BUNDLE / "explorer.sqlite"), uri=True)
    circ = pd.read_sql("SELECT cid, n, pass FROM circuit", ex).set_index("cid")
    rc = pd.read_sql("SELECT cid, layer, n_members, top25_share FROM reading_circuit", con).set_index("cid")
    m = pd.read_sql("SELECT cid, rank, share FROM reading_member", con)
    m = m.sort_values(["cid", "rank"])
    m["cum"] = m.groupby("cid").share.cumsum()
    need = {}
    for q in (0.5, 0.8, 0.9):
        need["n%d" % int(q * 100)] = m[m.cum < q].groupby("cid").size().reindex(rc.index).fillna(0) + 1
    d = rc.join(pd.DataFrame(need)).join(circ[["pass"]], how="left")
    d["band"] = d.layer.apply(band)
    d["frac80"] = d.n80 / d.n_members
    return d


def main():
    L = ["# 076 — how big is the bag?\n"]
    w = L.append
    # ---- causal truncations
    p = OUT76 / "rows.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in open(p)]
        d = pd.DataFrame([r for r in rows if "error" not in r and "skip" not in r]).drop_duplicates("seed", keep="last")
        ev = {}
        import glob
        for f in glob.glob(str(E62 / "out_full" / "out" / "main" / "eval.shard*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if "error" not in r and r.get("held") == "strong":
                    ev[r["seed"]] = r
        def prod_pass(k):
            r = ev.get(k)
            if r is None or r.get("vacuous_tk"):
                return None
            f = [r.get(x) for x in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")]
            return all(v is not None and 0.8 <= v <= 1.5 for v in f) and (r.get("phi_sup_blind_tk") or 0) >= 0.9
        d["prod_pass"] = d.seed.apply(prod_pass)
        d = d[d.prod_pass.notna()]
        w("## Causal attribution (gradient x alpha on train contexts), %d targets\n" % len(d))
        w("Members needed for 50 / 80 / 90%% of total |attribution| (medians), circuit size, and the share of "
          "attribution mass that is negative (members pushing the target down).\n")
        w("| band | n | circuit size | n50 | n80 | n90 | n80 / size | negative mass |")
        w("|---|---|---|---|---|---|---|---|")
        for b in BANDS + ["all"]:
            x = d if b == "all" else d[d.band == b]
            w("| %s | %d | %.0f | %.0f | %.0f | %.0f | %.2f | %.2f |" % (b, len(x), x.n.median(), x.n50.median(),
                                                                       x.n80.median(), x.n90.median(),
                                                                       (x.n80 / x.n).median(),
                                                                       x.frac_neg_attr.median()))
        w("\n**Truncations** (top-n by |attribution|, coefficients unchanged, no refit; held-out strongest). "
          "Share of targets with Z, A and C all in [0.8, 1.5], among production-PASSING circuits; and the median "
          "smallest grid n at which all three are in band.\n")
        ns = ["1", "2", "5", "10", "20", "50", "100", "200", "500"]
        w("| band | passing n | " + " | ".join("top %s" % n for n in ns) + " | full | median smallest n | median size |")
        w("|---|---|" + "---|" * len(ns) + "---|---|---|")
        for b in BANDS + ["all"]:
            x = d[d.prod_pass.astype(bool)]
            x = x if b == "all" else x[x.band == b]
            cells = []
            for n in ns + ["full"]:
                ok = []
                for _, r in x.iterrows():
                    t = r.trunc
                    key = str(r.n) if n == "full" else n
                    if key not in t:                         # circuit smaller than n: use the full circuit
                        key = str(r.n)
                    v = t.get(key)
                    ok.append(v is not None and all(s is not None and 0.8 <= s <= 1.5 for s in v.values()))
                cells.append("%.0f%%" % (100 * np.mean(ok)) if ok else "-")
            w("| %s | %d | %s | %s | %.0f |" % (b, len(x), " | ".join(cells),
                                               "%.0f" % x.n_min_band.dropna().median() if x.n_min_band.notna().any()
                                               else "-", x.n.median()))
        w("\nMedian Z / A / C faithfulness of top-n truncations, passing circuits, all bands:\n")
        w("| n | Z | A | C |")
        w("|---|---|---|---|")
        x = d[d.prod_pass.astype(bool)]
        for n in ns:
            vals = {f: [] for f in ("Z", "A", "C")}
            for _, r in x.iterrows():
                t = r.trunc.get(n) or r.trunc.get(str(r.n))
                for f in vals:
                    if t and t.get(f) is not None:
                        vals[f].append(t[f])
            w("| %s | %.2f | %.2f | %.2f |" % (n, *[np.median(vals[f]) if vals[f] else float("nan") for f in vals]))
    # ---- bundle concentration (all 15k, magnitude proxy)
    try:
        bc = bundle_concentration()
        w("\n## All ~15k circuits, magnitude-share proxy (explorer bundle)\n")
        w("share = |alpha| x anchor x decoder norm / circuit total. A write-size proxy, not causal; reported for "
          "coverage of the whole run.\n")
        w("| band | n | members | n50 | n80 | n90 | n80 / members | top-25 share |")
        w("|---|---|---|---|---|---|---|---|")
        for b in BANDS + ["all"]:
            x = bc if b == "all" else bc[bc.band == b]
            w("| %s | %d | %.0f | %.0f | %.0f | %.0f | %.2f | %.2f |" % (b, len(x), x.n_members.median(), x.n50.median(),
                                                                       x.n80.median(), x.n90.median(),
                                                                       x.frac80.median(), x.top25_share.median()))
    except Exception as e:  # noqa: BLE001
        w("\n(bundle concentration unavailable: %s)\n" % e)
    txt = "\n".join(L)
    (OUT76 / "report.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
