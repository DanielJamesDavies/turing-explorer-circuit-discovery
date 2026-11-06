"""089: print the markdown tables for report.md from out/rows.jsonl (no GPU)."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ARMS = ["prod", "prod_refit", "alpha_prod", "alpha_AC", "alpha_AC_Zhalf", "AC_only"]


def load():
    rows = {}
    for line in open(HERE / "out" / "rows.jsonl"):
        r = json.loads(line)
        if "error" in r or "skip" in r:
            continue
        rows[(r["seed"], r["arm"])] = r
    return rows


def f(x, d=2):
    return "-" if x is None else ("%.*f" % (d, x))


def main():
    rows = load()
    keys = list(dict.fromkeys(s for s, _ in rows))
    arms = [a for a in ARMS if any(a == b for _, b in rows)]
    print("### Held-out strongest (production scorer, post-Top-K read; A / C = sparsity-preserving fill, dense = the trained fill)\n")
    print("| target | arm | n | Z | A | C | nec | pass | A dense | C dense | Z_pre | A_pre | C_pre |"
          " alpha med / p90 / max |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        for a in arms:
            r = rows.get((k, a))
            if r is None:
                continue
            print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s / %s |" % (
                k, a, r.get("ho_n"), f(r.get("ho_free0_tk")), f(r.get("ho_freeM_topk_tk")),
                f(r.get("ho_freeN_topk_tk")), f(r.get("ho_phi_sup_blind_tk")), "Y" if r.get("ho_pass") else "n",
                f(r.get("ho_freeM_dense_tk")), f(r.get("ho_freeN_dense_tk")),
                f(r.get("ho_free0_pre")), f(r.get("ho_freeM_topk_pre")), f(r.get("ho_freeN_topk_pre")),
                f(r.get("alpha_med")), f(r.get("alpha_p90")), f(r.get("alpha_max"))))
    print("\n### Medians over targets: held-out (ho) vs training split (tr)\n")
    print("| arm | n targets | n | ho Z | ho A | ho C | ho nec | ho pass | tr Z | tr A | tr C | tr nec | tr pass |"
          " ho A dense | ho C dense | tr A dense | tr C dense | ho Z_pre | ho A_pre | ho C_pre | alpha med | alpha max |")
    print("|---|" + "---|" * 21)
    for a in arms:
        rs = [rows[(k, a)] for k in keys if (k, a) in rows]

        def med(field):
            v = [r.get(field) for r in rs if r.get(field) is not None]
            return float(np.median(v)) if v else None
        print("| %s | %d | %s | %s | %s | %s | %s | %d/%d | %s | %s | %s | %s | %d/%d | %s | %s | %s | %s |"
              " %s | %s | %s | %s | %s |" % (
            a, len(rs), f(med("ho_n"), 0), f(med("ho_free0_tk")), f(med("ho_freeM_topk_tk")),
            f(med("ho_freeN_topk_tk")), f(med("ho_phi_sup_blind_tk")), sum(bool(r.get("ho_pass")) for r in rs), len(rs),
            f(med("tr_free0_tk")), f(med("tr_freeM_topk_tk")), f(med("tr_freeN_topk_tk")),
            f(med("tr_phi_sup_blind_tk")), sum(bool(r.get("tr_pass")) for r in rs), len(rs),
            f(med("ho_freeM_dense_tk")), f(med("ho_freeN_dense_tk")), f(med("tr_freeM_dense_tk")),
            f(med("tr_freeN_dense_tk")), f(med("ho_free0_pre")), f(med("ho_freeM_topk_pre")), f(med("ho_freeN_topk_pre")), f(med("alpha_med")),
            f(med("alpha_max"))))
    print("\nDense-fill pass (Z, A dense, C dense in [0.8, 1.5], nec >= 0.9; the TRAINED A / C object):")
    for a in arms:
        rs = [rows[(k, a)] for k in keys if (k, a) in rows]

        def dpass(r, p):
            v = [r.get(p + x) for x in ("free0_tk", "freeM_dense_tk", "freeN_dense_tk")]
            nec = r.get(p + "phi_sup_blind_tk")
            return None not in v and nec is not None and all(0.8 <= x <= 1.5 for x in v) and nec >= 0.9
        print("- %s: held-out %d/%d, train %d/%d" % (a, sum(dpass(r, "ho_") for r in rs), len(rs),
                                                   sum(dpass(r, "tr_") for r in rs), len(rs)))
    print("\n### Training split per target (Z / A / C / nec)\n")
    print("| target | " + " | ".join(arms) + " |")
    print("|---|" + "---|" * len(arms))
    for k in keys:
        cells = []
        for a in arms:
            r = rows.get((k, a))
            cells.append("-" if r is None else "%s / %s / %s / %s" % (
                f(r.get("tr_free0_tk")), f(r.get("tr_freeM_topk_tk")), f(r.get("tr_freeN_topk_tk")),
                f(r.get("tr_phi_sup_blind_tk"))))
        print("| %s | %s |" % (k, " | ".join(cells)))


if __name__ == "__main__":
    main()
