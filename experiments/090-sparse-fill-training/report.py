"""090: rebuild out/tables.md from out/rows.jsonl (no GPU).  python experiments/090-sparse-fill-training/report.py"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "out"
ARMS = ["prod", "dense", "sp_hard", "sp_ste", "curriculum"]
GROUPS = [("all", None), ("fail89", "fail89"), ("deep attn", "attn"), ("controls", "control")]
FIT_ARMS = ["sp_hard", "sp_ste", "curriculum"]


def load(path):
    rows = {}
    if not path.exists():
        return rows
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if "error" in r or "skip" in r:
            continue
        rows[(r["seed"], r["arm"])] = r
    return rows


def f(x, d=2):
    return "-" if x is None else ("%.*f" % (d, x))


def med(rs, field):
    v = [r.get(field) for r in rs if r.get(field) is not None]
    return float(np.median(v)) if v else None


def main():
    rows = load(OUT / "rows.jsonl")
    keys = list(dict.fromkeys(s for s, _ in rows))
    arms = [a for a in ARMS if any(a == b for _, b in rows)]
    grp = {k: r.get("group") for (k, _), r in rows.items()}
    L = []
    p = L.append

    p("# 090: sparsity-preserving fill in WCM's A / C training terms\n")
    p("Held-out strongest 16 unless marked tr (the 089 train probe: strongest training rows). Production scorer, "
      "post-Top-K read. A / C = the RULE's sparsity-preserving fill; A / C dense = the dense mean fill (production's "
      "training fill). pass = rule pass (Z, A, C in [0.8, 1.5], nec >= 0.9); dpass = the same with dense A / C. "
      "_pre = pre-activation read. evict = share of active members at the held-out anchors that a LITERAL Top-K over "
      "members + means would drop (the scorer never drops a member); >K = share of (site, anchor) pairs with more than "
      "K active members (no mean fill there).\n")

    p("## Medians per arm and group\n")
    p("| group | arm | targets | n | Z | A | C | nec | pass | A dense | C dense | dpass | tr Z | tr A | tr C | tr pass |"
      " Z_pre | A_pre | C_pre | alpha med | alpha p90 | alpha max | evict | >K |")
    p("|---|" + "---|" * 24)
    for gname, g in GROUPS:
        for a in arms:
            rs = [rows[(k, a)] for k in keys if (k, a) in rows and (g is None or grp.get(k) == g)]
            if not rs:
                continue
            p("| %s | %s | %d | %s | %s | %s | %s | %s | %d/%d | %s | %s | %d/%d | %s | %s | %s | %d/%d | %s | %s | %s |"
              " %s | %s | %s | %s | %s |" % (
                  gname, a, len(rs), f(med(rs, "ho_n"), 0), f(med(rs, "ho_free0_tk")), f(med(rs, "ho_freeM_topk_tk")),
                  f(med(rs, "ho_freeN_topk_tk")), f(med(rs, "ho_phi_sup_blind_tk")),
                  sum(bool(r.get("ho_pass")) for r in rs), len(rs),
                  f(med(rs, "ho_freeM_dense_tk")), f(med(rs, "ho_freeN_dense_tk")),
                  sum(bool(r.get("ho_dpass")) for r in rs), len(rs),
                  f(med(rs, "tr_free0_tk")), f(med(rs, "tr_freeM_topk_tk")), f(med(rs, "tr_freeN_topk_tk")),
                  sum(bool(r.get("tr_pass")) for r in rs), len(rs),
                  f(med(rs, "ho_free0_pre")), f(med(rs, "ho_freeM_topk_pre")), f(med(rs, "ho_freeN_topk_pre")),
                  f(med(rs, "alpha_med")), f(med(rs, "alpha_p90")), f(med(rs, "alpha_max")),
                  f(med(rs, "ev_lit_evict_share"), 3), f(med(rs, "ev_over_k_share"), 3)))

    p("\n## Paired changes against `dense` (the in-run production refit)\n")
    p("Medians of (arm - dense) per target; n ratio = median of n_arm / n_dense; pass +/- = rule passes gained / lost; "
      "A up = targets whose rule A rose by > 0.05.\n")
    p("| group | arm | pairs | dZ | dA | dC | dnec | dA dense | dC dense | dZ_pre | dA_pre | dC_pre | tr dA | tr dC |"
      " n ratio | pass + / - | dpass + / - | A up | Z drop > 0.1 |")
    p("|---|" + "---|" * 18)
    for gname, g in GROUPS:
        for a in FIT_ARMS:
            pairs = [(rows[(k, a)], rows[(k, "dense")]) for k in keys
                     if (k, a) in rows and (k, "dense") in rows and (g is None or grp.get(k) == g)]
            if not pairs:
                continue

            def dm(field):
                v = [x.get(field) - y.get(field) for x, y in pairs if x.get(field) is not None and y.get(field) is not None]
                return float(np.median(v)) if v else None
            ratio = [x["ho_n"] / y["ho_n"] for x, y in pairs if x.get("ho_n") and y.get("ho_n")]
            p("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d / %d | %d / %d | %d | %d |" % (
                gname, a, len(pairs), f(dm("ho_free0_tk")), f(dm("ho_freeM_topk_tk")), f(dm("ho_freeN_topk_tk")),
                f(dm("ho_phi_sup_blind_tk")), f(dm("ho_freeM_dense_tk")), f(dm("ho_freeN_dense_tk")),
                f(dm("ho_free0_pre")), f(dm("ho_freeM_topk_pre")), f(dm("ho_freeN_topk_pre")),
                f(dm("tr_freeM_topk_tk")), f(dm("tr_freeN_topk_tk")),
                f(float(np.median(ratio)) if ratio else None),
                sum(bool(x.get("ho_pass")) and not y.get("ho_pass") for x, y in pairs),
                sum(bool(y.get("ho_pass")) and not x.get("ho_pass") for x, y in pairs),
                sum(bool(x.get("ho_dpass")) and not y.get("ho_dpass") for x, y in pairs),
                sum(bool(y.get("ho_dpass")) and not x.get("ho_dpass") for x, y in pairs),
                sum((x.get("ho_freeM_topk_tk") or -9) - (y.get("ho_freeM_topk_tk") or -9) > 0.05 for x, y in pairs),
                sum((y.get("ho_free0_tk") or 0) - (x.get("ho_free0_tk") or 0) > 0.1 for x, y in pairs)))

    p("\n## Deep attention: near-threshold (clean rank >= 64) vs not\n")
    p("| sub | arm | targets | n | Z | A | C | pass | A dense | C dense |")
    p("|---|" + "---|" * 9)
    for sub in ("near", "far"):
        for a in arms:
            rs = [rows[(k, a)] for k in keys if (k, a) in rows and rows[(k, a)].get("attn_sub") == sub]
            if rs:
                p("| %s | %s | %d | %s | %s | %s | %s | %d/%d | %s | %s |" % (
                    sub, a, len(rs), f(med(rs, "ho_n"), 0), f(med(rs, "ho_free0_tk")), f(med(rs, "ho_freeM_topk_tk")),
                    f(med(rs, "ho_freeN_topk_tk")), sum(bool(r.get("ho_pass")) for r in rs), len(rs),
                    f(med(rs, "ho_freeM_dense_tk")), f(med(rs, "ho_freeN_dense_tk"))))

    p("\n## Per target (held-out)\n")
    p("| target | group | arm | n | Z | A | C | nec | pass | A dense | C dense | dpass | tr Z / A / C | Z_pre | A_pre |"
      " C_pre | alpha med / p90 / max | evict | >K |")
    p("|---|" + "---|" * 18)
    for k in keys:
        for a in arms:
            r = rows.get((k, a))
            if r is None:
                continue
            g = r.get("group") or ""
            if r.get("attn_sub"):
                g += " (%s)" % r["attn_sub"]
            p("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s / %s | %s | %s | %s |"
              " %s / %s / %s | %s | %s |" % (
                  k, g, a, r.get("ho_n"), f(r.get("ho_free0_tk")), f(r.get("ho_freeM_topk_tk")),
                  f(r.get("ho_freeN_topk_tk")), f(r.get("ho_phi_sup_blind_tk")), "Y" if r.get("ho_pass") else "n",
                  f(r.get("ho_freeM_dense_tk")), f(r.get("ho_freeN_dense_tk")), "Y" if r.get("ho_dpass") else "n",
                  f(r.get("tr_free0_tk")), f(r.get("tr_freeM_topk_tk")), f(r.get("tr_freeN_topk_tk")),
                  f(r.get("ho_free0_pre")), f(r.get("ho_freeM_topk_pre")), f(r.get("ho_freeN_topk_pre")),
                  f(r.get("alpha_med")), f(r.get("alpha_p90")), f(r.get("alpha_max")),
                  f(r.get("ev_lit_evict_share"), 3), f(r.get("ev_over_k_share"), 3)))

    r89 = load(HERE.parent / "089-objective-tradeoff" / "out" / "rows.jsonl")
    both = [k for k in keys if (k, "dense") in rows and (k, "prod_refit") in r89]
    if both:
        p("\n## No-op check: 090 `dense` vs 089 `prod_refit` (same settings, same GPU, separate processes)\n")
        p("| target | n 089 / 090 | Z 089 / 090 | A 089 / 090 | C 089 / 090 |")
        p("|---|---|---|---|---|")
        for k in both:
            x, y = r89[(k, "prod_refit")], rows[(k, "dense")]
            p("| %s | %s / %s | %s / %s | %s / %s | %s / %s |" % (
                k, x.get("ho_n"), y.get("ho_n"), f(x.get("ho_free0_tk")), f(y.get("ho_free0_tk")),
                f(x.get("ho_freeM_topk_tk")), f(y.get("ho_freeM_topk_tk")), f(x.get("ho_freeN_topk_tk")),
                f(y.get("ho_freeN_topk_tk"))))

    fits = [r for (k, a), r in rows.items() if a != "prod" and r.get("t_fit")]
    if fits:
        p("\n## Fit cost\n")
        p("| arm | median fit s | max fit peak MB |")
        p("|---|---|---|")
        for a in arms[1:]:
            rs = [r for r in fits if r["arm"] == a]
            if rs:
                p("| %s | %s | %s |" % (a, f(med(rs, "t_fit"), 0), max((r.get("fit_peak_mb") or 0) for r in rs)))
    text = "\n".join(L) + "\n"
    (OUT / "tables.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
