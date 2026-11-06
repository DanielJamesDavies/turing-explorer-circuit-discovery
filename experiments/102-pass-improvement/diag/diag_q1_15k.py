"""102 diag: Q1 / Q2 / Q3 cross-checks on the v1 15k (062 out_full; v1 circuits, LEAKY held-out sets). CPU.

(a) Q1 with power: Z on held-out vs on the 48 strongest-train rows (free0_tr_tk), by band x kind.
(b) Top-K cut at scale: failing terms whose pre-activation read passes (Z on the 0 baseline = free0_raw_pre /
    a_pos_pre; A / C = freeM_topk_pre / freeN_topk_pre; necessity = (a_pos_pre - phi_sup_blind_raw_pre) / a_pos_pre),
    and pass rate by rank_clean bin.
(c) Is the diversity effect generalisation (held-out anchors unseen in training) or target difficulty? Pass rate and
    the held-out - train Z gap by anchor diversity x whether every held-out anchor token was seen in training.
Writes out/q1_15k.md.
"""
import numpy as np
import pandas as pd

from diag_common import DIAG, E62, band_of, fnum, inband, load_062_eval, md_table, prod_pass

OUT = DIAG / "out"


def main():
    ev = load_062_eval()
    ctx = pd.read_csv(OUT / "ctxfeat_15k.csv").set_index("seed")
    tg = pd.read_csv(E62 / "out_full" / "out" / "targets.csv").set_index("seed")
    rows = []
    for s, r in ev.items():
        P = prod_pass(r)
        if P is None or s not in ctx.index:
            continue
        L = int(s.split(".")[0])
        ap = fnum(r.get("a_pos_pre"))
        zr, nr = fnum(r.get("free0_raw_pre")), fnum(r.get("phi_sup_blind_raw_pre"))
        z_pre0 = zr / ap if (ap and ap > 1e-9 and zr is not None) else None
        nec_pre0 = (ap - nr) / ap if (ap and ap > 1e-9 and nr is not None) else None
        rows.append(dict(seed=s, band=band_of(L), kind=s.split(".")[1], P=P, Z=fnum(r.get("free0_tk")),
                         Ztr=fnum(r.get("free0_tr_tk")), A=fnum(r.get("freeM_topk_tk")), C=fnum(r.get("freeN_topk_tk")),
                         nec=fnum(r.get("phi_sup_blind_tk")), Zp=z_pre0, Ap=fnum(r.get("freeM_topk_pre")),
                         Cp=fnum(r.get("freeN_topk_pre")), necp=nec_pre0, rank=tg.rank_clean.get(s, np.nan),
                         div=float(ctx.at[s, "tr_anchor_distinct"]), seen=float(ctx.at[s, "ho_anchor_seen"]),
                         dup=float(ctx.at[s, "ho_dup16"]), top1=float(ctx.at[s, "tr_anchor_top1"])))
    d = pd.DataFrame(rows)
    md = ["# v1 15k cross-checks (diag_q1_15k.py; v1 circuits, floor 0.75, LEAKY held-out sets)\n",
          "n = %d targets, production pass %d (%.1f%%).\n" % (len(d), d.P.sum(), 100 * d.P.mean())]
    # (a)
    trs = []
    for (b, k), g in d.groupby(["band", "kind"]):
        zf = g[~g.Z.apply(inband)]
        trs.append(["%s/%s" % (b, k), len(g), "%.0f%%" % (100 * g.P.mean()), len(zf),
                    "%.0f%%" % (100 * (~zf.Ztr.apply(inband)).mean()) if len(zf) else "-",
                    "%.3f" % (g.Z - g.Ztr).median(), "%.3f" % (zf.Z - zf.Ztr).median() if len(zf) else "-"])
    zf = d[~d.Z.apply(inband)]
    trs.append(["all", len(d), "%.0f%%" % (100 * d.P.mean()), len(zf), "%.0f%%" % (100 * (~zf.Ztr.apply(inband)).mean()),
                "%.3f" % (d.Z - d.Ztr).median(), "%.3f" % (zf.Z - zf.Ztr).median()])
    md.append("## (a) Z held-out vs train (48 strongest-train rows)\n")
    md.append(md_table(["cell", "n", "prod pass", "Z fails held-out", "... of which fail train too",
                        "median Z ho - tr (all)", "median Z ho - tr (Z failures)"], trs))
    # (b)
    md.append("\n## (b) Top-K cut: failing terms whose pre-activation read passes\n")
    trs = []
    for t, tp, isnec in (("Z", "Zp", False), ("A", "Ap", False), ("C", "Cp", False), ("nec", "necp", True)):
        ok = (lambda v: v is not None and v == v and v >= 0.9) if isnec else inband
        f = d[~d[t].apply(ok)]
        cut = f[tp].apply(ok)
        trs.append([t, len(f), int(cut.sum()), "%.0f%%" % (100 * cut.mean()) if len(f) else "-"])
    fails = d[~d.P]
    allpre = fails.apply(lambda r: inband(r.Zp) and inband(r.Ap) and inband(r.Cp) and (r.necp or 0) >= 0.9, axis=1)
    trs.append(["all production failures passing every pre read", len(fails), int(allpre.sum()),
                "%.0f%%" % (100 * allpre.mean())])
    md.append(md_table(["term", "n failing post-Top-K", "pre read passes", "share"], trs))
    trs = []
    for lo, hi in ((1, 2), (2, 5), (5, 16), (16, 32), (32, 64), (64, 129)):
        g = d[(d["rank"] >= lo) & (d["rank"] < hi)]
        zc = g[~g.Z.apply(inband) & g.Zp.apply(inband)]
        trs.append(["[%d, %d)" % (lo, hi), len(g), "%.0f%%" % (100 * g.P.mean()) if len(g) else "-",
                    "%.0f%%" % (100 * len(zc) / len(g)) if len(g) else "-"])
    md.append("\nPass rate and Z-cut rate by clean rank of the target (K = 128):\n")
    md.append(md_table(["rank_clean", "n", "prod pass", "Z lost at the cut (post fails, pre passes)"], trs))
    # (c)
    md.append("\n## (c) diversity vs held-out novelty\n")
    md.append("div = share of distinct anchor tokens among the 48 training contexts (tertiles); seen = every held-out"
              " anchor token occurs among training anchors. Z gap = median Z held-out - Z train.\n")
    q = np.quantile(d["div"], [1 / 3, 2 / 3])
    d["divt"] = np.digitize(d["div"], q, right=True)
    trs = []
    for t in range(3):
        for seen in (True, False):
            g = d[(d.divt == t) & ((d.seen >= 0.999) == seen)]
            if len(g) == 0:
                continue
            trs.append(["%s (%.2f-%.2f)" % (["low", "mid", "high"][t], g["div"].min(), g["div"].max()),
                        "all seen" if seen else "some unseen", len(g), "%.0f%%" % (100 * g.P.mean()),
                        "%.3f" % (g.Z - g.Ztr).median(), "%.0f%%" % (100 * (~g.Z.apply(inband)).mean()),
                        "%.0f%%" % (100 * (~g.Ztr.apply(inband)).mean())])
    md.append(md_table(["anchor diversity", "held-out anchors", "n", "prod pass", "Z gap", "Z fail (held-out)",
                        "Z fail (train)"], trs))
    # same within band x kind: logistic of P on div + seen + cell dummies
    d["cell"] = d.band + "/" + d.kind
    d["unseen"] = (d.seen < 0.999).astype(float)
    d["logrank"] = np.log2(d["rank"].clip(lower=1))
    names = ["div", "unseen", "dup", "logrank"]
    cells = sorted(d.cell.unique())
    X = np.column_stack([np.ones(len(d))] + [(d.cell == c).astype(float).values for c in cells[1:]]
                        + [d[n].values.astype(float) for n in names])
    beta, se = newton_logit(X, d.P.astype(float).values)
    from scipy.stats import norm
    off = len(cells)
    md.append("\nLogit P ~ cell + div + unseen + ho_dup16 + log2 rank_clean (unpenalised, n %d):\n" % len(d))
    trs = [[n, beta[off + i], se[off + i], 2 * norm.sf(abs(beta[off + i] / se[off + i]))] for i, n in enumerate(names)]
    md.append(md_table(["term", "coef", "se", "p"], trs))
    (OUT / "q1_15k.md").write_text("\n".join(md))
    print("\n".join(md))


def newton_logit(X, y, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ b))
    H = X.T @ (X * (p * (1 - p))[:, None])
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


if __name__ == "__main__":
    main()
