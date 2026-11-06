"""102 diag Q4: how much do pass / fail flip from fit noise alone? CPU (loads circuits, no model).

Noise-only refit pairs available on disk:
  (a) 062 v1 (H100 15k run) vs 093's v1 arm (RTX, same records, same v1 protocol: lambda 1e-3, gamma 0.25,
      rank-keep 3e-3, 400 steps): 72 targets. Different GPU / process / batch shapes -> bf16 kernel noise.
  (b) 092 t1_trunc and t2_comb vs 090 sp_ste (v2-equivalent): both mathematically identical to the reference (092
      README), 8 targets x 2.
(100's and 095's re-fits are bit-identical on the same GPU: zero noise, nothing to learn.)
Then a simulation: perturb every v2CR term on 095's 72 by the measured per-term noise and count production / resample
pass flips; this is the noise band for a single 72-target arm comparison.
Writes out/q4.md.
"""
import numpy as np
import torch

from diag_common import DIAG, E62, E93, EXP, fnum, inband, load_062_eval, load_093, load_095, load_jsonl, md_table, \
    prod_pass

OUT = DIAG / "out"
RNG = np.random.default_rng(20261031)
TERMS = ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk")


def members(path):
    c = torch.load(path, weights_only=False)
    return {str(n.metadata.get("feature_id")): float(n.metadata.get("amplitude", 1.0))
            for n in c.nodes.values() if n.metadata.get("role") != "seed"}


def pair_stats(pairs):
    """pairs: list of (row_a, row_b, mem_a or None, mem_b or None)."""
    flips = 0
    n = 0
    d = {t: [] for t in TERMS}
    jac, sz = [], []
    for a, b, ma, mb in pairs:
        pa, pb = prod_pass(a), prod_pass(b)
        if pa is None or pb is None:
            continue
        n += 1
        flips += pa != pb
        for t in TERMS:
            x, y = fnum(a.get(t)), fnum(b.get(t))
            if x is not None and y is not None and abs(x) < 5 and abs(y) < 5:
                d[t].append(x - y)
        if ma is not None and mb is not None:
            ka, kb = set(ma), set(mb)
            jac.append(len(ka & kb) / max(len(ka | kb), 1))
            sz.append(len(ka) / max(len(kb), 1))
    return n, flips, d, jac, sz


def main():
    md = ["# Q4: fit-noise flips (diag_q4.py)\n"]
    # (a)
    ev = load_062_eval()
    r93 = load_093()["v1"]
    pairs = []
    for s, b in r93.items():
        a = ev.get(s)
        if a is None:
            continue
        pa = E62 / "out_full" / "out" / "main" / "circuits" / ("%s.pt" % s)
        pb = E93 / "out" / "circuits" / "v1" / ("%s.pt" % s)
        ma = members(pa) if pa.exists() else None
        mb = members(pb) if pb.exists() else None
        pairs.append((a, b, ma, mb))
    na, fa, da, ja, sa = pair_stats(pairs)
    pp_a = (sum(prod_pass(a) for a, _, _, _ in pairs if prod_pass(a) is not None),
            sum(prod_pass(b) for _, b, _, _ in pairs if prod_pass(b) is not None))
    # (b)
    ref = {r["seed"]: r for r in load_jsonl(EXP / "090-sparse-fill-training" / "out" / "rows.jsonl")
           if r.get("arm") == "sp_ste"} if (EXP / "090-sparse-fill-training" / "out" / "rows.jsonl").exists() else {}
    pairs_b = []
    for r in load_jsonl(EXP / "092-v2-speedups" / "out" / "rows.jsonl"):
        if r.get("arm") not in ("t1_trunc", "t2_comb"):
            continue
        a = {t: r.get("ho_" + t) for t in TERMS}
        a["vacuous_tk"] = r.get("ho_vacuous_tk")
        b = {t: (None if r.get("ho_" + t) is None or r.get("d_" + t) is None else r["ho_" + t] - r["d_" + t])
             for t in TERMS}
        b["vacuous_tk"] = r.get("ho_vacuous_tk")
        pairs_b.append((a, b, None, None))
        if r.get("jaccard") is not None:
            pass
    nb, fb, db, _, _ = pair_stats(pairs_b)
    jb = [r["jaccard"] for r in load_jsonl(EXP / "092-v2-speedups" / "out" / "rows.jsonl")
          if r.get("arm") in ("t1_trunc", "t2_comb") and r.get("jaccard") is not None]
    trs = [["(a) 062 v1 vs 093 v1 (72 targets, different GPU)", na, fa, "%d -> %d" % pp_a,
            "%.2f [%.2f, %.2f]" % tuple(np.percentile(ja, [50, 25, 75])) if ja else "-",
            "%.2f [%.2f, %.2f]" % tuple(np.percentile(sa, [50, 25, 75])) if sa else "-"]
           + ["%.3f" % np.median(np.abs(da[t])) for t in TERMS],
           ["(b) 092 t1/t2 vs 090 sp_ste (8 targets x 2, same GPU, kernel shapes)", nb, fb, "-",
            "%.2f [%.2f, %.2f]" % tuple(np.percentile(jb, [50, 25, 75])) if jb else "-", "-"]
           + ["%.3f" % np.median(np.abs(db[t])) if db[t] else "-" for t in TERMS]]
    md.append(md_table(["pair", "n", "production pass flips (floor 0.75)", "passes a -> b", "member Jaccard",
                        "size ratio a/b", "median |dZ|", "median |dA|", "median |dC|", "median |d nec|"], trs))
    # per-term noise sd (robust: 1.4826 * MAD of differences / sqrt 2 = per-fit sd)
    sd = {}
    for t in TERMS:
        x = np.array(da[t] + db[t])
        sd[t] = 1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(2)
    md.append("\nPer-fit noise SD per term (robust, from the pooled differences / sqrt 2): " +
              ", ".join("%s %.3f" % (t, sd[t]) for t in TERMS))
    # the flips are heavy-tailed: show the share of |d| > 0.1 and > 0.25
    md.append("\nShare of pairs with |d| > 0.10 / > 0.25 (pooled a + b): " + ", ".join(
        "%s %.0f%% / %.0f%%" % (t, 100 * np.mean(np.abs(np.array(da[t] + db[t])) > 0.10),
                                100 * np.mean(np.abs(np.array(da[t] + db[t])) > 0.25)) for t in TERMS))
    # (c) simulation on 095 v2CR: resample the observed pooled differences (empirical, heavy tails) / sqrt 2
    by = load_095()
    pool = {t: np.array(da[t] + db[t]) / np.sqrt(2) for t in TERMS}
    rs_pool = np.concatenate([pool["freeM_topk_tk"], pool["freeN_topk_tk"]])   # donor terms: no refit pairs; proxy
    sims = []

    def noisy(r):
        p = dict(r["prod"])
        for t in TERMS:
            if p.get(t) is not None:
                p[t] = p[t] + RNG.choice(pool[t])
        P = prod_pass(p)
        rsA = r.get("rs_A") + RNG.choice(rs_pool) if r.get("rs_A") is not None else None
        rsC = r.get("rs_C") + RNG.choice(rs_pool) if r.get("rs_C") is not None else None
        return bool(P), bool(P and inband(rsA) and inband(rsC))

    for arm in ("v2", "v2CR"):
        base = by[arm]
        fl_p, fl_r, net_p, net_r = [], [], [], []
        for _ in range(2000):
            cp = cr = dp = dr = 0
            for s, r in base.items():
                (p1, r1), (p2, r2) = noisy(r), noisy(r)       # two independent refits around the observed fit
                cp += p1 != p2
                cr += r1 != r2
                dp += int(p2) - int(p1)
                dr += int(r2) - int(r1)
            fl_p.append(cp); fl_r.append(cr); net_p.append(dp); net_r.append(dr)
        sims.append([arm, "%.1f [%d, %d]" % (np.mean(fl_p), *np.percentile(fl_p, [5, 95])),
                     "[%d, %d]" % tuple(np.percentile(net_p, [2.5, 97.5])),
                     "%.1f [%d, %d]" % (np.mean(fl_r), *np.percentile(fl_r, [5, 95])),
                     "[%d, %d]" % tuple(np.percentile(net_r, [2.5, 97.5]))])
    md.append("\n**Two simulated noise-only refits of the same arm on 095's 72** (every term of each refit = observed +"
              " a draw from the measured refit differences / sqrt 2, i.e. per-fit noise; donor terms use the A / C"
              " noise as a proxy: no donor refit pairs exist). 2,000 draws. Discordant targets mean [5th, 95th] and"
              " the 95% range of the net pass difference = the noise band any arm-vs-arm comparison on 72 must clear."
              " Check against the empirical pair (a): 10 discordant of 72.\n")
    md.append(md_table(["arm", "production discordant", "production net diff 95%", "resample discordant",
                        "resample net diff 95%"], sims))
    (OUT / "q4.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
