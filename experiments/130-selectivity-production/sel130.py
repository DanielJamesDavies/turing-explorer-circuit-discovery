"""130: selectivity reads of the 2026-11-07 production circuits (WCM v4; TuringLLM, Pythia-70m, GPT-2 small).

Analysis only (CPU, reads 109's unpacked archives; no model is loaded). Two reads that every production row already
carries but the pass rule ignores:

  silent_firing  (src/eval/donor_test.py:349-351; protocol.md §0.5 donor test, :289-290, :418)
      Bases include the target's 16 held-out contrast rows (close selector, verified post-Top-K silent at every
      position). Members keep their LIVE value on that silent base (x alpha in variant "fit"), non-members take a donor's
      natural code at the same position (pool A = 48 strongest-train activating rows, C = 16 held-out contrast rows,
      R = 512 random corpus windows), error live. silent_firing = share of the 16 x 64 silent-base positions where the
      target's post-Top-K read is > 0, per draw (8 draws). Here: mean over the 8 draws per pool.
      Reference values: full model on these bases = 0 by construction (verified silent). The "empty" circuit (every
      latent from the donor) is the background reference: how often the donor background alone switches the target
      on. A selective circuit keeps fit ~0 even when the background is activating (pool A).

  phi_ind = phi_cf_alpha_blind_tk  (protocol.md §0.9, :392-440, :750)
      On the 16 held-out contrast rows, every member is SET at every position to alpha x its pin (mean code at the
      anchors of the 48 strongest-train rows); the rest of the stream is live. Score (A-_int - A-)/(a_pos - A-) at the
      contrast anchors. Full-model reference: 0 = target stays at its natural (silent) value; 1 = the circuit's
      activating-state members turn the target on to its natural activating level.

Population = 109's headline: main eval rows, held-out strongest, error rows dropped, last row per (seed, held),
merge.vacuous excluded, pass = merge.passes (lib109.main_eval). The 124 paraphrase re-score is a subset check and
is NOT applied by the headline, so not here either.

Baselines present in the archives: harder nulls (084: random / anchor-active / contribution-matched member sets
fitted like WCM, and alpha = 1 re-scoring of the WCM members) carry phi_ind but no donor test; Pythia's unweighted
circuits (donor_unweighted) carry silent_firing.

  python -X utf8 experiments/130-selectivity-production/sel130.py
Writes experiments/130-selectivity-production/out/{summary.csv, nulls_ind.csv, worst.csv, per_target.csv, report.txt}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "109-pod-run-analysis"))
import lib109 as L  # noqa: E402

OUT = HERE / "out"
POOLS = ("A", "C", "R")
VARS = ("empty", "fit", "a1", "pinned")


def donor_frame(dirs):
    """Last good row per (seed, arm) of donor rows.shard*.jsonl, with mean-over-draws silent_firing per variant/pool."""
    last = {}
    for d in dirs:
        for p in sorted(Path(d).glob("rows.shard*.jsonl")) + [Path(d) / "rows.jsonl"]:
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if "error" in r or r.get("skip"):
                        continue
                    sf = r.get("silent_firing") or {}
                    row = dict(seed=r["seed"], arm=r.get("arm"), n_silent=r.get("n_silent"))
                    for v in VARS:
                        for ds in POOLS:
                            xs = [x for x in (sf.get(v) or {}).get(ds) or [] if x is not None]
                            row["sf_%s_%s" % (v, ds)] = float(np.mean(xs)) if xs else np.nan
                    for ds in POOLS:
                        row["donor_fires_" + ds] = (r.get("donor_fires_frac") or {}).get(ds)
                        row["rec_fit_" + ds] = ((r.get("recovery") or {}).get("fit") or {}).get(ds)
                    prod = r.get("prod") or {}
                    for k in L.HEAD + [L.NEC, L.IND, "a_pos_tk", "eM_topk_tk", "eN_topk_tk", "e0_tk", "n"]:
                        if k in prod:
                            row["prod_" + k] = prod[k]
                    last[(row["seed"], row["arm"])] = row
    return pd.DataFrame(list(last.values()))


def donor_dirs(model):
    if model == "turingllm":
        return {"wcm": L.find(model, "out", "donor/main")}
    return {"wcm": L.find(model, "out", "donor"), "unweighted": L.find(model, "out", "donor_unweighted")}


def q(s):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    if s.empty:
        return dict(n=0, med=np.nan, p25=np.nan, p75=np.nan, p90=np.nan)
    return dict(n=len(s), med=s.median(), p25=s.quantile(.25), p75=s.quantile(.75), p90=s.quantile(.9))


def sf_stats(s, pre="sf"):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    d = {"%s_%s" % (pre, k): v for k, v in q(s).items()}
    for t in (0.0, 0.01, 0.05, 0.10):
        d["%s_gt%g" % (pre, t)] = 100 * float((s > t).mean()) if len(s) else np.nan
    return d


def ind_stats(s, pre="ind"):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    d = {"%s_%s" % (pre, k): v for k, v in q(s).items()}
    if len(s):
        d["%s_lt0.1" % pre] = 100 * float((s < 0.1).mean())
        d["%s_ge0.5" % pre] = 100 * float((s >= 0.5).mean())
        d["%s_inband" % pre] = 100 * float(((s >= L.BAND[0]) & (s <= L.BAND[1])).mean())
    return d


def nulls_ind(model):
    """phi_ind of the 084 harder-null arms (draw 0, held split, non-vacuous), restricted to production-scored seeds."""
    sys.path.insert(0, str(ROOT / "experiments" / "084-harder-nulls"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("report84", ROOT / "experiments" / "084-harder-nulls" / "report.py")
    R84 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R84)
    dirs = L.find(model, "work", "harder_nulls") if model == "turingllm" else L.find(model, "out", "harder_nulls")
    rows = []
    for d in dirs:
        rows += R84.read_rows(d, "nulls")
    if not rows:
        return None
    df = R84.frame(rows)
    df["vacuous"] = L.vacuous(df).values
    df = df[(df.split == "held") & (df.draw == 0) & ~df.vacuous]
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summ, nul, per, lines = [], [], [], []
    for model in L.MODELS:
        ev = L.main_eval(model)
        s = ev[(ev.held == "strong") & ~ev.vac].copy()
        s[L.IND] = pd.to_numeric(s[L.IND], errors="coerce")
        s["ind_a1"] = pd.to_numeric(s.get("phi_cf_a1_role_tk"), errors="coerce")
        ctr = s["ctr_source"].value_counts().to_dict() if "ctr_source" in s else {}
        abase = pd.to_numeric(s.get("a_base_tk"), errors="coerce")
        lines.append("%s: scored %d, ctr_source %s, a_base_tk (natural target at contrast anchors) max %.4g"
                     % (L.LABEL[model], len(s), ctr, abase.max()))
        dd = {}
        for arm, dirs in donor_dirs(model).items():
            df = donor_frame(dirs)
            if df.empty:
                continue
            dd[arm] = df[df.arm == "main"] if arm == "wcm" and "main" in set(df.arm) else df
        s["alpha_median"] = pd.to_numeric(s.get("alpha_median"), errors="coerce")
        w = s.set_index("seed")[["layer", "kind", "band", "n", "pass", "alpha_median", L.IND, "ind_a1"] + L.HEAD
                                + [L.NEC]]
        if "wcm" in dd:
            dw = dd["wcm"].set_index("seed")
            w = w.join(dw[[c for c in dw if c.startswith(("sf_", "donor_fires", "rec_fit"))]], how="left")
        lines.append("  donor rows joined: %d / %d scored" % (int(w["sf_fit_A"].notna().sum()) if "sf_fit_A" in w else 0,
                                                              len(w)))
        w["model"] = L.LABEL[model]
        per.append(w.reset_index())
        groups = [("all", w)] + [("band " + b, w[w.band == b]) for b, _, _ in L.bands_of(model)] + \
                 [("pass", w[w["pass"]]), ("fail", w[~w["pass"]])] + \
                 [("deep pass", w[(w.band == L.bands_of(model)[-1][0]) & w["pass"]]),
                  ("deep fail", w[(w.band == L.bands_of(model)[-1][0]) & ~w["pass"]])] + \
                 [("kind " + k, w[w.kind == k]) for k in L.KIND_ORDER if k in set(w.kind)] + \
                 [("pass kind " + k, w[(w.kind == k) & w["pass"]]) for k in L.KIND_ORDER if k in set(w.kind)]
        for g, x in groups:
            r = dict(model=L.LABEL[model], circuits="WCM", group=g, n=len(x))
            r.update(ind_stats(x[L.IND]))
            r.update(ind_stats(x["ind_a1"], "ind_a1"))
            for v in ("fit", "empty", "a1"):
                for ds in POOLS:
                    c = "sf_%s_%s" % (v, ds)
                    if c in x:
                        r.update(sf_stats(x[c], "sf_%s_%s" % (v, ds)))
            if "sf_fit_A" in x:
                fa, ea = x["sf_fit_A"], x["sf_empty_A"]
                r["veto_share_A"] = 100 * float(((fa < ea)).mean())          # circuit at silent values fires less than empty
                r["selective_both"] = 100 * float(((x[L.IND] >= 0.5) & (x[[c for c in x if c.startswith("sf_fit_")]]
                                                                           .max(axis=1) <= 0.01)).mean())
            summ.append(r)
        # what drives the false firing: alpha (fit vs a1) and size, on passing circuits
        if "sf_fit_C" in w:
            x = w[w["pass"]]
            rho = lambda a, b: float(pd.Series(a).rank().corr(pd.Series(b).rank()))  # noqa: E731
            lines.append("  pass: Spearman sf_fit_C vs alpha_median %.2f, vs n %.2f, vs layer %.2f; sf_fit_C vs phi_ind %.2f"
                         % (rho(x.sf_fit_C, x.alpha_median), rho(x.sf_fit_C, x.n), rho(x.sf_fit_C, x.layer),
                            rho(x.sf_fit_C, x[L.IND])))
            t = pd.qcut(x.alpha_median, 3, labels=["low", "mid", "high"])
            for lab in ("low", "mid", "high"):
                y = x[t == lab]
                lines.append("    alpha_median tercile %-4s (%.2f-%.2f): n %d, sf_fit_C median %.3f, >0.01 %.1f%%; "
                             "sf_a1_C >0.01 %.1f%%" % (lab, y.alpha_median.min(), y.alpha_median.max(), len(y),
                                                       y.sf_fit_C.median(), 100 * (y.sf_fit_C > .01).mean(),
                                                       100 * (y.sf_a1_C > .01).mean()))
        # Pythia unweighted circuits (donor_unweighted): silent_firing only
        if "unweighted" in dd:
            u = dd["unweighted"]
            both = u.set_index("seed").join(w[["sf_fit_A", "sf_fit_C", "sf_fit_R", "pass"]], rsuffix="_wcm",
                                            how="inner")
            lines.append("  unweighted vs WCM, same %d seeds: sf_fit_A median %.3f vs %.3f; sf_fit_C %.3f vs %.3f; "
                         "sf_fit_C >0.01 %.1f%% vs %.1f%%" % (
                             len(both), both.sf_fit_A.median(), both.sf_fit_A_wcm.median(), both.sf_fit_C.median(),
                             both.sf_fit_C_wcm.median(), 100 * (both.sf_fit_C > .01).mean(),
                             100 * (both.sf_fit_C_wcm > .01).mean()))
            pr = pd.DataFrame({k.replace("prod_", ""): u[k] for k in u if k.startswith("prod_")})
            pr = pr.apply(pd.to_numeric, errors="coerce")
            vac = L.vacuous(pr).values
            u = u[~vac].copy()
            u_pass = np.zeros(len(u), bool)
            u_pass[:] = L.passes(pr[~vac]).values
            u["pass"] = u_pass
            for g, x in (("all", u), ("pass", u[u["pass"]])):
                r = dict(model=L.LABEL[model], circuits="unweighted (donor_unweighted)", group=g, n=len(x))
                if "prod_" + L.IND in x:
                    r.update(ind_stats(x["prod_" + L.IND]))
                for v in ("fit", "empty"):
                    for ds in POOLS:
                        r.update(sf_stats(x["sf_%s_%s" % (v, ds)], "sf_%s_%s" % (v, ds)))
                summ.append(r)
        nd = nulls_ind(model)
        if nd is not None:
            nd = nd[nd.seed.isin(w.index)]
            for arm in sorted(set(nd.arm)):
                x = pd.to_numeric(nd[nd.arm == arm][L.IND], errors="coerce")
                r = dict(model=L.LABEL[model], arm=arm, targets=int((nd.arm == arm).sum()))
                r.update(ind_stats(x))
                nul.append(r)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "summary.csv", index=False)
    N = pd.DataFrame(nul)
    N.to_csv(OUT / "nulls_ind.csv", index=False)
    P = pd.concat(per, ignore_index=True)
    P.to_csv(OUT / "per_target.csv", index=False)
    # worst passing circuits: highest silent_firing (fit, max over pools) and lowest phi_ind
    P["sf_fit_max"] = P[[c for c in P if c.startswith("sf_fit_")]].max(axis=1)
    pp = P[P["pass"]]
    cols = ["model", "seed", "layer", "kind", "band", "n", L.IND, "sf_fit_A", "sf_fit_C", "sf_fit_R", "sf_empty_A",
            "sf_empty_R"] + L.HEAD + [L.NEC]
    worst = pd.concat([g.sort_values(["sf_fit_C", "sf_fit_R"], ascending=False).head(5)
                       .assign(why="highest silent_firing (pool C)") for _, g in pp.groupby("model")] +
                      [g.sort_values([L.IND, "n"], ascending=[True, False]).head(3)
                       .assign(why="lowest phi_ind") for _, g in pp.groupby("model")])
    worst[["why"] + cols].to_csv(OUT / "worst.csv", index=False)
    with open(OUT / "report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    key = ["model", "circuits", "group", "n", "ind_med", "ind_p25", "ind_p75", "ind_lt0.1", "ind_ge0.5", "ind_inband",
           "sf_fit_A_med", "sf_fit_A_p90", "sf_fit_A_gt0.01", "sf_fit_A_gt0.05", "sf_empty_A_med", "sf_fit_R_gt0.01",
           "sf_empty_R_gt0.01", "sf_fit_C_gt0.01", "veto_share_A", "selective_both"]
    print(S[[c for c in key if c in S]].round(3).to_string(index=False))
    print(N.round(3).to_string(index=False))
    print(worst[["why"] + cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
