"""Experiment 080 target lists (CPU, no model).

Every target is an internal latent; the tier axis is how much it affects the next-token prediction.

Population: every 062 full-run target with a held-out-strongest eval row that is not vacuous (vacuous_tk false), a
production circuit (n >= 1) and a cached context record; all 15,004 are in 064's target_effects.parquet. Band: L0-4 /
L5-7 / L8-11. Each target also carries its legacy direct-path tier (dp_tier: common.tier_of on 064's boost_peak).

SAMPLING=causal (DEFAULT since the second 2026-10-18 review). Needs the measured in-context next-token effects
(next_token_effect.py -> effect/effects.parquet, via queue_effect.sh); REFUSES to run without them. Within each of the
9 kind x band groups' measured pool, the top tertile of effect_abs is "high next-token effect" and the bottom tertile
"low"; the middle is dropped (common.causal_pool). Each group gives one high-vs-low pair, layer-matched
(common.withinkind_pick with pairs=CAUSAL_PAIRS), 9 per tier (162); a group whose pool cannot fill 9 per tier, or not
layer-matched, is flagged. Calibration: one target per group-tier cell (18), disjoint, layer-matched within each pair,
a smoke target forced into its cell when it has a causal tier (common.withinkind_calib).
  -> targets.txt, targets_calib.txt, targets_meta.csv (targets_smoke.txt kept: common.SMOKE_TARGETS)
SAMPLING=withinkind: six within-kind, within-band pairs of adjacent direct-path tiers (common.PAIRS), 13 per cell (156),
layer-matched; calibration one per cell (12), smoke forced.
  -> targets_withinkind.txt, targets_calib_withinkind.txt, targets_meta_withinkind.csv
SAMPLING=tierband (the original design): 50 per direct-path tier, 16 / 17 / 17 over the bands (common.stratified_pick),
calibration common.calib_pick.
  -> targets_tierband.txt, targets_calib_tierband.txt, targets_meta_tierband.csv
SUFFIX overrides the output suffix (e.g. SUFFIX= with SAMPLING=withinkind rewrites the default-named files).

  PYTHONPATH=src python experiments/080-latent-vs-output/make_targets.py
  env: SAMPLING (causal | withinkind | tierband)  SUFFIX  EFFECTS_FILE (effect/effects.parquet)  SRC_OUT (062 out)
       N_PER_CAUSAL (9)  N_PER_CELL (13)  N_PER_TIER (50)
"""
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

SRC_OUT = Path(os.environ.get("SRC_OUT", str(C.EXP / "062-h100-protocol-v1" / "out_full" / "out")))
EFFECTS_064 = C.EXP / "064-prediction-circuits" / "results" / "target_effects.parquet"
EFFECTS_FILE = Path(os.environ.get("EFFECTS_FILE", str(C.EFFECTS_FILE)))
SAMPLING = os.environ.get("SAMPLING", "causal")
N_PER_TIER = int(os.environ.get("N_PER_TIER", C.N_PER_TIER))
N_PER_CELL = int(os.environ.get("N_PER_CELL", C.N_PER_CELL))
N_PER_CAUSAL = int(os.environ.get("N_PER_CAUSAL", C.N_PER_CAUSAL))
DEFAULT_SUFFIX = {"causal": "", "withinkind": "_withinkind", "tierband": "_tierband"}


def strong_rows():
    ev = {}
    for f in sorted(glob.glob(str(SRC_OUT / "main" / "eval.shard*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "error" not in r and r.get("held") == "strong":
                ev[r["seed"]] = r
    return ev


def population():
    """(meta by key, drop counts) for the filtered population; tier = the legacy direct-path tier."""
    import pandas as pd
    te = pd.read_parquet(EFFECTS_064)
    te["key"] = te.layer.astype(str) + "." + te.kind + "." + te.latent.astype(str)
    ev = strong_rows()
    meta, n_drop = {}, defaultdict(int)
    for r in te.itertuples():
        e = ev.get(r.key)
        if e is None:
            n_drop["no_eval_row"] += 1; continue
        if e.get("vacuous_tk"):
            n_drop["vacuous"] += 1; continue
        if not e.get("n"):
            n_drop["empty_circuit"] += 1; continue
        if not (SRC_OUT / "main" / "circuits" / ("%s.pt" % r.key)).exists():
            n_drop["no_circuit_file"] += 1; continue
        if not (SRC_OUT / "ctx" / ("%s.pt" % r.key)).exists():
            n_drop["no_ctx_file"] += 1; continue
        t, b = C.tier_of(float(r.boost_peak)), C.band(int(r.layer))
        p = C.pair_of(r.kind, b, t, C.PAIRS)
        meta[r.key] = dict(seed=r.key, layer=int(r.layer), kind=r.kind, tier=t, dp_tier=t, band=b,
                           pair=C.pair_name(r.kind, b) if p else "", cell=C.cell_name(r.kind, b, t) if p else "",
                           boost_peak=float(r.boost_peak), z1=float(r.z1), lat_n=int(e["n"]),
                           lat_pass=C.passes_latent(e))
    return meta, n_drop


def layer_quality(meta, keys, pairs):
    """Per pair: the layer histogram of each tier, mean layers, and the share of targets in matched layer slots."""
    import numpy as np
    rows = []
    for kind, b, hi, lo in pairs:
        ks = {t: [k for k in keys if meta[k]["cell"] == C.cell_name(kind, b, t)] for t in (hi, lo)}
        hist = {t: {l: sum(meta[k]["layer"] == l for k in ks[t]) for l in C.band_layers(b)} for t in (hi, lo)}
        matched = sum(min(hist[hi][l], hist[lo][l]) for l in C.band_layers(b))
        n = max(len(ks[hi]), len(ks[lo]), 1)
        mh = float(np.mean([meta[k]["layer"] for k in ks[hi]])) if ks[hi] else float("nan")
        ml = float(np.mean([meta[k]["layer"] for k in ks[lo]])) if ks[lo] else float("nan")
        rows.append(dict(pair=C.pair_name(kind, b), tiers="%s|%s" % (hi, lo), n_hi=len(ks[hi]), n_lo=len(ks[lo]),
                         layers_hi=" ".join("%d:%d" % kv for kv in hist[hi].items()),
                         layers_lo=" ".join("%d:%d" % kv for kv in hist[lo].items()),
                         mean_layer_hi=round(mh, 2), mean_layer_lo=round(ml, 2), d_mean_layer=round(mh - ml, 2),
                         matched_share=round(matched / n, 2)))
    return rows


def causal_meta(meta):
    """The measured pool with causal tiers: (meta restricted to measured targets, pool, cut-offs). Every measured
    target keeps its effect fields; tier / pair / cell are the causal ones (None for the dropped middle tertile)."""
    eff = C.load_effects(EFFECTS_FILE)
    measured = {}
    for k, m in meta.items():
        e = eff.get(k)
        if e is None:
            continue
        measured[k] = dict(m, effect_abs=float(e["effect_abs"]), effect_signed=float(e["effect_signed"]),
                           kl=float(e["kl"]), n_effect_contexts=int(e["n_contexts"]),
                           mean_act_at_anchor=e.get("mean_act_at_anchor"))
    pool, tiers, cuts = C.causal_pool(measured)
    for k, m in measured.items():
        t = tiers.get(k)
        m.update(tier=t or "", pair=C.pair_name(m["kind"], m["band"]) if t else "",
                 cell=C.cell_name(m["kind"], m["band"], t) if t else "")
    return measured, pool, cuts


def main():
    import pandas as pd
    if SAMPLING not in C.SAMPLINGS:
        sys.exit("SAMPLING must be one of %s" % (C.SAMPLINGS,))
    if SAMPLING == "causal" and not EFFECTS_FILE.exists():
        sys.exit("REFUSED: SAMPLING=causal needs %s, which does not exist yet. Measure the in-context next-token "
                 "effects first: bash experiments/080-latent-vs-output/queue_effect.sh (next_token_effect.py), or "
                 "use SAMPLING=withinkind / tierband. Nothing was written." % EFFECTS_FILE)
    suffix = os.environ.get("SUFFIX", DEFAULT_SUFFIX[SAMPLING])
    meta, n_drop = population()
    print("population %d (dropped %s) | sampling %s | output suffix %r" % (len(meta), dict(n_drop), SAMPLING, suffix))
    pairs = None
    if SAMPLING == "tierband":
        pool = defaultdict(list)
        for k, m in meta.items():
            pool[(m["tier"], m["band"])].append(k)
        main_ = C.stratified_pick(pool, N_PER_TIER)
        calib = C.calib_pick(pool, exclude=main_)
    else:
        if SAMPLING == "withinkind":
            pairs, n_cell = C.PAIRS, N_PER_CELL
            pool = defaultdict(list)
            for k, m in meta.items():
                pool[(m["kind"], m["tier"], m["layer"])].append(k)
        else:
            pairs, n_cell = C.CAUSAL_PAIRS, N_PER_CAUSAL
            meta, pool, cuts = causal_meta(meta)
            print("measured pool by group (effect_abs tertile cut-offs, nats):")
            for p, c in cuts.items():
                print("  %-12s pool %4d  low %3d (<= %s)  high %3d (>= %s)" % (
                    p, c["n_pool"], c["n_low"], "%.4f" % c["low_max"] if c["low_max"] is not None else "-",
                    c["n_high"], "%.4f" % c["high_min"] if c["high_min"] is not None else "-"))
        main_, info = C.withinkind_pick(pool, n_cell, exclude=C.SMOKE_TARGETS, pairs=pairs)
        calib = C.withinkind_calib(pool, exclude=main_, pairs=pairs)
        for p, i in info.items():
            un = {t: len(v) for t, v in i["unmatched"].items() if v}
            print("  %-12s layer-matched %d of %d per tier%s%s" % (
                p, i["n_matched"], n_cell, ("; unmatched top-up %s" % un) if un else "",
                ("; THIN: available %s" % i["n_available"]) if i["thin"] else ""))
    assert not set(main_) & set(calib)
    (HERE / ("targets%s.txt" % suffix)).write_text("\n".join(main_) + "\n")
    (HERE / ("targets_calib%s.txt" % suffix)).write_text("\n".join(calib) + "\n")
    smoke = list(C.SMOKE_TARGETS)
    if SAMPLING != "tierband":
        assert not set(smoke) & set(main_)
        if suffix == "":
            (HERE / "targets_smoke.txt").write_text("\n".join(smoke) + "\n")
    rows = ([dict(meta[k], role="main", sampling=SAMPLING) for k in main_]
            + [dict(meta[k], role="calib", sampling=SAMPLING) for k in calib])
    d = pd.DataFrame(rows)
    d.to_csv(HERE / ("targets_meta%s.csv" % suffix), index=False)
    print("\nmain %d, calib %d, smoke %s (in calibration: %s)" % (len(main_), len(calib), smoke,
                                                                 [k for k in smoke if k in set(calib)]))
    if SAMPLING == "tierband":
        print(d.groupby(["role", "tier", "band"]).size().unstack().to_string())
        return
    print(d.groupby(["cell", "role"]).size().unstack(fill_value=0).to_string())
    print("\nlayer matching within pairs (main list):")
    print(pd.DataFrame(layer_quality(meta, main_, pairs)).to_string(index=False))
    print("\nlayer matching within pairs (calibration list):")
    print(pd.DataFrame(layer_quality(meta, calib, pairs)).to_string(index=False))
    m = d[d.role == "main"]
    agg = dict(n=("seed", "size"), lat_n=("lat_n", "median"), lat_pass=("lat_pass", "mean"),
               boost=("boost_peak", "median"))
    if "effect_abs" in m:
        agg["effect_abs"] = ("effect_abs", "median")
    print("\nmain LAT size / pass by cell:")
    print(m.groupby("cell").agg(**agg).round(3).to_string())


if __name__ == "__main__":
    main()
