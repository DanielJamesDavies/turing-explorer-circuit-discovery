"""096 - is there a circuit target with no token expression? (Linear DAN-151, "wanted" example)

Reads the static per-target direct effects that 064's target_effects.py already computed
(064/results/target_effects.parquet + target_topk.npz; CPU, no model forward needed) for the three
DAN-151 candidates and one output-facing reference (the eponymous-law latent 10.resid.15497), and
places each against the population of 15,004 targets and the same-site random null.

Direct effect = decoder direction through the final RMSNorm (linearised at rms_typ) to the unembedding,
centred over the real vocabulary (064 README section 1). "boost" = activation x l_max = nats the top
promoted token gains over the average token, at the target's peak and at its mean activation.

Writes results/direct_effects.json and results/direct_effects.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
E064 = HERE.parent / "064-prediction-circuits"
sys.path.insert(0, str(E064))
import common as C  # noqa: E402

CANDIDATES = {
    "9.resid.7640": "negation pivot (candidate)",
    "8.resid.31095": "causal 'but also' (candidate)",
    "9.resid.17596": "importance as a comparison dimension (candidate)",
    "10.resid.15497": "eponymous law (output-facing reference)",
}
N_TOK = 10


def pct(series, v):
    return float((series.to_numpy() < v).mean() * 100)


def main():
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    tdf = pd.read_parquet(E064 / "results" / "target_effects.parquet")
    ndf = pd.read_parquet(E064 / "results" / "null_effects.parquet")
    tk = np.load(E064 / "results" / "target_topk.npz")
    row_of = {int(g): i for i, g in enumerate(tk["gid"])}
    tok = C.load_tokenizer()

    def dec(i):
        return tok.decode([int(i)])

    tdf["boost_mean"] = tdf.mean_act * tdf.lmax
    res = {"population": {
        "targets_median_boost_peak": float(tdf.boost_peak.median()),
        "targets_median_boost_mean": float(tdf.boost_mean.median()),
        "resid_L8_10_median_boost_peak": float(tdf[(tdf.kind == "resid") & tdf.layer.between(7, 9)].boost_peak.median()),
        "inference_median_best_fired_dla_top1": 0.041,  # 064 README, fig 2 (per-position best fired circuit)
    }, "latents": {}}
    for key, role in CANDIDATES.items():
        l, k, i = key.split(".")
        gid = C.gid_of(int(l), k, int(i))
        r = tdf[tdf.gid == gid]
        if r.empty:
            res["latents"][key] = {"role": role, "error": "not a 062 target"}
            continue
        r = r.iloc[0]
        j = row_of[gid]
        site_null = ndf[(ndf.layer == r.layer) & (ndf.kind == r.kind)]
        same = tdf[(tdf.layer == r.layer) & (tdf.kind == r.kind)]
        res["latents"][key] = {
            "role": role, "pass": bool(r["pass"]),
            "peak_act": float(r.peak), "mean_act": float(r.mean_act),
            "lmax": float(r.lmax), "lmin": float(r.lmin),
            "boost_peak_nats": float(r.boost_peak), "boost_mean_nats": float(r.boost_mean),
            "suppress_peak_nats": float(r.peak * r.lmin),
            "z1": float(r.z1), "z1_site_null_p95": float(site_null.z1.quantile(0.95)),
            "kurt": float(r["kurt"]), "gain": float(r["gain"]), "ctx_z": float(r["ctx_z"]),
            "boost_peak_pctile_all_targets": pct(tdf.boost_peak, r.boost_peak),
            "boost_peak_pctile_same_site": pct(same.boost_peak, r.boost_peak),
            "promoted": [[dec(t), round(float(v), 3)] for t, v in zip(tk["top_ids"][j][:N_TOK], tk["top_vals"][j][:N_TOK])],
            "suppressed": [[dec(t), round(float(v), 3)] for t, v in zip(tk["bot_ids"][j][:N_TOK], tk["bot_vals"][j][:N_TOK])],
        }
    (out_dir / "direct_effects.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    lines = ["| latent | role | pass | peak act | boost@peak (nats) | boost@mean | supp@peak | z1 (site null p95) | ctx_z | pctile (all / site) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, d in res["latents"].items():
        if "error" in d:
            lines.append(f"| {key} | {d['role']} | not a target |")
            continue
        lines.append(f"| {key} | {d['role']} | {d['pass']} | {d['peak_act']:.2f} | {d['boost_peak_nats']:.2f} | "
                     f"{d['boost_mean_nats']:.3f} | {d['suppress_peak_nats']:.2f} | {d['z1']:.1f} ({d['z1_site_null_p95']:.1f}) | "
                     f"{d['ctx_z']:.2f} | {d['boost_peak_pctile_all_targets']:.0f} / {d['boost_peak_pctile_same_site']:.0f} |")
    lines.append("")
    for key, d in res["latents"].items():
        if "error" in d:
            continue
        lines.append(f"**{key}** promoted: " + ", ".join(f"`{t}` {v:+.2f}" for t, v in d["promoted"]))
        lines.append(f"suppressed: " + ", ".join(f"`{t}` {v:+.2f}" for t, v in d["suppressed"]))
        lines.append("")
    lines.append("Population: " + json.dumps(res["population"]))
    (out_dir / "direct_effects.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
