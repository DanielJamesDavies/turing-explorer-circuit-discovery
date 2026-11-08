"""Table 7: the lambda pilot of both public models (30 pilot targets each, disjoint from main).

Reads   out_<model>/sweep/eval.shard*.jsonl (the pilot's W_<lam> arms; Pythia's U_0.001 rows are a different arm and
        are ignored by the pilot reader) and 082 targets_<model>_pilot.txt; out_<model>/pilot_lambda.json (as written
        on the pod)
Writes  paper/tables/run2026_pilot_lambda.tex, out/run2026_pilot_lambda.csv, out/run2026_pilot_lambda.json

The report is recomputed from the rows with 082 pilot_lambda.py's own report() (the pre-registered rule, sizes only:
keep 1e-3 unless the pooled median size leaves [100, 1000]) and checked against the pod's pilot_lambda.json.

  python -X utf8 experiments/109-pod-run-analysis/t7_pilot.py
"""
import importlib.util
import json

import pandas as pd

import lib109 as L

_spec = importlib.util.spec_from_file_location("pilot_lambda", L.ROOT / "experiments" / "082-public-runs" / "pilot_lambda.py")
PL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PL)


def main():
    rows, tex, js = [], [], {}
    for model in ("pythia-70m", "gpt2-small"):
        outs = L.find(model, "out")
        tg = L.find_file(model, "run", "targets_%s_pilot.txt" % model)
        if not outs or not list(outs[-1].glob("sweep/eval.*.jsonl")):
            L.not_available("%s pilot sweep rows" % model)
            continue
        r = PL.report(outs[-1], model, str(tg) if tg else None)
        pod = outs[-1] / "pilot_lambda.json"
        pod_r = json.load(open(pod, encoding="utf-8")) if pod.exists() else None
        same = None
        if pod_r:
            same = (pod_r.get("recommended_lam") == r["recommended_lam"] and
                    pod_r.get("pooled_median_size") == r["pooled_median_size"])
        js[model] = dict(recomputed=r, pod_json_matches=same)
        print("%-11s pilot: grid %s | pooled medians %s | recommended %g (%s) | pod json matches: %s | coverage %s" % (
            L.LABEL[model], r["grid"], r["pooled_median_size"], r["recommended_lam"], r["reason"], same,
            {k: v for k, v in (r["coverage"] or {}).items() if k != "not_in_list"}))
        if tex:
            tex.append([r"\midrule"])
        for p in r["per_lambda"]:
            s = p["pooled"]
            bands = " / ".join(L.tex_int(p["by_band"][b]["median_size"]) for b in r["bands"])
            rows.append(dict(model=L.LABEL[model], lam=p["lam"], circuits=s["circuits"], median_size=s["median_size"],
                             p25_size=s["p25_size"], p75_size=s["p75_size"], share_in_1e2_1e3=s["share_in_1e2_1e3"],
                             pass_rate_reported=s["pass_rate_reported"], vacuous=s["vacuous"],
                             **{"median_size_%s" % b: p["by_band"][b]["median_size"] for b in r["bands"]},
                             recommended=p["lam"] == r["recommended_lam"]))
            tex.append([L.LABEL[model] if p is r["per_lambda"][0] else "",
                        "%g%s" % (p["lam"], " (kept)" if p["lam"] == r["recommended_lam"] else ""),
                        L.tex_int(s["circuits"]), "%s (%s--%s)" % (L.tex_int(s["median_size"]), L.tex_int(s["p25_size"]),
                                                                   L.tex_int(s["p75_size"])),
                        "%.0f" % (100 * s["share_in_1e2_1e3"]), "%s (%s)" % (bands, ", ".join(r["bands"])),
                        "%.0f" % (100 * s["pass_rate_reported"]) if s["pass_rate_reported"] is not None else L.DASH])
    if not rows:
        return
    L.write_table("pilot_lambda", tex, ["Model", r"$\lambda$", "Circuits", "Size, median (IQR)",
                                        r"In $[10^2, 10^3]$ (\%)", "Median size by depth band", r"Pass (\%, reported)"],
                  "llrllll", note="pre-registered rule (082 pilot_lambda.py): keep 1e-3 unless the pooled median size "
                  "leaves [100, 1000]; pass rate reported, never used", csv_df=pd.DataFrame(rows))
    L.write_json("pilot_lambda", js)


if __name__ == "__main__":
    main()
