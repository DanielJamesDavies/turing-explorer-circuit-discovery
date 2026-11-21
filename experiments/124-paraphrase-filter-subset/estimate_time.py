"""124 local GPU time estimate (CPU): the subset's production t_eval (T3 status rows, H100 under co-tenancy) scaled by
the local / production ratio measured on the 115 pilot (the same 36 circuits scored on this RTX 5070 Ti, parsed from
115 out/rs_orig.log, out/rs_doc.log), plus the measured driver start-up and the 115 build cost.

  python -X utf8 experiments/124-paraphrase-filter-subset/estimate_time.py   -> out/time_estimate.json
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import para124 as P  # noqa: E402

C = P.C
sys.path.insert(0, str(C.EXP / "109-pod-run-analysis"))
import lib109 as L  # noqa: E402

LINE = re.compile(r"^\[(\d\d:\d\d:\d\d) shard0\]\s+\d+/\d+ (\S+)\s+ok .* eval ([\d.]+)s")


def parse(log):
    rows, t0, first = [], None, None
    for ln in Path(log).read_text(errors="replace").splitlines():
        m0 = re.match(r"^\[(\d\d:\d\d:\d\d) shard0\] MODE", ln)
        if m0 and t0 is None:
            t0 = m0.group(1)
        m = LINE.match(ln)
        if m:
            first = first or m.group(1)
            rows.append(dict(seed=m.group(2), t_local=float(m.group(3))))
    f = lambda s: datetime.strptime(s, "%H:%M:%S")
    startup = (f(first) - f(t0)).total_seconds() - rows[0]["t_local"]
    return pd.DataFrame(rows).set_index("seed"), startup


def main():
    dirs = [d / "main" for d in L.find("turingllm", "out") if (d / "main").exists()]
    prod = pd.DataFrame(L.read_rows(dirs, "status")).drop_duplicates("seed", keep="last").set_index("seed")
    e115 = C.EXP / "115-docdisjoint-rescore-pilot" / "out"
    ratios, startups, per = [], [], []
    for log in ("rs_orig.log", "rs_doc.log"):
        df, st = parse(e115 / log)
        j = df.join(prod[["t_eval"]], how="inner")
        ratios.append(j.t_local.sum() / j.t_eval.sum())
        startups.append(st)
        per.append(j.t_local.mean())
    ratio, startup = max(ratios), max(startups)
    keys = C.read_targets(P.subset_path())
    t_sub = prod.loc[keys, "t_eval"].sum()
    score_variant = ratio * t_sub + startup
    build = 100 + 35 + 0.5 * len(keys)            # 115 build: engine ~100 s, first target ~35 s, then ~0.4 s each
    total = build + 2 * score_variant + 120        # + CPU steps (check, prep, verify, report)
    out = dict(n=len(keys), prod_t_eval_subset_s=float(t_sub), local_over_prod=[float(r) for r in ratios],
               ratio_used=float(ratio), startup_s=float(startup), local_mean_eval_pilot_s=[float(x) for x in per],
               score_per_variant_min=score_variant / 60, build_min=build / 60, total_h=total / 3600,
               per_target_both_variants_s=2 * ratio * t_sub / len(keys),
               n_for_3h=int((3 * 3600 - 120 - build - 2 * startup) / (2 * ratio * t_sub / len(keys))))
    P.write_json(out, P.WORK / "time_estimate.json")
    for k, v in out.items():
        print("%-28s %s" % (k, v))


if __name__ == "__main__":
    main()
