"""111 a4: run facts per model: fit time and hardware; held-out set sizes; contrast-context counts.

  fit time   main/status rows (lib109.frame, last per seed, ok rows): t_fit (the 400-step fit), t_eval, t_spec, t_ctx;
             median and total; by depth band. These are per-process wall seconds with several fits sharing a GPU
             under CUDA MPS, not exclusive GPU time.
  hardware   the main stages' launch logs (logs/<model>/main*.launch.log: "all N shards returned (G GPUs [...] x P per
             GPU, T CPU threads each)") and the shard logs' timestamps (per launch segment: first to last timestamp,
             max over shards; a segment with no fitted target is ignored). GPU type: H100 SXM per the pod notes; the
             archives do not record it.
  held-out   main/contexts rows: len(held_strong_ids), len(held_mid_ids); contrast = contrast_train_ids +
             contrast_held_ids (all fitted targets, vacuous included); TuringLLM records/build rows for the targets
             with no record (skip reasons) and for the corpus-scan share (n_scan / n_strong, n_store).

Writes paper/tables/run2026_heldout_sizes.tex, paper/tables/run2026_fit_time.tex, out/run2026_heldout_sizes*.csv,
       out/run2026_fit_time.csv, out/a4.json

  python -X utf8 experiments/111-failure-anatomy/a4_runinfo.py
"""
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import lib111 as X

L = X.L
LOGS = {"pythia-70m": ("pub-pythia-70m-*", "pythia-70m", ["main"]),
        "gpt2-small": ("pub-gpt2-small-*", "gpt2-small", ["main"]),
        "turingllm": ("pub-turingllm-*", "turingllm", ["main_c%d" % i for i in range(8)])}
TS = re.compile(r"^\[(\d\d:\d\d:\d\d) shard\d+\]")
SHARDS = re.compile(r"all (\d+) shards returned \((\d+) GPUs \[([^\]]*)\] x (\d+) per GPU")


def log_dir(model):
    glob_, name, _ = LOGS[model]
    ds = [a / "experiments/082-public-runs/logs" / name for a in sorted(L.ARCHIVES.glob(glob_)) if a.is_dir()]
    ds = [d for d in ds if d.exists()]
    return ds[-1] if ds else None          # the latest archive holds the most complete (appended) logs


def segments(path):
    """(start, end, n_ok) per launch segment of one shard log (a segment starts at a 'MODE main' line)."""
    segs, cur = [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = TS.match(line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%H:%M:%S")
        if "MODE main" in line:
            if cur:
                segs.append(cur)
            cur = [t, t, 0]
            continue
        if cur is None:
            continue
        while t < cur[1] - timedelta(hours=12):       # past midnight (small out-of-order steps are not a wrap)
            t += timedelta(days=1)
        cur[1] = t
        if " ok n=" in line:
            cur[2] += 1
    if cur:
        segs.append(cur)
    return segs


EVENT = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s+(\S+) (\S+) on GPUs ([\d,]+)(?: \(P=(\d+)\))?")


def chain_events(model, stage, start):
    """(gpu list, P) of the chain.events launch line '<model> <stage> on GPUs ... (P=n)' within 2 min of start."""
    for a in sorted(L.ARCHIVES.glob("pub-*")):
        p = a / "experiments/082-public-runs/logs/chain.events"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            m = EVENT.match(line)
            if m and m.group(2) == LOGS[model][1] and m.group(3) == stage and m.group(5):
                t = datetime.strptime(m.group(1), "%H:%M:%S")
                if abs((t - start).total_seconds()) < 120:
                    return m.group(4), m.group(5)
    return None


def hardware(model):
    d = log_dir(model)
    if d is None:
        return []
    out = []
    for stage in LOGS[model][2]:
        launch = d / ("%s.launch.log" % stage)
        txt = launch.read_text(encoding="utf-8", errors="replace") if launch.exists() else ""
        runs = SHARDS.findall(txt)
        allseg = []
        for p in sorted((d / stage).glob("main.shard*.log")):
            allseg += segments(p)
        # cluster the shards' segments by launch time (time of day, within 10 min): one cluster per launch
        clusters = []
        for a, b, n in sorted(allseg, key=lambda x: x[0]):
            if clusters and (a - clusters[-1][0][0]).total_seconds() < 600:
                clusters[-1].append((a, b, n))
            else:
                clusters.append([(a, b, n)])
        for j, ss in enumerate(clusters):
            n_ok = sum(n for _, _, n in ss)
            if n_ok == 0:
                continue
            start = ss[0][0]
            wall = max((a - start).total_seconds() + (b - a).total_seconds() for a, b, _ in ss)
            match = [r_ for r_ in runs if int(r_[0]) == len(ss)]   # the launch whose shard count matches
            r = match[0] if match else None
            if r is None:                                           # launch log overwritten: chain.events' line
                ev = chain_events(model, stage, start)
                if ev:
                    r = (len(ss), len(ev[0].split(",")), ev[0].replace(",", " "), ev[1])
            out.append(dict(model=L.LABEL[model], stage=stage, segment=j, start=start.strftime("%H:%M"),
                            fitted=n_ok, wall_h=wall / 3600, shards=int(r[0]) if r else None,
                            gpus=int(r[1]) if r else None, gpu_ids=r[2] if r else None,
                            per_gpu=int(r[3]) if r else None))
    return out


def main():
    ht, ft, hrows, hgrp, frows, hw_rows = [], [], [], [], [], []
    summary = {}
    for model in L.MODELS:
        lab = L.LABEL[model]
        # ------------------------------------------------------------------- fit time
        st = L.frame(L.read_rows(L.find(model, "out", "main"), "status"), keys=("seed",))
        st = st[st.get("ok", True) == True].copy() if "ok" in st else st  # noqa: E712
        st = L.add_site(st, model)
        for c in ("t_fit", "t_eval", "t_spec", "t_ctx"):
            st[c] = pd.to_numeric(st.get(c), errors="coerce")
        hw = hardware(model)
        hw_rows += hw
        wall = sum(h["wall_h"] for h in hw)
        fr = dict(model=lab, fits=len(st), t_fit_median_s=st.t_fit.median(), t_fit_p10_s=st.t_fit.quantile(.1),
                  t_fit_p90_s=st.t_fit.quantile(.9), t_fit_total_h=st.t_fit.sum() / 3600,
                  t_eval_median_s=st.t_eval.median(), t_spec_median_s=st.t_spec.median(),
                  t_all_total_h=st[["t_fit", "t_eval", "t_spec", "t_ctx"]].sum().sum() / 3600,
                  main_wall_h=wall, main_segments=len(hw),
                  layout="; ".join("%d fits: %s GPUs x %s" % (n_, g_, p_) for (g_, p_), n_ in sorted(
                      pd.DataFrame(hw).groupby(["gpus", "per_gpu"]).fitted.sum().items())) if hw else "",
                  **{"t_fit_median_%s_s" % b: st[st.band == b].t_fit.median() for b, _, _ in L.bands_of(model)})
        frows.append(fr)
        ft.append([lab, L.tex_int(fr["fits"]), "%.0f (%.0f--%.0f)" % (fr["t_fit_median_s"], fr["t_fit_p10_s"],
                                                                      fr["t_fit_p90_s"]),
                   " / ".join("%.0f" % fr["t_fit_median_%s_s" % b] for b, _, _ in L.bands_of(model)),
                   "%.1f" % fr["t_fit_total_h"], "%.1f" % fr["main_wall_h"],
                   L.tex_escape(fr["layout"]).replace(" x ", r" $\times$ ")])
        # ------------------------------------------------------------------- held-out sizes
        cx = L.frame(L.read_rows(L.find(model, "out", "main"), "contexts"), keys=("seed",))
        cx = L.add_site(cx, model)
        ln = lambda c: cx[c].map(lambda v: len(v) if isinstance(v, list) else 0)  # noqa: E731
        cx["n_ho_strong"], cx["n_ho_mid"] = ln("held_strong_ids"), ln("held_mid_ids")
        cx["n_contrast"] = ln("contrast_train_ids") + ln("contrast_held_ids")
        cx["n_contrast_ho"] = ln("contrast_held_ids")
        thin = cx.n_mid.fillna(0).astype(int) < 8 if "n_mid" in cx else pd.Series(False, index=cx.index)
        hr = dict(model=lab, fitted=len(cx), strong16=int((cx.n_ho_strong == 16).sum()),
                  strong_lt16=int((cx.n_ho_strong < 16).sum()), strong_lt8=int((cx.n_ho_strong < 8).sum()),
                  strong_min=int(cx.n_ho_strong.min()), strong_median=float(cx.n_ho_strong.median()),
                  mid_none=int((cx.n_ho_mid == 0).sum()), mid_lt16=int(((cx.n_ho_mid < 16) & (cx.n_ho_mid > 0)).sum()),
                  mid_lt8_nonzero=int(((cx.n_ho_mid < 8) & (cx.n_ho_mid > 0)).sum()),
                  contrast_lt64=int((cx.n_contrast < 64).sum()), contrast_min=int(cx.n_contrast.min()),
                  contrast_ho_lt16=int((cx.n_contrast_ho < 16).sum()))
        if model == "turingllm":
            b = pd.DataFrame(L.read_rows(L.find(model, "work", "records"), "build")).drop_duplicates("seed", keep="last")
            sk = b[b.get("skip").notna()] if "skip" in b else b.iloc[:0]
            ok = b[b.get("skip").isna()] if "skip" in b else b
            hr.update(no_record=len(sk), no_record_reasons="; ".join("%s %d" % kv for kv in sk.skip.value_counts().items()),
                      no_contrast_n_contrast=",".join(str(int(x)) for x in sk.get("n_contrast", pd.Series()).dropna()),
                      records_contrast_lt64=int((pd.to_numeric(ok.n_neg) < 64).sum()),
                      scan_share_strong_pct=100 * pd.to_numeric(ok.n_scan).sum() / pd.to_numeric(ok.n_strong).sum(),
                      store_rows=int(pd.to_numeric(ok.n_store).sum()),
                      records_held_lt8=int((pd.to_numeric(ok.n_held) < 8).sum()),
                      records_held_lt16=int((pd.to_numeric(ok.n_held) < 16).sum()))
        hrows.append(hr)
        for gt, col in (("band", "band"), ("kind", "kind")):
            for g, d in cx.groupby(col):
                hgrp.append(dict(model=lab, group_type=gt, group=g, fitted=len(d),
                                 strong_lt16=int((d.n_ho_strong < 16).sum()), strong_lt8=int((d.n_ho_strong < 8).sum()),
                                 mid_none=int((d.n_ho_mid == 0).sum()), contrast_lt64=int((d.n_contrast < 64).sum())))
        for b_, _, _ in L.bands_of(model):
            for k in L.KIND_ORDER:
                d = cx[(cx.band == b_) & (cx.kind == k)]
                if len(d):
                    hgrp.append(dict(model=lab, group_type="band x kind", group="%s %s" % (b_, k), fitted=len(d),
                                     strong_lt16=int((d.n_ho_strong < 16).sum()),
                                     strong_lt8=int((d.n_ho_strong < 8).sum()), mid_none=int((d.n_ho_mid == 0).sum()),
                                     contrast_lt64=int((d.n_contrast < 64).sum())))
        ht.append([lab, L.tex_int(hr["fitted"]), L.tex_int(hr["strong16"]), L.tex_int(hr["strong_lt16"]),
                   L.tex_int(hr["strong_lt8"]), L.tex_int(hr["strong_min"]), L.tex_int(hr["mid_none"]),
                   L.tex_int(hr["mid_lt16"]), L.tex_int(hr["contrast_lt64"]), L.tex_int(hr["contrast_min"])])
        summary[model] = dict(fit=fr, heldout=hr)
        print("%-11s fit: %d fits | t_fit median %.1f s (p10 %.1f, p90 %.1f) by band %s | total %.1f process-h (all "
              "stages of a fit %.1f h) | main wall %.1f h over %d segment(s), %s" % (
                  lab, fr["fits"], fr["t_fit_median_s"], fr["t_fit_p10_s"], fr["t_fit_p90_s"],
                  " / ".join("%.0f" % fr["t_fit_median_%s_s" % b] for b, _, _ in L.bands_of(model)),
                  fr["t_fit_total_h"], fr["t_all_total_h"], wall, len(hw), fr["layout"]))
        print("            held-out: %s" % ", ".join("%s %s" % kv for kv in hr.items() if kv[0] != "model"))
    for h in hw_rows:
        print("  segment %-11s %-8s seg %d start %s fitted %d wall %.2f h | %s GPUs [%s] x %s" % (
            h["model"], h["stage"], h["segment"], h["start"], h["fitted"], h["wall_h"], h["gpus"], h["gpu_ids"],
            h["per_gpu"]))
    pd.DataFrame(hgrp).to_csv(X.OUT / "run2026_heldout_sizes_groups.csv", index=False)
    pd.DataFrame(hw_rows).to_csv(X.OUT / "run2026_fit_segments.csv", index=False)
    L.write_table("heldout_sizes", ht, ["Model", "Fitted", "Strongest 16", r"Strongest $<16$", r"Strongest $<8$",
                                        "Strongest min", "No mid-band", r"Mid-band $<16$", r"Contrast $<64$",
                                        "Contrast min"], "lrrrrrrrrr",
                  note="per fitted target, from main/contexts rows: held-out strongest / mid-band context counts and "
                       "contrast contexts (training + held-out)", csv_df=pd.DataFrame(hrows))
    L.write_table("fit_time", ft, ["Model", "Fits", "Fit time, s: median (p10--p90)", "Median by depth band, s",
                                   "Summed fit time (h)", "Main-stage wall clock (h)", "GPU layout"], "lrllrrl",
                  note="t_fit from main/status rows (per-process wall seconds, fits share GPUs under CUDA MPS); wall "
                       "clock from the shard logs; NVIDIA H100 SXM (80 GB) pod",
                  csv_df=pd.DataFrame(frows),
                  footer=[r"Fit times are per-process wall seconds on NVIDIA H100 SXM GPUs shared by several fits under CUDA MPS."])
    L.write_json("a4", summary)


if __name__ == "__main__":
    main()
