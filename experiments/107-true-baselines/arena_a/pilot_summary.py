"""Arena A (A9): the timing pilot's summary. CPU only, reads files, no model.

  fits      per (lambda, target): fit seconds, steps, s / step, micro-batch used, circuit size, eval / ctx seconds,
            errors (driver main/train.shard*.jsonl and main/status.shard*.jsonl of every OUT P/wcm_l*)
  memory    peak and median GPU memory.used per pilot step (P/gpu_mem.csv: epoch,step,MiB, sampled every 10 s by
            run_pilot_a.sh; the whole card, so another process on it would be counted too)
  steps     wall time per pilot step (P/steps.tsv: step, start, end, exit code)
Writes P/timing_summary.json and P/timing_summary.md, and prints the markdown.

  python experiments/107-true-baselines/arena_a/pilot_summary.py [P]   (default arena_a/out/pilot)
"""
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "out" / "pilot"
CARD_MIB = 16303                      # RTX 5070 Ti; ~15.6 GB usable before Windows spills to shared memory
SPILL_MIB = 15600


def rows(paths):
    out = []
    for p in paths:
        for line in open(p):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def main():
    fits = []
    for od in sorted(P.glob("wcm_l*")):
        lam = od.name[len("wcm_l"):]
        tr = {r.get("seed"): r for r in rows(sorted((od / "main").glob("train.shard*.jsonl")))}
        stt = {r.get("seed"): r for r in rows(sorted((od / "main").glob("status.shard*.jsonl")))}
        for seed in sorted(set(tr) | set(stt)):
            t, s = tr.get(seed, {}), stt.get(seed, {})
            steps = t.get("steps")
            t_fit = t.get("t_fit", s.get("t_fit"))
            fits.append(dict(lam=lam, seed=seed, t_fit=t_fit, steps=steps,
                             s_per_step=round(t_fit / steps, 3) if t_fit and steps else None,
                             batch=t.get("batch_size_used"), n_kept=t.get("n_kept"), n=s.get("n"),
                             t_ctx=s.get("t_ctx"), t_eval=s.get("t_eval"), ok=s.get("ok"), error=s.get("error"),
                             skip=s.get("skip"), wcm_version=s.get("wcm_version") or t.get("wcm_version")))
    mem = {}
    mp = P / "gpu_mem.csv"
    if mp.exists():
        for line in open(mp):
            parts = line.strip().split(",")
            if len(parts) != 3 or not parts[2].strip().isdigit():
                continue
            mem.setdefault(parts[1], []).append(int(parts[2]))
    steps = []
    sp = P / "steps.tsv"
    if sp.exists():
        for line in open(sp):
            p = line.rstrip("\n").split("\t")
            if len(p) == 4 and p[1].isdigit() and p[2].isdigit():
                steps.append(dict(step=p[0], wall_s=int(p[2]) - int(p[1]), exit=p[3]))
    spp = [f["s_per_step"] for f in fits if f["s_per_step"]]
    summary = dict(
        fits=fits,
        s_per_step=dict(n=len(spp), median=st.median(spp) if spp else None, max=max(spp) if spp else None),
        n_errors=sum(1 for f in fits if f["error"]),
        memory={k: dict(n=len(v), peak_mib=max(v), median_mib=int(st.median(v))) for k, v in mem.items()},
        peak_mib=max((max(v) for v in mem.values()), default=None),
        steps=steps)
    summary["spill_risk"] = summary["peak_mib"] is not None and summary["peak_mib"] > SPILL_MIB
    md = ["# Arena A GPU timing pilot: summary", "",
          "| lambda | target | fit s | steps | s / step | micro-batch | n kept | ctx s | eval s | status |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for f in fits:
        md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            f["lam"], f["seed"], f["t_fit"], f["steps"], f["s_per_step"], f["batch"], f["n_kept"] or f["n"],
            f["t_ctx"], f["t_eval"], f["error"] or f["skip"] or ("ok" if f["ok"] else "-")))
    md += ["", "s / step: median %s, max %s over %d fits; errors %d" % (
        summary["s_per_step"]["median"], summary["s_per_step"]["max"], len(spp), summary["n_errors"]), "",
        "| pilot step | wall s | exit | GPU mem peak MiB | median MiB | samples |", "|---|---|---|---|---|---|"]
    names = [s["step"] for s in steps] + [k for k in mem if k not in {s["step"] for s in steps}]
    by = {s["step"]: s for s in steps}
    for nme in names:
        s, m = by.get(nme, {}), summary["memory"].get(nme, {})
        md.append("| %s | %s | %s | %s | %s | %s |" % (nme, s.get("wall_s"), s.get("exit"), m.get("peak_mib"),
                                                       m.get("median_mib"), m.get("n")))
    md += ["", "Peak GPU memory.used (whole card): %s MiB of %d (spill threshold ~%d: %s)" % (
        summary["peak_mib"], CARD_MIB, SPILL_MIB, "OVER" if summary["spill_risk"] else "under")]
    P.mkdir(parents=True, exist_ok=True)
    (P / "timing_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (P / "timing_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
