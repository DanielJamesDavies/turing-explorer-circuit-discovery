"""Exp 128 ram_probe report: what the host RAM of one deep arena A WCM fit is used for.

  python -I ram_report.py --dir experiments/128-pod-preflight/out/ram_probe [--out .../ram_report.md]

Reads every probe sub-directory (mon.csv from ram_monitor.py, status.jsonl from probe_driver.py, KILLED if the
monitor killed it, wcm/main/status.shard0.jsonl from the driver) and writes ram_report.md (+ .json): per probe the
RSS / MemAvailable at the end of set-up, after the context load, at fit steps 1 / 50 / 100 / 200 / 400, the peaks
(process-tree RSS, VmHWM, nvidia-smi memory.used, torch's reserved CUDA memory), the CPU-tensor scans, the RSS slope
over fit steps, the memory used outside the process tree (MemAvailable drop minus RSS growth), and a verdict line per
hypothesis:
  H1  GPU allocations spill into system memory when the 16 GB card is full (WSL sysmem fallback);
  H2  something in the fit grows with steps (per-step state on the CPU);
  H3  something big is loaded up front (dictionaries, stores, contexts, fit set-up).
"""
import argparse
import csv
import json
from pathlib import Path

MILESTONE_STEPS = (1, 50, 100, 200, 400)
GIB = 2 ** 30


def read_csv(p):
    if not p.exists():
        return []
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if not r.get("mem_avail_mib"):
            continue
        for k, v in r.items():
            if k in ("phase", "key", "log_tail", "event"):
                continue
            try:
                r[k] = float(v) if v not in ("", None) else None
            except ValueError:
                r[k] = None
        out.append(r)
    return out


def read_jsonl(p):
    out = []
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def near(mon, t):
    """The monitor row closest in time to t."""
    if not mon or t is None:
        return None
    return min(mon, key=lambda r: abs((r["t"] or 0) - t))


def g(x, d=2):
    return "n/a" if x is None else ("%.*f" % (d, x))


def mib_to_g(x):
    return None if x is None else x / 1024.0


def slope(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return None if sxx == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def probe(d):
    mon, st = read_csv(d / "mon.csv"), read_jsonl(d / "status.jsonl")
    killed = (d / "KILLED").read_text().strip() if (d / "KILLED").exists() else None
    drv = read_jsonl(d / "wcm" / "main" / "status.shard0.jsonl")
    base = mon[0] if mon else None
    res = dict(name=d.name, killed=killed, n_mon=len(mon), n_status=len(st),
               driver_status=[{k: r.get(k) for k in ("seed", "ok", "error", "n", "t_fit", "t_eval")} for r in drv])
    if base is None:
        return res
    a0, r0 = base["mem_avail_mib"], base["tree_rss_mib"] or 0

    def at(t, rss_kb=None):
        m = near(mon, t)
        if m is None:
            return None
        rss = (rss_kb / 1024.0) if rss_kb else m["tree_rss_mib"]
        drop = a0 - m["mem_avail_mib"]
        return dict(rss_gib=mib_to_g(rss), tree_rss_gib=mib_to_g(m["tree_rss_mib"]), avail_gib=mib_to_g(m["mem_avail_mib"]),
                    nonproc_gib=mib_to_g(drop - ((m["tree_rss_mib"] or 0) - r0)), gpu_used_gib=mib_to_g(m["gpu_used_mib"]),
                    swap_gib=mib_to_g(m["swap_used_mib"]))

    ms = {}
    for ev, ph, label in (("exit", "runner_init", "end of set-up (model, bank, stores)"),
                          ("exit", "contexts", "after the context / record load"),
                          ("enter", "fit", "fit entered")):
        r = next((x for x in st if x.get("event") == ev and x.get("phase") == ph), None)
        if r:
            ms[label] = dict(at(r["t"], r.get("VmRSS")) or {}, cuda_reserved_gib=(r.get("cuda_reserved") or 0) / GIB)
    steps = [x for x in st if x.get("event") == "step"]
    last_fit = max((x.get("fit_i", 0) for x in steps), default=0)
    steps = [x for x in steps if x.get("fit_i") == last_fit]
    for s in MILESTONE_STEPS:
        r = next((x for x in steps if x.get("step") == s), None)
        if r:
            ms["step %d" % s] = dict(at(r["t"], r.get("VmRSS")) or {}, cuda_reserved_gib=(r.get("cuda_reserved") or 0) / GIB)
    r = next((x for x in st if x.get("event") == "exit" and x.get("phase") == "fit"), None)
    if r:
        ms["fit returned"] = dict(at(r["t"], r.get("VmRSS")) or {}, cuda_reserved_gib=(r.get("cuda_reserved") or 0) / GIB)
    if steps:
        ms["last logged step (%d)" % steps[-1]["step"]] = dict(at(steps[-1]["t"], steps[-1].get("VmRSS")) or {},
                                                               cuda_reserved_gib=(steps[-1].get("cuda_reserved") or 0) / GIB)
    res["milestones"] = ms
    res["start"] = dict(tree_rss_gib=mib_to_g(r0), avail_gib=mib_to_g(a0), gpu_used_gib=mib_to_g(base["gpu_used_mib"]))
    gpu_total = next((m["gpu_total_mib"] for m in mon if m.get("gpu_total_mib")), None)
    res["peak"] = dict(
        tree_rss_gib=mib_to_g(max((m["tree_rss_mib"] or 0) for m in mon)),
        vmhwm_gib=max([(x.get("VmHWM") or 0) / 2 ** 20 for x in st] + [mib_to_g(max((m["tree_hwm_mib"] or 0) for m in mon))]),
        min_avail_gib=mib_to_g(min(m["mem_avail_mib"] for m in mon)),
        max_nonproc_gib=mib_to_g(max(a0 - m["mem_avail_mib"] - ((m["tree_rss_mib"] or 0) - r0) for m in mon)),
        gpu_used_gib=mib_to_g(max((m["gpu_used_mib"] or 0) for m in mon)), gpu_total_gib=mib_to_g(gpu_total),
        cuda_reserved_gib=max([(x.get("cuda_max_reserved") or x.get("cuda_reserved") or 0) / GIB for x in st] + [0]),
        cuda_total_gib=next((x["cuda_total"] / GIB for x in st if x.get("cuda_total")), None),
        swap_used_gib=mib_to_g(max((m["swap_used_mib"] or 0) for m in mon)))
    res["scans"] = [dict(event=x["event"], phase=x["phase"], step=x["step"], gib=x["cpu_tensor_bytes"] / GIB,
                         pinned_gib=x.get("cpu_pinned_bytes", 0) / GIB, n=x.get("cpu_tensor_n"), top=x.get("cpu_top"),
                         rss_gib=(x.get("VmRSS") or 0) / 2 ** 20)
                    for x in st if "cpu_tensor_bytes" in x]
    xs = [x["step"] for x in steps if x.get("VmRSS") and x["step"] >= 5]
    ys = [x["VmRSS"] / 1024.0 for x in steps if x.get("VmRSS") and x["step"] >= 5]
    res["rss_mib_per_step"] = slope(xs, ys)
    res["n_steps_logged"] = steps[-1]["step"] if steps else 0
    res["rss_step1_gib"] = (steps[0]["VmRSS"] / 2 ** 20) if steps and steps[0].get("VmRSS") else None
    res["rss_laststep_gib"] = (steps[-1]["VmRSS"] / 2 ** 20) if steps and steps[-1].get("VmRSS") else None
    return res


def verdicts(res):
    out = []
    pk, ms = res.get("peak") or {}, res.get("milestones") or {}
    ct, cr = pk.get("cuda_total_gib"), pk.get("cuda_reserved_gib")
    gu, gt, npk = pk.get("gpu_used_gib"), pk.get("gpu_total_gib"), pk.get("max_nonproc_gib")
    if ct and cr is not None:
        if cr > 0.97 * ct:
            out.append("H1 SUPPORTED: torch reserved %.1f GiB on a %.1f GiB card (more than fits: sysmem fallback); "
                       "memory outside the process tree peaked at %s GiB" % (cr, ct, g(npk, 1)))
        elif gu is not None and gt and gu > gt - 0.5:
            out.append("H1 POSSIBLE: nvidia-smi shows the card full (%.1f of %.1f GiB) but torch reserved only %.1f GiB; "
                       "outside-tree memory peak %s GiB" % (gu, gt, cr, g(npk, 1)))
        else:
            out.append("H1 NOT SUPPORTED: the card never filled (torch reserved peak %.1f of %.1f GiB, nvidia-smi peak "
                       "%s GiB); outside-tree memory peak %s GiB" % (cr, ct, g(gu, 1), g(npk, 1)))
    else:
        out.append("H1 INCONCLUSIVE: no CUDA numbers from the driver")
    s1, sl = res.get("rss_step1_gib"), res.get("rss_laststep_gib")
    rate, n = res.get("rss_mib_per_step"), res.get("n_steps_logged", 0)
    peak = pk.get("tree_rss_gib")
    if s1 is not None and sl is not None and peak:
        grow = sl - s1
        tag = "SUPPORTED" if grow > 2 and grow > 0.3 * peak else ("WEAK" if grow > 0.5 else "NOT SUPPORTED")
        out.append("H2 %s: RSS %.1f GiB at step 1 -> %.1f GiB at step %d (%+.1f GiB; slope %s MiB / step); peak tree RSS "
                   "%.1f GiB" % (tag, s1, sl, n, grow, g(rate, 1), peak))
    else:
        out.append("H2 INCONCLUSIVE: no fit steps logged (killed before the first step?)")
    su = ms.get("end of set-up (model, bank, stores)", {}).get("rss_gib")
    sc = ms.get("after the context / record load", {}).get("rss_gib")
    if peak and (su is not None or s1 is not None):
        upfront = s1 if s1 is not None else (sc if sc is not None else su)
        tag = "SUPPORTED" if upfront > 0.5 * peak else "NOT SUPPORTED"
        out.append("H3 %s: RSS %s GiB after set-up, %s GiB after the contexts, %s GiB at fit step 1 (fit set-up), of a "
                   "%.1f GiB peak" % (tag, g(su, 1), g(sc, 1), g(s1, 1), peak))
    else:
        out.append("H3 INCONCLUSIVE: no set-up marker")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    D = Path(a.dir)
    out = Path(a.out) if a.out else D.parent / "ram_report.md"
    probes = [probe(d) for d in sorted(p for p in D.iterdir() if p.is_dir() and (p / "mon.csv").exists())] if D.is_dir() else []
    L = ["# Exp 128 ram_probe: host RAM of one deep arena A WCM fit (local RTX 5070 Ti, WSL)", ""]
    if not probes:
        L.append("No probe ran (no %s/*/mon.csv)." % D)
    for r in probes:
        r["verdicts"] = verdicts(r)
        L += ["## %s" % r["name"], ""]
        L.append("- killed by the monitor: %s" % (r["killed"] or "no (ended by itself)"))
        for s in r["driver_status"]:
            L.append("- driver status: %s" % json.dumps(s))
        if "start" in r:
            s = r["start"]
            L.append("- at launch: tree RSS %s GiB, MemAvailable %s GiB, GPU used %s GiB" % (
                g(s["tree_rss_gib"]), g(s["avail_gib"]), g(s["gpu_used_gib"])))
        L += ["", "| point | RSS GiB (driver) | tree RSS GiB | MemAvailable GiB | outside-tree GiB | nvidia-smi GiB | torch reserved GiB | swap GiB |",
              "|---|---|---|---|---|---|---|---|"]
        for k, m in (r.get("milestones") or {}).items():
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (k, g(m.get("rss_gib")), g(m.get("tree_rss_gib")),
                     g(m.get("avail_gib")), g(m.get("nonproc_gib")), g(m.get("gpu_used_gib")),
                     g(m.get("cuda_reserved_gib")), g(m.get("swap_gib"))))
        pk = r.get("peak") or {}
        if pk:
            L += ["", "Peaks: tree RSS %s GiB, VmHWM %s GiB, min MemAvailable %s GiB, outside-tree %s GiB, nvidia-smi %s of "
                  "%s GiB, torch reserved %s of %s GiB, swap %s GiB. RSS slope over fit steps: %s MiB / step (%d steps "
                  "logged)." % (g(pk["tree_rss_gib"]), g(pk["vmhwm_gib"]), g(pk["min_avail_gib"]), g(pk["max_nonproc_gib"]),
                                g(pk["gpu_used_gib"]), g(pk["gpu_total_gib"]), g(pk["cuda_reserved_gib"]),
                                g(pk["cuda_total_gib"]), g(pk["swap_used_gib"]), g(r.get("rss_mib_per_step"), 1),
                                r.get("n_steps_logged", 0))]
        if r.get("scans"):
            L += ["", "CPU tensor scans (live torch CPU storages):", ""]
            for s in r["scans"]:
                top = ", ".join("%s %s %s MB" % (t["shape"], t["dtype"].replace("torch.", ""), t["mb"]) for t in (s["top"] or [])[:4])
                L.append("- %s %s step %s: %.2f GiB in %s tensors (pinned %.2f GiB; RSS %.1f GiB); largest: %s" % (
                    s["event"], s["phase"], s["step"], s["gib"], s["n"], s["pinned_gib"], s["rss_gib"], top))
        L += [""] + ["- **%s**" % v for v in r["verdicts"]] + [""]
    L += ["Columns: RSS (driver) = the fit process's own VmRSS at that marker; tree RSS = the sum over its process tree "
          "(monitor, nearest 2 s sample); outside-tree = MemAvailable drop since launch minus the tree's RSS growth "
          "(memory the kernel / driver holds for no process: sysmem-fallback GPU pages would show here or in the "
          "process RSS, depending on how WSL maps them)."]
    out.write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(probes, indent=1, default=str))
    print("ram report -> %s (%d probes)" % (out, len(probes)))


if __name__ == "__main__":
    main()
