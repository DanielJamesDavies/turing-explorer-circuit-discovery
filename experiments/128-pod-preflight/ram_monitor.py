"""Exp 128 ram_probe: sample one job's memory every --every s into a CSV, and kill the job (its process group) when host
MemAvailable < --min-avail-mib, GPU memory.used > --gpu-max-mib, or it runs longer than --max-s. CPU only; reads
/proc and nvidia-smi, never the GPU itself.

  python -I ram_monitor.py --pid <process-group leader> --csv OUT.csv [--status probe.jsonl] [--log driver.log]
         [--every 2] [--min-avail-mib 3072] [--gpu-max-mib 15300] [--max-s 0] [--kill-file KILLED]

Columns: t, elapsed_s, n_proc, tree_rss_mib (sum of VmRSS over the process tree), tree_hwm_mib (max VmHWM), top_pid,
top_rss_mib, top_anon_mib / top_file_mib / top_shmem_mib (smaps_rollup of the largest process), top_swap_mib,
mem_avail_mib, mem_free_mib, cached_mib, shmem_mib, anon_pages_mib, mapped_mib, unevict_mib, mlocked_mib, swap_used_mib,
committed_mib, gpu_used_mib, gpu_total_mib, gpu_proc_mib (nvidia-smi compute-apps for the tree; often empty on WSL),
phase, step, fit_i, key, cuda_alloc_mib, cuda_reserved_mib (the driver's own last status line), log_tail, event.
Exit: 0 the job ended by itself; 9 killed (reason in --kill-file and the last CSV row).
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import time

COLS = ["t", "elapsed_s", "n_proc", "tree_rss_mib", "tree_hwm_mib", "top_pid", "top_rss_mib", "top_anon_mib",
        "top_file_mib", "top_shmem_mib", "top_swap_mib", "mem_avail_mib", "mem_free_mib", "cached_mib", "shmem_mib",
        "anon_pages_mib", "mapped_mib", "unevict_mib", "mlocked_mib", "swap_used_mib", "committed_mib", "gpu_used_mib",
        "gpu_total_mib", "gpu_proc_mib", "phase", "step", "fit_i", "key", "cuda_alloc_mib", "cuda_reserved_mib",
        "log_tail", "event"]


def kb_fields(path, keys):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                k = line.split(":", 1)[0]
                if k in keys:
                    out[k] = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return out


def tree(root):
    """root and every descendant (from /proc/<pid>/stat ppid)."""
    kids = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % d) as f:
                s = f.read()
            ppid = int(s[s.rfind(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        if p in out:
            continue
        out.append(p)
        todo += kids.get(p, [])
    return out


def nvsmi():
    used = total = None
    apps = {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        a, b = r.stdout.strip().splitlines()[0].split(",")
        used, total = int(a), int(b)
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        for line in r.stdout.strip().splitlines():
            try:
                p, m = line.split(",")
                apps[int(p)] = int(m)
            except ValueError:
                continue
    except Exception:  # noqa: BLE001
        pass
    return used, total, apps


def last_line(path, maxb=65536):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - maxb))
            lines = [x for x in f.read().decode("utf-8", "replace").splitlines() if x.strip()]
        return lines[-1] if lines else ""
    except OSError:
        return ""


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                                   # a zombie (exited, not yet reaped) counts as ended
        with open("/proc/%d/stat" % pid) as f:
            s = f.read()
        return s[s.rfind(")") + 2:].split()[0] != "Z"
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--status", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--min-avail-mib", type=int, default=3072)
    ap.add_argument("--gpu-max-mib", type=int, default=15300)
    ap.add_argument("--max-s", type=float, default=0)
    ap.add_argument("--kill-file", default="")
    ap.add_argument("--kill-grace", type=float, default=15)
    a = ap.parse_args()
    t0 = time.time()
    new = not os.path.exists(a.csv)
    fh = open(a.csv, "a", newline="", buffering=1)
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()
    killed = None
    mib = 1024.0
    while alive(a.pid):
        pids = tree(a.pid)
        rss_tot, hwm, top = 0, 0, (None, 0)
        for p in pids:
            st = kb_fields("/proc/%d/status" % p, ("VmRSS", "VmHWM", "VmSwap"))
            r = st.get("VmRSS", 0)
            rss_tot += r
            hwm = max(hwm, st.get("VmHWM", 0))
            if r > top[1]:
                top = (p, r, st.get("VmSwap", 0))
        sm = kb_fields("/proc/%d/smaps_rollup" % top[0], ("Anonymous", "Rss", "Shared_Clean", "Shared_Dirty",
                                                          "Private_Clean", "Private_Dirty")) if top[0] else {}
        st_top = kb_fields("/proc/%d/status" % top[0], ("RssAnon", "RssFile", "RssShmem")) if top[0] else {}
        mi = kb_fields("/proc/meminfo", ("MemAvailable", "MemFree", "Cached", "Shmem", "AnonPages", "Mapped",
                                         "Unevictable", "Mlocked", "SwapTotal", "SwapFree", "Committed_AS"))
        used, total, apps = nvsmi()
        stl = {}
        if a.status:
            try:
                stl = json.loads(last_line(a.status) or "{}")
            except ValueError:
                stl = {}
        row = dict(
            t=round(time.time(), 1), elapsed_s=round(time.time() - t0, 1), n_proc=len(pids),
            tree_rss_mib=round(rss_tot / mib), tree_hwm_mib=round(hwm / mib), top_pid=top[0],
            top_rss_mib=round(top[1] / mib),
            top_anon_mib=round(st_top.get("RssAnon", sm.get("Anonymous", 0)) / mib),
            top_file_mib=round(st_top.get("RssFile", 0) / mib), top_shmem_mib=round(st_top.get("RssShmem", 0) / mib),
            top_swap_mib=round(top[2] / mib),
            mem_avail_mib=round(mi.get("MemAvailable", 0) / mib), mem_free_mib=round(mi.get("MemFree", 0) / mib),
            cached_mib=round(mi.get("Cached", 0) / mib), shmem_mib=round(mi.get("Shmem", 0) / mib),
            anon_pages_mib=round(mi.get("AnonPages", 0) / mib), mapped_mib=round(mi.get("Mapped", 0) / mib),
            unevict_mib=round(mi.get("Unevictable", 0) / mib), mlocked_mib=round(mi.get("Mlocked", 0) / mib),
            swap_used_mib=round((mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) / mib),
            committed_mib=round(mi.get("Committed_AS", 0) / mib), gpu_used_mib=used, gpu_total_mib=total,
            gpu_proc_mib=sum(apps.get(p, 0) for p in pids) if apps else "",
            phase=stl.get("phase", ""), step=stl.get("step", ""), fit_i=stl.get("fit_i", ""), key=stl.get("key", ""),
            cuda_alloc_mib=round(stl["cuda_alloc"] / 2 ** 20) if "cuda_alloc" in stl else "",
            cuda_reserved_mib=round(stl["cuda_reserved"] / 2 ** 20) if "cuda_reserved" in stl else "",
            log_tail=(last_line(a.log)[:160].replace(",", ";") if a.log else ""), event="")
        avail = row["mem_avail_mib"]
        if mi.get("MemAvailable") is not None and avail < a.min_avail_mib:
            killed = "host MemAvailable %d MiB < %d MiB" % (avail, a.min_avail_mib)
        elif used is not None and used > a.gpu_max_mib:
            killed = "GPU memory.used %d MiB > %d MiB" % (used, a.gpu_max_mib)
        elif a.max_s and time.time() - t0 > a.max_s:
            killed = "time cap %.0f s reached" % a.max_s
        if killed:
            row["event"] = "KILL: " + killed
        w.writerow(row)
        if killed:
            break
        time.sleep(a.every)
    if killed:
        if a.kill_file:
            with open(a.kill_file, "w") as f:
                f.write("%s %s (pid %d, phase %s, step %s)\n" % (time.strftime("%F %T"), killed, a.pid,
                                                                 row.get("phase"), row.get("step")))
        for sig, wait in ((signal.SIGTERM, a.kill_grace), (signal.SIGKILL, 0)):
            try:
                os.killpg(a.pid, sig)
            except OSError:
                pass
            t1 = time.time()
            while wait and time.time() - t1 < wait and alive(a.pid):
                time.sleep(0.5)
        w.writerow(dict(t=round(time.time(), 1), elapsed_s=round(time.time() - t0, 1), event="killed: " + killed))
        fh.close()
        raise SystemExit(9)
    fh.close()


if __name__ == "__main__":
    main()
