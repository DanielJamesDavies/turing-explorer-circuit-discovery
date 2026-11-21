"""Exp 128 ram_probe: arena A's WCM driver (107/arena_a/driver_tcz.py -> 062 driver.py, both UNCHANGED) with phase and
memory markers, so the host-RAM curve of one deep fit can be read against what the fit is doing.

  (env as 062/launch.sh sets it for one shard, plus PROBE_STATUS=<jsonl>) python -X utf8 probe_driver.py

What it adds (observation only; no result changes):
  - observability.phases.phase is wrapped BEFORE anything imports it (learned_mask.py binds it at import), so every
    fit step (the "fit.opt" phase, once per optimizer step) is counted;
  - driver.Runner.__init__ / contexts / fit / score are wrapped with enter / exit markers;
  - each marker (and every PROBE_STEP_EVERY [5]-th step) writes one JSON line to PROBE_STATUS: time, event, phase,
    step, this process's VmRSS / VmHWM / VmSwap (kB), torch.cuda memory_allocated / memory_reserved /
    max_memory_reserved (bytes), and the device's total memory;
  - at runner_init exit, contexts exit, fit exit and at the steps in PROBE_SCAN_STEPS ["1,50,200"], a scan of the live
    CPU tensors (gc; storages counted once): total bytes, pinned bytes, count, the 6 largest (shape, dtype, MB).
    A scan pauses the fit for a few seconds.
Then driver_tcz's own __main__ block is replayed (its log line + driver.main()).
"""
import gc
import json
import os
import runpy
import sys
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DRIVER_TCZ = ROOT / "experiments" / "107-true-baselines" / "arena_a" / "driver_tcz.py"
sys.path.insert(0, str(ROOT / "src"))

STATUS_PATH = os.environ.get("PROBE_STATUS")
EVERY = max(1, int(os.environ.get("PROBE_STEP_EVERY", "5")))
SCAN_STEPS = {int(x) for x in os.environ.get("PROBE_SCAN_STEPS", "1,50,200").split(",") if x.strip()}
_fh = open(STATUS_PATH, "a", buffering=1) if STATUS_PATH else None
S = dict(phase="setup", step=0, fit_i=0, key=None)


def proc_status():
    out = {}
    try:
        for line in open("/proc/self/status"):
            k = line.split(":", 1)[0]
            if k in ("VmRSS", "VmHWM", "VmSwap"):
                out[k] = int(line.split()[1])
    except OSError:
        pass
    return out


def cuda_mem():
    try:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return {}
        return dict(cuda_alloc=torch.cuda.memory_allocated(), cuda_reserved=torch.cuda.memory_reserved(),
                    cuda_max_reserved=torch.cuda.max_memory_reserved(),
                    cuda_total=torch.cuda.get_device_properties(0).total_memory)
    except Exception:  # noqa: BLE001
        return {}


def cpu_tensor_scan():
    """Live CPU tensors (storages counted once): total / pinned bytes, count, the 6 largest."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return {}
    seen, tot, pin, n, big = set(), 0, 0, 0, []
    t0 = time.time()
    for o in gc.get_objects():
        try:
            if not torch.is_tensor(o) or o.device.type != "cpu":
                continue
            st = o.untyped_storage()
            p = st.data_ptr()
            if p in seen or p == 0:
                continue
            seen.add(p)
            b = st.nbytes()
            n += 1
            tot += b
            if o.is_pinned():
                pin += b
            big.append((b, list(o.shape), str(o.dtype)))
        except Exception:  # noqa: BLE001
            continue
    big.sort(key=lambda x: -x[0])
    return dict(cpu_tensor_bytes=tot, cpu_pinned_bytes=pin, cpu_tensor_n=n, scan_s=round(time.time() - t0, 2),
                cpu_top=[dict(mb=round(b / 2 ** 20, 1), shape=s, dtype=d) for b, s, d in big[:6]])


def emit(event, scan=False, **kw):
    if _fh is None:
        return
    row = dict(t=round(time.time(), 2), event=event, phase=S["phase"], step=S["step"], fit_i=S["fit_i"], key=S["key"])
    row.update(proc_status())
    row.update(cuda_mem())
    row.update(kw)
    if scan:
        row.update(cpu_tensor_scan())
    _fh.write(json.dumps(row) + "\n")


import observability.phases as _PH  # noqa: E402

_orig_phase = _PH.phase


@contextmanager
def _phase(name):
    with _orig_phase(name):
        yield
    if name == "fit.opt":
        S["step"] += 1
        st = S["step"]
        if st in SCAN_STEPS:
            emit("step", scan=True)
        elif st == 1 or st % EVERY == 0:
            emit("step")


_PH.phase = _phase


def wrap(cls, meth, label, scan_exit=False):
    f = getattr(cls, meth)

    def g(self, *a, **k):
        prev = S["phase"]
        S["phase"] = label
        if label == "fit":
            S["fit_i"] += 1
            S["step"] = 0
        if label in ("contexts", "fit") and a:
            S["key"] = a[0] if isinstance(a[0], str) else (a[2] if label == "fit" and len(a) > 2 else S["key"])
        emit("enter")
        try:
            return f(self, *a, **k)
        finally:
            emit("exit", scan=scan_exit)
            S["phase"] = "after_" + label if label != "score" else prev
    setattr(cls, meth, g)


def np_scan():
    """Live numpy arrays that own their memory: total bytes, count, the 4 largest."""
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return {}
    tot, n, big = 0, 0, []
    for o in gc.get_objects():
        try:
            if isinstance(o, np.ndarray) and o.base is None:
                tot += o.nbytes
                n += 1
                big.append((o.nbytes, list(o.shape), str(o.dtype)))
        except Exception:  # noqa: BLE001
            continue
    big.sort(key=lambda x: -x[0])
    return dict(np_bytes=tot, np_n=n, np_top=[dict(mb=round(b / 2 ** 20, 1), shape=s, dtype=d) for b, s, d in big[:4]])


def rss_watch():
    """2026-11-18: the 11-17 probes were killed inside fit set-up before step 1 (2 -> 27 GiB), where no scan ran.
    Each time VmRSS first crosses a mark in PROBE_RSS_MARKS_GIB, write the main thread's Python stack (the line that is
    allocating) plus CPU tensor and numpy scans."""
    import threading
    import traceback
    marks = sorted(float(x) for x in os.environ.get("PROBE_RSS_MARKS_GIB", "4,8,12,16,20,24").split(",") if x.strip())
    main_id = threading.main_thread().ident

    def run():
        i = 0
        while i < len(marks):
            rss = proc_status().get("VmRSS", 0) / 2 ** 20
            if rss >= marks[i]:
                fr = sys._current_frames().get(main_id)
                stack = [dict(file=f.filename.split("/src/")[-1].split("/experiments/")[-1], line=f.lineno, fn=f.name,
                              code=(f.line or "")[:160]) for f in traceback.extract_stack(fr)[-25:]] if fr else []
                row = dict(mark_gib=marks[i], rss_gib=round(rss, 2), stack=stack)
                row.update(np_scan())
                emit("rss_mark", scan=True, **row)
                while i < len(marks) and rss >= marks[i]:
                    i += 1
            time.sleep(0.25)

    threading.Thread(target=run, name="rss_watch", daemon=True).start()


emit("start", argv=sys.argv[1:], pid=os.getpid())
if _fh is not None:
    rss_watch()
# driver_tcz's module body (family registration, the fp32 + interleave + AA_RANK_W patches, import driver) without
# its __main__ block; the block is replayed below
G = runpy.run_path(str(DRIVER_TCZ), run_name="driver_tcz_probe")
driver = sys.modules["driver"]
wrap(driver.Runner, "__init__", "runner_init", scan_exit=True)
wrap(driver.Runner, "contexts", "contexts", scan_exit=True)
wrap(driver.Runner, "fit", "fit", scan_exit=True)
wrap(driver.Runner, "score", "score")
emit("imported")
WT = G["WT"]
driver.log("arena A (exp 128 probe_driver): family %s, scope %s, BOS %s, matmul precision after setup: %s" % (
    WT.FAMILY, WT.scope(), WT.bos_prefix(), "TF32 (ARENA_A_TF32=1)" if os.environ.get("ARENA_A_TF32") == "1"
    else "highest"))
try:
    driver.main()
finally:
    emit("end")
