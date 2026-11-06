"""106 profiler: where does one production v3 fit spend its time?

Per target, two short fits on the 062 records (production v3 settings, the driver's Runner / method):
  timing   STEPS_T steps, no profiler, one cuda sync per optimiser step: setup / per-step / post wall time, peak memory
  profile  torch.profiler (CPU + CUDA, record_shapes, with_stack) over PROF_ACTIVE steps after PROF_WAIT steps, with
           record_function labels on every engine piece:
             106:T:<Z|C|A|R>   one term's forward (the engine's _forward_preact under grad)
             106:B:<Z|C|A|R>   that term's backward;  106:B:pen  the penalty backward;  106:O  optimiser step
             106:S:<piece>     inside a forward: encode, dense (sparse_topk_to_dense), keep (sparse/donor keep mask),
                               code (sparse_delta's code + STE, keep excluded), decode, donor (DonorPool.get), tap
Kernel time is attributed to (term, piece) by the label enclosing the launching op (forward), or by the autograd
node and the backward label's time window (backward runs on the autograd thread). Output:
  out/profile/<variant>/<key>.json  (summary)   <key>.txt (tables)   <key>.trace.json.gz (chrome trace, optional)

  PYTHONPATH=src python experiments/106-speed-profile/profile_fit.py
  env: KEYS (comma list, default the 104 panel)  VARIANT (comma fastlm flags, default none = production)
       STEPS_T (100)  PROF_WAIT (12)  PROF_ACTIVE (4)  TRACE (1 = export the chrome trace)
"""
import gzip
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h106  # noqa: E402

VARIANT = [f for f in os.environ.get("VARIANT", "").split(",") if f]
VNAME = "+".join(VARIANT) if VARIANT else "base"
KEYS = [k for k in os.environ.get("KEYS", ",".join(h106.TARGETS3)).split(",") if k]
STEPS_T = int(os.environ.get("STEPS_T", "100"))
PROF_WAIT = int(os.environ.get("PROF_WAIT", "12"))
PROF_ACTIVE = int(os.environ.get("PROF_ACTIVE", "4"))
TRACE = os.environ.get("TRACE", "1") == "1"
OUT = HERE / "out" / "profile" / VNAME
SCRATCH = h106.SCRATCH / ("prof_%s_%d" % (VNAME, int(time.time())))     # outside the repo (throw-away fits)

h106.set_env(SCRATCH, steps=STEPS_T)

import torch  # noqa: E402
from torch.autograd.profiler import record_function  # noqa: E402

_ANN = {"on": False, "pat_order": {}, "last_term": None, "bwd_done": set(), "t_loop0": None, "fwd_n": 0}


def _lbl(name):
    return record_function("106:" + name)


def install_annotations(mod, bank):
    """Wrap the engine module's pieces with record_function labels (and loop-start timing)."""
    orig_fp = mod._forward_preact

    def term_of(p):
        if getattr(p, "donors", None) is not None:
            return "R"
        if not p.floors:
            return "Z"
        o = _ANN["pat_order"]
        if id(p) not in o:
            o[id(p)] = "C" if "C" not in o.values() else "A"
        return o[id(p)]

    def fp(inference, patcher, tokens, grad):
        if not grad:
            with _lbl("T:setup"):
                return orig_fp(inference, patcher, tokens, grad)
        if _ANN["t_loop0"] is None:
            torch.cuda.synchronize()
            _ANN["t_loop0"] = time.perf_counter()
        t = term_of(patcher)
        _ANN["last_term"] = t
        _ANN["fwd_n"] += 1
        with _lbl("T:" + t):
            return orig_fp(inference, patcher, tokens, grad)
    mod._forward_preact = fp

    def wrap_fn(name, f):
        def g(*a, **k):
            with _lbl("S:" + name):
                return f(*a, **k)
        return g
    mod.sparse_topk_to_dense = wrap_fn("dense", mod.sparse_topk_to_dense)
    mod.sparse_keep_mask = wrap_fn("keep", mod.sparse_keep_mask)
    mod.donor_keep_mask = wrap_fn("keep", mod.donor_keep_mask)
    mod.DonorPool.get = wrap_fn("donor", mod.DonorPool.get)
    mod.LearnedMaskPatcher._sparse_delta = wrap_fn("code", mod.LearnedMaskPatcher._sparse_delta)
    enc, dec = bank.encode, bank.decode
    bank.encode = wrap_fn("encode", enc)
    bank.decode = wrap_fn("decode", dec)
    orig_tr = mod.LearnedMaskPatcher.transform

    def transform(self, layer_idx, kind, x):
        if self.tap_seed and layer_idx == self.seed_layer and kind == self.seed_kind:
            with _lbl("S:tap"):
                return orig_tr(self, layer_idx, kind, x)
        return orig_tr(self, layer_idx, kind, x)
    mod.LearnedMaskPatcher.transform = transform

    orig_bw = torch.Tensor.backward

    def backward(self, *a, **k):
        t = _ANN["last_term"]
        lab = "B:pen" if (t is None or t in _ANN["bwd_done"]) else "B:" + t
        if t is not None:
            _ANN["bwd_done"].add(t)
        if lab == "B:pen":
            _ANN["bwd_done"].clear()
            _ANN["last_term"] = None
        with _lbl(lab):
            return orig_bw(self, *a, **k)
    torch.Tensor.backward = backward

    from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
    _ANN["opt_rf"] = None

    def pre(opt, args, kwargs):
        rf = _lbl("O")
        rf.__enter__()
        _ANN["opt_rf"] = rf

    def post(opt, args, kwargs):
        rf = _ANN.pop("opt_rf", None)
        if rf is not None:
            rf.__exit__(None, None, None)
        _ANN["opt_rf"] = None
        st = _ANN.get("step_rf")
        if st is not None:
            st.__exit__(None, None, None)
        if _ANN.get("prof") is not None:
            _ANN["prof"].step()
        st = _lbl("step")
        st.__enter__()
        _ANN["step_rf"] = st
    register_optimizer_step_pre_hook(pre)
    register_optimizer_step_post_hook(post)


KCAT = [("gemm", re.compile(r"gemm|cutlass|xmma|nvjet|sm\d+_|cublas|Kernel2|splitK|matmul", re.I)),
        ("topk/sort", re.compile(r"topk|radix|bitonic|sort|select", re.I)),
        ("attention", re.compile(r"flash|fmha|attention", re.I)),
        ("memcpy", re.compile(r"memcpy", re.I)),
        ("memset", re.compile(r"memset|fill", re.I)),
        ("reduce", re.compile(r"reduce|sum|norm", re.I)),
        ("scatter/gather/index", re.compile(r"scatter|gather|index", re.I)),
        ("elementwise", re.compile(r"elementwise|vectorized|unrolled|functor|CatArray|copy|where", re.I))]


def kcat(name):
    for c, rx in KCAT:
        if rx.search(name):
            return c
    return "other"


def summarise(prof, n_active):
    evs = list(prof.events())
    labels = [e for e in evs if e.name.startswith("106:")]
    bwin = sorted((e.time_range.start, e.time_range.end, e.name[4:]) for e in labels
                  if e.name.startswith("106:B:") or e.name == "106:O")
    steps = [e for e in labels if e.name == "106:step"]
    wall_us = sum(e.time_range.end - e.time_range.start for e in steps)
    if steps:
        n_active = len(steps)                       # whole steps inside the active window

    def window_label(t):
        best = None
        for s, e, n in bwin:
            if s <= t <= e:
                best = n
            elif s > t:
                break
        return best

    def anc(e):
        term = sub = node = None
        p = e.cpu_parent
        while p is not None:
            n = p.name
            if sub is None and n.startswith("106:S:"):
                sub = n[6:]
            if term is None and (n.startswith("106:T:") or n.startswith("106:B:") or n == "106:O"):
                term = n[4:]
            if node is None and n.startswith("autograd::engine::evaluate_function: "):
                node = n.split(": ", 1)[1]
            p = p.cpu_parent
        return term, sub, node

    by_term, by_ts, by_cat, by_tcat, by_node = (defaultdict(float) for _ in range(5))
    n_kern = defaultdict(int)
    tot = 0.0
    memcpy = defaultdict(int)
    for e in evs:
        ks = getattr(e, "kernels", None) or []
        if not ks:
            continue
        term, sub, node = anc(e)
        if term is None:
            term = window_label(e.time_range.start) or "other"
        if term.startswith("B:") or term == "O":
            sub = sub or ("node:" + node if node else "op:" + e.name)
        sub = sub or ("op:" + e.name if term.startswith("T:") else "-")
        for k in ks:
            d = float(k.duration)
            c = kcat(k.name)
            tot += d
            by_term[term] += d
            by_ts[(term, sub)] += d
            by_cat[c] += d
            by_tcat[(term, c)] += d
            n_kern[term] += 1
            if c == "memcpy":
                memcpy[k.name] += 1
    api = defaultdict(int)
    for e in evs:
        if e.name in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync", "cudaMemcpy",
                      "aten::_local_scalar_dense", "aten::item", "aten::nonzero", "cudaLaunchKernel",
                      "cudaHostAlloc", "cudaEventSynchronize", "cuLaunchKernel", "cudaLaunchKernelExC"):
            api[e.name] += 1
    sync_stacks = defaultdict(int)
    for e in evs:
        if e.name in ("aten::_local_scalar_dense", "cudaStreamSynchronize") or (
                e.name == "aten::copy_" and any("Pageable" in k.name for k in (getattr(e, "kernels", None) or []))):
            st = [f for f in (e.stack or []) if "106-speed-profile" not in f and "torch/" not in f][:3]
            sync_stacks[(e.name, " | ".join(st))] += 1
    per = float(max(n_active, 1))
    r = lambda d: {k if isinstance(k, str) else "%s/%s" % k: round(v / per / 1e3, 3)
                   for k, v in sorted(d.items(), key=lambda kv: -kv[1])}
    return dict(n_active=n_active, wall_ms_per_step=round(wall_us / per / 1e3, 3),
                kernel_ms_per_step=round(tot / per / 1e3, 3),
                gpu_busy_frac=round(tot / wall_us, 4) if wall_us else None,
                kernels_per_step={k: round(v / per, 1) for k, v in sorted(n_kern.items())},
                kernels_total_per_step=round(sum(n_kern.values()) / per, 1),
                ms_by_term=r(by_term), ms_by_category=r(by_cat), ms_by_term_category=r(by_tcat),
                ms_by_term_piece=r(by_ts),
                api_calls_per_step={k: round(v / per, 2) for k, v in sorted(api.items())},
                memcpy_kernels_per_step={k: round(v / per, 2) for k, v in memcpy.items()},
                sync_sites=[dict(op=k[0], stack=k[1], per_step=round(v / per, 2))
                            for k, v in sorted(sync_stacks.items(), key=lambda kv: -kv[1])[:25]])


def main():
    import fastlm
    OUT.mkdir(parents=True, exist_ok=True)
    R, M, driver = h106.runner()
    from config import config
    mod = fastlm.install(VARIANT)
    install_annotations(mod, R.G["bank"])
    lm = config.discovery.learned_mask
    for key in KEYS:
        js = OUT / ("%s.json" % key)
        if js.exists():
            print("skip", key, flush=True)
            continue
        res = dict(key=key, variant=VNAME, gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
        # ---- timing run
        lm.steps = STEPS_T
        _ANN["t_loop0"] = None
        _ANN["pat_order"] = {}
        sc = SCRATCH / key / "timing"
        sc.mkdir(parents=True, exist_ok=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        row = h106.fit_one(R, M, key, sc, "timing", sync_steps=True, save_params=False)
        t1 = time.perf_counter()
        ts = h106._OPT["t"]
        stp = sorted(b - a for a, b in zip(ts, ts[1:]))
        res["timing"] = dict(steps=STEPS_T, t_fit=row["t_fit"], peak_alloc_mb=row["peak_alloc_mb"],
                             peak_reserved_mb=row["peak_reserved_mb"], base_alloc_mb=row["base_alloc_mb"],
                             setup_s=round(_ANN["t_loop0"] - t0, 3) if _ANN["t_loop0"] else None,
                             step_s_median=round(stp[len(stp) // 2], 5), step_s_p10=round(stp[len(stp) // 10], 5),
                             step_s_p90=round(stp[(9 * len(stp)) // 10], 5),
                             post_s=round(t1 - ts[-1], 3), n=row["n"])
        tm = res["timing"]
        tm["est_400_fit_s"] = round(tm["setup_s"] + 400 * tm["step_s_median"] + tm["post_s"], 2)
        print(key, "timing", json.dumps(tm), flush=True)
        # ---- profile run
        lm.steps = PROF_WAIT + 1 + PROF_ACTIVE + 1
        _ANN["t_loop0"] = None
        _ANN["pat_order"] = {}
        sc = SCRATCH / key / "profile"
        sc.mkdir(parents=True, exist_ok=True)
        sched =torch.profiler.schedule(wait=PROF_WAIT, warmup=1, active=PROF_ACTIVE, repeat=1)
        holder = {}
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    schedule=sched, record_shapes=True, with_stack=True,
                                    on_trace_ready=lambda p: holder.__setitem__("p", p)) as prof:
            _ANN["prof"] = prof
            h106.fit_one(R, M, key, sc, "profile", sync_steps=False, save_params=False)
            _ANN["prof"] = None
        p = holder.get("p", prof)
        res["profile"] = summarise(p, PROF_ACTIVE)
        tbl = p.key_averages().table(sort_by="self_device_time_total", row_limit=60)
        tbl2 = p.key_averages(group_by_input_shape=True).table(sort_by="self_device_time_total", row_limit=40)
        (OUT / ("%s.txt" % key)).write_text(tbl + "\n\n" + tbl2)
        if TRACE:
            tp = SCRATCH / key / "trace.json"                         # raw trace outside the repo, gzip copy in out/
            p.export_chrome_trace(str(tp))
            with open(tp, "rb") as fi, gzip.open(str(OUT / ("%s.trace.json.gz" % key)), "wb") as fo:
                shutil.copyfileobj(fi, fo)
        js.write_text(json.dumps(res, indent=1))
        pr = res["profile"]
        print(key, "profile wall %.2f ms/step, kernels %.2f ms/step, busy %.2f, %d kernels/step" % (
            pr["wall_ms_per_step"], pr["kernel_ms_per_step"], pr["gpu_busy_frac"] or -1,
            pr["kernels_total_per_step"]), flush=True)
        print("  by term", pr["ms_by_term"], flush=True)
        print("  by cat", pr["ms_by_category"], flush=True)


if __name__ == "__main__":
    main()
