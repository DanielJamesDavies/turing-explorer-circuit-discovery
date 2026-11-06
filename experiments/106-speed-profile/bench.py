"""106 bench + equivalence: full 400-step production v3 fits of the 104 panel under each prototype variant.

Per variant (VARIANTS, ';'-separated, each a ','-list of fastlm flags; 'base' = production, unchanged) and target:
fit -> out/bench/<variant>/circuits/<key>.pt, final raw parameters -> params/<key>.pt, a row in fits.jsonl (wall
t_fit, peak memory, loss curve). Then compare.json: each variant vs base (same machine, same process) and vs the 104
bit-identity circuits (104 out/bitident: production v3, which matched 102 bit for bit):
  circuit identical (members + 4-dp amplitudes), members identical, Jaccard, max |alpha diff|,
  raw final thetas / psis max |diff| and how many tensors are exactly equal, loss curve identical.
Resumable: a (variant, key) with a row is skipped.

  PYTHONPATH=src python experiments/106-speed-profile/bench.py
  env: VARIANTS (default "base;early_exit,donor_gpu,rank_nosync;early_exit;donor_gpu,rank_nosync;base#2;
       early_exit,donor_gpu,rank_nosync,triton_topk")  KEYS (default the 104 panel)
  (a '#n' suffix names a repeat: same flags, separate run, for timing noise / run-to-run determinism)
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h106  # noqa: E402

DEFAULT = ("base;early_exit,donor_gpu,rank_nosync;early_exit,donor_gpu,rank_nosync,batch4;"
           "early_exit,donor_gpu,rank_nosync,no_cfeval;early_exit,donor_gpu,rank_nosync,compile_sp;"
           "early_exit,donor_gpu,rank_nosync,triton_topk;early_exit,donor_gpu,rank_nosync,batch2;"
           "early_exit;donor_gpu,rank_nosync;base#2")
VARIANTS = [v for v in os.environ.get("VARIANTS", DEFAULT).split(";") if v]
KEYS = [k for k in os.environ.get("KEYS", ",".join(h106.TARGETS3)).split(",") if k]
OUT = HERE / "out" / "bench"


def vname(v):
    return v.replace(",", "+")


METHOD_FLAGS = ("no_cfeval",)     # discovery-method patches (not engine): no_cfeval = skip discover()'s post-fit
#                                   counterfactual-faithfulness eval (mask modes never reject on it; it only fills
#                                   circuit.metadata counterfactual_faithfulness / posctx_suppression_score)


def flags_of(v):
    base = v.split("#")[0]
    return [] if base == "base" else [f for f in base.split(",") if f and f not in METHOD_FLAGS]


def method_flags_of(v):
    return [f for f in v.split("#")[0].split(",") if f in METHOD_FLAGS]


def set_method_flags(M, flags):
    if "no_cfeval" in flags:
        M._run_faithfulness_eval = lambda ctx, circuit, neg, layers: (float("nan"), float("nan"))
    elif "_run_faithfulness_eval" in M.__dict__:
        del M.__dict__["_run_faithfulness_eval"]


def _rows(d):
    p = Path(d) / "fits.jsonl"
    out = []
    if p.exists():
        for line in open(p):
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return out


def compare_all():
    import torch
    res = {}
    rows = {}
    for v in VARIANTS:
        p = OUT / vname(v) / "fits.jsonl"
        if p.exists():
            for line in open(p):
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                rows[(v, r["key"])] = r
    for v in VARIANTS:
        for k in KEYS:
            r = rows.get((v, k))
            if r is None:
                continue
            if "error" in r:
                res["%s|%s" % (vname(v), k)] = dict(error=r["error"])
                continue
            e = dict(t_fit=r["t_fit"], n=r["n"], peak_alloc_mb=r["peak_alloc_mb"],
                     peak_reserved_mb=r["peak_reserved_mb"])
            b = rows.get(("base", k))
            c_v = OUT / vname(v) / "circuits" / ("%s.pt" % k)
            if b is not None and v != "base":
                e["speedup_vs_base"] = round(b["t_fit"] / r["t_fit"], 4)
                e["vs_base"] = h106.compare_to(c_v, OUT / "base" / "circuits" / ("%s.pt" % k))
                pp, pb = OUT / vname(v) / "params" / ("%s.pt" % k), OUT / "base" / "params" / ("%s.pt" % k)
                if pp.exists() and pb.exists():
                    e["params_vs_base"] = h106.compare_params(pp, pb)
                e["loss_curve_identical"] = r.get("loss_curve") == b.get("loss_curve")
                if r.get("loss_curve") and b.get("loss_curve"):
                    e["loss_curve_max_abs_diff"] = max(abs(x - y) for x, y in zip(r["loss_curve"], b["loss_curve"]))
            ref = h106.REF104 / ("%s.pt" % k)
            if ref.exists() and c_v.exists():
                e["vs_104"] = h106.compare_to(c_v, ref)
            res["%s|%s" % (vname(v), k)] = e
    (OUT / "compare.json").write_text(json.dumps(res, indent=1))
    for kk, e in res.items():
        if "error" in e:
            print("%-60s ERROR %s" % (kk, e["error"][:120]), flush=True)
            continue
        vb, v104 = e.get("vs_base") or {}, e.get("vs_104") or {}
        print("%-60s t %7.1fs  x%-6s n %-5s  ==base %-5s jac %-7s  ==104 %-5s  dtheta %s  curve== %s" % (
            kk, e["t_fit"], e.get("speedup_vs_base", "-"), e["n"], vb.get("identical"), vb.get("jaccard"),
            v104.get("identical"), (e.get("params_vs_base") or {}).get("max_theta_diff"),
            e.get("loss_curve_identical")), flush=True)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("COMPARE_ONLY") == "1":
        h106.set_env(OUT / "_runner")
        compare_all()
        return
    h106.set_env(OUT / "_runner")
    R, M, driver = h106.runner()
    import fastlm
    from config import config
    lm = config.discovery.learned_mask
    # warm-up (kernel autotune / cuBLAS / allocator) so the first timed fit is not charged for it
    lm.steps = 20
    h106.fit_one(R, M, KEYS[0], h106.SCRATCH / ("bench_warmup_%d" % int(time.time())), "warmup", save_params=False)
    lm.steps = 400
    for v in VARIANTS:
        d = OUT / vname(v)
        d.mkdir(parents=True, exist_ok=True)
        fastlm.install(flags_of(v))
        set_method_flags(M, method_flags_of(v))
        if not all(any(r.get("key") == k and "error" not in r for r in _rows(d)) for k in KEYS):
            lm.steps = 10                                  # per-variant warm-up (compile / autotune not charged)
            h106.fit_one(R, M, KEYS[0], h106.SCRATCH / ("bench_warm_%s_%d" % (vname(v), int(time.time()))),
                         "warmup", save_params=False)
            lm.steps = 400
        for k in KEYS:
            t = time.time()
            try:
                row = h106.fit_one(R, M, k, d, vname(v), sync_steps=False, save_params=True)
            except Exception as e:  # noqa: BLE001  (e.g. OOM of a batched variant at depth): logged, next
                import torch
                err = dict(key=k, label=vname(v), error="%s: %s" % (type(e).__name__, str(e)[:300]))
                with open(d / "fits.jsonl", "a") as fh:
                    fh.write(json.dumps(err) + "\n")
                print("[%s] %-40s %-16s ERROR %s" % (time.strftime("%H:%M:%S"), vname(v), k, err["error"]), flush=True)
                torch.cuda.empty_cache()
                continue
            print("[%s] %-40s %-16s n=%s t_fit %.1fs (%.0fs wall incl. resume check) peak %.0f MB" % (
                time.strftime("%H:%M:%S"), vname(v), k, row["n"], row["t_fit"], time.time() - t,
                row["peak_reserved_mb"]), flush=True)
    fastlm.install(())
    set_method_flags(M, ())
    compare_all()


if __name__ == "__main__":
    main()
