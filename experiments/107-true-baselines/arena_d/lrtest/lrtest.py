"""Arena D lr-budget test (2026-11-06, agent dlrtest): does v4's optimisation budget (peak lr x3.15, warm-up 0.1,
geometric) help arena D's behavioural WCM? Experiment-local wrapper; nothing under src/, experiments/081-task-circuits/
or the production arena D files is modified. See README.md (question, design, pre-stated reading).

How the arms are applied (no engine edit):
  * one wcm_gemma.make_engine(...) per family and process (the production GemmaEngine, V3_LOGIT = arena C's behavioural
    v3: sparse fill + STE, R 1.0 k 1, rank-keep 0, geometric lr, 081 WCM_VERSION=v2);
  * the instance's engine_kwargs is wrapped: the production kwargs are built unchanged, then the arm's overrides are
    applied (C0: none but an explicit weight_decay 0.05, = 081's weight_decay(0.05); C1: warmup_frac 0.1 and
    weight_decay 0.05, v4's absolute decay, NOT 081's lr-rescaled one); the peak lr goes in through wcm_gemma.run(lr=);
  * each arm writes to its own --out (wcm_gemma.run's resume keys do not carry the lr), so circuits never mix.

Commands (WSL, repo root, .venv):
  CUDA_VISIBLE_DEVICES= ./.venv/bin/python <this> check      # CPU: imports, engine refusals, tiny CPU fits per arm
  CUDA_VISIBLE_DEVICES= ./.venv/bin/python <this> plan       # CPU: units, done state, budget projection
  ./.venv/bin/python <this> run --stage res_main|plt_main|res_seed2   # GPU (run.sh calls it; one process at a time)
env: STOP_EPOCH (no new unit after it), LRTEST_BUDGET_S (10800: GPU wall incl. loads, from out/ledger.jsonl),
     LRTEST_OUT (default <here>/out), LRTEST_DTYPE (bfloat16: the 16 GB card; the pod runs fp32 google weights)
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AD = HERE.parent
if str(AD) not in sys.path:
    sys.path.insert(0, str(AD))
OUT = Path(os.environ.get("LRTEST_OUT") or (HERE / "out"))
MEANS = AD / "out" / "caples_means" / "rc_gap6_bf16_SMOKE.pt"     # the local bf16 Caples means file (pilot step 2)
STEPS = 400                       # pod_stage_d.sh DW_STEPS (arena D's WCM on the pod), not the smoke's 50
BASE_LR = 0.05                    # 081 task_common.PROD_LR (asserted in check / run)
PEAK_MULT = 3.15                  # 062 driver.WCM_LR_MULT["v4"]
WARMUP = 0.1                      # 062 driver.WCM_VERSIONS["v4"]["warmup_frac"]
WD = 0.05                         # v4 keeps the config's absolute AdamW decay at the x3.15 peak (spec §3)
R_SEED2 = 20261018                # second R-donor seed (production 20261017)
S_PER_STEP_PRIOR = 0.65           # the Thursday pilot: 0.64 s / step on both families (bf16, 5070 Ti)
LOAD_PRIOR_S = 180.0              # pilot: model + dictionaries load, task build, means
SCORE_PRIOR_S = 15.0
BUDGET_S = float(os.environ.get("LRTEST_BUDGET_S") or 3 * 3600)

ARMS = {
    "C0": dict(lr=BASE_LR, kw=dict(weight_decay=WD),
               desc="current arena D WCM as the pod runs it (v2 + R, geometric, peak 0.05, no warm-up)"),
    "C1": dict(lr=BASE_LR * PEAK_MULT, kw=dict(warmup_frac=WARMUP, weight_decay=WD),
               desc="C0 + v4's lr budget (geometric, peak x3.15 = 0.1575, warm-up 0.1, wd 0.05)"),
    "C0s2": dict(lr=BASE_LR, kw=dict(weight_decay=WD, resample_seed=R_SEED2),
                 desc="C0 with R-donor seed %d (replicate)" % R_SEED2),
    "C1s2": dict(lr=BASE_LR * PEAK_MULT, kw=dict(warmup_frac=WARMUP, weight_decay=WD, resample_seed=R_SEED2),
                 desc="C1 with R-donor seed %d (replicate)" % R_SEED2),
}
# the smoke's wcm_plt / wcm_res steps (run_pilot_d.sh): same families, data, means, substrate and prompt lengths
FAMS = {
    "res": dict(family="gemma-2-2b-gsres-gap6", data="caples", mode="caples", means="per_position"),
    "plt": dict(family="gemma-2-2b-plt", data="sfc", mode="native", means="pooled"),
}
LAMS = (1e-3, 1e-4, 1e-2)         # three points of the pod's 9-lambda grid, 1e-3 (the smoke's) first
STAGES = {                        # in priority order inside each stage; run.sh runs the stages in STAGE_ORDER
    "res_main": ("res", [(a, "rc", l) for l in LAMS for a in ("C0", "C1")]),
    "plt_main": ("plt", [(a, t, l) for t in ("rc", "simple", "nounpp") for l in LAMS for a in ("C0", "C1")]),
    "res_seed2": ("res", [(a, "rc", 1e-3) for a in ("C0s2", "C1s2")]),
}
STAGE_ORDER = ("res_main", "plt_main", "res_seed2")


def say(*a):
    print("[lrtest %s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def unit_out(arm: str, fam: str) -> Path:
    return OUT / "runs" / arm / fam


def done_rows(arm: str, fam: str):
    p = unit_out(arm, fam) / "eval.jsonl"
    rows = []
    if p.exists():
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001  (a truncated last line after a kill)
                pass
    return rows


def is_done(arm: str, fam: str, task: str, lam: float) -> bool:
    for r in done_rows(arm, fam):
        if (r.get("family") == FAMS[fam]["family"] and r.get("sfc_task") == task and float(r["lam"]) == float(lam)
                and int(r["steps"]) == STEPS):
            return True
    return False


def ledger():
    p = OUT / "ledger.jsonl"
    out = []
    if p.exists():
        for line in open(p):
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return out


def ledger_add(**row):
    OUT.mkdir(parents=True, exist_ok=True)
    row["epoch"] = round(time.time(), 1)
    with open(OUT / "ledger.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")


def est_unit(fam: str) -> float:
    """Wall seconds of one fit + score on this family: the largest measured so far, else the pilot's prior."""
    w = [r["wall"] for r in ledger() if r.get("kind") == "unit" and r.get("fam") == fam]
    return max(w) if w else STEPS * S_PER_STEP_PRIOR + SCORE_PRIOR_S


def est_load() -> float:
    w = [r["wall"] for r in ledger() if r.get("kind") == "load"]
    return max(w) if w else LOAD_PRIOR_S


def spent() -> float:
    return sum(float(r.get("wall", 0)) for r in ledger())


def pending(stage: str):
    fam, units = STAGES[stage]
    return [u for u in units if not is_done(u[0], fam, u[1], u[2])]


def projection() -> dict:
    rem, n = 0.0, 0
    for st in STAGE_ORDER:
        p = pending(st)
        if p:
            rem += est_load() + len(p) * est_unit(STAGES[st][0])
            n += len(p)
    return dict(spent_s=round(spent()), remaining_units=n, remaining_s=round(rem),
                total_s=round(spent() + rem), budget_s=BUDGET_S)


def stop_epoch_passed() -> bool:
    s = os.environ.get("STOP_EPOCH")
    return bool(s) and time.time() >= float(s)


# ------------------------------------------------------------------------------------------------- arm wrapper
CUR: dict = {}


def wrap_engine(eng):
    """The arm overrides on top of the production engine_kwargs; one batch (and its means) per task for every arm."""
    orig_ek, orig_b = eng.engine_kwargs, eng.batch
    cache: dict = {}

    def engine_kwargs(B, *a, **kw):
        out = orig_ek(B, *a, **kw)
        if not CUR:
            raise RuntimeError("lrtest: no arm set")
        for k in ("lr_schedule", "rank_weight", "resample_weight", "resample_k", "train_fill", "sparse_grad"):
            if out.get(k) != CUR["prod"][k]:          # the production setting is what is built underneath
                raise RuntimeError("lrtest: production %s = %r, expected %r" % (k, out.get(k), CUR["prod"][k]))
        if "censor_temp" in out:
            raise RuntimeError("lrtest: censor_temp present in the production kwargs")
        out.update(CUR["kw"])
        return out

    def batch(task, contrast=None):
        k = task["name"]
        if k not in cache:
            cache[k] = orig_b(task, contrast)
        return cache[k]

    eng.engine_kwargs = engine_kwargs
    eng.batch = batch
    return eng


def set_arm(name: str, WM, base_v3: dict):
    a = ARMS[name]
    CUR.clear()
    CUR.update(a, name=name, prod=base_v3)
    # provenance only: wcm_gemma.run writes V3_LOGIT into WCM_VERSION and every circuit JSON (the engine's own copy
    # was taken at make_engine time, before this, and stays the production one)
    WM.V3_LOGIT.clear()
    WM.V3_LOGIT.update(base_v3)
    WM.V3_LOGIT.update(lrtest_arm=name, lrtest_lr_peak=a["lr"], **{"lrtest_" + k: v for k, v in a["kw"].items()})


def restore_v3(WM, base_v3: dict):
    WM.V3_LOGIT.clear()
    WM.V3_LOGIT.update(base_v3)


def write_arm_files():
    for arm, a in ARMS.items():
        for fam in FAMS:
            d = unit_out(arm, fam)
            if any(u[0] == arm and STAGES[s][0] == fam for s in STAGES for u in STAGES[s][1]):
                d.mkdir(parents=True, exist_ok=True)
                (d / "ARM.json").write_text(json.dumps(dict(arm=arm, lr_peak=a["lr"], overrides=a["kw"], desc=a["desc"],
                                                            steps=STEPS, **FAMS[fam]), indent=1) + "\n")


# ------------------------------------------------------------------------------------------------- run
def cmd_run(stage: str) -> int:
    import torch
    import wcm_gemma as W
    C, _ = W._import_081()
    assert C.WCM_VERSION == "v2", "081 WCM_VERSION must be v2 (as the pod pins it), got %s" % C.WCM_VERSION
    assert C.PROD_LR == BASE_LR and abs(C.weight_decay(BASE_LR) - WD) < 1e-12
    fam, _units = STAGES[stage]
    F = FAMS[fam]
    todo = pending(stage)
    if not todo:
        say("%s: every unit done" % stage)
        return 0
    if F["mode"] == "caples" and not MEANS.exists():
        say("MISSING the Caples means file %s" % MEANS)
        return 1
    pr = projection()
    say("%s: %d pending units; projection %s" % (stage, len(todo), pr))
    if stop_epoch_passed():
        say("STOP_EPOCH passed: no new unit")
        return 0
    if spent() + est_load() + est_unit(fam) > BUDGET_S:
        say("BUDGET: spent %.0f s + load + one unit > %.0f s: not started" % (spent(), BUDGET_S))
        return 0
    write_arm_files()
    WM = W._wcm_marks()
    base_v3 = dict(WM.V3_LOGIT)
    dtype = os.environ.get("LRTEST_DTYPE", "bfloat16") or None
    t0 = time.time()
    eng = W.make_engine(F["family"], F["mode"], F["means"], str(MEANS) if F["mode"] == "caples" else None,
                        dtype=dtype)
    wrap_engine(eng)
    load_s = time.time() - t0
    ledger_add(kind="load", stage=stage, fam=fam, wall=round(load_s, 1))
    say("%s: engine loaded in %.0f s (%s, %s)" % (stage, load_s, F["family"], dtype))
    first = True
    for arm, task, lam in todo:
        if stop_epoch_passed():
            say("STOP_EPOCH passed: no new unit")
            break
        if spent() + est_unit(fam) > BUDGET_S:
            say("BUDGET: spent %.0f s + est %.0f s > %.0f s: stopping before %s %s %g"
                % (spent(), est_unit(fam), BUDGET_S, arm, task, lam))
            break
        set_arm(arm, WM, base_v3)
        out = unit_out(arm, fam)
        say("unit %s %s %s lam %g (lr peak %g, overrides %s) -> %s" % (arm, F["family"], task, lam, ARMS[arm]["lr"],
                                                                         ARMS[arm]["kw"], out))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tu = time.time()
        try:
            rows = W.run([task], [lam], F["family"], F["data"], F["mode"], F["means"], STEPS, out,
                         lr=ARMS[arm]["lr"], eng=eng, log=lambda s: say("  " + s),
                         caples_means=str(MEANS) if F["mode"] == "caples" else None,
                         task_kw=dict(n_held=W.N_HELD_CAPLES) if F["data"] == "caples" else None)
        finally:
            restore_v3(WM, base_v3)
        wall = time.time() - tu
        if not rows:
            if is_done(arm, fam, task, lam):
                continue
            say("run returned nothing for %s %s %g (STOP_EPOCH?): stopping" % (arm, task, lam))
            break
        r = rows[0]
        ledger_add(kind="unit", stage=stage, fam=fam, arm=arm, task=task, lam=lam, wall=round(wall, 1),
                   t_fit=r.get("t_fit"), t_score=r.get("t_score"), s_per_step=round(r["t_fit"] / STEPS, 4)
                   if r.get("t_fit") else None, peak_fit_mb=r.get("peak_fit_mb"), peak_score_mb=r.get("peak_score_mb"),
                   n_latents=r.get("n_latents"))
        if first:
            first = False
            say("FIRST FIT %s: %.3f s / step (t_fit %.0f s, unit wall %.0f s, peak %s MB); projection %s"
                % (stage, r["t_fit"] / STEPS, r["t_fit"], wall, r.get("peak_fit_mb"), projection()))
    return 0


# ------------------------------------------------------------------------------------------------- plan
def cmd_plan() -> int:
    say("out %s | steps %d | budget %.0f s | dtype %s" % (OUT, STEPS, BUDGET_S, os.environ.get("LRTEST_DTYPE",
                                                                                              "bfloat16")))
    for arm, a in ARMS.items():
        say("arm %-5s lr peak %.4f overrides %s: %s" % (arm, a["lr"], a["kw"], a["desc"]))
    for st in STAGE_ORDER:
        fam, units = STAGES[st]
        p = set(pending(st))
        say("stage %s (%s: %s)" % (st, fam, FAMS[fam]))
        for u in units:
            say("   %-5s %-7s lam %-6g %s" % (u[0], u[1], u[2], "pending" if u in p else "done"))
    say("projection %s" % projection())
    return 0


# ------------------------------------------------------------------------------------------------- check (CPU)
def lr_sum(lr: float, steps: int, warmup: float, schedule: str = "geometric", min_frac: float = 0.05) -> float:
    """sum(lr) of learned_mask's lr_at (re-stated here; check compares it with a fit's own lr_sum)."""
    floor, n_w = lr * min_frac, int(round(warmup * steps))
    tot = 0.0
    for t in range(steps):
        if schedule == "constant":
            tot += lr
        elif t < n_w:
            tot += floor + (lr - floor) * (t + 1) / n_w
        else:
            tot += lr * min_frac ** ((t - n_w) / max(steps - n_w - 1, 1))
    return tot


def cmd_check() -> int:
    import torch
    fails = []

    def ok(cond, msg):
        say(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    ok(not torch.cuda.is_available(), "CPU only (CUDA not visible)")
    import wcm_gemma as W
    C, E = W._import_081()
    ok(C.WCM_VERSION == "v2", "081 WCM_VERSION = %s (v2 pinned, as the pod)" % C.WCM_VERSION)
    ok(C.PROD_LR == BASE_LR and abs(C.weight_decay(BASE_LR) - WD) < 1e-12,
       "081 PROD_LR %g, weight_decay(0.05) = %g" % (C.PROD_LR, C.weight_decay(BASE_LR)))
    ok(abs(C.weight_decay(BASE_LR * PEAK_MULT) - WD) > 1e-3,
       "081 would rescale wd to %.4f at lr 0.1575 (C1 overrides it to v4's absolute 0.05)"
       % C.weight_decay(BASE_LR * PEAK_MULT))
    WM = W._wcm_marks()
    v3 = dict(WM.V3_LOGIT)
    ok(v3.get("lr_schedule") == "geometric" and v3.get("rank_weight") == 0.0 and v3.get("resample_weight") == 1.0
       and "warmup_frac" not in v3 and "censor_temp" not in v3, "production V3_LOGIT %s" % v3)
    import circuit.instrument.learned_mask as LM
    src = inspect.getsource(LM)
    ok("censor_temp / rival_weight / rank_zero_mult need objective='pos'" in src,
       "engine refuses censor_temp for objective='logit' (C2 needs an engine change: skipped)")
    ok("offtarget_weight / rank_weight need objective='pos'" in src,
       "engine refuses rank_weight for objective='logit' (rank-keep skipped)")
    for k in ("warmup_frac", "weight_decay", "resample_seed", "lr", "lr_schedule"):
        ok(re_has_kw(src, k), "learned_mask accepts %s" % k)
    for name, fam in (("res", FAMS["res"]), ("plt", FAMS["plt"])):
        st = W.SPECS[fam["family"]]
        ok(st["state"] in (W.load_plt_state, W.load_res_state), "%s dictionaries load as JumpReLUSAE (%s)"
           % (fam["family"], st["state"].__name__))
    ok(MEANS.exists(), "Caples means file %s" % MEANS)
    for arm in ("C0", "C1"):
        a = ARMS[arm]
        s = lr_sum(a["lr"], STEPS, a["kw"].get("warmup_frac", 0.0))
        say("     %s at %d steps: sum(lr) %.2f, decay budget sum(lr) x wd %.3f, sparsity budget sum(lr) x lam(1e-3) "
            "%.4f" % (arm, STEPS, s, s * a["kw"]["weight_decay"], s * 1e-3))
    ok(abs(lr_sum(BASE_LR * PEAK_MULT, 400, WARMUP) - 21.40) < 0.01, "C1 sum(lr) = 21.40 (= the v4 spec, §3)")
    # tiny CPU fits through the same wrapper: the overrides reach the engine (the fit's own provenance)
    sys.path.insert(0, str(AD))
    import test_wcm_gemma as TW
    os.environ.pop("STOP_AT", None)
    os.environ.pop("STOP_EPOCH", None)
    inf, bank = TW.tiny_stack(TW.TINY_PLT, vocab=TW.GEMMA_VOCAB)
    eng = wrap_engine(W.make_engine(TW.TINY_PLT, "native", "pooled", shared=(inf, bank)))
    tsteps = 10
    for arm in ("C0", "C1", "C1s2"):
        set_arm(arm, WM, v3)
        out = OUT / "cpucheck" / arm
        try:
            rows = W.run(["rc"], [1e-3], TW.TINY_PLT, "sfc", "native", "pooled", tsteps, out, lr=ARMS[arm]["lr"],
                         eng=eng, log=lambda s: say("  " + s))
        finally:
            restore_v3(WM, v3)
        if not rows:
            rec = json.loads(next((out / "circuits").rglob("*.json")).read_text())
        else:
            rec = json.loads((out / rows[0]["circuit"]).read_text())
        f = rec["fit"]
        a = ARMS[arm]
        want_w = int(round(a["kw"].get("warmup_frac", 0.0) * tsteps))
        ok(abs(f["lr_peak"] - a["lr"]) < 1e-9 and f["warmup_steps"] == want_w
           and abs(f["weight_decay"] - WD) < 1e-12 and f["lr_schedule"] == "geometric",
           "tiny fit %s: lr_peak %g warmup_steps %s weight_decay %g schedule %s" % (
               arm, f["lr_peak"], f["warmup_steps"], f["weight_decay"], f["lr_schedule"]))
        ok(abs(f["lr_sum"] - lr_sum(a["lr"], tsteps, a["kw"].get("warmup_frac", 0.0))) < 1e-3,
           "tiny fit %s: engine lr_sum %.4f = re-stated %.4f" % (
               arm, f["lr_sum"], lr_sum(a["lr"], tsteps, a["kw"].get("warmup_frac", 0.0))))
        ok(f.get("resample_weight") == 1.0 and not f.get("rank_weight") and f.get("resample_seed") ==
           a["kw"].get("resample_seed", 20261017), "tiny fit %s: R 1.0, rank-keep off, R seed %s"
           % (arm, f.get("resample_seed")))
        ok(rec["wcm"].get("lrtest_arm") == arm, "tiny fit %s: arm recorded in the circuit JSON" % arm)
    ok(dict(WM.V3_LOGIT) == v3, "V3_LOGIT restored after the arms")
    say("check: %d failure(s)" % len(fails))
    return 1 if fails else 0


def re_has_kw(src: str, k: str) -> bool:
    import re
    return re.search(r"^\s+%s\s*:" % re.escape(k), src, re.M) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "plan", "run"))
    ap.add_argument("--stage", choices=STAGE_ORDER)
    a = ap.parse_args()
    if a.cmd == "run":
        if not a.stage:
            ap.error("run needs --stage")
        sys.exit(cmd_run(a.stage))
    sys.exit(cmd_check() if a.cmd == "check" else cmd_plan())


if __name__ == "__main__":
    main()
