"""092: speed-ups of v2 (090 sp_ste) and one training change, on 091's 8 targets.

  t1_trunc   v2 with every training micro-batch cut after its latest anchor (sx_engine SPEED truncate)
  t2_comb    v2 with the Z / C / A passes as ONE forward over 3 x the micro-batch (SPEED combine)
  t12_both   (optional, not in the default ARMS) t1 + t2 together
  t3b_ZAR_rawR  (optional) t3 with 091's R exactly (no sparsity-preserving donor fill): t3b vs 091 v2R differs
             ONLY in the dropped C term
  t3_ZAR     Z + A + R_C: 079's donor-resample term with the target's 48 TRAINING contrast contexts as donors (091's
             setup: one donor per row per forward, redrawn every forward, DonorFloors seed 20261017), under the
             sparsity-preserving donor fill (sp_donor), weight 0.25; the C term dropped; no silent term
Reference: 090's sp_ste circuits / rows (refits of 090 are deterministic: 091 CHECK, same circuit + same curve).
Scoring: production scorer R.score (held-out strongest) for all; t3 also 073 strict_eval.score_target, c_held=True
(C donors = the 16 held-out contrast contexts, R donors = 073's random pool), exactly as 091.

  wsl: PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/092-v2-speedups/run.py
  env: CHECK=1   step-0 equality on the first target (v2 vs truncate vs combine vs both: data terms + gradients)
       ARMS      comma list (default t1_trunc,t2_comb,t3_ZAR)   LIMIT  first N targets
       COMBINE_ROWS  rows per group per combined forward (default: the whole micro-batch; OOM falls back to 2)
"""
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
E90, E91 = EXP / "090-sparse-fill-training", EXP / "091-v2-resample-small"
sys.path.insert(0, str(HERE))
os.environ.setdefault("CURVE_STRIDE", "20")    # as 090

_spec = importlib.util.spec_from_file_location("run091", str(E91 / "run.py"))
R91 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R91)             # 091's helpers (imports 073 strict_eval -> 062 driver); its main() is not run

import torch  # noqa: E402

import sx_engine  # noqa: E402
import sp_engine  # noqa: E402

SE, driver = R91.SE, R91.driver
OUT = Path(os.environ.get("OUT92", str(HERE / "out")))
GAMMA, LAM, W_R = 0.25, 1e-3, 0.25
ALL_ARMS = ("t1_trunc", "t2_comb", "t3_ZAR")
WANT = [a for a in os.environ.get("ARMS", ",".join(ALL_ARMS)).split(",") if a]
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
COMBINE_ROWS = int(os.environ["COMBINE_ROWS"]) if os.environ.get("COMBINE_ROWS") else None
KEEP = R91.KEEP


def members(c):
    out = {}
    for n in c.nodes.values():
        if n.metadata.get("role") == "seed":
            continue
        f = n.metadata["feature_id"]
        out[(int(f.layer), str(f.kind), int(f.index))] = float(n.metadata.get("amplitude", 1.0))
    return out


def compare(c, ref):
    a, b = members(c), members(ref)
    inter = set(a) & set(b)
    uni = set(a) | set(b)
    da = [abs(a[k] - b[k]) for k in inter]
    da.sort()
    return dict(n_ref=len(b), jaccard=(len(inter) / len(uni)) if uni else 1.0, n_only_new=len(set(a) - set(b)),
                n_only_ref=len(set(b) - set(a)), dalpha_max=max(da) if da else None,
                dalpha_med=da[len(da) // 2] if da else None, identical=(a == b))


def arm_engine(arm):
    src = sx_engine.source_v2r() if arm.startswith("t3") else sx_engine.source_v2()
    return sx_engine.install(src, "092_" + arm)


def arm_switches(mod, arm, R, rec, key):
    sx_engine.reset(mod)
    sp_engine.set_mode(mod, "ste")
    info = {}
    if arm == "t1_trunc":
        mod.SPEED["truncate"] = True
    elif arm in ("t2_comb", "t12_both"):
        mod.SPEED["combine"] = True
        mod.SPEED["combine_rows"] = COMBINE_ROWS
        mod.SPEED["truncate"] = arm == "t12_both"
    elif arm in ("t3_ZAR", "t3b_ZAR_rawR"):
        mod.SPEED["drop_floor"] = True
        mod.SPEED["sp_donor"] = arm == "t3_ZAR"
        floors, _sfn, n_don = R91.setup_terms(mod, R, rec, key, W_R, 0.0)
        info.update(n_donors=n_don, _floors=floors)
    return info


# ------------------------------------------------------------------------------------------------ step-0 check
def check(R, key):
    """Step-0 data terms and data-term gradients: v2 vs truncate vs combine vs combine+truncate (one target)."""
    rec = R.contexts(key)
    tr_arm, _ = R.train_arm(rec)
    mod = sx_engine.install(sx_engine.source_v2(), "092_check")
    res = dict(key=key)
    got = {}
    stamp = time.strftime("%H%M%S")
    for tag, tr, cb in (("v2", False, False), ("trunc", True, False), ("comb", False, True), ("comb_trunc", True, True)):
        sx_engine.reset(mod)
        sp_engine.set_mode(mod, "ste")
        mod.SPEED.update(truncate=tr, combine=cb, grab=True, combine_rows=COMBINE_ROWS)
        M = R.method(GAMMA, LAM, True, steps=1)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        R.fit(M, rec, key, tr_arm, OUT / "_check" / stamp / tag / ("%s.pt" % key))
        g = mod.SPEED["grads"]
        got[tag] = g
        res["peak_mb_" + tag] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        res["secs_" + tag] = round(time.time() - t0, 1)
        res["terms_" + tag] = g["terms"]
        if tr:
            res["tok_frac_" + tag] = mod.SPEED["tok_used"] / max(mod.SPEED["tok_full"], 1)
    base = got["v2"]
    for tag in ("trunc", "comb", "comb_trunc"):
        g = got[tag]
        res["dterm_max_" + tag] = max(abs(x - y) for x, y in zip(g["terms"], base["terms"]))
        res["dloss_" + tag] = abs(sum(g["terms"]) - sum(base["terms"]))
        for part in ("theta", "amp"):
            dm = gm = 0.0
            for s, t in base[part].items():
                dm = max(dm, float((g[part][s] - t).abs().max()))
                gm = max(gm, float(t.abs().max()))
            res["dgrad_%s_max_%s" % (part, tag)] = dm
            res["grad_%s_scale" % part] = gm
    res["loss_v2"] = sum(base["terms"])
    sx_engine.reset(mod)
    print("092 CHECK %s" % json.dumps(res), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "check.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")


# ------------------------------------------------------------------------------------------------ main
def main():
    t = R91.pick_targets()                     # reads 091's out/targets.json (written by 091)
    keys = t["keys"][:LIMIT]
    R = driver.Runner()
    if os.environ.get("CHECK") == "1":
        check(R, keys[0])
        return
    G = dict(R.G)
    G["R"] = R
    dev = G["device"]
    eval_pool = None
    rows90 = {}
    for line in open(E90 / "out" / "rows.jsonl"):
        r = json.loads(line)
        if "error" not in r and r["arm"] == "sp_ste":
            rows90[r["seed"]] = r
    OUT.mkdir(parents=True, exist_ok=True)
    rows_path, fits_path = OUT / "rows.jsonl", OUT / "fits.jsonl"
    done, fits = set(), {}
    if rows_path.exists():
        for line in open(rows_path):
            r = json.loads(line)
            if "error" not in r:
                done.add((r["seed"], r["arm"]))
    if fits_path.exists():
        for line in open(fits_path):
            r = json.loads(line)
            if "error" not in r:
                fits[(r["seed"], r["arm"])] = r
    fh, ff, ftr = open(rows_path, "a"), open(fits_path, "a"), open(OUT / "train.jsonl", "a")
    t0 = time.time()
    print("092: %d targets %s, arms %s, steps %d" % (len(keys), keys, WANT, driver.STEPS), flush=True)
    for arm in WANT:
        mod = arm_engine(arm)
        M = R.method(GAMMA, LAM, True)
        for key in keys:
            if (key, arm) in done:
                continue
            row = dict(seed=key, arm=arm, group=R91.group_of(t, key), layer=int(key.split(".")[0]),
                       kind=key.split(".")[1])
            path = OUT / "circuits" / arm / ("%s.pt" % key)
            try:
                rec = R.contexts(key)
                tr_arm, _ = R.train_arm(rec)
                if path.exists() and (key, arm) in fits:
                    c = torch.load(path, weights_only=False)
                    f = fits[(key, arm)]
                else:
                    f = dict(seed=key, arm=arm)
                    for attempt in (COMBINE_ROWS, 2):
                        info = arm_switches(mod, arm, R, rec, key)
                        if arm in ("t2_comb", "t12_both"):
                            mod.SPEED["combine_rows"] = attempt
                        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                        try:
                            c, t_fit = R.fit(M, rec, key, tr_arm, path, train_fh=ftr, label=dict(arm092=arm))
                        except torch.cuda.OutOfMemoryError:
                            if arm not in ("t2_comb", "t12_both") or attempt == 2:
                                raise
                            traceback.print_exc()
                            f["oom_at_rows"] = attempt or 4
                            sx_engine.reset(mod)
                            torch.cuda.empty_cache()
                            continue
                        break
                    fl = info.get("_floors")
                    f.update(t_fit=round(t_fit, 1), fit_peak_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20),
                             combine_rows=(mod.SPEED.get("combine_rows") or "all") if arm in ("t2_comb", "t12_both")
                             else None,
                             tok_frac=(mod.SPEED["tok_used"] / mod.SPEED["tok_full"]) if mod.SPEED["tok_full"] else None,
                             sp_calls=mod.SPFILL.get("stats", {}).get("calls", 0),
                             donor_sp_calls=mod.SPFILL.get("stats", {}).get("donor_calls", 0),
                             n_donors=info.get("n_donors"), donor_draws=fl.n_draws if fl else None,
                             n=driver.n_nodes(c) if c is not None else None, rejected=c is None)
                    sx_engine.reset(mod)
                    ff.write(json.dumps(f) + "\n"); ff.flush()
                    fits[(key, arm)] = f
                row.update({k: v for k, v in f.items() if k not in ("seed", "arm")})
                if c is None:
                    row["skip"] = "rejected"
                else:
                    row.update(R91.alpha_summary(c))
                    ref = torch.load(E90 / "out" / "circuits" / "sp_ste" / ("%s.pt" % key), weights_only=False)
                    row.update(compare(c, ref))
                    p = R.score(c, rec, "strong")
                    row.update({"ho_" + k: p.get(k) for k in KEEP if k in p})
                    row["pass"] = R91.rule_pass(p)
                    row["dpass"] = R91.rule_pass(p, dense=True)
                    r90 = rows90.get(key, {})
                    row["ref_t_fit"], row["ref_peak_mb"], row["ref_pass"] = (r90.get("t_fit"), r90.get("fit_peak_mb"),
                                                                             r90.get("ho_pass"))
                    for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "freeM_dense_tk",
                              "freeN_dense_tk"):
                        if p.get(k) is not None and r90.get("ho_" + k) is not None:
                            row["d_" + k] = p[k] - r90["ho_" + k]
                    if arm.startswith("t3"):
                        if eval_pool is None:
                            eval_pool = torch.load(SE.OUT73 / "random_pool.pt")["tokens"].to(dev)
                            SE.RAND_DIR = OUT / "_no_random_null"
                        SE.MAIN_DIR = path.parent
                        s = SE.score_target(G, key, eval_pool, c_held=True)
                        rs = {k: v for k, v in s.items() if k.startswith("rs_") and not k.endswith("_draws")}
                        row.update(rs_A=rs.get("rs_A_wcm"), rs_C=rs.get("rs_C_wcm"), rs_R=rs.get("rs_R_wcm"),
                                   ap_C_r=s.get("ap_C_wcm_r"), ap_C_fp_mass=s.get("ap_C_wcm_fp_mass"))
                        row["rs_pass"] = bool(R91.ib(p.get("free0_tk")) and R91.ib(row["rs_A"]) and
                                              R91.ib(row["rs_C"]) and (p.get("phi_sup_blind_tk") or 0) >= 0.9)
                        row["resample"] = rs
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                sx_engine.reset(mod)
                torch.cuda.empty_cache()
            fh.write(json.dumps(row) + "\n"); fh.flush()
            g = lambda k: "%.3f" % row[k] if isinstance(row.get(k), (int, float)) else "-"  # noqa: E731
            print("[%5.0fs] %-8s %-14s n %s/%s J %s dα %s | Z %s A %s C %s nec %s pass %s | t_fit %s (ref %s) peak %s"
                  " (ref %s) tok %s | rsA %s rsC %s %s" % (
                      time.time() - t0, arm, key, row.get("n"), row.get("n_ref"), g("jaccard"), g("dalpha_max"),
                      g("ho_free0_tk"), g("ho_freeM_topk_tk"), g("ho_freeN_topk_tk"), g("ho_phi_sup_blind_tk"),
                      row.get("pass"), row.get("t_fit"), row.get("ref_t_fit"), row.get("fit_peak_mb"),
                      row.get("ref_peak_mb"), g("tok_frac"), g("rs_A"), g("rs_C"), row.get("error", "")), flush=True)
            torch.cuda.empty_cache()
        print("092: arm %s done %.0fs" % (arm, time.time() - t0), flush=True)
    print("092: DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
