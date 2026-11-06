"""091: does 079's donor-resample term R (contrast donors only), with or without the silent-position term S, help on
top of v2 (090's sp_ste: A / C trained under the sparsity-preserving fill, straight-through)?

Targets (8 of 090's 20, random.Random(20261025)): 4 of group fail89; 2 deep attention (1 near-threshold, 1 far);
2 controls that pass the rule under both 090 dense and 090 sp_ste.
Arms fitted here (production otherwise: 400 steps, lambda 1e-3, gamma_Z 1, gamma_A = gamma_C 0.25, rank-keep 3e-3):
  v2R     SPFILL ste + RESAMPLE weight 0.25, donors = the target's 48 TRAINING contrast contexts (079 DONORS=contrast:
          one donor per training row per forward, redrawn every forward, DonorFloors seed 20261017)
  v2R_S   v2R + SILENT weight 0.25 (079's RS0.25_S0.25)
Reused (not refitted): 090's dense and sp_ste circuits (and their fit time / peak memory).
Scoring, all four versions, held-out strongest:
  production scorer  driver.Runner.score (rule A / C sparsity-preserving, dense A / C, necessity, rule pass)
  resample (073)     strict_eval.score_target, c_held=True: C donors = the 16 HELD-OUT contrast contexts, R donors =
                     073's evaluation pool; resample pass = Z, rs A, rs C in [0.8, 1.5] and necessity >= 0.9 (079)

  wsl: PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/091-v2-resample-small/run.py
  env: CHECK=1 (equivalence checks on the first target at CHECK_STEPS steps, then exit)  LIMIT  ARMS (v2R,v2R_S)
       SCORE_ONLY=1 (skip fitting)
"""
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
E90 = EXP / "090-sparse-fill-training"
sys.path.insert(0, str(EXP / "073-strict-eval"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import strict_eval as SE  # noqa: E402  (imports 062's driver, OUT = the 15k run)
import engine  # noqa: E402
import sp_engine  # noqa: E402
import rs_engine  # noqa: E402

driver = SE.driver
OUT = HERE / "out"
SEED = 20261025
W_R, W_S = 0.25, 0.25
TRAIN_POOL_SEED = 20261017                     # 079's DonorFloors seed
ARMS = {"v2R": (W_R, 0.0), "v2R_S": (W_R, W_S)}
WANT = [a for a in os.environ.get("ARMS", "v2R,v2R_S").split(",") if a]
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
GAMMA, LAM = 0.25, 1e-3
KEEP = ("n", "free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "freeM_dense_tk", "freeN_dense_tk",
        "phi_cf_alpha_blind_tk", "vacuous_tk", "alpha_median", "alpha_p90", "a_pos_tk")


# ------------------------------------------------------------------------------------------------ targets
def pick_targets():
    p = OUT / "targets.json"
    if p.exists():
        return json.loads(p.read_text())
    g = json.loads((E90 / "targets.json").read_text())
    rows = [json.loads(l) for l in open(E90 / "out" / "rows.jsonl")]
    rule = {(r["seed"], r["arm"]): r.get("ho_pass") for r in rows if "error" not in r}
    rng = random.Random(SEED)
    fail = sorted(rng.sample(sorted(g["fail89"]), 4))
    near = rng.choice(sorted(k for k in g["attn"] if g["attn_sub"][k] == "near"))
    far = rng.choice(sorted(k for k in g["attn"] if g["attn_sub"][k] == "far"))
    ok = sorted(k for k in g["control"] if rule.get((k, "dense")) and rule.get((k, "sp_ste")))
    ctl = sorted(rng.sample(ok, 2))
    t = dict(seed=SEED, fail89=fail, attn_near=[near], attn_far=[far], control=ctl,
             control_pool=ok, keys=fail + [near, far] + ctl)
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(t, indent=1))
    return t


def group_of(t, key):
    for g in ("fail89", "attn_near", "attn_far", "control"):
        if key in t[g]:
            return g
    return None


# ------------------------------------------------------------------------------------------------ helpers
def donor_tokens(rec):
    """079 DONORS=contrast: the 48 training contrast contexts (rec['neg'][:64] split 48 / 16). Asserted equal to the
    C term's own training contrast rows and to the first slice of the scorer's contrast split."""
    from circuit.protocol_contexts import eval_probe, training_probe
    from eval.wcm_eval import contrast_split, split_n
    nt = rec["neg"][:64]
    nt_tr = nt[:split_n(int(nt.shape[0]))]
    pr = eval_probe(rec, "strong", "cpu")
    ntp = pr.neg_tokens[:64]
    n_ntr = contrast_split(int(ntp.shape[0]), pr.metadata)
    assert n_ntr == nt_tr.shape[0] and torch.equal(ntp[:n_ntr].cpu(), nt_tr.cpu()), "donors != scorer contrast-train"
    arm, _ = driver.Runner.train_arm(None, rec)
    tp = training_probe(rec, "cpu", arm)
    assert torch.equal(tp.neg_tokens[:nt_tr.shape[0]].cpu(), nt_tr.cpu()), "donors != the C term's contrast rows"
    return nt_tr


def setup_terms(mod, R, rec, key, w, sw, seed=TRAIN_POOL_SEED):
    """Arm the RESAMPLE / SILENT switches for one fit (as 079 fit_score.main). Returns (floors, silent_fn, n_donors)."""
    G = R.G
    inference, bank, dev = G["inference"], G["bank"], G["device"]
    from eval.ablation_faithfulness import upstream_sites
    from eval.wcm_eval import site_encoder
    l, kd, ti = key.split(".")
    sfn = floors = None
    n_don = None
    if sw:
        We, be = site_encoder(bank, int(l), kd)
        sfn = rs_engine.make_silent_fn(inference, bank, (int(l), kd), int(ti), We[int(ti)], be[int(ti)], sw)
        mod.SILENT.update(weight=float(sw), fn=sfn)
    if w:
        toks = donor_tokens(rec).to(dev)
        sites = sorted(upstream_sites(bank, int(l), kd))
        codes = SE.capture_codes(inference, bank, toks, sites)
        codes = {s: (a.cpu(), i.cpu()) for s, (a, i) in codes.items()}
        floors = rs_engine.DonorFloors(codes, bank.d_sae, dev, seed=seed)
        mod.RESAMPLE["make"] = rs_engine.make_factory(mod, floors)
        mod.RESAMPLE["weight"] = float(w)
        n_don = int(toks.shape[0])
    return floors, sfn, n_don


def alpha_summary(c):
    import numpy as np
    a = np.array([float(n.metadata.get("amplitude", 1.0)) for n in c.nodes.values() if n.metadata.get("role") != "seed"])
    if a.size == 0:
        return dict(alpha_med=None, alpha_p90=None, alpha_max=None)
    return dict(alpha_med=float(np.median(a)), alpha_p90=float(np.percentile(a, 90)), alpha_max=float(a.max()))


def ib(v):
    return v is not None and 0.8 <= v <= 1.5


def rule_pass(p, dense=False):
    a, c = ("freeM_dense_tk", "freeN_dense_tk") if dense else ("freeM_topk_tk", "freeN_topk_tk")
    nec = p.get("phi_sup_blind_tk")
    return bool(ib(p.get("free0_tk")) and ib(p.get(a)) and ib(p.get(c)) and nec is not None and nec >= 0.9)


def circuit_sig(c):
    """Exact identity of a fitted circuit: sorted (layer, kind, index, amplitude) of its non-seed nodes."""
    out = []
    for n in c.nodes.values():
        if n.metadata.get("role") == "seed":
            continue
        f = n.metadata["feature_id"]
        out.append((int(f.layer), str(f.kind), int(f.index), float(n.metadata.get("amplitude", 1.0))))
    return sorted(out)


# ------------------------------------------------------------------------------------------------ equivalence checks
def check(key, steps):
    """(1) combined, SPFILL ste, w_R = w_S = 0  vs  090 sp_engine ste;
    (2) combined, SPFILL dense, w_R = w_S = 0.25  vs  079 rs_engine RS0.25_S0.25.
    Each pair: same target, same steps, fresh module installs; compares the circuit (members + amplitudes, exact) and
    the per-step training loss curve (exact)."""
    d = OUT / "_check"
    res = {}
    R = driver.Runner()
    rec = R.contexts(key)
    tr_arm, _ = R.train_arm(rec)

    def fit(src, tag, mode, w, sw):
        mod = engine.install(src, tag)
        if hasattr(mod, "SPFILL"):
            sp_engine.set_mode(mod, mode)
        if w or sw:
            setup_terms(mod, R, rec, key, w, sw)
        M = R.method(GAMMA, LAM, True, steps=steps)
        path = d / tag / ("%s.pt" % key)
        if path.exists():
            path.unlink()
        c, _ = R.fit(M, rec, key, tr_arm, path)
        curve = (M.last_mask_provenance or {}).get("curve", {})
        engine.reset(mod)
        return c, curve

    c90, k90 = fit(sp_engine.patched_source(), "sp090", "ste", 0, 0)
    c91, k91 = fit(None, "v91_ste_w0", "ste", 0, 0)
    res["ste_w0_same_circuit"] = circuit_sig(c90) == circuit_sig(c91)
    res["ste_w0_same_curve"] = k90.get("loss") == k91.get("loss")
    res["ste_n"] = len(circuit_sig(c91))
    c79, k79 = fit(rs_engine.patched_source(), "rs079", "dense", W_R, W_S)
    c91b, k91b = fit(None, "v91_dense_RS", "dense", W_R, W_S)
    res["rs_dense_same_circuit"] = circuit_sig(c79) == circuit_sig(c91b)
    res["rs_dense_same_curve"] = k79.get("loss") == k91b.get("loss")
    res["rs_curve_keys"] = {k: (sorted(v) if isinstance(v, dict) else None) for k, v in k91b.items()}
    # (3) the full v2R_S fill actually differs from both parents (sanity: the combination is not a no-op)
    c3, k3 = fit(None, "v91_ste_RS", "ste", W_R, W_S)
    res["v2RS_differs_from_ste"] = k3.get("loss") != k91.get("loss")
    res["v2RS_differs_from_rs"] = k3.get("loss") != k91b.get("loss")
    res.update(key=key, steps=steps, loss_last=dict(sp090=k90.get("loss", [None])[-1], v91_ste_w0=k91.get("loss", [None])[-1],
                                                     rs079=k79.get("loss", [None])[-1],
                                                     v91_dense_RS=k91b.get("loss", [None])[-1],
                                                     v91_ste_RS=k3.get("loss", [None])[-1]))
    print("091 CHECK %s" % json.dumps(res, default=str), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "check.jsonl", "a") as fh:
        fh.write(json.dumps(res, default=str) + "\n")
    return res


# ------------------------------------------------------------------------------------------------ main
def main():
    t = pick_targets()
    keys = t["keys"][:LIMIT]
    if os.environ.get("CHECK") == "1":
        check(keys[0], int(os.environ.get("CHECK_STEPS", "6")))
        return
    mod = engine.install()
    R = driver.Runner()
    G = dict(R.G)
    G["R"] = R
    dev = G["device"]
    eval_pool = torch.load(SE.OUT73 / "random_pool.pt")["tokens"].to(dev)
    SE.RAND_DIR = OUT / "_no_random_null"
    rows90 = {}
    for l in open(E90 / "out" / "rows.jsonl"):
        r = json.loads(l)
        if "error" not in r and r["arm"] in ("dense", "sp_ste"):
            rows90[(r["seed"], r["arm"])] = r
    OUT.mkdir(parents=True, exist_ok=True)
    fits_path, rows_path = OUT / "fits.jsonl", OUT / "rows.jsonl"
    fits = {}
    if fits_path.exists():
        for l in open(fits_path):
            r = json.loads(l)
            fits[(r["seed"], r["arm"])] = r
    t0 = time.time()
    print("091: %d targets %s, arms %s, steps %d" % (len(keys), keys, WANT, driver.STEPS), flush=True)

    # ---- fits
    if os.environ.get("SCORE_ONLY") != "1":
        ftr = open(OUT / "train.jsonl", "a")
        ff = open(fits_path, "a")
        M = None
        for arm in WANT:
            w, sw = ARMS[arm]
            for key in keys:
                path = OUT / "circuits" / arm / ("%s.pt" % key)
                if path.exists() and (key, arm) in fits:
                    continue
                row = dict(seed=key, arm=arm, w_R=w, w_S=sw, donors="contrast_train48", fill="ste")
                try:
                    rec = R.contexts(key)
                    tr_arm, _ = R.train_arm(rec)
                    sp_engine.set_mode(mod, "ste")
                    floors, sfn, n_don = setup_terms(mod, R, rec, key, w, sw)
                    if M is None:
                        M = R.method(GAMMA, LAM, True)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    c, t_fit = R.fit(M, rec, key, tr_arm, path, train_fh=ftr,
                                     label=dict(arm091=arm, w_R=w, w_S=sw, fill="ste"))
                    row.update(t_fit=round(t_fit, 1), fit_peak_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20),
                               n_donors=n_don, donor_draws=floors.n_draws if floors else None,
                               sp_calls=mod.SPFILL.get("stats", {}).get("calls", 0),
                               silent_calls=dict(sfn.stats) if sfn else None,
                               n=driver.n_nodes(c) if c is not None else None, rejected=c is None)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                    torch.cuda.empty_cache()
                engine.reset(mod)
                ff.write(json.dumps(row) + "\n"); ff.flush()
                fits[(key, arm)] = row
                print("[%5.0fs] fit %-6s %-16s n %s t_fit %s peak %s MB draws %s sp %s %s" % (
                    time.time() - t0, arm, key, row.get("n"), row.get("t_fit"), row.get("fit_peak_mb"),
                    row.get("donor_draws"), row.get("sp_calls"), row.get("error", "")), flush=True)
            print("091: arm %s fitted %.0fs" % (arm, time.time() - t0), flush=True)

    # ---- scoring (all four versions; production engine state: every switch off)
    engine.reset(mod)
    done = set()
    if rows_path.exists():
        for l in open(rows_path):
            r = json.loads(l)
            if "error" not in r:
                done.add((r["seed"], r["arm"]))
    fh = open(rows_path, "a")
    for key in keys:
        rec = R.contexts(key)
        for arm in ("dense", "sp_ste") + tuple(WANT):
            if (key, arm) in done:
                continue
            cdir = (E90 / "out" / "circuits" / arm) if arm in ("dense", "sp_ste") else (OUT / "circuits" / arm)
            row = dict(seed=key, arm=arm, group=group_of(t, key), layer=int(key.split(".")[0]), kind=key.split(".")[1])
            ts = time.time()
            try:
                c = torch.load(cdir / ("%s.pt" % key), weights_only=False)
                if c is None:
                    row["skip"] = "rejected"
                else:
                    row["n"] = driver.n_nodes(c)
                    row.update(alpha_summary(c))
                    p = R.score(c, rec, "strong")
                    row.update({"ho_" + k: p.get(k) for k in KEEP if k in p})
                    row["pass"] = rule_pass(p)
                    row["dpass"] = rule_pass(p, dense=True)
                    if arm in ("dense", "sp_ste"):
                        r90 = rows90.get((key, arm), {})
                        row.update(t_fit=r90.get("t_fit"), fit_peak_mb=r90.get("fit_peak_mb"), fit_source="090")
                        diffs = [abs(p[k] - r90["ho_" + k]) for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk",
                                                                       "phi_sup_blind_tk")
                                 if p.get(k) is not None and r90.get("ho_" + k) is not None]
                        row["max_diff_vs_090_row"] = max(diffs) if diffs else None
                        row["pass_090_row"] = r90.get("ho_pass")
                    else:
                        f = fits.get((key, arm), {})
                        row.update(t_fit=f.get("t_fit"), fit_peak_mb=f.get("fit_peak_mb"), fit_source="091")
                    SE.MAIN_DIR = cdir
                    s = SE.score_target(G, key, eval_pool, c_held=True)
                    rs = {k: v for k, v in s.items() if k.startswith("rs_") and not k.endswith("_draws")}
                    row.update(rs_A=rs.get("rs_A_wcm"), rs_C=rs.get("rs_C_wcm"), rs_R=rs.get("rs_R_wcm"),
                               ap_C_r=s.get("ap_C_wcm_r"), ap_C_fp_mass=s.get("ap_C_wcm_fp_mass"))
                    row["rs_pass"] = bool(ib(p.get("free0_tk")) and ib(row["rs_A"]) and ib(row["rs_C"]) and
                                          (p.get("phi_sup_blind_tk") or 0) >= 0.9)
                    row["resample"] = rs
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                torch.cuda.empty_cache()
            row["score_secs"] = round(time.time() - ts, 1)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            g = lambda k: "%.2f" % row[k] if isinstance(row.get(k), (int, float)) else "-"  # noqa: E731
            print("[%5.0fs] score %-6s %-16s n %s Z %s A %s C %s nec %s pass %s | rsA %s rsC %s rsR %s rs_pass %s %s" % (
                time.time() - t0, arm, key, row.get("n"), g("ho_free0_tk"), g("ho_freeM_topk_tk"),
                g("ho_freeN_topk_tk"), g("ho_phi_sup_blind_tk"), row.get("pass"), g("rs_A"), g("rs_C"), g("rs_R"),
                row.get("rs_pass"), row.get("error", "")), flush=True)
            torch.cuda.empty_cache()
    print("091: DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
