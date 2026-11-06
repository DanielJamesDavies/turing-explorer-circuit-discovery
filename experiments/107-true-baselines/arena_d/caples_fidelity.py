"""Arena D2 fidelity: do our runs of Caples et al.'s unmodified script (run_caples.sh) reproduce their saved 2B anchors?
(README §0.3 / §5.2, spec caples.md §10.3; any Python with torch, CPU; reads files only.)

Anchors: third_party/ScalableSAECircuits/masks/sva/rc_train_saegap6_mean_error/logit_diff_<lam>_run/<lam>_run.json
(read only). Ours: <runs>/<grid>/src/masks/sva/rc_train_saegap6_mean_error/... written by their script in run_caples.sh's
clone copy, plus <runs>/<grid>/run.log. Grids (their --start/--end_threshold, 5 steps, their linear_map spacing):
  A (0.1, 10)  = scripts/sae_gap.sh as committed      B (0.005, 0.1) = the spec's anchors      C (5e-5, 0.005)
  D (0.01, 0.2) = the Colab grid: reported, never matched (README §0.3: another code state)
Checks per lambda (tolerances: plan D5 row; the size floor of 2 latents for near-empty circuits is ours):
  model LD  = logit_diff / faithfulness on their eval batches: 3.21 +- 0.02 (anchors: 3.2141 for every run)
  size      = total_density (their (SAE, position, latent) pairs): within +-30% of the anchor (or +-2)
  f         = faithfulness (vs the true model): within +-0.05 of the anchor
Reported, no pass bound:
  temperature at the final step (their tqdm postfix in run.log) vs the loop arithmetic: S = n_batches * portion,
    last i = first integer >= 1.1 S, T = 200 ** (i / S * 1.1)  (= 644 at 187 batches, portion 0.4)
  F5 latent identity per SAE: Jaccard of the latent sets vs the anchor's indices. FINDING (2026-11-05): grids A and D
    store LATENT indices, but grids B and C store POSITIONS (every index <= 6): their JSON writes
    torch.where(mask > 0)[1] at HEAD (latents) and the B / C runs evidently came from a code state that wrote [0].
    For B / C the per-SAE position histogram of our .pt masks is compared with the anchor's instead.
  F7 nesting: fraction of the (SAE, position, latent) members at lambda_k+1 that are also members at lambda_k (our .pt).

  python caples_fidelity.py --runs out/caples_runs [--grids A,B,C] [--out out/caples_runs/fidelity.json]
  python caples_fidelity.py --dry-run     (grid arithmetic vs the anchor folder names; anchors scored against
                                           themselves; the expected temperature; no run needed)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ANCHORS = ROOT / "third_party" / "ScalableSAECircuits" / "masks" / "sva" / "rc_train_saegap6_mean_error"
SAVE_TASK = "sva/rc_train_saegap6_mean_error"
GRIDS = {"A": (0.1, 10.0, 5), "B": (0.005, 0.1, 5), "C": (5e-05, 0.005, 5), "D": (0.01, 0.2, 5)}
MATCHED = ("A", "B", "C")
MODEL_LD, MODEL_LD_TOL = 3.21, 0.02
SIZE_REL, SIZE_FLOOR, F_TOL = 0.30, 2, 0.05
N_BATCHES, PORTION, EXAMPLE_LEN = 187, 0.4, 7
HOOKS = ["blocks.%d.hook_resid_post" % l for l in (0, 6, 12, 18, 24)]


def thresholds(start: float, end: float, n: int):
    """interp_bottlenecks.py:302-316, verbatim arithmetic (the folder names are str(threshold))."""
    def modify_fn(x):
        return x ** 2

    def linear_map(x, s, e):
        mod_start, mod_end = modify_fn(s), modify_fn(e)
        return (x - mod_start) / (mod_end - mod_start) * (e - s) + s
    delta = (end - start) / float(n)
    return [linear_map(modify_fn(start + i * delta), start, end) for i in range(n)]


def expected_temperature(n_batches: int = N_BATCHES, portion: float = PORTION) -> dict:
    total = n_batches * portion
    i = 0
    while not i >= total * 1.1:                               # utils.py:503-538: train step i, then break
        i += 1
    r = i / total * 1.1
    return dict(total_steps=total, last_step=i, steps=i + 1, ratio=r, temperature=200.0 ** r)


def run_dir(base: Path, lam: float) -> Path:
    return base / ("logit_diff_%s_run" % str(lam))


def load_run(base: Path, lam: float):
    d = run_dir(base, lam)
    f = d / ("%s_run.json" % str(lam))
    if not f.exists():
        return None
    j = json.loads(f.read_text())
    j["_dir"] = str(d)
    return j


def index_kind(j) -> str:
    mx = max((max(j[h]) for h in HOOKS if j.get(h)), default=-1)
    return "position" if 0 <= mx < EXAMPLE_LEN else ("latent" if mx >= EXAMPLE_LEN else "empty")


def pt_masks(j):
    """{hook: bool [T, d_sae]} from their sae_mask_<i>.pt (state_dict 'mask'), member = theta > 0; None if absent."""
    d = Path(j["_dir"])
    if not (d / "sae_mask_0.pt").exists():
        return None
    import torch
    out = {}
    for i, h in enumerate(HOOKS):
        sd = torch.load(d / ("sae_mask_%d.pt" % i), map_location="cpu", weights_only=False)
        out[h] = sd["mask"] > 0
    return out


def compare(ours, anc) -> dict:
    ld = ours["logit_diff"] / ours["faithfulness"] if ours["faithfulness"] else float("nan")
    s, sa = ours["total_density"], anc["total_density"]
    r = dict(model_ld=ld, model_ld_ok=abs(ld - MODEL_LD) <= MODEL_LD_TOL,
             size=s, anchor_size=sa, size_ok=abs(s - sa) <= max(SIZE_REL * sa, SIZE_FLOOR),
             size_ratio=(s / sa if sa else None), f=ours["faithfulness"], anchor_f=anc["faithfulness"],
             f_ok=abs(ours["faithfulness"] - anc["faithfulness"]) <= F_TOL,
             per_sae=[[len(ours.get(h, [])), len(anc.get(h, []))] for h in HOOKS],
             anchor_index_kind=index_kind(anc), ours_index_kind=index_kind(ours))
    r["passed"] = bool(r["model_ld_ok"] and r["size_ok"] and r["f_ok"])
    if r["anchor_index_kind"] == "latent" and r["ours_index_kind"] in ("latent", "empty"):
        jac = []
        for h in HOOKS:                                        # None: both empty at that SAE
            so, sa_ = set(ours.get(h, [])), set(anc.get(h, []))
            jac.append(len(so & sa_) / len(so | sa_) if (so | sa_) else None)
        r["latent_jaccard"] = jac
    elif r["anchor_index_kind"] == "position":
        m = pt_masks(ours)
        if m is not None:
            l1 = []
            for h in HOOKS:
                ca = Counter(anc.get(h, []))
                co = Counter(m[h].nonzero()[:, 0].tolist())
                tot = max(sum(ca.values()) + sum(co.values()), 1)
                l1.append(sum(abs(co[p] - ca[p]) for p in range(EXAMPLE_LEN)) / tot)
            r["position_hist_l1"] = l1                         # 0 = same per-position counts; 1 = disjoint
        else:
            r["position_hist_l1"] = "no .pt masks"
    return r


def parse_log(path: Path) -> dict:
    if not path.exists():
        return {}
    txt = path.read_text(errors="replace")
    out = {}
    for k, pat in (("full_model_ld", r"Average Full Model LD:\s*([-0-9.eE]+)"),
                   ("sae_ld", r"Average Model \+ SAEs LD:\s*([-0-9.eE]+)"),
                   ("sae_mean_error_ld", r"Average Model \+ SAEs \(Mean Error\) LD:\s*([-0-9.eE]+)")):
        m = re.search(pat, txt)
        out[k] = float(m.group(1)) if m else None
    temps = {}
    for seg in txt.split("doing a run with sparsity multiplier")[1:]:
        lam = seg.split()[0]
        t = re.findall(r"temperature=tensor\(([-0-9.eE+]+)", seg)
        temps[lam] = float(t[-1]) if t else None
    out["final_temperature"] = temps
    return out


def nesting(runs) -> list:
    """runs: [(lam, json)] ascending lam -> fraction of lam_{k+1}'s members also in lam_k (bigger circuit)."""
    out = []
    ms = [(lam, pt_masks(j)) for lam, j in runs if j is not None]
    for (l0, a), (l1, b) in zip(ms, ms[1:]):
        if a is None or b is None:
            continue
        nb = sum(int(b[h].sum()) for h in HOOKS)
        both = sum(int((a[h] & b[h]).sum()) for h in HOOKS)
        out.append(dict(lam_big=l0, lam_small=l1, n_small=nb, frac_in_big=(both / nb if nb else None)))
    return out


def score_grid(name, ours_base: Path, anc_base: Path, log: Path) -> dict:
    lams = thresholds(*GRIDS[name])
    rows, missing, runs = [], [], []
    for lam in lams:
        anc, ours = load_run(anc_base, lam), load_run(ours_base, lam)
        runs.append((lam, ours))
        if ours is None:
            missing.append(lam)
            continue
        if anc is None:
            rows.append(dict(lam=lam, size=ours["total_density"], f=ours["faithfulness"], anchor=None))
            continue
        rows.append(dict(lam=lam, **compare(ours, anc)))
    lg = parse_log(log)
    exp = expected_temperature()
    for r in rows:
        t = lg.get("final_temperature", {}).get(str(r["lam"]))
        r["final_temperature"] = t
        r["temperature_rel_err"] = None if t is None else abs(t - exp["temperature"]) / exp["temperature"]
    matched = name in MATCHED
    ok = bool(matched and not missing and rows and all(r.get("passed") for r in rows))
    return dict(grid=name, lams=lams, matched=matched, passed=ok if matched else None, missing=missing, rows=rows,
                log=lg, expected_temperature=exp, nesting=nesting(runs))


def show(g):
    print("grid %s (%s): %s%s" % (g["grid"], "matched" if g["matched"] else "reported only",
                                  {True: "PASS", False: "FAIL", None: "-"}[g["passed"]],
                                  "; missing %s" % g["missing"] if g["missing"] else ""))
    for r in g["rows"]:
        if r.get("anchor_size") is None:
            print("   lam %-10.4g size %6s f %.3f (no anchor)" % (r["lam"], r["size"], r["f"]))
            continue
        extra = ""
        if "latent_jaccard" in r:
            extra = " | latent Jaccard " + " ".join("-" if x is None else "%.2f" % x for x in r["latent_jaccard"])
        elif isinstance(r.get("position_hist_l1"), list):
            extra = " | pos-hist L1 " + " ".join("%.2f" % x for x in r["position_hist_l1"])
        t = r.get("final_temperature")
        print("   lam %-10.4g %s LD %.4f | size %6d vs %6d (x%.2f) | f %.3f vs %.3f | T_end %s%s" % (
            r["lam"], "ok  " if r["passed"] else "FAIL", r["model_ld"], r["size"], r["anchor_size"],
            r["size_ratio"] or float("nan"), r["f"], r["anchor_f"], "-" if t is None else "%.1f" % t, extra))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", default=str(HERE / "out" / "caples_runs"))
    p.add_argument("--grids", default="A,B,C")
    p.add_argument("--anchors", default=str(ANCHORS))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    anc = Path(a.anchors)
    grids = [g for g in a.grids.split(",") if g]
    res = dict(anchors=str(anc), expected_temperature=expected_temperature(), tolerances=dict(
        model_ld=[MODEL_LD, MODEL_LD_TOL], size_rel=SIZE_REL, size_floor=SIZE_FLOOR, f=F_TOL), grids={})
    if a.dry_run:
        names = {d.name for d in anc.iterdir() if d.is_dir()}
        want = {"logit_diff_%s_run" % str(l) for g in GRIDS.values() for l in thresholds(*g)}
        res["grid_names_match_anchor_folders"] = sorted(want) == sorted(names)
        res["anchor_folders_not_in_grids"] = sorted(names - want)
        res["grid_folders_not_in_anchors"] = sorted(want - names)
        print("grid arithmetic -> anchor folder names: %s (extra %s, missing %s)" % (
            res["grid_names_match_anchor_folders"], res["anchor_folders_not_in_grids"],
            res["grid_folders_not_in_anchors"]))
        for g in GRIDS:                                    # anchors scored against themselves: must pass (A-C)
            res["grids"][g] = score_grid(g, anc, anc, Path("/nonexistent"))
            res["grids"][g]["label"] = "DRY RUN: anchors vs anchors"
            show(res["grids"][g])
        e = res["expected_temperature"]
        print("expected: %d steps (last i %d, S %.1f), final ratio %.4f, T_end %.1f" % (
            e["steps"], e["last_step"], e["total_steps"], e["ratio"], e["temperature"]))
        print("ours would be read from %s/<grid>/src/masks/%s/" % (a.runs, SAVE_TASK))
        ok = res["grid_names_match_anchor_folders"] and all(res["grids"][g]["passed"] for g in MATCHED)
    else:
        runs = Path(a.runs)
        for g in grids:
            res["grids"][g] = score_grid(g, runs / g / "src" / "masks" / SAVE_TASK, anc, runs / g / "run.log")
            show(res["grids"][g])
        ok = all(res["grids"][g]["passed"] for g in grids if g in MATCHED)
    res["passed"] = bool(ok)
    out = Path(a.out) if a.out else (HERE / "out" / "dryrun" / "caples_fidelity_dryrun.json" if a.dry_run
                                     else Path(a.runs) / "fidelity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("VERDICT: %s -> %s" % ("PASS" if ok else "FAIL", out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
