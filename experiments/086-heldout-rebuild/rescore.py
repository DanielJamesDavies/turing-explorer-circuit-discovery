"""RE-SCORE QUEUE (DAN-157): the 15k run's FIXED circuits scored on the rebuilt held-out sets. No refits.

Every job reads its records through driver.Runner.contexts with CTX_DIR (default the rebuild, 062/out_full/out/ctx_v2),
so the scorer, the context injection and the explicit split (CTX_DIR forces EVAL_SPLIT=explicit) are the production
ones; on a rebuilt record the scorer's contrast held-out slice is the deduplicated neg_held_v2 (protocol_contexts.
eval_contrast; the contrast-train slice is the run's 48 rows). Rows go to new files only (OUT_RS/<job>/...); nothing
under 062/out_full is written. Every row carries the record's rule version, held-out sizes and below-floor flags;
below-floor targets are scored and flagged, never dropped; a target with no held-out row at all gets a skip row.

Jobs (JOBS, comma list; each resumable per row key, sharded by SHARD=i/k over its target list):
  main     every main circuit (062 out_full/out/main/circuits): held-out strongest + held-out mid-band eval rows
           (eval.shard<i>.jsonl, keyed seed / held) and 056 specificity on held-out strongest (spec.shard<i>.jsonl)
  sweep    every sweep arm circuit (out_full/out/sweep/circuits/<arm>/<key>.pt), held-out strongest
  alpha1   the alpha = 1 sample (the seeds of 062/out_alpha1/alpha1.jsonl): main circuits, every member at alpha 1
  random   (not in the default JOBS: 084's harder nulls score their DAN-17 references on ctx_v2 directly) the DAN-17
           random circuits (RN_DIRS, default 062/out_random_scale: <key>.d<draw>.pt), held-out strongest
The paired leakage effect: JOBS=main with CTX_DIR = the as-run records (062/out_full/out/ctx) and TARGETS =
targets_asrun1000.txt (subset.py: the frozen stratified 1,000) on the same machine, OUT_RS its own directory.
073 strict_eval, 083 and 084 read records through driver.Runner too: run them with the same CTX_DIR and their own new
output directories (README).

  CUDA_VISIBLE_DEVICES=0 SHARD=0/1 JOBS=main PYTHONPATH=src python experiments/086-heldout-rebuild/rescore.py
  env: JOBS (main,sweep,alpha1)  SHARD  LIMIT (first N targets per job, smoke)  TARGETS (restrict every job to
       this list)  CTX_DIR  OUT_RS (086/out/rescore_<CTX_DIR name>)  RN_DIRS  SKIP_SPEC=1  MID (all | none)
       REPORT=1 (no GPU): OUT_RS = the ctx_v2 re-score (headline over targets with >= 8 clean held-out strongest rows +
       worst case, below floor = fail; per band x kind), ASRUN_RS = the as-run subset re-score (paired ctx vs ctx_v2
       on that subset: the leakage effect), ASRUN_TARGETS (targets_asrun1000.txt) -> OUT_RS/report.md
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).parent
E062 = HERE.parent / "062-h100-protocol-v1"
os.environ.setdefault("OUT", str(E062 / "out_full" / "out"))
os.environ.setdefault("CTX_DIR", str(E062 / "out_full" / "out" / "ctx_v2"))
sys.path.insert(0, str(E062))
sys.path.insert(0, str(HERE))
import driver  # noqa: E402  (CTX_DIR -> EVAL_SPLIT=explicit, before the scorer is imported)
import rescore_alpha1 as RA  # noqa: E402
from passrates import FLOOR, HEAD, classify, headline, paired  # noqa: E402

import torch  # noqa: E402

JOBS = [j for j in os.environ.get("JOBS", "main,sweep,alpha1").split(",") if j]
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
ONLY = set(Path(os.environ["TARGETS"]).read_text().split()) if os.environ.get("TARGETS") else None
CTX_DIR = Path(os.environ["CTX_DIR"])
OUT_RS = Path(os.environ.get("OUT_RS", str(HERE / "out" / ("rescore_%s" % CTX_DIR.name))))
RN_DIRS = [Path(p) for p in os.environ.get("RN_DIRS", str(E062 / "out_random_scale")).split(",") if p]
SKIP_SPEC = os.environ.get("SKIP_SPEC") == "1"
MID = os.environ.get("MID", "all")
ASRUN_RS = Path(os.environ["ASRUN_RS"]) if os.environ.get("ASRUN_RS") else None
ASRUN_TARGETS = Path(os.environ.get("ASRUN_TARGETS", str(HERE / "targets_asrun1000.txt")))
MAIN = driver.OUT / "main" / "circuits"


def log(s):
    print("[%s rescore %s] %s" % (time.strftime("%H:%M:%S"), driver.TAG, s), flush=True)


def tags(rec, held):
    """Row labels from the record's provenance (heldout_v2, 086 rebuild)."""
    hv = rec.get("heldout_v2") or {}
    d = rec.get(held)
    n = len(d["held"]) if d is not None else 0
    nh = rec.get("neg_held_v2")
    nn = int(nh.shape[0]) if nh is not None else None          # None: the run's own 16 contrast held-out rows
    return dict(held_rule=hv.get("rule"), n_held=n, below_floor=n < FLOOR, n_neg_held=nn,
                neg_below_floor=None if nn is None else nn < FLOOR, ctx_dir=CTX_DIR.name)


def keyset(keys):
    keys = [k for k in keys if ONLY is None or k in ONLY][driver.SHARD_I::driver.SHARD_K]
    return keys[:LIMIT] if LIMIT else keys


def scalar(r):
    return {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None}


class Job:
    def __init__(self, name, fields):
        self.d = OUT_RS / name
        self.d.mkdir(parents=True, exist_ok=True)
        self.fields = fields
        self.path = driver.shard_jsonl(self.d, name)
        self.done = {tuple(r.get(f) for f in fields) for r in driver.read_jsonl_all(self.d, name) if "error" not in r}
        self.fh = open(self.path, "a")

    def write(self, row):
        self.fh.write(json.dumps(row) + "\n"); self.fh.flush()


def score_or_skip(R, c, rec, held, **kw):
    if rec.get(held) is None:
        return None
    t = tags(rec, held)
    if t["n_held"] == 0:
        return dict(skip="no_heldout", **t)
    if t["n_neg_held"] == 0:
        return dict(skip="no_contrast_heldout", **t)
    row = R.score(c, rec, held, **kw)
    row.update(t)
    return row


def job_main(R):
    J = Job("main", ("seed", "held"))
    S = None if SKIP_SPEC else Job("spec", ("seed",))
    keys = keyset(sorted(p.stem for p in MAIN.glob("*.pt")))
    log("main: %d targets (%d rows done)" % (len(keys), len(J.done)))
    for n_, key in enumerate(keys):
        helds = ["strong"] + (["mid"] if MID == "all" else [])
        if all((key, h) in J.done for h in helds) and (S is None or (key,) in S.done):
            continue
        t0 = time.time()
        try:
            c = torch.load(MAIN / ("%s.pt" % key), weights_only=False)
            if c is None:
                for h in helds:
                    if (key, h) not in J.done:
                        J.write(dict(seed=key, held=h, skip="rejected"))
                if S is not None and (key,) not in S.done:
                    S.write(dict(seed=key, skip="rejected"))
                continue
            rec = R.contexts(key)
            arm, thin = R.train_arm(rec)
            for h in helds:
                if (key, h) in J.done:
                    continue
                row = score_or_skip(R, c, rec, h)
                if row is not None:
                    row.update(seed=key, held=h, arm=arm, thin=thin, shard=driver.TAG)
                    J.write(row)
            if S is not None and (key,) not in S.done:
                ts_ = tags(rec, "strong")
                if ts_["n_held"] == 0 or ts_["n_neg_held"] == 0:
                    S.write(dict(seed=key, skip="no_heldout" if ts_["n_held"] == 0 else "no_contrast_heldout", **ts_))
                else:
                    for r in R.spec(c, rec, arm):
                        S.fh.write(json.dumps(dict(r, **tags(rec, "strong"))) + "\n")
                    S.fh.flush()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            J.write(dict(seed=key, held="strong", error="%s: %s" % (type(e).__name__, str(e)[:300])))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if n_ % 25 == 0:
            log("main %d/%d %s %.1fs" % (n_ + 1, len(keys), key, time.time() - t0))


def job_sweep(R):
    J = Job("sweep", ("seed", "arm"))
    root = driver.OUT / "sweep" / "circuits"
    arms = sorted(p.name for p in root.iterdir() if p.is_dir())
    log("sweep: arms %s" % arms)
    for arm in arms:
        keys = keyset(sorted(p.stem for p in (root / arm).glob("*.pt")))
        for key in keys:
            if (key, arm) in J.done:
                continue
            try:
                c = torch.load(root / arm / ("%s.pt" % key), weights_only=False)
                if c is None:
                    row = dict(skip="rejected")
                else:
                    rec = R.contexts(key)
                    row = score_or_skip(R, c, rec, "strong")
                row.update(seed=key, arm=arm, held="strong", shard=driver.TAG)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row = dict(seed=key, arm=arm, error="%s: %s" % (type(e).__name__, str(e)[:300]))
            J.write(row)
        log("sweep arm %s: %d targets" % (arm, len(keys)))


def job_random(R):
    J = Job("random", ("seed", "draw", "rn_dir"))
    for d in RN_DIRS:
        files = sorted((d / "circuits").glob("*.d*.pt"))
        keys = set(keyset(sorted({f.name.rsplit(".d", 1)[0] for f in files})))
        log("random %s: %d circuits, %d targets on this shard" % (d.name, len(files), len(keys)))
        for f in files:
            key, draw = f.name[:-3].rsplit(".d", 1)
            if key not in keys or (key, int(draw), d.name) in J.done:
                continue
            try:
                c = torch.load(f, weights_only=False)
                if c is None:
                    row = dict(skip="rejected")
                else:
                    row = score_or_skip(R, c, R.contexts(key), "strong")
                row.update(seed=key, draw=int(draw), rn_dir=d.name, held="strong", shard=driver.TAG)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row = dict(seed=key, draw=int(draw), rn_dir=d.name, error="%s: %s" % (type(e).__name__, str(e)[:300]))
            J.write(scalar(row))


def job_alpha1(R):
    J = Job("alpha1", ("seed",))
    src = [r for r in driver.read_jsonl_all(E062 / "out_alpha1", "alpha1") if "error" not in r and "skip" not in r]
    keys = keyset(sorted({r["seed"] for r in src}))
    log("alpha1: %d targets" % len(keys))
    for key in keys:
        if (key,) in J.done:
            continue
        try:
            c = torch.load(MAIN / ("%s.pt" % key), weights_only=False)
            row = score_or_skip(R, c, R.contexts(key), "strong", alphas=RA.ones(c))
            row.update(seed=key, held="strong", alpha="one", shard=driver.TAG)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            row = dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))
        J.write(scalar(row))


def report():
    """REPORT=1 (no GPU) -> OUT_RS/report.md (also printed).
    1. ctx_v2 headline (OUT_RS): pass rate over targets with >= 8 clean held-out strongest rows, and the worst case
       (below-floor targets = fails), overall and per band x kind; mid-band, sweep arms and alpha = 1 the same way.
    2. The paired leakage effect (ASRUN_RS = the as-run re-score of the frozen subset ASRUN_TARGETS, same machine):
       as-run vs ctx_v2 pass rates on the same targets, per band x kind, headline and worst case.
    3. Reproduction check against the run's own rows (062 out_full main eval): the as-run subset re-score should
       match them up to float noise (H100 vs H100); alpha = 1 against 062 out_alpha1 (ctx_v2: differs by design)."""
    import pandas as pd
    L = ["# 086 re-score report", "", "ctx_v2 re-score: %s" % OUT_RS, "as-run subset re-score: %s (targets %s)" % (
        ASRUN_RS, ASRUN_TARGETS if ASRUN_TARGETS.exists() else "missing"), "",
         "Pass = DAN-8 (Z, A, C in [0.8, 1.5], necessity >= 0.9) on held-out strongest unless stated; vacuous targets "
         "excluded; rejected circuits excluded. Headline = targets with >= %d clean held-out strongest rows; worst "
         "case = every below-floor target counted as a fail." % FLOOR, ""]
    pct = lambda x: "-" if x is None or x != x else "%.1f%%" % (100 * x)

    def hl_block(title, d):
        h = headline(d)
        L.extend(["## %s" % title, "", "headline %s (%d / %d) | below floor %d (their own pass rate %s) | worst case %s "
                  "(n %d)" % (pct(h["rate"]), h["passed"], h["n"], h["n_below"], pct(h["below_scored_rate"]),
                              pct(h["worst"]), h["n_worst"]), "",
                  "| band | kind | n | headline | below floor | worst case |", "|---|---|---|---|---|---|"])
        for (b, k), g in d.groupby(["band", "kind"]):
            x = headline(g)
            L.append("| %s | %s | %d | %s | %d | %s |" % (b, k, x["n"], pct(x["rate"]), x["n_below"], pct(x["worst"])))
        L.append("")

    v2 = classify(driver.read_jsonl_all(OUT_RS / "main", "main"), ("seed", "held"))
    if len(v2):
        for held in ("strong", "mid"):
            d = v2[v2.held == held]
            if len(d):
                hl_block("ctx_v2, held-out %s (%d targets)" % (held, len(d)), d)
    sw = classify(driver.read_jsonl_all(OUT_RS / "sweep", "sweep"), ("seed", "arm"))
    if len(sw):
        L += ["## ctx_v2, sweep arms (held-out strongest)", "", "| arm | n | headline | below floor | worst case |",
              "|---|---|---|---|---|"]
        for arm, g in sw.groupby("arm"):
            x = headline(g)
            L.append("| %s | %d | %s | %d | %s |" % (arm, x["n"], pct(x["rate"]), x["n_below"], pct(x["worst"])))
        L.append("")
    a1 = classify(driver.read_jsonl_all(OUT_RS / "alpha1", "alpha1"))
    if len(a1):
        hl_block("ctx_v2, alpha = 1 (%d targets)" % len(a1), a1)
    # 2. the paired leakage effect on the frozen subset
    if ASRUN_RS is not None:
        sub = set(ASRUN_TARGETS.read_text().split()) if ASRUN_TARGETS.exists() else None
        ar = classify(driver.read_jsonl_all(ASRUN_RS / "main", "main"), ("seed", "held"))
        for held in ("strong", "mid"):
            if not len(ar) or not len(v2):
                break
            a = ar[(ar.held == held) & (ar.seed.isin(sub) if sub else True)]
            b = v2[(v2.held == held) & (v2.seed.isin(sub) if sub else True)]
            if not len(a) or not len(b):
                continue
            ov, g = paired(a, b)
            L += ["## Leakage effect, held-out %s: as-run (ctx) vs rebuilt (ctx_v2), paired on the subset" % held, "",
                  "%d paired targets (%d at or above the floor, %d below). Headline pairs: as run %s, rebuilt %s, "
                  "effect (as run - rebuilt) %s. Worst case over all pairs: as run %s, rebuilt %s, effect %s." % (
                      ov["n_pairs"], ov["n_head"], ov["n_below"], pct(ov["pass_asrun"]), pct(ov["pass_v2"]),
                      pct(ov["effect"]), pct(ov["pass_asrun_all"]), pct(ov["pass_v2_worst"]), pct(ov["effect_worst"])),
                  "", "| band | kind | pairs | below | as run | rebuilt | effect | as run (all) | rebuilt worst | "
                  "effect worst |", "|---|---|---|---|---|---|---|---|---|---|"]
            for r in g.to_dict("records"):
                L.append("| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s |" % (
                    r["band"], r["kind"], r["n_pairs"], r["n_below"], pct(r["pass_asrun"]), pct(r["pass_v2"]),
                    pct(r["effect"]), pct(r["pass_asrun_all"]), pct(r["pass_v2_worst"]), pct(r["effect_worst"])))
            L.append("")
    # 3. reproduction checks against the run's own eval rows
    ev = []
    for p in sorted((driver.OUT / "main").glob("eval.shard*.jsonl")):
        ev += [json.loads(l) for l in open(p)]
    old = pd.DataFrame(ev)
    if len(old) and "error" in old:
        old = old[old["error"].isna()]
    if len(old) and ASRUN_RS is not None and len(ar):
        old = old[old.get("skip", pd.Series(index=old.index, dtype=object)).isna()]
        L += ["## Reproduction check: the as-run subset re-score vs the run's own main eval rows", "",
              "(same records; differences = hardware / float noise, plus the explicit split on mid-band sets shorter "
              "than 16, protocol-v1.md §3.6)", ""]
        for held in ("strong", "mid"):
            o = old[old.held == held].drop_duplicates("seed", keep="last").set_index("seed")
            n = ar[(ar.held == held) & ar.scored].set_index("seed")
            m = n[HEAD + ["pass"]].join(o[HEAD + ["phi_sup_blind_tk"]], rsuffix="_run", how="inner")
            if not len(m):
                continue
            dev = max(float((m[h] - m[h + "_run"]).abs().max()) for h in HEAD)
            run_pass = classify(o.reset_index().to_dict("records")).set_index("seed")["pass"].reindex(m.index)
            L.append("- held-out %s: %d paired rows; largest |re-score - run| over %s: %.4g; pass agreement %d / %d" % (
                held, len(m), "/".join(HEAD), dev, int((m["pass"] == run_pass).sum()), len(m)))
        L.append("")
    a1r = driver.read_jsonl_all(E062 / "out_alpha1", "alpha1")
    if len(a1) and a1r:
        o1 = classify(a1r).set_index("seed")
        m = a1[a1.scored].set_index("seed")[HEAD].join(o1[HEAD], rsuffix="_run", how="inner")
        L.append("- alpha1 (%s; differs from the run by design unless CTX_DIR is the as-run ctx): %d paired; largest "
                 "|re-score - run| %s" % (CTX_DIR.name,
            len(m), max(float((m[h] - m[h + "_run"]).abs().max()) for h in HEAD) if len(m) else None))
    OUT_RS.mkdir(parents=True, exist_ok=True)
    (OUT_RS / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    log("CTX_DIR %s | EVAL_SPLIT %s | jobs %s | out %s" % (CTX_DIR, os.environ.get("EVAL_SPLIT"), JOBS, OUT_RS))
    R = driver.Runner()
    for j in JOBS:
        t0 = time.time()
        {"main": job_main, "sweep": job_sweep, "random": job_random, "alpha1": job_alpha1}[j](R)
        log("job %s done in %.0fs" % (j, time.time() - t0))
    log("DONE")


if __name__ == "__main__":
    report() if os.environ.get("REPORT") else main()
