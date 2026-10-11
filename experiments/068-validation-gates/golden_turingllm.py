"""DAN-74 GOLDEN REGRESSION SET: does HEAD still reproduce the TuringLLM 15k-circuit run (062 out_full)? GPU.

The 15k run: experiments/062-h100-protocol-v1 MODE=main on 8xH100 (2026-09-27), circuits out_full/out/main/
circuits/<key>.pt, contexts out_full/out/ctx/<key>.pt, eval rows out_full/out/main/eval.shard*.jsonl. Code: the pod
pulled `multi-device` at 538cbed (stage 1's 1,416 targets ran at d5109bc / 564b68f, whose src/ is identical to
538cbed's; only 062/driver.py differs). It fitted through driver.Runner.fit and scored through driver.Runner.score
(049 amp_eval_pass_v2.score_circuit(c, {}, skip_roles=True), EVAL_SPLIT=as_run).

Phases (one process, resumable; OUT is this experiment's own directory, never out_full):
  select   32 targets, fixed seed: fitted, non-vacuous (held-out strongest row, wcm_eval.is_vacuous), non-thin
           where the cell allows, stratified over layer x kind: the 36 (layer, kind) cells minus 4 dropped cells
           (4 distinct layers, kinds rotating), one target per kept cell; an empty kept cell (0.attn: TuringLLM
           attn L0 has no upstream sites, so no circuits) is refilled from a dropped cell. Every layer >= 2 targets.
           -> OUT/selection.json (seed 74: attn 9 / mlp 11 / resid 12; dropped 5.attn, 8.mlp, 9.attn)
  fit      each target refitted locally REPS times (default r0, r1) with the 062 protocol by importing 062's
           driver.Runner (Runner.method(GAMMA, LAM, True) + Runner.fit), from the run's cached context records
           (read-only). -> OUT/fits/<rep>/circuits/<key>.pt, OUT/fits/<rep>/train.jsonl
  score    Runner.score (EVAL_SPLIT=as_run, as the run) on held-out strongest and mid-band for: each local refit,
           and the run's own H100 circuit scored twice ("h100a", "h100b": scorer determinism on this GPU).
           -> OUT/scores.jsonl
  report   (no GPU) -> OUT/report.md and OUT/stores/{r0,r1,h100}.pt (dict stores for 046/compare_stores.py)

Gates (DAN-74):
  noise floor   r0 vs r1 per target: Jaccard, max |d alpha| (046/compare_stores semantics), _tk row diffs
  memberships   HEAD (r0, r1) vs the H100 circuit, per target: EXACT (same set, |d alpha| = 0), EQUIV (Jaccard >= 0.98
                and |d alpha| <= 1e-2), FLOOR (both refits no farther from the H100 circuit than from each other:
                Jaccard >= J(r0, r1) - 0.02 and |d alpha| <= max(1e-2, 2 x d alpha(r0, r1))), else DIFFERENT.
                Pass: no DIFFERENT target.
  eval rows     _tk fields: h100a vs h100b must be identical (same circuit, same GPU); h100a vs the stored H100 rows
                and r0 vs stored are reported (exact / <=1e-6 / <=1e-3 / larger, DAN-8 verdict flips); r0 vs r1 is
                the fit-noise floor for the rows.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python experiments/068-validation-gates/golden_turingllm.py
  env: PHASES (select,fit,score,report)  REPS (r0,r1)  N (32)  SEL_SEED (74)  OUT (./out_golden)
       RUN (062/out_full/out)  HELDS (strong,mid)  LIMIT (first N targets only, smoke tests)
  Run from the repo root (the engine set-up reads outputs/ relative to it). Expect ~3 min start-up, then ~60 s per
  fit + score, ~5 s per extra scoring: 32 x 2 fits + 32 x 2 x 4 scorings ~ 1.5 h on the RTX 5070 Ti.
"""
import json
import math
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
EXP = HERE.parent
E062 = EXP / "062-h100-protocol-v1"
RUN = Path(os.environ.get("RUN", str(E062 / "out_full" / "out")))
OUT = Path(os.environ.get("OUT", str(HERE / "out_golden"))).resolve()
PHASES = [p for p in os.environ.get("PHASES", "select,fit,score,report").split(",") if p]
REPS = [r for r in os.environ.get("REPS", "r0,r1").split(",") if r]
HELDS = [h for h in os.environ.get("HELDS", "strong,mid").split(",") if h]
N = int(os.environ.get("N", 32))
SEL_SEED = int(os.environ.get("SEL_SEED", 74))
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
KINDS = ("attn", "mlp", "resid")
N_LAYER = 12
PINNED_COMMIT = "538cbed"
J_EQUIV, A_EQUIV = 0.98, 1e-2          # 046 compare_stores / DAN-74 fallback
IGNORE = {"held", "arm", "thin", "shard", "eval_split", "n_held_mid", "n_strong_train", "secs", "secs_roles",
          "peak_mem_mb", "error", "_src"}

if str(OUT).startswith(str(RUN.resolve())) or "out_full" in OUT.parts:
    raise SystemExit("OUT must not be inside the 15k run's directory: %s" % OUT)


def log(s):
    print("[%s golden] %s" % (time.strftime("%H:%M:%S"), s), flush=True)


# ----------------------------------------------------------------------------------------------- select
def read_eval_rows(keys=None, helds=("strong",)):
    """The run's eval rows (last per (seed, held), as merge.py keeps), optionally restricted to `keys`."""
    out = {}
    for f in sorted((RUN / "main").glob("eval.shard*.jsonl")):
        for line in open(f):
            if keys is not None and not any('"%s"' % k in line for k in keys):
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "error" in r or "skip" in r or r.get("held") not in helds:
                continue
            if keys is None or r.get("seed") in keys:
                out[(r["seed"], r["held"])] = r
    return out


def select():
    path = OUT / "selection.json"
    if path.exists():
        return json.loads(path.read_text())
    sys.path.insert(0, str(Path("src").resolve()))
    from eval.wcm_eval import is_vacuous
    cells = defaultdict(lambda: {"thick": [], "thin": []})
    for (seed, _), r in read_eval_rows().items():
        if is_vacuous(r) or not (RUN / "main" / "circuits" / ("%s.pt" % seed)).exists():
            continue
        cells[(int(r["layer"]), r["kind"])]["thin" if r.get("thin") else "thick"].append(seed)
    rng = random.Random(SEL_SEED)
    drop_layers = rng.sample(range(N_LAYER), 4)
    k0 = rng.randrange(3)
    dropped = {(l, KINDS[(k0 + j) % 3]) for j, l in enumerate(drop_layers)}
    keys, picked_from = [], {}
    for l in range(N_LAYER):
        for k in KINDS:
            if (l, k) in dropped:
                continue
            pool = sorted(cells[(l, k)]["thick"]) or sorted(cells[(l, k)]["thin"])
            if not pool:
                continue
            keys.append(rng.choice(pool))
            picked_from[keys[-1]] = dict(cell="%d.%s" % (l, k), pool=len(pool),
                                         thin=not cells[(l, k)]["thick"])
    empty = [c for c in ((l, k) for l in range(N_LAYER) for k in KINDS)
             if c not in dropped and not (cells[c]["thick"] or cells[c]["thin"])]
    while len(keys) < N and dropped:                           # refill from a dropped cell if a kept cell was empty
        c = sorted(dropped).pop(0); dropped.discard(c)
        pool = sorted(cells[c]["thick"]) or sorted(cells[c]["thin"])
        if pool:
            keys.append(rng.choice(pool)); picked_from[keys[-1]] = dict(cell="%d.%s" % c, pool=len(pool))
    sel = dict(seed=SEL_SEED, n=len(keys), keys=keys, picked_from=picked_from,
               dropped_cells=["%d.%s" % c for c in sorted(dropped)], empty_cells=["%d.%s" % c for c in empty],
               rule="fitted, non-vacuous (strong), non-thin where possible; one per (layer, kind) cell minus 4 "
                    "dropped cells (4 distinct layers, rotating kinds); random.Random(%d)" % SEL_SEED,
               pinned_commit=PINNED_COMMIT)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sel, indent=1))
    log("selected %d targets (dropped cells %s)" % (len(keys), sel["dropped_cells"]))
    return sel


# ----------------------------------------------------------------------------------------------- engine
_R = {}


def runner():
    """062's driver.Runner, imported with OUT pointed at this experiment (driver.OUT is read at import)."""
    if _R:
        return _R["R"], _R["D"]
    os.environ["OUT"] = str(OUT / "driver_out")
    os.environ.setdefault("MODE", "main")
    sys.path.insert(0, str(E062))
    import driver as D
    assert Path(D.OUT).resolve() == (OUT / "driver_out"), D.OUT
    R = D.Runner()
    _R.update(R=R, D=D)
    return R, D


def record(key):
    import torch
    return torch.load(RUN / "ctx" / ("%s.pt" % key), weights_only=False)[key]


def fit_phase(keys):
    R, D = runner()
    M = R.method(D.GAMMA, D.LAM, True)
    t0 = time.time()
    for rep in REPS:
        d = OUT / "fits" / rep
        (d / "circuits").mkdir(parents=True, exist_ok=True)
        with open(d / "train.jsonl", "a") as fh:
            for n_, key in enumerate(keys):
                path = d / "circuits" / ("%s.pt" % key)
                if path.exists():
                    continue
                try:
                    rec = record(key)
                    arm, thin = R.train_arm(rec)
                    c, t_fit = R.fit(M, rec, key, arm, path, train_fh=fh, label=dict(rep=rep))
                    log("fit %s %2d/%d %-16s n=%s %.0fs (total %.0f min)" % (
                        rep, n_ + 1, len(keys), key, None if c is None else D.n_nodes(c), t_fit,
                        (time.time() - t0) / 60))
                except Exception:  # noqa: BLE001
                    traceback.print_exc()


def score_phase(keys):
    import torch
    R, _ = runner()
    path = OUT / "scores.jsonl"
    done = set()
    if path.exists():
        done = {(r["seed"], r["held"], r["_src"]) for r in map(json.loads, open(path)) if "error" not in r}
    srcs = [(rep, OUT / "fits" / rep / "circuits") for rep in REPS]
    srcs += [("h100a", RUN / "main" / "circuits"), ("h100b", RUN / "main" / "circuits")]
    with open(path, "a") as fh:
        for key in keys:
            rec = record(key)
            for src, cdir in srcs:
                cp = cdir / ("%s.pt" % key)
                if not cp.exists():
                    continue
                c = None
                for held in HELDS:
                    if (key, held, src) in done or rec.get(held) is None:
                        continue
                    if c is None:
                        c = torch.load(cp, weights_only=False)
                    if c is None:
                        break
                    try:
                        row = R.score(c, rec, held)
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc()
                        row = dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))
                    fh.write(json.dumps(dict(row, held=held, _src=src)) + "\n"); fh.flush()
            log("scored %s" % key)


# ----------------------------------------------------------------------------------------------- report
def members(c):
    out = {}
    for n in c.nodes.values():
        if n.metadata.get("role") == "seed":
            continue
        f = n.metadata["feature_id"]
        out[(f.layer, f.kind, f.index)] = n.metadata.get("amplitude")
    return out


def cmp_members(a, b):
    """046 compare_stores semantics: Jaccard of the member sets, max |d alpha| over the intersection."""
    inter = set(a) & set(b)
    j = len(inter) / max(len(set(a) | set(b)), 1)
    da = max((abs((a[i] or 0) - (b[i] or 0)) for i in inter), default=0.0)
    return dict(j=j, da=da, na=len(a), nb=len(b), exact=(set(a) == set(b) and da == 0.0))


def _flat(v, prefix=""):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from _flat(x, "%s.%s" % (prefix, k))
    elif isinstance(v, (list, tuple)):
        for j, x in enumerate(v):
            yield from _flat(x, "%s[%d]" % (prefix, j))
    else:
        yield prefix, v


def cmp_rows(pairs, tk_only=True):
    """Field-level agreement of row pairs over the _tk fields (or all score fields): exact / <=1e-6 / <=1e-3 /
    larger, None mismatches, the largest |diff|, DAN-8 verdict flips on held-out strongest."""
    from eval.wcm_eval import is_vacuous, passes
    s = dict(rows=0, fields=0, exact=0, le_1e6=0, le_1e3=0, gt_1e3=0, none_mismatch=0, missing=0, max_abs=0.0,
             max_at=None, flips=0, identical_rows=0)
    for label, a, b in pairs:
        s["rows"] += 1
        fa = {k: v for f in a if f not in IGNORE and (not tk_only or "_tk" in f) for k, v in _flat(a[f], f)}
        fb = {k: v for f in b if f not in IGNORE and (not tk_only or "_tk" in f) for k, v in _flat(b[f], f)}
        s["missing"] += len(set(fa) ^ set(fb))
        same = not (set(fa) ^ set(fb))
        for f in set(fa) & set(fb):
            x, y = fa[f], fb[f]
            s["fields"] += 1
            if x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)):
                s["exact"] += 1
                continue
            same = False
            if x is None or y is None or isinstance(x, bool) or isinstance(y, bool) \
                    or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                s["none_mismatch"] += 1
                continue
            d = abs(float(x) - float(y))
            s["le_1e6" if d <= 1e-6 else ("le_1e3" if d <= 1e-3 else "gt_1e3")] += 1
            if d > s["max_abs"]:
                s["max_abs"], s["max_at"] = d, "%s %s" % (label, f)
        s["identical_rows"] += same
        if a.get("held") == "strong" or b.get("held") == "strong":
            s["flips"] += int((passes(a) and not is_vacuous(a)) != (passes(b) and not is_vacuous(b)))
    return s


def report(keys):
    import torch
    sys.path.insert(0, str(Path("src").resolve()))
    lines = ["# DAN-74 golden regression set (%s)\n" % time.strftime("%Y-%m-%d %H:%M"),
             "%d targets (selection.json, seed %d); the 15k run = %s (pinned commit %s). Local refits: %s.\n"
             % (len(keys), SEL_SEED, RUN, PINNED_COMMIT, ", ".join(REPS))]
    circ = {"h100": {}}
    for rep in REPS:
        circ[rep] = {}
    for key in keys:
        for src, cdir in [("h100", RUN / "main" / "circuits")] + [(r, OUT / "fits" / r / "circuits") for r in REPS]:
            p = cdir / ("%s.pt" % key)
            if p.exists():
                c = torch.load(p, weights_only=False)
                if c is not None:
                    circ[src][key] = c
    (OUT / "stores").mkdir(parents=True, exist_ok=True)
    for src, d in circ.items():
        if d:
            torch.save(d, OUT / "stores" / ("%s.pt" % src))

    # ---- memberships and alpha
    lines += ["## Memberships and α (046 compare_stores semantics)\n",
              "| target | n H100 | n r0 / r1 | J(r0,r1) | dα(r0,r1) | J(r0,H100) | dα(r0,H100) | J(r1,H100) | "
              "dα(r1,H100) | status |", "|---|---|---|---|---|---|---|---|---|---|"]
    status_count = defaultdict(int)
    floors_j, floors_a = [], []
    for key in keys:
        h = circ["h100"].get(key)
        loc = [circ[r].get(key) for r in REPS[:2]]
        if h is None or any(c is None for c in loc) or len(REPS) < 2:
            lines.append("| %s | %s | - | | | | | | | MISSING |" % (key, "-" if h is None else len(members(h))))
            status_count["MISSING"] += 1
            continue
        mh, m0, m1 = members(h), members(loc[0]), members(loc[1])
        fl, a0, a1 = cmp_members(m0, m1), cmp_members(m0, mh), cmp_members(m1, mh)
        floors_j.append(fl["j"]); floors_a.append(fl["da"])
        if a0["exact"] and a1["exact"]:
            st = "EXACT"
        elif min(a0["j"], a1["j"]) >= J_EQUIV and max(a0["da"], a1["da"]) <= A_EQUIV:
            st = "EQUIV"
        elif (min(a0["j"], a1["j"]) >= fl["j"] - 0.02
              and max(a0["da"], a1["da"]) <= max(A_EQUIV, 2 * fl["da"])):
            st = "FLOOR"
        else:
            st = "DIFFERENT"
        status_count[st] += 1
        lines.append("| %s | %d | %d / %d | %.4f | %.2g | %.4f | %.2g | %.4f | %.2g | %s |" % (
            key, len(mh), len(m0), len(m1), fl["j"], fl["da"], a0["j"], a0["da"], a1["j"], a1["da"], st))
    srt = sorted(floors_j)
    lines += ["", "Status counts: %s. Local noise floor J(r0, r1): min %.4f, median %.4f; dα(r0, r1) max %.2g. "
              "**Membership gate: %s**\n" % (
                  dict(status_count), srt[0] if srt else float("nan"), srt[len(srt) // 2] if srt else float("nan"),
                  max(floors_a) if floors_a else float("nan"),
                  "PASS" if floors_j and not status_count.get("DIFFERENT") and not status_count.get("MISSING")
                  else ("FAIL" if status_count.get("DIFFERENT") else "INCOMPLETE"))]

    # ---- eval rows
    sp = OUT / "scores.jsonl"
    mine = defaultdict(dict)
    if sp.exists():
        for r in map(json.loads, open(sp)):
            if "error" not in r:
                mine[(r["seed"], r["held"])][r["_src"]] = r
    stored = read_eval_rows(set(keys), helds=tuple(HELDS))
    lines += ["## Eval rows (049 v2 scorer, EVAL_SPLIT=as_run)\n",
              "| comparison | held | rows | identical rows | fields | exact | <=1e-6 | <=1e-3 | >1e-3 | None/type | "
              "missing | max abs diff (at) | pass flips |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    gate_rows = None
    for title, a_src, b_src in (("h100a vs h100b (same circuit, same GPU: must be identical)", "h100a", "h100b"),
                                ("h100a vs stored H100 row (scorer, RTX vs H100)", "h100a", "stored"),
                                ("r0 vs r1 (fit noise floor)", "r0", "r1"),
                                ("r0 vs stored H100 row (HEAD refit vs the run)", "r0", "stored"),
                                ("r1 vs stored H100 row", "r1", "stored")):
        for held in HELDS:
            pairs = []
            for key in keys:
                m = mine.get((key, held), {})
                a = m.get(a_src)
                b = stored.get((key, held)) if b_src == "stored" else m.get(b_src)
                if a is not None and b is not None:
                    pairs.append((key, a, b))
            s = cmp_rows(pairs)
            if a_src == "h100a" and b_src == "h100b" and s["rows"]:
                gate_rows = (gate_rows or 0) + (s["rows"] - s["identical_rows"])
            lines.append("| %s | %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | %.3g (%s) | %d |" % (
                title, held, s["rows"], s["identical_rows"], s["fields"], s["exact"], s["le_1e6"], s["le_1e3"],
                s["gt_1e3"], s["none_mismatch"], s["missing"], s["max_abs"], s["max_at"] or "-", s["flips"]))
    lines.append("\nScorer determinism (h100a vs h100b identical _tk rows): **%s**" % (
        "n/a" if gate_rows is None else ("PASS" if gate_rows == 0 else "FAIL (%d rows differ)" % gate_rows)))
    txt = "\n".join(lines)
    (OUT / "report.md").write_text(txt, encoding="utf-8")
    print(txt)
    log("compare_stores (046): python experiments/046-fit-profiling/compare_stores.py %s %s"
        % (OUT / "stores" / "r0.pt", OUT / "stores" / "h100.pt"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = select()
    keys = sel["keys"][:LIMIT] if LIMIT else sel["keys"]
    if "fit" in PHASES:
        fit_phase(keys)
    if "score" in PHASES:
        score_phase(keys)
    if "report" in PHASES:
        report(keys)


if __name__ == "__main__":
    main()
