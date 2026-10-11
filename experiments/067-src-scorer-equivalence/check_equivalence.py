"""DAN-73 GPU CHECK: the src scorer (src/eval/wcm_eval.py, wcm_specificity.py) reproduces the 062 run's scorer.

On a stratified handful of the 15k run's targets, with the run's own circuits and cached context records:

  (a) score each target on held-out strongest and mid-band with the experiment scorer (049 amp_eval_pass_v2,
      EVAL_SPLIT=as_run, exactly as 062 driver.Runner.score) and with the src scorer in split="as_run";
      specificity (held-out strongest) with 056 specificity.score and src specificity_rows
  (b) compare every field of the two (same GPU, same process): exact matches and max |diff|
  (c) compare both with the stored rows of the run (out_full/out/main/eval.shard*.jsonl, spec.shard*.jsonl). The run
      was on H100s, so expect float-level differences here; the report also counts pass-rule verdict flips
  (d) for the padded mid-band targets (§3.6: mid-band rows with n_pos < 64), score held-out mid-band in explicit mode
      (both scorers) and compare with the re-scored rows in 062/out_rescore_mid/eval_mid.jsonl

Resumable: every scored row is appended to OUT/rows.jsonl (OUT/spec_rows.jsonl) keyed by (seed, held, impl, split)
and skipped on a rerun; the selection is cached in OUT/selection.json. The report is printed and written to
OUT/report.md. Run from the repo root (the engine set-up reads outputs/ relative to it):

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python experiments/067-src-scorer-equivalence/check_equivalence.py
  env: RUN (default 062/out_full/out)  RESCORE (default 062/out_rescore_mid/eval_mid.jsonl)  OUT (default ./out)
       N_STRAT (12 targets over depth band x site kind)  N_PAD (4 padded mid-band targets)  KEYS (comma list:
       score exactly these, as stratified targets)  SPEC (1: also specificity; 0 skips)  REPORT=1 (report only, no GPU)
"""
import json
import math
import os
import sys
import time
import traceback
import zlib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
EXP = HERE.parent
E062 = EXP / "062-h100-protocol-v1"
RUN = Path(os.environ.get("RUN", str(E062 / "out_full" / "out")))
RESCORE = Path(os.environ.get("RESCORE", str(E062 / "out_rescore_mid" / "eval_mid.jsonl")))
OUT = Path(os.environ.get("OUT", str(HERE / "out")))
N_STRAT = int(os.environ.get("N_STRAT", 12))
N_PAD = int(os.environ.get("N_PAD", 4))
KEYS = [k for k in os.environ.get("KEYS", "").split(",") if k]
SPEC = os.environ.get("SPEC", "1") == "1"
ROWS, SPEC_ROWS, SEL = OUT / "rows.jsonl", OUT / "spec_rows.jsonl", OUT / "selection.json"

# fields that are not scores: run bookkeeping the driver added or overwrote, labels, timings
IGNORE = {"held", "arm", "thin", "shard", "eval_split", "n_held_mid", "n_strong_train", "_impl", "_split",
          "secs", "secs_roles", "peak_mem_mb", "error"}


def depth_band(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def stored(pattern, keys, fields=("seed", "held")):
    """The run's rows for `keys` (last row per key, as merge.py keeps), keyed by `fields`."""
    out = {}
    for f in sorted(pattern.parent.glob(pattern.name)):
        for line in open(f):
            if not any('"%s"' % k in line for k in keys):
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("seed") in keys and "error" not in r:
                out[tuple(r.get(x) for x in fields)] = r
    return out


# ----------------------------------------------------------------------------------------------- selection
def select():
    if SEL.exists():
        return json.loads(SEL.read_text())
    if KEYS:
        sel = dict(strat=KEYS, pad=[])
    else:
        cells, pad = defaultdict(list), []
        for f in sorted((RUN / "main").glob("eval.shard*.jsonl")):
            for line in open(f):
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if "error" in r or "skip" in r:
                    continue
                if r.get("held") == "strong":
                    cells[(depth_band(r["layer"]), r["kind"])].append(r["seed"])
                elif r.get("held") == "mid" and r["n_pos"] < 64:
                    pad.append(r["seed"])
        h = lambda k: zlib.crc32(k.encode())                       # deterministic, spread over the cell
        cells = {c: sorted(set(v), key=h) for c, v in sorted(cells.items())}
        strat, i = [], 0
        while len(strat) < N_STRAT and any(i < len(v) for v in cells.values()):
            strat += [v[i] for v in cells.values() if i < len(v)][:N_STRAT - len(strat)]
            i += 1
        rescored = {json.loads(l)["seed"] for l in open(RESCORE)} if RESCORE.exists() else set()
        pad = sorted(set(pad), key=lambda k: (k not in rescored, h(k)))[:N_PAD]
        sel = dict(strat=strat, pad=pad)
    OUT.mkdir(parents=True, exist_ok=True)
    SEL.write_text(json.dumps(sel, indent=1))
    return sel


# ----------------------------------------------------------------------------------------------- scoring
def done_keys(path, fields):
    if not path.exists():
        return set()
    return {tuple(r.get(f) for f in fields) for r in map(json.loads, open(path)) if "error" not in r}


def score_all(sel):
    import torch
    sys.path.insert(0, str(EXP / "049-circuit-graph"))
    sys.path.insert(0, str(EXP / "056-specificity"))
    os.environ.setdefault("TAG", "dan73_unused")
    os.environ["CTR_SOURCE"] = "close"                            # the 062 driver's setting (read at import)
    import amp_eval_pass_v2 as V
    import specificity as S056
    import store.context as SC
    from circuit.protocol_contexts import eval_contexts, training_arm
    from eval.wcm_eval import neg_ctx_store_fallback, score_circuit
    from eval.wcm_specificity import specificity_rows
    from pipeline.component_index import component_idx

    G = V.setup()                                                 # loads the stores (SC.neg_ctx) as the run did
    inference, bank, dev = G["inference"], G["bank"], G["device"]
    taps = S056.make_patchers(G)
    done = done_keys(ROWS, ("seed", "held", "_impl", "_split"))
    sdone = done_keys(SPEC_ROWS, ("seed", "_impl"))
    fh, fs = open(ROWS, "a"), open(SPEC_ROWS, "a")
    jobs = [(k, "strong", "as_run") for k in sel["strat"]] + [(k, "mid", "as_run") for k in sel["strat"] + sel["pad"]]
    jobs += [(k, "mid", "explicit") for k in sel["pad"]]
    t0 = time.time()
    for n_, (key, held, split) in enumerate(jobs):
        if (key, held, "exp", split) in done and (key, held, "src", split) in done:
            continue
        rec = torch.load(RUN / "ctx" / ("%s.pt" % key), weights_only=False)[key]
        if rec.get(held) is None:
            continue
        c = torch.load(RUN / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
        l, kd, i = key.split(".")
        comp = component_idx(int(l), G["KINDS"].index(kd), G["NK"])
        pd_, selr = eval_contexts(rec, held, dev)
        tag = dict(held=held, _split=split)
        for impl in ("exp", "src"):
            if (key, held, impl, split) in done:
                continue
            try:
                if impl == "exp":                                 # driver.Runner.score: protocol_harness injection
                    G["M0"].build_probe_dataset = lambda comp_, i_, _p=pd_: _p
                    G["M0"]._neg_context_selector = lambda _s=selr: _s
                    V.EVAL_SPLIT = split
                    row = V.score_circuit(c, {}, skip_roles=True)
                else:
                    row = score_circuit(inference, bank, c, pd_, selr.tokens, split=split,
                                        ctr_fallback=neg_ctx_store_fallback(SC.neg_ctx, comp, int(i)))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row = dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))
            fh.write(json.dumps(dict(row, _impl=impl, **tag)) + "\n"); fh.flush()
        if SPEC and held == "strong" and split == "as_run":
            arm = training_arm(rec)[0]
            for impl in ("exp", "src"):
                if (key, impl) in sdone:
                    continue
                try:
                    if impl == "exp":
                        G["M0"].build_probe_dataset = lambda comp_, i_, _p=pd_: _p
                        G["M0"]._neg_context_selector = lambda _s=selr: _s
                        rows = S056.score(G, c, V, taps[0], taps[1], arm)
                    else:
                        rows = specificity_rows(inference, bank, c, pd_, selr.tokens, arm=arm, split="as_run")
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    rows = [dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))]
                for r in rows:
                    fs.write(json.dumps(dict(r, _impl=impl)) + "\n")
                fs.flush()
        print("%3d/%d %-16s %-6s %-8s %.0fs" % (n_ + 1, len(jobs), key, held, split, time.time() - t0), flush=True)
    fh.close(); fs.close()


# ----------------------------------------------------------------------------------------------- comparison
def _flat(v, prefix=""):
    """Scalar leaves of a row value (lists element-wise, dicts by key)."""
    if isinstance(v, dict):
        for k, x in v.items():
            yield from _flat(x, "%s.%s" % (prefix, k))
    elif isinstance(v, (list, tuple)):
        for j, x in enumerate(v):
            yield from _flat(x, "%s[%d]" % (prefix, j))
    else:
        yield prefix, v


def compare(pairs):
    """pairs: [(label, row_a, row_b)]. Field-level agreement: exact, |diff| <= 1e-6 / 1e-3 / larger, None or type
    mismatches, missing fields; the largest |diff| with its field and row; pass-rule verdict flips."""
    from eval.wcm_eval import is_vacuous, passes
    s = dict(rows=0, fields=0, exact=0, le_1e6=0, le_1e3=0, gt_1e3=0, none_mismatch=0, other_mismatch=0,
             missing=0, max_abs=0.0, max_at=None, verdict_flips=0, big=defaultdict(int))
    for label, a, b in pairs:
        s["rows"] += 1
        fa = {k: v for f in a if f not in IGNORE for k, v in _flat(a[f], f)}
        fb = {k: v for f in b if f not in IGNORE for k, v in _flat(b[f], f)}
        s["missing"] += len(set(fa) ^ set(fb))
        for f in set(fa) & set(fb):
            x, y = fa[f], fb[f]
            s["fields"] += 1
            if x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)):
                s["exact"] += 1
            elif x is None or y is None:
                s["none_mismatch"] += 1; s["big"][f] += 1
            elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                d = abs(float(x) - float(y))
                key = "le_1e6" if d <= 1e-6 else ("le_1e3" if d <= 1e-3 else "gt_1e3")
                s[key] += 1
                if d > 1e-3:
                    s["big"][f] += 1
                if d > s["max_abs"]:
                    s["max_abs"], s["max_at"] = d, "%s %s" % (label, f)
            else:
                s["other_mismatch"] += 1; s["big"][f] += 1
        if "free0_tk" in a and "free0_tk" in b and a.get("held", b.get("held")) == "strong":
            va = passes(a) and not is_vacuous(a)
            vb = passes(b) and not is_vacuous(b)
            s["verdict_flips"] += int(va != vb)
    s["big"] = dict(sorted(s["big"].items(), key=lambda kv: -kv[1])[:12])
    return s


def report():
    sel = json.loads(SEL.read_text())
    keys = set(sel["strat"]) | set(sel["pad"])
    mine = defaultdict(dict)
    for r in map(json.loads, open(ROWS)):
        if "error" not in r:
            mine[(r["seed"], r["held"], r["_split"])][r["_impl"]] = r
    run = stored(RUN / "main" / "eval.shard*.jsonl", keys)
    resc = {r["seed"]: r for r in map(json.loads, open(RESCORE)) if "error" not in r} if RESCORE.exists() else {}
    errors = [json.loads(l) for l in open(ROWS) if '"error"' in l]
    sections = []

    def section(title, pairs):
        sections.append((title, compare(pairs)))

    for held in ("strong", "mid"):
        ks = [k for k in sorted(mine) if k[1] == held and k[2] == "as_run"]
        both = [k for k in ks if {"exp", "src"} <= set(mine[k])]
        section("(b) src vs experiment scorer, as_run, held %s (same GPU)" % held,
                [(k[0], mine[k]["exp"], mine[k]["src"]) for k in both])
        for impl in ("exp", "src"):
            section("(c) %s vs stored H100 rows, as_run, held %s" % (impl, held),
                    [(k[0], run[(k[0], held)], mine[k][impl]) for k in ks if impl in mine[k] and (k[0], held) in run])
    ks = [k for k in sorted(mine) if k[2] == "explicit"]
    section("(d) src vs experiment scorer, explicit, held mid (same GPU)",
            [(k[0], mine[k]["exp"], mine[k]["src"]) for k in ks if {"exp", "src"} <= set(mine[k])])
    for impl in ("exp", "src"):
        section("(d) %s explicit vs out_rescore_mid rows" % impl,
                [(k[0], resc[k[0]], mine[k][impl]) for k in ks if impl in mine[k] and k[0] in resc])
    if SPEC_ROWS.exists():
        sp = defaultdict(dict)
        for r in map(json.loads, open(SPEC_ROWS)):
            if "error" not in r:
                sp[(r["seed"], r["pi"])][r["_impl"]] = r
        run_sp = stored(RUN / "main" / "spec.shard*.jsonl", keys, ("seed", "pi"))
        section("(b) specificity: src vs 056 (same GPU)",
                [("%s %s" % k, v["exp"], v["src"]) for k, v in sorted(sp.items()) if {"exp", "src"} <= set(v)])
        section("(c) specificity: src vs stored H100 rows",
                [("%s %s" % k, run_sp[k], v["src"]) for k, v in sorted(sp.items()) if "src" in v and k in run_sp])

    lines = ["# DAN-73 src scorer equivalence (%s)\n" % time.strftime("%Y-%m-%d %H:%M"),
             "Targets: %d stratified (%s), %d padded mid-band (%s). Scoring errors: %d.\n" % (
                 len(sel["strat"]), ", ".join(sel["strat"]), len(sel["pad"]), ", ".join(sel["pad"]), len(errors)),
             "| comparison | rows | fields | exact | <=1e-6 | <=1e-3 | >1e-3 | None mismatch | other | missing | "
             "max abs diff (at) | pass flips |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for title, s in sections:
        lines.append("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | %.3g (%s) | %d |" % (
            title, s["rows"], s["fields"], s["exact"], s["le_1e6"], s["le_1e3"], s["gt_1e3"], s["none_mismatch"],
            s["other_mismatch"], s["missing"], s["max_abs"], s["max_at"] or "-", s["verdict_flips"]))
    lines.append("\nFields with a None / type mismatch or |diff| > 1e-3 (count of rows), per comparison:\n")
    for title, s in sections:
        if s["big"]:
            lines.append("- %s: %s" % (title, s["big"]))
    for e in errors:
        lines.append("- ERROR %s %s %s: %s" % (e["seed"], e.get("held"), e.get("_impl"), e["error"]))
    txt = "\n".join(lines)
    (OUT / "report.md").write_text(txt, encoding="utf-8")
    print(txt)


def main():
    if os.environ.get("REPORT") != "1":
        sel = select()
        print("selection: %d stratified, %d padded mid-band -> %s" % (len(sel["strat"]), len(sel["pad"]), OUT),
              flush=True)
        score_all(sel)
    report()


if __name__ == "__main__":
    main()
