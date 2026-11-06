"""104 part B, target draw (rule frozen 2026-11-01 before any 104 fit; no score is read): ~24 targets per public model,
8 per depth band, kinds mixed, from the 082 preliminary run's frame lists (082/prelim/<model>/targets_<model>_main.txt,
drawn by make_targets_public.py --frame fired --seed 0 on the local 069 smoke stores) with records built here from the
same local stores (062 driver Runner.contexts = src build_record, group-aware split; no data pass, no download).

Bands: gpt2-small (12 layers) L0-4 / L5-7 / L8-11, as TuringLLM (exp 093 / 102). pythia-70m (6 layers, L0 has resid
only: 0.attn / 0.mlp have no upstream SAE site) L0-1 / L2-3 / L4-5, equal thirds of depth.
Kinds per band: 8 slots dealt round robin over the model's kinds, starting one kind later in each band
(pythia attn / mlp / resid: 3-3-2, 3-3-2, 3-3-2 rotated -> 8 per kind overall; gpt2 attn / resid_mid / mlp / resid: 2 each).
Draw: per (band, kind) cell, sorted(cell) shuffled by one random.Random(SEED) over the cells in sorted order;
candidates are tried in that order. Eligible = a record with a strongest pool (strong not None), >= 8 held-out
strongest contexts and >= 32 contrast contexts (>= 8 held out); records on the smoke stores are often thin (n_mid 0,
fewer than 64 strongest), which the rule accepts. At most MAX_TRY candidates per cell; a cell that cannot fill its
slots passes them to the band's other kinds in deal order (logged). Every attempt is logged in out/<model>/select.jsonl.

  wsl (repo root): FAMILY=pythia-70m PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \\
      .venv/bin/python -u experiments/104-v3-cross-model/select_targets.py
  writes targets_<model>.txt (+ .json) next to this file; records in out/<model>/rec/ctx (resumable: cached records)
"""
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 20261101
PER_BAND = 8
MAX_TRY = 40
MIN_HELD, MIN_NEG = 8, 32
FAMILY = os.environ.get("FAMILY", "pythia-70m")
BANDS = {"pythia-70m": (("L0-1", 0, 1), ("L2-3", 2, 3), ("L4-5", 4, 5)),
         "gpt2-small": (("L0-4", 0, 4), ("L5-7", 5, 7), ("L8-11", 8, 11))}[FAMILY]
KINDS = {"pythia-70m": ("attn", "mlp", "resid"), "gpt2-small": ("attn", "resid_mid", "mlp", "resid")}[FAMILY]
OUT = HERE / "out" / FAMILY
os.environ.setdefault("DATA", str(ROOT / "data_public" / FAMILY))
os.environ.setdefault("STORES", str(ROOT / "outputs_public" / FAMILY))
os.environ["OUT"] = str(OUT / "rec")          # Runner builds records into OUT/ctx; v1 (no WCM marker written)
os.environ.pop("CTX_DIR", None)
os.environ.pop("WCM_VERSION", None)
os.environ.setdefault("SAE_PRELOAD", "0")


def band_of(layer):
    return next(b for b, lo, hi in BANDS if lo <= layer <= hi)


def main():
    frame = [k for k in (ROOT / "experiments" / "082-public-runs" / "prelim" / FAMILY /
                         ("targets_%s_main.txt" % FAMILY)).read_text().split() if k]
    cells = {}
    for k in frame:
        l, kind, _ = k.split(".")
        cells.setdefault((band_of(int(l)), kind), []).append(k)
    rng = random.Random(SEED)
    order = {c: rng.sample(sorted(v), len(v)) for c, v in sorted(cells.items())}
    quota = {}
    for bi, (b, _, _) in enumerate(BANDS):
        deal = [KINDS[(i + bi) % len(KINDS)] for i in range(PER_BAND)]
        for kind in deal:
            quota[(b, kind)] = quota.get((b, kind), 0) + 1
    sys.path.insert(0, str(ROOT / "experiments" / "062-h100-protocol-v1"))
    import driver
    assert driver.FAMILY == FAMILY and driver.WCM_VERSION == "v1"
    R = driver.Runner()
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "select.jsonl"
    seen = {}
    if log_path.exists():
        for line in open(log_path):
            try:
                r = json.loads(line)
                seen[r["seed"]] = r
            except ValueError:
                continue
    flog = open(log_path, "a")

    def attempt(key):
        if key in seen:
            return seen[key]
        t = time.time()
        r = dict(seed=key)
        try:
            rec = R.contexts(key)
            s, neg = rec.get("strong"), rec.get("neg")
            r.update(n_top=rec.get("n_top"), n_mid=rec.get("n_mid"),
                     n_held=None if s is None else len(s["held"]), n_train=None if s is None else len(s["train"]),
                     n_neg=None if neg is None else len(neg))
            r["eligible"] = bool(s is not None and r["n_held"] >= MIN_HELD and (r["n_neg"] or 0) >= MIN_NEG)
            if not r["eligible"]:
                r["why"] = ("no_strong_pool" if s is None else "held<%d" % MIN_HELD if r["n_held"] < MIN_HELD
                            else "neg<%d" % MIN_NEG)
        except Exception as e:  # noqa: BLE001
            r.update(eligible=False, why="error", error="%s: %s" % (type(e).__name__, str(e)[:300]))
        r["t"] = round(time.time() - t, 1)
        flog.write(json.dumps(r) + "\n"); flog.flush()
        seen[key] = r
        print("[%s] %-18s %s %s" % (time.strftime("%H:%M:%S"), key, "ok" if r["eligible"] else r.get("why"),
                                    {k: r.get(k) for k in ("n_top", "n_held", "n_neg")}), flush=True)
        return r

    picked, notes = {}, []
    for bi, (b, _, _) in enumerate(BANDS):
        short, pos = 0, {}                     # pos[kind] = next candidate index of the cell (deterministic on resume)
        for kind in KINDS:
            want, got, i = quota.get((b, kind), 0), [], 0
            cand = order.get((b, kind), [])
            while len(got) < want and i < min(MAX_TRY, len(cand)):
                if attempt(cand[i])["eligible"]:
                    got.append(cand[i])
                i += 1
            picked[(b, kind)], pos[kind] = got, i
            if len(got) < want:
                short += want - len(got)
                notes.append("%s/%s: %d of %d slots filled (%d tried)" % (b, kind, len(got), want, i))
        # a band's unfilled slots go to its other kinds, one at a time in deal order, up to 2 x MAX_TRY tries per cell
        k_i, idle = 0, 0
        while short > 0 and idle < len(KINDS):
            kind = KINDS[(k_i + bi) % len(KINDS)]
            k_i += 1
            cand, found = order.get((b, kind), []), False
            while pos[kind] < min(2 * MAX_TRY, len(cand)):
                key = cand[pos[kind]]
                pos[kind] += 1
                if attempt(key)["eligible"]:
                    picked[(b, kind)].append(key)
                    short -= 1
                    found = True
                    notes.append("%s: slot passed to %s (%s)" % (b, kind, key))
                    break
            idle = 0 if found else idle + 1
        if short > 0:
            notes.append("%s: %d slots unfilled" % (b, short))
    keys = [k for b, _, _ in BANDS for kind in KINDS for k in picked.get((b, kind), [])]
    (HERE / ("targets_%s.txt" % FAMILY)).write_text("\n".join(keys) + "\n")
    (HERE / ("targets_%s.json" % FAMILY)).write_text(json.dumps(dict(
        family=FAMILY, seed=SEED, per_band=PER_BAND, max_try=MAX_TRY, min_held=MIN_HELD, min_neg=MIN_NEG,
        bands=BANDS, kinds=KINDS, quota={"%s/%s" % c: q for c, q in quota.items()},
        cells={"%s/%s" % c: v for c, v in picked.items()}, notes=notes, n=len(keys),
        n_tried=len(seen), n_eligible_tried=sum(1 for r in seen.values() if r["eligible"])), indent=1))
    print("104 select %s: %d targets; %s" % (FAMILY, len(keys), notes), flush=True)


if __name__ == "__main__":
    main()
