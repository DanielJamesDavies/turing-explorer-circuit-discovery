"""The 16-target panel of experiment 108 (CPU; run once, writes targets.txt + targets.json).

Source: latent_pilot's 48 dev targets (../107-true-baselines/latent_pilot/targets_dev48.txt: 093's 72 dev targets minus
adaptive_lambda's 24, never used to choose anything), 16 per depth band (L1-4 shallow, L5-7 mid, L8-11 deep) over the
three site kinds. Eligible = the record (062 out_full/out/ctx, the CTX_DIR latent_pilot's local fits used) trains arm
B with exactly 48 rows (32 strongest + 16 mid-band) and has both held-out sets, so B48 = the full training set and
every arm scores on held-out strongest + mid. (The two thin dev48 targets, 7.attn.38584 and 8.attn.39566, fail this.)

Selection, random.Random(20261106): each band x kind cell shuffled (cells in the order deep-resid, then per kind attn /
mlp / resid the bands shallow / mid / deep); round 1 = one target per cell, round 2 = a second per cell; the panel is
round 1 then round 2, cut at 16 (so the first N of any cut spread over the cells). Forced: the deepest target 11.resid.10297 (35 upstream sites:
the VRAM and time worst case) is the deep-resid cell's round-1 pick and goes first; the first shallow target goes
second (the timing stage fits these two). The order is the run order: plan.py keeps the first N.

  PYTHONPATH=src .venv/bin/python experiments/108-batch-size/make_targets.py
"""
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "experiments" / "107-true-baselines" / "latent_pilot" / "targets_dev48.txt"
CTX = REPO / "experiments" / "062-h100-protocol-v1" / "out_full" / "out" / "ctx"
SEED = 20261106
DEEPEST = "11.resid.10297"
N = 16
KINDS = ("attn", "mlp", "resid")


def band(key):
    l = int(key.split(".")[0])
    return "shallow" if l <= 4 else ("mid" if l <= 7 else "deep")


def eligible(key):
    p = CTX / ("%s.pt" % key)
    if not p.exists():
        return False, "no record"
    rec = torch.load(p, weights_only=False, map_location="cpu")[key]
    if rec.get("strong") is None or rec.get("mid") is None:
        return False, "thin / no strongest pool"
    n_tr = len(rec["D_train"]) + len(rec["B_mid"])
    if n_tr != 48:
        return False, "%d training rows" % n_tr
    if not len(rec["strong"]["held"]) or not len(rec["mid"]["held"]):
        return False, "missing a held-out set"
    return True, "ok (strong held %d, mid held %d)" % (len(rec["strong"]["held"]), len(rec["mid"]["held"]))


def main():
    keys = [k for k in SRC.read_text().split() if k]
    assert len(keys) == 48 and DEEPEST in keys
    info, ok = {}, []
    for k in keys:
        e, why = eligible(k)
        info[k] = why
        if e:
            ok.append(k)
        else:
            print("  excluded %-16s %s" % (k, why))
    rng = random.Random(SEED)
    cells = []
    order = [("deep", "resid")] + [(b, kd) for kd in KINDS for b in ("shallow", "mid", "deep")
                                   if (b, kd) != ("deep", "resid")]
    for b, kd in order:
        c = sorted(k for k in ok if band(k) == b and k.split(".")[1] == kd)
        rng.shuffle(c)
        if b == "deep" and kd == "resid":
            c.remove(DEEPEST)
            c.insert(0, DEEPEST)
        cells.append(((b, kd), c))
    panel = [c[0] for _, c in cells if c] + [c[1] for _, c in cells if len(c) > 1]
    panel = panel[:N]
    assert panel[0] == DEEPEST and band(panel[1]) == "shallow", panel[:2]
    (HERE / "targets.txt").write_text("\n".join(panel) + "\n")
    (HERE / "targets.json").write_text(json.dumps(dict(
        seed=SEED, source=str(SRC.relative_to(REPO)), ctx=str(CTX.relative_to(REPO)), panel=panel,
        timing=panel[:2], cells={"%s.%s" % c: v for c, v in cells}, eligibility=info), indent=1) + "\n")
    for i, k in enumerate(panel):
        print("%2d %-16s %-7s %s" % (i + 1, k, band(k), info[k]))


if __name__ == "__main__":
    sys.exit(main())
