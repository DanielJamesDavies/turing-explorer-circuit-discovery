"""Arena A (A9 pilot): the 3 smoke targets (2 tc, 1 z) and their context records, from the pilot's data pass.

Rule (fixed before any fit; seeded, deterministic, resumable):
  slots    tc deep (layers 8-11), tc shallow / mid (layers 2-6), z deep (layers 8-11): the deep slots give the
           timing pilot its worst case (most upstream sites: memory and s / step), the shallow tc slot the cheap end.
  frame    per slot, every latent of those (layer, kind) components with a full top store (64 of 64) and >= 48 of 64
           mid-band contexts (arena C's panel rule), sorted, then shuffled by random.Random(SEED) (one rng, slots in
           the order above).
  accept   the first candidate whose record (062 driver Runner.contexts = src build_record, group-aware split, the
           same records the driver would build) has a strongest pool, >= 8 held-out strongest and >= 32 contrast
           contexts (104's rule); at most MAX_TRY candidates per slot.
Records are written to OUT/ctx (the driver's CTX_DIR for the fits); every attempt is logged in OUT/select.jsonl;
the targets file lists the picks; OUT/targets.json records the rule and the picks.

  STORES=<data pass out> OUT=<records dir> TARGETS_OUT=<file> HF_HUB_OFFLINE=1 PYTHONPATH=src \
      ./.venv/bin/python experiments/107-true-baselines/arena_a/make_pilot_targets.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEED = 20261104
MAX_TRY = int(os.environ.get("MAX_TRY", 20))
MIN_HELD, MIN_NEG = 8, 32
MIN_TOP, MIN_MID = 64, 48
SLOTS = (("tc", tuple(range(8, 12))), ("tc", tuple(range(2, 7))), ("z", tuple(range(8, 12))))
if os.environ.get("SLOTS"):           # CPU tests only: e.g. "tc:1,z:1" (kind:layer[+layer...])
    SLOTS = tuple((s.split(":")[0], tuple(int(x) for x in s.split(":")[1].split("+")))
                  for s in os.environ["SLOTS"].split(",") if s)
STORES = Path(os.environ.get("STORES", str(HERE / "out" / "pilot" / "datapass")))
OUT = Path(os.environ.get("OUT", str(HERE / "out" / "pilot" / "rec")))
TARGETS_OUT = Path(os.environ.get("TARGETS_OUT", str(HERE / "out" / "pilot" / "targets_pilot.txt")))
os.environ.update(STORES=str(STORES), OUT=str(OUT), MODE="main")
os.environ.pop("CTX_DIR", None)
os.environ["WCM_VERSION"] = "v1"        # records only: v1 writes no OUT marker (records are version-free)
os.environ.setdefault("DATA", str(ROOT / "data_public" / "gpt2-small"))
os.environ.setdefault("SAE_PRELOAD", "0")
sys.path.insert(0, str(HERE))
import driver_tcz  # noqa: E402  (registers the family, imports 062's driver)
import torch  # noqa: E402

import wcm_tcz as WT  # noqa: E402

driver = driver_tcz.driver


def main():
    top = torch.load(STORES / "top_ctx.pt", weights_only=False)["ctx_seq_idx"].cpu()
    mid = torch.load(STORES / "mid_ctx.pt", weights_only=False)["ctx_seq_idx"].cpu()
    rng = random.Random(SEED)
    frames = []
    for kind, layers in SLOTS:
        keys = []
        for l in layers:
            c = l * len(WT.KINDS) + WT.KINDS.index(kind)
            ok = (((top[c] > 0).sum(-1) >= MIN_TOP) & ((mid[c] > 0).sum(-1) >= MIN_MID)).nonzero().flatten().tolist()
            keys += ["%d.%s.%d" % (l, kind, i) for i in ok]
        keys = sorted(keys)
        frames.append(rng.sample(keys, len(keys)))
    del top, mid
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
        flog.write(json.dumps(r) + "\n")
        flog.flush()
        seen[key] = r
        print("[%s] %-16s %s %s" % (time.strftime("%H:%M:%S"), key, "ok" if r["eligible"] else r.get("why"),
                                    {k: r.get(k) for k in ("n_top", "n_mid", "n_held", "n_neg")}), flush=True)
        return r

    picks, notes = [], []
    for (kind, layers), frame in zip(SLOTS, frames):
        got = None
        for key in frame[:MAX_TRY]:
            if key not in picks and attempt(key)["eligible"]:
                got = key
                break
        if got is None:
            notes.append("slot %s L%s: unfilled (%d candidates, %d tried)" % (kind, layers, len(frame),
                                                                            min(MAX_TRY, len(frame))))
        else:
            picks.append(got)
        print("slot %s L%d-%d: %d candidates -> %s" % (kind, layers[0], layers[-1], len(frame), got), flush=True)
    TARGETS_OUT.parent.mkdir(parents=True, exist_ok=True)
    TARGETS_OUT.write_text("\n".join(picks) + "\n")
    (OUT / "targets.json").write_text(json.dumps(dict(
        family=WT.FAMILY, seed=SEED, slots=[[k, list(ls)] for k, ls in SLOTS], max_try=MAX_TRY, min_top=MIN_TOP,
        min_mid=MIN_MID, min_held=MIN_HELD, min_neg=MIN_NEG, stores=str(STORES), picks=picks, notes=notes,
        n_candidates=[len(f) for f in frames]), indent=1) + "\n")
    print("targets -> %s: %s%s" % (TARGETS_OUT, picks, (" | " + "; ".join(notes)) if notes else ""), flush=True)
    if len(picks) < len(SLOTS):
        raise SystemExit("only %d of %d slots filled" % (len(picks), len(SLOTS)))


if __name__ == "__main__":
    main()
