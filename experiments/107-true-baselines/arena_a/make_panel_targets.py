"""Arena A (2026-11-05 night): the 12-target v4 panel, 2 tc + 2 z in each layer band 1-3, 4-7, 8-11, from the pilot's
data pass. make_pilot_targets.py's rule (same frame filter: full top store, >= 48 of 64 mid; same acceptance: a strongest
pool, >= 8 held-out strongest, >= 32 contrast; records built by 062's Runner.contexts), with
  slots    (tc, 1-3) x2, (z, 1-3) x2, (tc, 4-7) x2, (z, 4-7) x2, (tc, 8-11) x2, (z, 8-11) x2, in that order
  seed     PANEL_SEED (default 20261105), one rng, one shuffled frame per (kind, band) shared by its two slots
  exclude  the pilot's targets (EXCLUDE file, default out/pilot/targets_pilot.txt) and earlier picks
Fixed before any panel fit. Records -> OUT/ctx, attempts -> OUT/select.jsonl, picks -> TARGETS_OUT and OUT/targets.json.

  STORES=<pilot datapass> OUT=<panel>/rec TARGETS_OUT=<panel>/targets_panel.txt HF_HUB_OFFLINE=1 PYTHONPATH=src \
      ./.venv/bin/python experiments/107-true-baselines/arena_a/make_panel_targets.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("STORES", str(HERE / "out" / "pilot" / "datapass"))
os.environ.setdefault("OUT", str(HERE / "out" / "panel" / "rec"))
os.environ.setdefault("TARGETS_OUT", str(HERE / "out" / "panel" / "targets_panel.txt"))
import make_pilot_targets as M  # noqa: E402  (sets the env, registers the family, imports 062's driver)
import torch  # noqa: E402

SEED = int(os.environ.get("PANEL_SEED", 20261105))
PER_BAND = 2
BANDS = ((1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))
KINDS = ("tc", "z")
EXCLUDE = Path(os.environ.get("EXCLUDE", str(HERE / "out" / "pilot" / "targets_pilot.txt")))


def main():
    top = torch.load(M.STORES / "top_ctx.pt", weights_only=False)["ctx_seq_idx"].cpu()
    mid = torch.load(M.STORES / "mid_ctx.pt", weights_only=False)["ctx_seq_idx"].cpu()
    excl = set(EXCLUDE.read_text().split()) if EXCLUDE.exists() else set()
    rng = random.Random(SEED)
    groups = []
    for layers in BANDS:
        for kind in KINDS:
            keys = []
            for l in layers:
                c = l * len(M.WT.KINDS) + M.WT.KINDS.index(kind)
                ok = (((top[c] > 0).sum(-1) >= M.MIN_TOP) & ((mid[c] > 0).sum(-1) >= M.MIN_MID)).nonzero()
                keys += ["%d.%s.%d" % (l, kind, i) for i in ok.flatten().tolist()]
            keys = sorted(k for k in keys if k not in excl)
            groups.append((kind, layers, rng.sample(keys, len(keys))))
    del top, mid
    R = M.driver.Runner()
    M.OUT.mkdir(parents=True, exist_ok=True)
    log_path = M.OUT / "select.jsonl"
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
            r["eligible"] = bool(s is not None and r["n_held"] >= M.MIN_HELD and (r["n_neg"] or 0) >= M.MIN_NEG)
            if not r["eligible"]:
                r["why"] = ("no_strong_pool" if s is None else "held<%d" % M.MIN_HELD if r["n_held"] < M.MIN_HELD
                            else "neg<%d" % M.MIN_NEG)
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
    for kind, layers, frame in groups:
        got, tried = [], 0
        for key in frame:
            if len(got) >= PER_BAND or tried >= M.MAX_TRY * PER_BAND:
                break
            if key in picks:
                continue
            tried += 1
            if attempt(key)["eligible"]:
                got.append(key)
                picks.append(key)
        if len(got) < PER_BAND:
            notes.append("%s L%d-%d: %d of %d filled (%d candidates, %d tried)" % (
                kind, layers[0], layers[-1], len(got), PER_BAND, len(frame), tried))
        print("band %s L%d-%d: %d candidates -> %s" % (kind, layers[0], layers[-1], len(frame), got), flush=True)
    M.TARGETS_OUT.parent.mkdir(parents=True, exist_ok=True)
    M.TARGETS_OUT.write_text("\n".join(picks) + "\n")
    (M.OUT / "targets.json").write_text(json.dumps(dict(
        family=M.WT.FAMILY, seed=SEED, bands=[list(b) for b in BANDS], kinds=list(KINDS), per_band=PER_BAND,
        exclude=sorted(excl), max_try=M.MAX_TRY, min_top=M.MIN_TOP, min_mid=M.MIN_MID, min_held=M.MIN_HELD,
        min_neg=M.MIN_NEG, stores=str(M.STORES), picks=picks, notes=notes,
        n_candidates=[len(f) for _, _, f in groups]), indent=1) + "\n")
    print("targets -> %s: %s%s" % (M.TARGETS_OUT, picks, (" | " + "; ".join(notes)) if notes else ""), flush=True)
    if len(picks) < PER_BAND * len(groups):
        raise SystemExit("only %d of %d slots filled" % (len(picks), PER_BAND * len(groups)))


if __name__ == "__main__":
    main()
