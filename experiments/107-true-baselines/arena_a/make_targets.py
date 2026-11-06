"""Arena A (A6, DAN-197): the 48 production targets and their prompts, from a data pass of the gpt2-small-tcz family.

Rule (fixed before any fit; seeded, deterministic, resumable) [ours, README section 6]:
  slots    24 slots = 12 layers x {tc, z}, in the order L0 tc, L0 z, L1 tc, ... L11 z; N_PER_SLOT = 2 targets each.
           A z slot's frame holds the (L, z) and, at L5 / L7, the (L, z2) latents (z2 = the upper half of Kissane's
           49,152-wide dictionary; wcm_tcz.orig_latent maps it back).
  frame    per slot, every latent with a full top store (MIN_TOP = 64 of 64) and >= MIN_MID = 48 of 64 mid-band contexts
           (the pilot's / arena C's panel rule), sorted, then shuffled by ONE random.Random(SEED) (slots in the order
           above, so a slot's frame does not depend on later slots).
  accept   in frame order, a candidate whose record (062 driver Runner.contexts = src build_record, group-aware split,
           exactly the record the WCM fit and the scorer use) has a strongest pool, >= 8 held-out strongest, >= 32
           contrast contexts (104's rule, as the pilot) and >= K strong training contexts; at most MAX_TRY per slot.
  prompts  per target, K = 8 activating prompts [ours]: the K highest-peak contexts of the strong part of the WCM
           training arm (arm B: D_train; arm A: strong train), ties by index, each truncated after its anchor
           (max-activating) position. Token ids are the corpus tokens WITHOUT BOS; the baselines prepend BOS (Ge A14,
           Dunefsky A15) and root at anchor + 1, the WCM site model (ARENA_A_BOS=1) does the same internally.
  held-out the record's own sets as the protocol builds them (held strongest, held mid-band, contrast held), listed
           by corpus sequence id and anchor via the 062 driver's manifest().
  exclude  EXCLUDE (optional, 2026-11-05, A13): a file of L.kind.i keys (whitespace-separated; the WCM tuning dev set
           + the pilot targets, tune_exclude.txt) that are never attempted. They are skipped in the already-shuffled
           frame and do not count as a try, so the shuffle and every pick that does not hit an excluded key are
           unchanged; unset = exactly the old rule and outputs (no new field in targets.json / MANIFEST.json).
Outputs in OUT: ctx/<key>.pt (the records: the driver's CTX_DIR for the fits), select.jsonl (every attempt),
targets.txt, prompts.jsonl, targets.json (rule, candidates per slot, picks), MANIFEST.json (md5 of every output that
must reproduce, plus the input stores' sizes and the data-pass manifest md5).

  STORES=<data pass out> OUT=<dir> DATA=<corpus> HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES= \
      ./.venv/bin/python -I -B experiments/107-true-baselines/arena_a/make_targets.py
"""
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(ROOT / "src"), str(HERE)):           # -I ignores PYTHONPATH
    if _p not in sys.path:
        sys.path.insert(0, _p)
SEED = int(os.environ.get("SEED", 20261105))
MAX_TRY = int(os.environ.get("MAX_TRY", 40))
N_PER_SLOT = int(os.environ.get("N_PER_SLOT", 2))
K = int(os.environ.get("K", 8))
MIN_HELD, MIN_NEG = 8, 32
MIN_TOP, MIN_MID = int(os.environ.get("MIN_TOP", 64)), int(os.environ.get("MIN_MID", 48))
LAYERS = tuple(int(x) for x in os.environ.get("LAYERS", ",".join(map(str, range(12)))).split(","))
SLOTS = tuple((l, kind) for l in LAYERS for kind in ("tc", "z"))
STORES = Path(os.environ.get("STORES", str(HERE / "out" / "pilot" / "datapass")))
OUT = Path(os.environ.get("OUT", str(HERE / "out" / "targets_smoke")))
EXCLUDE_FILE = os.environ.get("EXCLUDE") or None
EXCLUDE = frozenset(Path(EXCLUDE_FILE).read_text().split()) if EXCLUDE_FILE else frozenset()
os.environ.update(STORES=str(STORES), OUT=str(OUT), MODE="main")
os.environ.pop("CTX_DIR", None)
os.environ["WCM_VERSION"] = "v1"        # records only: v1 writes no OUT marker (records are version-free)
os.environ.setdefault("DATA", str(ROOT / "data_public" / "gpt2-small"))
os.environ.setdefault("SAE_PRELOAD", "0")
import driver_tcz  # noqa: E402  (registers the family, imports 062's driver)
import torch  # noqa: E402

import wcm_tcz as WT  # noqa: E402

driver = driver_tcz.driver


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def slot_kinds(kind, layer):
    return [k for k in WT.KINDS if (k == kind) or (kind == "z" and k == "z2" and layer in WT.WIDE)]


def prompts_for(key, rec):
    from circuit.protocol_contexts import training_arm
    arm, _thin = training_arm(rec)
    s = rec["strong"]
    idx = list(rec["D_train"]) if arm == "B" else list(s["train"])
    order = sorted(idx, key=lambda j: (-float(s["peak"][j]), j))[:K]
    out = []
    for j in order:
        a = int(s["arg"][j])
        out.append(dict(pool="strong", j=int(j), seq_id=None if s.get("ids") is None else int(s["ids"][j]),
                        anchor=a, peak=float(s["peak"][j]), tokens=[int(t) for t in s["pos"][j][: a + 1]]))
    return arm, out


def main():
    t_start = time.time()
    top = torch.load(STORES / "top_ctx.pt", weights_only=False, map_location="cpu")["ctx_seq_idx"]
    mid = torch.load(STORES / "mid_ctx.pt", weights_only=False, map_location="cpu")["ctx_seq_idx"]
    rng = random.Random(SEED)
    frames, n_cand = [], []
    for l, kind in SLOTS:
        keys, per_kind = [], {}
        for k in slot_kinds(kind, l):
            c = l * len(WT.KINDS) + WT.KINDS.index(k)
            ok = (((top[c] > 0).sum(-1) >= MIN_TOP) & ((mid[c] > 0).sum(-1) >= MIN_MID)).nonzero().flatten().tolist()
            per_kind[k] = len(ok)
            keys += ["%d.%s.%d" % (l, k, i) for i in ok]
        keys = sorted(keys)
        frames.append(rng.sample(keys, len(keys)))
        n_cand.append(dict(layer=l, kind=kind, n=len(keys), per_kind=per_kind))
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
            n_k = 0 if s is None else len(prompts_for(key, rec)[1])
            r["n_prompts"] = n_k
            r["eligible"] = bool(s is not None and r["n_held"] >= MIN_HELD and (r["n_neg"] or 0) >= MIN_NEG
                                 and n_k >= K)
            if not r["eligible"]:
                r["why"] = ("no_strong_pool" if s is None else "held<%d" % MIN_HELD if r["n_held"] < MIN_HELD
                            else "neg<%d" % MIN_NEG if (r["n_neg"] or 0) < MIN_NEG else "prompts<%d" % K)
        except Exception as e:  # noqa: BLE001
            r.update(eligible=False, why="error", error="%s: %s" % (type(e).__name__, str(e)[:300]))
        r["t"] = round(time.time() - t, 1)
        flog.write(json.dumps(r) + "\n")
        flog.flush()
        seen[key] = r
        print("[%s] %-16s %s %s" % (time.strftime("%H:%M:%S"), key, "ok" if r["eligible"] else r.get("why"),
                                    {k: r.get(k) for k in ("n_top", "n_mid", "n_held", "n_neg", "n_prompts")}),
              flush=True)
        return r

    picks, slot_of, notes, skipped = [], {}, [], []
    for (l, kind), frame, nc in zip(SLOTS, frames, n_cand):
        got, tried = [], 0
        for key in frame:
            if len(got) >= N_PER_SLOT or tried >= MAX_TRY:
                break
            if key in EXCLUDE:                      # never attempted, not a try (see the docstring)
                skipped.append(key)
                continue
            tried += 1
            if attempt(key)["eligible"]:
                got.append(key)
        nc.update(tried=tried, picked=got)
        if len(got) < N_PER_SLOT:
            notes.append("slot L%d %s: %d of %d filled (%d candidates, %d tried)" % (l, kind, len(got), N_PER_SLOT,
                                                                                   len(frame), tried))
        for k in got:
            slot_of[k] = (l, kind)
        picks += got
        print("slot L%d %-2s: %5d candidates %s, tried %d -> %s" % (l, kind, len(frame), nc["per_kind"], tried, got),
              flush=True)
    rows = []
    for key in picks:
        rec = R.contexts(key)
        l, k, i = key.split(".")
        arm, prompts = prompts_for(key, rec)
        dk, dl = WT.orig_latent(k, int(i))
        rows.append(dict(key=key, slot=list(slot_of[key]), layer=int(l), kind=k, latent=int(i), dict_kind=dk,
                         dict_latent=dl, arm=arm, prompts=prompts, heldout=driver.manifest(key, rec, arm)))
    (OUT / "targets.txt").write_text("".join(k + "\n" for k in picks))
    with open(OUT / "prompts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    tj = dict(family=WT.FAMILY, seed=SEED, slots=[list(s) for s in SLOTS], n_per_slot=N_PER_SLOT, k=K, max_try=MAX_TRY,
              min_top=MIN_TOP, min_mid=MIN_MID, min_held=MIN_HELD, min_neg=MIN_NEG, stores=str(STORES),
              candidates=n_cand, picks=picks, notes=notes)
    if EXCLUDE_FILE:                                # only when set: unset leaves targets.json byte-identical
        tj.update(exclude_file=str(EXCLUDE_FILE), exclude=sorted(EXCLUDE), excluded_skipped=skipped)
        assert not EXCLUDE & set(picks)
    (OUT / "targets.json").write_text(json.dumps(tj, indent=1, sort_keys=True) + "\n")
    files = ["targets.txt", "prompts.jsonl", "targets.json"] + ["ctx/%s.pt" % k for k in picks]
    man = dict(schema="arena_a_targets_v1", seed=SEED, files={f: md5(OUT / f) for f in files},
               inputs=dict(stores=str(STORES),
                           store_sizes={p.name: p.stat().st_size for p in sorted(STORES.glob("*.pt"))},
                           data_pass_manifest_md5=(md5(STORES / "data_pass_manifest.json")
                                                   if (STORES / "data_pass_manifest.json").exists() else None),
                           data=os.environ["DATA"]),
               n_targets=len(picks), seconds=round(time.time() - t_start, 1))
    if EXCLUDE_FILE:
        man["exclude"] = dict(file=str(EXCLUDE_FILE), md5=md5(Path(EXCLUDE_FILE)), n=len(EXCLUDE),
                              skipped_in_frames=skipped)
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print("targets -> %s: %d picks%s" % (OUT, len(picks), (" | " + "; ".join(notes)) if notes else ""), flush=True)
    if len(picks) < N_PER_SLOT * len(SLOTS):
        raise SystemExit("only %d of %d targets" % (len(picks), N_PER_SLOT * len(SLOTS)))


if __name__ == "__main__":
    main()
