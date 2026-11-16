"""115 DOCUMENT-DISJOINT HELD-OUT STRONGEST SETS for the production TuringLLM records (DAN-217 pilot). GPU (firing).

Per target, starting from the production record (T2 ctx_fresh/<key>.pt, read only), only the held-out STRONGEST set
is rebuilt; every other field (strong.train, D_train, mid pool and split, B_mid, neg, neg_ids) is the record's, byte
for byte, so the fixed production circuit is scored with no refit.

The rule is 086's DAN-157 held-out rule (circuit/heldout_rule.select_heldout through 086 rebuild.py's own helpers:
forbidden(), fire_fn_for(), stable(), Engine), plus ONE extra clause, the group clause:

  forbidden groups  the groups (document or topic, below) of every activating TRAINING row of the target:
                    strong.train (48, incl. the fit's D_train 32) and mid.train (incl. B_mid), each with all its corpus
                    twins (same 085 content hash), since one window text can sit in several files
  candidates        (1) the record's original held-out strongest rows, in pool order, at their stored anchors;
                    (2) the T1 scan's top-512 rows for the target (the production scan), strongest first by the scan
                    statistic (mean post-Top-K activation over 64 positions), rows of the record's own pool skipped,
                    anchors = recomputed argmax (086's order)
                    -> a candidate whose group (or any twin's group) is forbidden is dropped BEFORE the 086 rule
  then the 086 rule fire at the anchor (recomputed, protocol forward); not exact / anchor-prefix / near to the
                    forbidden rows F (strong.train + mid.train + all 64 contrast rows, 086 forbidden()) nor to an
                    accepted row; mid-band pool rows never enter; final in-batch firing check (stable / unstable_fire)
  fill              to the ORIGINAL held-out size (16 for every pilot record); floor 8 (below: flagged, still scored)
  stored as         086 does: accepted scan rows appended to strong.pos/tgt/arg/peak/ids; strong.held = kept original
                    indices (pool order) + appended indices; provenance in rec["docdisjoint"]

Groups (114 check.groups(): corpus id -> 085 corpus_src.npy -> data/text_paths.txt line):
  doc    the file path with the phi-3-mini / data-augmentation prefix stripped: a Phi-3 original + its T5 paraphrases
  topic  the dirname of that path: the (up to 6) sibling documents of one leaf topic (stricter)

  wsl: PYTHONPATH=src .venv/bin/python -X utf8 experiments/115-docdisjoint-rescore-pilot/build_heldout.py
  env: TARGETS (115/targets_pilot.txt)  GROUPS (doc,topic)  LIMIT
Writes out/ctx_<group>/<key>.pt (records for driver CTX_DIR) and out/build_<group>.jsonl (one summary row per target).
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiments"
ARCH = EXP / "109-pod-run-analysis" / "out" / "archives"
CTX_FRESH = ARCH / "pub-turingllm-T2-records" / "root/turing-tl/experiments/082-public-runs/out_turingllm/ctx_fresh"
SCAN = ARCH / "pub-turingllm-T1-scan" / "root/turing-tl/experiments/082-public-runs/out_turingllm/scan/scan_top512.pt"
CORPUS = EXP / "085-heldout-audit" / "out"
TARGETS = Path(os.environ.get("TARGETS", str(HERE / "targets_pilot.txt")))
GROUPS = [g for g in os.environ.get("GROUPS", "doc,topic").split(",") if g]
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
VERSION = "115-docdisjoint/2026-11-15 (086 DAN-157 rule + group clause vs activating training rows incl. twins)"

os.environ.setdefault("SCAN", str(SCAN))
os.environ.setdefault("CORPUS_DIR", str(CORPUS))
sys.path.insert(0, str(EXP / "086-heldout-rebuild"))
import rebuild as RB  # noqa: E402  (086: forbidden, fire_fn_for, stable, Engine)
import scan as SC  # noqa: E402     (086: load_scan)


def _load_114():
    spec = importlib.util.spec_from_file_location("check114", EXP / "114-paraphrase-check" / "check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(s):
    print("[%s 115 build] %s" % (time.strftime("%H:%M:%S"), s), flush=True)


class Groups:
    """Group label per corpus id, over all of an id's content twins."""

    def __init__(self):
        _, _, aug, doc, topic = _load_114().groups()
        self.src = np.load(CORPUS / "corpus_src.npy")
        self.hash = np.load(CORPUS / "corpus_hash.npy")
        hv = self.hash[1:]
        self.order = np.argsort(hv, kind="stable")
        self.sh = hv[self.order]
        self.lab = {"doc": doc, "topic": topic}
        self.aug = aug

    def twins(self, i):
        h = self.hash[int(i)]
        lo, hi = np.searchsorted(self.sh, h), np.searchsorted(self.sh, h, side="right")
        return self.order[lo:hi] + 1

    def of(self, i, g):
        return set(int(x) for x in self.lab[g][self.src[self.twins(i)]])

    def file_of(self, i):
        return int(self.src[int(i)])


def build_one(key, rec, scan_rows, E, G, group):
    from circuit import heldout_rule as HR
    s, m = rec["strong"], rec.get("mid")
    n0 = int(s["pos"].shape[0])
    n_want = len(s["held"])
    train_ids = [int(s["ids"][j]) for j in s["train"]] + ([int(m["ids"][j]) for j in m["train"]] if m is not None else [])
    forb = set()
    for i in train_ids:
        forb |= G.of(i, group)
    fb, finfo = RB.forbidden(rec)
    fire_cache, tok_cache = {}, {}
    fire = RB.fire_fn_for(E, key, fire_cache)
    pool_ids = set(int(x) for x in s["ids"])
    sc_ids = [int(x) for x in scan_rows["ids"].tolist()]

    def load_fn(cs):
        need = [c.id for c in cs if c.id not in tok_cache]
        if need:
            pos, tgt = E.load(need)
            for i, p, t in zip(need, pos, tgt):
                tok_cache[i] = (p.long(), t.long())
        for c in cs:
            c.tokens = tok_cache[c.id][0]

    # candidates in 086's order, with the group clause applied first (counted)
    counts = dict(orig_total=0, orig_allowed=0, scan_total=0, scan_allowed=0)
    base = []
    for j in sorted(int(x) for x in s["held"]):
        counts["orig_total"] += 1
        if G.of(s["ids"][j], group) & forb:
            continue
        counts["orig_allowed"] += 1
        base.append(HR.Candidate(tokens=s["pos"][j].long(), id=int(s["ids"][j]), origin="original",
                                 anchor=int(s["arg"][j]), info=dict(pool_index=int(j))))
    for r, i in enumerate(sc_ids):
        if i in pool_ids:
            continue
        counts["scan_total"] += 1
        if G.of(i, group) & forb:
            continue
        counts["scan_allowed"] += 1
        base.append(HR.Candidate(tokens=None, id=i, origin="scan",
                                 info=dict(scan_rank=r + 1, scan_mean=float(scan_rows["mean"][r]),
                                           scan_peak=float(scan_rows["peak"][r]), scan_arg=int(scan_rows["arg"][r]))))

    def fresh_cands(excl):
        out = []
        for c in base:
            if c.id in excl:
                continue
            out.append(HR.Candidate(tokens=c.tokens, id=c.id, origin=c.origin, anchor=c.anchor if c.origin == "original"
                                    else None, info=dict(c.info)))
        return out

    def final_silent(cands):
        if not cands:
            return set()
        A = E.acts(key, torch.stack([c.tokens for c in cands]))
        return {c.id for c, a in zip(cands, A) if not float(a[int(c.anchor)]) > 0}

    def strong_order(cs, acc):
        orig = {cs[i].info["pool_index"]: i for i in acc if cs[i].origin == "original"}
        return [orig[int(j)] for j in s["held"] if int(j) in orig] + [i for i in acc if cs[i].origin == "scan"]

    def sel(excl):
        cs_ = fresh_cands(excl)
        a_, e_ = HR.select_heldout(cs_, fb, fire, n_held=n_want, exclude_mid=True, load_fn=load_fn)
        return cs_, a_, e_

    cs, acc, ex, unstable = RB.stable(sel, strong_order, final_silent, "unstable_fire")
    kept = {cs[i].info["pool_index"] for i in acc if cs[i].origin == "original"}
    new = [cs[i] for i in acc if cs[i].origin == "scan"]
    s2 = dict(s)
    fr_new = [ex[cs.index(c)]["fire"] for c in new]
    if new:
        s2["pos"] = torch.cat([s["pos"], torch.stack([tok_cache[c.id][0] for c in new]).to(s["pos"].dtype)])
        s2["tgt"] = torch.cat([s["tgt"], torch.stack([tok_cache[c.id][1] for c in new]).to(s["tgt"].dtype)])
        s2["arg"] = torch.cat([s["arg"], torch.tensor([f["anchor"] for f in fr_new], dtype=s["arg"].dtype)])
        s2["peak"] = torch.cat([s["peak"], torch.tensor([f["peak"] for f in fr_new], dtype=s["peak"].dtype)])
        s2["ids"] = list(s["ids"]) + [c.id for c in new]
    s2["held"] = [int(j) for j in s["held"] if int(j) in kept] + list(range(n0, n0 + len(new)))
    # independent re-check of the group clause on the final set
    leak = [int(s2["ids"][j]) for j in s2["held"] if G.of(s2["ids"][j], group) & forb]
    assert not leak, "group clause violated: %s" % leak
    # reported, not a rule: held rows sharing a group with the contrast-TRAINING rows (rec["neg_ids"][:48], the
    # scorers' contrast-train slice; RB.neg_split)
    neg_ids = rec.get("neg_ids")
    if neg_ids is not None:
        cforb = set()
        for i in neg_ids[:RB.neg_split(rec)]:
            cforb |= G.of(i, group)
        share_c_new = sum(bool(G.of(s2["ids"][j], group) & cforb) for j in s2["held"])
        share_c_old = sum(bool(G.of(s["ids"][j], group) & cforb) for j in s["held"])
    else:
        share_c_new = share_c_old = None
    r2 = dict(rec)
    r2["strong"] = s2
    held_ids = [int(s2["ids"][j]) for j in s2["held"]]
    old_ids = [int(s["ids"][j]) for j in s["held"]]
    old_groups = [bool(G.of(i, group) & forb) for i in old_ids]
    pk_tr = s["peak"][s["train"]].float()
    pk_new = s2["peak"][s2["held"]].float()
    pk_old = s["peak"][s["held"]].float()
    prov = dict(version=VERSION, rule=HR.RULE_VERSION, group=group, n_forbidden_groups=len(forb),
                n_train_rows=len(train_ids), n_want=n_want, n_held=len(s2["held"]), below_floor=len(s2["held"]) < HR.FLOOR,
                n_orig_kept=len(kept), n_scan=len(new), counts=counts, reasons=HR.reason_counts(ex),
                unstable_fire=unstable, scan_rank=[c.info["scan_rank"] for c in new], held_ids=held_ids,
                old_held_ids=old_ids, old_in_forbidden_group=old_groups, forbidden=finfo)
    r2["docdisjoint"] = prov
    med = lambda x: float(np.median(x)) if len(x) else None
    l = int(key.split(".")[0])
    row = dict(seed=key, group=group, layer=l, kind=key.split(".")[1], n_want=n_want, n_held=len(s2["held"]),
               below_floor=prov["below_floor"], n_old_leaky=int(sum(old_groups)), n_orig_kept=len(kept),
               new_share_contrast=share_c_new, old_share_contrast=share_c_old,
               overlap_old=len(set(held_ids) & set(old_ids)), n_scan=len(new),
               cand_total=counts["orig_total"] + counts["scan_total"],
               cand_allowed=counts["orig_allowed"] + counts["scan_allowed"], **counts,
               n_examined=len(ex), reasons=prov["reasons"], unstable=len(unstable), n_forbidden_groups=len(forb),
               peak_train_med=med(pk_tr.numpy()), peak_old_med=med(pk_old.numpy()), peak_new_med=med(pk_new.numpy()),
               scan_rank_med=med(prov["scan_rank"]), scan_rank_max=max(prov["scan_rank"], default=None))
    return r2, row


def main():
    keys = [k for k in TARGETS.read_text().split() if k][:LIMIT] if LIMIT else [k for k in TARGETS.read_text().split() if k]
    G = Groups()
    log("groups ready")
    sc = SC.load_scan(SCAN)
    log("scan loaded (%d targets)" % len(sc["where"]))
    E = RB.Engine()
    log("engine ready")
    for group in GROUPS:
        out = HERE / "out" / ("ctx_%s" % group)
        out.mkdir(parents=True, exist_ok=True)
        with open(HERE / "out" / ("build_%s.jsonl" % group), "w") as fh:
            for n_, key in enumerate(keys):
                t = time.time()
                rec = torch.load(CTX_FRESH / ("%s.pt" % key), weights_only=False)[key]
                r2, row = build_one(key, rec, sc["rows"](key), E, G, group)
                RB.save_atomic({key: r2}, out / ("%s.pt" % key))
                row["t"] = round(time.time() - t, 1)
                fh.write(json.dumps(row) + "\n"); fh.flush()
                log("%s %2d/%d %-16s held %d/%d (kept %d orig, %d scan; old leaky %d) cands %d/%d %.1fs" % (
                    group, n_ + 1, len(keys), key, row["n_held"], row["n_want"], row["n_orig_kept"], row["n_scan"],
                    row["n_old_leaky"], row["cand_allowed"], row["cand_total"], row["t"]))
    log("DONE")


if __name__ == "__main__":
    main()
