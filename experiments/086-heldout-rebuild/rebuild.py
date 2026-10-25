"""HELD-OUT REBUILD (DAN-157): new held-out sets for the 15k run's FIXED circuits, under the approved rule.

Reads each 062 record (062/out_full/out/ctx/<key>.pt, never written) and the merged top-N scan (scan.py), and writes
one new record with the same schema to ctx_v2/<key>.pt (the only variant: the source-file rule was dropped completely,
Daniel 2026-10-24 final). Training-side fields are untouched (no refits): strong.train, D_train, mid.train, B_mid, neg,
neg_ids, n_top, n_mid and the first n_pool_orig rows of every strong.* tensor / list are the original's, byte for byte.

Held-out STRONGEST (the primary scored set), circuit/heldout_rule.select_heldout per target:
  forbidden F  strongest-train (48: the A-mean / pin reference, incl. the fit's 32), mid-band train (48), all 64
               contrast rows; plus their corpus twins (same corpus id or content hash)
  candidates   (1) the original held-out rows, in pool order (= the top store's rank by mean score), at their stored
               anchors; (2) the scan's rows for the target, strongest first by the scan statistic = the top store's
               ranking statistic (mean post-Top-K activation over the 64 positions), anchors = recomputed argmax.
               Mid-band pool rows (by id or content) never enter the strongest set (Daniel, 2026-10-24)
  each must    fire (post-Top-K activation > 0 at its anchor, recomputed here by circuit/protocol_contexts.
               target_activations: the protocol's own forward, batch 16), and not be an exact / anchor-prefix / near
               duplicate of F nor of an already-accepted held-out row.
               Firing is checked twice: in the selection forward, and on the FINAL set forwarded as the scorer sees it
               (one batch, record order). A row near the Top-K cut that is silent there is excluded (unstable_fire)
               and the set refilled
  fill         up to 16; floor 8 (below: flagged below_floor, reported per band x kind; scored and flagged, never
               dropped)
  stored as    accepted scan rows APPENDED to strong.pos / tgt / arg / peak / ids; strong.held = the kept original
               indices (original order) + the appended indices (acceptance order). heldout_v2 = provenance per row
               (origin, corpus id, activation, anchor, ranks), rule version, counts, reasons
Held-out MID-BAND (reported): the same rule on the original mid held rows (pool order); rows are only removed.

Held-out CONTRAST (the scorer's contrast held-out slice: rec["neg"][n_ntr:], n_ntr = split_n(64) = 48, as
eval/wcm_eval.score_circuit and 049 amp_eval_pass_v2 slice it), select_heldout(..., want="silent"):
  forbidden    the contrast-train rows rec["neg"][:48] and EVERY activating row of the rebuilt record (strongest pool incl.
               the appended scan rows, mid-band pool), by tokens, id and content hash
  candidates   (1) the original contrast held-out rows rec["neg"][48:], in record order; (2) refill: the close selector's
               next candidates (utils/neg_context_selector, the 15k run's selector and arguments, max_sequences = 64 +
               NEG_EXTRA; its first 64 are the run's contrast rows, checked as selector_prefix_ok), similarity order,
               minus the record's own contrast ids / contents. Fetched only when the originals leave the set short
  each must    be SILENT (the target's post-Top-K activation is 0 at every position, recomputed with the protocol
               forward), and not be exact / near (apfx: anchor = last position, so apfx == exact; a contrast row has
               no anchor) to the forbidden rows nor to an already-accepted contrast held-out row. The final set is
               forwarded once more as one batch; a row that fires there is excluded (unstable_silence) and refilled
  fill         up to 16; fewer are kept when the candidates run out (contrast_below_floor: < 8; reported)
  stored as    rec["neg_held_v2"] [h, 64] + rec["neg_held_v2_ids"]; rec["neg"] is NOT changed, so the fit (training
               probe: all 64 of rec["neg"], 48 train + 16 engine-holdout used only for its holdout-loss statistic) and
               every reader of rec["neg"] see the run's rows. Scoring: protocol_contexts.eval_probe uses
               rec["neg"][:48] + neg_held_v2 with metadata n_neg_train = 48 when the field is present (eval_contrast)

  MODE=build  PYTHONPATH=src python experiments/086-heldout-rebuild/rebuild.py     (GPU; resumable; SHARD=i/k)
  MODE=verify PYTHONPATH=src python experiments/086-heldout-rebuild/rebuild.py     (independent re-check, GPU fire)
  MODE=report PYTHONPATH=src python experiments/086-heldout-rebuild/rebuild.py     (no GPU) -> OUT_RB/report.md
  env: TARGETS (062 targets_full.txt)  SHARD (0/1)  LIMIT  CTX_IN (062/out_full/out/ctx)  CTX_OUT (062/out_full/out/ctx_v2)
       SCAN (086/out/scan/scan_top512.pt)  CORPUS_DIR (085/out: corpus_hash.npy)  OUT_RB (086/out/rebuild)
       FIRE=0 (verify: skip GPU)  STORES_DIR (outputs: top_ctx / mid_ctx / neg_ctx / seq_repr .pt, the contrast refill's
       selector)  NEG_EXTRA (128: refill candidates beyond the run's 64)
"""
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
E062 = REPO / "experiments" / "062-h100-protocol-v1"
MODE = os.environ.get("MODE", "build")
SHARD_I, SHARD_K = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
TARGETS = Path(os.environ.get("TARGETS", str(E062 / "targets_full.txt")))
CTX_IN = Path(os.environ.get("CTX_IN", str(E062 / "out_full" / "out" / "ctx")))
CTX_OUT = Path(os.environ.get("CTX_OUT", str(E062 / "out_full" / "out" / "ctx_v2")))
SCAN = Path(os.environ.get("SCAN", str(HERE / "out" / "scan" / "scan_top512.pt")))
CORPUS_DIR = Path(os.environ.get("CORPUS_DIR", str(REPO / "experiments" / "085-heldout-audit" / "out")))
OUT_RB = Path(os.environ.get("OUT_RB", str(HERE / "out" / "rebuild")))
FIRE = os.environ.get("FIRE", "1") == "1"
STORES_DIR = Path(os.environ.get("STORES_DIR", str(REPO / "outputs")))
NEG_EXTRA = int(os.environ.get("NEG_EXTRA", "128"))
MAX_ROUNDS = 5           # re-selections while the final set has rows that flip in their final batch (near the Top-K cut)
NEG_HOLDOUT_FRAC = 0.25  # the scorers' contrast split (split_n): rec["neg"][:48] train, [48:] held out
TAG = "shard%d" % SHARD_I


def log(s):
    print("[%s rebuild %s] %s" % (time.strftime("%H:%M:%S"), TAG, s), flush=True)


def band(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def load_rec(d, key):
    return torch.load(Path(d) / ("%s.pt" % key), weights_only=False)[key]


def save_atomic(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp-%d" % os.getpid())
    torch.save(obj, tmp)
    os.replace(tmp, path)


def neg_split(rec):
    """The scorers' contrast-train count for the run's record (split_n of its contrast rows: 48 of 64)."""
    n = int(rec["neg"].shape[0])
    return max(1, n - int(round(n * NEG_HOLDOUT_FRAC)))


class Corpus:
    """085 corpus_index's content hash per corpus id (twins = same hash; verify: an id must hash to its tokens)."""

    def __init__(self, d=CORPUS_DIR):
        self.hash = np.load(d / "corpus_hash.npy", mmap_mode="r")


def forbidden(rec):
    """The training-side rows of one record (heldout_rule.Forbidden) plus the counts for the provenance."""
    from circuit import heldout_rule as HR
    s, m = rec["strong"], rec.get("mid")
    rows = [s["pos"][s["train"]].long()]
    ids = [int(s["ids"][j]) for j in s["train"]]
    if m is not None:
        rows.append(m["pos"][m["train"]].long()); ids += [int(m["ids"][j]) for j in m["train"]]
    rows.append(rec["neg"].long())
    neg_ids = rec.get("neg_ids")
    ids += [int(x) for x in neg_ids] if neg_ids is not None else []
    F = torch.cat(rows, 0)
    hashes = set(int(x) for x in HR.window_hash(F))
    mid_ids = set(int(x) for x in m["ids"]) if m is not None else set()
    mid_h = set(int(x) for x in HR.window_hash(m["pos"])) if m is not None else set()
    fb = HR.Forbidden(tokens=F, ids=set(ids), hashes=hashes, mid_ids=mid_ids, mid_hashes=mid_h)
    info = dict(n_rows=int(F.shape[0]), n_ids=len(set(ids)), contrast_ids=neg_ids is not None,
                roles="strong.train + mid.train + neg (all)")
    return fb, info


def contrast_forbidden(rec, s2, m):
    """The contrast held-out set's forbidden rows: contrast-train + every activating row (rebuilt strongest pool incl.
    appended rows, mid-band pool)."""
    from circuit import heldout_rule as HR
    n_ntr = neg_split(rec)
    neg_ids = rec.get("neg_ids")
    rows = [rec["neg"][:n_ntr].long(), s2["pos"].long()] + ([m["pos"].long()] if m is not None else [])
    ids = ([int(x) for x in neg_ids[:n_ntr]] if neg_ids is not None else []) + [int(x) for x in s2["ids"]]
    ids += [int(x) for x in m["ids"]] if m is not None else []
    F = torch.cat(rows, 0)
    return HR.Forbidden(tokens=F, ids={i for i in ids if i > 0}, hashes=set(int(x) for x in HR.window_hash(F)))


class Engine:
    """GPU side: the model, the SAE bank and the loader, as build_record uses them; the close selector on demand."""

    def __init__(self):
        from data.loader import DataLoader
        from model.inference import Inference
        from sae.bank import SAEBank
        self.dev = torch.device("cuda")
        self.loader = DataLoader(device=self.dev)
        self.inference = Inference(device=self.dev)
        self.bank = SAEBank(device=self.dev, load_decoders=False)
        self.kinds = list(self.bank.kinds)
        self._sel = None

    def load(self, ids):
        """(pos, tgt) for ids, row-aligned (protocol_contexts._load: the loader path build_record used)."""
        from circuit.protocol_contexts import _load
        pos, tgt, got = _load(self.loader, [int(i) for i in ids])
        if got != [int(i) for i in ids]:
            raise RuntimeError("loader returned %d of %d ids" % (len(got), len(ids)))
        return pos.cpu(), tgt.cpu()

    def acts(self, key, tokens):
        from circuit.protocol_contexts import target_activations
        l, kd, i = key.split(".")
        return target_activations(self.inference, self.bank, int(l), self.kinds.index(kd), kd, int(i),
                                  tokens.to(self.dev))

    def selector(self):
        """The 15k run's close-contrast selector (protocol_contexts.make_selector) over the stores it ranks with:
        top_ctx / mid_ctx (excluded positives, the similarity reference), neg_ctx (the candidate ids), seq_repr."""
        if self._sel is None:
            import store.seq_repr as SR
            from circuit.protocol_contexts import make_selector
            from pipeline.discovery_artifacts import _seq_repr_from_payload
            from store.context import mid_ctx, neg_ctx, top_ctx
            t0 = time.time()
            for st, name in ((top_ctx, "top_ctx"), (mid_ctx, "mid_ctx"), (neg_ctx, "neg_ctx")):
                if not getattr(st, "_allocated", False):
                    st.load(str(STORES_DIR / ("%s.pt" % name)))
            if SR.seq_repr is None:
                SR.seq_repr = _seq_repr_from_payload(torch.load(STORES_DIR / "seq_repr.pt", map_location="cpu",
                                                                weights_only=False))
            self._sel = make_selector(self.inference, self.bank, self.loader)
            log("close selector ready (stores %s, %.0fs)" % (STORES_DIR, time.time() - t0))
        return self._sel

    def contrast_pool(self, key, n_max):
        """(ids, tokens [n, 64] cpu) of the close selector's first n_max verified-silent candidates (the run's
        selector arguments, protocol_contexts._SELECT_KW), similarity order."""
        from circuit.protocol_contexts import _SELECT_KW
        from pipeline.component_index import component_idx
        l, kd, i = key.split(".")
        cs = self.selector().select(component_idx(int(l), self.kinds.index(kd), len(self.kinds)), int(i), "close",
                                    max_sequences=int(n_max), **_SELECT_KW)
        if cs is None:
            return [], torch.zeros(0, 64, dtype=torch.long)
        return [int(x) for x in cs.sequence_ids], cs.tokens.cpu().long()


def fire_fn_for(E, key, cache):
    """heldout_rule fire_fn: one forward per chunk; act at the candidate's anchor (argmax when it has none). Cached by
    (content hash, anchor): ids may be unknown (-1) for contrast rows."""
    def fire(cands):
        from circuit.heldout_rule import window_hash
        ck = lambda c: (int(window_hash(c.tokens)[0]) if c.hash is None else int(c.hash), c.anchor)
        todo = [c for c in cands if ck(c) not in cache]
        if todo:
            A = E.acts(key, torch.stack([c.tokens for c in todo]))
            for c, a in zip(todo, A):
                am = int(a.argmax())
                anc = am if c.anchor is None else int(c.anchor)
                cache[ck(c)] = dict(act_at_anchor=float(a[anc]), anchor=anc, peak=float(a.max()),
                                    mean=float(a.mean()), argmax=am)
        return [cache[ck(c)] for c in cands]
    return fire


def stable(select, record_order, bad_of, reason):
    """select(excluded ids) -> (cands, acc, ex) until the final set passes bad_of (the rows that flip when the final
    set is forwarded as the scorer sees it). Rows still flipping after MAX_ROUNDS are dropped (no refill)."""
    srt = lambda x: sorted(x, key=lambda v: (isinstance(v, str), str(v).zfill(24)))   # ids, then "h<hash>" keys
    unstable = set()
    for _ in range(MAX_ROUNDS):
        cs, acc, ex = select(unstable)
        bad = bad_of([cs[i] for i in record_order(cs, acc)])
        if not bad:
            return cs, acc, ex, srt(unstable)
        unstable |= bad
    hit = lambda c: c.id in bad or ("h%d" % c.hash) in bad
    for i in acc:
        if hit(cs[i]):
            ex[i]["reason"] = reason
    acc = [i for i in acc if not hit(cs[i])]
    return cs, acc, ex, srt(unstable | bad)


def rebuild_contrast(key, rec, s2, m, E):
    """(neg_held_v2 tokens, ids, provenance) for one record: the contrast held-out rule (module docstring)."""
    from circuit import heldout_rule as HR
    n_ntr = neg_split(rec)
    neg = rec["neg"]
    neg_ids = rec.get("neg_ids")
    fb = contrast_forbidden(rec, s2, m)
    cache = {}
    fire = fire_fn_for(E, key, cache)
    own_ids = set(int(x) for x in neg_ids) if neg_ids is not None else set()
    own_h = set(int(x) for x in HR.window_hash(neg))
    pool = dict(ids=None, tokens=None, prefix_ok=None, n=0)

    def ckey(c):                                         # exclusion key: the id, or the content when the id is unknown
        return c.id if c.id > 0 else "h%d" % c.hash

    def originals():
        return [HR.Candidate(tokens=neg[j].long(), id=int(neg_ids[j]) if neg_ids is not None else -1,
                             origin="original", anchor=HR.CONTRAST_ANCHOR, info=dict(neg_index=int(j)))
                for j in range(n_ntr, int(neg.shape[0]))]

    def refill():
        if pool["ids"] is None:
            ids, tok = E.contrast_pool(key, int(neg.shape[0]) + NEG_EXTRA)
            hs = [int(x) for x in HR.window_hash(tok)] if len(ids) else []
            head = set(ids[:int(neg.shape[0])])
            pool["prefix_ok"] = (head == own_ids) if neg_ids is not None else (set(hs[:int(neg.shape[0])]) == own_h)
            keep = [r for r, (i, h) in enumerate(zip(ids, hs)) if i not in own_ids and h not in own_h]
            pool.update(ids=[ids[r] for r in keep], tokens=tok[keep] if keep else tok[:0], n=len(ids),
                        ranks=[r + 1 for r in keep])
        return [HR.Candidate(tokens=t, id=int(i), origin="refill", anchor=HR.CONTRAST_ANCHOR,
                             info=dict(sel_rank=int(r))) for i, t, r in zip(pool["ids"], pool["tokens"], pool["ranks"])]

    def select(excl):
        cs = originals()
        for c in cs:                                     # hashes first: the exclusion key of an id-less row
            c.hash = int(HR.window_hash(c.tokens)[0])
        cs = [c for c in cs if ckey(c) not in excl]
        acc, ex = HR.select_heldout(cs, fb, fire, n_held=HR.N_HELD, want="silent")
        if len(acc) < HR.N_HELD:                         # short: the selector's next candidates
            rf = refill()
            for c in rf:
                c.hash = int(HR.window_hash(c.tokens)[0])
            cs = cs + [c for c in rf if ckey(c) not in excl]
            acc, ex = HR.select_heldout(cs, fb, fire, n_held=HR.N_HELD, want="silent")
        return cs, acc, ex

    def order(cs, acc):
        o = sorted((i for i in acc if cs[i].origin == "original"), key=lambda i: cs[i].info["neg_index"])
        return o + [i for i in acc if cs[i].origin == "refill"]

    def fires_in_final(cands):
        if not cands:
            return set()
        A = E.acts(key, torch.stack([c.tokens for c in cands]))
        return {ckey(c) for c, a in zip(cands, A) if float(a.max()) > 0}

    cs, acc, ex, unstable = stable(select, order, fires_in_final, "unstable_silence")
    rows = order(cs, acc)
    k = {i: n for n, i in enumerate(acc)}
    tok = torch.stack([cs[i].tokens for i in rows]).to(neg.dtype) if rows else neg[:0].clone()
    ids = [int(cs[i].id) for i in rows]
    prov = [dict(origin=cs[i].origin, id=int(cs[i].id), accept_order=k[i], peak=ex[i]["fire"]["peak"],
                 hash=str(cs[i].hash), **cs[i].info) for i in rows]
    n_orig = int(neg.shape[0]) - n_ntr
    info = dict(n_held=len(rows), n_orig_held=n_orig, n_orig_kept=sum(p["origin"] == "original" for p in prov),
                n_refill=sum(p["origin"] == "refill" for p in prov), below_floor=len(rows) < HR.FLOOR,
                n_train=n_ntr, reasons=HR.reason_counts(ex), n_examined=len(ex), unstable_silence=unstable,
                refill_fetched=pool["ids"] is not None, refill_pool=len(pool["ids"] or []), selector_n=pool["n"],
                selector_prefix_ok=pool["prefix_ok"], neg_extra=NEG_EXTRA, rows=prov,
                forbidden=dict(n_rows=int(fb.tokens.shape[0]), roles="neg[:n_train] + strong (all rows, rebuilt) + mid"))
    return tok, ids, info


def build_one(key, rec, scan_rows, corpus, E, scan_meta):
    """(new record, summary row) for one target."""
    from circuit import heldout_rule as HR
    if rec.get("strong") is None:
        r2 = dict(rec)
        r2["heldout_v2"] = dict(rule=HR.RULE_VERSION, skip=True)
        return r2, dict(seed=key, skip=rec.get("no_contrast") and "no_contrast" or "too_few_contexts")
    s, m = rec["strong"], rec.get("mid")
    n0 = int(s["pos"].shape[0])
    assert len(s["ids"]) == n0, "ids not row-aligned"
    fb, finfo = forbidden(rec)
    fire_cache, tok_cache = {}, {}
    fire = fire_fn_for(E, key, fire_cache)
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

    def strong_cands():
        cs = [HR.Candidate(tokens=s["pos"][j].long(), id=int(s["ids"][j]), origin="original",
                           anchor=int(s["arg"][j]), info=dict(pool_index=int(j)))
              for j in sorted(int(x) for x in s["held"])]
        for r, i in enumerate(sc_ids):
            if i in pool_ids:                       # the pool's own rows: train = forbidden, held = visited above
                continue
            cs.append(HR.Candidate(tokens=None, id=i, origin="scan",
                                   info=dict(scan_rank=r + 1, scan_mean=float(scan_rows["mean"][r]),
                                             scan_peak=float(scan_rows["peak"][r]), scan_arg=int(scan_rows["arg"][r]))))
        return cs

    # S_tr strength by the ranking statistic, from the scan (looked up by content: pool ids may be twins)
    sc_hash = {int(corpus.hash[i]): float(v) for i, v in zip(sc_ids, scan_rows["mean"].tolist())}
    tr_means = [sc_hash.get(int(corpus.hash[int(s["ids"][j])])) for j in s["train"]]
    tr_means = [x for x in tr_means if x is not None]
    tr_peak = float(np.median(s["peak"][s["train"]].float().numpy()))

    def final_silent(cands):
        """Ids of the rows that are silent at their anchors when the final held-out set is forwarded as the scorer
        sees it (one batch of <= 16, record order: the last eval batch after the 48 reference rows). A row near the
        Top-K cut can fire in one batch composition and not in another; such rows are excluded and the set refilled."""
        if not cands:
            return set()
        A = E.acts(key, torch.stack([c.tokens for c in cands]))
        return {c.id for c, a in zip(cands, A) if not float(a[int(c.anchor)]) > 0}

    def strong_order(cs, acc):
        orig = {cs[i].info["pool_index"]: i for i in acc if cs[i].origin == "original"}
        return [orig[int(j)] for j in s["held"] if int(j) in orig] + [i for i in acc if cs[i].origin == "scan"]

    def sel_strong(excl):
        cs_ = [c for c in strong_cands() if c.id not in excl]
        a_, e_ = HR.select_heldout(cs_, fb, fire, n_held=HR.N_HELD, exclude_mid=True, load_fn=load_fn)
        return cs_, a_, e_
    cs, acc, ex, unstable_s = stable(sel_strong, strong_order, final_silent, "unstable_fire")
    kept_orig = {cs[i].info["pool_index"] for i in acc if cs[i].origin == "original"}
    new = [cs[i] for i in acc if cs[i].origin == "scan"]
    s2 = dict(s)
    if new:
        pos = torch.stack([tok_cache[c.id][0] for c in new]).to(s["pos"].dtype)
        tgt = torch.stack([tok_cache[c.id][1] for c in new]).to(s["tgt"].dtype)
        fr = [ex[cs.index(c)]["fire"] for c in new]
        s2["pos"] = torch.cat([s["pos"], pos]); s2["tgt"] = torch.cat([s["tgt"], tgt])
        s2["arg"] = torch.cat([s["arg"], torch.tensor([f["anchor"] for f in fr], dtype=s["arg"].dtype)])
        s2["peak"] = torch.cat([s["peak"], torch.tensor([f["peak"] for f in fr], dtype=s["peak"].dtype)])
        s2["ids"] = list(s["ids"]) + [c.id for c in new]
    s2["held"] = [int(j) for j in s["held"] if int(j) in kept_orig] + list(range(n0, n0 + len(new)))
    prov, order = [], {i: k for k, i in enumerate(acc)}
    idx_of = {cs[i].info["pool_index"]: i for i in acc if cs[i].origin == "original"}
    idx_of.update({n0 + k: cs.index(c) for k, c in enumerate(new)})
    for j in s2["held"]:
        c = cs[idx_of[j]]
        f = ex[idx_of[j]]["fire"]
        prov.append(dict(idx=int(j), origin=c.origin, id=int(c.id), anchor=int(c.anchor),
                         act_at_anchor=f["act_at_anchor"], peak=f["peak"], mean=f["mean"], argmax=f["argmax"],
                         accept_order=order[idx_of[j]], **c.info))
    examined_all = len(ex) == len(cs)
    sinfo = dict(n_held=len(s2["held"]), below_floor=len(s2["held"]) < HR.FLOOR, n_orig_held=len(s["held"]),
                 n_orig_kept=len(kept_orig), n_scan=len(new), rows=prov, reasons=HR.reason_counts(ex),
                 n_examined=len(ex), scan_exhausted=bool(examined_all and len(s2["held"]) < HR.N_HELD),
                 n_scan_rows=len(sc_ids), unstable_fire=unstable_s)
    r2 = dict(rec)
    r2["strong"] = s2
    minfo = None
    if m is not None:
        def sel_mid(excl):
            mc_ = [HR.Candidate(tokens=m["pos"][j].long(), id=int(m["ids"][j]), origin="original",
                                anchor=int(m["arg"][j]), info=dict(pool_index=int(j)))
                   for j in sorted(int(x) for x in m["held"])]
            mc_ = [c for c in mc_ if c.id not in excl]
            a_, e_ = HR.select_heldout(mc_, fb, fire, n_held=len(mc_), exclude_mid=False)
            return mc_, a_, e_

        def mid_order(mc_, a_):
            o = {mc_[i].info["pool_index"]: i for i in a_}
            return [o[int(j)] for j in m["held"] if int(j) in o]
        mc, macc, mex, unstable_m = stable(sel_mid, mid_order, final_silent, "unstable_fire")
        mk = {mc[i].info["pool_index"] for i in macc}
        m2 = dict(m)
        m2["held"] = [int(j) for j in m["held"] if int(j) in mk]
        r2["mid"] = m2
        minfo = dict(n_held=len(m2["held"]), n_orig_held=len(m["held"]), below_floor=len(m2["held"]) < HR.FLOOR,
                     changed=m2["held"] != list(m["held"]), reasons=HR.reason_counts(mex), unstable_fire=unstable_m,
                     dropped={mc[i].info["pool_index"]: v["reason"] for i, v in mex.items() if v["reason"]},
                     rows=[dict(idx=mc[i].info["pool_index"], id=mc[i].id, anchor=mc[i].anchor,
                                accept_order=k, act_at_anchor=mex[i]["fire"]["act_at_anchor"])
                           for k, i in enumerate(macc)])
    ntok, nids, cinfo = rebuild_contrast(key, rec, s2, m, E)
    r2["neg_held_v2"] = ntok
    r2["neg_held_v2_ids"] = nids
    r2["heldout_v2"] = dict(rule=HR.RULE_VERSION, n_pool_orig=n0, strong=sinfo, mid=minfo, contrast=cinfo,
                            forbidden=finfo,
                            scan=dict(file=str(SCAN.name), n_top=scan_meta.get("n_top"),
                                      partial=bool(scan_meta.get("partial")), version=scan_meta.get("version")),
                            ctx_in=str(CTX_IN))
    hp = [p["peak"] for p in prov]; hm = [p["mean"] for p in prov]
    sp = [p["peak"] for p in prov if p["origin"] == "scan"]; sm = [p["mean"] for p in prov if p["origin"] == "scan"]
    med = lambda x: float(np.median(x)) if len(x) else None
    l = int(key.split(".")[0])
    row = dict(seed=key, layer=l, kind=key.split(".")[1], band=band(l), n_held=sinfo["n_held"],
               below_floor=sinfo["below_floor"], n_orig_kept=sinfo["n_orig_kept"], n_scan=sinfo["n_scan"],
               n_orig_held=sinfo["n_orig_held"], scan_exhausted=sinfo["scan_exhausted"], n_examined=len(ex),
               n_scan_rows=len(sc_ids), reasons=sinfo["reasons"],
               tr_peak_med=tr_peak, ho_peak_med=med(hp), scan_peak_med=med(sp),
               tr_mean_med=med(tr_means), n_tr_mean=len(tr_means), ho_mean_med=med(hm), scan_mean_med=med(sm),
               scan_rank_med=med([p["scan_rank"] for p in prov if p["origin"] == "scan"]),
               scan_rank_max=max([p["scan_rank"] for p in prov if p["origin"] == "scan"], default=None),
               scan_arg_agree=(sum(p["argmax"] == p["scan_arg"] for p in prov if p["origin"] == "scan"), len(sp)),
               mid_n_held=minfo and minfo["n_held"], mid_n_orig=minfo and minfo["n_orig_held"],
               mid_changed=minfo and minfo["changed"], mid_below_floor=minfo and minfo["below_floor"],
               mid_reasons=minfo and minfo["reasons"],
               neg_n_held=cinfo["n_held"], neg_n_orig_kept=cinfo["n_orig_kept"], neg_n_refill=cinfo["n_refill"],
               neg_below_floor=cinfo["below_floor"], neg_reasons=cinfo["reasons"],
               neg_refill_fetched=cinfo["refill_fetched"], neg_refill_pool=cinfo["refill_pool"],
               neg_prefix_ok=cinfo["selector_prefix_ok"])
    return r2, row


def targets():
    ks = [t for t in TARGETS.read_text().split() if t][SHARD_I::SHARD_K]
    return ks[:LIMIT] if LIMIT else ks


def build():
    import scan as SC
    sc = SC.load_scan(SCAN)
    log("scan %s: %s" % (SCAN, {k: v for k, v in sc["meta"].items() if k != "files"}))
    corpus = Corpus()
    E = Engine()
    OUT_RB.mkdir(parents=True, exist_ok=True)
    path = OUT_RB / ("build.%s.jsonl" % TAG)
    fh = open(path, "a")
    keys = targets()
    t0, n_done = time.time(), 0
    for n_, key in enumerate(keys):
        if (CTX_OUT / ("%s.pt" % key)).exists():
            continue
        t = time.time()
        try:
            rec = load_rec(CTX_IN, key)
            rows = sc["rows"](key) if rec.get("strong") is not None else None
            r2, row = build_one(key, rec, rows, corpus, E, sc["meta"])
            save_atomic({key: r2}, CTX_OUT / ("%s.pt" % key))
            row["t"] = round(time.time() - t, 2)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            n_done += 1
            log("%4d/%d %-16s strong %s (orig %s + scan %s)  mid %s/%s  contrast %s (orig %s + refill %s)  %.1fs" % (
                n_ + 1, len(keys), key, row.get("n_held", row.get("skip")), row.get("n_orig_kept"), row.get("n_scan"),
                row.get("mid_n_held"), row.get("mid_n_orig"), row.get("neg_n_held"), row.get("neg_n_orig_kept"),
                row.get("neg_n_refill"), time.time() - t))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            fh.write(json.dumps(dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))) + "\n"); fh.flush()
    log("DONE %d built in %.0fs" % (n_done, time.time() - t0))


# ----------------------------------------------------------------------------------------------- verify
# An independent implementation of the predicates (numpy, loops), not heldout_rule's torch code.
def _hash_np(row):
    from circuit.heldout_rule import _HASH_W     # the shared corpus hash definition (085 corpus_index)
    with np.errstate(over="ignore"):
        return int((np.asarray(row, dtype=np.uint64) * _HASH_W[:len(row)]).sum(dtype=np.uint64))


def _lcr_vs(h, O):
    """Longest common run of h [64] against each row of O [r, 64] (row-vectorised DP over h's positions)."""
    r, L = O.shape
    prev = np.zeros((r, L + 1), dtype=np.int32)
    best = np.zeros(r, dtype=np.int32)
    for i in range(L):
        cur = np.zeros((r, L + 1), dtype=np.int32)
        hit = O == h[i]
        cur[:, 1:] = np.where(hit, prev[:, :-1] + 1, 0)
        best = np.maximum(best, cur.max(1))
        prev = cur
    return best


def _c8_vs(h, O):
    g = [tuple(h[k:k + 8]) for k in range(len(h) - 7)]
    out = []
    for o in O:
        s = {tuple(o[k:k + 8]) for k in range(len(o) - 7)}
        out.append(sum(x in s for x in g) / len(g))
    return np.asarray(out)


def _pfx_vs(h, O):
    neq = O != h[None, :]
    first = np.where(neq.any(1), neq.argmax(1), O.shape[1])
    return first


def violations(h, a, O, near_lcr=32, near_c8=0.5):
    """Reasons row h (anchor a) is a duplicate of some row of O: exact / apfx / near."""
    if len(O) == 0:
        return []
    out = []
    pf = _pfx_vs(h, O)
    if (pf == len(h)).any():
        out.append("exact")
    if (pf >= a + 1).any():
        out.append("apfx")
    if (_lcr_vs(h, O) >= near_lcr).any() or (_c8_vs(h, O) >= near_c8).any():
        out.append("near")
    return out


NEW_FIELDS = ("strong", "mid", "heldout_v2", "neg_held_v2", "neg_held_v2_ids")


def verify_one(key, orig, new, corpus, E):
    """Failures of one rebuilt record against the rule, recomputed from the original record."""
    fail = []
    hv = new.get("heldout_v2") or {}
    if orig.get("strong") is None:
        if new.get("strong") is not None:
            fail.append("skip record gained a strong pool")
        return fail
    s0, s1, m0, m1 = orig["strong"], new["strong"], orig.get("mid"), new.get("mid")
    n0 = int(s0["pos"].shape[0])
    # 1. training side untouched (rec["neg"] / neg_ids included: the fit's contrast rows)
    for k in set(orig) | set(new):
        if k in NEW_FIELDS:
            continue
        a, b = orig.get(k), new.get(k)
        same = torch.equal(a, b) if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) else a == b
        if not same:
            fail.append("field %s changed" % k)
    for t in ("pos", "tgt", "arg", "peak"):
        if not torch.equal(s1[t][:n0], s0[t]):
            fail.append("strong.%s[:n0] changed" % t)
    if list(s1["ids"][:n0]) != list(s0["ids"]) or list(s1["train"]) != list(s0["train"]):
        fail.append("strong ids / train changed")
    if (m0 is None) != (m1 is None):
        fail.append("mid presence changed")
    if m0 is not None:
        for t in m0:
            if t == "held":
                continue
            a, b = m0[t], m1[t]
            if not (torch.equal(a, b) if isinstance(a, torch.Tensor) else list(a) == list(b)):
                fail.append("mid.%s changed" % t)
        if not set(m1["held"]) <= set(int(x) for x in m0["held"]):
            fail.append("mid held gained rows")
    # 2. held structure
    H = [int(j) for j in s1["held"]]
    n1 = int(s1["pos"].shape[0])
    if len(H) != len(set(H)) or len(H) > 16 or any(j < 0 or j >= n1 for j in H):
        fail.append("strong held indices invalid")
    if [j for j in H if j < n0 and j not in set(int(x) for x in s0["held"])]:
        fail.append("strong held took an original non-held row")
    if sorted(j for j in H if j >= n0) != list(range(n0, n1)):
        fail.append("appended rows unused or not all held")
    if len(s1["ids"]) != n1 or s1["tgt"].shape[0] != n1 or s1["arg"].shape[0] != n1 or s1["peak"].shape[0] != n1:
        fail.append("strong tensors misaligned")
    if n1 > n0 and not bool((s1["tgt"][n0:, :63] == s1["pos"][n0:, 1:]).all()):
        fail.append("appended tgt is not pos shifted")
    sv = hv.get("strong") or {}
    if sv.get("n_held") != len(H) or sv.get("below_floor") != (len(H) < 8):
        fail.append("strong counts inconsistent")
    # 3. the rule, from scratch: forbidden rows / ids / hashes
    F = [s0["pos"][s0["train"]].numpy()]
    fids = [int(s0["ids"][j]) for j in s0["train"]]
    if m0 is not None:
        F.append(m0["pos"][m0["train"]].numpy()); fids += [int(m0["ids"][j]) for j in m0["train"]]
    F.append(orig["neg"].numpy()); fids += [int(x) for x in (orig.get("neg_ids") or [])]
    F = np.concatenate(F).astype(np.int64)
    fhash = {_hash_np(r) for r in F}
    mid_ids = set(int(x) for x in m0["ids"]) if m0 is not None else set()
    mid_h = {_hash_np(r) for r in m0["pos"].numpy()} if m0 is not None else set()

    def check_set(name, rows, ids, anchors, order, F_, fids_, fhash_, strongest, labels=None):
        """rows / ids / anchors / order (acceptance order) aligned; failures name labels[x] (default the position)."""
        rows_acc = []
        fids_ = set(fids_)
        for x in sorted(range(len(rows)), key=lambda x_: order[x_]):
            j = x if labels is None else labels[x]
            h, a, i = np.asarray(rows[x], dtype=np.int64), int(anchors[x]), int(ids[x])
            hh = _hash_np(h)
            if i > 0 and i in fids_:
                fail.append("%s %d: forbidden id" % (name, j))
            if hh in fhash_:
                fail.append("%s %d: forbidden content" % (name, j))
            if i > 0 and int(corpus.hash[i]) != hh:
                fail.append("%s %d: id does not hash to its tokens" % (name, j))
            if strongest and (i in mid_ids or hh in mid_h):
                fail.append("%s %d: mid-band row in the strongest set" % (name, j))
            for v in violations(h, a, F_):
                fail.append("%s %d: %s vs F" % (name, j, v))
            for v in violations(h, a, np.asarray(rows_acc)) if rows_acc else []:
                fail.append("%s %d: %s vs held" % (name, j, v))
            rows_acc.append(h)

    order_s = {p["idx"]: p["accept_order"] for p in sv.get("rows", [])}
    if set(order_s) != set(H):
        fail.append("strong provenance does not cover the held set")
    else:
        check_set("strong", [s1["pos"][j].numpy() for j in H], [s1["ids"][j] for j in H], [s1["arg"][j] for j in H],
                  [order_s[j] for j in H], F, fids, fhash, True, labels=H)
    Hm = [int(x) for x in m1["held"]] if m1 is not None else []
    if m1 is not None:
        mv = hv.get("mid") or {}
        order_m = {p["idx"]: p["accept_order"] for p in mv.get("rows", [])}
        if set(order_m) != set(Hm):
            fail.append("mid provenance does not cover the held set")
        else:
            check_set("mid", [m1["pos"][j].numpy() for j in Hm], [m1["ids"][j] for j in Hm], [m1["arg"][j] for j in Hm],
                      [order_m[j] for j in Hm], F, fids, fhash, False, labels=Hm)
    # 4. the contrast held-out set: same predicates, sides swapped (anchor = last position), silent everywhere
    nh, nid = new.get("neg_held_v2"), new.get("neg_held_v2_ids")
    cv = hv.get("contrast") or {}
    if nh is None or nid is None:
        fail.append("no rebuilt contrast held-out set")
    else:
        n_neg = int(orig["neg"].shape[0])
        n_ntr = max(1, n_neg - int(round(n_neg * 0.25)))
        if int(nh.shape[0]) != len(nid) or int(nh.shape[0]) > 16 or (int(nh.shape[0]) and nh.shape[1] != 64):
            fail.append("contrast held-out shape invalid")
        if cv.get("n_held") != int(nh.shape[0]) or cv.get("below_floor") != (int(nh.shape[0]) < 8) \
                or cv.get("n_train") != n_ntr:
            fail.append("contrast counts inconsistent")
        oids = list(orig.get("neg_ids") or [])
        own = {_hash_np(r) for r in orig["neg"].numpy()}
        for j, (r, i) in enumerate(zip(nh.numpy(), nid)):
            if int(i) > 0 and int(i) in set(oids[:n_ntr]):
                fail.append("contrast %d: a contrast-train id" % j)
            if int(i) > 0 and oids and int(i) in set(oids[n_ntr:]) and _hash_np(r) not in own:
                fail.append("contrast %d: original id with other tokens" % j)
        Fc = [orig["neg"][:n_ntr].numpy(), s1["pos"].numpy()] + ([m1["pos"].numpy()] if m1 is not None else [])
        Fc = np.concatenate(Fc).astype(np.int64)
        fcid = [int(x) for x in oids[:n_ntr]] + [int(x) for x in s1["ids"]] + \
            ([int(x) for x in m1["ids"]] if m1 is not None else [])
        crow = cv.get("rows", [])
        if len(crow) != int(nh.shape[0]) or [int(p["id"]) for p in crow] != [int(x) for x in nid]:
            fail.append("contrast provenance does not cover the held set")
        else:
            check_set("contrast", list(nh.numpy()), list(nid), [63] * len(nid), [p["accept_order"] for p in crow],
                      Fc, fcid, {_hash_np(r) for r in Fc}, False)
    # 5. firing / silence, recomputed (the protocol forward)
    if E is not None:
        for name, d, held in (("strong", s1, H), ("mid", m1, Hm)):
            if not held:
                continue
            A = E.acts(key, d["pos"][held])
            for j, a in zip(held, A):
                if not float(a[int(d["arg"][j])]) > 0:
                    fail.append("%s %d: silent at its anchor" % (name, j))
        if nh is not None and int(nh.shape[0]):
            A = E.acts(key, nh)
            for j, a in enumerate(A):
                if float(a.max()) > 0:
                    fail.append("contrast %d: fires" % j)
    return fail


def verify():
    corpus = Corpus()
    E = Engine() if FIRE else None
    OUT_RB.mkdir(parents=True, exist_ok=True)
    fh = open(OUT_RB / ("verify.%s.jsonl" % TAG), "w")
    tot = Counter()
    for n_, key in enumerate(targets()):
        orig = load_rec(CTX_IN, key)
        p = CTX_OUT / ("%s.pt" % key)
        if not p.exists():
            tot["missing"] += 1
            fh.write(json.dumps(dict(seed=key, missing=True)) + "\n")
            continue
        try:
            f = verify_one(key, orig, load_rec(CTX_OUT, key), corpus, E)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            f = ["error %s: %s" % (type(e).__name__, str(e)[:200])]
        tot["ok" if not f else "fail"] += 1
        fh.write(json.dumps(dict(seed=key, ok=not f, failures=f[:50])) + "\n"); fh.flush()
        if f:
            log("FAIL %s: %s" % (key, f[:5]))
        if n_ % 200 == 0:
            log("%d targets verified: %s" % (n_ + 1, dict(tot)))
    log("verify DONE: %s" % dict(tot))


# ----------------------------------------------------------------------------------------------- report
def report():
    import pandas as pd
    rows = []
    for p in sorted(OUT_RB.glob("build.shard*.jsonl")):
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    d = pd.DataFrame(rows)
    col = lambda name: d.get(name, pd.Series(index=d.index, dtype=object))
    d = d[col("error").isna()].drop_duplicates(["seed"], keep="last")
    col = lambda name: d.get(name, pd.Series(index=d.index, dtype=object))
    u = d[col("skip").isna()].copy()
    from circuit.heldout_rule import RULE_VERSION
    L = ["# 086 held-out rebuild report", "", "rule %s; %d targets (%d skip records copied)" % (
        RULE_VERSION, u.seed.nunique(), d[col("skip").notna()].seed.nunique()), ""]
    q = lambda s, x: float(s.quantile(x)) if len(s.dropna()) else float("nan")
    if len(u):
        L += ["## Held-out strongest", ""]
        L.append("| | value |")
        L.append("|---|---|")
        L.append("| targets | %d |" % len(u))
        L.append("| held-out strongest rows: median / p10 | %.0f / %.0f |" % (u.n_held.median(), q(u.n_held, 0.1)))
        L.append("| reach 16 | %.1f%% |" % (100 * (u.n_held >= 16).mean()))
        L.append("| at or above the floor (8) | %.1f%% |" % (100 * (u.n_held >= 8).mean()))
        L.append("| below the floor (scored and flagged) | %d (%.1f%%) |" % (int((u.n_held < 8).sum()),
                                                                          100 * (u.n_held < 8).mean()))
        L.append("| zero rows | %d |" % int((u.n_held == 0).sum()))
        L.append("| original rows kept: median | %.0f of %.0f |" % (u.n_orig_kept.median(), u.n_orig_held.median()))
        L.append("| scan rows: share of all held rows | %.1f%% |" % (100 * u.n_scan.sum() / max(1, u.n_held.sum())))
        L.append("| scan list exhausted before 16 | %d |" % int(u.scan_exhausted.sum()))
        L.append("| held / strongest-train peak (median of per-target medians) | %.3f |" % (u.ho_peak_med / u.tr_peak_med).median())
        L.append("| scan rows / strongest-train peak | %.3f |" % (u.scan_peak_med / u.tr_peak_med).median())
        L.append("| held / strongest-train mean score | %.3f |" % (u.ho_mean_med / u.tr_mean_med).median())
        L.append("| scan rows / strongest-train mean score | %.3f |" % (u.scan_mean_med / u.tr_mean_med).median())
        L.append("| scan rank of accepted scan rows: median of medians / max | %s / %s |" % (
            u.scan_rank_med.median(), u.scan_rank_max.max()))
        ag = [tuple(x) for x in u.scan_arg_agree.dropna()]
        L.append("| scan argmax == recomputed anchor | %d / %d |" % (sum(a for a, _ in ag), sum(b for _, b in ag)))
        rs = Counter()
        for r in u.reasons:
            rs.update(r)
        L.append("| candidate outcomes | %s |" % ", ".join("%s %d" % kv for kv in rs.most_common()))
        L += ["", "Per band x kind (share reaching 16 / at or above the floor / below the floor, n):", "",
              "| band | kind | n | =16 | >=8 | <8 | median rows |", "|---|---|---|---|---|---|---|"]
        for (b, k), g in u.groupby(["band", "kind"]):
            L.append("| %s | %s | %d | %.1f%% | %.1f%% | %d | %.0f |" % (b, k, len(g), 100 * (g.n_held >= 16).mean(),
                                                                       100 * (g.n_held >= 8).mean(), int((g.n_held < 8).sum()),
                                                                       g.n_held.median()))
        mu = u[u.mid_n_orig.notna()]
        if len(mu):
            mr = Counter()
            for r in mu.mid_reasons:
                mr.update(r or {})
            L += ["", "Mid-band held-out (reported set; rows only removed): %d targets with a mid pool; %d changed "
                  "(%.1f%%); rows %d -> %d; below the floor after: %d; outcomes: %s" % (
                      len(mu), int(mu.mid_changed.sum()), 100 * mu.mid_changed.mean(), int(mu.mid_n_orig.sum()),
                      int(mu.mid_n_held.sum()), int(mu.mid_below_floor.sum()),
                      ", ".join("%s %d" % kv for kv in mr.most_common())), ""]
        if "neg_n_held" in u:
            c = u[u.neg_n_held.notna()]
            cr = Counter()
            for r in c.neg_reasons:
                cr.update(r or {})
            pk = c.neg_prefix_ok.dropna()
            L += ["## Held-out contrast", "", "| | value |", "|---|---|",
                  "| targets | %d |" % len(c),
                  "| contrast held-out rows: median / p10 / min | %.0f / %.0f / %.0f |" % (
                      c.neg_n_held.median(), q(c.neg_n_held, 0.1), c.neg_n_held.min()),
                  "| reach 16 | %.1f%% |" % (100 * (c.neg_n_held >= 16).mean()),
                  "| below 8 (kept, flagged) | %d |" % int((c.neg_n_held < 8).sum()),
                  "| zero rows (contrast slice cannot be scored: skip row) | %d |" % int((c.neg_n_held == 0).sum()),
                  "| original contrast held-out rows kept: median | %.0f of 16 |" % c.neg_n_orig_kept.median(),
                  "| targets with every original kept | %d (%.1f%%) |" % (int((c.neg_n_orig_kept >= 16).sum()),
                                                                        100 * (c.neg_n_orig_kept >= 16).mean()),
                  "| refill rows: share of all contrast held rows | %.1f%% |" % (
                      100 * c.neg_n_refill.sum() / max(1, c.neg_n_held.sum())),
                  "| refill fetched (selector run) | %d |" % int(c.neg_refill_fetched.fillna(False).astype(bool).sum()),
                  "| selector's first 64 == the run's contrast rows | %d / %d |" % (
                      int(pk.astype(bool).sum()), len(pk)),
                  "| candidate outcomes | %s |" % ", ".join("%s %d" % kv for kv in cr.most_common()), ""]
            L += ["| band | kind | n | contrast =16 | <8 | median rows |", "|---|---|---|---|---|---|"]
            for (b, k), g in c.groupby(["band", "kind"]):
                L.append("| %s | %s | %d | %.1f%% | %d | %.0f |" % (b, k, len(g), 100 * (g.neg_n_held >= 16).mean(),
                                                                   int((g.neg_n_held < 8).sum()), g.neg_n_held.median()))
            L.append("")
    vf = []
    for p in sorted(OUT_RB.glob("verify.shard*.jsonl")):
        vf += [json.loads(l) for l in open(p)]
    if vf:
        v = pd.DataFrame(vf)
        L += ["## Verification (independent re-check)", "", "%d records checked: %d ok, %d failed, %d missing" % (
            len(v), int(v.get("ok", pd.Series(dtype=bool)).fillna(False).sum()),
            int((v.get("ok", pd.Series(dtype=bool)) == False).sum()), int(v.get("missing", pd.Series(dtype=bool)).fillna(False).sum()))]  # noqa: E712
    (OUT_RB / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    {"build": build, "verify": verify, "report": report}[MODE]()
