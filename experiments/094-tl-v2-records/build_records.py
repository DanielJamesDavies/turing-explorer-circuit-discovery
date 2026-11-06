"""FRESH TuringLLM CONTEXT RECORDS for the protocol-v2 refit (Daniel, 2026-10-25): leakage fixed at the source.

The 062 records split pools that held corpus twins and near copies (085: 66.8% of held-out strongest rows repeat a
training-side row). These records are built new, for new fits (the 8k refit), so the fix goes into the POOLS: every
activating row of a record is distinct from every other under the 086 rule, and the split then cannot leak.

Per target (L.kind.i), one record in the schema the driver reads (protocol_contexts.build_record's: strong / mid
pools with pos, tgt, arg, peak, ids, train, held; D_train, B_mid, neg, neg_ids, n_top, n_mid), so CTX_DIR can point
at OUT_DIR:

  strongest pool  candidates = the top store's 64 rows (top_ctx) merged with the 086 corpus scan's top 512 (scan.py:
                  the same ranking statistic, the mean post-Top-K activation over the 64 positions; canonical ids, one
                  per content hash, so corpus twins never take two slots), merged by content hash (the scan's value
                  wins), strongest first (mean desc, id asc). Walked in that order; a candidate is ACCEPTED iff it fires
                  (post-Top-K activation > 0 at its argmax, recomputed with the protocol forward,
                  protocol_contexts.target_activations) and is not an exact copy (tokens or content hash), an anchor-
                  prefix copy (under EITHER row's anchor) or a near duplicate (longest common run >= 32, or >= 50% of
                  EITHER row's 8-gram positions found in the other) of an already-accepted row: the symmetric
                  relation of heldout_rule.dup_groups, so the final pool is all singletons. Up to pool_size (64)
  mid-band pool   the mid store's rows (mid_ctx, store order), the same acceptance test against the strongest pool
                  and the mid rows accepted before. Up to 64; the mid store has no deeper source, so duplicates shrink
                  it
  stability       the final pools are forwarded as build_record does (_peak, batch 16, pool order): peak / anchor
                  come from that forward; a row silent there, or a duplicate under the final anchors (dup_groups), is
                  excluded and the pool re-walked (MAX_ROUNDS)
  split           the group-aware split (heldout_rule.group_split_ex over the joint dup_groups labels, as
                  protocol_contexts._group_split_pools; all groups are singletons here, so it is the stratified split):
                  strongest 3:1 (64 -> 48 train / 16 held; D_train = 32 of the 48, ranks 1-2 kept), mid-band 16 held
                  whenever it has >= 32 rows (MID_HELD=fixed16; else 3:1, MID_HELD=ratio), B_mid = 16 spread over
                  mid-train
  contrast        the run's close selector (utils/neg_context_selector, the run's arguments), contrast_count + NEG_EXTRA
                  candidates in similarity order; heldout_rule.select_heldout(want="silent"): silent at every position
                  (recomputed), not a whole-window exact copy (tokens, id, hash) nor a near duplicate of any activating
                  row of the record or of an accepted contrast row (no anchor-prefix rule for contrast rows, Daniel
                  2026-10-24). First 64 accepted, then build_record's stratified order (48 train / 16 held). The final
                  64 are forwarded in record order (batch 16); a row that fires there is excluded and refilled
  final guard     protocol_contexts._group_heldout_filter (the public path's guard): both held-out sets vs every
                  training-side row incl. the contrast rows, and firing in the scorer's own batch (unstable_fire)
  provenance      rec["fresh_v2"]: version, rule, per pool the walk (examined / accepted / reasons, origins store or
                  scan), the scan meta, the split sizes; rec["heldout"] as the public path writes it

  MODE=build   (GPU, resumable per target, SHARD=i/k)   MODE=verify (independent re-check; FIRE=0 skips the GPU part)
  MODE=report  (CPU) -> OUT_RB/report.md
  env: TARGETS (094/targets_tl8k.txt)  OUT_DIR (records, default 094/out/ctx_v2)  OUT_RB (rows / report, 094/out/build)
       SCAN (086 scan_top512.pt over these targets; unset or missing = store rows only, flagged)  CORPUS_DIR (085/out:
       corpus_hash.npy)  STORES_DIR (outputs)  NEG_EXTRA (128)  MID_HELD (fixed16 | ratio)  LIMIT  SHARD  FIRE
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
D86 = REPO / "experiments" / "086-heldout-rebuild"
sys.path.insert(0, str(D86))
MODE = os.environ.get("MODE", "build")
SHARD_I, SHARD_K = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
TARGETS = Path(os.environ.get("TARGETS", str(HERE / "targets_tl8k.txt")))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(HERE / "out" / "ctx_v2")))
OUT_RB = Path(os.environ.get("OUT_RB", str(HERE / "out" / "build")))
SCAN = os.environ.get("SCAN") or None
FIRE = os.environ.get("FIRE", "1") == "1"
NEG_EXTRA = int(os.environ.get("NEG_EXTRA", "128"))
MID_HELD = os.environ.get("MID_HELD", "fixed16")
MAX_ROUNDS = 5
CHUNK = 32
VERSION = "094-fresh-v2/2026-10-25 (dedup pools: store + scan512, symmetric exact/apfx/near; group split; mid %s; " \
          "contrast exact+near)" % MID_HELD
TAG = "shard%d" % SHARD_I
if MID_HELD not in ("fixed16", "ratio"):
    raise SystemExit("MID_HELD must be fixed16 or ratio, got %r" % MID_HELD)


def log(s):
    print("[%s fresh %s] %s" % (time.strftime("%H:%M:%S"), TAG, s), flush=True)


def band(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def targets():
    ks = [t for t in TARGETS.read_text().split() if t][SHARD_I::SHARD_K]
    return ks[:LIMIT] if LIMIT else ks


def save_atomic(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp-%d" % os.getpid())
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------- duplicate test
def dup_reason(h, a, O, aO):
    """Why row h (anchor a) duplicates some row of O (anchors aO): 'exact' | 'apfx' | 'near' | None. Symmetric, as
    heldout_rule.dup_groups links rows: apfx under either anchor, near with either row's 8-grams."""
    from circuit import heldout_rule as HR
    if O is None or int(O.shape[0]) == 0:
        return None, None
    ex, pfx, lcr, c8 = HR.pair_stats(h[None], O)
    _, _, _, c8r = HR.pair_stats(O, h[None])
    aO = np.asarray(aO)
    checks = (("exact", ex[0]), ("apfx", (pfx[0] >= a + 1) | (pfx[0] >= aO + 1)),
              ("near", HR.near_of(lcr[0], c8[0]) | HR.near_of(lcr[0], c8r[:, 0])))
    for name, hit in checks:
        if hit.any():
            return name, int(np.nonzero(hit)[0][0])
    return None, None


# ----------------------------------------------------------------------------------------------- candidates
class Sources:
    """The stores (top / mid ids and values), the corpus hash and the scan, for candidate lists."""

    def __init__(self, E):
        from store.context import mid_ctx, top_ctx
        E.selector()                                     # loads top_ctx / mid_ctx / neg_ctx / seq_repr (086 Engine)
        self.top, self.mid = top_ctx, mid_ctx
        self.hash = np.load(Path(os.environ.get("CORPUS_DIR", str(REPO / "experiments" / "085-heldout-audit" / "out")))
                            / "corpus_hash.npy", mmap_mode="r")
        self.scan, self.scan_meta = None, {}
        if SCAN and Path(SCAN).exists():
            import scan as SC
            self.scan = SC.load_scan(SCAN)
            self.scan_meta = {k: v for k, v in self.scan["meta"].items() if k not in ("files", "scanned_shards")}
        elif SCAN:
            log("SCAN %s missing: store rows only (flagged per record)" % SCAN)

    def strong(self, comp, lat, key):
        """[(id, mean, origin)] strongest first: the scan rows, plus store rows whose content the scan lacks."""
        rows, seen = [], set()
        has_scan = self.scan is not None and key in self.scan["where"]
        if has_scan:
            r = self.scan["rows"](key)
            for i, m in zip(r["ids"].tolist(), r["mean"].tolist()):
                h = int(self.hash[int(i)])
                if h not in seen:
                    seen.add(h); rows.append((int(i), float(m), "scan"))
        ids = self.top.ctx_seq_idx[comp, lat].tolist()
        vals = self.top.ctx_seq_val[comp, lat].float().tolist()
        for i, v in zip(ids, vals):
            if int(i) <= 0:
                continue
            h = int(self.hash[int(i)])
            if h not in seen:
                seen.add(h); rows.append((int(i), float(v), "store"))
        rows.sort(key=lambda x: (-x[1], x[0]))
        return rows, has_scan

    def mid_ids(self, comp, lat):
        out, seen = [], set()
        for i in self.mid.ctx_seq_idx[comp, lat].tolist():
            if int(i) > 0 and int(i) not in seen:
                seen.add(int(i)); out.append(int(i))
        return out


def walk(cands, n_want, fire, load_fn, excl, against=None):
    """Greedy pool selection (module docstring). cands: [Candidate] in rank order (tokens loaded on demand).
    against = (tokens [r, 64], anchors) rows every candidate must also differ from (the strongest pool, for mid).
    Returns (accepted indices, examined {index: dict(reason, dup_of, fire)})."""
    from circuit import heldout_rule as HR
    acc, ex = [], {}
    rows, ancs, hashes = [], [], set()
    if against is not None and int(against[0].shape[0]):
        rows = [t.long().cpu() for t in against[0]]
        ancs = [int(a) for a in against[1]]
        hashes = set(int(x) for x in HR.window_hash(against[0]))
    n_fixed = len(rows)
    i0 = 0
    while i0 < len(cands) and len(acc) < n_want:
        idx = list(range(i0, min(len(cands), i0 + CHUNK)))
        i0 = idx[-1] + 1
        need = [cands[i] for i in idx if cands[i].tokens is None]
        if need:
            load_fn(need)
        live = []
        for i in idx:
            c = cands[i]
            c.hash = int(HR.window_hash(c.tokens)[0])
            if c.id in excl or ("h%d" % c.hash) in excl:
                ex[i] = dict(reason="unstable")
            elif c.hash in hashes:
                ex[i] = dict(reason="exact")                    # a twin of an accepted row (same content)
            else:
                live.append(i)
        fires = fire([cands[i] for i in live]) if live else []
        for i, fr in zip(live, fires):
            c = cands[i]
            c.anchor = int(fr["anchor"])
            if len(acc) >= n_want:
                ex[i] = dict(reason="full", fire=fr); continue
            if not float(fr["peak"]) > 0:
                ex[i] = dict(reason="silent", fire=fr); continue
            O = torch.stack(rows) if rows else None
            why, j = dup_reason(c.tokens.long().cpu(), c.anchor, O, ancs)
            if why is None and c.hash in hashes:
                why, j = "exact", -1
            if why is not None:
                ex[i] = dict(reason=why, dup_of=("fixed" if j is not None and j < n_fixed else "pool"), fire=fr)
                continue
            ex[i] = dict(reason=None, fire=fr)
            acc.append(i)
            rows.append(c.tokens.long().cpu()); ancs.append(c.anchor); hashes.add(c.hash)
    return acc, ex


def held_frac(n, name, cfg):
    """The pool's held-out fraction: v1's 3:1, except a mid-band pool of >= 32 rows holds out exactly 16
    (MID_HELD=fixed16): the dedupe shrinks the 64-slot mid store, which has no deeper source."""
    if name == "mid" and MID_HELD == "fixed16" and n >= 32:
        return 16.0 / n
    return float(cfg.holdout_frac)


def split_pools(rec, cfg):
    """protocol_contexts._group_split_pools with a per-pool held-out fraction (held_frac): joint dup_groups labels,
    the strongest pool first, then the mid-band pool with the strongest pool's groups forced to their side."""
    from circuit.heldout_rule import dup_groups, group_split_ex
    names = [n for n in ("strong", "mid") if rec.get(n) is not None]
    lab = dup_groups(torch.cat([rec[n]["pos"] for n in names]), torch.cat([rec[n]["arg"] for n in names]))
    off, f_tr, f_ho = 0, set(), set()
    for n in names:
        d = rec[n]
        k = int(d["pos"].shape[0])
        g = lab[off:off + k]; off += k
        tr, ho, un = group_split_ex(d["peak"], g, holdout_frac=held_frac(k, n, cfg), keep_top=cfg.keep_top,
                                    force_train=f_tr, force_held=f_ho)
        d.update(train=tr, held=ho, unused=un, group=g, doc=[-1] * k)
        f_tr |= {g[j] for j in tr}; f_ho |= {g[j] for j in ho + un}
    return len(set(lab)) == len(lab)


# ----------------------------------------------------------------------------------------------- build
def build_one(key, E, S, cfg):
    from circuit import heldout_rule as HR
    from circuit import protocol_contexts as PC
    from circuit.context_split import stratified_order, stratified_split, stratified_subset
    from pipeline.component_index import component_idx
    import rebuild as RB
    l, kd, lat = key.split(".")
    l, lat = int(l), int(lat)
    kinds = E.kinds
    ki = kinds.index(kd)
    comp = component_idx(l, ki, len(kinds))
    n_pool = int(cfg.pool_size)
    fire = RB.fire_fn_for(E, key, {})
    tok = {}

    def load_fn(cs):
        need = [c.id for c in cs if c.id not in tok]
        for s0 in range(0, len(need), 64):
            part = need[s0:s0 + 64]
            pos, tgt = E.load(part)
            for i, p, t in zip(part, pos, tgt):
                tok[i] = (p.long(), t.long())
        for c in cs:
            c.tokens = tok[c.id][0]

    s_list, has_scan = S.strong(comp, lat, key)
    m_list = S.mid_ids(comp, lat)
    prov = dict(version=VERSION, rule=HR.RULE_VERSION, scan=bool(has_scan), scan_meta=S.scan_meta,
                n_strong_cands=len(s_list), n_strong_cands_scan=sum(o == "scan" for _, _, o in s_list),
                n_mid_store=len(m_list), mid_held=MID_HELD)
    excl_s, excl_m, rounds = set(), set(), 0
    while True:
        rounds += 1
        sc = [HR.Candidate(tokens=None, id=i, origin=o, info=dict(rank=r + 1, mean=m))
              for r, (i, m, o) in enumerate(s_list)]
        sa, sx = walk(sc, n_pool, fire, load_fn, excl_s)
        s_rows = [sc[i] for i in sa]
        mc = [HR.Candidate(tokens=None, id=i, origin="mid_store", info=dict(rank=r + 1)) for r, i in enumerate(m_list)]
        against = ((torch.stack([c.tokens for c in s_rows]), [c.anchor for c in s_rows]) if s_rows
                   else (torch.zeros(0, 64, dtype=torch.long), []))
        ma, mx = walk(mc, n_pool, fire, load_fn, excl_m, against=against)
        m_rows = [mc[i] for i in ma]
        rec = dict(n_top=len(s_rows), n_mid=len(m_rows))
        bad_s, bad_m = set(), set()
        for name, rows, bad in (("strong", s_rows, bad_s), ("mid", m_rows, bad_m)):
            if len(rows) < int(cfg.min_pool):
                rec[name] = None
                continue
            pos = torch.stack([tok[c.id][0] for c in rows])
            tgt = torch.stack([tok[c.id][1] for c in rows])
            v, a = PC._peak(E.inference, E.bank, l, ki, kd, lat, pos.to(E.dev))
            for c, vv in zip(rows, v.tolist()):
                if not vv > 0:
                    bad.add(c.id)
            rec[name] = dict(pos=pos, tgt=tgt, arg=a, peak=v, ids=[int(c.id) for c in rows],
                             origin=[c.origin for c in rows], rank=[int(c.info["rank"]) for c in rows])
        # duplicates under the FINAL anchors (the selection forward's argmax may differ near ties)
        names = [n for n in ("strong", "mid") if rec.get(n) is not None]
        if names and not bad_s and not bad_m:
            lab = HR.dup_groups(torch.cat([rec[n]["pos"] for n in names]), torch.cat([rec[n]["arg"] for n in names]))
            off = 0
            for n in names:
                k = int(rec[n]["pos"].shape[0])
                for j in range(k):
                    if lab[off + j] != off + j:                 # not its group's first row: a duplicate
                        (bad_s if n == "strong" else bad_m).add(int(rec[n]["ids"][j]))
                off += k
        if (not bad_s and not bad_m) or rounds >= MAX_ROUNDS:
            break
        excl_s |= bad_s; excl_m |= bad_m
    prov.update(rounds=rounds, unstable_strong=sorted(excl_s | bad_s), unstable_mid=sorted(excl_m | bad_m),
                strong_walk=dict(n_examined=len(sx), n_accepted=len(sa), reasons=HR.reason_counts(sx),
                                 n_scan=sum(c.origin == "scan" for c in s_rows),
                                 n_store=sum(c.origin == "store" for c in s_rows),
                                 last_rank=max([c.info["rank"] for c in s_rows], default=None),
                                 exhausted=len(sx) == len(sc) and len(sa) < n_pool),
                mid_walk=dict(n_examined=len(mx), n_accepted=len(ma), reasons=HR.reason_counts(mx)))
    for n, bad in (("strong", bad_s), ("mid", bad_m)):            # rows still flipping after MAX_ROUNDS: drop them
        d = rec.get(n)
        if d is None or not bad:
            continue
        keep = [j for j, i in enumerate(d["ids"]) if i not in bad]
        for f in ("pos", "tgt", "arg", "peak"):
            d[f] = d[f][keep]
        for f in ("ids", "origin", "rank"):
            d[f] = [d[f][j] for j in keep]
        if len(keep) < int(cfg.min_pool):
            rec[n] = None
    rec["fresh_v2"] = prov
    if rec.get("strong") is None:
        return rec, dict(seed=key, skip="too_few_contexts", n_top=rec["n_top"], scan=has_scan)
    prov["singletons"] = split_pools(rec, cfg)
    s = rec["strong"]
    hf = PC.strong_holdout_frac(n_pool, cfg.holdout_frac, cfg.train_strong)
    d_keep, _ = stratified_split(s["peak"][s["train"]], holdout_frac=hf, keep_top=cfg.keep_top)
    rec["D_train"] = [s["train"][j] for j in d_keep]
    if rec.get("mid") is not None:
        m = rec["mid"]
        rec["B_mid"] = [m["train"][j] for j in stratified_subset(m["peak"][m["train"]], int(cfg.train_mid))]
    # contrast
    n_c = int(cfg.contrast_count)
    act = [rec[n] for n in ("strong", "mid") if rec.get(n) is not None]
    F = torch.cat([d["pos"].long() for d in act])
    fb = HR.Forbidden(tokens=F, ids={int(i) for d in act for i in d["ids"]},
                      hashes=set(int(x) for x in HR.window_hash(F)))
    pool = {}

    def ckey(c):
        return c.id if c.id > 0 else "h%d" % c.hash

    def sel(excl):
        if "ids" not in pool:
            ids, t = E.contrast_pool(key, n_c + NEG_EXTRA)
            pool.update(ids=ids if len(ids) == int(t.shape[0]) else [-1] * int(t.shape[0]), tok=t)
        cs = [HR.Candidate(tokens=t.long(), id=int(i), origin="selector", anchor=HR.CONTRAST_ANCHOR,
                           info=dict(sel_rank=r + 1)) for r, (i, t) in enumerate(zip(pool["ids"], pool["tok"]))]
        for c in cs:
            c.hash = int(HR.window_hash(c.tokens)[0])
        cs = [c for c in cs if ckey(c) not in excl]
        acc, ex = HR.select_heldout(cs, fb, fire, n_held=n_c, want="silent")
        return cs, acc, ex

    def c_order(cs, acc):
        o = stratified_order(-torch.arange(len(acc), dtype=torch.float64), holdout_frac=cfg.holdout_frac,
                             keep_top=cfg.keep_top)
        return [acc[j] for j in o]

    def fires_in_final(cands):
        if not cands:
            return set()
        A = E.acts(key, torch.stack([c.tokens for c in cands]))      # record order, batch 16 (the protocol forward)
        return {ckey(c) for c, a in zip(cands, A) if float(a.max()) > 0}

    cs, cacc, cex, c_unstable = RB.stable(sel, c_order, fires_in_final, "unstable_silence")
    prov["contrast"] = dict(n_selector=len(pool.get("ids", [])), n_accepted=len(cacc), reasons=HR.reason_counts(cex),
                            unstable_silence=c_unstable, neg_extra=NEG_EXTRA,
                            ids_known=bool(pool.get("ids")) and pool["ids"][0] != -1 if pool.get("ids") else False)
    if len(cacc) < int(cfg.min_contrast):
        rec["strong"] = None; rec["no_contrast"] = True
        return rec, dict(seed=key, skip="no_contrast", n_contrast=len(cacc), scan=has_scan)
    rows = c_order(cs, cacc)
    rec["neg"] = torch.stack([cs[i].tokens for i in rows]).cpu()
    ids = [int(cs[i].id) for i in rows]
    rec["neg_ids"] = ids if all(i > 0 for i in ids) else None
    prov["contrast"]["sel_rank"] = [int(cs[i].info["sel_rank"]) for i in rows]
    # final guard (the public path's): held-out sets vs every training-side row incl. contrast; firing in-batch
    PC._group_heldout_filter(rec, PC.GroupSplit(), final_acts=lambda t: E.acts(key, t))
    s, m = rec["strong"], rec.get("mid")
    med = lambda x: float(np.median(x)) if len(x) else None
    tr_pk, ho_pk = s["peak"][s["train"]].float().numpy(), s["peak"][s["held"]].float().numpy()
    mean_of = {int(i): float(mm) for i, mm, _ in s_list}
    row = dict(seed=key, layer=l, kind=kd, band=band(l), scan=has_scan, n_strong=int(s["pos"].shape[0]),
               n_train=len(s["train"]), n_held=len(s["held"]), n_dtrain=len(rec["D_train"]),
               n_mid=int(m["pos"].shape[0]) if m is not None else 0, mid_train=len(m["train"]) if m is not None else 0,
               mid_held=len(m["held"]) if m is not None else 0, n_bmid=len(rec.get("B_mid") or []),
               n_neg=int(rec["neg"].shape[0]), n_scan=prov["strong_walk"]["n_scan"],
               n_store=prov["strong_walk"]["n_store"], last_rank=prov["strong_walk"]["last_rank"],
               strong_reasons=prov["strong_walk"]["reasons"], mid_reasons=prov["mid_walk"]["reasons"],
               neg_reasons=prov["contrast"]["reasons"], rounds=rounds, singletons=prov["singletons"],
               tr_peak_med=med(tr_pk), ho_peak_med=med(ho_pk),
               tr_mean_med=med([mean_of[int(s["ids"][j])] for j in s["train"]]),
               ho_mean_med=med([mean_of[int(s["ids"][j])] for j in s["held"]]),
               guard_dropped=sum(len((rec["heldout"].get(n) or {}).get("dropped") or {}) for n in ("strong", "mid")))
    return rec, row


def build():
    from config import config
    import rebuild as RB
    cfg = config.discovery.context_v1
    E = RB.Engine()
    S = Sources(E)
    log("scan %s | %s" % (SCAN, S.scan_meta))
    OUT_RB.mkdir(parents=True, exist_ok=True)
    fh = open(OUT_RB / ("build.%s.jsonl" % TAG), "a")
    keys = targets()
    t0, n_done = time.time(), 0
    for n_, key in enumerate(keys):
        if (OUT_DIR / ("%s.pt" % key)).exists():
            continue
        t = time.time()
        try:
            try:
                rec, row = build_one(key, E, S, cfg)
            except Exception as e:  # noqa: BLE001  (one retry: a transient CUDA / IO error must not leave a gap
                log("%s: %s: %s; retrying once" % (key, type(e).__name__, str(e)[:200]))      # for the verify gate)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                rec, row = build_one(key, E, S, cfg)
            save_atomic({key: rec}, OUT_DIR / ("%s.pt" % key))
            row["t"] = round(time.time() - t, 2)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            n_done += 1
            log("%4d/%d %-16s %s strong %s (train %s held %s; scan %s store %s, last rank %s) mid %s/%s neg %s  %.1fs" % (
                n_ + 1, len(keys), key, row.get("skip", "ok"), row.get("n_strong"), row.get("n_train"),
                row.get("n_held"), row.get("n_scan"), row.get("n_store"), row.get("last_rank"), row.get("mid_train"),
                row.get("mid_held"), row.get("n_neg"), time.time() - t))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            fh.write(json.dumps(dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))) + "\n"); fh.flush()
    log("DONE %d built in %.0fs" % (n_done, time.time() - t0))


# ----------------------------------------------------------------------------------------------- verify
def verify_one(key, rec, corpus, E, cfg, warn=None):
    """Failures of one fresh record, from scratch with 086's independent numpy predicates (rebuild.violations).
    warn (a list) collects non-rule observations (recomputed peak drift)."""
    import rebuild as RB
    fail = []
    if rec.get("strong") is None:
        return fail if rec.get("fresh_v2") else ["skip record without provenance"]
    s, m = rec["strong"], rec.get("mid")
    pools = [("strong", s)] + ([("mid", m)] if m is not None else [])
    # 1. structure
    for name, d in pools:
        n = int(d["pos"].shape[0])
        if n > int(cfg.pool_size) or d["tgt"].shape[0] != n or d["arg"].shape[0] != n or d["peak"].shape[0] != n \
                or len(d["ids"]) != n:
            fail.append("%s: fields misaligned" % name)
        if not bool((d["tgt"][:, :63] == d["pos"][:, 1:]).all()):
            fail.append("%s: tgt is not pos shifted" % name)
        tr, ho, un = [int(j) for j in d["train"]], [int(j) for j in d["held"]], [int(j) for j in d.get("unused", [])]
        if set(tr) & set(ho) or sorted(tr + ho + un) != list(range(n)):
            fail.append("%s: train / held / unused do not partition the pool" % name)
        want = 16 if (name == "strong" and n == int(cfg.pool_size)) or (name == "mid" and MID_HELD == "fixed16"
                                                                         and n >= 32) else None
        if want is not None and len(ho) + len(un) != want:
            fail.append("%s: %d held + %d unused, expected %d" % (name, len(ho), len(un), want))
        if len(ho) > 16:
            fail.append("%s: more than 16 held" % name)
    if not set(rec["D_train"]) <= set(int(j) for j in s["train"]):
        fail.append("D_train outside strong.train")
    if int(s["pos"].shape[0]) == int(cfg.pool_size) and (len(s["train"]) != 48 or len(rec["D_train"]) != 32):
        fail.append("full strongest pool without the 48 / 32 split")
    if m is not None:
        if not set(rec["B_mid"]) <= set(int(j) for j in m["train"]):
            fail.append("B_mid outside mid.train")
        if len(m["train"]) >= 16 and len(rec["B_mid"]) != 16:
            fail.append("B_mid is not 16")
    # 2. the rule from scratch over every activating row: ids hash to their tokens, all contents distinct,
    #    no pair exact / apfx (either anchor) / near (either direction)
    P = np.concatenate([d["pos"].numpy() for _, d in pools]).astype(np.int64)
    A = np.concatenate([d["arg"].numpy() for _, d in pools]).astype(np.int64)
    I = [int(i) for _, d in pools for i in d["ids"]]
    lab = ["%s %d" % (name, j) for name, d in pools for j in range(int(d["pos"].shape[0]))]
    H = [RB._hash_np(r) for r in P]
    if len(set(H)) != len(H) or len(set(I)) != len(I):
        fail.append("activating rows share a content hash or an id")
    for x, (i, h) in enumerate(zip(I, H)):
        if i <= 0 or int(corpus.hash[i]) != h:
            fail.append("%s: id %d does not hash to its tokens" % (lab[x], i))
    for x in range(1, len(P)):
        v = RB.violations(P[x], int(A[x]), P[:x])                   # x vs earlier rows, x's anchor, x's 8-grams
        O = P[:x]
        pf = RB._pfx_vs(P[x], O)
        if (pf >= A[:x] + 1).any() and "apfx" not in v:              # apfx under the OTHER row's anchor
            v.append("apfx(other anchor)")
        c8r = np.asarray([RB._c8_vs(o, P[x][None])[0] for o in O])   # the other rows' 8-grams in x
        if (c8r >= 0.5).any() and "near" not in v:
            v.append("near(other 8-grams)")
        for w in v:
            fail.append("%s: %s vs an earlier activating row" % (lab[x], w))
    # 3. contrast: silent (below), distinct, not exact / near to any activating row nor to each other
    N = rec["neg"].numpy().astype(np.int64)
    if N.shape[0] < int(cfg.min_contrast) or N.shape[0] > int(cfg.contrast_count):
        fail.append("contrast size %d" % N.shape[0])
    nids = rec.get("neg_ids")
    if nids is not None and (len(nids) != N.shape[0] or len(set(nids)) != len(nids) or set(nids) & set(I)):
        fail.append("contrast ids misaligned, repeated or activating")
    if nids is not None:
        for j, (r, i) in enumerate(zip(N, nids)):
            if int(corpus.hash[int(i)]) != RB._hash_np(r):
                fail.append("contrast %d: id does not hash to its tokens" % j)
    fh = set(H)
    for j in range(N.shape[0]):
        if RB._hash_np(N[j]) in fh:
            fail.append("contrast %d: same content as an activating row" % j)
        for w in RB.violations(N[j], 63, P):
            if w != "apfx":                                          # anchor 63: apfx == exact
                fail.append("contrast %d: %s vs an activating row" % (j, w))
        for w in RB.violations(N[j], 63, N[:j]) if j else []:
            if w != "apfx":
                fail.append("contrast %d: %s vs an earlier contrast row" % (j, w))
    # 4. firing / silence recomputed (the protocol forward): pools as built (batch 16, pool order), held sets as the
    #    scorer forwards them (one batch, record order), the 64 contrast rows in record order
    if E is not None:
        for name, d in pools:
            acts = E.acts(key, d["pos"])
            pk = acts.max(-1).values
            if warn is not None and not torch.allclose(pk, d["peak"].float(), rtol=1e-3, atol=1e-3):
                # numerical noise, not the rule: reported (verify rows' "warnings"), never a failure
                warn.append("%s: recomputed peak differs (max |d| %.3g)" % (name, float((pk - d["peak"]).abs().max())))
            for j, a in enumerate(acts):
                if not float(a[int(d["arg"][j])]) > 0:
                    fail.append("%s %d: silent at its anchor" % (name, j))
            held = [int(j) for j in d["held"]]
            if held:
                Ah = E.acts(key, d["pos"][held])
                for j, a in zip(held, Ah):
                    if not float(a[int(d["arg"][j])]) > 0:
                        fail.append("%s %d: silent at its anchor in the held-out batch" % (name, j))
        An = E.acts(key, rec["neg"])
        for j, a in enumerate(An):
            if float(a.max()) > 0:
                fail.append("contrast %d: fires" % j)
    return fail


def verify():
    from config import config
    import rebuild as RB
    cfg = config.discovery.context_v1
    corpus = RB.Corpus(Path(os.environ.get("CORPUS_DIR", str(REPO / "experiments" / "085-heldout-audit" / "out"))))
    E = RB.Engine() if FIRE else None
    OUT_RB.mkdir(parents=True, exist_ok=True)
    fh = open(OUT_RB / ("verify.%s.jsonl" % TAG), "w")
    tot = Counter()
    keys = targets()
    for n_, key in enumerate(keys):
        p = OUT_DIR / ("%s.pt" % key)
        if not p.exists():
            log("%4d/%d %-16s verify MISSING (no record)" % (n_ + 1, len(keys), key))
            tot["missing"] += 1
            fh.write(json.dumps(dict(seed=key, missing=True)) + "\n")
            continue
        w = []
        try:
            f = verify_one(key, torch.load(p, weights_only=False)[key], corpus, E, cfg, warn=w)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            f = ["error %s: %s" % (type(e).__name__, str(e)[:200])]
        tot["ok" if not f else "fail"] += 1
        tot["warned"] += bool(w)
        fh.write(json.dumps(dict(seed=key, ok=not f, failures=f[:50], warnings=w[:10])) + "\n"); fh.flush()
        log("%4d/%d %-16s verify %s" % (n_ + 1, len(keys), key, "ok" if not f else "FAIL %s" % f[:3]))  # progress line
        if n_ % 200 == 0:
            log("%d verified: %s" % (n_ + 1, dict(tot)))
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
    if not rows:
        raise SystemExit("no build rows in %s" % OUT_RB)
    d = pd.DataFrame(rows)
    col = lambda name: d.get(name, pd.Series(index=d.index, dtype=object))
    d = d[col("error").isna()].drop_duplicates(["seed"], keep="last")
    col = lambda name: d.get(name, pd.Series(index=d.index, dtype=object))
    u = d[col("skip").isna()].copy()
    L = ["# 094 fresh TuringLLM records (protocol v2)", "", VERSION, "",
         "%d targets built: %d usable, %d skip records (%s)" % (
             len(d), len(u), len(d) - len(u), dict(Counter(col("skip").dropna())))]
    if len(u):
        rs, ms, ns = Counter(), Counter(), Counter()
        for a, b, c in zip(u.strong_reasons, u.mid_reasons, u.neg_reasons):
            rs.update(a or {}); ms.update(b or {}); ns.update(c or {})
        q = lambda s, x: float(s.quantile(x))
        L += ["", "| | value |", "|---|---|",
              "| scan rows available (targets) | %d / %d |" % (int(u.scan.sum()), len(u)),
              "| strongest pool = 64 | %.1f%% (min %d) |" % (100 * (u.n_strong >= 64).mean(), u.n_strong.min()),
              "| held-out strongest = 16 | %.1f%%; below the floor (8): %d |" % (100 * (u.n_held >= 16).mean(),
                                                                             int((u.n_held < 8).sum())),
              "| strongest-train 48 / D_train 32 | %.1f%% / %.1f%% |" % (100 * (u.n_train == 48).mean(),
                                                                       100 * (u.n_dtrain == 32).mean()),
              "| mid pool rows: median / p10 / min | %.0f / %.0f / %d |" % (u.n_mid.median(), q(u.n_mid, 0.1),
                                                                         u.n_mid.min()),
              "| held-out mid = 16 | %.1f%%; below 8: %d; no mid pool (thin): %d |" % (
                  100 * (u.mid_held >= 16).mean(), int(((u.mid_held < 8) & (u.n_mid > 0)).sum()), int((u.n_mid == 0).sum())),
              "| B_mid = 16 | %.1f%% |" % (100 * (u.n_bmid == 16).mean()),
              "| contrast = 64 | %.1f%% (min %d) |" % (100 * (u.n_neg >= 64).mean(), u.n_neg.min()),
              "| strongest pool rows from the scan (beyond the store's 64) | %.1f%% |" % (
                  100 * u.n_scan.sum() / max(1, u.n_strong.sum())),
              "| deepest candidate rank used: median / max | %.0f / %d |" % (u.last_rank.median(), u.last_rank.max()),
              "| held / train peak (median of per-target medians) | %.3f |" % (u.ho_peak_med / u.tr_peak_med).median(),
              "| held / train mean score | %.3f |" % (u.ho_mean_med / u.tr_mean_med).median(),
              "| all activating rows distinct (singleton groups) | %d / %d |" % (int(u.singletons.sum()), len(u)),
              "| rows dropped by the final guard | %d |" % int(u.guard_dropped.sum()),
              "| strongest walk outcomes | %s |" % ", ".join("%s %d" % kv for kv in rs.most_common()),
              "| mid walk outcomes | %s |" % ", ".join("%s %d" % kv for kv in ms.most_common()),
              "| contrast outcomes | %s |" % ", ".join("%s %d" % kv for kv in ns.most_common()),
              "| build seconds per target: median | %.1f |" % u.t.median() if "t" in u else "", ""]
        L += ["| band | kind | n | held 16 | mid held 16 | mid rows median | held/train peak |",
              "|---|---|---|---|---|---|---|"]
        for (b, k), g in u.groupby(["band", "kind"]):
            L.append("| %s | %s | %d | %.1f%% | %.1f%% | %.0f | %.3f |" % (
                b, k, len(g), 100 * (g.n_held >= 16).mean(), 100 * (g.mid_held >= 16).mean(), g.n_mid.median(),
                (g.ho_peak_med / g.tr_peak_med).median()))
    vf = []
    for p in sorted(OUT_RB.glob("verify.shard*.jsonl")):
        vf += [json.loads(x) for x in open(p) if x.strip()]
    if vf:
        ok = sum(bool(r.get("ok")) for r in vf)
        L += ["", "## Verification (independent re-check)", "",
              "%d records: %d ok, %d failed, %d missing" % (len(vf), ok, sum(r.get("ok") is False for r in vf),
                                                         sum(bool(r.get("missing")) for r in vf))]
    (OUT_RB / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    {"build": build, "verify": verify, "report": report}[MODE]()
