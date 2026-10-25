"""HELD-OUT AUDIT (085): what is wrong with the 15k run's held-out contexts, measured on the stored records.

Ground truth = experiments/062-h100-protocol-v1/out_full/out/ctx/<key>.pt (what the run fitted and scored on; the
sweep, random nulls and every companion read the same records). Read-only. No GPU, no model.

Sets per record (indices into each pool; see README for the rules the code implements):
  S      strongest pool (top store, <= 64)      S_tr = strongest-train (48: the scorer's A-mean / pin reference = REF)
                                                S_ho = held-out strongest (16, PRIMARY)
                                                D    = the fit's strongest (32 of S_tr, arm B)   U = S_tr \\ D (unused)
  M      mid-band pool (mid store, <= 64)       M_tr, M_ho (16, reported), Bm = the fit's mid-band (16 of M_tr)
  C      contrast (close, verified silent)      C_tr = first n - round(n/4) (48), C_ho = the rest (16)
  FIT    = D + Bm (arm B) or S_tr (arm A, thin)
  TRAIN-SIDE = S_tr + M_tr + C_tr  (everything on the training side of any split; includes FIT and REF)

Per held-out row (S_ho, M_ho, C_ho) against every other row of the record:
  exact   token-identical 64-token window
  apfx    anchor-prefix identical: tokens[0 .. anchor] equal at the same positions. The model is causal and every
          read is at the anchor, so for the scored quantity this IS the same input even if later tokens differ
  lcr     longest common contiguous token run, any alignment (boilerplate shifted inside the window)
  c8      containment: share of the held row's 8-grams that occur in the other row
  near    = lcr >= 32 or c8 >= 0.5 (the near-duplicate rule; thresholds also tabulated at 16/24/48)
  src     same corpus source-file index (the shard segment's first element) as the other row, different id
plus id integrity (corpus content hash of the stored id == hash of the stored tokens), token sanity, anchors and
peaks, and the clean held-out counts under nested removal rules R0..R3 (README).

  PYTHONPATH=src .venv/bin/python experiments/085-heldout-audit/audit.py          (all records)
  env: SUBSET=1 (10 per layer x kind, seed 0)  WORKERS (12)  LIMIT (first N files, smoke)
"""
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
CTX = REPO / "experiments" / "062-h100-protocol-v1" / "out_full" / "out" / "ctx"
OUT = HERE / "out"
SUBSET = os.environ.get("SUBSET") == "1"
WORKERS = int(os.environ.get("WORKERS", "12"))
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
NEAR_LCR, NEAR_C8 = 32, 0.5
VOCAB = 32064                                   # Phi-3 tokenizer (TuringLLM); 32000 = <|endoftext|>
EOT = 32000

_W = np.random.default_rng(0).integers(1, 2 ** 62, size=64, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
CORPUS = {}


def load_corpus():
    for k in ("src", "hash", "mult"):
        p = OUT / ("corpus_%s.npy" % k)
        CORPUS[k] = np.load(p, mmap_mode="r") if p.exists() else None


def band(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def chash(rows):
    with np.errstate(over="ignore"):
        return (rows.astype(np.uint64) * _W[None, :]).sum(axis=1, dtype=np.uint64)


def pair_stats(H, O):
    """H [h,64], O [r,64] int64 (torch). Returns (exact, pfx, lcr, c8), each [h, r] numpy."""
    import torch
    eq_al = H[:, None, :] == O[None, :, :]                                  # aligned equality [h,r,64]
    pfx = eq_al.int().cumprod(-1).sum(-1)
    exact = pfx == 64
    eq = H[:, None, :, None] == O[None, :, None, :]                         # [h,r,64,64]
    L = torch.zeros(H.shape[0], O.shape[0], 64, dtype=torch.int16)
    best = torch.zeros(H.shape[0], O.shape[0], dtype=torch.int16)
    for i in range(64):
        prev = torch.zeros_like(L)
        prev[..., 1:] = L[..., :-1]
        L = (prev + 1) * eq[:, :, i, :]
        best = torch.maximum(best, L.max(-1).values)
    w = torch.tensor([1000003 ** k % (2 ** 61 - 1) for k in range(8)], dtype=torch.int64)
    def grams(T):
        return sum(T[:, k:k + 57] * w[k] for k in range(8))                # [n,57]
    hg, og = grams(H), grams(O)
    c8 = (hg[:, None, :, None] == og[None, :, None, :]).any(-1).float().mean(-1)
    return exact.numpy(), pfx.numpy(), best.numpy().astype(np.int32), c8.numpy()


def audit_one(path):
    import torch
    torch.set_num_threads(1)
    key, rec = next(iter(torch.load(path, weights_only=False).items()))
    l, kind, i = key.split(".")
    row = dict(key=key, layer=int(l), kind=kind, band=band(int(l)), n_top=rec.get("n_top"), n_mid=rec.get("n_mid"))
    ex = []
    s, m = rec.get("strong"), rec.get("mid")
    if s is None:
        row["skip"] = "no_contrast" if rec.get("no_contrast") else "too_few_contexts"
        return row, ex
    arm = "B" if m is not None else "A"
    row.update(arm=arm, thin=arm == "A")
    S_tr, S_ho = [int(x) for x in s["train"]], [int(x) for x in s["held"]]
    D = [int(x) for x in rec["D_train"]]
    U = [x for x in S_tr if x not in set(D)]
    nS = int(s["pos"].shape[0])
    if m is not None:
        M_tr, M_ho, Bm = [int(x) for x in m["train"]], [int(x) for x in m["held"]], [int(x) for x in rec["B_mid"]]
        nM = int(m["pos"].shape[0])
    else:
        M_tr, M_ho, Bm, nM = [], [], [], 0
    neg = rec["neg"]
    nC = int(neg.shape[0])
    n_ctr = nC - int(round(nC * 0.25))
    C_tr, C_ho = list(range(n_ctr)), list(range(n_ctr, nC))
    FIT_S = D if arm == "B" else S_tr
    row.update(n_S=nS, n_S_tr=len(S_tr), n_S_ho=len(S_ho), n_D=len(D), n_U=len(U), n_fit_S=len(FIT_S),
               n_M=nM, n_M_tr=len(M_tr), n_M_ho=len(M_ho), n_Bm=len(Bm), n_fit=len(FIT_S) + len(Bm),
               n_C=nC, n_C_tr=len(C_tr), n_C_ho=len(C_ho), neg_ids_present=rec.get("neg_ids") is not None)

    # ---- rows of the record, one global table: pool label, index, roles
    Sp, Mp = s["pos"].long(), (m["pos"].long() if m is not None else torch.zeros(0, 64, dtype=torch.long))
    ALL = torch.cat([Sp, Mp, neg.long()], 0)
    g_S = list(range(nS)); g_M = [nS + j for j in range(nM)]; g_C = [nS + nM + j for j in range(nC)]
    gs = lambda idx: [g_S[j] for j in idx]
    gm = lambda idx: [g_M[j] for j in idx]
    gc = lambda idx: [g_C[j] for j in idx]
    R = int(ALL.shape[0])
    role = {name: np.zeros(R, bool) for name in ("FIT", "REF", "TRAIN", "C", "ACT", "S_ho", "M_ho", "C_ho", "C_tr",
                                                  "U", "Mrest")}
    role["FIT"][gs(FIT_S) + gm(Bm)] = True
    role["REF"][gs(S_tr)] = True
    role["TRAIN"][gs(S_tr) + gm(M_tr) + gc(C_tr)] = True
    role["C"][g_C] = True
    role["C_tr"][gc(C_tr)] = True
    role["ACT"][g_S + g_M] = True
    role["S_ho"][gs(S_ho)] = True
    role["M_ho"][gm(M_ho)] = True
    role["C_ho"][gc(C_ho)] = True
    role["U"][gs(U)] = True
    role["Mrest"][gm([j for j in M_tr if j not in set(Bm)])] = True

    # ---- integrity / token sanity
    allnp = ALL.numpy()
    row["tok_neg"] = int((allnp < 0).sum()); row["tok_oov"] = int((allnp >= VOCAB).sum())
    row["tok_unk0_rows"] = int((allnp == 0).any(1).sum())
    row["eot_rows_S_ho"] = int((Sp[S_ho] == EOT).any(1).sum()) if S_ho else 0
    row["eot_rows_all"] = int((allnp == EOT).any(1).sum())
    for name, d in (("S", s), ("M", m)):
        if d is None:
            continue
        tg = d["tgt"].long()
        row["tgt_shift_ok_%s" % name] = bool((tg[:, :63] == d["pos"][:, 1:].long()).all())
        row["tgt_lastcol_zero_%s" % name] = bool((tg[:, 63] == 0).all())
        pk = d["peak"].float()
        row["peak_nonfinite_%s" % name] = int((~torch.isfinite(pk)).sum())
        row["peak_le0_%s" % name] = int((pk <= 0).sum())
        row["ids_len_ok_%s" % name] = len(d["ids"]) == int(d["pos"].shape[0])
        row["ids_unique_%s" % name] = len(set(d["ids"])) == len(d["ids"])
    ids_S = list(s["ids"]); ids_M = list(m["ids"]) if m is not None else []
    ids_C = list(rec["neg_ids"]) if rec.get("neg_ids") is not None else None
    if CORPUS.get("hash") is not None:
        for name, ids, tok in (("S", ids_S, Sp), ("M", ids_M, Mp), ("C", ids_C, neg.long())):
            if ids is None or len(ids) == 0:
                continue
            ch = CORPUS["hash"][np.asarray(ids, dtype=np.int64)]
            row["id_hash_mismatch_%s" % name] = int((ch != chash(tok.numpy())).sum())
    all_ids = ids_S + ids_M + (ids_C or [])
    row["id_overlap_act_ctr"] = len(set(ids_S + ids_M) & set(ids_C or []))
    row["id_overlap_S_M"] = len(set(ids_S) & set(ids_M))

    # ---- exact identity between pools (hash of tokens)
    h_all = chash(allnp)
    hS, hM, hC = set(h_all[g_S].tolist()), set(h_all[g_M].tolist()), set(h_all[g_C].tolist())
    row["exact_ctr_eq_act_rows"] = int(sum(h in (hS | hM) for h in h_all[g_C].tolist()))   # contradiction
    row["exact_M_eq_S_rows"] = int(sum(h in hS for h in h_all[g_M].tolist()))
    row["dup_rows_in_S_pool"] = nS - len(hS); row["dup_rows_in_M_pool"] = nM - len(hM)
    row["dup_rows_in_C"] = nC - len(hC)
    row["n_unique_S_pool"] = len(hS)

    # ---- anchors and peaks
    pkS, aS = s["peak"].float().numpy(), s["arg"].numpy()
    row.update(peak_le0_S_ho=int((pkS[S_ho] <= 0).sum()), anchor0_S_ho=int((aS[S_ho] == 0).sum()), anchor_le3_S_ho=int((aS[S_ho] <= 3).sum()),
               anchor0_S_tr=int((aS[S_tr] == 0).sum()), anchor_med_S_ho=float(np.median(aS[S_ho])),
               anchor_med_S_tr=float(np.median(aS[S_tr])))
    row.update(peak_med_S_ho=float(np.median(pkS[S_ho])), peak_med_S_tr=float(np.median(pkS[S_tr])),
               peak_med_FIT_S=float(np.median(pkS[FIT_S])), peak_max_S=float(pkS.max()), peak_min_S=float(pkS.min()),
               peak_ties_S=int(nS - len(np.unique(pkS))))
    if m is not None:
        pkM, aM = m["peak"].float().numpy(), m["arg"].numpy()
        row.update(peak_med_M_ho=float(np.median(pkM[M_ho])), peak_med_M_tr=float(np.median(pkM[M_tr])),
                   anchor0_M_ho=int((aM[M_ho] == 0).sum()), mid_peak_over_S=float(np.median(pkM) / max(np.median(pkS), 1e-9)),
                   mid_ge_Smin=int((pkM > pkS.min()).sum()))
    anchors = np.full(R, -1, dtype=np.int64)
    anchors[g_S] = aS
    if m is not None:
        anchors[g_M] = m["arg"].numpy()
    peaks = np.full(R, np.nan); peaks[g_S] = pkS
    if m is not None:
        peaks[g_M] = m["peak"].float().numpy()

    # ---- pairwise: held rows (S_ho, M_ho, C_ho) x all rows
    Hg = gs(S_ho) + gm(M_ho) + gc(C_ho)
    exact, pfx, lcr, c8 = pair_stats(ALL[Hg], ALL)
    for a, gidx in enumerate(Hg):
        exact[a, gidx] = False; pfx[a, gidx] = 0; lcr[a, gidx] = 0; c8[a, gidx] = 0.0         # self
    src = CORPUS.get("src")
    if src is not None:
        gid = np.zeros(R, np.int64)
        gid[g_S] = ids_S
        if ids_M:
            gid[g_M] = ids_M
        if ids_C is not None:
            gid[g_C] = ids_C
        srcv = np.where(gid > 0, np.asarray(src)[gid], -1)
        same_src = (srcv[Hg][:, None] == srcv[None, :]) & (srcv[None, :] >= 0) & (gid[Hg][:, None] != gid[None, :])
        mult = np.asarray(CORPUS["mult"])[gid[Hg]] if CORPUS.get("mult") is not None else None
    else:
        same_src, mult = np.zeros_like(exact), None
    apfx = np.zeros_like(exact)
    apfx_sa = np.zeros_like(exact)              # same prefix AND the other row is anchored at the same position
    for a, gidx in enumerate(Hg):
        if anchors[gidx] >= 0:
            apfx[a] = pfx[a] >= anchors[gidx] + 1
            apfx_sa[a] = apfx[a] & (anchors == anchors[gidx])
    if src is not None:
        for name, rows_ in (("S", g_S), ("TRAIN", list(np.flatnonzero(role["TRAIN"]))), ("FIT", list(np.flatnonzero(role["FIT"])))):
            v = srcv[rows_]
            row["n_src_%s" % name] = int(len(set(v[v >= 0].tolist())))

    def held_block(prefix, rows_g, mask_name_groups):
        """Per held set: counts of rows that hit each group under each criterion + the R0..R3 clean counts."""
        sel = [Hg.index(g) for g in rows_g]
        if not sel:
            return
        crit = dict(exact=exact, apfx=apfx, apfxsa=apfx_sa, near=(lcr >= NEAR_LCR) | (c8 >= NEAR_C8), lcr16=lcr >= 16,
                    lcr24=lcr >= 24, lcr48=lcr >= 48, c8_50=c8 >= 0.5, src=same_src)
        for gname, gmask in mask_name_groups.items():
            for cname, cm in crit.items():
                row["%s_%s_%s" % (prefix, cname, gname)] = int(cm[sel][:, gmask].any(1).sum())
        row["%s_maxlcr_TRAIN_med" % prefix] = float(np.median(lcr[sel][:, role["TRAIN"]].max(1)))
        # within-held duplicates (a row equal / near to an EARLIER held row of the same set)
        earlier = np.zeros((len(sel), R), bool)
        for t, a in enumerate(sel):
            earlier[t, rows_g[:t]] = True
        for cname, cm in (("exact", exact), ("apfx", apfx), ("near", (lcr >= NEAR_LCR) | (c8 >= NEAR_C8))):
            row["%s_%s_within" % (prefix, cname)] = int((cm[sel] & earlier).any(1).sum())
        # nested clean rules against TRAIN-SIDE + all contrast (activating held) and earlier kept held rows
        bad_g = role["TRAIN"] | role["C"]
        rules = {"R0": exact, "R1": exact | apfx, "R2": exact | apfx | (lcr >= NEAR_LCR) | (c8 >= NEAR_C8)}
        rules["R3"] = rules["R2"] | same_src
        rules["R2sa"] = exact | apfx_sa | (lcr >= NEAR_LCR) | (c8 >= NEAR_C8)     # anchor-prefix only at the same anchor
        for rname, cm in rules.items():
            kept = []
            for t, a in enumerate(sel):
                if cm[a][bad_g].any():
                    continue
                if kept and cm[a][[rows_g[k] for k in kept]].any():
                    continue
                kept.append(t)
            row["%s_clean_%s" % (prefix, rname)] = len(kept)
            if rname == "R2" and kept:
                row["%s_clean_R2_peak_med" % prefix] = float(np.nanmedian(peaks[[rows_g[k] for k in kept]]))
        if mult is not None:
            mu = mult[sel]
            row["%s_corpus_mult_gt1" % prefix] = int((mu > 1).sum())
            row["%s_corpus_mult_ge10" % prefix] = int((mu >= 10).sum())
            row["%s_corpus_mult_max" % prefix] = int(mu.max())
        # worst example rows for the examples file
        for t, a in enumerate(sel):
            gidx = rows_g[t]
            hit = bad_g & ((exact[a]) | apfx[a] | (lcr[a] >= NEAR_LCR))
            if hit.any():
                j = int(np.argmax(np.where(hit, lcr[a] + 100 * exact[a] + 50 * apfx[a], -1)))
                ex.append(dict(key=key, set=prefix, row=int(gidx), other=int(j),
                               other_role=[r for r in ("FIT", "REF", "TRAIN", "C") if role[r][j]],
                               exact=bool(exact[a, j]), apfx=bool(apfx[a, j]), lcr=int(lcr[a, j]),
                               c8=round(float(c8[a, j]), 3), anchor=int(anchors[gidx]), pfx=int(pfx[a, j]),
                               peak=None if np.isnan(peaks[gidx]) else round(float(peaks[gidx]), 4)))

    groups_act = dict(FIT=role["FIT"], REF=role["REF"], TRAIN=role["TRAIN"], C=role["C"])
    held_block("Sho", gs(S_ho), dict(groups_act, Mho=role["M_ho"]))
    if M_ho:
        held_block("Mho", gm(M_ho), dict(groups_act, Sho=role["S_ho"]))
    # contrast held: vs contrast train and vs any activating row (contradiction / near)
    selC = [Hg.index(g) for g in gc(C_ho)]
    if selC:
        nearm = (lcr >= NEAR_LCR) | (c8 >= NEAR_C8)
        for gname, gmask in (("Ctr", role["C_tr"]), ("ACT", role["ACT"])):
            for cname, cm in (("exact", exact), ("near", nearm), ("src", same_src)):
                row["Cho_%s_%s" % (cname, gname)] = int(cm[selC][:, gmask].any(1).sum())

    # ---- top-up candidates already inside the record (no new corpus scan):
    #   strongest rows the fit never saw (S_ho + U) that are R2-clean vs FIT + all contrast, deduped among themselves
    #   mid rows the fit never saw (M_ho + M_tr \\ Bm), same rule
    cand = {"S": gs(S_ho) + gs(U), "M": gm(M_ho) + gm([j for j in M_tr if j not in set(Bm)])}
    for name, rows_g in cand.items():
        if not rows_g:
            row["cand_%s_R2" % name] = 0
            continue
        Hc = ALL[rows_g]
        e2, p2, l2, c2 = pair_stats(Hc, ALL)
        bad = (e2 | (l2 >= NEAR_LCR) | (c2 >= NEAR_C8))
        ap2 = np.zeros_like(e2)
        for t, gidx in enumerate(rows_g):
            ap2[t] = p2[t] >= anchors[gidx] + 1
            bad[t, gidx] = False; ap2[t, gidx] = False
        bad = bad | ap2
        forbid = role["FIT"] | role["C"]
        kept = []
        for t, gidx in enumerate(rows_g):
            if bad[t][forbid].any() or (kept and bad[t][[rows_g[k] for k in kept]].any()):
                continue
            kept.append(t)
        row["cand_%s_R2" % name] = len(kept)
        if kept:
            row["cand_%s_R2_peak_med" % name] = float(np.nanmedian(peaks[[rows_g[k] for k in kept]]))
    return row, ex


def pick_subset(files):
    cells = defaultdict(list)
    for f in files:
        k = Path(f).stem
        l, kind, _ = k.split(".")
        cells[(int(l), kind)].append(f)
    rng = random.Random(0)
    return sorted(x for c in sorted(cells) for x in rng.sample(sorted(cells[c]), min(10, len(cells[c]))))


def _init():
    load_corpus()


def main():
    load_corpus()
    files = sorted(glob.glob(str(CTX / "*.pt")))
    if SUBSET:
        files = pick_subset(files)
    if LIMIT:
        files = files[:LIMIT]
    tag = "subset" if SUBSET else ("limit%d" % LIMIT if LIMIT else "all")
    print("audit %d records (%s), corpus index %s" % (len(files), tag, "yes" if CORPUS["hash"] is not None else "NO"),
          flush=True)
    t0 = time.time()
    rows, exs = [], []
    with Pool(WORKERS, initializer=_init) as pool:
        for n, (r, e) in enumerate(pool.imap_unordered(audit_one, files, chunksize=4)):
            rows.append(r); exs += e
            if n % 1000 == 0:
                print("%d/%d %.0fs" % (n, len(files), time.time() - t0), flush=True)
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("key")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / ("audit_%s.csv" % tag), index=False)
    with open(OUT / ("examples_%s.jsonl" % tag), "w") as fh:
        for e in exs:
            fh.write(json.dumps(e) + "\n")
    print("wrote %d rows, %d example hits, %.0fs" % (len(df), len(exs), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
