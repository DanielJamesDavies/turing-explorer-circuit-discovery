"""At inference, which fired circuits drive the prediction? DLA ranking vs the explorer score + causal check.

Sequences: 18 hand-written prompts (BOS + prompt, < 64 tokens) and 16 random 64-token corpus sequences
(bundle tokens.npy, fed as stored). At every non-BOS position:
  - the model's top-5 next tokens;
  - every fired circuit (target active, post-TopK > 0) with the explorer's score (target activation / target
    peak x member support) and its direct logit attribution (DLA) to log p(top-1) and to log p(true next):
    a x (u_tok - E_p[u]) . J d, J = the exact final-RMSNorm Jacobian at that position (common.logprob_dla);
  - the same DLA for EVERY active latent (36 sites x 128), to ask whether the latents that drive the prediction
    are circuit targets at all.
Causal check on 48 sampled positions (>= 12 fired circuits): single-target ablations of
  top-3 by DLA to top-1 | top-3 by explorer score | top-3 by activation (largest write, |a| since ||d|| = 1)
  | 3 random fired (outside the DLA and score top-3)
Ablation = zero the latent in its SAE reconstruction at that position only and splice back with the SAE error
term kept: out = decode(z with z_i = 0) + (x - decode(z)) = x - a d (decode is affine; checked numerically
against the explicit encode / decode path on the first case). Measured at that position: change of the top-1
logit and log-prob and KL(base || ablated). Plus the three set-joint ablations.

  PYTHONPATH=src python experiments/064-prediction-circuits/inference_dla.py
Writes results/{dla_positions,dla_fired,causal}.parquet and results/inference_summary.json.
"""
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import common as C
from model.hooks import multi_patch

PROMPTS = [
    "The capital of France is Paris, and the capital of Germany is",
    "When Mary and John went to the store, John gave a drink to",
    "1, 2, 3, 4, 5, 6, 7,",
    "The acid reacted with the base to form a salt and",
    "def add(a, b):\n    return",
    "She opened the door, looked outside, and saw that it was raining. She grabbed her",
    "The Eiffel Tower is located in the city of",
    "In 1969, Neil Armstrong became the first person to walk on the",
    "The opposite of hot is",
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose and",
    "He was not happy; in fact, he was very",
    "The war ended in 1945, and the soldiers returned home to their",
    "Dear Sir or Madam, I am writing to",
    "The quick brown fox jumps over the lazy",
    "Water boils at 100 degrees Celsius and freezes at",
    "To be or not to be, that is the",
    "The mitochondria is the powerhouse of the",
    "Once upon a time, there was a little girl who lived in a small",
]
N_CORPUS = 16
N_CAUSAL = 48
MIN_FIRED = 12


def sequences(tok):
    seqs = [("prompt", i, [C.BOS] + tok.encode(p)) for i, p in enumerate(PROMPTS)]
    toks = np.load(C.BUNDLE / "tokens" / "tokens.npy", mmap_mode="r")
    rng = np.random.default_rng(640)
    for j, s in enumerate(rng.choice(toks.shape[0], N_CORPUS, replace=False)):
        seqs.append(("corpus", int(s) + 1, [int(t) for t in toks[s]]))   # seq_id is 1-based
    return seqs


def all_active_dla(bank, vals, idx, t, x, probs, W_U, g, eps, tok):
    """DLA to `tok` of every active latent at position t -> (gids, dla)."""
    gids, dirs, acts = [], [], []
    for c in range(vals.shape[0]):
        l, k = divmod(c, 3)
        v = vals[c, t]
        keep = v > 0
        ii = idx[c, t][keep]
        W = bank.saes[C.KINDS[k]][l].decoder.weight
        dirs.append(W[:, ii].T.detach())
        acts.append(v[keep])
        gids.append(ii.numpy() + c * C.N_LAT)
    with torch.no_grad():
        dla = C.logprob_dla(x, probs, W_U, g, eps, tok, torch.cat(dirs), torch.cat(acts))
    return np.concatenate(gids), dla.numpy()


def ablate_logits(model, ids, t, items):
    """items: list of (layer, kind, act, dir) -> logits at t with each latent removed at position t."""
    by_site = {}
    for l, k, a, d in items:
        by_site.setdefault((l, k), []).append(a * d)

    def tf(layer, kind, x):
        ds = by_site.get((layer, kind))
        if ds is None:
            return None
        out = x.clone()
        out[0, t] = out[0, t] - sum(ds)
        return out

    with torch.no_grad(), multi_patch(model, tf):
        lg, _ = model(torch.tensor([ids]), return_all_logits=True)
    return lg[0, t].float()


def check_splice(model, bank, ids, t, l, k, gid):
    """Explicit SAE splice (encode, zero, decode + error) vs x - a d: max |logit diff| at t."""
    from sae.dense import sparse_topk_to_dense
    lat = C.split_gid(gid)[2]

    def tf(layer, kind, x):
        if (layer, kind) != (l, k):
            return None
        a, i = bank.encode(x, kind, layer)
        z = sparse_topk_to_dense(a, i, bank.d_sae, dtype=x.dtype)
        err = x - bank.decode(z, kind, layer)
        z2 = z.clone()
        z2[0, t, lat] = 0
        patched = bank.decode(z2, kind, layer) + err
        out = x.clone()
        out[0, t] = patched[0, t]
        return out

    with torch.no_grad(), multi_patch(model, tf):
        lg, _ = model(torch.tensor([ids]), return_all_logits=True)
    return lg[0, t].float()


def main():
    t0 = time.time()
    model = C.load_model()
    tok = C.load_tokenizer()
    bank = C.load_bank()
    W_U, g, eps = C.unembed(model)
    con = C.connect()
    circ = C.load_circuits(con)
    tt = C.TargetTable(circ)
    mi = C.MemberIndex(con, circ)
    D_all = C.decoder_dirs(bank, circ.seed_gid.to_numpy())
    print(f"loaded {time.time() - t0:.0f}s")

    pos_rows, fired_rows, cache = [], [], {}
    for si, (src, sid, ids) in enumerate(sequences(tok)):
        logits, acts = C.forward_capture(model, ids)
        vals, idx = C.encode_all(bank, acts)
        probs = F.softmax(logits, -1)
        xf = acts[11, 2]
        cache[si] = (ids, logits, vals, idx)
        for t in range(len(ids)):
            if ids[t] == C.BOS:
                continue
            p = probs[t]
            top5p, top5 = p.topk(5)
            top1 = int(top5[0])
            true_next = ids[t + 1] if t + 1 < len(ids) else -1
            gids, avals = C.active_at(vals, idx, t)
            rows, tg, ta, sup, score = C.fired_circuits(tt, mi, gids, avals)
            dirs = D_all[torch.as_tensor(rows)] if len(rows) else torch.zeros(0, 1024)
            dla1 = C.logprob_dla(xf[t], p, W_U, g, eps, top1, dirs, ta).numpy() if len(rows) else np.zeros(0)
            dlan = (C.logprob_dla(xf[t], p, W_U, g, eps, true_next, dirs, ta).numpy()
                    if len(rows) and true_next >= 0 else np.full(len(rows), np.nan))
            ag, adla = all_active_dla(bank, vals, idx, t, xf[t], p, W_U, g, eps, top1)
            order = np.argsort(-adla)
            top10_all = ag[order[:10]]
            pos_rows.append({
                "seq": si, "src": src, "sid": sid, "t": t, "token": ids[t], "true_next": true_next,
                "top1": top1, "p_top1": float(top5p[0]), "top5": top5.tolist(), "top5_p": top5p.tolist(),
                "entropy": float(-(p * torch.log(p.clamp_min(1e-30))).sum()),
                "n_active": len(ag), "n_fired": len(rows),
                "best_all_dla": float(adla[order[0]]), "best_all_gid": int(ag[order[0]]),
                "best_all_is_target": bool(tt.is_target[ag[order[0]]]),
                "top10_all_target_frac": float(tt.is_target[top10_all].mean()),
                "active_target_frac": float(tt.is_target[ag].mean()),
                "best_target_rank_all": int(np.flatnonzero(tt.is_target[ag[order]])[0]) if tt.is_target[ag].any() else -1,
                "sum_pos_dla_all": float(adla[adla > 0].sum()),
                "sum_pos_dla_targets": float(dla1[dla1 > 0].sum()) if len(rows) else 0.0,
            })
            for r, gg, a, s, sc, d1, dn in zip(rows, tg, ta, sup, score, dla1, dlan):
                l, k, _ = C.split_gid(gg)
                fired_rows.append({"seq": si, "t": t, "row": int(r), "gid": int(gg), "layer": l, "kind": k,
                                   "act": float(a), "support": float(s), "score": float(sc),
                                   "dla_top1": float(d1), "dla_true": float(dn), "pass": bool(circ["pass"].iat[r])})
        print(f"seq {si} ({src}) T={len(ids)} {time.time() - t0:.0f}s")

    P = pd.DataFrame(pos_rows)
    Fd = pd.DataFrame(fired_rows)
    C.RESULTS.mkdir(exist_ok=True)
    P.to_parquet(C.RESULTS / "dla_positions.parquet")
    Fd.to_parquet(C.RESULTS / "dla_fired.parquet")

    # ------------------------------------------------------------------ causal check
    rng = np.random.default_rng(7)
    elig = P[P.n_fired >= MIN_FIRED]
    half = N_CAUSAL // 2
    samp = pd.concat([elig[elig.src == "prompt"].sample(half, random_state=7),
                      elig[elig.src == "corpus"].sample(half, random_state=8)])
    crow, checked = [], False
    for _, pr in samp.iterrows():
        ids, logits, vals, idx = cache[pr.seq]
        t = int(pr.t)
        base = logits[t]
        lp0 = F.log_softmax(base, -1)
        top1 = int(pr.top1)
        f = Fd[(Fd.seq == pr.seq) & (Fd.t == t)].reset_index(drop=True)
        sets = {"dla": f.nlargest(3, "dla_top1").index.tolist(),
                "score": f.nlargest(3, "score").index.tolist(),
                "act": f.nlargest(3, "act").index.tolist()}
        pool = [i for i in f.index if i not in set(sets["dla"]) | set(sets["score"])]
        sets["random"] = rng.choice(pool, 3, replace=False).tolist()
        for name, members in sets.items():
            items_all = []
            for rank, i in enumerate(members):
                r = f.loc[i]
                item = (int(r.layer), r.kind, float(r.act), D_all[int(r.row)])
                items_all.append(item)
                lg = ablate_logits(model, ids, t, [item])
                if not checked:
                    lg2 = check_splice(model, bank, ids, t, int(r.layer), r.kind, int(r.gid))
                    print("splice check: max |logit diff| =", float((lg - lg2).abs().max()))
                    checked = True
                lp = F.log_softmax(lg, -1)
                crow.append({"seq": pr.seq, "t": t, "src": pr.src, "set": name, "rank": rank, "joint": False,
                             "gid": int(r.gid), "layer": int(r.layer), "kind": r.kind, "act": float(r.act),
                             "score": float(r.score), "dla_top1": float(r.dla_top1), "pass": bool(r["pass"]),
                             "d_logit_top1": float(lg[top1] - base[top1]),
                             "d_logp_top1": float(lp[top1] - lp0[top1]),
                             "kl": float((lp0.exp() * (lp0 - lp)).sum()),
                             "top1_changed": bool(int(lg.argmax()) != top1)})
            lg = ablate_logits(model, ids, t, items_all)
            lp = F.log_softmax(lg, -1)
            crow.append({"seq": pr.seq, "t": t, "src": pr.src, "set": name, "rank": -1, "joint": True,
                         "dla_top1": float(f.loc[members, "dla_top1"].sum()),
                         "d_logit_top1": float(lg[top1] - base[top1]),
                         "d_logp_top1": float(lp[top1] - lp0[top1]),
                         "kl": float((lp0.exp() * (lp0 - lp)).sum()),
                         "top1_changed": bool(int(lg.argmax()) != top1)})
    Cz = pd.DataFrame(crow)
    Cz.to_parquet(C.RESULTS / "causal.parquet")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
