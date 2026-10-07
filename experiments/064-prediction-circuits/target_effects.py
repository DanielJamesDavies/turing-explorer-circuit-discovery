"""Per-target direct output effect: which of the 15,004 circuit targets look like next-token predictors?

For every circuit target, push its SAE decoder direction d (unit norm) through the final RMSNorm and the
unembedding: l = W_U (g * d) / rms_typ over the real vocabulary (ids < 32,064), centred by its vocab mean.
rms_typ = median final-residual rms over 64 random corpus sequences (positions >= 1). This is the DIRECT
PATH only: attn/mlp directions add into the residual at their layer and resid directions are the residual,
but every later block can transform them (quantified causally in inference_dla.py).

Per target: top-20 promoted / suppressed tokens; concentration (top-1 z-score, excess kurtosis);
unembedding gain (std of l relative to random unit directions); magnitude (max centred logit x mean active
activation, and x the explorer's target peak); agreement with logit_ctx (empirical next tokens when the
latent is active at the last position of corpus sequences; top-probability first, deduplicated).
A null of 1,024 random live NON-target latents per site gets the same metrics.

  PYTHONPATH=src python experiments/064-prediction-circuits/target_effects.py
Writes results/{rms_typ.json, target_effects.parquet, null_effects.parquet, target_topk.npz}.
"""
import json
import time

import numpy as np
import pandas as pd
import torch

import common as C

TOPN = 20
N_NULL_PER_SITE = 1024
N_RMS_SEQ = 64
CHUNK = 1024


def measure_rms(model, eps):
    toks = np.load(C.BUNDLE / "tokens" / "tokens.npy", mmap_mode="r")
    rng = np.random.default_rng(64)
    seqs = rng.choice(toks.shape[0], N_RMS_SEQ, replace=False)
    rs = []
    for s in seqs:
        ids = [int(t) for t in toks[s]]
        _, acts = C.forward_capture(model, ids)
        rs.append(C.rms_of(acts[11, 2], eps)[1:])
    r = torch.cat(rs).numpy()
    out = {"rms_typ": float(np.median(r)), "q10": float(np.quantile(r, 0.1)), "q90": float(np.quantile(r, 0.9)),
           "n_positions": int(r.size), "n_seq": N_RMS_SEQ}
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    C.RMS_TYP_FILE.write_text(json.dumps(out, indent=1))
    return out


def effects(dirs, g, W_Ur, rms_typ, rand_std):
    """dirs [n, d] -> dict of metrics + top-k ids/vals (centred logits per unit activation)."""
    rec = {k: [] for k in ("z1", "kurt", "gain", "lmax", "lmin", "top_ids", "top_vals", "bot_ids", "bot_vals")}
    for a in range(0, len(dirs), CHUNK):
        L = (dirs[a:a + CHUNK] * g) @ W_Ur.T / rms_typ          # [c, V]
        L = L - L.mean(1, keepdim=True)
        sd = L.std(1)
        tv, ti = L.topk(TOPN, dim=1)
        bv, bi = (-L).topk(TOPN, dim=1)
        rec["z1"].append(tv[:, 0] / sd)
        rec["kurt"].append((L / sd[:, None]).pow(4).mean(1) - 3)
        rec["gain"].append(sd / rand_std)
        rec["lmax"].append(tv[:, 0])
        rec["lmin"].append(-bv[:, 0])
        rec["top_ids"].append(ti); rec["top_vals"].append(tv)
        rec["bot_ids"].append(bi); rec["bot_vals"].append(-bv)
    return {k: torch.cat(v).numpy() for k, v in rec.items()}


def ctx_agreement(top_ids, L_z_fn, gids, lc_tok, lc_prob, lc_cnt, pair=None):
    """overlap20: |direct top-20 ∩ empirical top-10 distinct|; ctx_z: mean z-score of the direct logit over the
    empirical top-10 distinct next tokens; hit1: direct top-1 token among the empirical top-32.
    pair: optional permutation - use latent pair[j]'s empirical tokens against latent j's direct logits (null)."""
    n = len(gids)
    ov, cz, hit, has = np.zeros(n), np.full(n, np.nan), np.zeros(n, bool), np.zeros(n, bool)
    emp = []
    for g in gids:
        emp.append([t for t, _ in C.distinct_next_tokens(lc_tok, lc_prob, g, 32) if t < C.N_REAL_VOCAB])
    for j in range(n):
        e = emp[pair[j]] if pair is not None else emp[j]
        if not e:
            continue
        has[j] = True
        e10 = e[:10]
        ov[j] = len(set(top_ids[j].tolist()) & set(e10))
        hit[j] = int(top_ids[j, 0]) in set(e)
        cz[j] = float(np.mean(L_z_fn(j, e10)))
    return ov, cz, hit, has


def main():
    t0 = time.time()
    model = C.load_model()
    W_U, g, eps = C.unembed(model)
    W_Ur = W_U[:C.N_REAL_VOCAB]
    rms = measure_rms(model, eps)
    print("rms_typ", rms, f"{time.time() - t0:.0f}s")
    bank = C.load_bank()
    con = C.connect()
    circ = C.load_circuits(con)
    lat = pd.read_sql_query("SELECT gid, mean, active_count FROM latent", con).set_index("gid")

    # random unit-direction baseline for the unembedding gain
    torch.manual_seed(0)
    R = torch.randn(1000, 1024)
    R = R / R.norm(dim=1, keepdim=True)
    LR = (R * g) @ W_Ur.T / rms["rms_typ"]
    rand_std = float((LR - LR.mean(1, keepdim=True)).std(1).median())

    lc_tok, lc_prob, lc_cnt = C.load_logit_ctx()

    def run(gids, name):
        dirs = C.decoder_dirs(bank, gids)
        E = effects(dirs, g, W_Ur, rms["rms_typ"], rand_std)
        # z-scored direct logit of arbitrary tokens, recomputed per latent on demand (cheap: 1 x V)
        cache = {}

        def lz(j, toks):
            if j not in cache:
                cache.clear()
                L = (dirs[j] * g) @ W_Ur.T / rms["rms_typ"]
                L = L - L.mean()
                cache[j] = (L / L.std()).numpy()
            return cache[j][toks]
        ov, cz, hit, has = ctx_agreement(E["top_ids"], lz, gids, lc_tok, lc_prob, lc_cnt)
        # null pairing: shuffle within site
        comp = np.asarray(gids) // C.N_LAT
        perm = np.arange(len(gids))
        rng = np.random.default_rng(1)
        for c in np.unique(comp):
            idx = np.flatnonzero(comp == c)
            perm[idx] = rng.permutation(idx)
        ov0, cz0, hit0, _ = ctx_agreement(E["top_ids"], lz, gids, lc_tok, lc_prob, lc_cnt, pair=perm)
        l, k, i = zip(*[C.split_gid(x) for x in gids])
        df = pd.DataFrame({"gid": gids, "layer": l, "kind": k, "latent": i,
                           "z1": E["z1"], "kurt": E["kurt"], "gain": E["gain"], "lmax": E["lmax"], "lmin": E["lmin"],
                           "mean_act": lat["mean"].reindex(gids).to_numpy(),
                           "lc_count": lc_cnt.reshape(-1)[np.asarray(gids)],
                           "ctx_overlap20": ov, "ctx_z": cz, "ctx_hit1": hit, "has_ctx": has,
                           "ctx_overlap20_shuf": ov0, "ctx_z_shuf": cz0, "ctx_hit1_shuf": hit0})
        df["boost_mean"] = df.mean_act * df.lmax
        print(f"{name}: {len(df)} latents, {time.time() - t0:.0f}s")
        return df, E

    tdf, TE = run(circ.seed_gid.to_numpy(), "targets")
    tdf = tdf.merge(circ[["seed_gid", "cid", "key", "pass", "peak", "near_threshold", "amp_any", "n_nodes",
                          "free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk"]],
                    left_on="gid", right_on="seed_gid").drop(columns="seed_gid")
    tdf["boost_peak"] = tdf.peak * tdf.lmax
    tdf.to_parquet(C.RESULTS / "target_effects.parquet")
    np.savez_compressed(C.RESULTS / "target_topk.npz", gid=circ.seed_gid.to_numpy(),
                        top_ids=TE["top_ids"].astype(np.int32), top_vals=TE["top_vals"].astype(np.float32),
                        bot_ids=TE["bot_ids"].astype(np.int32), bot_vals=TE["bot_vals"].astype(np.float32))

    # null: random live non-targets per site
    is_t = np.zeros(36 * C.N_LAT, bool); is_t[circ.seed_gid.to_numpy()] = True
    live = np.zeros(36 * C.N_LAT, bool)
    live[lat.index.to_numpy()[lat.active_count.to_numpy() > 0]] = True
    rng = np.random.default_rng(2)
    null = []
    for c in range(36):
        cand = np.arange(c * C.N_LAT, (c + 1) * C.N_LAT)
        cand = cand[live[cand] & ~is_t[cand]]
        null.append(rng.choice(cand, N_NULL_PER_SITE, replace=False))
    ndf, _ = run(np.concatenate(null), "null")
    ndf.to_parquet(C.RESULTS / "null_effects.parquet")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
