"""Is the 11.mlp.30743 residual relay ONE feature passed through the layers? Two checks plus a per-layer census.

1. Dictionary geometry. Decoder cosine between consecutive chain latents (R_l -> R_l'), the rank of R_l' among ALL
   latents of its dictionary by decoder cosine to R_l (1 = nearest neighbour), the read-write alignment
   cos(encoder row of R_l', decoder column of R_l), and a random baseline (median |cos| of random cross-dictionary
   pairs).
2. Behaviour on a RANDOM corpus sample (not the target's contexts). Per latent: positions fired (act > 0),
   precision on "iv" tokens (share of its firings on an "iv" token), recall (share of "iv" tokens it fires on).
   Per pair: Jaccard of firing positions and Pearson correlation of activations over all positions.
3. Census. For every layer's residual dictionary (and the target's MLP site), the latent with the best F1 as an
   "iv" detector on the corpus: does every layer have one, and did the circuit pick it?

  PYTHONPATH=src python experiments/057-wcm-edges/check_relay.py   -> results/check_relay.md (+ .json)
  env: N_SEQ (4096 random corpus sequences)  SEED (0)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent / "049-circuit-graph"
N_SEQ = int(os.environ.get("N_SEQ", 4096))
RNG = np.random.default_rng(int(os.environ.get("SEED", 0)))
TARGET = (11, "mlp", 30743)
CHAIN = [(2, "resid", 1164), (3, "resid", 21272), (4, "resid", 22454), (5, "resid", 38187), (6, "resid", 1448),
         (7, "resid", 14260), (9, "resid", 37889)]
CONTROL = [(10, "resid", 36122)]
BS = 64


def main():
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("TAG", "relay_unused")
    import amp_eval_pass_v2 as V
    from model.tokenizer import Tokenizer

    G = V.setup()
    inference, bank, M0, KINDS, D = (G[k] for k in ("inference", "bank", "M0", "KINDS", "D"))
    tok = Tokenizer(); k2i = {k: i for i, k in enumerate(KINDS)}
    sel = M0._neg_context_selector()
    loader = sel.loader

    # "iv" token ids: every vocabulary entry that decodes to "iv" (with or without surrounding whitespace)
    vocab = int(inference.model.lm_head.weight.shape[0])
    iv_ids = sorted(t for t in range(vocab) if tok.decode([t]).strip() == "iv")
    print("iv token ids:", iv_ids, [repr(tok.decode([t])) for t in iv_ids], flush=True)

    # random corpus sample, uniform over global sequence ids
    ranges = loader.shard_id_ranges
    sizes = np.array([e - s + 1 for s, e in ranges], dtype=np.float64)
    shards = RNG.choice(len(ranges), size=N_SEQ, p=sizes / sizes.sum())
    ids = sorted({int(ranges[s][0] + RNG.integers(0, sizes[s])) for s in shards})
    ids, tokens = sel.load_tokens(ids, max_length=64)
    print("corpus sample: %d sequences, %d positions" % (tokens.shape[0], tokens.numel()), flush=True)
    is_iv = torch.isin(tokens.cpu(), torch.tensor(iv_ids))

    tracked = CHAIN + CONTROL + [TARGET]
    census_sites = [(l, "resid") for l in range(12)] + [(TARGET[0], TARGET[1])]
    fire_all = {s: torch.zeros(D, dtype=torch.long) for s in census_sites}
    fire_iv = {s: torch.zeros(D, dtype=torch.long) for s in census_sites}
    acts = {t: [] for t in tracked}
    need = set(census_sites) | {(t[0], t[1]) for t in tracked}
    batch_iv = []

    def hook(layer_idx, activations):
        for kd in KINDS:
            s = (layer_idx, kd)
            if s not in need:
                continue
            ta, ti = bank.encode(activations[k2i[kd]], kd, layer_idx)                   # [B, T, k]
            ta, ti = ta.float().cpu(), ti.long().cpu()
            on = ta > 0
            if s in fire_all:
                fire_all[s] += torch.bincount(ti[on], minlength=D)
                m_iv = on & batch_iv[-1].unsqueeze(-1)
                fire_iv[s] += torch.bincount(ti[m_iv], minlength=D)
            for t in tracked:
                if (t[0], t[1]) == s:
                    acts[t].append(((ti == t[2]) * ta).sum(-1))                            # [B, T] dense value

    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(tokens.shape[0]), BS):
                batch_iv.append(is_iv[s0:s0 + BS])
                inference.forward(tokens[s0:s0 + BS], activations_callback=hook, return_activations=False,
                                  tokenize_final=False)
    finally:
        inference.enable_compile()
    A = {t: torch.cat(v).flatten() for t, v in acts.items()}
    ivf = is_iv.flatten()
    n_iv = int(ivf.sum())

    rep = ["# Is the 11.mlp.30743 relay one feature? (random corpus sample: %d sequences, %d positions, %d \"iv\" "
           "tokens)\n" % (tokens.shape[0], tokens.numel(), n_iv)]
    out = dict(n_seq=int(tokens.shape[0]), n_pos=int(tokens.numel()), n_iv=n_iv, iv_ids=iv_ids)

    # 2a. per-latent behaviour
    rep.append("## Behaviour per latent on the random sample\n")
    rep.append("| latent | fires (positions) | precision on \"iv\" | recall of \"iv\" | top tokens it fires on (count) |")
    rep.append("|---|---|---|---|---|")
    beh = {}
    flat_tok = tokens.cpu().flatten()
    for t in tracked:
        on = A[t] > 0
        n = int(on.sum()); p = float((on & ivf).sum()) / max(n, 1); r = float((on & ivf).sum()) / max(n_iv, 1)
        tc = torch.bincount(flat_tok[on], minlength=vocab)
        top = [(tok.decode([int(i)]), int(tc[i])) for i in tc.topk(6).indices.tolist() if tc[i] > 0]
        beh["%d.%s.%d" % t] = dict(fires=n, precision=p, recall=r, top_tokens=top)
        rep.append("| %s%d/%d%s | %d | %.3f | %.3f | %s |" % ("R" if t[1] == "resid" else "M", t[0], t[2],
                                                             " (target)" if t == TARGET else "", n, p, r,
                                                             ", ".join("%r×%d" % x for x in top).replace("|", "/")))
    out["behaviour"] = beh

    # 2b. pairwise co-firing
    rep.append("\n## Co-firing between consecutive chain latents (and each with the target)\n")
    rep.append("| pair | Jaccard of firing positions | Pearson r of activations |")
    rep.append("|---|---|---|")
    pairs = list(zip(CHAIN[:-1], CHAIN[1:])) + [(c, TARGET) for c in CHAIN] + [(CHAIN[0], CHAIN[-1])]
    co = {}
    for a, b in pairs:
        oa, ob = A[a] > 0, A[b] > 0
        j = float((oa & ob).sum()) / max(1, int((oa | ob).sum()))
        r = float(np.corrcoef(A[a].numpy(), A[b].numpy())[0, 1])
        co["%d.%s.%d->%d.%s.%d" % (a + b)] = dict(jaccard=j, pearson=r)
        rep.append("| R%d/%d → %s%d/%d | %.3f | %.3f |" % (a[0], a[2], "R" if b[1] == "resid" else "M", b[0], b[2], j, r))
    out["cofire"] = co

    # 1. geometry
    rep.append("\n## Dictionary geometry\n")
    rep.append("| pair | decoder cos | rank of downstream among its dictionary (by decoder cos) | read-write cos (enc_d · dec_u) |")
    rep.append("|---|---|---|---|")
    dec_of = lambda t: bank.saes[t[1]][t[0]].decoder.weight.detach().float()                 # [d_model, D]
    enc_of = lambda t: bank.saes[t[1]][t[0]].encoder.weight.detach().float()                 # [D, d_model]
    geo = {}
    for a, b in zip(CHAIN[:-1], CHAIN[1:]):
        da = dec_of(a)[:, a[2]]; da = da / da.norm()
        Db = dec_of(b); Db = Db / Db.norm(dim=0, keepdim=True)
        cos_all = (da.to(Db.device) @ Db).cpu()
        c = float(cos_all[b[2]]); rank = int((cos_all > c).sum()) + 1
        eb = enc_of(b)[b[2]]; rw = float((eb / eb.norm()).to(da.device) @ da)
        geo["%d->%d" % (a[0], b[0])] = dict(cos=c, rank=rank, readwrite=rw)
        rep.append("| R%d/%d → R%d/%d | %.3f | %d of %d | %.3f |" % (a[0], a[2], b[0], b[2], c, rank, D, rw))
    # random baseline: random latent pairs across the same dictionaries
    base = []
    for a, b in zip(CHAIN[:-1], CHAIN[1:]):
        Da, Db = dec_of(a), dec_of(b)
        ia = torch.tensor(RNG.integers(0, D, 2000)); ib = torch.tensor(RNG.integers(0, D, 2000))
        x = Da[:, ia] / Da[:, ia].norm(dim=0); y = Db[:, ib] / Db[:, ib].norm(dim=0)
        base.append((x * y).sum(0).abs().cpu())
    b_med = float(torch.cat(base).median()); b_p99 = float(torch.cat(base).quantile(0.99))
    rep.append("\nRandom cross-dictionary pairs: median |cos| %.3f, 99th percentile %.3f." % (b_med, b_p99))
    out["geometry"] = geo; out["geometry_baseline"] = dict(median=b_med, p99=b_p99)

    # 3. census
    rep.append("\n## Census: the best \"iv\" detector in each dictionary (F1 on the random sample)\n")
    rep.append("| site | best latent | precision | recall | F1 | in the circuit's chain? |")
    rep.append("|---|---|---|---|---|---|")
    chain_idx = {(t[0], t[1]): t[2] for t in CHAIN + [TARGET]}
    cen = {}
    for s in census_sites:
        fa, fi = fire_all[s].double(), fire_iv[s].double()
        prec = fi / fa.clamp(min=1); rec = fi / max(n_iv, 1)
        f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
        j = int(f1.argmax())
        chosen = chain_idx.get(s)
        mark = "yes" if chosen == j else ("no (circuit has %d, F1 %.2f)" % (chosen, float(f1[chosen])) if chosen is not None
                                          else "no chain node at this site")
        cen["%d.%s" % s] = dict(best=j, precision=float(prec[j]), recall=float(rec[j]), f1=float(f1[j]), chain=chosen)
        rep.append("| %s%d | %d | %.3f | %.3f | %.3f | %s |" % ("R" if s[1] == "resid" else "M", s[0], j, float(prec[j]),
                                                              float(rec[j]), float(f1[j]), mark))
    out["census"] = cen
    txt = "\n".join(rep)
    print(txt)
    (HERE / "results" / "check_relay.md").write_text(txt, encoding="utf-8")
    json.dump(out, open(HERE / "results" / "check_relay.json", "w"), indent=1)


if __name__ == "__main__":
    main()
