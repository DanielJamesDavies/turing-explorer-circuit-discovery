"""TIDE OR STEERING? What does an amplitude-fitted circuit do to the
REST of the seed's SAE layer?

The worry (Daniel's): if fitted amplitudes reconstruct the seed by
raising activity wholesale -- amplified upstream latents writing a big
unspecific vector into the residual -- then every latent at the seed's
layer floats up and the seed just rides the tide. That would make
coact_amp's faithfulness (and possibly ours) much less meaningful than
"the circuit computes the seed".

Directly measurable. Under the zero-fill exam on HELD-OUT windows,
capture the full pre-activation vector at the seed's layer at the
anchor, natural vs intervened, for both tri-amp and coact_amp:

  * seed delta vs the delta DISTRIBUTION over all 16,384 latents
    (median, p99, max, and the seed's rank among deltas)
  * fraction of latents pushed up vs down
  * rank preservation: the seed's natural value-rank at the anchor vs
    its rank under the intervention (a tide preserves ranks; targeted
    steering moves the seed up relative to others)
  * Daniel's sub-question: how high does the seed naturally sit in its
    own SAE at its anchors (value rank among active latents)?

coact_amp's alphas are read from the members dump the arm itself wrote
(the arm was rerun with dumping enabled), so this analyses the genuine
fitted object rather than a reimplementation of it.

  TIER=2 PYTHONPATH=. python coact_tide.py
"""
import json

import torch

import ours_gemma as O


def full_preacts(transforms, toks, L):
    """Pre-activations of EVERY latent at layer L's anchor read-out,
    under the given upstream transforms."""
    cap = {}
    hd = O.block(L).post_feedforward_layernorm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("p", O.pre_acts(L, o)))
    with torch.no_grad(), O.Runner(transforms):
        O.model(toks.to(O.DEV))
    hd.remove()
    return cap["p"]


def zero_tf(ma):
    """The zero-fill exam's transform: members at alpha*c, rest zeroed."""
    out = {}
    for layer, d in ma.items():
        idx = torch.tensor(sorted(d), device=O.DEV, dtype=torch.long)
        av = torch.tensor([d[int(i)] for i in idx.tolist()],
                          device=O.DEV, dtype=O.DTYPE)
        def fn(c, _i=idx, _a=av):
            chat = torch.zeros_like(c)
            chat[..., _i] = c[..., _i] * _a
            return chat
        out[layer] = fn
    return out


def main():
    sc = torch.load(O.HERE / ("scan_gemma_t%d.pt" % O.TIER),
                    weights_only=False)
    tri, co = {}, {}
    for line in open(O.HERE / "ours_gemma_members_t2.jsonl"):
        r = json.loads(line)
        d = {int(k): {int(i): float(a) for i, a in v.items()}
             for k, v in r["alphas"].items()}
        (tri if r.get("arm") == "triamp400" else co)[
            (r["layer"], r["latent"])] = d
    stored = {}
    for line in open(O.HERE / "ours_gemma_rows_t2.jsonl"):
        r = json.loads(line)
        if r.get("arm") == "coact_amp":
            stored[(r["layer"], r["latent"])] = r["ampF0"]

    print("seed | nat rank@anchor (of active) || arm: seed_dz | layer dz "
          "med/p99/max | seed dz-rank | frac up | rank nat->int")
    for key, S in sorted(sc["seeds"].items()):
        L, sl = S["layer"], S["latent"]
        if (L, sl) not in tri:
            continue
        toks = sc["tokens"]
        pos = toks[S["pos_windows"]]
        pos_tr, pos_ho = pos[:O.N_TRAIN], pos[O.N_TRAIN:]
        UP = list(range(L))

        # anchors, train (for selection/refit) and held-out (for reading)
        def anchors_of(t):
            cap = {}
            hd = O.block(L).post_feedforward_layernorm \
                .register_forward_hook(
                    lambda m, i, o: cap.__setitem__(
                        "f", O.features(L, o)))
            with torch.no_grad():
                O.model(t.to(O.DEV))
            hd.remove()
            nat = cap["f"][..., sl]
            nat[:, 0] = -float("inf")
            return cap["f"], nat.argmax(dim=1)

        f_ho, an_ho = anchors_of(pos_ho)
        B = pos_ho.shape[0]
        bi = torch.arange(B, device=O.DEV)

        # Daniel's sub-question: seed's natural standing in its own SAE
        at_anchor = f_ho[bi, an_ho.to(O.DEV)]           # [B, D_TC]
        active = (at_anchor > 0)
        vrank = (at_anchor > at_anchor[:, sl:sl + 1]).sum(-1).float() + 1
        print("\n[L%d %d] naturally rank %.1f of %.0f active latents at "
              "its anchor (median over %d held-out windows)"
              % (L, sl, vrank.median(), active.sum(-1).float().median(), B))

        coact = co[(L, sl)]


        p_nat = full_preacts({}, pos_ho, L)[bi, an_ho.to(O.DEV)]
        nat_rank = (p_nat > p_nat[:, sl:sl + 1]).sum(-1).float() + 1
        for label, ma in [("triamp", tri[(L, sl)]), ("coact_amp", coact)]:
            p_int = full_preacts(zero_tf(ma), pos_ho, L)[bi, an_ho.to(O.DEV)]
            dz = (p_int - p_nat).float()
            sdz = dz[:, sl]
            drank = (dz > sdz.unsqueeze(-1)).sum(-1).float() + 1
            irank = (p_int > p_int[:, sl:sl + 1]).sum(-1).float() + 1
            # reproduction check for the refit arm
            note = ""
            if label == "coact_amp":
                a_ho = float(torch.relu(p_int[:, sl]).mean())
                note = " | stored arm ampF0 %.3f" % stored.get((L, sl),
                                                               float("nan"))
            print("  %-9s seed dz %+7.3f | layer dz med %+6.3f p99 %+6.3f "
                  "max %+7.3f | seed dz-rank %5.1f/16384 | up %4.1f%% | "
                  "rank %5.1f -> %5.1f%s"
                  % (label, sdz.median(), dz.median(),
                     dz.quantile(0.99, dim=-1).median(),
                     dz.max(-1).values.median(), drank.median(),
                     100 * (dz > 0).float().mean(), nat_rank.median(),
                     irank.median(), note))


if __name__ == "__main__":
    main()
