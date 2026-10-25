"""079 diagnostic: WHERE do RS circuits over-fire? (all-position correlation drop, attn / mlp targets)

For each target in KEYS and each arm (W0, RS0.25, RS1), the circuit-only target read under the C fill (contrast mean,
sparsity-preserving: freeN_topk's fill) vs the natural read, at every position of the 16 held-out activating and 16
held-out contrast contexts. Activation mass (sum of the post-Top-K read) and squared error are split into:
  anchor      the held-out anchors (what training constrains)
  act_fire    other positions of activating contexts where the target naturally fires
  act_silent  positions of activating contexts where it is naturally silent
  ctr         every position of the contrast contexts
plus the distance (in tokens) from the anchor of the circuit's false firings, and the members that are active at those
positions: are they the members the RS term added (not in W0)?

  PYTHONPATH=src python experiments/079-resample-training/trace.py     env: KEYS  OUT79   -> out*/trace.json
"""
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "073-strict-eval"))
import strict_eval as SE  # noqa: E402

OUT79 = Path(os.environ.get("OUT79", str(HERE / "out_smoke4")))
KEYS = os.environ.get("KEYS", "3.mlp.24963,4.mlp.2484,7.mlp.24470,4.attn.9697").split(",")
ARMS = ("W0", "RS0.25", "RS1")


def main():
    from circuit.protocol_contexts import eval_probe
    from eval.ablation_faithfulness import upstream_sites
    from eval.floors import collect_site_means
    from eval.wcm_eval import NaturalTap, TapCircuitOnly, circuit_alphas, contrast_split, site_encoder
    R = SE.driver.Runner()
    G = R.G
    inference, bank, dev = G["inference"], G["bank"], G["device"]
    K, D = int(bank.k), int(bank.d_sae)
    res = {}
    for key in KEYS:
        rec = R.contexts(key)
        probe = eval_probe(rec, "strong", dev)
        n_tr = int(probe.metadata["n_train"])
        pt, pa = probe.pos_tokens[:64], probe.pos_argmax[:64]
        pt_ho, pa_ho = pt[n_tr:], pa[n_tr:]
        nt = probe.neg_tokens[:64]
        n_ntr = contrast_split(int(nt.shape[0]), probe.metadata)      # 086 rebuilt contrast: n_neg_train
        nt_tr, nt_ho = nt[:n_ntr], nt[n_ntr:]
        l, kd, i = key.split(".")
        layer, sl = int(l), int(i)
        site = (layer, kd)
        W, b = site_encoder(bank, layer, kd)
        w_seed, b_seed = W[sl], b[sl]
        UPS = set(upstream_sites(bank, layer, kd))
        means_neg = collect_site_means(inference, bank, nt_tr, UPS)
        both = torch.cat([pt_ho, nt_ho])
        n_act = int(pt_ho.shape[0])
        nat = SE.run_full(inference, lambda a, b_: NaturalTap(bank, site, w_seed, b_seed, sl), both)
        anc = torch.zeros_like(nat, dtype=torch.bool)
        anc[torch.arange(n_act), pa_ho.cpu().clamp(0, nat.shape[1] - 1)] = True
        regions = {
            "anchor": anc,
            "act_fire": (~anc) & (nat > 0) & (torch.arange(nat.shape[0])[:, None] < n_act).to(nat.device),
            "act_silent": (nat <= 0) & (torch.arange(nat.shape[0])[:, None] < n_act).to(nat.device),
            "ctr": (torch.arange(nat.shape[0])[:, None] >= n_act).to(nat.device).expand_as(nat),
        }
        out = {"nat_mass": {k: float(nat[m].sum()) for k, m in regions.items()},
               "n_pos": {k: int(m.sum()) for k, m in regions.items()}}
        members = {}
        for arm in ARMS:
            c = torch.load(OUT79 / "circuits" / arm / ("%s.pt" % key), weights_only=False)
            al = circuit_alphas(c)
            members[arm] = {(s, j) for s, d in al.items() for j in d}
            kt = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in al.items() if d}
            ks = {}
            for s, t in kt.items():
                sv = torch.ones(D, device=dev)
                sv[t] = torch.tensor([al[s][int(j)] for j in t.tolist()], device=dev)
                ks[s] = sv
            anc_idx = torch.cat([pa_ho, torch.zeros(nt_ho.shape[0], dtype=pa_ho.dtype, device=pa_ho.device)])
            circ = SE.run_full(inference, lambda a, b_: TapCircuitOnly(
                bank=bank, keep_indices={s: set(t.tolist()) for s, t in kt.items()}, in_scope=UPS, seed_layer=layer,
                seed_kind=kd, seed_latent_idx=sl, pos_argmax=anc_idx[a:b_], site_means=means_neg, respect_topk=True,
                topk=K, keep_tensors=kt, keep_scales=ks, w_seed=w_seed, b_seed=b_seed), both)
            err = (circ - nat) ** 2
            o = {"n": sum(len(d) for d in al.values()),
                 "mass": {k: float(circ[m].sum()) for k, m in regions.items()},
                 "sq_err_share": {k: float(err[m].sum() / err.sum()) for k, m in regions.items()},
                 "anchor_ratio": float(circ[regions["anchor"]].mean() / nat[regions["anchor"]].mean())}
            # false firings in activating contexts: distance from the anchor
            fp = regions["act_silent"] & (circ > 0)
            rows_, cols = fp.nonzero(as_tuple=True)
            dist = (cols - pa_ho.to(cols.device)[rows_]).float()
            o["fp_act_n"] = int(fp.sum())
            o["fp_act_dist_median"] = float(dist.median()) if dist.numel() else None
            o["fp_act_after_anchor_share"] = float((dist > 0).float().mean()) if dist.numel() else None
            o["fp_act_mass_vs_nat_fire_mass"] = float(circ[fp].sum() / max(nat[nat > 0].sum(), 1e-9))
            o["fp_ctr_rate"] = float((circ[regions["ctr"]] > 0).float().mean())
            out[arm] = o
        added = members["RS1"] - members["W0"]
        out["members"] = dict(W0=len(members["W0"]), RS1=len(members["RS1"]), added_in_RS1=len(added),
                              kept_from_W0=len(members["RS1"] & members["W0"]))
        res[key] = out
        print(json.dumps({key: out}, indent=1), flush=True)
    (OUT79 / "trace.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
