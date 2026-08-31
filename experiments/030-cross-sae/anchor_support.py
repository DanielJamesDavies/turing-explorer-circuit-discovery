"""The corrected mechanism measurement: ANCHOR SUPPORT.

The eval-time top-k projection (topkeval.py) did NOT repair the dense
null under the zero fill, and the reason identifies the real mechanism:
under a zero fill the null's members are the only nonzero entries, so
projecting the RECONSTRUCTED code cannot remove them. What kills the
null at home is the ENCODER: a Top-K encoder emits exactly zero for a
latent outside the top-k at that position, so its fitted amplitude
multiplies zero and buys nothing. A ReLU encoder emits a nonzero value
almost everywhere, so amplitudes have purchase.

This script measures that quantity directly, on the dense dictionaries:
for the probe anchor positions, what fraction of latents drawn from the
live pool are NONZERO at the anchor (i.e. usable by a fitted
amplitude)? Reported alongside the Top-K expectation k/d_sae, and the
same statistic for the discovered members.

  PYTHONPATH=. python anchor_support.py
"""
import json

import torch

import crosssae as C

DEV = C.DEV
HERE = C.HERE
K_HOME, D_HOME = 128, 40960          # the paper's Top-K bank


def main():
    data = torch.load(HERE / "scan.pt", weights_only=False)
    toks, seeds = data["tokens"], data["seeds"]
    out = []
    for key, S in seeds.items():
        L, sl = S["layer"], S["latent"]
        UP = C.upstream_sites(L)
        saes = {site: C.load_sae(C.sub_name(site)) for site in UP}
        seed_sae = C.load_sae(C.sub_name(("resid", L)))
        pos = toks[S["pos_windows"]]
        pos_tr = pos[:C.N_TRAIN]
        nat = C.forward(pos_tr, C.Hooks({}, ("resid", L), seed_sae, sl))
        anchors = nat.argmax(dim=1)

        # per site: code at the anchor positions -> support statistics
        frac_nonzero, n_live, l0_at_anchor = [], [], []
        for site in UP:
            cap = {}
            def grab(mod, inp, out, _s=site):
                x = out[0] if isinstance(out, tuple) else out
                f = C.encode(saes[_s], x)
                B = f.shape[0]
                cap[_s] = f[torch.arange(B, device=DEV),
                            anchors.to(DEV)]              # [B, D_SAE]
            hd = C.site_module(site).register_forward_hook(grab)
            with torch.no_grad():
                C.model(pos_tr.to(DEV), use_cache=False)
            hd.remove()
            at_anchor = cap[site]
            live_mask = (at_anchor > 0).any(0)            # live anywhere
            # fraction of LIVE latents nonzero at a given anchor, averaged
            nz = (at_anchor > 0).float()
            frac_nonzero.append(float(nz[:, live_mask].mean())
                                if live_mask.any() else 0.0)
            n_live.append(int(live_mask.sum()))
            l0_at_anchor.append(float(nz.sum(1).mean()))
        rec = {"layer": L, "latent": sl,
               "sites": len(UP),
               "mean_live_per_site": round(sum(n_live) / len(n_live), 1),
               "anchor_support_rate": round(
                   sum(frac_nonzero) / len(frac_nonzero), 4),
               "mean_L0_at_anchor": round(
                   sum(l0_at_anchor) / len(l0_at_anchor), 1),
               "topk_expectation": round(K_HOME / D_HOME, 4)}
        out.append(rec)
        print("L%d/%-5d  live/site %8.0f | ANCHOR SUPPORT %.3f "
              "(L0@anchor %.0f) vs Top-K expectation %.4f"
              % (L, sl, rec["mean_live_per_site"],
                 rec["anchor_support_rate"], rec["mean_L0_at_anchor"],
                 rec["topk_expectation"]), flush=True)
        del saes, seed_sae
        torch.cuda.empty_cache()
    with (HERE / "anchor_support.json").open("w") as f:
        json.dump(out, f, indent=1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
