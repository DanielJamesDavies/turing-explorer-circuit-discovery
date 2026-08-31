"""TOKEN-DRIVEN seed selection: which TuringLLM latents fire hardest ON
the relativity token itself?

No auto-interp anywhere. We take training sequences that contain the
'relativ' stem in a PHYSICS context, run the model, and at exactly the
anchor position read every SAE latent at every (layer, kind) site. A
latent that tops the list is, by construction, a latent whose highest
activation in these contexts is on the word we care about.

Two controls, both necessary:
  * PHYSICS vs DISTRACTOR: the same stem appears in "linguistic
    relativity" and "relatively". Sequences are split by whether the
    window carries physics vocabulary, and we report each latent's
    score on BOTH. A latent that fires equally on the linguistic sense
    is a string detector, not a physics one.
  * SPECIFICITY: mean activation at the anchor vs mean over all other
    positions in the same sequences. A latent that fires everywhere is
    not about relativity.

  PYTHONPATH=src python experiments/043-relativity/find_seeds.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, "src")
from data.loader import DataLoader                       # noqa: E402
from hardware import detect_devices, is_fast_memory, should_compile  # noqa: E402
from model.inference import Inference                    # noqa: E402
from model.tokenizer import Tokenizer                    # noqa: E402
from pipeline.component_index import (                   # noqa: E402
    component_idx, layer_component_bounds, split_component_idx)
from sae.bank import SAEBank                             # noqa: E402

HERE = Path(__file__).parent
N_SEQ = int(os.environ.get("N_SEQ", 96))     # per group
BATCH = int(os.environ.get("BATCH", 8))
TOPN = int(os.environ.get("TOPN", 25))
# physics context words (token ids resolved below); a window counts as
# physics if any appears anywhere in it
PHYS = ["Einstein", "gravity", "spacetime", "光"][:3] + [
    "light", "space", "physics", "energy", "mass", "velocity", "quantum"]


def main():
    tok = Tokenizer()
    phys_ids = set()
    for w in PHYS:
        for form in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            for i in tok.encode(form):
                if i > 2:
                    phys_ids.add(i)

    rows = [json.loads(l) for l in open(HERE / "relativity_seqs.jsonl")
            if '"relativ"' in l]
    phys, dist = [], []
    for r in rows:
        t = [x for x in r["tokens"] if x >= 0]
        (phys if phys_ids & set(t) else dist).append(r)
    print("relativ windows: %d physics-context, %d distractor"
          % (len(phys), len(dist)), flush=True)
    phys, dist = phys[:N_SEQ], dist[:N_SEQ]

    devices = detect_devices(); device = devices[0]
    DataLoader(device=device, pin_memory=is_fast_memory())
    inference = Inference(device=device, compile=should_compile())
    bank = SAEBank(devices=devices, load_decoders=False,
                   compile=should_compile())
    n_kinds = len(bank.kinds)

    D_SAE = bank.d_sae
    n_comp = bank.n_layer * n_kinds

    def scan(group):
        """Mean activation AT the anchor, and mean over all other
        positions, for every latent at every site.

        Accumulated with index_add_ on dense [n_comp, d_sae] buffers:
        the obvious per-element Python loop is ~2M float() calls per
        batch and leaves the GPU idle at 8% util.
        """
        at = torch.zeros((n_comp, D_SAE), device=device, dtype=torch.float32)
        off = torch.zeros((n_comp, D_SAE), device=device, dtype=torch.float32)
        n = 0
        for s0 in range(0, len(group), BATCH):
            chunk = group[s0:s0 + BATCH]
            toks = torch.tensor([[max(t, 0) for t in r["tokens"]]
                                 for r in chunk], dtype=torch.long,
                                device=device)
            anch = torch.tensor([r["pos"] for r in chunk], dtype=torch.long,
                                device=device)

            def cb(layer_idx, activations):
                with torch.no_grad():
                    B = toks.shape[0]
                    ar = torch.arange(B, device=device)
                    for ki, kind in enumerate(bank.kinds):
                        ta, ti = bank.encode(activations[ki], kind, layer_idx)
                        ta = ta.float()
                        c = component_idx(layer_idx, ki, n_kinds)
                        T = ti.shape[1]
                        a_i = ti[ar, anch, :].reshape(-1)
                        a_v = ta[ar, anch, :].reshape(-1)
                        at[c].index_add_(0, a_i, a_v)
                        m = torch.ones((B, T), dtype=torch.bool, device=device)
                        m[ar, anch] = False
                        o_i = ti[m].reshape(-1)
                        o_v = ta[m].reshape(-1) / max(1, T - 1)
                        off[c].index_add_(0, o_i, o_v)
            inference.forward(toks, num_gen=1, tokenize_final=False,
                              activations_callback=cb, return_activations=False)
            n += len(chunk)
        at /= max(1, n); off /= max(1, n)
        return at.cpu(), off.cpu(), n

    print("scanning physics windows...", flush=True)
    p_at, p_off, n_p = scan(phys)
    print("scanning distractor windows...", flush=True)
    d_at, _d_off, n_d = scan(dist)

    out = []
    nz = (p_at > 0).nonzero()
    for c, lat in nz.tolist():
        v = float(p_at[c, lat])
        layer, ki = split_component_idx(c, n_kinds)
        o = float(p_off[c, lat]); dd = float(d_at[c, lat])
        out.append({"comp_idx": c, "latent": lat, "layer": layer,
                    "kind": bank.kinds[ki],
                    "phys_anchor": round(v, 4),
                    "phys_offanchor": round(o, 4),
                    "dist_anchor": round(dd, 4),
                    "specificity": round(v / (o + 1e-6), 2),
                    "phys_vs_dist": round(v / (dd + 1e-6), 2)})
    out.sort(key=lambda r: -r["phys_anchor"])
    with open(HERE / "relativity_latents.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")

    print("\n%d physics / %d distractor windows scanned" % (n_p, n_d))
    print("\n%-6s %-5s %-7s %8s %8s %8s %7s %8s"
          % ("comp", "lat", "site", "anchor", "offanch", "distanc", "spec",
             "phys/dis"))
    for r in out[:TOPN]:
        print("%-6d %-5d L%-2d%-4s %8.3f %8.3f %8.3f %7.1f %8.1f"
              % (r["comp_idx"], r["latent"], r["layer"], r["kind"],
                 r["phys_anchor"], r["phys_offanchor"], r["dist_anchor"],
                 r["specificity"], r["phys_vs_dist"]))
    print("\n-> relativity_latents.jsonl")


if __name__ == "__main__":
    main()
