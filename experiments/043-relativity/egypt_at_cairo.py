"""Do the EGYPT-dedicated latents fire at the CAIRO token?

Reads the seven strongest Egypt-anchor latents (from city_scan) at the
Cairo anchor positions, plus at Tokyo anchors as the control city --
if they fire at Cairo but not Tokyo, the model is retrieving
country-of-city at the city token: the "Cairo -> Egypt" association,
seen in the forward pass.

  PYTHONPATH=src python experiments/043-relativity/egypt_at_cairo.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from hardware import detect_devices, should_compile
from model.inference import Inference
from pipeline.component_index import component_idx, split_component_idx
from sae.bank import SAEBank

HERE = Path(__file__).parent
EGYPT_LATENTS = [(32, 32384), (35, 23662), (29, 39412), (29, 20684),
                 (35, 33148), (32, 22545), (26, 29346)]
BATCH = 16


def main():
    devices = detect_devices()
    device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    bank = SAEBank(devices=devices, load_decoders=False,
                   compile=should_compile())
    n_kinds = len(bank.kinds)
    wins = dict(torch.load(HERE / "city_windows.pt", weights_only=False))
    wins.update(torch.load(HERE / "egypt_windows.pt", weights_only=False))

    want = {}
    for c, l in EGYPT_LATENTS:
        want.setdefault(c, []).append(l)

    for group in ("Cairo", "Tokyo", "Paris", "Egypt"):
        ws = wins.get(group)
        if not ws:
            continue
        acc = {k: [0.0, 0] for k in EGYPT_LATENTS}   # sum, n_fired
        n = 0
        for s0 in range(0, len(ws), BATCH):
            chunk = ws[s0:s0 + BATCH]
            toks = torch.tensor([[max(t, 0) for t in w] for w, _ in chunk],
                                dtype=torch.long, device=device)
            anch = torch.tensor([p for _, p in chunk], dtype=torch.long,
                                device=device)

            def cb(layer_idx, activations):
                with torch.no_grad():
                    B = toks.shape[0]
                    ar = torch.arange(B, device=device)
                    for ki, kind in enumerate(bank.kinds):
                        c = component_idx(layer_idx, ki, n_kinds)
                        if c not in want:
                            continue
                        ta, ti = bank.encode(activations[ki], kind, layer_idx)
                        av = ta[ar, anch, :].float()
                        iv = ti[ar, anch, :]
                        for lat in want[c]:
                            hit = (iv == lat)
                            vals = torch.where(hit, av,
                                               torch.zeros_like(av)).amax(-1)
                            acc[(c, lat)][0] += float(vals.sum())
                            acc[(c, lat)][1] += int((vals > 0).sum())
            inference.forward(toks, num_gen=1, tokenize_final=False,
                              activations_callback=cb,
                              return_activations=False)
            n += len(chunk)
        print("\n%s anchors (n=%d): Egypt-latent activity AT the anchor"
              % (group, n))
        for (c, l), (s, k) in acc.items():
            layer, ki = split_component_idx(c, n_kinds)
            print("  L%-2d %-6s %-6d mean %6.2f  fires in %2d/%d windows"
                  % (layer, bank.kinds[ki], l, s / max(n, 1), k, n))


if __name__ == "__main__":
    main()
