"""THE control for country-at-city retrieval: does the country latent
fire at the city token even when the country is NOWHERE in the window?

Splits each city's windows into country-mentioned vs country-absent and
reads the country-specific latents at the city anchor (p..p+2) in each
subset. Retrieval that survives the country-absent subset is a stored
association; retrieval only in the mentioned subset is context echo.

  PYTHONPATH=src python .../retrieval_control.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from hardware import detect_devices, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer
from pipeline.component_index import component_idx, split_component_idx
from sae.bank import SAEBank

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from city_scan import gather_windows

# country-SPECIFIC latents only (verified near-zero at control cities)
TESTS = [
    ("Cairo", "Egypt", [(32, 32384), (35, 23662), (29, 39412),
                        (29, 20684), (32, 22545)]),
    ("Vienna", "Austria", [(29, 3049), (35, 33404), (26, 10050)]),
]
BATCH = 16


def main():
    tok = Tokenizer()
    devices = detect_devices()
    device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    bank = SAEBank(devices=devices, load_decoders=False,
                   compile=should_compile())
    n_kinds = len(bank.kinds)

    for city, country, latents in TESTS:
        wins = gather_windows(tok, city)
        c_ids = [i for i in tok.encode(" " + country)
                 if i > 2 and i != 29871]
        c_last = c_ids[-1]
        with_c = [w for w in wins if c_last in w[0]]
        without_c = [w for w in wins if c_last not in w[0]]
        print("\n### %s: %d windows | country '%s' mentioned in %d, "
              "absent in %d" % (city, len(wins), country, len(with_c),
                                len(without_c)), flush=True)
        want = {}
        for c, l in latents:
            want.setdefault(c, []).append(l)

        def read(subset):
            acc = {k: [0.0, 0, 0] for k in latents}   # sum, events, windows
            n = 0
            for s0 in range(0, len(subset), BATCH):
                chunk = subset[s0:s0 + BATCH]
                toks = torch.tensor([[max(t, 0) for t in w]
                                     for w, _ in chunk],
                                    dtype=torch.long, device=device)
                T = toks.shape[1]
                anch = torch.tensor([p for _, p in chunk],
                                    dtype=torch.long, device=device)

                def cb(layer_idx, activations):
                    with torch.no_grad():
                        B = toks.shape[0]
                        ar = torch.arange(B, device=device)
                        for ki, kind in enumerate(bank.kinds):
                            c = component_idx(layer_idx, ki, n_kinds)
                            if c not in want:
                                continue
                            ta, ti = bank.encode(activations[ki], kind,
                                                 layer_idx)
                            winhit = {lat: torch.zeros(B, dtype=torch.bool,
                                                       device=device)
                                      for lat in want[c]}
                            for off in (0, 1, 2):
                                pos = (anch + off).clamp(0, T - 1)
                                av = ta[ar, pos, :].float()
                                iv = ti[ar, pos, :]
                                for lat in want[c]:
                                    hit = (iv == lat)
                                    vals = torch.where(
                                        hit, av,
                                        torch.zeros_like(av)).amax(-1)
                                    acc[(c, lat)][0] += float(vals.sum()) / 3
                                    acc[(c, lat)][1] += int((vals > 0).sum())
                                    winhit[lat] |= (vals > 0)
                            for lat in want[c]:
                                acc[(c, lat)][2] += int(winhit[lat].sum())
                inference.forward(toks, num_gen=1, tokenize_final=False,
                                  activations_callback=cb,
                                  return_activations=False)
                n += len(chunk)
            return acc, n

        for name, subset in (("country MENTIONED", with_c),
                             ("country ABSENT", without_c)):
            if len(subset) < 8:
                print("  [%s] too thin (%d)" % (name, len(subset)))
                continue
            acc, n = read(subset)
            print("  [%s] n=%d" % (name, n))
            for (c, l), (s, k, wn) in acc.items():
                layer, ki = split_component_idx(c, n_kinds)
                print("    L%-2d %-6s %-6d mean %5.2f  in %d/%d windows"
                      % (layer, bank.kinds[ki], l, s / max(n, 1), wn, n))


if __name__ == "__main__":
    main()
