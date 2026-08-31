"""Does the model retrieve COUNTRY at a CITY token? The general test.

For each (country, city) pair: find the country's dedicated latents
(top anchor latents at the country token that fire in >=90% of its own
windows), then measure them at the city's anchor and the two positions
after it, with the OTHER cities as controls.

  PYTHONPATH=src python .../country_retrieval.py
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
from city_scan import gather_windows            # reuse the window finder

PAIRS = [("Japan", "Tokyo"), ("Austria", "Vienna"), ("Egypt", "Cairo"),
         ("France", "Paris")]
BATCH = 16
N_DED = 6


def main():
    tok = Tokenizer()
    devices = detect_devices()
    device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    bank = SAEBank(devices=devices, load_decoders=False,
                   compile=should_compile())
    n_kinds = len(bank.kinds)
    n_comp = bank.n_layer * n_kinds

    def scan_top(wins):
        at = torch.zeros((n_comp, bank.d_sae), device=device)
        fired = torch.zeros((n_comp, bank.d_sae), device=device)
        n = 0
        for s0 in range(0, len(wins), BATCH):
            chunk = wins[s0:s0 + BATCH]
            toks = torch.tensor([[max(t, 0) for t in w] for w, _ in chunk],
                                dtype=torch.long, device=device)
            anch = torch.tensor([p for _, p in chunk], dtype=torch.long,
                                device=device)

            def cb(layer_idx, activations):
                with torch.no_grad():
                    B = toks.shape[0]
                    ar = torch.arange(B, device=device)
                    for ki, kind in enumerate(bank.kinds):
                        ta, ti = bank.encode(activations[ki], kind, layer_idx)
                        c = component_idx(layer_idx, ki, n_kinds)
                        a_i = ti[ar, anch, :].reshape(-1)
                        a_v = ta[ar, anch, :].reshape(-1).float()
                        at[c].index_add_(0, a_i, a_v)
                        fired[c].index_add_(0, a_i, (a_v > 0).float())
            inference.forward(toks, num_gen=1, tokenize_final=False,
                              activations_callback=cb,
                              return_activations=False)
            n += len(chunk)
        return at / max(n, 1), fired, n

    def read_at(wins, targets, offsets=(0, 1, 2)):
        want = {}
        for c, l in targets:
            want.setdefault(c, []).append(l)
        acc = {k: [0.0, 0] for k in targets}
        n = 0
        for s0 in range(0, len(wins), BATCH):
            chunk = wins[s0:s0 + BATCH]
            toks = torch.tensor([[max(t, 0) for t in w] for w, _ in chunk],
                                dtype=torch.long, device=device)
            T = toks.shape[1]
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
                        best = None
                        for off in offsets:
                            pos = (anch + off).clamp(0, T - 1)
                            av = ta[ar, pos, :].float()
                            iv = ti[ar, pos, :]
                            for lat in want[c]:
                                hit = (iv == lat)
                                vals = torch.where(
                                    hit, av, torch.zeros_like(av)).amax(-1)
                                key = (c, lat)
                                if best is None:
                                    pass
                                acc[key][0] += float(vals.sum()) / len(offsets)
                                acc[key][1] += int((vals > 0).sum())
            inference.forward(toks, num_gen=1, tokenize_final=False,
                              activations_callback=cb,
                              return_activations=False)
            n += len(chunk)
        return acc, n

    wins = {}
    for country, city in PAIRS:
        for name in (country, city):
            if name not in wins:
                wins[name] = gather_windows(tok, name)
                print("%s: %d windows" % (name, len(wins[name])), flush=True)

    for country, city in PAIRS:
        cw = wins[country]
        if len(cw) < 24 or len(wins[city]) < 24:
            print("\n### %s/%s: too thin, skipped" % (country, city))
            continue
        at, fired, n = scan_top(cw)
        flat = at.flatten()
        ded = []
        for v, gi in zip(*[t.tolist() for t in flat.topk(40)]):
            c, lat = gi // bank.d_sae, gi % bank.d_sae
            if fired[c, lat] >= 0.9 * n:
                ded.append((c, lat, v))
            if len(ded) >= N_DED:
                break
        print("\n### %s dedicated latents (fire >=90%% at own anchor):"
              % country)
        for c, lat, v in ded:
            layer, ki = split_component_idx(c, n_kinds)
            print("    L%-2d %-6s %-6d mean %.2f"
                  % (layer, bank.kinds[ki], lat, v))
        targets = [(c, l) for c, l, _ in ded]
        for probe in [city] + [c2 for _, c2 in PAIRS if c2 != city][:2]:
            acc, np_ = read_at(wins[probe], targets)
            tot_fire = sum(k for _, k in acc.values())
            print("  at %-8s anchors (n=%d, pos p..p+2): "
                  "mean sum %.2f, fire-events %d"
                  % (probe, np_, sum(s for s, _ in acc.values()), tot_fire))
            for (c, l), (s, k) in acc.items():
                if k > 0.1 * np_:
                    layer, ki = split_component_idx(c, n_kinds)
                    print("      L%-2d %-6s %-6d mean %5.2f fires %d"
                          % (layer, bank.kinds[ki], l, s / max(np_, 1), k))


if __name__ == "__main__":
    main()
