"""Which latents fire AT a city's token? The dedicated-vs-compositional
test across a frequency ladder (Paris ~23k mentions, Tokyo ~2k,
Cairo ~200).

For each city: gather windows whose text contains the city name (full
sub-token match, anchored on the LAST sub-token, position > 0), run the
model, and accumulate every latent's activation at the anchor across
all (layer, kind) sites. Report the top latents per city. If a top
latent is city-specific (readable via its top_ctx later), the city has
dedicated hardware; if the top latents are generic (country, city-ness,
geography), the concept is COMPOSED -- which is what we want to seed.

Windows are saved to city_windows.pt for later probe use.

  PYTHONPATH=src python experiments/043-relativity/city_scan.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from hardware import detect_devices, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer
from pipeline.component_index import component_idx, split_component_idx
from sae.bank import SAEBank

HERE = Path(__file__).parent
SEQ = 64
N_WIN = 96
BATCH = 16
import os
CITIES = os.environ.get("CITIES", "Paris,Tokyo,Cairo").split(",")
TOPN = int(os.environ.get("TOPN", 15))


def gather_windows(tok, city, max_shards=3030):
    ids = [i for i in tok.encode(" " + city) if i > 2 and i != 29871]
    last = ids[-1]
    shards = sorted(Path("data").glob("*.npy"))[:max_shards]
    out = []
    for sp in shards:
        sh = np.asarray(np.load(sp, mmap_mode="r"))
        hits = np.where(sh == last)[0]
        if len(ids) > 1 and len(hits):
            ok = np.ones(len(hits), dtype=bool)
            for k, t in enumerate(reversed(ids)):
                idx = hits - k
                good = idx >= 0
                good[good] = sh[idx[good]] == t
                ok &= good
            hits = hits[ok]
        if not len(hits):
            continue
        sep = np.where(sh == -1)[0]
        st = np.concatenate([[0], sep + 1]) + 1
        en = np.concatenate([sep, [len(sh)]])
        keep = (en - st) == SEQ
        st, en = st[keep], en[keep]
        for h in hits:
            j = np.searchsorted(st, h, side="right") - 1
            if j < 0 or h >= en[j]:
                continue
            pos = int(h - st[j])
            if pos == 0:
                continue
            out.append((sh[st[j]:en[j]].tolist(), pos))
            if len(out) >= N_WIN:
                return out
    return out


def main():
    tok = Tokenizer()
    devices = detect_devices()
    device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    bank = SAEBank(devices=devices, load_decoders=False,
                   compile=should_compile())
    n_kinds = len(bank.kinds)
    n_comp = bank.n_layer * n_kinds

    store = {}
    for city in CITIES:
        wins = gather_windows(tok, city)
        print("\n### %s: %d windows" % (city, len(wins)), flush=True)
        if len(wins) < 24:
            print("  too thin, skipping")
            continue
        store[city] = wins
        at = torch.zeros((n_comp, bank.d_sae), device=device)
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
            inference.forward(toks, num_gen=1, tokenize_final=False,
                              activations_callback=cb,
                              return_activations=False)
            n += len(chunk)
        at /= max(1, n)
        flat = at.flatten()
        top = flat.topk(TOPN)
        print("  top latents at the %s anchor:" % city)
        for v, gi in zip(top.values.tolist(), top.indices.tolist()):
            c, lat = gi // bank.d_sae, gi % bank.d_sae
            layer, ki = split_component_idx(c, n_kinds)
            print("    comp %-3d L%-2d %-6s latent %-6d mean %.2f"
                  % (c, layer, bank.kinds[ki], lat, v))
        sample = store[city][0]
        txt = tok.decode([t for t in sample[0] if t >= 0])
        print("  sample window: %s" % txt[:200].replace("\n", " "))
    torch.save(store, HERE / os.environ.get("WINOUT", "city_windows.pt"))
    print("\nwindows saved -> city_windows.pt")


if __name__ == "__main__":
    main()
