"""What does a TuringLLM latent actually fire on? Read it off the
training data directly -- no auto-interp involved.

Scans a BROAD sample of training sequences (not just the relativity
windows the latent was selected from, which would only ever show it in
relativity contexts) and prints its highest-activating tokens in
context. This is the TuringLLM advantage over the Gemma arena: the
corpus is ours and decodes cleanly, so a latent can be characterised by
observation instead of by a label.

  PYTHONPATH=src python .../show_ctx.py 29/3736 35/13633 ...
  env: NSEQ (sequences sampled, default 3000), TOPK (default 8)
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from hardware import detect_devices, should_compile          # noqa: E402
from model.inference import Inference                        # noqa: E402
from model.tokenizer import Tokenizer                        # noqa: E402
from pipeline.component_index import (                       # noqa: E402
    component_idx, split_component_idx)
from sae.bank import SAEBank                                 # noqa: E402

HERE = Path(__file__).parent
SEQ = 64
NSEQ = int(os.environ.get("NSEQ", 3000))
TOPK = int(os.environ.get("TOPK", 8))
BATCH = int(os.environ.get("BATCH", 16))
CTX = int(os.environ.get("CTX", 10))


def sample_sequences(n):
    """Random training sequences, segmented exactly as DataLoader does."""
    shards = sorted(Path("data").glob("*.npy"))
    rng = np.random.default_rng(0)
    out = []
    for sp in rng.choice(shards, size=min(60, len(shards)), replace=False):
        sh = np.asarray(np.load(sp, mmap_mode="r"))
        sep = np.where(sh == -1)[0]
        st = np.concatenate([[0], sep + 1]) + 1
        en = np.concatenate([sep, [len(sh)]])
        keep = (en - st) == SEQ
        st, en = st[keep], en[keep]
        for a, b in zip(st[:n // 50 + 1], en[:n // 50 + 1]):
            out.append(sh[a:b].tolist())
        if len(out) >= n:
            break
    return out[:n]


def main():
    targets = []
    for a in sys.argv[1:]:
        c, l = a.split("/")
        targets.append((int(c), int(l)))
    tok = Tokenizer()
    devices = detect_devices(); device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    bank = SAEBank(devices=devices, load_decoders=False, compile=should_compile())
    n_kinds = len(bank.kinds)

    seqs = sample_sequences(NSEQ)
    print("scanning %d random training sequences for %d latents\n"
          % (len(seqs), len(targets)), flush=True)
    want = {}
    for c, l in targets:
        want.setdefault(c, []).append(l)
    best = {t: [] for t in targets}

    for s0 in range(0, len(seqs), BATCH):
        chunk = seqs[s0:s0 + BATCH]
        toks = torch.tensor([[max(t, 0) for t in s] for s in chunk],
                            dtype=torch.long, device=device)

        def cb(layer_idx, activations):
            with torch.no_grad():
                for ki, kind in enumerate(bank.kinds):
                    c = component_idx(layer_idx, ki, n_kinds)
                    if c not in want:
                        continue
                    ta, ti = bank.encode(activations[ki], kind, layer_idx)
                    ta = ta.float()
                    for lat in want[c]:
                        hit = (ti == lat)
                        if not bool(hit.any()):
                            continue
                        vals = torch.where(hit, ta,
                                           torch.zeros_like(ta)).amax(-1)
                        for b in range(vals.shape[0]):
                            v, p = vals[b].max(0)
                            if float(v) > 0:
                                best[(c, lat)].append(
                                    (float(v), s0 + b, int(p)))
        inference.forward(toks, num_gen=1, tokenize_final=False,
                          activations_callback=cb, return_activations=False)

    for (c, lat) in targets:
        layer, ki = split_component_idx(c, n_kinds)
        hits = sorted(best[(c, lat)], reverse=True)[:TOPK]
        print("=== comp %d (L%d %s) latent %d -- %d/%d sequences fire"
              % (c, layer, bank.kinds[ki], lat,
                 len(best[(c, lat)]), len(seqs)))
        for v, si, p in hits:
            s = seqs[si]
            lo, hi = max(0, p - CTX), min(len(s), p + CTX + 1)
            # decode SPANS, not single tokens: this tokenizer encodes
            # leading spaces, which per-token decoding silently drops
            # and runs every word together.
            cl = lambda a, b: tok.decode([t for t in s[a:b] if t >= 0])
            txt = "%s[[%s]]%s" % (cl(lo, p), cl(p, p + 1), cl(p + 1, hi))
            print("  %7.2f  ...%s..." % (v, txt.replace("\n", " ")))
        print()


if __name__ == "__main__":
    main()
