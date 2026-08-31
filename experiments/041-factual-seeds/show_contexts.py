"""Show what a feature ACTUALLY fires on: the top-activating tokens in
their wikitext context. Neuronpedia labels are auto-interp and can be
wrong or vague; this reads the activations directly, so a factual claim
about a seed rests on observed behaviour rather than on a label.

  PYTHONPATH=. python show_contexts.py 8/10364 8/13404 ...
  PYTHONPATH=. python show_contexts.py --all      (every screened seed)

Prints, per feature: the top-k (window, position) activations with the
firing token marked [[like this]] inside a window of surrounding text.
"""
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
GEMMA = HERE.parent / "038-transcoder-compare-gemma"
sys.path.insert(0, str(GEMMA))
os.chdir(GEMMA)

import ours_gtc as G

K = int(os.environ.get("K", 12))
CTX = int(os.environ.get("CTX", 9))


def contexts(toks, layer, feat, k=K):
    """Top-k firing positions for one feature over the given windows."""
    best = []                      # (act, window_idx, pos)
    with torch.no_grad():
        for s0 in range(0, len(toks), G.BATCH):
            cap = {}
            hd = G.block(layer).pre_feedforward_layernorm.register_forward_hook(
                lambda m, i, o: cap.__setitem__("f", G.features(layer, o)))
            G.model(toks[s0:s0 + G.BATCH].to(G.DEV))
            hd.remove()
            f = cap["f"][:, :, feat].float().cpu()      # [B, T]
            for b in range(f.shape[0]):
                a, p = f[b].max(0)
                if float(a) > 0:
                    best.append((float(a), s0 + b, int(p)))
    best.sort(reverse=True)
    return best[:k]


def render(toks, hits):
    out = []
    for act, w, p in hits:
        lo, hi = max(0, p - CTX), min(toks.shape[1], p + CTX + 1)
        parts = []
        for i in range(lo, hi):
            t = G.tok.decode([int(toks[w, i])])
            parts.append("[[%s]]" % t if i == p else t)
        out.append("  %6.2f  ...%s..." % (act, "".join(parts).replace("\n", " ")))
    return out


def main():
    src = HERE / os.environ.get("SEEDS", "run_seeds.pt")
    if not src.exists():
        src = HERE / "factual_seeds.pt"
    data = torch.load(src, weights_only=False)
    print("# reading seeds from %s" % src.name)
    toks, seeds = data["tokens"], data["seeds"]
    args = sys.argv[1:]
    keys = sorted(seeds) if (not args or args[0] == "--all") else args
    for key in keys:
        S = seeds.get(key)
        if S is None:
            print("!! %s not in the screened set" % key); continue
        # read on the seed's OWN positive windows: that is the corpus the
        # circuit is fitted and scored on, so it is what the labels must
        # describe for the anatomy to mean anything.
        pos = toks[S["pos_windows"]]
        hits = contexts(pos, S["layer"], S["latent"])
        print("\n=== %s | %s" % (key, S.get("label", "(no label)")))
        print("    fire_frac %.4f  max_act %.2f"
              % (S.get("fire_frac", float("nan")), S.get("max_act", float("nan"))))
        print("\n".join(render(pos, hits)) or "  (no positive activations)")


if __name__ == "__main__":
    main()
