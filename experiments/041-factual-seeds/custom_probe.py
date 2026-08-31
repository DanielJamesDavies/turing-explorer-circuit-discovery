"""Test a hypothesis about what a latent detects, on TEXT WE WRITE
rather than on corpus windows it was selected from.

The screen picks a seed by its wikitext behaviour, and the circuit is
fitted on that same slice -- so any reading of "what the seed means"
risks being a story about 64 windows. Custom probes break that loop:
state a hypothesis, write positives and negatives that separate it from
the obvious confound, and read the activation per token.

  PYTHONPATH=. python custom_probe.py 8/2016 probes_carbonyl.txt

Probe file: one sentence per line; "#" comments; a line starting with
"== " opens a labelled group (e.g. "== POSITIVE (predict: fires)").
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

TOPN = int(os.environ.get("TOPN", 4))


def main():
    layer, feat = (int(x) for x in sys.argv[1].split("/"))
    lines = [l.rstrip("\n") for l in open(HERE / sys.argv[2], encoding="utf-8")]
    group = ""
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("== "):
            group = line[3:].strip()
            print("\n### %s" % group)
            continue
        ids = [G.tok.bos_token_id] + G.tok(line, add_special_tokens=False)["input_ids"]
        t = torch.tensor([ids], dtype=torch.long)
        cap = {}
        hd = G.block(layer).pre_feedforward_layernorm.register_forward_hook(
            lambda m, i, o: cap.__setitem__("f", G.features(layer, o)))
        with torch.no_grad():
            G.model(t.to(G.DEV))
        hd.remove()
        a = cap["f"][0, :, feat].float().cpu()
        a[0] = 0.0                     # BOS carries anomalous norm; excluded
        top = a.topk(min(TOPN, len(a)))
        hits = [(G.tok.decode([ids[p]]), float(v))
                for v, p in zip(top.values.tolist(), top.indices.tolist())
                if v > 0]
        print("  max %5.2f  %s" % (float(a.max()), line[:74]))
        if hits:
            print("           fires on: %s"
                  % ", ".join("%r=%.1f" % (t_, v) for t_, v in hits))


if __name__ == "__main__":
    main()
