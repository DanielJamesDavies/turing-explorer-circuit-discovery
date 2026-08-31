"""Activation screen for the Neuronpedia factual candidates.

Runs each candidate feature over the SAME 20k cached wikitext windows
the Gemma arena uses (scan_gtc.pt), and keeps those in the method's
operating band -- fire_frac in (0.005, 0.05), the same band the
original scan used, so a factual seed is not a different kind of
object from the six we already studied.

For each survivor writes the seed record the harness needs
(layer, latent, pos_windows, neg_windows) plus the top-activating
token in context, so we can SEE what the feature responds to.

  PYTHONPATH=. python screen_factual.py -> factual_seeds.pt + screen.md
"""
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
GEMMA = HERE.parent / "038-transcoder-compare-gemma"
sys.path.insert(0, str(GEMMA))
os.chdir(GEMMA)                      # ours_gtc.py resolves paths from HERE

N_POS = int(os.environ.get("N_POS", 64))   # matches the harness (N_POS=64, N_TRAIN=48)
BAND = (0.005, 0.05)

import ours_gtc as G                 # model, transcoders, features(), block()

data = torch.load(GEMMA / "scan_gtc.pt", weights_only=False)
toks = data["tokens"]
print("windows:", tuple(toks.shape), flush=True)

CANDS = os.environ.get("CANDS", "candidates.jsonl")
OUT = os.environ.get("OUT", "factual_seeds.pt")
cands = [json.loads(l) for l in open(HERE / CANDS)]
by_layer = {}
for c in cands:
    by_layer.setdefault(c["layer"], []).append(c)

out_seeds, report = {}, ["# Factual seed screen", "",
                         "Candidates from Neuronpedia search, screened on the "
                         "same 20k wikitext windows and the same firing band "
                         "(%.3f-%.3f) as the original scan." % BAND, "",
                         "| layer | feat | fire_frac | max_act | in band | label |",
                         "|---|---|---|---|---|---|"]

for L in sorted(by_layer):
    feats = sorted({c["feat"] for c in by_layer[L]})
    lab = {c["feat"]: c["label"] for c in by_layer[L]}
    idx = torch.tensor(feats, dtype=torch.long, device=G.DEV)
    fire = torch.zeros(len(feats), device=G.DEV)
    mx = torch.zeros(len(feats), device=G.DEV)
    chunks = []
    with torch.no_grad():
        for s0 in range(0, len(toks), G.BATCH):
            cap = {}
            hd = G.block(L).pre_feedforward_layernorm.register_forward_hook(
                lambda m, i, o: cap.__setitem__("f", G.features(L, o)))
            G.model(toks[s0:s0 + G.BATCH].to(G.DEV))
            hd.remove()
            wmax = cap["f"].amax(dim=1)[:, idx]
            fire += (wmax > 0).float().sum(0)
            mx = torch.maximum(mx, wmax.amax(0))
            chunks.append(wmax.half().cpu())
    frac = (fire / len(toks)).cpu()
    wmax_all = torch.cat(chunks)
    for i, f in enumerate(feats):
        ok = BAND[0] < float(frac[i]) < BAND[1]
        report.append("| %d | %d | %.4f | %.2f | %s | %s |" % (
            L, f, frac[i], mx[i], "YES" if ok else "-", lab[f][:70]))
        if not ok:
            continue
        col = wmax_all[:, i]
        top = col.argsort(descending=True)[:N_POS].tolist()
        silent = (col == 0).nonzero(as_tuple=True)[0]
        if len(silent) < N_POS:
            continue
        neg = silent[torch.randperm(len(silent))[:N_POS]].tolist()
        out_seeds["%d/%d" % (L, f)] = {
            "layer": L, "latent": int(f),
            "fire_frac": round(float(frac[i]), 4),
            "max_act": round(float(mx[i]), 3),
            "label": lab[f],
            "pos_windows": top, "neg_windows": neg}
    print("L%d: %d candidates, %d in band"
          % (L, len(feats), sum(1 for k in out_seeds if k.startswith("%d/" % L))),
          flush=True)
    del wmax_all, chunks

torch.save({"tokens": toks, "seeds": out_seeds}, HERE / OUT)
(HERE / "screen.md").write_text("\n".join(report), encoding="utf-8", newline="")
print("\n%d seeds in band -> factual_seeds.pt" % len(out_seeds))
for k, v in sorted(out_seeds.items()):
    print("  %-10s frac %.4f max %5.2f  %s"
          % (k, v["fire_frac"], v["max_act"], v["label"][:60]))
