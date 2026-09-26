"""LABEL the hub / reduced / core latents from their own top contexts
(item 7, GPU-light): for each latent, build its probe dataset (64 top and
mid contexts with the argmax position), decode the token AT the peak and
the 3 tokens before it, and report the token histogram at the peak
("top-token consistency" = share of the most common token) plus 3
example windows. No LLM labeller: this is the activation-gated evidence
the auto-interp labels would have to agree with.

  N_LAT=80 PYTHONPATH=src python experiments/049-circuit-graph/label_hubs.py
"""
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer
from pipeline.component_index import component_idx as comp_of
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

HERE = Path(__file__).parent; R = Path(os.environ.get("OUT", str(HERE / "results")))
N_LAT = int(os.environ.get("N_LAT", 80))
N_FAM_LABEL = int(os.environ.get("N_FAM_LABEL", 12))
FAMS_LABEL = [int(x) for x in os.environ.get("FAMS", "").split(",") if x]   # label these families' cores instead of the largest
torch.set_float32_matmul_precision("high")
load_discovery_artifacts("outputs", candidates_path="outputs/candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
pb = ProbeDatasetBuilder(inference, bank, loader)
KINDS = list(bank.kinds); NK = len(KINDS)
avg_acts = torch.zeros((bank.n_layer * NK, bank.d_sae), device=bank.device)
config.discovery.probe_sequence_count = 64; config.discovery.probe_batch_size = 4
M0 = _build_mode_method("counterfactual_gradient", "local", inference, bank, avg_acts, pb)
tok = Tokenizer()

todo = []
SEEDS_CSV = os.environ.get("SEEDS_CSV")
if SEEDS_CSV:
    # label an explicit list of seeds (e.g. interesting_top400.csv) instead of hubs/families
    _s = pd.read_csv(SEEDS_CSV)
    for _, r in _s.head(N_LAT).iterrows():
        todo.append((r["skey"], "score %.2f" % r["score"] if "score" in r else "listed"))
fo = pd.read_csv(R / "fanout_latents.csv")
for x in (fo["latent"].head(40) if not SEEDS_CSV else []):
    todo.append((x, "hub fan-out %d" % int(fo.loc[fo["latent"] == x, "fanout"].iloc[0])))
if not SEEDS_CSV:
    ap = pd.read_csv(R / "amplitude_profiles.csv")
    P = ap[ap["n"] >= 10]
    for _, r in P.sort_values("median").head(8).iterrows():
        todo.append((r["lkey"], "reduced median gain %.2f (n %d)" % (r["median"], r["n"])))
    for _, r in P.sort_values("median", ascending=False).head(8).iterrows():
        todo.append((r["lkey"], "boosted median gain %.2f (n %d)" % (r["median"], r["n"])))
    fams = json.load(open(R / "families_nohub.json"))
    if FAMS_LABEL:
        fams = [f for f in fams if f["family"] in FAMS_LABEL]
    for f in fams[:N_FAM_LABEL]:
        for x in f["core"][:5]:
            todo.append((x, "family %d core (n %d)" % (f["family"], f["n"])))
seen = set(); todo2 = []
for x, why in todo:
    if x not in seen:
        seen.add(x); todo2.append((x, why))
todo = todo2[:N_LAT]
print("labelling %d latents" % len(todo), flush=True)

out = []
for lkey, why in todo:
    l, k, i = lkey.split("."); l, i = int(l), int(i)
    try:
        pd_ = M0.build_probe_dataset(comp_of(l, KINDS.index(k), NK), i)
    except Exception as e:
        print("  %s: probes failed %s" % (lkey, e)); continue
    if pd_ is None or pd_.pos_tokens.shape[0] == 0:
        print("  %-16s | %s | NO CONTEXTS" % (lkey, why)); continue
    pt, pa = pd_.pos_tokens.cpu(), pd_.pos_argmax.cpu()
    peak = Counter(); prev = Counter(); windows = []
    for b in range(pt.shape[0]):
        p = int(pa[b]); ids = pt[b].tolist()
        peak[tok.decode([ids[p]])] += 1
        if p > 0:
            prev[tok.decode([ids[p - 1]])] += 1
        if len(windows) < 3:
            windows.append(tok.decode(ids[max(0, p - 6):p]) + " [[" + tok.decode([ids[p]]) + "]]")
    n = pt.shape[0]; top = peak.most_common(3)
    row = dict(latent=lkey, why=why, n=n, peak_top=top, peak_consistency=top[0][1] / n, prev_top=prev.most_common(3), windows=windows)
    out.append(row)
    print("  %-16s | %-34s | peak %s (%.0f%%) %s | prev %s | e.g. %s"
          % (lkey, why, repr(top[0][0]), 100 * top[0][1] / n, [repr(t) for t, _ in top[1:]], [repr(t) for t, _ in prev.most_common(2)],
             " || ".join(w.replace("\n", "\\n") for w in windows[:2])), flush=True)
tag = os.environ.get("TAG", "")
json.dump(out, open(R / ("labels%s.json" % tag), "w"), indent=1)
print("->", R / ("labels%s.json" % tag))
