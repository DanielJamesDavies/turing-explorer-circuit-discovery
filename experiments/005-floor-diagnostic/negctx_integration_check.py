"""End-to-end check that floor_source="negctx" actually works on the live path.

The unit tests in tests/eval/test_ablation_faithfulness.py exercise
resolve_site_floors directly with stubs. They CANNOT catch the real integration
risk: gradient_base sets self._floor_neg_tokens inside _discover(), so a wiring
mistake there leaves it None and every negctx discovery raises — invisible to
any test that does not go through discover(). This runs the real thing.

Three checks, cheapest seed (L2, 8 upstream sites):

  1. ROUTING   — resolve_site_floors under floor_source="negctx" returns
                 exactly collect_site_means(neg_tokens), and NOT the posctx
                 means it was handed.
  2. PLUMBING  — a real discover() completes under negctx, proving
                 self._floor_neg_tokens is set and reaches the resolver.
  3. EFFECT    — the negctx circuit DIFFERS from the posctx circuit. If the
                 floor changed but attribution did not, the knob is decorative.

  PYTHONPATH=src python experiments/005-floor-diagnostic/negctx_integration_check.py
"""
import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import upstream_sites
from eval.floors import collect_site_means, resolve_site_floors
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from pathlib import Path

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
SC_IDX, LATENT = 8, 30122          # L2 resid — 8 upstream sites, cheapest seed
N_SEQ, EVAL_BS, D_SAE = 64, 16, 40960
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
avg_acts = torch.zeros((bank.n_layer * len(bank.kinds), D_SAE), device=bank.device)
n_kinds = len(bank.kinds)

_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = True
disc.position_aware_select = "abs_pctl"
disc.position_aware_threshold = 90.0
disc.magnitude_prune = False
disc.recurrence_prune = False
disc.min_faithfulness = -100.0
config.discovery.ablation_gradient.negative_roles = "include"

layer, ki = split_component_idx(SC_IDX, n_kinds)
kind = bank.kinds[ki]
up = upstream_sites(bank, layer, kind)
failures = []


def check(name, ok, detail=""):
    print("  %-9s %s  %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        failures.append(name)


# ---- 1. ROUTING ---------------------------------------------------------
print("\n[1] routing", flush=True)
meth = _build_mode_method("ablation_gradient", "ig_mean", inference, bank,
                          avg_acts, probe_builder)
pd_ = meth.build_probe_dataset(SC_IDX, LATENT)
pt = pd_.pos_tokens[:N_SEQ]
nt = pd_.neg_tokens[:N_SEQ]
print("    pos_tokens %s  neg_tokens %s" % (tuple(pt.shape), tuple(nt.shape)), flush=True)

posctx = collect_site_means(inference, bank, pt, up)
expected = collect_site_means(inference, bank, nt, up)

disc.floor_source = "negctx"
routed = resolve_site_floors(inference, bank, up, posctx_means=posctx,
                             loader=loader, neg_tokens=nt)
same_as_neg = all(torch.allclose(routed[s], expected[s], atol=1e-5) for s in up)
diff_from_pos = any(not torch.allclose(routed[s], posctx[s], atol=1e-5) for s in up)
check("negctx == means(neg_tokens)", same_as_neg)
check("negctx != means(pos_tokens)", diff_from_pos)

# The claim the whole design rests on: the posctx floor drives the seed, the
# negctx floor does not. Reported as max mean latent value per floor.
mx_pos = max(float(posctx[s].max()) for s in up)
mx_neg = max(float(expected[s].max()) for s in up)
print("    max mean latent — posctx %.4f | negctx %.4f" % (mx_pos, mx_neg), flush=True)

# ---- 2/3. PLUMBING + EFFECT --------------------------------------------
print("\n[2] plumbing: real discover() under each floor", flush=True)
members = {}
for source in ("posctx", "negctx"):
    disc.floor_source = source
    m = _build_mode_method("ablation_gradient", "ig_mean", inference, bank,
                           avg_acts, probe_builder)
    try:
        circ = m.discover(SC_IDX, LATENT)
    except Exception as exc:
        check("discover() under %s" % source, False,
              "%s: %s" % (type(exc).__name__, exc))
        continue
    if circ is None:
        check("discover() under %s" % source, False, "returned None")
        continue
    ms = set()
    for node in circ.nodes.values():
        f = node.feature_id
        if node.metadata.get("role") == "seed" or f is None:
            continue
        if (f.layer, f.kind) in up:
            ms.add((f.layer, f.kind, int(f.index)))
    members[source] = ms
    check("discover() under %s" % source, True, "%d upstream members" % len(ms))
    del circ
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n[3] effect", flush=True)
if len(members) == 2:
    a, b = members["posctx"], members["negctx"]
    inter = len(a & b)
    union = len(a | b)
    jac = inter / union if union else 1.0
    check("negctx circuit differs from posctx", a != b,
          "|posctx|=%d |negctx|=%d shared=%d Jaccard=%.3f" % (len(a), len(b), inter, jac))
else:
    check("negctx circuit differs from posctx", False, "a discovery failed above")

disc.floor_source = "posctx"
print("\n" + ("ALL CHECKS PASSED" if not failures
              else "FAILURES: %s" % ", ".join(failures)), flush=True)
raise SystemExit(1 if failures else 0)
