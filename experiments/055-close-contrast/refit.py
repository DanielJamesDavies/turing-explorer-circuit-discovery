"""REFIT WITH CLOSE CONTRAST CONTEXTS (DAN-64): does WCM change when C is the intended contrast set?

The 15k production circuits were fitted with floor_negctx_mode="store": C read the neg_ctx kNN store, which gives
38% of targets one shared fallback list and anchors the rest on ~1 of their 128 activating contexts. This refits
N_SEEDS pilot targets through the SAME production path (AblationGradientDiscovery, attribution_mode="mask",
config-h100-triamp.yaml's learned_mask values) with floor_negctx_mode="close", i.e. NegContextSelector: activating
contexts forwarded when missing from seq_repr, own activating contexts excluded, candidates kept only if the target
stays out of Top-K.

  PYTHONPATH=src python experiments/055-close-contrast/refit.py
      -> data_<mode>/discovered_circuits.shard0.pt (key -> Circuit), seeds.txt (shared by every mode), refit_<mode>.log
  env: N_SEEDS (16), FLOOR_NEGCTX_MODE (close | random | distant | store), GAMMA_C / GAMMA_A (0.25 / 0.25),
       LAM (1e-3), OFFTARGET (gamma_S of the off-target term, 0 = off), ARM (output name, default = the mode)

Then score old and new circuits under the same corrected evaluation (amp_eval_pass_v2.py, CTR_SOURCE=close); see README.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parent / "049-circuit-graph"
N_SEEDS = int(os.environ.get("N_SEEDS", 16))
MODE = os.environ.get("FLOOR_NEGCTX_MODE", "close")
GAMMA_C = float(os.environ.get("GAMMA_C", 0.25))     # dual_floor_weight: weight on the contrast-context (C) term
GAMMA_A = float(os.environ.get("GAMMA_A", 0.25))     # triple_floor_weight: weight on the activating-context (A) term
LAM = float(os.environ.get("LAM", 1e-3))
OFFTARGET = float(os.environ.get("OFFTARGET", 0.0))  # gamma_S, the off-target term (056); 0 = off
OT_MODE = os.environ.get("OT_MODE", "all")           # all | cut (only lifts above the clean Top-K cut)
MARGIN = int(os.environ["MARGIN"]) if os.environ.get("MARGIN") else None   # firing-margin k (e.g. 128); None = off
SEED_FILE = Path(os.environ.get("SEED_FILE", str(HERE / "seeds.txt")))       # the shared target list
ARM = os.environ.get("ARM", MODE)                    # output name; defaults to the contrast mode
OUT = HERE / ("data_%s" % ARM)


def pick_seeds():
    """Half fallback-store targets, half retrieved, spread over depth (pilot 60, sorted by layer)."""
    import amp_eval_pass_v2 as V
    rows = [json.loads(l) for l in open(ROOT / "results_full" / "amp_eval_v3_pilot.jsonl")]
    rows = [r for r in rows if "skip" not in r]
    rows.sort(key=lambda r: (int(r["seed"].split(".")[0]), r["seed"]))
    KINDS = V.G["KINDS"]
    fb, rt = [], []
    for r in rows:
        l, k, i = r["seed"].split(".")
        (fb if V.neg_ctx_store_fallback(int(l) * len(KINDS) + KINDS.index(k), int(i)) else rt).append(r["seed"])
    take = lambda xs, n: [xs[round(j * (len(xs) - 1) / max(1, n - 1))] for j in range(n)] if len(xs) > n else xs
    return take(fb, N_SEEDS // 2) + take(rt, N_SEEDS - N_SEEDS // 2), set(fb)


def main():
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("TAG", "refit_unused")
    import amp_eval_pass_v2 as V
    from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
    from config import config

    G = V.setup()
    disc = config.discovery
    lm = disc.learned_mask                              # config-h100-triamp.yaml values (the 15k run)
    lm.mask_floor_source = "triple"; lm.free_amplitude = True; lm.l1_lambda = LAM; lm.deep_site_threshold = 99
    lm.dual_floor_weight = GAMMA_C; lm.triple_floor_weight = GAMMA_A; lm.offtarget_weight = OFFTARGET; lm.offtarget_mode = OT_MODE
    lm.margin_topk = MARGIN
    lm.margin_plain_norm = os.environ.get("MARGIN_PLAIN_NORM") == "1"
    lm.rank_weight = float(os.environ.get("RANK", 0.0))            # Top-K hinge weight gamma_R; 0 = off
    lm.rank_delta = float(os.environ.get("RANK_DELTA", 0.0))       # safety gap above the cut, pre-activation units
    lm.rank_mode = os.environ.get("RANK_MODE", "cut")              # cut | keep | top
    lm.rank_temp = float(os.environ.get("RANK_TEMP", 0.05))        # "top": soft-rank temperature (fraction of seed pre)
    lm.rank_power = float(os.environ.get("RANK_POWER", 2.0))       # "top": exponent p of ((rank - 1) / k)^p
    lm.floors_train_only = os.environ.get("FLOORS_TRAIN_ONLY", "1") == "1"   # 0 = the pre-2026-09-24 leak, for reproduction
    disc.floor_negctx_mode = MODE
    disc.eval_batch_size = 64
    M = _build_mode_method("ablation_gradient", "mask", G["inference"], G["bank"], G["avg_acts"], G["M0"].probe_builder)
    seeds, fb = pick_seeds()
    if SEED_FILE.exists():                          # every mode refits the same targets
        seeds = [s for s in SEED_FILE.read_text().split() if s]
    elif SEED_FILE == HERE / "seeds.txt":           # first run only: freeze the shared list (never overwrite it)
        SEED_FILE.write_text("\n".join(seeds) + "\n")
    if os.environ.get("ONLY"):                      # smoke tests: a subset of the shared targets
        seeds = [s for s in seeds if s in os.environ["ONLY"].split(",")]
    OUT.mkdir(exist_ok=True)
    print("refit %d seeds, arm %s: floor_negctx_mode=%s gamma_C=%g gamma_A=%g lambda=%g gamma_S=%g (%s) margin_topk=%s "
          "rank=%s (%s) (%d fallback-store targets)" % (len(seeds), ARM, MODE, GAMMA_C, GAMMA_A, LAM, OFFTARGET, OT_MODE,
                                                        MARGIN, os.environ.get("RANK", "0"),
                                                        os.environ.get("RANK_MODE", "cut"), sum(s in fb for s in seeds)),
          flush=True)
    KINDS = G["KINDS"]; found = {}
    for s in seeds:
        l, k, i = s.split("."); comp = int(l) * len(KINDS) + KINDS.index(k)
        ts = time.time()
        try:
            c = M.discover(comp, int(i))
        except Exception as e:  # noqa: BLE001
            print("  %-15s ERROR %s: %s" % (s, type(e).__name__, str(e)[:200]), flush=True); continue
        if c is None:
            print("  %-15s rejected (%.0fs)" % (s, time.time() - ts), flush=True); continue
        found[s] = c
        n = sum(1 for nd in c.nodes.values() if nd.metadata.get("role") != "seed")
        print("  %-15s %s  %4d nodes  %.0fs" % (s, "fallback " if s in fb else "retrieved", n, time.time() - ts), flush=True)
        torch.save(found, OUT / "discovered_circuits.shard0.pt")
    print("saved %d circuits -> %s" % (len(found), OUT / "discovered_circuits.shard0.pt"))


if __name__ == "__main__":
    main()
