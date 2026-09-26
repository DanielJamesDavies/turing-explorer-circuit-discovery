"""SPECIFICITY DIAGNOSTIC: is a circuit a mechanism for its target latent, or a concept amplifier?

Faithfulness, necessity and the random baseline all read the target alone, so a circuit that restores the target by
pushing up its whole neighbourhood (concept, topic, context type) passes them. This reads the WHOLE code at the
target's site (every latent's pre-activation and the actual Top-K set, at the target's anchor) in three runs per
ablation method pi in {Z, A (sparsity-preserving), C (sparsity-preserving, close contrast contexts)}:

  clean     the unmodified model
  empty     no circuit: every upstream latent ablated per pi
  circuit   the fitted circuit (nodes at alpha x live value, everything else ablated per pi)

on the 16 held-out activating contexts, and reports per (circuit, pi):

  rank        the target's rank among the site's pre-activations (1 = top), clean vs circuit; in_topk share
  turnover    Jaccard of the circuit's Top-K set with the clean one, and the same for the empty circuit (does the
              circuit restore the clean code, or distort it?); n_new = latents in the circuit's Top-K not in clean's
  switched_on non-target latents the circuit lifts above the CLEAN Top-K cut that were below it clean
  siblings    latents that co-fire with the target (in the clean Top-K at the anchor on >= SIB_FRAC of the TRAIN
              contexts). Each sibling's faithfulness uses the target's own form, (a_circ - a_empty)/(a_clean - a_empty)
              on the pre-activation. A concept amplifier restores siblings as well as the target; a specific circuit
              restores the target clearly better. Control: other latents active in the clean held-out Top-K.

Everything runs through the engine's CircuitOnlyPatcher, exactly as the v3 eval pass does (every upstream site
ablated, SAE error preserved, fitted alphas applied); contrast contexts come from NegContextSelector("close").

  PYTHONPATH=src python experiments/056-specificity/specificity.py
      -> results/specificity.jsonl (one row per arm x target x pi), then the summary
  env: ARMS (old,close,eq2)  SEEDS (file, default 055 seeds.txt)  SIB_FRAC (0.5)  MIN_DEN (0.05)
       SMOKE=1 (first target of the first arm only)  SUMMARY=1 (re-print from the jsonl)
"""
import contextlib
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent / "049-circuit-graph"
E055 = HERE.parent / "055-close-contrast"
RES = HERE / "results"
ARMS = [a for a in os.environ.get("ARMS", "old,close,eq2").split(",") if a]
SEEDS_FILE = Path(os.environ.get("SEEDS", str(E055 / "seeds.txt")))
SIB_FRAC = float(os.environ.get("SIB_FRAC", 0.5))
MIN_DEN = float(os.environ.get("MIN_DEN", 0.05))      # a latent's faithfulness needs (clean - empty) >= MIN_DEN * clean
SMOKE = os.environ.get("SMOKE") == "1"
OUT = RES / ("specificity_smoke.jsonl" if SMOKE else "specificity.jsonl")
N_SEQ, EVAL_BS = 64, 16
PIS = ("Z", "A", "C")


def arm_data(arm):
    """(circuit shard paths, key filter) for a training arm; 'old' = the stored 15k circuits."""
    if arm == "old":
        import pandas as pd
        ct = pd.read_parquet(ROOT / "tables_full" / "circuits.parquet")
        ct["skey"] = ct.seed_layer.astype(str) + "." + ct.seed_kind.astype(str) + "." + ct.seed_index.astype(str)
        return ct, [ROOT / "data_full" / ("discovered_circuits.shard%d.pt" % i) for i in sorted(set(ct["shard"]))]
    return None, [E055 / ("data_%s" % arm) / "discovered_circuits.shard0.pt"]


def load_circuits(arm, seeds, V):
    ct, paths = arm_data(arm)
    if ct is not None:
        paths = [ROOT / "data_full" / ("discovered_circuits.shard%d.pt" % i) for i in sorted(set(ct[ct.skey.isin(seeds)]["shard"]))]
    found = {}
    for p in paths:
        for c in torch.load(p, weights_only=False, map_location="cpu").values():
            k = V.key_of(c)
            if k in seeds:
                found[k] = c
    return found


class SiteTap:
    """Mixin: at the target site, capture every latent's pre-activation and the SAE's Top-K set at the anchors.
    The stream there is passed through untouched (the target site is never in the ablation scope)."""

    def capture(self, x):
        bank = self.bank
        B = x.shape[0]
        rr = torch.arange(B, device=x.device)
        anc = self.anchors.to(x.device).clamp(0, x.shape[1] - 1)
        xa = x[rr, anc].detach()                                                  # [B, d_model]
        sae = bank.saes[self.site[1]][self.site[0]]
        W = sae.encoder.weight.detach().to(xa.dtype); b = sae._get_bias_eff().detach().to(xa.dtype)
        self.pre_all = (xa @ W.T + b).float()                                     # [B, D], same form as the target tap
        with torch.no_grad():
            ta, ti = bank.encode(xa.unsqueeze(1), self.site[1], self.site[0])     # the SAE's own Top-K
        ta, ti = ta[:, 0], ti[:, 0]
        self.topk_sets = [set(int(i) for i, a in zip(ti[j].tolist(), ta[j].tolist()) if a > 0) for j in range(B)]


def make_patchers(G):
    from eval.ablation_faithfulness import CircuitOnlyPatcher
    from model.hooks import multi_patch

    class CleanTap(SiteTap):
        def __init__(self, bank, site, anchors):
            self.bank, self.site, self.anchors = bank, site, anchors

        def __call__(self, model):
            return multi_patch(model, self.transform)

        def transform(self, layer_idx, kind, x):
            if (layer_idx, kind) == self.site:
                self.capture(x)
            return x

    class CircuitTap(CircuitOnlyPatcher, SiteTap):
        def __init__(self, *a, anchors=None, **k):
            super().__init__(*a, **k)
            self.site, self.anchors = (self.seed_layer, self.seed_kind), anchors

        def transform(self, layer_idx, kind, x):
            if (layer_idx, kind) == self.site:
                self.capture(x)
            return super().transform(layer_idx, kind, x)

    return CleanTap, CircuitTap


def run(G, make, tokens, anchors):
    """Forward `tokens` in chunks; returns (pre_all [N, D] on CPU, list of Top-K sets)."""
    inference = G["inference"]
    pres, sets = [], []
    inference.disable_compile()
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
                p = make(s0, min(s0 + EVAL_BS, int(tokens.shape[0])))
                inference.forward(tokens[s0:s0 + EVAL_BS], patcher=p, grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
                pres.append(p.pre_all.cpu()); sets.extend(p.topk_sets)
    finally:
        inference.enable_compile()
    return torch.cat(pres), sets


def latent_faith(circ, empty, clean):
    """Per-latent faithfulness on the pre-activation, target form; NaN where the denominator is too small."""
    c, e, n = (torch.relu(t).mean(0) for t in (circ, empty, clean))
    den = n - e
    out = (c - e) / den
    out[den < MIN_DEN * n.clamp(min=1e-9)] = float("nan")
    return out


def score(G, c, V, CleanTap, CircuitTap, arm):
    from eval.ablation_faithfulness import upstream_sites
    from eval.floors import collect_site_anchors, collect_site_means
    from pipeline.component_index import component_idx as comp_of

    bank, M0, KINDS, NK, D, K = (G[k] for k in ("bank", "M0", "KINDS", "NK", "D", "K"))
    seed = V.seed_of(c); sf = seed.metadata["feature_id"]; layer, kind, sl = sf.layer, sf.kind, sf.index
    key = "%d.%s.%d" % (layer, kind, sl); site = (layer, kind)
    alphas = defaultdict(dict)
    for n in c.nodes.values():
        if n is not seed:
            f = n.metadata["feature_id"]; alphas[(f.layer, f.kind)][int(f.index)] = float(n.metadata.get("amplitude", 1.0))
    alphas = dict(alphas)
    comp = comp_of(layer, KINDS.index(kind), NK)
    pd_ = M0.build_probe_dataset(comp, sl)
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    sel = M0._neg_context_selector().select(comp, sl, "close", max_sequences=N_SEQ, batch_size=EVAL_BS, exact=False,
                                            non_activation_threshold=0.0, filter_batch_size=32, load_window_size=256)
    nt = sel.tokens[:N_SEQ]
    n_tr = V.split_n(int(pt.shape[0])); n_ntr = V.split_n(int(nt.shape[0]))
    pt_tr, pa_tr, pt_ho, pa_ho = pt[:n_tr], pa[:n_tr], pt[n_tr:], pa[n_tr:]
    UPS = set(upstream_sites(bank, layer, kind)); dev = pt.device
    means = {"Z": None,
             "A": collect_site_anchors(G["inference"], bank, pt_tr, UPS, pa_tr, pin_position_specific=False)[0],
             "C": collect_site_means(G["inference"], bank, nt[:n_ntr], UPS)}
    keep = {s: set(d) for s, d in alphas.items() if d}
    kt = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in keep.items()}
    scales = {}
    for s, d in alphas.items():
        sv = torch.ones(D, device=dev, dtype=torch.float32)
        sv[kt[s]] = torch.tensor([d[int(i)] for i in kt[s].tolist()], device=dev, dtype=torch.float32)
        scales[s] = sv

    def circuit_run(keep_, pi, tokens, anchors):
        return run(G, lambda a, b: CircuitTap(bank=bank, keep_indices=keep_, in_scope=UPS, seed_layer=layer, seed_kind=kind,
                                              seed_latent_idx=sl, pos_argmax=anchors[a:b], site_means=means[pi],
                                              respect_topk=(pi != "Z"), topk=K,
                                              keep_tensors={s: kt[s] for s in keep_}, keep_scales=(scales if keep_ else None),
                                              anchors=anchors[a:b]), tokens, anchors)

    # siblings: co-fire with the target in the clean Top-K at the anchor on >= SIB_FRAC of the TRAIN contexts
    _, tr_sets = run(G, lambda a, b: CleanTap(bank, site, pa_tr[a:b]), pt_tr, pa_tr)
    cnt = defaultdict(int)
    for s_ in tr_sets:
        for j in s_:
            if j != sl:
                cnt[j] += 1
    siblings = sorted(j for j, n_ in cnt.items() if n_ >= SIB_FRAC * len(tr_sets))

    clean_pre, clean_sets = run(G, lambda a, b: CleanTap(bank, site, pa_ho[a:b]), pt_ho, pa_ho)
    clean_active = set().union(*clean_sets) - {sl}
    control = sorted(clean_active - set(siblings))
    tau_clean = torch.tensor([min(float(clean_pre[j, i]) for i in s_) if s_ else float("inf")
                              for j, s_ in enumerate(clean_sets)])                  # clean Top-K cut per anchor

    def rank_of(pre):                                                               # 1 = highest pre-activation
        return (pre > pre[:, sl:sl + 1]).sum(1).float() + 1

    rows = []
    for pi in PIS:
        empty_pre, empty_sets = circuit_run({}, pi, pt_ho, pa_ho)
        circ_pre, circ_sets = circuit_run(keep, pi, pt_ho, pa_ho)
        jac = lambda A, B: float(np.mean([len(a & b) / max(1, len(a | b)) for a, b in zip(A, B)]))
        others = torch.ones(D, dtype=torch.bool); others[sl] = False
        switched = ((circ_pre > tau_clean[:, None]) & (clean_pre <= tau_clean[:, None]) & others[None]).sum(1).float()
        switched_empty = ((empty_pre > tau_clean[:, None]) & (clean_pre <= tau_clean[:, None]) & others[None]).sum(1).float()
        f_all = latent_faith(circ_pre, empty_pre, clean_pre)
        tf = float(f_all[sl])
        sib_f = f_all[siblings] if siblings else torch.tensor([])
        ctl_f = f_all[control] if control else torch.tensor([])
        med = lambda t: float(np.nanmedian(t.numpy())) if t.numel() and not torch.isnan(t).all() else None
        n_ok = lambda t: int((~torch.isnan(t)).sum()) if t.numel() else 0
        rows.append(dict(
            arm=arm, seed=key, pi=pi, layer=layer, kind=kind, n_nodes=sum(len(d) for d in alphas.values()), K=K,
            a_clean_pre=float(torch.relu(clean_pre[:, sl]).mean()),
            target_faith_pre=None if np.isnan(tf) else tf,
            rank_clean=float(rank_of(clean_pre).median()), rank_circuit=float(rank_of(circ_pre).median()),
            rank_empty=float(rank_of(empty_pre).median()),
            in_topk_clean=float(np.mean([sl in s_ for s_ in clean_sets])),
            in_topk_circuit=float(np.mean([sl in s_ for s_ in circ_sets])),
            jaccard_circuit=jac(circ_sets, clean_sets), jaccard_empty=jac(empty_sets, clean_sets),
            n_new_circuit=float(np.mean([len(a - b) for a, b in zip(circ_sets, clean_sets)])),
            n_new_empty=float(np.mean([len(a - b) for a, b in zip(empty_sets, clean_sets)])),
            switched_on_circuit=float(switched.median()), switched_on_empty=float(switched_empty.median()),
            n_siblings=len(siblings), n_siblings_scored=n_ok(sib_f), sibling_faith_median=med(sib_f),
            n_control=len(control), n_control_scored=n_ok(ctl_f), control_faith_median=med(ctl_f),
            specificity_gap=(None if np.isnan(tf) or med(sib_f) is None else tf - med(sib_f)),
            siblings=siblings[:50]))
    return rows


def summarise():
    import pandas as pd
    df = pd.DataFrame([json.loads(l) for l in open(OUT)])
    if "error" in df.columns:
        df = df[df["error"].isna()]
    lines = ["# 056 specificity diagnostic (%d rows)\n" % len(df)]

    def emit(s=""):
        print(s); lines.append(s)

    cols = ["target_faith_pre", "sibling_faith_median", "control_faith_median", "specificity_gap", "in_topk_circuit",
            "rank_circuit", "jaccard_circuit", "jaccard_empty", "n_new_circuit", "switched_on_circuit", "n_siblings"]
    for pi in PIS:
        emit("\n## %s ablation: medians over targets" % pi)
        emit(df[df.pi == pi].groupby("arm")[cols].median().reindex([a for a in ARMS if a in set(df.arm)]).round(3).to_string())
    emit("\n## Concept-amplifier flags (sibling faithfulness >= 0.8 x target faithfulness, target faithfulness >= 0.5)")
    fl = df[(df.target_faith_pre >= 0.5) & (df.sibling_faith_median >= 0.8 * df.target_faith_pre)]
    emit(fl[["arm", "seed", "pi", "target_faith_pre", "sibling_faith_median", "control_faith_median", "n_siblings",
             "in_topk_circuit", "switched_on_circuit"]].round(3).to_string(index=False) if len(fl) else "(none)")
    emit("\n## Per target (C ablation, first arm)")
    first = df[(df.pi == "C") & (df.arm == ARMS[0])]
    emit(first[["seed", "n_nodes", "target_faith_pre", "sibling_faith_median", "control_faith_median", "rank_clean",
                "rank_circuit", "in_topk_clean", "in_topk_circuit", "jaccard_circuit", "jaccard_empty",
                "switched_on_circuit"]].round(3).to_string(index=False))
    (RES / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    RES.mkdir(exist_ok=True)
    if os.environ.get("SUMMARY") == "1":
        summarise(); return
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("TAG", "specificity_unused")
    import amp_eval_pass_v2 as V
    G = V.setup()
    CleanTap, CircuitTap = make_patchers(G)
    seeds = [s for s in SEEDS_FILE.read_text().split() if s]
    done = set()
    if OUT.exists():
        done = {(r["arm"], r["seed"]) for r in map(json.loads, open(OUT)) if "pi" in r}   # failed targets are retried
    with open(OUT, "a") as fh:
        for arm in ARMS:
            circuits = load_circuits(arm, set(seeds), V)
            print("arm %s: %d / %d circuits" % (arm, len(circuits), len(seeds)), flush=True)
            for s in seeds:
                if (arm, s) in done or s not in circuits:
                    continue
                ts = time.time()
                try:
                    rows = score(G, circuits[s], V, CleanTap, CircuitTap, arm)
                except Exception as e:  # noqa: BLE001
                    rows = [dict(arm=arm, seed=s, error="%s: %s" % (type(e).__name__, str(e)[:300]))]
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                r0 = next((r for r in rows if r.get("pi") == "C"), rows[0])
                print("  %-8s %-15s %s  %.0fs" % (arm, s, r0.get("error") or "faith %s sib %s in_topk %s switched %s" % (
                    r0["target_faith_pre"], r0["sibling_faith_median"], r0["in_topk_circuit"], r0["switched_on_circuit"]),
                    time.time() - ts), flush=True)
                if SMOKE:
                    return
    summarise()


if __name__ == "__main__":
    main()
