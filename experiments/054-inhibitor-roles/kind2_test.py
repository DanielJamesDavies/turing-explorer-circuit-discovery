"""KIND-2 SUBSET TEST (DAN-72 follow-up): do CONTEXT-DETECTOR inhibitors form a real brake set?

Node inspection (results/node_report.md) found that most leave-one-out "inhibitory" nodes are ubiquitous hub
latents or redundant copies, while a smaller kind looks like a genuine inhibitor: latents that fire MORE on the
target's contrast contexts than on its activating contexts, and are not hubs. This script tests that kind alone,
on the circuits with leave-one-out roles in results/roles_pilot.jsonl.

  kind 2   L_C < 0 (inhibitory by leave-one-out under C) AND contrast-preferring (mean per-sequence max
           activation higher on the target's contrast contexts than on its activating contexts) AND not a hub
           (in < HUB_FRAC of all production circuits; 049 results/fanout_latents.csv).
  tests    on the 16 held-out activating contexts, circuit-only execution under C and A:
             release(kind 2)            removing the kind-2 set should RAISE the target;
             release(random control)    same-size random draws from the OTHER leave-one-out inhibitory nodes
                                        (N_DRAWS, median);
             release(hub inhibitors)    the hub nodes among the leave-one-out inhibitory nodes;
             additivity                 predicted (sum of single L_C effects) vs actual for kind 2;
           plus a semantic check over ALL non-hub nodes: share inhibitory among contrast-preferring vs the rest.

  PYTHONPATH=src python experiments/054-inhibitor-roles/kind2_test.py   -> results/kind2.jsonl + summary
  env: HUB_FRAC (0.01), N_DRAWS (5), SUMMARY=1
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent / "049-circuit-graph"
RES = HERE / "results"
OUT_FILE = RES / os.environ.get("KIND2_OUT", "kind2.jsonl")
HUB_FRAC = float(os.environ.get("HUB_FRAC", 0.01))
N_DRAWS = int(os.environ.get("N_DRAWS", 5))


def hub_fracs():
    import pandas as pd
    f = pd.read_csv(ROOT / "results" / "fanout_latents.csv", usecols=["latent", "frac_of_circuits"])
    return dict(zip(f.latent, f.frac_of_circuits))


def score(row, c, fracs):
    import torch
    import amp_eval_pass_v2 as V
    from eval.ablation_faithfulness import upstream_sites
    from eval.floors import collect_site_anchors, collect_site_means
    from pipeline.component_index import component_idx as comp_of
    from sae.dense import sparse_topk_to_dense

    G = V.G
    inference, bank, M0, KINDS, NK, D, K = (G[k] for k in ("inference", "bank", "M0", "KINDS", "NK", "D", "K"))
    TapCO, _ = V._engine_classes()
    k2i = {k: i for i, k in enumerate(KINDS)}
    ts = time.time()
    seed = V.seed_of(c); sf = seed.metadata["feature_id"]; layer, kind, sl = sf.layer, sf.kind, sf.index
    alphas = {}
    for n in c.nodes.values():
        if n is not seed:
            f = n.metadata["feature_id"]
            alphas.setdefault((f.layer, f.kind), {})[int(f.index)] = float(n.metadata.get("amplitude", 1.0))
    SITES = sorted(alphas)
    nodes = [(s, i) for s in SITES for i in sorted(alphas[s])]
    names = ["%d.%s.%d" % (s[0], s[1], i) for s, i in nodes]
    pd_ = M0.build_probe_dataset(comp_of(layer, KINDS.index(kind), NK), sl)
    pt, pa, nt = pd_.pos_tokens[:V.N_SEQ], pd_.pos_argmax[:V.N_SEQ], pd_.neg_tokens[:V.N_SEQ]
    n_tr = V.split_n(int(pt.shape[0])); n_ntr = V.split_n(int(nt.shape[0]))
    pt_tr, pa_tr, pt_ho, pa_ho = pt[:n_tr], pa[:n_tr], pt[n_tr:], pa[n_tr:]
    sae = bank.saes[kind][layer]; w_seed = sae.encoder.weight[sl].detach(); b_seed = sae._get_bias_eff()[sl].detach()
    UPS = set(upstream_sites(bank, layer, kind)); site = (layer, kind); dev = pt.device
    means = {"A": collect_site_anchors(inference, bank, pt_tr, UPS, pa_tr, pin_position_specific=False)[0],
             "C": collect_site_means(inference, bank, nt[:n_ntr], UPS)}
    SC = {}
    for s in SITES:
        sv = torch.ones(D, device=dev, dtype=torch.float32)
        sv[torch.tensor(sorted(alphas[s]), device=dev)] = torch.tensor([alphas[s][i] for i in sorted(alphas[s])], device=dev, dtype=torch.float32)
        SC[s] = sv

    def co(drop, pi):
        keep_ = {s: set(alphas[s]) for s in SITES}
        for j in drop:
            s, i = nodes[j]; keep_[s].discard(i)
        keep_ = {s: v for s, v in keep_.items() if v}
        kt = {s: torch.tensor(sorted(v), device=dev, dtype=torch.long) for s, v in keep_.items()}
        return V.read(lambda a, b: TapCO(bank=bank, keep_indices=keep_, in_scope=UPS, seed_layer=layer, seed_kind=kind,
                                         seed_latent_idx=sl, pos_argmax=pa_ho[a:b], site_means=means[pi], topk=K, keep_tensors=kt,
                                         keep_scales={s: SC[s] for s in keep_} or None, w_seed=w_seed, b_seed=b_seed), pt_ho, pa_ho)["pre"]

    # ---- every node's firing on the target's activating vs contrast contexts (mean per-sequence max)
    want = {}
    for j, (s, i) in enumerate(nodes):
        want.setdefault(s, []).append((j, i))
    fire = np.zeros((len(nodes), 2))
    for col, tokens in enumerate((pt, nt)):
        cap = {s: [] for s in want}

        def hook(layer_idx, activations):
            for kd in KINDS:
                s = (layer_idx, kd)
                if s in want:
                    ta, ti = bank.encode(activations[k2i[kd]], kd, layer_idx)
                    idx = torch.tensor([i for _, i in want[s]], device=ta.device)
                    cap[s].append(sparse_topk_to_dense(ta, ti, D, dtype=torch.float32)[..., idx].amax(dim=1).cpu())
        inference.disable_compile()
        try:
            with torch.no_grad():
                for s0 in range(0, int(tokens.shape[0]), 16):
                    inference.forward(tokens[s0:s0 + 16], activations_callback=hook, return_activations=False, tokenize_final=False)
        finally:
            inference.enable_compile()
        for s, lst in want.items():
            A = torch.cat(cap[s], 0).mean(0)
            for c_, (j, _) in enumerate(lst):
                fire[j, col] = float(A[c_])

    lc = np.array(row["vals"]["L_C"]) / row["a_pos_pre"]
    hub = np.array([fracs.get(nm, 0.0) >= HUB_FRAC for nm in names])
    ctr_pref = fire[:, 1] > fire[:, 0]
    inh = lc < 0
    k2 = np.where(inh & ctr_pref & ~hub)[0].tolist()
    other_inh = np.where(inh & ~(ctr_pref & ~hub))[0].tolist()
    hub_inh = np.where(inh & hub)[0].tolist()
    a_pos = row["a_pos_pre"]
    out = dict(seed=row["seed"], layer=layer, kind=kind, n=len(nodes), n_inh=int(inh.sum()), n_k2=len(k2), n_hub_inh=len(hub_inh),
               n_ctr_pref_nonhub=int((ctr_pref & ~hub).sum()),
               inh_share_ctr_pref_nonhub=float(inh[ctr_pref & ~hub].mean()) if (ctr_pref & ~hub).any() else None,
               inh_share_rest_nonhub=float(inh[~ctr_pref & ~hub].mean()) if (~ctr_pref & ~hub).any() else None,
               inh_share_hub=float(inh[hub].mean()) if hub.any() else None,
               k2_nodes=[names[j] for j in k2])
    rng = random.Random(row["seed"])
    for pi in ("C", "A"):
        full = co([], pi)
        out["k2_release_" + pi] = (co(k2, pi) - full) / a_pos if k2 else None
        out["k2_predicted_" + pi] = float(-lc[k2].sum()) if k2 else None     # additivity: sum of single effects (L_C)
        draws = []
        if k2 and len(other_inh) >= len(k2):
            for _ in range(N_DRAWS):
                draws.append((co(rng.sample(other_inh, len(k2)), pi) - full) / a_pos)
        out["ctrl_release_" + pi] = float(np.median(draws)) if draws else None
        out["hub_release_" + pi] = (co(hub_inh, pi) - full) / a_pos if hub_inh else None
    out["secs"] = round(time.time() - ts, 1)
    return out


def summarise():
    rows = [json.loads(l) for l in open(OUT_FILE)]
    rows = [r for r in rows if "error" not in r]
    med = lambda xs: float(np.median(xs)) if xs else float("nan")
    print("circuits %d | hub threshold: in >= %.3f of production circuits" % (len(rows), HUB_FRAC))
    print("kind-2 set size: median %d (of median %d inhibitory, %d nodes); circuits with >= 1 kind-2 node: %d"
          % (med([r["n_k2"] for r in rows]), med([r["n_inh"] for r in rows]), med([r["n"] for r in rows]), sum(r["n_k2"] > 0 for r in rows)))
    print("\n=== semantic check (non-hub nodes): share inhibitory by leave-one-out under C ===")
    for k, lab in (("inh_share_ctr_pref_nonhub", "fire more on contrast contexts"), ("inh_share_rest_nonhub", "fire more on activating contexts"),
                   ("inh_share_hub", "hub latents (for reference)")):
        xs = [r[k] for r in rows if r.get(k) is not None]
        print("  %-34s median %.2f  (n=%d)" % (lab, med(xs), len(xs)))
    print("\n=== set-level release on held-out contexts (share of a_pos; > 0 = target rises) ===")
    for pi in ("C", "A"):
        k2 = [r for r in rows if r.get("k2_release_" + pi) is not None]
        rel = [r["k2_release_" + pi] for r in k2]
        ctrl = [r["ctrl_release_" + pi] for r in k2 if r.get("ctrl_release_" + pi) is not None]
        hub = [r["hub_release_" + pi] for r in rows if r.get("hub_release_" + pi) is not None]
        pred = [r["k2_predicted_" + pi] for r in k2]
        print("  under %s:" % pi)
        print("    kind 2            median %+.4f, rises in %.2f of circuits (n=%d)" % (med(rel), np.mean(np.array(rel) > 0), len(rel)))
        print("    random control    median %+.4f, rises in %.2f  (same-size draws from the other inhibitory nodes, n=%d)"
              % (med(ctrl), np.mean(np.array(ctrl) > 0) if ctrl else float("nan"), len(ctrl)))
        print("    hub inhibitors    median %+.4f, rises in %.2f (n=%d)" % (med(hub), np.mean(np.array(hub) > 0) if hub else float("nan"), len(hub)))
        print("    additivity (kind 2): predicted median %+.4f vs actual %+.4f" % (med(pred), med(rel)))


def main():
    if os.environ.get("SUMMARY") == "1":
        summarise(); return
    rows = [json.loads(l) for l in open(RES / "roles_pilot.jsonl")]
    rows = [r for r in rows if r.get("loo") and "L_C" in r.get("vals", {})]
    os.environ["SEEDS"] = ",".join(r["seed"] for r in rows)
    os.environ.setdefault("TAG", "kind2_unused")
    sys.path.insert(0, str(ROOT))
    import amp_eval_pass_v2 as V
    items, _, _, _ = V.load_items()
    V.setup()
    fracs = hub_fracs()
    by_key = {V.key_of(c): c for c in items}
    print("kind-2 test | %d circuits | hub >= %.3f" % (len(rows), HUB_FRAC), flush=True)
    with open(OUT_FILE, "w") as fh:
        for r in rows:
            try:
                o = score(r, by_key[r["seed"]], fracs)
            except Exception as e:  # noqa: BLE001
                o = dict(seed=r["seed"], error="%s: %s" % (type(e).__name__, str(e)[:300]))
            fh.write(json.dumps(o) + "\n"); fh.flush()
            print("  %-16s %s" % (r["seed"], o.get("error") or "k2 %d / inh %d | release C %s ctrl %s | %.0fs"
                  % (o["n_k2"], o["n_inh"], o["k2_release_C"], o["ctrl_release_C"], o["secs"])), flush=True)
    summarise()


if __name__ == "__main__":
    main()
