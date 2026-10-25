"""C2 (DAN-159): the Ge et al. (2024) hierarchical-attribution arm under protocol v1, tau bisected to matched size,
every bisection point kept (free DAN-77 curve points), scored by the production v1 scorer.

The discovery is 034-ge-replication/runner.py's, unchanged: GeHierInstrument (every upstream site rewritten as
decode(code) + detached error; latents gated out at already-processed downstream sites detached, the site-level form
of Ge's during-backward stop), hier_run (sites downstream -> upstream, attribution act x grad of the target's
pre-activation at the anchors, summed over contexts and positions, gate |attr| >= tau), and the log-space tau
bisection (bracket 1e-8 .. 1e2, 7 steps, +-10% of the target size; tau = 0 first as the all-active reference). What
changes is the protocol around it:
  - contexts: the target's v1 TRAINING contexts (the rows driver.Runner.fit trains on), not 034's 48 pre-v1 probes;
  - target size: the target's production WCM circuit (062 sweep W_0.001), not the pre-v1 panel's triamp400 size;
  - scoring: driver.Runner.score on held-out strongest (034's AmpCircuitPatcher, deleted by DAN-67, left every
    site with no member live);
  - resumable mid-bisection: every sweep is saved (ge/points/<key>.<train hash>.pt) and the bisection replays from
    it (cmp_common.ge_plan); contexts come through driver.Runner.contexts (the driver's CTX_DIR applies).
Every point with n <= SCORE_MAX_N is scored (SCORE_POINTS=all) or only the returned one (SCORE_POINTS=final).

  CUDA_VISIBLE_DEVICES=0 SHARD=0/1 PYTHONPATH=src python experiments/083-v1-comparison/ge_hier.py
  env: BISECT_STEPS (7)  SIZE_TOL (0.10)  SCORE_POINTS (all | final)  SCORE_MAX_N (20000)  TARGETS  LIMIT  SHARD
"""
import os
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cmp_common as C  # noqa: E402

BISECT_STEPS = int(os.environ.get("BISECT_STEPS", 7))
SIZE_TOL = float(os.environ.get("SIZE_TOL", 0.10))
SCORE_POINTS = os.environ.get("SCORE_POINTS", "all")
SCORE_MAX_N = int(os.environ.get("SCORE_MAX_N", 20000))


def log(s):
    print("[%s %s] %s" % (time.strftime("%H:%M:%S"), C.TAG, s), flush=True)


class GeHierInstrument:
    """034's reverse-causal-sweep hierarchical attribution instrument (verbatim; the bank is passed in).

    Every upstream site's stream is rewritten as decode(code) + detached error, so all gradient flows through SAE
    codes (errors detached, as Ge prescribes). Latents pruned at already-processed downstream sites have their code
    coordinates detached, gating gradient flow exactly as Ge's during-backward stop. The current score site's code
    retains grad for the attribution readout."""

    def __init__(self, bank, sites, keep_masks, score_site, seed_site, w_seed, b_seed):
        self.bank = bank
        self.sites = set(sites)
        self.keep_masks = keep_masks      # {site: float mask [d_sae]} 1=keep
        self.score_site = score_site
        self.seed_site = seed_site
        self.w_seed, self.b_seed = w_seed, b_seed
        self.seed_pre = None
        self.scored = None                # dense_g at score_site (retains grad)

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        from sae.dense import sparse_topk_to_dense
        bank = self.bank
        site = (layer_idx, kind)
        if site == self.seed_site:
            w = self.w_seed.to(device=x.device, dtype=x.dtype)
            b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre = x @ w + b
            return x
        if site not in self.sites:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        keep = self.keep_masks.get(site)
        if keep is None:
            dense_g = dense
        else:
            k = keep.to(device=dense.device, dtype=dense.dtype)
            dense_g = dense * k + dense.detach() * (1.0 - k)
        if site == self.score_site:
            dense_g.retain_grad()
            self.scored = dense_g
        recon = bank.decode(dense_g, kind, layer_idx, add_bias=False)
        recon_ref = bank.decode(dense.detach(), kind, layer_idx, add_bias=False)
        err = (x - recon_ref).detach()
        return (recon + err).to(x.dtype)


def reverse_order(kinds, sites):
    """Downstream -> upstream: later layers first; within a layer, later-in-causal-order kinds first."""
    return sorted(sites, key=lambda s: (s[0], list(kinds).index(s[1])), reverse=True)


def hier_run(G, tau, UP, seed_site, w_seed, b_seed, pt_tr, pa_tr, disc_bs):
    """034's hier_run: one full hierarchical sweep at threshold tau. Returns keep_masks (float [d_sae] per site,
    CPU) and the membership size."""
    inference, bank, dev = G["inference"], G["bank"], G["device"]
    keep_masks = {}
    inference.disable_compile()
    try:
        for site in reverse_order(G["KINDS"], UP):
            attr = torch.zeros(bank.d_sae, device=dev)
            for s0 in range(0, int(pt_tr.shape[0]), disc_bs):
                tk = pt_tr[s0:s0 + disc_bs]
                pa = pa_tr[s0:s0 + disc_bs]
                inst = GeHierInstrument(bank, UP, keep_masks, site, seed_site, w_seed, b_seed)
                with torch.enable_grad():
                    inference.forward(tk, patcher=inst, grad_enabled=True, return_activations=False,
                                      tokenize_final=False)
                    pre = inst.seed_pre
                    B = pre.shape[0]
                    rr = torch.arange(B, device=pre.device)
                    anc = pa.to(pre.device).clamp(0, pre.shape[1] - 1)
                    metric = pre[rr, anc].mean()
                    metric.backward()
                if inst.scored is not None and inst.scored.grad is not None:
                    g = inst.scored.grad
                    attr += (inst.scored.detach() * g).sum(dim=(0, 1)).float().to(dev)
                del inst
            keep_masks[site] = (attr.abs() >= tau).float().cpu()
        n = int(sum(int(m.sum()) for m in keep_masks.values()))
        return keep_masks, n
    finally:
        inference.enable_compile()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def discover(G, key, pos, arg, n_target, path):
    """Bisect tau to n_target, resuming from the saved points. Returns (points, final index)."""
    from eval.ablation_faithfulness import upstream_sites
    layer, kind, i = C.parse(key)
    UP = C.site_order(G["KINDS"], upstream_sites(G["bank"], layer, kind))
    disc_bs = 2 if len(UP) >= 25 else 4                      # 034's depth-aware batch (result-identical ranking)
    sae = G["bank"].saes[kind][layer]
    w_seed, b_seed = sae.encoder.weight[i].detach(), sae._get_bias_eff()[i].detach()
    state = torch.load(path, weights_only=False) if path.exists() else dict(
        key=key, n_target=int(n_target), n_sites=len(UP), disc_bs=disc_bs, steps=BISECT_STEPS, tol=SIZE_TOL, points=[])
    if state["n_target"] != int(n_target) or state.get("steps") != BISECT_STEPS or state.get("tol") != SIZE_TOL:
        raise ValueError("%s: saved Ge points were bisected for n_target %s / steps %s / tol %s" % (
            key, state["n_target"], state.get("steps"), state.get("tol")))
    while True:
        what, x = C.ge_plan(state["points"], int(n_target), BISECT_STEPS, SIZE_TOL)
        if what == "done":
            return state, x
        t = time.time()
        masks, n = hier_run(G, x, UP, (layer, kind), w_seed, b_seed, pos, arg, disc_bs)
        members = None
        if n <= SCORE_MAX_N:
            members = {s: m.nonzero(as_tuple=True)[0].numpy().astype("int32") for s, m in masks.items() if m.sum() > 0}
        state["points"].append(dict(tau=float(x), n=int(n), members=members, secs=round(time.time() - t, 1)))
        C.save_ranking_file(path, state)
        log("  %-16s ge point %d: tau %.3e n %d (target %d) %.0fs" % (key, len(state["points"]) - 1, x, n, n_target,
                                                                       time.time() - t))


def main():
    driver = C.import_driver()
    targets = C.shard_targets()
    sizes = C.sizes_by_arm(C.sweep_rows())
    d = C.OUT / "ge"
    d.mkdir(parents=True, exist_ok=True)
    rows_path = d / ("rows.%s.jsonl" % C.TAG)
    done = C.done_keys(rows_path, ("seed", "point", "ctx_dir"))
    log("ge hierarchical | %d targets | bisect %d steps, tol %.2f | score %s (n <= %d)"
        % (len(targets), BISECT_STEPS, SIZE_TOL, SCORE_POINTS, SCORE_MAX_N))
    R = driver.Runner()
    ctx_dir = str(R.ctx_dir)
    fh = open(rows_path, "a")
    for n_, key in enumerate(targets):
        n_target = sizes.get(key, {}).get(C.REF_ARM)
        if n_target is None:
            continue
        rec = R.contexts(key)
        if rec.get("strong") is None:
            continue
        arm, thin, pos, arg, h = C.training_contexts(R, rec)
        try:
            state, final = discover(R.G, key, pos, arg, n_target, d / "points" / ("%s.%s.pt" % (key, h)))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            C.write_row(fh, dict(seed=key, point=None, method="ge_hier", error="%s: %s" % (type(e).__name__, str(e)[:300])))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        secs = round(sum(p["secs"] for p in state["points"]), 1)
        for j, p in enumerate(state["points"]):
            if (key, j, ctx_dir) in done or (SCORE_POINTS == "final" and j != final):
                continue
            row = dict(seed=key, point=j, method="ge_hier", ranking="bisect", size_arm=C.REF_ARM if j == final else "tau",
                       final=j == final, tau=p["tau"], n_target=int(n_target), n_sites=state["n_sites"],
                       disc_bs=state["disc_bs"], t_discover=secs, train_arm=arm, thin=thin, held="strong", shard=C.TAG,
                       **C.provenance(R, h))
            if p["members"] is None:
                row.update(skip="too_large", n=p["n"])
            else:
                try:
                    t = time.time()
                    members = {s: [int(x) for x in v] for s, v in p["members"].items()}
                    r = C.scalar_row(R.score(C.make_circuit(key, members), rec, "strong"))
                    r.update(row)
                    r["t_score"] = round(time.time() - t, 1)
                    row = r
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            C.write_row(fh, row)
            log("%4d/%d %-16s point %d%s tau %.2e n=%s/%d Z %s A %s C %s nec %s" % (
                n_ + 1, len(targets), key, j, " (final)" if j == final else "", p["tau"], row.get("n"), n_target,
                *("%.2f" % row[c] if isinstance(row.get(c), float) else "-" for c in C.HEAD + [C.NEC])))
    log("DONE")


if __name__ == "__main__":
    main()
