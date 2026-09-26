"""Is the train / held-out split ordered? (DAN-12 / DAN-15). Read-only: no fitting, no writes outside results/.

The engine uses contexts[:48] for training and contexts[48:] as held-out. The code says both lists arrive sorted:
activating contexts are top_ctx (kept by torch.topk, descending) then mid_ctx, capped at 64; close contrast
contexts are ranked by similarity. This measures it.

  (a) top_ctx.ctx_seq_val per target: is the stored order monotone non-increasing?
  (b) the target's activation at its anchor (per-sequence max) on contexts[:48] vs [48:], and on the stratified
      alternative (rank by that activation, every 4th to held-out); Spearman of list position vs activation
  (c) how many targets have fewer than 64 valid top contexts, i.e. how often mid-band contexts enter at all
      (over the 15k production seed list and over every latent in the bank)
  (d) close contrast contexts: max cosine to the activating set on contexts[:48] vs [48:], and monotonicity

Targets: the 16 pilot targets (055 seeds.txt) plus N_SAMPLE from the 15k production seed list (seeded).

  PYTHONPATH=src python experiments/058-split-ordering/check_split.py   -> results/check_split.md (+ .jsonl)
  env: N_SAMPLE (200)  N_NEG_SAMPLE (48; (d) is slower, it runs the silence check)  SEED (0)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent / "049-circuit-graph"
E055 = HERE.parent / "055-close-contrast"
RES = HERE / "results"
N_SAMPLE = int(os.environ.get("N_SAMPLE", 200))
N_NEG_SAMPLE = int(os.environ.get("N_NEG_SAMPLE", 48))
RNG = np.random.default_rng(int(os.environ.get("SEED", 0)))
N_SEQ, N_TR = 64, 48


def spearman(x):
    """Spearman correlation of list position (0 = first) with the values; negative = descending order."""
    if len(x) < 3:
        return None
    r = np.argsort(np.argsort(np.asarray(x)))
    return float(np.corrcoef(np.arange(len(x)), r)[0, 1])


def stratified_heldout(n, n_hold):
    """Held-out positions under the stratified split: rank by strength, every (n // n_hold)-th to held-out."""
    step = n / n_hold
    return sorted({int(i * step + step - 1) for i in range(n_hold)})


def main():
    RES.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("TAG", "split_unused")
    import pandas as pd
    import amp_eval_pass_v2 as V
    from pipeline.component_index import component_idx as comp_of
    from sae.dense import target_latent_activations

    G = V.setup()
    inference, bank, M0, KINDS, NK = (G[k] for k in ("inference", "bank", "M0", "KINDS", "NK"))
    from store.context import top_ctx
    sel = M0._neg_context_selector()

    ct = pd.read_parquet(ROOT / "tables_full" / "circuits.parquet")
    prod = sorted(set(ct.seed_layer.astype(str) + "." + ct.seed_kind.astype(str) + "." + ct.seed_index.astype(str)))
    pilot = [s for s in (E055 / "seeds.txt").read_text().split() if s]
    sample = [s for s in RNG.choice([s for s in prod if s not in pilot], size=min(N_SAMPLE, len(prod)), replace=False)]
    parse = lambda s: (int(s.split(".")[0]), s.split(".")[1], int(s.split(".")[2]))

    # (c) over the whole production list and the whole bank: valid top contexts per latent
    valid = (top_ctx.ctx_seq_idx > 0).sum(-1).cpu()                                  # [n_comp, d_sae]
    prod_valid = torch.tensor([int(valid[comp_of(l, KINDS.index(k), NK), i]) for l, k, i in map(parse, prod)])
    fired = valid > 0

    def target_acts(l, k, i, tokens):
        """Per-sequence max activation of the target (the anchor is the argmax)."""
        out = []

        def hook(layer_idx, activations):
            if layer_idx == l:
                ta, ti = bank.encode(activations[KINDS.index(k)], k, layer_idx)
                out.append(target_latent_activations(ta, ti, i).float().max(-1).values.cpu())
        inference.disable_compile()
        try:
            with torch.no_grad():
                for s0 in range(0, int(tokens.shape[0]), 16):
                    inference.forward(tokens[s0:s0 + 16], activations_callback=hook, return_activations=False,
                                      tokenize_final=False)
        finally:
            inference.enable_compile()
        return torch.cat(out).numpy()

    rows = []
    targets = [(s, "pilot") for s in pilot] + [(s, "sample") for s in sample]
    neg_keys = set(pilot) | set(sample[:N_NEG_SAMPLE])
    for n_, (key, group) in enumerate(targets):
        l, k, i = parse(key); comp = comp_of(l, KINDS.index(k), NK)
        r = dict(seed=key, group=group, n_valid_top=int(valid[comp, i]))
        try:
            # (a) stored order
            vals = top_ctx.ctx_seq_val[comp, i].float().cpu().numpy()
            v = vals[top_ctx.ctx_seq_idx[comp, i].cpu().numpy() > 0]
            r["store_monotone"] = bool(np.all(np.diff(v) <= 1e-6))
            r["store_n_violations"] = int((np.diff(v) > 1e-6).sum())
            # (b) activation on the engine's contexts
            pd_ = M0.build_probe_dataset(comp, i)
            pt = pd_.pos_tokens[:N_SEQ]
            a = target_acts(l, k, i, pt)
            n = len(a); ntr = V.split_n(n)
            r.update(n_ctx=n, act_train=float(a[:ntr].mean()), act_held=float(a[ntr:].mean()),
                     act_ratio=float(a[ntr:].mean() / max(a[:ntr].mean(), 1e-9)), act_spearman=spearman(a),
                     act_min_train=float(a[:ntr].min()), act_max_held=float(a[ntr:].max()))
            order = np.argsort(-a); ho = set(order[stratified_heldout(n, n - ntr)].tolist())
            tr = [j for j in range(n) if j not in ho]
            r.update(strat_act_train=float(a[tr].mean()), strat_act_held=float(a[sorted(ho)].mean()),
                     strat_ratio=float(a[sorted(ho)].mean() / max(a[tr].mean(), 1e-9)))
            # (d) close contrast contexts
            if key in neg_keys:
                s_ = sel.select(comp, i, "close", max_sequences=N_SEQ, batch_size=16, exact=False,
                                non_activation_threshold=0.0, filter_batch_size=32, load_window_size=256)
                ref, _ = sel.topctx_reference_reprs(comp, i)
                dev = sel._ranking_device()
                sims = sel._positive_set_max_similarity(sel._get_repr_for_ids(s_.sequence_ids),
                                                        sel._normalized_reference_reps(ref, dev), dev).numpy()
                m = len(sims); mtr = V.split_n(m)
                r.update(neg_n=m, sim_train=float(sims[:mtr].mean()), sim_held=float(sims[mtr:].mean()),
                         sim_spearman=spearman(sims), sim_monotone=bool(np.all(np.diff(sims) <= 1e-6)))
        except Exception as e:  # noqa: BLE001
            r["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        rows.append(r)
        if n_ % 20 == 0:
            print("%d/%d %s" % (n_ + 1, len(targets), key), flush=True)

    with open(RES / "check_split.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    df = pd.DataFrame(rows)
    ok = df[df.get("error").isna()] if "error" in df.columns else df
    lines = ["# Split ordering check (DAN-12 / DAN-15)\n",
             "Targets: %d pilot + %d sampled from the %d production seeds; %d errors.\n"
             % (len(pilot), len(sample), len(prod), len(df) - len(ok))]
    lines.append("## (a) Stored top-context order\n")
    lines.append("- monotone non-increasing: %d / %d targets (median violations %.0f)"
                 % (int(ok.store_monotone.sum()), len(ok), ok.store_n_violations.median()))
    lines.append("\n## (b) Target activation at its anchor, engine split (first 48 train / last 16 held-out)\n")
    for g in ("pilot", "sample"):
        x = ok[ok.group == g]
        lines.append("- **%s (n=%d):** held-out / train mean activation, median %.3f (IQR %.3f–%.3f); held-out "
                     "weaker on %d / %d; position-vs-activation Spearman median %.2f; every held-out context "
                     "below the weakest training context on %d / %d"
                     % (g, len(x), x.act_ratio.median(), x.act_ratio.quantile(.25), x.act_ratio.quantile(.75),
                        int((x.act_ratio < 1).sum()), len(x), x.act_spearman.median(),
                        int((x.act_max_held <= x.act_min_train).sum()), len(x)))
        lines.append("  - stratified split instead: held-out / train median %.3f (IQR %.3f–%.3f)"
                     % (x.strat_ratio.median(), x.strat_ratio.quantile(.25), x.strat_ratio.quantile(.75)))
    lines.append("\n## (c) How often mid-band contexts enter (fewer than 64 valid top contexts)\n")
    lines.append("- production seeds: %d / %d (%.1f%%) have < 64; median valid %d"
                 % (int((prod_valid < N_SEQ).sum()), len(prod_valid), 100 * float((prod_valid < N_SEQ).float().mean()),
                    int(prod_valid.median())))
    lines.append("- whole bank, latents that ever fired: %d / %d (%.1f%%) have < 64"
                 % (int(((valid < N_SEQ) & fired).sum()), int(fired.sum()),
                    100 * float(((valid < N_SEQ) & fired).sum()) / max(1, int(fired.sum()))))
    if "sim_train" in ok.columns:
        x = ok[ok.sim_train.notna()]
        lines.append("\n## (d) Close contrast contexts: max cosine to the activating set, first 48 vs last 16\n")
        lines.append("- n=%d: train %.3f vs held-out %.3f (medians of per-target means); held-out less similar on "
                     "%d / %d; monotone %d / %d; Spearman median %.2f"
                     % (len(x), x.sim_train.median(), x.sim_held.median(), int((x.sim_held < x.sim_train).sum()),
                        len(x), int(x.sim_monotone.sum()), len(x), x.sim_spearman.median()))
    txt = "\n".join(lines)
    print(txt)
    (RES / "check_split.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
