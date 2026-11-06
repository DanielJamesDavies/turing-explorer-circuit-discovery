"""Arena C (DAN-194), Finding 2 diagnostic: why the zero-ablated empty circuit (the scorer's e0) explodes on latent
targets of the Marks dictionaries. Forward passes only, no fits.

Panel: ~2 latents per (layer 0-5) x {attn, mlp, resid} site with full top + mid stores in the smoke data pass, seeded.
Per target, on its protocol contexts (eval_probe(rec, "strong"): read at the held-out strongest anchors, as the scorer):
  clean       the target pre-activation (median over anchors)
  e0          every upstream code zeroed, errors kept (the scorer's empty Z circuit)
  e0_<kind>   only that kind's upstream sites zeroed (embed / attn / mlp / resid)
  e0_pos0     e0 with position 0 left clean;  e0_noembed  e0 without the embed site
  vacuous     protocol section 12.4's rule (eval.wcm_eval.vacuous_v2) on the post-ReLU reads (= the Top-K read at
              k = d_sae): a_pos, e0, and the dense A / C empty circuits (eM / eN: every upstream code at the scorer's A /
              C mean, eval_reference rows; with k = d_sae the sparsity-preserving fill is the dense fill)
  ln          per layer <= target layer: the LayerNorm input std (input_layernorm; GPT-NeoX's parallel LNs both read the
              layer input), min over positions and at the anchors, clean vs e0

  FAMILY etc. are set here; run in WSL from the repo root:
  HF_HUB_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python experiments/107-true-baselines/arena_c/diag_zero_empty.py
"""
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STORES = HERE / "out" / "datapass_smoke"
OUT = HERE / "out" / "diag_z"
os.environ.update(FAMILY="pythia-70m-deduped-marks", DATA=str(ROOT / "data_public" / "pythia-70m"),
                  STORES=str(STORES), OUT=str(OUT), MODE="main", WCM_VERSION="v3", HF_HUB_OFFLINE="1")
sys.path.insert(0, str(HERE))
import driver_marks  # noqa: E402,F401  (registers the family, imports 062's driver)
import torch  # noqa: E402

import wcm_marks as WM  # noqa: E402

driver = driver_marks.driver
PER_SITE = int(os.environ.get("PER_SITE", 2))
SEED = 20261101


def panel():
    if os.environ.get("KEYS"):
        return [k for k in os.environ["KEYS"].split(",") if k]
    top =torch.load(STORES / "top_ctx.pt", weights_only=False)["ctx_seq_idx"].cpu()
    mid = torch.load(STORES / "mid_ctx.pt", weights_only=False)["ctx_seq_idx"].cpu()
    rng = random.Random(SEED)
    keys = []
    for l in range(6):
        for k in ("attn", "mlp", "resid"):
            c = l * len(WM.KINDS) + WM.KINDS.index(k)
            ok = (((top[c] > 0).sum(-1) >= 64) & ((mid[c] > 0).sum(-1) >= 48)).nonzero().flatten().tolist()
            keys += ["%d.%s.%d" % (l, k, i) for i in rng.sample(ok, min(PER_SITE, len(ok)))]
    return keys


class Tap:
    """Read the target's pre-activation at its site; at `zero` sites replace every code by `fill` (0 or a mean),
    errors kept (x <- x + (fill - c) W_D^T), optionally leaving position 0 clean."""

    def __init__(self, bank, site, w, b, zero=(), fill=None, keep_pos0=False):
        self.bank, self.site, self.w, self.b = bank, site, w, b
        self.zero, self.fill, self.keep_pos0 = set(zero), fill or {}, keep_pos0
        self.pre = None

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.t)

    def t(self, l, k, x):
        if (l, k) == self.site:
            self.pre = x @ self.w.to(x.dtype) + self.b.to(x.dtype)
            return x
        if (l, k) not in self.zero:
            return x
        ta, _ = self.bank.encode(x, k, l)
        f = self.fill.get((l, k))
        d = (-ta) if f is None else (f.to(ta.dtype).view(1, 1, -1) - ta)
        if self.keep_pos0:
            d = d.clone()
            d[:, 0] = 0
        return x + self.bank.decode(d, k, l, add_bias=False).to(x.dtype)


def run_tap(inf, tap, toks, anchors, ln_layers=None):
    """(pre at the anchors [N], LN-input std per layer {l: (min over positions, median at the anchors)})."""
    stds, hs = {}, []
    if ln_layers is not None:
        for l in ln_layers:
            ln = inf.model.hf.gpt_neox.layers[l].input_layernorm

            def pre(_m, args, _l=l):
                stds[_l] = args[0].float().std(-1, unbiased=False)
            hs.append(ln.register_forward_pre_hook(pre))
    try:
        with torch.no_grad():
            inf.forward(toks, patcher=tap, return_activations=False, tokenize_final=False)
    finally:
        for h in hs:
            h.remove()
    rr = torch.arange(toks.shape[0], device=tap.pre.device)
    a = tap.pre[rr, anchors.to(tap.pre.device)].float()
    ln = {l: (float(s.min()), float(s[rr, anchors.to(s.device)].median())) for l, s in stds.items()}
    return a, ln


def main():
    from circuit.protocol_contexts import eval_probe, eval_reference, skip_reason
    from eval.ablation_faithfulness import upstream_sites
    from eval.floors import collect_site_means
    from eval.wcm_eval import eval_split, vacuous_v2
    R = driver.Runner()
    inf, bank, dev = R.G["inference"], R.G["bank"], R.G["device"]
    OUT.mkdir(parents=True, exist_ok=True)
    # MARKS_EMBED=0 (the production default since 2026-11-01): (0, embed) is a NullSAE, so "upstream" excludes the
    # embedding exactly as the driver's latent path now does; MARKS_EMBED=1 reproduces the first panel
    fh = open(OUT / os.environ.get("RESULTS", "results.jsonl"), "w")
    rows = []
    for key in panel():
        rec = R.contexts(key)
        if rec.get("strong") is None or skip_reason(rec) is not None:
            print("%s: no usable record (%s)" % (key, skip_reason(rec)))
            continue
        l, k, i = (int(key.split(".")[0]), key.split(".")[1], int(key.split(".")[2]))
        site = (l, k)
        w, b = bank.preact_vector(k, l, i)
        pd_ = eval_probe(rec, "strong", dev)
        n_tr = eval_split(int(pd_.pos_tokens.shape[0]), pd_.metadata, "explicit")
        toks, anch = pd_.pos_tokens[n_tr:], pd_.pos_argmax[n_tr:]
        ups = sorted(upstream_sites(bank, l, k), key=lambda s: (s[0], WM.KINDS.index(s[1])))
        real_ups = [s for s in ups if s in set(WM.REAL_SITES) and not getattr(bank.saes[s[1]][s[0]], "null", False)]
        lnl = list(range(0, l + 1))
        res = dict(key=key, n_ho=int(toks.shape[0]), n_up=len(real_ups))
        a_clean, ln_c = run_tap(inf, Tap(bank, site, w, b), toks, anch, lnl)
        a_e0, ln_z = run_tap(inf, Tap(bank, site, w, b, real_ups), toks, anch, lnl)
        res.update(clean=float(a_clean.median()), e0=float(a_e0.median()))
        for kd in ("embed", "attn", "mlp", "resid"):
            zs = [s for s in real_ups if s[1] == kd]
            res["e0_" + kd] = (float(run_tap(inf, Tap(bank, site, w, b, zs), toks, anch)[0].median())
                               if zs else None)
        res["e0_pos0"] = float(run_tap(inf, Tap(bank, site, w, b, real_ups, keep_pos0=True), toks, anch)[0].median())
        res["e0_noembed"] = float(run_tap(inf, Tap(bank, site, w, b, [s for s in real_ups if s[1] != "embed"]),
                                          toks, anch)[0].median())
        # the scorer's vacuous rule (post-ReLU = post-Top-K reads at k = d_sae; means over the anchors)
        ref_a, ref_c = eval_reference(rec, dev)
        mA = collect_site_means(inf, bank, ref_a, set(real_ups))
        mC = collect_site_means(inf, bank, ref_c, set(real_ups))
        eM = run_tap(inf, Tap(bank, site, w, b, real_ups, fill=mA), toks, anch)[0]
        eN = run_tap(inf, Tap(bank, site, w, b, real_ups, fill=mC), toks, anch)[0]
        tk = lambda v: float(torch.relu(v).mean())  # noqa: E731
        res.update(eM=float(eM.median()), eN=float(eN.median()), a_pos_tk=tk(a_clean), e0_tk=tk(a_e0),
                   eM_tk=tk(eM), eN_tk=tk(eN))
        res["vacuous_v2"] = vacuous_v2(res["a_pos_tk"], [res["e0_tk"], res["eM_tk"], res["eN_tk"]])
        res["vacuous_z_only"] = vacuous_v2(res["a_pos_tk"], [res["e0_tk"]])
        res["ln_clean"] = {str(x): [round(v, 4) for v in y] for x, y in ln_c.items()}
        res["ln_e0"] = {str(x): [round(v, 6) for v in y] for x, y in ln_z.items()}
        rows.append(res)
        fh.write(json.dumps(res) + "\n")
        fh.flush()
        print("%-14s clean %8.3g e0 %10.3g | embed %9.3g attn %9.3g mlp %9.3g resid %9.3g | pos0 %9.3g noemb %9.3g | "
              "eM %9.3g eN %9.3g vac %s | LN std e0 min/layer %s" % (
                  key, res["clean"], res["e0"], *(res["e0_" + kd] if res["e0_" + kd] is not None else float("nan")
                                                   for kd in ("embed", "attn", "mlp", "resid")),
                  res["e0_pos0"], res["e0_noembed"], res["eM"], res["eN"], res["vacuous_v2"],
                  [round(ln_z[x][0], 4) for x in lnl]), flush=True)
    print("vacuous_v2: %d of %d (Z-only rule: %d)" % (sum(r["vacuous_v2"] for r in rows), len(rows),
                                                      sum(r["vacuous_z_only"] for r in rows)))


if __name__ == "__main__":
    main()
