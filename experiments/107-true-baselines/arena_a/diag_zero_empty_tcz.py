"""Arena A (item A4): arena C's Finding 2 diagnostic (../arena_c/diag_zero_empty.py) on the gpt2-small-tcz family:
does the zero-ablated empty circuit (the scorer's e0) explode on latent targets? Forward passes only, no fits.

Panel: PER_SITE (default 1) latents per (layer 0-11) x {z, tc} site with full top + mid stores in the data pass,
seeded (24 targets by default); KEYS=a,b,... overrides it. Per target, on its protocol contexts (eval_probe(rec,
"strong"): read at the held-out strongest anchors, as the scorer):
  clean       the target pre-activation (median over anchors)
  e0          every upstream code zeroed (z, z2, tc sites below the target, and z / z2 at its own layer for a tc
              target), errors kept (the scorer's empty Z circuit)
  e0_<kind>   only that kind's upstream sites zeroed (z = z + z2 halves together, tc)
  vacuous     protocol section 12.4's rule (eval.wcm_eval.vacuous_v2) on the post-ReLU reads (= the Top-K read at
              k = d_sae): a_pos, e0, and the dense A / C empty circuits (eM / eN: every upstream code at the scorer's A /
              C mean, eval_reference rows)
  ln          per layer <= target layer: the ln_1 input std (the residual stream entering the block), min over
              positions and median at the anchors, clean vs e0
BOS (ARENA_A_BOS=1) is hidden from every site by the site model, so position 0 of a site tensor is the first real
token: arena C's e0_pos0 / e0_noembed rows have no counterpart here (no embed site; BOS is never edited).

  run in WSL from the repo root (STORES = the data pass's --out; OUT gets ctx/ records and results.jsonl):
  STORES=... OUT=... HF_HUB_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python \
      experiments/107-true-baselines/arena_a/diag_zero_empty_tcz.py
"""
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STORES = Path(os.environ.get("STORES", str(HERE / "out" / "pilot" / "datapass")))
OUT = Path(os.environ.get("OUT", str(HERE / "out" / "pilot" / "diag_z")))
os.environ.update(STORES=str(STORES), OUT=str(OUT), MODE="main", HF_HUB_OFFLINE="1")
os.environ.setdefault("DATA", str(ROOT / "data_public" / "gpt2-small"))
os.environ.setdefault("WCM_VERSION", "v4")
sys.path.insert(0, str(HERE))
import driver_tcz  # noqa: E402  (registers the family, imports 062's driver, float32 without TF32)
import torch  # noqa: E402

import wcm_tcz as WT  # noqa: E402

driver = driver_tcz.driver
PER_SITE = int(os.environ.get("PER_SITE", 1))
PANEL_KINDS = ("z", "tc")
SEED = 20261104
MIN_TOP, MIN_MID = 64, 48                     # arena C's panel rule (top store full, mid store >= 48 of 64)


def eligible(top, mid, layer, kind):
    c = layer * len(WT.KINDS) + WT.KINDS.index(kind)
    return (((top[c] > 0).sum(-1) >= MIN_TOP) & ((mid[c] > 0).sum(-1) >= MIN_MID)).nonzero().flatten().tolist()


def panel():
    if os.environ.get("KEYS"):
        return [k for k in os.environ["KEYS"].split(",") if k]
    top = torch.load(STORES / "top_ctx.pt", weights_only=False)["ctx_seq_idx"].cpu()
    mid = torch.load(STORES / "mid_ctx.pt", weights_only=False)["ctx_seq_idx"].cpu()
    rng = random.Random(SEED)
    keys = []
    for l in range(WT.N_LAYER):
        for k in PANEL_KINDS:
            ok = eligible(top, mid, l, k)
            keys += ["%d.%s.%d" % (l, k, i) for i in rng.sample(ok, min(PER_SITE, len(ok)))]
    return keys


class Tap:
    """Read the target's pre-activation at its site; at `zero` sites replace every code by `fill` (0 or a mean),
    errors kept (x <- x + (fill - c) W_D^T)."""

    def __init__(self, bank, site, w, b, zero=(), fill=None):
        self.bank, self.site, self.w, self.b = bank, site, w, b
        self.zero, self.fill = set(zero), fill or {}
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
        return x + self.bank.decode(d, k, l, add_bias=False).to(x.dtype)


def run_tap(inf, tap, toks, anchors, ln_layers=None):
    """(pre at the anchors [N], ln_1-input std per layer {l: (min over positions, median at the anchors)}). The ln_1
    input carries the BOS row at position 0 (the hook sees the full stream), so it is dropped: positions line up with
    the site tensors."""
    stds, hs = {}, []
    if ln_layers is not None:
        for l in ln_layers:
            ln = inf.model.hf.transformer.h[l].ln_1

            def pre(_m, args, _l=l):
                x = args[0]
                if WT.bos_prefix():
                    x = x[:, 1:]
                stds[_l] = x.float().std(-1, unbiased=False)
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
    real = set(WT.real_sites())
    fh = open(OUT / os.environ.get("RESULTS", "results.jsonl"), "w")
    rows = []
    for key in panel():
        rec = R.contexts(key)
        if rec.get("strong") is None or skip_reason(rec) is not None:
            print("%s: no usable record (%s)" % (key, skip_reason(rec)), flush=True)
            continue
        l, k, i = (int(key.split(".")[0]), key.split(".")[1], int(key.split(".")[2]))
        site = (l, k)
        w, b = bank.preact_vector(k, l, i)
        pd_ = eval_probe(rec, "strong", dev)
        n_tr = eval_split(int(pd_.pos_tokens.shape[0]), pd_.metadata, "explicit")
        toks, anch = pd_.pos_tokens[n_tr:], pd_.pos_argmax[n_tr:]
        ups = sorted(upstream_sites(bank, l, k), key=lambda s: (s[0], WT.KINDS.index(s[1])))
        real_ups = [s for s in ups if s in real and not getattr(bank.saes[s[1]][s[0]], "null", False)]
        lnl = list(range(0, l + 1))
        res = dict(key=key, n_ho=int(toks.shape[0]), n_up=len(real_ups))
        a_clean, ln_c = run_tap(inf, Tap(bank, site, w, b), toks, anch, lnl)
        a_e0, ln_z = run_tap(inf, Tap(bank, site, w, b, real_ups), toks, anch, lnl)
        res.update(clean=float(a_clean.median()), e0=float(a_e0.median()))
        for kd, kset in (("z", ("z", "z2")), ("tc", ("tc",))):
            zs = [s for s in real_ups if s[1] in kset]
            res["e0_" + kd] = (float(run_tap(inf, Tap(bank, site, w, b, zs), toks, anch)[0].median())
                               if zs else None)
        ref_a, ref_c = eval_reference(rec, dev)
        if real_ups:
            mA = collect_site_means(inf, bank, ref_a, set(real_ups))
            mC = collect_site_means(inf, bank, ref_c, set(real_ups))
            eM = run_tap(inf, Tap(bank, site, w, b, real_ups, fill=mA), toks, anch)[0]
            eN = run_tap(inf, Tap(bank, site, w, b, real_ups, fill=mC), toks, anch)[0]
        else:                                    # (0, z): nothing upstream, every empty circuit is the clean read
            eM = eN = a_clean
        tk = lambda v: float(torch.relu(v).mean())  # noqa: E731
        res.update(eM=float(eM.median()), eN=float(eN.median()), a_pos_tk=tk(a_clean), e0_tk=tk(a_e0),
                   eM_tk=tk(eM), eN_tk=tk(eN))
        res["e0_over_clean_tk"] = res["e0_tk"] / res["a_pos_tk"] if res["a_pos_tk"] > 0 else None
        res["vacuous_v2"] = vacuous_v2(res["a_pos_tk"], [res["e0_tk"], res["eM_tk"], res["eN_tk"]])
        res["vacuous_z_only"] = vacuous_v2(res["a_pos_tk"], [res["e0_tk"]])
        res["ln_clean"] = {str(x): [round(v, 4) for v in y] for x, y in ln_c.items()}
        res["ln_e0"] = {str(x): [round(v, 6) for v in y] for x, y in ln_z.items()}
        rows.append(res)
        fh.write(json.dumps(res) + "\n")
        fh.flush()
        nan = float("nan")
        print("%-14s n_up %2d clean %8.3g e0 %10.3g (tk ratio %s) | z %9.3g tc %9.3g | eM %9.3g eN %9.3g vac %s | "
              "LN std e0 min/layer %s" % (
                  key, res["n_up"], res["clean"], res["e0"],
                  "%.3g" % res["e0_over_clean_tk"] if res["e0_over_clean_tk"] is not None else "-",
                  *(res["e0_" + kd] if res["e0_" + kd] is not None else nan for kd in ("z", "tc")),
                  res["eM"], res["eN"], res["vacuous_v2"], [round(ln_z[x][0], 4) for x in lnl]), flush=True)
    ratios = sorted(r["e0_over_clean_tk"] for r in rows if r["e0_over_clean_tk"] is not None)
    print("panel: %d targets; vacuous_v2: %d (Z-only rule: %d); e0/clean (post-ReLU means) median %s, max %s" % (
        len(rows), sum(r["vacuous_v2"] for r in rows), sum(r["vacuous_z_only"] for r in rows),
        "%.3g" % ratios[len(ratios) // 2] if ratios else "-", "%.3g" % ratios[-1] if ratios else "-"), flush=True)


if __name__ == "__main__":
    main()
