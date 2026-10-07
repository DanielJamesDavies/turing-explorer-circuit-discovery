"""CASE-STUDY DATA for the paper's diagrammed circuit (DAN-138 / DAN-149; default 9.resid.20419, a date or place read
as classical Greek antiquity). Everything the figure and the section quote comes from this one script.

Role groups are named from each member's own top contexts and logit effect (the 60-member report in
results_case/reports/), not from auto-interp. Four measurements, all on the full-run circuit and its protocol contexts:

  1. GROUP SHARES. Each group's share of the circuit's total contribution at the target's anchor (members table:
     alpha x mean activation at the anchor x decoder norm), and its layer span.
  2. KNOCK-OUTS (as validate.py). Re-score with one group removed (its members ablated like non-members) through the
     run's scorer (held-out strongest, activation read); control = the same number of members of similar contribution
     rank (+-RANK_WIN), N_DRAWS draws, excluding every named member and the generic core.
  3. GROUP EDGES. Circuit-only runs under the activating-mean fill (A) on the held-out strongest contexts. A member's
     value is its live encoding at its site, which depends on the members upstream of it. Edge g -> h =
     1 - sum_h(value with g removed) / sum_h(value, full circuit), at the target's anchor positions. Also g -> target.
  4. PROBES. Hand-written sentences and a year sweep. The target's maximum over the sentence (post-Top-K), in the
     clean model and in circuit-only runs under each fill (Z zero, A activating mean, C contrast mean; means from the
     target's training contexts), so the figure can show that the circuit alone reproduces the contrast. Each fill
     is also read with the EMPTY circuit (everything upstream ablated), the floor the fill itself produces.

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src python experiments/063-case-studies/case_study.py
  env: KEY (9.resid.20419)  N_DRAWS (8)  RANK_WIN (15)  SKIP_KO=1  -> results_case/<KEY>.json
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
EXP = HERE.parent
os.environ.setdefault("OUT", str(EXP / "062-h100-protocol-v1" / "out_full" / "out"))
sys.path.insert(0, str(HERE))
import validate as VAL  # noqa: E402  (puts 062 / 059 / 049 / 056 on the path)
import attn_heads as AH  # noqa: E402  (062: the scorer's preamble, setup_target)

KEY = os.environ.get("KEY", "9.resid.20419")
N_DRAWS = int(os.environ.get("N_DRAWS", 8))
RANK_WIN = int(os.environ.get("RANK_WIN", 15))
MEMBERS = HERE / "results_case" / "members" / ("%s.csv" % KEY)
RESULTS = HERE / "results_case"

# ---- role groups for 9.resid.20419, from the 60-member report (own peaks, own contexts, logit effect, target vs
# contrast engagement). Ordered roughly by layer and role; "generic core" and "historical period" are the controls.
GROUPS = {
    "dates / eras": ["7.attn.24922", "7.mlp.19617", "8.mlp.5180", "8.resid.22626"],
    "early civilisations": ["7.resid.10003", "8.resid.13018", "7.attn.23646", "7.attn.39627", "2.resid.16478",
                            "7.mlp.17292", "7.resid.17459"],
    "classical Greece": ["7.resid.37523", "8.resid.4704", "9.mlp.11623", "9.mlp.21754", "8.resid.34691",
                         "8.mlp.3889", "7.mlp.17081", "8.mlp.4335", "9.attn.36812", "8.resid.1005"],
    "philosophy / rhetoric": ["8.resid.21699", "8.resid.26127", "7.resid.38584", "0.mlp.21906", "7.attn.10587",
                              "7.attn.23071", "8.attn.3327", "7.attn.2593"],
    "government / empire": ["8.resid.42", "7.resid.35780", "7.resid.23336", "8.resid.4747"],
    "historical period (control)": ["8.resid.13141", "7.resid.25644", "7.attn.15390", "7.attn.24506",
                                    "8.attn.6856", "7.attn.24779"],
    "generic core (control)": ["0.resid.6299", "0.attn.5663", "0.mlp.1349", "1.attn.13899", "1.attn.16130",
                               "1.mlp.621", "5.attn.33630"],
}
STORY = ["dates / eras", "early civilisations", "classical Greece"]

PROBES = {
    "classical Greece": [
        "In the 5th century BC, the philosopher taught in the city of",
        "In 450 BC, the scholar lived in the city of",
        "In ancient Greece, the scholar lived in the city of",
        "In ancient Athens, the scholar lived in the city of",
    ],
    "other ancient": [
        "In 10000 BC, the scholar lived in the city of",
        "In ancient Egypt, the scholar lived in the city of",
        "In ancient Rome, the scholar lived in the city of",
        "In ancient China, the scholar lived in the city of",
    ],
    "later eras": [
        "In medieval France, the scholar lived in the city of",
        "In the 19th century, the philosopher taught in the city of",
        "In 1950, the scholar lived in the city of",
    ],
    "lexical control": ["The BC Lions played football in the city of"],
}
YEARS_BC = [5000, 3000, 2000, 1500, 1000, 800, 600, 500, 450, 400, 300, 200, 100]
YEARS_AD = [100, 300, 500, 800, 1000, 1200, 1500, 1800, 1900, 2000]


def year_sentence(y, era):
    return ("In %d BC, the scholar lived in the city of" % y) if era == "BC" else (
        "In %d AD, the scholar lived in the city of" % y)


def parse(k):
    l, kd, i = k.split(".")
    return int(l), kd, int(i)


class Case:
    def __init__(self):
        self.Vd = VAL.Validator()
        self.R, self.G, self.V, self.H = self.Vd.R, self.Vd.G, self.Vd.V, self.Vd.H
        self.c = torch.load(VAL.OUT / "main" / "circuits" / ("%s.pt" % KEY), weights_only=False)
        self.rec = self.R.contexts(KEY)
        self.base = VAL.alpha_map(self.c)
        self.present = {"%d.%s.%d" % (l, k, i) for (l, k), d in self.base.items() for i in d}
        self.mem = pd.read_csv(MEMBERS)
        for g, ms in GROUPS.items():
            miss = [m for m in ms if m not in self.present]
            assert not miss, "%s: not in the circuit: %s" % (g, miss)

    # -------------------------------------------------------------------------------- 1. shares
    def shares(self):
        tot = float(self.mem.contrib.clip(lower=0).sum())
        out = {}
        for g, ms in GROUPS.items():
            m = self.mem[self.mem.node.isin(ms)]
            out[g] = dict(n=len(ms), share=float(m.contrib.clip(lower=0).sum()) / tot,
                          layers=sorted({parse(x)[0] for x in ms}), members=ms)
        named = {x for ms in GROUPS.values() for x in ms}
        rest = self.mem[~self.mem.node.isin(named)]
        out["_rest"] = dict(n=int(len(rest)), share=float(rest.contrib.clip(lower=0).sum()) / tot)
        out["_total_members"] = int(len(self.mem))
        return out

    # -------------------------------------------------------------------------------- 2. knock-outs
    def knockouts(self):
        rank = {n: r for r, n in enumerate(self.mem.node)}
        named = {m for ms in GROUPS.values() for m in ms}
        pool = [n for n in self.mem.node if n not in named and n not in VAL.GENERIC]
        out = {"full": self.Vd.score(self.c, self.rec, self.base)}
        groups = dict(GROUPS)
        groups["all story groups"] = [m for g in STORY for m in GROUPS[g]]
        rng = random.Random(0)
        for g, keys in groups.items():
            ko = self.Vd.score(self.c, self.rec, VAL.without(self.base, keys))
            draws = []
            for _ in range(N_DRAWS):
                pick = set()
                for k in keys:
                    cand = [n for n in pool if abs(rank[n] - rank[k]) <= RANK_WIN and n not in pick]
                    if cand:
                        pick.add(rng.choice(cand))
                draws.append(self.Vd.score(self.c, self.rec, VAL.without(self.base, pick)))
            out[g] = dict(n=len(keys), ko=ko, random=draws)
            print("  knock-out %-28s A %.2f (random %.2f)" % (
                g, ko["A"] if ko["A"] is not None else float("nan"),
                np.mean([d["A"] for d in draws if d["A"] is not None])), flush=True)
        return out

    # -------------------------------------------------------------------------------- 3. edges
    def edges(self):
        self.H.patch_eval_contexts(self.G, self.rec, "strong")
        S = AH.setup_target(self.R, KEY, self.c)
        TapCO, _ = self.V._engine_classes()
        bank, K = self.G["bank"], int(self.G["K"])
        watch = {}
        for g, ms in GROUPS.items():
            for m in ms:
                l, kd, i = parse(m)
                watch.setdefault((l, kd), []).append((g, i))
        pa = S["pa"]

        class Cap(TapCO):
            def transform(self_, layer_idx, kind, x):
                w = watch.get((layer_idx, kind))
                if w:
                    from sae.dense import sparse_topk_to_dense
                    ta, ti = bank.encode(x, kind, layer_idx)
                    rr = torch.arange(x.shape[0], device=x.device)
                    anc = pa.to(x.device).clamp(0, x.shape[1] - 1)[:x.shape[0]]
                    idx = torch.tensor([i for _, i in w], device=x.device)
                    vals = sparse_topk_to_dense(ta, ti, self.G["D"], dtype=torch.float32)[rr, anc][:, idx]
                    for (g, _), v in zip(w, vals.T):
                        self_.grp[g] = self_.grp.get(g, 0.0) + float(v.sum())
                return super().transform(layer_idx, kind, x)

        def run(alphas):
            keep = {s: set(d) for s, d in alphas.items() if d}
            msets = {s: torch.tensor(sorted(d), device=pa.device, dtype=torch.long) for s, d in keep.items()}
            scales = {}
            for s, v in msets.items():
                sv = torch.ones(self.G["D"], device=pa.device, dtype=torch.float32)
                sv[v] = torch.tensor([alphas[s][int(i)] for i in v.tolist()], device=pa.device, dtype=torch.float32)
                scales[s] = sv
            p = Cap(bank=bank, keep_indices=keep, in_scope=S["UPS"], seed_layer=S["layer"], seed_kind=S["kind"],
                    seed_latent_idx=S["sl"], pos_argmax=pa, site_means=S["means"], respect_topk=True, topk=K,
                    keep_tensors=msets, keep_scales=scales, w_seed=S["w"], b_seed=S["b"])
            p.grp = {}
            inf = self.G["inference"]
            inf.disable_compile()
            try:
                with torch.no_grad():
                    inf.forward(S["pt"], patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
            finally:
                inf.enable_compile()
            rr = torch.arange(int(S["pt"].shape[0]), device=p.tap_tk.device)
            tgt = float(p.tap_tk[rr, pa.to(p.tap_tk.device).clamp(0, p.tap_tk.shape[1] - 1)].mean())
            return p.grp, tgt

        full, tfull = run(self.base)
        out = dict(full_group_values=full, target_full=tfull, edges={}, to_target={})
        for g, ms in GROUPS.items():
            vals, t = run(VAL.without(self.base, ms))
            out["edges"][g] = {h: (1 - vals.get(h, 0.0) / full[h]) if full.get(h, 0) > 1e-6 else None
                               for h in GROUPS if h != g}
            out["to_target"][g] = 1 - t / tfull if tfull > 0 else None
        return out

    # -------------------------------------------------------------------------------- 4. probes
    def probes(self):
        from eval.floors import collect_site_means
        from model.tokenizer import Tokenizer
        tok = Tokenizer()
        self.H.patch_eval_contexts(self.G, self.rec, "strong")
        S = AH.setup_target(self.R, KEY, self.c)
        V, G = self.V, self.G
        site = (S["layer"], S["kind"])
        ref = V.read(lambda a, b: V.AmpInjectPatcher({}, site, S["w"], S["b"], S["sl"]), S["pt"], S["pa"])["tk"]
        TapCO, _ = V._engine_classes()
        bank, K = G["bank"], int(G["K"])
        nt = self.rec["neg"].to(G["device"])
        means_neg = collect_site_means(G["inference"], bank, nt[:V.split_n(int(nt.shape[0]))], S["UPS"])
        keep = {s: set(d) for s, d in S["alphas"].items() if d}
        fills = {"Z": (None, False), "A": (S["means"], True), "C": (means_neg, True)}
        inf = G["inference"]

        def read(ids, fill, empty=False):
            t = torch.tensor([ids], device=G["device"])
            anc = torch.tensor([len(ids) - 1])
            if fill is None:
                p = V.AmpInjectPatcher({}, (S["layer"], S["kind"]), S["w"], S["b"], S["sl"])
            else:
                mm, topk = fills[fill]
                p = TapCO(bank=bank, keep_indices={} if empty else keep, in_scope=S["UPS"], seed_layer=S["layer"],
                          seed_kind=S["kind"], seed_latent_idx=S["sl"], pos_argmax=anc, site_means=mm,
                          respect_topk=topk, topk=K, keep_tensors={} if empty else dict(S["msets"]),
                          keep_scales=None if empty else S["scales"], w_seed=S["w"], b_seed=S["b"])
            inf.disable_compile()
            try:
                with torch.no_grad():
                    inf.forward(t, patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
            finally:
                inf.enable_compile()
            a = p.tap_tk[0].float().cpu()
            j = int(a.argmax())
            return float(a[j]), tok.decode([ids[j]])

        def one(sentence, cat):
            ids = tok.encode(sentence)
            row = dict(sentence=sentence, category=cat)
            for f in (None, "Z", "A", "C"):
                v, at = read(ids, f)
                row["clean" if f is None else f] = v
                if f is None:
                    row["at"] = at
                else:                                   # the empty circuit under the same fill: the fill's own floor
                    row[f + "_empty"] = read(ids, f, empty=True)[0]
            return row

        rows = [one(s, cat) for cat, ss in PROBES.items() for s in ss]
        years = [dict(year=-y, **one(year_sentence(y, "BC"), "year")) for y in YEARS_BC] + \
                [dict(year=y, **one(year_sentence(y, "AD"), "year")) for y in YEARS_AD]
        return dict(ref_a_pos=ref, sentences=rows, years=years)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    C = Case()
    out = dict(key=KEY, groups=C.shares())
    out["probes"] = C.probes(); print("probes done %.0fs" % (time.time() - t0), flush=True)
    out["edges"] = C.edges(); print("edges done %.0fs" % (time.time() - t0), flush=True)
    if not os.environ.get("SKIP_KO"):
        out["knockouts"] = C.knockouts(); print("knock-outs done %.0fs" % (time.time() - t0), flush=True)
    path = RESULTS / ("%s.json" % KEY)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
