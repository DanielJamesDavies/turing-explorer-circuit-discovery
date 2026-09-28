"""CASE-STUDY VALIDATION for the chosen circuits (063 shortlist; Daniel picked 8.attn.29991 and 9.resid.7640).

Three checks per circuit, all on the protocol-v1 stage-1 outputs (OUT, default the stage-1 download):
  1. INGREDIENT KNOCK-OUTS. Re-score the circuit with one named group of "story" members removed (they are then
     ablated like any non-member), using the run's own scorer (amp_eval_pass_v2.score_circuit via override_alphas, on
     the held-out strongest contexts, activation read). Control: remove the same number of members drawn at random
     from members of SIMILAR CONTRIBUTION RANK (+-RANK_WIN), excluding every named member and the generic core, so the
     control is not just deleting negligible layer-0 members. N_DRAWS draws; mean and spread reported.
  2. PROBE SENTENCES. Hand-written sentences, in the clean model (no circuit): the target's activation (post-Top-K)
     at every position; the sentence max and its token, and for the negation target the value at the first comma or
     semicolon. Reference scale: the target's mean activation at its anchors on its held-out strongest contexts.
  3. SPECIFICITY. From the run's 056 rows (spec.shard*.jsonl): target vs sibling vs control faithfulness and the
     latents lifted above the site's Top-K cut (circuit vs empty circuit), per ablation method.

  PYTHONPATH=src python experiments/063-case-studies/validate.py      -> results_validate/<target>.md
  env: OUT  KEYS (default both)  N_DRAWS (8)  RANK_WIN (15)  SKIP_KO=1 (probes + specificity only)
"""
import glob
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
EXP = HERE.parent
OUT = Path(os.environ.get("OUT", str(EXP / "062-h100-protocol-v1" / "out_stage1" / "out")))
os.environ["OUT"] = str(OUT)
for d in ("062-h100-protocol-v1", "059-context-pool", "049-circuit-graph", "056-specificity"):
    sys.path.insert(0, str(EXP / d))
RESULTS = HERE / "results_validate"
N_DRAWS = int(os.environ.get("N_DRAWS", 8))
RANK_WIN = int(os.environ.get("RANK_WIN", 15))
FAITH = [("Z", "free0_tk"), ("A", "freeM_topk_tk"), ("C", "freeN_topk_tk"), ("necessity", "phi_sup_blind_tk")]
# members that fire about equally on target and contrast in (nearly) every circuit: never drawn as random controls
GENERIC = {"0.mlp.1349", "0.resid.6299", "0.attn.5663", "0.attn.712", "1.attn.13899", "1.attn.34495", "0.mlp.26380",
           "1.mlp.621", "1.attn.16130", "1.resid.39697", "5.attn.33630"}

# ---------------------------------------------------------------------------------------------- the two case studies
CASES = {
    "8.attn.29991": dict(
        title="acid-base dissociation",
        groups={
            "dissociation in water": ["5.resid.38102", "6.resid.29630"],
            "bases / hydroxide": ["4.resid.24519", "7.resid.10184", "6.resid.34419", "3.resid.12724"],
            "acidic / basic pH": ["7.resid.35144"],
            "equilibrium constant": ["5.resid.37972"],
            "acid (lexical)": ["7.resid.34146", "7.resid.6145"],
            "generic chemistry (control group)": ["7.resid.9342", "7.attn.20363", "2.resid.29543"],
        },
        all_story=["dissociation in water", "bases / hydroxide", "acidic / basic pH", "equilibrium constant",
                   "acid (lexical)"],
        probes={
            "acid-base": [
                "When hydrochloric acid dissolves in water, it donates a proton to form hydronium ions.",
                "A weak acid only partially dissociates in water, so its pH is higher than that of a strong acid.",
                "According to the Brønsted-Lowry definition, a base is a proton acceptor.",
                "Adding sodium hydroxide to the solution neutralises the acid and raises the pH.",
                "A buffer resists changes in pH when small amounts of acid or base are added.",
            ],
            "other chemistry": [
                "In a redox reaction, the reducing agent loses electrons and is oxidised.",
                "Ionic bonds form when electrons are transferred from a metal to a non-metal.",
                "Alkanes are saturated hydrocarbons with only single carbon-carbon bonds.",
                "To balance the equation, adjust the coefficients so each element appears equally on both sides.",
                "The ideal gas law relates the pressure, volume and temperature of a gas.",
            ],
            "same words, not chemistry": [
                "The team will base its strategy on the results of last year's survey.",
                "Her acid wit made the whole audience laugh.",
                "The professional athlete signed a new contract with the club.",
                "The dissolution of the partnership was announced yesterday.",
            ],
        }),
    "9.resid.7640": dict(
        title="the pivot in 'not X, but rather Y'",
        groups={
            "negation scope / 'more than just'": ["8.resid.12223", "7.resid.5408", "6.resid.27144", "8.resid.29816",
                                                  "8.resid.2652", "4.resid.26922"],
            "generic comma (control group)": ["6.resid.29283", "7.resid.853"],
        },
        all_story=["negation scope / 'more than just'"],
        probes={
            "negated clause, then pivot": [
                "Leadership isn't about giving orders, but rather about listening to your team.",
                "Cooking is not just about following recipes; it is about understanding flavours.",
                "The goal is not to win every argument, but to understand the other side.",
                "Good design isn't about decoration, but about solving problems.",
                "Science is not merely a collection of facts; rather, it is a way of thinking.",
            ],
            "same comma, no negation": [
                "Leadership is about giving orders, but also about listening to your team.",
                "Cooking involves following recipes, and it rewards creativity.",
                "The weather was cold, but the sun was shining all afternoon.",
                "She bought apples, oranges and pears at the market.",
            ],
            "negation, no contrastive pivot": [
                "He did not go to the party, because he was tired.",
                "The results were not significant, which surprised the authors.",
            ],
        }),
}


def parse(key):
    l, k, i = key.split(".")
    return int(l), k, int(i)


def alpha_map(c):
    out = defaultdict(dict)
    for n in c.nodes.values():
        md = n.metadata
        if md.get("role") == "seed":
            continue
        f = md["feature_id"]
        out[(int(f.layer), str(f.kind))][int(f.index)] = float(md.get("amplitude", 1.0))
    return out


def without(alphas, keys):
    drop = {parse(k) for k in keys}
    return {s: {i: a for i, a in d.items() if (s[0], s[1], i) not in drop} for s, d in alphas.items()}


class Validator:
    def __init__(self):
        import driver
        self.R = driver.Runner()
        self.G, self.V, self.H = self.R.G, self.R.V, self.R.H

    def score(self, c, rec, alphas):
        self.H.patch_eval_contexts(self.G, rec, "strong")
        row = self.V.score_circuit(c, {}, override_alphas=alphas, skip_roles=True)
        return {name: row.get(col) for name, col in FAITH}

    # ------------------------------------------------------------------------------------------- 1. knock-outs
    def knockouts(self, key, case, lines):
        c = torch.load(OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
        rec = self.R.contexts(key)
        base = alpha_map(c)
        present = {"%d.%s.%d" % (l, k, i) for (l, k), d in base.items() for i in d}
        mem = pd.read_csv(HERE / "results_deep" / "members" / ("%s.csv" % key))
        rank = {n: r for r, n in enumerate(mem.node)}
        named = {m for g in case["groups"].values() for m in g}
        pool = [n for n in mem.node if n not in named and n not in GENERIC]
        full = self.score(c, rec, base)
        rows = [("full circuit", len(present), full, None)]
        groups = dict(case["groups"])
        groups["ALL story ingredients"] = [m for g in case["all_story"] for m in case["groups"][g]]
        rng = random.Random(0)
        for gname, keys in groups.items():
            keys = [k for k in keys if k in present]
            missing = [k for k in case["groups"].get(gname, keys) if k not in present]
            if missing:
                lines.append("- note: %s not in the circuit, skipped: %s" % (gname, ", ".join(missing)))
            if not keys:
                continue
            ko = self.score(c, rec, without(base, keys))
            draws = []
            for _ in range(N_DRAWS):
                pick = set()
                for k in keys:                                   # a random member of similar contribution rank
                    r = rank.get(k, len(mem) // 2)
                    cand = [n for n in pool if abs(rank[n] - r) <= RANK_WIN and n not in pick]
                    if cand:
                        pick.add(rng.choice(cand))
                draws.append(self.score(c, rec, without(base, pick)))
            rows.append((gname, len(keys), ko, draws))
            print("  %-40s done" % gname, flush=True)

        def fmt(v):
            return "—" if v is None else "%.2f" % v
        lines += ["", "### 1. Ingredient knock-outs (held-out strongest contexts, activation read)", "",
                  "Removed members are ablated like non-members. Random = the same number of members of similar "
                  "contribution rank (±%d), mean ± sd over %d draws." % (RANK_WIN, N_DRAWS), "",
                  "| removed | n | Z | A | C | necessity | random (same n): Z | A | C |", "|---|---|---|---|---|---|---|---|---|"]
        for gname, n, s, draws in rows:
            if draws is None:
                lines.append("| **%s** | %d | %s | %s | %s | %s | | | |" % (gname, n, *[fmt(s[x]) for x, _ in FAITH]))
                continue
            rnd = []
            for x in ("Z", "A", "C"):
                v = [d[x] for d in draws if d[x] is not None]
                rnd.append("%.2f ± %.2f" % (np.mean(v), np.std(v)) if v else "—")
            lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (gname, n, *[fmt(s[x]) for x, _ in FAITH], *rnd))

    # ------------------------------------------------------------------------------------------- 2. probes
    def target_acts(self, key, token_rows):
        from sae.dense import sparse_topk_to_dense
        l, k, i = parse(key)
        G = self.G
        k2i = {kd: n for n, kd in enumerate(G["KINDS"])}
        out = []
        for toks in token_rows:
            cap = {}

            def hook(layer_idx, activations):
                if layer_idx == l:
                    ta, ti = G["bank"].encode(activations[k2i[k]], k, layer_idx)
                    cap["a"] = sparse_topk_to_dense(ta, ti, G["D"], dtype=torch.float32)[0, :, i].cpu()
            G["inference"].disable_compile()
            try:
                with torch.no_grad():
                    G["inference"].forward(torch.tensor([toks], device=G["device"]), activations_callback=hook,
                                           return_activations=False, tokenize_final=False)
            finally:
                G["inference"].enable_compile()
            out.append(cap["a"])
        return out

    def probes(self, key, case, lines):
        from model.tokenizer import Tokenizer
        tok = Tokenizer()
        dec = lambda t: tok.decode([int(t)]).replace("\n", "\\n")
        rec = self.R.contexts(key)
        pos, arg = rec["strong"]["pos"], rec["strong"]["arg"].long()
        held = rec["strong"]["held"]
        ref_rows = self.target_acts(key, [pos[j].tolist() for j in held])
        ref = float(np.mean([float(a[int(arg[j])]) for a, j in zip(ref_rows, held)]))
        lines += ["", "### 2. Probe sentences (clean model, no circuit)", "",
                  "Reference: the target's mean activation at its anchors on its held-out strongest contexts = "
                  "**%.2f**. Values are post-Top-K activations (0 = the target is not in its site's Top-K)." % ref, ""]
        neg = key == "9.resid.7640"
        for group, sents in case["probes"].items():
            lines.append("**%s**" % group)
            ids = [tok.encode(s) for s in sents]
            acts = self.target_acts(key, ids)
            for s, t, a in zip(sents, ids, acts):
                j = int(a.argmax())
                extra = ""
                if neg:
                    cp = [p for p, x in enumerate(t) if dec(x).strip() in (",", ";")]
                    extra = " | at first ,/; %.2f" % float(a[cp[0]]) if cp else " | no ,/;"
                lines.append("- max %.2f (%.0f%% of ref) on %r%s — %s" % (float(a.max()), 100 * float(a.max()) / ref,
                                                                      dec(t[j]), extra, s))
            lines.append("")

    # ------------------------------------------------------------------------------------------- 3. specificity
    def specificity(self, key, lines):
        rows = [json.loads(l) for f in glob.glob(str(OUT / "main" / "spec.shard*.jsonl")) for l in open(f)]
        rows = [r for r in rows if r.get("seed") == key]
        lines += ["### 3. Specificity (the run's 056 rows)", "",
                  "| ablation | target faith (pre) | sibling faith (median) | control faith (median) | latents lifted "
                  "above Top-K cut: circuit / empty | target rank clean → circuit |", "|---|---|---|---|---|---|"]
        f = lambda v: "—" if v is None else "%.2f" % v
        for r in sorted(rows, key=lambda r: r["pi"]):
            lines.append("| %s | %s | %s | %s | %.0f / %.0f | %.0f → %.0f |" % (
                r["pi"], f(r.get("target_faith_pre")), f(r.get("sibling_faith_median")),
                f(r.get("control_faith_median")), r.get("switched_on_circuit", float("nan")),
                r.get("switched_on_empty", float("nan")), r.get("rank_clean", float("nan")),
                r.get("rank_circuit", float("nan"))))


def main():
    keys = [k for k in os.environ.get("KEYS", ",".join(CASES)).split(",") if k]
    RESULTS.mkdir(parents=True, exist_ok=True)
    V = Validator()
    for key in keys:
        case = CASES[key]
        lines = ["# %s — %s" % (key, case["title"])]
        V.specificity(key, lines)
        V.probes(key, case, lines)
        if not os.environ.get("SKIP_KO"):
            V.knockouts(key, case, lines)
        txt = "\n".join(lines) + "\n"
        (RESULTS / ("%s.md" % key)).write_text(txt, encoding="utf-8")
        print(txt, flush=True)


if __name__ == "__main__":
    main()
