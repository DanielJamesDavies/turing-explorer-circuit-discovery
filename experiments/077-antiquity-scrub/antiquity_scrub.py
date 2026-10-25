"""077 — causal scrub of the antiquity case study (9.resid.20419), the hypothesis-structured test.

The case study's interpretation: the target reads an "is this classical Greek antiquity?" variable carried by two
member groups ("ancient / early times", "classical Greek"; 063 validate.py). Causal scrubbing asks whether that
interpretation survives swaps it says should not matter, and fails swaps it says should.

Prompts: "In <X>, the scholar lived in the city of" and "In <X>, the philosopher taught in the city of", X classical
(CLASSICAL) or not (OTHER). Base and donor share the suffix, so suffix positions are aligned from the end. In the live
model (every other latent live, SAE error preserved), the named group's latents at the suffix positions are SET to
their values in the donor prompt; the target is read at the base's anchor (its peak over the suffix).

  same-class swap   classical base <- classical donor: the interpretation predicts the target is KEPT
  cross-class swap  classical base <- other donor: predicts it DROPS toward the donor's level
  induce            other base <- classical donor: predicts it RISES toward the classical level
Controls: random member groups of the same size and similar contribution rank (+-15, 063's rule), N_CTRL draws.
Suffix positions = the token tail shared by every prompt with that suffix (computed, not assumed).

  PYTHONPATH=src python experiments/077-antiquity-scrub/antiquity_scrub.py      -> out/rows.jsonl, out/report.md
"""
import contextlib
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "073-strict-eval"))
import strict_eval as SE  # noqa: E402

KEY = "9.resid.20419"
GROUPS = {
    "ancient / early times": ["7.resid.10003", "8.resid.13018", "7.attn.24922"],
    "classical Greek": ["7.resid.37523", "8.resid.4704", "9.mlp.11623", "9.mlp.21754", "8.resid.13141",
                        "7.resid.25644", "8.resid.34691"],
}
GROUPS["both"] = GROUPS["ancient / early times"] + GROUPS["classical Greek"]
CLASSICAL = ["the 5th century BC", "450 BC", "ancient Greece", "classical Athens", "the age of Pericles",
             "ancient Sparta"]
OTHER = ["the 19th century", "1950", "medieval France", "modern Germany", "the 1980s", "Victorian England"]
SUFFIXES = [", the scholar lived in the city of", ", the philosopher taught in the city of"]
MEMBERS = EXP / "063-case-studies" / "results_full_deep" / "members" / ("%s.csv" % KEY)
N_CTRL = int(os.environ.get("N_CTRL", 8))
RANK_WIN = 15
OUT = HERE / "out"


def parse(k):
    l, kd, i = k.split(".")
    return (int(l), kd), int(i)


class GroupSet:
    """Live stream; at each named site, the named latents at positions ``pos`` SET to ``vals`` [len(pos), n].
    Also taps the target's post-Top-K read at its site (all positions)."""

    def __init__(self, bank, edits, target_site, target_idx):
        self.bank, self.edits, self.ts, self.ti = bank, edits, target_site, target_idx
        self.tap = None

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        from sae.dense import sparse_topk_to_dense, target_latent_activations
        s = (layer_idx, kind)
        bank = self.bank
        if s == self.ts:
            ta, ti = bank.encode(x, kind, layer_idx)
            self.tap = target_latent_activations(ta, ti, self.ti)[0].float().cpu()
        e = self.edits.get(s)
        if e is None:
            return x
        idx, pos, vals = e
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        err = x - bank.decode(dense, kind, layer_idx)
        new = dense.clone()
        new[0, pos[:, None], idx[None, :]] = vals.to(new.dtype)
        return bank.decode(new, kind, layer_idx) + err


def main():
    from model.tokenizer import Tokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    R = SE.driver.Runner()
    G = R.G
    inference, bank, dev = G["inference"], G["bank"], G["device"]
    tok = Tokenizer()
    tsite, tidx = parse(KEY)
    kinds = list(bank.kinds)
    k2i = {k: i for i, k in enumerate(kinds)}

    mem = pd.read_csv(MEMBERS)
    rank = {n: r for r, n in enumerate(mem.node)}
    named = set(GROUPS["both"])
    pool = [n for n in mem.node if n not in named]

    def clean(ids):
        """Target read and every member's code at all positions, one prompt."""
        cap = {}
        want = {}
        for n in mem.node:
            s, i = parse(n)
            want.setdefault(s, []).append(i)

        def hook(layer_idx, acts):
            from sae.dense import sparse_topk_to_dense
            for kd in kinds:
                s = (layer_idx, kd)
                if s in want or s == tsite:
                    ta, ti = bank.encode(acts[k2i[kd]], kd, layer_idx)
                    dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=torch.float32)[0]
                    if s in want:
                        cap[s] = (torch.tensor(want[s]), dense[:, want[s]].cpu())
                    if s == tsite:
                        cap["target"] = dense[:, tidx].cpu()
                    if s in want and s != tsite:
                        cap.setdefault("_active_" + str(s), (dense > 0).any(0).nonzero().flatten().cpu())

        inference.disable_compile()
        try:
            with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                inference.forward(torch.tensor([ids], device=dev), activations_callback=hook, return_activations=False,
                                  tokenize_final=False)
        finally:
            inference.enable_compile()
        return cap

    prompts = {}
    for cls, xs in (("classical", CLASSICAL), ("other", OTHER)):
        for x in xs:
            for sfx in SUFFIXES:
                text = "In %s%s" % (x, sfx)
                ids = tok.encode(text)
                prompts[text] = dict(cls=cls, x=x, sfx=sfx, ids=ids)
    for sfx in SUFFIXES:
        group = [p["ids"] for p in prompts.values() if p["sfx"] == sfx]
        n = 0
        while all(len(g) > n and g[-1 - n] == group[0][-1 - n] for g in group):
            n += 1
        for p in prompts.values():
            if p["sfx"] == sfx:
                p["n_sfx"] = n
        print("suffix %r: %d shared tail tokens" % (sfx, n), flush=True)
    for p in prompts.values():
        p["cap"] = clean(p["ids"])
        L = len(p["ids"])
        p["sfx_pos"] = list(range(L - p["n_sfx"], L))
        t = p["cap"]["target"]
        p["anchor"] = max(p["sfx_pos"], key=lambda j: float(t[j]))
        p["t"] = float(t[p["anchor"]])
    print("prompts:", flush=True)
    for text, p in prompts.items():
        print("  %-9s %6.2f  %s" % (p["cls"], p["t"], text), flush=True)

    def patched(base, donor, nodes):
        n_al = min(len(base["sfx_pos"]), len(donor["sfx_pos"]))
        bpos = torch.tensor(base["sfx_pos"][-n_al:], device=dev)
        dpos = donor["sfx_pos"][-n_al:]
        by_site = {}
        for n in nodes:
            s, i = parse(n)
            by_site.setdefault(s, []).append(i)
        edits = {}
        for s, idx in by_site.items():
            if s in donor["cap"]:
                allidx, codes = donor["cap"][s]
                col = [int((allidx == i).nonzero()[0]) for i in idx]
                vals = codes[dpos][:, col]
            else:                                      # a non-member control latent: capture it now
                vals = None
            if vals is None:
                continue
            edits[s] = (torch.tensor(idx, device=dev), bpos, vals.to(dev))
        p = GroupSet(bank, edits, tsite, tidx)
        inference.disable_compile()
        try:
            with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                inference.forward(torch.tensor([base["ids"]], device=dev), patcher=p, grad_enabled=False,
                                  return_activations=False, tokenize_final=False)
        finally:
            inference.enable_compile()
        return float(p.tap[base["anchor"]])

    rng = random.Random(0)

    def ctrl_groups(keys):
        out = []
        for _ in range(N_CTRL):
            pick = []
            for k in keys:
                r = rank.get(k, len(mem) // 2)
                cand = [n for n in pool if abs(rank[n] - r) <= RANK_WIN and n not in pick]
                if cand:
                    pick.append(rng.choice(cand))
            out.append(pick)
        return out

    ctrls = {g: ctrl_groups(keys) for g, keys in GROUPS.items()}
    rows = []
    texts = list(prompts)
    for bt in texts:
        b = prompts[bt]
        for dt in texts:
            d = prompts[dt]
            if dt == bt or d["sfx"] != b["sfx"]:
                continue
            cond = {("classical", "classical"): "same", ("classical", "other"): "cross",
                    ("other", "classical"): "induce"}.get((b["cls"], d["cls"]))
            if cond is None:
                continue
            for g, keys in GROUPS.items():
                v = patched(b, d, keys)
                cv = [patched(b, d, ck) for ck in ctrls[g]]
                rows.append(dict(cond=cond, base=bt, donor=dt, group=g, n=len(keys), t_base=b["t"], t_donor=d["t"],
                                 t_patched=v, t_ctrl=cv))
    with open(OUT / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # report
    df = pd.DataFrame(rows)
    df["ctrl_mean"] = df.t_ctrl.apply(np.mean)
    t_cls = np.mean([p["t"] for p in prompts.values() if p["cls"] == "classical"])
    t_oth = np.mean([p["t"] for p in prompts.values() if p["cls"] == "other"])
    L = ["# 077 — antiquity causal scrub (%s)\n" % KEY,
         "Clean target at the anchor: classical prompts mean %.2f, other prompts mean %.2f.\n" % (t_cls, t_oth),
         "Retained = t_patched / t_base (same / cross); induced = t_patched / mean classical t (induce). Medians over "
         "base x donor pairs; controls = %d random member groups of matched size and contribution rank.\n" % N_CTRL,
         "| condition | group | n | pairs | story group | random members | prediction |",
         "|---|---|---|---|---|---|---|"]
    pred = {"same": "kept (~1)", "cross": "drops (toward %.2f)" % (t_oth / t_cls), "induce": "rises"}
    for cond in ("same", "cross", "induce"):
        for g in GROUPS:
            x = df[(df.cond == cond) & (df.group == g)]
            if cond == "induce":
                s, c = (x.t_patched / t_cls).median(), (x.ctrl_mean / t_cls).median()
            else:
                s, c = (x.t_patched / x.t_base).median(), (x.ctrl_mean / x.t_base).median()
            L.append("| %s | %s | %d | %d | %.2f | %.2f | %s |" % (cond, g, len(GROUPS[g]), len(x), s, c, pred[cond]))
    L.append("\nPrompt reads:\n")
    for text, p in prompts.items():
        L.append("- %s %.2f — %s" % (p["cls"], p["t"], text))
    txt = "\n".join(L)
    (OUT / "report.md").write_text(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    main()
