"""Named-law pathway ("Newton's Third Law"): which of the candidate latents fire on which token, on minimal pairs.

The 065 descriptions suggest a relay across tokens: an apostrophe latent after "Newton" that pushes "Second"/"Third",
an ordinal latent on "Second"/"Third" that pushes "Law", and a latent on "Law" after "Third". The circuits cannot show
whether the first step depends on WHICH name (knowledge) or on any name (entity kind), so this runs the model on
minimal pairs (real vs made-up names, physics vs non-physics text, law vs non-law ordinals) and prints each latent's
TopK activation per token, plus the model's own next-token guesses at the key positions.

    python experiments/066-newton-chain/probe.py            # CPU, ~1 min
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "064-prediction-circuits"))
import common as C  # noqa: E402  (064 helpers: model, SAE bank, capture; sets cwd to the repo root)

LATENTS = {
    "9.resid.29881": "Newton-apostrophe (-> Second/Third)",
    "7.resid.4492": "Second/Third in Newton's (-> Law)",
    "9.attn.28565": "Law after Third",
    "5.attn.4921": "Zeroth Law",
    "10.resid.15497": "eponym 's -> Law (063)",
    "9.resid.2881": "apostrophe after proper names",
    "9.resid.22174": "Gutenberg 's",
}

PROMPTS = [
    # the apostrophe step: which name?
    "In classical mechanics, Newton's Third Law states that every action has a reaction.",
    "In classical mechanics, Kepler's Third Law states that the orbit period grows with distance.",
    "In classical mechanics, Zorbel's Third Law states that every action has a reaction.",
    "In classical mechanics, Einstein's theory replaced the older view of gravity.",
    "In the story, Newton's cat slept all afternoon on the warm windowsill.",
    "Shakespeare's plays are still performed in theatres around the world.",
    # the ordinal step: which ordinal, which noun follows?
    "In classical mechanics, Newton's Second Law relates force to acceleration.",
    "In classical mechanics, Newton's First Law describes inertia.",
    "The Third Law of Thermodynamics concerns absolute zero.",
    "The Zeroth Law of Thermodynamics defines thermal equilibrium.",
    "The third chapter of the book describes the war.",
    "After the Second World War, Europe was rebuilt.",
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    model, bank, tok = C.load_model(), C.load_bank(), C.load_tokenizer()
    dec = C.Dec(tok)
    gids = {k: C.parse_target(k) for k in LATENTS}
    sites = {}
    for k, g in gids.items():
        l, kind, i = C.split_gid(g)
        sites.setdefault((l, kind), []).append((k, i))

    for prompt in PROMPTS:
        ids = [C.BOS] + tok.encode(prompt)
        logits, acts = C.forward_capture(model, ids)
        probs = torch.softmax(logits[:, : C.N_REAL_VOCAB], dim=-1)
        T = len(ids)
        act = {k: torch.zeros(T) for k in LATENTS}
        for (l, kind), items in sites.items():
            v, idx = bank.encode(acts[l, C.KINDS.index(kind)], kind, l)   # [T, k]
            for k, i in items:
                hit = (idx == i)
                act[k] = (v.float() * hit).sum(-1)
        print(f"\n=== {prompt}")
        for t in range(1, T):
            fired = [(k, float(act[k][t])) for k in LATENTS if act[k][t] > 0]
            if not fired:
                continue
            top = torch.topk(probs[t], 3)
            guess = " ".join(f"{dec(int(j))!r} {float(p):.2f}" for p, j in zip(top.values, top.indices))
            print(f"  [{t:2d}] {dec(ids[t])!r:14s} " + "; ".join(f"{LATENTS[k].split(' (')[0]} {a:.1f}" for k, a in fired)
                  + f"   | next: {guess}")


if __name__ == "__main__":
    main()
