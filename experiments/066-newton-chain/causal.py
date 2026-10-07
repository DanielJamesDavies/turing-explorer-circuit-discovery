"""Causal check: remove a latent's contribution (activation x decoder direction, SAE error term kept) at the
apostrophe and 's' of "<Name>'s" and measure the model's next-token probabilities after the 's'.

  ordinal = p(' First') + p(' Second') + p(' Third')      (Newton-specific knowledge: Newton's Second/Third Law)
  law     = p(' Law') + p(' law') + p(' La') + p(' laws')  (generic "<name>'s -> Law")

Ablations: the Newton apostrophe latent (9.resid.29881), the generic eponym latent (10.resid.15497, 063 case study),
both, and a control: the other active latent at 9.resid with the closest activation at the 's'.

    python experiments/066-newton-chain/causal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "064-prediction-circuits"))
import common as C  # noqa: E402
from model.hooks import multi_patch  # noqa: E402

NEWTON, EPONYM = "9.resid.29881", "10.resid.15497"
PROMPTS = [
    "In classical mechanics, Newton's",
    "In the story, Newton's",
    "In classical mechanics, Kepler's",
    "In classical mechanics, Einstein's",
    "In classical mechanics, Zorbel's",
]


def latent_acts(bank, acts, l, kind, positions):
    """{latent index: [activation at each position]} for all latents active at any of the positions."""
    v, idx = bank.encode(acts[l, C.KINDS.index(kind)], kind, l)
    out = {}
    for j, t in enumerate(positions):
        for a, i in zip(v[t].float().tolist(), idx[t].tolist()):
            if a > 0:
                out.setdefault(i, [0.0] * len(positions))[j] = a
    return out


def run(model, ids, removals, t_read):
    """removals: list of (layer, kind, positions, acts per position, direction). -> probs [V] at t_read."""
    def tf(layer, kind, x):
        mine = [r for r in removals if (r[0], r[1]) == (layer, kind)]
        if not mine:
            return None
        out = x.clone()
        for _, _, ps, a, d in mine:
            for p, ap in zip(ps, a):
                out[0, p] = out[0, p] - ap * d
        return out
    with torch.no_grad(), multi_patch(model, tf):
        lg, _ = model(torch.tensor([ids]), return_all_logits=True)
    return torch.softmax(lg[0, t_read, : C.N_REAL_VOCAB].float(), -1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    model, bank, tok = C.load_model(), C.load_bank(), C.load_tokenizer()
    tid = lambda w: tok.encode("x " + w)[-1]
    ORD = [tid(w) for w in ("First", "Second", "Third")]
    LAW = [tid("Law"), tid("law"), tid("laws")] + [tok.encode("x La")[-1]]
    dirs = {k: C.decoder_dirs(bank, [C.parse_target(k)])[0] for k in (NEWTON, EPONYM)}

    print(f"{'prompt':38s} {'ablation':20s} {'ordinal':>8s} {'law':>7s}   (acts at ' and s)")
    for prompt in PROMPTS:
        ids = [C.BOS] + tok.encode(prompt)
        t_s = len(ids) - 1                     # the 's'
        pos = [t_s - 1, t_s]                   # apostrophe and 's'
        _, acts = C.forward_capture(model, ids)
        a9 = latent_acts(bank, acts, 9, "resid", pos)
        a10 = latent_acts(bank, acts, 10, "resid", pos)
        n_idx, e_idx = C.split_gid(C.parse_target(NEWTON))[2], C.split_gid(C.parse_target(EPONYM))[2]
        a_new, a_ep = a9.get(n_idx, [0.0, 0.0]), a10.get(e_idx, [0.0, 0.0])
        # control: another 9.resid latent active at the 's' with the closest activation to the Newton latent's
        ref = a_new[1] if a_new[1] > 0 else 10.0
        ctrl_i = min((i for i in a9 if i != n_idx and a9[i][1] > 0), key=lambda i: abs(a9[i][1] - ref))
        ctrl_dir = C.decoder_dirs(bank, [C.gid_of(9, "resid", ctrl_i)])[0]
        cases = {
            "clean": [],
            "- Newton latent": [(9, "resid", pos, a_new, dirs[NEWTON])],
            "- eponym latent": [(10, "resid", pos, a_ep, dirs[EPONYM])],
            "- both": [(9, "resid", pos, a_new, dirs[NEWTON]), (10, "resid", pos, a_ep, dirs[EPONYM])],
            f"- control 9.resid.{ctrl_i}": [(9, "resid", pos, a9[ctrl_i], ctrl_dir)],
        }
        for name, rem in cases.items():
            p = run(model, ids, rem, t_s)
            acts_note = (f"Newton {a_new[0]:.1f}/{a_new[1]:.1f}  eponym {a_ep[0]:.1f}/{a_ep[1]:.1f}"
                         if name == "clean" else "")
            print(f"{prompt:38s} {name:20s} {float(p[ORD].sum()):8.3f} {float(p[LAW].sum()):7.3f}   {acts_note}")
        print()


if __name__ == "__main__":
    main()
