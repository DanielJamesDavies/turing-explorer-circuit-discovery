"""Add the differential-seed (seed_vector) override to learned_mask.
Run from repo root: python experiments/043-relativity/patch_seed_vector.py
"""
import ast

P = "src/circuit/instrument/learned_mask.py"
s = open(P, encoding="utf-8").read()

old = ("    w_seed = sae.encoder.weight[seed_latent_idx].detach()\n"
       "    b_seed = sae._get_bias_eff()[seed_latent_idx].detach()")
assert s.count(old) == 1
new = (old + "\n"
       "    # DIFFERENTIAL SEED (2026-08-28): seed_vector=(w, b) overrides\n"
       "    # the encoder-row derivation, so a circuit can be fitted against\n"
       "    # a VIRTUAL direction -- e.g. w_A - w_B, whose reconstruction\n"
       "    # target is the difference signal between two same-site latents.\n"
       "    # Topic-shared composition cancels by construction; the circuit\n"
       "    # contains the differentia. Everything downstream (patchers,\n"
       "    # floors, scoring taps) consumes (w_seed, b_seed) verbatim.\n"
       "    if seed_vector is not None:\n"
       "        w_seed = seed_vector[0].detach()\n"
       "        b_seed = seed_vector[1].detach()")
s = s.replace(old, new)

old2 = "    seed_latent_idx: int,"
assert s.count(old2) == 1
s = s.replace(old2, old2 + "\n    seed_vector: "
              '"Optional[Tuple[torch.Tensor, torch.Tensor]]" = None,')

open(P, "w", encoding="utf-8", newline="").write(s)
ast.parse(s)
print("engine: seed_vector override added")
